"""Phase 29: one MultiStreamManager under every live channel of a process (engine unification).

Before this module the deployed runner gave each channel its own FFmpeg child, reader thread and backoff loop inside
`LiveStreamWorker`, and `MultiStreamManager` (the supervised, asyncio fleet engine with stall detection, jittered backoff,
bounded queues, fail-over and a RAM report) only ran in tests. Now `FleetIngest` owns the single manager and a channel's
`LiveStreamWorker` becomes a *consumer* of it: the manager spawns FFmpeg, supervises and reconnects; the worker receives
the audio (and the connect / disconnect events) through an `Inbox` and keeps doing what it has always done with it
(ring buffer, matching, discovery, evidence, ledger).

    fleet = FleetIngest()
    worker = LiveStreamWorker("Echorouk TV", resolver, db, fleet=fleet, channel_id="echorouk_tv").start()
"""
import collections
import logging
import re
import threading

from kihc.config import LIVE_SAMPLE_RATE
from kihc.ingest import multi_stream_manager as msm

log = logging.getLogger("kihc.fleet")

INBOX_MAX_SEGMENTS = 100            # 100 x 0.2 s = 20 s of audio per channel before the oldest segment is shed
SEGMENT_SECONDS = 0.2               # = the worker's CHUNK_SAMPLES (4410 samples at 22 050 Hz): one segment, one chunk


def channel_slug(name: str) -> str:
    """A catalogue-valid channel id ([a-z0-9_]+) from a display name or key."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "channel"


class Inbox:
    """Thread-safe hand-over from the manager's loop thread to one worker: events are never dropped, audio may be.

    `put_*` never blocks (it runs on the ingest loop): when the worker falls behind by more than INBOX_MAX_SEGMENTS the
    OLDEST segment is shed and counted, exactly like the manager's own queue, so one slow channel cannot stall FFmpeg or
    another channel."""

    def __init__(self, max_segments: int = INBOX_MAX_SEGMENTS):
        self._dq: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._segs = 0
        self.max_segments = max_segments
        self.dropped = 0

    def put_event(self, *ev) -> None:
        with self._cv:
            self._dq.append(ev)
            self._cv.notify()

    def put_segment(self, seg) -> None:
        with self._cv:
            if self._segs >= self.max_segments:
                for i, ev in enumerate(self._dq):
                    if ev[0] == "seg":
                        del self._dq[i]
                        self._segs -= 1
                        self.dropped += 1
                        break
            self._dq.append(("seg", seg))
            self._segs += 1
            self._cv.notify()

    def get(self, timeout: float = 0.5):
        with self._cv:
            if not self._dq:
                self._cv.wait(timeout)
            if not self._dq:
                return None
            ev = self._dq.popleft()
            if ev[0] == "seg":
                self._segs -= 1
            return ev

    def __len__(self) -> int:
        return len(self._dq)


class _Listener:
    """The manager-side callbacks for one worker: translate them into Inbox events (and let the worker read FFmpeg's log)."""

    def __init__(self, inbox: Inbox, on_stderr=None):
        self.inbox, self._on_stderr = inbox, on_stderr

    def on_attempt(self):
        self.inbox.put_event("attempt")

    def on_connect(self, url):
        self.inbox.put_event("connect", url)

    def on_segment(self, seg):
        self.inbox.put_segment(seg)

    def congested(self) -> bool:
        """True while the worker is far enough behind that reading more would shed audio: the manager then pauses this
        channel's reader (the same back-pressure the single-worker reader applied by blocking on its ring)."""
        return len(self.inbox) >= self.inbox.max_segments - 5

    def on_stderr(self, line):
        return bool(self._on_stderr and self._on_stderr(line))

    def on_disconnect(self, error, retry_in):
        self.inbox.put_event("disc", error, retry_in)


class _Source:
    __slots__ = ("source", "headers", "input_args", "info_for")

    def __init__(self, source, headers, input_args, info_for=None):
        self.source, self.headers, self.input_args, self.info_for = source, dict(headers or {}), input_args, info_for


class FleetIngest:
    """The process-wide MultiStreamManager plus the registry that maps a channel id to its source and FFmpeg options."""

    def __init__(self, settings: dict | None = None):
        cfg = dict(sample_rate=LIVE_SAMPLE_RATE, segment_seconds=SEGMENT_SECONDS, queue_max_segments=600,
                   stall_timeout_sec=20.0, backoff_base_sec=5.0, backoff_max_sec=120.0, healthy_after_sec=30.0,
                   unresolved_retry_sec=30.0, max_active=64, resolve_timeout_sec=30.0)
        cfg.update(settings or {})
        catalogue = msm.Catalogue({}, {**msm.DEFAULTS, **cfg}, ())
        self._sources: dict[str, _Source] = {}
        self._lock = threading.Lock()
        self.manager = msm.MultiStreamManager(catalogue, pools=(), resolver=self._resolve, command_builder=self._command,
                                              settings=cfg)
        self._started = False

    # ---- manager callbacks (blocking resolver runs in the manager's executor)
    def _resolve(self, ch: msm.ChannelProfile) -> str:
        src = self._sources[ch.id].source
        return src() if callable(src) else src

    def _command(self, url: str, ch: msm.ChannelProfile) -> list[str]:
        s, cfg = self._sources[ch.id], self.manager.cfg
        return msm.ffmpeg_command(url, stall_timeout=cfg["stall_timeout_sec"], live_start_index=cfg["live_start_index"],
                                  sample_rate=int(cfg["sample_rate"]), input_args=s.input_args,
                                  user_agent=s.headers.get("User-Agent"), referer=s.headers.get("Referer"),
                                  loglevel="level+info" if s.info_for and s.info_for(url) else "error")

    # ---- lifecycle
    def start(self) -> "FleetIngest":
        with self._lock:
            if not self._started:
                self.manager.start()
                self._started = True
        return self

    def stop(self, timeout: float = 10.0) -> None:
        with self._lock:
            if self._started:
                self.manager.stop(timeout)
                self._started = False

    # ---- channel ownership
    def attach(self, channel_id: str, name: str, source, inbox: Inbox, *, headers: dict | None = None,
               input_args: list[str] | None = None, tier: str = "priority", on_stderr=None, info_for=None,
               fresh: bool = True):
        """Make `inbox` the consumer of the channel and (re)start its supervised FFmpeg. -> the listener (hand it to detach()).

        `fresh` forces a new connection even if the channel is already running (a supervisor restart after a freeze): the
        new listener is installed first, so the previous worker's late `detach` finds itself replaced and does nothing."""
        self.start()
        with self._lock:
            self._sources[channel_id] = _Source(source, headers, input_args, info_for)
            if channel_id not in self.manager.catalogue.channels:
                self.manager.register_channel(msm.ChannelProfile(channel_id, name, tier))
            listener = _Listener(inbox, on_stderr)
            self.manager.set_listener(channel_id, listener)
        if fresh:
            self.manager.deactivate(channel_id)
        if not self.manager.activate(channel_id) and channel_id not in self.manager.active():
            log.warning("fleet: %s could not be activated (max_active=%s)", channel_id, self.manager.cfg["max_active"])
        return listener

    def detach(self, channel_id: str, listener=None) -> None:
        """Stop the channel, unless a newer worker has taken it over."""
        if self.manager.clear_listener(channel_id, listener):
            self.manager.deactivate(channel_id)

    # ---- observability
    def stat(self, channel_id: str):
        return self.manager.stats.get(channel_id)

    def memory_report(self) -> dict:
        return self.manager.memory_report()
