"""Multi-stream, audio-only ingestion of the Algerian channel fleet.

    python -m kihc.ingest.multi_stream_manager --pools priority --seconds 60     # smoke run, prints status
    python -m kihc.ingest.multi_stream_manager --list                            # catalogue and pools

One asyncio event loop in one thread supervises every channel. Each active channel is a coroutine that owns one
FFmpeg child, started as strict raw-audio extraction (-vn -acodec pcm_s16le -ar 16000 -ac 1, HLS
-live_start_index -2) and read in fixed segments (default 2 s = 64 KB). Every segment is stamped with its
capture time and its channel_id and put on ONE bounded, thread-safe queue (queue.Queue): consumers in any thread
call `manager.queue.get()`. A full queue drops its oldest segment (counted in `dropped`), so a slow consumer can never
back-pressure FFmpeg or another channel.

Why asyncio and not one process per channel: a Python process costs 25-30 MB before it does anything, an
asyncio task a few KB plus one 64 KB segment buffer. Isolation is per channel anyway: a channel's failure (resolver
error, network cut, FFmpeg crash, stall) is caught inside its own coroutine and retried with exponential backoff and
jitter; it never touches the others.

Memory, honestly: the *manager* (this Python process, all channel buffers, the queue's share) is budgeted well under
250 MB (see memory_report()); the FFmpeg children are separate OS processes and are reported separately
(`ffmpeg_total_mb`) - their cost per channel depends on the FFmpeg build and the stream, and is NOT under 10 MB
for a typical HLS demux; the 10 MB/channel budget applies to the Python-side state of a channel.

Pools: channels carry a tier ('priority' | 'secondary'). `active_pools` in the catalogue (or `pools=`) chooses which
tiers start; the rest stay in stand-by (no process, no memory) and `activate(id)` / `deactivate(id)` /
`failover(from_id, to_id)` switch them at run time, capped by `max_active`.

The legacy El-Heddaf pipeline (live_worker / pilot, 22.05 kHz, discovery + OCR) is untouched and independent: nothing
imports this module.
"""
import argparse
import asyncio
import collections
import json
import logging
import queue
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from kihc.config import DATA_DIR

log = logging.getLogger("kihc.ingest")

CATALOGUE_PATH = DATA_DIR / "channels.json"
OVERRIDES_PATH = DATA_DIR / "channel_overrides.json"
PCM_RATE, PCM_BYTES_PER_SAMPLE = 16000, 2
STATES = ("standby", "resolving", "connecting", "live", "backoff", "unresolved", "stopped")
DEFAULTS = dict(sample_rate=PCM_RATE, segment_seconds=2.0, queue_max_segments=600, stall_timeout_sec=20.0,
                backoff_base_sec=2.0, backoff_max_sec=60.0, healthy_after_sec=30.0, unresolved_retry_sec=300.0,
                live_start_index=-2, max_active=12, resolve_timeout_sec=30.0)


# ---------------------------------------------------------------- catalogue

@dataclass(frozen=True)
class ChannelProfile:
    id: str
    name: str
    tier: str = "secondary"            # priority | secondary
    group: str = ""
    enabled: bool = True
    source: dict = field(default_factory=dict)


@dataclass
class Catalogue:
    channels: dict                      # id -> ChannelProfile (file order)
    defaults: dict
    active_pools: tuple

    def select(self, pools=None, only=None) -> list[ChannelProfile]:
        """Enabled channels of the given tiers, or exactly `only` ids."""
        if only:
            missing = [c for c in only if c not in self.channels]
            if missing:
                raise KeyError(f"unknown channel id(s): {', '.join(missing)}")
            return [self.channels[c] for c in only if self.channels[c].enabled]
        pools = set(pools if pools is not None else self.active_pools)
        return [c for c in self.channels.values() if c.enabled and c.tier in pools]


def load_catalogue(path=None) -> Catalogue:
    data = json.loads(Path(path or CATALOGUE_PATH).read_text(encoding="utf-8"))
    chans = {}
    for row in data["channels"]:
        cid = str(row["id"]).strip()
        if not re.fullmatch(r"[a-z0-9_]+", cid) or cid in chans:
            raise ValueError(f"invalid or duplicate channel id: {cid!r}")
        if row.get("tier", "secondary") not in ("priority", "secondary"):
            raise ValueError(f"{cid}: tier must be priority or secondary")
        chans[cid] = ChannelProfile(cid, row.get("name", cid), row.get("tier", "secondary"), row.get("group", ""),
                                    bool(row.get("enabled", True)), dict(row.get("source") or {}))
    return Catalogue(chans, {**DEFAULTS, **data.get("defaults", {})}, tuple(data.get("active_pools", ("priority",))))


# ---------------------------------------------------------------- pieces

@dataclass(frozen=True)
class AudioSegment:
    channel_id: str
    seq: int                            # per channel, restarts never reset it
    wall_start: float                   # epoch seconds (capture clock) of the first sample
    sample_rate: int
    pcm: bytes                          # s16le mono

    @property
    def duration(self) -> float:
        return len(self.pcm) / PCM_BYTES_PER_SAMPLE / self.sample_rate


class SegmentQueue:
    """Bounded thread-safe queue; put() never blocks: when full the oldest segment is discarded."""

    def __init__(self, maxsize: int):
        self._q = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self.dropped = collections.Counter()

    def put(self, seg: AudioSegment) -> None:
        with self._lock:
            while True:
                try:
                    self._q.put_nowait(seg)
                    return
                except queue.Full:
                    try:
                        old = self._q.get_nowait()
                        self.dropped[old.channel_id] += 1
                    except queue.Empty:
                        pass

    def get(self, timeout: float | None = None) -> AudioSegment:
        return self._q.get(timeout=timeout)

    def get_nowait(self) -> AudioSegment:
        return self._q.get_nowait()

    def qsize(self) -> int:
        return self._q.qsize()


def backoff_delay(attempt: int, base: float, cap: float, rng=random) -> float:
    """Exponential backoff with equal jitter: half the nominal delay is fixed, half random (de-synchronises a fleet
    that lost the network at the same moment). attempt 0,1,2.. with base 2: 1-2 s, 2-4 s, 4-8 s ... capped at `cap`."""
    nominal = min(cap, base * (2.0 ** attempt))
    return nominal * (0.5 + 0.5 * rng.random())


def ffmpeg_command(url: str, *, stall_timeout: float, live_start_index: int, ffmpeg: str | None = None) -> list[str]:
    """Raw audio only: no video, subtitles or data streams; 16 kHz / 16-bit / mono PCM on stdout."""
    if ffmpeg is None:
        from kihc.core.audio import ffmpeg_exe
        ffmpeg = ffmpeg_exe()
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-loglevel", "error", "-threads", "1"]
    if re.match(r"https?://", url):
        cmd += ["-probesize", "262144", "-analyzeduration", "1000000", "-max_delay", "500000",   # capped probe memory
                "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_on_network_error", "1",
                "-reconnect_delay_max", "5", "-rw_timeout", str(int(stall_timeout * 1e6)),
                "-live_start_index", str(live_start_index)]       # input options: before -i
    cmd += ["-i", url, "-vn", "-sn", "-dn", "-acodec", "pcm_s16le", "-ar", str(PCM_RATE), "-ac", "1",
            "-f", "s16le", "pipe:1"]
    return cmd


class ChannelUnresolved(RuntimeError):
    pass


class DefaultResolver:
    """channel -> stream URL: channel_overrides.json, then the catalogue source. Blocking (run in an executor)."""

    PLAYLIST_TTL = 3600.0

    def __init__(self, overrides_path=OVERRIDES_PATH):
        self.overrides_path = Path(overrides_path)
        self._playlist, self._playlist_at = [], 0.0
        self._lock = threading.Lock()

    def _overrides(self) -> dict:
        from kihc.connectors.iptv_manager import parse_overrides
        try:
            return parse_overrides(json.loads(self.overrides_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return {}

    def _playlist_channels(self):
        with self._lock:
            if not self._playlist or time.time() - self._playlist_at > self.PLAYLIST_TTL:
                from kihc.connectors.iptv_manager import fallback_channels, fetch_channels
                try:
                    self._playlist = fetch_channels()
                except Exception:
                    self._playlist = fallback_channels()
                self._playlist_at = time.time()
            return self._playlist

    def __call__(self, ch: ChannelProfile) -> str:
        from kihc.connectors.iptv_manager import squash_key
        want = squash_key(ch.id)                        # Phase 28: the dedicated feed wins, no scraper is ever consulted
        override = next((u for k, u in self._overrides().items() if squash_key(k) == want), None)
        if override:
            return override
        from kihc.connectors.iptv_manager import iptv_only
        if iptv_only():
            raise ChannelUnresolved(f"{ch.name}: no dedicated IPTV feed (KIHC_IPTV_ONLY: no scraper fallback)")
        src = ch.source or {}
        kind = src.get("type")
        if kind == "hls" and src.get("url"):
            return src["url"]
        if kind == "broadcaster":
            from kihc.connectors.iptv_manager import resolve_channel_stream
            r = resolve_channel_stream(src["key"])
            if r.ok:
                return r.url
            raise ChannelUnresolved(f"{ch.name}: not live ({'; '.join(a[2][:60] for a in r.attempts)})")
        if kind == "playlist":
            for want in src.get("match", ()):
                rx = re.compile(r"(?<![\w])" + re.escape(want) + r"(?![\w])", re.I)
                for c in self._playlist_channels():
                    if rx.search(c.name):
                        return c.url
        raise ChannelUnresolved(f"{ch.name}: no stream URL known (add one to channel_overrides.json)")


# ---------------------------------------------------------------- manager

@dataclass
class _Stat:
    state: str = "standby"
    since: float = field(default_factory=time.time)
    segments: int = 0
    bytes: int = 0
    connects: int = 0
    failures: int = 0
    attempt: int = 0
    last_segment_at: float | None = None
    last_error: str = ""
    pid: int | None = None
    seq: int = 0
    next_retry_in: float = 0.0


class MultiStreamManager:
    def __init__(self, catalogue: Catalogue, *, pools=None, only=None, resolver=None, command_builder=None,
                 settings: dict | None = None, rng=None):
        self.catalogue = catalogue
        self.cfg = {**catalogue.defaults, **(settings or {})}
        self.resolver = resolver or DefaultResolver()
        self.command_builder = command_builder or (
            lambda url, ch: ffmpeg_command(url, stall_timeout=self.cfg["stall_timeout_sec"],
                                           live_start_index=self.cfg["live_start_index"]))
        self.rng = rng or random.Random()
        self.queue = SegmentQueue(int(self.cfg["queue_max_segments"]))
        self.stats = {cid: _Stat() for cid in catalogue.channels}
        self._initial = [c.id for c in catalogue.select(pools, only)]
        self._tasks: dict[str, asyncio.Task] = {}
        self._procs: dict[str, object] = {}
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._stop_evt = None

    # ---- lifecycle
    def start(self) -> "MultiStreamManager":
        if self._thread:
            return self
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), name="kihc-ingest", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("ingest loop did not start")
        return self

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._stop_evt = asyncio.Event()
        for cid in self._initial[: int(self.cfg["max_active"])]:
            self._spawn_task(cid)
        self._ready.set()
        await self._stop_evt.wait()
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self, timeout: float = 10.0) -> None:
        if self._thread and self._loop:
            self._loop.call_soon_threadsafe(self._stop_evt.set)
            self._thread.join(timeout)
        self._thread = None

    def _call(self, fn):
        async def run():
            return fn()
        return asyncio.run_coroutine_threadsafe(run(), self._loop).result(10)

    # ---- pool control (thread-safe)
    def active(self) -> list[str]:
        return sorted(self._tasks)

    def activate(self, channel_id: str) -> bool:
        """Start a stand-by channel. False if unknown, disabled, already active or the max_active cap is reached."""
        ch = self.catalogue.channels.get(channel_id)
        if ch is None or not ch.enabled:
            return False
        return self._call(lambda: self._spawn_task(channel_id))

    def deactivate(self, channel_id: str) -> bool:
        return self._call(lambda: self._cancel_task(channel_id))

    def failover(self, from_id: str, to_id: str) -> bool:
        """Swap a failed/blocked channel for a stand-by one."""
        self.deactivate(from_id)
        return self.activate(to_id)

    def _spawn_task(self, cid: str) -> bool:
        if cid in self._tasks or len(self._tasks) >= int(self.cfg["max_active"]):
            return False
        self.stats[cid] = _Stat(state="resolving")
        self._tasks[cid] = asyncio.get_running_loop().create_task(self._run_channel(self.catalogue.channels[cid]),
                                                                  name=f"ingest:{cid}")
        return True

    def _cancel_task(self, cid: str) -> bool:
        t = self._tasks.pop(cid, None)
        if t is None:
            return False
        t.cancel()
        self.stats[cid].state = "standby"
        self.stats[cid].since = time.time()
        return True

    # ---- per-channel supervision: nothing escapes this coroutine except cancellation
    async def _run_channel(self, ch: ChannelProfile):
        st, cfg = self.stats[ch.id], self.cfg
        loop = asyncio.get_running_loop()
        attempt = 0
        try:
            while True:
                st.state = "resolving"
                try:
                    url = await asyncio.wait_for(loop.run_in_executor(None, self.resolver, ch),
                                                 cfg["resolve_timeout_sec"])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    st.state, st.last_error, st.failures = "unresolved", f"{type(exc).__name__}: {exc}", st.failures + 1
                    await self._sleep(st, backoff_delay(0, cfg["unresolved_retry_sec"], cfg["unresolved_retry_sec"], self.rng))
                    continue
                started = time.monotonic()
                try:
                    await self._stream_once(ch, url, st)
                    st.last_error = st.last_error or "stream ended"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    st.last_error = f"{type(exc).__name__}: {exc}"
                st.failures += 1
                if time.monotonic() - started >= cfg["healthy_after_sec"]:
                    attempt = 0
                st.attempt = attempt
                delay = backoff_delay(attempt, cfg["backoff_base_sec"], cfg["backoff_max_sec"], self.rng)
                attempt += 1
                st.state = "backoff"
                log.warning("%s: %s - retry in %.1f s", ch.id, st.last_error, delay)
                await self._sleep(st, delay)
        finally:
            if self._tasks.get(ch.id) is asyncio.current_task():
                del self._tasks[ch.id]

    @staticmethod
    async def _sleep(st: _Stat, delay: float):
        st.next_retry_in = delay
        await asyncio.sleep(delay)
        st.next_retry_in = 0.0

    async def _stream_once(self, ch: ChannelProfile, url: str, st: _Stat):
        cfg = self.cfg
        sr = int(cfg["sample_rate"])
        seg_bytes = int(sr * cfg["segment_seconds"]) * PCM_BYTES_PER_SAMPLE
        st.state = "connecting"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        kw = dict(creationflags=flags) if flags else {}
        proc = await asyncio.create_subprocess_exec(*self.command_builder(url, ch), stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                    limit=seg_bytes * 2, **kw)
        st.pid, st.connects = proc.pid, st.connects + 1
        err_tail = collections.deque(maxlen=3)

        async def drain():
            while line := await proc.stderr.readline():
                err_tail.append(line.decode(errors="replace").strip()[:200])
        drain_task = asyncio.get_running_loop().create_task(drain())
        n = 0
        anchor = 0.0
        try:
            while True:
                try:
                    pcm = await asyncio.wait_for(proc.stdout.readexactly(seg_bytes), cfg["stall_timeout_sec"])
                except asyncio.IncompleteReadError:
                    rc = await asyncio.wait_for(proc.wait(), 5)
                    raise ConnectionError(f"ffmpeg exited rc={rc}: {' | '.join(err_tail)}"[:300])
                except asyncio.TimeoutError:
                    raise ConnectionError(f"no audio for {cfg['stall_timeout_sec']:.0f} s (stall)")
                now = time.time()
                dur = len(pcm) / PCM_BYTES_PER_SAMPLE / sr
                if n == 0:
                    anchor, st.state = now - dur, "live"
                # capture clock: contiguous from the anchor, but never in the future (FFmpeg bursts at connect)
                wall0 = min(anchor + n * dur, now - dur)
                if now - (anchor + n * dur) > 5.0:         # fell behind real time (consumer-side stall): re-anchor
                    anchor, n, wall0 = now - dur, 0, now - dur
                st.seq += 1
                self.queue.put(AudioSegment(ch.id, st.seq, wall0, sr, pcm))
                n += 1
                st.segments, st.bytes, st.last_segment_at = st.segments + 1, st.bytes + len(pcm), now
        finally:
            drain_task.cancel()
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), 5)
                except Exception:
                    pass
            transport = getattr(proc, "_transport", None)      # close pipes now: the loop may end right after a cancel
            if transport is not None:
                transport.close()
            st.pid = None

    # ---- observability
    def status(self) -> dict:
        now = time.time()
        out = {}
        for cid, ch in self.catalogue.channels.items():
            s = self.stats[cid]
            out[cid] = dict(name=ch.name, tier=ch.tier, enabled=ch.enabled, state=s.state,
                            active=cid in self._tasks, segments=s.segments, connects=s.connects, failures=s.failures,
                            dropped=self.queue.dropped[cid], last_error=s.last_error, ffmpeg_pid=s.pid,
                            seconds_since_audio=round(now - s.last_segment_at, 1) if s.last_segment_at else None,
                            next_retry_in=round(s.next_retry_in, 1))
        return out

    def memory_report(self) -> dict:
        from kihc.service.sysinfo import memory_mb
        rss, peak = memory_mb()
        children = {cid: memory_mb(s.pid)[0] for cid, s in self.stats.items() if s.pid}
        n = max(1, len(self._tasks))
        return dict(manager_rss_mb=round(rss, 1), manager_peak_mb=round(peak, 1), active_channels=len(self._tasks),
                    ffmpeg_total_mb=round(sum(children.values()), 1),
                    ffmpeg_mb_per_channel={k: round(v, 1) for k, v in children.items()},
                    queue_segments=self.queue.qsize(),
                    queue_mb=round(self.queue.qsize() * self.cfg["segment_seconds"] * self.cfg["sample_rate"]
                                   * PCM_BYTES_PER_SAMPLE / 2 ** 20, 1),
                    python_budget_mb=250.0, python_mb_per_channel=round(rss / n, 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", default=None)
    ap.add_argument("--pools", nargs="*", default=None, help="tiers to start (default: catalogue active_pools)")
    ap.add_argument("--channels", nargs="*", default=None, help="exact channel ids (overrides --pools)")
    ap.add_argument("--seconds", type=float, default=60.0, help="run time; 0 = until Ctrl+C")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    cat = load_catalogue(a.catalogue)
    if a.list:
        for c in cat.channels.values():
            print(f"{c.id:28s} {c.tier:9s} {c.group:11s} {c.source.get('type', '-'):11s} {c.name}")
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    mgr = MultiStreamManager(cat, pools=a.pools, only=a.channels).start()
    end = time.time() + a.seconds if a.seconds else float("inf")
    counts = collections.Counter()
    next_print = time.time() + 10
    try:
        while time.time() < end:
            try:
                counts[mgr.queue.get(timeout=1.0).channel_id] += 1
            except queue.Empty:
                pass
            if time.time() >= next_print:
                next_print = time.time() + 10
                print(json.dumps({k: (v["state"], v["segments"]) for k, v in mgr.status().items() if v["active"]}),
                      json.dumps(mgr.memory_report()), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
    print("segments consumed:", dict(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
