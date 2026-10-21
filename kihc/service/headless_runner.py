"""KIHC 24/7 headless monitoring service (no Streamlit, no browser).

    python -m kihc.cli run-service --channel el_heddaf_tv [--channel echorouk_tv ...]
    python -m kihc.cli run-service                    # channels from data/service_channels.json
    python -m kihc.cli stop-service                   # graceful stop from any console

Supervisor responsibilities (every tick, default 5 s):
  * heartbeat row + per-channel telemetry in SQLite (read by the dashboard)
  * restart a worker whose threads died, whose stream delivered no audio data for
    `freeze_after` seconds (default 20) while it claimed to be listening, or that has been
    stuck connecting (no first audio) for `connect_timeout` seconds (default 45)
    (data flow, not loudness: a channel airing silence is not frozen; the radar flags dead air)
  * gc.collect() every `gc_every` seconds (default 60 min)
  * graceful shutdown on SIGINT / SIGTERM / SIGBREAK or a stop-request file:
    workers are stopped so pending matches are confirmed, then the lock is released
Single instance per database via an OS file lock.
"""
import gc
import json
import logging
import logging.handlers
import os
import re
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from kihc.config import DATA_DIR, DEFAULT_DB_PATH, EVIDENCE_DIR
from kihc.core.live_worker import LiveStreamWorker
from kihc.database.schema import init_db
from kihc.service.lock import ProcessLock, service_paths
from kihc.service.sysinfo import memory_mb

log = logging.getLogger("kihc.service")

CONFIG_FILE = DATA_DIR / "service_channels.json"
RAM_BUDGET_MB = 250.0
EXIT_OK, EXIT_ALREADY_RUNNING, EXIT_NO_CHANNELS = 0, 3, 2


class ServiceAlreadyRunning(RuntimeError):
    pass


@dataclass
class ChannelSpec:
    name: str
    source: str | Callable[[], str]
    headers: dict = field(default_factory=dict)
    key: str = ""
    feed_label: str = ""          # shown in telemetry, e.g. "Direct Web Feed"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def resolve_channel_arg(arg: str) -> ChannelSpec:
    """'name=url_or_path' | national key (el_heddaf_tv, or prefix 'el_heddaf') | IPTV channel slug."""
    from kihc.connectors import iptv_manager as iptv

    if "=" in arg:
        name, src = arg.split("=", 1)
        return ChannelSpec(name.strip(), src.strip(), key=_slug(name))
    key = _slug(arg)
    if iptv.override_url(key):                          # Phase 28: a dedicated IPTV feed exists for this channel
        return ChannelSpec(iptv.display_name(key), iptv.channel_resolver(key), key=key, feed_label="Dedicated IPTV feed")
    squash = key.replace("_", "")                       # 'elheddaf' == 'el_heddaf'
    matches = [k for k in iptv.NATIONAL_BROADCASTERS if k == key or k.startswith(key)
               or k.replace("_", "").startswith(squash) or _slug(iptv.NATIONAL_BROADCASTERS[k].name) == key]
    if len(matches) == 1:
        b = iptv.NATIONAL_BROADCASTERS[matches[0]]
        return ChannelSpec(b.name, iptv.channel_resolver(b.key), key=b.key,
                           feed_label="Dedicated IPTV feed" if iptv.override_url(b.key) else "")
    if len(matches) > 1:
        raise ValueError(f"'{arg}' is ambiguous: {matches}")
    for ch in iptv.fetch_channels():
        if _slug(ch.name) == key or _slug(ch.name).startswith(key):
            return ChannelSpec(ch.name, ch.url, ch.headers, key=_slug(ch.name))
    raise ValueError(f"Unknown channel '{arg}'. Use a key such as {list(iptv.NATIONAL_BROADCASTERS)} "
                     f"or 'Name=https://.../index.m3u8'.")


def load_config_channels(path=CONFIG_FILE) -> list[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [c for c in data.get("channels", []) if isinstance(c, str)]
    except (OSError, ValueError, AttributeError):
        return []


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Supervisor:
    def __init__(self, channels: list[ChannelSpec], db_path=None, *, tick: float = 5.0,
                 freeze_after: float = 20.0, connect_timeout: float = 45.0, gc_every: float = 3600.0,
                 evidence_dir=EVIDENCE_DIR, worker_factory: Callable[[ChannelSpec], object] | None = None,
                 worker_kwargs: dict | None = None):
        if not channels:
            raise ValueError("no channels to monitor")
        self.channels = channels
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self.tick_seconds, self.freeze_after, self.gc_every = tick, freeze_after, gc_every
        self.connect_timeout = connect_timeout
        self._connecting_since: dict[int, float] = {}
        self.evidence_dir = evidence_dir
        self.worker_kwargs = worker_kwargs or {}
        self._factory = worker_factory or self._default_factory
        self.paths = service_paths(self.db_path)
        self.lock = ProcessLock(self.paths["lock"])
        self._stop = threading.Event()
        self.stop_reason = ""
        self.workers: dict[str, object] = {}
        self.restarts: dict[str, int] = {c.name: 0 for c in channels}
        self.restart_log: list[tuple[float, str, str]] = []
        self.frame_ingests: dict[str, object] = {}      # channel -> FrameIngest, owned (started/stopped) by the caller
        self.gc_runs, self.last_gc_at = 0, None
        self._last_gc = time.monotonic()
        self.started_at = None
        self.ticks = 0

    # ------------------------------------------------------------ lifecycle
    def _default_factory(self, spec: ChannelSpec):
        return LiveStreamWorker(spec.name, spec.source, self.db_path, headers=spec.headers,
                                evidence_dir=self.evidence_dir, **self.worker_kwargs)

    def request_stop(self, reason: str = "requested"):
        if not self._stop.is_set():
            self.stop_reason = reason
            log.info("stop requested: %s", reason)
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def start(self):
        if not self.lock.acquire():
            raise ServiceAlreadyRunning(f"another KIHC service holds {self.paths['lock']}")
        try:
            self.paths["stop"].unlink(missing_ok=True)      # stale stop request from a previous run
            self.started_at = _now_iso()
            self._write_heartbeat("starting")
            for spec in self.channels:
                self.workers[spec.name] = self._factory(spec).start()
                log.info("started worker for %s", spec.name)
        except BaseException:
            for w in self.workers.values():
                w.stop()
            self._write_heartbeat("stopped", message="startup failed")
            self.lock.release()
            raise

    def run(self) -> int:
        """Blocking supervisor loop. Returns a process exit code."""
        self.start()
        try:
            while not self._stop.is_set():
                self.tick()
                self._stop.wait(self.tick_seconds)
        finally:
            self.shutdown()
        return EXIT_OK

    def shutdown(self):
        self._stop.set()
        self._write_heartbeat("stopping")
        stoppers = [threading.Thread(target=w.stop, daemon=True) for w in self.workers.values()]
        for t in stoppers:                                   # parallel: each stop flushes pending matches
            t.start()
        for t in stoppers:
            t.join(15)
        self._write_channels()
        self._write_heartbeat("stopped", message=f"graceful shutdown ({self.stop_reason or 'requested'})")
        log.info("service stopped (%s)", self.stop_reason or "requested")
        self.paths["stop"].unlink(missing_ok=True)
        self.lock.release()

    # ------------------------------------------------------------ supervision
    def tick(self):
        self.ticks += 1
        if self.paths["stop"].exists():
            self.request_stop("stop-request file")
            return
        for spec in self.channels:
            w = self.workers.get(spec.name)
            reason = self._needs_restart(w)
            if reason:
                self._restart(spec, reason)
        if time.monotonic() - self._last_gc >= self.gc_every:
            freed = gc.collect()
            self.gc_runs += 1
            self.last_gc_at = _now_iso()
            self._last_gc = time.monotonic()
            log.info("gc.collect(): %d objects freed", freed)
        self._write_channels()
        self._write_heartbeat("running")

    def _needs_restart(self, w) -> str | None:
        if w is None:
            return "worker missing"
        if not w.is_alive():
            return f"worker threads ended (state={w.state}, last_error={w.last_error[:80]!r})"
        s = w.status()
        since = s.get("seconds_since_data")
        if s["state"] == "listening" and since is not None and since > self.freeze_after:
            return f"frozen: no audio data for {since:.0f}s"
        # A pipe can also hang before the first audio arrives (server stalls mid-probe).
        key = id(w)
        if s["state"] == "connecting":
            first = self._connecting_since.setdefault(key, time.monotonic())
            if time.monotonic() - first > self.connect_timeout:
                self._connecting_since.pop(key, None)
                return f"stuck connecting for {self.connect_timeout:.0f}s with no audio"
        else:
            self._connecting_since.pop(key, None)
        return None

    def _restart(self, spec: ChannelSpec, reason: str):
        old = self.workers.get(spec.name)
        self.restarts[spec.name] += 1
        self.restart_log.append((time.time(), spec.name, reason))
        log.warning("restarting %s (#%d): %s", spec.name, self.restarts[spec.name], reason)
        if old is not None:   # stop in the background so one hung pipe can't stall the supervisor
            threading.Thread(target=old.stop, daemon=True, name=f"kihc-reap-{spec.name}").start()
        self.workers[spec.name] = self._factory(spec).start()

    # ------------------------------------------------------------ telemetry
    def _ffmpeg_rss(self, statuses) -> float:
        return sum(memory_mb(s["ffmpeg_pid"])[0] for s in statuses if s.get("ffmpeg_pid"))

    def _write_channels(self):
        rows, statuses = [], []
        refs = {c.name: (c.source if isinstance(c.source, str) else f"channel:{c.key}" if c.key else None)
                for c in self.channels}
        labels = {c.name: c.feed_label for c in self.channels}
        for name, w in self.workers.items():
            s = w.status()
            s["level_history"] = s.get("level_history", [])[-150:]
            s["stream_ref"] = refs.get(name)        # lets the dashboard offer a client-side preview
            s["feed_label"] = labels.get(name, "")
            fi = self.frame_ingests.get(name)
            if fi is not None:                      # Phase 25: live 2 fps video telemetry next to the audio status
                try:
                    from kihc.analytics import pod_boundary_detector as pbd
                    s["video"] = dict(fi.status(), sensing=pbd.get_detector(name).sensing_status())
                except Exception:
                    log.exception("video telemetry failed")
            statuses.append(s)
            rows.append((name, s["state"], time.time(), self.restarts.get(name, 0), json.dumps(s)))
        self._last_statuses = statuses
        self._db(lambda c: c.executemany(
            "INSERT INTO service_channel_status (channel, state, updated_epoch, restarts, status_json) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(channel) DO UPDATE SET state=excluded.state, "
            "updated_epoch=excluded.updated_epoch, restarts=excluded.restarts, status_json=excluded.status_json",
            rows))

    def _write_heartbeat(self, state: str, message: str = ""):
        rss, peak = memory_mb()
        ffmpeg = self._ffmpeg_rss(getattr(self, "_last_statuses", []))
        if rss + ffmpeg > RAM_BUDGET_MB:
            message = message or f"over RAM budget: {rss + ffmpeg:.0f} MB > {RAM_BUDGET_MB:.0f} MB"
            log.warning(message)
        self._db(lambda c: c.execute(
            "INSERT INTO service_heartbeat (service_id, pid, hostname, state, started_at, heartbeat_at, "
            "heartbeat_epoch, tick_seconds, rss_mb, peak_rss_mb, ffmpeg_rss_mb, channels, restarts, gc_runs, "
            "last_gc_at, message) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(service_id) DO UPDATE SET pid=excluded.pid, hostname=excluded.hostname, "
            "state=excluded.state, started_at=excluded.started_at, heartbeat_at=excluded.heartbeat_at, "
            "heartbeat_epoch=excluded.heartbeat_epoch, tick_seconds=excluded.tick_seconds, rss_mb=excluded.rss_mb, "
            "peak_rss_mb=excluded.peak_rss_mb, ffmpeg_rss_mb=excluded.ffmpeg_rss_mb, channels=excluded.channels, "
            "restarts=excluded.restarts, gc_runs=excluded.gc_runs, last_gc_at=excluded.last_gc_at, "
            "message=excluded.message",
            (os.getpid(), socket.gethostname(), state, self.started_at, _now_iso(), time.time(),
             self.tick_seconds, round(rss, 1), round(peak, 1), round(ffmpeg, 1), len(self.channels),
             sum(self.restarts.values()), self.gc_runs, self.last_gc_at, message)))

    def _db(self, fn):
        try:
            conn = init_db(self.db_path)
            try:
                with conn:
                    fn(conn)
            finally:
                conn.close()
        except Exception:   # telemetry must never take the service down
            log.exception("telemetry write failed")


# ---------------------------------------------------------------- service status (for the dashboard / CLI)

def read_service_status(db_path=None, stale_after: float | None = None) -> dict | None:
    """Heartbeat + channel telemetry, with `active` = lock held AND heartbeat fresh."""
    from kihc.service.lock import is_locked
    db_path = str(db_path or DEFAULT_DB_PATH)
    conn = init_db(db_path)
    try:
        hb = conn.execute("SELECT * FROM service_heartbeat WHERE service_id = 1").fetchone()
        chans = conn.execute("SELECT * FROM service_channel_status ORDER BY channel").fetchall()
    finally:
        conn.close()
    if hb is None:
        return None
    hb = dict(hb)
    stale_after = stale_after or 3 * (hb["tick_seconds"] or 5.0) + 5.0
    age = time.time() - (hb["heartbeat_epoch"] or 0)
    locked = is_locked(service_paths(db_path)["lock"])
    hb.update(heartbeat_age_s=round(age, 1), lock_held=locked,
              active=locked and hb["state"] in ("starting", "running") and age <= stale_after,
              channel_status=[dict(name=c["channel"], restarts=c["restarts"], **json.loads(c["status_json"]))
                              for c in chans] if hb["state"] in ("starting", "running") else [])
    return hb


# ---------------------------------------------------------------- entry point

def _setup_logging(log_path: Path, verbose: bool = False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s")
    root = logging.getLogger("kihc")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=5 * 2 ** 20, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def install_signal_handlers(sup: Supervisor):
    def handler(signum, _frame):
        sup.request_stop(signal.Signals(signum).name)
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):          # SIGBREAK: Ctrl+Break / console close on Windows
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)


def run_service(channel_args: list[str] | None = None, db_path=None, *, config=CONFIG_FILE, tick: float = 5.0,
                freeze_after: float = 20.0, connect_timeout: float = 45.0, gc_every: float = 3600.0,
                verbose: bool = False) -> int:
    db_path = str(db_path or DEFAULT_DB_PATH)
    _setup_logging(service_paths(db_path)["log"], verbose)
    args = channel_args or load_config_channels(config)
    if not args:
        log.error("no channels given (use --channel or %s)", config)
        return EXIT_NO_CHANNELS
    specs = []
    for a in args:
        try:
            specs.append(resolve_channel_arg(a))
        except ValueError as exc:
            log.error("%s", exc)
            return EXIT_NO_CHANNELS
    sup = Supervisor(specs, db_path, tick=tick, freeze_after=freeze_after, connect_timeout=connect_timeout,
                     gc_every=gc_every)
    if os.environ.get("KIHC_FRAMES", "on").lower() not in ("off", "0", "false"):
        from kihc.stream import elheddaf_stream as heddaf
        for s in specs:                              # El-Heddaf only: the one feed with a verified video rendition
            if s.name == heddaf.CHANNEL_NAME and not s.feed_label.startswith("Dedicated"):   # the video rendition is the web feed's
                sup.frame_ingests[s.name] = heddaf.FrameIngest(s.name).start()
                log.info("video boundary sensing ON for %s: %dx%d @ %d fps", s.name, heddaf.FRAME_W, heddaf.FRAME_H,
                         heddaf.FRAME_FPS)
    install_signal_handlers(sup)
    log.info("KIHC service starting: pid=%d db=%s channels=%s", os.getpid(), db_path, [s.name for s in specs])
    try:
        return sup.run()
    except ServiceAlreadyRunning as exc:
        log.error("%s", exc)
        return EXIT_ALREADY_RUNNING
    finally:
        for fi in sup.frame_ingests.values():
            fi.stop()
