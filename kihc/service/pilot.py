"""Unified pilot: background ingest + discovery supervisor and the web dashboard in one process.

    python -m kihc.cli pilot-run --channel elheddaf --port 8502

* Ingest: DiscoveryWorker (reference-ad matching + unsupervised discovery) under the headless Supervisor:
  heartbeat, frozen-pipe restart after 20 s without audio, stuck-connect restart after 45 s,
  gc.collect() every 30 min, single-instance DB lock.
* El-Heddaf: verified direct web feed (audio-only rendition), FFmpeg reconnect flags, backoff 5 -> 15 -> 30 s.
* Console: one line per event, no tracebacks; full tracebacks go to the rotating log file.
* Web: FastAPI dashboard on 127.0.0.1:<port>; stopping the server (Ctrl+C) stops ingest gracefully.
"""
import logging
import logging.handlers
import os
import threading
import time
from pathlib import Path

from kihc.connectors import iptv_manager
from kihc.config import DEFAULT_DB_PATH, DISCOVERED_DIR, EVIDENCE_DIR
from kihc.core.discovery import DiscoveryEngine
from kihc.core.discovery_worker import DiscoveryWorker
from kihc.core.reference_worker import ReferenceDiscoveryWorker
from kihc.ingest.fleet import channel_slug
from kihc.service.headless_runner import (ChannelSpec, ServiceAlreadyRunning, Supervisor,
                                          resolve_channel_arg)
from kihc.service.lock import service_paths
from kihc.service.sysinfo import memory_mb
from kihc.stream import elheddaf_stream as heddaf

log = logging.getLogger("kihc.pilot")
ELHEDDAF_ALIASES = {"elheddaf", "el_heddaf", "elheddaftv", "el_heddaf_tv", "el-heddaf", "el-heddaf tv"}


class OneLineFormatter(logging.Formatter):
    """Console formatter: a single line per record, never a traceback."""

    def format(self, record):
        record = logging.makeLogRecord(record.__dict__)
        msg = record.getMessage().replace("\n", " ")
        if record.exc_info and record.exc_info[1] is not None:
            msg += f" ({type(record.exc_info[1]).__name__}: {str(record.exc_info[1])[:160]})"
        record.msg, record.args, record.exc_info, record.exc_text, record.stack_info = msg, None, None, None, None
        return super().format(record)


def configure_logging(log_path: Path, console_level=logging.INFO):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("kihc")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=5 * 2 ** 20, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(OneLineFormatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    ch.setLevel(console_level)
    root.addHandler(fh)
    root.addHandler(ch)
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "streamlink"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def pilot_channel(channel: str) -> tuple[ChannelSpec, dict]:
    """-> (ChannelSpec, extra LiveStreamWorker kwargs)."""
    if channel.strip().casefold() in ELHEDDAF_ALIASES and not iptv_manager.override_url("el_heddaf_tv"):
        return (ChannelSpec(heddaf.CHANNEL_NAME, heddaf.elheddaf_resolver(), key="el_heddaf_tv",
                            feed_label=heddaf.FEED_LABEL),
                dict(ffmpeg_input_args=heddaf.FFMPEG_INPUT_ARGS, backoff_base=5.0, backoff_factor=3.0,
                     backoff_max=30.0))
    return resolve_channel_arg(channel), dict(backoff_base=5.0, backoff_factor=3.0, backoff_max=30.0)


def split_channels(channel: str) -> list[str]:
    """'a,b,c' -> ['a', 'b', 'c']; the token '@overrides' expands to every dedicated feed in channel_overrides.json."""
    out: list[str] = []
    if "=" in str(channel):                           # 'Name=URL' may legitimately contain commas
        return [str(channel).strip()]
    for tok in (t.strip() for t in str(channel).split(",")):
        if tok == "@overrides":
            out += [k for k in iptv_manager._overrides() if k not in out]
        elif tok and tok not in out:
            out.append(tok)
    return out


def build_supervisor(channel: str, db_path, *, history_hours: float = 4.0, gc_every: float = 1800.0,
                     tick: float = 5.0, discovery_kwargs: dict | None = None, worker_kwargs: dict | None = None,
                     packshot=None, reference=None, fleet="default"):
    """`channel` may name several channels ('a,b,c' / '@overrides'): one supervisor, one worker + discovery engine each.
    `fleet`: the shared MultiStreamManager ('default' = the production engine, None = a private FFmpeg per worker)."""
    chosen = [pilot_channel(c) for c in split_channels(channel)]
    if not chosen:
        raise ValueError("no channels to monitor")
    specs = [s for s, _ in chosen]
    spec, per_channel_kw = specs[0], {s.name: dict(kw, **(worker_kwargs or {})) for s, kw in chosen}
    if fleet == "default":
        from kihc.service.headless_runner import default_fleet
        fleet = default_fleet()
    dkw = dict(history_hours=history_hours, **(discovery_kwargs or {}))
    engines = {}                                      # one discovery engine per channel for the pilot's lifetime

    def factory(s: ChannelSpec):
        wkw = dict(per_channel_kw[s.name])
        if fleet is not None:                         # Phase 29: the channel's audio comes from the shared manager
            wkw.update(fleet=fleet, channel_id=s.key or channel_slug(s.name))
        engine = engines.get(s.name)
        if engine is None:
            engine = engines[s.name] = DiscoveryEngine(s.name, db_path, **dkw)
            restored = engine.load_state(history_state_path(db_path, s.name))
            if restored:
                log.info("discovery history restored: %.0f min of %s", restored / 60, s.name)
        # A supervisor restart (frozen stream) reuses the engine: before Phase 11.4 every restart wiped the
        # look-back window, so ads seen before it could not be recognised as repeats afterwards.
        if reference is not None:    # exact reference matcher + the unchanged discovery loop on the same audio
            w = ReferenceDiscoveryWorker(s.name, s.source, db_path, headers=s.headers, evidence_dir=EVIDENCE_DIR,
                                         discovery=engine, reference=reference, **wkw)
        else:
            w = DiscoveryWorker(s.name, s.source, db_path, headers=s.headers, evidence_dir=EVIDENCE_DIR,
                                discovery=engine, **wkw)
        if packshot is not None:
            packshot.attach(engine)
        return w
    sup = Supervisor(specs, db_path, tick=tick, freeze_after=20.0, connect_timeout=45.0, gc_every=gc_every,
                     worker_factory=factory, fleet=fleet)
    sup.discovery_engines = engines
    return sup, spec


def history_state_path(db_path, channel: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in channel)
    return service_paths(db_path)["log"].parent.parent / "state" / f"discovery_{safe}.npz"


class TranscriptionScheduler:
    """Launches a background batch (default: speech-to-text, kihc.analytics.transcriber) as a separate
    low-priority process whenever it has pending work. Heavy models never load into the monitoring process,
    and schedulers share one slot so at most one such process runs at a time (memory peaks don't stack)."""

    _heavy_slot = threading.Lock()

    def __init__(self, db_path, model="base", memory_cap_mb=480.0, interval=120.0, log_path=None, *,
                 module="kihc.analytics.transcriber", pending_fn=None, extra_args=None, name="transcription"):
        self.db_path, self.model, self.cap, self.interval = str(db_path), model, memory_cap_mb, interval
        self.log_path = Path(log_path) if log_path else None
        self.module, self.name = module, name
        self.extra_args = ["--model", model] if extra_args is None else list(extra_args)
        if pending_fn is None:
            from kihc.analytics.transcriber import pending_assets as pending_fn
        self.pending_fn = pending_fn
        self._stop = threading.Event()
        self._proc = None
        self._holding = False
        self._failures = 0
        self._thread = threading.Thread(target=self._loop, name=f"kihc-{name}-scheduler", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._thread.join(10)
        self._release()

    def _release(self):
        if self._holding:
            self._holding = False
            self._heavy_slot.release()

    def _loop(self):
        while not self._stop.wait(5 if self._proc is None else self.interval / 4):
            try:
                if self._proc is not None:
                    rc = self._proc.poll()
                    if rc is None:
                        continue
                    self._proc = None
                    self._release()
                    if rc == 0:
                        self._failures = 0
                        log.info("%s batch finished", self.name)
                    else:
                        self._failures += 1
                        why = "memory cap reached" if rc == 75 else f"exit code {rc}"
                        log.warning("%s batch failed (%s); see %s", self.name, why, self.log_path)
                if self._failures >= 2:                                   # back off 30 min after repeated failures
                    if self._stop.wait(1800):
                        return
                    self._failures = 0
                pending = self.pending_fn(self.db_path)
                if pending:
                    if not self._heavy_slot.acquire(blocking=False):
                        continue                                          # another batch is running: retry soon
                    self._holding = True
                    try:
                        self._launch(len(pending))
                    except Exception:
                        self._release()
                        raise
                elif self._stop.wait(self.interval):
                    return
            except Exception as exc:                                      # never take the pilot down
                log.warning("%s scheduler: %s", self.name, exc)

    def _launch(self, n):
        import subprocess
        import sys
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
        out = open(self.log_path, "a", encoding="utf-8") if self.log_path else subprocess.DEVNULL
        self._proc = subprocess.Popen([sys.executable, "-m", self.module, "--db", self.db_path, *self.extra_args,
                                       "--memory-cap-mb", str(self.cap)],
                                      stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      creationflags=flags)
        if out is not subprocess.DEVNULL:
            out.close()
        log.info("%s batch started for %d asset(s) (%s, separate process pid %d)",
                 self.name, n, self.module.rsplit(".", 1)[-1], self._proc.pid)


def run_pilot(channel: str = "elheddaf", port: int = 8502, host: str = "127.0.0.1", db_path=None,
              history_hours: float = 4.0, gc_every: float = 1800.0, transcribe: bool = True,
              transcribe_model: str = "base", transcribe_memory_cap_mb: float = 480.0, packshots: bool = True,
              ocr_memory_cap_mb: float = 300.0, reference_matcher: bool = True, frames: bool = True) -> int:
    import uvicorn

    from kihc.analytics import visual_ocr
    from kihc.app.api_server import create_app
    from kihc.core.discovery import backfill_nominal_durations, dedupe_assets
    from kihc.stream.packshot import PackshotGrabber
    db_path = str(db_path or DEFAULT_DB_PATH)
    configure_logging(service_paths(db_path)["log"])
    backfill_nominal_durations(db_path)

    def startup_dedupe():                             # background: never delays ingest start
        try:
            for m in dedupe_assets(db_path)["merged"]:
                log.info("de-duplication: %s merged into %s (%d same airing(s), %d added)",
                         m["duplicate"], m["master"], m["same_airings"], m["airings_added"])
            visual_ocr.mark_visual_needed(db_path)
        except Exception as exc:
            log.warning("startup de-duplication skipped: %s", exc)
    threading.Thread(target=startup_dedupe, name="kihc-dedupe", daemon=True).start()
    grabber = None
    web_feed = (any(c.casefold() in ELHEDDAF_ALIASES for c in split_channels(channel))
                and not iptv_manager.override_url("el_heddaf_tv"))
    if packshots and web_feed:   # only feed with a verified video rendition
        grabber = PackshotGrabber(db_path, heddaf.AUDIO_HLS, heddaf.VIDEO_HLS, DISCOVERED_DIR).start()
    frame_ingest = None
    if frames and web_feed:      # live 2 fps visual cues for pod boundaries
        frame_ingest = heddaf.FrameIngest(heddaf.CHANNEL_NAME).start()
        log.info("video boundary sensing ON: %dx%d @ %d fps, ring depth %d", heddaf.FRAME_W, heddaf.FRAME_H,
                 heddaf.FRAME_FPS, heddaf.RING_DEPTH)
    reference = None
    if reference_matcher:
        from kihc.analytics.reference_matcher import ReferenceMatcher, make_packshot_frame_grabber
        reference = ReferenceMatcher(db_path, frame_grabber=make_packshot_frame_grabber(grabber, db_path) if grabber else None,
                                    evidence_dir=EVIDENCE_DIR)       # confirmed-airing clips are persisted here
        log.info("reference matcher ON: %d signature(s), %.2f MB index", len(reference.index),
                 reference.index.memory_bytes() / 2 ** 20)
    sup, spec = build_supervisor(channel, db_path, history_hours=history_hours, gc_every=gc_every, packshot=grabber,
                                 reference=reference)
    if frame_ingest is not None:
        sup.frame_ingests[heddaf.CHANNEL_NAME] = frame_ingest     # video telemetry rides on the channel status
    try:
        sup.start()                                   # takes the single-instance lock: fail fast
    except ServiceAlreadyRunning as exc:
        log.error("%s - stop it first (python -m kihc.cli stop-service) or use another --db", exc)
        return 3

    server_box = {}

    def save_history():
        for name, engine in getattr(sup, "discovery_engines", {}).items():
            try:
                if engine.save_state(history_state_path(db_path, name)):
                    log.info("discovery history saved (%.0f min of %s)", engine._fp.size / 86.13 / 60, name)
            except Exception as exc:
                log.warning("discovery history not saved: %s", exc)

    def supervise():
        try:
            while not sup.stopping:
                try:
                    sup.tick()                            # also honours `kihc.cli stop-service` (stop file)
                except Exception as exc:                  # supervision must never die
                    log.warning("supervisor tick failed: %s", exc, exc_info=True)
                sup._stop.wait(sup.tick_seconds)
        finally:
            sup.shutdown()
            save_history()                            # stop-service / stop file path
            if "server" in server_box:                    # ingest stopped -> stop the web server too
                server_box["server"].should_exit = True

    thread = threading.Thread(target=supervise, name="kihc-pilot-supervisor", daemon=True)

    schedulers = []
    logs_dir = service_paths(db_path)["log"].parent
    if transcribe:
        schedulers.append(TranscriptionScheduler(db_path, transcribe_model, transcribe_memory_cap_mb,
                                                 log_path=logs_dir / "transcriber.log").start())
    if grabber is not None:
        schedulers.append(TranscriptionScheduler(db_path, memory_cap_mb=ocr_memory_cap_mb, interval=60.0,
                                                 log_path=logs_dir / "visual_ocr.log", module="kihc.analytics.visual_ocr",
                                                 pending_fn=visual_ocr.pending, extra_args=[], name="packshot OCR").start())

    alerts_stop = threading.Event()

    def alert_loop():                                 # near-real-time compliance alerts (<= 60 s after detection)
        from kihc.analytics.compliance_audit import refresh_alerts
        while not alerts_stop.wait(60):
            try:
                n = refresh_alerts(db_path)
                if n:
                    log.info("compliance: %d new alert(s)", n)
            except Exception as exc:                  # never take the pilot down
                log.warning("compliance alerts: %s", exc)
    threading.Thread(target=alert_loop, name="kihc-compliance-alerts", daemon=True).start()

    def resolve_loop():                               # zero-input brand resolution of new assets (lookups only, no model)
        from kihc.analytics.auto_resolver import resolve_pending
        while not alerts_stop.wait(300):
            try:
                r = resolve_pending(db_path, backup=False)
                if r["promoted"]:
                    log.info("brand resolver: %d asset(s) confirmed (%s)", r["promoted"],
                             ", ".join(sorted({d["brand"] for d in r["decisions"] if d["promoted"]})))
            except Exception as exc:                  # never take the pilot down
                log.warning("brand resolver: %s", exc)
    threading.Thread(target=resolve_loop, name="kihc-brand-resolver", daemon=True).start()

    def stop_ingest():
        alerts_stop.set()
        for s in schedulers:
            s.stop()
        if grabber is not None:
            grabber.stop()
        if frame_ingest is not None:
            frame_ingest.stop()
        save_history()
        sup.request_stop("web server stopped")
        thread.join(30)

    app = create_app(db_path, on_shutdown=stop_ingest)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False))
    server_box["server"] = server
    thread.start()
    url = f"http://{host}:{port}"
    log.info("KIHC pilot: monitoring %s (%s) | dashboard %s | db %s | pid %d | RAM %.0f MB",
             spec.name, spec.feed_label or "stream", url, db_path, os.getpid(), memory_mb()[0])
    print(f"\n  KIHC pilot running  ->  {url}\n  Stop: Ctrl+C, or `python -m kihc.cli stop-service` from any console.\n",
          flush=True)
    try:
        server.run()
    finally:
        if thread.is_alive():
            stop_ingest()
    return 0
