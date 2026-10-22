"""Phase 3.5: headless 24/7 service - supervisor, heartbeat, restarts, signals, scripts."""
import signal
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from kihc.core.ingest import ingest_reference
from kihc.core.proof import verify_row
from kihc.database.schema import init_db
from kihc.service import headless_runner as hr
from kihc.service.headless_runner import (ChannelSpec, ServiceAlreadyRunning,
                                          Supervisor, install_signal_handlers,
                                          read_service_status,
                                          resolve_channel_arg)
from kihc.service.lock import ProcessLock, is_locked, service_paths
from kihc.tests.generate_test_assets import generate

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def _wait(pred, timeout, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return False


# ------------------------------------------------------------------ fakes for deterministic supervisor tests

class FakeWorker:
    instances = []

    def __init__(self, spec):
        self.spec, self.state, self.last_error = spec, "idle", ""
        self.alive, self.stopped, self.since = False, False, 0.5
        FakeWorker.instances.append(self)

    def start(self):
        self.alive, self.state = True, "listening"
        return self

    def stop(self):
        self.alive, self.stopped, self.state = False, True, "stopped"

    def is_alive(self):
        return self.alive

    def status(self):
        return dict(channel=self.spec.name, state=self.state, seconds_since_data=self.since, ffmpeg_pid=None,
                    level_dbfs=-20.0, level_history=[(time.time(), -20.0)] * 3, received_seconds=1.0,
                    detections=0, last_error=self.last_error, buffered_seconds=30.0, last_audio_epoch=time.time())


@pytest.fixture
def fake_sup(tmp_path):
    FakeWorker.instances = []
    sup = Supervisor([ChannelSpec("TV A", "x"), ChannelSpec("TV B", "y")], tmp_path / "svc.db",
                     tick=0.05, freeze_after=20, gc_every=3600, worker_factory=FakeWorker)
    yield sup
    if sup.lock.held:
        sup.shutdown()


def test_heartbeat_and_channel_telemetry(fake_sup):
    fake_sup.start()
    fake_sup.tick()
    st = read_service_status(fake_sup.db_path)
    assert st["active"] and st["lock_held"] and st["state"] == "running"
    assert st["channels"] == 2 and st["rss_mb"] > 0 and st["heartbeat_age_s"] < 5
    assert {c["name"] for c in st["channel_status"]} == {"TV A", "TV B"}
    assert all(c["level_dbfs"] == -20.0 for c in st["channel_status"])


def test_frozen_stream_is_restarted(fake_sup):
    fake_sup.start()
    frozen = fake_sup.workers["TV A"]
    frozen.since = 25.0                                    # > freeze_after while 'listening'
    fake_sup.tick()
    assert fake_sup.restarts == {"TV A": 1, "TV B": 0}
    assert _wait(lambda: frozen.stopped, 2) and fake_sup.workers["TV A"] is not frozen
    assert "frozen" in fake_sup.restart_log[-1][2]


def test_silent_but_flowing_stream_is_not_restarted(fake_sup):
    fake_sup.start()
    fake_sup.workers["TV A"].since = 0.2                   # data flowing (level may be dead air)
    for _ in range(3):
        fake_sup.tick()
    assert sum(fake_sup.restarts.values()) == 0


def test_stuck_connecting_is_restarted(fake_sup):
    fake_sup.connect_timeout = 0.2
    fake_sup.start()
    stuck = fake_sup.workers["TV A"]
    stuck.state, stuck.since = "connecting", None
    fake_sup.tick()                                         # first sighting: timer starts
    assert fake_sup.restarts["TV A"] == 0
    time.sleep(0.3)
    fake_sup.tick()
    assert fake_sup.restarts["TV A"] == 1 and "stuck connecting" in fake_sup.restart_log[-1][2]
    assert fake_sup.restarts["TV B"] == 0


def test_dead_worker_is_restarted(fake_sup):
    fake_sup.start()
    fake_sup.workers["TV B"].alive = False
    fake_sup.tick()
    assert fake_sup.restarts["TV B"] == 1 and fake_sup.workers["TV B"].is_alive()


def test_gc_collect_runs_on_schedule(fake_sup, monkeypatch):
    calls = []
    monkeypatch.setattr(hr.gc, "collect", lambda: calls.append(1) or 0)
    fake_sup.gc_every = 0.0
    fake_sup.start()
    fake_sup.tick()
    fake_sup.tick()
    assert len(calls) == 2 and fake_sup.gc_runs == 2 and fake_sup.last_gc_at


def test_stop_file_shuts_down_gracefully(fake_sup):
    t = threading.Thread(target=fake_sup.run)
    t.start()
    assert _wait(lambda: read_service_status(fake_sup.db_path) is not None, 5)
    service_paths(fake_sup.db_path)["stop"].write_text("stop")
    t.join(10)
    assert not t.is_alive()
    st = read_service_status(fake_sup.db_path)
    assert st["state"] == "stopped" and not st["active"] and not st["lock_held"]
    assert "stop-request file" in st["message"]
    assert all(w.stopped for w in FakeWorker.instances)
    assert not service_paths(fake_sup.db_path)["stop"].exists()


def test_signal_handler_requests_graceful_stop(fake_sup):
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        install_signal_handlers(fake_sup)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert fake_sup.stopping and fake_sup.stop_reason == "SIGTERM"
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


def test_single_instance_lock(fake_sup, tmp_path):
    fake_sup.start()
    fake_sup.tick()
    other = Supervisor([ChannelSpec("TV C", "z")], fake_sup.db_path, worker_factory=FakeWorker)
    with pytest.raises(ServiceAlreadyRunning):
        other.start()
    st = read_service_status(fake_sup.db_path)
    assert st["pid"] and st["channels"] == 2                   # the loser did not overwrite the heartbeat


def test_lock_is_released_when_holder_dies(tmp_path):
    lock = tmp_path / "x.lock"
    code = (f"from kihc.service.lock import ProcessLock; import os; "
            f"assert ProcessLock(r'{lock}').acquire(); os._exit(9)")   # dies without releasing
    subprocess.run([PY, "-c", code], cwd=ROOT, check=False)
    assert not is_locked(lock)
    assert ProcessLock(lock).acquire()


def test_startup_failure_releases_lock(tmp_path):
    def boom(spec):
        raise RuntimeError("bad channel")
    sup = Supervisor([ChannelSpec("Bad", "x")], tmp_path / "b.db", worker_factory=boom)
    with pytest.raises(RuntimeError):
        sup.start()
    assert not sup.lock.held and not is_locked(sup.paths["lock"])


def test_channel_argument_resolution(monkeypatch):
    from kihc.connectors import iptv_manager
    monkeypatch.setattr(iptv_manager, "fetch_channels", iptv_manager.fallback_channels)
    spec = resolve_channel_arg("el_heddaf")
    assert spec.name == "El-Heddaf TV" and spec.key == "el_heddaf_tv" and callable(spec.source)
    assert resolve_channel_arg("Echorouk TV").key == "echorouk_tv"
    direct = resolve_channel_arg("My TV=https://example.com/live/index.m3u8")
    assert (direct.name, direct.source) == ("My TV", "https://example.com/live/index.m3u8")
    assert resolve_channel_arg("tv2").name == "TV2 (Algeria)"              # IPTV list slug
    with pytest.raises(ValueError):
        resolve_channel_arg("el")                                         # ambiguous: el_bilad / el_heddaf
    with pytest.raises(ValueError):
        resolve_channel_arg("no_such_channel")


# ------------------------------------------------------------------ real workers

@pytest.fixture(scope="module")
def assets():
    return generate()


def test_real_supervisor_detects_and_flushes_on_shutdown(assets, tmp_path):
    ad_a, _, broadcast = assets
    db = tmp_path / "real.db"
    ingest_reference(ad_a, "Acme", "Ad A", db_path=db)
    sup = Supervisor([ChannelSpec("SynthTV", str(broadcast))], db, tick=0.2, evidence_dir=tmp_path / "ev",
                     worker_kwargs=dict(backoff_base=0.1, backoff_max=0.2))
    t = threading.Thread(target=sup.run)
    t.start()
    try:
        seen = {}

        def detected():           # a file source ends in seconds; the Phase 29.2 contract then reports 'stalled' (no level), so
            st = read_service_status(db)   # capture the snapshot in which detection and a live level coincide
            ok = st and st["channel_status"] and st["channel_status"][0]["detections"] >= 1                 and st["channel_status"][0]["level_dbfs"] is not None
            if ok:
                seen["st"] = st
            return ok
        assert _wait(detected, 30)
        st = seen["st"]
        ch = st["channel_status"][0]
        print(f"\n[SERVICE] heartbeat age {st['heartbeat_age_s']}s rss {st['rss_mb']} MB "
              f"ffmpeg {st['ffmpeg_rss_mb']} MB · {ch['name']} level {ch['level_dbfs']} dBFS")
        assert st["active"] and ch["level_dbfs"] is not None
    finally:
        sup.request_stop("test")
        t.join(30)
    conn = init_db(db)
    rows = conn.execute("SELECT * FROM ad_occurrences").fetchall()
    conn.close()
    assert rows and all(r["source_name"] == "live:SynthTV" and verify_row(r) is True for r in rows)
    assert read_service_status(db)["state"] == "stopped"


class _StallingStream:
    """HTTP 'live stream' that sends 8 s of WAV audio, then goes quiet without closing (a frozen pipe).

    8 s, not 1 s: FFmpeg probes ~5 s of input before emitting audio, so less would stall mid-probe
    (that case is covered by the 'stuck connecting' test).
    """

    def __init__(self, seconds: int = 8):
        self.release = threading.Event()
        self.connections = 0
        sr = 22050
        pcm = (np.sin(2 * np.pi * 440 * np.arange(sr * seconds) / sr) * 8000).astype("<i2").tobytes()
        header = (b"RIFF" + struct.pack("<I", 0x7FFFFFF0) + b"WAVEfmt " +
                  struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" + struct.pack("<I", 0x7FFFFFC0))
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.connections += 1
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                try:
                    self.wfile.write(header + pcm)
                    self.wfile.flush()
                    outer.release.wait(60)               # stall: connection open, no more data
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/live.wav"

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def test_real_frozen_pipe_is_restarted(tmp_path):
    stream = _StallingStream()
    # Worker's own stall watchdog disabled (999 s) so the supervisor is what must recover the frozen pipe.
    sup = Supervisor([ChannelSpec("FrozenTV", stream.url)], tmp_path / "f.db", tick=0.2, freeze_after=2.0,
                     worker_kwargs=dict(stall_timeout=999))
    t = threading.Thread(target=sup.run)
    t.start()
    try:
        assert _wait(lambda: sup.restarts["FrozenTV"] >= 1, 20), sup.restart_log
        assert _wait(lambda: stream.connections >= 2, 10)            # a fresh pipe reconnected
        reason = sup.restart_log[0][2]
        print(f"\n[FROZEN] restarts={sup.restarts['FrozenTV']} reason={reason!r} connections={stream.connections}")
        assert reason.startswith("frozen: no audio data for")
    finally:
        sup.request_stop("test")
        stream.close()
        t.join(30)
    assert not t.is_alive()


def _spawn_service(db, *channels, tick="0.5"):
    args = [PY, "-m", "kihc.cli", "run-service", "--db", str(db), "--tick", tick]
    for c in channels:
        args += ["--channel", c]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    return subprocess.Popen(args, cwd=ROOT, creationflags=flags, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def test_subprocess_graceful_signal_and_memory_budget(assets, tmp_path):
    ad_a, ad_b, broadcast = assets
    db = tmp_path / "proc.db"
    ingest_reference(ad_a, "Acme", "Ad A", db_path=db)
    ingest_reference(ad_b, "Decoy Co", "Ad B", db_path=db)
    proc = _spawn_service(db, f"SynthOne={broadcast}", f"SynthTwo={broadcast}")
    try:
        assert _wait(lambda: (st := read_service_status(db)) and st["active"]
                     and len(st["channel_status"]) == 2 and st["ffmpeg_rss_mb"] > 0, 30)
        peak = 0.0
        for _ in range(10):                                         # sample while both pipes run
            st = read_service_status(db)
            peak = max(peak, st["rss_mb"] + st["ffmpeg_rss_mb"])
            time.sleep(0.3)
        print(f"\n[SERVICE RAM] python {st['rss_mb']} MB (peak {st['peak_rss_mb']}) + ffmpeg "
              f"{st['ffmpeg_rss_mb']} MB; max sampled total {peak:.1f} MB (budget 250)")
        assert peak < 250 and st["peak_rss_mb"] < 250
        proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
        assert proc.wait(30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    st = read_service_status(db)
    expected = "SIGBREAK" if sys.platform == "win32" else "SIGTERM"
    assert st["state"] == "stopped" and expected in st["message"] and not st["lock_held"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch scripts")
def test_batch_scripts_start_and_stop(assets, tmp_path):
    _, _, broadcast = assets
    db = tmp_path / "bat.db"
    out = tmp_path / "start.log"
    # Not a pipe: `start` hands the script's handles to the long-lived service, so a pipe would never close.
    with open(out, "w") as fh:
        rc = subprocess.run(["cmd", "/c", str(ROOT / "scripts" / "start_monitoring_service.bat"),
                             "--db", str(db), "--tick", "0.5", "--channel", f"BatTV={broadcast}"],
                            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, timeout=30).returncode
    assert rc == 0 and "launched" in out.read_text()
    try:
        assert _wait(lambda: (st := read_service_status(db)) and st["active"], 30)
    finally:
        stop = subprocess.run(["cmd", "/c", str(ROOT / "scripts" / "stop_monitoring_service.bat"),
                               "--db", str(db)], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert "Service stopped" in stop.stdout, stop.stdout + stop.stderr
    assert read_service_status(db)["state"] == "stopped"
