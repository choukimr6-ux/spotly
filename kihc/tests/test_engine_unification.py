"""Phase 29: engine unification, telemetry truth, incremental DSP and persistent sealed evidence."""
import json
import sqlite3
import sys
import threading
import time
import wave

import numpy as np
import pytest

from kihc.config import HOP_SIZE, LIVE_SAMPLE_RATE
from kihc.core import incremental_dsp as idsp
from kihc.core.fingerprint import StreamingFingerprinter

SR22 = LIVE_SAMPLE_RATE


# ------------------------------------------------------------------------------------------ incremental DSP parity
def _signal(seconds=23.0, seed=5):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR22)
    t = np.arange(n) / SR22
    x = 0.2 * np.sin(2 * np.pi * (300 + 40 * t) * t) + 0.05 * rng.standard_normal(n)
    return x.astype(np.float32)


def _full_window(x, g_start=0):
    """What the pre-Phase-29 worker did on one pass: FIR-decimate the whole window, align to the frame grid, fingerprint."""
    y = np.convolve(x, idsp.design_taps(), mode="same")[::idsp.DECIM]
    s0 = g_start // idsp.DECIM
    s_aligned = -(-s0 // HOP_SIZE) * HOP_SIZE
    y = y[s_aligned - s0:]
    return y, StreamingFingerprinter().feed(y), s_aligned // HOP_SIZE


def _incremental(x, pieces, g_start=0, normalize=False):
    s = idsp.IncrementalFingerprintStream(g_start, normalize=normalize)
    deltas, i = [], 0
    for n in pieces:
        deltas.append(s.push(x[i:i + n]))
        i += n
    deltas.append(s.push(x[i:]))
    deltas.append(s.flush())
    return s, deltas


@pytest.mark.parametrize("pieces", [[4410] * 30, [1, 7, 4409, 13000, 99, 22050], [100000], [3, 3, 3]])
def test_incremental_decimation_and_fingerprints_equal_the_full_window_rerun(pieces):
    x = _signal()
    y_full, fp_full, base_full = _full_window(x)
    s, deltas = _incremental(x, pieces)
    y_inc = np.concatenate([d.x11 for d in deltas])
    fp_inc = np.concatenate([d.fp for d in deltas])
    assert y_inc.size == y_full.size
    assert np.allclose(y_inc, y_full, atol=1e-6)
    # sub-fingerprints: identical bit for bit (a float rounding tie may flip a bit when the filter order differs)
    assert fp_inc.size == fp_full.size
    flipped = np.bitwise_count(fp_inc ^ fp_full).sum()
    assert flipped <= max(2, 1e-4 * fp_full.size * 32)
    assert deltas[0].base == base_full
    # deltas are contiguous: each starts exactly where the previous one ended
    nxt = base_full
    for d in deltas:
        if d.fp.size:
            assert d.base == nxt
            nxt += d.fp.size
    assert s.frame_end == base_full + fp_full.size


def test_incremental_parity_with_an_odd_global_start_and_unaligned_stream():
    x = _signal(14.0)
    g0 = 22050 * 91 + 1357                     # a connection that starts mid-stream, on an odd global sample
    s, deltas = _incremental(x, [4410] * 20, g_start=g0)
    y_ref, fp_ref, base_ref = _full_window(x[g0 % 2:], g_start=g0 + g0 % 2)
    assert deltas[0].base == base_ref and deltas[0].s_start == base_ref * HOP_SIZE
    y_inc = np.concatenate([d.x11 for d in deltas])
    assert np.allclose(y_inc, y_ref, atol=1e-6)
    assert np.concatenate([d.fp for d in deltas]).size == fp_ref.size


def test_with_the_normaliser_on_the_fingerprints_stay_within_a_hair_of_the_full_window():
    # a -10 dB level step every 3 s (the normaliser re-gains each piece): only frames straddling a step may differ
    x = _signal() * np.where(np.arange(_signal().size) % (SR22 * 6) < SR22 * 3, 1.0, 0.3).astype(np.float32)
    _, fp_full, _ = _full_window(x)
    _, deltas = _incremental(x, [4410 * 15] * 5, normalize=True)
    fp_inc = np.concatenate([d.fp for d in deltas])
    ber = np.bitwise_count(fp_inc[:fp_full.size] ^ fp_full).sum() / (fp_full.size * 32)
    assert fp_inc.size == fp_full.size and ber < 0.02


def test_each_sample_is_filtered_once_work_is_linear_in_new_audio():
    """The decimator never touches more than (new samples + one filter length) per push: no re-convolution of the history."""
    x = _signal(30.0)
    dec = idsp.IncrementalDecimator(idsp.design_taps())
    calls = []
    real = np.lib.stride_tricks.sliding_window_view

    def spy(a, w, *args, **kw):
        calls.append(a.size)
        return real(a, w, *args, **kw)

    np.lib.stride_tricks.sliding_window_view = spy
    try:
        for i in range(0, x.size, 4410):
            dec.push(x[i:i + 4410])
    finally:
        np.lib.stride_tricks.sliding_window_view = real
    assert max(calls) <= 4410 + 2 * dec.half + 1               # a 0.2 s piece plus its filter context, never the 30 s ring
    assert dec._buf.size <= 2 * dec.half + 4410


def test_a_huge_backlog_push_keeps_transient_memory_bounded():
    """The first pass after a reconnect may hold 30 s+ of audio: the stream must still allocate in small blocks."""
    import tracemalloc
    x = _signal(60.0)
    s = idsp.IncrementalFingerprintStream(0, normalize=False)
    tracemalloc.start()
    s.push(x)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 30 * 2 ** 20


# ------------------------------------------------------------------------------------------ telemetry truth
from kihc.database.schema import init_db                                             # noqa: E402
from kihc.service import headless_runner as hr                                       # noqa: E402
from kihc.service.lock import ProcessLock, service_paths                             # noqa: E402


def _status_json(level=-20.0, buffered=30.0, audio_age=0.5):
    return json.dumps(dict(state="listening", level_dbfs=level, last_audio_epoch=time.time() - audio_age, peak_dbfs_10s=level + 4, silent=False,
                           connected_for_s=321.0, disconnects_last_hour=0, buffered_seconds=buffered, received_seconds=321.0,
                           detections=2, references=13, connect_attempts=1, next_retry_at=None, last_error="",
                           level_history=[[time.time(), level]] * 5, feed_label="Dedicated IPTV feed"))


def _seed_engine(db, rows, pid=None):
    """A running engine (heartbeat now, lock held by this test) plus raw channel rows: (name, age_s, owner_pid)."""
    pid = pid or __import__("os").getpid()
    now = time.time()
    conn = init_db(db)
    try:
        with conn:
            conn.execute("INSERT OR REPLACE INTO service_heartbeat (service_id, pid, hostname, state, started_at, heartbeat_at, "
                         "heartbeat_epoch, tick_seconds, rss_mb, peak_rss_mb, ffmpeg_rss_mb, channels, restarts, gc_runs, message) "
                         "VALUES (1, ?, 'h', 'running', 'x', 'x', ?, 5.0, 100, 110, 10, ?, 0, 0, '')", (pid, now, len(rows)))
            conn.execute("DELETE FROM service_channel_status")
            for name, age, owner in rows:
                conn.execute("INSERT INTO service_channel_status (channel, state, updated_epoch, restarts, status_json, owner_pid) "
                             "VALUES (?, 'listening', ?, 0, ?, ?)", (name, now - age, _status_json(), owner))
    finally:
        conn.close()


@pytest.fixture
def engine_db(tmp_path):
    db = tmp_path / "t.db"
    lock = ProcessLock(service_paths(db)["lock"])
    assert lock.acquire()
    yield db
    lock.release()


def test_stale_and_orphan_channel_rows_never_reach_the_live_fleet(engine_db):
    me = __import__("os").getpid()
    _seed_engine(engine_db, [("Fresh TV", 2.0, me), ("Slow TV", 20.0, me), ("Dead TV", 120.0, me),
                             ("Ghost TV", 1.0, me + 100000),          # written by a different (dead) process
                             ("Legacy Ghost", 8000.0, None)])         # pre-Phase-29 row nobody refreshed for 2 h
    st = hr.read_service_status(engine_db)
    by = {c["name"]: c for c in st["channel_status"]}
    assert set(by) == {"Fresh TV", "Slow TV", "Dead TV"}                # Ghost TV / Legacy Ghost are not served at all
    assert st["orphan_rows_ignored"] == 2
    assert by["Fresh TV"]["link"] == "live" and by["Fresh TV"]["state"] == "listening" and by["Fresh TV"]["level_dbfs"] == -20.0
    assert by["Slow TV"]["link"] == "stale" and by["Slow TV"]["state"] == "listening"          # shown, flagged
    dead = by["Dead TV"]
    assert dead["link"] == "offline" and dead["state"] == hr.OFFLINE_LABEL and dead["offline"] is True
    assert dead["level_dbfs"] is None and dead["connected_for_s"] is None and dead["level_history"] == []
    assert (st["live_channels"], st["offline_channels"]) == (2, 1)


def test_telemetry_endpoint_labels_offline_channels_and_drops_orphans(engine_db):
    from fastapi.testclient import TestClient
    from kihc.app.api_server import create_app
    me = __import__("os").getpid()
    _seed_engine(engine_db, [("Fresh TV", 1.0, me), ("Dead TV", 300.0, me), ("Ghost TV", 1.0, me + 100000)])
    with TestClient(create_app(engine_db)) as client:
        body = client.get("/api/telemetry").json()
    chans = {c["name"]: c for c in body["channels"]}
    assert set(chans) == {"Fresh TV", "Dead TV"}
    assert chans["Fresh TV"]["state"] == "listening" and chans["Fresh TV"]["link"] == "live" and not chans["Fresh TV"]["offline"]
    assert chans["Dead TV"]["state"] == "OFFLINE / DISCONNECTED" and chans["Dead TV"]["offline"] is True
    assert chans["Dead TV"]["level_dbfs"] is None and chans["Dead TV"]["level_history"] == []
    assert body["engine"]["live_channels"] == 1 and body["engine"]["offline_channels"] == 1


def test_a_retrying_channel_with_a_fresh_row_is_stalled_not_live(engine_db):
    """Phase 29.2 contract: link=live needs buffered audio AND a chunk ingested < 15 s ago; a row that merely refreshes is not proof."""
    import sqlite3
    me = __import__("os").getpid()
    _seed_engine(engine_db, [("Flowing TV", 1.0, me), ("Retry TV", 1.0, me), ("Dry TV", 1.0, me), ("Late TV", 1.0, me)])
    rows = {"Retry TV": _status_json(buffered=0.0, audio_age=400.0),        # retry loop: nothing buffered, last chunk long ago
            "Dry TV": _status_json(buffered=30.0, audio_age=400.0),         # ring not yet drained but no chunk for 400 s
            "Late TV": _status_json(buffered=30.0, audio_age=16.0)}         # just past the 15 s freshness window
    conn = sqlite3.connect(engine_db)
    with conn:
        for name, js in rows.items():
            conn.execute("UPDATE service_channel_status SET status_json = ? WHERE channel = ?", (js, name))
    conn.close()
    st = hr.read_service_status(engine_db)
    by = {c["name"]: c for c in st["channel_status"]}
    assert by["Flowing TV"]["link"] == "live" and by["Flowing TV"]["state"] == "listening"
    for n in ("Retry TV", "Dry TV", "Late TV"):
        assert by[n]["link"] == "stalled" and by[n]["state"] == "connecting" and by[n]["level_dbfs"] is None
        assert by[n]["offline"] is False and by[n]["audio_age_s"] is not None
    assert (st["live_channels"], st["offline_channels"]) == (1, 3)


def test_a_dead_engines_channels_are_not_reported_at_all(tmp_path):
    db = tmp_path / "d.db"
    _seed_engine(db, [("Echorouk TV", 2.0, __import__("os").getpid())])      # but nobody holds the service lock
    st = hr.read_service_status(db)
    assert st["active"] is False
    from fastapi.testclient import TestClient
    from kihc.app.api_server import create_app
    with TestClient(create_app(db)) as client:
        assert client.get("/api/telemetry").json()["channels"] == []


class _IdleWorker:
    state, last_error = "listening", ""

    def __init__(self, spec):
        self.spec = spec

    def start(self):
        return self

    def stop(self, *a):
        pass

    def is_alive(self):
        return True

    def status(self):
        return dict(state="listening", seconds_since_data=0.1, ffmpeg_pid=None, level_dbfs=-20.0, level_history=[],
                    buffered_seconds=30.0, last_audio_epoch=time.time())


def test_supervisor_startup_purges_rows_a_previous_process_left_behind(tmp_path):
    db = tmp_path / "s.db"
    _seed_engine(db, [("Echorouk TV", 8000.0, 4242), ("Ennahar TV", 8000.0, None)], pid=4242)
    sup = hr.Supervisor([hr.ChannelSpec("Only TV", "x")], db, worker_factory=_IdleWorker)
    sup.start()
    try:
        conn = init_db(db)
        names = {r[0] for r in conn.execute("SELECT channel FROM service_channel_status")}
        conn.close()
        assert names == set(), "phantom rows survived the startup purge"
        sup.tick()
        st = hr.read_service_status(db)
        assert {c["name"] for c in st["channel_status"]} == {"Only TV"}
        assert st["channel_status"][0]["link"] == "live"
    finally:
        sup.request_stop("test")
        sup.shutdown()


# ------------------------------------------------------------------------------------------ persistent sealed evidence
import hashlib                                                                        # noqa: E402

from kihc.analytics import evidence_recorder as er                                   # noqa: E402
from kihc.config import EVIDENCE_DIR                                                  # noqa: E402
from kihc.core.ingest import ingest_reference                                         # noqa: E402
from kihc.core.live_worker import LiveStreamWorker                                    # noqa: E402
from kihc.core.proof import verify_row                                                # noqa: E402
from kihc.tests.generate_test_assets import generate                                  # noqa: E402

CH = "Evidence TV"
T0 = 1_790_000_000.0


@pytest.fixture
def recorder(tmp_path):
    er.reset()
    yield tmp_path / "ev"
    er.reset()


def _feed(channel, t_from, seconds, amp=8000):
    n = int(seconds * SR22)
    x = (amp * np.sin(2 * np.pi * 600 * np.arange(n) / SR22)).astype(np.int16)
    for i in range(0, n, 4410):
        er.feed(channel, t_from + i / SR22, x[i:i + 4410])


def test_confirmed_airing_clip_is_written_hashed_and_immutable(recorder):
    _feed(CH, T0, 40.0)
    a = er.persist_airing_clip(CH, T0 + 10, T0 + 25, key="ad7", evidence_dir=recorder)
    assert a and a["created"] and a["complete"]
    path = recorder / a["path"] if not a["path"].startswith(str(recorder)) else __import__("pathlib").Path(a["path"])
    assert path.is_file() and path.parent.name == er.AIRING_DIR
    with wave.open(str(path), "rb") as w:                                       # the 11 025 Hz clip the spec asks for
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 11025)
        assert w.getnframes() / 11025 == pytest.approx(15 + 2 * er.PAD_SECONDS, abs=0.01)
    assert a["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    before = path.read_bytes()
    b = er.persist_airing_clip(CH, T0 + 10, T0 + 25, key="ad7", evidence_dir=recorder)
    assert b["created"] is False and b["sha256"] == a["sha256"] and path.read_bytes() == before     # never rewritten


def test_a_clip_is_clipped_to_the_buffered_audio_not_padded_with_silence(recorder):
    _feed(CH, T0, 20.0)                                  # the post-roll has not arrived yet
    a = er.persist_airing_clip(CH, T0 + 10, T0 + 19, key="ad1", evidence_dir=recorder)
    assert a and a["complete"] is False
    assert a["clip_seconds"] == pytest.approx(9 + er.PAD_SECONDS + 1.0, abs=0.01)     # 3 s pre-roll + 9 s + the 1 s that exists
    assert er.persist_airing_clip("Nobody TV", T0 + 10, T0 + 19, key="x", evidence_dir=recorder) is None
    assert er.persist_airing_clip(CH, T0 + 500, T0 + 520, key="gone", evidence_dir=recorder) is None


def test_snapshot_fallback_saves_an_airing_the_live_ring_already_scrolled_past(recorder):
    _feed(CH, T0, 30.0)
    snap = er.snapshot(CH, T0 + 5 - er.PAD_SECONDS, float("inf"))
    _feed(CH, T0 + 30, 200.0)                            # the ring is now far past the airing
    assert er.persist_airing_clip(CH, T0 + 5, T0 + 25, key="old", evidence_dir=recorder) is None
    ev = er.persist_airing_clip(CH, T0 + 5, T0 + 25, key="old", evidence_dir=recorder, fallback=snap)
    assert ev and ev["covered_seconds"] >= 20.0


@pytest.fixture(scope="module")
def assets_and_db(tmp_path_factory):
    ad_a, ad_b, broadcast = generate()
    return ad_a, ad_b, broadcast


def _run_worker(assets, tmp_path, name, **kw):
    ad_a, ad_b, broadcast = assets
    db = tmp_path / f"{name}.db"
    ingest_reference(ad_a, "Acme", "Ad A", db_path=db)
    ingest_reference(ad_b, "Decoy Co", "Ad B", db_path=db)
    w = LiveStreamWorker(name, str(broadcast), db, max_connects=1, **kw).start()
    assert w.wait(120)
    return db, w


def _rows(db):
    conn = init_db(db)
    try:
        return conn.execute("SELECT * FROM ad_occurrences ORDER BY airing_start").fetchall()
    finally:
        conn.close()


@pytest.mark.parametrize("custom_root", [True, False])
def test_live_confirmation_persists_a_clip_and_chains_its_sha256_into_the_seal(assets_and_db, tmp_path, custom_root):
    root = tmp_path / "ev" if custom_root else EVIDENCE_DIR
    db, w = _run_worker(assets_and_db, tmp_path, "EvidLive", evidence_dir=root)
    rows = _rows(db)
    assert rows and w.detections, "the synthetic broadcast must produce a confirmed airing"
    row = rows[0]
    clip = __import__("pathlib").Path(row["evidence_path"])
    clip = clip if clip.is_absolute() else EVIDENCE_DIR / clip
    assert clip.is_file() and clip.suffix == ".wav"
    assert clip.is_relative_to(tmp_path) == custom_root                    # a custom root is stored absolute, the default relative
    digest = hashlib.sha256(clip.read_bytes()).hexdigest()
    assert row["seal_clip_sha256"] == digest and row["evidence_sha256"] == digest
    assert verify_row(row) is True                                        # the seal covers the clip + the spectrogram
    assert row["evidence_frame_path"] and __import__("pathlib").Path(row["evidence_frame_path"]).is_file()
    # tamper with ONE byte of the persisted audio: the airing's verification hash no longer matches
    original = clip.read_bytes()
    try:
        data = bytearray(original)
        data[len(data) // 2] ^= 0xFF
        clip.write_bytes(bytes(data))
        assert verify_row(row) is False
        clip.unlink()
        assert verify_row(row) is False                                   # a deleted clip breaks the seal too
    finally:
        clip.write_bytes(original)
    assert verify_row(row) is True


def test_the_legacy_matcher_writes_through_the_shared_queue_not_a_private_connection(assets_and_db, tmp_path):
    from kihc.data import db_queue as dq
    db, w = _run_worker(assets_and_db, tmp_path, "QueueLive", evidence_dir=tmp_path / "ev")
    q = dq.get_queue(db)
    assert q.flush(10) and q.stats["committed"] >= len(_rows(db)) >= 1
    assert q.stats["failed"] == 0 and q.stats["deadlettered"] == 0


def test_a_duplicate_airing_leaves_no_orphan_clip(assets_and_db, tmp_path):
    from kihc.data import reference_db as rdb
    db, w = _run_worker(assets_and_db, tmp_path, "DupLive", evidence_dir=tmp_path / "ev")
    row = _rows(db)[0]
    clips_before = sorted((tmp_path / "ev" / er.AIRING_DIR).glob("*.wav"))
    assert rdb.commit_airing(db, asset_id="ad:1", ad_id=row["ad_id"], source_name=row["source_name"],
                             airing_start=row["airing_start"] + 0.5, airing_end=row["airing_end"] + 0.5, duration=10.0,
                             confidence=0.99, algeria_timestamp="x", verification_hash="h", detected_at="d") is None
    assert sorted((tmp_path / "ev" / er.AIRING_DIR).glob("*.wav")) == clips_before


# ------------------------------------------------------------------------------------------ engine unification
from kihc.ingest.fleet import FleetIngest, Inbox                                     # noqa: E402

FAKE_FFMPEG = r"""
import sys, time
seg = int(sys.argv[1]); delay = float(sys.argv[2]); out = sys.stdout.buffer
while True:
    out.write(b"\x10\x00" * seg); out.flush(); time.sleep(delay)
"""


def _wait(cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _fake_fleet(delay=0.02):
    fleet = FleetIngest(dict(stall_timeout_sec=3.0, backoff_base_sec=0.1, backoff_max_sec=0.3, unresolved_retry_sec=0.3))
    fleet.manager.command_builder = lambda url, ch: [sys.executable, "-c", FAKE_FFMPEG, "4410", str(delay)]
    return fleet.start()


def test_managed_worker_consumes_the_shared_managers_audio(tmp_path):
    fleet = _fake_fleet()
    try:
        w = LiveStreamWorker("Managed TV", "fake://x", tmp_path / "m.db", evidence_dir=tmp_path, fleet=fleet,
                             channel_id="managed_tv").start()
        assert _wait(lambda: w.status()["state"] == "listening" and w.status()["received_seconds"] > 2.0)
        s = w.status()
        assert s["engine"] == "fleet" and s["ffmpeg_pid"] and s["connect_attempts"] == 1
        assert fleet.manager.status()["managed_tv"]["state"] == "live"          # the manager, not the worker, owns FFmpeg
        pid = s["ffmpeg_pid"]
        w.stop()
        assert _wait(lambda: fleet.stat("managed_tv").pid is None)               # stopping the worker stops the child
        assert "managed_tv" not in fleet.manager.active()
        assert pid not in __import__("psutil").pids() if _has_psutil() else True
    finally:
        fleet.stop()


def _has_psutil():
    try:
        import psutil  # noqa: F401
        return True
    except ImportError:
        return False


def test_a_restarted_worker_takes_the_channel_over_and_the_old_stop_does_not_kill_it(tmp_path):
    fleet = _fake_fleet()
    try:
        old = LiveStreamWorker("Handover TV", "fake://x", tmp_path / "h.db", evidence_dir=tmp_path, fleet=fleet,
                               channel_id="handover_tv").start()
        assert _wait(lambda: old.status()["received_seconds"] > 1.0)
        new = LiveStreamWorker("Handover TV", "fake://x", tmp_path / "h.db", evidence_dir=tmp_path, fleet=fleet,
                               channel_id="handover_tv").start()                 # supervisor restart: new worker, same channel
        threading.Thread(target=old.stop, daemon=True).start()                   # ... and the old one is reaped in the background
        assert _wait(lambda: new.status()["state"] == "listening" and new.status()["received_seconds"] > 1.0)
        before = new.status()["received_seconds"]
        time.sleep(1.0)
        assert new.status()["received_seconds"] > before + 0.5, "the old worker's stop() cancelled the new worker's stream"
        assert "handover_tv" in fleet.manager.active()
        new.stop()
    finally:
        fleet.stop()


def test_a_slow_consumer_gets_back_pressure_not_dropped_audio(tmp_path):
    inbox = Inbox(max_segments=20)
    fleet = _fake_fleet(delay=0.0)                                               # a source that floods as fast as it can
    try:
        fleet.attach("flood", "Flood", "fake://x", inbox, fresh=False)
        assert _wait(lambda: len(inbox) >= 15)
        time.sleep(0.5)
        assert len(inbox) <= 20 and inbox.dropped == 0                           # the reader paused instead of shedding
        seqs = []
        for _ in range(60):
            ev = inbox.get(0.5)
            if ev and ev[0] == "seg":
                seqs.append(ev[1].seq)
        assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))                 # contiguous: nothing was lost
    finally:
        fleet.stop()


def test_managed_and_legacy_engines_agree_on_what_was_detected(assets_and_db, tmp_path):
    ad_a, ad_b, broadcast = assets_and_db
    db_l, wl = _run_worker(assets_and_db, tmp_path, "Parity", evidence_dir=tmp_path / "ev_l")
    fleet = FleetIngest(dict(backoff_base_sec=0.1, backoff_max_sec=0.3)).start()
    try:
        db_m = tmp_path / "ParityManaged.db"
        ingest_reference(ad_a, "Acme", "Ad A", db_path=db_m)
        ingest_reference(ad_b, "Decoy Co", "Ad B", db_path=db_m)
        wm = LiveStreamWorker("ParityManaged", str(broadcast), db_m, max_connects=1, evidence_dir=tmp_path / "ev_m",
                              fleet=fleet, channel_id="parity_managed").start()
        assert wm.wait(180), wm.status()
    finally:
        fleet.stop()
    legacy, managed = _rows(db_l), _rows(db_m)
    assert managed and len(managed) == len(legacy)
    for a, b in zip(legacy, managed):
        assert a["ad_id"] == b["ad_id"] and a["status"] == b["status"] == "CONFIRMED"
        assert abs((a["airing_end"] - a["airing_start"]) - (b["airing_end"] - b["airing_start"])) < 0.3
        assert abs(a["confidence_score"] - b["confidence_score"]) < 0.03
    offs_l = [d.stream_offset_sec for d in wl.detections]
    offs_m = [d.stream_offset_sec for d in wm.detections]
    assert all(abs(x - y) < 1.0 for x, y in zip(offs_l, offs_m)), (offs_l, offs_m)
    assert all(verify_row(r) is True for r in managed)
    assert wm.status()["engine"] == "fleet" and wl.status()["engine"] == "legacy"
    # the capture timeline survives the manager: every airing has its sealed clip, and the clip really holds the airing
    for rows, root in ((legacy, tmp_path / "ev_l"), (managed, tmp_path / "ev_m")):
        for r in rows:
            assert r["evidence_path"] and r["seal_clip_sha256"] == hashlib.sha256(
                __import__("pathlib").Path(r["evidence_path"]).read_bytes()).hexdigest()
            with wave.open(r["evidence_path"], "rb") as w:
                assert w.getnframes() / w.getframerate() >= 0.9 * (r["airing_end"] - r["airing_start"])
    assert [round(r["airing_start"] - managed[0]["airing_start"], 0) for r in managed] ==         [round(r["airing_start"] - legacy[0]["airing_start"], 0) for r in legacy]


def test_pilot_channel_list_syntax():
    from kihc.service.pilot import split_channels
    assert split_channels("a,b, c,a") == ["a", "b", "c"]
    assert split_channels("Canal=http://h/x.m3u8?a=1,2") == ["Canal=http://h/x.m3u8?a=1,2"]          # one 'Name=URL'
    assert split_channels("Canal=http://h/x.m3u8?a=1,2;elheddaf") == ["Canal=http://h/x.m3u8?a=1,2", "elheddaf"]
