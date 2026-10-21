"""Phase 11: DB hardening, El-Heddaf direct feed, unsupervised discovery, pilot runner, discovery APIs."""
import logging
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from kihc.analytics.financial_intelligence import build_financial_market_report
from kihc.config import HOP_SIZE
from kihc.config import SAMPLE_RATE as SR
from kihc.connectors.iptv_manager import ChannelUnavailable
from kihc.core import live_worker as lw
from kihc.core.discovery import FPS, POST_ROLL_SEC, DiscoveryEngine, ignore_asset, promote_asset
from kihc.core.discovery_report import discovered_airings, list_assets
from kihc.core.discovery_worker import DiscoveryWorker
from kihc.core.fingerprint import StreamingFingerprinter
from kihc.core.proof import verify_row
from kihc.database import schema
from kihc.database.schema import get_db, init_db
from kihc.service.headless_runner import read_service_status, resolve_channel_arg
from kihc.service.pilot import OneLineFormatter, pilot_channel
from kihc.stream import elheddaf_stream as heddaf
from kihc.tests.synthetic_broadcast import build_schedule, creative, programme, render

ROOT = Path(__file__).resolve().parents[2]
T0 = 1_791_900_000.0


# ================================================================== DB hardening

def test_wal_busy_timeout_and_synchronous(tmp_path):
    conn = init_db(tmp_path / "w.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 15000
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1           # NORMAL
    finally:
        conn.close()


def test_get_db_commits_rolls_back_and_closes(tmp_path):
    db = tmp_path / "g.db"
    with get_db(db) as conn:
        conn.execute("INSERT INTO rate_card_overrides (id, json) VALUES (1, '{}')")
    with pytest.raises(RuntimeError):
        with get_db(db) as conn:
            conn.execute("DELETE FROM rate_card_overrides")
            raise RuntimeError("boom")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")                                               # closed
    with get_db(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM rate_card_overrides").fetchone()[0] == 1   # rolled back


def test_concurrent_writers_and_readers_never_lock(tmp_path):
    db = tmp_path / "c.db"
    init_db(db).close()
    errors = []

    def writer(w):
        try:
            for i in range(80):
                with get_db(db) as conn:
                    conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint) "
                                 "VALUES (?, 'X', 20, x'00')", (f"T{w}_AD_{i:03d}",))
        except Exception as exc:
            errors.append(repr(exc))

    def reader():
        try:
            for _ in range(150):
                with get_db(db) as conn:
                    conn.execute("SELECT COUNT(*) FROM discovered_assets").fetchone()
        except Exception as exc:
            errors.append(repr(exc))
    threads = [threading.Thread(target=writer, args=(w,)) for w in range(6)] + \
              [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert errors == []
    with get_db(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM discovered_assets").fetchone()[0] == 480


def test_schema_init_runs_once_and_again_for_a_replaced_file(tmp_path):
    db = tmp_path / "r.db"
    init_db(db).close()
    assert schema._INITIALISED.get(str(Path(db).resolve()).replace("\\", "\\")) or \
        any(k.endswith("r.db") for k in schema._INITIALISED)
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    sqlite3.connect(db).close()                                                 # new empty file, same path
    with get_db(db) as conn:                                                    # schema recreated
        assert conn.execute("SELECT COUNT(*) FROM reference_ads").fetchone()[0] == 0


# ================================================================== stream + worker plumbing

def test_backoff_5_15_30():
    assert [lw.backoff_delay(i, 5, 30, 3) for i in range(5)] == [5, 15, 30, 30, 30]
    assert [lw.backoff_delay(i) for i in range(4)] == [2, 4, 8, 16]                 # defaults unchanged


class _Resp:
    def __init__(self, ok, text="#EXTM3U\n"):
        self.ok, self.text = ok, text


def test_elheddaf_resolution_order(monkeypatch):
    calls = []

    def fake_get(url, timeout, headers):
        calls.append(url)
        return _Resp(up.get(url, False))
    monkeypatch.setattr(heddaf.requests, "get", fake_get)
    up = {heddaf.AUDIO_HLS: True}
    assert heddaf.resolve_elheddaf_stream() == (heddaf.AUDIO_HLS, "audio-only")
    up = {heddaf.MASTER_HLS: True}
    assert heddaf.resolve_elheddaf_stream() == (heddaf.MASTER_HLS, "master")
    up = {}
    with pytest.raises(ChannelUnavailable):
        heddaf.elheddaf_resolver()()
    assert heddaf.elheddaf_resolver().__name__ == "elheddaf:direct-web-feed"


def test_ffmpeg_flags_and_worker_uses_them(monkeypatch, tmp_path):
    args = heddaf.FFMPEG_INPUT_ARGS
    assert "-reconnect" in args and "-reconnect_streamed" in args and "-reconnect_delay_max" in args
    assert "-reconnect_at_eof" not in args                                       # measured harmful on HLS
    seen = {}

    class FakeProc:
        pid = 1
        stderr = type("E", (), {"readline": lambda self: b"", "close": lambda self: None})()

        def poll(self):
            return 0
    lw.ffmpeg_exe()                       # resolve (and cache) the binary before Popen is faked
    monkeypatch.setattr(lw.subprocess, "Popen", lambda cmd, **kw: seen.setdefault("cmd", cmd) and FakeProc())
    w = lw.LiveStreamWorker("X", "https://h/x.m3u8", tmp_path / "x.db", ffmpeg_input_args=args)
    w._spawn("https://h/x.m3u8")
    cmd = seen["cmd"]
    assert cmd[cmd.index("-i") - len(args):cmd.index("-i")] == args and "-rw_timeout" in cmd


def test_pilot_channel_and_alias(tmp_path, monkeypatch):
    from kihc.connectors import iptv_manager
    monkeypatch.setattr(iptv_manager, "OVERRIDES_FILE", tmp_path / "none.json")     # the legacy web feed: no dedicated override
    spec, wkw = pilot_channel("elheddaf")
    assert spec.name == "El-Heddaf TV" and spec.feed_label == "Direct Web Feed" and callable(spec.source)
    assert wkw["ffmpeg_input_args"] == heddaf.FFMPEG_INPUT_ARGS and wkw["backoff_factor"] == 3.0
    assert resolve_channel_arg("elheddaf").key == "el_heddaf_tv"
    spec2, _ = pilot_channel("Test TV=https://example.com/x.m3u8")
    assert spec2.name == "Test TV"


def test_console_formatter_never_prints_tracebacks():
    try:
        raise ConnectionResetError("peer reset")
    except ConnectionResetError:
        rec = logging.getLogger("kihc.x").makeRecord("kihc.x", logging.WARNING, __file__, 1, "pipe died\nsecond line",
                                                     None, sys.exc_info())
    out = OneLineFormatter("%(levelname)s %(message)s").format(rec)
    assert "\n" not in out and "Traceback" not in out and "ConnectionResetError: peer reset" in out


# ================================================================== discovery engine (synthetic broadcast)

SPOTS = {"A20": (11, 20), "B30": (22, 30), "C45": (33, 45), "ID10": (44, 10)}


def _feed(engine, chunks, piece_seconds: float = 5.0):
    """Feed like the live worker does: <= 5 s pieces (bounded FFT memory), contiguous sample positions."""
    fper, gidx = StreamingFingerprinter(), 0
    piece = int(piece_seconds * SR)
    for c0, x in chunks:
        for p0 in range(0, x.size, piece):
            y = x[p0:p0 + piece]
            fp = fper.feed(y)
            engine.feed(fp, gidx, T0 + np.arange(gidx, gidx + fp.size) * HOP_SIZE / SR,
                        (np.clip(y, -1, 1) * 32767).astype(np.int16), c0 + p0)
            gidx += fp.size
    engine.flush()


@pytest.fixture(scope="module")
def discovered(tmp_path_factory):
    d = tmp_path_factory.mktemp("disc")
    spots = {k: creative(seed, secs) for k, (seed, secs) in SPOTS.items()}
    events, total = build_schedule(spots, n_breaks=5)
    eng = DiscoveryEngine("El-Heddaf TV", d / "disc.db", audio_dir=d / "ads")
    t = time.perf_counter()
    _feed(eng, render(spots, events, total))
    elapsed = time.perf_counter() - t
    truth = {}
    for name, s, e in events:
        truth.setdefault(str(name), []).append(s)
    return dict(db=d / "disc.db", dir=d, engine=eng, truth=truth, total=total, elapsed=elapsed)


def _assets(db):
    with get_db(db) as conn:
        out = {}
        for a in conn.execute("SELECT * FROM discovered_assets ORDER BY asset_id"):
            occ = [r[0] - T0 for r in conn.execute("SELECT airing_start FROM discovered_occurrences WHERE asset_id = ? "
                                                   "ORDER BY airing_start", (a["asset_id"],))]
            out[a["asset_id"]] = dict(row=dict(a), starts=occ)
    return out


def test_discovers_each_recurring_spot_with_every_airing(discovered):
    assets = _assets(discovered["db"])
    print(f"\n[DISCOVERY] {discovered['total']/60:.0f} min processed in {discovered['elapsed']:.1f}s; "
          f"{ {k: (v['row']['duration_sec'], len(v['starts'])) for k, v in assets.items()} }")
    assert sorted(assets) == ["ELHEDDAF_AD_001", "ELHEDDAF_AD_002", "ELHEDDAF_AD_003"]
    by_len = {round(v["row"]["duration_sec"] / 5) * 5: v for v in assets.values()}
    for name, secs in (("A20", 20), ("B30", 30), ("C45", 45)):
        a = by_len[secs]
        assert a["row"]["duration_sec"] == pytest.approx(secs, abs=1.0)
        truth = discovered["truth"][name]
        assert len(a["starts"]) == len(truth)                                    # every airing, no extras
        assert all(abs(s - t) <= 1.0 for s, t in zip(a["starts"], truth))
        assert a["row"]["status"] == "unclassified" and a["row"]["occurrences"] == len(truth)


def test_short_idents_and_programme_are_not_catalogued(discovered):
    assets = _assets(discovered["db"])
    assert all(v["row"]["duration_sec"] >= 15 for v in assets.values())       # 10 s ident rejected
    assert discovered["engine"].counters["rejected_length"] >= 4


def test_asset_audio_and_fingerprint_cached(discovered):
    for aid, v in _assets(discovered["db"]).items():
        wav_path = Path(v["row"]["audio_path"])
        with wave.open(str(wav_path)) as w:
            assert w.getframerate() == SR and w.getnchannels() == 1
            # Phase 11.2: the clip keeps a post-roll (up to 3.5 s) after the measured spot end
            pad = v["row"]["audio_post_pad_sec"]
            assert pad is not None and 0 <= pad <= POST_ROLL_SEC + 1e-6
            assert w.getnframes() / SR == pytest.approx(v["row"]["duration_sec"] + pad, abs=0.6)
        fp_file = discovered["dir"] / "ads" / f"{aid}.fp"
        assert fp_file.stat().st_size == len(v["row"]["fingerprint"]) > 0


def test_review_clips_keep_post_roll_and_nominal_length_is_snapped(discovered):
    for v in _assets(discovered["db"]).values():
        row = v["row"]
        assert row["audio_post_pad_sec"] == pytest.approx(POST_ROLL_SEC, abs=0.05)     # audio after the spot existed
        assert row["nominal_duration_sec"] == round(row["duration_sec"] / 5) * 5 and row["duration_basis"].startswith("snapped")


def test_promote_trims_reference_to_nominal_length(discovered, tmp_path):
    db = tmp_path / "trim.db"
    with sqlite3.connect(discovered["db"]) as a, sqlite3.connect(db) as b:
        a.backup(b)
    b30 = next(k for k, v in _assets(db).items() if round(v["row"]["duration_sec"]) == 30)
    res = promote_asset(db, b30, "Atlas Mobile", "Fiber 30s")
    with get_db(db) as conn:
        dur = conn.execute("SELECT duration_sec FROM reference_ads WHERE ad_id = ?", (res["ad_id"],)).fetchone()[0]
    assert dur == pytest.approx(30.0, abs=0.05)                     # post-roll not in the reference / billing length


def test_backfill_nominal_durations_leaves_occurrences_untouched(discovered, tmp_path):
    from kihc.core.discovery import backfill_nominal_durations
    db = tmp_path / "bf.db"
    with sqlite3.connect(discovered["db"]) as a, sqlite3.connect(db) as b:
        a.backup(b)
    with get_db(db) as conn:
        conn.execute("UPDATE discovered_assets SET nominal_duration_sec = NULL, duration_basis = NULL")
        before_occ = [tuple(r) for r in conn.execute("SELECT * FROM discovered_occurrences ORDER BY id")]
        before_assets = [tuple(r) for r in conn.execute(
            "SELECT asset_id, duration_sec, first_seen, last_seen, occurrences, status FROM discovered_assets")]
    done = backfill_nominal_durations(db)
    assert len(done) == 3 and backfill_nominal_durations(db) == []                    # idempotent
    with get_db(db) as conn:
        assert [tuple(r) for r in conn.execute("SELECT * FROM discovered_occurrences ORDER BY id")] == before_occ
        assert [tuple(r) for r in conn.execute(
            "SELECT asset_id, duration_sec, first_seen, last_seen, occurrences, status FROM discovered_assets")] == before_assets


def test_discovered_airings_report_placement_and_provisional_value(discovered):
    rows = discovered_airings(discovered["db"], limit=0)
    assert len(rows) == 13
    assert all(r["daypart"] and r["break_id"].startswith("BRK-") and r["valuation_basis"].startswith("PROVISIONAL")
               and r["provisional_gross"] > 0 for r in rows)
    first_break = sorted((r for r in rows if r["break_index"] == 1), key=lambda r: r["airing_start"])
    assert [r["position"] for r in first_break] == ["FIRST", "MIDDLE", "LAST"]   # A20, B30, C45 at 120/142/174 s
    assert first_break[0]["break_ads"] == 3


def test_classify_and_ignore_flow(discovered, tmp_path):
    db = tmp_path / "copy.db"
    with sqlite3.connect(discovered["db"]) as a, sqlite3.connect(db) as b:
        a.backup(b)
    assets = _assets(db)
    b30 = next(k for k, v in assets.items() if round(v["row"]["duration_sec"]) == 30)
    a20 = next(k for k, v in assets.items() if round(v["row"]["duration_sec"]) == 20)
    res = promote_asset(db, b30, "Atlas Mobile", "Fiber 30s", industry="Telecom")
    assert res["occurrences_backfilled"] == 4 and res["ad_id"]
    with get_db(db) as conn:
        rows = conn.execute("SELECT * FROM ad_occurrences WHERE ad_id = ?", (res["ad_id"],)).fetchall()
        st = conn.execute("SELECT status, label FROM discovered_assets WHERE asset_id = ?", (b30,)).fetchone()
    assert len(rows) == 4 and all(verify_row(r) is True for r in rows) and tuple(st) == ("classified", "Atlas Mobile – Fiber 30s")
    assert all(r["asset_id"] != b30 for r in discovered_airings(db, limit=0))  # now tracked as a known ad
    rep = build_financial_market_report(db)
    assert {b["brand"] for b in rep["brands"]} == {"Atlas Mobile"} and rep["totals"]["priced_airings"] == 4
    ignore_asset(db, a20, "station promo")
    assert {a["asset_id"]: a["status"] for a in list_assets(db)}[a20] == "ignored"
    assert all(r["asset_id"] != a20 for r in discovered_airings(db, limit=0))
    with pytest.raises(KeyError):
        ignore_asset(db, "NOPE_AD_999")


def test_back_to_back_spots_split_by_digital_silence(tmp_path):
    a, b = creative(51, 20), creative(52, 30)
    gap = np.zeros(2 * SR, np.float32)                                          # digital silence between spots

    def chunks():
        t = 0
        for k in range(3):                                                       # A+gap+B in the same order 3x
            prog = programme(240 if k else 120, seed=900 + k)
            for x in (prog, a, gap, b):
                y = x if x is gap else x + 0.004 * np.random.default_rng(t).standard_normal(x.size).astype(np.float32)
                yield t, y.astype(np.float32)
                t += x.size
        yield t, programme(60, seed=999)
    eng = DiscoveryEngine("Gap TV", tmp_path / "gap.db", audio_dir=tmp_path / "ads")
    _feed(eng, chunks())
    durations = sorted(round(v["row"]["duration_sec"]) for v in _assets(tmp_path / "gap.db").values())
    assert durations == [20, 30], durations                                     # not one 52 s merged asset


def test_history_is_bounded(tmp_path):
    eng = DiscoveryEngine("Mem TV", tmp_path / "m.db", audio_dir=tmp_path, history_hours=0.02)   # 72 s
    rng = np.random.default_rng(3)
    for i in range(0, int(600 * FPS), 4000):
        fp = rng.integers(1, 2 ** 32, size=4000, dtype=np.uint32)
        eng.feed(fp, i, T0 + (i + np.arange(4000)) / FPS, rng.integers(-99, 99, 4000 * HOP_SIZE, dtype=np.int16),
                 i * HOP_SIZE)
    assert eng._fp.size <= eng.cap * 1.1 and eng._t.size == eng._fp.size
    assert sum(x.size for _, x in eng._pcm) <= eng._pcm_cap + 4000 * HOP_SIZE
    assert eng.stats()["assets"] == 0                                            # random data: nothing recurs


# ================================================================== live path through FFmpeg

@pytest.fixture(scope="module")
def synthetic_wav(tmp_path_factory):
    d = tmp_path_factory.mktemp("wav")
    spots = {k: creative(seed, secs) for k, (seed, secs) in SPOTS.items() if k != "ID10"}
    events, total = build_schedule(spots, n_breaks=4)
    path = d / "broadcast.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        for _, x in render(spots, events, total):
            w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    truth = {}
    for name, s, e in events:
        truth.setdefault(str(name), []).append(s)
    return path, truth


def test_discovery_worker_end_to_end_through_ffmpeg(synthetic_wav, tmp_path):
    path, truth = synthetic_wav
    db = tmp_path / "live.db"
    w = DiscoveryWorker("Synth TV", str(path), db, evidence_dir=tmp_path / "ev", max_connects=1,
                        discovery_kwargs=dict(audio_dir=tmp_path / "ads"))
    w.start()
    assert w.wait(240)
    st = w.status()["discovery"]
    print(f"\n[LIVE DISCOVERY] {st}")
    assets = _assets(db)
    assert len(assets) == 3
    for v in assets.values():                                                    # spacing between airings matches truth
        secs = round(v["row"]["duration_sec"] / 5) * 5
        name = {20: "A20", 30: "B30", 45: "C45"}[secs]
        starts = v["starts"]
        assert len(starts) == len(truth[name])
        gaps = np.diff(starts) - np.diff(truth[name])
        assert np.all(np.abs(gaps) <= 1.0)


# ================================================================== API

@pytest.fixture
def api(discovered, tmp_path):
    from kihc.app.api_server import create_app
    db = tmp_path / "api.db"
    with sqlite3.connect(discovered["db"]) as a, sqlite3.connect(db) as b:
        a.backup(b)
    app = create_app(db, channel_loader=lambda: [])
    app.state.audio_root = discovered["dir"] / "ads"
    with TestClient(app) as c:
        yield c


def test_discovery_api(api):
    r = api.get("/api/discovery/assets").json()
    assert r["count"] == 3 and all("audio_path" not in a and a["audio_url"] for a in r["assets"])
    assert "not necessarily" in r["note"] or "may be ads" in r["note"]
    aid = r["assets"][0]["asset_id"]
    au = api.get(f"/api/discovery/assets/{aid}/audio")
    assert au.status_code == 200 and au.headers["content-type"] == "audio/wav" and au.content[:4] == b"RIFF"
    assert api.get("/api/discovery/assets/..%2F..%2Fsecret/audio").status_code == 404
    assert api.get("/api/discovery/assets/NOPE_AD_999/audio").status_code == 404
    occ = api.get("/api/discovery/occurrences", params={"limit": 50}).json()["occurrences"]
    assert len(occ) == 13 and {"break_id", "position", "daypart", "provisional_gross"} <= set(occ[0])
    bad = api.post(f"/api/discovery/assets/{aid}/classify", json={"brand": "", "campaign": "x"})
    assert bad.status_code == 422
    ok = api.post(f"/api/discovery/assets/{aid}/classify", json={"brand": "Atlas Mobile", "campaign": "Spot"})
    assert ok.status_code == 200 and ok.json()["ad_id"]
    assert api.post(f"/api/discovery/assets/{aid}/ignore", json={"label": "x"},
                    headers={"origin": "https://evil.example"}).status_code == 403


def test_dashboard_has_discovery_card_and_stream_pill(api):
    html = api.get("/").text
    for element_id in ("disc-body", "disc-stats", "stream-pill"):
        assert f'id="{element_id}"' in html
    js = api.get("/static/app.js").text
    assert "loadDiscovery" in js and "HIDDEN_EVERY" in js and "Direct" not in js.split("renderStreamPill")[0][-50:]


# ================================================================== unified pilot command (real process)

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pilot_run_starts_ingest_and_dashboard(synthetic_wav, tmp_path):
    path, _ = synthetic_wav
    db, port = tmp_path / "pilot.db", _free_port()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen([sys.executable, "-m", "kihc.cli", "pilot-run", "--channel", f"Synth TV={path}",
                             "--port", str(port), "--db", str(db)], cwd=ROOT, creationflags=flags,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if requests.get(f"{base}/healthz", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.3)
        t = {}
        while time.monotonic() < deadline:
            t = requests.get(f"{base}/api/telemetry", timeout=5).json()
            if t["engine"] and t["engine"]["active"] and t["channels"] and t["channels"][0].get("discovery"):
                break
            time.sleep(0.5)
        assert t["engine"]["active"] and t["channels"][0]["name"] == "Synth TV"
        assert "frames" in t["channels"][0]["discovery"]
        assert 'id="disc-body"' in requests.get(base, timeout=5).text
        assert read_service_status(db)["lock_held"]
    finally:
        proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT if sys.platform == "win32" else subprocess.signal.SIGINT)
        try:
            proc.wait(40)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(10)
    out = proc.stdout.read().decode(errors="replace")
    assert "Traceback" not in out, out[-2000:]
    assert not read_service_status(db)["lock_held"]


def test_stop_service_stops_the_whole_pilot(synthetic_wav, tmp_path):
    path, _ = synthetic_wav
    db, port = tmp_path / "stop.db", _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "kihc.cli", "pilot-run", "--channel", f"Synth TV={path}",
                             "--port", str(port), "--db", str(db)], cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not (read_service_status(db) or {}).get("active"):
            time.sleep(0.3)
        stop = subprocess.run([sys.executable, "-m", "kihc.cli", "stop-service", "--db", str(db)], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)
        assert "Service stopped" in stop.stdout, stop.stdout + stop.stderr
        assert proc.wait(30) == 0                                                # web server exited too
    finally:
        if proc.poll() is None:
            proc.kill()
    with pytest.raises(requests.RequestException):
        requests.get(f"http://127.0.0.1:{port}/healthz", timeout=2)
