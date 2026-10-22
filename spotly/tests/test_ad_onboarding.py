"""Phase 16: pre-broadcast master onboarding, silence trimming, robust packshot, zero-downtime registry hot-reload."""
import base64
import json
import logging
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from spotly.analytics import ad_onboarding as ob
from spotly.analytics import compliance_audit as ca
from spotly.analytics import reference_matcher as rm
from spotly.core.audio import ffmpeg_exe
from spotly.data import reference_db as rdb
from spotly.database.schema import get_db
from spotly.tests.test_reference_matcher import noise, synth

SR = rm.SR
LEAD_S, BODY_S, TAIL_S = 1.0, 6.0, 0.5


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def master_audio(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(full master with 1.0 s digital silence in front and 0.5 s behind, the sound part alone)."""
    body = synth(seed, BODY_S)
    return np.concatenate([np.zeros(int(LEAD_S * SR), np.float32), body, np.zeros(int(TAIL_S * SR), np.float32)]), body


def make_mp4(path: Path, audio: np.ndarray, fade_start: float = 2.0, fade_len: float = 3.0) -> float:
    """H.263-class test video that fades to black well before the end, with `audio` as its soundtrack."""
    wav = path.with_suffix(".src.wav")
    write_wav(wav, audio)
    dur = audio.size / SR
    p = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
                        "-f", "lavfi", "-i", f"testsrc2=s=320x240:r=10:d={dur}", "-i", str(wav),
                        "-vf", f"fade=t=out:st={fade_start}:d={fade_len}", "-c:v", "mpeg4", "-c:a", "aac", "-b:a", "128k",
                        "-shortest", str(path)], capture_output=True)
    assert p.returncode == 0, p.stderr.decode()
    return dur


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "k.db"
    with get_db(path) as conn:
        rdb.ensure_schema(conn)
    return path


@pytest.fixture
def mp4(tmp_path):
    full, body = master_audio(11)
    f = tmp_path / "spot.mp4"
    dur = make_mp4(f, full)
    return f, full, body, dur


def feed_all(m, channel, audio, epoch0=1_700_000_000.0, piece=0.2):
    out, step = [], int(piece * SR)
    for i in range(0, audio.size, step):
        out += m.feed(channel, audio[i:i + step], epoch0 + i / SR)
    return out


# ----------------------------------------------------------------------------- trimming / extraction
def test_trim_uses_absolute_minus_50_dbfs_floor():
    quiet = np.concatenate([np.full(SR, 0.002, np.float32),             # -54 dBFS: digital silence for this rule
                            synth(1, 2.0), np.zeros(SR // 2, np.float32)])
    y, lead = rm.trim_silence(quiet, margin_s=0.02, floor_db=-50.0, absolute=True)
    assert 0.97 <= lead <= 1.01 and 1.95 <= y.size / SR <= 2.1
    loud_hiss = np.concatenate([np.full(SR, 0.01, np.float32), synth(1, 2.0)])         # -40 dBFS: kept (not silence)
    assert rm.trim_silence(loud_hiss, margin_s=0.02, floor_db=-50.0, absolute=True)[1] == 0.0


def test_register_mp4_trims_leading_silence_and_stores_everything(db, mp4):
    f, full, body, dur = mp4
    r = ob.register_ad(f, "Djezzy", "Telecom", separation_window=180, position="FIRST_OR_LAST", rate_override=250000,
                       db_path=db)
    assert 0.95 <= r["lead_trim_s"] <= 1.02 and 0.4 <= r["tail_trim_s"] <= 0.55
    assert abs(r["trimmed_s"] - BODY_S) < 0.15 and abs(r["duration_s"] - dur) < 0.1 and r["hashes"] > 200
    with get_db(db) as conn:
        s = conn.execute("SELECT * FROM reference_signatures").fetchone()
        ad = conn.execute("SELECT * FROM reference_ads WHERE ad_id = ?", (r["ad_id"],)).fetchone()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert s["ad_id"] == r["ad_id"] and s["canonical_sector"] == "Telecom" and s["visual_p_hash"]
    _, _, lead, trimmed = rdb.unpack_blob(s["audio_fingerprint_blob"])
    assert abs(lead - r["lead_trim_s"]) < 2e-3 and abs(trimmed - r["trimmed_s"]) < 1e-2
    assert ad["industry"] == "Djezzy" or ad["industry"] == "Telecom"
    c = next(c for c in ca.get_contracts(db) if c["brand"] == "Djezzy")
    assert (c["booked_position"], c["competitor_separation_seconds"], c["rate_override_dzd"]) == ("FIRST_OR_LAST", 180, 250000)


def test_packshot_skips_fade_to_black(db, mp4):
    f, _, _, dur = mp4
    shot = ob.register_ad(f, "Djezzy", "Telecom", db_path=db)["packshot"]
    first_at, first_entropy, first_luma = shot["tried"][0]
    assert abs(first_at - (dur - 1.5)) < 0.01                               # the spec'd candidate was tried first ...
    assert first_luma < ob.MIN_MEAN_LUMA                                    # ... found black, and was refused
    assert shot["at"] < dur - 1.5 and shot["entropy"] >= ob.MIN_ENTROPY_BITS and shot["luma"] >= ob.MIN_MEAN_LUMA


def test_audio_only_master_has_no_packshot_and_wav_mp3_accepted(db, tmp_path):
    w = tmp_path / "radio.wav"
    write_wav(w, master_audio(12)[0])
    r = ob.register_ad(w, "Mobilis", "Telecom", db_path=db)
    assert r["packshot"] is None and r["lead_trim_s"] > 0.9
    w2, mp3 = tmp_path / "radio2.wav", tmp_path / "radio.mp3"          # different audio: the same creative is refused as a duplicate
    write_wav(w2, master_audio(14)[0])
    p = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", str(w2), str(mp3)], capture_output=True)
    if p.returncode == 0:                                                    # an ffmpeg build without an mp3 encoder
        assert ob.register_ad(mp3, "Ooredoo", "Telecom", db_path=db)["hashes"] > 100


@pytest.mark.parametrize("kw,msg", [(dict(brand=""), "required"), (dict(position="MIDDLE"), "position"),
                                    (dict(separation_window=0), "separation"), (dict(rate_override=-1), "rate")])
def test_bad_input_is_refused_before_any_write(db, tmp_path, kw, msg):
    w = tmp_path / "a.wav"
    write_wav(w, master_audio(13)[0])
    args = dict(brand="B", sector="S", db_path=db)
    args.update(kw)
    brand = args.pop("brand")
    with pytest.raises(ob.OnboardingError, match=msg):
        ob.register_ad(w, brand, args.pop("sector"), **args)
    with get_db(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM reference_signatures").fetchone()[0] == 0


def test_rejects_unsupported_extension_and_silent_file(db, tmp_path):
    t = tmp_path / "x.txt"
    t.write_text("no")
    with pytest.raises(ob.OnboardingError, match="unsupported"):
        ob.register_ad(t, "B", "S", db_path=db)
    w = tmp_path / "silent.wav"
    write_wav(w, np.zeros(3 * SR, np.float32))
    with pytest.raises(ob.OnboardingError, match="landmarks"):
        ob.register_ad(w, "B", "S", db_path=db)


# ----------------------------------------------------------------------------- live match of a pre-registered master
def test_registered_master_is_recognised_in_a_stream_at_once(db, mp4):
    f, full, body, dur = mp4
    r = ob.register_ad(f, "Djezzy", "Telecom", db_path=db)
    m = rm.ReferenceMatcher(db, compliance=False)
    assert m.covered_ad_ids() == {r["ad_id"]}
    on_air_at = 8.0                                                         # master (incl. its 1 s lead) starts here
    stream = np.concatenate([noise(on_air_at, 0.02), full * 0.6, noise(6, 0.02, seed=5)])
    t0 = 1_700_000_000.0
    cands = feed_all(m, "El-Heddaf TV", stream, t0) + m.end_stream("El-Heddaf TV")
    hit = [c for c in cands if c.asset_id == r["asset_id"]]
    assert len(hit) == 1
    c = hit[0]
    assert c.decision == "confirmed" and c.confidence >= rm.ACCEPT_CONF and c.occurrence_id
    assert abs(c.airing_start - (t0 + on_air_at)) < 0.2                     # aligned to the first on-air frame (lead added back)
    rep = m.memory_report()
    assert rep["lookup_ms_p50"] < 30 and rep["lookup_ms_p95"] < 30 and rep["lookup_ms_max"] < 30, rep
    with get_db(db) as conn:
        row = conn.execute("SELECT ad_id, source_name FROM ad_occurrences").fetchone()
        assert (row["ad_id"], row["source_name"]) == (r["ad_id"], "live:El-Heddaf TV")
        assert conn.execute("SELECT match_count FROM reference_signatures").fetchone()[0] == 1


# ----------------------------------------------------------------------------- hot reload
def _two_masters(db, tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    write_wav(a, master_audio(21)[0])
    write_wav(b, master_audio(22)[0])
    return a, b, master_audio(21)[0], master_audio(22)[0]


def test_hot_reload_mid_airing_drops_no_frames_and_logs(db, tmp_path, caplog):
    a, b, fa, fb = _two_masters(db, tmp_path)
    ra = ob.register_ad(a, "BrandA", "Telecom", db_path=db)
    m = rm.ReferenceMatcher(db, compliance=False)
    t0 = 1_700_000_000.0
    stream = np.concatenate([noise(5, 0.02), fa, noise(4, 0.02, seed=6), fb, noise(6, 0.02, seed=7)])
    cut = int((5 + fa.size / SR / 2) * SR)                                 # halfway through BrandA's airing
    out = feed_all(m, "Ch", stream[:cut], t0)
    sm_before = m._streams["Ch"]
    ob.register_ad(b, "BrandB", "Telecom", db_path=db)                      # a new campaign lands while A is on air
    old_index = m.index
    with caplog.at_level(logging.INFO, logger="spotly.reference"):
        assert m.atomic_reload_signatures() is True
    assert any("[REGISTRY] Hot-reloaded asset: BrandB (" in r.getMessage() and r.getMessage().endswith("s)")
               for r in caplog.records)
    assert m.index is not old_index and len(m.index) == 2 and len(old_index) == 1   # shadow built, pointer swapped
    out += feed_all(m, "Ch", stream[cut:], t0 + cut / SR)
    out += m.end_stream("Ch")
    assert m._streams.get("Ch") is None and sm_before.index is m.index              # same stream object adopted the swap
    got = {c.brand_name: c for c in out if c.decision in ("confirmed", "confirmed_visual")}
    assert set(got) == {"BrandA", "BrandB"}, [(c.brand_name, c.decision, c.confidence) for c in out]
    assert got["BrandA"].confidence >= rm.ACCEPT_CONF and got["BrandB"].confidence >= rm.ACCEPT_CONF
    assert abs(got["BrandA"].airing_start - (t0 + 5.0)) < 0.2                       # A straddled the swap unharmed
    assert m.atomic_reload_signatures() is False                                    # nothing new: no-op


def test_reload_from_another_thread_while_feeding(db, tmp_path):
    """The registry is rebuilt and swapped over and over from a second thread while the live stream is being fed."""
    a, b, fa, fb = _two_masters(db, tmp_path)
    ob.register_ad(a, "BrandA", "Telecom", db_path=db)
    m = rm.ReferenceMatcher(db, compliance=False)                           # live monitor knows A only
    ob.register_ad(b, "BrandB", "Telecom", db_path=db)                      # B is onboarded; the monitor has not seen it yet
    stream = np.concatenate([noise(4, 0.02), fa, noise(3, 0.02, seed=6), fb, noise(5, 0.02, seed=7)])
    errors, out, done = [], [], threading.Event()

    def feeder():
        try:
            step, t0 = int(0.2 * SR), 1_700_000_000.0
            for i in range(0, stream.size, step):
                out.extend(m.feed("Ch", stream[i:i + step], t0 + i / SR))
            out.extend(m.end_stream("Ch"))
        except Exception as exc:                                              # pragma: no cover
            errors.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=feeder)
    t.start()
    swaps = 0
    while not done.is_set():
        swaps += m.atomic_reload_signatures(force=True)                       # fresh shadow index + pointer swap each time
        time.sleep(0.05)
    t.join(120)
    assert not errors and swaps >= 2
    got = {c.brand_name: c for c in out if c.decision == "confirmed"}
    assert set(got) == {"BrandA", "BrandB"} and all(c.confidence >= rm.ACCEPT_CONF for c in got.values())


def test_reregistering_the_same_master_updates_in_place(db, tmp_path):
    w = tmp_path / "a.wav"
    write_wav(w, master_audio(31)[0])
    r1 = ob.register_ad(w, "BrandA", "Telecom", campaign="C1", db_path=db)
    m = rm.ReferenceMatcher(db, compliance=False)
    r2 = ob.register_ad(w, "BrandA", "Telecom", campaign="C1", separation_window=300, db_path=db)
    assert r1["asset_id"] == r2["asset_id"] and r1["ad_id"] == r2["ad_id"]
    assert m.atomic_reload_signatures() is True and len(m.index) == 1
    assert len([c for c in ca.get_contracts(db) if c["brand"] == "BrandA"]) == 1


# ----------------------------------------------------------------------------- rate override flows into spend
def test_contract_rate_override_replaces_the_daypart_rate(db):
    from spotly.analytics.rate_engine import RateCardEngine, compute_airing_financials
    ca.save_contracts(db, [dict(brand="B", campaign="*", rate_override_dzd=100000, agency_discount=0.0)])
    card = RateCardEngine.load(db_path=db)
    f = compute_airing_financials(dict(brand="B", campaign="C", channel="El-Heddaf TV", start_epoch=1_700_000_000.0,
                                       spot_seconds=30, pod_position="MIDDLE"), card)
    assert f["priced"] and f["rate_source"] == "contract rate override" and str(f["spot_value"]) == "100000.00"


# ----------------------------------------------------------------------------- CLI + API
def test_cli_register_ad(db, tmp_path, capsys):
    from spotly import cli
    w = tmp_path / "a.wav"
    write_wav(w, master_audio(41)[0])
    cli.main(["register-ad", "--file", str(w), "--brand", "Condor", "--sector", "Electronics", "--separation-window", "120",
              "--position", "LAST", "--rate-card-override", "90000", "--db", str(db)])
    out = capsys.readouterr().out
    assert "Registered Condor" in out and "separation 120s, position LAST, rate 90,000 DZD" in out
    with pytest.raises(SystemExit, match="register-ad failed"):
        cli.main(["register-ad", "--file", str(tmp_path / "nope.wav"), "--brand", "X", "--sector", "Y", "--db", str(db)])


def test_dashboard_has_the_upload_component():
    root = Path(rm.__file__).parents[1] / "app"
    html = (root / "templates/index.html").read_text(encoding="utf-8")
    js = (root / "static/app.js").read_text(encoding="utf-8")
    for needle in ('id="master-open"', 'id="master-drop"', 'id="master-wave"', 'id="master-sector"', 'id="master-position"',
                   'id="master-sep"', 'accept=".mp4,.wav,.mp3'):
        assert needle in html
    assert "/api/ads/register" in js and "decodeAudioData" in js and "initMaster()" in js


def test_api_register_master(db, tmp_path):
    from fastapi.testclient import TestClient

    from spotly.app.api_server import create_app
    w = tmp_path / "a.wav"
    write_wav(w, master_audio(51)[0])
    body = dict(filename="spot.wav", data_b64=base64.b64encode(w.read_bytes()).decode(), brand="Cevital", sector="Food",
                position="FIRST", separation_window=180)
    with TestClient(create_app(db)) as c:
        r = c.post("/api/ads/register", json=body)
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["asset_id"].startswith("MASTER_") and j["lead_trim_s"] > 0.9 and j["contract"]["booked_position"] == "FIRST"
        assert c.post("/api/ads/register", json=dict(body, filename="x.exe")).status_code == 422
        assert c.post("/api/ads/register", json=dict(body, data_b64="!!!notbase64!!!")).status_code == 422
        assert any(a["brand_name"] == "Cevital" for a in c.get("/api/ads").json()["ads"])
    assert rm.ReferenceMatcher(db, compliance=False).covered_ad_ids() == {j["ad_id"]}


# ----------------------------------------------------------------------------- memory: a live-ingest process
CHILD = r"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
from spotly.analytics import ad_onboarding as ob, reference_matcher as rm
from spotly.service.sysinfo import memory_mb
from spotly.tests.test_ad_onboarding import make_mp4, master_audio, feed_all
from spotly.tests.test_reference_matcher import noise
tmp = Path(sys.argv[1]); db = tmp / "m.db"
full, _ = master_audio(61); f = tmp / "s.mp4"; make_mp4(f, full)
m = rm.ReferenceMatcher(db, compliance=False)               # live monitor already running on an empty registry
feed_all(m, "El-Heddaf TV", noise(3, 0.02))
ob.register_ad(f, "Djezzy", "Telecom", db_path=db)          # pre-broadcast onboarding
m.atomic_reload_signatures()
c = feed_all(m, "El-Heddaf TV", np.concatenate([noise(4, 0.02), full, noise(5, 0.02, seed=3)])) + m.end_stream("El-Heddaf TV")
cur, peak = memory_mb()
print(json.dumps(dict(peak_mb=peak, cur_mb=cur, hits=[(x.brand_name, x.decision, round(x.confidence, 3)) for x in c])))
"""


def test_live_ingest_process_stays_under_180_mb(tmp_path):
    p = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r); exec(%r)"
                        % (str(Path(rm.__file__).parents[2]), CHILD), str(tmp_path)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-800:]
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["hits"] and res["hits"][0][1] == "confirmed"
    assert 0 < res["peak_mb"] < 180, res
