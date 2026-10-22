"""Phase 2.5: loudness normalization, radar telemetry, SHA-256 proof, PDF certificates."""
import re
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from spotly.core.audio import ffmpeg_exe, iter_pcm
from spotly.core.fingerprint import fingerprint_samples
from spotly.core.ingest import ingest_reference
from spotly.core.live_worker import LiveStreamWorker
from spotly.core.matcher import scan_source
from spotly.core.normalize import (SILENCE_DBFS, AdaptiveNormalizer, gain_for,
                                 normalize_rms, rms_dbfs)
from spotly.core.proof import (ALGERIA_TZ, algeria_iso,
                             compute_verification_hash, verify_row)
from spotly.database.schema import init_db
from spotly.reports.certificate import certificate_for
from spotly.tests.generate_test_assets import generate

SR = 11025


def _tone_plus_noise(dbfs, seconds=3.0, seed=0):
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.default_rng(seed).standard_normal(t.size)
    x = x / np.sqrt(np.mean(x ** 2)) * 10 ** (dbfs / 20)
    return x.astype(np.float32)


# ------------------------------------------------------------------ normalization

@pytest.mark.parametrize("level", [-35.0, -10.0])
def test_normalize_brings_weak_and_loud_to_target(level):
    x = _tone_plus_noise(level)
    assert rms_dbfs(x) == pytest.approx(level, abs=0.05)
    assert rms_dbfs(normalize_rms(x)) == pytest.approx(-20.0, abs=0.05)


def test_silence_is_not_amplified():
    x = _tone_plus_noise(-75.0)
    assert gain_for(x) == 1.0
    assert rms_dbfs(normalize_rms(x)) == pytest.approx(-75.0, abs=0.05)


def test_adaptive_normalizer_ramps_without_steps():
    norm = AdaptiveNormalizer(SR)
    a, b = norm.process(_tone_plus_noise(-35.0, 2, 1)), norm.process(_tone_plus_noise(-10.0, 2, 2))
    assert rms_dbfs(a) == pytest.approx(-20.0, abs=0.1)
    assert rms_dbfs(b[SR:]) == pytest.approx(-20.0, abs=0.1)      # settled after the 0.5 s ramp
    g = np.abs(b[:200]) / (np.abs(_tone_plus_noise(-10.0, 2, 2)[:200]) + 1e-9)
    assert g.max() < 10 ** (15 / 20)                              # no jump to the full -35 dB gain


def test_fingerprint_is_level_invariant():
    """Why normalization does not move match scores: constant gain leaves the bits unchanged."""
    x = _tone_plus_noise(-20.0, 5)
    fp = fingerprint_samples(x)
    for level in (-35.0, -10.0):
        diff = np.bitwise_count(fingerprint_samples(x * 10 ** ((level + 20) / 20)) ^ fp).sum()
        assert diff <= 0.0001 * fp.size * 32


def _relevel_wav(src, dest, gain_db):
    subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn",
                    "-af", f"volume={gain_db:.2f}dB", "-c:a", "pcm_s16le", str(dest)], check=True)


@pytest.fixture(scope="module")
def level_ctx(tmp_path_factory):
    ad_a, _, broadcast = generate()
    d = tmp_path_factory.mktemp("levels")
    db = d / "lvl.db"
    ingest_reference(ad_a, "Acme", "Ad A", db_path=db)
    pcm = np.concatenate(list(iter_pcm(broadcast, 30)))
    return dict(dir=d, db=db, broadcast=broadcast, pcm=pcm, rms=rms_dbfs(pcm))


def test_weak_signal_minus35_detected(level_ctx, tmp_path):
    """Weak IPTV feed: 16-bit file at -35 dBFS RMS through the normal file pipeline."""
    p = tmp_path / "weak.wav"
    _relevel_wav(level_ctx["broadcast"], p, -35 - level_ctx["rms"])
    measured = rms_dbfs(np.concatenate(list(iter_pcm(p, 30))))
    db = tmp_path / "w.db"
    ingest_reference(generate()[0], "Acme", "Ad A", db_path=db)
    dets = scan_source(p, db_path=db, evidence_dir=tmp_path)
    print(f"\n[LEVEL] weak {measured:.1f} dBFS -> {[(d.airing_start, d.confidence_score) for d in dets]}")
    assert measured == pytest.approx(-35, abs=0.5)
    assert len(dets) == 1 and 24.0 <= dets[0].airing_start <= 26.0 and dets[0].confidence_score > 0.85


def test_loud_signal_minus10_detected(level_ctx):
    """Loud feed at -10 dBFS RMS, unclipped (float, as a decoder delivers it), through normalize + matcher."""
    from spotly.core.fingerprint import StreamingFingerprinter
    from spotly.core.matcher import _load_refs
    x = level_ctx["pcm"] * 10 ** ((-10 - level_ctx["rms"]) / 20)
    assert rms_dbfs(x) == pytest.approx(-10, abs=0.05) and np.abs(x).max() > 1.0   # would clip in 16-bit
    norm, fper = AdaptiveNormalizer(SR), StreamingFingerprinter()
    fp = np.concatenate([fper.feed(norm.process(c)) for c in np.array_split(x, 2)])
    conn = init_db(level_ctx["db"])
    ref = _load_refs(conn, 0.82)[0]
    conn.close()
    ref.scan(fp, 0)
    ref.flush()
    det = ref.detection(*ref.found[0], "loud")
    print(f"\n[LEVEL] loud -10.0 dBFS (unclipped) -> ({det.airing_start}, {det.confidence_score})")
    assert len(ref.found) == 1 and 24.0 <= det.airing_start <= 26.0 and det.confidence_score > 0.85


def test_saturated_signal_characterisation(level_ctx, tmp_path):
    """KNOWN LIMITATION: pushing this 20 dB-crest signal to -10 dBFS in 16-bit hard-clips ~1.3% of samples.

    Normalization cannot undo clipping; confidence falls to ~0.80 (below the 0.82 file threshold).
    This test pins the measured behaviour: the airing is still located correctly at a lower threshold.
    """
    p = tmp_path / "saturated.wav"
    _relevel_wav(level_ctx["broadcast"], p, -10 - level_ctx["rms"])
    x = np.concatenate(list(iter_pcm(p, 30)))
    clipped = (np.abs(x) >= 0.9999).mean()
    db = tmp_path / "s.db"
    ingest_reference(generate()[0], "Acme", "Ad A", db_path=db)
    dets = scan_source(p, db_path=db, evidence_dir=tmp_path, threshold=0.70)
    print(f"\n[LEVEL] saturated {rms_dbfs(x):.1f} dBFS, {clipped * 100:.2f}% clipped -> "
          f"{[(d.airing_start, d.confidence_score) for d in dets]}")
    assert clipped > 0.005
    assert len(dets) == 1 and 24.0 <= dets[0].airing_start <= 26.0 and dets[0].confidence_score >= 0.75


# ------------------------------------------------------------------ live: radar + proof

@pytest.fixture(scope="module")
def live_run(tmp_path_factory):
    ad_a, ad_b, broadcast = generate()
    d = tmp_path_factory.mktemp("live25")
    db = d / "live.db"
    ingest_reference(ad_a, "Acme", "Ad A", db_path=db)
    ingest_reference(ad_b, "Decoy Co", "Ad B", db_path=db)
    w = LiveStreamWorker("El-Heddaf TV", str(broadcast), db, evidence_dir=d / "ev", max_connects=1)
    w.start()
    radar_seen = []
    while w.is_alive():
        r = w.radar()
        if r["level_dbfs"] is not None:
            radar_seen.append(r)
        time.sleep(0.02)
    return dict(db=db, worker=w, radar=radar_seen, broadcast=broadcast, dir=d)


def test_radar_reports_input_level(live_run):
    expected = rms_dbfs(np.concatenate(list(iter_pcm(live_run["broadcast"], 30))))
    hist = [l for _, l in live_run["worker"].radar()["level_history"]]
    avg = 10 * np.log10(np.mean([10 ** (l / 10) for l in hist]))
    print(f"\n[RADAR] file RMS {expected:.1f} dBFS, radar mean {avg:.1f} dBFS over {len(hist)} samples")
    assert hist and avg == pytest.approx(expected, abs=1.5)
    connected = [r for r in live_run["radar"] if r["connected_for_s"] is not None]
    assert connected, "radar never reported a connected stream with a level"
    assert all(r["silent"] is False for r in connected)
    assert live_run["worker"].radar()["connected_for_s"] is None        # stream ended -> disconnected


def test_radar_flags_silence(tmp_path):
    silent = tmp_path / "silence.wav"
    subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "anullsrc=r=22050:cl=mono", "-t", "3", str(silent)], check=True)
    w = LiveStreamWorker("DeadAir", str(silent), tmp_path / "s.db", evidence_dir=tmp_path,
                         max_connects=1, buffer_seconds=30).start()
    seen = []
    while w.is_alive():
        seen.append(w.radar())
        time.sleep(0.01)
    seen.append(w.radar())
    assert any(r["silent"] for r in seen if r["peak_dbfs_10s"] is not None)
    assert max(r["peak_dbfs_10s"] for r in seen if r["peak_dbfs_10s"] is not None) < SILENCE_DBFS


def _row(db, occurrence_id=None):
    conn = init_db(db)
    try:
        sql = ("SELECT * FROM ad_occurrences WHERE occurrence_id = ?" if occurrence_id
               else "SELECT * FROM ad_occurrences ORDER BY occurrence_id LIMIT 1")
        return conn.execute(sql, (occurrence_id,) if occurrence_id else ()).fetchone()
    finally:
        conn.close()


def test_live_occurrence_has_algeria_time_and_hash(live_run):
    row = _row(live_run["db"])
    ts = datetime.fromisoformat(row["algeria_timestamp"])
    assert ts.utcoffset().total_seconds() == 3600                         # CET, UTC+1
    assert ts.timestamp() == pytest.approx(row["airing_start"], abs=0.001)
    assert re.fullmatch(r"[0-9a-f]{64}", row["verification_hash"])
    det = live_run["worker"].detections[0]
    assert det.verification_hash == row["verification_hash"] and det.algeria_timestamp == row["algeria_timestamp"]
    assert verify_row(row) is True


def test_hash_is_deterministic_and_field_sensitive():
    args = (1, "live:El-Heddaf TV", algeria_iso(1791900000.0), 1791900010.0, 9.985, 0.9258, b"\x89PNG...")
    h = compute_verification_hash(*args)
    assert h == compute_verification_hash(*args)
    for i, changed in enumerate([2, "live:Echorouk TV", algeria_iso(1791900001.0), 1791900011.0, 9.0, 0.99,
                                 b"\x89PNG..!"]):
        mutated = list(args)
        mutated[i] = changed
        assert compute_verification_hash(*mutated) != h, f"field {i} not covered by the hash"


@pytest.mark.parametrize("tamper", ["confidence", "channel", "timestamp", "evidence"])
def test_tamper_detection(live_run, tamper, tmp_path):
    src = live_run["db"]
    db = tmp_path / "copy.db"
    with sqlite3.connect(src) as a, sqlite3.connect(db) as b:
        a.backup(b)
    row = _row(db)
    ev = tmp_path / "evidence.png"
    ev.write_bytes(Path(row["evidence_frame_path"]).read_bytes())
    with sqlite3.connect(db) as c:
        c.execute("UPDATE ad_occurrences SET evidence_frame_path = ?", (str(ev),))
    assert verify_row(_row(db)) is True                                  # intact copy verifies
    with sqlite3.connect(db) as c:
        if tamper == "confidence":
            c.execute("UPDATE ad_occurrences SET confidence_score = 0.99")
        elif tamper == "channel":
            c.execute("UPDATE ad_occurrences SET source_name = 'live:Echorouk TV'")
        elif tamper == "timestamp":
            c.execute("UPDATE ad_occurrences SET algeria_timestamp = '2026-01-01T20:00:00.000+01:00'")
    if tamper == "evidence":
        data = bytearray(ev.read_bytes())
        data[len(data) // 2] ^= 0xFF
        ev.write_bytes(bytes(data))
    assert verify_row(_row(db)) is False


def test_file_scan_rows_are_hashed_too(tmp_path):
    ad_a, _, broadcast = generate()
    db = tmp_path / "f.db"
    ingest_reference(ad_a, "Acme", "Ad A", db_path=db)
    scan_source(broadcast, db_path=db, evidence_dir=tmp_path)
    row = _row(db)
    assert row["algeria_timestamp"] is None and verify_row(row) is True


def test_schema_migrates_old_database(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as c:   # Day 1 schema, without the Phase 2.5 columns
        c.executescript("""
            CREATE TABLE reference_ads (ad_id INTEGER PRIMARY KEY AUTOINCREMENT, brand_name TEXT NOT NULL,
              campaign_name TEXT NOT NULL, duration_sec REAL NOT NULL, fingerprint BLOB NOT NULL,
              fp_algorithm TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT 'x', UNIQUE (brand_name, campaign_name));
            CREATE TABLE ad_occurrences (occurrence_id INTEGER PRIMARY KEY AUTOINCREMENT, ad_id INTEGER NOT NULL,
              source_name TEXT NOT NULL, airing_start REAL NOT NULL, airing_end REAL NOT NULL,
              duration_detected REAL NOT NULL, confidence_score REAL NOT NULL, evidence_frame_path TEXT,
              human_verified INTEGER NOT NULL DEFAULT 0, detected_at TEXT NOT NULL DEFAULT 'x',
              UNIQUE (ad_id, source_name, airing_start));
            INSERT INTO reference_ads (brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm)
              VALUES ('Old', 'Row', 10, x'00', 'kihc-subband-v2-masked');
            INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected,
              confidence_score) VALUES (1, 'old.mp4', 1.0, 11.0, 10.0, 0.9);
        """)
    conn = init_db(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ad_occurrences)")}
    row = conn.execute("SELECT * FROM ad_occurrences").fetchone()
    conn.close()
    assert {"verification_hash", "algeria_timestamp"} <= cols
    assert row["confidence_score"] == 0.9 and row["verification_hash"] is None
    assert verify_row(row) is None                                       # legacy row: no hash, not "tampered"
    init_db(db).close()                                                  # idempotent


# ------------------------------------------------------------------ certificates

def test_pdf_certificate_fast_and_valid(live_run):
    row = _row(live_run["db"])
    t0 = time.perf_counter()
    pdf = certificate_for(live_run["db"], row["occurrence_id"])
    elapsed = time.perf_counter() - t0
    print(f"\n[PDF] {len(pdf)} bytes in {elapsed:.3f}s")
    assert elapsed < 1.5
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    assert len(re.findall(rb"/Type\s*/Page\b", pdf)) == 1                 # single page
    assert re.search(rb"/Subtype\s*/Image", pdf)                         # evidence embedded
    (live_run["dir"] / "certificate.pdf").write_bytes(pdf)


def test_pdf_reports_tampering(live_run, tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(live_run["db"]) as a, sqlite3.connect(db) as b:
        a.backup(b)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE ad_occurrences SET confidence_score = 0.99")
    row = _row(db)
    assert verify_row(row) is False
    pdf = certificate_for(db, row["occurrence_id"])
    assert pdf.startswith(b"%PDF-")


def test_pdf_missing_occurrence_raises_keyerror(live_run):
    with pytest.raises(KeyError):
        certificate_for(live_run["db"], 999_999)


def test_algeria_time_is_utc_plus_one():
    assert ALGERIA_TZ.utcoffset(None).total_seconds() == 3600
    assert algeria_iso(0.0) == "1970-01-01T01:00:00.000+01:00"
