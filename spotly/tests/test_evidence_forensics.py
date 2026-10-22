"""Phase 19: tamper-evident breach evidence clips (3.0 s pre/post-roll, SHA-256 sealed, range-streamed)."""
import json
import time
import wave
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spotly.analytics import certificate_authority as cert
from spotly.analytics import compliance_audit as ca
from spotly.analytics import contracts as ctr
from spotly.analytics import evidence_recorder as er
from spotly.app.api_server import create_app
from spotly.core.proof import ALGERIA_TZ
from spotly.database.schema import get_db

CH = "El-Heddaf TV"
T = datetime(2026, 10, 13, 20, 0, tzinfo=ALGERIA_TZ).timestamp()
SR22 = 22050
CONTRACT = dict(contract_id="CTR-2026-MOB-001", brand_name="Mobilis", channel_id=CH, net_rate_per_second=10.0,
                separation_window_seconds=180, valid_from="2026-10-01", valid_to="2026-10-31")


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    er.reset()
    monkeypatch.setattr(er, "EVIDENCE_DIR", tmp_path / "ev")
    yield
    er.flush(5)
    er.reset()


def broadcast(t_from, seconds, ad=(None, None), rng_seed=3):
    """22.05 kHz int16 broadcast: faint programme noise, with a loud 1 kHz creative between `ad` (wall epochs)."""
    n = int(seconds * SR22)
    rng = np.random.default_rng(rng_seed)
    x = rng.standard_normal(n) * 30.0                                     # ~ -60 dBFS programme floor
    if ad[0] is not None:
        a, b = int(round((ad[0] - t_from) * SR22)), int(round((ad[1] - t_from) * SR22))
        t = np.arange(b - a) / SR22
        x[max(a, 0):b] = 12000 * np.sin(2 * np.pi * 1000 * t)[max(0, -a):]
    return x.astype(np.int16)


def feed_stream(channel, t_from, pcm, chunk_s=1.0):
    step = int(chunk_s * SR22)
    for i in range(0, pcm.size, step):
        er.feed(channel, t_from + i / SR22, pcm[i:i + step])


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, er.CLIP_RATE)
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def loud(pcm, win=11):
    return np.abs(np.convolve(np.abs(pcm.astype(float)), np.ones(win) / win, "same")) > 3000


def test_clip_has_exact_three_second_padding_both_sides(tmp_path):
    a0, a1 = T + 20.0, T + 35.0                                           # a 15 s airing
    feed_stream(CH, T, broadcast(T, 60.0, (a0, a1)))
    ev = er.capture_breach_evidence(CH, a0, a1, "separation", occurrence_id=7)
    assert ev["path"].startswith("breaches/") and ev["path"].endswith(".wav") and ev["complete"]
    assert ev["clip_seconds"] == pytest.approx(15.0 + 2 * er.PAD_SECONDS, abs=1 / er.CLIP_RATE)
    pcm = read_wav(tmp_path / "ev" / ev["path"])
    on = np.flatnonzero(loud(pcm))
    assert on[0] / er.CLIP_RATE == pytest.approx(3.0, abs=0.002)          # 3.0 s of programme, then the creative
    assert (on[-1] + 1) / er.CLIP_RATE == pytest.approx(18.0, abs=0.002)   # creative ends 3.0 s before the clip does
    assert not loud(pcm[: int(2.99 * er.CLIP_RATE)]).any() and not loud(pcm[int(18.01 * er.CLIP_RATE):]).any()
    assert ev["sha256"] == er.sha256_file(tmp_path / "ev" / ev["path"])


def test_identical_audio_gives_identical_hash(tmp_path):
    feed_stream(CH, T, broadcast(T, 60.0, (T + 20, T + 35)))
    a = er.capture_breach_evidence(CH, T + 20, T + 35, "position", occurrence_id=1)
    b = er.capture_breach_evidence(CH, T + 20, T + 35, "position", occurrence_id=1)
    assert a["sha256"] == b["sha256"]
    c = er.capture_breach_evidence(CH, T + 21, T + 35, "position", occurrence_id=1)
    assert c["sha256"] != a["sha256"]


def test_candidate_waits_for_postroll_and_nothing_hits_disk_unflagged(tmp_path):
    a0, a1 = T + 10.0, T + 20.0
    pcm = broadcast(T, 40.0, (a0, a1))
    feed_stream(CH, T, pcm[: int(21 * SR22)])                              # detection fires 1 s after the airing ends
    er.note_airing(CH, 5, a0, a1)
    assert 5 not in er._STASH                                              # +3 s post-roll has not arrived yet
    feed_stream(CH, T + 21, pcm[int(21 * SR22):])
    assert 5 in er._STASH and er._STASH[5].covered == pytest.approx(16.0, abs=0.01)
    assert not (tmp_path / "ev").exists()                                  # zero disk bloat until a breach is flagged
    er.reset()                                                             # ring gone ... the stashed candidate is not
    feed_stream(CH, T, pcm)
    er.note_airing(CH, 6, a0, a1)
    er._STASH.pop(6)
    with pytest.raises(er.EvidenceUnavailable):
        er.capture_breach_evidence("Unknown TV", a0, a1, "position", occurrence_id=6)


def test_ring_is_bounded():
    feed_stream(CH, T, broadcast(T, 200.0))
    ring = er._RINGS[CH]
    held = sum(c[1].size for c in ring._chunks) / er.CLIP_RATE
    assert held <= er.RING_SECONDS + 1.0
    with pytest.raises(er.EvidenceUnavailable):                            # the airing scrolled out of the ring
        er.capture_breach_evidence(CH, T + 5, T + 20, "position")


def _db_with_breach(tmp_path):
    db = tmp_path / "c.db"
    with get_db(db) as conn:
        for ad_id, brand, role in ((1, "Mobilis", "self"), (2, "Djezzy", "competitor")):
            conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, "
                         "fp_algorithm, industry, ad_type) VALUES (?, ?, 'Spot', 30, x'00', 't', 'Telecom', ?)",
                         (ad_id, brand, role))
        for ad_id, start in ((1, T), (2, T + 90)):
            conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                         "confidence_score, human_verified) VALUES (?, ?, ?, ?, 30, 0.95, 0)",
                         (ad_id, f"live:{CH}", start, start + 30))
    return db


def test_breach_triggers_async_capture_binds_hash_and_links_certificate(tmp_path):
    db = _db_with_breach(tmp_path)
    ctr.ingest_contract(db, CONTRACT)
    feed_stream(CH, T - 10, broadcast(T - 10, 50.0, (T, T + 30)))
    er.note_airing(CH, 1, T, T + 30)
    t0 = time.perf_counter()
    assert ca.refresh_alerts(db, now=T + 3600) >= 1
    enqueue_ms = (time.perf_counter() - t0) * 1000
    assert er.flush(20)
    ev = er.evidence_for(db, 1)
    assert ev and ev["path"].startswith("breaches/") and len(ev["sha256"]) == 64
    assert er.verify_evidence(db, 1)["status"] == "PASS"
    assert er.evidence_for(db, 2) is None                                  # Djezzy's airing was never buffered
    log = [e for e in ctr.audit_log(db) if e["entity"] == "breach_evidence"]
    assert len(log) == 1 and json.loads(log[0]["snapshot"])["evidence_sha256"] == ev["sha256"]
    assert ctr.verify_audit_chain(db)
    rows = {r["occurrence_id"]: r for r in ca.compliance_report(db)["airings"]}
    assert rows[1]["evidence_available"] and rows[1]["evidence_sha256"] == ev["sha256"]
    assert any(a["evidence_available"] for a in ca.list_alerts(db) if a["occurrence_id"] == 1)
    ca.refresh_alerts(db, now=T + 3601)                                    # idempotent: no second clip / log entry
    er.flush(20)
    assert len([e for e in ctr.audit_log(db) if e["entity"] == "breach_evidence"]) == 1
    assert len(list((tmp_path / "ev" / "breaches").glob("*.wav"))) == 1
    pdf = cert.generate_airing_certificate_pdf(1, db)
    assert b"BREACH EVIDENCE CLIP" in pdf and ev["sha256"].encode() in pdf
    assert enqueue_ms < 500                                                # the alert loop does not wait for media work


def test_tamper_one_byte_invalidates_hash_and_blocks_playback(tmp_path):
    db = _db_with_breach(tmp_path)
    feed_stream(CH, T - 10, broadcast(T - 10, 50.0, (T, T + 30)))
    ev = er.capture_breach_evidence(CH, T, T + 30, "position", occurrence_id=1)
    er.bind_evidence(db, 1, ev, "position")
    path = tmp_path / "ev" / ev["path"]
    assert er.verify_evidence(db, 1)["status"] == "PASS"
    c = TestClient(create_app(db_path=db))
    assert c.get("/api/evidence/play/1").status_code == 200
    original = path.read_bytes()
    mutated = bytearray(original)
    mutated[len(mutated) // 2] ^= 0x01                                     # a single flipped bit in the audio payload
    path.write_bytes(bytes(mutated))
    res = er.verify_evidence(db, 1)
    assert res["status"] == "FAIL" and res["actual"] != res["recorded"] == ev["sha256"]
    assert c.get("/api/evidence/play/1").status_code == 409
    assert c.get("/api/evidence/download/1").status_code == 409
    path.write_bytes(original)
    assert er.verify_evidence(db, 1)["status"] == "PASS"
    path.unlink()
    assert er.verify_evidence(db, 1)["status"] == "MISSING" and c.get("/api/evidence/play/1").status_code == 404


def test_streaming_endpoints_support_byte_ranges_and_download(tmp_path, monkeypatch):
    db = _db_with_breach(tmp_path)
    feed_stream(CH, T - 10, broadcast(T - 10, 50.0, (T, T + 30)))
    ev = er.capture_breach_evidence(CH, T, T + 30, "separation", occurrence_id=1)
    er.bind_evidence(db, 1, ev, "separation")
    raw = (tmp_path / "ev" / ev["path"]).read_bytes()
    c = TestClient(create_app(db_path=db))
    full = c.get("/api/evidence/play/1")
    assert full.status_code == 200 and full.content == raw and full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-type"] == "audio/wav" and full.headers["x-evidence-sha256"] == ev["sha256"]
    part = c.get("/api/evidence/play/1", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == raw[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(raw)}"
    assert c.get("/api/evidence/play/1", headers={"Range": "bytes=-50"}).content == raw[-50:]
    assert c.get("/api/evidence/play/1", headers={"Range": f"bytes={len(raw) - 10}-"}).content == raw[-10:]
    assert c.get("/api/evidence/play/1", headers={"Range": f"bytes={len(raw) + 5}-"}).status_code == 416
    dl = c.get("/api/evidence/download/1")
    assert dl.status_code == 200 and dl.content == raw and "attachment" in dl.headers["content-disposition"]
    assert c.get("/api/evidence/play/2").status_code == 404                # no clip recorded for this airing
    air = {a["occurrence_id"]: a for a in c.get("/api/airings").json()["airings"]}
    assert air[1]["evidence_available"] and not air[2]["evidence_available"]
    monkeypatch.setenv("SPOTLY_EVIDENCE_TOKEN", "s3cret")                    # optional bearer protection
    assert c.get("/api/evidence/play/1").status_code == 401
    assert c.get("/api/evidence/play/1", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/api/evidence/play/1?token=s3cret").status_code == 200


def test_evidence_path_cannot_escape_breach_folder(tmp_path):
    with pytest.raises(ValueError):
        er.resolve_path("../spotly.db")
    with pytest.raises(ValueError):
        er.resolve_path("breaches/../../secret.wav")


def test_benchmark_capture_latency_and_footprint(tmp_path):
    feed_stream(CH, T, broadcast(T, 60.0, (T + 20, T + 50)))
    t0 = time.perf_counter()
    ev = er.capture_breach_evidence(CH, T + 20, T + 50, "position", occurrence_id=9)
    ms = (time.perf_counter() - t0) * 1000
    size = (tmp_path / "ev" / ev["path"]).stat().st_size
    ring_bytes = sum(c[1].nbytes for c in er._RINGS[CH]._chunks)
    print(f"\nEVIDENCE BENCHMARK  capture+hash {ms:.1f} ms | clip {ev['clip_seconds']:.1f} s = {size / 1024:.0f} KiB "
          f"({size / ev['clip_seconds'] / 1024:.1f} KiB/s) | ring {ring_bytes / 1e6:.2f} MB / channel")
    assert ms < 250 and size < 1_000_000 and ring_bytes < 2_000_000



def test_legacy_airings_are_retired_once_and_never_retried_or_logged(tmp_path, caplog, monkeypatch):
    """Phase 29.2: an airing with no clip whose audio left the ring is flagged UNAVAILABLE_LEGACY and leaves the retry queue."""
    import logging
    db = _db_with_breach(tmp_path)
    monkeypatch.setattr(er, "GONE_AFTER_S", -1.0)                          # every unbuffered airing is past the ring
    caplog.set_level(logging.DEBUG, logger="spotly.evidence")
    er.submit_breach(db, 2, CH, T + 90, T + 120, "UNDER_DELIVERY")
    assert er.flush(20)
    assert er.is_unavailable(db, 2) and not er.is_unavailable(db, 1)
    assert not [r for r in caplog.records if r.levelno >= logging.INFO and "no evidence clip" in r.getMessage()]
    pending = er._PENDING
    er.submit_breach(db, 2, CH, T + 90, T + 120, "UNDER_DELIVERY")        # second time: not even enqueued
    assert er._PENDING == pending and er._Q.empty()
    with get_db(db) as conn:                                              # the startup sweep catches the rest in one UPDATE
        assert conn.execute("SELECT evidence_status FROM ad_occurrences WHERE occurrence_id = 1").fetchone()[0] is None
    monkeypatch.setattr(er, "GONE_AFTER_S", 60.0)
    assert er.retire_legacy(db, now=T + 100000) == 1                      # occurrence 1 (2 is already retired)
    assert er.retire_legacy(db, now=T + 100000) == 0                      # idempotent
