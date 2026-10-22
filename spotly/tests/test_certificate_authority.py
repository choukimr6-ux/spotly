"""Phase 17: tamper-evident proof-of-performance certificates - content, speed, memory, seal, batch export, API."""
import base64
import hashlib
import io
import re
import time
import tracemalloc
import zipfile
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from spotly.analytics import certificate_authority as cert
from spotly.analytics import compliance_audit as ca
from spotly.analytics.reference_matcher import phash_image
from spotly.core.proof import ALGERIA_TZ, algeria_iso, compute_verification_hash
from spotly.data import reference_db as rdb
from spotly.database.schema import get_db

T0 = datetime(2026, 10, 20, 20, 0, 0, tzinfo=ALGERIA_TZ).timestamp()


def _frame(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)).save(path, "JPEG", quality=90)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "c.db"
    shot, ev = tmp_path / "shot.jpg", tmp_path / "ev.jpg"
    _frame(shot, 1)
    _frame(ev, 2)
    with get_db(path) as conn:
        rdb.ensure_schema(conn)
        for ad_id, brand in ((1, "Djezzy"), (2, "Mobilis"), (3, "Cevital")):
            conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                         "industry) VALUES (?, ?, 'Camp', 30, x'00', 't', ?)", (ad_id, brand, "Food" if ad_id == 3 else "Telecom"))
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, status, packshot_path, "
                     "promoted_ad_id) VALUES ('A1', 'El-Heddaf TV', 30, x'00', 'classified', ?, 1)", (str(shot),))
        with Image.open(shot) as im:
            ph = phash_image(im)
        conn.execute("INSERT INTO reference_signatures (asset_id, brand_name, nominal_duration, audio_fingerprint_blob, "
                     "visual_p_hash, ad_id) VALUES ('A1', 'Djezzy', 30, x'00', ?, 1)", (ph,))
        # one break: Djezzy, Mobilis (same sector, 20 s later -> separation breach for Djezzy), Cevital
        for i, (ad_id, start, ev_path) in enumerate(((1, T0, str(ev)), (2, T0 + 38, None), (3, T0 + 76, None))):
            ts = algeria_iso(start)
            h = compute_verification_hash(ad_id, "live:El-Heddaf TV", ts, start + 30, 30.0, 0.99, ev and ev_path and open(ev_path, "rb").read())
            conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                         "confidence_score, evidence_frame_path, verification_hash, algeria_timestamp) "
                         "VALUES (?, 'live:El-Heddaf TV', ?, ?, 30.0, 0.99, ?, ?, ?)", (ad_id, start, start + 30, ev_path, h, ts))
    ca.save_contracts(path, [dict(brand="Djezzy", campaign="*", booked_position="LAST", competitor_separation_seconds=180,
                                  separation_credit=0.5)])
    return path


def test_certificate_content_and_seal(db):
    pdf = cert.generate_airing_certificate_pdf(1, db)
    assert pdf.startswith(b"%PDF-") and len(re.findall(rb"/Type /Page\b", pdf)) == 1          # single page
    rep = cert.verify_certificate_pdf(pdf, db)
    assert rep["status"] == "PASS" and all(rep["checks"].values()), rep
    p = rep["payload"]
    assert list(sorted(p)) == sorted(cert.PAYLOAD_FIELDS)
    assert (p["airing_id"], p["channel"], p["brand"], p["exact_duration"], p["matched_confidence"]) == ("1", "El-Heddaf TV", "Djezzy", "30.000", "0.9900")
    assert p["timestamp_cet"].endswith("+01:00") and re.fullmatch(r"\d+\.\d\d", p["calculated_cost_dzd"])
    # the seal is exactly SHA-256 over the seven specified fields
    body = cert.canonical_payload(p)
    assert hashlib.sha256(body).hexdigest() == rep["sha256"] and b'"calculated_cost_dzd"' in body
    for text in (b"Regulatory Broadcast Verification", b"EXACT MEASURED DURATION", b"POD POSITION", b"CLUTTER INDEX",
                 b"PERCEPTUAL HASH", b"GROSS VALUE", b"APPLIED SLA", b"PASS / VERIFIED", b"NOT a measurement of audience",
                 rep["sha256"].encode()):
        assert text in pdf, text
    assert b"Packshot frame not available" not in pdf
    # Djezzy: LAST booked but airs FIRST -> position breach; Mobilis 20 s after in same sector -> separation breach
    assert pdf.count(b"BREACH") >= 1 and b"Separation breach" in pdf and b"Position breach" in pdf


def test_pod_position_and_unpriced_file_scan(db):
    assert b"FIRST" in cert.generate_airing_certificate_pdf(1, db)
    assert b"LAST" in cert.generate_airing_certificate_pdf(3, db)
    with get_db(db) as conn:
        conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                     "confidence_score) VALUES (3, 'rec.mp4', 12.0, 42.0, 30.0, 0.97)")
        rid = conn.execute("SELECT MAX(occurrence_id) FROM ad_occurrences").fetchone()[0]
    pdf = cert.generate_airing_certificate_pdf(rid, db)                       # no clock time, no hash, no packshot
    assert cert.verify_certificate_pdf(pdf, db)["status"] == "PASS" and b"into recorded file" in pdf
    with pytest.raises(KeyError):
        cert.generate_airing_certificate_pdf(99999, db)


def test_generation_latency_under_650_ms_and_memory_under_35_mb(db):
    cert.generate_airing_certificate_pdf(1, db)                               # warm: imports, fonts, rate card file
    times = []
    for i in (1, 2, 3, 1, 2, 3):
        t = time.perf_counter()
        cert.generate_airing_certificate_pdf(i, db)
        times.append(time.perf_counter() - t)
    assert max(times) < 0.65, times        # Phase 29.4: headroom for CPU contention from the live ingest engine
    tracemalloc.start()
    cert.generate_airing_certificate_pdf(1, db)
    peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
    tracemalloc.stop()
    assert peak < 35, peak
    print(f"\n[phase17 bench] certificate latency mean {1000 * sum(times) / len(times):.0f} ms, max {1000 * max(times):.0f} ms; "
          f"peak allocation {peak:.1f} MB")


def _keyword_span(pdf):
    m = cert._KW.search(pdf)
    return m.start(), m.end()


def test_every_single_byte_tamper_of_the_seal_fails(db):
    pdf = cert.generate_airing_certificate_pdf(1, db)
    assert cert.verify_certificate_pdf(pdf)["status"] == "PASS"
    a, b = _keyword_span(pdf)
    start = a + len(b"/Keywords (" + cert.CERT_VERSION.encode()) + 1
    fails = 0
    for pos in range(start, b - 1):                                          # payload b64, sha256 and hmac, byte by byte
        t = bytearray(pdf)
        t[pos] = t[pos] ^ 0x01 if t[pos] ^ 0x01 not in (0x28, 0x29) else t[pos] ^ 0x02
        rep = cert.verify_certificate_pdf(bytes(t))
        assert rep["status"] == "FAIL", f"tamper at byte {pos} was not detected"
        fails += 1
    assert fails > 400


def test_tampered_visible_hash_and_stripped_seal_fail(db):
    pdf = cert.generate_airing_certificate_pdf(1, db)
    digest = cert.verify_certificate_pdf(pdf)["sha256"].encode()
    first, second = pdf.index(digest), pdf.rindex(digest)
    assert first != second
    t = bytearray(pdf)
    t[second] ^= 0x01                                                       # the hash printed on the page
    assert cert.verify_certificate_pdf(bytes(t))["status"] == "FAIL"
    assert cert.verify_certificate_pdf(pdf.replace(b"/Keywords", b"/Xeywords"))["status"] == "FAIL"
    assert cert.verify_certificate_pdf(b"%PDF-1.4 nothing")["status"] == "FAIL"


def test_forger_who_recomputes_the_hash_still_fails_the_hmac(db):
    pdf = cert.generate_airing_certificate_pdf(1, db)
    s = cert.extract_seal(pdf)
    p = dict(s["payload"], calculated_cost_dzd="1.00")                      # lower the cost, re-hash like a forger
    body = cert.canonical_payload(p)
    digest = hashlib.sha256(body).hexdigest()
    forged = pdf.replace(base64.b64encode(s["body"]), base64.b64encode(body)).replace(s["sha256"].encode(), digest.encode())
    rep = cert.verify_certificate_pdf(forged)
    assert rep["checks"]["sha256_matches_payload"] and rep["checks"]["hmac_valid"] is False and rep["status"] == "FAIL"
    assert cert.verify_certificate_pdf(pdf, key=b"another installation")["checks"]["hmac_valid"] is False


def test_record_edited_after_issue_is_detected(db):
    pdf = cert.generate_airing_certificate_pdf(1, db)
    assert cert.verify_certificate_pdf(pdf, db)["status"] == "PASS"
    with get_db(db) as conn:
        conn.execute("UPDATE ad_occurrences SET confidence_score = 0.5 WHERE occurrence_id = 1")
    rep = cert.verify_certificate_pdf(pdf, db)
    assert rep["status"] == "FAIL" and rep["checks"]["matches_current_record"] is False
    assert any("matched_confidence" in n for n in rep["notes"])
    assert b"FAIL" in cert.generate_airing_certificate_pdf(1, db)           # a fresh one reports the broken detection hash


def test_evidence_file_swapped_marks_certificate_fail(db, tmp_path):
    _frame(tmp_path / "ev.jpg", 77)                                          # replace the evidence image after detection
    pdf = cert.generate_airing_certificate_pdf(1, db)
    assert b"Do not rely on this certificate" in pdf


def test_batch_export_zip(db):
    data, info = cert.export_batch_zip(db)
    assert info == dict(count=3, matched=3, truncated=False)
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    assert sorted(n for n in names if n.endswith(".pdf")) == [f"SPOTLY-certificate-{i:06d}.pdf" for i in (1, 2, 3)]
    manifest = z.read("MANIFEST.csv").decode().splitlines()
    assert len(manifest) == 4 and manifest[0].startswith("airing_id,brand")
    for n in names:
        if n.endswith(".pdf"):
            pdf = z.read(n)
            assert cert.verify_certificate_pdf(pdf, db)["status"] == "PASS"
            assert hashlib.sha256(pdf).hexdigest() in "\n".join(manifest)
    assert cert.export_batch_zip(db, brand="mobilis")[1]["count"] == 1
    assert cert.export_batch_zip(db, start_date="2026-10-20", end_date="2026-10-20")[1]["count"] == 3
    assert cert.export_batch_zip(db, start_date="2026-10-21")[1]["count"] == 0
    assert cert.export_batch_zip(db, limit=2)[1] == dict(count=2, matched=3, truncated=True)


def test_api_certificate_batch_and_verify(db):
    from fastapi.testclient import TestClient

    from spotly.app.api_server import create_app
    with TestClient(create_app(db)) as c:
        r = c.get("/api/certificate/1")
        assert r.status_code == 200 and r.content.startswith(b"%PDF-") and cert.verify_certificate_pdf(r.content, db)["status"] == "PASS"
        assert c.get("/api/certificate/4242").status_code == 404
        z = c.get("/api/certificates/export-batch", params=dict(start_date="2026-10-20", end_date="2026-10-20", brand="Djezzy"))
        assert z.status_code == 200 and z.headers["content-type"] == "application/zip" and z.headers["x-spotly-certificates"] == "1"
        assert zipfile.ZipFile(io.BytesIO(z.content)).testzip() is None
        assert c.get("/api/certificates/export-batch", params=dict(brand="Nobody")).status_code == 404
        assert c.get("/api/certificates/export-batch", params=dict(start_date="20/10/2026")).status_code == 422
        good = c.post("/api/certificates/verify", json=dict(pdf_b64=base64.b64encode(r.content).decode())).json()
        assert good["status"] == "PASS"
        bad = bytearray(r.content)
        a, b = _keyword_span(r.content)
        bad[a + 40] ^= 1
        assert c.post("/api/certificates/verify", json=dict(pdf_b64=base64.b64encode(bytes(bad)).decode())).json()["status"] == "FAIL"


def test_dashboard_wires_the_certificate_actions():
    from pathlib import Path
    root = Path(cert.__file__).parents[1] / "app"
    js, html = (root / "static/app.js").read_text(encoding="utf-8"), (root / "templates/index.html").read_text(encoding="utf-8")
    assert "📄 Certificate (PDF)" in js and "/api/certificate/${a.occurrence_id}" in js
    assert "/api/certificates/export-batch" in html and 'id="cb-go"' in html
