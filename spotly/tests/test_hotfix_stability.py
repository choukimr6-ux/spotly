"""Hotfix 2026-10-21: crash-proof certificates, name sanitisation, transcript quality gate, catalogue consolidation."""
import json
import sqlite3
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spotly.analytics import brand_extractor as be
from spotly.analytics import catalog_cleanup as cc
from spotly.analytics import certificate_authority as cert
from spotly.analytics import reference_matcher as rm
from spotly.analytics import transcriber as tr
from spotly.app.api_server import create_app
from spotly.core import discovery as disc
from spotly.core.discovery_report import list_assets
from spotly.core.proof import ALGERIA_TZ, algeria_iso, compute_verification_hash, verify_row
from spotly.data import reference_db as rdb
from spotly.database.schema import get_db
from spotly.tests.test_api_server import FakeChannel, FakeWorker
from spotly.tests.test_reference_matcher import synth

T0 = datetime(2026, 10, 20, 21, 0, 0, tzinfo=ALGERIA_TZ).timestamp()


def make_db(path, brand="Amiral", campaign="Spot", frame=None):
    with get_db(path) as conn:
        rdb.ensure_schema(conn)
        conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                     "industry) VALUES (1, ?, ?, 30, x'00', 't', 'Food & FMCG')", (brand, campaign))
        ts = algeria_iso(T0)
        h = compute_verification_hash(1, "live:El-Heddaf TV", ts, T0 + 30, 30.0, 0.97, None)
        conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                     "confidence_score, evidence_frame_path, verification_hash, algeria_timestamp) "
                     "VALUES (1, 'live:El-Heddaf TV', ?, ?, 30.0, 0.97, ?, ?, ?)", (T0, T0 + 30, frame, h, ts))
    return path


# ====================================================================================== certificates never crash
@pytest.mark.parametrize("case,sql", [
    ("markup in names", "UPDATE reference_ads SET brand_name='A <b>& co', campaign_name='Promo <i & \U0001F525'"),
    ("arabic names", "UPDATE reference_ads SET brand_name='\u0623\u0645\u064a\u0631\u0627\u0644', campaign_name='\u0643\u0644 \u064a\u0646\u0637\u0648\u0649'"),
    ("text drift", "UPDATE ad_occurrences SET clock_drift_seconds='abc', clock_drift_margin_s='x', clock_drift_basis='pts-relative'"),
    ("infinite drift", "UPDATE ad_occurrences SET clock_drift_seconds=1e999, clock_drift_margin_s=1e999"),
    ("absurd start", "UPDATE ad_occurrences SET airing_start=1e15, airing_end=1e15+30"),
    ("missing evidence file", "UPDATE ad_occurrences SET evidence_frame_path='C:/definitely/not/here.png'"),
    ("null stamp and hash", "UPDATE ad_occurrences SET algeria_timestamp=NULL, verification_hash=NULL"),
    ("garbage phash", "INSERT INTO reference_signatures (asset_id, brand_name, nominal_duration, audio_fingerprint_blob, "
                      "visual_p_hash, ad_id) VALUES ('X', 'B', 30, x'00', 'zzzz', 1)"),
])
def test_certificate_renders_whatever_the_row_holds(tmp_path, case, sql):
    db = make_db(tmp_path / "c.db")
    with get_db(db) as conn:
        conn.execute(sql)
    pdf = cert.generate_airing_certificate_pdf(1, db)
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF"), case


def test_corrupt_evidence_image_is_skipped_not_fatal(tmp_path):
    bad = tmp_path / "half.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n-- writer was interrupted --")
    pdf = cert.generate_airing_certificate_pdf(1, make_db(tmp_path / "c.db", frame=str(bad)))
    assert pdf.startswith(b"%PDF-") and b"Spotly" in pdf


def test_unknown_airing_is_still_a_clean_keyerror(tmp_path):
    with pytest.raises(KeyError):
        cert.generate_airing_certificate_pdf(999, make_db(tmp_path / "c.db"))


def test_simplified_certificate_is_issued_when_the_layout_engine_fails(tmp_path, monkeypatch):
    db = make_db(tmp_path / "c.db")
    monkeypatch.setattr(cert, "render", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("layout exploded")))
    pdf = cert.generate_airing_certificate_pdf(1, db)
    assert pdf.startswith(b"%PDF-") and b"simplified" in pdf and b"Amiral" in pdf


def test_endpoint_never_answers_500(tmp_path, monkeypatch):
    db = make_db(tmp_path / "c.db")
    app = create_app(db, worker_factory=FakeWorker, channel_loader=lambda: [FakeChannel("TV2", "http://x/index.m3u8")])
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/api/certificate/1").status_code == 200
        assert c.get("/api/certificate/999").status_code == 404
        monkeypatch.setattr(cert, "_generate_full", lambda *a, **k: (_ for _ in ()).throw(MemoryError("boom")))
        r = c.get("/api/certificate/1")
        assert r.status_code == 200 and r.content.startswith(b"%PDF-")                # the simplified certificate
        monkeypatch.setattr(cert, "generate_airing_certificate_pdf", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
        r = c.get("/api/certificate/1")
        assert r.status_code == 200 and r.content.startswith(b"%PDF-")                # endpoint-level net as well


# ====================================================================================== names
def test_stitched_fragment_brand_is_repaired():
    assert be.repair_stitched_brand("MON", "miral NOW EXCELLENCE MON") == ("Amiral", "NOW EXCELLENCE MON")
    assert be.repair_stitched_brand("Amiral", "miral NOW") == ("Amiral", "miral NOW")      # a registry brand is never rewritten
    assert be.repair_stitched_brand("Acme", "spring sale") == ("Acme", "spring sale")      # nothing to repair


@pytest.mark.parametrize("raw,clean", [
    ("Auto-discovered ELHEDDAF_AD_117", ""), ("Auto-resolved ELHEDDAF_AD_013", ""),
    ("Djezzy \u2013 Auto-resolved ELHEDDAF_AD_013", "Djezzy"), ("ELHEDDAF_AD_004", ""), ("MASTER_0007_djezzy", ""),
    ("Ramadan 2026", "Ramadan 2026"), ("auto-discovered Ramadan", "Ramadan")])
def test_internal_identifiers_never_survive(raw, clean):
    assert be.strip_internal(raw) == clean


def test_display_label_is_clean_and_never_repeats_the_brand():
    assert be.display_label("amiral", "Auto-discovered ELHEDDAF_AD_117", 45.0) == "Amiral \u00b7 45 s spot"
    assert be.display_label("Amiral", "Amiral Ramadan") == "Amiral \u00b7 Ramadan"
    assert be.display_label("Djezzy", None, 29.95) == "Djezzy \u00b7 30 s spot"
    assert be.canonical_brand("AMIRAL ") == "Amiral" and be.canonical_brand("unknown co") == "unknown co"


def test_promote_names_are_sanitised_and_unique(tmp_path):
    db = make_db(tmp_path / "n.db", brand="Amiral", campaign="45 s spot")
    assert disc.clean_names(db, "MON", "miral NOW EXCELLENCE MON", 45.0) == ("Amiral", "NOW EXCELLENCE MON")
    assert disc.clean_names(db, "amiral", "Auto-discovered ELHEDDAF_AD_117", 45.0) == ("Amiral", "45 s spot #2")
    assert disc.clean_names(db, "Djezzy", "", 29.9) == ("Djezzy", "30 s spot")
    with pytest.raises(ValueError):
        disc.clean_names(db, "Auto-resolved ELHEDDAF_AD_001", "", 30)


def test_user_facing_endpoints_show_clean_names(tmp_path):
    db = make_db(tmp_path / "u.db", brand="MON", campaign="Auto-discovered ELHEDDAF_AD_117")
    app = create_app(db, worker_factory=FakeWorker, channel_loader=lambda: [FakeChannel("TV2", "http://x/index.m3u8")])
    with TestClient(app) as c:
        a = c.get("/api/airings?verify=false").json()["airings"][0]
        assert a["campaign"] == "30 s spot" and "ELHEDDAF" not in json.dumps(c.get("/api/ads").json())


# ====================================================================================== transcript gate
SPEECH = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645 \u0641\u064a \u0639\u0631\u0636 \u0631\u0645\u0636\u0627\u0646 \u0627\u0644\u062c\u062f\u064a\u062f \u0645\u0646 \u0623\u0645\u064a\u0631\u0627\u0644"


@pytest.mark.parametrize("text,logprob,shown", [
    (SPEECH, None, True), (SPEECH, -0.4, True), (SPEECH, -0.9, False),                    # poor mean logprob
    ("la la la la la la la la la la", None, False), ("a b c d e f g h", None, False),      # loops / fragments
    ("dah dah dah dah dah dah dah", None, False), ("oui", None, False), ("", None, False),
    ("one 2 3 4 5 6 7", None, False)])
def test_transcript_gate(text, logprob, shown):
    assert tr.assess_transcript(text, logprob)[0] is shown


def test_asset_listing_hides_incoherent_text_and_cleans_labels(tmp_path):
    db = make_db(tmp_path / "t.db", brand="Amiral", campaign="Auto-discovered ELHEDDAF_AD_117")
    with get_db(db) as conn:
        for aid, text, lp in (("A1", SPEECH, -0.3), ("A2", "x y z a b c d e f", None), ("A3", SPEECH, -1.2), ("A4", "", None)):
            conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, status, label, "
                         "promoted_ad_id, transcript, transcribed_at, transcript_logprob) VALUES (?, 'Ch', 30, x'00', "
                         "'classified', 'Amiral \u2013 Auto-discovered ELHEDDAF_AD_117', 1, ?, '2026-10-20', ?)", (aid, text, lp))
    rows = {r["asset_id"]: r for r in list_assets(db)}
    assert rows["A1"]["transcript_quality"] == "speech" and rows["A1"]["transcript"] == SPEECH
    for k in ("A2", "A3"):
        assert rows[k]["transcript_quality"] == "soundtrack" and rows[k]["transcript"] == ""
        assert rows[k]["transcript_display"] == "Audio: Commercial Soundtrack & Sound Effects"
    assert rows["A4"]["transcript_quality"] == "none"
    assert all("ELHEDDAF" not in r["label"] and "Auto-" not in r["label"] for r in rows.values())


def test_mean_logprob_is_persisted_by_the_batch(tmp_path):
    db = make_db(tmp_path / "p.db")
    with get_db(db) as conn:
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint) VALUES ('A', 'Ch', 30, x'00')")
    tr.save_result(db, "A", "Ch", SPEECH, "ar", "base", avg_logprob=-0.42)
    with get_db(db) as conn:
        assert conn.execute("SELECT transcript_logprob FROM discovered_assets WHERE asset_id='A'").fetchone()[0] == pytest.approx(-0.42)


# ====================================================================================== consolidation
@pytest.fixture
def messy(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "DATA_DIR", tmp_path)
    monkeypatch.setattr("spotly.analytics.auto_resolver.DATA_DIR", tmp_path)
    db = tmp_path / "m.db"
    master, other = synth(31, 30.0), synth(32, 30.0)
    blob = lambda x: rdb.pack_blob(*rm.signature_from_audio(x))                        # noqa: E731
    with get_db(db) as conn:
        rdb.ensure_schema(conn)
        for ad_id, brand, camp, dur in ((1, "MON", "miral NOW EXCELLENCE MON", 30), (2, "Amiral", "Auto-discovered X_AD_2", 15),
                                        (3, "amiral", "Auto-resolved X_AD_3", 30), (4, "Dayzy", "Auto-resolved X_AD_4", 30)):
            conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                         "industry) VALUES (?, ?, ?, ?, x'00', 't', 'Food & FMCG')", (ad_id, brand, camp, dur))
        rdb.upsert_signature(conn, "M", "MON", "Food", 30.0, blob(master), None, ad_id=1)
        rdb.upsert_signature(conn, "C", "Amiral", "Food", 15.0, blob(master[:15 * rm.SR]), None, ad_id=2)
        rdb.upsert_signature(conn, "D", "Dayzy", "Food", 30.0, blob(other), None, ad_id=4)
        ts = algeria_iso(T0)
        for ad in (1, 2):
            h = compute_verification_hash(ad, "live:Ch", ts, T0 + ad * 100 + 30, 30.0, 0.97, None)
            conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                         "confidence_score, verification_hash, algeria_timestamp) VALUES (?, 'live:Ch', ?, ?, 30, 0.97, ?, ?)",
                         (ad, T0 + ad * 100, T0 + ad * 100 + 30, h, ts))
    return db


def test_dry_run_changes_nothing(messy):
    with get_db(messy) as conn:
        before = [tuple(r) for r in conn.execute("SELECT * FROM reference_ads")]
    rep = cc.consolidate_catalog(messy, dry_run=True)
    assert rep["renamed_ads"] and rep["backup"] is None
    with get_db(messy) as conn:
        assert [tuple(r) for r in conn.execute("SELECT * FROM reference_ads")] == before


def test_consolidation_cleans_names_links_families_and_keeps_every_seal(messy, tmp_path):
    rep = cc.consolidate_catalog(messy)
    assert rep["backup"] and (tmp_path / "backups").exists() and rep["audit_file"]
    with get_db(messy) as conn:
        ads = {r["ad_id"]: (r["brand_name"], r["campaign_name"]) for r in conn.execute("SELECT * FROM reference_ads")}
        sigs = {r["asset_id"]: (r["brand_name"], r["master_asset_id"], r["cut_type"]) for r in conn.execute("SELECT * FROM reference_signatures")}
        rows = [dict(r) for r in conn.execute("SELECT o.*, r.brand_name FROM ad_occurrences o JOIN reference_ads r USING (ad_id)")]
    assert ads[1] == ("Amiral", "30 s spot")                                           # 'MON' + 'miral ...' repaired
    assert ads[3] == ("Amiral", "30 s spot #2") and ads[4] == ("Dayzy", "30 s spot")
    assert ads[2] == ("Amiral", "30 s spot [CUT_DOWN]")                                # child of the parent campaign
    assert sigs["C"] == ("Amiral", "M", "CUT_DOWN") and sigs["M"][0] == "Amiral" and sigs["D"][1] is None
    assert all("Auto-" not in c and "_AD_" not in c for _, c in ads.values())
    assert all(verify_row(r) is not False for r in rows)                                # airing seals untouched
    assert len(rows) == 2


def test_consolidation_is_idempotent(messy):
    cc.consolidate_catalog(messy)
    again = cc.consolidate_catalog(messy)
    assert again["renamed_ads"] == [] and again["brand_fixes"] == []


def test_different_creatives_of_one_brand_are_not_merged(messy):
    rep = cc.consolidate_catalog(messy)
    assert not any(f["asset_id"] == "D" or f["master"] == "D" for f in rep["families"])
