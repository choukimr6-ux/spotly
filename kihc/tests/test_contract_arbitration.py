"""Phase 18: dual-track contract ingestion & dispute-proof compliance (Track A verified / Track B illustrative)."""
import base64
import json
import sqlite3
from datetime import datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from kihc.analytics import certificate_authority as cert
from kihc.analytics import compliance_audit as ca
from kihc.analytics import contracts as ctr
from kihc.app.api_server import create_app
from kihc.core.proof import ALGERIA_TZ
from kihc.database.schema import get_db

CH = "El-Heddaf TV"
T = datetime(2026, 10, 13, 20, 0, tzinfo=ALGERIA_TZ).timestamp()        # Prime Time, CET
CONTRACT = dict(contract_id="CTR-2026-MOB-001", brand_name="Mobilis", channel_id=CH, booked_position="RUN_OF_BREAK",
                separation_window_seconds=180, net_rate_per_second=10.0, demotion_penalty_pct=0.30,
                separation_breach_penalty_pct=1.0, valid_from="2026-10-01", valid_to="2026-10-31")


def _ad(conn, ad_id, brand, sector, role="competitor"):
    conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                 "industry, ad_type) VALUES (?, ?, 'Spot', 30, x'00', 't', ?, ?)", (ad_id, brand, sector, role))


def _air(conn, ad_id, start, seconds=30.0, channel=CH):
    conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                 "confidence_score, human_verified) VALUES (?, ?, ?, ?, ?, 0.95, 0)",
                 (ad_id, f"live:{channel}", start, start + seconds, seconds))


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "c.db"
    with get_db(path) as conn:
        _ad(conn, 1, "Mobilis", "Telecom", role="self")
        _ad(conn, 2, "Djezzy", "Telecom")
        _ad(conn, 3, "Bellat", "Cold cuts")
        _air(conn, 1, T)                      # Mobilis 20:00:00-20:00:30
        _air(conn, 2, T + 90)                 # Djezzy 60 s after it ended: inside any 180 s window
    return path


def _mob(rep):
    return next(r for r in rep["airings"] if r["brand"] == "Mobilis")


def test_same_airing_verified_contract_vs_illustrative_estimate(db):
    ca.save_contracts(db, [dict(brand="Mobilis", competitor_separation_seconds=180, separation_credit=0.5)])
    before = _mob(ca.compliance_report(db))
    assert before["audit_status"] == "ILLUSTRATIVE_MARKET_ESTIMATE" and before["contract_id"] is None
    assert before["separation"] == "breach" and before["separation_credit"] > 0
    card_credit = before["separation_credit"]

    ctr.ingest_contract(db, CONTRACT, origin="test")
    rep = ca.compliance_report(db)
    after = _mob(rep)
    assert after["audit_status"] == "VERIFIED_CONTRACT_AUDIT" and after["contract_id"] == "CTR-2026-MOB-001"
    assert after["estimated_net_spend"] == Decimal("300.00")                     # 10 DZD/s x 30 s, not the card
    assert after["separation"] == "breach" and after["separation_credit"] == Decimal("300.00")
    assert after["separation_credit"] != card_credit
    djezzy = next(r for r in rep["airings"] if r["brand"] == "Djezzy")
    assert djezzy["audit_status"] == "ILLUSTRATIVE_MARKET_ESTIMATE"              # same slot, no contract -> Track B
    assert rep["kpis"]["verified_contract_airings"] == 1 and rep["kpis"]["illustrative_airings"] == 1


def test_contract_demotion_penalty_and_alert_reference(db):
    with get_db(db) as conn:
        _air(conn, 3, T - 30)                                        # Bellat, then Mobilis, then Djezzy: Mobilis is MIDDLE
    ctr.ingest_contract(db, dict(CONTRACT, booked_position="FIRST_IN_POD", separation_window_seconds=0))
    row = _mob(ca.compliance_report(db))
    assert row["position_compliance"] == "breach" and row["booked_position"] == "FIRST"
    assert row["position_credit"] == Decimal("90.00")                           # 300 x 30 %
    ca.refresh_alerts(db, now=T + 3600)
    msgs = [a["message"] for a in ca.list_alerts(db)]
    assert any("CTR-2026-MOB-001" in m for m in msgs)
    assert "CTR-2026-MOB-001" in ca.claim_sheet_csv(ca.compliance_report(db))


def test_expiry_boundaries_are_half_open(db):
    ctr.ingest_contract(db, dict(CONTRACT, valid_from="2026-10-13T20:00:00+01:00", valid_to="2026-10-13T20:01:00+01:00"))
    with get_db(db) as conn:
        _air(conn, 1, T + 3600)
        _air(conn, 1, T + 59.0, 1.0)
    active = ctr.load_active(db)
    m = lambda t: ctr.match_contract(active, CH, "mobilis", t)               # noqa: E731  (case-insensitive)
    assert m(T)["contract_id"] == "CTR-2026-MOB-001"                         # valid_from inclusive
    assert m(T + 59.9) and m(T + 60) is None                                  # valid_to exclusive
    assert m(T - 1) is None
    ctr.ingest_contract(db, dict(CONTRACT, status="EXPIRED", valid_from="2026-10-13T20:00:00+01:00",
                                 valid_to="2026-10-13T20:01:00+01:00"))
    assert ctr.match_contract(ctr.load_active(db), CH, "Mobilis", T) is None  # EXPIRED never binds
    assert _mob(ca.compliance_report(db))["audit_status"] == "ILLUSTRATIVE_MARKET_ESTIMATE"


def test_date_only_valid_to_covers_whole_day():
    f = ctr._normalise(dict(CONTRACT, valid_from="2026-10-13", valid_to="2026-10-13"))
    assert ctr.to_epoch(f["valid_to"]) - ctr.to_epoch(f["valid_from"]) == 86400


def test_conflicting_date_ranges_rejected_but_adjacent_allowed(db):
    ctr.ingest_contract(db, CONTRACT)
    with pytest.raises(ctr.ContractError, match="conflicting date range"):
        ctr.ingest_contract(db, dict(CONTRACT, contract_id="CTR-2026-MOB-002", valid_from="2026-10-20", valid_to="2026-11-10"))
    ctr.ingest_contract(db, dict(CONTRACT, contract_id="CTR-2026-MOB-003", valid_from="2026-11-01", valid_to="2026-11-30"))
    ctr.ingest_contract(db, dict(CONTRACT, contract_id="CTR-2026-DJZ-001", brand_name="Djezzy"))   # other brand
    assert {c["contract_id"] for c in ctr.list_contracts(db)} == {"CTR-2026-MOB-001", "CTR-2026-MOB-003", "CTR-2026-DJZ-001"}
    with pytest.raises(ctr.ContractError, match="net_rate_per_second"):
        ctr.ingest_contract(db, dict(CONTRACT, contract_id="X", net_rate_per_second=0))


def test_audit_log_is_append_only_and_hash_chained(db):
    ctr.ingest_contract(db, CONTRACT, origin="alice")
    ctr.ingest_contract(db, dict(CONTRACT, net_rate_per_second=12.5), origin="bob")
    ctr.ingest_contract(db, dict(CONTRACT, status="EXPIRED"), origin="bob")
    ctr.log_change(db, "rate_card", "ILLUSTRATIVE-v2", "UPDATE", {"prime_time": 99}, origin="carol")
    log = ctr.audit_log(db, "CTR-2026-MOB-001")
    assert [(r["action"], r["version"]) for r in log] == [("CREATE", 1), ("UPDATE", 2), ("EXPIRE", 3)]
    assert json.loads(log[0]["snapshot"])["net_rate_per_second"] == 10.0          # old state still reproducible
    assert log[0]["origin_hash"] != log[1]["origin_hash"] and len(log[0]["origin_hash"]) == 64
    assert ctr.verify_audit_chain(db)
    with get_db(db) as conn:
        for sql in ("UPDATE contract_audit_log SET action = 'CREATE' WHERE id = 1", "DELETE FROM contract_audit_log"):
            with pytest.raises(sqlite3.DatabaseError, match="append-only"):
                conn.execute(sql)
    conn = sqlite3.connect(db)                                                    # tamper behind the triggers' back
    conn.execute("DROP TRIGGER contract_audit_log_no_update")
    conn.execute("UPDATE contract_audit_log SET snapshot = '{}' WHERE id = 1")
    conn.commit()
    conn.close()
    assert not ctr.verify_audit_chain(db)                                         # ... and the chain exposes it


def test_ingest_api_metadata_document_and_conflict(db):
    c = TestClient(create_app(db_path=db))
    r = c.post("/api/contracts/ingest", json=CONTRACT)
    assert r.status_code == 200 and r.json()["action"] == "CREATE" and r.json()["version"] == 1
    r = c.post("/api/contracts/ingest", json=dict(CONTRACT, contract_id="CTR-2026-MOB-009", valid_from="2026-10-15"))
    assert r.status_code == 409 and "conflicting" in r.json()["detail"]
    doc = dict(CONTRACT, contract_id="CTR-2026-DJZ-002", brand_name="Djezzy", valid_from="2026-10-01", valid_to="2026-10-31")
    r = c.post("/api/contracts/ingest", json=dict(document_name="djezzy.json", document_b64=base64.b64encode(
        json.dumps(doc).encode()).decode(), net_rate_per_second=11.0))            # explicit field overrides the document
    body = r.json()
    assert r.status_code == 200 and body["brand_name"] == "Djezzy" and body["net_rate_per_second"] == 11.0
    assert body["contract_document_path"].endswith(".json")
    assert c.post("/api/contracts/ingest", json=dict(document_name="x.exe", document_b64="AAAAAAAA")).status_code == 422
    assert c.post("/api/contracts/ingest", json=dict(CONTRACT, contract_id="BAD", net_rate_per_second=-1)).status_code == 422
    assert c.post("/api/contracts/ingest", data="{}").status_code == 415            # CSRF guard still applies
    assert len(c.get("/api/contracts").json()["contracts"]) == 2
    log = c.get("/api/contracts/audit-log").json()
    assert log["intact"] and len(log["entries"]) == 2
    rep = c.get("/api/compliance/overview").json()
    assert rep["kpis"]["verified_contract_airings"] == 2      # Mobilis + Djezzy both bound


def test_certificate_banner_follows_the_track(db):
    pdf = cert.generate_airing_certificate_pdf(1, db)
    assert b"MARKET ESTIMATE" in pdf and b"Independent Benchmark Telemetry" in pdf and b"Bound to Order" not in pdf
    ctr.ingest_contract(db, CONTRACT)
    pdf = cert.generate_airing_certificate_pdf(1, db)
    assert b"CERTIFIED CONTRACT AUDIT" in pdf and b"Bound to Order #CTR-2026-MOB-001" in pdf
    assert b"Independent Benchmark Telemetry" not in pdf
    for needle in (b"Factual", b"SHA-256", b"tamper-evident", b"UTC+1", b"reconciliation."):   # footnote wraps lines
        assert needle in pdf
    assert cert.verify_certificate_pdf(pdf, db)["status"] == "PASS"
