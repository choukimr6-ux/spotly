"""Phase 12.1: contract compliance & penalty reclaim, competitor separation, dynamic SOV, alerts, API."""
import csv
import io
from datetime import datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from kihc.analytics import compliance_audit as ca
from kihc.analytics.rate_engine import RateCardEngine, RateCardError, compute_airing_financials
from kihc.core.proof import ALGERIA_TZ
from kihc.database.schema import get_db

CH = "El-Heddaf TV"
T = datetime(2026, 10, 13, 20, 0, tzinfo=ALGERIA_TZ).timestamp()        # Prime Time


def _ad(conn, ad_id, brand, sector, role="competitor", seconds=30.0):
    conn.execute("INSERT INTO reference_ads (ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                 "industry, ad_type) VALUES (?, ?, 'Spot', ?, x'00', 't', ?, ?)", (ad_id, brand, seconds, sector, role))


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
        _ad(conn, 4, "Amiral", "Biscuits")
        _ad(conn, 5, "Mystery", None)
    return path


def _rows(rep):
    return {(r["brand"], round(r["airing_start"] - T)): r for r in rep["airings"]}


def test_separation_breach_credit_equals_contracted_share_of_net_spot(db):
    ca.save_contracts(db, [dict(brand="Mobilis", competitor_separation_seconds=180, separation_credit=1.0)])
    with get_db(db) as conn:
        _air(conn, 1, T)                       # Mobilis 20:00:00-20:00:30
        _air(conn, 2, T + 90)                  # Djezzy 60 s after Mobilis ended  -> breach
        _air(conn, 1, T + 3600)                # Mobilis alone an hour later       -> compliant
        _air(conn, 2, T + 3600 + 30 + 200)     # Djezzy 200 s later                -> outside the clause
    rep = ca.compliance_report(db)
    rows = _rows(rep)
    hit = rows[("Mobilis", 0)]
    assert hit["separation"] == "breach" and hit["separation_window_s"] == 180
    card = RateCardEngine.load(db_path=db)
    fin = compute_airing_financials(dict(brand="Mobilis", campaign="Spot", channel=CH, start_epoch=T, spot_seconds=30,
                                         pod_position=hit["delivered_position"]), card)
    assert hit["separation_credit"] == fin["estimated_net_spend"] > 0
    assert rows[("Mobilis", 3600)]["separation"] == "compliant"
    assert rows[("Djezzy", 90)]["separation"] == "not contracted"            # the competitor has no clause
    k = rep["kpis"]
    assert k["separation_breaches"] == 1 and k["reclaimable_net"] == hit["separation_credit"]
    assert k["judged_airings"] == 2 and k["compliance_pct"] == 50.0
    assert rep["rate_card"]["illustrative"]                                   # money labelled honestly


def test_partial_credit_and_different_sectors_never_conflict(db):
    ca.save_contracts(db, [dict(brand="Mobilis", competitor_separation_seconds=180, separation_credit=0.5),
                           dict(brand="Bellat", competitor_separation_seconds=180)])
    with get_db(db) as conn:
        _air(conn, 1, T)
        _air(conn, 2, T + 40)
        _air(conn, 3, T + 7200)                # Bellat (cold cuts) next to Amiral (biscuits): different sectors
        _air(conn, 4, T + 7235)
    rows = _rows(ca.compliance_report(db))
    m = rows[("Mobilis", 0)]
    assert m["separation_credit"] == (m["estimated_net_spend"] * Decimal("0.5")).quantize(Decimal("0.01"))
    assert rows[("Bellat", 7200)]["separation"] == "compliant" and not rows[("Bellat", 7200)]["conflicts"]


def test_conflicts_without_clause_are_reported_but_never_claimed(db):
    with get_db(db) as conn:
        _air(conn, 1, T)
        _air(conn, 2, T + 100)
        _air(conn, 2, T + 100, channel="Echorouk TV")                       # other channel: not a conflict
    rep = ca.compliance_report(db)
    assert rep["kpis"]["separation_conflicts"] == 1 and rep["kpis"]["reclaimable_net"] == 0
    assert rep["setup"]["needs_contracts"] and "No contracts configured" in rep["setup"]["note"]
    assert rep["conflicts"][0]["sector"] == "Telecom" and rep["conflicts"][0]["gap_seconds"] == 70.0


def test_unknown_sector_is_not_assessable(db):
    ca.save_contracts(db, [dict(brand="Mystery", competitor_separation_seconds=180)])
    with get_db(db) as conn:
        _air(conn, 5, T)
        _air(conn, 2, T + 40)
    rep = ca.compliance_report(db)
    assert _rows(rep)[("Mystery", 0)]["separation"] == "not assessable: sector unknown"
    assert rep["setup"]["brands_without_sector"] == ["Mystery"] and rep["kpis"]["breaches"] == 0


def test_position_breach_joins_the_reclaim_and_claim_sheet(db):
    ca.save_contracts(db, [dict(brand="Mobilis", booked_position="FIRST")])
    with get_db(db) as conn:
        _air(conn, 3, T)                       # Bellat, Mobilis, Amiral back-to-back: Mobilis is MIDDLE
        _air(conn, 1, T + 32)
        _air(conn, 4, T + 64)
    rep = ca.compliance_report(db)
    m = _rows(rep)[("Mobilis", 32)]
    assert m["position_compliance"] == "breach" and m["position_credit"] > 0
    assert rep["kpis"]["position_breaches"] == 1 and rep["kpis"]["reclaimable_net"] == m["position_credit"]
    rows = list(csv.DictReader(io.StringIO(ca.claim_sheet_csv(rep))))
    assert len(rows) == 1 and rows[0]["rule"] == "booked position" and rows[0]["brand"] == "Mobilis"
    assert rows[0]["rate_card_status"] == "ILLUSTRATIVE" and "as-run" in rows[0]["basis"]


def test_provisional_brands_flagged_never_billed(db):
    ca.save_contracts(db, [dict(brand="Mobilis", competitor_separation_seconds=180)])
    with get_db(db) as conn:
        _air(conn, 1, T)
        conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, status, suggested_brand) "
                     "VALUES ('ELHEDDAF_AD_001', ?, 30, x'01000000', 'unclassified', 'Djezzy')", (CH,))
        conn.execute("INSERT INTO discovered_occurrences (asset_id, channel, airing_start, airing_end, duration_sec, "
                     "confidence) VALUES ('ELHEDDAF_AD_001', ?, ?, ?, 30, 0.9)", (CH, T + 50, T + 80))
    plain = ca.compliance_report(db)
    assert plain["kpis"]["separation_conflicts"] == 0 and plain["kpis"]["provisional_conflicts"] == 0
    rep = ca.compliance_report(db, include_provisional=True)
    assert rep["kpis"]["provisional_conflicts"] == 1 and rep["kpis"]["separation_breaches"] == 0
    assert rep["kpis"]["reclaimable_net"] == 0 and rep["conflicts"][0]["provisional"]


def test_sov_timeline_daily_airtime_and_24h_shift(db):
    now = T + 3600
    with get_db(db) as conn:
        for i in range(3):
            _air(conn, 1, now - 3600 * (i + 1))             # Mobilis: 3 spots in the last 24 h
        _air(conn, 2, now - 7200 + 10)                      # Djezzy: 1 in the last 24 h ...
        for i in range(4):
            _air(conn, 2, now - 86400 - 3600 * (i + 1))     # ... 4 the day before
    s = ca.sov_timeline(db, days=3, now=now)
    b = {x["brand"]: x for x in s["brands"]}
    assert b["Mobilis"]["share_24h"] == 75.0 and b["Djezzy"]["share_24h"] == 25.0
    assert b["Djezzy"]["share_prev_24h"] == 100.0 and b["Djezzy"]["change_pts"] == -75.0
    assert s["self_share_24h"] == 75.0 and len(s["days"]) == 3 and sum(b["Mobilis"]["airtime"]) == 90.0
    assert ca.sov_timeline(db, days=3, now=now, sector="Biscuits")["brands"] == []


def test_alerts_are_created_once_and_acknowledged(db):
    ca.save_contracts(db, [dict(brand="Mobilis", competitor_separation_seconds=180)])
    now = T + 600
    with get_db(db) as conn:
        _air(conn, 1, T)
        _air(conn, 2, T + 60)
    assert ca.refresh_alerts(db, now=now) == 2                              # breach + the conflict itself
    assert ca.refresh_alerts(db, now=now) == 0                              # idempotent
    alerts = ca.list_alerts(db)
    crit = next(a for a in alerts if a["kind"] == "separation_breach")
    assert crit["severity"] == "critical" and "Djezzy" in crit["message"] and Decimal(crit["credit_dzd"]) > 0
    assert ca.acknowledge_alert(db, crit["id"]) and not ca.acknowledge_alert(db, crit["id"])
    assert crit["id"] not in {a["id"] for a in ca.list_alerts(db)}


def test_contract_clause_validation(db):
    for bad, word in ((dict(brand="X", competitor_separation_seconds=0), "competitor_separation_seconds"),
                      (dict(brand="X", separation_credit=2), "separation_credit"), (dict(brand=" "), "without a brand")):
        card = RateCardEngine.load(overrides=dict(contracts=[bad]))          # never half-applied: falls back
        assert card.errors and word in card.errors[-1] and card.config["contracts"] == []
        with pytest.raises(RateCardError):
            ca.save_contracts(db, [bad])
    assert ca.get_contracts(db) == []


def test_compliance_api(db):
    from kihc.app.api_server import create_app
    app = create_app(db, channel_loader=lambda: [])
    with TestClient(app) as c:
        assert c.post("/api/compliance/contracts", json={"contracts": [{"brand": "Mobilis",
                       "competitor_separation_seconds": 5000}]}).status_code == 422
        ok = c.post("/api/compliance/contracts", json={"contracts": [{"brand": "Mobilis", "booked_position": "ANY",
                     "competitor_separation_seconds": 180, "separation_credit": 1.0}]})
        assert ok.status_code == 200 and c.get("/api/compliance/contracts").json()["contracts"][0]["brand"] == "Mobilis"
        assert c.post("/api/compliance/contracts", json={"contracts": []},
                      headers={"origin": "https://evil.example"}).status_code == 403
        r = c.post("/api/ads/5/sector", json={"sector": "Telecom"})
        assert r.status_code == 200
        ads = {a["ad_id"]: a for a in c.get("/api/ads").json()["ads"]}
        assert ads[5]["sector"] == "Telecom" and ads[5]["role"] == "competitor"           # role kept
        with get_db(db) as conn:
            _air(conn, 1, T)
            _air(conn, 5, T + 45)
        ov = c.get("/api/compliance/overview").json()
        assert ov["kpis"]["separation_breaches"] == 1 and ov["kpis"]["reclaimable_net"] > 0
        assert ov["flagged_airings"] and ov["rate_card"]["status"] == "ILLUSTRATIVE"
        assert c.get("/api/compliance/export/csv").text.count("competitor separation") == 1
        assert c.get("/api/compliance/sov", params={"days": 7}).status_code == 200
        ca.refresh_alerts(db, now=T + 600)
        alert = c.get("/api/compliance/alerts").json()["alerts"][0]
        assert c.post(f"/api/compliance/alerts/{alert['id']}/ack", json={}).status_code == 200
        assert c.post(f"/api/compliance/alerts/{alert['id']}/ack", json={}).status_code == 404
        html = c.get("/").text
        for i in ("comp-pct", "comp-reclaim", "comp-alerts", "sov-chart", "comp-conflict-list"):
            assert f'id="{i}"' in html
