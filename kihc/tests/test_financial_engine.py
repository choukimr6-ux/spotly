"""Phase 8: rate cards, per-airing valuation, market report, CSV audit, financial APIs.

Expected money values are computed by hand in each test (Decimal, ROUND_HALF_UP to centimes).
"""
import copy
import csv
import io
import json
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from kihc.analytics.financial_intelligence import (CSV_COLUMNS, build_financial_market_report,
                                                   report_csv)
from kihc.analytics.rate_engine import (BUILTIN, DEFAULT_PATH, RateCardEngine, RateCardError,
                                        clear_db_override, compute_airing_financials,
                                        save_db_override, validate)
from kihc.app.api_server import create_app
from kihc.core.proof import ALGERIA_TZ
from kihc.tests.test_pod_analyzer import Market

D = Decimal


def at(y, mo, d, h, mi=0, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=ALGERIA_TZ).timestamp()


def card(tmp_path, **changes):
    cfg = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    for k, v in changes.items():
        cfg[k] = v
    p = tmp_path / "rates.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return RateCardEngine.load(path=p)


def value(engine, start, seconds=30, position=None, brand="Atlas Mobile", campaign="Spot", channel="El-Heddaf TV"):
    return compute_airing_financials(dict(brand=brand, campaign=campaign, channel=channel, start_epoch=start,
                                          spot_seconds=seconds, pod_position=position), engine)


@pytest.fixture
def eng(tmp_path):
    return card(tmp_path)


# ------------------------------------------------------------------ rate lookup

@pytest.mark.parametrize("hms, daypart, rate", [
    ((19, 29, 59), "Daytime", "90000"), ((19, 30, 0), "Prime Time", "220000"),
    ((23, 0, 59), "Prime Time", "220000"), ((23, 1, 0), "Off-Peak / Late Night", "45000"),
    ((0, 0, 0), "Off-Peak / Late Night", "45000"), ((11, 59, 59), "Off-Peak / Late Night", "45000"),
    ((12, 0, 0), "Daytime", "90000"),
])
def test_daypart_boundaries(eng, hms, daypart, rate):
    f = value(eng, at(2026, 10, 10, *hms))
    assert f["daypart"] == daypart and f["base_rate_30s"] == D(rate) and f["gross_spend"] == D(rate)


def test_daypart_uses_algeria_time_not_utc(eng):
    # 18:45 UTC = 19:45 Algeria (UTC+1) -> Prime Time
    assert value(eng, datetime(2026, 10, 10, 18, 45, tzinfo=ALGERIA_TZ.__class__.utc).timestamp())["daypart"] == "Prime Time"


@pytest.mark.parametrize("seconds, start_h, gross, net", [
    (20, 20, "146666.67", "102666.67"),     # 220000 * 20/30 ; * 0.70
    (45, 15, "135000.00", "94500.00"),      # 90000 * 45/30
    (10, 2, "15000.00", "10500.00"),        # 45000 * 10/30
])
def test_duration_proportionality(eng, seconds, start_h, gross, net):
    f = value(eng, at(2026, 10, 10, start_h), seconds=seconds)
    assert f["gross_spend"] == D(gross) and f["estimated_net_spend"] == D(net) and f["spot_seconds"] == seconds


def test_nominal_length_is_rounded_to_whole_seconds(eng):
    assert value(eng, at(2026, 10, 10, 20), seconds=10.008)["gross_spend"] == D("73333.33")   # 220000 * 10/30


def test_seasonality_multiplier_inclusive_dates(eng):
    assert value(eng, at(2026, 3, 1, 21))["gross_spend"] == D("550000.00")                  # Ramadan 2.5x prime
    first = value(eng, at(2026, 2, 18, 0, 30))
    assert first["season"] == "Ramadan 2026" and first["gross_spend"] == D("112500.00")     # 45000 * 2.5
    assert value(eng, at(2026, 3, 19, 23, 30))["season"] == "Ramadan 2026"
    assert value(eng, at(2026, 3, 20, 0, 30))["season"] == "Standard"


def test_channel_multiplier_own_dayparts_and_fallback(tmp_path):
    channels = {"Echorouk TV": {"rate_multiplier": 1.5},
                "Ennahar TV": {"dayparts": [{"name": "All Day", "start": "00:00", "end": "24:00", "rate": 100000}]}}
    e = card(tmp_path, channels=channels)
    assert value(e, at(2026, 10, 10, 20), channel="Echorouk TV")["gross_spend"] == D("330000.00")
    assert value(e, at(2026, 10, 10, 3), channel="ennahar tv")["daypart"] == "All Day"            # case-insensitive
    unknown = value(e, at(2026, 10, 10, 20), channel="New Channel TV")
    assert unknown["rate_source"] == "default" and unknown["gross_spend"] == D("220000.00")


def test_file_scan_is_unpriced(eng):
    f = value(eng, None)
    assert not f["priced"] and "no broadcast clock" in f["unpriced_reason"] and f["gross_spend"] == D("0.00")


# ------------------------------------------------------------------ contracts, breaches, bonus

CONTRACTS = [{"brand": "Atlas Mobile", "campaign": "*", "booked_position": "FIRST"},
             {"brand": "Atlas Mobile", "campaign": "Closer", "booked_position": "LAST"},
             {"brand": "Oasis Net", "campaign": "*", "booked_position": "FIRST", "agency_discount": 0.20,
              "adjacency": True}]


@pytest.fixture
def ceng(tmp_path):
    return card(tmp_path, contracts=CONTRACTS)


def test_premium_booked_but_aired_middle_is_a_breach(ceng):
    f = value(ceng, at(2026, 10, 10, 20), position="MIDDLE")
    # spot 220000; FIRST surcharge 20% = 44000 billed; gross 264000; net 70% = 184800
    assert (f["booked_position"], f["position_compliance"]) == ("FIRST", "breach")
    assert f["premium_value_applied"] == D("44000.00") and f["gross_spend"] == D("264000.00")
    assert f["estimated_net_spend"] == D("184800.00")
    assert f["contract_loss_credit_gross"] == D("44000.00") and f["contract_loss_credit"] == D("30800.00")


@pytest.mark.parametrize("delivered, status", [("FIRST", "compliant"), ("SOLO", "compliant"),
                                               ("LAST", "breach"), (None, "unknown position")])
def test_first_booking_compliance(ceng, delivered, status):
    f = value(ceng, at(2026, 10, 10, 20), position=delivered)
    assert f["position_compliance"] == status
    assert f["contract_loss_credit"] == (D("30800.00") if status == "breach" else D("0.00"))


def test_exact_campaign_contract_beats_wildcard(ceng):
    f = value(ceng, at(2026, 10, 10, 20), position="FIRST", campaign="Closer")
    # booked LAST (15%) but delivered FIRST -> breach, credit 220000*0.15*0.70
    assert f["booked_position"] == "LAST" and f["position_compliance"] == "breach"
    assert f["contract_loss_credit"] == D("23100.00")


def test_contract_discount_and_unverifiable_adjacency(ceng):
    f = value(ceng, at(2026, 10, 10, 20), position="FIRST", brand="oasis net")
    # 220000 * (1 + 0.20 + 0.10) = 286000 gross; 20% discount -> 228800
    assert f["gross_spend"] == D("286000.00") and f["estimated_net_spend"] == D("228800.00")
    assert f["adjacency"].startswith("billed, unverifiable")


def test_unbooked_premium_position_is_bonus_not_billed(ceng):
    f = value(ceng, at(2026, 10, 10, 20), position="FIRST", brand="Sahara Telecom")
    assert f["gross_spend"] == D("220000.00") and f["bonus_premium_value"] == D("44000.00")
    assert f["position_compliance"] == "not booked" and f["contract_loss_credit"] == D("0.00")
    assert value(ceng, at(2026, 10, 10, 20), position="LAST", brand="Sahara Telecom")["bonus_premium_value"] == D("33000.00")


# ------------------------------------------------------------------ engine loading / validation

def test_shipped_rate_card_is_valid_and_flagged_illustrative():
    e = RateCardEngine.load(path=DEFAULT_PATH)
    assert not e.errors and e.is_illustrative and e.status == "ILLUSTRATIVE"
    assert {dp["name"]: dp["rate"] for dp in e.config["defaults"]["dayparts"]} == {
        "Prime Time": 220000, "Daytime": 90000, "Off-Peak / Late Night": 45000}
    assert e.config["defaults"]["agency_discount"] == 0.30
    assert {"El-Heddaf TV", "Echorouk TV", "Ennahar TV"} <= set(e.config["channels"])


def test_missing_and_invalid_files_fall_back(tmp_path):
    e = RateCardEngine.load(path=tmp_path / "nope.json")
    assert e.sources == ["built-in"] and "not found" in e.errors[0]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert "unreadable" in RateCardEngine.load(path=bad).errors[0]


@pytest.mark.parametrize("dayparts, msg", [
    ([{"name": "A", "start": "00:00", "end": "12:00", "rate": 1}], "12:00 is covered by no daypart"),
    ([{"name": "A", "start": "00:00", "end": "13:00", "rate": 1},
      {"name": "B", "start": "12:00", "end": "00:00", "rate": 1}], "12:00 is covered by ['A', 'B']"),
])
def test_gaps_and_overlaps_are_rejected(tmp_path, dayparts, msg):
    cfg = copy.deepcopy(BUILTIN)
    cfg["defaults"]["dayparts"] = dayparts
    with pytest.raises(RateCardError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        validate(cfg)
    e = card(tmp_path, defaults=cfg["defaults"])
    assert e.sources == ["built-in (fallback)"] and "fell back" in e.errors[-1]


def test_env_var_db_and_runtime_overrides(tmp_path, monkeypatch):
    alt = tmp_path / "alt.json"
    alt.write_text(json.dumps({"_meta": {"status": "VERIFIED"}}), encoding="utf-8")
    monkeypatch.setenv("KIHC_RATE_CARDS", str(alt))
    e = RateCardEngine.load()
    assert str(alt) in e.sources and not e.is_illustrative
    db = tmp_path / "o.db"
    save_db_override(db, {"defaults": {"agency_discount": 0.25}, "contracts": CONTRACTS[:1]})
    e = RateCardEngine.load(db_path=db)
    assert "database override" in e.sources and e.config["defaults"]["agency_discount"] == 0.25
    assert e.config["defaults"]["dayparts"]                                          # deep-merged, not replaced
    assert value(e, at(2026, 10, 10, 20), position="MIDDLE")["contract_loss_credit"] == D("33000.00")  # 44000*0.75
    e = RateCardEngine.load(db_path=db, overrides={"defaults": {"agency_discount": 0.0}})
    assert e.sources[-1] == "runtime override" and e.config["defaults"]["agency_discount"] == 0.0
    clear_db_override(db)
    assert "database override" not in RateCardEngine.load(db_path=db).sources
    with pytest.raises(RateCardError):
        save_db_override(db, {"defaults": {"agency_discount": 1.5}})


# ------------------------------------------------------------------ market report + CSV

@pytest.fixture
def market(tmp_path):
    m = Market(tmp_path / "fin.db")
    m.ad("Atlas Mobile", "self")
    m.ad("Sahara Telecom", "competitor")
    base = at(2026, 10, 10, 20)
    m.chain(["Atlas Mobile", "Sahara Telecom", "Atlas Mobile"], base, seconds=30, gap=2)   # FIRST, MIDDLE, LAST
    m.air("Sahara Telecom", at(2026, 10, 10, 14), 30)                                      # Daytime SOLO
    m.air("Sahara Telecom", at(2026, 10, 10, 21), 30, review=-1)                           # rejected
    m.air("Atlas Mobile", 100.0, 30, channel="Echorouk TV · rec.mp4", live=False)          # file scan: unpriced
    m.conn.execute("UPDATE reference_ads SET industry = 'Telecom'")
    m.commit()
    m.conn.close()
    return m.db, card(tmp_path, contracts=CONTRACTS[:1])


def test_market_report_figures(market):
    db, eng = market
    rep = build_financial_market_report(db, rate_card=eng)
    t = rep["totals"]
    # Atlas: FIRST compliant 264000 + LAST breach 264000 = 528000 gross, net 369600, credit 30800
    # Sahara: MIDDLE prime 220000 + SOLO daytime 90000 (bonus 18000) = 310000 gross, net 217000
    assert t["gross_spend"] == D("838000.00") and t["estimated_net_spend"] == D("586600.00")
    assert t["contract_loss_credit"] == D("30800.00") and t["breaches"] == 1
    assert t["bonus_premium_value"] == D("18000.00")
    assert t["priced_airings"] == 4 and t["unpriced_airings"] == 1
    b = {x["brand"]: x for x in rep["brands"]}
    assert rep["top_spender"]["brand"] == "Atlas Mobile"
    assert b["Atlas Mobile"]["gross_spend"] == D("528000.00") and b["Atlas Mobile"]["compliance_pct"] == 50.0
    assert b["Atlas Mobile"]["sos_pct"] == pytest.approx(528000 / 838000 * 100)
    assert b["Atlas Mobile"]["sov_pct"] == pytest.approx(50.0)                         # 60 s of 120 s
    assert b["Sahara Telecom"]["compliance_pct"] is None                               # nothing booked
    assert sum(x["sos_pct"] for x in rep["brands"]) == pytest.approx(100)
    assert [mg["delivered_position"] for mg in rep["make_goods"]] == ["LAST"]
    assert rep["rate_card"]["illustrative"] is True


def test_report_filters_and_determinism(market):
    db, eng = market
    day = build_financial_market_report(db, "2026-10-10", "2026-10-10", rate_card=eng)
    assert day["totals"]["priced_airings"] == 4 and list(day["unpriced"].values()) == [1]
    assert "date filter" in next(iter(day["unpriced"]))
    assert build_financial_market_report(db, "2026-10-11", None, rate_card=eng)["totals"]["gross_spend"] == D("0.00")
    assert build_financial_market_report(db, industry="Dairy", rate_card=eng)["brands"] == []
    a = build_financial_market_report(db, rate_card=eng)["export"]
    assert a == build_financial_market_report(db, rate_card=eng)["export"]           # deterministic


def test_csv_export(market):
    db, eng = market
    text = report_csv(build_financial_market_report(db, rate_card=eng))
    assert text.startswith("﻿")
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    assert list(rows[0].keys()) == CSV_COLUMNS and len(rows) == 4
    breach = next(r for r in rows if r["position_compliance"] == "breach")
    assert (breach["gross_spend_dzd"], breach["contract_loss_credit_net_dzd"]) == ("264000.00", "30800.00")
    assert breach["delivered_position"] == "LAST" and breach["booked_position"] == "FIRST"
    assert all(r["rate_card_status"] == "ILLUSTRATIVE" and r["basis"].startswith("estimate") for r in rows)
    assert sum(D(r["gross_spend_dzd"]) for r in rows) == D("838000.00")


# ------------------------------------------------------------------ API

@pytest.fixture
def api(market):
    db, _ = market
    save_db_override(db, {"contracts": CONTRACTS[:1]})               # contracts via the DB override path
    with TestClient(create_app(db, channel_loader=lambda: [])) as c:
        yield c


def test_financial_endpoints(api):
    o = api.get("/api/financials/overview").json()
    assert o["gross_spend"] == 838000.0 and o["contract_loss_credit"] == 30800.0 and o["breaches"] == 1
    assert o["top_spender"]["brand"] == "Atlas Mobile" and o["rate_card"]["illustrative"] is True
    br = api.get("/api/financials/brands").json()
    assert [x["brand"] for x in br["brands"]] == ["Atlas Mobile", "Sahara Telecom"]
    assert br["brands"][0]["compliance_pct"] == 50.0 and len(br["make_goods"]) == 1
    r = api.get("/api/financials/export/csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"] and r.content.startswith(b"\xef\xbb\xbf")
    assert api.get("/api/financials/overview", params={"start": "not-a-date"}).status_code == 422
    assert api.get("/api/financials/overview", params={"start": "2026-10-11"}).json()["gross_spend"] == 0.0
    s = api.get("/api/stats").json()
    assert s["financials"]["gross_spend"] == 838000.0


def test_dashboard_has_financial_cards(api):
    html = api.get("/").text
    for element_id in ("fin-total", "fin-top", "fin-sos", "fin-sov", "fin-credit", "fin-credit-badge", "fin-export",
                       "fin-status"):
        assert f'id="{element_id}"' in html, element_id
    assert 'href="/api/financials/export/csv"' in html and "Export Financial Audit (.CSV)" in html
    assert 'data-fin-mode="gross"' in html and 'data-fin-mode="net"' in html
