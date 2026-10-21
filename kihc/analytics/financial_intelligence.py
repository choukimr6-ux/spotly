"""Financial market report: estimated spend, Share of Spend vs Share of Voice, make-good credits.

All figures are ESTIMATES = configured rate card x detected airings. They are only as good as the
rate card (see its `status`) and the reference library (unindexed ads are invisible).

Basis (also written into every report and CSV row):
  * Priced airings = live detections (wall-clock start known), human-rejected excluded.
    File-scan detections have no broadcast clock, so they are counted as unpriced.
  * SOS % = brand gross / total gross;  SOV % = brand detected airtime / total detected airtime,
    both over the same priced airings so the two are directly comparable.
  * Positions come from pod reconstruction among *detected* ads; make-good credits are therefore
    "claimable pending verification" against the broadcaster's as-run log.
"""
import csv
import io
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal

from kihc.analytics.campaign_family import family_map
from kihc.analytics.rate_engine import RateCardEngine, compute_airing_financials, money
from kihc.core.analytics import channel_of
from kihc.stream.time_synchronizer import broadcast_epoch
from kihc.core.pod_analyzer import cluster_occurrences_into_pods
from kihc.core.proof import ALGERIA_TZ
from kihc.database.schema import init_db

_SQL = """
SELECT o.occurrence_id, o.source_name, o.airing_start, o.airing_end, o.duration_detected, o.confidence_score,
       o.human_verified, o.verification_hash, o.algeria_timestamp, o.mcr_drift_delta_seconds,
       r.brand_name, r.campaign_name, r.duration_sec, r.industry, r.ad_type, o.ad_id
FROM ad_occurrences o JOIN reference_ads r USING (ad_id)
WHERE o.human_verified {review}
ORDER BY o.airing_start, o.occurrence_id
"""

CSV_COLUMNS = [
    "occurrence_id", "brand", "campaign", "industry", "role", "channel", "airing_start_cet", "daypart", "season",
    "season_multiplier", "rate_source", "base_rate_30s_dzd", "spot_seconds", "spot_value_dzd",
    "premium_value_applied_dzd", "gross_spend_dzd", "agency_discount", "estimated_net_spend_dzd",
    "booked_position", "delivered_position", "position_compliance", "adjacency",
    "contract_loss_credit_net_dzd", "contract_loss_credit_gross_dzd", "bonus_premium_value_dzd",
    "confidence", "human_review", "verification_hash", "rate_card_status", "basis",
]
BASIS = ("estimate: rate card x detected airing; position among detected ads; "
         "credits claimable pending as-run verification")


def _as_date(v):
    if v is None or v == "":
        return None
    return v if isinstance(v, date) and not isinstance(v, datetime) else date.fromisoformat(str(v)[:10])


def _pct(part: Decimal, whole: Decimal) -> float | None:
    return float(part / whole * 100) if whole else None


def build_financial_market_report(db_path, start_date=None, end_date=None, *, industry=None,
                                  rate_card: RateCardEngine | None = None, verified_only: bool = False,
                                  max_gap_seconds: float = 12.0) -> dict:
    rate_card = rate_card or RateCardEngine.load(db_path=db_path)
    d0, d1 = _as_date(start_date), _as_date(end_date)
    pod_ads = {a["occurrence_id"]: a
               for p in cluster_occurrences_into_pods(db_path=db_path, max_gap_seconds=max_gap_seconds)
               for a in p.ads}
    positions = {k: a["position"] for k, a in pod_ads.items()}
    conn = init_db(db_path)
    try:
        rows = conn.execute(_SQL.format(review="= 1" if verified_only else "!= -1")).fetchall()
    finally:
        conn.close()

    want_ind = industry.strip().casefold() if industry else None
    airings, unpriced = [], defaultdict(int)
    brands = defaultdict(lambda: dict(gross=Decimal(0), net=Decimal(0), airings=0, airtime=Decimal(0),
                                      premium=Decimal(0), credit=Decimal(0), credit_gross=Decimal(0),
                                      bonus=Decimal(0), booked=0, compliant=0, breaches=0, role="competitor",
                                      industry=None, channels=set()))
    families = family_map(db_path)
    campaigns = defaultdict(lambda: dict(gross=Decimal(0), net=Decimal(0), airings=0, airtime=Decimal(0),
                                         variants=defaultdict(int)))
    for r in rows:
        if want_ind and (r["industry"] or "").strip().casefold() != want_ind:
            continue
        live = r["source_name"].startswith("live:")
        if live and (d0 or d1):
            day = datetime.fromtimestamp(r["airing_start"], ALGERIA_TZ).date()
            if (d0 and day < d0) or (d1 and day > d1):
                continue
        elif not live and (d0 or d1):
            unpriced["outside date filter: no broadcast date (file scan)"] += 1
            continue
        channel = channel_of(r["source_name"])
        fin = compute_airing_financials(dict(
            brand=r["brand_name"], campaign=r["campaign_name"], channel=channel,
            start_epoch=broadcast_epoch(r["airing_start"], r["mcr_drift_delta_seconds"]) if live else None,
            spot_seconds=r["duration_sec"], pod_position=positions.get(r["occurrence_id"]),
            placement_status=(pod_ads.get(r["occurrence_id"]) or {}).get("placement_status")), rate_card)
        if not fin["priced"]:
            unpriced[fin["unpriced_reason"]] += 1
            continue
        fam = families.get(r["ad_id"])
        parent_campaign = fam["parent_campaign"] if fam else r["campaign_name"]
        c = campaigns[(r["brand_name"], parent_campaign)]            # Phase 21: rolled up to the Parent Campaign
        c["gross"] += fin["gross_spend"]
        c["net"] += fin["estimated_net_spend"]
        c["airings"] += 1
        c["airtime"] += Decimal(str(r["duration_detected"]))
        c["variants"][(r["campaign_name"], fam["cut_type"] if fam else "MASTER")] += 1
        b = brands[r["brand_name"]]
        b["gross"] += fin["gross_spend"]
        b["net"] += fin["estimated_net_spend"]
        b["airings"] += 1
        b["airtime"] += Decimal(str(r["duration_detected"]))
        b["premium"] += fin["premium_value_applied"]
        b["credit"] += fin["contract_loss_credit"]
        b["credit_gross"] += fin["contract_loss_credit_gross"]
        b["bonus"] += fin["bonus_premium_value"]
        b["booked"] += fin["booked_position"] is not None
        b["compliant"] += fin["position_compliance"] == "compliant"
        b["breaches"] += fin["position_compliance"] == "breach"
        if (r["ad_type"] or "self") == "self":
            b["role"] = "self"
        b["industry"] = b["industry"] or r["industry"]
        b["channels"].add(channel)
        airings.append(dict(fin, occurrence_id=r["occurrence_id"], brand=r["brand_name"],
                            campaign=r["campaign_name"], industry=r["industry"], role=r["ad_type"] or "self",
                            channel=channel, airing_start=r["airing_start"], airing_end=r["airing_end"],
                            broadcast_start=broadcast_epoch(r["airing_start"], r["mcr_drift_delta_seconds"]),
                            mcr_drift_delta_seconds=r["mcr_drift_delta_seconds"],
                            airing_start_cet=r["algeria_timestamp"] or datetime.fromtimestamp(
                                r["airing_start"], ALGERIA_TZ).isoformat(timespec="seconds"),
                            confidence=r["confidence_score"],
                            human_review={1: "verified", 0: "unreviewed"}.get(r["human_verified"], "other"),
                            verification_hash=r["verification_hash"],
                            boundary_confidence=(pod_ads.get(r["occurrence_id"]) or {}).get("boundary_confidence"),
                            boundary_cues=(pod_ads.get(r["occurrence_id"]) or {}).get("boundary_cues") or [],
                            placement_note=(pod_ads.get(r["occurrence_id"]) or {}).get("placement_note")))

    total_gross = sum((b["gross"] for b in brands.values()), Decimal(0))
    total_net = sum((b["net"] for b in brands.values()), Decimal(0))
    total_airtime = sum((b["airtime"] for b in brands.values()), Decimal(0))
    brand_rows = []
    for name, b in brands.items():
        judged = b["compliant"] + b["breaches"]
        brand_rows.append(dict(
            brand=name, role=b["role"], industry=b["industry"], airings=b["airings"],
            channels=sorted(b["channels"]), gross_spend=money(b["gross"]), estimated_net_spend=money(b["net"]),
            premium_value_applied=money(b["premium"]), contract_loss_credit=money(b["credit"]),
            contract_loss_credit_gross=money(b["credit_gross"]), bonus_premium_value=money(b["bonus"]),
            booked_premium_airings=b["booked"], compliant_airings=b["compliant"], breaches=b["breaches"],
            compliance_pct=100.0 * b["compliant"] / judged if judged else None,
            sos_pct=_pct(b["gross"], total_gross), sov_pct=_pct(b["airtime"], total_airtime),
            airtime_seconds=float(b["airtime"])))
    brand_rows.sort(key=lambda x: (-x["gross_spend"], x["brand"]))
    campaign_rows = sorted((dict(brand=b, parent_campaign=name, airings=c["airings"], gross_spend=money(c["gross"]),
                                 estimated_net_spend=money(c["net"]), airtime_seconds=float(c["airtime"]),
                                 sov_pct=_pct(c["airtime"], total_airtime),
                                 variants=[dict(campaign=v, cut_type=t, airings=n)
                                           for (v, t), n in sorted(c["variants"].items())])
                            for (b, name), c in campaigns.items()), key=lambda x: (-x["gross_spend"], x["brand"]))
    make_goods = [a for a in airings if a["position_compliance"] == "breach"]
    report = dict(
        currency=rate_card.currency,
        rate_card=dict(status=rate_card.status, illustrative=rate_card.is_illustrative, sources=rate_card.sources,
                       errors=rate_card.errors),
        filters=dict(start_date=d0.isoformat() if d0 else None, end_date=d1.isoformat() if d1 else None,
                     industry=industry, verified_only=verified_only),
        totals=dict(gross_spend=money(total_gross), estimated_net_spend=money(total_net),
                    priced_airings=len(airings), unpriced_airings=sum(unpriced.values()),
                    contract_loss_credit=money(sum((b["credit"] for b in brands.values()), Decimal(0))),
                    contract_loss_credit_gross=money(sum((b["credit_gross"] for b in brands.values()), Decimal(0))),
                    bonus_premium_value=money(sum((b["bonus"] for b in brands.values()), Decimal(0))),
                    breaches=len(make_goods)),
        top_spender=brand_rows[0] if brand_rows else None,
        brands=brand_rows, campaigns=campaign_rows, make_goods=make_goods, airings=airings, unpriced=dict(unpriced),
        basis=BASIS, generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    report["export"] = dict(columns=CSV_COLUMNS, rows=[_csv_row(a, rate_card) for a in airings])
    return report


def _fmt(v):
    if isinstance(v, Decimal):
        return f"{v:.2f}"
    if v is None:
        return ""
    return v


def _csv_row(a: dict, rate_card: RateCardEngine) -> list:
    m = dict(a, base_rate_30s_dzd=a["base_rate_30s"], spot_value_dzd=a["spot_value"],
             premium_value_applied_dzd=a["premium_value_applied"], gross_spend_dzd=a["gross_spend"],
             estimated_net_spend_dzd=a["estimated_net_spend"], contract_loss_credit_net_dzd=a["contract_loss_credit"],
             contract_loss_credit_gross_dzd=a["contract_loss_credit_gross"],
             bonus_premium_value_dzd=a["bonus_premium_value"], rate_card_status=rate_card.status, basis=BASIS)
    return [_fmt(m.get(c)) for c in CSV_COLUMNS]


def report_csv(report: dict) -> str:
    """Airing-by-airing audit CSV (UTF-8 with BOM so Excel opens Arabic/French text correctly)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(report["export"]["columns"])
    w.writerows(report["export"]["rows"])
    return "﻿" + buf.getvalue()


def jsonable(obj):
    """Decimals -> float (2 dp), sets -> sorted lists; for JSON APIs."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(obj)
    return obj
