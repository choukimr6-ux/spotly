"""Contract compliance & penalty reclaim, competitor separation, dynamic share of voice, alerts (Phase 12.1).

Rules audited per confirmed airing (ad_occurrences of human-classified / indexed ads, live clock known):
  * Booked position - a contract's booked FIRST/LAST not delivered (rate_engine: credit = position surcharge).
  * Competitor separation - a contract clause `competitor_separation_seconds` (e.g. 180) protects the brand's
    airing from a different brand of the SAME SECTOR airing within that many seconds before or after it on the
    same channel (gap measured end-to-start). Breach credit = estimated net spot value x `separation_credit`
    (default 1.0 = full make-good). Without a clause, same-sector spots closer than DEFAULT_SEPARATION_SECONDS
    are reported as conflicts but no money is claimed.

Sector = reference_ads.industry as set by the reviewer (e.g. "Biscuits", "Telecom"). A brand without a sector is
"not assessable" for separation - KIHC never guesses a sector to manufacture a breach. Auto-suggested
(unconfirmed) discovered brands can be included as *provisional* conflicts, never with money.

All DZD figures inherit the rate card's status: while it is ILLUSTRATIVE they are illustrative, and every credit
is "claimable pending verification against the broadcaster's as-run log" (positions are among detected ads).
"""
import csv
import io
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from kihc.analytics import contracts as ctr
from kihc.analytics.financial_intelligence import build_financial_market_report
from kihc.analytics.rate_engine import RateCardEngine, money
from kihc.core.analytics import detect_competitor_surges
from kihc.core.proof import ALGERIA_TZ, algeria_iso
from kihc.database.schema import get_db

DEFAULT_SEPARATION_SECONDS = 180.0
CLAIM_COLUMNS = ["occurrence_id", "brand", "campaign", "channel", "airing_start_cet", "rule", "detail",
                 "estimated_net_spot_dzd", "credit_rate", "credit_net_dzd", "rate_card_status", "basis"]
CLAIM_BASIS = "claimable pending verification against the broadcaster's as-run log"


DEMOTION_DEFAULT_PCT = 0.30            # Track B: illustrative placement penalty when no contract binds the airing
_POSITION_MAP = {"FIRST_IN_POD": "FIRST", "LAST_IN_POD": "LAST", "FIRST_OR_LAST": "FIRST_OR_LAST"}


def _bind_contract(row: dict, a: dict, c: dict) -> None:
    """Track A: price and judge one airing strictly against its ingested contract (never the rate card)."""
    seconds = Decimal(str(a.get("spot_seconds") or (a["airing_end"] - a["airing_start"])))
    net = money(Decimal(str(c["net_rate_per_second"])) * seconds)
    booked = _POSITION_MAP.get(c["booked_position"])
    delivered, edge = a["delivered_position"], booked == "FIRST_OR_LAST"
    if not booked:
        compliance = "not booked"
    elif delivered is None:
        compliance = "unknown position"
    elif delivered == booked or delivered == "SOLO" or (edge and delivered in ("FIRST", "LAST")):
        compliance = "compliant"
    elif a.get("placement_status") == "demoted_ambiguous":
        compliance = "unverified placement"           # Phase 20: weak boundary cues never become a claim
    else:
        compliance = "breach"
    window = float(c["separation_window_seconds"] or 0)
    row.update(estimated_net_spend=net, booked_position=booked, position_compliance=compliance,
               position_credit=money(net * Decimal(str(c["demotion_penalty_pct"]))) if compliance == "breach" else money(0),
               separation_window_s=window or None, audit_status=ctr.STATUS_VERIFIED, contract_id=c["contract_id"],
               separation_penalty_pct=c["separation_breach_penalty_pct"])


def _sector(value) -> str | None:
    s = (value or "").strip()
    return s.casefold() if s else None


def _gap(a: dict, b: dict) -> float:
    """Seconds between two airings (0 if they overlap)."""
    first, second = (a, b) if a["airing_start"] <= b["airing_start"] else (b, a)
    return max(0.0, second["airing_start"] - first["airing_end"])


def provisional_airings(db_path, t0: float | None = None, t1: float | None = None) -> list[dict]:
    """Airings of unclassified discovered assets that carry an auto-suggested brand (unconfirmed)."""
    from kihc.analytics.brand_extractor import BRANDS
    sector_of = {b.name: (b.product or b.industry) for b in BRANDS}
    q = ("SELECT o.id, o.channel, o.airing_start, o.airing_end, o.duration_sec, a.asset_id, a.suggested_brand "
         "FROM discovered_occurrences o JOIN discovered_assets a USING (asset_id) "
         "WHERE a.status = 'unclassified' AND a.suggested_brand IS NOT NULL")
    with get_db(db_path) as conn:
        rows = [dict(r) for r in conn.execute(q)]
    out = []
    for r in rows:
        if (t0 is not None and r["airing_start"] < t0) or (t1 is not None and r["airing_start"] >= t1):
            continue
        out.append(dict(occurrence_id=f"D{r['id']}", asset_id=r["asset_id"], brand=r["suggested_brand"],
                        industry=sector_of.get(r["suggested_brand"]), channel=r["channel"], role="unconfirmed",
                        airing_start=r["airing_start"], airing_end=r["airing_end"],
                        duration=r["duration_sec"], provisional=True))
    return out


def compliance_report(db_path, start_date=None, end_date=None, *, rate_card: RateCardEngine | None = None,
                      include_provisional: bool = False) -> dict:
    rate_card = rate_card or RateCardEngine.load(db_path=db_path)
    fin = build_financial_market_report(db_path, start_date, end_date, rate_card=rate_card)
    airings = [dict(a, provisional=False) for a in fin["airings"]]
    contracts = rate_card.config.get("contracts") or []
    rows = {a["occurrence_id"]: dict(
        occurrence_id=a["occurrence_id"], brand=a["brand"], campaign=a["campaign"], role=a["role"],
        sector=a["industry"], channel=a["channel"], airing_start=a["airing_start"],
        airing_start_cet=a["airing_start_cet"], airing_end=a["airing_end"], daypart=a["daypart"],
        estimated_net_spend=a["estimated_net_spend"], booked_position=a["booked_position"],
        delivered_position=a["delivered_position"], position_compliance=a["position_compliance"],
        position_credit=a["contract_loss_credit"], separation="not contracted", separation_credit=money(0),
        separation_window_s=None, conflicts=[], audit_status=ctr.STATUS_ESTIMATE, contract_id=None,
        separation_penalty_pct=None, placement_status=a.get("placement_status"),
        boundary_confidence=a.get("boundary_confidence"), boundary_cues=a.get("boundary_cues") or [],
        placement_note=a.get("placement_note")) for a in airings}
    with get_db(db_path) as conn:                        # Phase 19: sealed breach clips already bound to these airings
        clips = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT occurrence_id, evidence_path, evidence_sha256 FROM ad_occurrences WHERE evidence_path IS NOT NULL")}
    for oid, row in rows.items():
        path, digest = clips.get(oid, (None, None))
        row.update(evidence_available=bool(path), evidence_sha256=digest)
    active = ctr.load_active(db_path)
    for a in airings:                                    # Track A where an ACTIVE contract matches [channel, brand, time]
        c = ctr.match_contract(active, a["channel"], a["brand"], a.get("broadcast_start", a["airing_start"]))
        if c:
            _bind_contract(rows[a["occurrence_id"]], a, c)

    pool = list(airings)
    if include_provisional:
        t = [a["airing_start"] for a in airings]
        lo, hi = (min(t) - 3600, max(t) + 3600) if t else (None, None)    # only unconfirmed spots near confirmed ones
        pool += provisional_airings(db_path, lo, hi)
    by_channel = defaultdict(list)
    for a in pool:
        by_channel[a["channel"]].append(a)

    conflicts = []
    for ch, items in by_channel.items():
        items.sort(key=lambda x: x["airing_start"])
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if b["airing_start"] - a["airing_end"] > 3600:
                    break                                           # sorted: nothing closer later
                sa, sb = _sector(a["industry"]), _sector(b["industry"])
                if not sa or sa != sb or a["brand"].casefold() == b["brand"].casefold():
                    continue
                gap = _gap(a, b)
                windows = [(rows[x["occurrence_id"]]["separation_window_s"] or 0.0)
                           if rows.get(x["occurrence_id"], {}).get("contract_id") else
                           float((rate_card.contract(x["brand"], x.get("campaign", "*")) or {})
                                 .get("competitor_separation_seconds") or 0) for x in (a, b)]
                if gap >= max(windows + [DEFAULT_SEPARATION_SECONDS]):
                    continue
                conflicts.append(dict(channel=ch, sector=a["industry"], gap_seconds=round(gap, 1),
                                      provisional=a["provisional"] or b["provisional"],
                                      a=dict(brand=a["brand"], occurrence_id=a["occurrence_id"],
                                             start_cet=algeria_iso(a["airing_start"])),
                                      b=dict(brand=b["brand"], occurrence_id=b["occurrence_id"],
                                             start_cet=algeria_iso(b["airing_start"]))))
                for x, y, w in ((a, b, windows[0]), (b, a, windows[1])):
                    row = rows.get(x["occurrence_id"]) if not x["provisional"] else None
                    if row is None:
                        continue
                    row["conflicts"].append(dict(brand=y["brand"], gap_seconds=round(gap, 1), provisional=y["provisional"]))
                    if w and gap < w and not y["provisional"]:
                        pct = row["separation_penalty_pct"] if row["contract_id"] is not None else \
                            (rate_card.contract(x["brand"], x.get("campaign", "*")) or {}).get("separation_credit", 1.0)
                        credit = money(row["estimated_net_spend"] * Decimal(str(pct)))
                        row.update(separation="breach", separation_window_s=w,
                                   separation_credit=max(row["separation_credit"], credit))

    # separation status for contracted airings without a breach
    for row in rows.values():
        c = rate_card.contract(row["brand"], row["campaign"]) or {}
        w = row["separation_window_s"] if row["contract_id"] else c.get("competitor_separation_seconds")
        if w and row["separation"] != "breach":
            row["separation_window_s"] = float(w)
            row["separation"] = "compliant" if row["sector"] else "not assessable: sector unknown"

    audited = [r for r in rows.values() if r["booked_position"] or r["separation_window_s"]]
    breached = [r for r in audited if r["position_compliance"] == "breach" or r["separation"] == "breach"]
    judged = [r for r in audited if r["position_compliance"] in ("compliant", "breach")
              or r["separation"] in ("compliant", "breach")]
    pos_credit = sum((r["position_credit"] for r in rows.values()), Decimal(0))
    sep_credit = sum((r["separation_credit"] for r in rows.values()), Decimal(0))
    no_sector = sorted({r["brand"] for r in rows.values() if not r["sector"]})
    return dict(
        currency=rate_card.currency,
        rate_card=dict(status=rate_card.status, illustrative=rate_card.is_illustrative, errors=rate_card.errors),
        contracts=len(contracts), client_contracts=len(active), filters=fin["filters"], basis=CLAIM_BASIS,
        kpis=dict(airings=len(rows),
                  verified_contract_airings=sum(r["audit_status"] == ctr.STATUS_VERIFIED for r in rows.values()),
                  illustrative_airings=sum(r["audit_status"] == ctr.STATUS_ESTIMATE for r in rows.values()), audited_airings=len(audited), judged_airings=len(judged),
                  compliant_airings=len(judged) - len(breached), breaches=len(breached),
                  compliance_pct=round(100.0 * (len(judged) - len(breached)) / len(judged), 1) if judged else None,
                  position_breaches=sum(r["position_compliance"] == "breach" for r in rows.values()),
                  separation_breaches=sum(r["separation"] == "breach" for r in rows.values()),
                  separation_conflicts=sum(not c["provisional"] for c in conflicts),
                  provisional_conflicts=sum(c["provisional"] for c in conflicts),
                  reclaimable_net=money(pos_credit + sep_credit), position_credit=money(pos_credit),
                  separation_credit=money(sep_credit)),
        setup=dict(brands_without_sector=no_sector,
                   needs_contracts=not contracts and not active,
                   note=("No contracts configured: nothing is owed by definition. Add clauses (booked position, "
                         "competitor separation) to audit them." if not contracts and not active else None)),
        conflicts=conflicts, airings=sorted(rows.values(), key=lambda r: r["airing_start"], reverse=True),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def claim_sheet_csv(report: dict) -> str:
    """One row per reclaimable breach - the make-good claim to send to the broadcaster."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CLAIM_COLUMNS)
    status = report["rate_card"]["status"]
    for r in report["airings"]:
        ref = f"{ctr.STATUS_VERIFIED}: contract {r['contract_id']}" if r.get("contract_id") else None
        if r["position_compliance"] == "breach":
            w.writerow([r["occurrence_id"], r["brand"], r["campaign"], r["channel"], r["airing_start_cet"],
                        "booked position", f"booked {r['booked_position']}, delivered {r['delivered_position']}",
                        f"{r['estimated_net_spend']:.2f}", "position surcharge", f"{r['position_credit']:.2f}",
                        status, ref or report["basis"]])
        if r["separation"] == "breach":
            near = ", ".join(f"{c['brand']} ({c['gap_seconds']:.0f} s)" for c in r["conflicts"] if not c["provisional"])
            w.writerow([r["occurrence_id"], r["brand"], r["campaign"], r["channel"], r["airing_start_cet"],
                        "competitor separation", f"within {r['separation_window_s']:.0f} s of {near}",
                        f"{r['estimated_net_spend']:.2f}", "separation_credit", f"{r['separation_credit']:.2f}",
                        status, ref or report["basis"]])
    return buf.getvalue()


# ---------------------------------------------------------------- dynamic share of voice

def sov_timeline(db_path, days: int = 7, *, sector: str | None = None, include_provisional: bool = False,
                 now: float | None = None) -> dict:
    """Daily airtime per brand (Algeria calendar days) + share in the last 24 h vs the 24 h before."""
    now = now if now is not None else datetime.now(timezone.utc).timestamp()
    t0 = now - days * 86400
    fin = build_financial_market_report(db_path)
    pool = [dict(brand=a["brand"], role=a["role"], industry=a["industry"], start=a["airing_start"],
                 seconds=a["airing_end"] - a["airing_start"], provisional=False) for a in fin["airings"]]
    if include_provisional:
        pool += [dict(brand=f"{p['brand']} (unconfirmed)", role="unconfirmed", industry=p["industry"],
                      start=p["airing_start"], seconds=p["duration"], provisional=True)
                 for p in provisional_airings(db_path, t0, now)]
    want = _sector(sector)
    pool = [p for p in pool if p["start"] >= t0 and p["start"] <= now and (not want or _sector(p["industry"]) == want)]
    days_axis = [(datetime.fromtimestamp(now, ALGERIA_TZ).date() - timedelta(days=i)).isoformat()
                 for i in range(days - 1, -1, -1)]
    series = defaultdict(lambda: defaultdict(float))
    roles = {}
    for p in pool:
        day = datetime.fromtimestamp(p["start"], ALGERIA_TZ).date().isoformat()
        series[p["brand"]][day] += p["seconds"]
        roles[p["brand"]] = p["role"]

    def share(lo, hi):
        tot = defaultdict(float)
        for p in pool:
            if lo <= p["start"] < hi:
                tot[p["brand"]] += p["seconds"]
        s = sum(tot.values())
        return {b: round(100 * v / s, 1) for b, v in tot.items()} if s else {}
    last, prev = share(now - 86400, now + 1), share(now - 2 * 86400, now - 86400)
    brands = sorted(series, key=lambda b: -sum(series[b].values()))
    return dict(days=days_axis, sector=sector, include_provisional=include_provisional,
                brands=[dict(brand=b, role=roles[b], airtime=[round(series[b].get(d, 0.0), 1) for d in days_axis],
                             share_24h=last.get(b), share_prev_24h=prev.get(b),
                             change_pts=round(last.get(b, 0) - prev.get(b, 0), 1) if (b in last or b in prev) else None)
                        for b in brands],
                self_share_24h=round(sum(v for b, v in last.items() if roles.get(b) == "self"), 1) if last else None)


# ---------------------------------------------------------------- alerts

def _insert_alert(conn, key, kind, severity, message, brand=None, channel=None, start=None, credit=None,
                  meta=None) -> bool:
    cur = conn.execute("INSERT OR IGNORE INTO compliance_alerts (key, kind, severity, brand, channel, airing_start, "
                       "message, credit_dzd, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (key, kind, severity, brand, channel, start, message, None if credit is None else f"{credit:.2f}",
                        json.dumps(meta, default=str) if meta else None))
    return cur.rowcount == 1


def _alert_meta(r: dict) -> dict:
    """Boundary / audit-track facts stored with the alert so a challenged claim can show its sensing basis."""
    return dict(audit_status=r["audit_status"], contract_id=r["contract_id"], placement_status=r["placement_status"],
                boundary_confidence=r["boundary_confidence"], boundary_cues=r["boundary_cues"],
                placement_note=r["placement_note"])


def _record_evidence(db_path, r: dict, breach_type: str) -> None:
    """Hand the breach to the background evidence recorder (non-blocking; the clip is cut from the memory ring)."""
    from kihc.analytics import evidence_recorder
    evidence_recorder.submit_breach(db_path, r["occurrence_id"], r["channel"], r["airing_start"], r["airing_end"],
                                    breach_type, r.get("contract_id"))


def refresh_alerts(db_path, lookback_days: int = 2, now: float | None = None) -> int:
    """Turn new breaches/conflicts/surges into alerts (idempotent). -> number of new alerts."""
    now = now if now is not None else datetime.now(timezone.utc).timestamp()
    start = datetime.fromtimestamp(now - lookback_days * 86400, ALGERIA_TZ).date().isoformat()
    rep = compliance_report(db_path, start_date=start)
    cur = rep["rate_card"]["status"]
    new = 0
    with get_db(db_path) as conn:
        try:                                       # Phase 22: unconfirmed cut-downs are logged but never claimed
            unconfirmed = {r[0] for r in conn.execute(
                "SELECT occurrence_id FROM ad_occurrences WHERE status LIKE 'UNCONFIRMED%'")}
        except sqlite3.OperationalError:
            unconfirmed = set()
        for r in rep["airings"]:
            if r["occurrence_id"] in unconfirmed:
                continue
            if r["separation"] == "breach" and not r["evidence_available"]:
                _record_evidence(db_path, r, "separation")
            if r["position_compliance"] == "breach" and not r["evidence_available"]:
                _record_evidence(db_path, r, "position")
            if r["separation"] == "breach":
                near = ", ".join(c["brand"] for c in r["conflicts"] if not c["provisional"])
                basis = f"contract {r['contract_id']}" if r["contract_id"] else f"{cur.lower()} rate card"
                new += _insert_alert(conn, f"sep:{r['occurrence_id']}", "separation_breach", "critical",
                                     f"{r['brand']} aired within {r['separation_window_s']:.0f} s of {near} on {r['channel']} "
                                     f"({r['airing_start_cet'][11:16]} CET): separation clause breached, "
                                     f"{r['separation_credit']:.2f} {rep['currency']} reclaimable ({basis})",
                                     r["brand"], r["channel"], r["airing_start"], r["separation_credit"],
                                     meta=_alert_meta(r))
            if r["position_compliance"] == "breach":
                ref = f" [Contract {r['contract_id']}]" if r["contract_id"] else ""
                new += _insert_alert(conn, f"pos:{r['occurrence_id']}", "position_breach", "warning",
                                     f"{r['brand']} booked {r['booked_position']} but aired {r['delivered_position']} on "
                                     f"{r['channel']} ({r['airing_start_cet'][11:16]} CET): {r['position_credit']:.2f} "
                                     f"{rep['currency']} make-good{ref}", r["brand"], r["channel"], r["airing_start"],
                                     r["position_credit"], meta=_alert_meta(r))
        for c in rep["conflicts"]:
            if c["provisional"] or {c["a"]["occurrence_id"], c["b"]["occurrence_id"]} & unconfirmed:
                continue
            key = f"conf:{min(c['a']['occurrence_id'], c['b']['occurrence_id'])}:{max(c['a']['occurrence_id'], c['b']['occurrence_id'])}"
            new += _insert_alert(conn, key, "separation_conflict", "info",
                                 f"{c['a']['brand']} and {c['b']['brand']} ({c['sector']}) aired {c['gap_seconds']:.0f} s apart "
                                 f"on {c['channel']} ({c['b']['start_cet'][11:16]} CET)", None, c["channel"])
        for s in detect_competitor_surges(24, now=now, db_path=db_path):
            day = datetime.fromtimestamp(now, ALGERIA_TZ).date().isoformat()
            change = "new activity" if s["change_pct"] is None else f"+{s['change_pct']:.0f}%"
            new += _insert_alert(conn, f"surge:{s['brand']}:{day}", "competitor_surge", "warning",
                                 f"Competitor surge: {s['brand']} {s['previous_airings']} -> {s['current_airings']} "
                                 f"airings in 24 h ({change})", s["brand"])
    return new


def list_alerts(db_path, include_acknowledged: bool = False, limit: int = 50) -> list[dict]:
    q = "SELECT * FROM compliance_alerts" + ("" if include_acknowledged else " WHERE acknowledged_at IS NULL") + \
        " ORDER BY id DESC LIMIT ?"
    with get_db(db_path) as conn:
        out = [dict(r) for r in conn.execute(q, (limit,))]
        have = {r[0] for r in conn.execute("SELECT occurrence_id FROM ad_occurrences WHERE evidence_path IS NOT NULL")}
    for a in out:
        m = re.fullmatch(r"(?:sep|pos):(\d+)", a["key"] or "")
        a["occurrence_id"] = int(m.group(1)) if m else None
        a["evidence_available"] = a["occurrence_id"] in have
    return out


def acknowledge_alert(db_path, alert_id: int) -> bool:
    with get_db(db_path) as conn:
        return conn.execute("UPDATE compliance_alerts SET acknowledged_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                            "WHERE id = ? AND acknowledged_at IS NULL", (alert_id,)).rowcount == 1


# ---------------------------------------------------------------- contracts (DB override of the rate card)

def get_contracts(db_path) -> list[dict]:
    return RateCardEngine.load(db_path=db_path).config.get("contracts") or []


def save_contracts(db_path, contracts: list[dict]) -> list[dict]:
    """Replace the contract list (validated with the whole card; invalid input raises RateCardError)."""
    from kihc.analytics.rate_engine import _deep_merge, _load_db_override, save_db_override, validate
    current = _load_db_override(db_path) or {}
    override = dict(current, contracts=contracts)
    card = RateCardEngine.load(db_path=None)
    validate(_deep_merge(card.config, override))
    save_db_override(db_path, override)
    return contracts
