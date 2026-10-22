"""Dynamic rate cards and per-airing spend valuation (deterministic Decimal arithmetic).

Configuration sources, later ones deep-merged over earlier ones (lists are replaced, not merged):
  1. built-in fallback (same shape as spotly/data/rate_cards.json)
  2. JSON file: explicit path, else $SPOTLY_RATE_CARDS, else spotly/data/rate_cards.json
  3. database override: table rate_card_overrides (save_db_override / clear_db_override)
  4. runtime overrides dict passed to RateCardEngine.load(..., overrides=...)
An invalid configuration is never half-applied: the engine falls back to the built-in card and
records why in `.errors` (surfaced by the API and reports).

Valuation of one airing (all money in DZD, ROUND_HALF_UP to centimes):
  spot           = daypart_rate * channel_multiplier * season_multiplier * spot_seconds / base_spot_seconds
  premium        = spot * (booked position surcharge + booked adjacency surcharge)   [contract terms]
  gross          = spot + premium                                                    [tarif plaquette]
  net            = gross * (1 - agency_discount)
  breach credit  = spot * booked position surcharge * (1 - discount), when the booked First/Last
                   position was not delivered (SOLO satisfies both: it is first and last in its break)
  bonus value    = spot * surcharge of a First/Last position delivered without being booked (not billed)
Adjacency cannot be verified (no programme schedule): it is billed when contracted, status 'unverifiable'.
"""
import copy
import json
import os
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from spotly.config import DATA_DIR
from spotly.core.proof import ALGERIA_TZ
from spotly.database.schema import init_db

DEFAULT_PATH = DATA_DIR / "rate_cards.json"
ENV_VAR = "SPOTLY_RATE_CARDS"
CENT = Decimal("0.01")
POSITIONS = ("FIRST", "MIDDLE", "LAST", "SOLO")

BUILTIN = {
    "_meta": {"version": 1, "currency": "DZD", "base_spot_seconds": 30, "status": "ILLUSTRATIVE",
              "status_note": "Built-in fallback card (illustrative, not published tariffs)."},
    "defaults": {
        "agency_discount": 0.30, "rate_multiplier": 1.0,
        "dayparts": [{"name": "Prime Time", "start": "19:30", "end": "23:01", "rate": 220000},
                     {"name": "Daytime", "start": "12:00", "end": "19:30", "rate": 90000},
                     {"name": "Off-Peak / Late Night", "start": "23:01", "end": "12:00", "rate": 45000}],
        "position_surcharges": {"FIRST": 0.20, "LAST": 0.15, "ADJACENCY": 0.10}},
    "channels": {}, "seasons": [], "contracts": [],
}


class RateCardError(ValueError):
    pass


def money(x) -> Decimal:
    return Decimal(x).quantize(CENT, rounding=ROUND_HALF_UP)


def _dec(x) -> Decimal:
    return Decimal(str(x))


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    h, m = int(h), int(m)
    if not (0 <= h <= 24 and 0 <= m < 60) or h * 60 + m > 1440:
        raise RateCardError(f"invalid time {hhmm!r}")
    return h * 60 + m


def _covers(dp, minute: int) -> bool:
    s, e = _minutes(dp["start"]), _minutes(dp["end"])
    return s <= minute < e if s < e else (minute >= s or minute < e)       # wraps past midnight


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def validate(config: dict) -> None:
    """Raise RateCardError unless every daypart set tiles the 24 h day exactly once, values are sane."""
    meta = config.get("_meta", {})
    if int(meta.get("base_spot_seconds", 30)) <= 0:
        raise RateCardError("base_spot_seconds must be positive")
    d = config.get("defaults") or {}
    sets = [("defaults", d.get("dayparts"))] + [(f"channel {c}", v.get("dayparts"))
                                                for c, v in (config.get("channels") or {}).items() if v.get("dayparts")]
    for label, dps in sets:
        if not dps:
            raise RateCardError(f"{label}: no dayparts")
        for dp in dps:
            if _dec(dp["rate"]) < 0:
                raise RateCardError(f"{label}: negative rate in {dp['name']}")
        for minute in range(1440):
            hits = [dp["name"] for dp in dps if _covers(dp, minute)]
            if len(hits) != 1:
                hhmm = f"{minute // 60:02d}:{minute % 60:02d}"
                raise RateCardError(f"{label}: {hhmm} is covered by {hits or 'no daypart'} (must be exactly one)")
    if not 0 <= float(d.get("agency_discount", 0)) < 1:
        raise RateCardError("agency_discount must be in [0, 1)")
    for c in config.get("contracts") or []:
        if str(c.get("booked_position", "ANY")).upper() not in ("FIRST", "LAST", "ANY", "FIRST_OR_LAST"):
            raise RateCardError(f"contract for {c.get('brand')}: booked_position must be FIRST, LAST, FIRST_OR_LAST or ANY")
        if "pod_edge_credit" in c and not 0 <= float(c["pod_edge_credit"]) <= 1:
            raise RateCardError(f"contract for {c.get('brand')}: pod_edge_credit must be in [0, 1]")
        if "agency_discount" in c and not 0 <= float(c["agency_discount"]) < 1:
            raise RateCardError(f"contract for {c.get('brand')}: agency_discount must be in [0, 1)")
        # Phase 12.1 compliance clauses (spotly/analytics/compliance_audit.py)
        if "competitor_separation_seconds" in c and not 0 < float(c["competitor_separation_seconds"]) <= 3600:
            raise RateCardError(f"contract for {c.get('brand')}: competitor_separation_seconds must be in (0, 3600]")
        if "rate_override_dzd" in c and float(c["rate_override_dzd"]) < 0:
            raise RateCardError(f"contract for {c.get('brand')}: rate_override_dzd must not be negative")
        if "separation_credit" in c and not 0 <= float(c["separation_credit"]) <= 1:
            raise RateCardError(f"contract for {c.get('brand')}: separation_credit must be in [0, 1]")
        if not str(c.get("brand", "")).strip():
            raise RateCardError("contract without a brand")
    for s in config.get("seasons") or []:
        if date.fromisoformat(s["start"]) > date.fromisoformat(s["end"]) or float(s["multiplier"]) <= 0:
            raise RateCardError(f"season {s.get('name')}: invalid dates or multiplier")


# ---------------------------------------------------------------- DB overrides

def save_db_override(db_path, override: dict) -> None:
    """Store a validated override (deep-merged over the file card at load time)."""
    validate(_deep_merge(BUILTIN, override))
    conn = init_db(db_path)
    try:
        with conn:
            conn.execute("INSERT INTO rate_card_overrides (id, json, updated_at) VALUES (1, ?, "
                         "strftime('%Y-%m-%dT%H:%M:%SZ','now')) ON CONFLICT(id) DO UPDATE SET json=excluded.json, "
                         "updated_at=excluded.updated_at", (json.dumps(override),))
    finally:
        conn.close()


def clear_db_override(db_path) -> None:
    conn = init_db(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM rate_card_overrides")
    finally:
        conn.close()


def _load_db_override(db_path):
    conn = init_db(db_path)
    try:
        row = conn.execute("SELECT json FROM rate_card_overrides WHERE id = 1").fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


# ---------------------------------------------------------------- engine

class RateCardEngine:
    def __init__(self, config: dict, sources: list[str], errors: list[str] | None = None):
        self.config = config
        self.sources = sources
        self.errors = errors or []
        meta = config.get("_meta", {})
        self.currency = meta.get("currency", "DZD")
        self.base_seconds = _dec(meta.get("base_spot_seconds", 30))
        self.status = str(meta.get("status", "ILLUSTRATIVE")).upper()

    @property
    def is_illustrative(self) -> bool:
        return self.status != "VERIFIED"

    @classmethod
    def load(cls, path=None, db_path=None, overrides: dict | None = None) -> "RateCardEngine":
        config, sources, errors = copy.deepcopy(BUILTIN), ["built-in"], []
        file = Path(path or os.environ.get(ENV_VAR) or DEFAULT_PATH)
        try:
            config = _deep_merge(config, json.loads(file.read_text(encoding="utf-8")))
            sources.append(str(file))
        except FileNotFoundError:
            errors.append(f"rate card file not found: {file} (using built-in card)")
        except (ValueError, OSError) as exc:
            errors.append(f"rate card file unreadable: {file}: {exc} (using built-in card)")
        if db_path is not None:
            try:
                ov = _load_db_override(db_path)
                if ov:
                    config = _deep_merge(config, ov)
                    sources.append("database override")
            except (ValueError, OSError) as exc:
                errors.append(f"database override ignored: {exc}")
        if overrides:
            config = _deep_merge(config, overrides)
            sources.append("runtime override")
        try:
            validate(config)
        except (RateCardError, KeyError, ValueError, TypeError) as exc:
            errors.append(f"invalid rate card ({exc}); fell back to the built-in card")
            config, sources = copy.deepcopy(BUILTIN), ["built-in (fallback)"]
        return cls(config, sources, errors)

    # -------------------------------------------------------- lookups
    def channel_card(self, channel: str) -> dict:
        d = self.config["defaults"]
        ch = (self.config.get("channels") or {}).get(channel)
        if ch is None:   # case-insensitive match
            ch = next((v for k, v in (self.config.get("channels") or {}).items() if k.casefold() == channel.casefold()),
                      None)
        return dict(dayparts=(ch or {}).get("dayparts") or d["dayparts"],
                    multiplier=_dec((ch or {}).get("rate_multiplier", d.get("rate_multiplier", 1.0))),
                    rate_source="channel" if ch else "default")

    def daypart(self, channel: str, epoch: float) -> tuple[str, Decimal, str]:
        local = datetime.fromtimestamp(epoch, ALGERIA_TZ)
        minute = local.hour * 60 + local.minute
        card = self.channel_card(channel)
        dp = next(dp for dp in card["dayparts"] if _covers(dp, minute))      # validated: exactly one
        return dp["name"], _dec(dp["rate"]) * card["multiplier"], card["rate_source"]

    def season(self, epoch: float) -> tuple[str, Decimal]:
        day = datetime.fromtimestamp(epoch, ALGERIA_TZ).date()
        for s in self.config.get("seasons") or []:
            if date.fromisoformat(s["start"]) <= day <= date.fromisoformat(s["end"]):
                return s["name"], _dec(s["multiplier"])
        return "Standard", Decimal(1)

    def contract(self, brand: str, campaign: str) -> dict | None:
        b = (brand or "").casefold()
        matches = [c for c in self.config.get("contracts") or [] if str(c.get("brand", "")).casefold() == b
                   and c.get("campaign", "*") in ("*", campaign)]
        matches.sort(key=lambda c: c.get("campaign", "*") == "*")          # exact campaign beats '*'
        return matches[0] if matches else None

    def surcharge(self, key: str) -> Decimal:
        return _dec(self.config["defaults"]["position_surcharges"].get(key, 0))

    def discount(self, contract: dict | None) -> Decimal:
        return _dec((contract or {}).get("agency_discount", self.config["defaults"].get("agency_discount", 0.30)))


def compute_airing_financials(airing: dict, rate_card: RateCardEngine) -> dict:
    """Value one airing.

    airing keys: brand, campaign, channel, start_epoch (None = no broadcast clock, e.g. file scan),
    spot_seconds (nominal creative length), pod_position (FIRST|MIDDLE|LAST|SOLO|None).
    """
    out = dict(priced=False, unpriced_reason=None, daypart=None, season=None, rate_source=None,
               base_rate_30s=None, spot_value=money(0), gross_spend=money(0), estimated_net_spend=money(0),
               premium_value_applied=money(0), contract_loss_credit=money(0), contract_loss_credit_gross=money(0),
               bonus_premium_value=money(0), booked_position=None, delivered_position=airing.get("pod_position"),
               position_compliance="not booked", adjacency="not booked", agency_discount=None,
               rate_card_status=rate_card.status, placement_status=airing.get("placement_status"))
    if airing.get("start_epoch") is None:
        out["unpriced_reason"] = "no broadcast clock time (file scan offsets)"
        return out
    seconds = int(round(float(airing.get("spot_seconds") or 0)))
    if seconds <= 0:
        out["unpriced_reason"] = "unknown spot length"
        return out
    dp_name, rate, source = rate_card.daypart(airing["channel"], airing["start_epoch"])
    season_name, season_mult = rate_card.season(airing["start_epoch"])
    contract = rate_card.contract(airing.get("brand"), airing.get("campaign"))
    discount = rate_card.discount(contract)
    if contract and contract.get("rate_override_dzd") is not None:      # negotiated 30 s rate replaces the daypart card
        rate, source = _dec(contract["rate_override_dzd"]), "contract rate override"
    spot = rate * season_mult * Decimal(seconds) / rate_card.base_seconds

    booked = str((contract or {}).get("booked_position", "ANY")).upper()
    booked = booked if booked in ("FIRST", "LAST", "FIRST_OR_LAST") else None
    edge = booked == "FIRST_OR_LAST"               # either end of the pod protected; a MIDDLE airing is the breach
    booked_pct = (max(rate_card.surcharge("FIRST"), rate_card.surcharge("LAST")) if edge
                  else rate_card.surcharge(booked)) if booked else Decimal(0)
    adj_booked = bool((contract or {}).get("adjacency"))
    adj_pct = rate_card.surcharge("ADJACENCY") if adj_booked else Decimal(0)
    premium = spot * (booked_pct + adj_pct)
    gross = spot + premium
    delivered = airing.get("pod_position")

    credit_gross = Decimal(0)
    if booked:
        if delivered is None:
            compliance = "unknown position"
        elif delivered == booked or delivered == "SOLO" or (edge and delivered in ("FIRST", "LAST")):
            compliance = "compliant"
        elif airing.get("placement_status") == "demoted_ambiguous":
            compliance = "unverified placement"     # boundary cues too weak to prove a breach: no claim (Phase 20)
        else:
            compliance = "breach"
            # FIRST_OR_LAST contracts state the make-good as a share of the spot value (default 30 %)
            credit_gross = spot * (_dec(contract.get("pod_edge_credit", 0.30)) if edge else booked_pct)
    else:
        compliance = "not booked"
    bonus = Decimal(0)
    if not booked and delivered in ("FIRST", "LAST", "SOLO"):
        bonus = spot * rate_card.surcharge("FIRST" if delivered in ("FIRST", "SOLO") else "LAST")

    out.update(priced=True, daypart=dp_name, season=season_name, season_multiplier=float(season_mult),
               rate_source=source, base_rate_30s=money(rate), spot_seconds=seconds, spot_value=money(spot),
               premium_value_applied=money(premium), gross_spend=money(gross),
               estimated_net_spend=money(gross * (Decimal(1) - discount)), agency_discount=float(discount),
               booked_position=booked, position_compliance=compliance,
               adjacency="billed, unverifiable (no programme schedule)" if adj_booked else "not booked",
               contract_loss_credit_gross=money(credit_gross),
               contract_loss_credit=money(credit_gross * (Decimal(1) - discount)),
               bonus_premium_value=money(bonus))
    return out
