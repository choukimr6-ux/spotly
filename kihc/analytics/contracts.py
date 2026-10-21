"""Phase 18: client insertion-order contracts (Track A) - ingestion, matching and the append-only audit log.

Dual-track status carried by every audited airing:
  * VERIFIED_CONTRACT_AUDIT      - bound to an ACTIVE ingested contract (green badge)
  * ILLUSTRATIVE_MARKET_ESTIMATE - priced from the public rate card, no contract (amber badge)

Validity window is half-open: valid_from <= airing_start < valid_to. A naive timestamp is read as Algeria time
(CET, UTC+1) and stored as UTC ISO-8601; a date-only `valid_to` covers that whole day. Every create/update/expire
appends a hash-chained row to `contract_audit_log` (SQLite triggers refuse UPDATE/DELETE on that table), so any
past state can be reproduced.
"""
import hashlib
import json
from datetime import datetime, timezone

from kihc.core.proof import ALGERIA_TZ
from kihc.data import db_queue
from kihc.database.schema import get_db

POSITIONS = ("FIRST_IN_POD", "LAST_IN_POD", "FIRST_OR_LAST", "RUN_OF_BREAK")
STATUS_VERIFIED = "VERIFIED_CONTRACT_AUDIT"
STATUS_ESTIMATE = "ILLUSTRATIVE_MARKET_ESTIMATE"
BADGES = {STATUS_VERIFIED: ("VERIFIED CONTRACT AUDIT", "green"),
          STATUS_ESTIMATE: ("ILLUSTRATIVE MARKET ESTIMATE", "amber")}
FIELDS = ("contract_id", "brand_name", "channel_id", "booked_position", "separation_window_seconds",
          "net_rate_per_second", "demotion_penalty_pct", "separation_breach_penalty_pct",
          "contract_document_path", "status", "valid_from", "valid_to")


class ContractError(ValueError):
    pass


def origin_hash(origin: str | None) -> str:
    return hashlib.sha256((origin or "unknown").encode("utf-8")).hexdigest()


def to_epoch(value, *, end_of_day: bool = False) -> float | None:
    """ISO date/datetime (or epoch number) -> UTC epoch seconds; None stays None."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise ContractError(f"invalid timestamp {value!r}; use ISO-8601 (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ALGERIA_TZ)
    return dt.timestamp() + (86400 if (len(text) <= 10 and end_of_day) else 0)


def _iso(epoch: float | None) -> str | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise(fields: dict) -> dict:
    f = {k: fields.get(k) for k in FIELDS}
    for k in ("contract_id", "brand_name", "channel_id"):
        f[k] = (f[k] or "").strip()
        if not f[k]:
            raise ContractError(f"{k} is required")
    f["booked_position"] = (f["booked_position"] or "RUN_OF_BREAK").upper()
    if f["booked_position"] not in POSITIONS:
        raise ContractError(f"booked_position must be one of {', '.join(POSITIONS)}")
    f["status"] = (f["status"] or "ACTIVE").upper()
    if f["status"] not in ("ACTIVE", "EXPIRED"):
        raise ContractError("status must be ACTIVE or EXPIRED")
    try:
        f["net_rate_per_second"] = float(f["net_rate_per_second"])
        f["separation_window_seconds"] = int(180 if f["separation_window_seconds"] is None
                                             else f["separation_window_seconds"])
        f["demotion_penalty_pct"] = float(0.30 if f["demotion_penalty_pct"] is None else f["demotion_penalty_pct"])
        f["separation_breach_penalty_pct"] = float(1.00 if f["separation_breach_penalty_pct"] is None
                                                   else f["separation_breach_penalty_pct"])
    except (TypeError, ValueError):
        raise ContractError("rate, window and penalty fields must be numbers")
    if f["net_rate_per_second"] <= 0:
        raise ContractError("net_rate_per_second must be > 0")
    if not 0 <= f["separation_window_seconds"] <= 3600:
        raise ContractError("separation_window_seconds must be within 0-3600")
    for k in ("demotion_penalty_pct", "separation_breach_penalty_pct"):
        if not 0 <= f[k] <= 1:
            raise ContractError(f"{k} must be within 0-1")
    lo, hi = to_epoch(f["valid_from"]), to_epoch(f["valid_to"], end_of_day=True)
    if lo is not None and hi is not None and hi <= lo:
        raise ContractError("valid_to must be after valid_from")
    f["valid_from"], f["valid_to"] = _iso(lo), _iso(hi)
    return f


def _entry_digest(prev, entity, entity_id, action, version, who, body) -> str:
    return hashlib.sha256(f"{prev}|{entity}|{entity_id}|{action}|{version}|{who}|{body}".encode()).hexdigest()


def _append_log(conn, entity_id: str, action: str, snapshot: dict, origin: str | None,
                entity: str = "client_contract") -> int:
    last = conn.execute("SELECT entry_hash FROM contract_audit_log ORDER BY id DESC LIMIT 1").fetchone()
    version = conn.execute("SELECT COUNT(*) FROM contract_audit_log WHERE entity = ? AND entity_id = ?",
                           (entity, entity_id)).fetchone()[0] + 1
    body = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    who, prev = origin_hash(origin), (last[0] if last else None)
    conn.execute("INSERT INTO contract_audit_log (entity, entity_id, action, version, snapshot, origin_hash, "
                 "prev_hash, entry_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (entity, entity_id, action, version, body, who, prev,
                  _entry_digest(prev, entity, entity_id, action, version, who, body)))
    return version


def log_change(db_path, entity: str, entity_id: str, action: str, snapshot: dict, origin: str | None = None) -> int:
    """Record a rate-card / compliance-rule change (or any other versioned change) in the append-only log."""
    with get_db(db_path) as conn:
        return _append_log(conn, entity_id, action, snapshot, origin, entity)


def _overlap(conn, f: dict) -> str | None:
    lo, hi = to_epoch(f["valid_from"]), to_epoch(f["valid_to"])
    rows = conn.execute("SELECT contract_id, valid_from, valid_to FROM client_contracts WHERE status = 'ACTIVE' "
                        "AND lower(brand_name) = lower(?) AND lower(channel_id) = lower(?) AND contract_id != ?",
                        (f["brand_name"], f["channel_id"], f["contract_id"])).fetchall()
    for r in rows:
        rlo, rhi = to_epoch(r["valid_from"]), to_epoch(r["valid_to"])
        if (hi is None or rlo is None or rlo < hi) and (rhi is None or lo is None or lo < rhi):
            return r["contract_id"]
    return None


def _apply_contract(conn, *, fields: dict, origin: str | None = None) -> dict:
    """Writer-side body (inside the DatabaseWriteQueue transaction): validate, upsert, append the audit row."""
    f = _normalise(fields)
    if f["status"] == "ACTIVE":
        clash = _overlap(conn, f)
        if clash:
            raise ContractError(f"conflicting date range: {f['brand_name']} on {f['channel_id']} already has "
                                f"active contract {clash} overlapping this period")
    old = conn.execute("SELECT 1 FROM client_contracts WHERE contract_id = ?", (f["contract_id"],)).fetchone()
    if old:
        conn.execute("UPDATE client_contracts SET brand_name=?, channel_id=?, booked_position=?, "
                     "separation_window_seconds=?, net_rate_per_second=?, demotion_penalty_pct=?, "
                     "separation_breach_penalty_pct=?, contract_document_path=?, status=?, valid_from=?, valid_to=? "
                     "WHERE contract_id = ?", (*[f[k] for k in FIELDS[1:]], f["contract_id"]))
        action = "EXPIRE" if f["status"] == "EXPIRED" else "UPDATE"
    else:
        conn.execute(f"INSERT INTO client_contracts ({', '.join(FIELDS)}) VALUES ({', '.join('?' * len(FIELDS))})",
                     [f[k] for k in FIELDS])
        action = "CREATE"
    return dict(f, version=_append_log(conn, f["contract_id"], action, f, origin), action=action)


db_queue.register("contract_ingest", _apply_contract)


def ingest_contract(db_path, fields: dict, origin: str | None = None, timeout: float = 60.0) -> dict:
    """Persist through the single-writer DatabaseWriteQueue (WAL); waits until the batch is committed."""
    return db_queue.get_queue(db_path).submit("contract_ingest", urgent=True, fields=fields,
                                              origin=origin).result(timeout)


def list_contracts(db_path, status: str | None = None) -> list[dict]:
    q = "SELECT * FROM client_contracts" + (" WHERE status = ?" if status else "") + " ORDER BY created_at, contract_id"
    with get_db(db_path) as conn:
        return [dict(r) for r in conn.execute(q, (status,) if status else ())]


def audit_log(db_path, entity_id: str | None = None) -> list[dict]:
    q = "SELECT * FROM contract_audit_log" + (" WHERE entity_id = ?" if entity_id else "") + " ORDER BY id"
    with get_db(db_path) as conn:
        return [dict(r) for r in conn.execute(q, (entity_id,) if entity_id else ())]


def verify_audit_chain(db_path) -> bool:
    """True when every audit row's chain link and digest still match (detects tampering with the file)."""
    prev = None
    for r in audit_log(db_path):
        want = _entry_digest(prev, r["entity"], r["entity_id"], r["action"], r["version"], r["origin_hash"],
                             r["snapshot"])
        if r["prev_hash"] != prev or r["entry_hash"] != want:
            return False
        prev = r["entry_hash"]
    return True


def load_active(db_path) -> list[dict]:
    with get_db(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM client_contracts WHERE status = 'ACTIVE'")]


def match_contract(contracts: list[dict], channel_id: str, brand_name: str, timestamp: float) -> dict | None:
    """Active contract for [channel_id, brand_name, timestamp] (case-insensitive); valid_from <= t < valid_to."""
    ch, br = (channel_id or "").casefold(), (brand_name or "").casefold()
    for c in contracts:
        if c["status"] != "ACTIVE" or c["channel_id"].casefold() != ch or c["brand_name"].casefold() != br:
            continue
        lo, hi = to_epoch(c["valid_from"]), to_epoch(c["valid_to"])
        if (lo is None or timestamp >= lo) and (hi is None or timestamp < hi):
            return c
    return None
