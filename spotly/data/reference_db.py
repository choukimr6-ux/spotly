"""Reference acoustic signatures (Phase 13.1): one row per verified commercial.

    from spotly.data.reference_db import ensure_schema, upsert_signature, load_signatures

audio_fingerprint_blob layout (little-endian, 6 bytes per hash):
    header  : magic b"KRS1", uint32 n, float32 lead_trim_s, float32 trimmed_s
    payload : n x (uint32 hash, uint16 t_cs)      t_cs = anchor time in 10 ms from the *trimmed* start
`lead_trim_s` is the silence removed before the first sound; the matcher adds it back so the snapped airing
starts where the creative's (untrimmed) first frame was on air.

All writes go through spotly.database.schema (WAL, busy_timeout, commit/rollback): a reader never sees half of
a match (occurrence row without its match_count bump) and a failed write leaves nothing behind.
"""
import sqlite3
import struct
from dataclasses import dataclass

import numpy as np

from spotly.data import db_queue
from spotly.database.schema import get_db

DDL = """
CREATE TABLE IF NOT EXISTS reference_signatures (
    asset_id               TEXT PRIMARY KEY,
    brand_name             TEXT NOT NULL,
    canonical_sector       TEXT,
    nominal_duration       REAL NOT NULL,
    audio_fingerprint_blob BLOB NOT NULL,
    visual_p_hash          TEXT,
    created_at             TIMESTAMP NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    last_matched_at        TIMESTAMP,
    match_count            INTEGER NOT NULL DEFAULT 0,
    ad_id                  INTEGER REFERENCES reference_ads(ad_id) ON DELETE SET NULL,  -- master assets (Phase 16)
    master_asset_id        TEXT REFERENCES reference_signatures(asset_id) ON DELETE SET NULL,   -- Phase 21 family
    cut_type               TEXT NOT NULL DEFAULT 'MASTER' CHECK (cut_type IN ('MASTER','CUT_DOWN','TEASER','TAGGED')),
    voice_ratio            REAL,                                   -- Phase 22: estimated share of voice-over (0..1)
    is_music_dominant      INTEGER NOT NULL DEFAULT 0              -- Phase 22: 1 = packshot confirmation is mandatory
);
"""

CUT_TYPES = ("MASTER", "CUT_DOWN", "TEASER", "TAGGED")
MAGIC = b"KRS1"
_HEADER = struct.Struct("<4sIff")
_ROW = np.dtype([("h", "<u4"), ("t", "<u2")])
MAX_T_CS = 0xFFFF                      # 655.35 s: longer creatives are not commercials


@dataclass
class Signature:
    asset_id: str
    brand_name: str
    canonical_sector: str | None
    nominal_duration: float
    codes: np.ndarray            # uint32 hashes
    t_cs: np.ndarray             # uint16 anchor times (10 ms) from the trimmed start
    lead_trim_s: float
    trimmed_s: float
    visual_p_hash: str | None = None
    match_count: int = 0
    last_matched_at: str | None = None
    ad_id: int | None = None
    master_asset_id: str | None = None      # Phase 21: parent of a cut-down / teaser / tagged edit
    cut_type: str = "MASTER"
    voice_ratio: float | None = None
    is_music_dominant: bool = False


def family_tree(signatures) -> dict:
    """asset_id -> {root, parent, cut_type, depth}: the campaign trees, resolved in memory (hot-reloaded with the index).

    A parent that is missing from the registry (deleted, not yet registered) or a cycle never raises: the asset is
    simply treated as the root of its own family, so a bad row can't take the live monitor down."""
    parent = {s.asset_id: getattr(s, "master_asset_id", None) for s in signatures}
    kind = {s.asset_id: getattr(s, "cut_type", None) or "MASTER" for s in signatures}
    out = {}
    for a in parent:
        seen, cur = [a], a
        while parent.get(cur) in parent and parent[cur] not in seen:
            cur = parent[cur]
            seen.append(cur)
        out[a] = dict(root=cur, parent=parent[a] if parent[a] in parent else None, cut_type=kind[a], depth=len(seen) - 1)
    return out


def pack_blob(codes: np.ndarray, t_cs: np.ndarray, lead_trim_s: float, trimmed_s: float) -> bytes:
    if codes.size != t_cs.size:
        raise ValueError("codes and times must have the same length")
    if t_cs.size and int(t_cs.max()) > MAX_T_CS:
        raise ValueError("signature longer than 655 s")
    rows = np.empty(codes.size, dtype=_ROW)
    rows["h"], rows["t"] = codes, t_cs
    return _HEADER.pack(MAGIC, codes.size, lead_trim_s, trimmed_s) + rows.tobytes()


def unpack_blob(blob: bytes):
    """-> (codes uint32, t_cs uint16, lead_trim_s, trimmed_s)"""
    if len(blob) < _HEADER.size:
        raise ValueError("fingerprint blob truncated")
    magic, n, lead, trimmed = _HEADER.unpack_from(blob)
    if magic != MAGIC or len(blob) != _HEADER.size + n * _ROW.itemsize:
        raise ValueError("not a KRS1 fingerprint blob (wrong magic or length)")
    rows = np.frombuffer(blob, dtype=_ROW, count=n, offset=_HEADER.size)
    return rows["h"].astype(np.uint32), rows["t"].astype(np.uint16), float(lead), float(trimmed)


def ensure_schema(conn: sqlite3.Connection) -> None:
    if conn.execute("PRAGMA query_only").fetchone()[0]:      # Phase 30 read-only dashboard connection: the writer ensured it
        return
    conn.executescript(DDL)
    if "ad_id" not in {r[1] for r in conn.execute("PRAGMA table_info(reference_signatures)")}:
        conn.execute("ALTER TABLE reference_signatures ADD COLUMN ad_id INTEGER")       # tables from Phase 13.1
    have = {r[1] for r in conn.execute("PRAGMA table_info(reference_signatures)")}
    if "master_asset_id" not in have:                                                  # tables from before Phase 21
        conn.execute("ALTER TABLE reference_signatures ADD COLUMN master_asset_id TEXT "
                     "REFERENCES reference_signatures(asset_id) ON DELETE SET NULL")
    if "cut_type" not in have:
        conn.execute("ALTER TABLE reference_signatures ADD COLUMN cut_type TEXT NOT NULL DEFAULT 'MASTER' "
                     "CHECK (cut_type IN ('MASTER','CUT_DOWN','TEASER','TAGGED'))")
    if "voice_ratio" not in have:                                                      # tables from before Phase 22
        conn.execute("ALTER TABLE reference_signatures ADD COLUMN voice_ratio REAL")
    if "is_music_dominant" not in have:
        conn.execute("ALTER TABLE reference_signatures ADD COLUMN is_music_dominant INTEGER NOT NULL DEFAULT 0")


def upsert_signature(conn: sqlite3.Connection, asset_id: str, brand_name: str, canonical_sector: str | None,
                     nominal_duration: float, blob: bytes, visual_p_hash: str | None, ad_id: int | None = None,
                     master_asset_id: str | None = None, cut_type: str | None = None,
                     voice_ratio: float | None = None, music_dominant: bool | None = None) -> None:
    """Insert or refresh a signature. A rebuild keeps the airing statistics (match_count, last_matched_at).
    `ad_id` links a master asset (registered before it ever aired) to its reference ad; discovered assets get theirs
    from discovered_assets.promoted_ad_id. created_at is refreshed so signature_version() notices a re-registration.
    `master_asset_id` / `cut_type` (Phase 21) place the asset in a campaign family; a rebuild that passes neither keeps them.
    `voice_ratio` / `music_dominant` (Phase 22) record the vocal-activity analysis; music-dominant spots need a packshot
    match before an airing is CONFIRMED. A rebuild that passes neither keeps the stored values."""
    if cut_type is not None and cut_type not in CUT_TYPES:
        raise ValueError(f"cut_type must be one of {CUT_TYPES}")
    if master_asset_id is not None and master_asset_id == asset_id:
        raise ValueError("an asset cannot be its own master")
    conn.execute(
        "INSERT INTO reference_signatures (asset_id, brand_name, canonical_sector, nominal_duration, "
        "audio_fingerprint_blob, visual_p_hash, ad_id, master_asset_id, cut_type, voice_ratio, is_music_dominant) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, 'MASTER'), ?, COALESCE(?, 0)) "
        "ON CONFLICT(asset_id) DO UPDATE SET brand_name = excluded.brand_name, "
        "canonical_sector = excluded.canonical_sector, nominal_duration = excluded.nominal_duration, "
        "audio_fingerprint_blob = excluded.audio_fingerprint_blob, visual_p_hash = excluded.visual_p_hash, "
        "ad_id = COALESCE(excluded.ad_id, ad_id), master_asset_id = COALESCE(excluded.master_asset_id, master_asset_id), "
        "cut_type = CASE WHEN ? IS NULL THEN cut_type ELSE excluded.cut_type END, "
        "voice_ratio = COALESCE(excluded.voice_ratio, voice_ratio), "
        "is_music_dominant = CASE WHEN ? IS NULL THEN is_music_dominant ELSE excluded.is_music_dominant END, "
        "created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        (asset_id, brand_name, canonical_sector, float(nominal_duration), blob, visual_p_hash, ad_id,
         master_asset_id, cut_type, voice_ratio, None if music_dominant is None else int(bool(music_dominant)), cut_type,
         None if music_dominant is None else int(bool(music_dominant))))


def load_signatures(db_path=None) -> list[Signature]:
    with get_db(db_path) as conn:
        ensure_schema(conn)
        rows = conn.execute("SELECT * FROM reference_signatures ORDER BY asset_id").fetchall()
    out = []
    for r in rows:
        codes, t_cs, lead, trimmed = unpack_blob(r["audio_fingerprint_blob"])
        out.append(Signature(r["asset_id"], r["brand_name"], r["canonical_sector"], r["nominal_duration"],
                             codes, t_cs, lead, trimmed, r["visual_p_hash"], r["match_count"], r["last_matched_at"], r["ad_id"],
                             r["master_asset_id"], r["cut_type"] or "MASTER", r["voice_ratio"],
                             bool(r["is_music_dominant"])))
    return out


def signature_version(db_path=None) -> tuple:
    """Cheap change detector for hot-reloading the in-memory index: (row count, newest created_at, total blob bytes)."""
    with get_db(db_path) as conn:
        ensure_schema(conn)
        r = conn.execute("SELECT COUNT(*), MAX(created_at), COALESCE(SUM(LENGTH(audio_fingerprint_blob)), 0) "
                         "FROM reference_signatures").fetchone()
    return tuple(r)


def _apply_airing(conn, *, asset_id: str, ad_id: int, source_name: str, airing_start: float, airing_end: float,
                  duration: float, confidence: float, algeria_timestamp: str, verification_hash: str,
                  detected_at: str, penalty: dict | None = None, status: str = "CONFIRMED",
                  clock_drift_seconds: float | None = None, clock_drift_margin_s: float | None = None,
                  clock_drift_basis: str | None = None, broadcast_start_cet: str | None = None,
                  broadcast_end_cet: str | None = None, mcr_drift_delta_seconds: float | None = None,
                  evidence_frame_path: str | None = None, evidence_path: str | None = None,
                  evidence_sha256: str | None = None, seal_clip_sha256: str | None = None) -> int | None:
    """Writer-side body (runs inside the queue's BEGIN IMMEDIATE, so there is no check-then-insert race):
    occurrence row + match_count / last_matched_at bump + optional contract-penalty alert, all or nothing.

    Phase 29: `evidence_path` / `evidence_sha256` / `seal_clip_sha256` bind the 11 025 Hz clip written at confirmation; the
    caller has already chained the same digest into `verification_hash`. `evidence_frame_path` is the spectrogram PNG of the
    legacy BER path."""
    conn.execute(DDL)                      # one statement; executescript() would COMMIT the writer's open transaction
    if conn.execute("SELECT 1 FROM ad_occurrences WHERE ad_id = ? AND source_name = ? AND human_verified != -1 "
                    "AND airing_start < ? AND airing_end > ? LIMIT 1",
                    (ad_id, source_name, airing_end - 2.0, airing_start + 2.0)).fetchone():
        return None
    cur = conn.execute(
        "INSERT OR IGNORE INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
        "confidence_score, detected_at, verification_hash, algeria_timestamp, status, clock_drift_seconds, "
        "clock_drift_margin_s, clock_drift_basis, broadcast_start_cet, broadcast_end_cet, mcr_drift_delta_seconds, "
        "evidence_frame_path, evidence_path, evidence_sha256, seal_clip_sha256) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ad_id, source_name, airing_start, airing_end, duration, confidence, detected_at, verification_hash,
         algeria_timestamp, status, clock_drift_seconds, clock_drift_margin_s, clock_drift_basis,
         broadcast_start_cet, broadcast_end_cet, mcr_drift_delta_seconds,
         evidence_frame_path, evidence_path, evidence_sha256, seal_clip_sha256))
    if not cur.rowcount:
        return None
    conn.execute("UPDATE reference_signatures SET match_count = match_count + 1, "
                 "last_matched_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE asset_id = ?", (asset_id,))
    if penalty:
        _apply_penalty(conn, **penalty)
    return cur.lastrowid


def _apply_penalty(conn, *, key: str, kind: str, severity: str, message: str, brand=None, channel=None,
                   airing_start=None, credit_dzd=None) -> bool:
    cur = conn.execute("INSERT OR IGNORE INTO compliance_alerts (key, kind, severity, brand, channel, airing_start, "
                       "message, credit_dzd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (key, kind, severity, brand, channel, airing_start, message, credit_dzd))
    return cur.rowcount == 1


def _apply_signature_touch(conn, *, asset_id: str, n: int = 1) -> None:
    conn.execute("UPDATE reference_signatures SET match_count = match_count + ?, "
                 "last_matched_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE asset_id = ?", (n, asset_id))


AIRING_STATUSES = ("CONFIRMED", "UNCONFIRMED_CUTDOWN_NO_DIP", "UNCONFIRMED_MUSIC_NO_PACKSHOT", "REJECTED")


def _apply_airing_status(conn, *, occurrence_id: int, status: str) -> bool:
    if status not in AIRING_STATUSES:
        raise ValueError(f"status must be one of {AIRING_STATUSES}")
    return conn.execute("UPDATE ad_occurrences SET status = ? WHERE occurrence_id = ?", (status, occurrence_id)).rowcount == 1


db_queue.register("airing", _apply_airing)
db_queue.register("airing_status", _apply_airing_status)
db_queue.register("penalty", _apply_penalty)
db_queue.register("signature_touch", _apply_signature_touch)


def submit_airing(db_path, *, urgent: bool = False, **fields):
    """Fire-and-forget: enqueue one airing (+ optional `penalty=` dict) and return its Ticket without waiting."""
    return db_queue.get_queue(db_path).submit("airing", urgent=urgent, **fields)


def commit_airing(db_path, *, asset_id: str, ad_id: int, source_name: str, airing_start: float, airing_end: float,
                  duration: float, confidence: float, algeria_timestamp: str, verification_hash: str,
                  detected_at: str, penalty: dict | None = None, status: str = "CONFIRMED",
                  clock_drift_seconds: float | None = None, clock_drift_margin_s: float | None = None,
                  clock_drift_basis: str | None = None, broadcast_start_cet: str | None = None,
                  broadcast_end_cet: str | None = None, mcr_drift_delta_seconds: float | None = None,
                  evidence_frame_path: str | None = None, evidence_path: str | None = None,
                  evidence_sha256: str | None = None, seal_clip_sha256: str | None = None) -> int | None:
    """Log one confirmed airing, bump the signature's match statistics and (optionally) record a contract
    penalty, as ONE atomic unit routed through the central write queue (spotly.data.db_queue).

    Returns the new occurrence_id, or None when that airing is already on file: either the same row
    (UNIQUE(ad_id, source, start)) or the same ad overlapping it on the same channel, e.g. logged by the
    BER matcher or backfilled by promote_asset with a slightly different start. The counters only move for a
    genuinely new row, so re-delivery never inflates match_count or double-counts spend.
    Blocks until the batch containing it is committed (use submit_airing() to avoid waiting); a failing event
    (e.g. unknown ad_id) raises its own error and leaves nothing behind.
    """
    return submit_airing(db_path, urgent=True, asset_id=asset_id, ad_id=ad_id, source_name=source_name,
                         airing_start=airing_start, airing_end=airing_end, duration=duration, confidence=confidence,
                         algeria_timestamp=algeria_timestamp, verification_hash=verification_hash,
                         detected_at=detected_at, penalty=penalty, status=status,
                         clock_drift_seconds=clock_drift_seconds, clock_drift_margin_s=clock_drift_margin_s,
                         clock_drift_basis=clock_drift_basis, broadcast_start_cet=broadcast_start_cet,
                         broadcast_end_cet=broadcast_end_cet, mcr_drift_delta_seconds=mcr_drift_delta_seconds,
                         evidence_frame_path=evidence_frame_path, evidence_path=evidence_path,
                         evidence_sha256=evidence_sha256, seal_clip_sha256=seal_clip_sha256).result()


def set_airing_status(db_path, occurrence_id: int, status: str) -> bool:
    """Resolve an airing's status (e.g. UNCONFIRMED_CUTDOWN_NO_DIP -> CONFIRMED once the packshot is seen)."""
    return db_queue.get_queue(db_path).submit("airing_status", urgent=True, occurrence_id=occurrence_id,
                                              status=status).result()


def promoted_ad_id(db_path, asset_id: str) -> int | None:
    """reference_ads.ad_id a confirmed discovered asset was promoted to (the id compliance/financials key on)."""
    with get_db(db_path) as conn:
        r = conn.execute("SELECT ad_id FROM reference_signatures WHERE asset_id = ?", (asset_id,)).fetchone()
        if r and r["ad_id"] is not None:
            return r["ad_id"]
        r = conn.execute("SELECT promoted_ad_id FROM discovered_assets WHERE asset_id = ?", (asset_id,)).fetchone()
    return r["promoted_ad_id"] if r else None
