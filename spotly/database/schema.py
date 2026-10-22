"""SQLite persistence for Spotly.

Occurrence rows record *detected airings* only (source, timestamps, confidence).
They are not audience impressions and must never be reported as such.
"""
import contextvars
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from spotly.config import DEFAULT_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS reference_ads (
    ad_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_name      TEXT    NOT NULL,
    campaign_name   TEXT    NOT NULL,
    duration_sec    REAL    NOT NULL,
    fingerprint     BLOB    NOT NULL,          -- little-endian uint32 sub-fingerprints
    fp_algorithm    TEXT    NOT NULL,          -- guards against mixing incompatible fingerprints
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (brand_name, campaign_name)
);

CREATE TABLE IF NOT EXISTS ad_occurrences (
    occurrence_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ad_id               INTEGER NOT NULL REFERENCES reference_ads(ad_id) ON DELETE CASCADE,
    source_name         TEXT    NOT NULL,
    airing_start        REAL    NOT NULL,      -- seconds from start of source
    airing_end          REAL    NOT NULL,      -- seconds from start of source
    duration_detected   REAL    NOT NULL,
    confidence_score    REAL    NOT NULL,      -- 1 - bit error rate over the full ad window
    evidence_frame_path TEXT,
    human_verified      INTEGER NOT NULL DEFAULT 0,  -- 0 = unreviewed, 1 = verified, -1 = rejected
    detected_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (ad_id, source_name, airing_start)
);

CREATE INDEX IF NOT EXISTS idx_occ_ad ON ad_occurrences(ad_id);
CREATE INDEX IF NOT EXISTS idx_occ_source ON ad_occurrences(source_name, airing_start);

-- (service_heartbeat / service_channel_status live in spotly_telemetry.db since Phase 30: see spotly/database/telemetry_store.py)

-- Unsupervised discovery (Phase 11): recurring audio segments found without reference files.
-- 'unclassified' assets may be ads, promos, idents or programme openers; they enter spend
-- analytics only after a human classifies them (promoted_ad_id -> reference_ads).
CREATE TABLE IF NOT EXISTS discovered_assets (
    asset_id        TEXT PRIMARY KEY,                -- e.g. ELHEDDAF_AD_001
    channel         TEXT NOT NULL,
    duration_sec    REAL NOT NULL,
    fingerprint     BLOB NOT NULL,                   -- little-endian uint32 sub-fingerprints
    audio_path      TEXT,                            -- cached segment audio (WAV) for review
    status          TEXT NOT NULL DEFAULT 'unclassified',   -- unclassified | classified | ignored
    label           TEXT,
    promoted_ad_id  INTEGER,
    first_seen      REAL,
    last_seen       REAL,
    occurrences     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS discovered_occurrences (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id          TEXT NOT NULL REFERENCES discovered_assets(asset_id) ON DELETE CASCADE,
    channel           TEXT NOT NULL,
    airing_start      REAL NOT NULL,                 -- epoch seconds (capture clock)
    airing_end        REAL NOT NULL,
    duration_sec      REAL NOT NULL,
    confidence        REAL NOT NULL,                 -- 1 - BER between the two matched airings
    algeria_timestamp TEXT,
    detected_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (asset_id, channel, airing_start)
);
CREATE INDEX IF NOT EXISTS idx_disc_occ_time ON discovered_occurrences(channel, airing_start);

-- Phase 12.1: compliance / competitive alerts. `key` makes generation idempotent (one alert per event).
CREATE TABLE IF NOT EXISTS compliance_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    key             TEXT NOT NULL UNIQUE,
    kind            TEXT NOT NULL,      -- separation_breach | separation_conflict | position_breach | competitor_surge
    severity        TEXT NOT NULL,      -- critical | warning | info
    brand           TEXT,
    channel         TEXT,
    airing_start    REAL,
    message         TEXT NOT NULL,
    credit_dzd      TEXT,               -- Decimal as text; NULL when no contract clause applies
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    acknowledged_at TEXT
);

-- Phase 11.3: audit trail of asset merges (duplicate catalogue entries of the same airings).
-- The duplicate's own occurrence rows are never deleted; copied_ids lists the rows added to the master,
-- so a merge can be undone exactly (discovery.unmerge_asset).
CREATE TABLE IF NOT EXISTS asset_merges (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    master_id     TEXT NOT NULL,
    duplicate_id  TEXT NOT NULL,
    similarity    REAL NOT NULL,     -- best agreement of informative 3 s windows
    coverage      REAL NOT NULL,     -- share of the shorter asset found in the other at one consistent offset
    offset_sec    REAL NOT NULL,     -- duplicate start relative to master start
    reason        TEXT NOT NULL,
    same_airings  INTEGER NOT NULL,  -- duplicate airings that were the master's airings (not re-counted)
    copied_ids    TEXT NOT NULL,     -- JSON: discovered_occurrences ids created under the master
    merged_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    undone_at     TEXT
);

-- Financial rate-card override (Phase 8): one JSON document deep-merged over spotly/data/rate_cards.json.
CREATE TABLE IF NOT EXISTS rate_card_overrides (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    json        TEXT NOT NULL,
    updated_at  TEXT
);

-- Phase 18: dual-track contract matrix. A row here makes matching airings 'VERIFIED_CONTRACT_AUDIT' (Track A);
-- airings without a matching ACTIVE contract stay 'ILLUSTRATIVE_MARKET_ESTIMATE' (Track B, public rate cards).
CREATE TABLE IF NOT EXISTS client_contracts (
    contract_id                    TEXT PRIMARY KEY,
    brand_name                     TEXT NOT NULL,
    channel_id                     TEXT NOT NULL,
    booked_position                TEXT NOT NULL DEFAULT 'RUN_OF_BREAK'
        CHECK (booked_position IN ('FIRST_IN_POD', 'LAST_IN_POD', 'FIRST_OR_LAST', 'RUN_OF_BREAK')),
    separation_window_seconds      INTEGER NOT NULL DEFAULT 180,
    net_rate_per_second            REAL NOT NULL,
    demotion_penalty_pct           REAL NOT NULL DEFAULT 0.30,
    separation_breach_penalty_pct  REAL NOT NULL DEFAULT 1.00,
    contract_document_path         TEXT,
    status                         TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'EXPIRED')),
    created_at                     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    valid_from                     TIMESTAMP,
    valid_to                       TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_client_contracts_match ON client_contracts (channel_id, brand_name);

-- Append-only: every contract / rate card / rule change is a new row (triggers refuse UPDATE and DELETE).
CREATE TABLE IF NOT EXISTS contract_audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    entity       TEXT NOT NULL DEFAULT 'client_contract',   -- client_contract | rate_card | compliance_rule
    entity_id    TEXT NOT NULL,
    action       TEXT NOT NULL,                              -- CREATE | UPDATE | EXPIRE
    version      INTEGER NOT NULL,
    snapshot     TEXT NOT NULL,                              -- JSON of the full record after the change
    origin_hash  TEXT NOT NULL,                              -- SHA-256 of the requesting user/origin
    prev_hash    TEXT,                                       -- hash chain: previous row's entry_hash
    entry_hash   TEXT NOT NULL,
    recorded_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TRIGGER IF NOT EXISTS contract_audit_log_no_update BEFORE UPDATE ON contract_audit_log
BEGIN SELECT RAISE(ABORT, 'contract_audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS contract_audit_log_no_delete BEFORE DELETE ON contract_audit_log
BEGIN SELECT RAISE(ABORT, 'contract_audit_log is append-only'); END;
"""



BUSY_TIMEOUT_MS = 15_000          # wait up to 15 s for a lock instead of failing with "database is locked"

# path -> (inode, ctime) of the file whose schema this process already initialised
_INITIALISED: dict[str, tuple] = {}
_INIT_LOCK = threading.Lock()


_READ_ONLY: contextvars.ContextVar[bool] = contextvars.ContextVar("spotly_read_only", default=False)


@contextmanager
def read_only():
    """Every get_db/init_db inside this block (this thread / task only) opens the ledger `mode=ro` with `query_only = ON`:
    a dashboard read path can neither take the write lock nor mutate evidence, whatever code it calls."""
    token = _READ_ONLY.set(True)
    try:
        yield
    finally:
        _READ_ONLY.reset(token)


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA query_only = ON")
    return conn


def connect(db_path=None) -> sqlite3.Connection:
    """Plain connection with concurrency-safe settings (no schema work)."""
    path = Path(db_path or DEFAULT_DB_PATH)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA synchronous = NORMAL")       # safe with WAL; fsync at checkpoints, not every commit
    return conn


# Additive migrations: (table, column, type). Applied only when the column is missing,
# so databases created by earlier versions upgrade in place and keep all rows.
MIGRATIONS = [
    ("ad_occurrences", "verification_hash", "TEXT"),   # SHA-256 proof-of-performance hash (see core/proof.py)
    ("ad_occurrences", "algeria_timestamp", "TEXT"),   # airing start in Algeria time (CET, UTC+1), ISO 8601
    # Competitive intelligence (Phase 4.1)
    ("reference_ads", "industry", "TEXT"),                          # e.g. Telecom, Dairy, Electronics
    ("reference_ads", "ad_type", "TEXT NOT NULL DEFAULT 'self'"),   # 'self' (client) | 'competitor'
    ("reference_ads", "target_competitor_of", "TEXT"),              # client brand a competitor ad competes with
    # Commercial pods (Phase 4.2): optional cache written by pod_analyzer.materialize_pods(); pods are
    # recomputed from occurrences at query time, so these may be NULL or stale until re-materialized.
    ("ad_occurrences", "pod_id", "TEXT"),           # POD-YYYYMMDD-CHANNEL-NNN | POD-REC-<recording>-NNN
    ("ad_occurrences", "pod_position", "TEXT"),     # FIRST | MIDDLE | LAST | SOLO
    ("ad_occurrences", "pod_density", "INTEGER"),   # detected ads in the pod (lower bound on true density)
    # Discovery enrichment (Phase 11.2). Occurrence timestamps are never rewritten: duration_sec stays the
    # measured length; nominal_duration_sec is the snapped spot length used when an asset is promoted.
    ("discovered_assets", "nominal_duration_sec", "REAL"),
    ("discovered_assets", "duration_basis", "TEXT"),        # 'measured' | 'snapped to 30 s (measured 28.4 s)'
    ("discovered_assets", "audio_post_pad_sec", "REAL"),    # seconds of post-roll included in the cached clip
    ("discovered_assets", "transcript", "TEXT"),
    ("discovered_assets", "transcript_language", "TEXT"),
    ("discovered_assets", "transcription_model", "TEXT"),
    ("discovered_assets", "transcribed_at", "TEXT"),
    ("discovered_assets", "suggested_brand", "TEXT"),
    ("discovered_assets", "suggested_industry", "TEXT"),
    ("discovered_assets", "brand_confidence", "REAL"),
    ("discovered_assets", "brand_evidence", "TEXT"),        # JSON: matched aliases and how they matched
    ("discovered_assets", "content_hint", "TEXT"),          # e.g. 'station self-promo (El-Heddaf TV)'
    # Phase 11.3: de-duplication + visual packshot fallback
    ("discovered_assets", "merged_into", "TEXT"),           # status 'merged': its airings now count under this asset
    ("discovered_assets", "related_assets", "TEXT"),        # JSON: assets sharing part of the audio (not merged)
    ("discovered_assets", "suggestion_source", "TEXT"),     # 'audio' (speech) | 'packshot' (on-screen text)
    ("discovered_assets", "visual_status", "TEXT"),         # needed | captured | ocr_brand | ocr_no_brand | not_needed
    ("discovered_assets", "packshot_path", "TEXT"),         # one JPEG: frame 1.5 s before the spot's end
    ("discovered_assets", "packshot_at", "TEXT"),
    ("discovered_assets", "packshot_conf", "REAL"),          # acoustic agreement that located the spot's end
    ("discovered_assets", "packshot_attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("discovered_assets", "packshot_last_error", "TEXT"),
    ("discovered_assets", "ocr_text", "TEXT"),
    ("discovered_assets", "ocr_at", "TEXT"),
    # Phase 11.4: visual creative features of the packshot (one frame) and the cropped brand region
    ("discovered_assets", "visual_features", "TEXT"),       # JSON: programme_overlay, sharpness, logo_box/area, dhash...
    ("discovered_assets", "logo_path", "TEXT"),
    # Phase 19: tight breach-evidence clip (airing -3 s .. +3 s) and its SHA-256, written only when a breach is flagged
    ("ad_occurrences", "evidence_path", "TEXT"),      # relative to the evidence dir, e.g. breaches/<file>.wav
    ("ad_occurrences", "evidence_sha256", "TEXT"),
    # Phase 20: stored break-edge sensing for the airing (entry / exit context incl. boundary_confidence and cues)
    ("ad_occurrences", "boundary_json", "TEXT"),
    ("compliance_alerts", "metadata", "TEXT"),        # JSON: boundary_confidence, cues, contract / audit status
    ("discovered_assets", "transcript_logprob", "REAL"),   # hotfix: mean Whisper logprob of the kept speech (quality gate)
    # Phase 22: airing status (CONFIRMED | UNCONFIRMED_CUTDOWN_NO_DIP | REJECTED) and the broadcast-clock reading
    ("ad_occurrences", "status", "TEXT NOT NULL DEFAULT 'CONFIRMED'"),
    ("ad_occurrences", "clock_drift_seconds", "REAL"),   # capture time - broadcaster (MCR) time at this airing
    ("ad_occurrences", "clock_drift_margin_s", "REAL"),  # tolerance of that reading
    ("ad_occurrences", "clock_drift_basis", "TEXT"),     # program-date-time | pts-relative
    # Phase 26: calibrated MCR timeline NEXT TO the immutable ingest stamps (airing_start / algeria_timestamp / detected_at)
    ("ad_occurrences", "broadcast_start_cet", "TEXT"),   # ISO CET: ingest start + mcr_drift_delta_seconds
    ("ad_occurrences", "broadcast_end_cet", "TEXT"),
    ("ad_occurrences", "mcr_drift_delta_seconds", "REAL"),   # broadcast time - ingest time (< 0: broadcaster clock is earlier)
    # Phase 29: the SHA-256 of the 11 025 Hz audio clip written at confirmation, chained into verification_hash (NULL = the
    # row was sealed before clips were persisted, or no audio was buffered); and which process owns a channel-status row
    ("ad_occurrences", "seal_clip_sha256", "TEXT"),
    # Phase 29.2: 'UNAVAILABLE_LEGACY' = no clip exists and none can ever be cut (the in-RAM ring is gone): never retried
    ("ad_occurrences", "evidence_status", "TEXT"),
]
AD_TYPES = ("self", "competitor")


LEGACY_TELEMETRY_TABLES = ("service_heartbeat", "service_channel_status")      # moved out of the ledger in Phase 30


def _migrate(conn: sqlite3.Connection) -> None:
    for table in LEGACY_TELEMETRY_TABLES:         # operational state only (never evidence): the ledger holds no transient rows
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    for table, column, ctype in MIGRATIONS:
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")


def _file_identity(path: Path):
    try:
        st = os.stat(path)
        return st.st_ino, st.st_ctime_ns
    except OSError:
        return None


def init_db(db_path=None) -> sqlite3.Connection:
    """Connection to an initialised database.

    Schema creation, migrations and the switch to WAL run once per process per database file
    (re-run if the file is replaced). Previously they ran on every connection, so every web poll
    and worker write briefly contended for the write lock.
    """
    path = Path(db_path or DEFAULT_DB_PATH)
    memory = str(path) == ":memory:"
    key = str(path) if memory else os.path.realpath(path)
    ident = None if memory else _file_identity(path)
    if _READ_ONLY.get() and not memory:
        if ident is None or _INITIALISED.get(key) != ident:      # first touch in this process: schema work needs a writer, once
            token = _READ_ONLY.set(False)
            try:
                init_db(path).close()
            finally:
                _READ_ONLY.reset(token)
        return _connect_ro(path)
    conn = connect(path)
    if memory or ident is None or _INITIALISED.get(key) != ident:
        with _INIT_LOCK:
            if not memory:
                conn.execute("PRAGMA journal_mode = WAL")    # persistent: readers never block the writer
            conn.executescript(SCHEMA)
            _migrate(conn)
            conn.commit()
            if not memory:
                _INITIALISED[key] = _file_identity(path)
    return conn


@contextmanager
def get_db(db_path=None):
    """Managed connection: commit on success, roll back on error, always close.

        with get_db(path) as conn:
            conn.execute(...)
    """
    conn = init_db(db_path)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
