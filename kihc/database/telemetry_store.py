"""Phase 30: volatile service telemetry lives in its own SQLite file, never in the forensic ledger.

`kihc.db` holds only persistent, tamper-evident domain records. The supervisor's heartbeat (every tick) and per-channel
radar status (levels, buffers, retry state, every few seconds) are operational state: losing them costs nothing, but writing
them next to the ledger made every status tick contend for the ledger's single write lock (SQLITE_BUSY).

    <db>.db  ->  <db>_telemetry.db      (WAL, synchronous = NORMAL, busy_timeout 15 s; safe to delete at any time)

One writer (the supervisor), many readers (the 8502 API, the CLI); readers and writer never touch the ledger's locks.
"""
import os
import sqlite3
import threading
from pathlib import Path

from kihc.config import DEFAULT_DB_PATH
from kihc.database.schema import BUSY_TIMEOUT_MS

SCHEMA = """
CREATE TABLE IF NOT EXISTS service_heartbeat (
    service_id      INTEGER PRIMARY KEY CHECK (service_id = 1),   -- single row
    pid             INTEGER,
    hostname        TEXT,
    state           TEXT,          -- starting | running | stopping | stopped
    started_at      TEXT,
    heartbeat_at    TEXT,
    heartbeat_epoch REAL,
    tick_seconds    REAL,
    rss_mb          REAL,
    peak_rss_mb     REAL,
    ffmpeg_rss_mb   REAL,
    channels        INTEGER,
    restarts        INTEGER,
    gc_runs         INTEGER,
    last_gc_at      TEXT,
    message         TEXT
);

CREATE TABLE IF NOT EXISTS service_channel_status (
    channel         TEXT PRIMARY KEY,
    state           TEXT,
    updated_epoch   REAL,
    restarts        INTEGER,
    status_json     TEXT,          -- LiveStreamWorker.status() snapshot (radar telemetry)
    owner_pid       INTEGER        -- the supervisor process that wrote the row
);
"""

_READY: dict[str, tuple] = {}
_LOCK = threading.Lock()


def telemetry_path(db_path=None) -> Path:
    """kihc.db -> kihc_telemetry.db (same folder); ':memory:' stays private to the connection."""
    path = Path(db_path or DEFAULT_DB_PATH)
    if str(path) == ":memory:":
        return path
    return path.with_name(f"{path.stem}_telemetry{path.suffix or '.db'}")


def _identity(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_ctime_ns)


def connect(db_path=None) -> sqlite3.Connection:
    """Connection to the telemetry database next to `db_path` (created on first use)."""
    path = telemetry_path(db_path)
    memory = str(path) == ":memory:"
    if not memory:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA synchronous = NORMAL")
    key = str(path) if memory else os.path.realpath(path)
    if memory or _READY.get(key) != _identity(path):
        with _LOCK:
            if not memory:
                conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            conn.commit()
            if not memory:
                _READY[key] = _identity(path)
    return conn
