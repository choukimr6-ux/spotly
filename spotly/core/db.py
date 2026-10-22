"""Database access entry point.

    from spotly.core.db import get_db
    with get_db(db_path) as conn:          # WAL, busy_timeout=10 s, commit/rollback, always closed
        conn.execute(...)
"""
from spotly.database.schema import BUSY_TIMEOUT_MS, connect, get_db, init_db  # noqa: F401
