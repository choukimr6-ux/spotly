"""Single-instance OS file lock.

The OS releases the lock when the holding process exits for any reason (crash,
kill, power loss), so a dead service can never leave a stale "running" flag.
"""
import os
import sys
from pathlib import Path


class ProcessLock:
    def __init__(self, path):
        self.path = Path(path)
        self._fh = None

    def acquire(self) -> bool:
        """Non-blocking. True if this process now holds the lock."""
        if self._fh:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self):
        if not self._fh:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._fh.close()
            self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None


def is_locked(path) -> bool:
    """True if another holder (any process, or another handle in this one) has the lock."""
    if not Path(path).exists():
        return False
    probe = ProcessLock(path)
    if probe.acquire():
        probe.release()
        return False
    return True


def service_paths(db_path) -> dict:
    """Lock / stop-request / log locations derived from the database path (one service per database)."""
    db = Path(db_path).resolve()
    return dict(lock=db.with_suffix(".service.lock"), stop=db.with_suffix(".service.stop"),
                log=db.parent / "logs" / f"{db.stem}_service.log")


def request_stop(db_path) -> Path:
    p = service_paths(db_path)["stop"]
    p.write_text(f"stop requested by pid {os.getpid()}\n", encoding="utf-8")
    return p
