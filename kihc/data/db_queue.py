"""Phase 15: single-writer, batched, retrying write queue in front of SQLite.

    q = get_queue(db_path)
    ticket = q.submit("airing", urgent=True, **fields)      # returns immediately
    occurrence_id = ticket.result()                         # optional: wait until the batch is committed

Capture workers never touch the write lock: they enqueue plain-dict events. ONE writer thread drains the queue and
applies each batch inside a single BEGIN IMMEDIATE transaction, committing at 50 events or 500 ms after the first
event, whichever comes first. An `urgent` event (a caller that waits for its result) cuts the window short, but
whatever is already queued still rides in the same commit, so contention turns into bigger batches, not more locks.

Guarantees
  * zero lost events: the queue is bounded only by backpressure (put blocks, never drops); a transient lock
    ("database is locked/busy") re-runs the whole batch with exponential full-jitter back-off; a batch that still
    fails after the retries is appended to `<db>.deadletter.jsonl` and every ticket gets the error;
  * one bad event never poisons its batch: each event runs in its own SAVEPOINT and only it is rolled back;
  * tickets resolve only AFTER the COMMIT, so a waiter never sees a result that is not durable;
  * a thread (not a process): sqlite3 releases the GIL, a second process would cost 20-30 MB of interpreter
    for no gain at this write rate, and the whole queue stays well under the 15 MB budget (see memory_bytes()).
The writer exits after `idle_exit_s` without work and is restarted by the next submit().
"""
import atexit
import json
import logging
import os
import queue
import random
import sqlite3
import threading
import time
from pathlib import Path

from kihc.database.schema import init_db

log = logging.getLogger(__name__)

MAX_BATCH = 50
FLUSH_S = 0.5
MAX_RETRIES = 8
BACKOFF_BASE_S = 0.05
BACKOFF_CAP_S = 2.0

HANDLERS: dict = {}


def register(kind: str, fn) -> None:
    """fn(conn, **payload) -> result; runs inside the writer's transaction (inside a SAVEPOINT)."""
    HANDLERS[kind] = fn


class Ticket:
    __slots__ = ("_done", "value", "error")

    def __init__(self):
        self._done = threading.Event()
        self.value = None
        self.error = None

    def done(self) -> bool:
        return self._done.is_set()

    def result(self, timeout: float | None = 60.0):
        if not self._done.wait(timeout):
            raise TimeoutError("database write not committed in time")
        if self.error is not None:
            raise self.error
        return self.value


class _Event:
    __slots__ = ("kind", "payload", "ticket", "urgent")

    def __init__(self, kind, payload, urgent):
        self.kind, self.payload, self.urgent, self.ticket = kind, payload, urgent, Ticket()


def _transient(exc: Exception) -> bool:
    m = str(exc).lower()
    return isinstance(exc, sqlite3.OperationalError) and ("locked" in m or "busy" in m)


class DatabaseWriteQueue:
    def __init__(self, db_path, *, max_batch: int = MAX_BATCH, flush_s: float = FLUSH_S, max_retries: int = MAX_RETRIES,
                 idle_exit_s: float = 30.0, maxsize: int = 50_000):
        self.db_path = Path(db_path) if db_path is not None else None
        self.max_batch, self.flush_s, self.max_retries, self.idle_exit_s = max_batch, flush_s, max_retries, idle_exit_s
        self._q: queue.Queue = queue.Queue(maxsize)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._inflight = 0                                   # dequeued but not yet resolved
        self._idle = threading.Condition()
        self._stopping = False
        self.stats = dict(submitted=0, committed=0, failed=0, batches=0, retries=0, lock_errors=0,
                          max_batch=0, peak_depth=0, deadlettered=0)

    # ---- producer side
    def submit(self, kind: str, *, urgent: bool = False, **payload) -> Ticket:
        if kind not in HANDLERS:
            raise KeyError(f"no handler registered for {kind!r}")
        ev = _Event(kind, payload, urgent)
        with self._idle:
            self._inflight += 1
        self._q.put(ev)                                      # blocks (backpressure) instead of dropping
        self.stats["submitted"] += 1
        self.stats["peak_depth"] = max(self.stats["peak_depth"], self._q.qsize())
        self._ensure_thread()
        return ev.ticket

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stopping = False
                self._thread = threading.Thread(target=self._run, name="kihc-db-writer", daemon=True)
                self._thread.start()

    def flush(self, timeout: float = 60.0) -> bool:
        """Block until every submitted event has been committed (or dead-lettered)."""
        end = time.monotonic() + timeout
        with self._idle:
            while self._inflight:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._idle.wait(left)
        return True

    def close(self, timeout: float = 60.0) -> bool:
        ok = self.flush(timeout)
        self._stopping = True
        t = self._thread
        if t is not None:
            t.join(2.0)
        return ok

    def memory_bytes(self) -> int:
        """Rough footprint of what is queued right now (payload reprs)."""
        with self._q.mutex:
            return sum(len(repr(e.payload)) + 200 for e in self._q.queue)

    # ---- writer side
    def _collect(self, first: _Event) -> list:
        batch = [first]
        deadline = time.monotonic() + self.flush_s
        urgent = first.urgent
        while len(batch) < self.max_batch:
            try:
                if urgent:
                    ev = self._q.get_nowait()                # a waiter is blocked: take what is there and go
                else:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    ev = self._q.get(timeout=left)
            except queue.Empty:
                break
            batch.append(ev)
            urgent = urgent or ev.urgent
        return batch

    def _run(self) -> None:
        conn = None
        try:
            while True:
                try:
                    first = self._q.get(timeout=min(self.idle_exit_s, 0.5) if self._stopping else self.idle_exit_s)
                except queue.Empty:
                    with self._lock:
                        if self._q.empty():
                            self._thread = None
                            return
                    continue
                batch = self._collect(first)
                conn = self._write(batch, conn)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass

    def _open(self):
        conn = init_db(self.db_path)
        conn.isolation_level = None                          # we issue BEGIN/COMMIT/SAVEPOINT ourselves
        return conn

    def _write(self, batch: list, conn):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                if conn is None:
                    conn = self._open()
                conn.execute("BEGIN IMMEDIATE")
                outcomes = []
                for ev in batch:
                    conn.execute("SAVEPOINT ev")
                    try:
                        outcomes.append((HANDLERS[ev.kind](conn, **ev.payload), None))
                        conn.execute("RELEASE ev")
                    except Exception as exc:
                        if _transient(exc):
                            raise
                        conn.execute("ROLLBACK TO ev")       # only this event is undone
                        conn.execute("RELEASE ev")
                        outcomes.append((None, exc))
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                last_exc = exc
                try:
                    if conn is not None and conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    conn = None
                if not _transient(exc) or attempt == self.max_retries:
                    break
                self.stats["retries"] += 1
                self.stats["lock_errors"] += 1
                time.sleep(random.uniform(0, min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** attempt)))   # full jitter
                continue
            else:
                self.stats["batches"] += 1
                self.stats["max_batch"] = max(self.stats["max_batch"], len(batch))
                for ev, (val, err) in zip(batch, outcomes):
                    ev.ticket.value, ev.ticket.error = val, err
                    self.stats["failed" if err else "committed"] += 1
                self._resolve(batch)
                return conn
        self._deadletter(batch, last_exc)
        return None if conn is None else conn

    def _deadletter(self, batch: list, exc) -> None:
        log.error("db writer gave up on %d event(s): %s", len(batch), exc)
        try:
            path = Path(str(self.db_path) + ".deadletter.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                for ev in batch:
                    f.write(json.dumps(dict(kind=ev.kind, payload=ev.payload, error=str(exc)), default=str) + "\n")
        except OSError as e:
            log.error("dead-letter write failed too: %s", e)
        self.stats["deadlettered"] += len(batch)
        for ev in batch:
            ev.ticket.error = exc if isinstance(exc, Exception) else RuntimeError("write failed")
        self._resolve(batch)

    def _resolve(self, batch: list) -> None:
        for ev in batch:
            ev.ticket._done.set()
        with self._idle:
            self._inflight -= len(batch)
            self._idle.notify_all()


_QUEUES: dict[str, DatabaseWriteQueue] = {}
_QLOCK = threading.Lock()


def get_queue(db_path, **kw) -> DatabaseWriteQueue:
    """The shared queue for a database file (one writer per file per process)."""
    key = os.path.realpath(db_path) if db_path is not None else ""
    with _QLOCK:
        q = _QUEUES.get(key)
        if q is None:
            from kihc.database.schema import DEFAULT_DB_PATH
            q = _QUEUES[key] = DatabaseWriteQueue(db_path or DEFAULT_DB_PATH, **kw)
        return q


def shutdown_all(timeout: float = 30.0) -> None:
    with _QLOCK:
        queues = list(_QUEUES.values())
    for q in queues:
        q.close(timeout)


atexit.register(shutdown_all)
