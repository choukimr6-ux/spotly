"""Phase 15: concurrent write queue - 10 channels x 100 airings, no lock errors, nothing lost, bounded memory."""
import sqlite3
import threading
import time
import tracemalloc

import pytest

from spotly.data import db_queue as dq
from spotly.data import reference_db as rdb
from spotly.database.schema import get_db

CHANNELS, PER_CHANNEL = 10, 100


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "c.db"
    with get_db(path) as conn:
        rdb.ensure_schema(conn)
        for i in range(CHANNELS):
            conn.execute("INSERT INTO reference_ads (brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm) "
                         "VALUES (?, 'c', 30, x'00', 'test')", (f"Brand{i}",))
            rdb.upsert_signature(conn, f"A{i}", f"Brand{i}", None, 30.0, rdb.pack_blob(
                __import__("numpy").zeros(0, "<u4"), __import__("numpy").zeros(0, "<u2"), 0.0, 0.0), None)
    yield path
    dq.get_queue(path).close()


def _fields(ch, n):
    start = 1000.0 + n * 60
    return dict(asset_id=f"A{ch}", ad_id=ch + 1, source_name=f"live:Ch{ch}", airing_start=start, airing_end=start + 30,
                duration=30.0, confidence=0.99, algeria_timestamp="2026-01-01T00:00:00+01:00",
                verification_hash=f"h{ch}-{n}", detected_at="2026-01-01T00:00:00+00:00")


def test_ten_channels_hundred_airings_each_no_lock_errors(db, capsys):
    errors, tickets = [], []
    barrier = threading.Barrier(CHANNELS)
    tlock = threading.Lock()

    def channel(ch):
        barrier.wait()
        for n in range(PER_CHANNEL):
            try:
                pen = None
                if n % 10 == 0:
                    pen = dict(key=f"pen-{ch}-{n}", kind="position_breach", severity="warning", message="m",
                               brand=f"Brand{ch}", channel=f"Ch{ch}", airing_start=1000.0 + n * 60, credit_dzd="100.00")
                t = rdb.submit_airing(db, **_fields(ch, n), penalty=pen)       # fire-and-forget producers
                with tlock:
                    tickets.append(t)
            except Exception as exc:                                            # pragma: no cover
                errors.append(exc)

    # a rogue writer outside the queue keeps grabbing the write lock, as another process would
    stop = threading.Event()
    rogue_hits = []

    def rogue():
        conn = sqlite3.connect(db, timeout=15, isolation_level=None)
        while not stop.is_set():
            conn.execute("BEGIN IMMEDIATE")
            time.sleep(0.05)
            conn.execute("COMMIT")
            rogue_hits.append(1)
            time.sleep(0.01)
        conn.close()

    tracemalloc.start()
    rt = threading.Thread(target=rogue)
    rt.start()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=channel, args=(c,)) for c in range(CHANNELS)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    q = dq.get_queue(db)
    assert q.flush(60)
    elapsed = time.perf_counter() - t0
    stop.set()
    rt.join()
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    results = [t.result() for t in tickets]                                     # raises on any "database is locked"
    assert not errors and len(results) == CHANNELS * PER_CHANNEL
    assert all(isinstance(r, int) for r in results)
    with get_db(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0] == 1000
        assert conn.execute("SELECT COUNT(DISTINCT occurrence_id) FROM ad_occurrences").fetchone()[0] == 1000
        assert conn.execute("SELECT SUM(match_count) FROM reference_signatures").fetchone()[0] == 1000
        assert all(r[0] == PER_CHANNEL for r in conn.execute("SELECT match_count FROM reference_signatures"))
        assert conn.execute("SELECT COUNT(*) FROM compliance_alerts").fetchone()[0] == CHANNELS * 10
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 15000
    assert q.stats["committed"] == 1000 and q.stats["failed"] == 0 and q.stats["deadlettered"] == 0
    assert q.stats["max_batch"] <= dq.MAX_BATCH
    assert peak_mb < 15, f"queue peak {peak_mb:.1f} MB"
    with capsys.disabled():
        print(f"\n[phase15 stress] events=1000 persisted=1000 lock_errors_seen_by_callers=0 "
              f"batches={q.stats['batches']} max_batch={q.stats['max_batch']} retries={q.stats['retries']} "
              f"peak_queue_depth={q.stats['peak_depth']} rogue_lock_holds={len(rogue_hits)} "
              f"traced_peak={peak_mb:.2f}MB elapsed={elapsed:.2f}s ({1000 / elapsed:.0f} ev/s)")


def test_synchronous_commit_airing_is_not_delayed_by_the_500ms_window(db):
    t0 = time.perf_counter()
    assert rdb.commit_airing(db, **_fields(0, 0)) is not None
    assert time.perf_counter() - t0 < 0.4


def test_batches_commit_at_50_items(db):
    q = dq.get_queue(db)
    tickets = [rdb.submit_airing(db, **_fields(n % CHANNELS, n)) for n in range(120)]
    assert q.flush(30)
    assert all(t.result() for t in tickets) and q.stats["max_batch"] <= 50 and q.stats["batches"] >= 3


def test_transient_lock_is_retried_with_backoff_and_loses_nothing(db, monkeypatch):
    calls = {"n": 0}

    def flaky(conn, **kw):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise sqlite3.OperationalError("database is locked")
        return rdb._apply_airing(conn, **kw)

    monkeypatch.setitem(dq.HANDLERS, "flaky", flaky)
    monkeypatch.setattr(dq, "BACKOFF_BASE_S", 0.001)
    q = dq.get_queue(db)
    t = q.submit("flaky", urgent=True, **_fields(1, 1))
    assert isinstance(t.result(10), int) and q.stats["retries"] == 3
    with get_db(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0] == 1


def test_bad_event_is_isolated_and_the_rest_of_the_batch_commits(db):
    q = dq.get_queue(db)
    good1 = rdb.submit_airing(db, **_fields(0, 1))
    bad = rdb.submit_airing(db, **dict(_fields(0, 2), ad_id=999))             # FK violation
    good2 = rdb.submit_airing(db, **_fields(0, 3))
    q.flush(10)
    assert good1.result() and good2.result()
    with pytest.raises(sqlite3.IntegrityError):
        bad.result()
    with get_db(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0] == 2
        assert conn.execute("SELECT match_count FROM reference_signatures WHERE asset_id='A0'").fetchone()[0] == 2


def test_exhausted_retries_go_to_dead_letter_not_lost(db, monkeypatch):
    def locked(conn, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setitem(dq.HANDLERS, "locked", locked)
    monkeypatch.setattr(dq, "BACKOFF_BASE_S", 0.0001)
    q = dq.DatabaseWriteQueue(db, max_retries=2)
    t = q.submit("locked", urgent=True, x=1)
    with pytest.raises(sqlite3.OperationalError):
        t.result(10)
    assert q.stats["deadlettered"] == 1
    dl = db.parent / (db.name + ".deadletter.jsonl")
    assert dl.exists() and '"x": 1' in dl.read_text(encoding="utf-8")
    q.close()


def test_idle_writer_exits_and_restarts_on_next_submit(db):
    q = dq.DatabaseWriteQueue(db, idle_exit_s=0.2)
    assert q.submit("signature_touch", urgent=True, asset_id="A0").result(5) is None
    time.sleep(0.8)
    assert q._thread is None
    q.submit("signature_touch", urgent=True, asset_id="A0").result(5)
    with get_db(db) as conn:
        assert conn.execute("SELECT match_count FROM reference_signatures WHERE asset_id='A0'").fetchone()[0] == 2
    q.close()


def test_phase30_ledger_stays_lock_free_under_10hz_telemetry_and_dashboard_polling(db, capsys):
    """10 channels x 100 airings through the queue WHILE 10 Hz heartbeat/status writes and the 8502 API polls run:
    zero 'database is locked', every airing persisted with a distinct id, and no transient table inside the ledger."""
    import json
    import os
    from fastapi.testclient import TestClient
    from spotly.app.api_server import create_app
    from spotly.database import telemetry_store

    errors, tickets, tlock = [], [], threading.Lock()
    stop = threading.Event()
    counts = dict(telemetry=0, polls=0, ro_polls=0)

    def telemetry():                                   # the supervisor: 10 Hz, heartbeat + one status row per channel
        conn = telemetry_store.connect(db)
        try:
            while not stop.is_set():
                try:
                    with conn:
                        conn.execute("INSERT INTO service_heartbeat (service_id, pid, state, heartbeat_epoch, channels) "
                                     "VALUES (1, ?, 'running', ?, ?) ON CONFLICT(service_id) DO UPDATE SET "
                                     "heartbeat_epoch=excluded.heartbeat_epoch", (os.getpid(), time.time(), CHANNELS))
                        conn.executemany(
                            "INSERT INTO service_channel_status (channel, state, updated_epoch, restarts, status_json, owner_pid) "
                            "VALUES (?, 'listening', ?, 0, ?, ?) ON CONFLICT(channel) DO UPDATE SET "
                            "updated_epoch=excluded.updated_epoch, status_json=excluded.status_json",
                            [(f"Ch{c}", time.time(), json.dumps(dict(level_dbfs=-20.0, level_history=[-20.0] * 150)), os.getpid())
                             for c in range(CHANNELS)])
                    counts["telemetry"] += 1
                except Exception as exc:
                    errors.append(("telemetry", exc))
                time.sleep(0.1)
        finally:
            conn.close()

    def poller():                                      # the dashboard: telemetry + historical reads (mode=ro)
        with TestClient(create_app(db)) as client:
            while not stop.is_set():
                try:
                    r = client.get("/api/telemetry")
                    assert r.status_code == 200, r.text
                    counts["polls"] += 1
                    for url in ("/api/airings?limit=50&verify=false", "/api/stats"):
                        r = client.get(url)
                        assert r.status_code == 200, (url, r.text[:200])
                        counts["ro_polls"] += 1
                except Exception as exc:
                    errors.append(("poll", exc))
                time.sleep(0.05)

    def channel(ch, barrier):
        barrier.wait()
        for n in range(PER_CHANNEL):
            try:
                t = rdb.submit_airing(db, **_fields(ch, n))
                with tlock:
                    tickets.append(t)
            except Exception as exc:
                errors.append(("producer", exc))

    barrier = threading.Barrier(CHANNELS)
    bg = [threading.Thread(target=telemetry), threading.Thread(target=poller)]
    [t.start() for t in bg]
    producers = [threading.Thread(target=channel, args=(c, barrier)) for c in range(CHANNELS)]
    t0 = time.perf_counter()
    [t.start() for t in producers]
    [t.join() for t in producers]
    assert dq.get_queue(db).flush(90)
    elapsed = time.perf_counter() - t0
    time.sleep(max(0.5, 4.0 - elapsed))                 # keep the background load running for a fixed minimum window
    stop.set()
    [t.join() for t in bg]

    ids = [t.result() for t in tickets]
    assert not errors, errors[:3]
    assert not [e for e in errors if "locked" in str(e)]
    assert len(ids) == CHANNELS * PER_CHANNEL and len(set(ids)) == len(ids) and all(isinstance(i, int) for i in ids)
    with get_db(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0] == CHANNELS * PER_CHANNEL
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert not tables & {"service_heartbeat", "service_channel_status"}          # ledger purity
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 15000
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1                 # NORMAL
    assert counts["telemetry"] >= 20 and counts["polls"] >= 3 and counts["ro_polls"] >= 6
    with capsys.disabled():
        print(f"\n[PHASE30] {len(ids)} airings / {elapsed:.1f} s, {counts['telemetry']} telemetry ticks (10 Hz target), "
              f"{counts['polls']} /api/telemetry + {counts['ro_polls']} read-only polls, 0 lock errors")


def test_readonly_dashboard_connections_cannot_write_the_ledger(db):
    from spotly.database.schema import read_only
    with read_only():
        with get_db(db) as conn:
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("DELETE FROM ad_occurrences")
    with get_db(db) as conn:                                                       # outside the block: normal read-write again
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 0
