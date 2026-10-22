"""Phase 4.1: competitive tags, Share of Voice, placement, dayparts, surges (fictional brands)."""
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from spotly.core.analytics import (channel_of, channel_placement,
                                 compute_share_of_voice, daypart_distribution,
                                 detect_competitor_surges)
from spotly.core.library import ingest_media, set_ad_tags, validate_tags
from spotly.core.matcher import scan_source
from spotly.core.proof import ALGERIA_TZ, verify_row
from spotly.database.schema import init_db
from spotly.tests.generate_test_assets import generate

ROOT = Path(__file__).resolve().parents[2]
DAY = datetime(2026, 10, 10, tzinfo=ALGERIA_TZ)


def _at(day_offset, hour, minute=0):
    return (DAY + timedelta(days=day_offset, hours=hour, minutes=minute)).timestamp()


def _ad(conn, brand, campaign, industry, ad_type="self", target=None):
    conn.execute("INSERT INTO reference_ads (brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm, "
                 "industry, ad_type, target_competitor_of) VALUES (?, ?, 30, x'00', 'test', ?, ?, ?)",
                 (brand, campaign, industry, ad_type, target))
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _air(conn, ad_id, channel, start, seconds, review=0, live=True):
    source = f"live:{channel}" if live else channel
    conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                 "confidence_score, human_verified) VALUES (?, ?, ?, ?, ?, 0.93, ?)",
                 (ad_id, source, start, start + seconds, seconds, review))


@pytest.fixture
def market(tmp_path):
    """Telecom: Atlas Mobile (client) 4x30s, Sahara Telecom 3x20s, Oasis Net 1x20s; Dairy: Dune Dairy."""
    db = tmp_path / "market.db"
    conn = init_db(db)
    with conn:
        atlas = _ad(conn, "Atlas Mobile", "4G Ramadan", "Telecom")
        sahara = _ad(conn, "Sahara Telecom", "Fiber Promo", "Telecom", "competitor", "Atlas Mobile")
        oasis = _ad(conn, "Oasis Net", "Roaming", "Telecom", "competitor", "Atlas Mobile")
        dune = _ad(conn, "Dune Dairy", "Yogurt", "Dairy")
        for h in (8, 13, 20):
            _air(conn, atlas, "Echorouk TV", _at(0, h), 30)
        _air(conn, atlas, "Ennahar TV", _at(1, 20), 30)
        _air(conn, sahara, "Echorouk TV", _at(0, 20, 5), 20)
        _air(conn, sahara, "Echorouk TV", _at(1, 20, 5), 20)
        _air(conn, sahara, "Ennahar TV", _at(1, 21), 20)
        _air(conn, sahara, "Ennahar TV", _at(1, 22), 20, review=-1)          # rejected: must not count
        _air(conn, oasis, "Ennahar TV", _at(1, 9), 20)
        _air(conn, dune, "Echorouk TV", _at(0, 12), 15)
    conn.close()
    return dict(db=db, atlas=atlas, sahara=sahara, oasis=oasis, dune=dune)


def _by_brand(sov):
    return {b["brand"]: b for b in sov["brands"]}


# ------------------------------------------------------------------ schema migration

def test_migration_adds_tag_columns_to_existing_reference_ads(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as c:   # pre-4.1 reference_ads, with data and a hashed occurrence
        c.executescript("""
            CREATE TABLE reference_ads (ad_id INTEGER PRIMARY KEY AUTOINCREMENT, brand_name TEXT NOT NULL,
              campaign_name TEXT NOT NULL, duration_sec REAL NOT NULL, fingerprint BLOB NOT NULL,
              fp_algorithm TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT 'x', UNIQUE (brand_name, campaign_name));
            INSERT INTO reference_ads (brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm)
              VALUES ('Legacy Brand', 'Old Spot', 20, x'0102', 'kihc-subband-v2-masked');
        """)
    conn = init_db(db)
    cols = {r[1]: r for r in conn.execute("PRAGMA table_info(reference_ads)")}
    row = conn.execute("SELECT * FROM reference_ads").fetchone()
    conn.close()
    assert {"industry", "ad_type", "target_competitor_of"} <= set(cols)
    assert row["brand_name"] == "Legacy Brand" and row["fingerprint"] == b"\x01\x02"     # data intact
    assert row["ad_type"] == "self" and row["industry"] is None and row["target_competitor_of"] is None
    init_db(db).close()                                                                  # idempotent


def test_tagging_does_not_touch_occurrence_hashes(tmp_path):
    ad_a, _, broadcast = generate()
    db = tmp_path / "h.db"
    ingest_media(ad_a, "Atlas Mobile", "Spot A", db_path=db)
    scan_source(broadcast, db_path=db, evidence_dir=tmp_path)
    set_ad_tags(1, "Telecom", "competitor", "Someone Else", db_path=db)
    conn = init_db(db)
    row = conn.execute("SELECT * FROM ad_occurrences").fetchone()
    conn.close()
    assert verify_row(row) is True


# ------------------------------------------------------------------ tagging

def test_tag_validation():
    assert validate_tags("  Telecom  ", "Competitor", " Atlas ") == ("Telecom", "competitor", "Atlas")
    with pytest.raises(ValueError):
        validate_tags("Telecom", "partner")
    with pytest.raises(ValueError):
        validate_tags("Telecom", "self", "Atlas")                 # only competitors compete with someone
    with pytest.raises(KeyError):
        set_ad_tags(999, "Telecom", db_path=":memory:")


def test_ingest_with_tags_and_reingest_keeps_them(tmp_path):
    ad_a, _, _ = generate()
    db = tmp_path / "t.db"
    res = ingest_media(ad_a, "Sahara Telecom", "Fiber", db_path=db, industry="Telecom", ad_type="competitor",
                       target_competitor_of="Atlas Mobile")
    assert res.committed
    ingest_media(ad_a, "Sahara Telecom", "Fiber", db_path=db)    # re-index without tags
    conn = init_db(db)
    r = conn.execute("SELECT industry, ad_type, target_competitor_of FROM reference_ads").fetchone()
    conn.close()
    assert tuple(r) == ("Telecom", "competitor", "Atlas Mobile")


# ------------------------------------------------------------------ share of voice

def test_sov_airtime_and_airings_shares(market):
    sov = compute_share_of_voice("Telecom", db_path=market["db"])
    b = _by_brand(sov)
    assert sov["total_airings"] == 8 and sov["total_airtime_sec"] == pytest.approx(200.0)
    assert [x["brand"] for x in sov["brands"]] == ["Atlas Mobile", "Sahara Telecom", "Oasis Net"]
    assert (b["Atlas Mobile"]["airings"], b["Atlas Mobile"]["airtime_sec"]) == (4, 120.0)
    assert (b["Sahara Telecom"]["airings"], b["Sahara Telecom"]["airtime_sec"]) == (3, 60.0)  # rejected excluded
    assert b["Atlas Mobile"]["sov_airtime_pct"] == pytest.approx(60.0)
    assert b["Sahara Telecom"]["sov_airtime_pct"] == pytest.approx(30.0)
    assert b["Oasis Net"]["sov_airtime_pct"] == pytest.approx(10.0)
    assert b["Atlas Mobile"]["sov_airings_pct"] == pytest.approx(50.0)
    assert b["Sahara Telecom"]["sov_airings_pct"] == pytest.approx(37.5)
    assert b["Oasis Net"]["sov_airings_pct"] == pytest.approx(12.5)
    assert sum(x["sov_airtime_pct"] for x in sov["brands"]) == pytest.approx(100.0)
    assert sov["self_sov_airtime_pct"] == pytest.approx(60.0)
    assert sov["competitor_sov_airtime_pct"] == pytest.approx(40.0)
    assert b["Atlas Mobile"]["role"] == "self" and b["Oasis Net"]["role"] == "competitor"
    assert b["Atlas Mobile"]["channels"] == ["Echorouk TV", "Ennahar TV"]


def test_industries_are_isolated_and_case_insensitive(market):
    dairy = compute_share_of_voice("dairy", db_path=market["db"])
    assert [x["brand"] for x in dairy["brands"]] == ["Dune Dairy"] and dairy["brands"][0]["sov_airtime_pct"] == 100
    assert compute_share_of_voice("TELECOM", db_path=market["db"])["total_airings"] == 8
    assert compute_share_of_voice(None, db_path=market["db"])["total_airings"] == 9    # all industries


def test_channel_filter(market):
    sov = compute_share_of_voice("Telecom", channel="echorouk tv", db_path=market["db"])
    b = _by_brand(sov)
    assert sov["total_airtime_sec"] == pytest.approx(130.0)                  # Atlas 3x30 + Sahara 2x20
    assert b["Atlas Mobile"]["sov_airtime_pct"] == pytest.approx(90 / 130 * 100)
    assert "Oasis Net" not in b


def test_date_range_and_undated_file_scans(market):
    conn = init_db(market["db"])
    with conn:
        _air(conn, market["atlas"], "Echorouk TV · rec.mp4", 12.0, 30, live=False)   # recording offset, no date
    conn.close()
    assert compute_share_of_voice("Telecom", db_path=market["db"])["total_airings"] == 9   # no filter: included
    day1 = compute_share_of_voice("Telecom", date_from=date(2026, 10, 11), date_to=date(2026, 10, 11),
                                  db_path=market["db"])
    b = _by_brand(day1)
    assert day1["excluded_undated"] == 1
    assert (b["Atlas Mobile"]["airings"], b["Sahara Telecom"]["airings"], b["Oasis Net"]["airings"]) == (1, 2, 1)
    assert day1["total_airtime_sec"] == pytest.approx(30 + 40 + 20)
    day0 = compute_share_of_voice("Telecom", date_from="2026-10-10", date_to="2026-10-10", db_path=market["db"])
    assert day0["total_airings"] == 4                                        # Atlas x3 + Sahara x1


def test_verified_only_and_client_scope(market):
    conn = init_db(market["db"])
    with conn:
        conn.execute("UPDATE ad_occurrences SET human_verified = 1 WHERE ad_id = ?", (market["oasis"],))
        other = _ad(conn, "Zenith Phones", "Launch", "Telecom", "competitor", "Some Other Client")
        _air(conn, other, "Echorouk TV", _at(0, 21), 40)
    conn.close()
    verified = compute_share_of_voice("Telecom", db_path=market["db"], verified_only=True)
    assert [x["brand"] for x in verified["brands"]] == ["Oasis Net"]
    scoped = compute_share_of_voice("Telecom", db_path=market["db"], client_brand="atlas mobile")
    assert "Zenith Phones" not in _by_brand(scoped) and scoped["total_airings"] == 8
    assert "Zenith Phones" in _by_brand(compute_share_of_voice("Telecom", db_path=market["db"]))


def test_empty_selection(market):
    sov = compute_share_of_voice("Automotive", db_path=market["db"])
    assert sov["brands"] == [] and sov["total_airings"] == 0 and sov["self_sov_airtime_pct"] == 0


# ------------------------------------------------------------------ placement & dayparts

def test_channel_placement(market):
    rows = channel_placement("Telecom", db_path=market["db"])
    atlas = {r["channel"]: r for r in rows if r["brand"] == "Atlas Mobile"}
    assert atlas["Echorouk TV"]["share_of_brand_pct"] == pytest.approx(75.0)
    assert atlas["Ennahar TV"]["share_of_brand_pct"] == pytest.approx(25.0)
    sahara = {r["channel"]: r for r in rows if r["brand"] == "Sahara Telecom"}
    assert sahara["Echorouk TV"]["airtime_sec"] == pytest.approx(40) and sahara["Ennahar TV"]["airings"] == 1
    for brand in {r["brand"] for r in rows}:
        assert sum(r["share_of_brand_pct"] for r in rows if r["brand"] == brand) == pytest.approx(100)


def test_daypart_hours_are_algeria_time(market):
    conn = init_db(market["db"])
    with conn:
        _air(conn, market["atlas"], "Echorouk TV · rec.mp4", 50.0, 30, live=False)    # no clock: not counted
    conn.close()
    rows = daypart_distribution("Telecom", db_path=market["db"])
    atlas = {r["hour"]: r["airings"] for r in rows if r["brand"] == "Atlas Mobile"}
    assert atlas == {8: 1, 13: 1, 20: 2}
    assert {r["hour"] for r in rows if r["brand"] == "Sahara Telecom"} == {20, 21}


def test_channel_of():
    assert channel_of("live:Echorouk TV") == "Echorouk TV"
    assert channel_of("Ennahar TV 20:00 · rec.mp4") == "Ennahar TV 20:00"
    assert channel_of("broadcast_60s.mp4") == "broadcast_60s.mp4"


# ------------------------------------------------------------------ surges

@pytest.fixture
def surge_db(tmp_path):
    db = tmp_path / "surge.db"
    now = _at(2, 12)
    conn = init_db(db)

    def series(brand, ad_type, prev, cur, target="Atlas Mobile"):
        ad = _ad(conn, brand, "Spot", "Telecom", ad_type, target if ad_type == "competitor" else None)
        for i in range(prev):
            _air(conn, ad, "Echorouk TV", now - 36 * 3600 + i * 60, 20)       # previous 24 h window
        for i in range(cur):
            _air(conn, ad, "Echorouk TV", now - 12 * 3600 + i * 60, 20)       # current 24 h window
    with conn:
        series("Doubling Co", "competitor", 2, 4)       # +100%   -> surge
        series("Steady Co", "competitor", 3, 3)         # 0%      -> no
        series("Tiny Co", "competitor", 1, 2)           # +100% but only 2 airings -> below min volume
        series("Newcomer Co", "competitor", 0, 3)       # new activity
        series("Edge30 Co", "competitor", 10, 13)       # exactly +30% -> not "> 30%"
        series("Edge40 Co", "competitor", 10, 14)       # +40%    -> surge
        series("Atlas Mobile", "self", 1, 9)            # client's own ads never flagged
    conn.close()
    return db, now


def test_detect_competitor_surges(surge_db):
    db, now = surge_db
    found = {s["brand"]: s for s in detect_competitor_surges(24, now=now, db_path=db)}
    assert set(found) == {"Doubling Co", "Newcomer Co", "Edge40 Co"}
    assert found["Doubling Co"]["change_pct"] == pytest.approx(100.0) and found["Doubling Co"]["kind"] == "surge"
    assert found["Newcomer Co"]["change_pct"] is None and found["Newcomer Co"]["kind"] == "new activity"
    assert found["Edge40 Co"]["change_pct"] == pytest.approx(40.0)
    assert found["Doubling Co"]["current_airtime_sec"] == pytest.approx(80.0)
    assert found["Doubling Co"]["target_competitor_of"] == "Atlas Mobile"


def test_surge_window_is_configurable(surge_db):
    db, now = surge_db
    # 6 h window: 'current' airings (12 h ago) fall outside both windows -> nothing flagged
    assert detect_competitor_surges(6, now=now, db_path=db) == []


# ------------------------------------------------------------------ CLI

def test_cli_sov_and_tag(market):
    run = lambda *a: subprocess.run([sys.executable, "-m", "spotly.cli", "--db", str(market["db"]), *a],
                                    capture_output=True, text=True, cwd=ROOT)
    out = run("sov", "--industry", "Telecom")
    assert out.returncode == 0, out.stderr
    assert "Atlas Mobile" in out.stdout and "60.00% airtime" in out.stdout and "37.50% airings" in out.stdout
    tag = run("tag-ad", str(market["dune"]), "--industry", "Telecom", "--role", "competitor",
              "--competes-with", "Atlas Mobile")
    assert tag.returncode == 0, tag.stderr
    assert "Dune Dairy" in _by_brand(compute_share_of_voice("Telecom", db_path=market["db"]))
