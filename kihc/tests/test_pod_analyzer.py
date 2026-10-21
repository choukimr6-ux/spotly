"""Phase 4.2: commercial pod reconstruction, in-break positions, clutter (synthetic occurrences)."""
import re
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from kihc.core.ingest import ingest_reference
from kihc.core.matcher import scan_source
from kihc.core.pod_analyzer import (Pod, calculate_ad_positions,
                                    channel_clutter, clutter_score,
                                    cluster_occurrences_into_pods,
                                    materialize_pods, position_distribution)
from kihc.core.proof import ALGERIA_TZ, verify_row
from kihc.database.schema import init_db
from kihc.tests.generate_test_assets import generate

ROOT = Path(__file__).resolve().parents[2]
DAY = datetime(2026, 10, 10, tzinfo=ALGERIA_TZ)


def T(hour, minute=0, second=0.0, day=0):
    return (DAY + timedelta(days=day, hours=hour, minutes=minute, seconds=second)).timestamp()


class Market:
    def __init__(self, db):
        self.db, self.conn, self.ads = db, init_db(db), {}

    def ad(self, brand, ad_type="self"):
        if brand not in self.ads:
            self.conn.execute("INSERT INTO reference_ads (brand_name, campaign_name, duration_sec, fingerprint, "
                              "fp_algorithm, ad_type) VALUES (?, 'Spot', 30, x'00', 'test', ?)", (brand, ad_type))
            self.ads[brand] = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return self.ads[brand]

    def air(self, brand, start, seconds, channel="Echorouk TV", live=True, review=0, ad_type="self"):
        src = f"live:{channel}" if live else channel
        self.conn.execute("INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, "
                          "duration_detected, confidence_score, human_verified) VALUES (?, ?, ?, ?, ?, 0.9, ?)",
                          (self.ad(brand, ad_type), src, start, start + seconds, seconds, review))
        return start + seconds                                   # end, for chaining

    def chain(self, brands, start, seconds=30, gap=2.0, **kw):
        t = start
        for b in brands:
            t = self.air(b, t, seconds, **kw) + gap
        return t

    def commit(self):
        self.conn.commit()


@pytest.fixture
def m(tmp_path):
    mk = Market(tmp_path / "pods.db")
    yield mk
    mk.conn.close()


def pods(m, *a, **k):
    m.commit()
    return cluster_occurrences_into_pods(*a, db_path=m.db, **k)


# ------------------------------------------------------------------ clustering

def test_two_second_gaps_form_one_pod_and_ten_minutes_split(m):
    m.chain(["A", "B", "C", "D"], T(20), seconds=30, gap=2.0)
    m.air("E", T(20, 12), 30)                                    # ~10 min after the break ends
    ps = pods(m, "Echorouk TV", date(2026, 10, 10))
    assert [p.total_ads_in_pod for p in ps] == [4, 1]
    first = ps[0]
    assert first.pod_start == pytest.approx(T(20)) and first.pod_end == pytest.approx(T(20) + 4 * 30 + 3 * 2)
    assert first.pod_duration_seconds == pytest.approx(126.0)
    assert [a["brand"] for a in first.ads] == ["A", "B", "C", "D"]


def test_positions_first_middle_last_solo(m):
    m.chain(["A", "B", "C", "D"], T(20))
    m.chain(["P", "Q"], T(21))
    m.air("S", T(22), 30)
    ps = pods(m, "Echorouk TV", "2026-10-10")
    assert [a["position"] for a in ps[0].ads] == ["FIRST", "MIDDLE", "MIDDLE", "LAST"]
    assert [a["position_of"] for a in ps[0].ads] == ["1 of 4", "2 of 4", "3 of 4", "4 of 4"]
    assert [a["position_label"] for a in ps[1].ads] == ["First-in-Break", "Last-in-Break"]   # N=2: no middle
    assert ps[2].ads[0]["position"] == "SOLO" and ps[2].ads[0]["position_label"] == "Solo"
    assert ps[0].ads[2]["offset_in_pod"] == pytest.approx(64.0)


def test_gap_boundary_is_inclusive(m):
    m.air("A", T(20), 30)
    m.air("B", T(20) + 30 + 12.0, 30)                            # gap exactly 12 s -> same pod
    m.air("C", T(21), 30)
    m.air("D", T(21) + 30 + 12.5, 30)                            # 12.5 s -> new pod
    ps = pods(m, "Echorouk TV", "2026-10-10")
    assert [p.total_ads_in_pod for p in ps] == [2, 1, 1]
    assert [p.total_ads_in_pod for p in pods(m, "Echorouk TV", "2026-10-10", max_gap_seconds=13)] == [2, 2]


def test_back_to_back_same_ad(m):
    m.chain(["A", "A", "B"], T(20), gap=0.0)
    (p,) = pods(m, "Echorouk TV", "2026-10-10")
    assert [(a["brand"], a["position"]) for a in p.ads] == [("A", "FIRST"), ("A", "MIDDLE"), ("B", "LAST")]
    assert len({a["occurrence_id"] for a in p.ads}) == 3 and not p.has_overlap


def test_overlapping_timestamps(m):
    m.air("Long", T(20), 60)
    m.air("Short", T(20, 0, 10), 15)                             # starts and ends inside 'Long'
    m.air("After", T(20, 1, 5), 20)                              # 5 s after Long ends
    (p,) = pods(m, "Echorouk TV", "2026-10-10")
    assert p.has_overlap and p.total_ads_in_pod == 3
    assert [a["brand"] for a in p.ads] == ["Long", "Short", "After"]
    assert p.pod_end == pytest.approx(T(20, 1, 25))              # max end, not the last-started ad's end


def test_empty_periods(m):
    m.air("A", T(20), 30)
    assert pods(m, "Echorouk TV", "2026-10-11") == []            # nothing that day
    assert pods(m, "Ennahar TV", "2026-10-10") == []             # nothing on that channel
    assert pods(m, "Echorouk TV", (T(1), T(2))) == []           # empty window
    assert cluster_occurrences_into_pods(db_path=m.db.parent / "fresh.db") == []   # empty database


def test_channels_never_merge_and_rejected_ignored(m):
    m.air("A", T(20), 30, channel="Echorouk TV")
    m.air("B", T(20, 0, 31), 30, channel="Ennahar TV")           # 1 s later but another channel
    m.air("C", T(20, 0, 32), 30, channel="Echorouk TV", review=-1)   # rejected: must not join
    ps = pods(m, None, "2026-10-10")
    assert sorted((p.channel, p.total_ads_in_pod) for p in ps) == [("Echorouk TV", 1), ("Ennahar TV", 1)]


def test_recordings_cluster_per_file(m):
    m.air("A", 100.0, 30, channel="Echorouk TV · rec1.mp4", live=False)
    m.air("B", 132.0, 30, channel="Echorouk TV · rec1.mp4", live=False)
    m.air("C", 133.0, 30, channel="Echorouk TV · rec2.mp4", live=False)   # same offsets, other file
    ps = pods(m)                                                           # no date: recordings included
    assert sorted(p.total_ads_in_pod for p in ps) == [1, 2]
    assert all(re.fullmatch(r"POD-REC-[A-Z0-9_]+-\d{3}", p.pod_id) and not p.is_live for p in ps)
    assert pods(m, None, "2026-10-10") == []                     # recordings have no broadcast date


# ------------------------------------------------------------------ ids, midnight, stability

def test_pod_ids_are_stable_and_dated(m):
    m.chain(["A", "B"], T(9))
    m.chain(["C", "D"], T(20))
    full = pods(m, "Echorouk TV", "2026-10-10")
    assert [p.pod_id for p in full] == ["POD-20261010-ECHOROUK_TV-001", "POD-20261010-ECHOROUK_TV-002"]
    evening = pods(m, "Echorouk TV", (T(19), T(23)))             # narrower window: same id for that pod
    assert [p.pod_id for p in evening] == ["POD-20261010-ECHOROUK_TV-002"]


def test_pod_straddling_midnight_belongs_to_start_day_whole(m):
    m.chain(["A", "B", "C"], T(23, 59, 0), seconds=30, gap=2.0)  # 23:59:00 -> 00:00:34
    day0 = pods(m, "Echorouk TV", "2026-10-10")
    assert len(day0) == 1 and day0[0].total_ads_in_pod == 3 and day0[0].pod_date == "2026-10-10"
    assert pods(m, "Echorouk TV", "2026-10-11") == []


# ------------------------------------------------------------------ clutter

@pytest.mark.parametrize("n, dur, expected", [(3, 90, "Low"), (4, 120, "Medium"), (8, 230, "Medium"),
                                              (9, 200, "High"), (3, 270, "High"), (1, 30, "Low")])
def test_clutter_score(n, dur, expected):
    assert clutter_score(n, dur) == expected


def test_clutter_on_reconstructed_pods(m):
    m.chain([f"B{i}" for i in range(9)], T(20), seconds=20, gap=1.0)    # 9 ads -> High
    m.chain(["X", "Y", "Z"], T(21), seconds=90, gap=1.0)                # 3 ads, 272 s -> High (duration)
    m.chain(["P", "Q", "R", "S", "T"], T(22), seconds=15)               # 5 ads -> Medium
    m.chain(["U", "V"], T(23), seconds=15)                              # Low
    assert [p.clutter for p in pods(m, "Echorouk TV", "2026-10-10")] == ["High", "High", "Medium", "Low"]


def test_calculate_ad_positions_on_raw_pod():
    ads = [dict(occurrence_id=i, brand=b, ad_type="self", start=s, end=s + 10, duration=10)
           for i, (b, s) in enumerate([("B", 12.0), ("A", 0.0), ("C", 24.0)])]
    pod = calculate_ad_positions(Pod("POD-X", "ch", "live:ch", True, 0.0, 34.0, ads))
    assert [(a["brand"], a["position"]) for a in pod.ads] == [("A", "FIRST"), ("B", "MIDDLE"), ("C", "LAST")]
    assert pod.clutter == "Low"


# ------------------------------------------------------------------ aggregates

def test_position_distribution_and_channel_clutter(m):
    m.chain(["Atlas", "Sahara", "Oasis"], T(20))                 # Atlas FIRST, Sahara MIDDLE, Oasis LAST
    m.chain(["Sahara", "Atlas"], T(21))                          # Sahara FIRST, Atlas LAST
    m.air("Atlas", T(22), 30)                                    # Atlas SOLO
    m.chain(["Oasis", "Sahara"], T(20), channel="Ennahar TV")
    ps = pods(m, None, "2026-10-10")
    dist = {d["brand"]: d for d in position_distribution(ps)}
    atlas = dist["Atlas"]
    assert (atlas["first"], atlas["middle"], atlas["last"], atlas["solo"]) == (1, 0, 1, 1)
    assert atlas["premium_pct"] == pytest.approx(100.0)           # both in-break airings in premium slots
    assert atlas["solo_pct"] == pytest.approx(100 / 3)
    sahara = dist["Sahara"]
    assert (sahara["first"], sahara["middle"], sahara["last"]) == (1, 1, 1)
    assert sahara["premium_pct"] == pytest.approx(200 / 3)
    clutter = {c["channel"]: c for c in channel_clutter(ps)}
    assert clutter["Echorouk TV"]["pods"] == 3 and clutter["Echorouk TV"]["avg_ads_per_pod"] == pytest.approx(2.0)
    assert clutter["Echorouk TV"]["max_ads_per_pod"] == 3
    assert clutter["Ennahar TV"]["avg_pod_seconds"] == pytest.approx(62.0)


# ------------------------------------------------------------------ persistence

def test_materialize_is_idempotent_and_keeps_hashes(tmp_path):
    ad_a, _, broadcast = generate()
    db = tmp_path / "mat.db"
    ingest_reference(ad_a, "Acme", "Ad A", db_path=db)
    scan_source(broadcast, db_path=db, evidence_dir=tmp_path)
    assert materialize_pods(db_path=db) == 1
    assert materialize_pods(db_path=db) == 1
    conn = init_db(db)
    row = conn.execute("SELECT * FROM ad_occurrences").fetchone()
    conn.close()
    assert row["pod_id"].startswith("POD-REC-") and row["pod_position"] == "SOLO" and row["pod_density"] == 1
    assert verify_row(row) is True                               # pod cache is outside the hashed fields


def test_schema_migration_adds_pod_columns(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as c:
        c.executescript("""
            CREATE TABLE ad_occurrences (occurrence_id INTEGER PRIMARY KEY AUTOINCREMENT, ad_id INTEGER NOT NULL,
              source_name TEXT NOT NULL, airing_start REAL NOT NULL, airing_end REAL NOT NULL,
              duration_detected REAL NOT NULL, confidence_score REAL NOT NULL, evidence_frame_path TEXT,
              human_verified INTEGER NOT NULL DEFAULT 0, detected_at TEXT NOT NULL DEFAULT 'x',
              UNIQUE (ad_id, source_name, airing_start));
            INSERT INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected,
              confidence_score) VALUES (1, 'live:Old TV', 1.0, 31.0, 30.0, 0.9);
        """)
    conn = init_db(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ad_occurrences)")}
    row = conn.execute("SELECT * FROM ad_occurrences").fetchone()
    conn.close()
    assert {"pod_id", "pod_position", "pod_density"} <= cols
    assert row["confidence_score"] == 0.9 and row["pod_id"] is None


# ------------------------------------------------------------------ scale + CLI

def test_scale_one_month_of_breaks(m):
    rng_brands = [f"Brand{i:02d}" for i in range(20)]
    t0 = T(0)
    for d in range(30):                                          # 30 days x 24 breaks x 6 ads = 4320 airings
        for h in range(24):
            m.chain([rng_brands[(d + h + k) % 20] for k in range(6)], t0 + d * 86400 + h * 3600, seconds=20)
    m.commit()
    t = time.perf_counter()
    ps = cluster_occurrences_into_pods(db_path=m.db)
    elapsed = time.perf_counter() - t
    print(f"\n[PODS] {sum(p.total_ads_in_pod for p in ps)} airings -> {len(ps)} pods in {elapsed:.2f}s")
    assert len(ps) == 720 and all(p.total_ads_in_pod == 6 for p in ps) and elapsed < 5


def test_cli_pods(m):
    m.chain(["Atlas", "Sahara", "Oasis"], T(20))
    m.commit()
    out = subprocess.run([sys.executable, "-m", "kihc.cli", "--db", str(m.db), "pods", "--channel", "Echorouk TV",
                          "--date", "2026-10-10", "--materialize"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "POD-20261010-ECHOROUK_TV-001" in out.stdout and "1 of 3" in out.stdout and "Last-in-Break" in out.stdout
    assert "Cached pod fields on 3 occurrence(s)" in out.stdout
