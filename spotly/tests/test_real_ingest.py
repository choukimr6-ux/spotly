"""Phase 3: real-media ingestion (audio + video containers), duplicate protection, recording scans."""
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from spotly.core.audio import ffmpeg_exe
from spotly.core.library import (ingest_bytes, ingest_media, scan_bytes,
                               scan_recording, timecode)
from spotly.core.memsource import InMemorySource
from spotly.core.proof import verify_row
from spotly.database.schema import init_db
from spotly.reports.certificate import certificate_for
from spotly.tests.generate_test_assets import generate

ROOT = Path(__file__).resolve().parents[2]

# container -> ffmpeg output args (all derived from the synthetic Ad A creative)
FORMATS = {
    "mp4": ["-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "128k"],   # moov at end
    "m4a": ["-vn", "-c:a", "aac", "-b:a", "128k"],
    "mp3": ["-vn", "-c:a", "libmp3lame", "-b:a", "128k"],
    "wav": ["-vn", "-c:a", "pcm_s16le"],
    "aac": ["-vn", "-c:a", "aac", "-b:a", "96k", "-f", "adts"],
}


def _ff(*args):
    subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y", *map(str, args)], check=True)


def _count(db, table):
    conn = init_db(db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    ad_a, ad_b, broadcast = generate()
    d = tmp_path_factory.mktemp("real_media")
    out = {"ad_a": ad_a, "ad_b": ad_b, "broadcast": broadcast, "dir": d}
    for ext, args in FORMATS.items():
        p = d / f"ad_a.{ext}"
        _ff("-i", ad_a, *args, p)
        out[ext] = p
    out["cutdown"] = d / "ad_a_cutdown_5s.m4a"
    _ff("-i", ad_a, "-vn", "-t", "5", "-c:a", "aac", out["cutdown"])
    return out


# ------------------------------------------------------------------ in-memory source

def test_inmemory_source_supports_ranges_and_is_private():
    data = bytes(range(256)) * 100
    with InMemorySource(data, "x.bin") as url:
        assert requests.get(url, timeout=5).content == data
        r = requests.get(url, headers={"Range": "bytes=10-19"}, timeout=5)
        assert r.status_code == 206 and r.content == data[10:20]
        assert r.headers["Content-Range"] == f"bytes 10-19/{len(data)}"
        assert requests.get(url, headers={"Range": "bytes=-5"}, timeout=5).content == data[-5:]
        assert requests.get(url.rsplit("/", 2)[0] + "/guess/x.bin", timeout=5).status_code == 404
        assert url.startswith("http://127.0.0.1:")
    with pytest.raises(requests.ConnectionError):
        requests.get(url, timeout=2)                                  # server gone after the block


def test_mp4_fixture_is_non_faststart(media):
    raw = media["mp4"].read_bytes()
    assert raw.find(b"moov") > raw.find(b"mdat") > 0                   # index at the end: unpipeable


# ------------------------------------------------------------------ ingestion from containers

@pytest.mark.parametrize("ext", list(FORMATS))
def test_ingest_from_bytes_and_detect(media, ext, tmp_path):
    db = tmp_path / f"{ext}.db"
    t0 = time.perf_counter()
    res = ingest_bytes(media[ext].read_bytes(), media[ext].name, "Acme", f"Ad A ({ext})",
                       expected_duration=10, db_path=db)
    elapsed = time.perf_counter() - t0
    assert res.committed and res.ad_id and not res.warnings, res
    assert res.measured_duration == pytest.approx(10.0, abs=0.15)
    rows = scan_recording(media["broadcast"], "broadcast", db_path=db, evidence_dir=tmp_path)
    print(f"\n[INGEST] {ext}: {elapsed:.2f}s, {res.measured_duration:.3f}s audio -> "
          f"{[(r['airing_start'], r['confidence_score']) for r in rows]}")
    assert len(rows) == 1 and 24.0 <= rows[0]["airing_start"] <= 26.0
    assert rows[0]["confidence_score"] > 0.85


def test_ingest_from_video_path(media, tmp_path):
    db = tmp_path / "path.db"
    res = ingest_media(media["mp4"], "Acme", "Ad A", db_path=db)
    assert res.committed and _count(db, "reference_ads") == 1


def test_expected_duration_mismatch_warns_but_indexes(media, tmp_path):
    res = ingest_media(media["m4a"], "Acme", "Ad A", expected_duration=30, db_path=tmp_path / "d.db")
    assert res.committed and any("differs from the expected" in w for w in res.warnings)


def test_file_without_audio_is_rejected(tmp_path):
    silent_video = tmp_path / "video_only.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=d=3:s=160x90:r=5", "-an", "-c:v", "libx264", "-preset", "ultrafast",
        silent_video)
    with pytest.raises(RuntimeError):                                  # FFmpeg: no audio stream to decode
        ingest_bytes(silent_video.read_bytes(), silent_video.name, "X", "Y", db_path=tmp_path / "n.db")
    assert _count(tmp_path / "n.db", "reference_ads") == 0


# ------------------------------------------------------------------ duplicate protection

@pytest.fixture
def library_db(media, tmp_path):
    db = tmp_path / "lib.db"
    assert ingest_media(media["ad_a"], "Acme", "Ad A", db_path=db).committed
    assert ingest_media(media["ad_b"], "Decoy Co", "Ad B", db_path=db).committed
    return db


@pytest.mark.parametrize("ext", ["mp3", "wav"])
def test_reencoded_copy_is_refused(media, library_db, ext):
    res = ingest_bytes(media[ext].read_bytes(), media[ext].name, "Acme", f"Ad A copy ({ext})", db_path=library_db)
    assert not res.committed and res.ad_id is None
    assert res.duplicate_of["campaign_name"] == "Ad A" and res.duplicate_of["relation"] == "same creative"
    assert _count(library_db, "reference_ads") == 2


def test_cutdown_is_flagged(media, library_db):
    res = ingest_media(media["cutdown"], "Acme", "Ad A 5s", db_path=library_db)
    assert not res.committed and res.duplicate_of["relation"] == "cut-down of existing ad"


def test_allow_duplicate_overrides(media, library_db):
    res = ingest_media(media["mp3"], "Acme", "Ad A radio cut", db_path=library_db, allow_duplicate=True)
    assert res.committed and res.duplicate_of is not None and _count(library_db, "reference_ads") == 3


def test_reingest_same_name_updates_in_place(media, library_db):
    res = ingest_media(media["m4a"], "Acme", "Ad A", db_path=library_db)   # same brand+campaign: not a dup
    assert res.committed and res.duplicate_of is None and res.ad_id == 1
    assert _count(library_db, "reference_ads") == 2


def test_distinct_creative_is_not_flagged(media, tmp_path):
    db = tmp_path / "one.db"
    ingest_media(media["ad_a"], "Acme", "Ad A", db_path=db)
    res = ingest_media(media["ad_b"], "Decoy Co", "Ad B", db_path=db)
    assert res.committed and res.duplicate_of is None


# ------------------------------------------------------------------ recording scans + dedup

def test_scanning_twice_does_not_duplicate(media, library_db, tmp_path):
    data = media["broadcast"].read_bytes()
    first = scan_bytes(data, "broadcast_60s.mp4", "Ennahar TV · rec1", db_path=library_db, evidence_dir=tmp_path)
    second = scan_bytes(data, "broadcast_60s.mp4", "Ennahar TV · rec1", db_path=library_db, evidence_dir=tmp_path)
    assert [r["is_new"] for r in first] == [True] and [r["is_new"] for r in second] == [False]
    assert first[0]["occurrence_id"] == second[0]["occurrence_id"]
    assert first[0]["verification_hash"] == second[0]["verification_hash"]
    assert _count(library_db, "ad_occurrences") == 1
    conn = init_db(library_db)
    row = conn.execute("SELECT * FROM ad_occurrences").fetchone()
    conn.close()
    assert verify_row(row) is True
    assert certificate_for(library_db, row["occurrence_id"]).startswith(b"%PDF-")


@pytest.fixture(scope="module")
def ten_minute_recording(media):
    """10-min broadcast (small video + AAC): Ad A at 02:00 and 07:10, Ad B at 05:00, over ambient noise."""
    p = media["dir"] / "recording_10min.mp4"
    graph = ("anoisesrc=c=pink:a=0.08:r=44100:d=600:seed=5[bed];"
             "[0:a]asplit=2[a1][a2];[a1]adelay=120000:all=1[x1];[a2]adelay=430000:all=1[x2];"
             "[1:a]adelay=300000:all=1[x3];"
             "[bed][x1][x2][x3]amix=inputs=4:duration=first:normalize=0[aout]")
    _ff("-i", media["ad_a"], "-i", media["ad_b"], "-f", "lavfi", "-i", "testsrc2=d=600:s=160x90:r=5",
        "-filter_complex", graph, "-map", "2:v", "-map", "[aout]", "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "64k", "-t", "600", p)
    return p


def test_ten_minute_recording_scan(media, library_db, ten_minute_recording, tmp_path):
    t0 = time.perf_counter()
    rows = scan_recording(ten_minute_recording, "Echorouk TV · 10min", db_path=library_db, evidence_dir=tmp_path)
    elapsed = time.perf_counter() - t0
    found = [(r["campaign_name"], timecode(r["airing_start"]), r["confidence_score"]) for r in rows]
    print(f"\n[SCAN 10min] {elapsed:.1f}s -> {found}")
    by_ad = sorted((r["campaign_name"], r["airing_start"]) for r in rows)
    assert [c for c, _ in by_ad] == ["Ad A", "Ad A", "Ad B"]
    starts = {c: [s for cc, s in by_ad if cc == c] for c in ("Ad A", "Ad B")}
    assert starts["Ad A"][0] == pytest.approx(120, abs=1) and starts["Ad A"][1] == pytest.approx(430, abs=1)
    assert starts["Ad B"][0] == pytest.approx(300, abs=1)
    assert all(r["confidence_score"] > 0.85 and r["is_new"] for r in rows)
    conn = init_db(library_db)
    stored = conn.execute("SELECT * FROM ad_occurrences").fetchall()
    conn.close()
    assert len(stored) == 3 and all(verify_row(r) is True for r in stored)


def test_peak_memory_within_phase3_budget():
    from spotly.tests.test_connectors import _proc_peak_ws
    import os
    if sys.platform != "win32":
        pytest.skip("peak working set measured via Win32 API")
    peak = _proc_peak_ws(os.getpid())
    print(f"\n[MEMORY] test process peak working set {peak / 1024 ** 2:.1f} MiB (budget 300 MiB)")
    assert 0 < peak < 300 * 1024 ** 2


# ------------------------------------------------------------------ CLI

def test_cli_ingest_and_duplicate_exit_code(media, tmp_path):
    db = tmp_path / "cli.db"
    run = lambda *a: subprocess.run([sys.executable, "-m", "spotly.cli", "--db", str(db), "ingest", *map(str, a)],
                                    capture_output=True, text=True, cwd=ROOT)
    ok = run(media["mp4"], "--brand", "Acme", "--campaign", "Ad A", "--expected-duration", "10")
    assert ok.returncode == 0 and "Indexed ad_id=1" in ok.stdout, ok.stderr
    dup = run(media["mp3"], "--brand", "Acme", "--campaign", "Ad A mp3")
    assert dup.returncode == 1 and "same creative" in dup.stdout
    forced = run(media["mp3"], "--brand", "Acme", "--campaign", "Ad A mp3", "--allow-duplicate")
    assert forced.returncode == 0 and _count(db, "reference_ads") == 2
