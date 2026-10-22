"""Unit + synthetic end-to-end acceptance tests for Spotly.

Run:  python -m pytest spotly/tests -v -s
"""
import ctypes
import sys

import numpy as np
import pytest

from spotly.config import MATCH_THRESHOLD, SAMPLE_RATE
from spotly.core.fingerprint import (StreamingFingerprinter, ber,
                                   fingerprint_samples, from_blob, to_blob)
from spotly.core.ingest import ingest_reference
from spotly.core.matcher import scan_source
from spotly.database.schema import init_db
from spotly.tests.generate_test_assets import generate

RAM_BUDGET_BYTES = int(1.5 * 1024 ** 3)


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError(ctypes.get_last_error())
        return pmc.PeakWorkingSetSize
    import resource
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


# ---------------------------------------------------------------- unit tests

def _noise(seconds, seed):
    return np.random.default_rng(seed).standard_normal(int(seconds * SAMPLE_RATE)).astype(np.float32) * 0.1


def test_schema_creates_tables(tmp_path):
    conn = init_db(tmp_path / "s.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"reference_ads", "ad_occurrences"} <= tables
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ad_occurrences)")}
    assert {"occurrence_id", "ad_id", "source_name", "airing_start", "airing_end", "duration_detected",
            "confidence_score", "evidence_frame_path", "human_verified"} <= cols
    conn.close()


def test_fingerprint_is_deterministic():
    x = _noise(3, seed=1)
    assert np.array_equal(fingerprint_samples(x), fingerprint_samples(x))
    assert ber(fingerprint_samples(x), fingerprint_samples(x)) == 0.0


def test_streaming_matches_one_shot():
    x = _noise(5, seed=2)
    fper = StreamingFingerprinter()
    chunked = np.concatenate([fper.feed(c) for c in np.array_split(x, 7)])
    assert np.array_equal(chunked, fingerprint_samples(x))


def test_unrelated_audio_ber_is_near_half():
    a, b = fingerprint_samples(_noise(5, 3)), fingerprint_samples(_noise(5, 4))
    n = min(a.size, b.size)
    assert 0.45 < ber(a[:n], b[:n]) < 0.55


def test_blob_roundtrip():
    fp = np.random.default_rng(5).integers(0, 2 ** 32, size=100, dtype=np.uint32)
    mask = fp[::-1].copy()
    fp2, mask2 = from_blob(to_blob(fp, mask))
    assert np.array_equal(fp2, fp) and np.array_equal(mask2, mask)


# ------------------------------------------------------- acceptance (E2E)

@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    ad_a, ad_b, broadcast = generate()
    work = tmp_path_factory.mktemp("e2e")
    db = work / "spotly_test.db"
    ad_a_id = ingest_reference(ad_a, "Acme", "Ad A", db_path=db)
    ad_b_id = ingest_reference(ad_b, "Decoy Co", "Ad B (never aired)", db_path=db)
    detections = scan_source(broadcast, source_name="synthetic_broadcast_60s",
                             db_path=db, evidence_dir=work / "evidence")
    return dict(db=db, ad_a_id=ad_a_id, ad_b_id=ad_b_id, detections=detections, broadcast=broadcast)


def test_ad_a_detected_in_window(e2e):
    hits = [d for d in e2e["detections"] if d.ad_id == e2e["ad_a_id"]]
    assert len(hits) == 1, f"expected exactly one airing of Ad A, got {hits}"
    d = hits[0]
    print(f"\n[ACCEPTANCE] Ad A start={d.airing_start:.3f}s end={d.airing_end:.3f}s "
          f"duration={d.duration_detected:.3f}s confidence={d.confidence_score:.4f}")
    assert 24.0 <= d.airing_start <= 26.0
    assert 34.0 <= d.airing_end <= 36.0
    assert d.confidence_score > 0.85
    assert d.confidence_score >= MATCH_THRESHOLD


def test_decoy_not_detected(e2e):
    assert not [d for d in e2e["detections"] if d.ad_id == e2e["ad_b_id"]]


def test_occurrence_persisted_with_evidence(e2e):
    conn = init_db(e2e["db"])
    rows = conn.execute("SELECT * FROM ad_occurrences").fetchall()
    conn.close()
    assert len(rows) == 1
    row = rows[0]
    assert row["ad_id"] == e2e["ad_a_id"]
    assert row["human_verified"] == 0
    from pathlib import Path
    ev = Path(row["evidence_frame_path"])
    assert ev.exists() and ev.stat().st_size > 1000
    print(f"\n[EVIDENCE] {ev} ({ev.stat().st_size} bytes)")


def test_rescan_is_idempotent(e2e):
    scan_source(e2e["broadcast"], source_name="synthetic_broadcast_60s",
                db_path=e2e["db"], evidence_dir=e2e["db"].parent / "evidence")
    conn = init_db(e2e["db"])
    assert conn.execute("SELECT COUNT(*) FROM ad_occurrences").fetchone()[0] == 1
    conn.close()


def test_peak_memory_within_budget(e2e):
    peak = peak_rss_bytes()
    print(f"\n[MEMORY] peak working set {peak / 1024 ** 2:.1f} MiB (budget 1536 MiB)")
    assert 0 < peak < RAM_BUDGET_BYTES
