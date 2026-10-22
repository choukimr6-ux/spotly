"""Reference-ad library and recording scans for real media (UI + CLI entry points).

ingest_media / ingest_bytes : index a creative from a path or uploaded bytes, refusing
                              likely duplicates of an existing reference unless forced.
scan_recording / scan_bytes : scan a recorded broadcast with the unchanged chunked matcher
                              and return per-airing rows (new vs already recorded).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spotly.config import EVIDENCE_DIR, FP_ALGORITHM, MATCH_THRESHOLD
from spotly.core.fingerprint import from_blob
from spotly.core.ingest import fingerprint_file, ingest_reference
from spotly.core.matcher import scan_source
from spotly.core.memsource import InMemorySource
from spotly.database.schema import AD_TYPES, init_db

MAX_UPLOAD_MB = 100              # uploads live in RAM; larger recordings should be scanned from a path
DURATION_TOLERANCE_SEC = 1.0
MEDIA_TYPES = ("mp4", "m4a", "mp3", "wav", "aac", "mov", "mkv", "ts")


@dataclass
class IngestResult:
    ad_id: int | None
    measured_duration: float
    committed: bool
    duplicate_of: dict | None = None          # {ad_id, brand_name, campaign_name, confidence, relation}
    warnings: list = field(default_factory=list)


def _best_alignment(query_fp, query_mask, ref_fp, ref_mask):
    """Max masked confidence of the shorter fingerprint slid across the longer one.

    The shorter side's own reliability mask is used, exactly as in matching.
    """
    if query_fp.size >= ref_fp.size:
        long_fp, short_fp, short_mask = query_fp, ref_fp, ref_mask
    else:
        long_fp, short_fp, short_mask = ref_fp, query_fp, query_mask
    bits = int(np.bitwise_count(short_mask).sum())
    if bits == 0 or short_fp.size == 0:
        return 0.0
    win = np.lib.stride_tricks.sliding_window_view(long_fp, short_fp.size)
    best = 0.0
    for b0 in range(0, win.shape[0], 2048):
        errs = np.bitwise_count((win[b0:b0 + 2048] ^ short_fp) & short_mask).sum(axis=1)
        best = max(best, 1.0 - errs.min() / bits)
    return float(best)


def find_duplicate(fp, mask, duration, db_path=None, threshold=MATCH_THRESHOLD, exclude=None):
    """Existing reference whose audio matches this creative (same, cut-down or extended version)."""
    conn = init_db(db_path)
    try:
        rows = conn.execute("SELECT ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm "
                            "FROM reference_ads").fetchall()
    finally:
        conn.close()
    best = None
    for r in rows:
        if r["fp_algorithm"] != FP_ALGORITHM or (exclude and (r["brand_name"], r["campaign_name"]) == exclude):
            continue
        ref_fp, ref_mask = from_blob(r["fingerprint"])
        conf = _best_alignment(fp, mask, ref_fp, ref_mask)
        if conf >= threshold and (best is None or conf > best["confidence"]):
            ratio = duration / r["duration_sec"] if r["duration_sec"] else 1.0
            relation = ("same creative" if 0.9 <= ratio <= 1.1 else
                        "cut-down of existing ad" if ratio < 0.9 else "extended version of existing ad")
            best = dict(ad_id=r["ad_id"], brand_name=r["brand_name"], campaign_name=r["campaign_name"],
                        confidence=round(conf, 4), relation=relation)
    return best


def validate_tags(industry=None, ad_type="self", target_competitor_of=None):
    """Normalise competitive-intel tags; raises ValueError on an invalid combination."""
    ad_type = (ad_type or "self").strip().lower()
    if ad_type not in AD_TYPES:
        raise ValueError(f"ad_type must be one of {AD_TYPES}, got {ad_type!r}")
    industry = " ".join(industry.split()) if industry and industry.strip() else None
    target = target_competitor_of.strip() if target_competitor_of and target_competitor_of.strip() else None
    if ad_type == "self" and target:
        raise ValueError("target_competitor_of only applies to competitor ads")
    return industry, ad_type, target


def set_ad_tags(ad_id: int, industry=None, ad_type="self", target_competitor_of=None, db_path=None):
    industry, ad_type, target = validate_tags(industry, ad_type, target_competitor_of)
    conn = init_db(db_path)
    try:
        with conn:
            cur = conn.execute("UPDATE reference_ads SET industry = ?, ad_type = ?, target_competitor_of = ? "
                               "WHERE ad_id = ?", (industry, ad_type, target, int(ad_id)))
        if cur.rowcount == 0:
            raise KeyError(f"reference ad {ad_id} not found")
    finally:
        conn.close()


def ingest_media(source, brand_name: str, campaign_name: str, *, expected_duration: float | None = None,
                 db_path=None, allow_duplicate: bool = False, industry: str | None = None,
                 ad_type: str = "self", target_competitor_of: str | None = None) -> IngestResult:
    """Index a creative from a path or URL. Duplicates are reported and not committed unless forced."""
    validate_tags(industry, ad_type, target_competitor_of)          # fail fast, before any decoding
    fp, mask, duration = fingerprint_file(source)
    result = IngestResult(None, round(duration, 3), False)
    if fp.size < 64:
        result.warnings.append(f"Audio too short to fingerprint ({duration:.2f}s) or no audio stream found.")
        return result
    if expected_duration and abs(duration - expected_duration) > DURATION_TOLERANCE_SEC:
        result.warnings.append(f"Measured audio duration {duration:.2f}s differs from the expected "
                               f"{expected_duration:.2f}s by more than {DURATION_TOLERANCE_SEC:.0f}s "
                               f"(wrong file, slate/padding, or a different cut?).")
    result.duplicate_of = find_duplicate(fp, mask, duration, db_path, exclude=(brand_name, campaign_name))
    if result.duplicate_of and not allow_duplicate:
        d = result.duplicate_of
        result.warnings.append(f"Not indexed: {d['relation']} '{d['brand_name']} – {d['campaign_name']}' "
                               f"(ad_id {d['ad_id']}, match {d['confidence']:.1%}). Indexing it again would "
                               f"double-count every airing. Re-submit with allow_duplicate to override.")
        return result
    conn = init_db(db_path)
    try:
        existed = conn.execute("SELECT 1 FROM reference_ads WHERE brand_name = ? AND campaign_name = ?",
                               (brand_name, campaign_name)).fetchone() is not None
    finally:
        conn.close()
    result.ad_id = ingest_reference(source, brand_name, campaign_name, db_path=db_path)
    # Re-indexing an existing creative keeps its tags unless new ones are given (use set_ad_tags to change).
    if not existed or industry or ad_type != "self" or target_competitor_of:
        set_ad_tags(result.ad_id, industry, ad_type, target_competitor_of, db_path=db_path)
    result.committed = True
    return result


def ingest_bytes(data: bytes, filename: str, brand_name: str, campaign_name: str, **kwargs) -> IngestResult:
    """Same as ingest_media for an in-memory upload (never written to disk)."""
    with InMemorySource(data, filename) as url:
        return ingest_media(url, brand_name, campaign_name, **kwargs)


def scan_recording(source, source_name: str, *, db_path=None, threshold=MATCH_THRESHOLD,
                   evidence_dir=EVIDENCE_DIR) -> list[dict]:
    """Scan a recorded broadcast; one row per detected airing with its occurrence id and hash."""
    dets = scan_source(source, source_name=source_name, db_path=db_path, threshold=threshold,
                       evidence_dir=evidence_dir)
    conn = init_db(db_path)
    try:
        rows = []
        for d in dets:
            row = conn.execute(
                "SELECT occurrence_id, verification_hash, human_verified FROM ad_occurrences "
                "WHERE ad_id = ? AND source_name = ? AND airing_start = ?",
                (d.ad_id, d.source_name, d.airing_start)).fetchone()
            rows.append(dict(occurrence_id=row["occurrence_id"] if row else None,
                             is_new=d.occurrence_id is not None, ad_id=d.ad_id, brand_name=d.brand_name,
                             campaign_name=d.campaign_name, source_name=d.source_name,
                             airing_start=d.airing_start, airing_end=d.airing_end,
                             duration_detected=d.duration_detected, confidence_score=d.confidence_score,
                             evidence_frame_path=d.evidence_frame_path,
                             verification_hash=row["verification_hash"] if row else None))
    finally:
        conn.close()
    return rows


def scan_bytes(data: bytes, filename: str, source_name: str | None = None, **kwargs) -> list[dict]:
    with InMemorySource(data, filename) as url:
        return scan_recording(url, source_name or Path(filename).name, **kwargs)


def timecode(seconds: float) -> str:
    """Recording offset as HH:MM:SS.mmm"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
