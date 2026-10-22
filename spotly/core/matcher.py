"""Sliding-window BER matcher over a streamed target.

The target is decoded through an FFmpeg pipe in fixed-size chunks and
fingerprinted incrementally; only ~max(ad length) of fingerprints is kept
in memory, so RAM stays bounded for arbitrarily long streams.

Search per reference ad:
  1. coarse pass: BER on every COARSE_STRIDE-th sub-fingerprint at every offset
  2. exact pass: full-window BER only at offsets that pass the coarse gate
  3. non-max suppression: one occurrence per airing (best offset wins)
"""
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spotly.config import (CHUNK_SECONDS, EVIDENCE_DIR, FP_ALGORITHM,
                         MATCH_THRESHOLD, SAMPLE_RATE)
from spotly.core.audio import ffmpeg_exe, has_video_stream, iter_pcm
from spotly.core.fingerprint import (StreamingFingerprinter, fps_to_seconds,
                                   from_blob)
from spotly.core.normalize import AdaptiveNormalizer
from spotly.core.proof import compute_verification_hash
from spotly.database.schema import get_db

COARSE_STRIDE = 4
COARSE_MARGIN = 0.06       # coarse gate = threshold - margin (coarse estimate is noisier)
BLOCK_SECONDS = 1.0        # granularity for start/end refinement
BLOCK_THRESHOLD = 0.70     # per-block confidence counted as "ad audio present"
OFFSET_BATCH = 4096


@dataclass
class Detection:
    ad_id: int
    brand_name: str
    campaign_name: str
    source_name: str
    airing_start: float
    airing_end: float
    duration_detected: float
    confidence_score: float
    offset_index: int = field(repr=False, default=0)
    evidence_frame_path: str | None = None
    occurrence_id: int | None = None


class _RefScanner:
    def __init__(self, ad_id, brand, campaign, duration, fp, mask, threshold):
        self.ad_id, self.brand, self.campaign = ad_id, brand, campaign
        self.duration = duration
        self.fp, self.mask = fp, mask
        self.L = fp.size
        self.mask_bits = np.bitwise_count(mask).astype(np.int64)      # reliable bits per frame
        self.coarse_fp = fp[::COARSE_STRIDE]
        self.coarse_mask = mask[::COARSE_STRIDE]
        self.coarse_bits = int(self.mask_bits[::COARSE_STRIDE].sum())
        self.total_bits = int(self.mask_bits.sum())
        self.threshold = threshold
        self.next_offset = 0                 # global index of next window start to test
        self.min_gap = self.L                # an ad cannot overlap another airing of itself
        self.pending = None                  # best candidate of the current airing
        self.found = []
        self.block_len = max(1, int(round(BLOCK_SECONDS / fps_to_seconds(1))))

    def scan(self, buf, base):
        last = base + buf.size - self.L      # last testable global offset
        if last < self.next_offset:
            return
        windows = np.lib.stride_tricks.sliding_window_view(buf, self.L)
        start = self.next_offset - base
        stop = last - base + 1
        for b0 in range(start, stop, OFFSET_BATCH):
            b1 = min(stop, b0 + OFFSET_BATCH)
            coarse = (windows[b0:b1, ::COARSE_STRIDE] ^ self.coarse_fp) & self.coarse_mask
            coarse_conf = 1.0 - np.bitwise_count(coarse).sum(axis=1) / self.coarse_bits
            for rel in np.flatnonzero(coarse_conf >= self.threshold - COARSE_MARGIN):
                local = b0 + int(rel)
                errs = np.bitwise_count((windows[local] ^ self.fp) & self.mask).astype(np.int64)
                conf = 1.0 - errs.sum() / self.total_bits
                if conf >= self.threshold:
                    self._candidate(base + local, float(conf), errs)
        self.next_offset = last + 1

    def _candidate(self, offset, conf, errs):
        if self.pending and offset - self.pending[0] < self.min_gap:
            if conf > self.pending[1]:
                self.pending = (offset, conf, errs)
            return
        self.flush()
        self.pending = (offset, conf, errs)

    def flush(self):
        if self.pending:
            self.found.append(self.pending)
            self.pending = None

    def detection(self, offset, conf, errs, source_name):
        n_blocks = -(-self.L // self.block_len)
        pad = n_blocks * self.block_len - self.L
        block_errs = np.pad(errs, (0, pad)).reshape(n_blocks, -1).sum(axis=1)
        block_bits = np.pad(self.mask_bits, (0, pad)).reshape(n_blocks, -1).sum(axis=1)
        block_conf = 1.0 - block_errs / np.maximum(block_bits, 1)
        present = np.flatnonzero((block_conf >= BLOCK_THRESHOLD) & (block_bits > 0))
        first, last = (int(present[0]), int(present[-1])) if present.size else (0, n_blocks - 1)
        block_sec = fps_to_seconds(self.block_len)
        win_start = fps_to_seconds(offset)
        start = win_start + first * block_sec
        end = win_start + min(self.duration, (last + 1) * block_sec)
        return Detection(self.ad_id, self.brand, self.campaign, source_name,
                         round(start, 3), round(end, 3), round(end - start, 3),
                         round(conf, 4), offset_index=offset)


def _load_refs(conn, threshold, ad_ids=None):
    rows = conn.execute(
        "SELECT ad_id, brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm FROM reference_ads"
    ).fetchall()
    refs = []
    for r in rows:
        if ad_ids is not None and r["ad_id"] not in ad_ids:
            continue
        if r["fp_algorithm"] != FP_ALGORITHM:
            continue  # incompatible fingerprint family; never compare across algorithms
        fp, mask = from_blob(r["fingerprint"])
        refs.append(_RefScanner(r["ad_id"], r["brand_name"], r["campaign_name"],
                                r["duration_sec"], fp, mask, threshold))
    return refs


def extract_evidence(source, det: Detection, evidence_dir) -> str | None:
    """Keyframe at the airing midpoint; spectrogram of the segment for audio-only sources."""
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{det.source_name}_ad{det.ad_id}_{det.airing_start:09.3f}")
    if has_video_stream(source):
        out = evidence_dir / f"{stem}.jpg"
        mid = (det.airing_start + det.airing_end) / 2
        cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-ss", f"{mid:.3f}", "-i", str(source),
               "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "3", str(out)]
    else:
        out = evidence_dir / f"{stem}.png"
        cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-ss", f"{det.airing_start:.3f}",
               "-t", f"{det.duration_detected:.3f}", "-i", str(source),
               "-lavfi", "showspectrumpic=s=640x240:legend=0", str(out)]
    proc = subprocess.run(cmd, capture_output=True)
    return str(out) if proc.returncode == 0 and out.exists() else None


def scan_source(source, source_name=None, db_path=None, threshold=MATCH_THRESHOLD,
                evidence_dir=EVIDENCE_DIR, ad_ids=None) -> list[Detection]:
    """Scan a media file/stream URL for all indexed ads and persist occurrences."""
    source_name = source_name or Path(str(source)).name
    # Short-lived connection for the references; none is held during the (possibly long) decode.
    with get_db(db_path) as conn:
        refs = _load_refs(conn, threshold, ad_ids)
    if not refs:
        return []

    fper = StreamingFingerprinter()
    norm = AdaptiveNormalizer(SAMPLE_RATE)       # level-consistent input; see core/normalize.py
    buf = np.zeros(0, dtype=np.uint32)
    base = 0
    for chunk in iter_pcm(source, CHUNK_SECONDS):
        new = fper.feed(norm.process(chunk))
        buf = np.concatenate([buf, new])
        for ref in refs:
            ref.scan(buf, base)
        keep_from = min(ref.next_offset for ref in refs)    # oldest window still untested
        drop = max(0, keep_from - base)
        buf, base = buf[drop:], base + drop
    for ref in refs:
        ref.flush()

    detections = []
    for ref in refs:
        for offset, conf, errs in ref.found:
            det = ref.detection(offset, conf, errs, source_name)
            with get_db(db_path) as conn:
                existing = conn.execute(
                    "SELECT occurrence_id, evidence_frame_path FROM ad_occurrences "
                    "WHERE ad_id = ? AND source_name = ? AND airing_start = ?",
                    (det.ad_id, det.source_name, det.airing_start)).fetchone()
            if existing:   # already recorded: never regenerate evidence a stored hash covers
                det.evidence_frame_path = existing["evidence_frame_path"]
                detections.append(det)
                continue
            # FFmpeg evidence extraction runs with no transaction open (it can take seconds).
            det.evidence_frame_path = extract_evidence(source, det, evidence_dir)
            evidence = Path(det.evidence_frame_path).read_bytes() if det.evidence_frame_path else None
            vhash = compute_verification_hash(det.ad_id, det.source_name, f"file+{det.airing_start:.3f}s",
                                              det.airing_end, det.duration_detected,
                                              det.confidence_score, evidence)
            with get_db(db_path) as conn:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO ad_occurrences
                       (ad_id, source_name, airing_start, airing_end, duration_detected,
                        confidence_score, evidence_frame_path, verification_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (det.ad_id, det.source_name, det.airing_start, det.airing_end,
                     det.duration_detected, det.confidence_score, det.evidence_frame_path, vhash),
                )
                det.occurrence_id = cur.lastrowid if cur.rowcount else None
            detections.append(det)
    detections.sort(key=lambda d: d.airing_start)
    return detections
