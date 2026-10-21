"""Proof-of-performance integrity hashing.

verification_hash = SHA-256 over a canonical, versioned record:

    "KIHC-POP-v1\n" + canonical JSON {ad_id, channel, timestamp_cet, airing_end,
                                       duration, confidence, evidence_sha256}

which covers the fields requested (ad_id + channel + Algeria timestamp + evidence
bytes) and additionally the airing end, duration and confidence, so none of the
figures printed on a certificate can be edited without breaking the hash.

Scope: this is tamper *evidence* against accidental or casual edits. Anyone with
write access to both the database and the evidence folder can recompute a valid
hash; non-repudiation needs a secret-keyed signature (HMAC / Ed25519) or an
external timestamp authority on top.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Algeria observes CET (UTC+1) all year with no daylight saving.
ALGERIA_TZ = timezone(timedelta(hours=1), "CET")
HASH_VERSION = "KIHC-POP-v1"


def algeria_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, ALGERIA_TZ).isoformat(timespec="milliseconds")


def canonical_payload(ad_id, channel, timestamp_cet, airing_end, duration, confidence,
                      evidence_bytes: bytes | None, evidence_sha256: str | None = None) -> bytes:
    """`evidence_sha256` seals a clip that is already on disk (its SHA-256, computed over the written file) without
    holding its bytes; it is the same value `sha256(evidence_bytes)` would give, so both spellings produce one hash."""
    record = {
        "ad_id": int(ad_id),
        "channel": str(channel),
        "timestamp_cet": str(timestamp_cet),
        "airing_end": f"{float(airing_end):.3f}",
        "duration": f"{float(duration):.3f}",
        "confidence": f"{float(confidence):.4f}",
        "evidence_sha256": evidence_sha256 or (hashlib.sha256(evidence_bytes).hexdigest() if evidence_bytes else None),
    }
    return (HASH_VERSION + "\n" + json.dumps(record, sort_keys=True, separators=(",", ":"))).encode("utf-8")


def compute_verification_hash(ad_id, channel, timestamp_cet, airing_end, duration, confidence,
                              evidence_bytes: bytes | None, evidence_sha256: str | None = None) -> str:
    return hashlib.sha256(canonical_payload(ad_id, channel, timestamp_cet, airing_end, duration,
                                            confidence, evidence_bytes, evidence_sha256)).hexdigest()


def record_timestamp(row) -> str:
    """The timestamp field that was hashed: Algeria time for live rows, file offset otherwise."""
    return row["algeria_timestamp"] or f"file+{float(row['airing_start']):.3f}s"


def file_sha256(path) -> str | None:
    """SHA-256 of a file's bytes, or None if it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def combine_evidence(png_sha256: str | None, clip_sha256: str) -> str:
    """One evidence digest for an airing that has BOTH a spectrogram image and a persisted audio clip (Phase 29): editing,
    replacing or deleting either one changes it, so the seal covers both."""
    return hashlib.sha256(f"KIHC-EVD-v2|{png_sha256 or '-'}|{clip_sha256}".encode("utf-8")).hexdigest()


def _sealed_clip_sha256(row) -> str | None:
    """The clip digest the row was sealed with (Phase 29), or None for rows sealed over a spectrogram / no evidence."""
    keys = row.keys() if hasattr(row, "keys") else ()
    return row["seal_clip_sha256"] if "seal_clip_sha256" in keys else None


def _clip_path(row, evidence_dir=None) -> Path | None:
    """Absolute clip path; anything that resolves outside the evidence folder counts as missing."""
    from kihc.config import EVIDENCE_DIR
    rel = row["evidence_path"] if "evidence_path" in row.keys() else None
    if not rel:
        return None
    if Path(rel).is_absolute():                  # a clip written under a custom evidence root keeps its absolute path
        return Path(rel)
    root = Path(evidence_dir or EVIDENCE_DIR).resolve()
    p = (root / rel).resolve()
    return p if root in p.parents else None


def hash_for_row(row, evidence_dir=None) -> str:
    """Recompute the hash from a stored ad_occurrences row and its evidence file on disk.

    A row sealed over a persisted clip (`seal_clip_sha256`) is re-verified against the clip file as it is NOW: the digest
    is recomputed from the bytes on disk, so an edited, replaced or deleted clip breaks the seal even if the database row
    still carries the original digest."""
    sealed = _sealed_clip_sha256(row)
    if sealed:
        p = _clip_path(row, evidence_dir)
        actual = file_sha256(p) if p is not None else None
        png = row["evidence_frame_path"]
        png_sha = file_sha256(png) if png else None
        return compute_verification_hash(row["ad_id"], row["source_name"], record_timestamp(row), row["airing_end"],
                                         row["duration_detected"], row["confidence_score"], None,
                                         evidence_sha256=combine_evidence(png_sha, actual or "CLIP-MISSING"))
    path = row["evidence_frame_path"]
    evidence = Path(path).read_bytes() if path and Path(path).exists() else None
    return compute_verification_hash(row["ad_id"], row["source_name"], record_timestamp(row),
                                     row["airing_end"], row["duration_detected"],
                                     row["confidence_score"], evidence)


def verify_row(row, evidence_dir=None) -> bool | None:
    """True = intact, False = tampered/evidence changed, None = no hash recorded."""
    stored = row["verification_hash"]
    if not stored:
        return None
    return hash_for_row(row, evidence_dir) == stored
