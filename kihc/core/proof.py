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
                      evidence_bytes: bytes | None) -> bytes:
    record = {
        "ad_id": int(ad_id),
        "channel": str(channel),
        "timestamp_cet": str(timestamp_cet),
        "airing_end": f"{float(airing_end):.3f}",
        "duration": f"{float(duration):.3f}",
        "confidence": f"{float(confidence):.4f}",
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest() if evidence_bytes else None,
    }
    return (HASH_VERSION + "\n" + json.dumps(record, sort_keys=True, separators=(",", ":"))).encode("utf-8")


def compute_verification_hash(ad_id, channel, timestamp_cet, airing_end, duration, confidence,
                              evidence_bytes: bytes | None) -> str:
    return hashlib.sha256(canonical_payload(ad_id, channel, timestamp_cet, airing_end, duration,
                                            confidence, evidence_bytes)).hexdigest()


def record_timestamp(row) -> str:
    """The timestamp field that was hashed: Algeria time for live rows, file offset otherwise."""
    return row["algeria_timestamp"] or f"file+{float(row['airing_start']):.3f}s"


def hash_for_row(row) -> str:
    """Recompute the hash from a stored ad_occurrences row and its evidence file on disk."""
    path = row["evidence_frame_path"]
    evidence = Path(path).read_bytes() if path and Path(path).exists() else None
    return compute_verification_hash(row["ad_id"], row["source_name"], record_timestamp(row),
                                     row["airing_end"], row["duration_detected"],
                                     row["confidence_score"], evidence)


def verify_row(row) -> bool | None:
    """True = intact, False = tampered/evidence changed, None = no hash recorded."""
    stored = row["verification_hash"]
    if not stored:
        return None
    return hash_for_row(row) == stored
