"""Index a reference ad: decode audio via pipe, fingerprint, store in SQLite."""
import numpy as np

from kihc.config import CHUNK_SECONDS, FP_ALGORITHM, SAMPLE_RATE
from kihc.core.audio import iter_pcm
from kihc.core.fingerprint import (StreamingFingerprinter, reliability_mask,
                                   to_blob)
from kihc.core.normalize import AdaptiveNormalizer
from kihc.database.schema import get_db


def fingerprint_file(path):
    """Return (fingerprint, reliability mask, duration seconds) for a media file."""
    fper = StreamingFingerprinter()
    norm = AdaptiveNormalizer(SAMPLE_RATE)       # same level conditioning as the matcher
    fps, strengths, n_samples = [], [], 0
    for chunk in iter_pcm(path, CHUNK_SECONDS):
        n_samples += chunk.size
        fp, strength = fper.feed(norm.process(chunk), with_strength=True)
        fps.append(fp)
        strengths.append(strength)
    if not fps:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32), 0.0
    fp = np.concatenate(fps)
    return fp, reliability_mask(np.concatenate(strengths)), n_samples / SAMPLE_RATE


def ingest_reference(path, brand_name: str, campaign_name: str, db_path=None) -> int:
    fp, mask, duration = fingerprint_file(path)
    if fp.size < 64:
        raise ValueError(f"{path}: too short to fingerprint ({duration:.2f}s)")
    with get_db(db_path) as conn:
        conn.execute(
            """INSERT INTO reference_ads (brand_name, campaign_name, duration_sec, fingerprint, fp_algorithm)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (brand_name, campaign_name) DO UPDATE SET
                 duration_sec = excluded.duration_sec,
                 fingerprint  = excluded.fingerprint,
                 fp_algorithm = excluded.fp_algorithm""",
            (brand_name, campaign_name, duration, to_blob(fp, mask), FP_ALGORITHM),
        )
        ad_id = conn.execute(
            "SELECT ad_id FROM reference_ads WHERE brand_name = ? AND campaign_name = ?",
            (brand_name, campaign_name),
        ).fetchone()[0]
    return ad_id
