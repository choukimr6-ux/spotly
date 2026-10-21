"""Adaptive RMS loudness normalization.

Note on what this does and does not buy: the sub-band fingerprint compares
*relative* band energies, so a constant gain leaves its bits unchanged (measured:
<= 1 of 164,928 bits differ between -35, -20 and -10 dBFS copies of the test
broadcast). Normalization therefore does not change match scores for constant
level differences; it keeps every downstream stage (fingerprint, evidence
spectrograms, telemetry) working on a consistent level. It cannot undo clipping
that happened upstream of KIHC.
"""
import numpy as np

TARGET_DBFS = -20.0
SILENCE_DBFS = -60.0       # below this, do not amplify (would only raise noise / dead air)
MAX_GAIN_DB = 30.0
MIN_GAIN_DB = -30.0
RAMP_SECONDS = 0.5         # gain changes are ramped, never stepped, between chunks
_EPS = 1e-12


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS of float audio in [-1, 1]; -inf-safe (floors at -120)."""
    if x.size == 0:
        return -120.0
    return float(10.0 * np.log10(max(np.mean(np.square(x, dtype=np.float64)), _EPS)))


def gain_for(x: np.ndarray, target_dbfs: float = TARGET_DBFS) -> float:
    """Linear gain bringing x to target_dbfs, gated on silence and clamped."""
    level = rms_dbfs(x)
    if level < SILENCE_DBFS:
        return 1.0
    gain_db = float(np.clip(target_dbfs - level, MIN_GAIN_DB, MAX_GAIN_DB))
    return 10.0 ** (gain_db / 20.0)


def normalize_rms(x: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """One-shot normalization of a whole buffer (float32 out, no clipping applied)."""
    return (x * gain_for(x, target_dbfs)).astype(np.float32, copy=False)


class AdaptiveNormalizer:
    """Per-chunk RMS normalization for streams, with a smooth gain ramp between chunks.

    Each chunk's gain is computed from its own RMS; the first RAMP_SECONDS of a
    chunk glide from the previous gain so frames straddling a chunk boundary do
    not see a step in level.
    """

    def __init__(self, sample_rate: int, target_dbfs: float = TARGET_DBFS):
        self.target_dbfs = target_dbfs
        self.ramp = max(1, int(RAMP_SECONDS * sample_rate))
        self.prev_gain = None
        self.last_input_dbfs = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        self.last_input_dbfs = rms_dbfs(chunk)
        gain = gain_for(chunk, self.target_dbfs)
        if self.prev_gain is None or chunk.size == 0:
            out = chunk * gain
        else:
            n = min(self.ramp, chunk.size)
            g = np.full(chunk.size, gain, dtype=np.float32)
            g[:n] = np.linspace(self.prev_gain, gain, n, dtype=np.float32)
            out = chunk * g
        self.prev_gain = gain
        return out.astype(np.float32, copy=False)
