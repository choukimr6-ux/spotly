"""Sub-band energy fingerprint (Haitsma-Kalker style), pure numpy.

Each frame yields one uint32 sub-fingerprint: bit b is the sign of the
time-derivative of the energy difference between adjacent bands b and b+1.
Two aligned fingerprints are compared by Bit Error Rate (BER).

This is the in-memory fallback used when Chromaprint's native library / fpcalc
is unavailable. It is deterministic and needs no external binaries.
"""
import numpy as np

from spotly.config import (BAND_HIGH_HZ, BAND_LOW_HZ, FRAME_SIZE, HOP_SIZE,
                         MASK_KEEP_FRACTION, N_BANDS, SAMPLE_RATE)

BITS_PER_FP = N_BANDS - 1
_BIT_WEIGHTS = (np.uint64(1) << np.arange(BITS_PER_FP, dtype=np.uint64)).astype(np.uint64)


def _band_matrix() -> np.ndarray:
    freqs = np.fft.rfftfreq(FRAME_SIZE, d=1.0 / SAMPLE_RATE)
    edges = np.geomspace(BAND_LOW_HZ, BAND_HIGH_HZ, N_BANDS + 1)
    m = np.zeros((freqs.size, N_BANDS), dtype=np.float32)
    for b in range(N_BANDS):
        m[(freqs >= edges[b]) & (freqs < edges[b + 1]), b] = 1.0
    return m


_BANDS = _band_matrix()
_WINDOW = np.hanning(FRAME_SIZE).astype(np.float32)


class StreamingFingerprinter:
    """Incrementally converts PCM chunks into sub-fingerprints.

    Sub-fingerprint i corresponds to the frame starting at sample (i + 1) * HOP_SIZE.
    The constant offset cancels when aligning reference and target, so a match
    at target index m means the ad starts at sample m * HOP_SIZE.
    """

    def __init__(self):
        self._pending = np.zeros(0, dtype=np.float32)
        self._prev_diff = None

    def feed(self, samples: np.ndarray, with_strength: bool = False):
        """Return uint32 sub-fingerprints; with_strength also returns |derivative| per bit."""
        buf = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        n_frames = 0 if buf.size < FRAME_SIZE else 1 + (buf.size - FRAME_SIZE) // HOP_SIZE
        if n_frames == 0:
            self._pending = buf
            empty = np.zeros(0, dtype=np.uint32)
            return (empty, np.zeros((0, BITS_PER_FP), np.float32)) if with_strength else empty
        frames = np.lib.stride_tricks.sliding_window_view(buf, FRAME_SIZE)[::HOP_SIZE][:n_frames]
        spec = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)).astype(np.float32) ** 2
        energy = spec @ _BANDS                                   # (n_frames, N_BANDS)
        diff = energy[:, :-1] - energy[:, 1:]                    # (n_frames, 32)
        if self._prev_diff is not None:
            diff = np.vstack([self._prev_diff, diff])
        delta = diff[1:] - diff[:-1]
        self._prev_diff = diff[-1:]
        self._pending = buf[n_frames * HOP_SIZE:]
        fp = pack_bits(delta > 0)
        return (fp, np.abs(delta)) if with_strength else fp


def pack_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint64) @ _BIT_WEIGHTS).astype(np.uint32)


def reliability_mask(strength: np.ndarray, keep_fraction: float = MASK_KEEP_FRACTION) -> np.ndarray:
    """Mark the strongest bits of a clean reference as reliable.

    Bits whose energy derivative is near zero (stationary tones, silence, empty
    bands) flip on the slightest noise; excluding them from BER is what makes
    sparse-spectrum creatives (jingles, pure tones) matchable through a noisy channel.
    """
    cutoff = np.quantile(strength, 1.0 - keep_fraction)
    return pack_bits(strength >= cutoff)


def fingerprint_samples(samples: np.ndarray) -> np.ndarray:
    return StreamingFingerprinter().feed(samples)


def fps_to_seconds(n: int) -> float:
    return n * HOP_SIZE / SAMPLE_RATE


def to_blob(fp: np.ndarray, mask: np.ndarray) -> bytes:
    """Serialise as [fp words][mask words], little-endian uint32."""
    return np.concatenate([fp, mask]).astype("<u4").tobytes()


def from_blob(blob: bytes):
    words = np.frombuffer(blob, dtype="<u4").astype(np.uint32)
    half = words.size // 2
    return words[:half], words[half:]


def ber(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Bit error rate between two equal-length fingerprints (optionally over masked bits only)."""
    x = a ^ b
    if mask is None:
        return float(np.bitwise_count(x).sum()) / (a.size * BITS_PER_FP)
    return float(np.bitwise_count(x & mask).sum()) / float(np.bitwise_count(mask).sum())
