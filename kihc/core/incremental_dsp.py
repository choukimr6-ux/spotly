"""Phase 29: incremental live DSP - every PCM sample is filtered and fingerprinted exactly once.

The live worker used to re-run, on every 3 s match pass, the RMS normalisation, the 101-tap FIR decimation and the
sub-band fingerprint over the WHOLE ring (30 s and more): ~90 % of each pass re-did audio that an earlier pass had already
processed. `IncrementalFingerprintStream` keeps just the state that is needed to continue a stream (the FIR's left/right
context, the fingerprinter's partial frame and previous band-energy difference) and turns each newly received piece of PCM
into a *delta*: the new sub-fingerprints, their global frame index and the new 11 025 Hz audio.

Parity: with the normaliser off, the concatenation of all deltas (plus `flush()`) is bit-identical to running
`np.convolve(x, taps, "same")[::DECIM]` and one `StreamingFingerprinter().feed()` over the whole stream. The frame grid is the
global one (frame k starts at sample k * HOP_SIZE of the 11 025 Hz clock), as before.
"""
from dataclasses import dataclass

import numpy as np

from kihc.config import HOP_SIZE, LIVE_SAMPLE_RATE, SAMPLE_RATE
from kihc.core.fingerprint import StreamingFingerprinter
from kihc.core.normalize import AdaptiveNormalizer

DECIM = LIVE_SAMPLE_RATE // SAMPLE_RATE
assert DECIM * SAMPLE_RATE == LIVE_SAMPLE_RATE, "live rate must be an integer multiple of the fingerprint rate"
FP_FEED_SAMPLES = 5 * SAMPLE_RATE          # fingerprint in 5 s pieces to bound transient RAM


def design_taps(n_taps: int = 101, cutoff: float = 0.45 / DECIM) -> np.ndarray:
    """Windowed-sinc low-pass (identical to the filter the worker has always used)."""
    k = np.arange(n_taps) - n_taps // 2
    h = 2 * cutoff * np.sinc(2 * cutoff * k) * np.hamming(n_taps)
    return (h / h.sum()).astype(np.float32)


class IncrementalDecimator:
    """Streaming `np.convolve(x, taps, "same")[::decim]` over a stream whose first sample has global index `g_start`.

    Output j is centred on stream sample j * decim (zero padded before the first sample, and after the last one on `flush()`),
    so it needs `half` samples of look-ahead: `push()` returns only the outputs whose context is complete, `flush()` the rest.
    """

    BLOCK_OUTPUTS = 4096                # filter outputs computed per matmul: ~1.6 MB of windows, whatever the push size

    def __init__(self, taps: np.ndarray, decim: int = DECIM):
        self.taps = np.asarray(taps, np.float32)
        assert self.taps.size % 2 == 1, "an odd tap count keeps the filter centred on a sample"
        self.half = self.taps.size // 2
        self.decim = decim
        self._kernel = self.taps[::-1].copy()              # window . reversed taps == convolution output
        self._buf = np.zeros(0, np.float32)
        self._buf_start = 0                                 # stream index of _buf[0]
        self._next = 0                                      # stream index (multiple of decim) of the next output centre

    @property
    def samples_in(self) -> int:
        return self._buf_start + self._buf.size

    def _emit(self, n_last: int) -> np.ndarray:
        """Outputs centred on stream samples _next, _next + decim, ... <= n_last (missing context reads as zeros)."""
        if n_last < self._next:
            return np.zeros(0, np.float32)
        lo, hi = self._next - self.half, n_last + self.half + 1
        seg = np.zeros(hi - lo, np.float32)
        a, b = max(lo, self._buf_start), min(hi, self.samples_in)
        if b > a:
            seg[a - lo:b - lo] = self._buf[a - self._buf_start:b - self._buf_start]
        n_out = (n_last - self._next) // self.decim + 1
        out = np.empty(n_out, np.float32)
        for a in range(0, n_out, self.BLOCK_OUTPUTS):   # a matmul over strided windows copies them: bound that copy
            m = min(self.BLOCK_OUTPUTS, n_out - a)
            piece = seg[a * self.decim: a * self.decim + (m - 1) * self.decim + self.taps.size]
            out[a:a + m] = np.lib.stride_tricks.sliding_window_view(piece, self.taps.size)[::self.decim] @ self._kernel
        self._next += self.decim * n_out
        return out

    def _trim(self) -> None:
        keep_from = max(self._buf_start, self._next - self.half)
        if keep_from > self._buf_start:
            self._buf = self._buf[keep_from - self._buf_start:]
            self._buf_start = keep_from

    def push(self, x: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, np.asarray(x, np.float32)])
        out = self._emit(self.samples_in - 1 - self.half)
        self._trim()
        return out

    def flush(self) -> np.ndarray:
        out = self._emit(self.samples_in - 1)
        self._trim()
        return out


@dataclass
class Delta:
    """What one `push()` / `flush()` produced. `fp[i]` is global frame index `base + i`; `x11` starts at 11 025 Hz sample `s_start`."""
    fp: np.ndarray
    base: int
    x11: np.ndarray
    s_start: int

    @property
    def empty(self) -> bool:
        return self.fp.size == 0 and self.x11.size == 0


class IncrementalFingerprintStream:
    """Normalise (optional) -> decimate -> fingerprint one contiguous stream, piece by piece.

    `g_start` is the global 22 050 Hz index of the first sample that will be pushed. The first decimator output sits at the first
    even global sample and fingerprinting starts at the first 11 025 Hz sample on the global frame grid, exactly as the
    full-window passes did (`s_aligned`).
    """

    def __init__(self, g_start: int, taps: np.ndarray | None = None, normalize: bool = True):
        self.taps = design_taps() if taps is None else taps
        self._skip_in = g_start % DECIM                     # keep the 2:1 phase on the global clock
        s0 = (g_start + self._skip_in) // DECIM
        self.s_aligned = -(-s0 // HOP_SIZE) * HOP_SIZE
        self._skip_out = self.s_aligned - s0                # decimated samples before the first frame boundary
        self._dec = IncrementalDecimator(self.taps)
        self._fper = StreamingFingerprinter()
        self._norm = AdaptiveNormalizer(LIVE_SAMPLE_RATE) if normalize else None
        self._base0 = self.s_aligned // HOP_SIZE
        self._emitted = 0                                   # sub-fingerprints produced so far
        self._x11_out = 0                                   # 11 025 Hz samples returned so far (after the alignment skip)
        self.closed = False

    def _finish(self, y: np.ndarray) -> Delta:
        if self._skip_out:
            drop = min(self._skip_out, y.size)
            y, self._skip_out = y[drop:], self._skip_out - drop
        s_start = self.s_aligned + self._x11_out
        # the fingerprinter's FFT works on (frames x 2048) arrays: feed it in 5 s pieces so a long backlog (the first pass
        # after a reconnect or a reference reload) never allocates hundreds of MB at once
        parts = [self._fper.feed(y[i:i + FP_FEED_SAMPLES]) for i in range(0, y.size, FP_FEED_SAMPLES)]
        fp = np.concatenate(parts) if parts else np.zeros(0, np.uint32)
        d = Delta(fp, self._base0 + self._emitted, y, s_start)
        self._emitted += fp.size
        self._x11_out += y.size
        return d

    def push(self, pcm_f32: np.ndarray) -> Delta:
        x = np.asarray(pcm_f32, np.float32)
        if self._skip_in:
            x, self._skip_in = x[self._skip_in:], 0
        if self._norm is not None and x.size:
            x = self._norm.process(x)
        return self._finish(self._dec.push(x))

    def flush(self) -> Delta:
        self.closed = True
        return self._finish(self._dec.flush())

    @property
    def frame_end(self) -> int:
        """Global index one past the newest sub-fingerprint produced."""
        return self._base0 + self._emitted
