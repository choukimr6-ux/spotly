"""Deterministic synthetic broadcast for discovery tests (11025 Hz, generated in memory in chunks).

Creatives are broadband like real spots: a moving chord with vibrato, speech-like amplitude-modulated
noise, and percussion. Programme content between breaks never repeats. Every airing gets its own
noise floor, and spots are separated by short near-silent gaps.
"""
import numpy as np

from spotly.config import SAMPLE_RATE as SR


def creative(seed: int, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    step = rng.uniform(0.45, 0.8)
    notes = rng.uniform(300, 2400, size=int(seconds / step) + 2)
    for i, f0 in enumerate(notes):                                    # melody/chord changing every `step`
        a, b = int(i * step * SR), min(n, int((i + 1) * step * SR))
        if a >= n:
            break
        tt = t[a:b]
        for mult, amp in ((1.0, 0.3), (1.25, 0.18), (1.5, 0.12)):
            x[a:b] += amp * np.sin(2 * np.pi * f0 * mult * tt + 3 * np.sin(2 * np.pi * 5 * tt))
    env = np.repeat(rng.uniform(0, 1, size=int(seconds * 4) + 1), SR // 4)[:n]          # syllable envelope
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < 300) | (f > 3400)] = 0
    x += 0.25 * env * np.fft.irfft(spec, n) / (np.abs(np.fft.irfft(spec, n)).max() + 1e-9)
    clicks = np.zeros(n)
    clicks[np.arange(0, n, int(0.5 * SR))] = 1.0
    x += 0.3 * np.convolve(clicks, np.exp(-np.arange(300) / 40) * rng.standard_normal(300), "same")
    return (x / np.abs(x).max() * 0.6).astype(np.float32)


def programme(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec /= np.sqrt(np.maximum(np.arange(spec.size), 1))
    x = np.fft.irfft(spec, n)
    x = x / np.abs(x).max() * 0.25
    t = np.arange(n) / SR
    for _ in range(int(seconds / 3)):                                  # random non-repeating tone gestures
        a = rng.integers(0, max(1, n - SR))
        b = min(n, a + int(rng.uniform(0.3, 2.0) * SR))
        x[a:b] += 0.2 * np.sin(2 * np.pi * rng.uniform(200, 3000) * t[a:b] * (1 + rng.uniform(-0.2, 0.2) * t[a:b]))
    return x.astype(np.float32)


def build_schedule(spots: dict, n_breaks: int = 5, spacing: float = 300.0, first: float = 120.0, seed: int = 1):
    """Breaks every `spacing` s, each a shuffled subset of spots with 2 s gaps. -> (events, total_seconds)."""
    rng = np.random.default_rng(seed)
    events, t = [], first
    names = list(spots)
    for k in range(n_breaks):
        order = list(rng.permutation(names))
        if k % 2 == 1:                                                 # vary the line-up between breaks
            order = order[:-1] if len(order) > 2 else order
        cur = t
        for name in order:
            events.append((name, cur, cur + spots[name].size / SR))
            cur += spots[name].size / SR + 2.0
        t += spacing
    return events, t + 60.0


def render(spots: dict, events, total_seconds: float, seed: int = 7, chunk_seconds: float = 30.0):
    """Yield (start_sample, float32 chunk) of the whole broadcast without holding it all in memory."""
    rng = np.random.default_rng(seed)
    n_total = int(total_seconds * SR)
    chunk = int(chunk_seconds * SR)
    for c0 in range(0, n_total, chunk):
        c1 = min(n_total, c0 + chunk)
        x = programme((c1 - c0) / SR, seed=1000 + c0 // chunk)
        for name, s, e in events:
            a, b = int(s * SR), int(s * SR) + spots[name].size
            lo, hi = max(a - 2 * SR, c0), min(b + 2 * SR, c1)          # clear programme around the spot
            if lo < hi:
                x[lo - c0:hi - c0] = 0.003 * rng.standard_normal(hi - lo)
            lo, hi = max(a, c0), min(b, c1)
            if lo < hi:
                x[lo - c0:hi - c0] += spots[name][lo - a:hi - a]
        x += 0.004 * rng.standard_normal(x.size).astype(np.float32)      # per-airing channel noise
        yield c0, x.astype(np.float32)
