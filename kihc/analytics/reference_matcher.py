"""Exact-match engine for verified commercials: peak-constellation hashing (Phase 13.1).

Why a second fingerprint next to core/fingerprint.py: the sub-band BER fingerprint must be slid over every
offset of the buffer; a constellation of spectral peaks is looked up in an inverted index instead, so cost
does not grow with the number of references and the stream may join an ad at any point.

Pipeline
  PCM 11025 Hz -> log-spectrogram (1024/256) -> local-maximum peaks (level-invariant: each must be within
  REL_DB of its +-1 s neighbourhood) -> pairs (anchor, next FANOUT peaks within 0.1-2 s) -> 19-bit hash
  [ f1 bin | log2(f2/f1) bin | log2(dt) bin ] -> sorted-array inverted index {hash -> (asset, t_ref)}.

Time-offset invariance: only *differences* matter. Every hit votes for u = t_stream - t_ref / s, the stream
time at which the reference would have started; a real airing piles its hits into one bin (BIN_W) whatever the
absolute time, the point where the capture joined the ad, or the broadcast latency.

Speed/pitch jitter (+-5 %): the frequency ratio log2(f2/f1) is invariant under tape-style speed change, f1 and
dt shift by < 0.07 octave, which the query absorbs by probing the neighbouring f1/dt bins; the slope s is
searched on a 2 % grid so the drift over a 55 s spot stays inside the vote bin.

Confidence = coverage x support x ambiguity (each in 0..1):
  coverage  : share of the overlapped reference seconds (those that carry hashes) holding >= 2 hits that agree
              with the winning alignment - a chance alignment never covers the ad end to end;
  support   : 1 - exp(-hits / K_SAT), saturating evidence volume;
  ambiguity : drops when a second, unrelated alignment of the same asset is nearly as strong.
It is a calibrated score, not a probability; thresholds are exercised in tests/test_reference_matcher.py.

Hybrid gate (ReferenceMatcher): >= 0.96 acoustic alone; 0.85 .. <0.96 needs the packshot perceptual hash to
agree; below 0.85 nothing is claimed and the caller falls back to the discovery loop. No STT, no OCR here.
"""
import collections
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from kihc.core.proof import algeria_iso, compute_verification_hash
from kihc.analytics.brand_extractor import compute_voice_ratio, is_music_dominant
from kihc.data import reference_db as rdb
from kihc.stream import clock_sync, time_synchronizer

log = logging.getLogger("kihc.reference")

SR = 11025
N_FFT = 1024
HOP = 256
FPS = SR / HOP
F_LO_HZ, F_HI_HZ = 250.0, 4000.0
K_LO = int(math.ceil(F_LO_HZ * N_FFT / SR))
K_HI = int(F_HI_HZ * N_FFT / SR)
_WIN = np.hanning(N_FFT).astype(np.float32)

# peak picking
CTX_FR = 44                         # frames of +-1 s context for the relative threshold
PEAK_T_HALF, PEAK_F_HALF = 8, 16    # a peak is the maximum of +-190 ms x +-172 Hz
REL_DB = 28.0
FLOOR_DB = -85.0

# pairing / hashing
DT_MIN_S, DT_MAX_S = 0.10, 2.0
DT_MIN_FR, DT_MAX_FR = int(math.ceil(DT_MIN_S * FPS)), int(DT_MAX_S * FPS)
FANOUT, KSCAN = 4, 12
F0_HZ, W_F = 250.0, 0.2             # f1: 0.2-octave bins
R_MAX, W_R = 4.0, 0.04              # log2(f2/f1) in -4..4, 0.04-octave bins
W_D = 0.08                          # log2(dt / 0.1 s) in 0.08-octave bins
N_FB, N_RB, N_DB = 32, 256, 64

# voting / decision
SLOPES = np.arange(0.93, 1.0701, 0.02)
BIN_W = 0.6
MIN_HITS = 10
K_SAT = 12.0
MIN_REF_SLOT_HASHES = 3
MIN_SLOT_HITS = 2
MAX_POSTING = 400
ENV_DT_S = 0.05                     # Phase 21: stream energy envelope resolution
FAMILY_DIP_DB = 12.0                # a cut-down boundary is real only if the stream level drops this much (or goes silent)
FAMILY_SILENCE_DBFS = -50.0
FAMILY_TOL_S = 1.0                  # +- around the cut-down's nominal end
FAMILY_MIN_CONT_HITS = 6            # longer-variant hits past the cut-down boundary that prove the audio carries on ...
FAMILY_MIN_CONT_RATE = 10.0         # ... at >= this many per second (a true continuation gives ~40/s, chance ~2/s)
EDGE_TOL_S = 3.0                    # cut-downs differ from their parent by a fade / tag at the edges
STEP_S = 1.0
STALE_S = 4.0
ACCEPT_CONF = 0.96
VERIFY_FLOOR = 0.85
VISUAL_MAX_BITS = 10                # packshot pHash Hamming distance (of 64) that counts as "same packshot"
MUSIC_VISUAL_MAX_BITS = 12          # Phase 22: music-dominant creatives may share a stock track with another brand's spot
UNCONFIRMED_MUSIC = "UNCONFIRMED_MUSIC_NO_PACKSHOT"  # music-dominant spot with no packshot to check: logged, not claimed
UNCONFIRMED_CUTDOWN = "UNCONFIRMED_CUTDOWN_NO_DIP"   # cut-down snapped without the >= 12 dBFS trailing dip: logged, not claimed
VISUAL_SECONDS_BEFORE_END = 1.5     # discovery captures the packshot this long before the spot's end
LOOKUP_BUDGET_MS = 30.0
INDEX_BUDGET_MB = 50.0


# --------------------------------------------------------------------------------------- DSP
def _maxfilt(a: np.ndarray, half: int, axis: int) -> np.ndarray:
    """Sliding maximum over [i-half, i+half] along `axis` in O(log width) passes (no scipy)."""
    w, n = 2 * half + 1, a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (half, half)
    m = np.pad(a, pad, constant_values=-np.inf)

    def sl(x, lo, hi):
        idx = [slice(None)] * x.ndim
        idx[axis] = slice(lo, hi)
        return x[tuple(idx)]

    ln = 1
    while ln * 2 <= w:
        m = np.maximum(sl(m, 0, m.shape[axis] - ln), sl(m, ln, None))
        ln *= 2
    return np.maximum(sl(m, 0, n), sl(m, w - ln, w - ln + n))


def extract_peaks(x: np.ndarray, frame_offset: int = 0, final_lo: int | None = None, final_hi: int | None = None):
    """Spectral peaks of mono float audio whose first sample is the start of frame `frame_offset`.

    Returns (frames int64 absolute, hz float32); only frames in [final_lo, final_hi) are reported so a streaming
    caller can ask for frames whose +-CTX_FR context is complete.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size < N_FFT:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    n = 1 + (x.size - N_FFT) // HOP
    fr = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:n]
    mag = np.abs(np.fft.rfft(fr * _WIN, axis=1))[:, K_LO - 1:K_HI + 2]
    full = (20.0 * np.log10(mag / (N_FFT / 4) + 1e-9)).astype(np.float32)
    core = full[:, 1:-1]
    level = _maxfilt(core.max(axis=1), CTX_FR, 0)
    is_peak = (core == _maxfilt(_maxfilt(core, PEAK_F_HALF, 1), PEAK_T_HALF, 0))
    keep = is_peak & (core >= level[:, None] - REL_DB) & (core > FLOOR_DB)
    lo = 0 if final_lo is None else max(0, final_lo - frame_offset)
    hi = n if final_hi is None else min(n, final_hi - frame_offset)
    keep[:lo], keep[hi:] = False, False
    r, c = np.nonzero(keep)
    y0, y1, y2 = full[r, c], full[r, c + 1], full[r, c + 2]
    den = y0 - 2 * y1 + y2
    delta = np.clip(np.where(den < -1e-6, 0.5 * (y0 - y2) / np.where(den < -1e-6, den, -1.0), 0.0), -0.5, 0.5)
    hz = ((K_LO + c + delta) * SR / N_FFT).astype(np.float32)
    return (r + frame_offset).astype(np.int64), hz


def _pairs(frames: np.ndarray, lo: int, hi: int):
    """(anchor idx, target idx) for anchors with lo <= frame < hi: the next FANOUT peaks 0.1-2 s later."""
    n = frames.size
    idx = np.arange(n)
    anchor_ok = (frames >= lo) & (frames < hi)
    cnt = np.zeros(n, np.int64)
    a_out, t_out = [], []
    for k in range(1, KSCAN + 1):
        j = idx + k
        jj = np.minimum(j, n - 1)
        dt = frames[jj] - frames
        valid = (j < n) & anchor_ok & (dt >= DT_MIN_FR) & (dt <= DT_MAX_FR) & (cnt < FANOUT)
        a_out.append(idx[valid])
        t_out.append(jj[valid])
        cnt += valid
    return np.concatenate(a_out), np.concatenate(t_out)


def _positions(frames, hz, ai, tj):
    f1, f2 = hz[ai].astype(np.float64), hz[tj].astype(np.float64)
    dt = (frames[tj] - frames[ai]) / FPS
    return (np.log2(f1 / F0_HZ) / W_F, (np.log2(f2 / f1) + R_MAX) / W_R, np.log2(dt / DT_MIN_S) / W_D)


def _encode(fb, rb, db):
    return ((fb.astype(np.uint32) << 14) | (rb.astype(np.uint32) << 6) | db.astype(np.uint32))


def make_hashes(frames: np.ndarray, hz: np.ndarray, lo: int = 0, hi: int | None = None):
    """Index-side hashes (one per pair) -> (codes uint32, anchor frame int64)."""
    if frames.size < 2:
        return np.zeros(0, np.uint32), np.zeros(0, np.int64)
    ai, tj = _pairs(frames, lo, frames[-1] + 1 if hi is None else hi)
    if ai.size == 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.int64)
    pf, pr, pd = _positions(frames, hz, ai, tj)
    fb = np.clip(np.floor(pf), 0, N_FB - 1).astype(np.int64)
    rb = np.clip(np.floor(pr), 0, N_RB - 1).astype(np.int64)
    db = np.clip(np.floor(pd), 0, N_DB - 1).astype(np.int64)
    return _encode(fb, rb, db), frames[ai]


def make_query_hashes(frames: np.ndarray, hz: np.ndarray, lo: int, hi: int):
    """Query-side hashes with neighbour-bin probes -> (codes uint32, anchor frame int64).

    f1 and dt are probed at +-1 bin (speed/pitch jitter moves them by < 0.07 octave, a bin is 0.2 / 0.08);
    the ratio is invariant to speed so it is probed only when it sits near a bin edge (peak-frequency noise).
    """
    if frames.size < 2:
        return np.zeros(0, np.uint32), np.zeros(0, np.int64)
    ai, tj = _pairs(frames, lo, hi)
    if ai.size == 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.int64)
    pf, pr, pd = _positions(frames, hz, ai, tj)
    fb = np.floor(pf).astype(np.int64)
    rb = np.floor(pr).astype(np.int64)
    db = np.floor(pd).astype(np.int64)
    frac_f, frac_r = pf - fb, pr - rb
    codes, anchors = [], []
    for dfb in (-1, 0, 1):
        for ddb in (-1, 0, 1):
            for drb in (-1, 0, 1):
                ok = ((drb == 0) | ((drb == -1) & (frac_r < 0.3)) | ((drb == 1) & (frac_r > 0.7)))
                ok &= (dfb == 0) | ((dfb == -1) & (frac_f < 0.4)) | ((dfb == 1) & (frac_f > 0.6))
                f, r, d = fb + dfb, rb + drb, db + ddb
                ok &= (f >= 0) & (f < N_FB) & (r >= 0) & (r < N_RB) & (d >= 0) & (d < N_DB)
                codes.append(_encode(f[ok], r[ok], d[ok]))
                anchors.append(frames[ai][ok])
    return np.concatenate(codes), np.concatenate(anchors)


def trim_silence(x: np.ndarray, margin_s: float = 0.04, floor_db: float = -45.0, absolute: bool = False):
    """Strip leading/trailing near-silence. -> (trimmed, lead_s). Frames are 20 ms; the floor is relative to the
    loudest frame, or in dBFS (full scale = 1.0) when `absolute` - the master-asset rule: digital silence < -50 dBFS."""
    x = np.asarray(x, dtype=np.float32)
    n = int(0.02 * SR)
    m = x.size // n
    if m == 0:
        return x, 0.0
    rms = np.sqrt((x[:m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12) if absolute else 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    loud = np.nonzero(db > floor_db)[0]
    if loud.size == 0:
        return x, 0.0
    a = max(0, int(loud[0] * n - margin_s * SR))
    b = min(x.size, int((loud[-1] + 1) * n + margin_s * SR))
    return x[a:b], a / SR


def signature_from_audio(x: np.ndarray, *, trim: bool = True, floor_dbfs: float | None = None):
    """Reference signature of a clean creative -> (codes uint32, t_cs uint16, lead_trim_s, trimmed_s).
    floor_dbfs: trim by an absolute level (e.g. -50) instead of relative to the loudest frame."""
    x = np.asarray(x, dtype=np.float32)
    lead = 0.0
    if trim:
        x, lead = (trim_silence(x) if floor_dbfs is None else
                   trim_silence(x, margin_s=0.02, floor_db=floor_dbfs, absolute=True))
    frames, hz = extract_peaks(x)
    codes, anchor = make_hashes(frames, hz)
    t_cs = np.clip(np.round(anchor / FPS * 100), 0, rdb.MAX_T_CS).astype(np.uint16)
    return codes, t_cs, float(lead), x.size / SR


# --------------------------------------------------------------------------------------- visual hash
def phash_image(img) -> str:
    """64-bit DCT perceptual hash as 16 hex chars; identical to imagehash.phash(img) (hash_size 8, highfreq 4)."""
    try:
        import imagehash
        return str(imagehash.phash(img))
    except ImportError:
        pass
    from PIL import Image
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    n = 32
    k = np.arange(n)
    dct = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    low = (dct @ g @ dct.T)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def _as_image(frame):
    """PIL image from JPEG/PNG bytes (what the packshot grabber returns) or an image passed through."""
    if isinstance(frame, (bytes, bytearray)):
        import io

        from PIL import Image
        return Image.open(io.BytesIO(bytes(frame)))
    return frame


def make_packshot_frame_grabber(packshot_grabber, db_path):
    """frame_grabber(channel, at, asset_id) -> JPEG bytes | None, backed by the pilot's PackshotGrabber.

    Uses the matched asset's own sub-band fingerprint to find the spot's end in the live HLS and takes the frame
    1.5 s before it (the same method that produced the stored packshots). None when the stream no longer
    publishes that part: the candidate then stays 'needs_visual' and is not logged.
    """
    def grab(channel, at, asset_id=None):
        if not asset_id:
            return None
        from kihc.database.schema import get_db
        with get_db(db_path) as conn:
            r = conn.execute("SELECT fingerprint FROM discovered_assets WHERE asset_id = ?", (asset_id,)).fetchone()
        if not r:
            return None
        fp = np.frombuffer(r["fingerprint"], dtype="<u4").astype(np.uint32)
        try:
            return packshot_grabber.grab(fp, at + VISUAL_SECONDS_BEFORE_END).jpeg
        except Exception as exc:                                  # PackshotMiss, network, ffmpeg
            log.info("visual check frame unavailable for %s: %s", asset_id, exc)
            return None
    return grab


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# --------------------------------------------------------------------------------------- index
@dataclass
class AssetMeta:
    asset_id: str
    brand_name: str
    sector: str | None
    nominal: float
    lead: float
    trimmed: float
    t_end: float                       # time of the last anchor (s from trimmed start)
    slot_counts: np.ndarray            # hashes per 1 s reference slot
    phash: str | None
    n_hashes: int
    root: int = -1                     # Phase 21: index of the family root (the Parent Campaign master)
    cut_type: str = "MASTER"
    parent: str | None = None
    music_dominant: bool = False       # Phase 22: voice ratio < 20 % -> an airing needs the packshot to be CONFIRMED


class ReferenceIndex:
    """In-memory inverted index over all signatures: three parallel sorted-by-hash arrays (8 bytes per hash)."""

    def __init__(self, signatures: list):
        keys, assets, ts, self.meta = [], [], [], []
        for i, s in enumerate(signatures):
            keys.append(s.codes)
            ts.append(s.t_cs)
            assets.append(np.full(s.codes.size, i, np.uint16))
            slots = np.bincount((s.t_cs // 100).astype(np.int64), minlength=1).astype(np.uint16)
            self.meta.append(AssetMeta(s.asset_id, s.brand_name, s.canonical_sector, s.nominal_duration,
                                       s.lead_trim_s, s.trimmed_s, float(s.t_cs.max()) / 100 if s.t_cs.size else 0.0,
                                       slots, s.visual_p_hash, int(s.codes.size)))
        self.position = {m.asset_id: i for i, m in enumerate(self.meta)}
        tree = rdb.family_tree(signatures)               # hot-reload rebuilds the campaign trees with the index
        for m in self.meta:
            t = tree[m.asset_id]
            m.root, m.cut_type, m.parent = self.position[t["root"]], t["cut_type"], t["parent"]
        for m, s in zip(self.meta, signatures):
            m.music_dominant = bool(getattr(s, "is_music_dominant", False))
        k = np.concatenate(keys) if keys else np.zeros(0, np.uint32)
        order = np.argsort(k, kind="stable")
        self.keys = k[order]
        self.asset = (np.concatenate(assets) if assets else np.zeros(0, np.uint16))[order]
        self.t_cs = (np.concatenate(ts) if ts else np.zeros(0, np.uint16))[order]
        self.max_dur = max([m.trimmed for m in self.meta], default=0.0)
        self._relations = None
        self._containers = {}

    @classmethod
    def from_db(cls, db_path=None) -> "ReferenceIndex":
        return cls(rdb.load_signatures(db_path))

    def __len__(self):
        return len(self.meta)

    def memory_bytes(self) -> int:
        return int(self.keys.nbytes + self.asset.nbytes + self.t_cs.nbytes + sum(m.slot_counts.nbytes for m in self.meta))

    def lookup(self, q: np.ndarray):
        """-> (query index, asset index, t_ref in cs) for every posting of every query hash (over-common hashes skipped)."""
        if q.size == 0 or self.keys.size == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.uint16), np.zeros(0, np.uint16)
        lo = np.searchsorted(self.keys, q, "left")
        n = np.searchsorted(self.keys, q, "right") - lo
        n[n > MAX_POSTING] = 0
        total = int(n.sum())
        if total == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.uint16), np.zeros(0, np.uint16)
        qi = np.repeat(np.arange(q.size), n)
        pos = np.repeat(lo, n) + (np.arange(total) - np.repeat(np.cumsum(n) - n, n))
        return qi, self.asset[pos], self.t_cs[pos]

    def relations(self) -> list[dict]:
        """Pairs of assets that share audio, found by replaying every signature against the index (cached).

        {x, y, conf, x_in_y}: x's audio matches y's at `conf`, x starting `x_in_y` s into y. kind:
          contained  x lies wholly inside the longer y (a cut-down / tag-on version),
          duplicate  same audio, about the same length,
          overlap    they only share part of their audio.
        Real broadcasts re-use creatives this way (30 s cut inside a 45 s spot); without arbitration one airing
        would be logged under both.
        """
        if self._relations is not None:
            return self._relations
        out = []
        if len(self.meta) > 1:
            order = np.argsort(self.asset, kind="stable")
            bounds = np.searchsorted(self.asset[order], np.arange(len(self.meta) + 1))
            tmp = StreamMatcher(self)
            for xi, mx in enumerate(self.meta):
                sel = order[bounds[xi]:bounds[xi + 1]]
                qi, asset, tr = self.lookup(self.keys[sel])
                if qi.size == 0:
                    continue
                tq = self.t_cs[sel][qi].astype(np.float64) / 100.0
                tmp._q_hi = mx.t_end + 0.5
                for yi in np.unique(asset):
                    if int(yi) == xi:
                        continue
                    m = asset == yi
                    if m.sum() < MIN_HITS:
                        continue
                    sc = tmp._score(int(yi), tq[m], tr[m], 0.0)
                    if sc is None or sc["conf"] < VERIFY_FLOOR:
                        continue
                    my = self.meta[int(yi)]
                    x_in_y = -sc["start"] * sc["speed"]
                    inside = x_in_y >= -EDGE_TOL_S and x_in_y + mx.t_end <= my.t_end + EDGE_TOL_S
                    if inside and my.t_end > mx.t_end + 1.0:
                        kind = "contained"
                    elif inside and abs(my.t_end - mx.t_end) <= 1.0 + EDGE_TOL_S:
                        kind = "duplicate"
                    else:
                        kind = "overlap"
                    out.append(dict(x=xi, y=int(yi), conf=round(sc["conf"], 3), x_in_y=round(x_in_y, 2), kind=kind))
        self._relations = out
        self._containers = {}
        for r in out:
            if r["kind"] == "contained":
                self._containers.setdefault(r["x"], []).append((r["y"], r["x_in_y"]))
        return out

    def same_family(self, a: int, b: int) -> bool:
        return a != b and self.meta[a].root == self.meta[b].root

    def families(self) -> dict:
        """root asset_id -> [member asset_ids] for every campaign with more than one variant."""
        out: dict = {}
        for m in self.meta:
            out.setdefault(self.meta[m.root].asset_id, []).append(m.asset_id)
        return {k: v for k, v in out.items() if len(v) > 1}

    def containers(self, x: int) -> list[tuple[int, float]]:
        """[(y, offset of x inside y)] for every longer asset that wholly contains asset x."""
        self.relations()
        return self._containers.get(x, [])


# --------------------------------------------------------------------------------------- streaming matcher
@dataclass
class Candidate:
    asset_id: str
    brand_name: str
    sector: str | None
    nominal_duration: float
    confidence: float
    airing_start: float            # epoch, snapped: start of the creative's first frame on the capture clock
    airing_end: float              # airing_start + nominal duration
    hits: int
    coverage: float
    overlap_s: float
    speed_ratio: float
    runner_up: int
    complete: bool                 # True: the whole ad passed; False: the stream stopped delivering it midway
    packshot_at: float = 0.0
    decision: str = ""
    visual_distance: int | None = None
    occurrence_id: int | None = None
    ad_id: int | None = None
    lookup_ms: float = 0.0
    family_root: str | None = None      # Phase 21: Parent Campaign master this variant rolls up to
    cut_type: str = "MASTER"
    family_note: str = ""               # how a master / cut-down ambiguity was resolved
    no_dip: bool = False                # cut-down snapped without a trailing level dip (Phase 22 policy)
    music_dominant: bool = False        # the creative's voice ratio is under 20 %
    status: str = "CONFIRMED"           # CONFIRMED | UNCONFIRMED_CUTDOWN_NO_DIP (what is stored on the airing row)


class StreamMatcher:
    """One continuous PCM stream (11025 Hz mono) against a ReferenceIndex. Call feed() with consecutive audio."""

    def __init__(self, index: ReferenceIndex):
        self.index = index
        self.pending_index = None      # set by another thread (copy-on-write registry swap); adopted at the next feed()
        self.history_s = index.max_dur + 15.0
        self.lookup_ms = []            # per step, the index query only (what the 30 ms budget refers to)
        self.reset(None)

    def reset(self, epoch0):
        self.epoch0 = epoch0
        self._buf = np.zeros(0, np.float32)
        self._buf_off = 0              # absolute sample index of _buf[0]
        self._n = 0
        self._peak_frame = 0           # first frame whose peaks are not final yet
        self._anchor_frame = 0         # first anchor not hashed yet
        self._pk_f = np.zeros(0, np.int64)
        self._pk_hz = np.zeros(0, np.float32)
        self._hits = {}                # asset idx -> (tq float64, tr_cs uint16)
        self._last_hit_q = {}
        self._q_hi = 0.0
        self._done_until = {}          # asset idx -> stream time before which that asset's hits are spent
        self._claims = []              # stream spans (start, end) of airings already decided
        self._stepped = 0
        self._env = np.zeros(0, np.float32)    # dBFS per ENV_DT_S block of the continuous stream (Phase 21)
        self._env_off = 0                      # blocks dropped from the front of _env
        self._env_rem = np.zeros(0, np.float32)
        self._family_note = {}                 # asset idx -> how its family ambiguity was resolved

    @property
    def stream_seconds(self) -> float:
        return self._n / SR

    def adopt(self, new: "ReferenceIndex") -> None:
        """Switch to a new index WITHOUT touching the audio state: buffered samples, peaks and the hits already
        gathered survive (their asset numbers are re-mapped by asset_id), so no frame is dropped or re-analysed.
        Hits of an asset whose signature changed are discarded; a new asset is matched from the next step on."""
        old = self.index
        remap = {}
        for i, m in enumerate(old.meta):
            j = new.position.get(m.asset_id)
            if j is not None and new.meta[j].n_hashes == m.n_hashes and new.meta[j].trimmed == m.trimmed:
                remap[i] = j
        self._hits = {remap[a]: v for a, v in self._hits.items() if a in remap}
        self._last_hit_q = {remap[a]: v for a, v in self._last_hit_q.items() if a in remap}
        self._done_until = {remap[a]: v for a, v in self._done_until.items() if a in remap}
        self._family_note = {}
        self.index = new
        self.history_s = new.max_dur + 15.0

    def feed(self, samples: np.ndarray, epoch: float | None = None) -> list[Candidate]:
        pending, self.pending_index = self.pending_index, None
        if pending is not None:
            self.adopt(pending)
        x = np.asarray(samples)
        if x.dtype.kind in "iu":
            x = x.astype(np.float32) / 32768.0
        x = x.astype(np.float32, copy=False)
        out = []
        if self.epoch0 is None:
            self.epoch0 = epoch if epoch is not None else 0.0
        elif epoch is not None and abs(epoch - (self.epoch0 + self._n / SR)) > 1.0:
            out += self.flush()                                   # capture clock jumped: audio is not contiguous
            self.reset(epoch)
        self._buf = np.concatenate([self._buf, x])
        self._n += x.size
        self._update_env(x)
        while self._n - self._stepped * int(STEP_S * SR) >= int(STEP_S * SR):
            self._stepped += 1
            out += self._step(final=False)
        return out

    def _update_env(self, x: np.ndarray) -> None:
        blk = int(ENV_DT_S * SR)
        if self._env_rem.size:
            x = np.concatenate([self._env_rem, x])
        n = x.size // blk
        self._env_rem = x[n * blk:].copy()
        if n:
            rms = np.sqrt((x[:n * blk].reshape(n, blk) ** 2).mean(axis=1))
            self._env = np.concatenate([self._env, (20.0 * np.log10(np.maximum(rms, 1e-7))).astype(np.float32)])
            cap = int((self.history_s + 30.0) / ENV_DT_S)
            if self._env.size > cap:
                self._env_off += self._env.size - cap
                self._env = self._env[-cap:]

    def energy_dip(self, t: float, tol: float = FAMILY_TOL_S) -> tuple[bool, float]:
        """(dip?, depth dB) of the continuous stream level within t +- tol (stream seconds): >= 12 dB under the level
        of the preceding seconds, or silence. Needs audio up to t + tol; (False, 0) when it isn't there yet."""
        blk = ENV_DT_S
        i0 = int((t - tol) / blk) - self._env_off
        i1 = int((t + tol) / blk) - self._env_off
        r0 = int((t - 4.0) / blk) - self._env_off
        r1 = int((t - tol) / blk) - self._env_off
        if i0 < 0 or i1 > self._env.size or r1 - max(r0, 0) < 4:
            return False, 0.0
        ref = float(np.median(self._env[max(r0, 0):r1]))
        win = self._env[i0:i1]
        low = float(np.min(np.convolve(win, np.ones(2) / 2, mode="valid"))) if win.size > 2 else float(win.min())
        depth = ref - low
        return bool(low < FAMILY_SILENCE_DBFS or depth >= FAMILY_DIP_DB), round(depth, 2)

    def flush(self) -> list[Candidate]:
        """Stream ended: decide every alignment that still holds enough evidence."""
        return self._step(final=True)

    # -- one analysis step
    def _step(self, final: bool) -> list[Candidate]:
        n_frames = 1 + (self._n - N_FFT) // HOP if self._n >= N_FFT else 0
        limit = n_frames if final else n_frames - CTX_FR
        if limit > self._peak_frame:
            start_fr = max(0, self._peak_frame - CTX_FR)
            a0 = start_fr * HOP - self._buf_off
            fr, hz = extract_peaks(self._buf[a0:], start_fr, self._peak_frame, limit)
            self._pk_f, self._pk_hz = np.concatenate([self._pk_f, fr]), np.concatenate([self._pk_hz, hz])
            self._peak_frame = limit
        anchor_limit = self._peak_frame if final else self._peak_frame - DT_MAX_FR
        t0 = time.perf_counter()
        if anchor_limit > self._anchor_frame and self._pk_f.size:
            q, qa = make_query_hashes(self._pk_f, self._pk_hz, self._anchor_frame, anchor_limit)
            qi, asset, tr = self.index.lookup(q)
            if qi.size:
                tq = qa[qi] / FPS
                for a in np.unique(asset):
                    m = asset == a
                    pt, pr = self._hits.get(int(a), (np.zeros(0), np.zeros(0, np.uint16)))
                    self._hits[int(a)] = (np.concatenate([pt, tq[m]]), np.concatenate([pr, tr[m]]))
                    self._last_hit_q[int(a)] = float(tq[m].max())
            self._q_hi = anchor_limit / FPS
            self._anchor_frame = anchor_limit
        self.lookup_ms.append((time.perf_counter() - t0) * 1000.0)
        lookup_ms = self.lookup_ms[-1]
        keep_from = (self._peak_frame - CTX_FR) * HOP
        drop = keep_from - self._buf_off
        if drop > 0:
            self._buf, self._buf_off = self._buf[drop:], self._buf_off + drop
        keep_pk = self._pk_f >= self._anchor_frame
        self._pk_f, self._pk_hz = self._pk_f[keep_pk], self._pk_hz[keep_pk]
        return self._decide(final, lookup_ms)

    def _decide(self, final: bool, lookup_ms: float) -> list[Candidate]:
        q_lo = max(0.0, self._q_hi - self.history_s)
        ready = []
        for a in list(self._hits):
            tq, tr = self._hits[a]
            m = tq >= max(q_lo, self._done_until.get(a, 0.0))
            tq, tr = tq[m], tr[m]
            self._hits[a] = (tq, tr)
            if tq.size < MIN_HITS:
                if tq.size == 0:
                    del self._hits[a]
                continue
            sc = self._score(a, tq, tr, q_lo)
            if sc is None:
                continue
            meta = self.index.meta[a]
            ended = sc["r_hi"] >= meta.t_end - 0.3
            stale = self._q_hi - self._last_hit_q.get(a, 0.0) >= STALE_S
            if not (ended or stale or final):
                continue
            # A join part-way through an ad (capture started inside it) is fine; an ad that stops part-way is not:
            # unless the whole creative passed, 80 % of it must be explained.
            need = min(max(4.0, 0.5 * meta.t_end), 0.9 * meta.t_end) if ended else 0.8 * meta.t_end
            if sc["overlap"] >= need and sc["conf"] >= VERIFY_FLOOR:
                ready.append((a, sc, ended))
            elif ended or stale:
                keep = tq > self._q_hi - 1.0
                self._hits[a] = (tq[keep], tr[keep])                # attempt over, drop it
        # Arbitration between assets that share audio: whoever explains the most stream audio goes first, a tie goes
        # to the shorter creative (Occam), and whatever lies inside an already claimed stretch is not a second airing.
        ready.sort(key=lambda r: (-round(r[1]["overlap"] / 3.0), self.index.meta[r[0]].t_end))
        out = []
        for a, sc, ended in ready:
            meta, s = self.index.meta[a], sc["speed"]
            span = (sc["start"], sc["start"] + meta.t_end / s)
            if self._covered_by_claim(span):
                self._done_until[a] = span[1] + 1.0
                self._hits.pop(a, None)
                continue
            if not final and self._inside_unfinished_container(a, sc):
                continue                                            # wait: a longer creative may still claim this audio
            if a not in self._family_note and any(self.index.same_family(a, y) for y, _ in self.index.containers(a)):
                self._note_snap(a, sc["start"] + meta.t_end / s)      # decided without waiting (bulk feed / flush): still measure
            start_stream = sc["start"] - meta.lead / s
            start = self.epoch0 + start_stream
            out.append(Candidate(meta.asset_id, meta.brand_name, meta.sector, meta.nominal,
                                 round(sc["conf"], 4), round(start, 3), round(start + meta.nominal, 3),
                                 sc["k"], round(sc["coverage"], 3), round(sc["overlap"], 2), round(s, 3),
                                 sc["runner"], complete=ended, lookup_ms=lookup_ms,
                                 packshot_at=round(start + meta.nominal - VISUAL_SECONDS_BEFORE_END, 3),
                                 family_root=self.index.meta[meta.root].asset_id if meta.root >= 0 else None,
                                 cut_type=meta.cut_type, family_note=(note := self._family_note.pop(a, "")),
                                 no_dip="without a level dip" in note, music_dominant=meta.music_dominant))
            self._claims.append(span)
            self._done_until[a] = span[1] + 1.0
            self._hits.pop(a, None)
            self._spend_hits(span[1] - 0.3)
        self._claims = [c for c in self._claims if c[1] > self._q_hi - self.history_s]
        return out

    def _spend_hits(self, until: float) -> None:
        """The audio up to `until` now belongs to a decided airing: hits other assets collected on it (a cut-down
        sharing its sound) are explained and must not seed - or pollute - a later alignment."""
        for b in list(self._hits):
            tq, tr = self._hits[b]
            keep = tq >= until
            if keep.all():
                continue
            if keep.any():
                self._hits[b] = (tq[keep], tr[keep])
            else:
                del self._hits[b]

    def _covered_by_claim(self, span) -> bool:
        length = span[1] - span[0]
        return any(min(span[1], c1) - max(span[0], c0) >= 0.7 * min(length, c1 - c0) for c0, c1 in self._claims)

    def _inside_unfinished_container(self, x: int, sc: dict) -> bool:
        """True while a longer creative that wholly contains asset x is still unfolding over the same audio."""
        for y, x_in_y in self.index.containers(x):
            hy = self._hits.get(y)
            if hy is None or hy[0].size < MIN_HITS:
                continue
            scy = self._score(y, hy[0], hy[1], max(0.0, self._q_hi - self.history_s))
            if scy is None:
                continue
            my = self.index.meta[y]
            same_place = abs(scy["start"] + x_in_y / scy["speed"] - sc["start"]) <= 1.0
            if same_place and self.index.same_family(x, y):
                # No confidence floor here: once the cut-down's end has passed, the longer variant's coverage of the
                # stream so far FALLS (nothing matches after the cut), which is exactly the case to decide on.
                if self._family_wait(x, y, sc, scy, x_in_y, hy):
                    return True
                continue                                    # snapped to the cut-down; unrelated containers may still object
            if scy["conf"] < VERIFY_FLOOR:
                continue
            unfinished = (scy["r_hi"] < my.t_end - 0.3 and self._q_hi - self._last_hit_q.get(y, 0.0) < STALE_S
                          and self._q_hi < scy["start"] + my.t_end / scy["speed"] + 2.0)
            if same_place and unfinished:
                return True
        return False

    def _family_wait(self, x: int, y: int, sc: dict, scy: dict, x_in_y: float, hy) -> bool:
        """Trailing-edge disambiguation between a cut-down x and the longer variant y of the same campaign.

        True  -> keep waiting: the longer variant's audio carries on past x's end, or its evidence isn't in yet.
        False -> snap to x: the stream level dips >= 12 dB (or goes silent) at x's nominal end +- 1 s and the longer
                 variant does not continue. With no dip but no longer-variant evidence either, x is still logged
                 (an ad that really aired must not vanish) and the note says so."""
        mx = self.index.meta[x]
        bnd = sc["start"] + mx.t_end / sc["speed"]                     # x's nominal end on the stream clock
        tq, tr = hy
        beyond = ((tq > bnd + 0.3) & (tr.astype(np.float64) / 100.0 > x_in_y + mx.t_end + 0.3)
                  & (np.abs(tq - (scy["a"] + scy["b"] * tr.astype(np.float64) / 100.0)) < 0.4))
        span = self._q_hi - bnd                                         # seconds of hashed evidence past the boundary
        if span < FAMILY_TOL_S:                                         # not enough of it in yet to tell
            return True
        n = int(beyond.sum())
        if n >= FAMILY_MIN_CONT_HITS and n / span >= FAMILY_MIN_CONT_RATE:   # audio persists: track the longer variant
            return True
        self._note_snap(x, bnd)
        return False

    def _note_snap(self, x: int, bnd: float) -> None:
        """Record how a cut-down was told apart from its master: the measured trailing level dip, or its absence."""
        mx = self.index.meta[x]
        dip, depth = self.energy_dip(bnd)
        self._family_note[x] = (f"snapped to {mx.cut_type} {mx.t_end:.0f}s: level dip {depth:.1f} dB at its end" if dip else
                                f"snapped to {mx.cut_type} {mx.t_end:.0f}s without a level dip (longer variant absent)")

    def _score(self, a: int, tq: np.ndarray, tr_cs: np.ndarray, q_lo: float):
        meta = self.index.meta[a]
        tr = tr_cs.astype(np.float64) / 100.0
        best = None
        for s in SLOPES:
            u = tq - tr / s
            b = np.floor(u / BIN_W).astype(np.int64)
            b0 = int(b.min())
            c = np.bincount(b - b0)
            c2 = c.copy()
            c2[:-1] += c[1:]
            j = int(c2.argmax())
            if best is None or c2[j] > best[0]:
                best = (int(c2[j]), float(s), u, b - b0, j, c, c2)
        k, s, u, b, j, c, c2 = best
        if k < MIN_HITS:
            return None
        k_bin = k
        mask = (b == j) | (b == j + 1)
        # Refine the coarse slope grid: robust line fit tq = a + (1/s) * tr through the winning bin's hits, then
        # re-collect every hit within 0.4 s of that line. Gives start to ~tens of ms and the true speed ratio.
        x, y = tr[mask], tq[mask]
        y0 = float(np.median(y))
        a_fit, b_fit = y0 - float(np.median(x)) / s, 1.0 / s
        for _ in range(2):
            if x.size < 8:
                break
            b_new, a_new = np.polyfit(x, y - y0, 1)
            if not 0.9 < 1.0 / b_new < 1.1:
                break
            a_fit, b_fit = y0 + a_new, b_new
            keep = np.abs(y - (a_fit + b_fit * x)) < 0.25
            if keep.sum() < 8:
                break
            x, y = x[keep], y[keep]
        s = 1.0 / b_fit
        start = float(a_fit)
        mask = np.abs(tq - (a_fit + b_fit * tr)) < 0.4
        k = int(mask.sum())
        c3 = c2.copy()
        c3[max(0, j - 3):j + 4] = 0
        runner = int(c3.max()) if c3.size else 0
        r_lo = max(0.0, s * (q_lo - start))
        r_hi = min(meta.t_end, s * (self._q_hi - start))
        overlap = r_hi - r_lo
        coverage = 0.0
        if overlap > 0:
            first, last = int(math.ceil(r_lo)), int(math.floor(r_hi))
            slots = np.arange(first, max(first, last))
            slots = slots[slots < meta.slot_counts.size]
            applicable = slots[meta.slot_counts[slots] >= MIN_REF_SLOT_HASHES] if slots.size else slots
            if applicable.size:
                hit_slots = np.floor(tr[mask]).astype(np.int64)
                per_slot = np.bincount(hit_slots, minlength=int(applicable.max()) + 1)
                # Chance hits (a noisy channel throws up hundreds of peaks) also land near the winning line: a slot
                # counts as covered only when it beats the background expected there by 3 sigma.
                bg_bin = max(0, tq.size - k_bin) / max(1, c.size)
                bg_slot = 2.0 * bg_bin / max(1, meta.slot_counts.size)
                need_hits = max(MIN_SLOT_HITS, int(math.ceil(bg_slot + 3.0 * math.sqrt(bg_slot))))
                coverage = float((per_slot[applicable] >= need_hits).mean())
        support = 1.0 - math.exp(-k / K_SAT)
        ratio = runner / k
        ambiguity = 1.0 if ratio <= 0.35 else max(0.0, 1.0 - (ratio - 0.35) / 0.35)
        return dict(conf=coverage * support * ambiguity, k=k, coverage=coverage, runner=runner, speed=s,
                    start=start, r_hi=r_hi, overlap=overlap, a=float(a_fit), b=float(b_fit))


# --------------------------------------------------------------------------------------- orchestration
class ReferenceMatcher:
    """Index + one StreamMatcher per channel + hybrid gate + logging + compliance feed.

        rm = ReferenceMatcher(db_path)
        for cand in rm.feed("Echorouk TV", pcm_11025, epoch_of_first_sample): ...

    Decisions on a Candidate:
      confirmed         conf >= 0.96, logged
      confirmed_visual  0.85 <= conf < 0.96 and the packshot pHash agrees, logged
      rejected_visual   0.85 <= conf < 0.96 but the packshot differs: NOT logged (falls back to discovery)
      needs_visual      0.85 <= conf < 0.96 and no frame could be grabbed: NOT logged
    Music-dominant creatives (voice ratio < 20 %: a stock track can be shared by other brands' spots) take the visual
    route at ANY confidence, with a Hamming limit of 12 bits instead of 10: a packshot that differs REJECTS the attribution;
    an exact acoustic match with no packshot to check is logged as UNCONFIRMED_MUSIC_NO_PACKSHOT (never CONFIRMED).
    A cut-down snapped without the trailing level dip is logged with status UNCONFIRMED_CUTDOWN_NO_DIP (excluded from
    penalty claims) unless its packshot matches, which resolves it to CONFIRMED.
      duplicate         the airing is already on file (BER matcher / backfill / re-delivery)
      unlinked          the asset has no promoted reference ad, so no billable identity: NOT logged
    Anything below 0.85 never leaves the matcher: the caller keeps running its normal discovery loop.
    """

    def __init__(self, db_path=None, *, index: ReferenceIndex | None = None, frame_grabber=None,
                 accept: float = ACCEPT_CONF, verify_floor: float = VERIFY_FLOOR, compliance: bool = True,
                 compliance_min_interval: float = 30.0):
        self.db_path = db_path
        self.index = index if index is not None else ReferenceIndex.from_db(db_path)
        self._version = rdb.signature_version(db_path) if index is None else None
        self.frame_grabber = frame_grabber
        self.accept, self.verify_floor = accept, verify_floor
        self.compliance = compliance
        self.compliance_min_interval = compliance_min_interval
        self._streams: dict[str, StreamMatcher] = {}
        self._reload_lock = threading.Lock()
        self._lookup_ms = collections.deque(maxlen=5000)      # per analysis step, all channels (also after a stream ends)
        self._ad_ids: dict[str, int | None] = {}
        self._airings: dict[str, list[tuple[float, float]]] = {}
        self._compliance_dirty: dict[str, float] = {}
        self._last_compliance = 0.0
        self.stats = dict(confirmed=0, visual=0, rejected_visual=0, needs_visual=0, duplicates=0, unlinked=0,
                          unconfirmed_cutdown=0)

    # -- registry
    def reload_if_changed(self) -> bool:
        return self.atomic_reload_signatures()

    def atomic_reload_signatures(self, force: bool = False) -> bool:
        """Copy-on-write hot reload of the registry; safe to call from any thread while channels are being fed.

        The fresh index is built entirely off to the side (the live one is never mutated), then published with a
        single reference assignment. Each running stream notices the new pointer at its next feed() (StreamMatcher.adopt keeps its
        audio buffer, peaks and gathered hits), so the live monitor never pauses, restarts or loses a frame.
        -> True if a new registry was published.
        """
        if self._version is None and not force:
            return False
        with self._reload_lock:                                  # one reloader at a time; feeders never take it
            v = rdb.signature_version(self.db_path)
            if v == self._version and not force:
                return False
            old = {m.asset_id: m for m in self.index.meta}
            shadow = ReferenceIndex.from_db(self.db_path)
            self.index, self._version = shadow, v                # the atomic pointer swap
            self._ad_ids = {}                                    # replaced, not cleared in place
        for m in shadow.meta:
            prev = old.get(m.asset_id)
            if prev is None or prev.n_hashes != m.n_hashes or prev.phash != m.phash:
                log.info("[REGISTRY] Hot-reloaded asset: %s (%.1fs)", m.brand_name, m.nominal)
        return True

    def covered_ad_ids(self) -> set[int]:
        ids = {self._ad_id(m.asset_id) for m in self.index.meta}
        return {i for i in ids if i is not None}

    def _ad_id(self, asset_id: str):
        if asset_id not in self._ad_ids:
            self._ad_ids[asset_id] = rdb.promoted_ad_id(self.db_path, asset_id)
        return self._ad_ids[asset_id]

    def memory_report(self) -> dict:
        b = self.index.memory_bytes()
        lat = list(self._lookup_ms) + [x for s in self._streams.values() for x in s.lookup_ms]
        return dict(assets=len(self.index), hashes=int(self.index.keys.size), index_mb=round(b / 2 ** 20, 3),
                    index_budget_mb=INDEX_BUDGET_MB, bytes_per_hash=8,
                    max_assets_in_budget=int(INDEX_BUDGET_MB * 2 ** 20 / 8 / max(1, self.index.keys.size / max(1, len(self.index)))),
                    lookup_ms_p50=round(float(np.median(lat)), 3) if lat else None,
                    lookup_ms_p95=round(float(np.percentile(lat, 95)), 3) if lat else None,
                    lookup_ms_max=round(float(max(lat)), 3) if lat else None, lookup_budget_ms=LOOKUP_BUDGET_MS)

    # -- streaming
    def feed(self, channel: str, samples: np.ndarray, epoch: float | None = None) -> list[Candidate]:
        sm = self._streams.get(channel)
        if sm is None:
            sm = self._streams[channel] = StreamMatcher(self.index)
        elif sm.index is not self.index:                         # registry was swapped: adopt on the feeding thread
            sm.pending_index = self.index
        return [self._gate(channel, c) for c in sm.feed(samples, epoch)]

    def end_stream(self, channel: str) -> list[Candidate]:
        sm = self._streams.pop(channel, None)
        out = [self._gate(channel, c) for c in (sm.flush() if sm else [])]
        if sm:
            self._lookup_ms.extend(sm.lookup_ms[:-1])               # the last step is the end-of-stream flush
        return out

    def in_reference_airing(self, channel: str, epoch: float) -> bool:
        """True while `epoch` lies inside a logged reference airing: nothing heavy (STT/OCR) is needed for it."""
        return any(a - 0.5 <= epoch <= b + 0.5 for a, b in self._airings.get(channel, ()))

    # -- hybrid safety gate
    def _packshot_distance(self, channel: str, c: Candidate, meta) -> int | None:
        """Hamming distance of the on-air packshot to the stored one; None when no frame / no reference could be had."""
        frame = None
        if self.frame_grabber is not None and meta.phash:
            try:
                frame = self.frame_grabber(channel, c.packshot_at, c.asset_id)
            except Exception as exc:                           # a broken grabber must not break matching
                log.warning("[%s] frame grabber failed: %s", channel, exc)
        if frame is None or not meta.phash:
            return None
        return hamming_hex(frame if isinstance(frame, str) else phash_image(_as_image(frame)), meta.phash)

    def _gate(self, channel: str, c: Candidate) -> Candidate:
        meta = next(m for m in self.index.meta if m.asset_id == c.asset_id)
        c.music_dominant = meta.music_dominant
        if c.confidence >= self.accept and not meta.music_dominant:
            c.decision = "confirmed"
            if c.no_dip:                                       # cut-down without a trailing dip: only the packshot can confirm
                d = self._packshot_distance(channel, c, meta)
                c.visual_distance = d
                c.status = "CONFIRMED" if d is not None and d <= VISUAL_MAX_BITS else UNCONFIRMED_CUTDOWN
        else:
            limit = MUSIC_VISUAL_MAX_BITS if meta.music_dominant else VISUAL_MAX_BITS
            c.visual_distance = self._packshot_distance(channel, c, meta)
            if c.visual_distance is not None:
                c.decision = "confirmed_visual" if c.visual_distance <= limit else "rejected_visual"
            elif c.confidence >= self.accept:                  # music-dominant, acoustic match is exact but unverifiable:
                c.decision, c.status = "confirmed", UNCONFIRMED_MUSIC   # keep the evidence, never claim it
            else:
                c.decision = "needs_visual"
        if c.decision in ("confirmed", "confirmed_visual"):
            self._log(channel, c)
        key = {"confirmed": "confirmed", "confirmed_visual": "visual", "rejected_visual": "rejected_visual",
               "needs_visual": "needs_visual", "duplicate": "duplicates", "unlinked": "unlinked"}[c.decision]
        self.stats[key] += 1
        return c

    def _log(self, channel: str, c: Candidate) -> None:
        ad_id = self._ad_id(c.asset_id)
        if ad_id is None:
            c.decision = "unlinked"
            return
        c.ad_id = ad_id
        source = f"live:{channel}"
        ts_cet = algeria_iso(c.airing_start)
        vhash = compute_verification_hash(ad_id, source, ts_cet, c.airing_end, c.nominal_duration, c.confidence, None)
        c.occurrence_id = rdb.commit_airing(
            self.db_path, asset_id=c.asset_id, ad_id=ad_id, source_name=source, airing_start=c.airing_start,
            airing_end=c.airing_end, duration=c.nominal_duration, confidence=c.confidence,
            algeria_timestamp=ts_cet, verification_hash=vhash,
            detected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), status=c.status,
            **clock_sync.occurrence_fields(channel),
            **time_synchronizer.occurrence_fields(channel, c.airing_start, c.airing_end))
        if c.occurrence_id is None:
            c.decision = "duplicate"
            return
        if c.family_note:
            log.info("[FAMILY] %s %s: %s", channel, c.asset_id, c.family_note)
        if c.status != "CONFIRMED":
            self.stats["unconfirmed_cutdown"] += 1
            log.info("[%s] %s %s logged as %s: no packshot match to confirm it - excluded from penalty claims",
                     channel, c.asset_id, c.cut_type, c.status)
        self._airings.setdefault(channel, []).append((c.airing_start, c.airing_end))
        self._airings[channel] = self._airings[channel][-50:]
        if self.compliance:
            self._compliance_dirty[channel] = c.airing_start
            self.run_compliance()

    def run_compliance(self, force: bool = False) -> int:
        """Pod positions + DZD breach alerts for channels with new airings (rate-limited; force ignores the limit)."""
        if not self._compliance_dirty or (not force and time.monotonic() - self._last_compliance
                                          < self.compliance_min_interval):
            return 0
        from kihc.analytics.compliance_audit import refresh_alerts
        from kihc.core.pod_analyzer import materialize_pods
        dirty, self._compliance_dirty = self._compliance_dirty, {}
        self._last_compliance = time.monotonic()
        for channel, t in dirty.items():
            materialize_pods(channel, (t - 3600, t + 3600), db_path=self.db_path)
        return refresh_alerts(self.db_path, lookback_days=1)


def pcm_from_wav(path) -> np.ndarray:
    """Mono float32 at 11025 Hz from a 16-bit WAV written by the discovery engine."""
    import wave
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono WAV")
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if rate != SR:
        pcm = np.interp(np.arange(0, pcm.size, rate / SR), np.arange(pcm.size), pcm).astype(np.float32)
    return pcm


@dataclass
class RegistryReport:
    built: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    confusable: list = field(default_factory=list)
    index_mb: float = 0.0


def build_registry(db_path=None, *, dry_run: bool = False, min_hashes: int = 60) -> RegistryReport:
    """Build reference_signatures from the confirmed (classified + promoted) discovered assets.

    Only clean assets: status 'classified', a promoted reference ad, cached audio on disk and enough peaks.
    The clip is cut to the nominal duration (drops the post-roll) then stripped of leading/trailing silence.
    Pairs of assets that share audio are reported (confusable): a cut-down inside a longer spot, or the same
    creative indexed twice. The matcher arbitrates them at run time (one airing -> one row), the report tells
    the reviewer which assets are involved.
    """
    from pathlib import Path
    from kihc.database.schema import get_db
    rep = RegistryReport()
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT a.asset_id, a.audio_path, a.packshot_path, a.nominal_duration_sec, a.duration_sec, "
            "r.brand_name, r.industry FROM discovered_assets a JOIN reference_ads r ON r.ad_id = a.promoted_ad_id "
            "WHERE a.status = 'classified' AND a.promoted_ad_id IS NOT NULL ORDER BY a.asset_id").fetchall()
    sigs = []
    for r in rows:
        if not r["audio_path"] or not Path(r["audio_path"]).exists():
            rep.skipped.append((r["asset_id"], "no cached audio"))
            continue
        nominal = float(r["nominal_duration_sec"] or r["duration_sec"])
        x = pcm_from_wav(r["audio_path"])[: int(round(nominal * SR))]
        codes, t_cs, lead, trimmed = signature_from_audio(x)
        if codes.size < min_hashes:
            rep.skipped.append((r["asset_id"], f"only {codes.size} hashes (too sparse/short to be exact)"))
            continue
        phash = None
        if r["packshot_path"] and Path(r["packshot_path"]).exists():
            from PIL import Image
            with Image.open(r["packshot_path"]) as im:
                phash = phash_image(im)
        vr = compute_voice_ratio(x)
        sigs.append(rdb.Signature(r["asset_id"], r["brand_name"], r["industry"], nominal, codes, t_cs, lead,
                                  trimmed, phash, voice_ratio=vr, is_music_dominant=is_music_dominant(vr)))
        rep.built.append(dict(asset_id=r["asset_id"], brand=r["brand_name"], sector=r["industry"], nominal=nominal,
                              trimmed=round(trimmed, 2), lead_trim=round(lead, 2), hashes=int(codes.size),
                              phash=phash, voice_ratio=vr, music_dominant=is_music_dominant(vr)))
    rep.confusable = _confusable(sigs)
    rep.index_mb = round(ReferenceIndex(sigs).memory_bytes() / 2 ** 20, 3)
    if not dry_run and sigs:
        with get_db(db_path) as conn:                  # one transaction: all signatures or none
            rdb.ensure_schema(conn)
            for s in sigs:
                rdb.upsert_signature(conn, s.asset_id, s.brand_name, s.canonical_sector, s.nominal_duration,
                                     rdb.pack_blob(s.codes, s.t_cs, s.lead_trim_s, s.trimmed_s), s.visual_p_hash,
                                     voice_ratio=s.voice_ratio, music_dominant=s.is_music_dominant)
    return rep


def _confusable(sigs: list) -> list:
    """(asset, other asset, confidence, kind, offset) for every pair that shares audio - see ReferenceIndex.relations."""
    idx = ReferenceIndex(sigs)
    return [(idx.meta[r["x"]].asset_id, idx.meta[r["y"]].asset_id, r["conf"], r["kind"], r["x_in_y"])
            for r in idx.relations()]
