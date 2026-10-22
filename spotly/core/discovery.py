"""Unsupervised discovery of recurring audio segments (candidate commercials) - no reference files.

How it works (all on the live worker's 32-bit sub-fingerprints, ~86 frames/s):
  1. History: a bounded ring of recent sub-fingerprints + capture times (default 4 h).
  2. Probe: every 5 s of new audio is compared against every history offset at least
     `min_gap_seconds` older. Exact fingerprint hashes don't survive a broadcast codec chain
     (measured: 0 % exact 32-bit matches between two airings), so the search is correlation-based:
     a 1-in-8 subsample of the block (54 frames, 1728 bits) is scored at every offset in batches,
     then the best candidates are verified on the full block. Random alignments score
     0.500 +/- 0.012; repeats score well above.
  3. Track: verified alignments (delta = time between the two airings) are followed across blocks.
     When a repeat ends, its boundaries are refined at 0.5 s resolution and segments of
     `min_seconds`..`max_seconds` are kept (shorter: idents/stings; longer: programme reruns).
  4. Catalogue: each segment is matched against known assets (same creative, possibly with slightly
     different measured boundaries); otherwise a new asset ELHEDDAF_AD_### is created, its audio
     cached as WAV and both airings recorded. Later airings add occurrences.

What it can and cannot know (also surfaced in the UI):
  * A recurring segment is not necessarily an ad: promos, idents, programme openers and theme music
    recur too. Assets start 'unclassified' and only enter spend analytics after a human classifies
    them (promote_asset -> reference_ads, after which the regular matcher tracks them).
  * Two ads that always air back-to-back in the same order are one asset until they air apart.
  * Segments overlapping an airing the regular matcher already detected are skipped (already known).
"""
import io
import json
import logging
import os
import re
import threading
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from spotly.config import DISCOVERED_DIR, HOP_SIZE, SAMPLE_RATE
from spotly.core.fingerprint import from_blob
from spotly.core.proof import algeria_iso, compute_verification_hash
from spotly.database.schema import get_db

log = logging.getLogger("spotly.discovery")

FPS = SAMPLE_RATE / HOP_SIZE                   # sub-fingerprints per second (~86.1)
BITS = 32
_BATCH = 65536

SPOT_BRACKETS = (15, 20, 30, 45, 60)           # standard commercial lengths (seconds)
SNAP_EXTEND_MAX = 3.0                          # measured may fall short of a bracket by up to 3 s (quiet tail)
SNAP_TRIM_MAX = 1.0                            # ... or overshoot it by up to 1 s (boundary step)
POST_ROLL_SEC = 3.5                            # extra audio kept after the spot in the cached review clip
TAIL_EXTEND_MAX_SEC = 2.5                      # fade-aware extension of the matched tail
FADE_DROP_DB = 3.0                             # a fade step is quieter than the spot's median level ...
FADE_MIN_DECAY_DB = 1.0                        # ... and keeps decaying, by at least this much per 0.5 s step
FADE_STEP_DB = 10.0                            # ... but never this much in one step (a cliff = gap after the spot)
FADE_MIN_STEPS = 2                             # >= 1 s of continuous decay; a quieter plateau is programme, not a fade
SILENCE_DBFS = -55.0                           # below this the tail is silence/gap, not part of the spot


BUMPER_MAX_SEC = 10.0                          # stings / bumpers are legitimately short and irregular
STANDARD_STEP_SEC = 5.0                        # broadcast spots are sold in 5 s steps ...
STANDARD_TOL_SEC = 1.5                         # ... and a measured length is within this of one


def commercial_duration_ok(duration: float) -> bool:
    """Commercial duration enforcement for NEW candidates: a bumper (<= 10 s) or a length within 1.5 s of a multiple
    of 5 s (so 35, 40, 55 s spots pass, not only the snap brackets). Interview and programme excerpts have arbitrary
    lengths; verified assets are never re-judged by this."""
    if duration <= BUMPER_MAX_SEC:
        return True
    nearest = round(duration / STANDARD_STEP_SEC) * STANDARD_STEP_SEC
    return abs(duration - nearest) <= STANDARD_TOL_SEC


def snap_duration(measured: float, brackets=SPOT_BRACKETS) -> tuple[float, str]:
    """Nominal spot length: the closest standard bracket if `measured` is within -1 s / +3 s of it."""
    near = [b for b in brackets if -SNAP_TRIM_MAX <= b - measured <= SNAP_EXTEND_MAX]
    if near:
        b = min(near, key=lambda x: abs(x - measured))
        return float(b), f"snapped to {b} s (measured {measured:.1f} s)"
    return round(float(measured), 2), "measured"


SHARE_WIN_SEC = 3.0                            # window used to compare two assets' audio
SHARE_MATCH = 0.75                             # window agreement that counts as "same audio" (chance ~0.53)
MERGE_SIMILARITY = 0.90                        # best window must be near-identical ...
MERGE_COVERAGE_SAME_AIRING = 0.60              # ... and >= 60 % shared when their airings overlap in time
MERGE_COVERAGE_SAME_CREATIVE = 0.90            # ... or >= 90 % shared + same length when they aired apart
RELATED_SIMILARITY = 0.80                      # partial sharing (jingle, back-to-back pair): flagged only


def shared_audio(a: np.ndarray, b: np.ndarray) -> dict:
    """How much of the shorter fingerprint occurs in the other at one consistent offset (silence ignored).

    -> similarity (best 3 s window agreement), coverage (share of the shorter's informative windows found at
    the dominant offset), offset_sec (start of `a` relative to start of `b`; meaningful when coverage > 0).
    Duration alone is not evidence: on 2026-10-13 pairs with identical measured lengths (17.97 s, 38.94 s)
    scored 0.53, i.e. unrelated audio; lengths cluster because boundaries are refined on a 0.5 s grid.
    """
    swap = a.size > b.size
    s, l = (b, a) if swap else (a, b)
    win = int(SHARE_WIN_SEC * FPS)
    if s.size < win or l.size < win:
        return dict(similarity=0.0, coverage=0.0, offset_sec=0.0)
    lw = np.lib.stride_tricks.sliding_window_view(l, win)
    l_ok = np.count_nonzero(lw, axis=1) >= 0.9 * win
    best, offsets, n_inf = 0.0, [], 0
    for st in range(0, s.size - win + 1, int(FPS)):
        core = s[st:st + win]
        if np.count_nonzero(core) < 0.9 * win:
            continue                                            # silent/zero window: not evidence
        n_inf += 1
        conf = 1.0 - np.bitwise_count(lw ^ core).sum(axis=1) / (win * BITS)
        conf[~l_ok] = 0.0
        i = int(conf.argmax())
        best = max(best, float(conf[i]))
        if conf[i] >= SHARE_MATCH:
            offsets.append(i - st)                              # where s[0] sits inside l
    if not offsets or not n_inf:
        return dict(similarity=round(best, 3), coverage=0.0, offset_sec=0.0)
    vals, counts = np.unique(np.round(np.array(offsets) / 4).astype(int), return_counts=True)
    mode = vals[counts.argmax()] * 4
    agree = sum(1 for o in offsets if abs(o - mode) <= 4)
    off = float(np.median([o for o in offsets if abs(o - mode) <= 4])) / FPS   # s start rel. l start
    # offset of a relative to b: if a is the shorter (s) it is +off; otherwise b sits at +off inside a
    return dict(similarity=round(best, 3), coverage=round(agree / n_inf, 3), offset_sec=round(off if not swap else -off, 2))


def _popcount_conf(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.bitwise_count(a ^ b).sum()) / (a.size * BITS)


def asset_prefix(channel: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "", channel.upper().replace("TV", ""))
    return f"{slug or 'CHANNEL'}_AD"


class DiscoveryEngine:
    def __init__(self, channel: str, db_path=None, *, prefix: str | None = None, audio_dir=DISCOVERED_DIR,
                 history_hours: float = 4.0, min_seconds: float = 15.0, max_seconds: float = 60.0,
                 min_gap_seconds: float = 180.0, probe_seconds: float = 5.0, coarse_conf: float = 0.57,
                 verify_conf: float = 0.58, edge_conf: float = 0.56, asset_match_conf: float = 0.62,
                 max_assets: int = 500, pcm_seconds: float = 100.0):
        self.channel = channel
        self.source_name = f"live:{channel}"
        self.db_path = db_path
        self.prefix = prefix or asset_prefix(channel)
        self.audio_dir = Path(audio_dir)
        self.min_frames = int(min_seconds * FPS)
        self.max_frames = int(max_seconds * FPS)
        self.min_gap = int(min_gap_seconds * FPS)
        self.block = int(probe_seconds * FPS)
        self.sub = max(8, self.block // 54)                      # coarse subsample stride (~54 frames per block)
        self.coarse_conf, self.verify_conf, self.edge_conf = coarse_conf, verify_conf, edge_conf
        self.asset_match_conf, self.max_assets = asset_match_conf, max_assets
        self.cap = int(history_hours * 3600 * FPS)
        self._fp = np.zeros(0, np.uint32)
        self._t = np.zeros(0, np.float64)
        self._base = None                                        # global frame index of _fp[0]
        self._next_probe = None
        self._runs: list[dict] = []
        self._pcm = deque()                                      # (global sample start, int16 array)
        self._pending_audio: list[dict] = []                     # clips waiting for their post-roll
        self._pcm_cap = int(pcm_seconds * SAMPLE_RATE)
        self._lock = threading.Lock()
        self.counters = dict(frames=0, probes=0, repeats_found=0, rejected_length=0, skipped_known=0,
                             assets_created=0, occurrences_recorded=0)
        self.last_discovery_at = None
        # on_airing(asset_id, later_airing_fp, end_epoch): called (under the engine lock: must only enqueue)
        # when an unclassified asset has just aired - used for live packshot capture.
        self.on_airing = None
        # Survive worker restarts (Phase 11.4): a new worker's frame indices restart at 0, so they are shifted
        # past the history; feeds from a replaced worker are ignored.
        self._source = None
        self._index_offset = 0
        self._remap_pending = False
        self.assets = self._load_assets()

    # ------------------------------------------------------------ continuity across restarts
    def begin_new_source(self, source) -> None:
        """A new worker (new ffmpeg, indices from 0) will feed this engine from now on."""
        with self._lock:
            self._finish_all()
            if self._pending_audio:
                self._flush_pending_audio(force=True)
            self._pcm.clear()                                    # the PCM ring must stay contiguous
            self._source, self._remap_pending = source, self._base is not None

    def save_state(self, path) -> bool:
        """Persist the fingerprint history (no audio) so a pilot restart keeps its look-back window."""
        import time as _time
        with self._lock:
            if self._fp.size == 0:
                return False
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(path) + ".tmp.npz")
            np.savez(tmp, fp=self._fp, t=self._t, channel=np.array(self.channel), saved_at=np.array(_time.time()))
            os.replace(tmp, path)
            return True

    def load_state(self, path, max_age_s: float = 1800.0) -> float:
        """Restore a saved history if it is for this channel and recent. -> seconds restored (0 if none)."""
        import time as _time
        p = Path(path)
        if not p.is_file():
            return 0.0
        try:
            with np.load(p) as d:
                if str(d["channel"]) != self.channel or _time.time() - float(d["saved_at"]) > max_age_s:
                    return 0.0
                fp, t = d["fp"].astype(np.uint32), d["t"].astype(np.float64)
        except Exception as exc:
            log.warning("discovery history not restored (%s)", exc)
            return 0.0
        with self._lock:
            n = min(fp.size, self.cap)
            self._fp, self._t = fp[-n:].copy(), t[-n:].copy()
            self._base, self._next_probe = 0, n
            self._remap_pending = True
        return n / FPS

    def _remap(self, first_index: int) -> int:
        if self._remap_pending:
            end = self._base + self._fp.size
            self._index_offset = end + 1 - first_index           # one-frame gap: never bridge two streams
            self._next_probe = max(self._next_probe, end + 1)
            self._remap_pending = False
        return first_index + self._index_offset

    # ------------------------------------------------------------ state
    def _load_assets(self) -> list[dict]:
        with get_db(self.db_path) as conn:
            rows = conn.execute("SELECT asset_id, duration_sec, fingerprint, status FROM discovered_assets "
                                "WHERE channel = ? ORDER BY asset_id", (self.channel,)).fetchall()
        return [dict(asset_id=r["asset_id"], duration=r["duration_sec"],
                     fp=np.frombuffer(r["fingerprint"], dtype="<u4").astype(np.uint32), status=r["status"])
                for r in rows]

    def stats(self) -> dict:
        hist_s = self._fp.size / FPS
        out = dict(self.counters, history_seconds=round(hist_s, 1), active_repeats=len(self._runs),
                   assets=len(self.assets), unclassified=sum(a["status"] == "unclassified" for a in self.assets),
                   last_discovery_at=self.last_discovery_at)
        if getattr(self, "packshot_stats", None):
            out["packshots"] = self.packshot_stats()
        return out

    def recent_fingerprint(self, seconds: float = 4.0):
        """(newest `seconds` of sub-fingerprints, capture epoch at their end) or None - for feed calibration."""
        with self._lock:
            n = int(seconds * FPS)
            if self._fp.size < n or np.isnan(self._t[-1]):
                return None
            return self._fp[-n:].copy(), float(self._t[-1]) + 1.0 / FPS

    # ------------------------------------------------------------ input
    def feed(self, fp: np.ndarray, first_index: int, epochs: np.ndarray, pcm: np.ndarray | None = None,
             pcm_start: int | None = None, source=None):
        """Append sub-fingerprints fp (global frame indices first_index..) with capture epochs, and the
        matching 11025 Hz int16 audio starting at global sample pcm_start. Overlap is ignored."""
        with self._lock:
            if source is not None and self._source is not None and source is not self._source:
                return                                             # a replaced worker still flushing
            first_index = self._remap(first_index)
            if pcm_start is not None:
                pcm_start += self._index_offset * HOP_SIZE
            if self._base is None:
                self._base, self._next_probe = first_index, first_index
            end = self._base + self._fp.size
            if first_index > end:                                  # small gap (reconnect): pad, break runs
                pad = first_index - end
                self._fp = np.concatenate([self._fp, np.zeros(pad, np.uint32)])
                self._t = np.concatenate([self._t, np.full(pad, np.nan)])
                self._finish_all()
                end = first_index
            skip = end - first_index
            if skip < fp.size:
                self._fp = np.concatenate([self._fp, fp[skip:].astype(np.uint32)])
                self._t = np.concatenate([self._t, np.asarray(epochs[skip:], np.float64)])
                self.counters["frames"] += fp.size - skip
            if pcm is not None and pcm_start is not None:
                self._add_pcm(pcm, pcm_start)
            self._compact()
            while self._next_probe + self.block <= self._base + self._fp.size:
                self._probe(self._next_probe)
                self._next_probe += self.block
            if self._pending_audio:
                self._flush_pending_audio()

    def flush(self):
        """End of stream: finalize repeats still being tracked, write pending clips with what audio exists."""
        with self._lock:
            self._finish_all()
            if self._pending_audio:
                self._flush_pending_audio(force=True)

    def _add_pcm(self, pcm, start):
        have = self._pcm[-1][0] + self._pcm[-1][1].size if self._pcm else start
        if start + pcm.size <= have:
            return
        cut = max(0, have - start)
        self._pcm.append((start + cut, np.asarray(pcm[cut:], np.int16).copy()))
        total = sum(a.size for _, a in self._pcm)
        while self._pcm and total - self._pcm[0][1].size >= self._pcm_cap:
            total -= self._pcm.popleft()[1].size

    def _compact(self):
        if self._fp.size > self.cap * 1.1:
            drop = self._fp.size - self.cap
            self._fp, self._t = self._fp[drop:].copy(), self._t[drop:].copy()
            self._base += drop
            for r in self._runs:                                   # runs referencing dropped history end
                if r["start"] - r["delta"] < self._base:
                    r["dead"] = True

    # ------------------------------------------------------------ search
    def _probe(self, g0: int):
        self.counters["probes"] += 1
        i0 = g0 - self._base
        block = self._fp[i0:i0 + self.block]
        hist_end = i0 - self.min_gap                               # only history at least min_gap older
        matched_deltas = []
        if hist_end > self.block and np.count_nonzero(block):
            probe_sub = block[::self.sub]
            n_offsets = hist_end - self.block + 1
            # strided view (no copy): row o = history[o], history[o + sub], ... aligned with probe_sub
            windows = np.lib.stride_tricks.sliding_window_view(self._fp[:hist_end], self.block)[:, ::self.sub]
            best = []
            for b0 in range(0, n_offsets, _BATCH):
                w = windows[b0:b0 + _BATCH]
                conf = 1.0 - np.bitwise_count(w ^ probe_sub).sum(axis=1) / (probe_sub.size * BITS)
                hits = np.flatnonzero(conf >= self.coarse_conf)
                if hits.size:
                    best.extend(zip(conf[hits].tolist(), (hits + b0).tolist()))
            best.sort(reverse=True)
            taken = []
            for c, off in best:
                if len(taken) >= 6:
                    break
                if any(abs(off - t) < 20 for t in taken):
                    continue
                taken.append(off)
                if _popcount_conf(block, self._fp[off:off + self.block]) >= self.verify_conf:
                    matched_deltas.append(i0 - off)
        alive = []
        for d in matched_deltas:
            run = next((r for r in self._runs if abs(r["delta"] - d) <= 3 and not r.get("seen")), None)
            if run:
                run["end"], run["seen"] = g0 + self.block, True
            else:
                self._runs.append(dict(delta=d, start=g0, end=g0 + self.block, seen=True))
        for r in self._runs:
            if r.pop("seen", False) and not r.get("dead"):
                alive.append(r)
            else:
                self._finish(r)
        self._runs = alive

    def _finish_all(self):
        for r in self._runs:
            self._finish(r)
        self._runs = []

    def _finish(self, run):
        if run.get("dead"):
            return
        self.counters["repeats_found"] += 1
        d = run["delta"]
        lo = max(run["start"] - self.block, self._base + d)
        hi = min(run["end"] + self.block, self._base + self._fp.size)
        step = max(1, int(0.5 * FPS))
        ks = list(range(lo, hi - step + 1, step))
        if not ks:
            return
        good = []
        for k in ks:
            a_blk = self._fp[k - self._base:k - self._base + step]
            b_blk = self._fp[k - d - self._base:k - d - self._base + step]
            # all-zero frames (digital silence between spots) "match" trivially: they are not evidence
            informative = np.count_nonzero(a_blk) >= step // 2 and np.count_nonzero(b_blk) >= step // 2
            good.append(informative and _popcount_conf(a_blk, b_blk) >= self.edge_conf)
        # Every maximal good stretch (one-step dips tolerated) is a candidate segment: a repeat region can
        # hold several spots that aired in the same order twice, split by uninformative silence.
        i = 0
        while i < len(ks):
            if not good[i]:
                i += 1
                continue
            a = b = i
            while b + 1 < len(ks) and (good[b + 1] or (b + 2 < len(ks) and good[b + 2])):
                b += 1 if good[b + 1] else 2
            i = b + 1
            s, e = ks[a], ks[b] + step                              # global frames of the later airing
            e = self._extend_fade_tail(s, e, step)
            frames = e - s
            if not self.min_frames <= frames <= self.max_frames:
                self.counters["rejected_length"] += 1
                continue
            seg = self._fp[s - self._base:e - self._base].copy()
            t_b, t_a = self._t[s - self._base], self._t[s - d - self._base]
            if np.isnan(t_b) or np.isnan(t_a):
                continue
            conf = _popcount_conf(seg, self._fp[s - d - self._base:e - d - self._base])
            self._record(seg, frames / FPS, conf, [(float(t_a), None), (float(t_b), (s, e))])

    # ------------------------------------------------------------ boundary hardening
    def _pcm_slice(self, s_sample: int, e_sample: int):
        """int16 audio for global samples [s_sample, e_sample) from the PCM ring, or None if not (fully) held."""
        if not self._pcm or s_sample < self._pcm[0][0]:
            return None
        end = self._pcm[-1][0] + self._pcm[-1][1].size
        if e_sample > end:
            return None
        buf = np.concatenate([a for _, a in self._pcm])
        base = self._pcm[0][0]
        return buf[s_sample - base:e_sample - base]

    @staticmethod
    def _rms_db(x) -> float:
        if x is None or x.size == 0:
            return -120.0
        return float(10 * np.log10(max(np.mean((x.astype(np.float64) / 32768.0) ** 2), 1e-12)))

    def _extend_fade_tail(self, s: int, e: int, step: int) -> int:
        """Extend the end through a quiet-but-audible fade (packshot/voice-over tail) of the later airing.

        Fingerprint agreement drops in quiet tails (bits become noise-dominated), so the matched boundary
        often closes early. Up to TAIL_EXTEND_MAX_SEC is added while each 0.5 s step is below the spot's
        median level, above SILENCE_DBFS, and 1-10 dB quieter than the step before. Fewer than
        FADE_MIN_STEPS such steps is not a fade: a quieter plateau after the spot is programme or the next
        spot, so the end is left where matching put it.
        """
        body = self._pcm_slice(s * HOP_SIZE, e * HOP_SIZE)
        if body is None or body.size < SAMPLE_RATE:
            return e
        win = step * HOP_SIZE
        levels = [self._rms_db(body[i:i + win]) for i in range(0, body.size - win + 1, win)]
        spot_db = float(np.median(levels))
        prev = levels[-1]
        limit = min(e + int(TAIL_EXTEND_MAX_SEC * FPS), self._base + self._fp.size)
        k = e
        while k + step <= limit:
            lvl = self._rms_db(self._pcm_slice(k * HOP_SIZE, (k + step) * HOP_SIZE))
            if (lvl <= SILENCE_DBFS or lvl > spot_db - FADE_DROP_DB
                    or not prev - FADE_STEP_DB <= lvl <= prev - FADE_MIN_DECAY_DB):
                break
            prev = lvl
            k += step
        return k if k - e >= FADE_MIN_STEPS * step else e

    # ------------------------------------------------------------ catalogue
    def _match_asset(self, seg: np.ndarray, duration: float):
        best, best_conf = None, 0.0
        for a in self.assets:
            if abs(a["duration"] - duration) > max(2.0, 0.15 * duration):
                continue
            short, long_ = (seg, a["fp"]) if seg.size <= a["fp"].size else (a["fp"], seg)
            n = int(short.size * 0.8)
            if n < 8:
                continue
            core = short[(short.size - n) // 2:(short.size - n) // 2 + n]
            w = np.lib.stride_tricks.sliding_window_view(long_, n)
            c = float((1.0 - np.bitwise_count(w ^ core).sum(axis=1) / (n * BITS)).max())
            if c > best_conf:
                best, best_conf = a, c
        return (best, best_conf) if best_conf >= self.asset_match_conf else (None, best_conf)

    def _known_airing(self, conn, start: float, end: float) -> bool:
        return conn.execute("SELECT 1 FROM ad_occurrences WHERE source_name = ? AND human_verified != -1 "
                            "AND airing_start < ? AND airing_end > ? LIMIT 1",
                            (self.source_name, end - 2.0, start + 2.0)).fetchone() is not None

    def _current(self, conn, asset):
        """Asset with its *current* status (a reviewer may have classified/ignored/merged it while we run);
        a merged asset resolves to its master."""
        for _ in range(5):                                          # bounded: merge chains are short
            row = conn.execute("SELECT status, merged_into FROM discovered_assets WHERE asset_id = ?",
                               (asset["asset_id"],)).fetchone()
            if row is None:
                return None
            asset["status"] = row["status"]
            if row["status"] != "merged" or not row["merged_into"]:
                return asset
            master = next((a for a in self.assets if a["asset_id"] == row["merged_into"]), None)
            if master is None:
                r = conn.execute("SELECT asset_id, duration_sec, fingerprint, status FROM discovered_assets "
                                 "WHERE asset_id = ?", (row["merged_into"],)).fetchone()
                if r is None:
                    return None
                master = dict(asset_id=r["asset_id"], duration=r["duration_sec"], status=r["status"],
                              fp=np.frombuffer(r["fingerprint"], dtype="<u4").astype(np.uint32))
                self.assets.append(master)
            asset = master
        return asset

    def _same_airing_asset(self, conn, seg, start, end):
        """An unclassified asset that already holds an airing overlapping [start, end) with the same audio:
        the same physical airing segmented differently (the cause of the 2026-10-13 duplicates)."""
        rows = conn.execute("SELECT DISTINCT o.asset_id FROM discovered_occurrences o JOIN discovered_assets a "
                            "USING (asset_id) WHERE o.channel = ? AND a.status = 'unclassified' "
                            "AND o.airing_start < ? AND o.airing_end > ?", (self.channel, end, start)).fetchall()
        best, best_key = None, None
        for r in rows:
            a = next((x for x in self.assets if x["asset_id"] == r["asset_id"]), None)
            if a is None:
                continue
            sh = shared_audio(seg, a["fp"])
            if sh["similarity"] >= MERGE_SIMILARITY and sh["coverage"] >= MERGE_COVERAGE_SAME_AIRING:
                key = (sh["coverage"], a["duration"])                # most complete, then longest segmentation
                if best_key is None or key > best_key:
                    best, best_key = a, key
        return best

    def _record(self, seg, duration, conf, airings):
        asset, _ = self._match_asset(seg, duration)
        with get_db(self.db_path) as conn:
            if asset is not None:
                asset = self._current(conn, asset)
            if asset is not None and asset["status"] != "unclassified":
                return                                              # classified -> matcher; ignored -> skip
            if any(self._known_airing(conn, t, t + duration) for t, _ in airings):
                self.counters["skipped_known"] += 1
                return
            if asset is None:
                t_last = airings[-1][0]
                asset = self._same_airing_asset(conn, seg, t_last, t_last + duration)
                if asset is not None:
                    self.counters["same_airing_resegmented"] = self.counters.get("same_airing_resegmented", 0) + 1
            if asset is None:
                if len(self.assets) >= self.max_assets:
                    return
                if not commercial_duration_ok(duration):
                    self.counters["rejected_duration"] = self.counters.get("rejected_duration", 0) + 1
                    log.info("[%s] rejected candidate: %.1f s is not a commercial length", self.channel, duration)
                    return
                # COUNT(*)+1 collides once rows are removed/renamed (ids stop being contiguous) and crash-looped the worker,
                # so take the highest numeric suffix in use and probe until the id is free.
                used = {r[0] for r in conn.execute("SELECT asset_id FROM discovered_assets WHERE asset_id LIKE ?",
                                                   (f"{self.prefix}_%",))}
                n = max((int(s.rsplit("_", 1)[1]) for s in used if s.rsplit("_", 1)[1].isdigit()), default=0)
                n += 1
                while f"{self.prefix}_{n:03d}" in used:
                    n += 1
                asset_id = f"{self.prefix}_{n:03d}"
                nominal, basis = snap_duration(duration)
                conn.execute("INSERT INTO discovered_assets (asset_id, channel, duration_sec, fingerprint, "
                             "first_seen, last_seen, nominal_duration_sec, duration_basis) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (asset_id, self.channel, round(duration, 2), seg.astype("<u4").tobytes(),
                              airings[0][0], airings[-1][0], nominal, basis))
                # The clip is written once POST_ROLL_SEC of audio after the spot has arrived (see feed()).
                self._pending_audio.append(dict(asset_id=asset_id, frames=airings[-1][1], seg=seg))
                asset = dict(asset_id=asset_id, duration=round(duration, 2), fp=seg, status="unclassified")
                self.assets.append(asset)
                self.counters["assets_created"] += 1
                log.info("[%s] discovered %s (%.1f s, match %.2f)", self.channel, asset_id, duration, conf)
            for t, _ in airings:
                # same airing already logged for this asset (start within 2 s, or a re-segmentation that
                # overlaps it by more than half): never count one airing twice
                dup = conn.execute("SELECT 1 FROM discovered_occurrences WHERE asset_id = ? AND channel = ? "
                                   "AND (abs(airing_start - ?) <= 2.0 OR (min(airing_end, ?) - max(airing_start, ?)) "
                                   "> 0.5 * min(airing_end - airing_start, ?))",
                                   (asset["asset_id"], self.channel, t, t + duration, t, duration)).fetchone()
                if dup:
                    continue
                conn.execute("INSERT OR IGNORE INTO discovered_occurrences (asset_id, channel, airing_start, "
                             "airing_end, duration_sec, confidence, algeria_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (asset["asset_id"], self.channel, round(t, 3), round(t + duration, 3),
                              round(duration, 2), round(conf, 4), algeria_iso(t)))
                self.counters["occurrences_recorded"] += 1
            conn.execute("UPDATE discovered_assets SET occurrences = (SELECT COUNT(*) FROM discovered_occurrences "
                         "WHERE asset_id = ?), last_seen = max(coalesce(last_seen, 0), ?) WHERE asset_id = ?",
                         (asset["asset_id"], airings[-1][0], asset["asset_id"]))
        self.last_discovery_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.on_airing is not None:
            try:                                                    # just aired: its packshot may still be published
                self.on_airing(asset["asset_id"], seg, airings[-1][0] + duration)
            except Exception as exc:
                log.warning("on_airing hook failed: %s", exc)

    def _flush_pending_audio(self, force: bool = False):
        """Write review clips once their post-roll has arrived (or after ~20 s / at flush: whatever exists)."""
        keep = []
        heard_up_to = (self._base or 0) + self._fp.size                  # global frame index of newest audio
        for p in self._pending_audio:
            give_up = force or heard_up_to >= p["frames"][1] + int(20 * FPS)
            saved = self._save_audio(p["asset_id"], p["frames"], p["seg"], require_full_pad=not give_up)
            if saved is None and not give_up:
                keep.append(p)
                continue
            path, pad = saved if saved else (None, 0.0)
            with get_db(self.db_path) as conn:
                conn.execute("UPDATE discovered_assets SET audio_path = ?, audio_post_pad_sec = ? WHERE asset_id = ?",
                             (path, round(pad, 2), p["asset_id"]))
        self._pending_audio = keep

    def _save_audio(self, asset_id: str, frames, seg: np.ndarray, require_full_pad: bool = True):
        """Cache the later airing as an 11025 Hz mono 16-bit WAV with up to POST_ROLL_SEC of post-roll.

        -> (path, post_roll_seconds) or None when the audio isn't (yet) available.
        """
        if frames is None or not self._pcm:
            return None
        s_sample, e_sample = frames[0] * HOP_SIZE, frames[1] * HOP_SIZE
        ring_start = self._pcm[0][0]
        ring_end = self._pcm[-1][0] + self._pcm[-1][1].size
        if s_sample < ring_start or e_sample > ring_end:
            return None
        want_end = e_sample + int(POST_ROLL_SEC * SAMPLE_RATE)
        if require_full_pad and want_end > ring_end:
            return None
        clip = self._pcm_slice(s_sample, min(want_end, ring_end))
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        path = self.audio_dir / f"{asset_id}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(clip.astype("<i2").tobytes())
        (self.audio_dir / f"{asset_id}.fp").write_bytes(seg.astype("<u4").tobytes())   # acoustic fingerprint
        return str(path), (min(want_end, ring_end) - e_sample) / SAMPLE_RATE


def backfill_nominal_durations(db_path) -> list[tuple[str, float, str]]:
    """Set nominal_duration_sec / duration_basis for assets that predate snapping.

    Only these two asset columns are written: measured duration_sec, occurrences, timestamps and
    counts are left exactly as they are.
    """
    out = []
    with get_db(db_path) as conn:
        for r in conn.execute("SELECT asset_id, duration_sec FROM discovered_assets WHERE nominal_duration_sec IS NULL"):
            nominal, basis = snap_duration(r["duration_sec"])
            conn.execute("UPDATE discovered_assets SET nominal_duration_sec = ?, duration_basis = ? WHERE asset_id = ?",
                         (nominal, basis, r["asset_id"]))
            out.append((r["asset_id"], nominal, basis))
    return out


# ---------------------------------------------------------------- de-duplication

def _overlaps(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def _may_share(a: np.ndarray, b: np.ndarray, probes: int = 6) -> bool:
    """Cheap pre-check: does any of `probes` 3 s windows spread over the shorter reach SHARE_MATCH in the
    other? Unrelated pairs (the vast majority) are rejected at ~1/5 of the cost of shared_audio()."""
    s, l = (a, b) if a.size <= b.size else (b, a)
    win = int(SHARE_WIN_SEC * FPS)
    if s.size < win or l.size < win:
        return False
    lw = np.lib.stride_tricks.sliding_window_view(l, win)
    for st in np.linspace(0, s.size - win, probes).astype(int):
        core = s[st:st + win]
        if np.count_nonzero(core) < 0.9 * win:
            continue
        if (1.0 - np.bitwise_count(lw ^ core).sum(axis=1) / (win * BITS)).max() >= SHARE_MATCH:
            return True
    return False


def find_duplicate_assets(db_path, channel=None) -> list[dict]:
    """Pairs of unclassified assets that share audio, classified as:

    * same_airings   - >= 60 % of the shorter found in the other (near-identical windows) AND their airings
                       overlap in time: one physical airing catalogued twice with different boundaries. Merge.
    * same_creative  - >= 90 % shared and lengths within 2 s, aired at different times: the engine missed
                       the match. Merge.
    * related        - some near-identical audio but neither rule above (shared jingle/music bed, or one
                       asset is two spots aired back-to-back). Flagged for the reviewer, never merged:
                       merging different creatives would corrupt airing counts.
    Matching duration alone is never a reason (see shared_audio).
    """
    q = "SELECT asset_id, channel, duration_sec, fingerprint, occurrences FROM discovered_assets WHERE status = 'unclassified'"
    args = ()
    if channel:
        q, args = q + " AND channel = ?", (channel,)
    with get_db(db_path) as conn:
        assets = [dict(r) for r in conn.execute(q + " ORDER BY asset_id", args)]
        occ = {}
        for r in conn.execute("SELECT asset_id, airing_start, airing_end FROM discovered_occurrences"):
            occ.setdefault(r[0], []).append((r[1], r[2]))
    for a in assets:
        a["fp"] = np.frombuffer(a.pop("fingerprint"), dtype="<u4").astype(np.uint32)
    out = []
    for i, a in enumerate(assets):
        for b in assets[i + 1:]:
            if a["channel"] != b["channel"] or not _may_share(a["fp"], b["fp"]):
                continue
            sh = shared_audio(a["fp"], b["fp"])
            if sh["similarity"] < RELATED_SIMILARITY:
                continue
            overlapping = [(round(x0, 1), round(y0, 1)) for x0, x1 in occ.get(a["asset_id"], [])
                           for y0, y1 in occ.get(b["asset_id"], []) if _overlaps(x0, x1, y0, y1)]
            if sh["similarity"] >= MERGE_SIMILARITY and sh["coverage"] >= MERGE_COVERAGE_SAME_AIRING and overlapping:
                kind = "same_airings"
            elif (sh["coverage"] >= MERGE_COVERAGE_SAME_CREATIVE
                  and abs(a["duration_sec"] - b["duration_sec"]) <= 2.0):
                # two *different* airings of one creative agree ~0.8 (codec noise), so the evidence here is
                # near-complete coverage at one consistent offset plus the same length, not a 0.9 window
                kind = "same_creative"
            else:
                kind = "related"
            # master: the longer (more complete) segmentation; then more airings; then the older id
            m, d = sorted((a, b), key=lambda x: (-x["duration_sec"], -x["occurrences"], x["asset_id"]))
            # shared_audio(a, b)["offset_sec"] = a's start relative to b's start; we need dup relative to master
            off = sh["offset_sec"] if d is a else -sh["offset_sec"]
            out.append(dict(kind=kind, master=m["asset_id"], duplicate=d["asset_id"], similarity=sh["similarity"],
                            coverage=sh["coverage"], offset_sec=off, overlapping_airings=overlapping))
    return sorted(out, key=lambda r: (r["kind"] == "related", -r["coverage"], -r["similarity"]))


def merge_assets(db_path, master: str, duplicate: str, *, similarity: float, coverage: float, offset_sec: float,
                 reason: str) -> dict:
    """Fold `duplicate` into `master` so repeat airings aggregate under one asset id.

    Airings of the duplicate that overlap one of the master's are the same airing and are not re-counted;
    the others are added to the master (start shifted by the measured audio offset, master length).
    The duplicate keeps its row and its own occurrence rows (status 'merged', excluded from reports);
    everything is recorded in asset_merges so unmerge_asset() can restore it exactly.
    """
    with get_db(db_path) as conn:
        m = conn.execute("SELECT * FROM discovered_assets WHERE asset_id = ?", (master,)).fetchone()
        d = conn.execute("SELECT * FROM discovered_assets WHERE asset_id = ?", (duplicate,)).fetchone()
        if m is None or d is None:
            raise KeyError(master if m is None else duplicate)
        if m["status"] != "unclassified" or d["status"] != "unclassified":
            raise ValueError(f"only unclassified assets can be merged ({master}: {m['status']}, {duplicate}: {d['status']})")
        m_occ = [(r["airing_start"], r["airing_end"]) for r in
                 conn.execute("SELECT airing_start, airing_end FROM discovered_occurrences WHERE asset_id = ?", (master,))]
        same, copied = 0, []
        for o in conn.execute("SELECT * FROM discovered_occurrences WHERE asset_id = ? ORDER BY airing_start",
                              (duplicate,)).fetchall():
            if any(_overlaps(o["airing_start"], o["airing_end"], s, e) for s, e in m_occ):
                same += 1
                continue
            start = round(o["airing_start"] - offset_sec, 3)            # master starts offset_sec before the dup
            end = round(start + m["duration_sec"], 3)
            cur = conn.execute("INSERT OR IGNORE INTO discovered_occurrences (asset_id, channel, airing_start, airing_end, "
                               "duration_sec, confidence, algeria_timestamp, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (master, o["channel"], start, end, m["duration_sec"], o["confidence"], algeria_iso(start),
                                o["detected_at"]))
            if cur.rowcount:
                copied.append(cur.lastrowid)
                m_occ.append((start, end))
        conn.execute("UPDATE discovered_assets SET status = 'merged', merged_into = ?, label = ? WHERE asset_id = ?",
                     (master, f"duplicate of {master}", duplicate))
        _refresh_counts(conn, master)
        conn.execute("INSERT INTO asset_merges (master_id, duplicate_id, similarity, coverage, offset_sec, reason, "
                     "same_airings, copied_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (master, duplicate, similarity, coverage, offset_sec, reason, same, json.dumps(copied)))
        n = conn.execute("SELECT occurrences FROM discovered_assets WHERE asset_id = ?", (master,)).fetchone()[0]
    log.info("merged %s into %s (%s; %d same airing(s), %d added)", duplicate, master, reason, same, len(copied))
    return dict(master=master, duplicate=duplicate, same_airings=same, airings_added=len(copied), master_occurrences=n)


def _refresh_counts(conn, asset_id):
    conn.execute("UPDATE discovered_assets SET occurrences = (SELECT COUNT(*) FROM discovered_occurrences WHERE asset_id = ?1), "
                 "first_seen = (SELECT MIN(airing_start) FROM discovered_occurrences WHERE asset_id = ?1), "
                 "last_seen = (SELECT MAX(airing_start) FROM discovered_occurrences WHERE asset_id = ?1) "
                 "WHERE asset_id = ?1", (asset_id,))


def unmerge_asset(db_path, duplicate: str) -> dict:
    """Undo the latest merge of `duplicate`: remove the airings it added to the master, restore its status."""
    with get_db(db_path) as conn:
        mg = conn.execute("SELECT * FROM asset_merges WHERE duplicate_id = ? AND undone_at IS NULL "
                          "ORDER BY id DESC LIMIT 1", (duplicate,)).fetchone()
        if mg is None:
            raise KeyError(f"{duplicate} has no active merge")
        ids = json.loads(mg["copied_ids"])
        for i in ids:
            conn.execute("DELETE FROM discovered_occurrences WHERE id = ? AND asset_id = ?", (i, mg["master_id"]))
        conn.execute("UPDATE discovered_assets SET status = 'unclassified', merged_into = NULL, label = NULL "
                     "WHERE asset_id = ?", (duplicate,))
        _refresh_counts(conn, mg["master_id"])
        conn.execute("UPDATE asset_merges SET undone_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?", (mg["id"],))
    return dict(master=mg["master_id"], duplicate=duplicate, airings_removed=len(ids))


def dedupe_assets(db_path, apply: bool = True, channel=None) -> dict:
    """Find duplicates; merge the safe ones (same_airings / same_creative) and flag 'related' pairs."""
    pairs = find_duplicate_assets(db_path, channel)
    merged, related, gone = [], {}, set()
    for p in pairs:
        if p["kind"] == "related":
            for x, y in ((p["master"], p["duplicate"]), (p["duplicate"], p["master"])):
                related.setdefault(x, []).append(dict(asset_id=y, similarity=p["similarity"], coverage=p["coverage"]))
            continue
        if not apply or p["master"] in gone or p["duplicate"] in gone:
            continue
        merged.append(merge_assets(db_path, p["master"], p["duplicate"], similarity=p["similarity"],
                                   coverage=p["coverage"], offset_sec=p["offset_sec"],
                                   reason=f"{p['kind']} (similarity {p['similarity']:.2f}, coverage {p['coverage']:.0%})"))
        gone.add(p["duplicate"])
    if apply:
        with get_db(db_path) as conn:
            for aid, rel in related.items():
                if aid not in gone:
                    conn.execute("UPDATE discovered_assets SET related_assets = ? WHERE asset_id = ?",
                                 (json.dumps([r for r in rel if r["asset_id"] not in gone]), aid))
    return dict(pairs=pairs, merged=merged, related=related)


# ---------------------------------------------------------------- classification

def ignore_asset(db_path, asset_id: str, label: str = "not an ad") -> None:
    with get_db(db_path) as conn:
        if conn.execute("UPDATE discovered_assets SET status = 'ignored', label = ? WHERE asset_id = ?",
                        (label, asset_id)).rowcount == 0:
            raise KeyError(asset_id)


def clean_names(db_path, brand: str, campaign: str | None, duration: float | None = None) -> tuple[str, str]:
    """Names a promoted asset is stored under: registry spelling of the brand, stitched fragments repaired, internal
    bookkeeping text ('Auto-discovered ELHEDDAF_AD_117') never stored, and a unique readable default ('45 s spot') when
    the caller gave nothing meaningful. UNIQUE(brand, campaign) is kept by numbering ('45 s spot #2')."""
    from spotly.analytics.brand_extractor import clean_campaign, repair_stitched_brand, strip_internal
    brand, campaign = repair_stitched_brand(brand, campaign)
    if not brand:
        raise ValueError("a brand name is required")
    campaign = strip_internal(campaign) or clean_campaign(brand, "", duration)
    with get_db(db_path) as conn:
        taken = {r[0] for r in conn.execute("SELECT campaign_name FROM reference_ads WHERE brand_name = ?", (brand,))}
    if campaign.endswith(" spot"):                       # a generated default: number it rather than collide
        base, n = campaign, 1
        while campaign in taken:
            n += 1
            campaign = f"{base} #{n}"
    return brand, campaign


def promote_asset(db_path, asset_id: str, brand: str, campaign: str, *, industry=None, ad_type="self",
                  target_competitor_of=None) -> dict:
    """Classify a discovered asset as a commercial: index its cached audio as a reference ad and backfill
    its discovered airings into ad_occurrences (with SHA-256 hashes, no evidence image)."""
    from spotly.core.library import ingest_bytes
    with get_db(db_path) as conn:
        a = conn.execute("SELECT * FROM discovered_assets WHERE asset_id = ?", (asset_id,)).fetchone()
        occ = conn.execute("SELECT * FROM discovered_occurrences WHERE asset_id = ? ORDER BY airing_start",
                           (asset_id,)).fetchall()
    if a is None:
        raise KeyError(asset_id)
    if not a["audio_path"] or not Path(a["audio_path"]).exists():
        raise ValueError(f"{asset_id} has no cached audio to build a reference from")
    # Reference = the spot itself: trim the review clip's post-roll back to the nominal length, so the next
    # spot's audio never enters the reference fingerprint or the billed duration.
    keep_sec = a["nominal_duration_sec"] or a["duration_sec"]
    brand, campaign = clean_names(db_path, brand, campaign, keep_sec)
    with wave.open(a["audio_path"], "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(min(w.getnframes(), int(round(keep_sec * rate))))
        params = (w.getnchannels(), w.getsampwidth(), rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(params[0])
        w.setsampwidth(params[1])
        w.setframerate(params[2])
        w.writeframes(frames)
    res = ingest_bytes(buf.getvalue(), f"{asset_id}.wav", brand, campaign, db_path=db_path, industry=industry,
                       ad_type=ad_type, target_competitor_of=target_competitor_of)
    ad_id = res.ad_id if res.committed else (res.duplicate_of or {}).get("ad_id")
    if ad_id is None:
        raise ValueError("; ".join(res.warnings) or "could not index the asset audio")
    source = f"live:{a['channel']}"
    added = 0
    with get_db(db_path) as conn:
        for o in occ:
            ts = o["algeria_timestamp"] or algeria_iso(o["airing_start"])
            vhash = compute_verification_hash(ad_id, source, ts, o["airing_end"], o["duration_sec"],
                                              o["confidence"], None)
            cur = conn.execute(
                "INSERT OR IGNORE INTO ad_occurrences (ad_id, source_name, airing_start, airing_end, duration_detected, "
                "confidence_score, detected_at, verification_hash, algeria_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ad_id, source, o["airing_start"], o["airing_end"], o["duration_sec"], o["confidence"],
                 o["detected_at"], vhash, ts))
            added += cur.rowcount
        conn.execute("UPDATE discovered_assets SET status = 'classified', label = ?, promoted_ad_id = ? "
                     "WHERE asset_id = ?", (f"{brand} – {campaign}", ad_id, asset_id))
    return dict(asset_id=asset_id, ad_id=ad_id, occurrences_backfilled=added,
                reused_existing_reference=not res.committed)
