"""Phase 26: MCR timecode synchronisation - the broadcaster's own timeline for every detected airing.

Every airing is first stamped on the ingest host's clock (NTP, shown as CET): `airing_start`, `algeria_timestamp`,
`detected_at`. Those raw values are the immutable audit trail and are never rewritten. This module derives, next to them, the
time the spot left the broadcaster's master control room (MCR):

    mcr_drift_delta_seconds = broadcast_time - ingest_time        (< 0: the broadcaster's clock reads EARLIER than our
                                                                    capture clock, i.e. packaging + CDN + look-behind latency)
    broadcast_start_cet     = ingest_start + mcr_drift_delta_seconds

(The Phase 22 columns `clock_drift_*` use the opposite sign, capture - broadcaster; both are stored, they are consistent.)

Sources, best first - the basis is reported with every reading:
  program-date-time  absolute: the HLS #EXT-X-PROGRAM-DATE-TIME of the segment the stream starts with versus the wall time
                     of the first audio received. Median of recent readings, margin = rtt/2 + 95th-percentile deviation.
  pts-ema            fallback when the manifest carries no PDT (or a static one that does not advance): the drift of
                     (arrival wall-clock - PTS) since the first observation, smoothed by an exponential moving average
                     (alpha = 0.15) so network jitter does not move the estimate in steps. PTS only gives a RELATIVE clock,
                     so the result is added to the last absolute (PDT) anchor when one exists; without it the reading is
                     reported as relative and the channel is never shown as locked.

    sync = get_synchronizer()
    sync.observe_manifest(channel, capture_epoch, program_date_time_epoch, rtt_s)
    sync.observe_pts(channel, arrival_epoch, pts_s, rtt_s)
    calibrated_epoch, jitter_margin_s, drift_s = sync.calibrate_timestamp(ingest_epoch, channel)

Cost: a few floats and two 120-entry deques per channel; no thread, no I/O, nothing on the audio path.
"""
import collections
import math
import threading
import time

ALPHA = 0.15                       # EMA weight of the newest PTS-based reading
MIN_MARGIN_S = 0.2
FRESH_S = 600.0                    # a reading older than this is not used to stamp a new airing
LOCK_MARGIN_S = 1.0                # a PDT reading at least this tight counts as "locked"
MAX_PLAUSIBLE_S = 600.0
STATIC_PDT_GAP_S = 5.0             # capture advanced this much while the manifest clock did not -> static tag
HISTORY = 120
REF_WARMUP = 5

BASIS_PDT, BASIS_EMA = "program-date-time", "pts-ema"
LOCKED, TRACKING, UNSYNCED = "LOCKED", "TRACKING", "UNSYNCED"


def lock_status(basis: str | None, margin_s: float | None) -> str:
    """LOCKED: absolute PDT anchor, tight margin. TRACKING: only a relative / loose estimate. UNSYNCED: nothing measured."""
    if basis in (BASIS_PDT, "program-date-time") and margin_s is not None and margin_s <= LOCK_MARGIN_S:
        return LOCKED
    if basis and margin_s is not None:
        return TRACKING
    return UNSYNCED


class _Channel:
    def __init__(self):
        self.pdt = collections.deque(maxlen=HISTORY)          # broadcast - ingest, from the manifest
        self.pdt_rtt = 0.0
        self.pdt_updated: float | None = None
        self._last_pdt: tuple[float, float] | None = None     # (capture, broadcaster) of the previous manifest reading
        self.static_pdt = 0
        self.pts_ref: tuple[float, float] | None = None       # (arrival, pts) reference: mean of the first REF_WARMUP pairs
        self.pts_warm: list[tuple[float, float]] = []
        self.ema: float | None = None
        self.ema_var = 0.0
        self.ema_n = 0
        self.ema_rtt = 0.0
        self.ema_updated: float | None = None
        self.anchor: float | None = None                      # last absolute PDT delta: base for the relative PTS drift


class MCRTimeSynchronizer:
    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self._lock = threading.Lock()
        self._ch: dict[str, _Channel] = {}

    # ---------------------------------------------------------------- inputs
    def _get(self, channel: str) -> _Channel:
        return self._ch.setdefault(channel, _Channel())

    def observe_manifest(self, channel: str, capture_epoch: float, broadcaster_epoch: float, rtt_s: float = 0.0) -> bool:
        """One #EXT-X-PROGRAM-DATE-TIME reading: `broadcaster_epoch` is the manifest clock for the audio first received at
        `capture_epoch`. Implausible or static readings are ignored (-> False) so the PTS fallback stays in charge."""
        delta = broadcaster_epoch - capture_epoch
        if not math.isfinite(delta) or abs(delta) > MAX_PLAUSIBLE_S:
            return False
        with self._lock:
            c = self._get(channel)
            last = c._last_pdt
            c._last_pdt = (capture_epoch, broadcaster_epoch)
            if last and capture_epoch - last[0] >= STATIC_PDT_GAP_S and broadcaster_epoch == last[1]:
                c.static_pdt += 1                                 # the tag did not advance while time did: not a clock
                return False
            c.pdt.append(delta)
            c.pdt_rtt, c.pdt_updated = float(rtt_s), time.time()
            c.anchor = sorted(c.pdt)[len(c.pdt) // 2]
        return True

    def observe_pts(self, channel: str, arrival_epoch: float, pts_s: float, rtt_s: float = 0.0) -> float | None:
        """One (arrival wall-clock, PTS) pair. The first pair is the reference; afterwards the growth of (arrival - PTS) is
        the latency drift, EMA-smoothed. Returns the smoothed relative delta (broadcast - ingest change)."""
        with self._lock:
            c = self._get(channel)
            if len(c.pts_warm) < REF_WARMUP:                      # the reference is the mean of the first few pairs, not one
                c.pts_warm.append((arrival_epoch, pts_s))         # jittery sample whose noise would become a constant bias
                n = len(c.pts_warm)
                c.pts_ref = (sum(a for a, _ in c.pts_warm) / n, sum(p for _, p in c.pts_warm) / n)
            ref_a, ref_p = c.pts_ref
            x = -((arrival_epoch - ref_a) - (pts_s - ref_p))      # latency grew -> broadcast moves further before ingest
            if not math.isfinite(x) or abs(x) > MAX_PLAUSIBLE_S:
                return None
            if c.ema is None:
                c.ema, c.ema_var = x, 0.0
            else:
                err = x - c.ema
                c.ema += self.alpha * err
                c.ema_var = (1 - self.alpha) * (c.ema_var + self.alpha * err * err)   # EW variance of the residual
            c.ema_n += 1
            c.ema_rtt, c.ema_updated = float(rtt_s), time.time()
            return c.ema

    # ---------------------------------------------------------------- outputs
    def _estimate(self, channel: str, now: float):
        """(delta, margin, basis, absolute) from the best fresh source, else None."""
        c = self._ch.get(channel)
        if c is None:
            return None
        if c.pdt and c.pdt_updated is not None and now - c.pdt_updated <= FRESH_S:
            d = sorted(c.pdt)
            med = d[len(d) // 2]
            dev = sorted(abs(x - med) for x in d)
            p95 = dev[min(len(dev) - 1, int(len(dev) * 0.95))]
            return med, max(MIN_MARGIN_S, c.pdt_rtt / 2 + p95), BASIS_PDT, True
        if c.ema is not None and c.ema_updated is not None and now - c.ema_updated <= FRESH_S:
            absolute = c.anchor is not None
            delta = (c.anchor or 0.0) + c.ema
            margin = max(MIN_MARGIN_S, c.ema_rtt / 2 + 1.96 * math.sqrt(c.ema_var))
            return delta, margin, BASIS_EMA, absolute
        return None

    def calibrate_timestamp(self, ingest_timestamp: float, channel_id: str, now: float | None = None):
        """-> (calibrated_epoch, jitter_margin_s, drift_seconds); drift = broadcast - ingest.

        With no fresh reading the raw timestamp is returned with margin and drift None: an unmeasured clock is unknown, never 0."""
        with self._lock:
            est = self._estimate(channel_id, time.time() if now is None else now)
        if est is None:
            return float(ingest_timestamp), None, None
        delta, margin, _, _ = est
        return float(ingest_timestamp) + delta, round(margin, 3), round(delta, 3)

    def status(self, channel_id: str, now: float | None = None) -> dict:
        with self._lock:
            est = self._estimate(channel_id, time.time() if now is None else now)
            c = self._ch.get(channel_id)
            static = c.static_pdt if c else 0
        if est is None:
            return dict(channel=channel_id, lock=UNSYNCED, basis=None, drift_s=None, margin_s=None, absolute=False,
                        static_pdt_ignored=static)
        delta, margin, basis, absolute = est
        lock = lock_status(basis, margin) if absolute else TRACKING
        return dict(channel=channel_id, lock=lock, basis=basis, drift_s=round(delta, 3), margin_s=round(margin, 3),
                    absolute=absolute, static_pdt_ignored=static)

    def reset(self) -> None:
        with self._lock:
            self._ch.clear()


_SYNC = MCRTimeSynchronizer()


def get_synchronizer() -> MCRTimeSynchronizer:
    return _SYNC


def reset() -> None:
    _SYNC.reset()


# ---------------------------------------------------------------- airing columns
def occurrence_fields(channel: str, start_epoch: float, end_epoch: float) -> dict:
    """Columns for an ad_occurrences row: broadcast_start_cet / broadcast_end_cet / mcr_drift_delta_seconds (all None while the
    channel has no fresh calibration). The raw ingest timestamps are written by the caller and are never touched here."""
    from kihc.core.proof import algeria_iso
    cal_start, _, delta = _SYNC.calibrate_timestamp(start_epoch, channel)
    if delta is None:
        return dict(broadcast_start_cet=None, broadcast_end_cet=None, mcr_drift_delta_seconds=None)
    return dict(broadcast_start_cet=algeria_iso(cal_start), broadcast_end_cet=algeria_iso(end_epoch + delta),
                mcr_drift_delta_seconds=delta)


def broadcast_epoch(airing_start: float, mcr_drift_delta_seconds: float | None) -> float:
    """The instant daypart, season and contract-date rules must be evaluated at: the broadcast start when it is known,
    otherwise (legacy rows, uncalibrated channel) the ingest start."""
    return airing_start if mcr_drift_delta_seconds is None else airing_start + float(mcr_drift_delta_seconds)


def airing_mcr(row) -> dict:
    """Calibrated view of one ad_occurrences row for the API / dashboard. The stored Phase 26 columns win; legacy rows that
    only carry the Phase 22 reading (capture - broadcaster) are derived from it. The raw ingest columns are not touched."""
    from kihc.core.proof import algeria_iso
    get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row.keys() else d)
    delta = get("mcr_drift_delta_seconds")
    if delta is None and get("clock_drift_seconds") is not None:
        delta = -float(get("clock_drift_seconds"))
    margin = get("clock_drift_margin_s")
    out = dict(mcr_drift_delta_seconds=delta, margin_s=margin, basis=get("clock_drift_basis"),
               lock=lock_status(get("clock_drift_basis"), margin) if delta is not None else UNSYNCED,
               broadcast_start_cet=get("broadcast_start_cet"), broadcast_end_cet=get("broadcast_end_cet"))
    if delta is not None and get("airing_start") is not None:
        out["broadcast_start_cet"] = out["broadcast_start_cet"] or algeria_iso(float(get("airing_start")) + delta)
        out["broadcast_end_cet"] = out["broadcast_end_cet"] or algeria_iso(float(get("airing_end")) + delta)
    return out


def backfill_calibrated(db_path=None) -> int:
    """Fill the Phase 26 columns of airings that carry a Phase 22 clock reading but no calibrated times yet:
    delta = -clock_drift_seconds, broadcast = ingest + delta. Only NULL columns are written; the raw ingest stamps, hashes and
    every other column stay as they are. -> rows updated."""
    from kihc.core.proof import algeria_iso
    from kihc.database.schema import get_db
    n = 0
    with get_db(db_path) as conn:
        rows = conn.execute("SELECT occurrence_id, airing_start, airing_end, clock_drift_seconds FROM ad_occurrences "
                            "WHERE clock_drift_seconds IS NOT NULL AND mcr_drift_delta_seconds IS NULL").fetchall()
        for r in rows:
            delta = -float(r["clock_drift_seconds"])
            n += conn.execute("UPDATE ad_occurrences SET mcr_drift_delta_seconds = ?, broadcast_start_cet = ?, "
                              "broadcast_end_cet = ? WHERE occurrence_id = ? AND mcr_drift_delta_seconds IS NULL",
                              (delta, algeria_iso(r["airing_start"] + delta), algeria_iso(r["airing_end"] + delta),
                               r["occurrence_id"])).rowcount
    return n
