"""Commercial pod (ad break) reconstruction, in-break positions and clutter.

A pod is a run of detected airings on one channel where each ad starts no more than
`max_gap_seconds` after the previous one ended. Pods are computed from ad_occurrences at
query time; materialize_pods() can optionally cache pod_id / pod_position / pod_density.

Limits (also shown in the dashboard):
  * Only ads in the reference library are detected. An unindexed ad inside a break leaves a
    gap that splits the pod, so `total_ads_in_pod` is a lower bound on true break clutter and
    FIRST / LAST mean first / last *among detected ads*.
  * Live occurrences carry wall-clock time (epoch) and are grouped per channel and Algeria day.
    File-scan occurrences carry offsets into one recording and are grouped per recording.
  * Human-rejected occurrences (human_verified = -1) are ignored.
"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta

from kihc.core.analytics import channel_of
from kihc.core.proof import ALGERIA_TZ
from kihc.database.schema import init_db

MAX_GAP_SECONDS = 12.0
HIGH_ADS, HIGH_SECONDS, MEDIUM_ADS = 8, 240.0, 4     # High: > 8 ads or > 4 min; Medium: 4-8 ads; Low: < 4
POSITION_LABELS = {"FIRST": "First-in-Break", "MIDDLE": "Middle", "LAST": "Last-in-Break", "SOLO": "Solo"}

_SQL = """
SELECT o.occurrence_id, o.ad_id, o.source_name, o.airing_start, o.airing_end, o.duration_detected,
       o.confidence_score, o.human_verified, o.boundary_json, r.brand_name, r.campaign_name, r.industry, r.ad_type
FROM ad_occurrences o JOIN reference_ads r USING (ad_id)
WHERE o.human_verified != -1 {where}
ORDER BY o.source_name, o.airing_start, o.airing_end, o.occurrence_id
"""


@dataclass
class Pod:
    pod_id: str
    channel: str
    source_name: str            # 'live:<channel>' or the recording's source name
    is_live: bool
    pod_start: float            # epoch (live) or seconds into the recording
    pod_end: float
    ads: list = field(default_factory=list)
    pod_date: str | None = None
    has_overlap: bool = False   # two detections overlap in time (data-quality flag)
    clutter: str = ""

    @property
    def pod_duration_seconds(self) -> float:
        return round(self.pod_end - self.pod_start, 3)

    @property
    def total_ads_in_pod(self) -> int:
        return len(self.ads)

    def as_dict(self) -> dict:
        return dict(pod_id=self.pod_id, channel=self.channel, source_name=self.source_name, is_live=self.is_live,
                    pod_date=self.pod_date, pod_start=self.pod_start, pod_end=self.pod_end,
                    pod_duration_seconds=self.pod_duration_seconds, total_ads_in_pod=self.total_ads_in_pod,
                    clutter=self.clutter, has_overlap=self.has_overlap, ads=self.ads)


def clutter_score(n_ads: int, duration_s: float) -> str:
    if n_ads > HIGH_ADS or duration_s > HIGH_SECONDS:
        return "High"
    if n_ads >= MEDIUM_ADS:
        return "Medium"
    return "Low"


def calculate_ad_positions(pod: Pod) -> Pod:
    """Order ads by start time and assign FIRST / MIDDLE / LAST / SOLO plus the pod's clutter score."""
    pod.ads.sort(key=lambda a: (a["start"], a["end"], a["occurrence_id"]))
    n = len(pod.ads)
    for i, ad in enumerate(pod.ads, start=1):
        code = "SOLO" if n == 1 else "FIRST" if i == 1 else "LAST" if i == n else "MIDDLE"
        ad.update(position=code, position_label=POSITION_LABELS[code], position_index=i,
                  position_of=f"{i} of {n}", offset_in_pod=round(ad["start"] - pod.pod_start, 3))
    pod.clutter = clutter_score(n, pod.pod_duration_seconds)
    return pod


def refine_positions(pod: Pod) -> Pod:
    """Phase 20 ground-truth guards on a live pod (see pod_boundary_detector): FIRST needs a proven break entry
    <= 2 s before the spot with nothing unclassified in between, LAST a proven exit <= 2 s after it; otherwise the
    spot is demoted to MIDDLE. placement_status: confirmed | demoted_confirmed (edge proven, other material in
    between: a real finding) | demoted_ambiguous (cues too weak: certification withheld, no claim) | unsensed
    (no boundary sensing for this airing: legacy detected-ads-only meaning, or demoted when KIHC_STRICT_PLACEMENT=1)
    | not_edge (a middle spot, nothing to certify)."""
    from kihc.analytics.pod_boundary_detector import CERTIFY_CONFIDENCE, strict
    if not pod.is_live:
        return pod
    for ad in pod.ads:
        legacy = ad["position"]
        first, last = legacy in ("FIRST", "SOLO"), legacy in ("LAST", "SOLO")
        rec = ad.get("boundary")
        ad.update(placement_status="not_edge" if not (first or last) else "unsensed", boundary_confidence=None,
                  boundary_cues=[], placement_note=None)
        if not (first or last):
            continue
        entry, exit_ = (rec or {}).get("entry") or {}, (rec or {}).get("exit") or {}
        sensed = bool(rec) and (entry.get("sensed") or exit_.get("sensed"))
        if not sensed:
            if strict():
                ad.update(position="MIDDLE", position_label="Middle-in-Pod (demoted)", placement_status="demoted_ambiguous",
                          placement_note="no boundary sensing for this airing (strict placement)")
            continue
        sides = []                                    # (wanted?, side record, name)
        if first:
            sides.append(("entry", entry))
        if last:
            sides.append(("exit", exit_))
        failed = [(n, s) for n, s in sides if not (s.get("is_break_entry") and not s.get("unclassified_material"))]
        conf = min((s.get("boundary_confidence") or 0.0) for _, s in sides)
        cues = sorted({c for _, s in sides for c in s.get("detected_cues", [])})
        ad.update(boundary_confidence=round(conf, 3), boundary_cues=cues)
        if not failed:
            ad["placement_status"] = "confirmed"
            continue
        keep_first = first and ("entry", entry) not in failed
        keep_last = last and ("exit", exit_) not in failed
        code = "SOLO" if keep_first and keep_last else "FIRST" if keep_first else "LAST" if keep_last else "MIDDLE"
        proven = all(s.get("too_far") and s.get("unclassified_material") or
                     (s.get("is_break_entry") and s.get("unclassified_material")) for _, s in failed)
        notes = [f"{n}: " + ("unclassified material between the break " + n + " and the spot" if s.get("unclassified_material")
                             else f"confidence {s.get('boundary_confidence', 0):.2f} < {CERTIFY_CONFIDENCE}")
                 for n, s in failed]
        ad.update(position=code, position_label=POSITION_LABELS[code] if code != "MIDDLE" else "Middle-in-Pod (demoted)",
                  placement_status="demoted_confirmed" if proven else "demoted_ambiguous", placement_note="; ".join(notes))
    for ad in pod.ads:                                # empirical name of the same verdict (once frames flow: never UNSENSED)
        ad["physical_status"] = PHYSICAL_STATUS.get(ad.get("placement_status"))
    return pod


PHYSICAL_STATUS = {"confirmed": "CONFIRMED_PHYSICAL", "demoted_confirmed": "DEMOTED_CONFIRMED",
                   "demoted_ambiguous": "DEMOTED_AMBIGUOUS", "unsensed": "UNSENSED", "not_edge": "NOT_EDGE"}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")[:40] or "UNKNOWN"


def _day_bounds(date_or_window):
    """-> (load_from, load_to, keep_from, keep_to) epochs. Loading always covers whole Algeria days
    (plus a margin) so pod ids are identical whatever window is asked for."""
    if date_or_window is None:
        return None
    if isinstance(date_or_window, str):
        date_or_window = date.fromisoformat(date_or_window)
    if isinstance(date_or_window, date) and not isinstance(date_or_window, datetime):
        start = datetime.combine(date_or_window, dtime.min, ALGERIA_TZ)
        keep = (start.timestamp(), (start + timedelta(days=1)).timestamp())
    else:
        a, b = date_or_window
        to_ts = lambda v: v if isinstance(v, (int, float)) else \
            (v if v.tzinfo else v.replace(tzinfo=ALGERIA_TZ)).timestamp()
        keep = (to_ts(a), to_ts(b))
    first_day = datetime.fromtimestamp(keep[0], ALGERIA_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    last_day = datetime.fromtimestamp(keep[1], ALGERIA_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    margin = 3600.0                                   # a pod straddling midnight is loaded whole
    return (first_day.timestamp() - margin, (last_day + timedelta(days=1)).timestamp() + margin) + keep


def _load(channel, bounds, db_path):
    where, params = "", []
    if bounds is not None:                            # wall-clock window: live occurrences only
        where = "AND o.source_name LIKE 'live:%' AND o.airing_start >= ? AND o.airing_start < ?"
        params = [bounds[0], bounds[1]]
    conn = init_db(db_path)
    try:
        rows = [dict(r) for r in conn.execute(_SQL.format(where=where), params)]
    finally:
        conn.close()
    if channel:
        want = channel.strip().casefold()
        rows = [r for r in rows if channel_of(r["source_name"]).casefold() == want
                or r["source_name"].casefold() == want]
    return rows


def cluster_occurrences_into_pods(channel=None, date_or_window=None, max_gap_seconds: float = MAX_GAP_SECONDS,
                                  *, db_path=None) -> list[Pod]:
    """Reconstruct ad breaks.

    channel: channel name ('Echorouk TV'), a recording's source name, or None for all.
    date_or_window: an Algeria calendar date (date or 'YYYY-MM-DD'), a (start, end) pair of
        datetimes/epochs, or None for everything (including recordings, which have no date).
    """
    bounds = _day_bounds(date_or_window)
    rows = _load(channel, bounds, db_path)
    streams = defaultdict(list)                       # one timeline per live channel / per recording
    for r in rows:
        streams[r["source_name"]].append(r)

    pods: list[Pod] = []
    for source, items in streams.items():
        live = source.startswith("live:")
        chan = channel_of(source)
        current: Pod | None = None
        for r in items:                               # already ordered by airing_start
            ad = dict(occurrence_id=r["occurrence_id"], ad_id=r["ad_id"], brand=r["brand_name"],
                      campaign=r["campaign_name"], ad_type=r["ad_type"] or "self", industry=r["industry"],
                      start=float(r["airing_start"]), end=float(r["airing_end"]),
                      duration=float(r["duration_detected"]), confidence=float(r["confidence_score"]),
                      boundary=json.loads(r["boundary_json"]) if r["boundary_json"] else None)
            if current is not None and ad["start"] - current.pod_end <= max_gap_seconds:
                if ad["start"] < current.pod_end:
                    current.has_overlap = True
                current.ads.append(ad)
                current.pod_end = max(current.pod_end, ad["end"])
            else:
                current = Pod("", chan, source, live, ad["start"], ad["end"], [ad])
                pods.append(current)

    counters = defaultdict(int)
    for pod in sorted(pods, key=lambda p: (p.source_name, p.pod_start)):
        calculate_ad_positions(pod)
        if pod.is_live:
            refine_positions(pod)
            day = datetime.fromtimestamp(pod.pod_start, ALGERIA_TZ)
            pod.pod_date = day.date().isoformat()
            key = (pod.source_name, pod.pod_date)
            counters[key] += 1
            pod.pod_id = f"POD-{day:%Y%m%d}-{_slug(pod.channel)}-{counters[key]:03d}"
        else:
            counters[pod.source_name] += 1
            pod.pod_id = f"POD-REC-{_slug(pod.source_name)}-{counters[pod.source_name]:03d}"

    if bounds is not None:
        pods = [p for p in pods if bounds[2] <= p.pod_start < bounds[3]]
    pods.sort(key=lambda p: (p.channel, p.pod_start))
    return pods


# ---------------------------------------------------------------- aggregates for the dashboard / CLI

def position_distribution(pods: list[Pod]) -> list[dict]:
    """Per brand: airings in FIRST / MIDDLE / LAST / SOLO and the premium share (FIRST + LAST)."""
    per = defaultdict(lambda: defaultdict(int))
    roles = {}
    for p in pods:
        for a in p.ads:
            per[a["brand"]][a["position"]] += 1
            roles[a["brand"]] = "self" if a["ad_type"] == "self" or roles.get(a["brand"]) == "self" else "competitor"
    out = []
    for brand, counts in per.items():
        total = sum(counts.values())
        in_breaks = total - counts["SOLO"]
        out.append(dict(brand=brand, role=roles[brand], airings=total,
                        **{k.lower(): counts[k] for k in POSITION_LABELS},
                        **{f"{k.lower()}_pct": 100.0 * counts[k] / total for k in POSITION_LABELS},
                        premium_pct=100.0 * (counts["FIRST"] + counts["LAST"]) / in_breaks if in_breaks else 0.0))
    return sorted(out, key=lambda x: (-x["airings"], x["brand"]))


def channel_clutter(pods: list[Pod]) -> list[dict]:
    """Per channel: number of pods, average/max ads and duration per pod, clutter mix."""
    per = defaultdict(list)
    for p in pods:
        per[p.channel].append(p)
    out = []
    for chan, ps in per.items():
        n = len(ps)
        mix = defaultdict(int)
        for p in ps:
            mix[p.clutter] += 1
        out.append(dict(channel=chan, pods=n,
                        avg_ads_per_pod=sum(p.total_ads_in_pod for p in ps) / n,
                        max_ads_per_pod=max(p.total_ads_in_pod for p in ps),
                        avg_pod_seconds=sum(p.pod_duration_seconds for p in ps) / n,
                        max_pod_seconds=max(p.pod_duration_seconds for p in ps),
                        high=mix["High"], medium=mix["Medium"], low=mix["Low"],
                        overall=clutter_score(round(sum(p.total_ads_in_pod for p in ps) / n),
                                              sum(p.pod_duration_seconds for p in ps) / n)))
    return sorted(out, key=lambda x: (-x["avg_ads_per_pod"], x["channel"]))


def materialize_pods(channel=None, date_or_window=None, max_gap_seconds: float = MAX_GAP_SECONDS,
                     *, db_path=None) -> int:
    """Cache pod_id / pod_position / pod_density on ad_occurrences for the given scope. Idempotent.

    These columns are not covered by the verification hash, so caching never affects integrity.
    """
    pods = cluster_occurrences_into_pods(channel, date_or_window, max_gap_seconds, db_path=db_path)
    updates = [(p.pod_id, a["position"], p.total_ads_in_pod, a["occurrence_id"]) for p in pods for a in p.ads]
    conn = init_db(db_path)
    try:
        with conn:
            conn.executemany("UPDATE ad_occurrences SET pod_id = ?, pod_position = ?, pod_density = ? "
                             "WHERE occurrence_id = ?", updates)
    finally:
        conn.close()
    return len(updates)
