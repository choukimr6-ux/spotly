"""KIHC Streamlit dashboard.

Launch:  streamlit run kihc/app/dashboard.py
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kihc.app.devreload import ensure_hot_reload, evict_stale_modules  # noqa: E402

ensure_hot_reload(_ROOT)     # so edits to kihc/core, kihc/connectors, … reload instead of leaving stale imports
evict_stale_modules(_ROOT)   # …including for browser sessions opened after the edit

import altair as alt
import pandas as pd
import streamlit as st

import time

from kihc.cli import EXPORT_SQL, export_csv
from kihc.config import DEFAULT_DB_PATH, MATCH_THRESHOLD
from kihc.connectors.iptv_manager import (FALLBACK_VERIFIED_ON, FEATURED_KEYS,
                                          NATIONAL_BROADCASTERS,
                                          channel_resolver, check_streams,
                                          fetch_channels,
                                          resolve_channel_stream)
from kihc.connectors.youtube_live import youtube_resolver
from datetime import datetime, timedelta

from kihc.app.components.visual_player import DISCLAIMER as PREVIEW_DISCLAIMER
from kihc.app.components.visual_player import (render_live_player,
                                               resolve_preview_source)
from kihc.connectors.iptv_manager import load_overrides
from kihc.core.analytics import (channel_of, channel_placement,
                                 compute_share_of_voice, daypart_distribution,
                                 detect_competitor_surges)
from kihc.core.pod_analyzer import (HIGH_ADS, HIGH_SECONDS, MEDIUM_ADS,
                                    channel_clutter,
                                    cluster_occurrences_into_pods,
                                    position_distribution)
from kihc.core.library import (MAX_UPLOAD_MB, MEDIA_TYPES, ingest_bytes,
                               scan_bytes, scan_recording, set_ad_tags,
                               timecode)
from kihc.core.proof import ALGERIA_TZ
from kihc.database.schema import AD_TYPES
from kihc.core.live_worker import LiveStreamWorker
from kihc.core.proof import verify_row
from kihc.database.schema import get_db, init_db
from kihc.reports.certificate import ROW_SQL as CERT_ROW_SQL
from kihc.reports.certificate import certificate_for
from kihc.service.headless_runner import read_service_status
from kihc.app import overview as ov
from kihc.app.theme import (brand_html, card_header, engine_badge_html,
                            inject_theme, kpi_card, microbar_svg, pill,
                            ratio_list_html, section_html, semi_donut_svg)

# Fixed categorical order (validated reference palette); colour follows ad_id, never rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRACK = "#d9d8d4"
REVIEW_LABEL = {1: "✅ Verified", 0: "⏳ Unreviewed", -1: "❌ Rejected"}

st.set_page_config(page_title="KIHC · Broadcast Intelligence", page_icon="📡", layout="wide")
inject_theme()


def load(db_path):
    with get_db(db_path) as conn:
        df = pd.read_sql_query(EXPORT_SQL, conn)
        ads = pd.read_sql_query("SELECT ad_id, brand_name, campaign_name, duration_sec, created_at FROM reference_ads",
                                conn)
    df["ad_label"] = df["brand_name"] + " – " + df["campaign_name"]
    df["review"] = df["human_verified"].map(REVIEW_LABEL)
    # Live occurrences (source "live:<channel>") store epoch seconds; file scans store seconds into the file.
    df["is_live"] = df["source_name"].str.startswith("live:")
    df["start_utc"] = pd.to_datetime(df["airing_start"].where(df["is_live"]), unit="s", utc=True)
    df["end_utc"] = pd.to_datetime(df["airing_end"].where(df["is_live"]), unit="s", utc=True)
    df["start_display"] = [
        s.strftime("%Y-%m-%d %H:%M:%S UTC") if live else f"{a:.3f}s"
        for live, s, a in zip(df["is_live"], df["start_utc"], df["airing_start"])]
    return df, ads


@st.cache_resource
def worker_registry() -> dict:
    """Process-wide {channel: LiveStreamWorker}; survives Streamlit reruns and browser reloads."""
    return {}


@st.cache_data(ttl=600, show_spinner="Checking Algerian streams…")
def load_channels():
    return [dict(name=c.name, url=c.url, headers=c.headers, active=bool(c.active),
                 source=c.source, detail=c.check_detail)
            for c in check_streams(fetch_channels())]


@st.cache_data(ttl=60, show_spinner="Resolving stream…")
def resolve_featured(channel_key: str) -> dict:
    r = resolve_channel_stream(channel_key)
    return dict(url=r.url, stream_type=r.stream_type, attempts=r.attempts)


RESOLVE_LABEL = {"hls": "🟢 Direct HLS stream", "youtube_live": "🟢 YouTube Live stream",
                 "unavailable": "🔴 Not live right now"}

STATE_LABEL = {"listening": "🟢 Listening", "connecting": "🟡 Connecting…", "backoff": "🟠 Reconnecting",
               "stopped": "⚪ Stopped", "idle": "⚪ Idle"}


def live_sidebar(db_path):
    st.sidebar.divider()
    st.sidebar.markdown(section_html("📺 Live Channels"), unsafe_allow_html=True)
    channels = load_channels()
    show_all = st.sidebar.toggle("Include unreachable channels", value=False)
    listed = [c for c in channels if c["active"] or show_all]
    src = channels[0]["source"] if channels else "none"
    st.sidebar.caption(f"{sum(c['active'] for c in channels)}/{len(channels)} reachable · list source: {src}"
                       + (f" (verified {FALLBACK_VERIFIED_ON})" if src == "fallback" else ""))
    if st.sidebar.button("↻ Re-check streams", width="stretch"):
        load_channels.clear()
        st.rerun()

    # Featured national broadcasters first (always listed; resolved on selection), then other IPTV channels.
    featured = {f"⭐ {NATIONAL_BROADCASTERS[k].name}": k for k in FEATURED_KEYS}
    featured_names = {NATIONAL_BROADCASTERS[k].name for k in FEATURED_KEYS}
    by_label = {f"{'🟢' if c['active'] else '🔴'} {c['name']}": c for c in listed if c["name"] not in featured_names}
    pick = st.sidebar.selectbox("Algerian channel", list(featured) + list(by_label),
                                help="⭐ Featured channels are resolved automatically: direct HLS if reachable, "
                                     "otherwise the broadcaster's YouTube Live stream.")
    yt_url = st.sidebar.text_input("…or YouTube Live URL", placeholder="https://www.youtube.com/@channel/live")

    if yt_url.strip():
        key, url, headers = f"YouTube {yt_url.strip()}", youtube_resolver(yt_url.strip()), {}
    elif pick in featured:
        ch_key = featured[pick]
        res = resolve_featured(ch_key)
        st.sidebar.caption(RESOLVE_LABEL[res["stream_type"]]
                           + ("" if res["url"] else " — monitoring will connect automatically once it goes live"))
        if not res["url"]:
            with st.sidebar.expander("Resolution attempts"):
                for method, target, outcome in res["attempts"]:
                    st.caption(f"{method}: {target} → {outcome[:120]}")
        if st.sidebar.button("↻ Re-resolve", width="stretch"):
            resolve_featured.clear()
            st.rerun()
        # The worker re-resolves on every (re)connect, so expiring YouTube URLs and late starts are handled.
        key, url, headers = NATIONAL_BROADCASTERS[ch_key].name, channel_resolver(ch_key), {}
    elif pick in by_label:
        ch = by_label[pick]
        key, url, headers = ch["name"], ch["url"], ch["headers"]
    else:
        key = None

    registry = worker_registry()
    worker = registry.get(key) if key else None
    running = bool(worker and worker.is_alive())
    engine = read_service_status(db_path)
    engine_active = bool(engine and engine["active"])
    c1, c2 = st.sidebar.columns(2)
    # While the 24/7 engine runs, the dashboard is read-only for live monitoring: no duplicate stream threads.
    if c1.button("▶ Start", disabled=key is None or running or engine_active, type="primary", width="stretch"):
        registry[key] = LiveStreamWorker(key, url, db_path, headers=headers).start()
        st.rerun()
    if c2.button("■ Stop", disabled=not running, width="stretch"):
        worker.stop()
        st.rerun()
    if engine_active:
        st.sidebar.caption("In-app monitoring is disabled while the background engine runs. "
                           "Stop it with `scripts\\stop_monitoring_service.bat` to monitor from here.")
    st.sidebar.caption("Live radar, signal health and video preview: dark card on the main canvas.")


RADAR_FLOOR_DBFS = -60.0


def _fmt_duration(seconds):
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _stability(s):
    if s["state"] != "listening":
        return "⚪ Not connected"
    if s["disconnects_last_hour"] >= 3:
        return f"🔴 Unstable · {s['disconnects_last_hour']} drops in the last hour"
    if s["disconnects_last_hour"] or (s["connected_for_s"] or 0) < 60:
        return f"🟡 Settling · {s['disconnects_last_hour']} drop(s) in the last hour"
    return "🟢 Stable · no drops in the last hour"


def _level_sparkline(history):
    df = pd.DataFrame(history, columns=["t", "dbfs"])
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["dbfs"] = df["dbfs"].clip(lower=RADAR_FLOOR_DBFS)
    return alt.Chart(df).mark_line(strokeWidth=2, color="#10B981").encode(
        x=alt.X("t:T", axis=None),
        y=alt.Y("dbfs:Q", scale=alt.Scale(domain=[RADAR_FLOOR_DBFS, 0]),
                axis=alt.Axis(values=[-60, -40, -20, 0], title=None, labelFontSize=9, grid=True,
                              labelColor="#9FC2BC", gridColor="#1E3B39", domain=False, ticks=False)),
        tooltip=[alt.Tooltip("t:T", title="Time (UTC)", format="%H:%M:%S"),
                 alt.Tooltip("dbfs:Q", title="Level (dBFS)", format=".1f")],
    ).properties(height=70).configure_view(strokeWidth=0).configure(background="transparent")


def _stream_ref(worker):
    """What a worker monitors: a URL, or 'channel:<key>' / 'youtube:<url>' for re-resolving sources."""
    src = getattr(worker, "_url", None)
    return src if isinstance(src, str) else getattr(src, "__name__", None)


def _video_preview(name, stream_ref):
    """Collapsed, dormant, client-side preview. Source picked without any network call (radar refreshes
    every 3 s): overrides -> channel registry -> configured URL / cached IPTV list."""
    with st.expander("📺 Live Video Preview (Visual Check)", expanded=False):
        st.caption(f"🛈 {PREVIEW_DISCLAIMER}")
        url, is_yt = resolve_preview_source(name, stream_ref, overrides=load_overrides(),
                                            iptv_urls={c["name"]: c["url"] for c in load_channels()})
        render_live_player(url, name, is_youtube=is_yt, show_badge=False)


def _radar_card(name, s, restarts=None, stream_ref=None):
    with st.container(border=True):
        label = STATE_LABEL.get(s["state"], s["state"])
        if s["state"] == "backoff" and "ChannelUnavailable" in s["last_error"]:
            label = "🟠 Waiting for channel to go live · next check"
        if s["state"] == "backoff" and s["next_retry_at"]:
            label += f" in {max(0, s['next_retry_at'] - time.time()):.0f}s"
        st.markdown(f"**{name}**  \n{label}")
        st.caption(_stability(s) + f" · connected {_fmt_duration(s['connected_for_s'])}"
                   + (f" · supervisor restarts {restarts}" if restarts is not None else ""))
        level = s["level_dbfs"]
        if s["state"] == "listening" and level is not None:
            shown = max(RADAR_FLOOR_DBFS, min(0.0, level))
            tag = "🔇 Silent / dead air" if s["silent"] else "🔊 Audio present"
            st.progress((shown - RADAR_FLOOR_DBFS) / -RADAR_FLOOR_DBFS,
                        text=f"{tag} · {level:.1f} dBFS (peak 10 s: {s['peak_dbfs_10s']:.1f})")
            if len(s["level_history"]) > 2:
                st.altair_chart(_level_sparkline(s["level_history"]), width="stretch")
        st.caption(f"buffer {s['buffered_seconds']:.0f}s · received {s['received_seconds']:.0f}s · "
                   f"{s['references']} refs · {s['detections']} detections · attempts {s['connect_attempts']}")
        if s["last_error"] and s["state"] != "listening":
            st.caption(f"⚠ {s['last_error'][:160]}")
        _video_preview(name, stream_ref)


@st.fragment(run_every=3)
def live_status(db_path):
    st.subheader("📡 Live Broadcast Radar")
    engine = read_service_status(db_path)
    if engine and engine["active"]:
        st.success("24/7 Background Engine Active", icon="🟢")
        st.caption(f"pid {engine['pid']} on {engine['hostname']} · up since {engine['started_at']} · "
                   f"heartbeat {engine['heartbeat_age_s']:.0f}s ago · RAM {engine['rss_mb']:.0f} MB "
                   f"(+ffmpeg {engine['ffmpeg_rss_mb']:.0f} MB) · restarts {engine['restarts']}")
        for c in engine["channel_status"]:
            _radar_card(c["name"], c, restarts=c["restarts"], stream_ref=c.get("stream_ref"))
    elif engine and engine["state"] in ("starting", "running"):
        st.warning(f"⚠️ Background engine not responding: last heartbeat {engine['heartbeat_age_s']:.0f}s ago "
                   f"(pid {engine['pid']}); its lock is {'held' if engine['lock_held'] else 'released'}.")
    workers = worker_registry()
    if not workers and not (engine and engine["active"]):
        st.caption("No live monitors running. Start one from the sidebar, or run the 24/7 engine: "
                   "`scripts\\start_monitoring_service.bat`.")
    for name, w in list(workers.items()):
        _radar_card(name, w.status(), stream_ref=_stream_ref(w))


def occurrence_integrity(db_path, occurrence_id):
    conn = init_db(db_path)
    try:
        row = conn.execute(CERT_ROW_SQL, (occurrence_id,)).fetchone()
    finally:
        conn.close()
    return verify_row(row) if row else None


@st.cache_data(max_entries=32, show_spinner=False)
def certificate_pdf(db_path, occurrence_id, review_state, integrity_state) -> bytes:
    """In-memory PDF; cache key includes review + integrity state so the certificate never goes stale."""
    return certificate_for(db_path, occurrence_id)


def _review_state(db_path, occurrence_id):
    conn = init_db(db_path)
    try:
        row = conn.execute("SELECT human_verified FROM ad_occurrences WHERE occurrence_id = ?",
                           (occurrence_id,)).fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def _distinct(db_path, sql):
    conn = init_db(db_path)
    try:
        return [r[0] for r in conn.execute(sql) if r[0]]
    finally:
        conn.close()


def known_industries(db_path):
    return _distinct(db_path, "SELECT DISTINCT industry FROM reference_ads WHERE industry IS NOT NULL "
                              "ORDER BY industry COLLATE NOCASE")


def client_brands(db_path):
    return _distinct(db_path, "SELECT DISTINCT brand_name FROM reference_ads WHERE ad_type = 'self' "
                              "ORDER BY brand_name COLLATE NOCASE")


def ingest_panel(db_path):
    with st.container(key="bento-ingest"):
        st.markdown(card_header("📥 Ingest Reference Ad",
                                "Upload a creative; audio is fingerprinted in memory and added to the library."),
                    unsafe_allow_html=True)
        with st.form("ingest_form", clear_on_submit=False):
            up = st.file_uploader("Creative file", type=list(MEDIA_TYPES),
                                  help=f"Audio is extracted in memory; nothing is written to disk. "
                                       f"Max {MAX_UPLOAD_MB} MB.")
            brand = st.text_input("Brand name")
            campaign = st.text_input("Campaign name")
            expected = st.number_input("Expected duration (s)", min_value=0.0, max_value=600.0, value=0.0,
                                       step=1.0, help="0 = don't check. Warns if the audio length differs by >1 s.")
            known_ind = known_industries(db_path)
            industry = st.text_input("Industry", placeholder=", ".join(known_ind[:4]) or "e.g. Telecom, Dairy",
                                     help="Used to group brands for Share of Voice. Existing: "
                                          + (", ".join(known_ind) or "none yet"))
            role = st.radio("Role", ["My Brand [self]", "Competitor Ad [competitor]"], horizontal=False)
            clients = client_brands(db_path)
            competes = st.text_input("Competes with (competitor ads only)",
                                     placeholder=", ".join(clients[:3]) or "client brand name",
                                     help="Which of your brands this competitor ad is benchmarked against.")
            force = st.checkbox("Index even if it duplicates an existing reference",
                                help="Two references for the same creative double-count every airing.")
            submitted = st.form_submit_button("Index ad", type="primary", width="stretch")
        if submitted:
            ad_type = "competitor" if role.startswith("Competitor") else "self"
            if not (up and brand.strip() and campaign.strip()):
                st.error("A file, a brand name and a campaign name are required.")
            elif ad_type == "self" and competes.strip():
                st.error("'Competes with' only applies to competitor ads; clear it or switch the role.")
            elif up.size > MAX_UPLOAD_MB * 1024 ** 2:
                st.error(f"File is {up.size / 1024 ** 2:.0f} MB; the in-memory limit is {MAX_UPLOAD_MB} MB.")
            else:
                with st.spinner("Fingerprinting…"):
                    res = ingest_bytes(up.getvalue(), up.name, brand.strip(), campaign.strip(),
                                       expected_duration=expected or None, db_path=db_path,
                                       allow_duplicate=force, industry=industry, ad_type=ad_type,
                                       target_competitor_of=competes if ad_type == "competitor" else None)
                for w in res.warnings:
                    st.warning(w)
                if res.committed:
                    st.success(f"Indexed **{brand} – {campaign}** as ad #{res.ad_id} "
                               f"({res.measured_duration:.2f} s). Live monitors pick it up within one pass.")


def scan_section(db_path):
    with st.container(key="bento-scan"):
        st.markdown(card_header("📼 Scan Broadcast Recording",
                                "Same chunked matcher as live monitoring. Times are offsets into the recording; "
                                "every new airing is stored with a SHA-256 integrity hash."), unsafe_allow_html=True)
        mode = st.radio("Recording source", ["Upload file", "Local file path"], horizontal=True,
                        help=f"Uploads are held in memory (max {MAX_UPLOAD_MB} MB). For long recordings use a "
                             f"local path: FFmpeg streams it from disk without loading it into RAM.")
        label = st.text_input("Channel / source label", placeholder="e.g. Echorouk TV 2026-10-12 20:00")
        up = path = None
        if mode == "Upload file":
            up = st.file_uploader("Recording", type=list(MEDIA_TYPES), key="rec_upload")
        else:
            path = st.text_input("Path to recording on this machine")
        if st.button("🔎 Scan recording", type="primary"):
            try:
                with st.spinner("Scanning…"):
                    if up is not None:
                        if up.size > MAX_UPLOAD_MB * 1024 ** 2:
                            raise ValueError(f"Upload is {up.size / 1024 ** 2:.0f} MB; use a local path for "
                                             f"recordings over {MAX_UPLOAD_MB} MB.")
                        name = f"{label.strip()} · {up.name}" if label.strip() else up.name
                        rows = scan_bytes(up.getvalue(), up.name, name, db_path=db_path)
                    elif path and Path(path.strip().strip('"')).expanduser().is_file():
                        p = Path(path.strip().strip('"')).expanduser()
                        name = f"{label.strip()} · {p.name}" if label.strip() else p.name
                        rows = scan_recording(str(p), name, db_path=db_path)
                    else:
                        raise ValueError("Choose a recording to scan.")
                st.session_state["last_scan"] = dict(source=name, rows=rows)
            except Exception as exc:   # bad media, unreadable path: show it, keep the app alive
                st.error(f"Scan failed: {exc}")

        scan = st.session_state.get("last_scan")
        if not scan:
            return
        rows = scan["rows"]
        st.markdown(f"**{scan['source']}**: {len(rows)} airing(s) detected · "
                    f"{sum(r['is_new'] for r in rows)} new, {sum(not r['is_new'] for r in rows)} already recorded")
        if not rows:
            return
        sdf = pd.DataFrame(rows)
        sdf["ad_label"] = sdf["brand_name"] + " – " + sdf["campaign_name"]
        sdf["start_tc"] = sdf["airing_start"].map(timecode)
        sdf["end_tc"] = sdf["airing_end"].map(timecode)
        ad_ids = sorted(set(ads["ad_id"]))
        dom = [sdf.loc[sdf["ad_id"] == a, "ad_label"].iloc[0] for a in ad_ids if a in set(sdf["ad_id"])]
        rng_ = [SERIES[ad_ids.index(a) % len(SERIES)] for a in ad_ids if a in set(sdf["ad_id"])]
        chart = alt.Chart(sdf).mark_bar(height=18, cornerRadius=4).encode(
            x=alt.X("airing_start:Q", title="Seconds into recording"), x2="airing_end:Q",
            y=alt.Y("ad_label:N", title=None),
            color=alt.Color("ad_label:N", scale=alt.Scale(domain=dom, range=rng_), legend=None),
            tooltip=[alt.Tooltip("ad_label:N", title="Ad"), alt.Tooltip("start_tc:N", title="Start"),
                     alt.Tooltip("end_tc:N", title="End"),
                     alt.Tooltip("confidence_score:Q", title="Confidence", format=".1%")],
        ).properties(height=max(60, 36 * sdf["ad_label"].nunique()))
        st.altair_chart(chart, width="stretch")
        for r in rows:
            occ = r["occurrence_id"]
            c1, c2, c3 = st.columns([4, 2, 2])
            c1.markdown(f"**{r['brand_name']} – {r['campaign_name']}**  \n"
                        f"{timecode(r['airing_start'])} → {timecode(r['airing_end'])} "
                        f"({r['duration_detected']:.2f} s)")
            c2.markdown(f"Confidence **{r['confidence_score']:.1%}**  \n"
                        f"{'🆕 new' if r['is_new'] else '↺ already recorded'} · #{occ}")
            if occ:
                integrity = occurrence_integrity(db_path, occ)
                c3.download_button("📜 Certificate (PDF)",
                                   data=certificate_pdf(db_path, occ, _review_state(db_path, occ), str(integrity)),
                                   file_name=f"KIHC-certificate-{occ:06d}.pdf", mime="application/pdf",
                                   key=f"scan_cert_{occ}", width="stretch")


def set_review(db_path, occurrence_id, value):
    with get_db(db_path) as conn:
        conn.execute("UPDATE ad_occurrences SET human_verified = ? WHERE occurrence_id = ?",
                     (value, int(occurrence_id)))


def gauge(value, color):
    data = pd.DataFrame({"part": ["confidence", "remaining"], "v": [value, 1 - value]})
    arc = alt.Chart(data).mark_arc(innerRadius=52, outerRadius=70, cornerRadius=4, padAngle=0.02).encode(
        theta=alt.Theta("v:Q", stack=True, sort=None),
        color=alt.Color("part:N", scale=alt.Scale(domain=["confidence", "remaining"], range=[color, TRACK]),
                        legend=None),
        order=alt.Order("part:N", sort="ascending"),
        tooltip=[alt.Tooltip("v:Q", title="share", format=".1%")],
    )
    label = alt.Chart(pd.DataFrame({"t": [f"{value:.1%}"]})).mark_text(fontSize=20, fontWeight="bold").encode(text="t:N")
    return (arc + label).properties(width=160, height=160)


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown(brand_html(), unsafe_allow_html=True)
st.sidebar.markdown(section_html("🎛️ Filters"), unsafe_allow_html=True)
# ?db=<path> preselects a database (e.g. a demo DB) without touching the default production one.
db_path = st.sidebar.text_input("Database", st.query_params.get("db", str(DEFAULT_DB_PATH)))
df, ads = load(db_path)

color_of = {aid: SERIES[i % len(SERIES)] for i, aid in enumerate(sorted(ads["ad_id"]))}
min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, float(MATCH_THRESHOLD), 0.01)
ad_choices = sorted(df["ad_label"].unique())
pick_ads = st.sidebar.multiselect("Ads", ad_choices, default=ad_choices)
pick_review = st.sidebar.multiselect("Review status", list(REVIEW_LABEL.values()), default=list(REVIEW_LABEL.values()))
view = df[(df["confidence_score"] >= min_conf) & df["ad_label"].isin(pick_ads) & df["review"].isin(pick_review)]
live_sidebar(db_path)
st.sidebar.markdown(engine_badge_html(read_service_status(db_path)), unsafe_allow_html=True)


# ------------------------------------------------------------------ bento overview (presentation only)

# Thin adapters over kihc/app/overview.py so this dashboard and the FastAPI web app compute identical figures.

def _detections_per_day(frame, days=14):
    per_day = ov.detections_per_day(frame.to_dict("records"), days)
    return [d["count"] for d in per_day], [pd.Timestamp(d["date"]).strftime("%b %d") for d in per_day]


def _confidence_delta(frame):
    c = ov.confidence_summary(frame.to_dict("records"))
    return c["avg"], None if c["delta_pts"] is None else c["delta_pts"] / 100, c["recent_avg"]


def _premium_share(pods):
    p = ov.premium_share(pods)
    return p["pct"], p["in_break_airings"], p["scope"]


def bento_overview():
    engine = read_service_status(db_path)
    h1, h2 = st.columns([3, 1], vertical_alignment="bottom")
    h1.markdown('<div class="hero"><div><h1>Broadcast Intelligence</h1><p>Detected airings, share of voice and '
                'break placement across monitored channels. Airings are not audience impressions.</p></div></div>',
                unsafe_allow_html=True)
    # Default to the industry with the most detected airings (alphabetical order would favour trivial 100% cases).
    industries = sorted(known_industries(db_path),
                        key=lambda i: -compute_share_of_voice(i, db_path=db_path)["total_airings"])
    industry = h2.selectbox("SOV industry", (industries or []) + ["All industries"], key="bento_industry",
                            help="Industry used for the Share of Voice card (default: most detected airings).")
    industry = None if industry == "All industries" else industry

    # ---------------- row 1: executive KPIs
    k1, k2, k3 = st.columns(3)
    counts, labels = _detections_per_day(view)
    k1.markdown(kpi_card("Total airings", f"{len(view):,}", sub="Detections logged per day, last 14 days",
                         badge=pill(f"{int(sum(counts))} in 14 days", "neutral"),
                         visual=microbar_svg(counts, labels)), unsafe_allow_html=True)
    overall, delta, _ = _confidence_delta(view)
    if overall is None:
        conf_badge = pill("no detections yet", "neutral")
    elif delta is None:
        conf_badge = pill("no prior 7-day period to compare", "neutral")
    else:
        conf_badge = pill(f"{'▲' if delta >= 0 else '▼'} {abs(delta) * 100:.1f} pts vs prior 7 days",
                          "" if delta >= 0 else "down")
    k2.markdown(kpi_card("Avg match confidence", "—" if overall is None else f"{overall * 100:.1f}", unit="%",
                         sub="1 − bit error rate · threshold 82% (files) / 80% (live)", badge=conf_badge),
                unsafe_allow_html=True)
    sov = compute_share_of_voice(industry, db_path=db_path)
    if sov["total_airings"]:
        top = sov["brands"][0]
        sov_badge = pill(f"leader: {top['brand']} {top['sov_airtime_pct']:.0f}%", "neutral")
        k3.markdown(kpi_card(f"Client share of voice · {industry or 'all industries'}",
                             f"{sov['self_sov_airtime_pct']:.1f}", unit="%",
                             sub=f"of {sov['total_airtime_sec'] / 60:.1f} min detected airtime · "
                                 f"{sov['total_airings']} airings", badge=sov_badge), unsafe_allow_html=True)
    else:
        k3.markdown(kpi_card("Client share of voice", "—", sub="Tag ads with an industry and role (📥 Ingest or "
                             "⚔️ Competitive tab) to compute SOV.", badge=pill("no tagged airings", "neutral")),
                    unsafe_allow_html=True)

    # ---------------- row 2: intelligence grid
    pods = cluster_occurrences_into_pods(db_path=db_path)
    c1, c2, c3 = st.columns([1, 1, 2])
    premium, n_in_breaks, who = _premium_share(pods)
    c1.markdown('<div class="bento-card">' + card_header("Premium pod placement", f"First + Last in break · {who}")
                + semi_donut_svg(premium, f"{n_in_breaks} in-break airings" if n_in_breaks else "no multi-ad breaks yet")
                + '</div>', unsafe_allow_html=True)
    if view.empty:
        ratios = []
    else:
        ch = view["source_name"].map(channel_of).value_counts()
        top5 = ch.head(5)
        ratios = [(name, 100.0 * n / ch.sum(), n) for name, n in top5.items()]
        if len(ch) > 5:
            ratios.append(("Other", 100.0 * ch.iloc[5:].sum() / ch.sum(), int(ch.iloc[5:].sum())))
    c2.markdown('<div class="bento-card">' + card_header("Channel distribution", "Share of filtered airings")
                + ratio_list_html(ratios) + '</div>', unsafe_allow_html=True)
    with c3.container(key="bento-clutter"):
        live_pods = [p for p in pods if p.is_live]
        if live_pods:
            per_hour = pd.DataFrame([dict(hour=datetime.fromtimestamp(p.pod_start, ALGERIA_TZ).hour,
                                          ads=p.total_ads_in_pod, dur=p.pod_duration_seconds) for p in live_pods])
            agg = per_hour.groupby("hour", as_index=False).agg(pods=("ads", "size"), avg_ads=("ads", "mean"),
                                                               avg_dur=("dur", "mean"))
            peak = agg.sort_values(["avg_ads", "pods"], ascending=False).iloc[0]
            st.markdown(card_header("Commercial break clutter by hour",
                                    f"Live breaks, Algeria time · peak intensity {int(peak['hour']):02d}:00 "
                                    f"({peak['avg_ads']:.1f} detected ads/break)"), unsafe_allow_html=True)
            chart = alt.Chart(agg).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#10B981").encode(
                x=alt.X("hour:O", title="Hour (Algeria)", scale=alt.Scale(domain=list(range(24)))),
                y=alt.Y("avg_ads:Q", title="Ads / break"),
                tooltip=[alt.Tooltip("hour:O", title="Hour"), alt.Tooltip("pods:Q", title="Breaks"),
                         alt.Tooltip("avg_ads:Q", title="Avg ads / break", format=".1f"),
                         alt.Tooltip("avg_dur:Q", title="Avg break (s)", format=".0f")])
            st.altair_chart(chart.properties(height=170).configure_view(strokeWidth=0)
                            .configure(background="transparent"), width="stretch")
        elif pods:
            cc = pd.DataFrame(channel_clutter(pods))
            st.markdown(card_header("Commercial break clutter",
                                    "Recordings only so far (no clock time): average detected ads per break"),
                        unsafe_allow_html=True)
            chart = alt.Chart(cc).mark_bar(cornerRadiusEnd=6, height=16, color="#10B981").encode(
                x=alt.X("avg_ads_per_pod:Q", title="Avg detected ads / break"), y=alt.Y("channel:N", title=None),
                tooltip=[alt.Tooltip("channel:N"), alt.Tooltip("pods:Q", title="Breaks"),
                         alt.Tooltip("avg_ads_per_pod:Q", title="Avg ads", format=".1f"),
                         alt.Tooltip("avg_pod_seconds:Q", title="Avg break (s)", format=".0f")])
            st.altair_chart(chart.properties(height=max(90, 30 * len(cc))).configure_view(strokeWidth=0)
                            .configure(background="transparent"), width="stretch")
        else:
            st.markdown(card_header("Commercial break clutter", "Appears once ads are detected"),
                        unsafe_allow_html=True)
            st.caption("No breaks reconstructed yet.")

    # ---------------- row 3: radar + certificate authority
    r1, r2 = st.columns([3, 2])
    with r1.container(key="bento-dark-radar"):
        live_status(db_path)
    with r2.container(key="bento-emerald-cert"):
        st.markdown("### 🔐 Airing Certificate Authority")
        st.caption("SHA-256 tamper-evident proof of performance for any detected airing, as a one-page PDF.")
        if view.empty:
            st.caption("No airings to certify yet.")
        else:
            recent = view.sort_values("detected_at", ascending=False).head(50)
            opts = {f"#{r.occurrence_id} · {r.ad_label} · {channel_of(r.source_name)} · {r.start_display}": r
                    for r in recent.itertuples()}
            sel = opts[st.selectbox("Airing", list(opts), key="cert_pick")]
            integrity = occurrence_integrity(db_path, int(sel.occurrence_id))
            st.markdown({True: "🔐 Integrity check: **PASS**",
                         False: "⚠️ Integrity check: **FAIL** — record or evidence changed",
                         None: "🔓 No hash recorded (detected before hashing existed)"}[integrity])
            st.download_button("📜 Generate SHA-256 Airing Certificate (PDF)",
                               data=certificate_pdf(db_path, int(sel.occurrence_id), int(sel.human_verified),
                                                    str(integrity)),
                               file_name=f"KIHC-certificate-{int(sel.occurrence_id):06d}.pdf",
                               mime="application/pdf", width="stretch", key="cert_banner_dl")


def monitor_view():
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Occurrences (filtered)", len(view))
    k2.metric("Reference ads indexed", len(ads))
    k3.metric("Human-verified", int((view["human_verified"] == 1).sum()))
    k4.metric("Rejected (hidden from analytics)", int((df["human_verified"] == -1).sum()))

    if df.empty:
        st.info("No occurrences yet. Index a creative (📥 Ingest tab) and scan a recording "
                "(📼 Recording Scanner tab), or run `python -m kihc.cli demo`.")
        return

    # -------------------------------------------------------------- timeline
    st.subheader("Detection timeline")
    domain = [a for a in ad_choices]
    rng = [color_of[int(df.loc[df["ad_label"] == a, "ad_id"].iloc[0])] for a in domain]
    tooltip = [alt.Tooltip("ad_label:N", title="Ad"), alt.Tooltip("source_name:N", title="Source"),
               alt.Tooltip("airing_start:Q", title="Start (s)", format=".3f"),
               alt.Tooltip("airing_end:Q", title="End (s)", format=".3f"),
               alt.Tooltip("confidence_score:Q", title="Confidence", format=".1%"),
               alt.Tooltip("review:N", title="Review")]
    file_view, live_view = view[~view["is_live"]], view[view["is_live"]]
    ad_color = alt.Color("ad_label:N", title="Ad", scale=alt.Scale(domain=domain, range=rng),
                         legend=alt.Legend(orient="top"))
    if not file_view.empty:
        timeline = alt.Chart(file_view.drop(columns=["start_utc", "end_utc"])).mark_bar(
            height=18, cornerRadius=4).encode(
            x=alt.X("airing_start:Q", title="Seconds from start of source file"),
            x2="airing_end:Q",
            y=alt.Y("source_name:N", title=None),
            color=ad_color,
            tooltip=tooltip,
        ).properties(height=max(90, 50 * file_view["source_name"].nunique()))
        st.altair_chart(timeline, width="stretch")
    if not live_view.empty:
        live_tip = [alt.Tooltip("ad_label:N", title="Ad"), alt.Tooltip("source_name:N", title="Channel"),
                    alt.Tooltip("start_utc:T", title="Start (UTC)", format="%Y-%m-%d %H:%M:%S"),
                    alt.Tooltip("end_utc:T", title="End (UTC)", format="%H:%M:%S"),
                    alt.Tooltip("confidence_score:Q", title="Confidence", format=".1%"),
                    alt.Tooltip("review:N", title="Review")]
        live_tl = alt.Chart(live_view).mark_bar(height=18, cornerRadius=4).encode(
            x=alt.X("start_utc:T", title="Capture time (UTC)", axis=alt.Axis(format="%b %d %H:%M", labelOverlap=True)),
            x2="end_utc:T",
            y=alt.Y("source_name:N", title=None),
            color=ad_color,
            tooltip=live_tip,
        ).properties(height=max(90, 50 * live_view["source_name"].nunique()))
        st.altair_chart(live_tl, width="stretch")
        st.caption("Live times are capture times; HLS delivery typically lags the on-air time by 10–30 s.")

    # -------------------------------------------------------------- confidence
    st.subheader("Confidence per occurrence")
    conf_chart = alt.Chart(view).mark_bar(size=14, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("occurrence_id:O", title="Occurrence"),
        y=alt.Y("confidence_score:Q", title="Confidence (1 − BER)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("ad_label:N", scale=alt.Scale(domain=domain, range=rng), legend=None),
        tooltip=tooltip,
    )
    rule = alt.Chart(pd.DataFrame({"y": [MATCH_THRESHOLD]})).mark_rule(strokeDash=[4, 4], color="#8a8984").encode(
        y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Match threshold")])
    st.altair_chart((conf_chart + rule).properties(height=220), width="stretch")
    st.caption(f"Dashed line: match threshold {MATCH_THRESHOLD:.2f}. Unrelated audio scores ≈ 0.50.")

    # -------------------------------------------------------------- table
    st.subheader("Occurrences")
    st.dataframe(view[["occurrence_id", "ad_label", "source_name", "start_display",
                       "duration_detected", "confidence_score", "review", "detected_at"]],
                 hide_index=True, width="stretch")

    # -------------------------------------------------------------- evidence + review
    st.subheader("Evidence & human review")
    if view.empty:
        st.info("No occurrences match the current filters.")
    else:
        opts = {f"#{r.occurrence_id} · {r.ad_label} · {r.source_name} @ {r.start_display}": r
                for r in view.itertuples()}
        sel = opts[st.selectbox("Occurrence", list(opts))]
        left, mid, right = st.columns([2, 1, 1])
        with left:
            ev = sel.evidence_frame_path
            if ev and Path(ev).exists():
                st.image(ev, caption=f"Evidence frame – {Path(ev).name}", width="stretch")
            else:
                st.warning("Evidence file missing for this occurrence.")
        with mid:
            st.altair_chart(gauge(float(sel.confidence_score), color_of[int(sel.ad_id)]))
            span = (f"{sel.start_utc:%Y-%m-%d %H:%M:%S} → {sel.end_utc:%H:%M:%S} UTC" if sel.is_live
                    else f"{sel.airing_start:.3f}s → {sel.airing_end:.3f}s")
            st.markdown(f"**{sel.ad_label}**  \n{span}  \n"
                        f"Detected duration {sel.duration_detected:.3f}s")
        with right:
            st.markdown(f"Status: **{sel.review}**")
            if st.button("✅ Verify", type="primary", width="stretch"):
                set_review(db_path, sel.occurrence_id, 1)
                st.rerun()
            if st.button("❌ Reject", width="stretch"):
                set_review(db_path, sel.occurrence_id, -1)
                st.rerun()
            if st.button("↺ Reset to unreviewed", width="stretch"):
                set_review(db_path, sel.occurrence_id, 0)
                st.rerun()
            integrity = occurrence_integrity(db_path, int(sel.occurrence_id))
            st.markdown({True: "🔐 Integrity: **PASS**",
                         False: "⚠️ Integrity: **FAIL** — record or evidence changed",
                         None: "🔓 Integrity: no hash recorded"}[integrity])
            st.download_button(
                "📜 Download Airing Certificate (PDF)",
                data=certificate_pdf(db_path, int(sel.occurrence_id), int(sel.human_verified), str(integrity)),
                file_name=f"KIHC-certificate-{int(sel.occurrence_id):06d}.pdf", mime="application/pdf",
                width="stretch")

    # -------------------------------------------------------------- export
    st.subheader("Export")
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download filtered CSV",
                       view.drop(columns=["ad_label", "is_live", "start_display"]).to_csv(index=False)
                       .encode("utf-8"),
                       file_name="kihc_occurrences.csv", mime="text/csv", width="stretch")
    if c2.button("💾 Save full report to kihc/reports/", width="stretch"):
        out, n = export_csv(db_path)
        c2.success(f"Wrote {n} rows to {out}")


ROLE_COLORS = {"self": SERIES[0], "competitor": SERIES[1]}   # blue = client, orange = competitors
DAYPART_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # sequential blue, light -> dark


def competitive_view():
    st.caption("**Share of Voice** = share of *detected* airtime and airings among indexed creatives on the "
               "channels and recordings KIHC monitors. It is not share of spend or audience; brands without an "
               "indexed creative are invisible to it. Rejected detections are excluded.")
    industries = known_industries(db_path)
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
    industry = f1.selectbox("Industry", ["All industries"] + industries)
    industry = None if industry == "All industries" else industry
    use_dates = f2.toggle("Filter by date (live airings only)", value=False)
    date_from = date_to = None
    if use_dates:
        today = datetime.now(ALGERIA_TZ).date()
        rng_ = f2.date_input("Date range (Algeria time)", value=(today - timedelta(days=6), today))
        if isinstance(rng_, tuple) and len(rng_) == 2:
            date_from, date_to = rng_
    client = f3.selectbox("Client brand", ["All"] + client_brands(db_path),
                          help="Show this client's own ads plus competitor ads benchmarked against it.")
    client = None if client == "All" else client
    verified_only = f4.toggle("Human-verified only", value=False)

    kw = dict(db_path=db_path, verified_only=verified_only, client_brand=client)
    sov = compute_share_of_voice(industry, None, date_from, date_to, **kw)

    surges = detect_competitor_surges(24, db_path=db_path, industry=industry)
    if surges:
        for s in surges:
            change = "new activity" if s["change_pct"] is None else f"+{s['change_pct']:.0f}%"
            st.warning(f"**{s['brand']}** competitor surge: {s['previous_airings']} → {s['current_airings']} airings "
                       f"in the last 24 h ({change})"
                       + (f" · competes with {s['target_competitor_of']}" if s["target_competitor_of"] else ""),
                       icon="📈")
    else:
        st.caption("📈 No competitor surges in the last 24 h (live airings, > 30 % over the previous 24 h, "
                   "minimum 3 airings).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Airings", sov["total_airings"])
    m2.metric("Detected airtime", f"{sov['total_airtime_sec'] / 60:.1f} min")
    m3.metric("Client SOV (airtime)", f"{sov['self_sov_airtime_pct']:.1f}%")
    m4.metric("Competitor SOV (airtime)", f"{sov['competitor_sov_airtime_pct']:.1f}%")
    if sov["excluded_undated"]:
        st.caption(f"{sov['excluded_undated']} file-scan airing(s) excluded by the date filter "
                   f"(recordings carry no broadcast date).")
    if not sov["brands"]:
        st.info("No airings for this selection. Tag reference ads with an industry and role "
                "(ingest form or the tag editor below), then monitor or scan.")
        tag_editor()
        return

    sdf = pd.DataFrame(sov["brands"])
    sdf["role_label"] = sdf["role"].map({"self": "Client", "competitor": "Competitor"})
    st.subheader("Share of Voice")
    bar = alt.Chart(sdf).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("sov_airtime_pct:Q", title="Share of detected airtime (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("brand:N", sort="-x", title=None),
        color=alt.Color("role_label:N", title="Role",
                        scale=alt.Scale(domain=["Client", "Competitor"],
                                        range=[ROLE_COLORS["self"], ROLE_COLORS["competitor"]]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("brand:N", title="Brand"), alt.Tooltip("role_label:N", title="Role"),
                 alt.Tooltip("sov_airtime_pct:Q", title="SOV airtime", format=".1f"),
                 alt.Tooltip("sov_airings_pct:Q", title="SOV airings", format=".1f"),
                 alt.Tooltip("airings:Q", title="Airings"), alt.Tooltip("airtime_sec:Q", title="Airtime (s)",
                                                                        format=".0f")],
    )
    labels = bar.mark_text(align="left", dx=4, color="#52514e").encode(
        text=alt.Text("sov_airtime_pct:Q", format=".1f"), color=alt.value("#52514e"))
    st.altair_chart((bar + labels).properties(height=max(80, 32 * len(sdf))), width="stretch")
    st.dataframe(sdf[["brand", "role_label", "airings", "airtime_sec", "sov_airtime_pct", "sov_airings_pct"]]
                 .rename(columns={"role_label": "role", "airtime_sec": "airtime (s)",
                                  "sov_airtime_pct": "SOV airtime %", "sov_airings_pct": "SOV airings %"}),
                 hide_index=True, width="stretch",
                 column_config={"SOV airtime %": st.column_config.NumberColumn(format="%.1f"),
                                "SOV airings %": st.column_config.NumberColumn(format="%.1f"),
                                "airtime (s)": st.column_config.NumberColumn(format="%.0f")})

    st.subheader("Channel placement")
    pl = pd.DataFrame(channel_placement(industry, date_from, date_to, **kw))
    channels = sorted(pl["channel"].unique())
    top = channels[:7] + (["Other"] if len(channels) > 7 else [])
    pl["channel"] = pl["channel"].where(pl["channel"].isin(channels[:7]), "Other")
    pl = pl.groupby(["brand", "role", "channel"], as_index=False)[["airings", "airtime_sec", "share_of_brand_pct"]].sum()
    place = alt.Chart(pl).mark_bar(height=18, stroke="white", strokeWidth=2).encode(
        x=alt.X("share_of_brand_pct:Q", title="Share of the brand's airtime (%)", stack="zero",
                scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("brand:N", title=None),
        color=alt.Color("channel:N", title="Channel",
                        scale=alt.Scale(domain=top, range=SERIES[:len(top)]), legend=alt.Legend(orient="top")),
        order=alt.Order("channel:N"),
        tooltip=[alt.Tooltip("brand:N", title="Brand"), alt.Tooltip("channel:N", title="Channel"),
                 alt.Tooltip("airings:Q", title="Airings"), alt.Tooltip("airtime_sec:Q", title="Airtime (s)",
                                                                        format=".0f"),
                 alt.Tooltip("share_of_brand_pct:Q", title="% of brand airtime", format=".1f")],
    ).properties(height=max(80, 32 * pl["brand"].nunique()))
    st.altair_chart(place, width="stretch")

    st.subheader("Ad frequency through the day (Algeria time)")
    dp = pd.DataFrame(daypart_distribution(industry, date_from, date_to, **kw))
    if dp.empty:
        st.caption("No live airings in this selection: time of day is only known for live monitoring "
                   "(recordings carry offsets, not clock times).")
    else:
        heat = alt.Chart(dp).mark_rect(cornerRadius=2, stroke="white", strokeWidth=2).encode(
            x=alt.X("hour:O", title="Hour of day", scale=alt.Scale(domain=list(range(24)))),
            y=alt.Y("brand:N", title=None),
            color=alt.Color("airings:Q", title="Airings", scale=alt.Scale(range=DAYPART_RAMP)),
            tooltip=[alt.Tooltip("brand:N", title="Brand"), alt.Tooltip("hour:O", title="Hour"),
                     alt.Tooltip("airings:Q", title="Airings")],
        ).properties(height=max(80, 32 * dp["brand"].nunique()))
        st.altair_chart(heat, width="stretch")
    tag_editor()


def tag_editor():
    with st.expander("🏷️ Reference ad tags (industry · role · competes with)"):
        conn = init_db(db_path)
        try:
            tags = pd.read_sql_query("SELECT ad_id, brand_name, campaign_name, industry, ad_type, "
                                     "target_competitor_of FROM reference_ads ORDER BY ad_id", conn)
        finally:
            conn.close()
        if tags.empty:
            st.caption("No reference ads indexed yet.")
            return
        edited = st.data_editor(
            tags, hide_index=True, width="stretch", key="tag_editor",
            disabled=["ad_id", "brand_name", "campaign_name"],
            column_config={"ad_type": st.column_config.SelectboxColumn("role", options=list(AD_TYPES), required=True),
                           "target_competitor_of": st.column_config.TextColumn("competes with"),
                           "industry": st.column_config.TextColumn("industry")})
        if st.button("Save tags", type="primary"):
            errors, saved = [], 0
            for old, new in zip(tags.itertuples(), edited.itertuples()):
                if (old.industry, old.ad_type, old.target_competitor_of) == \
                        (new.industry, new.ad_type, new.target_competitor_of):
                    continue
                try:
                    set_ad_tags(new.ad_id, new.industry if isinstance(new.industry, str) else None, new.ad_type,
                                new.target_competitor_of if isinstance(new.target_competitor_of, str) else None,
                                db_path=db_path)
                    saved += 1
                except ValueError as exc:
                    errors.append(f"ad #{new.ad_id}: {exc}")
            for e in errors:
                st.error(e)
            if saved:
                st.success(f"Saved tags for {saved} ad(s).")
                st.rerun()


POSITION_ORDER = ["First-in-Break", "Middle", "Last-in-Break", "Solo"]
POSITION_COLORS = [SERIES[0], "#b8b7b0", SERIES[2], SERIES[6]]      # premium slots in colour, middle muted
CLUTTER_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}


def _pod_time(p, t):
    return (datetime.fromtimestamp(t, ALGERIA_TZ).strftime("%H:%M:%S") if p.is_live else timecode(t))


def pod_view():
    st.caption("A **pod** is a run of detected ads on one channel with ≤ the max gap between them. KIHC only "
               "detects ads in its reference library: an unindexed ad inside a break splits the pod, so ad counts "
               "are a **lower bound** and First/Last mean first/last *among detected ads*. Rejected detections "
               "are ignored. Live pods use Algeria time; recordings use offsets into the file.")
    conn = init_db(db_path)
    try:
        sources = [r[0] for r in conn.execute("SELECT DISTINCT source_name FROM ad_occurrences "
                                              "WHERE human_verified != -1 ORDER BY source_name")]
    finally:
        conn.close()
    channels = sorted({channel_of(s) for s in sources if s.startswith("live:")})
    recordings = sorted(s for s in sources if not s.startswith("live:"))
    f1, f2, f3 = st.columns([3, 2, 2])
    scope = f1.selectbox("Channel / recording", ["All channels & recordings"] + channels + recordings,
                         key="pod_scope")
    scope = None if scope.startswith("All ") else scope
    day = None
    if f2.toggle("Single day (live only)", key="pod_day_toggle"):
        day = f2.date_input("Day (Algeria time)", value=datetime.now(ALGERIA_TZ).date(), key="pod_day")
    gap = f3.slider("Max gap between ads (s)", 2.0, 30.0, 12.0, 0.5, key="pod_gap")
    pods = cluster_occurrences_into_pods(scope, day, gap, db_path=db_path)
    if not pods:
        st.info("No pods in this scope yet: pods appear once ads are detected on a channel or in a recording.")
        return

    multi = [p for p in pods if p.total_ads_in_pod > 1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pods (breaks)", len(pods), help="Solo detections count as single-ad pods.")
    k2.metric("Avg detected ads / pod", f"{sum(p.total_ads_in_pod for p in pods) / len(pods):.1f}")
    k3.metric("Avg pod duration", f"{sum(p.pod_duration_seconds for p in pods) / len(pods):.0f} s")
    k4.metric("High-clutter pods", f"{100 * sum(p.clutter == 'High' for p in pods) / len(pods):.0f}%",
              help="High: > 8 ads or > 4 min · Medium: 4–8 ads · Low: < 4 ads")

    st.subheader("Clutter severity by channel")
    cc = pd.DataFrame(channel_clutter(pods))
    cc_tip = [alt.Tooltip("channel:N", title="Channel"), alt.Tooltip("pods:Q", title="Pods"),
              alt.Tooltip("avg_ads_per_pod:Q", title="Avg ads / pod", format=".1f"),
              alt.Tooltip("max_ads_per_pod:Q", title="Max ads / pod"),
              alt.Tooltip("avg_pod_seconds:Q", title="Avg pod (s)", format=".0f"),
              alt.Tooltip("overall:N", title="Clutter"), alt.Tooltip("high:Q", title="High pods"),
              alt.Tooltip("medium:Q", title="Medium pods"), alt.Tooltip("low:Q", title="Low pods")]
    clutter_color = alt.Color("overall:N", title="Clutter level",
                              scale=alt.Scale(domain=list(CLUTTER_COLORS), range=list(CLUTTER_COLORS.values())),
                              legend=alt.Legend(orient="top"))
    h = max(80, 34 * len(cc))
    c1, c2 = st.columns(2)
    ads_bar = alt.Chart(cc).mark_bar(height=18, cornerRadiusEnd=4).encode(
        x=alt.X("avg_ads_per_pod:Q", title="Average detected ads per pod"), y=alt.Y("channel:N", title=None),
        color=clutter_color, tooltip=cc_tip)
    ads_rules = alt.Chart(pd.DataFrame({"x": [MEDIUM_ADS, HIGH_ADS]})).mark_rule(
        strokeDash=[4, 4], color="#8a8984").encode(x="x:Q")
    c1.altair_chart((ads_bar + ads_rules).properties(height=h), width="stretch")
    dur_bar = alt.Chart(cc).mark_bar(height=18, cornerRadiusEnd=4).encode(
        x=alt.X("avg_pod_seconds:Q", title="Average pod duration (s)"), y=alt.Y("channel:N", title=None),
        color=clutter_color, tooltip=cc_tip)
    dur_rule = alt.Chart(pd.DataFrame({"x": [HIGH_SECONDS]})).mark_rule(strokeDash=[4, 4], color="#8a8984").encode(
        x="x:Q")
    c2.altair_chart((dur_bar + dur_rule).properties(height=h), width="stretch")
    st.caption(f"Dashed lines: Medium from {MEDIUM_ADS} ads, High above {HIGH_ADS} ads or {HIGH_SECONDS / 60:.0f} min.")

    st.subheader("Position in break by brand")
    pdist = pd.DataFrame(position_distribution(pods))
    long = pdist.melt(id_vars=["brand", "role", "airings", "premium_pct"],
                      value_vars=["first_pct", "middle_pct", "last_pct", "solo_pct"],
                      var_name="code", value_name="pct")
    long["position"] = long["code"].map({"first_pct": "First-in-Break", "middle_pct": "Middle",
                                         "last_pct": "Last-in-Break", "solo_pct": "Solo"})
    long["order"] = long["position"].map({p: i for i, p in enumerate(POSITION_ORDER)})
    pos_chart = alt.Chart(long).mark_bar(height=18, stroke="white", strokeWidth=2).encode(
        x=alt.X("pct:Q", title="Share of the brand's detected airings (%)", stack="zero",
                scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("brand:N", title=None, sort=alt.EncodingSortField("premium_pct", order="descending")),
        color=alt.Color("position:N", title="Position",
                        scale=alt.Scale(domain=POSITION_ORDER, range=POSITION_COLORS), legend=alt.Legend(orient="top")),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("brand:N", title="Brand"), alt.Tooltip("position:N", title="Position"),
                 alt.Tooltip("pct:Q", title="% of airings", format=".1f"), alt.Tooltip("airings:Q", title="Airings"),
                 alt.Tooltip("premium_pct:Q", title="Premium (First+Last) in breaks %", format=".1f")],
    ).properties(height=max(80, 34 * pdist["brand"].nunique()))
    st.altair_chart(pos_chart, width="stretch")
    st.dataframe(pdist[["brand", "role", "airings", "first", "middle", "last", "solo", "premium_pct"]]
                 .rename(columns={"premium_pct": "premium % (First+Last, multi-ad pods)"}),
                 hide_index=True, width="stretch",
                 column_config={"premium % (First+Last, multi-ad pods)": st.column_config.NumberColumn(format="%.1f")})

    st.subheader("Pod inspector")
    if not multi:
        st.caption("All detected airings are solo so far: no multi-ad breaks to inspect.")
        return
    ordered = sorted(multi, key=lambda p: (-p.total_ads_in_pod, -p.pod_start))
    labels = {f"{p.pod_id} · {p.channel} · {_pod_time(p, p.pod_start)} · {p.total_ads_in_pod} ads · "
              f"{p.pod_duration_seconds:.0f}s · {p.clutter}": p for p in ordered}
    pod = labels[st.selectbox("Break", list(labels), key="pod_pick")]
    if pod.has_overlap:
        st.warning("Two detections in this break overlap in time (e.g. a cut-down and full version both "
                   "indexed, or a false match). Review them in the Monitor tab.", icon="⚠️")
    adf = pd.DataFrame(pod.ads)
    adf["end_offset"] = adf["offset_in_pod"] + (adf["end"] - adf["start"])
    adf["label"] = adf["brand"] + " – " + adf["campaign"]
    adf["slot"] = adf["position_of"] + " · " + adf["position_label"]
    brands = sorted(adf["brand"].unique())
    bars = alt.Chart(adf).mark_bar(height=20, cornerRadius=4).encode(
        x=alt.X("offset_in_pod:Q", title=f"Seconds into break (starts {_pod_time(pod, pod.pod_start)})"),
        x2="end_offset:Q",
        y=alt.Y("slot:N", title=None, sort=alt.EncodingSortField("position_index", order="ascending")),
        color=alt.Color("brand:N", title="Brand", scale=alt.Scale(domain=brands, range=SERIES[:len(brands)]
                                                                    if len(brands) <= len(SERIES) else None),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("label:N", title="Ad"), alt.Tooltip("slot:N", title="Position"),
                 alt.Tooltip("offset_in_pod:Q", title="Starts at (s)", format=".1f"),
                 alt.Tooltip("duration:Q", title="Duration (s)", format=".1f"),
                 alt.Tooltip("confidence:Q", title="Confidence", format=".1%")])
    text = alt.Chart(adf).mark_text(align="left", dx=6, color="#52514e").encode(
        x="end_offset:Q", y=alt.Y("slot:N", sort=alt.EncodingSortField("position_index", order="ascending")),
        text="label:N")
    st.altair_chart((bars + text).properties(height=max(90, 34 * len(adf))), width="stretch")


bento_overview()

radar_tab, scan_tab, pod_tab, ci_tab, ingest_tab = st.tabs(
    ["📡 Radar & Review", "📼 Recording Scanner", "📺 Commercial Pod & Clutter Analysis",
     "⚔️ Competitive Intelligence", "📥 Ingest"])
with radar_tab:
    monitor_view()
with scan_tab:
    scan_section(db_path)
with pod_tab:
    pod_view()
with ci_tab:
    competitive_view()
with ingest_tab:
    ingest_panel(db_path)
