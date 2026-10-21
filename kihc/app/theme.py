"""Executive bento-grid theme for the KIHC dashboard (presentation only).

inject_theme() adds the CSS; the *_html / *_svg helpers build static card markup from values
the dashboard has already computed. Every dynamic string is HTML-escaped. Cards that hold
Streamlit widgets are keyed containers (st.container(key="bento-…")) styled via their
`st-key-…` class, so widgets stay native and fully functional.
"""
import html

import streamlit as st

INK, MUTED, ACCENT, ACCENT_INK, MINT = "#111827", "#6B7280", "#10B981", "#047857", "#E6F7F0"
TRACK, DARK = "#E5EBEA", "#0B1E1E"
SIDEBAR_GRADIENT = "linear-gradient(180deg, #072A27 0%, #051F1D 100%)"

CSS = f"""
<style>
  /* ---------- canvas ---------- */
  [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: #F4F7F6; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1440px; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}

  /* ---------- sidebar ---------- */
  [data-testid="stSidebar"] > div:first-child {{ background: {SIDEBAR_GRADIENT}; }}
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: #FFFFFF; }}
  [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.10); }}
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #9FC2BC; }}
  .kihc-nav {{ display:flex; flex-direction:column; gap:4px; margin: 4px 0 12px; }}
  .kihc-nav span {{ color:#CFE3DF; font-size:0.86rem; padding:7px 12px; border-radius:12px; }}
  .kihc-nav span.active {{ color:#FFFFFF; background: rgba(16,185,129,0.16);
                           box-shadow: 0 0 0 1px rgba(16,185,129,0.35), 0 0 18px rgba(16,185,129,0.18); }}
  .kihc-brand {{ display:flex; align-items:center; gap:10px; margin: 2px 0 14px; }}
  .kihc-brand .logo {{ width:34px; height:34px; border-radius:10px; background:{ACCENT}; color:#04251F;
                       display:flex; align-items:center; justify-content:center; font-weight:800; }}
  .kihc-brand .name {{ color:#FFFFFF; font-weight:700; font-size:1.05rem; line-height:1.1; }}
  .kihc-brand .sub {{ color:#9FC2BC; font-size:0.75rem; }}
  .engine-badge {{ margin-top: 18px; padding: 12px 14px; border-radius: 16px; font-size: 0.85rem;
                   background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10); color:#E8F3F1; }}
  .engine-badge.on {{ border-color: rgba(16,185,129,0.45); box-shadow: 0 0 18px rgba(16,185,129,0.15); }}
  .engine-badge small {{ display:block; color:#9FC2BC; margin-top: 2px; font-size: 0.74rem; }}

  /* ---------- bento cards ---------- */
  .bento-card, div[class*="st-key-bento-"] {{
      background: #FFFFFF; border: 1px solid {TRACK}; border-radius: 20px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.04); padding: 24px; }}
  .bento-card {{ height: 100%; min-height: 168px; }}
  div[class*="st-key-bento-dark"] {{ background: {DARK}; border-color: #13302E; color: #E8F3F1; }}
  div[class*="st-key-bento-dark"] h1, div[class*="st-key-bento-dark"] h2, div[class*="st-key-bento-dark"] h3,
  div[class*="st-key-bento-dark"] p, div[class*="st-key-bento-dark"] li,
  div[class*="st-key-bento-dark"] [data-testid="stMarkdownContainer"] {{ color: #E8F3F1; }}
  div[class*="st-key-bento-dark"] [data-testid="stCaptionContainer"] {{ color: #9FC2BC; }}
  div[class*="st-key-bento-dark"] [data-testid="stVerticalBlockBorderWrapper"],
  div[class*="st-key-bento-dark"] [data-testid="stExpander"] details {{
      background: rgba(255,255,255,0.03); border-color: rgba(255,255,255,0.10); border-radius: 16px; }}
  div[class*="st-key-bento-dark"] [data-testid="stExpander"] summary p {{ color: #E8F3F1; }}
  div[class*="st-key-bento-emerald"] {{
      background: linear-gradient(135deg, #065F46 0%, #10B981 100%); border: none; color: #FFFFFF;
      box-shadow: 0 10px 30px rgba(16,185,129,0.25); }}
  div[class*="st-key-bento-emerald"] p, div[class*="st-key-bento-emerald"] h3,
  div[class*="st-key-bento-emerald"] label, div[class*="st-key-bento-emerald"] [data-testid="stMarkdownContainer"],
  div[class*="st-key-bento-emerald"] [data-testid="stCaptionContainer"] {{ color: #FFFFFF; }}
  div[class*="st-key-bento-emerald"] [data-testid="stDownloadButton"] button {{
      background: #FFFFFF; color: #065F46; border: none; font-weight: 700; }}
  div[class*="st-key-bento-emerald"] [data-testid="stDownloadButton"] button p {{ color: #065F46; font-weight: 700; }}
  div[class*="st-key-bento-emerald"] [data-testid="stDownloadButton"] button:hover {{ background: #ECFDF5; }}

  .card-label {{ color: {MUTED}; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.02em;
                 text-transform: uppercase; margin-bottom: 6px; }}
  .card-title {{ color: {INK}; font-size: 1.02rem; font-weight: 700; margin-bottom: 2px; }}
  .card-sub {{ color: {MUTED}; font-size: 0.82rem; }}
  .metric {{ color: {INK}; font-size: 2.2rem; font-weight: 700; line-height: 1.1; margin: 4px 0 6px; }}
  .metric small {{ font-size: 1rem; color: {MUTED}; font-weight: 600; margin-left: 4px; }}
  .kpi-row {{ display:flex; align-items:flex-end; justify-content:space-between; gap: 12px; }}
  .pill {{ display:inline-block; background: {MINT}; color: {ACCENT_INK}; border-radius: 12px;
           font-size: 0.8rem; font-weight: 600; padding: 3px 10px; }}
  .pill.neutral {{ background: #EEF2F1; color: #4B5563; }}
  .pill.down {{ background: #FEECEC; color: #B42318; }}
  .pill.dark {{ background: rgba(16,185,129,0.15); color: #6EE7B7; }}
  .ratio-list {{ list-style:none; padding:0; margin: 10px 0 0; }}
  .ratio-list li {{ margin: 0 0 12px; }}
  .ratio-list .row {{ display:flex; justify-content:space-between; font-size:0.88rem; color:{INK}; }}
  .ratio-list .row b {{ font-weight: 700; }}
  .ratio-list .bar {{ height: 8px; border-radius: 6px; background: {TRACK}; margin-top: 5px; overflow:hidden; }}
  .ratio-list .bar span {{ display:block; height:100%; border-radius: 6px; background: {ACCENT}; }}
  .hero {{ display:flex; align-items:flex-end; justify-content:space-between; gap: 16px; flex-wrap: wrap;
           margin-bottom: 6px; }}
  .hero h1 {{ font-size: 1.9rem; font-weight: 800; margin: 0; padding: 0; }}
  .hero p {{ color: {MUTED}; margin: 4px 0 0; font-size: 0.92rem; }}

  /* ---------- tabs as floating pills (Streamlit >= 1.6x: role=tablist / stTab; older: baseweb) ---------- */
  [data-testid="stTabs"] [role="tablist"] {{ gap: 8px; background: transparent; border: none !important;
      box-shadow: none !important; flex-wrap: wrap; padding: 6px 0 12px; }}
  [data-testid="stTabs"] [role="tablist"]::after, [data-testid="stTabs"] [role="tablist"]::before {{ display: none; }}
  [data-testid="stTabs"] [role="tab"] {{ background: #FFFFFF; border: 1px solid {TRACK} !important;
      border-radius: 999px; padding: 8px 18px !important; height: auto; margin: 0 !important;
      box-shadow: 0 2px 10px rgba(0,0,0,0.03); transition: background .15s, border-color .15s; }}
  [data-testid="stTabs"] [role="tab"]:hover {{ border-color: {ACCENT} !important; }}
  [data-testid="stTabs"] [role="tab"] p {{ font-weight: 600; color: #374151; }}
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ background: #0C1A1A; border-color: #0C1A1A !important;
      box-shadow: 0 6px 18px rgba(12,26,26,0.18); }}
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: #FFFFFF; }}
  [data-testid="stTabs"] [role="tab"]::after, [data-testid="stTabs"] [role="tab"]::before {{ display: none !important; }}
  [data-testid="stTabs"] [data-baseweb="tab-highlight"],
  [data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}

  /* ---------- native metrics / buttons ---------- */
  [data-testid="stMetricValue"] {{ font-weight: 700; color: {INK}; }}
  .stButton > button, .stDownloadButton > button {{ border-radius: 12px; }}

  @media (max-width: 640px) {{
    .block-container {{ padding-left: 16px; padding-right: 16px; }}
    .metric {{ font-size: 1.8rem; }}
    .bento-card, div[class*="st-key-bento-"] {{ padding: 18px; border-radius: 18px; }}
  }}
</style>
"""


def inject_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pill(text, tone: str = "") -> str:
    return f'<span class="pill {esc(tone)}">{esc(text)}</span>'


def microbar_svg(values, labels=None, width=132, height=40, color=ACCENT) -> str:
    """Tiny bar chart; each bar has a native hover tooltip (<title>)."""
    values = list(values) or [0]
    labels = list(labels) if labels else [str(i) for i in range(len(values))]
    n, top = len(values), max(max(values), 1)
    gap = 3
    bw = max(2.0, (width - gap * (n - 1)) / n)
    bars = []
    for i, (v, lab) in enumerate(zip(values, labels)):
        h = max(2.0, (height - 2) * v / top) if v else 2.0
        x = i * (bw + gap)
        fill = color if v else TRACK
        bars.append(f'<rect x="{x:.1f}" y="{height - h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" '
                    f'fill="{fill}"><title>{esc(lab)}: {esc(v)}</title></rect>')
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="trend">{"".join(bars)}</svg>')


def semi_donut_svg(pct, label: str = "", size: int = 220, color=ACCENT) -> str:
    """Semi-circular gauge for a single 0-100 value (None -> empty track with an em dash)."""
    r, sw = size / 2 - 14, 18
    cx, cy = size / 2, size / 2 + 4
    length = 3.14159265 * r
    frac = 0.0 if pct is None else max(0.0, min(1.0, pct / 100.0))
    path = f"M {cx - r:.1f} {cy:.1f} A {r:.1f} {r:.1f} 0 0 1 {cx + r:.1f} {cy:.1f}"
    value = "—" if pct is None else f"{pct:.0f}%"
    return (f'<svg width="100%" viewBox="0 0 {size} {size / 2 + 24:.0f}" role="img" '
            f'aria-label="{esc(label)} {esc(value)}" style="max-width:{size}px;display:block;margin:auto">'
            f'<path d="{path}" fill="none" stroke="{TRACK}" stroke-width="{sw}" stroke-linecap="round"/>'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-dasharray="{frac * length:.1f} {length:.1f}"/>'
            f'<text x="{cx:.1f}" y="{cy - 10:.1f}" text-anchor="middle" font-size="34" font-weight="700" '
            f'fill="{INK}">{esc(value)}</text>'
            f'<text x="{cx:.1f}" y="{cy + 14:.1f}" text-anchor="middle" font-size="12" fill="{MUTED}">'
            f'{esc(label)}</text></svg>')


def kpi_card(label, value, unit="", sub="", badge="", visual="") -> str:
    """badge / visual are pre-built HTML from pill() / microbar_svg()."""
    unit_html = f"<small>{esc(unit)}</small>" if unit else ""
    return (f'<div class="bento-card"><div class="card-label">{esc(label)}</div>'
            f'<div class="kpi-row"><div><div class="metric">{esc(value)}{unit_html}</div>{badge}</div>'
            f'<div>{visual}</div></div><div class="card-sub" style="margin-top:10px">{esc(sub)}</div></div>')


def ratio_list_html(items) -> str:
    """items: [(name, pct, count)] -> labelled horizontal ratio bars."""
    if not items:
        return '<div class="card-sub">No detections in the current filters.</div>'
    rows = "".join(
        f'<li><div class="row"><span>{esc(name)}</span><span><b>{pct:.0f}%</b> · {esc(count)}</span></div>'
        f'<div class="bar"><span style="width:{max(0.0, min(100.0, pct)):.1f}%"></span></div></li>'
        for name, pct, count in items)
    return f'<ul class="ratio-list">{rows}</ul>'


def card_header(title, sub="") -> str:
    return f'<div class="card-title">{esc(title)}</div>' + (f'<div class="card-sub">{esc(sub)}</div>' if sub else "")


def brand_html() -> str:
    return ('<div class="kihc-brand"><div class="logo">K</div><div><div class="name">KIHC</div>'
            '<div class="sub">Broadcast Intelligence</div></div></div>')


def section_html(title: str, active: bool = True) -> str:
    """Sidebar section header rendered as a glowing pill (labels a section; not a fake nav link)."""
    return f'<div class="kihc-nav"><span class="{"active" if active else ""}">{esc(title)}</span></div>'


def engine_badge_html(engine) -> str:
    if engine and engine.get("active"):
        return (f'<div class="engine-badge on">🟢 24/7 Engine Active'
                f'<small>{esc(engine.get("channels", 0))} channel(s) · heartbeat '
                f'{esc(round(engine.get("heartbeat_age_s") or 0))}s ago · {esc(round(engine.get("rss_mb") or 0))} MB</small>'
                f'</div>')
    if engine and engine.get("state") in ("starting", "running"):
        return ('<div class="engine-badge">🟠 Engine not responding'
                f'<small>last heartbeat {esc(round(engine.get("heartbeat_age_s") or 0))}s ago</small></div>')
    return ('<div class="engine-badge">⚪ 24/7 Engine offline'
            '<small>start: scripts\\start_monitoring_service.bat</small></div>')
