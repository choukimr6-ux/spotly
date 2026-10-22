/* Spotly SPA (Phase 33 light SaaS design): in-place polling (no page reloads), Chart.js updates without re-creation.
 * All server strings are written with textContent / created elements, never innerHTML. */
(() => {
  "use strict";
  const MINT = "#10B981", INDIGO = "#4F46E5", INDIGO_SOFT = "#A5B4FC", TRACK = "#E2E8F0", GRID = "#F1F5F9", INK = "#0F172A", MUTED = "#64748B";
  const POLL = { telemetry: 3000, stats: 5000, airings: 10000, discovery: 10000, channels: 60000,
                 compliance: 30000, alerts: 15000, sov: 60000, setup: 60000 };
  const $ = (id) => document.getElementById(id);
  const charts = {};
  const state = { industry: "", channels: [], airings: [], engineActive: false, online: true, busy: false,
                  finMode: "gross", lastFin: null };

  // ------------------------------------------------------------ helpers
  async function api(path, opts = {}) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), opts.timeout || 8000);
    try {
      const r = await fetch(path, { ...opts, signal: ctl.signal, headers: { "Accept": "application/json", ...(opts.headers || {}) } });
      const body = r.headers.get("content-type")?.includes("json") ? await r.json() : null;
      if (!r.ok) throw Object.assign(new Error(body?.detail || r.statusText), { status: r.status });
      setOnline(true);
      return body;
    } catch (e) {
      if (e.name === "AbortError" || e instanceof TypeError) setOnline(false);
      throw e;
    } finally { clearTimeout(t); }
  }
  // Nothing raw reaches the screen: no "undefined"/"NaN", and internal asset codes (AL24NEWS_AD_005) become "Spot 005".
  const RAW_CODE = /\b[A-Z0-9]+_AD_0*(\d+)\b/g;
  function clean(text) {
    return String(text ?? "").replace(RAW_CODE, "Spot $1").replace(/\bundefined\b/g, "").replace(/\bNaN\b/g, "—")
      .replace(/\bnull\b/g, "").replace(/\s{2,}/g, " ").trim();
  }
  function setText(el, text) { text = clean(text); if (el && el.textContent !== text) el.textContent = text; }
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
  const TONES = { good: "bg-emerald-50 text-emerald-700", neutral: "bg-slate-100 text-slate-600", bad: "bg-rose-50 text-rose-700",
                  dark: "bg-emerald-50 text-emerald-700", warn: "bg-amber-50 text-amber-700", info: "bg-indigo-50 text-indigo-700",
                  violet: "bg-violet-50 text-violet-700" };
  function setPill(el, text, tone) {
    el.className = `pill ${TONES[tone] || TONES.neutral}`;
    setText(el, text);
  }
  function setOnline(on) {
    if (state.online === on) return;
    state.online = on;
    $("offline-banner").classList.toggle("hidden", on);
    $("conn-dot").className = `w-2 h-2 rounded-full ${on ? "bg-emerald-500" : "bg-amber-400"}`;
  }
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = clean(text); return e; }
  function icons() { if (window.lucide) window.lucide.createIcons(); }

  // Polling loop: setTimeout chain (no overlap). A hidden tab keeps polling, slowly (browsers throttle timers
  // there anyway); coming back refreshes immediately. Failures never stop the loop (auto-reconnect).
  const HIDDEN_EVERY = 30000;
  function loop(fn, base) {
    let timer = null, every = base;
    const run = async () => {
      timer = null;
      try { await fn(); } catch (e) { console.debug("poll failed:", fn.name, e); /* offline banner covers the network case; next tick retries */ }
      every = document.hidden ? Math.max(base, HIDDEN_EVERY) : base;
      timer = setTimeout(run, every);
    };
    document.addEventListener("visibilitychange", () => { if (!document.hidden && timer) { clearTimeout(timer); run(); } });
    run();
  }

  // ------------------------------------------------------------ charts (created once, then updated in place)
  // Lazily: Chart.js loads async, so the first stats may arrive before it does (numbers render regardless).
  function ensureCharts() {
    if (charts.perday) return true;
    if (!window.Chart) return false;
    Chart.defaults.font.family = "ui-sans-serif, system-ui, Segoe UI, Helvetica Neue, Arial, sans-serif";
    Chart.defaults.color = MUTED;
    charts.perday = new Chart($("chart-perday"), {
      type: "bar", data: { labels: [], datasets: [{ data: [], backgroundColor: INDIGO_SOFT, hoverBackgroundColor: INDIGO, borderRadius: 4, borderSkipped: false, barPercentage: 0.75, categoryPercentage: 0.9 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false }, tooltip: { displayColors: false } },
        scales: { x: { display: false }, y: { display: false, beginAtZero: true } } } });
    charts.premium = new Chart($("chart-premium"), {
      type: "doughnut", data: { labels: ["First + Last", "Middle"], datasets: [{ data: [0, 1], backgroundColor: [INDIGO, TRACK], borderWidth: 0, borderRadius: 6 }] },
      options: { responsive: true, maintainAspectRatio: false, circumference: 180, rotation: -90, cutout: "74%",
        animation: { duration: 500 }, plugins: { legend: { display: false },
          tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.raw} airings` } } } } });
    charts.clutter = new Chart($("chart-clutter"), {
      type: "bar", data: { labels: [], datasets: [{ label: "Avg detected ads / break", data: [], backgroundColor: "#818CF8", hoverBackgroundColor: INDIGO, borderRadius: 8, borderSkipped: false, maxBarThickness: 26 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
        plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c) => {
          const r = c.dataset.meta?.[c.dataIndex]; return r ? `${r.pods} breaks · avg ${Math.round(r.avg_seconds)} s` : ""; } } } },
        scales: { x: { grid: { display: false } }, y: { beginAtZero: true, border: { display: false }, grid: { color: GRID }, title: { display: true, text: "Ads / break" } } } } });
    return true;
  }

  // ------------------------------------------------------------ stats
  async function loadStats() {
    const q = state.industry ? `?industry=${encodeURIComponent(state.industry)}` : "";
    renderStats(await api(`/api/stats${q}`));
  }

  function renderStats(s) {
    state.lastStats = s;
    ensureCharts();
    // KPI 1
    setText($("kpi-total"), s.total_airings.toLocaleString());
    const recent = s.per_day.reduce((a, d) => a + d.count, 0);
    setPill($("kpi-total-badge"), s.total_airings === 0 ? "Analyzing live blocks…" : s.total_airings === 1 ? "First airing logged"
            : `${recent} in the last ${s.per_day.length} days`, s.total_airings ? "info" : "neutral");
    if (charts.perday) {
      charts.perday.data.labels = s.per_day.map((d) => d.date);
      charts.perday.data.datasets[0].data = s.per_day.map((d) => d.count);
      charts.perday.update("none");
    }
    // KPI 2
    const c = s.confidence;
    const noConf = c.avg === null;
    setText($("kpi-conf"), noConf ? "Analyzing…" : fmt(c.avg * 100));
    setText($("kpi-conf-unit"), noConf ? "" : "%"); setText($("kpi-conf-tail"), noConf ? "" : " match");
    $("kpi-conf-pill").className = `pill ${noConf ? TONES.neutral : TONES.good}`;
    if (c.delta_pts === null) setPill($("kpi-conf-delta"), noConf ? "No detections yet" : "Baseline forming", "neutral");
    else setPill($("kpi-conf-delta"), `${c.delta_pts >= 0 ? "▲" : "▼"} ${fmt(Math.abs(c.delta_pts))} pts vs prior 7 days`, c.delta_pts >= 0 ? "good" : "bad");
    // KPI 3 + industry selector
    syncIndustries(s.sov);
    setText($("kpi-sov-industry"), s.sov.industry || "all industries");
    if (s.sov.total_airings) {
      setText($("kpi-sov"), fmt(s.sov.client_pct));
      setPill($("kpi-sov-leader"), s.sov.leader ? (Math.round(s.sov.leader.pct) >= 100 ? `Solo spot · ${s.sov.leader.brand}`
              : `Leader · ${s.sov.leader.brand} ${Math.round(s.sov.leader.pct)}%`) : "Building baseline", "violet");
      setText($("kpi-sov-sub"), `of ${fmt(s.sov.airtime_min)} min detected airtime · ${s.sov.total_airings} airings`);
    } else {
      setText($("kpi-sov"), "—");
      setPill($("kpi-sov-leader"), "Awaiting tagged ads", "neutral");
      setText($("kpi-sov-sub"), "Tag ads with an industry and role to compute your share of voice.");
    }
    // Premium gauge
    const p = s.premium;
    setText($("premium-sub"), `First + Last in break · ${p.scope}`);
    setText($("premium-pct"), p.pct === null ? "—" : `${Math.round(p.pct)}%`);
    setText($("premium-n"), p.in_break_airings ? `${p.in_break_airings} in-break airings` : "Analyzing breaks…");
    if (charts.premium) {
      charts.premium.data.datasets[0].data = p.in_break_airings ? [p.premium_airings, p.in_break_airings - p.premium_airings] : [0, 1];
      charts.premium.update();
    }
    renderChannels(s.channels);
    renderClutter(s.clutter);
    if (s.financials) renderFinancials(s.financials);
    setText($("updated"), `updated ${new Date().toLocaleTimeString()}`);
  }

  // ------------------------------------------------------------ financial estimates
  const dzd = (v) => (v === null || v === undefined) ? "—" : Math.round(v).toLocaleString("en-US");
  function renderFinancials(f) {
    state.lastFin = f;
    const net = state.finMode === "net";
    document.querySelectorAll(".fin-mode").forEach((b) => {
      const on = b.dataset.finMode === (net ? "net" : "gross");
      b.className = `fin-mode px-2.5 py-0.5 rounded-full font-semibold transition-colors ${on ? "bg-indigo-600 text-white shadow-sm" : "text-slate-500 hover:text-slate-800"}`;
      b.setAttribute("aria-pressed", String(on));
    });
    const st = $("fin-status");
    st.className = `pill ${f.rate_card.illustrative ? TONES.warn : TONES.good}`;
    setText(st, f.rate_card.illustrative ? "Illustrative rate card" : "Verified rate card");
    st.title = (f.rate_card.errors || []).join("\n") || `sources: ${(f.rate_card.sources || []).join(" → ")}`;
    // Card A
    setText($("fin-basis-label"), net ? "(est. net)" : "(gross)");
    setText($("fin-total"), dzd(net ? f.estimated_net_spend : f.gross_spend));
    setPill($("fin-total-badge"), f.priced_airings ? `${f.priced_airings} priced airing${f.priced_airings === 1 ? "" : "s"}` : "Awaiting priced airings", f.priced_airings ? "info" : "neutral");
    setText($("fin-total-sub"), net ? `gross ${dzd(f.gross_spend)} DZD before agency discount`
      : `est. net ${dzd(f.estimated_net_spend)} DZD after agency discount` +
        (f.unpriced_airings ? ` · ${f.unpriced_airings} unpriced (no broadcast time)` : ""));
    // Card B
    const top = f.top_spender;
    setText($("fin-top"), top ? top.brand : "Analyzing…");
    setPill($("fin-sos"), top ? `SOS ${fmt(top.sos_pct)}%` : "SOS —", "violet");
    setPill($("fin-sov"), top ? `SOV ${fmt(top.sov_pct)}%` : "SOV —", "neutral");
    setText($("fin-top-sub"), top ? `${dzd(net ? top.estimated_net_spend : top.gross_spend)} DZD ${net ? "est. net" : "gross"} · ` +
      "SOS = share of spend, SOV = share of airtime" : "Appears with the first priced airing.");
    // Card C
    setText($("fin-credit"), dzd(f.contract_loss_credit));
    if (f.breaches) setPill($("fin-credit-badge"), `💰 ${f.breaches} breach${f.breaches === 1 ? "" : "es"} to claim`, "bad");
    else setPill($("fin-credit-badge"), "All placements on track", "good");
    setText($("fin-credit-sub"), "Value of booked First/Last positions not delivered · claimable pending as-run verification" +
      (f.bonus_premium_value ? ` · bonus premium received ${dzd(f.bonus_premium_value)} DZD` : ""));
  }

  function syncIndustries(sov) {
    const sel = $("industry-select");
    const want = [...sov.industries, ""];
    const have = [...sel.options].map((o) => o.value);
    if (JSON.stringify(want) !== JSON.stringify(have)) {
      sel.replaceChildren(...want.map((v) => { const o = el("option", "", v || "All industries"); o.value = v; return o; }));
    }
    if (!state.industry && sov.industry) sel.value = sov.industry;
  }

  function renderChannels(rows) {
    const list = $("channel-list");
    if (!rows.length) { list.replaceChildren(el("li", "text-sm text-muted", "Monitoring active · waiting for the first airings.")); return; }
    const items = rows.map((r) => {
      const li = el("li");
      const head = el("div", "flex justify-between text-sm");
      head.append(el("span", "truncate pr-2", r.name));
      const val = el("span", "num whitespace-nowrap"); val.append(el("b", "", `${Math.round(r.pct)}%`), document.createTextNode(` · ${r.count}`));
      head.append(val);
      const bar = el("div", "h-2 rounded-full bg-slate-100 mt-1.5 overflow-hidden");
      const fill = el("div", "bar-fill h-full rounded-full bg-gradient-to-r from-indigo-400 to-indigo-500"); fill.style.width = `${Math.min(100, r.pct)}%`;
      bar.append(fill); li.append(head, bar); return li;
    });
    list.replaceChildren(...items);
  }

  function renderClutter(cl) {
    if (!charts.clutter) return;
    const ds = charts.clutter.data.datasets[0];
    if (cl.mode === "hourly") {
      const byHour = Object.fromEntries(cl.rows.map((r) => [r.hour, r]));
      charts.clutter.data.labels = [...Array(24).keys()].map((h) => `${String(h).padStart(2, "0")}h`);
      ds.data = [...Array(24).keys()].map((h) => byHour[h] ? +byHour[h].avg_ads.toFixed(2) : 0);
      ds.meta = [...Array(24).keys()].map((h) => byHour[h] || null);
      setText($("clutter-sub"), `Live breaks by hour (Algeria time) · peak intensity ${String(cl.peak.hour).padStart(2, "0")}:00 (${fmt(cl.peak.avg_ads)} detected ads/break)`);
    } else if (cl.mode === "recordings") {
      charts.clutter.data.labels = cl.rows.map((r) => r.channel);
      ds.data = cl.rows.map((r) => +r.avg_ads.toFixed(2));
      ds.meta = cl.rows;
      setText($("clutter-sub"), "Recordings only so far (no clock time): average detected ads per break");
    } else {
      charts.clutter.data.labels = []; ds.data = []; ds.meta = [];
      setText($("clutter-sub"), "Appears once commercial breaks are reconstructed from detections.");
    }
    charts.clutter.update("none");
    $("clutter-empty").classList.toggle("hidden", ds.data.length > 0 && ds.data.some((v) => v > 0));
  }

  // ------------------------------------------------------------ telemetry / radar
  const STATE_LABEL = { listening: "Listening", connecting: "Connecting…", backoff: "Reconnecting", stopped: "Stopped", idle: "Idle" };
  const radarRows = new Map();

  async function loadTelemetry() {
    const t = await api("/api/telemetry");
    renderEngine(t.engine);
    const list = $("radar-list"), seen = new Set();
    for (const ch of t.channels) {
      const key = `${ch.source}:${ch.name}`; seen.add(key);
      let row = radarRows.get(key);
      if (!row) { row = makeRadarRow(ch); radarRows.set(key, row); list.append(row.root); icons(); }
      updateRadarRow(row, ch);
    }
    for (const [key, row] of radarRows) if (!seen.has(key)) { row.root.remove(); radarRows.delete(key); }
    $("radar-empty").classList.toggle("hidden", t.channels.length > 0);
    setPill($("radar-count"), `${t.channels.length} channel${t.channels.length === 1 ? "" : "s"}`, "dark");
    renderFleet(t.channels);
    renderMonitors(t.channels.filter((c) => c.source === "app"));
    renderStreamPill(t.channels);
  }

  // KPI 1: how many monitored channels are really delivering audio right now (link "live", state "listening")
  function renderFleet(channels) {
    const live = channels.filter((c) => c.state === "listening" && c.link !== "stalled" && !c.offline).length, total = channels.length;
    setText($("kpi-live"), String(live));
    if (!total) { setPill($("kpi-live-badge"), "No channels yet", "neutral"); setText($("kpi-live-sub"), "Start the 24/7 engine to begin monitoring."); return; }
    setPill($("kpi-live-badge"), live === total ? "Fleet healthy" : live ? `${live} of ${total} connected` : "Reconnecting…",
            live === total ? "good" : live ? "warn" : "bad");
    const waiting = channels.filter((c) => !(c.state === "listening" && c.link !== "stalled" && !c.offline)).map((c) => c.name);
    setText($("kpi-live-sub"), waiting.length ? `Reconnecting: ${waiting.join(", ")}` : `All ${total} monitored channels are streaming audio.`);
  }

  function renderStreamPill(channels) {
    const pill = $("stream-pill");
    const live = channels.find((c) => c.state === "listening");
    const pending = channels.find((c) => c.state === "connecting" || c.state === "backoff");
    if (live) {
      setPill(pill, `🟢 ${live.name} Live (${live.feed_label || "stream"})`, "good");
    } else if (pending) {
      const why = pending.state === "backoff" && pending.next_retry_at
        ? ` · retry in ${Math.max(0, Math.round(pending.next_retry_at - Date.now() / 1000))}s` : "";
      setPill(pill, `🟠 ${pending.name} reconnecting${why}`, "warn");
      pill.title = pending.last_error || "";
    } else {
      setPill(pill, "⚪ No live feed", "neutral");
    }
    const d = channels.map((c) => c.discovery).find((x) => x);
    if (d) {
      const hist = d.history_seconds >= 3600 ? `${fmt(d.history_seconds / 3600)} h` : `${Math.round(d.history_seconds / 60)} min`;
      setPill($("disc-stats"), `listening history ${hist} · ${d.assets} asset${d.assets === 1 ? "" : "s"} (${d.unclassified} unclassified)` +
        (d.last_discovery_at ? ` · last found ${d.last_discovery_at.slice(11, 16)} UTC` : ""), "good");
      $("disc-stats").classList.remove("mt-2");
    }
  }

  // ------------------------------------------------------------ creative intelligence cards (keyed by id: audio keeps playing)
  const discRows = new Map();
  const DISC_FILTERS = { open: (a) => a.status === "unclassified", done: (a) => a.status !== "unclassified", all: () => true };
  let lastAssets = [];
  async function loadDiscovery() {
    const r = await api("/api/discovery/assets");
    lastAssets = r.assets;
    renderDiscovery();
  }
  function renderDiscovery() {
    const body = $("disc-body");
    const keep = DISC_FILTERS[$("disc-filter")?.value || "open"] || DISC_FILTERS.all;
    // reviewable first: identified brands, then the rest by most recent airing (server order)
    const shown = lastAssets.filter(keep).sort((x, y) => (y.status === "unclassified" && !!y.suggested_brand) -
                                                         (x.status === "unclassified" && !!x.suggested_brand));
    if (!lastAssets.length) return;
    if (!discRows.size) body.replaceChildren();
    const seen = new Set();
    shown.forEach((a, i) => {
      seen.add(a.asset_id);
      let row = discRows.get(a.asset_id);
      if (!row) { row = makeDiscRow(a); discRows.set(a.asset_id, row); }
      if (body.children[i] !== row.tr) body.insertBefore(row.tr, body.children[i] || null);
      updateDiscRow(row, a);
    });
    for (const [id, row] of discRows) if (!seen.has(id)) { row.tr.remove(); discRows.delete(id); }
    if (!shown.length && !body.querySelector("[data-empty]")) {
      const p = el("p", "text-sm text-muted", "Nothing in this view."); p.dataset.empty = "1"; body.append(p);
    } else if (shown.length) body.querySelector("[data-empty]")?.remove();
  }
  $("disc-filter")?.addEventListener("change", renderDiscovery);

  const SCORE_PARTS = [["visual_dominance", "Brand visual dominance"], ["av_sync", "Audio-visual sync"],
                       ["pod_quality", "Pod quality"], ["identification", "Brand identification"]];
  const SOURCE_LABEL = { audio: "heard", packshot: "read on screen", logo: "logo match", "same packshot": "same packshot as a classified ad" };
  const scoreTone = (s) => s === null || s === undefined ? "neutral" : s >= 70 ? "good" : s >= 40 ? "warn" : "bad";

  async function post(path, payload, timeout) {
    return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), timeout });
  }
  function refreshAfterClassify() { loadStats().catch(() => {}); loadAirings().catch(() => {}); loadDiscovery().catch(() => {}); }

  // internal asset codes never reach the UI: "AL24NEWS_AD_005" -> "Spot 005"
  const spotName = (assetId) => { const m = /_(\d+)$/.exec(String(assetId || "")); return m ? `Spot ${m[1]}` : "Creative"; };
  // decorative, deterministic soft waveform (seeded by the asset id) shown behind / instead of a missing packshot
  function waveGraphic(seed) {
    const ns = "http://www.w3.org/2000/svg";
    let h = 0; for (const ch of String(seed)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 160 60"); svg.setAttribute("class", "absolute inset-x-4 top-3 bottom-3 w-[calc(100%-2rem)] h-[calc(100%-1.5rem)]");
    svg.setAttribute("preserveAspectRatio", "none"); svg.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 32; i++) {
      h = (h * 1103515245 + 12345) >>> 0;
      const v = 0.2 + 0.8 * Math.abs(Math.sin(i * 0.35 + (h % 7))) * (0.5 + ((h >> 8) % 50) / 100);
      const r = document.createElementNS(ns, "rect"), bh = Math.max(4, v * 52);
      r.setAttribute("x", String(i * 5 + 1)); r.setAttribute("y", String((60 - bh) / 2)); r.setAttribute("width", "3"); r.setAttribute("height", String(bh)); r.setAttribute("rx", "1.5");
      r.setAttribute("fill", i % 5 === 0 ? "#818CF8" : "#C7D2FE"); svg.append(r);
    }
    return svg;
  }

  function makeDiscRow(a) {
    const id = encodeURIComponent(a.asset_id);
    const tr = el("article", "lift rounded-2xl border border-line bg-white overflow-hidden flex flex-col");
    // ---- media header: packshot thumbnail, or a soft waveform graphic when no frame was captured yet
    const media = el("div", "relative aspect-video m-3 mb-0 rounded-xl overflow-hidden bg-gradient-to-br from-indigo-50 via-white to-violet-50 border border-indigo-100/70 flex items-center justify-center");
    media.append(waveGraphic(a.asset_id));
    const shot = el("a", "absolute inset-0 hidden"); shot.target = "_blank"; shot.rel = "noopener";
    const shotImg = el("img", "w-full h-full object-cover"); shotImg.loading = "lazy"; shotImg.alt = `${spotName(a.asset_id)} packshot`;
    shot.append(shotImg);
    const placeholder = el("div", "absolute bottom-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-white/85 px-2.5 py-0.5 text-indigo-500 text-[11px] font-medium shadow-sm");
    const logoWrap = el("div", "absolute top-2 left-2 hidden rounded-lg bg-white/95 p-1 shadow-sm");
    const logoImg = el("img", "h-8 max-w-[7rem] object-contain"); logoImg.alt = "brand region"; logoWrap.append(logoImg);
    const health = el("span", "pill absolute top-2 right-2 shadow");
    const durBadge = el("span", "absolute bottom-2 left-2 rounded-full bg-white/90 text-slate-700 border border-line text-[11px] font-semibold px-2.5 py-0.5 shadow-sm");
    media.append(placeholder, shot, logoWrap, health, durBadge);
    // ---- body
    const bodyEl = el("div", "p-4 flex flex-col gap-2 flex-1");
    const head = el("div", "flex items-center justify-between gap-2");
    const idEl = el("span", "text-[11px] font-semibold text-slate-400 uppercase tracking-wide", spotName(a.asset_id));
    const status = el("span", "pill");
    head.append(idEl, status);
    const tag = el("div", "font-bold text-sm leading-snug text-ink");
    const tagNote = el("div", "text-[11px] text-muted -mt-1");
    const review = el("div", "text-xs font-semibold text-amber-700 bg-amber-50 rounded-lg px-2 py-1 hidden");
    const meta = el("div", "text-[11px] text-muted");
    const scores = el("div", "grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]");
    const bars = {};
    SCORE_PARTS.forEach(([k, label]) => {
      const cell = el("div"); const top = el("div", "flex justify-between"); const name = el("span", "text-muted", label);
      const val = el("span", "num font-semibold"); top.append(name, val);
      const track = el("div", "h-1.5 rounded-full bg-slate-100 overflow-hidden mt-0.5"); const fill = el("div", "bar-fill h-full rounded-full bg-indigo-400"); fill.style.width = "0%";
      track.append(fill); cell.append(top, track); scores.append(cell); bars[k] = { cell, val, fill };
    });
    const transcript = el("div", "text-xs text-slate-600 leading-snug"); transcript.dir = "auto";
    const ocr = el("div", "text-[11px] text-muted leading-snug"); ocr.dir = "auto";
    const related = el("div", "text-[11px] text-muted leading-snug");
    let audio = null;
    if (a.audio_url) { audio = el("audio", "w-full h-8"); audio.controls = true; audio.preload = "none"; audio.src = a.audio_url; }
    // ---- one-click classification
    const actions = el("div", "mt-auto pt-2 border-t border-line flex flex-col gap-2");
    const btns = el("div", "grid grid-cols-2 gap-2");
    const mine = el("button", "btn-primary rounded-full py-1.5 text-xs", "🏢 My Brand"); mine.type = "button";
    const comp = el("button", "btn-soft rounded-full py-1.5 text-xs !text-indigo-700 !border-indigo-200 hover:!bg-indigo-50", "⚔️ Competitor"); comp.type = "button";
    btns.append(mine, comp);
    const named = el("div", "hidden grid grid-cols-[1fr_auto] gap-2");
    const brandIn = el("input", "field !py-1 !text-xs"); brandIn.placeholder = "Advertiser name";
    const go = el("button", "btn-soft px-3 text-xs", "Save"); go.type = "button";
    named.append(brandIn, go);
    const ignore = el("button", "text-[11px] text-slate-400 hover:text-slate-600 underline self-start", "Not an ad (promo / ident / programme)"); ignore.type = "button";
    const msg = el("div", "text-[11px] text-muted min-h-[1rem]");
    actions.append(btns, named, ignore, msg);
    let pendingRole = null;
    const classify = async (role) => {
      const cur = discRows.get(a.asset_id)?.asset || a;
      if (cur.suggested_brand) {
        mine.disabled = comp.disabled = true;
        try {
          const res = await post(`/api/discovery/assets/${id}/confirm-brand`, { role });
          setText(msg, `✔ ${cur.suggested_brand} as ${role === "self" ? "my brand" : "competitor"}: ${res.occurrences_backfilled} airing(s) now in rate-card audit.`);
          refreshAfterClassify();
        } catch (e) { setText(msg, `⚠ ${e.message}`); } finally { mine.disabled = comp.disabled = false; }
        return;
      }
      pendingRole = role;                               // no suggestion: one field, then save
      named.classList.remove("hidden");
      brandIn.focus();
      setText(msg, `Name the advertiser to save it as ${role === "self" ? "my brand" : "a competitor"}.`);
    };
    mine.addEventListener("click", () => classify("self"));
    comp.addEventListener("click", () => classify("competitor"));
    const saveNamed = async () => {
      const brand = brandIn.value.trim();
      if (!brand || !pendingRole) { setText(msg, "Advertiser name required."); return; }
      go.disabled = true;
      try {
        const res = await post(`/api/discovery/assets/${id}/classify`, { brand, campaign: "", role: pendingRole });
        setText(msg, `✔ Saved ${brand}: ${res.occurrences_backfilled} airing(s) added to analytics.`);
        refreshAfterClassify();
      } catch (e) { setText(msg, `⚠ ${e.message}`); } finally { go.disabled = false; }
    };
    go.addEventListener("click", saveNamed);
    brandIn.addEventListener("keydown", (e) => { if (e.key === "Enter") saveNamed(); });
    ignore.addEventListener("click", async () => {
      try { await post(`/api/discovery/assets/${id}/ignore`, { label: (discRows.get(a.asset_id)?.asset.content_hint) || "not an ad" }); refreshAfterClassify(); }
      catch (e) { setText(msg, `⚠ ${e.message}`); }
    });
    bodyEl.append(head, tag, tagNote, review, meta, scores, transcript, ocr, related);
    if (audio) bodyEl.append(audio);
    bodyEl.append(actions);
    tr.append(media, bodyEl);
    return { tr, asset: a, shot, shotImg, placeholder, logoWrap, logoImg, health, durBadge, status, tag, tagNote, review,
             meta, scores, bars, transcript, ocr, related, actions, named, brandIn };
  }

  function updateDiscRow(row, a) {
    row.asset = a;
    const open = a.status === "unclassified";
    const cr = a.creative || {};                        // absent until the server runs Phase 11.4 code
    // ---- media
    row.shot.classList.toggle("hidden", !a.packshot_url);
    if (a.packshot_url && row.shot.dataset.src !== a.packshot_url) {
      row.shot.dataset.src = a.packshot_url; row.shot.href = a.packshot_url; row.shotImg.src = a.packshot_url;
    }
    row.shot.title = a.packshot_at ? `Packshot: 1.5 s before the spot ends, captured ${a.packshot_at}` : "";
    setText(row.placeholder, a.packshot_url ? "" : (a.transcribed_at && !(a.transcript || "").trim()
      ? "🎵 Music / audio spot" : "Packshot arrives with the next airing"));
    row.logoWrap.classList.toggle("hidden", !a.logo_url);
    if (a.logo_url && row.logoImg.dataset.src !== a.logo_url) { row.logoImg.dataset.src = a.logo_url; row.logoImg.src = a.logo_url; }
    setPill(row.health, cr.health === null || cr.health === undefined ? "Analyzing…" : `Health ${cr.health}%`, scoreTone(cr.health));
    row.health.classList.add("absolute", "top-2", "right-2", "shadow-sm");
    row.health.title = cr.assessed ? `Creative health: weighted mean of ${cr.assessed} assessable components` : "";
    const dur = cr.label || `${fmt(a.nominal_duration_sec || a.duration_sec, 0)}s`;
    setText(row.durBadge, cr.daypart ? `${dur} | ${cr.daypart}` : dur);
    // ---- advertiser tag
    let ev = [];
    try { ev = JSON.parse(a.brand_evidence || "[]"); } catch (_) { ev = []; }
    const best = ev.find((m) => m.brand === a.suggested_brand) || {};
    if (a.status === "classified") {
      setText(row.tag, `🏷️ ${a.label || "classified"}`); setText(row.tagNote, "confirmed by a reviewer");
    } else if (a.suggested_brand) {
      setText(row.tag, `🏷️ ${a.suggested_brand}${best.product ? ` — ${best.product}` : ""} · ${Math.round((a.brand_confidence || 0) * 100)}% match`);
      const said = (best.evidence || []).map((x) => (Array.isArray(x) ? x[0] : x)).filter(Boolean).slice(0, 2).join(", ");
      setText(row.tagNote, `${SOURCE_LABEL[a.suggestion_source || "audio"] || "suggested"}${said ? ` “${said}”` : ""} · auto-suggested, verify before confirming`);
    } else {
      setText(row.tag, a.status === "ignored" ? `Ignored · ${a.label || "not an ad"}` : "Identifying advertiser…");
      setText(row.tagNote, open ? "Listen or check the packshot, then classify" : "");
    }
    setPill(row.status, { unclassified: "To review", classified: "Classified", ignored: "Ignored" }[a.status] || a.status,
            a.status === "classified" ? "good" : open ? "warn" : "neutral");
    setText(row.review, open ? (a.review_label || "") : "");
    row.review.classList.toggle("hidden", !open || !a.review_label);
    setText(row.meta, `${a.occurrences === 1 ? "First airing" : `${a.occurrences} airings`}${a.last_seen_cet ? ` · last ${a.last_seen_cet.replace("T", " ").slice(5, 16)} CET` : ""}` +
      (a.duration_basis && a.duration_basis.startsWith("snapped") ? ` · measured ${fmt(a.duration_sec)} s` : ""));
    // ---- creative metrics
    const comps = cr.components || {};
    row.scores.classList.toggle("hidden", !cr.components);
    SCORE_PARTS.forEach(([k]) => {
      const c = comps[k] || {}, b = row.bars[k];
      setText(b.val, c.score === null || c.score === undefined ? "Analyzing…" : (c.display || `${c.score}`));
      b.fill.style.width = `${c.score || 0}%`;
      b.fill.className = `bar-fill h-full rounded-full ${c.score >= 70 ? "bg-emerald-400" : c.score >= 40 ? "bg-amber-300" : "bg-rose-300"}`;
      b.cell.title = c.note || "";
    });
    // ---- evidence
    const tx = (a.transcript || "").trim();
    setText(row.transcript, a.transcribed_at
      ? (a.transcript_quality === "soundtrack" ? `🎵 ${a.transcript_display}`   // incoherent / hallucinated text is never shown
        : tx ? `🎙 “${tx.length > 140 ? tx.slice(0, 140) + "…" : tx}”` : "🎙 Music-led · no clear speech")
      : "🎙 Transcribing…");
    row.transcript.title = tx;
    setText(row.ocr, a.ocr_at ? (a.ocr_text ? `📷 on screen: ${a.ocr_text.slice(0, 120)}` : "📷 no readable text on the packshot") : "");
    let rel = [];
    try { rel = JSON.parse(a.related_assets || "[]"); } catch (_) { rel = []; }
    const same = ((a.visual_features || {}).same_packshot_as || []).map((x) => x.slice(-6));
    setText(row.related, [rel.length ? `shares audio with ${rel.map((r) => `${r.asset_id.slice(-6)} (${Math.round(r.coverage * 100)}%)`).join(", ")}` : "",
                          same.length ? `same packshot as ${same.join(", ")}` : ""].filter(Boolean).join(" · "));
    row.actions.classList.toggle("hidden", !open);
    if (!row.brandIn.value && !a.suggested_brand && a.ocr_text) {
      const latin = a.ocr_text.split(/\s+/).find((w) => /^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'&-]{2,}$/.test(w));
      if (latin) row.brandIn.placeholder = `Advertiser name (on screen: ${latin})`;
    }
  }

  function makeRadarRow(ch) {
    const root = el("div", "rounded-2xl bg-slate-50/70 border border-slate-100 p-4 transition-colors hover:bg-slate-50");
    const top = el("div", "flex items-center justify-between gap-3 flex-wrap");
    const left = el("div", "min-w-0");
    const nameRow = el("div", "flex items-center gap-2");
    const dot = el("span", "w-2 h-2 rounded-full bg-slate-300 shrink-0 transition-colors");
    const name = el("div", "font-semibold text-ink truncate", ch.name);
    const badge = el("span", "pill bg-slate-100 text-slate-600 transition-colors", "Checking…");
    nameRow.append(dot, name, badge);
    const status = el("div", "text-xs text-muted mt-1");
    left.append(nameRow, status);
    const right = el("div", "flex items-center gap-2");
    const db = el("span", "num text-sm font-semibold text-slate-700 min-w-[5.5rem] text-right");
    const btn = el("button", "btn-soft rounded-full px-3 py-1 text-xs flex items-center gap-1.5");
    btn.type = "button"; btn.innerHTML = '<i data-lucide="tv"></i>'; btn.append(document.createTextNode("Preview"));
    btn.addEventListener("click", () => openPreview(ch.name));
    right.append(db, btn); top.append(left, right);
    const meter = el("div", "h-2.5 rounded-full bg-slate-200/70 mt-3 overflow-hidden");
    const fill = el("div", "bar-fill h-full rounded-full bg-emerald-400"); fill.style.width = "0%"; meter.append(fill);
    const canvas = el("canvas", "w-full h-12 mt-2 block"); canvas.height = 48;
    const foot = el("div", "text-[11px] text-muted mt-1");
    root.append(top, meter, canvas, foot);
    return { root, status, db, btn, fill, canvas, foot, dot, badge };
  }

  function updateRadarRow(row, ch) {
    let label = STATE_LABEL[ch.state] || ch.state;
    if (ch.state === "backoff" && (ch.last_error || "").includes("ChannelUnavailable")) label = "Waiting for channel to go live";
    if (ch.state === "backoff" && ch.next_retry_at) label += ` · retry in ${Math.max(0, Math.round(ch.next_retry_at - Date.now() / 1000))}s`;
    const stab = ch.state !== "listening" ? "" : ch.disconnects_last_hour >= 3 ? " · unstable" :
      (ch.disconnects_last_hour || (ch.connected_for_s || 0) < 60) ? " · settling" : " · stable";
    const stalled = ch.link === "stalled";
    if (stalled) label = "Connecting · no audio";
    setText(row.status, ch.offline ? `${label} · no report for ${Math.round(ch.data_age_s || 0)} s`
      : stalled ? `${label}${ch.audio_age_s != null ? ` for ${ch.audio_age_s >= 120 ? `${Math.round(ch.audio_age_s / 60)} min` : `${Math.round(ch.audio_age_s)} s`}` : ""} · ${ch.source === "engine" ? "24/7 engine" : "this app"}`
      : `${label}${stab}${ch.link === "stale" ? " · data stale" : ""} · ${ch.source === "engine" ? "24/7 engine" : "this app"}`);
    row.status.classList.toggle("text-rose-600", !!ch.offline); row.status.classList.toggle("text-amber-600", stalled && !ch.offline);
    const tone = ch.offline ? ["Offline", "bg-rose-50 text-rose-700", "bg-rose-400"] : stalled ? ["Connecting", "bg-amber-50 text-amber-700", "bg-amber-400"]
      : ch.state === "listening" ? (ch.silent ? ["Silent", "bg-amber-50 text-amber-700", "bg-amber-400"] : ["Live", "bg-emerald-50 text-emerald-700", "bg-emerald-500"])
      : ["Connecting", "bg-amber-50 text-amber-700", "bg-amber-400"];
    row.badge.className = `pill transition-colors ${tone[1]}`; setText(row.badge, tone[0]);
    row.dot.className = `w-2 h-2 rounded-full shrink-0 transition-colors ${tone[2]}`;
    const lvl = ch.level_dbfs;
    setText(row.db, lvl === null || ch.state !== "listening" || stalled ? "— dBFS" : `${fmt(lvl)} dBFS`);
    const pct = lvl === null ? 0 : Math.max(0, Math.min(100, (lvl + 60) / 60 * 100));
    row.fill.style.width = `${ch.state === "listening" && !stalled ? pct : 0}%`;
    row.fill.className = `bar-fill h-full rounded-full ${ch.silent ? "bg-amber-300" : pct > 88 ? "bg-gradient-to-r from-emerald-400 to-amber-300" : "bg-gradient-to-r from-emerald-300 to-emerald-500"}`;
    row.btn.disabled = !ch.preview; row.btn.classList.toggle("opacity-40", !ch.preview);
    row.btn.title = ch.preview ? "Open client-side video preview" : "No browser-playable stream for this channel";
    setText(row.foot, (ch.state === "listening" && !stalled ? `Peak ${fmt(ch.peak_dbfs_10s)} dBFS · ${Math.round(ch.buffered_seconds || 0)} s buffered · ` : "") +
      `${ch.detections || 0} detection${ch.detections === 1 ? "" : "s"} · ${ch.references || 0} reference${ch.references === 1 ? "" : "s"}` +
      (ch.state !== "listening" || stalled ? " · reconnecting automatically" : ""));
    drawLevels(row.canvas, ch.state === "listening" && !stalled && !ch.offline ? (ch.level_history || []) : []);
  }

  function drawLevels(canvas, levels) {
    const w = canvas.clientWidth || 300, h = canvas.height;
    if (canvas.width !== w) canvas.width = w;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (!levels.length) return;
    const n = 60, bw = w / n;
    const data = levels.slice(-n);
    data.forEach((db, i) => {
      const v = Math.max(0, Math.min(1, (db + 60) / 60));
      const bh = Math.max(1, v * (h - 4));
      ctx.fillStyle = db < -60 ? "rgba(251,191,36,.6)" : `rgba(99,102,241,${0.25 + 0.6 * v})`;
      ctx.fillRect((n - data.length + i) * bw + 1, (h - bh) / 2, Math.max(1, bw - 2), bh);
    });
  }

  function renderEngine(e) {
    const dot = $("engine-dot"), pill = $("engine-pill");
    state.engineActive = !!(e && e.active);
    if (state.engineActive) {
      dot.className = "w-2.5 h-2.5 rounded-full bg-emerald-500 ring-4 ring-emerald-100";
      setText($("engine-label"), "24/7 Engine Active");
      setText($("engine-sub"), `${e.channels} channel(s) · heartbeat ${Math.round(e.heartbeat_age_s)} s ago · ${Math.round(e.rss_mb)} MB`);
      pill.classList.add("!bg-emerald-50/60", "!border-emerald-200");
    } else if (e && (e.state === "running" || e.state === "starting")) {
      dot.className = "w-2.5 h-2.5 rounded-full bg-amber-400 ring-4 ring-amber-100";
      setText($("engine-label"), "Engine not responding");
      setText($("engine-sub"), `last heartbeat ${Math.round(e.heartbeat_age_s)} s ago`);
      pill.classList.remove("!bg-emerald-50/60", "!border-emerald-200");
    } else {
      dot.className = "w-2.5 h-2.5 rounded-full bg-slate-300";
      setText($("engine-label"), "24/7 Engine offline");
      setText($("engine-sub"), "start: scripts\\start_monitoring_service.bat");
      pill.classList.remove("!bg-emerald-50/60", "!border-emerald-200");
    }
    const diag = $("diag");
    const pairs = e ? [["state", e.state], ["pid", e.pid], ["host", e.hostname], ["heartbeat", `${Math.round(e.heartbeat_age_s)} s`],
      ["RAM", `${Math.round(e.rss_mb)} MB`], ["ffmpeg", `${Math.round(e.ffmpeg_rss_mb || 0)} MB`], ["restarts", e.restarts]] : [["engine", "never started"]];
    diag.replaceChildren(...pairs.flatMap(([k, v]) => [el("dt", "", k), el("dd", "text-right text-slate-700 font-medium", String(v ?? "—"))]));
    syncToggle();
  }

  // ------------------------------------------------------------ channels + toggle
  async function loadChannels() {
    const r = await api("/api/channels");
    state.channels = r.channels;
    const sel = $("channel-select"), keep = sel.value;
    sel.replaceChildren(...r.channels.map((c) => { const o = el("option", "", `${c.featured ? "★ " : ""}${c.name}`); o.value = c.id; return o; }));
    if (keep && r.channels.some((c) => c.id === keep)) sel.value = keep;
    syncToggle();
  }

  function renderMonitors(appChannels) {
    const running = new Set(appChannels.map((c) => c.name));
    state.channels.forEach((c) => { c.running = running.has(c.name); });
    $("monitor-list").replaceChildren(...appChannels.map((c) => {
      const li = el("li", "flex items-center gap-2 text-slate-600");
      li.append(el("span", `w-2 h-2 rounded-full ${c.state === "listening" ? "bg-emerald-500" : "bg-amber-400"}`), el("span", "truncate", c.name));
      return li;
    }));
    syncToggle();
  }

  function syncToggle() {
    const btn = $("toggle-btn"), sel = $("channel-select");
    const ch = state.channels.find((c) => c.id === sel.value);
    const running = !!(ch && ch.running);
    btn.disabled = state.busy || !ch || (state.engineActive && !running);
    const label = running ? "Stop monitoring" : "Start monitoring";
    if (btn.lastChild.textContent !== label) {
      btn.replaceChildren(); const i = document.createElement("i"); i.setAttribute("data-lucide", running ? "square" : "play");
      btn.append(i, el("span", "", label)); icons();
    }
    btn.className = `${running ? "btn-soft" : "btn-primary"} w-full py-2 text-sm flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed`;
    if (state.engineActive && !running) setText($("toggle-msg"), "The 24/7 engine owns live monitoring; in-app start is disabled.");
    else if ($("toggle-msg").dataset.sticky !== "1") setText($("toggle-msg"), "");
  }

  async function toggle() {
    const sel = $("channel-select"), ch = state.channels.find((c) => c.id === sel.value);
    if (!ch) return;
    state.busy = true; syncToggle();
    const msg = $("toggle-msg");
    try {
      const r = await api("/api/stream/toggle", { method: "POST", headers: { "Content-Type": "application/json" },
                                                  body: JSON.stringify({ channel: ch.id, action: ch.running ? "stop" : "start" }) });
      ch.running = r.running;
      setText(msg, r.running ? `Monitoring ${r.channel}…` : `Stopped ${r.channel}.`);
    } catch (e) {
      setText(msg, `⚠ ${e.message}`);
    } finally {
      msg.dataset.sticky = "1"; setTimeout(() => { msg.dataset.sticky = ""; }, 6000);
      state.busy = false; syncToggle(); loadTelemetry().catch(() => {});
    }
  }

  // ------------------------------------------------------------ airings + certificate
  function syncCell(when, raw, mcr) {
    const td = el("td", "pr-3 num whitespace-nowrap", when);
    const lock = mcr.lock || "UNSYNCED";
    const dot = el("span", `inline-block w-2 h-2 rounded-full ml-1.5 align-middle ${lock === "LOCKED" ? "bg-emerald-500" : lock === "TRACKING" ? "bg-amber-400" : "bg-slate-300"}`);
    const drift = mcr.mcr_drift_delta_seconds === null || mcr.mcr_drift_delta_seconds === undefined ? "not calibrated"
      : `MCR drift ${Number(mcr.mcr_drift_delta_seconds).toFixed(1)} s${mcr.margin_s != null ? ` ±${Number(mcr.margin_s).toFixed(1)} s` : ""}`;
    td.title = `MCR sync ${lock} · ${drift} · ingest clock ${raw} (raw, unaltered)`;
    td.append(dot);
    return td;
  }

  async function loadAirings() {
    const r = await api("/api/airings?limit=25");
    state.airings = r.airings;
    const body = $("airings-body");
    body.replaceChildren(...r.airings.map((a) => {
      const tr = el("tr", "border-b border-slate-100 last:border-0 hover:bg-slate-50/70 transition-colors");
      const mcr = a.mcr || {};
      const raw = a.algeria_timestamp ? a.algeria_timestamp.replace("T", " ").slice(0, 19) + " CET" : `${fmt(a.airing_start, 3)} s into file`;
      // Phase 26: the broadcast (MCR-calibrated) time leads; the raw ingest stamp stays visible in the tooltip
      const when = mcr.broadcast_start_cet ? mcr.broadcast_start_cet.replace("T", " ").slice(0, 19) + " CET" : raw;
      const integ = a.integrity === true ? ["Verified", "bg-emerald-50 text-emerald-700"] : a.integrity === false ? ["Check failed", "bg-rose-50 text-rose-700"] : ["Pre-seal record", "bg-slate-100 text-slate-600"];
      const link = el("a", "text-indigo-600 font-semibold hover:underline whitespace-nowrap", "📄 Certificate (PDF)"); link.title = "Tamper-evident proof-of-performance certificate"; link.href = `/api/certificate/${a.occurrence_id}`;
      link.addEventListener("click", (ev) => { ev.preventDefault(); openCertificate(a); });
      tr.dataset.search = `${a.brand} ${a.campaign} ${a.channel}`.toLowerCase();
      const cells = [el("td", "py-2.5 pr-3 num text-muted", String(a.occurrence_id)), el("td", "pr-3 font-medium", `${a.brand} · ${a.campaign}`),
        el("td", "pr-3 text-slate-600", a.channel), syncCell(when, raw, mcr), el("td", "pr-3 text-right num font-semibold", `${fmt(a.confidence * 100)}%`)];
      const ic = el("td", "pr-3"); ic.append(el("span", `pill ${integ[1]}`, integ[0]));
      const lc = el("td", "whitespace-nowrap"); lc.append(link);
      if (a.evidence_available) lc.append(" ", evidenceButton(a.occurrence_id, `${a.brand} · ${a.channel}`, a.evidence_sha256));
      tr.append(...cells, ic, lc); return tr;
    }));
    if (!r.airings.length) {
      const tr = el("tr"), td = el("td", "py-6 text-center text-muted", "Monitoring active · confirmed airings will appear here as they are detected.");
      td.colSpan = 7; tr.append(td); body.replaceChildren(tr);
    }
    applySearch();
    syncCertificates();
  }

  // header search pill: filters the detections table and the discovered-asset cards in place
  function applySearch() {
    const q = ($("global-search")?.value || "").trim().toLowerCase();
    document.querySelectorAll("#airings-body tr[data-search]").forEach((tr) => tr.classList.toggle("hidden", !!q && !tr.dataset.search.includes(q)));
    for (const row of discRows.values()) {
      const a = row.asset, hay = `${a.suggested_brand || ""} ${a.label || ""} ${a.transcript || ""} ${spotName(a.asset_id)}`.toLowerCase();
      row.tr.classList.toggle("hidden", !!q && !hay.includes(q));
    }
  }

  // certificate modal (frosted glass): seal details + one-click PDF download
  function openCertificate(a) {
    setText($("cert-modal-title"), `${a.brand} · ${a.campaign}`);
    setText($("cert-modal-meta"), `${a.channel} · airing #${a.occurrence_id}${a.algeria_timestamp ? ` · ${a.algeria_timestamp.replace("T", " ").slice(0, 19)} CET` : ""}`);
    setPill($("cert-modal-integrity"), a.integrity === true ? "🔐 Integrity verified" : a.integrity === false ? "⚠️ Integrity check failed" : "🔓 Pre-seal record",
            a.integrity === true ? "good" : a.integrity === false ? "bad" : "neutral");
    setPill($("cert-modal-conf"), `${fmt(a.confidence * 100)}% match`, "good");
    setText($("cert-modal-hash"), a.verification_hash || "No SHA-256 seal recorded for this airing.");
    $("cert-modal-download").href = `/api/certificate/${a.occurrence_id}`;
    const ev = $("cert-modal-evidence"); ev.replaceChildren();
    if (a.evidence_available) ev.append(evidenceButton(a.occurrence_id, `${a.brand} · ${a.channel}`, a.evidence_sha256));
    $("cert-modal").showModal(); icons();
  }

  function syncCertificates() {
    const sel = $("cert-select"), keep = sel.value;
    const ids = state.airings.map((a) => String(a.occurrence_id));
    const have = [...sel.options].map((o) => o.value);
    if (JSON.stringify(ids) !== JSON.stringify(have)) {
      sel.replaceChildren(...state.airings.map((a) => { const o = el("option", "", `#${a.occurrence_id} · ${a.brand} · ${a.channel}`); o.value = a.occurrence_id; return o; }));
      if (!ids.length) sel.replaceChildren(el("option", "", "No airings yet"));
      if (keep && ids.includes(keep)) sel.value = keep;
    }
    renderCertificate();
  }

  function renderCertificate() {
    const a = state.airings.find((x) => String(x.occurrence_id) === $("cert-select").value);
    const dl = $("cert-download");
    if (!a) { dl.setAttribute("aria-disabled", "true"); dl.removeAttribute("href"); setText($("cert-integrity"), "Certificates unlock with the first confirmed airing."); setText($("cert-hash"), ""); return; }
    dl.setAttribute("aria-disabled", "false"); dl.href = `/api/certificate/${a.occurrence_id}`;
    setText($("cert-integrity"), a.integrity === true ? "🔐 Integrity verified" : a.integrity === false ? "⚠️ Integrity check failed — record or evidence changed"
      : "🔓 Pre-seal record (detected before hashing existed)");
    setText($("cert-hash"), a.verification_hash ? `SHA-256 ${a.verification_hash}` : "");
  }

  // ------------------------------------------------------------ preview modal
  function openPreview(name) {
    setText($("preview-title"), `${name} · live preview`);
    $("preview-frame").src = `/api/preview?channel=${encodeURIComponent(name)}`;
    $("preview-modal").showModal();
  }
  function closePreview() { $("preview-frame").src = "about:blank"; $("preview-modal").close(); }   // stops playback

  // ------------------------------------------------------------ compliance, separation, SOV, alerts (Phase 12.1)
  const SOV_COLORS = ["#4F46E5", "#34D399", "#FBBF24", "#A78BFA", "#F472B6", "#60A5FA", "#FB923C", "#2DD4BF", "#94A3B8"];
  const dzdUnit = (v) => v === null || v === undefined ? "—" : `${dzd(Number(v))} DZD`;   // financial dzd() + unit
  const compState = { lastAlertId: null, ads: [], contracts: [] };

  async function loadCompliance() {
    const c = await api("/api/compliance/overview?limit=40");
    const k = c.kpis;
    setPill($("comp-card-status"), `rate card ${c.rate_card.status}`, c.rate_card.illustrative ? "warn" : "good");
    setPill($("comp-pct"), k.compliance_pct === null ? "Awaiting contract clauses" : `Compliance ${fmt(k.compliance_pct)}%`,
            k.compliance_pct === null ? "neutral" : k.compliance_pct >= 95 ? "good" : "bad");
    setPill($("comp-breaches"), `Breaches ${k.breaches} (position ${k.position_breaches} · separation ${k.separation_breaches})`,
            k.breaches ? "bad" : "neutral");
    setPill($("comp-reclaim"), `Reclaimable ${dzdUnit(k.reclaimable_net)}${c.rate_card.illustrative ? " (illustrative)" : ""}`,
            Number(k.reclaimable_net) > 0 ? "good" : "neutral");
    setPill($("comp-conflicts"), `Separation conflicts ${k.separation_conflicts}` +
            (k.provisional_conflicts ? ` (+${k.provisional_conflicts} unconfirmed)` : ""), k.separation_conflicts ? "bad" : "neutral");
    const hints = [];
    if (c.setup.needs_contracts) hints.push("no contract clauses yet");
    if (c.setup.brands_without_sector.length) hints.push(`${c.setup.brands_without_sector.length} brand(s) without a sector`);
    const verifiedN = Number(k.verified_contract_airings ?? k.verified_contracts_count ?? 0) || 0;
    const illustrativeN = Number(k.illustrative_airings ?? k.illustrative_estimates_count ?? 0) || 0;
    setPill($("comp-tracks"), `Verified contract: ${verifiedN} · Illustrative estimate: ${illustrativeN}`,
            verifiedN ? "good" : illustrativeN ? "warn" : "neutral");
    setText($("comp-setup-hint"), hints.length ? `· ${hints.join(" · ")}` : "");
    const list = $("comp-conflict-list");
    list.replaceChildren();
    if (!c.conflicts.length) list.append(el("li", "text-muted", c.setup.brands_without_sector.length
      ? "None detected — brands without a sector can't be compared." : "None detected."));
    c.conflicts.slice(0, 12).forEach((x) => list.append(el("li", x.provisional ? "text-muted" : "",
      `${x.b.start_cet.slice(5, 16).replace("T", " ")} · ${x.channel} · ${x.a.brand} ↔ ${x.b.brand} (${x.sector}) ${x.gap_seconds.toFixed(0)} s apart` +
      (x.provisional ? " · unconfirmed brand" : ""))));
  }

  // ------------------------------------------------------------ breach evidence player (Phase 19)
  function evidenceButton(occurrenceId, label, sha) {
    const b = el("button", "text-[11px] font-semibold text-indigo-600 hover:underline whitespace-nowrap", "\u25B6\uFE0F Play Evidence");
    b.type = "button"; b.title = "Tamper-evident audio clip, airing \u22123 s to +3 s";
    b.addEventListener("click", () => openEvidence(occurrenceId, label, sha));
    return b;
  }
  function openEvidence(id, label, sha) {
    const audio = $("evidence-audio");
    setText($("evidence-title"), `Evidence \u00b7 airing #${id}`);
    setText($("evidence-hash"), sha ? `SHA-256 ${sha}` : "");
    setText($("evidence-error"), "");
    $("evidence-download").href = `/api/evidence/download/${id}`;
    audio.onerror = () => setText($("evidence-error"), "Clip unavailable or failed its integrity check.");
    audio.src = `/api/evidence/play/${id}`;
    $("evidence-modal").showModal();
    audio.play().catch(() => {});
  }
  function initEvidence() {
    const close = () => { const a = $("evidence-audio"); a.pause(); a.removeAttribute("src"); a.load(); $("evidence-modal").close(); };
    $("evidence-close").addEventListener("click", close);
    $("evidence-modal").addEventListener("cancel", () => { $("evidence-audio").pause(); });
  }

  async function loadAlerts() {
    const r = await api("/api/compliance/alerts?limit=30");
    const ul = $("comp-alerts");
    setPill($("comp-alert-count"), String(r.alerts.length), r.alerts.some((a) => a.severity === "critical") ? "bad" : r.alerts.length ? "warn" : "neutral");
    const bell = $("bell-badge"); bell.classList.toggle("hidden", !r.alerts.length); setText(bell, r.alerts.length > 9 ? "9+" : String(r.alerts.length));
    const maxId = r.alerts.reduce((m, a) => Math.max(m, a.id), 0);
    if (compState.lastAlertId !== null) r.alerts.filter((a) => a.id > compState.lastAlertId).forEach(toast);
    compState.lastAlertId = Math.max(compState.lastAlertId || 0, maxId);
    ul.replaceChildren();
    if (!r.alerts.length) { ul.append(el("li", "text-muted text-xs", "No open alerts.")); return; }
    const icon = { critical: "🚨", warning: "⚠️", info: "ℹ️" };
    r.alerts.forEach((a) => {
      const li = el("li", `rounded-xl border px-3 py-2 flex items-start justify-between gap-2 ${a.severity === "critical" ? "border-rose-200 bg-rose-50/70" : "border-amber-100 bg-amber-50/50"}`);
      const txt = el("div");
      txt.append(el("div", "text-xs", `${icon[a.severity] || "•"} ${a.message}`),
                 el("div", "text-[11px] text-muted", `${a.kind.replace("_", " ")} · ${a.created_at.replace("T", " ").slice(5, 16)} UTC`));
      const ack = el("button", "text-[11px] underline text-slate-500 hover:text-slate-800 shrink-0", "Acknowledge"); ack.type = "button";
      ack.addEventListener("click", async () => {
        try { await post(`/api/compliance/alerts/${a.id}/ack`, {}); loadAlerts().catch(() => {}); } catch (e) { setText(ack, "⚠ retry"); }
      });
      const acts = el("div", "flex flex-col items-end gap-1 shrink-0");
      if (a.evidence_available) acts.append(evidenceButton(a.occurrence_id, a.message.slice(0, 60)));
      acts.append(ack);
      li.append(txt, acts); ul.append(li);
    });
  }

  function toast(a) {
    const t = el("div", `fixed bottom-4 right-4 z-50 max-w-sm rounded-2xl shadow-xl px-4 py-3 text-sm backdrop-blur-md border ${a.severity === "critical" ? "bg-rose-50/95 border-rose-200 text-rose-800" : "bg-white/95 border-line text-slate-700"}`,
                 `${a.severity === "critical" ? "🚨" : "🔔"} ${a.message}`);
    t.setAttribute("role", "status");
    document.body.append(t);
    setTimeout(() => t.remove(), 9000);
  }

  async function loadSov() {
    const prov = $("sov-provisional")?.checked ? "&provisional=true" : "";
    const s = await api(`/api/compliance/sov?days=7${prov}`);
    setPill($("comp-sov"), s.self_share_24h === null ? "My SOV · building baseline" : `My SOV 24 h ${fmt(s.self_share_24h)}%`, s.self_share_24h === null ? "neutral" : "info");
    const top = s.brands.slice(0, 8);
    $("sov-empty").classList.toggle("hidden", s.brands.length > 0);
    setText($("sov-note"), s.brands.length ? top.map((b) => `${b.brand}${b.role === "self" ? " (mine)" : ""}: ${b.share_24h ?? 0}% 24 h` +
      (b.change_pts ? ` (${b.change_pts > 0 ? "+" : ""}${b.change_pts} pts)` : "")).join(" · ") : "");
    if (!ensureCharts()) return;
    const data = { labels: s.days.map((d) => d.slice(5)), datasets: top.map((b, i) => ({
      label: b.brand, data: b.airtime.map((x) => Math.round(x)), stack: "sov",
      backgroundColor: b.role === "self" ? SOV_COLORS[0] : SOV_COLORS[(i % (SOV_COLORS.length - 1)) + 1], borderRadius: 6, borderSkipped: false })) };
    if (!charts.sov) {
      charts.sov = new Chart($("sov-chart"), { type: "bar", data, options: { responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { position: "bottom", labels: { boxWidth: 10, boxHeight: 10, usePointStyle: true, pointStyle: "rectRounded" } }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.raw} s` } } },
        scales: { x: { stacked: true, grid: { display: false }, border: { display: false } }, y: { stacked: true, beginAtZero: true, border: { display: false }, grid: { color: GRID }, title: { display: true, text: "airtime (s)" } } } } });
    } else { charts.sov.data = data; charts.sov.update(); }
  }

  async function loadComplianceSetup() {
    if ($("comp-setup-box").contains(document.activeElement)) return;     // don't wipe what the user is typing
    const [ads, cts] = await Promise.all([api("/api/ads"), api("/api/compliance/contracts")]);
    compState.ads = ads.ads; compState.contracts = cts.contracts;
    const sec = $("comp-sectors"); sec.replaceChildren();
    if (!ads.ads.length) sec.append(el("li", "text-muted", "No confirmed brands yet — classify discovered assets first."));
    ads.ads.forEach((a) => {
      const li = el("li", "flex items-center gap-2");
      const input = el("input", "field !py-0.5 !px-2 !text-xs w-36"); input.value = a.sector || ""; input.placeholder = "sector";
      const save = el("button", "underline text-indigo-600", "save"); save.type = "button";
      save.addEventListener("click", async () => {
        try { await post(`/api/ads/${a.ad_id}/sector`, { sector: input.value }); setText(save, "✔"); loadCompliance().catch(() => {}); }
        catch (e) { setText(save, `⚠ ${e.message}`); }
      });
      li.append(el("span", "w-40 truncate", `${a.brand_name} (${a.role}, ${a.airings})`), input, save);
      sec.append(li);
    });
    const brandSel = $("cc-brand"); const keep = brandSel.value; brandSel.replaceChildren();
    [...new Set(ads.ads.map((a) => a.brand_name))].forEach((b) => { const o = el("option", "", b); o.value = b; brandSel.append(o); });
    if (keep) brandSel.value = keep;
    const ul = $("comp-contracts"); ul.replaceChildren();
    if (!cts.contracts.length) ul.append(el("li", "text-muted", "No clauses: nothing can be breached or reclaimed."));
    cts.contracts.forEach((c, i) => {
      const li = el("li", "flex items-center justify-between gap-2");
      li.append(el("span", "", `${c.brand}${c.campaign && c.campaign !== "*" ? ` / ${c.campaign}` : ""}: ` +
        [c.booked_position && c.booked_position !== "ANY" ? `booked ${c.booked_position}` : null,
         c.competitor_separation_seconds ? `separation ${c.competitor_separation_seconds} s @ ${Math.round((c.separation_credit ?? 1) * 100)}% credit` : null]
          .filter(Boolean).join(" · ") || "no clauses"));
      const rm = el("button", "underline text-slate-500", "remove"); rm.type = "button";
      rm.addEventListener("click", () => saveContracts(compState.contracts.filter((_, j) => j !== i)));
      li.append(rm); ul.append(li);
    });
  }

  async function saveContracts(list) {
    try {
      await post("/api/compliance/contracts", { contracts: list });
      setText($("cc-msg"), "✔ saved — re-auditing"); await loadComplianceSetup(); loadCompliance().catch(() => {});
    } catch (e) { setText($("cc-msg"), `⚠ ${e.message}`); }
  }

  function addClause() {
    const brand = $("cc-brand").value;
    if (!brand) { setText($("cc-msg"), "Classify a brand first."); return; }
    const sep = Number($("cc-sep").value), credit = Number($("cc-credit").value);
    const clause = { brand, campaign: "*", booked_position: $("cc-position").value };
    if (sep > 0) { clause.competitor_separation_seconds = sep; clause.separation_credit = Math.min(100, Math.max(0, credit)) / 100; }
    saveContracts([...compState.contracts.filter((c) => !(c.brand === brand && (c.campaign || "*") === "*")), clause]);
  }


  // ------------------------------------------------------------ master asset onboarding (Phase 16)
  const SECTORS = ["Telecom", "Food & Beverage", "Dairy", "Retail", "Automotive", "Banking & Finance", "Health & Beauty",
                   "Electronics", "Real Estate", "Media", "Other"];
  const master = { file: null };

  function drawWave(buf) {
    const cv = $("master-wave");
    cv.classList.remove("hidden");
    const W = cv.width = Math.max(200, cv.clientWidth), H = cv.height;
    const ctx = cv.getContext("2d"), data = buf.getChannelData(0), step = Math.max(1, Math.floor(data.length / W));
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#6366F1";
    for (let x = 0; x < W; x++) {
      let lo = 1, hi = -1;
      for (let i = 0, base = x * step; i < step && base + i < data.length; i += Math.max(1, step >> 5)) {
        const v = data[base + i]; if (v < lo) lo = v; if (v > hi) hi = v;
      }
      if (hi < lo) { hi = 0; lo = 0; }
      ctx.fillRect(x, (1 - hi) * H / 2, 1, Math.max(1, (hi - lo) * H / 2));
    }
    setText($("master-wave-info"), `${buf.duration.toFixed(2)} s · ${buf.sampleRate} Hz preview (client-side; silence below -50 dBFS is trimmed on indexing)`);
    $("master-wave-info").classList.remove("hidden");
  }

  async function pickMaster(file) {
    master.file = null; $("master-submit").disabled = true; setText($("master-status"), "");
    if (!file) return;
    if (!/\.(mp4|wav|mp3)$/i.test(file.name)) { setText($("master-status"), "⚠ use a .mp4, .wav or .mp3 file"); return; }
    if (file.size > 16 * 2 ** 20) { setText($("master-status"), "⚠ over 16 MB: index it with `python -m spotly.cli register-ad`"); return; }
    master.file = file; setText($("master-drop-text"), `${file.name} · ${(file.size / 2 ** 20).toFixed(1)} MB`);
    if (!$("master-brand").value) $("master-brand").focus();
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const buf = await ctx.decodeAudioData(await file.arrayBuffer()); ctx.close(); drawWave(buf);
    } catch (e) { setText($("master-wave-info"), "Waveform preview unavailable in this browser; indexing still works."); $("master-wave-info").classList.remove("hidden"); }
    $("master-submit").disabled = false;
  }

  function fileB64(file) {
    return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result).split(",")[1]); r.onerror = rej; r.readAsDataURL(file); });
  }

  async function submitMaster(ev) {
    ev.preventDefault();
    if (!master.file) return;
    const st = $("master-status"); $("master-submit").disabled = true; setText(st, "Extracting audio, trimming silence, hashing…");
    try {
      const rate = $("master-rate").value;
      const r = await post("/api/ads/register", { filename: master.file.name, data_b64: await fileB64(master.file),
        brand: $("master-brand").value.trim(), sector: $("master-sector").value, position: $("master-position").value,
        separation_window: Number($("master-sep").value), rate_override: rate ? Number(rate) : null }, 180000);
      st.replaceChildren(el("span", "pill bg-emerald-50 text-emerald-700", "✔ Indexed"),
        el("span", "ml-2 text-xs text-muted", `${r.asset_id} · ${r.duration_s}s (lead ${r.lead_trim_s}s trimmed) · ${r.hashes} landmarks · ` +
          (r.packshot ? `packshot pHash ${r.packshot.phash}` : "no packshot") + " · live monitor hot-reloads it, no restart"));
      loadComplianceSetup().catch(() => {}); loadStats().catch(() => {});
    } catch (e) { st.replaceChildren(el("span", "pill bg-rose-50 text-rose-700", `⚠ ${e.message}`)); }
    finally { $("master-submit").disabled = !master.file; }
  }

  function initMaster() {
    const sel = $("master-sector"); SECTORS.forEach((s) => { const o = el("option", "", s); o.value = s; sel.append(o); });
    api("/api/ads").then((d) => { [...new Set(d.ads.map((a) => a.brand_name))].forEach((b) => { const o = el("option"); o.value = b; $("master-brands").append(o); }); }).catch(() => {});
    $("master-open").addEventListener("click", () => $("master-modal").showModal());
    $("master-close").addEventListener("click", () => $("master-modal").close());
    $("master-file").addEventListener("change", (e) => pickMaster(e.target.files[0]));
    const drop = $("master-drop");
    ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("border-indigo-400", "bg-indigo-50"); }));
    ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("border-indigo-400", "bg-indigo-50"); }));
    drop.addEventListener("drop", (e) => { pickMaster(e.dataTransfer.files[0]); });
    $("master-form").addEventListener("submit", submitMaster);
  }

  // ------------------------------------------------------------ contract modal (Phase 18 dual-track)
  function contractMode() { return document.querySelector('input[name="contract-mode"]:checked').value; }
  function syncContractMode() {
    const manual = contractMode() === "manual";
    $("contract-fields").classList.toggle("hidden", !manual);
    $("contract-estimate-note").classList.toggle("hidden", manual);
    $("contract-submit").textContent = manual ? "Bind contract" : "Use market estimate";
    ["contract-id", "contract-brand", "contract-channel", "contract-rate"].forEach((id) => { $(id).required = manual; });
  }
  const fileToB64 = (f) => new Promise((ok, no) => {
    const r = new FileReader(); r.onerror = () => no(r.error);
    r.onload = () => ok(String(r.result).split(",")[1]); r.readAsDataURL(f);
  });
  async function submitContract(e) {
    e.preventDefault();
    const st = $("contract-status");
    if (contractMode() === "estimate") { $("contract-modal").close(); return; }
    const body = { contract_id: $("contract-id").value.trim(), brand_name: $("contract-brand").value.trim(),
                   channel_id: $("contract-channel").value.trim(), net_rate_per_second: Number($("contract-rate").value),
                   booked_position: $("contract-position").value, separation_window_seconds: Number($("contract-sep").value || 180) };
    if ($("contract-from").value) body.valid_from = $("contract-from").value;
    if ($("contract-to").value) body.valid_to = $("contract-to").value;
    $("contract-submit").disabled = true;
    try {
      const f = $("contract-file").files[0];
      if (f) { body.document_name = f.name; body.document_b64 = await fileToB64(f); }
      const r = await post("/api/contracts/ingest", body, 60000);
      st.replaceChildren(el("span", "pill bg-emerald-50 text-emerald-700", `✓ ${r.contract_id} ${r.action.toLowerCase()}d · audit v${r.version}`));
      loadCompliance().catch(() => {});
    } catch (err) { st.replaceChildren(el("span", "pill bg-rose-50 text-rose-700", `⚠ ${err.message}`)); }
    finally { $("contract-submit").disabled = false; }
  }
  function initContract() {
    $("contract-open").addEventListener("click", () => { $("contract-status").replaceChildren(); $("contract-modal").showModal(); });
    $("contract-close").addEventListener("click", () => $("contract-modal").close());
    document.querySelectorAll('input[name="contract-mode"]').forEach((r) => r.addEventListener("change", syncContractMode));
    $("contract-form").addEventListener("submit", submitContract);
    syncContractMode();
  }

  // ------------------------------------------------------------ boot
  // Called by the async <script onload> of Chart.js / Lucide, whenever they arrive.
  window.spotlyLibsReady = () => {
    icons();
    if (ensureCharts() && state.lastStats) renderStats(state.lastStats);
  };

  // sidebar: highlight the section in view (soft indigo pill)
  function initScrollSpy() {
    const links = [...document.querySelectorAll(".nav-link[data-nav]")];
    if (!window.IntersectionObserver || !links.length) return;
    const byId = new Map(links.map((l) => [l.dataset.nav, l]));
    const io = new IntersectionObserver((entries) => {
      entries.filter((e) => e.isIntersecting).forEach((e) => {
        links.forEach((l) => l.classList.remove("active")); byId.get(e.target.id)?.classList.add("active");
      });
    }, { rootMargin: "-20% 0px -70% 0px" });
    byId.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
  }

  function boot() {
    window.spotlyLibsReady();          // libraries that finished before this script ran
    $("industry-select").addEventListener("change", (e) => { state.industry = e.target.value; loadStats().catch(() => {}); });
    $("channel-select").addEventListener("change", syncToggle);
    $("toggle-btn").addEventListener("click", toggle);
    $("cert-select").addEventListener("change", renderCertificate);
    document.querySelectorAll(".fin-mode").forEach((b) => b.addEventListener("click", () => {
      state.finMode = b.dataset.finMode;
      if (state.lastFin) renderFinancials(state.lastFin);
    }));
    $("global-search").addEventListener("input", applySearch);
    $("bell-btn").addEventListener("click", () => $("comp-alerts").scrollIntoView({ behavior: "smooth", block: "center" }));
    $("cert-modal-close").addEventListener("click", () => $("cert-modal").close());
    $("preview-close").addEventListener("click", closePreview);
    initScrollSpy();
    $("preview-modal").addEventListener("cancel", (e) => { e.preventDefault(); closePreview(); });
    loop(loadTelemetry, POLL.telemetry);
    loop(loadStats, POLL.stats);
    loop(loadAirings, POLL.airings);
    loop(loadDiscovery, POLL.discovery);
    loop(loadChannels, POLL.channels);
    loop(loadCompliance, POLL.compliance);
    loop(loadAlerts, POLL.alerts);
    loop(loadSov, POLL.sov);
    loop(loadComplianceSetup, POLL.setup);
    $("cc-add").addEventListener("click", addClause);
    initMaster(); initContract(); initEvidence();
    const syncBatch = () => {
      const q = new URLSearchParams();
      [["start_date", "cb-start"], ["end_date", "cb-end"], ["brand", "cb-brand"]].forEach(([k, id]) => { const v = $(id).value.trim(); if (v) q.set(k, v); });
      $("cb-go").href = "/api/certificates/export-batch" + (q.toString() ? `?${q}` : "");
    };
    ["cb-start", "cb-end", "cb-brand"].forEach((id) => $(id).addEventListener("input", syncBatch));
    $("sov-provisional").addEventListener("change", () => loadSov().catch(() => {}));
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
