/* KIHC bento SPA: in-place polling (no page reloads), Chart.js updates without re-creation.
 * All server strings are written with textContent / created elements, never innerHTML. */
(() => {
  "use strict";
  const MINT = "#10B981", TRACK = "#E5EBEA", INK = "#111827", MUTED = "#6B7280";
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
  function setText(el, text) { if (el && el.textContent !== String(text)) el.textContent = text; }
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
  function setPill(el, text, tone) {
    const tones = { good: "bg-mintsoft text-minting", neutral: "bg-gray-100 text-gray-600", bad: "bg-red-50 text-red-700",
                    dark: "bg-emerald-400/15 text-emerald-300", warn: "bg-amber-100 text-amber-800" };
    el.className = `pill ${tones[tone] || tones.neutral} mt-2`;
    setText(el, text);
  }
  function setOnline(on) {
    if (state.online === on) return;
    state.online = on;
    $("offline-banner").classList.toggle("hidden", on);
    $("conn-dot").className = `w-2 h-2 rounded-full ${on ? "bg-mint" : "bg-amber-400"}`;
  }
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  function icons() { if (window.lucide) window.lucide.createIcons(); }

  // Polling loop: setTimeout chain (no overlap). A hidden tab keeps polling, slowly (browsers throttle timers
  // there anyway); coming back refreshes immediately. Failures never stop the loop (auto-reconnect).
  const HIDDEN_EVERY = 30000;
  function loop(fn, base) {
    let timer = null, every = base;
    const run = async () => {
      timer = null;
      try { await fn(); } catch (e) { /* surfaced via offline banner; next tick retries */ }
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
    Chart.defaults.font.family = "Inter, system-ui, Segoe UI, sans-serif";
    Chart.defaults.color = MUTED;
    charts.perday = new Chart($("chart-perday"), {
      type: "bar", data: { labels: [], datasets: [{ data: [], backgroundColor: MINT, borderRadius: 3, barPercentage: 0.8, categoryPercentage: 0.9 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false }, tooltip: { displayColors: false } },
        scales: { x: { display: false }, y: { display: false, beginAtZero: true } } } });
    charts.premium = new Chart($("chart-premium"), {
      type: "doughnut", data: { labels: ["First + Last", "Middle"], datasets: [{ data: [0, 1], backgroundColor: [MINT, TRACK], borderWidth: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, circumference: 180, rotation: -90, cutout: "74%",
        animation: { duration: 500 }, plugins: { legend: { display: false },
          tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.raw} airings` } } } } });
    charts.clutter = new Chart($("chart-clutter"), {
      type: "bar", data: { labels: [], datasets: [{ label: "Avg detected ads / break", data: [], backgroundColor: MINT, borderRadius: 6, maxBarThickness: 28 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
        plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c) => {
          const r = c.dataset.meta?.[c.dataIndex]; return r ? `${r.pods} breaks · avg ${Math.round(r.avg_seconds)} s` : ""; } } } },
        scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: "#EEF2F1" }, title: { display: true, text: "Ads / break" } } } } });
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
    setPill($("kpi-total-badge"), `${recent} in ${s.per_day.length} days`, "neutral");
    if (charts.perday) {
      charts.perday.data.labels = s.per_day.map((d) => d.date);
      charts.perday.data.datasets[0].data = s.per_day.map((d) => d.count);
      charts.perday.update("none");
    }
    // KPI 2
    const c = s.confidence;
    setText($("kpi-conf"), c.avg === null ? "—" : fmt(c.avg * 100));
    if (c.delta_pts === null) setPill($("kpi-conf-delta"), c.basis, "neutral");
    else setPill($("kpi-conf-delta"), `${c.delta_pts >= 0 ? "▲" : "▼"} ${fmt(Math.abs(c.delta_pts))} pts vs prior 7 days`, c.delta_pts >= 0 ? "good" : "bad");
    // KPI 3 + industry selector
    syncIndustries(s.sov);
    setText($("kpi-sov-industry"), s.sov.industry || "all industries");
    if (s.sov.total_airings) {
      setText($("kpi-sov"), fmt(s.sov.client_pct));
      setPill($("kpi-sov-leader"), s.sov.leader ? `leader: ${s.sov.leader.brand} ${Math.round(s.sov.leader.pct)}%` : "—", "neutral");
      setText($("kpi-sov-sub"), `of ${fmt(s.sov.airtime_min)} min detected airtime · ${s.sov.total_airings} airings`);
    } else {
      setText($("kpi-sov"), "—");
      setPill($("kpi-sov-leader"), "no tagged airings", "neutral");
      setText($("kpi-sov-sub"), "Tag ads with an industry and role to compute SOV.");
    }
    // Premium gauge
    const p = s.premium;
    setText($("premium-sub"), `First + Last in break · ${p.scope}`);
    setText($("premium-pct"), p.pct === null ? "—" : `${Math.round(p.pct)}%`);
    setText($("premium-n"), p.in_break_airings ? `${p.in_break_airings} in-break airings` : "no multi-ad breaks yet");
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
      b.className = `fin-mode px-3 py-1 rounded-lg font-semibold ${on ? "bg-[#0C1A1A] text-white" : "text-gray-600 hover:bg-gray-50"}`;
      b.setAttribute("aria-pressed", String(on));
    });
    const st = $("fin-status");
    st.className = `pill normal-case tracking-normal ${f.rate_card.illustrative ? "bg-amber-50 text-amber-800" : "bg-mintsoft text-minting"}`;
    setText(st, f.rate_card.illustrative ? "illustrative rate card — replace with contracted tariffs" : "verified rate card");
    st.title = (f.rate_card.errors || []).join("\n") || `sources: ${(f.rate_card.sources || []).join(" → ")}`;
    // Card A
    setText($("fin-basis-label"), net ? "(est. net)" : "(gross)");
    setText($("fin-total"), dzd(net ? f.estimated_net_spend : f.gross_spend));
    setPill($("fin-total-badge"), `${f.priced_airings} priced airings`, "neutral");
    setText($("fin-total-sub"), net ? `gross ${dzd(f.gross_spend)} DZD before agency discount`
      : `est. net ${dzd(f.estimated_net_spend)} DZD after agency discount` +
        (f.unpriced_airings ? ` · ${f.unpriced_airings} unpriced (no broadcast time)` : ""));
    // Card B
    const top = f.top_spender;
    setText($("fin-top"), top ? top.brand : "—");
    setPill($("fin-sos"), top ? `SOS ${fmt(top.sos_pct)}%` : "SOS —", "good");
    setPill($("fin-sov"), top ? `SOV ${fmt(top.sov_pct)}%` : "SOV —", "neutral");
    $("fin-sos").classList.remove("mt-2"); $("fin-sov").classList.remove("mt-2");
    setText($("fin-top-sub"), top ? `${dzd(net ? top.estimated_net_spend : top.gross_spend)} DZD ${net ? "est. net" : "gross"} · ` +
      "SOS = share of estimated spend, SOV = share of detected airtime (same airings)" : "No priced airings yet.");
    // Card C
    setText($("fin-credit"), dzd(f.contract_loss_credit));
    if (f.breaches) setPill($("fin-credit-badge"), `💰 Make-Good Value · ${f.breaches} breach${f.breaches === 1 ? "" : "es"}`, "bad");
    else setPill($("fin-credit-badge"), "no position breaches", "good");
    setText($("fin-credit-sub"), "Net value of booked First/Last positions not delivered; claimable pending as-run verification" +
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
    if (!rows.length) { list.replaceChildren(el("li", "text-sm text-muted", "No detections yet.")); return; }
    const items = rows.map((r) => {
      const li = el("li");
      const head = el("div", "flex justify-between text-sm");
      head.append(el("span", "truncate pr-2", r.name));
      const val = el("span", "num whitespace-nowrap"); val.append(el("b", "", `${Math.round(r.pct)}%`), document.createTextNode(` · ${r.count}`));
      head.append(val);
      const bar = el("div", "h-2 rounded-full bg-line mt-1.5 overflow-hidden");
      const fill = el("div", "bar-fill h-full rounded-full bg-mint"); fill.style.width = `${Math.min(100, r.pct)}%`;
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
      setText($("clutter-sub"), "Appears once breaks are reconstructed from detections.");
    }
    charts.clutter.update("none");
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
    $("radar-count").classList.remove("mt-2");
    renderMonitors(t.channels.filter((c) => c.source === "app"));
    renderStreamPill(t.channels);
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
      setPill(pill, `🟠 ${pending.name} reconnecting${why}`, "neutral");
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
  const scoreTone = (s) => s === null || s === undefined ? "neutral" : s >= 70 ? "good" : s >= 40 ? "neutral" : "bad";

  async function post(path, payload, timeout) {
    return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), timeout });
  }
  function refreshAfterClassify() { loadStats().catch(() => {}); loadAirings().catch(() => {}); loadDiscovery().catch(() => {}); }

  function makeDiscRow(a) {
    const id = encodeURIComponent(a.asset_id);
    const tr = el("article", "rounded-2xl border border-line bg-white overflow-hidden flex flex-col");
    // ---- media header: packshot (or audio placeholder) with overlay badges
    const media = el("div", "relative aspect-video bg-gradient-to-br from-petrol to-petrol2 flex items-center justify-center");
    const shot = el("a", "absolute inset-0 hidden"); shot.target = "_blank"; shot.rel = "noopener";
    const shotImg = el("img", "w-full h-full object-cover"); shotImg.loading = "lazy"; shotImg.alt = `${a.asset_id} packshot`;
    shot.append(shotImg);
    const placeholder = el("div", "text-center text-emerald-100/80 text-xs px-4");
    const logoWrap = el("div", "absolute top-2 left-2 hidden rounded-lg bg-white/95 p-1 shadow");
    const logoImg = el("img", "h-8 max-w-[7rem] object-contain"); logoImg.alt = "brand region"; logoWrap.append(logoImg);
    const health = el("span", "pill absolute top-2 right-2 shadow");
    const durBadge = el("span", "absolute bottom-2 left-2 rounded-md bg-black/65 text-white text-[11px] font-semibold px-2 py-0.5");
    media.append(placeholder, shot, logoWrap, health, durBadge);
    // ---- body
    const bodyEl = el("div", "p-4 flex flex-col gap-2 flex-1");
    const head = el("div", "flex items-center justify-between gap-2");
    const idEl = el("span", "font-mono text-[11px] text-muted", a.asset_id);
    const status = el("span", "pill");
    head.append(idEl, status);
    const tag = el("div", "font-bold text-sm leading-snug");
    const tagNote = el("div", "text-[11px] text-muted -mt-1");
    const review = el("div", "text-xs font-semibold text-amber-700 hidden");
    const meta = el("div", "text-[11px] text-muted");
    const scores = el("div", "grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]");
    const bars = {};
    SCORE_PARTS.forEach(([k, label]) => {
      const cell = el("div"); const top = el("div", "flex justify-between"); const name = el("span", "text-muted", label);
      const val = el("span", "num font-semibold"); top.append(name, val);
      const track = el("div", "h-1.5 rounded-full bg-gray-100 overflow-hidden mt-0.5"); const fill = el("div", "bar-fill h-full bg-mint"); fill.style.width = "0%";
      track.append(fill); cell.append(top, track); scores.append(cell); bars[k] = { cell, val, fill };
    });
    const transcript = el("div", "text-xs text-gray-600 leading-snug"); transcript.dir = "auto";
    const ocr = el("div", "text-[11px] text-muted leading-snug"); ocr.dir = "auto";
    const related = el("div", "text-[11px] text-muted leading-snug");
    let audio = null;
    if (a.audio_url) { audio = el("audio", "w-full h-8"); audio.controls = true; audio.preload = "none"; audio.src = a.audio_url; }
    // ---- one-click classification
    const actions = el("div", "mt-auto pt-2 border-t border-line flex flex-col gap-2");
    const btns = el("div", "grid grid-cols-2 gap-2");
    const mine = el("button", "rounded-lg bg-mint text-petrol2 font-bold py-1.5 text-xs", "🏢 My Brand"); mine.type = "button";
    const comp = el("button", "rounded-lg bg-petrol text-white font-bold py-1.5 text-xs", "⚔️ Competitor"); comp.type = "button";
    btns.append(mine, comp);
    const named = el("div", "hidden grid grid-cols-[1fr_auto] gap-2");
    const brandIn = el("input", "border border-line rounded-lg px-2 py-1 text-xs"); brandIn.placeholder = "Advertiser name";
    const go = el("button", "rounded-lg border border-line px-2 text-xs font-semibold", "Save"); go.type = "button";
    named.append(brandIn, go);
    const ignore = el("button", "text-[11px] text-muted underline self-start", "Not an ad (promo / ident / programme)"); ignore.type = "button";
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
      ? "🎵 Music / audio spot — visual needed" : "📷 Packshot will be captured at the next airing"));
    row.logoWrap.classList.toggle("hidden", !a.logo_url);
    if (a.logo_url && row.logoImg.dataset.src !== a.logo_url) { row.logoImg.dataset.src = a.logo_url; row.logoImg.src = a.logo_url; }
    setPill(row.health, cr.health === null || cr.health === undefined ? "Health n/a" : `Health ${cr.health}%`, scoreTone(cr.health));
    row.health.classList.remove("mt-2"); row.health.classList.add("absolute", "top-2", "right-2", "shadow");
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
      setText(row.tag, `🏷️ ${a.suggested_brand}${best.product ? ` — ${best.product}` : ""} (${Math.round((a.brand_confidence || 0) * 100)}%)`);
      const said = (best.evidence || []).map((x) => (Array.isArray(x) ? x[0] : x)).filter(Boolean).slice(0, 2).join(", ");
      setText(row.tagNote, `${SOURCE_LABEL[a.suggestion_source || "audio"] || "suggested"}${said ? ` “${said}”` : ""} · auto-suggested, verify before confirming`);
    } else {
      setText(row.tag, a.status === "ignored" ? `ignored · ${a.label || ""}` : "Advertiser not identified");
      setText(row.tagNote, open ? "listen / look at the packshot, then classify" : "");
    }
    setPill(row.status, { unclassified: "to review", classified: "classified", ignored: "ignored" }[a.status] || a.status,
            a.status === "classified" ? "good" : "neutral");
    row.status.classList.remove("mt-2");
    setText(row.review, open ? (a.review_label || "") : "");
    row.review.classList.toggle("hidden", !open || !a.review_label);
    setText(row.meta, `${a.occurrences} airing(s) · last ${a.last_seen_cet ? a.last_seen_cet.replace("T", " ").slice(5, 16) : "—"} CET` +
      (a.duration_basis && a.duration_basis.startsWith("snapped") ? ` · measured ${fmt(a.duration_sec)} s` : ""));
    // ---- creative metrics
    const comps = cr.components || {};
    row.scores.classList.toggle("hidden", !cr.components);
    SCORE_PARTS.forEach(([k]) => {
      const c = comps[k] || {}, b = row.bars[k];
      setText(b.val, c.score === null || c.score === undefined ? "n/a" : (c.display || `${c.score}`));
      b.fill.style.width = `${c.score || 0}%`;
      b.fill.className = `bar-fill h-full ${c.score >= 70 ? "bg-mint" : c.score >= 40 ? "bg-amber-400" : "bg-red-400"}`;
      b.cell.title = c.note || "";
    });
    // ---- evidence
    const tx = (a.transcript || "").trim();
    setText(row.transcript, a.transcribed_at
      ? (a.transcript_quality === "soundtrack" ? `🎵 ${a.transcript_display}`   // incoherent / hallucinated text is never shown
        : tx ? `🎙 “${tx.length > 140 ? tx.slice(0, 140) + "…" : tx}”` : "🎙 no clear speech")
      : "🎙 transcript pending…");
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
    const root = el("div", "rounded-2xl bg-white/[0.04] border border-white/10 p-4");
    const top = el("div", "flex items-center justify-between gap-3 flex-wrap");
    const left = el("div");
    const name = el("div", "font-semibold text-white", ch.name);
    const status = el("div", "text-xs text-emerald-200/70 mt-0.5");
    left.append(name, status);
    const right = el("div", "flex items-center gap-2");
    const db = el("span", "num font-mono text-sm text-emerald-100");
    const btn = el("button", "rounded-lg px-2.5 py-1 text-xs font-semibold bg-white/10 hover:bg-white/20 flex items-center gap-1");
    btn.type = "button"; btn.innerHTML = '<i data-lucide="tv"></i>'; btn.append(document.createTextNode(" Preview"));
    btn.addEventListener("click", () => openPreview(ch.name));
    right.append(db, btn); top.append(left, right);
    const meter = el("div", "h-2.5 rounded-full bg-white/10 mt-3 overflow-hidden");
    const fill = el("div", "bar-fill h-full rounded-full bg-mint"); fill.style.width = "0%"; meter.append(fill);
    const canvas = el("canvas", "w-full h-12 mt-2 block"); canvas.height = 48;
    const foot = el("div", "text-xs text-emerald-200/60 mt-1");
    root.append(top, meter, canvas, foot);
    return { root, status, db, btn, fill, canvas, foot };
  }

  function updateRadarRow(row, ch) {
    let label = STATE_LABEL[ch.state] || ch.state;
    if (ch.state === "backoff" && (ch.last_error || "").includes("ChannelUnavailable")) label = "Waiting for channel to go live";
    if (ch.state === "backoff" && ch.next_retry_at) label += ` · retry in ${Math.max(0, Math.round(ch.next_retry_at - Date.now() / 1000))}s`;
    const stab = ch.state !== "listening" ? "" : ch.disconnects_last_hour >= 3 ? " · unstable" :
      (ch.disconnects_last_hour || (ch.connected_for_s || 0) < 60) ? " · settling" : " · stable";
    setText(row.status, `${label}${stab} · ${ch.source === "engine" ? "24/7 engine" : "this app"}`);
    const lvl = ch.level_dbfs;
    setText(row.db, lvl === null || ch.state !== "listening" ? "— dBFS" : `${fmt(lvl)} dBFS${ch.silent ? " · silent" : ""}`);
    const pct = lvl === null ? 0 : Math.max(0, Math.min(100, (lvl + 60) / 60 * 100));
    row.fill.style.width = `${ch.state === "listening" ? pct : 0}%`;
    row.fill.className = `bar-fill h-full rounded-full ${ch.silent ? "bg-amber-400" : "bg-mint"}`;
    row.btn.disabled = !ch.preview; row.btn.classList.toggle("opacity-40", !ch.preview);
    row.btn.title = ch.preview ? "Open client-side video preview" : "No browser-playable stream for this channel";
    setText(row.foot, `peak 10 s ${fmt(ch.peak_dbfs_10s)} dBFS · buffer ${Math.round(ch.buffered_seconds || 0)} s · ` +
      `${ch.detections || 0} detections · ${ch.references || 0} refs` + (ch.last_error && ch.state !== "listening" ? ` · ${ch.last_error.slice(0, 90)}` : ""));
    drawWave(row.canvas, ch.level_history || []);
  }

  function drawWave(canvas, levels) {
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
      ctx.fillStyle = db < -60 ? "rgba(251,191,36,.7)" : `rgba(16,185,129,${0.35 + 0.65 * v})`;
      ctx.fillRect((n - data.length + i) * bw + 1, (h - bh) / 2, Math.max(1, bw - 2), bh);
    });
  }

  function renderEngine(e) {
    const dot = $("engine-dot"), pill = $("engine-pill");
    state.engineActive = !!(e && e.active);
    if (state.engineActive) {
      dot.className = "w-2.5 h-2.5 rounded-full bg-mint shadow-[0_0_10px_#10B981]";
      setText($("engine-label"), "24/7 Engine Active");
      setText($("engine-sub"), `${e.channels} channel(s) · heartbeat ${Math.round(e.heartbeat_age_s)} s ago · ${Math.round(e.rss_mb)} MB`);
      pill.classList.add("border-mint/50");
    } else if (e && (e.state === "running" || e.state === "starting")) {
      dot.className = "w-2.5 h-2.5 rounded-full bg-amber-400";
      setText($("engine-label"), "Engine not responding");
      setText($("engine-sub"), `last heartbeat ${Math.round(e.heartbeat_age_s)} s ago`);
      pill.classList.remove("border-mint/50");
    } else {
      dot.className = "w-2.5 h-2.5 rounded-full bg-gray-400";
      setText($("engine-label"), "24/7 Engine offline");
      setText($("engine-sub"), "start: scripts\\start_monitoring_service.bat");
      pill.classList.remove("border-mint/50");
    }
    const diag = $("diag");
    const pairs = e ? [["state", e.state], ["pid", e.pid], ["host", e.hostname], ["heartbeat", `${Math.round(e.heartbeat_age_s)} s`],
      ["RAM", `${Math.round(e.rss_mb)} MB`], ["ffmpeg", `${Math.round(e.ffmpeg_rss_mb || 0)} MB`], ["restarts", e.restarts]] : [["engine", "never started"]];
    diag.replaceChildren(...pairs.flatMap(([k, v]) => [el("dt", "", k), el("dd", "text-right text-emerald-100", String(v ?? "—"))]));
    syncToggle();
  }

  // ------------------------------------------------------------ channels + toggle
  async function loadChannels() {
    const r = await api("/api/channels");
    state.channels = r.channels;
    const sel = $("channel-select"), keep = sel.value;
    sel.replaceChildren(...r.channels.map((c) => { const o = el("option", "", `${c.featured ? "★ " : ""}${c.name}`); o.value = c.id; return o; }));
    sel.style.color = "white";
    [...sel.options].forEach((o) => { o.style.color = INK; });
    if (keep && r.channels.some((c) => c.id === keep)) sel.value = keep;
    syncToggle();
  }

  function renderMonitors(appChannels) {
    const running = new Set(appChannels.map((c) => c.name));
    state.channels.forEach((c) => { c.running = running.has(c.name); });
    $("monitor-list").replaceChildren(...appChannels.map((c) => {
      const li = el("li", "flex items-center gap-2 text-emerald-100/90");
      li.append(el("span", `w-2 h-2 rounded-full ${c.state === "listening" ? "bg-mint" : "bg-amber-400"}`), el("span", "truncate", c.name));
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
    btn.className = `w-full rounded-xl font-bold py-2 text-sm flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed ${running ? "bg-white/10 text-white border border-white/20 hover:bg-white/20" : "bg-mint text-petrol2 hover:brightness-110"}`;
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
    const dot = el("span", `inline-block w-2 h-2 rounded-full ml-1.5 align-middle ${lock === "LOCKED" ? "bg-mint" : lock === "TRACKING" ? "bg-amber-400" : "bg-gray-300"}`);
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
      const tr = el("tr", "border-b border-line/70 last:border-0");
      const mcr = a.mcr || {};
      const raw = a.algeria_timestamp ? a.algeria_timestamp.replace("T", " ").slice(0, 19) + " CET" : `${fmt(a.airing_start, 3)} s into file`;
      // Phase 26: the broadcast (MCR-calibrated) time leads; the raw ingest stamp stays visible in the tooltip
      const when = mcr.broadcast_start_cet ? mcr.broadcast_start_cet.replace("T", " ").slice(0, 19) + " CET" : raw;
      const integ = a.integrity === true ? ["PASS", "bg-mintsoft text-minting"] : a.integrity === false ? ["FAIL", "bg-red-50 text-red-700"] : ["no hash", "bg-gray-100 text-gray-600"];
      const link = el("a", "text-minting font-semibold hover:underline whitespace-nowrap", "📄 Certificate (PDF)"); link.title = "Tamper-evident proof-of-performance certificate"; link.href = `/api/certificate/${a.occurrence_id}`;
      const cells = [el("td", "py-2 pr-3 num text-muted", String(a.occurrence_id)), el("td", "pr-3", `${a.brand} · ${a.campaign}`),
        el("td", "pr-3", a.channel), syncCell(when, raw, mcr), el("td", "pr-3 text-right num", `${fmt(a.confidence * 100)}%`)];
      const ic = el("td", "pr-3"); ic.append(el("span", `pill ${integ[1]}`, integ[0]));
      const lc = el("td", "whitespace-nowrap"); lc.append(link);
      if (a.evidence_available) lc.append(" ", evidenceButton(a.occurrence_id, `${a.brand} · ${a.channel}`, a.evidence_sha256));
      tr.append(...cells, ic, lc); return tr;
    }));
    if (!r.airings.length) {
      const tr = el("tr"), td = el("td", "py-3 text-muted", "No detections yet.");
      td.colSpan = 7; tr.append(td); body.replaceChildren(tr);
    }
    syncCertificates();
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
    if (!a) { dl.setAttribute("aria-disabled", "true"); dl.removeAttribute("href"); setText($("cert-integrity"), "—"); setText($("cert-hash"), ""); return; }
    dl.setAttribute("aria-disabled", "false"); dl.href = `/api/certificate/${a.occurrence_id}`;
    setText($("cert-integrity"), a.integrity === true ? "🔐 Integrity check: PASS" : a.integrity === false ? "⚠️ Integrity check: FAIL — record or evidence changed"
      : "🔓 No hash recorded (detected before hashing existed)");
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
  const SOV_COLORS = ["#10B981", "#072A27", "#F59E0B", "#6366F1", "#EF4444", "#0EA5E9", "#A855F7", "#84CC16", "#94A3B8"];
  const dzdUnit = (v) => v === null || v === undefined ? "—" : `${dzd(Number(v))} DZD`;   // financial dzd() + unit
  const compState = { lastAlertId: null, ads: [], contracts: [] };

  async function loadCompliance() {
    const c = await api("/api/compliance/overview?limit=40");
    const k = c.kpis;
    setPill($("comp-card-status"), `rate card ${c.rate_card.status}`, c.rate_card.illustrative ? "neutral" : "good");
    $("comp-card-status").classList.remove("mt-2");
    setPill($("comp-pct"), k.compliance_pct === null ? "Compliance n/a (no judged clauses)" : `Compliance ${fmt(k.compliance_pct)}%`,
            k.compliance_pct === null ? "neutral" : k.compliance_pct >= 95 ? "good" : "bad");
    setPill($("comp-breaches"), `Breaches ${k.breaches} (position ${k.position_breaches} · separation ${k.separation_breaches})`,
            k.breaches ? "bad" : "neutral");
    setPill($("comp-reclaim"), `Reclaimable ${dzdUnit(k.reclaimable_net)}${c.rate_card.illustrative ? " (illustrative)" : ""}`,
            Number(k.reclaimable_net) > 0 ? "good" : "neutral");
    setPill($("comp-conflicts"), `Separation conflicts ${k.separation_conflicts}` +
            (k.provisional_conflicts ? ` (+${k.provisional_conflicts} unconfirmed)` : ""), k.separation_conflicts ? "bad" : "neutral");
    ["comp-pct", "comp-breaches", "comp-reclaim", "comp-conflicts"].forEach((id) => $(id).classList.remove("mt-2"));
    const hints = [];
    if (c.setup.needs_contracts) hints.push("no contract clauses yet");
    if (c.setup.brands_without_sector.length) hints.push(`${c.setup.brands_without_sector.length} brand(s) without a sector`);
    const verifiedN = Number(k.verified_contract_airings ?? k.verified_contracts_count ?? 0) || 0;
    const illustrativeN = Number(k.illustrative_airings ?? k.illustrative_estimates_count ?? 0) || 0;
    setPill($("comp-tracks"), `Verified contract: ${verifiedN} · Illustrative estimate: ${illustrativeN}`,
            verifiedN ? "good" : illustrativeN ? "warn" : "neutral");
    $("comp-tracks").classList.remove("mt-2");
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
    const b = el("button", "text-[11px] font-semibold text-minting hover:underline whitespace-nowrap", "\u25B6\uFE0F Play Evidence");
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
    setPill($("comp-alert-count"), String(r.alerts.length), r.alerts.some((a) => a.severity === "critical") ? "bad" : "neutral");
    $("comp-alert-count").classList.remove("mt-2");
    const maxId = r.alerts.reduce((m, a) => Math.max(m, a.id), 0);
    if (compState.lastAlertId !== null) r.alerts.filter((a) => a.id > compState.lastAlertId).forEach(toast);
    compState.lastAlertId = Math.max(compState.lastAlertId || 0, maxId);
    ul.replaceChildren();
    if (!r.alerts.length) { ul.append(el("li", "text-muted text-xs", "No open alerts.")); return; }
    const icon = { critical: "🚨", warning: "⚠️", info: "ℹ️" };
    r.alerts.forEach((a) => {
      const li = el("li", `rounded-xl border px-3 py-2 flex items-start justify-between gap-2 ${a.severity === "critical" ? "border-red-200 bg-red-50" : "border-line"}`);
      const txt = el("div");
      txt.append(el("div", "text-xs", `${icon[a.severity] || "•"} ${a.message}`),
                 el("div", "text-[11px] text-muted", `${a.kind.replace("_", " ")} · ${a.created_at.replace("T", " ").slice(5, 16)} UTC`));
      const ack = el("button", "text-[11px] underline text-muted shrink-0", "Acknowledge"); ack.type = "button";
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
    const t = el("div", `fixed bottom-4 right-4 z-50 max-w-sm rounded-xl shadow-lg px-4 py-3 text-sm text-white ${a.severity === "critical" ? "bg-red-600" : "bg-petrol"}`,
                 `${a.severity === "critical" ? "🚨" : "🔔"} ${a.message}`);
    t.setAttribute("role", "status");
    document.body.append(t);
    setTimeout(() => t.remove(), 9000);
  }

  async function loadSov() {
    const prov = $("sov-provisional")?.checked ? "&provisional=true" : "";
    const s = await api(`/api/compliance/sov?days=7${prov}`);
    setPill($("comp-sov"), s.self_share_24h === null ? "My SOV 24 h n/a" : `My SOV 24 h ${fmt(s.self_share_24h)}%`, "neutral");
    $("comp-sov").classList.remove("mt-2");
    const top = s.brands.slice(0, 8);
    setText($("sov-note"), s.brands.length ? top.map((b) => `${b.brand}${b.role === "self" ? " (mine)" : ""}: ${b.share_24h ?? 0}% 24 h` +
      (b.change_pts ? ` (${b.change_pts > 0 ? "+" : ""}${b.change_pts} pts)` : "")).join(" · ") : "No detected airings in the last 7 days.");
    if (!ensureCharts()) return;
    const data = { labels: s.days.map((d) => d.slice(5)), datasets: top.map((b, i) => ({
      label: b.brand, data: b.airtime.map((x) => Math.round(x)), stack: "sov",
      backgroundColor: b.role === "self" ? MINT : SOV_COLORS[(i % (SOV_COLORS.length - 1)) + 1], borderRadius: 3 })) };
    if (!charts.sov) {
      charts.sov = new Chart($("sov-chart"), { type: "bar", data, options: { responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.raw} s` } } },
        scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, beginAtZero: true, title: { display: true, text: "airtime (s)" } } } } });
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
      const input = el("input", "border border-line rounded-lg px-2 py-0.5 w-36"); input.value = a.sector || ""; input.placeholder = "sector";
      const save = el("button", "underline", "save"); save.type = "button";
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
      const rm = el("button", "underline text-muted", "remove"); rm.type = "button";
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
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#10B981";
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
    if (file.size > 16 * 2 ** 20) { setText($("master-status"), "⚠ over 16 MB: index it with `python -m kihc.cli register-ad`"); return; }
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
      st.replaceChildren(el("span", "pill bg-mintsoft text-minting", "✔ Indexed"),
        el("span", "ml-2 text-xs text-muted", `${r.asset_id} · ${r.duration_s}s (lead ${r.lead_trim_s}s trimmed) · ${r.hashes} landmarks · ` +
          (r.packshot ? `packshot pHash ${r.packshot.phash}` : "no packshot") + " · live monitor hot-reloads it, no restart"));
      loadComplianceSetup().catch(() => {}); loadStats().catch(() => {});
    } catch (e) { st.replaceChildren(el("span", "pill bg-red-50 text-red-700", `⚠ ${e.message}`)); }
    finally { $("master-submit").disabled = !master.file; }
  }

  function initMaster() {
    const sel = $("master-sector"); SECTORS.forEach((s) => { const o = el("option", "", s); o.value = s; sel.append(o); });
    api("/api/ads").then((d) => { [...new Set(d.ads.map((a) => a.brand_name))].forEach((b) => { const o = el("option"); o.value = b; $("master-brands").append(o); }); }).catch(() => {});
    $("master-open").addEventListener("click", () => $("master-modal").showModal());
    $("master-close").addEventListener("click", () => $("master-modal").close());
    $("master-file").addEventListener("change", (e) => pickMaster(e.target.files[0]));
    const drop = $("master-drop");
    ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("border-mint"); }));
    ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("border-mint"); }));
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
    } catch (err) { st.replaceChildren(el("span", "pill bg-red-50 text-red-700", `⚠ ${err.message}`)); }
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
  window.kihcLibsReady = () => {
    icons();
    if (ensureCharts() && state.lastStats) renderStats(state.lastStats);
  };

  function boot() {
    window.kihcLibsReady();          // libraries that finished before this script ran
    $("industry-select").addEventListener("change", (e) => { state.industry = e.target.value; loadStats().catch(() => {}); });
    $("channel-select").addEventListener("change", syncToggle);
    $("toggle-btn").addEventListener("click", toggle);
    $("cert-select").addEventListener("change", renderCertificate);
    document.querySelectorAll(".fin-mode").forEach((b) => b.addEventListener("click", () => {
      state.finMode = b.dataset.finMode;
      if (state.lastFin) renderFinancials(state.lastFin);
    }));
    $("preview-close").addEventListener("click", closePreview);
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
