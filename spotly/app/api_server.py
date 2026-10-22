"""Spotly web app: FastAPI JSON API + a static single-page bento dashboard (no Node.js).

    python -m spotly.app.api_server [--port 8502] [--host 127.0.0.1] [--db path]
    python -m spotly.cli web

Endpoints
  GET  /                         SPA (templates/index.html; static assets under /static)
  GET  /api/stats                executive overview (KPIs, SOV, channels, premium pods, clutter)
  GET  /api/telemetry            engine + live channel telemetry (dBFS, peak, state); DB/memory reads only
  GET  /api/airings              recent detections with SHA-256 hashes and integrity check
  GET  /api/certificate/{id}     one-page PDF proof-of-performance
  GET  /api/channels             channels that can be monitored from this app
  POST /api/stream/toggle        start/stop an in-app monitor  {"channel": "...", "action": "start"|"stop"}
  GET  /api/preview              client-side video preview document for a channel (iframe srcdoc)
  GET  /api/financials/overview  estimated spend KPIs (rate card x detected airings), top spender, credits
  GET  /api/financials/brands    per-brand gross/net, SOS vs SOV, position compliance, make-good list
  GET  /api/financials/export/csv  airing-by-airing financial audit CSV
  POST /api/contracts/ingest     bind a client insertion order (metadata JSON or document upload) -> Track A audit
  GET  /api/contracts            ingested contracts   |   GET /api/contracts/audit-log   append-only change log
  GET  /api/campaigns/families   parent-campaign rollup (airings / airtime / SOV) with each cut-down variant listed
  GET  /api/evidence/play/{occurrence_id}      breach evidence clip, byte-range streaming (in-browser playback)
  GET  /api/evidence/download/{occurrence_id}  same clip as an attachment (SHA-256 re-verified on every read)
  GET  /healthz

Security: there is no login, so the server binds to 127.0.0.1 by default. The only state-changing
endpoint requires a JSON body and a same-origin Origin header, so other websites open in the same
browser cannot drive it (cross-site POST / CSRF). Put an authenticating reverse proxy in front before
exposing it on a network.
"""
import argparse
import functools
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from datetime import datetime, timezone

from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spotly.core.discovery import DISCOVERED_DIR, ignore_asset, promote_asset
from spotly.core.discovery_report import discovered_airings, list_assets

from spotly.analytics import compliance_audit as ca
from spotly.database.schema import get_db, read_only
from spotly.analytics.financial_intelligence import (build_financial_market_report,
                                                   jsonable, report_csv)
from spotly.app import overview as ov
from spotly.app.components.visual_player import build_player_html, resolve_preview_source
from spotly.config import DEFAULT_DB_PATH
from spotly.connectors import iptv_manager
from spotly.analytics.brand_extractor import canonical_brand, clean_campaign
from spotly.stream import clock_sync
from spotly.stream.time_synchronizer import airing_mcr

log = logging.getLogger("spotly.api")
from spotly.core.live_worker import LiveStreamWorker
from spotly.core.proof import algeria_iso, verify_row
from spotly.reports.certificate import certificate_for
from spotly.service.headless_runner import read_service_status

APP_DIR = Path(__file__).resolve().parent
TEMPLATE = APP_DIR / "templates" / "index.html"
STATIC_DIR = APP_DIR / "static"
CHANNEL_CACHE_SECONDS = 600
HISTORY_POINTS = 60                       # level samples per channel sent to the browser (~12 s)


class ToggleRequest(BaseModel):
    channel: str
    action: str = "toggle"                # start | stop | toggle


class ClassifyRequest(BaseModel):
    brand: str = Field(min_length=1, max_length=120)
    campaign: str = Field("", max_length=160)          # blank: the server names it ("45 s spot")
    industry: str | None = Field(None, max_length=80)
    role: str = Field("self", pattern="^(self|competitor)$")
    competes_with: str | None = Field(None, max_length=120)


class ConfirmBrandRequest(BaseModel):
    # Client vs competitor is not knowable from audio and changes SOV, so the caller must choose it.
    role: str = Field(pattern="^(self|competitor)$")
    campaign: str | None = Field(None, max_length=160)
    competes_with: str | None = Field(None, max_length=120)


class ContractClause(BaseModel):
    brand: str = Field(min_length=1, max_length=120)
    campaign: str = Field("*", max_length=160)
    booked_position: str = Field("ANY", pattern="^(FIRST|LAST|FIRST_OR_LAST|ANY)$")
    adjacency: bool | None = None
    agency_discount: float | None = Field(None, ge=0, lt=1)
    competitor_separation_seconds: float | None = Field(None, gt=0, le=3600)
    separation_credit: float | None = Field(None, ge=0, le=1)
    rate_override_dzd: float | None = Field(None, ge=0)
    note: str | None = Field(None, max_length=300)


class ContractsRequest(BaseModel):
    contracts: list[ContractClause] = Field(max_length=200)


class ContractIngestRequest(BaseModel):
    """Either explicit metadata, or a structured document (`document_b64`; a .json document carries the metadata)."""
    contract_id: str | None = Field(None, max_length=60)
    brand_name: str | None = Field(None, max_length=120)
    channel_id: str | None = Field(None, max_length=120)
    booked_position: str | None = Field(None, pattern="^(FIRST_IN_POD|LAST_IN_POD|FIRST_OR_LAST|RUN_OF_BREAK)$")
    separation_window_seconds: int | None = Field(None, ge=0, le=3600)
    net_rate_per_second: float | None = Field(None, gt=0)
    demotion_penalty_pct: float | None = Field(None, ge=0, le=1)
    separation_breach_penalty_pct: float | None = Field(None, ge=0, le=1)
    status: str | None = Field(None, pattern="^(ACTIVE|EXPIRED)$")
    valid_from: str | None = Field(None, max_length=40)
    valid_to: str | None = Field(None, max_length=40)
    document_name: str | None = Field(None, max_length=200)
    document_b64: str | None = Field(None, min_length=8, max_length=22_000_000)


CONTRACT_DOC_EXT = (".json", ".pdf", ".csv", ".txt")


class RegisterAdRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    data_b64: str = Field(min_length=8, max_length=22_000_000)           # JSON body (CSRF guard): ~16 MB of media; bigger masters use the CLI
    brand: str = Field(min_length=1, max_length=120)
    sector: str = Field(min_length=1, max_length=80)
    campaign: str | None = Field(None, max_length=160)
    position: str = Field("ANY", pattern="^(FIRST|LAST|FIRST_OR_LAST|ANY)$")
    separation_window: float | None = Field(180, gt=0, le=3600)
    rate_override: float | None = Field(None, ge=0)
    parent_id: str | None = Field(None, max_length=120)
    cut_type: str | None = Field(None, pattern="^(MASTER|CUT_DOWN|TEASER|TAGGED)$")
    music_dominant: bool | None = None          # None: measured from the audio (voice-over share < 20 %)


class VerifyCertificateRequest(BaseModel):
    pdf_b64: str = Field(min_length=8, max_length=8_000_000)


class SectorRequest(BaseModel):
    sector: str = Field(min_length=1, max_length=80)


class AckRequest(BaseModel):
    note: str | None = Field(None, max_length=200)


class IgnoreRequest(BaseModel):
    label: str = Field("not an ad", min_length=1, max_length=120)


ASSET_ID_RE = re.compile(r"[A-Z0-9_]{1,40}_AD_\d{3,6}")


def _stream_ref(worker):
    src = getattr(worker, "_url", None)
    return src if isinstance(src, str) else getattr(src, "__name__", None)


def _ro(fn):
    """Historical dashboard reads: the ledger is opened `mode=ro` + `query_only`, so a poll can never take the write lock."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with read_only():
            return fn(*args, **kwargs)
    return run


def _slim_status(name: str, s: dict, source: str, stream_ref=None, restarts=None) -> dict:
    keys = ("state", "level_dbfs", "peak_dbfs_10s", "silent", "connected_for_s", "disconnects_last_hour",
            "buffered_seconds", "received_seconds", "detections", "references", "connect_attempts",
            "next_retry_at", "last_error")
    out = {k: s.get(k) for k in keys}
    # Phase 29 telemetry truth: how old the supervisor's last report is, and whether it is still a live channel at all
    out.update(link=s.get("link", "live"), data_age_s=s.get("data_age_s"), offline=bool(s.get("offline")))
    out.update(last_audio_epoch=s.get("last_audio_epoch"), audio_age_s=s.get("audio_age_s"))
    out.update(feed_label=s.get("feed_label") or "", discovery=s.get("discovery"), video=s.get("video"))
    out.update(name=name, source=source, restarts=restarts,
               level_history=[lvl for _, lvl in (s.get("level_history") or [])[-HISTORY_POINTS:]],
               preview=bool(resolve_preview_source(name, stream_ref)[0]))
    return out


def create_app(db_path=None, worker_factory=None, channel_loader=None, on_shutdown=None) -> FastAPI:
    db_path = str(db_path or DEFAULT_DB_PATH)
    factory = worker_factory or (lambda name, source, headers: LiveStreamWorker(name, source, db_path,
                                                                               headers=headers))
    loader = channel_loader or iptv_manager.fetch_channels

    @asynccontextmanager
    async def lifespan(app):
        yield
        for w in list(app.state.workers.values()):     # stop in-app monitors (pending matches are flushed)
            w.stop()
        if on_shutdown is not None:                    # e.g. the pilot's ingest supervisor
            on_shutdown()

    app = FastAPI(title="Spotly Broadcast Intelligence", version="7.0", lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None)
    app.state.db_path = db_path
    try:                                           # read-only endpoints (Phase 30) cannot create tables: do it once, here
        from spotly.data import reference_db as _rdb
        with get_db(db_path) as _conn:
            _rdb.ensure_schema(_conn)
    except Exception:
        log.exception("reference schema check failed")
    app.state.audio_root = DISCOVERED_DIR          # discovered-asset audio is served only from here
    app.state.workers = {}
    app.state.lock = threading.Lock()
    app.state.channel_cache = (0.0, [])
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------ CSRF guard for state-changing calls
    @app.middleware("http")
    async def same_origin_json_only(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
            if not request.headers.get("content-type", "").startswith("application/json"):
                return JSONResponse({"detail": "JSON body required"}, status_code=415)
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    # ------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(TEMPLATE, media_type="text/html")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "db": app.state.db_path}

    # ------------------------------------------------------------ data
    @app.get("/api/stats")
    @_ro
    def stats(industry: str | None = Query(None, description="industry for Share of Voice; default = busiest"),
              days: int = Query(14, ge=1, le=90)):
        out = ov.overview(app.state.db_path, industry if industry else "__auto__", days)
        out["financials"] = _financial_overview(build_financial_market_report(app.state.db_path))
        return out

    # ------------------------------------------------------------ financials (estimates from the rate card)
    def _report(start, end, industry, verified_only):
        try:
            return build_financial_market_report(app.state.db_path, start, end, industry=industry or None,
                                                 verified_only=verified_only)
        except ValueError as exc:                      # e.g. malformed date
            raise HTTPException(422, str(exc))

    def _financial_overview(rep):
        top = rep["top_spender"]
        return jsonable(dict(
            currency=rep["currency"], rate_card=rep["rate_card"], basis=rep["basis"], filters=rep["filters"],
            gross_spend=rep["totals"]["gross_spend"], estimated_net_spend=rep["totals"]["estimated_net_spend"],
            priced_airings=rep["totals"]["priced_airings"], unpriced_airings=rep["totals"]["unpriced_airings"],
            unpriced=rep["unpriced"], contract_loss_credit=rep["totals"]["contract_loss_credit"],
            contract_loss_credit_gross=rep["totals"]["contract_loss_credit_gross"],
            bonus_premium_value=rep["totals"]["bonus_premium_value"], breaches=rep["totals"]["breaches"],
            top_spender=dict(brand=top["brand"], gross_spend=top["gross_spend"],
                             estimated_net_spend=top["estimated_net_spend"], sos_pct=top["sos_pct"],
                             sov_pct=top["sov_pct"]) if top else None))

    @app.get("/api/financials/overview")
    @_ro
    def financials_overview(start: str | None = None, end: str | None = None, industry: str | None = None,
                            verified_only: bool = False):
        return _financial_overview(_report(start, end, industry, verified_only))

    @app.get("/api/financials/brands")
    @_ro
    def financials_brands(start: str | None = None, end: str | None = None, industry: str | None = None,
                          verified_only: bool = False):
        rep = _report(start, end, industry, verified_only)
        return jsonable(dict(currency=rep["currency"], rate_card=rep["rate_card"], basis=rep["basis"],
                             brands=rep["brands"], make_goods=rep["make_goods"]))

    @app.get("/api/financials/export/csv")
    def financials_csv(start: str | None = None, end: str | None = None, industry: str | None = None,
                       verified_only: bool = False):
        text = report_csv(_report(start, end, industry, verified_only))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return StreamingResponse(iter([text.encode("utf-8")]), media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition":
                                          f'attachment; filename="SPOTLY-financial-audit-{stamp}.csv"'})

    # ------------------------------------------------------------ compliance, separation, SOV, alerts (Phase 12.1)
    def _compliance(start, end, provisional):
        try:
            return ca.compliance_report(app.state.db_path, start, end, include_provisional=provisional)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/api/compliance/overview")
    def compliance_overview(start: str | None = None, end: str | None = None, provisional: bool = False,
                            limit: int = Query(50, ge=1, le=500)):
        rep = _compliance(start, end, provisional)
        flagged = [a for a in rep["airings"] if a["conflicts"] or a["position_compliance"] == "breach"
                   or a["separation"] == "breach"]
        return jsonable(dict({k: rep[k] for k in ("currency", "rate_card", "contracts", "filters", "basis", "kpis",
                                                  "setup", "generated_at")},
                             conflicts=rep["conflicts"][:limit], flagged_airings=flagged[:limit]))

    @app.get("/api/compliance/export/csv")
    def compliance_csv(start: str | None = None, end: str | None = None):
        text = ca.claim_sheet_csv(_compliance(start, end, False))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return StreamingResponse(iter([text.encode("utf-8")]), media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition":
                                          f'attachment; filename="SPOTLY-make-good-claims-{stamp}.csv"'})

    @app.get("/api/compliance/sov")
    def compliance_sov(days: int = Query(7, ge=1, le=90), sector: str | None = None, provisional: bool = False):
        return ca.sov_timeline(app.state.db_path, days, sector=sector or None, include_provisional=provisional)

    @app.get("/api/compliance/alerts")
    def compliance_alerts(include_acknowledged: bool = False, limit: int = Query(30, ge=1, le=200)):
        return {"alerts": ca.list_alerts(app.state.db_path, include_acknowledged, limit)}

    @app.post("/api/compliance/alerts/{alert_id}/ack")
    def compliance_ack(alert_id: int, req: AckRequest):
        if not ca.acknowledge_alert(app.state.db_path, alert_id):
            raise HTTPException(404, "unknown or already acknowledged alert")
        return {"id": alert_id, "acknowledged": True}

    @app.get("/api/compliance/contracts")
    def compliance_contracts():
        return {"contracts": ca.get_contracts(app.state.db_path)}

    @app.post("/api/compliance/contracts")
    def compliance_save_contracts(req: ContractsRequest):
        from spotly.analytics.rate_engine import RateCardError
        try:
            saved = ca.save_contracts(app.state.db_path, [c.model_dump(exclude_none=True) for c in req.contracts])
        except (RateCardError, KeyError, ValueError) as exc:
            raise HTTPException(422, str(exc))
        return {"contracts": saved}

    # ------------------------------------------------------------ dual-track contracts (Phase 18)
    @app.post("/api/contracts/ingest")
    def contracts_ingest(req: ContractIngestRequest, request: Request):
        import base64
        import binascii
        import hashlib
        import json as _json

        from spotly.analytics import contracts as ctr
        from spotly.config import DATA_DIR
        fields = req.model_dump(exclude_none=True, exclude={"document_name", "document_b64"})
        if req.document_b64:
            suffix = Path(req.document_name or "contract.json").suffix.lower()
            if suffix not in CONTRACT_DOC_EXT:
                raise HTTPException(422, f"unsupported document type; use one of {', '.join(CONTRACT_DOC_EXT)}")
            try:
                raw = base64.b64decode(req.document_b64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(422, "document payload is not valid base64")
            if suffix == ".json":
                try:
                    doc = _json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    raise HTTPException(422, "contract document is not valid UTF-8 JSON")
                if not isinstance(doc, dict):
                    raise HTTPException(422, "contract document must be a JSON object")
                fields = {**{k: v for k, v in doc.items() if k in ctr.FIELDS}, **fields}   # explicit fields win
            if not fields.get("contract_id"):
                raise HTTPException(422, "contract_id is required to store the document")
            folder = Path(DATA_DIR) / "contracts"
            folder.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(fields["contract_id"]))[:60]
            target = folder / f"{safe_id}-{hashlib.sha256(raw).hexdigest()[:12]}{suffix}"
            target.write_bytes(raw)
            fields["contract_document_path"] = str(target)
        origin = f"{request.client.host if request.client else '?'}|{request.headers.get('user-agent', '')}"
        try:
            return ctr.ingest_contract(app.state.db_path, fields, origin=origin)
        except ctr.ContractError as exc:
            raise HTTPException(409 if "conflicting" in str(exc) else 422, str(exc))

    @app.get("/api/contracts")
    def contracts_list(status: str | None = Query(None, pattern="^(ACTIVE|EXPIRED)$")):
        from spotly.analytics import contracts as ctr
        return {"contracts": ctr.list_contracts(app.state.db_path, status)}

    @app.get("/api/contracts/audit-log")
    def contracts_audit_log(contract_id: str | None = None):
        from spotly.analytics import contracts as ctr
        return {"intact": ctr.verify_audit_chain(app.state.db_path),
                "entries": ctr.audit_log(app.state.db_path, contract_id)}

    @app.post("/api/ads/register")
    def register_master(req: RegisterAdRequest):
        """Pre-broadcast onboarding of a master creative (.mp4/.wav/.mp3); live monitors hot-reload the signature."""
        import base64
        import binascii
        import tempfile

        from spotly.analytics.ad_onboarding import ALLOWED_EXT, OnboardingError, register_ad
        suffix = Path(req.filename).suffix.lower()
        if suffix not in ALLOWED_EXT:
            raise HTTPException(422, f"unsupported file type; use one of {', '.join(ALLOWED_EXT)}")
        try:
            raw = base64.b64decode(req.data_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(422, "file payload is not valid base64")
        with tempfile.TemporaryDirectory(prefix="spotly_upload_") as tmp:
            path = Path(tmp) / f"master{suffix}"
            path.write_bytes(raw)
            del raw
            try:
                return register_ad(path, req.brand, req.sector, campaign=req.campaign, position=req.position,
                                   separation_window=req.separation_window, rate_override=req.rate_override,
                                   db_path=app.state.db_path, original_name=Path(req.filename).name,
                                   parent_id=req.parent_id, cut_type=req.cut_type,
                                   music_dominant=req.music_dominant)
            except OnboardingError as exc:
                raise HTTPException(422, str(exc))

    @app.get("/api/campaigns/families")
    @_ro
    def campaign_families(days: int | None = Query(None, ge=1, le=365)):
        """Parent-campaign rollup: airings / airtime / SOV per campaign with each cut-down variant listed under it."""
        import time as _t

        from spotly.analytics.campaign_family import campaign_rollup
        return campaign_rollup(app.state.db_path, since_epoch=_t.time() - days * 86400 if days else None)

    @app.post("/api/ads/{ad_id}/sector")
    def ad_sector(ad_id: int, req: SectorRequest):
        from spotly.core.library import set_ad_tags
        from spotly.database.schema import get_db
        with get_db(app.state.db_path) as conn:
            row = conn.execute("SELECT ad_type, target_competitor_of FROM reference_ads WHERE ad_id = ?", (ad_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "unknown reference ad")
        try:                                            # keep role / target, change only the sector
            set_ad_tags(ad_id, req.sector.strip(), row["ad_type"] or "self", row["target_competitor_of"],
                        db_path=app.state.db_path)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"ad_id": ad_id, "sector": req.sector.strip()}

    @app.get("/api/ads")
    @_ro
    def ads():
        from spotly.database.schema import get_db
        with get_db(app.state.db_path) as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT r.ad_id, r.brand_name, r.campaign_name, r.industry AS sector, r.ad_type AS role, r.duration_sec, "
                "(SELECT COUNT(*) FROM ad_occurrences o WHERE o.ad_id = r.ad_id) AS airings "
                "FROM reference_ads r ORDER BY r.brand_name")]
        for r in rows:                                   # user-facing names only: no internal ids, registry brand spelling
            r["brand_name"] = canonical_brand(r["brand_name"])
            r["campaign_name"] = clean_campaign(r["brand_name"], r["campaign_name"], r.pop("duration_sec"))
        return {"ads": rows}

    @app.get("/api/airings")
    @_ro
    def airings(limit: int = Query(25, ge=1, le=200), verify: bool = Query(True)):
        rows = ov.load_airings(app.state.db_path)[:limit]
        from spotly.database.schema import get_db
        with get_db(app.state.db_path) as conn:
            clips = {r[0]: r[1] for r in conn.execute(
                "SELECT occurrence_id, evidence_sha256 FROM ad_occurrences WHERE evidence_path IS NOT NULL")}
        out = []
        for r in rows:
            live = r["source_name"].startswith("live:")
            # Live rows store epoch seconds; rows from before Algeria timestamps were stored get one derived
            # exactly as the live worker does (not part of any hash: such rows report integrity "no hash").
            cet = r["algeria_timestamp"] or (algeria_iso(r["airing_start"]) if live else None)
            out.append(dict(
                occurrence_id=r["occurrence_id"], brand=canonical_brand(r["brand_name"]),
                campaign=clean_campaign(canonical_brand(r["brand_name"]), r["campaign_name"], r["duration_detected"]),
                channel=ov.channel_of(r["source_name"]), source=r["source_name"],
                live=live, airing_start=r["airing_start"],
                airing_end=r["airing_end"], algeria_timestamp=cet,
                duration=r["duration_detected"], confidence=r["confidence_score"],
                review={1: "verified", 0: "unreviewed"}.get(r["human_verified"], "unknown"),
                detected_at=r["detected_at"], verification_hash=r["verification_hash"],
                integrity=verify_row(r) if verify else None,
                mcr=airing_mcr(r),                                        # Phase 26: calibrated broadcast time + sync lock
                evidence_available=r["occurrence_id"] in clips, evidence_sha256=clips.get(r["occurrence_id"])))
        return {"airings": out, "count": len(out)}

    # ------------------------------------------------------------ breach evidence clips (Phase 19)
    def _evidence_file(occurrence_id: int, request: Request) -> tuple[Path, str]:
        """Authorise, locate and integrity-check the clip. 404 = none recorded; 409 = file no longer matches its seal."""
        import hmac
        import os

        from spotly.analytics import evidence_recorder as er
        token = os.environ.get("SPOTLY_EVIDENCE_TOKEN")           # optional bearer secret (the app has no user login)
        if token:
            given = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or \
                request.query_params.get("token", "")
            if not hmac.compare_digest(given, token):
                raise HTTPException(401, "evidence access requires a valid token")
        rec = er.evidence_for(app.state.db_path, occurrence_id)
        if rec is None:
            raise HTTPException(404, "no evidence clip recorded for this airing")
        check = er.verify_evidence(app.state.db_path, occurrence_id)
        if check["status"] == "MISSING":
            raise HTTPException(404, "evidence file is missing from disk")
        if check["status"] != "PASS":
            raise HTTPException(409, "evidence file FAILED its SHA-256 integrity check (altered after sealing)")
        return er.resolve_path(rec["path"]), rec["sha256"]

    def _serve_clip(path: Path, sha: str, request: Request, attachment: bool, occurrence_id: int):
        size = path.stat().st_size
        headers = {"Accept-Ranges": "bytes", "X-Evidence-SHA256": sha,
                   "Content-Disposition": (f'attachment; filename="SPOTLY-evidence-{occurrence_id:06d}.wav"' if attachment
                                           else f'inline; filename="SPOTLY-evidence-{occurrence_id:06d}.wav"')}
        start, end, status = 0, size - 1, 200
        rng = request.headers.get("range")
        if rng and not attachment:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if not m or not (m.group(1) or m.group(2)):
                raise HTTPException(416, "bad range", headers={"Content-Range": f"bytes */{size}"})
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:                                                 # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            if start >= size or start > end:
                raise HTTPException(416, "range not satisfiable", headers={"Content-Range": f"bytes */{size}"})
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = end - start + 1
        headers["Content-Length"] = str(length)

        def body():
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    block = f.read(min(1 << 16, left))
                    if not block:
                        break
                    left -= len(block)
                    yield block
        return StreamingResponse(body(), status_code=status, media_type="audio/wav", headers=headers)

    @app.get("/api/evidence/play/{occurrence_id}")
    def evidence_play(occurrence_id: int, request: Request):
        path, sha = _evidence_file(occurrence_id, request)
        return _serve_clip(path, sha, request, False, occurrence_id)

    @app.get("/api/evidence/download/{occurrence_id}")
    def evidence_download(occurrence_id: int, request: Request):
        path, sha = _evidence_file(occurrence_id, request)
        return _serve_clip(path, sha, request, True, occurrence_id)

    @app.get("/api/certificate/{occurrence_id}")
    def certificate(occurrence_id: int):
        from spotly.analytics.certificate_authority import generate_airing_certificate_pdf
        try:
            pdf = generate_airing_certificate_pdf(occurrence_id, app.state.db_path)
        except KeyError:
            raise HTTPException(404, f"occurrence {occurrence_id} not found")
        except Exception:                                  # a certificate request is never answered with a 500
            log.exception("certificate %s failed outright", occurrence_id)
            from spotly.analytics.certificate_authority import fallback_pdf
            try:
                pdf = fallback_pdf(occurrence_id, app.state.db_path, "unexpected error")
            except KeyError:
                raise HTTPException(404, f"occurrence {occurrence_id} not found")
        return Response(pdf, media_type="application/pdf", headers={
            "Content-Disposition": f'attachment; filename="SPOTLY-certificate-{occurrence_id:06d}.pdf"'})

    @app.get("/api/certificates/export-batch")
    def certificates_export_batch(start_date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
                                  end_date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
                                  brand: str | None = Query(None, max_length=120),
                                  channel: str | None = Query(None, max_length=120)):
        """ZIP of one tamper-evident certificate per verified airing in the filter, plus MANIFEST.csv (max 500)."""
        from spotly.analytics.certificate_authority import export_batch_zip
        data, info = export_batch_zip(app.state.db_path, start_date, end_date, brand, channel)
        if not info["count"]:
            raise HTTPException(404, "no airings match the filter")
        return Response(data, media_type="application/zip", headers={
            "Content-Disposition": 'attachment; filename="SPOTLY-certificates.zip"',
            "X-SPOTLY-Certificates": str(info["count"]), "X-SPOTLY-Truncated": str(info["truncated"]).lower()})

    @app.post("/api/certificates/verify")
    def certificates_verify(req: VerifyCertificateRequest):
        import base64
        import binascii

        from spotly.analytics.certificate_authority import verify_certificate_pdf
        try:
            pdf = base64.b64decode(req.pdf_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(422, "pdf_b64 is not valid base64")
        rep = verify_certificate_pdf(pdf, app.state.db_path)
        rep.pop("payload", None)
        return rep

    # ------------------------------------------------------------ live
    @app.get("/api/telemetry")
    def telemetry():
        engine = read_service_status(app.state.db_path)
        channels = []
        if engine and engine["active"]:
            for c in engine["channel_status"]:
                channels.append(_slim_status(c["name"], c, "engine", c.get("stream_ref"), c.get("restarts")))
        with app.state.lock:
            workers = dict(app.state.workers)
        for name, w in workers.items():
            channels.append(_slim_status(name, w.status(), "app", _stream_ref(w)))
        eng = None
        if engine:
            eng = {k: engine.get(k) for k in ("active", "state", "pid", "hostname", "started_at", "heartbeat_age_s",
                                              "rss_mb", "ffmpeg_rss_mb", "restarts", "channels", "message",
                                              "live_channels", "offline_channels", "orphan_rows_ignored")}
        clocks = clock_sync.load_all(app.state.db_path)           # Phase 22: capture-vs-broadcaster offset per channel
        for c in clocks:
            c["age_s"] = round(time.time() - c["updated_epoch"], 1)
            c["stale"] = c["age_s"] > 600.0                       # time_synchronizer.FRESH_S: no longer used to stamp airings
        return {"engine": eng, "channels": channels, "clock": clocks, "server_time": time.time()}

    def _channel_catalog():
        stamp, cached = app.state.channel_cache
        if time.monotonic() - stamp > CHANNEL_CACHE_SECONDS or not cached:
            featured_names = {iptv_manager.NATIONAL_BROADCASTERS[k].name for k in iptv_manager.FEATURED_KEYS}
            catalog = [dict(id=f"channel:{k}", name=iptv_manager.NATIONAL_BROADCASTERS[k].name, featured=True)
                       for k in iptv_manager.FEATURED_KEYS]
            try:
                extra = [c for c in loader() if c.name not in featured_names]
            except Exception:
                extra = []
            catalog += [dict(id=f"iptv:{c.name}", name=c.name, featured=False, url=c.url, headers=c.headers)
                        for c in extra]
            app.state.channel_cache = (time.monotonic(), catalog)
        return app.state.channel_cache[1]

    @app.get("/api/channels")
    def channels():
        with app.state.lock:
            running = {n for n, w in app.state.workers.items() if w.is_alive()}
        engine = read_service_status(app.state.db_path)
        return {"channels": [dict(id=c["id"], name=c["name"], featured=c["featured"], running=c["name"] in running)
                             for c in _channel_catalog()],
                "engine_active": bool(engine and engine["active"])}

    @app.post("/api/stream/toggle")
    def toggle(req: ToggleRequest):
        entry = next((c for c in _channel_catalog() if req.channel in (c["id"], c["name"])), None)
        if entry is None:
            raise HTTPException(404, f"unknown channel {req.channel!r}")
        name = entry["name"]
        with app.state.lock:
            worker = app.state.workers.get(name)
            running = bool(worker and worker.is_alive())
            action = req.action if req.action in ("start", "stop") else ("stop" if running else "start")
            if action == "stop":
                if not running:
                    raise HTTPException(409, f"{name} is not being monitored by this app")
                app.state.workers.pop(name, None)
            else:
                engine = read_service_status(app.state.db_path)
                if engine and engine["active"]:
                    raise HTTPException(409, "the 24/7 background engine is running; it owns live monitoring")
                if running:
                    raise HTTPException(409, f"{name} is already being monitored")
                if entry["id"].startswith("channel:"):
                    source, headers = iptv_manager.channel_resolver(entry["id"][len("channel:"):]), {}
                else:
                    source, headers = entry["url"], entry.get("headers") or {}
                app.state.workers[name] = factory(name, source, headers).start()
        if action == "stop":
            threading.Thread(target=worker.stop, daemon=True).start()     # don't block the request
        return {"channel": name, "action": action, "running": action == "start"}

    # ------------------------------------------------------------ discovery (unsupervised recurring assets)
    @app.get("/api/discovery/assets")
    def discovery_assets(channel: str | None = None, include_ignored: bool = True, include_merged: bool = False):
        assets = list_assets(app.state.db_path, channel, include_ignored, include_merged, with_metrics=True)
        for a in assets:
            a.pop("audio_path", None)                     # never expose filesystem paths
            a.pop("packshot_path", None)
            a.pop("logo_path", None)
            a["audio_url"] = f"/api/discovery/assets/{a['asset_id']}/audio" if a["has_audio"] else None
            a["packshot_url"] = f"/api/discovery/assets/{a['asset_id']}/packshot" if a["has_packshot"] else None
            a["logo_url"] = f"/api/discovery/assets/{a['asset_id']}/logo" if a["has_logo"] else None
        return {"assets": assets, "count": len(assets),
                "note": "Recurring segments found without reference files. Unclassified assets may be ads, "
                        "promos, idents or programme openers; they enter spend analytics only once classified."}

    @app.get("/api/discovery/occurrences")
    def discovery_occurrences(channel: str | None = None, limit: int = Query(100, ge=1, le=1000)):
        return {"occurrences": discovered_airings(app.state.db_path, channel, limit)}

    def _asset_row(asset_id: str):
        if not ASSET_ID_RE.fullmatch(asset_id):
            raise HTTPException(404, "unknown asset")
        rows = [a for a in list_assets(app.state.db_path, include_merged=True) if a["asset_id"] == asset_id]
        if not rows:
            raise HTTPException(404, "unknown asset")
        return rows[0]

    def _cached_file(stored: str | None, what: str) -> Path:
        if not stored:
            raise HTTPException(404, f"{what} not available")
        path = Path(stored).resolve()
        # Only files inside the discovery cache are ever served (defence against a tampered DB path).
        if not path.is_relative_to(Path(app.state.audio_root).resolve()) or not path.is_file():
            raise HTTPException(404, f"{what} not available")
        return path

    @app.get("/api/discovery/assets/{asset_id}/audio")
    def discovery_audio(asset_id: str):
        path = _cached_file(_asset_row(asset_id)["audio_path"], "audio")
        return FileResponse(path, media_type="audio/wav", filename=f"{asset_id}.wav")

    @app.get("/api/discovery/assets/{asset_id}/packshot")
    def discovery_packshot(asset_id: str):
        path = _cached_file(_asset_row(asset_id)["packshot_path"], "packshot")
        return FileResponse(path, media_type="image/jpeg", filename=f"{asset_id}_packshot.jpg")

    @app.get("/api/discovery/assets/{asset_id}/logo")
    def discovery_logo(asset_id: str):
        path = _cached_file(_asset_row(asset_id)["logo_path"], "logo crop")
        return FileResponse(path, media_type="image/jpeg", filename=f"{asset_id}_logo.jpg")

    @app.post("/api/discovery/assets/{asset_id}/classify")
    def discovery_classify(asset_id: str, req: ClassifyRequest):
        _asset_row(asset_id)
        try:
            return promote_asset(app.state.db_path, asset_id, req.brand.strip(), req.campaign.strip(),
                                 industry=req.industry, ad_type=req.role, target_competitor_of=req.competes_with)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/discovery/assets/{asset_id}/confirm-brand")
    def discovery_confirm_brand(asset_id: str, req: ConfirmBrandRequest):
        """One click: promote the asset with its *suggested* brand into reference ads / rate-card auditing."""
        a = _asset_row(asset_id)
        if not a.get("suggested_brand"):
            raise HTTPException(422, "no brand suggestion for this asset; classify it manually")
        if a["status"] != "unclassified":
            raise HTTPException(409, f"asset is already {a['status']}")
        try:
            return promote_asset(app.state.db_path, asset_id, a["suggested_brand"],
                                 req.campaign or "", industry=a.get("suggested_industry"),
                                 ad_type=req.role, target_competitor_of=req.competes_with)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/discovery/assets/{asset_id}/ignore")
    def discovery_ignore(asset_id: str, req: IgnoreRequest):
        _asset_row(asset_id)
        ignore_asset(app.state.db_path, asset_id, req.label)
        return {"asset_id": asset_id, "status": "ignored", "label": req.label}

    @app.get("/api/preview", response_class=HTMLResponse)
    def preview(channel: str):
        ref = None
        with app.state.lock:
            w = app.state.workers.get(channel)
        if w is not None:
            ref = _stream_ref(w)
        else:
            engine = read_service_status(app.state.db_path)
            if engine and engine["active"]:
                ref = next((c.get("stream_ref") for c in engine["channel_status"] if c["name"] == channel), None)
        cat = {c["name"]: c.get("url") for c in _channel_catalog() if c.get("url")}
        url, is_yt = resolve_preview_source(channel, ref, overrides=iptv_manager.load_overrides(), iptv_urls=cat)
        return HTMLResponse(build_player_html(url, channel, is_yt))

    return app


app = create_app()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Spotly web app (FastAPI + bento SPA)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    ap.add_argument("--port", type=int, default=8502)
    ap.add_argument("--db", default=None, help="database path (default: spotly/data/spotly.db)")
    a = ap.parse_args(argv)
    import uvicorn
    print(f"Spotly web app: http://{a.host}:{a.port}  (API docs: /api/docs)")
    uvicorn.run(create_app(a.db), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
