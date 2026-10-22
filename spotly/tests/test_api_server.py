"""Phase 7: FastAPI web app - endpoints, figures parity, certificates, toggle safety, SPA delivery."""
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from spotly.app import overview as ov
from spotly.app.api_server import create_app
from spotly.core.analytics import compute_share_of_voice
from spotly.core.ingest import ingest_reference
from spotly.core.matcher import scan_source
from spotly.core.pod_analyzer import cluster_occurrences_into_pods, position_distribution
from spotly.service.headless_runner import ChannelSpec, Supervisor
from spotly.tests.generate_test_assets import generate
from spotly.tests.test_pod_analyzer import Market, T

ROOT = Path(__file__).resolve().parents[2]
SAME_ORIGIN = {"origin": "http://testserver", "content-type": "application/json"}


class FakeWorker:
    def __init__(self, name, source, headers):
        self.name, self._url, self.headers, self.alive, self.stopped = name, source, headers, False, False

    def start(self):
        self.alive = True
        return self

    def stop(self):
        self.alive, self.stopped = False, True

    def is_alive(self):
        return self.alive

    def status(self):
        return dict(state="listening", level_dbfs=-21.5, peak_dbfs_10s=-12.0, silent=False, connected_for_s=120.0,
                    disconnects_last_hour=0, buffered_seconds=30.0, received_seconds=300.0, detections=2,
                    references=3, connect_attempts=1, next_retry_at=None, last_error="",
                    level_history=[(time.time() - i, -20.0 - i % 5) for i in range(100)])


class FakeChannel:
    def __init__(self, name, url):
        self.name, self.url, self.headers = name, url, {}


@pytest.fixture
def seeded(tmp_path):
    """Live breaks (fictional brands, tagged) + one real scanned airing with hash + one rejected airing."""
    m = Market(tmp_path / "web.db")
    m.ad("Atlas Mobile", "self")
    m.ad("Sahara Telecom", "competitor")
    m.chain(["Atlas Mobile", "Sahara Telecom", "Atlas Mobile"], T(20))
    m.chain(["Sahara Telecom", "Atlas Mobile"], T(21), channel="Ennahar TV")
    m.air("Sahara Telecom", T(22), 30, review=-1)                          # rejected: excluded everywhere
    m.conn.execute("UPDATE reference_ads SET industry = 'Telecom'")
    m.conn.execute("UPDATE reference_ads SET target_competitor_of = 'Atlas Mobile' WHERE ad_type = 'competitor'")
    m.commit()
    m.conn.close()
    ad_a, _, broadcast = generate()
    ingest_reference(ad_a, "Acme", "Spot A", db_path=m.db)
    scan_source(broadcast, "Echorouk TV · rec.mp4", db_path=m.db, evidence_dir=tmp_path / "ev")
    return m.db


@pytest.fixture
def client(seeded):
    app = create_app(seeded, worker_factory=FakeWorker,
                     channel_loader=lambda: [FakeChannel("TV2 (Algeria)", "http://185.9.2.18/chid_347/index.m3u8")])
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------ pages

def test_spa_and_static_assets(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    for element_id in ("kpi-total", "kpi-conf", "kpi-sov", "chart-premium", "channel-list", "chart-clutter",
                       "radar-list", "cert-select", "cert-download", "preview-modal", "toggle-btn", "engine-pill"):
        assert f'id="{element_id}"' in html, element_id
    assert all(f"/static/vendor/{lib}" in html for lib in ("tailwind.play.js", "chart.umd.min.js", "lucide.min.js"))   # Phase 31.1: vendored
    assert 'sandbox="allow-scripts allow-presentation allow-popups"' in html     # no allow-same-origin
    import re
    for lib in ("chart.umd.min.js", "lucide.min.js"):                           # optional libs never block the app
        tag = re.search(rf'<script[^>]*{re.escape(lib)}[^>]*>', html).group(0)
        assert " async" in tag and " defer" not in tag, tag
    assert all(re.search(r'class="relative[^"]*"><canvas id="' + c, html) for c in
               ("chart-perday", "chart-premium", "chart-clutter"))                # Chart.js responsive sizing
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "setTimeout(run, every)" in js.text and "innerHTML = '<i" in js.text
    assert client.get("/healthz").json()["status"] == "ok"


def test_spa_never_injects_server_strings_as_html():
    js = (ROOT / "spotly" / "app" / "static" / "app.js").read_text(encoding="utf-8")
    # the only innerHTML use is a static icon literal; data goes through textContent/createElement
    import re
    assignments = re.findall(r"\.innerHTML\s*=\s*([^;]+);", js)
    assert assignments == ["'<i data-lucide=\"tv\"></i>'"]


# ------------------------------------------------------------------ stats parity

def test_stats_match_core_analytics(client, seeded):
    s = client.get("/api/stats").json()
    rows = ov.load_airings(seeded)
    assert s["total_airings"] == len(rows) == 6                              # 5 live + 1 scanned; rejected excluded
    assert sum(d["count"] for d in s["per_day"]) == 6 and len(s["per_day"]) == 14
    sov = compute_share_of_voice("Telecom", db_path=seeded)
    assert s["sov"]["industry"] == "Telecom" and s["sov"]["client_pct"] == pytest.approx(sov["self_sov_airtime_pct"])
    assert s["sov"]["leader"]["brand"] == sov["brands"][0]["brand"]
    dist = [d for d in position_distribution(cluster_occurrences_into_pods(db_path=seeded)) if d["role"] == "self"]
    in_breaks = sum(d["first"] + d["middle"] + d["last"] for d in dist)
    premium = sum(d["first"] + d["last"] for d in dist)
    assert s["premium"]["in_break_airings"] == in_breaks and s["premium"]["pct"] == pytest.approx(100 * premium / in_breaks)
    assert {c["name"] for c in s["channels"]} == {"Echorouk TV", "Ennahar TV"}
    assert sum(c["pct"] for c in s["channels"]) == pytest.approx(100)
    assert s["clutter"]["mode"] == "hourly" and s["clutter"]["peak"]["hour"] == 20
    assert s["confidence"]["delta_pts"] is None                               # everything logged today: no prior week


def test_stats_industry_override(client):
    s = client.get("/api/stats", params={"industry": "Dairy"}).json()
    assert s["sov"]["industry"] == "Dairy" and s["sov"]["total_airings"] == 0


def test_confidence_delta_against_prior_week():
    now = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
    rows = [dict(detected_at=(now - timedelta(days=d)).isoformat(), confidence_score=c)
            for d, c in [(1, 0.95), (2, 0.93), (9, 0.88), (10, 0.86)]]
    c = ov.confidence_summary(rows, now=now)
    assert c["delta_pts"] == pytest.approx((0.94 - 0.87) * 100) and c["avg"] == pytest.approx(0.905)


# ------------------------------------------------------------------ airings + certificates

def test_airings_with_hashes_and_integrity(client):
    a = client.get("/api/airings", params={"limit": 50}).json()["airings"]
    assert len(a) == 6
    scanned = [x for x in a if x["source"] == "Echorouk TV · rec.mp4"]
    assert len(scanned) == 1 and len(scanned[0]["verification_hash"]) == 64 and scanned[0]["integrity"] is True
    seeded_rows = [x for x in a if x["live"]]
    assert all(x["integrity"] is None for x in seeded_rows)                  # inserted without hashes: "no hash"
    # live rows from before Algeria timestamps were stored get one derived from their epoch start
    assert all(datetime.fromisoformat(x["algeria_timestamp"]).timestamp() == pytest.approx(x["airing_start"], abs=0.001)
               and x["algeria_timestamp"].endswith("+01:00") for x in seeded_rows)
    assert scanned[0]["algeria_timestamp"] is None                           # file offsets have no clock time
    assert client.get("/api/airings", params={"limit": 500}).status_code == 422   # bounded


def test_certificate_pdf(client):
    occ = next(x for x in client.get("/api/airings").json()["airings"] if x["verification_hash"])
    r = client.get(f"/api/certificate/{occ['occurrence_id']}")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF-") and "attachment" in r.headers["content-disposition"]
    assert client.get("/api/certificate/999999").status_code == 404


# ------------------------------------------------------------------ live: telemetry + toggle

def test_toggle_start_telemetry_stop(client):
    chans = client.get("/api/channels").json()
    assert [c["name"] for c in chans["channels"][:4]] == ["Echorouk TV", "Ennahar TV", "El Bilad TV", "El-Heddaf TV"]
    assert any(c["name"] == "TV2 (Algeria)" for c in chans["channels"])
    r = client.post("/api/stream/toggle", json={"channel": "channel:el_heddaf_tv", "action": "start"},
                    headers={"origin": "http://testserver"})
    assert r.status_code == 200 and r.json() == {"channel": "El-Heddaf TV", "action": "start", "running": True}
    t = client.get("/api/telemetry").json()
    (ch,) = t["channels"]
    assert ch["name"] == "El-Heddaf TV" and ch["level_dbfs"] == -21.5 and ch["source"] == "app"
    assert len(ch["level_history"]) == 60 and ch["preview"] is True
    assert client.post("/api/stream/toggle", json={"channel": "El-Heddaf TV", "action": "start"}).status_code == 409
    r = client.post("/api/stream/toggle", json={"channel": "El-Heddaf TV"})            # toggle -> stop
    assert r.status_code == 200 and r.json()["running"] is False
    assert client.get("/api/telemetry").json()["channels"] == []
    assert client.post("/api/stream/toggle", json={"channel": "El-Heddaf TV", "action": "stop"}).status_code == 409
    assert client.post("/api/stream/toggle", json={"channel": "No Such TV"}).status_code == 404


def test_toggle_refused_while_engine_runs(client, seeded, tmp_path):
    class Idle(FakeWorker):
        def __init__(self, spec):
            super().__init__(spec.name, spec.source, {})
    sup = Supervisor([ChannelSpec("Engine TV", "x")], seeded, worker_factory=Idle)
    sup.start()
    sup.tick()
    try:
        t = client.get("/api/telemetry").json()
        assert t["engine"]["active"] is True and [c["name"] for c in t["channels"]] == ["Engine TV"]
        assert t["channels"][0]["source"] == "engine"
        r = client.post("/api/stream/toggle", json={"channel": "El-Heddaf TV", "action": "start"})
        assert r.status_code == 409 and "24/7 background engine" in r.json()["detail"]
        assert client.get("/api/channels").json()["engine_active"] is True
    finally:
        sup.shutdown()


def test_cross_site_post_is_refused(client):
    r = client.post("/api/stream/toggle", json={"channel": "El-Heddaf TV"}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/stream/toggle", content='{"channel": "El-Heddaf TV"}', headers={"content-type": "text/plain"})
    assert r.status_code == 415                                              # "simple" CORS request shapes blocked
    assert client.get("/api/telemetry").json()["channels"] == []            # nothing was started


def test_preview_document_is_escaped_player(client):
    client.post("/api/stream/toggle", json={"channel": "TV2 (Algeria)", "action": "start"})
    r = client.get("/api/preview", params={"channel": "TV2 (Algeria)"})
    assert r.status_code == 200 and "<video" in r.text and "http://185.9.2.18/chid_347/index.m3u8" in r.text
    r = client.get("/api/preview", params={"channel": "<script>alert(1)</script>"})
    assert r.status_code == 200 and "<script>alert(1)</script>" not in r.text and "No browser-playable" in r.text


# ------------------------------------------------------------------ real server boot

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_boots_with_uvicorn(seeded):
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "spotly.app.api_server", "--port", str(port), "--db", str(seeded)],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        ok = False
        for _ in range(60):
            try:
                ok = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=1).json()["status"] == "ok"
                break
            except requests.RequestException:
                time.sleep(0.25)
        assert ok, "server did not come up"
        stats = requests.get(f"http://127.0.0.1:{port}/api/stats", timeout=10).json()
        assert stats["total_airings"] == 6
        assert requests.get(f"http://127.0.0.1:{port}/", timeout=5).text.startswith("<!doctype html>")
    finally:
        proc.terminate()
        proc.wait(10)
