"""Phase 31.1: the dashboard works air-gapped - no CDN URL anywhere in the page, every vendored asset served locally."""
import re

import pytest
from fastapi.testclient import TestClient

from kihc.app.api_server import STATIC_DIR, create_app

VENDOR = ("tailwind.play.js", "chart.umd.min.js", "lucide.min.js")
FORBIDDEN = ("cdn.jsdelivr.net", "cdn.tailwindcss.com", "unpkg.com", "cdnjs.cloudflare.com", "fonts.googleapis.com")


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "w.db")) as c:
        yield c


def test_index_has_no_external_urls(client):
    html = client.get("/").text
    assert not [h for h in FORBIDDEN if h in html]
    # no remote <script src> / <link href>; the only http(s) text allowed is the SVG xmlns inside the data: favicon
    remote = re.findall(r'<(?:script|link|img|iframe)[^>]+(?:src|href)=["\'](https?:)?//[^"\']+', html)
    assert not remote
    for lib in VENDOR:
        assert f"/static/vendor/{lib}" in html or lib == "tailwind.play.js" and "/static/vendor/tailwind.play.js" in html


def test_app_js_has_no_external_urls():
    assert not [h for h in FORBIDDEN if h in (STATIC_DIR / "app.js").read_text(encoding="utf-8")]


@pytest.mark.parametrize("name", VENDOR)
def test_vendor_assets_are_served_locally(client, name):
    r = client.get(f"/static/vendor/{name}")
    assert r.status_code == 200 and len(r.content) > 100_000
    assert "javascript" in r.headers["content-type"]


def test_lucide_and_chart_boot_hooks_stay_wired(client):
    html = client.get("/").text
    assert html.count("kihcLibsReady") >= 2                      # chart + lucide onload hooks
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "lucide.createIcons" in js
