"""The dashboard script must parse. A single syntax error (2026-10-14: a duplicate `const dzd`) stops the whole
SPA, leaving 'Checking engine...' on screen while the engine runs fine; string-presence tests can't see that."""
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "app" / "static" / "app.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed (only used to syntax-check app.js)")
def test_dashboard_javascript_parses():
    p = subprocess.run(["node", "--check", str(APP_JS)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
