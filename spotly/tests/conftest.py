"""Test-session isolation: every writable Spotly location points at a throwaway directory.

Set before any spotly module is imported (pytest imports this file first), and inherited by the
subprocesses the service/pilot tests launch, so the suite never writes into production
spotly/data/spotly.db, spotly/data/evidence, spotly/data/discovered_ads, spotly/data/logos or spotly/reports.
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="spotly_test_sandbox_"))
for _var, _sub in (("SPOTLY_DB_PATH", "spotly.db"), ("SPOTLY_EVIDENCE_DIR", "evidence"),
                   ("SPOTLY_DISCOVERED_DIR", "discovered_ads"), ("SPOTLY_REPORTS_DIR", "reports"),
                   ("SPOTLY_LOGO_DIR", "logos")):
    os.environ[_var] = str(_SANDBOX / _sub)
atexit.register(shutil.rmtree, _SANDBOX, ignore_errors=True)
