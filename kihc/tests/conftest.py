"""Test-session isolation: every writable KIHC location points at a throwaway directory.

Set before any kihc module is imported (pytest imports this file first), and inherited by the
subprocesses the service/pilot tests launch, so the suite never writes into production
kihc/data/kihc.db, kihc/data/evidence, kihc/data/discovered_ads, kihc/data/logos or kihc/reports.
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="kihc_test_sandbox_"))
for _var, _sub in (("KIHC_DB_PATH", "kihc.db"), ("KIHC_EVIDENCE_DIR", "evidence"),
                   ("KIHC_DISCOVERED_DIR", "discovered_ads"), ("KIHC_REPORTS_DIR", "reports"),
                   ("KIHC_LOGO_DIR", "logos")):
    os.environ[_var] = str(_SANDBOX / _sub)
atexit.register(shutil.rmtree, _SANDBOX, ignore_errors=True)
