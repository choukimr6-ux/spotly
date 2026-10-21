"""Dashboard hot-reload: library modules must be watched by Streamlit (regression for stale ImportError)."""
import os
from pathlib import Path

from streamlit import file_util

from kihc.app.devreload import ensure_hot_reload

ROOT = Path(__file__).resolve().parents[2]
LIB_MODULE = str(ROOT / "kihc" / "connectors" / "iptv_manager.py")


def test_library_modules_become_watchable(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert not file_util.file_in_pythonpath(LIB_MODULE)          # the bug: not watched -> never reloaded
    assert ensure_hot_reload(str(ROOT)) is True
    assert file_util.file_in_pythonpath(LIB_MODULE)              # Streamlit now evicts it on change
    assert file_util.file_in_pythonpath(str(ROOT / "kihc" / "core" / "matcher.py"))


def test_idempotent_and_preserves_existing_entries(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert ensure_hot_reload(str(ROOT)) is True
    assert ensure_hot_reload(str(ROOT)) is False                 # reruns don't grow PYTHONPATH
    parts = os.environ["PYTHONPATH"].split(os.pathsep)
    assert parts == [str(ROOT), str(tmp_path)]


def test_evict_stale_modules(tmp_path, monkeypatch):
    import importlib
    import sys
    import time
    from kihc.app import devreload
    pkg = tmp_path / "kihc"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod_file = pkg / "probe_mod.py"
    mod_file.write_text("VALUE = 1\n")
    for name in [n for n in sys.modules if n == "kihc" or n.startswith("kihc.")]:
        monkeypatch.delitem(sys.modules, name)                      # isolate: use the temp 'kihc' package
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(devreload, "_SEEN_MTIME", {})
    importlib.invalidate_caches()
    import kihc.probe_mod as probe
    assert probe.VALUE == 1
    assert devreload.evict_stale_modules(str(tmp_path)) == []       # first sighting: baseline only
    mod_file.write_text("VALUE = 2\n")
    future = time.time() + 5
    os.utime(mod_file, (future, future))
    assert devreload.evict_stale_modules(str(tmp_path)) == ["kihc.probe_mod"]
    importlib.invalidate_caches()
    assert importlib.import_module("kihc.probe_mod").VALUE == 2     # fresh code after eviction
    assert devreload.evict_stale_modules(str(tmp_path)) == []       # re-baselined, no churn


def test_dashboard_registers_root_before_kihc_imports():
    src = (ROOT / "kihc" / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert src.index("ensure_hot_reload(_ROOT)") < src.index("from kihc.cli import")
    assert src.index("evict_stale_modules(_ROOT)") < src.index("from kihc.cli import")
