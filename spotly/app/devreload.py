"""Make Streamlit hot-reload Spotly's library modules, not just the dashboard script.

Streamlit only evicts changed modules that live under the main script's folder (spotly/app/) or
under a directory listed in the PYTHONPATH *environment variable*; entries added to sys.path
are ignored. Without this, a long-running `streamlit run` keeps old copies of spotly/core,
spotly/connectors, ... after they are edited, and a dashboard that imports a newly added name
fails with ImportError until the server is restarted.
"""
import os
import sys

# module name -> source mtime when this process first saw it loaded
_SEEN_MTIME: dict[str, float] = {}


def evict_stale_modules(root: str, package: str = "spotly") -> list[str]:
    """Drop already-imported `package` modules whose source changed since they were loaded.

    Streamlit queues evictions on the watcher of the browser session that saw the change; a
    session opened *after* an edit starts with nothing queued while sys.modules (process-wide)
    still holds the old code. Calling this at the top of every dashboard run closes that gap.
    The first sighting of a module records its current mtime as the baseline.
    """
    pkg_dir = os.path.normcase(os.path.join(os.path.abspath(root), package))
    evicted = []
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if not path or name == __name__ or not (name == package or name.startswith(package + ".")):
            continue
        if not os.path.normcase(os.path.abspath(path)).startswith(pkg_dir):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        seen = _SEEN_MTIME.setdefault(name, mtime)
        if mtime > seen:
            sys.modules.pop(name, None)
            _SEEN_MTIME.pop(name, None)          # re-baselined when the fresh import is next seen
            evicted.append(name)
    return evicted


def ensure_hot_reload(root: str) -> bool:
    """Add `root` to PYTHONPATH (front, existing entries kept). Returns True if it was added."""
    root = os.path.abspath(root)
    current = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if any(os.path.normcase(os.path.abspath(p)) == os.path.normcase(root) for p in current):
        return False
    os.environ["PYTHONPATH"] = os.pathsep.join([root, *current])
    return True
