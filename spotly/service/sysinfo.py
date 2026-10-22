"""Process memory readings (current + peak working set) without third-party deps."""
import os
import sys


def memory_mb(pid: int | None = None) -> tuple[float, float]:
    """(current, peak) resident memory in MiB for pid (default: this process); (0, 0) if unknown."""
    if sys.platform == "win32":
        return _win_memory(pid)
    return _posix_memory(pid)


def _win_memory(pid):
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                    ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    own = pid is None or pid == os.getpid()
    h = k32.GetCurrentProcess() if own else k32.OpenProcess(0x1000 | 0x0010, False, pid)
    if not h:
        return 0.0, 0.0
    try:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not k32.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return 0.0, 0.0
        return pmc.WorkingSetSize / 2 ** 20, pmc.PeakWorkingSetSize / 2 ** 20
    finally:
        if not own:
            k32.CloseHandle(h)


def _posix_memory(pid):
    pid = pid or os.getpid()
    try:
        with open(f"/proc/{pid}/status") as f:
            fields = dict(line.split(":", 1) for line in f if ":" in line)
        kb = lambda k: float(fields.get(k, "0 kB").split()[0])
        return kb("VmRSS") / 1024, kb("VmHWM") / 1024
    except OSError:
        if pid == os.getpid():
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_mb = peak / 2 ** 20 if sys.platform == "darwin" else peak / 1024
            return peak_mb, peak_mb
        return 0.0, 0.0
