"""Hardware and environment configuration for Virtuoso Architect.

Detects the host operating system, CPU architecture, and Python version.
Designed for macOS (Intel / x86_64), CPU-first operation.
No heavy imports; no GPU assumptions.
"""

import platform
import sys


def setup_hardware() -> dict:
    """Detect and print the current execution environment.

    Returns:
        dict with keys ``os``, ``arch``, ``python_version``.
    """
    os_name: str = sys.platform  # "darwin" on macOS
    arch: str = platform.machine()  # "x86_64" on Intel Mac
    py_version: str = platform.python_version()

    info = {
        "os": os_name,
        "arch": arch,
        "python_version": py_version,
    }

    print("──── Virtuoso Architect — Environment ────")
    print(f"  OS            : {os_name}")
    print(f"  Architecture  : {arch}")
    print(f"  Python        : {py_version}")

    if os_name != "darwin":
        print("  ⚠  Expected macOS (darwin), got:", os_name)
    if arch != "x86_64":
        print("  ⚠  Expected Intel x86_64, got:", arch)

    print("  Device        : CPU (default)")
    print("──────────────────────────────────────────")
    return info
