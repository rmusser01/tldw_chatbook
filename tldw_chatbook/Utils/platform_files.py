"""Explicitly imported filesystem/lock interface; never alters stdlib modules."""

import os as _os

__all__ = ["fcntl", "os"]

if _os.name == "nt":
    from .windows_files import WindowsLocks, WindowsOS

    os = WindowsOS()
    fcntl = WindowsLocks()
else:
    import fcntl

    os = _os


def supports_native_process_groups() -> bool:
    """Check the platform primitives needed for native process-group custody."""
    return _os.name == "posix" and hasattr(_os, "killpg")
