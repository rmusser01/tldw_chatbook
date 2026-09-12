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
