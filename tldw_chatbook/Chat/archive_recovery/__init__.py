"""Local archive discovery and explicit recovery (ADR-166)."""

import sys
from types import FunctionType, ModuleType

_archive_root = sys.modules["tldw_chatbook"]
_archive_parent = sys.modules["tldw_chatbook.Chat"]
_package = sys.modules[__name__]
if (
    type(_package) is not ModuleType
    or vars(_package) is not globals()
    or type(_archive_root) is not ModuleType
    or type(_archive_parent) is not ModuleType
    or vars(_archive_root).get("Chat") is not _archive_parent
    or type(_lookup := vars(_archive_parent).get("__getattr__")) is not FunctionType
    or _lookup.__globals__ is not vars(_archive_parent)
):
    raise RuntimeError("ordinary archive package changed")
if _archive_parent._ordinary_archive_package is None:
    # Each original child publishes independently on first import.
    _ordinary_issuer = None
    _ordinary_declarations = None
    _archive_parent._ordinary_archive_package = _package
elif _archive_parent._ordinary_archive_package is not _package:
    raise RuntimeError("ordinary archive package changed")
