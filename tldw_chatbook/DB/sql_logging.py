# sql_logging.py
# Description: Shared helpers for cheap, BLOB-safe SQL debug logging.
"""
SQL debug-logging helpers
--------------------------

Query parameters can carry multi-megabyte values -- raw image BLOBs
(``messages.image_data``), full ingested document text, etc. Building a
``str(params)`` preview for *every* query -- even when nothing will ever be
emitted -- is measurably expensive (up to ~14ms per query for a 3MB BLOB
param; see Docs/Design/2026-07-16-performance-audit.md, finding A1).

Two independent guarantees are needed to make debug logging free at runtime:

1. **Laziness**: nothing here should be built unless the active logging sink
   actually admits DEBUG records. ``loguru`` has no ``isEnabledFor()`` (unlike
   stdlib ``logging``), so callers must use ``logger.opt(lazy=True).debug(...)``
   with callables -- the callables (and therefore this module's helpers) are
   only invoked if some configured sink's level is <= DEBUG.
2. **Boundedness**: even when DEBUG *is* enabled, a preview must never fully
   stringify a large value. Bytes-like values are summarized by length only;
   long strings are truncated; the whole rendered preview is capped too, so a
   pathological params collection (huge tuple/dict) can't blow up log volume.

Usage at a call site::

    from .sql_logging import preview_params

    logger.opt(lazy=True).debug(
        "Executing SQL (script={}): {}",
        lambda: script,
        lambda: f"{query[:300]}... Params: {preview_params(params)}",
    )

Do **not** restore an ``if logger.isEnabledFor(logging.DEBUG):`` guard around
a plain ``logger.debug(f"...")`` call for loguru loggers -- ``logger`` objects
from ``loguru`` do not have ``isEnabledFor`` and this raises ``AttributeError``.
"""

from typing import Any, Dict, Optional, Tuple, Union

# Individual param values longer than this are truncated.
_MAX_ITEM_CHARS = 80
# The fully-rendered preview string is capped to this length regardless of
# how many/how large the individual params are.
_MAX_PREVIEW_CHARS = 200


def _preview_value(value: Any) -> str:
    """Render a single SQL param value as a short, cheap-to-build string.

    Bytes-like values (``bytes``/``bytearray``/``memoryview``) are *never*
    stringified in full -- only their length is reported, regardless of size.
    Long strings are truncated. Everything else falls back to ``repr()``,
    which is fine for the small scalar types (int/float/bool/None/etc.) SQL
    params are normally built from.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        if len(value) > _MAX_ITEM_CHARS:
            return f"{value[:_MAX_ITEM_CHARS]}...<{len(value)} chars>"
        return value
    return repr(value)


def preview_params(params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]]) -> str:
    """Build a cheap, BLOB-safe preview string for SQL params, for debug logs.

    Args:
        params: The tuple or dict of query parameters (or ``None``).

    Returns:
        A short human-readable preview. Bytes-like values are summarized as
        ``<N bytes>`` rather than stringified; the whole preview is capped to
        ``_MAX_PREVIEW_CHARS``. Never raises -- a param collection that can't
        be rendered for any reason yields a fixed placeholder string instead
        of propagating an exception into the logging path.
    """
    if params is None:
        return "None"
    try:
        if isinstance(params, dict):
            items = ", ".join(f"{k}={_preview_value(v)}" for k, v in params.items())
            text = f"{{{items}}}"
        else:
            items = ", ".join(_preview_value(v) for v in params)
            text = f"({items})"
    except Exception:
        # Debug-log preview building must never break the query path.
        return "<unrepr-able params>"
    if len(text) > _MAX_PREVIEW_CHARS:
        return text[:_MAX_PREVIEW_CHARS] + "..."
    return text
