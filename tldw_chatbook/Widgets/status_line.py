"""One status-line helper for the ``_set_status`` family (TASK-32861).

Twenty-plus widgets hand-rolled the same "update a ``Static`` status line by
selector, tolerating the line not existing yet / already torn down" body,
with drift in which missing-widget errors each suppresses (``NoMatches``,
``QueryError``, bare ``Exception`` -- all the same event). This helper is
the single owner; adopters keep their ``_set_status`` signatures and any
genuinely extra behavior (error styling, announcements, notice panels) and
compose it around this call.

Semantics:
- ``missing_ok=True`` (the majority contract): a missing/stale status
  widget is silently ignored -- the surface will be recomposed anyway.
- ``missing_ok=False``: the lookup error propagates (for sites that treat
  a missing line as a bug).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.widgets import Static

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from textual.widget import Widget


def set_status_line(
    widget: Widget,
    selector: str | Any,
    text: str,
    *,
    missing_ok: bool = True,
) -> bool:
    """Update a ``Static`` status line identified by ``selector``.

    Args:
        widget: The owning widget (``self`` at the call site).
        selector: The ``query_one`` selector for the status ``Static``
            (a ``#dom-id`` string or any Textual selector).
        text: The status copy to display.
        missing_ok: When True (default), suppress lookup errors for a
            missing/stale line and return False instead.

    Returns:
        True when the line was found and updated.

    Raises:
        Exception: Whatever ``query_one`` raises, when ``missing_ok`` is
            False (``NoMatches`` / ``QueryError`` depending on the query).
    """
    if missing_ok:
        try:
            widget.query_one(selector, Static).update(text)
        except Exception:  # noqa: BLE001 - missing/stale status line; recompose follows
            return False
        return True
    widget.query_one(selector, Static).update(text)
    return True
