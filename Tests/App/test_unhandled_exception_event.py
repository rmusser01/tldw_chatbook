"""TASK-1240: a crash names its exception type in the persistent log."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_unhandled_exception_is_recorded(monkeypatch):
    from Tests.UI.test_screen_navigation import _build_test_app

    recorded: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append(
            {"component": component, "event": event, **fields}
        ),
    )

    app = _build_test_app()
    try:
        app._handle_exception(RuntimeError("secret detail"))
    except Exception:
        # Textual's implementation re-raises; that behaviour must be preserved.
        pass

    crashes = [r for r in recorded if r["event"] == "unhandled_exception"]
    assert crashes, f"no unhandled_exception recorded, got {recorded}"
    assert crashes[-1]["exception_type"] == "RuntimeError"
    assert "secret detail" not in str(crashes[-1])


def test_the_override_still_delegates_to_textual():
    """Must not swallow: Textual sets the return code and re-raises for tests."""
    from Tests.UI.test_screen_navigation import _build_test_app

    app = _build_test_app()
    try:
        app._handle_exception(RuntimeError("boom"))
    except Exception:
        pass
    assert app._return_code == 1
