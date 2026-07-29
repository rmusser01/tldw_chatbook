"""TASK-1240: the app records that it started and that it stopped cleanly."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_mounting_the_app_records_app_started(monkeypatch):
    """Boot the real app and assert the event fired.

    Asserted by capturing the call rather than scanning source: this repo has
    been burned by name-matching guards that pass against unwired code.
    """
    from Tests.UI.test_screen_navigation import _build_test_app

    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append((component, event)),
    )

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

    assert ("app", "app_started") in recorded
    assert ("app", "app_stopping") in recorded
