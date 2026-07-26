"""Non-resumable conversation rows must be honest (TASK-717).

Live UAT (workspace-settings review): a membership row whose conversation
record does not exist rendered exactly like an openable row, and a failed
open produced misleading feedback - two stacked toasts, one promising a
Library affordance that does not exist for a missing record - while the row
stayed clickable-identical forever.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app


def _ghost_row(console) -> Button:
    rows = [
        button
        for button in console.query(".console-workspace-conversation-row")
        if "Ghost" in str(button.label)
    ]
    assert rows, "expected the ghost membership row to render"
    return rows[0]


@pytest.mark.asyncio
async def test_failed_resume_marks_row_broken_with_honest_single_toast() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    active = service.get_active_workspace()
    service.link_membership(
        active.workspace_id,
        item_type="conversation",
        item_id="conv-ghost",
        title="Ghost chat",
        role="source",
    )
    # A scope service that answers (no transient failure) but has no record
    # for the ghost conversation - the permanent missing-record class.
    from types import SimpleNamespace

    app.chat_conversation_scope_service = SimpleNamespace(
        get_conversation_tree=lambda *args, **kwargs: {}
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.3)
        console = host.screen_stack[-1]
        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        _ghost_row(console).press()
        await pilot.pause(0.6)

        # Exactly one toast, and it tells the truth: the record is missing.
        # No false "open it from Library" promise - Library has no affordance
        # for a conversation record that does not exist.
        assert len(notifications) == 1, notifications
        assert "could not be loaded" in notifications[0]
        assert "Library" not in notifications[0]

        # The row is now visibly broken and no longer pretends to be openable.
        row = _ghost_row(console)
        assert row.disabled is True
        assert row.has_class("console-workspace-conversation-row-broken")
