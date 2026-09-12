from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.watchlists_operation_card import (
    WatchlistsOperationCard,
)


class CardHarness(ConsolidatedCSSApp):
    CSS_PATH = [str(BUNDLED_STYLESHEET)]

    def compose(self) -> ComposeResult:
        yield WatchlistsOperationCard("local:watchlist_run:7")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "indicator"),
    [
        ("queued", "QUEUED"),
        ("running", "RUNNING"),
        ("completed", "COMPLETE"),
        ("empty", "EMPTY"),
        ("failed", "FAILED"),
        ("cancelled", "CANCELLED"),
    ],
)
async def test_card_renders_non_color_status_and_exact_supported_actions(
    status, indicator
):
    app = CardHarness()
    async with app.run_test(size=(100, 20)) as pilot:
        card = app.query_one(WatchlistsOperationCard)
        card.set_operation(
            {
                "id": "local:watchlist_run:7",
                "status_detail": status,
                "source": {"name": "Threat feed"},
                "destination": "runs",
                "retry_capable": status == "failed",
                "cancel_capable": status in {"queued", "running"},
                "error_category": "x" * 500 if status == "failed" else None,
            }
        )
        await pilot.pause()

        text = " ".join(str(item.renderable) for item in card.query(Static))
        labels = {str(button.label) for button in card.query(Button) if button.display}
        assert indicator in text
        assert "Runs" in labels and "Stop following" in labels
        assert ("Retry" in labels) is (status == "failed")
        assert ("Cancel" in labels) is (status in {"queued", "running"})
        assert len(card.query_one(".watchlists-operation-error", Static).renderable) <= 164


def test_task_state_persists_only_valid_canonical_receipt_identity():
    restored = TaskResumeState.from_dict(
        {
            "followed_watchlists_operations": [
                "local:briefing:9",
                "local:watchlist_run:7",
                "https://secret.example/?token=bad",
            ],
            "tool_arguments": {"token": "must-not-survive"},
            "body": "must-not-survive",
        }
    )

    assert restored.followed_watchlists_operations == (
        "local:briefing:9",
        "local:watchlist_run:7",
    )
    assert set(restored.to_dict()) == {
        "summary",
        "last_step",
        "pending_approval",
        "pending_skill_install",
        "pending_skill_script",
        "diff_summary",
        "next_action",
        "followed_watchlists_operations",
    }
    assert "secret" not in str(restored.to_dict())


@pytest.mark.asyncio
async def test_card_uses_compact_production_geometry_from_consolidated_css():
    app = CardHarness()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        card = app.query_one(WatchlistsOperationCard)
        actions = card.query_one(".watchlists-operation-actions")

        assert card.region.height < 24
        assert actions.region.height <= 3
        assert card.styles.border.top[0] != ""
        await pilot.press("tab")
        assert isinstance(app.focused, Button)
