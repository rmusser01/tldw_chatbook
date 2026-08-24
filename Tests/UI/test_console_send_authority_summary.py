from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleInspectorState,
)
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    ConsoleSendAuthoritySummary,
    project_console_send_authority,
)


def _state(**overrides) -> ConsoleInspectorState:
    values = {
        "rows": (
            ConsoleDisplayRow("Workspace", "Research Lab"),
            ConsoleDisplayRow("Selected conversation", "Research conversation 3"),
            ConsoleDisplayRow("Provider", "ready", status="ready"),
            ConsoleDisplayRow("Sources", "ready", status="ready"),
        ),
    }
    values.update(overrides)
    return ConsoleInspectorState(**values)


def test_projection_is_one_frozen_five_fact_value_for_saved_conversation() -> None:
    projection = project_console_send_authority(_state())

    assert projection.where == "Research Lab › Research conversation 3"
    assert projection.scope == "Everything available"
    assert projection.run == "Ready"
    assert projection.sources == "None staged"
    assert projection.approvals == "None pending"
    assert tuple(projection.__dataclass_fields__) == (
        "where",
        "scope",
        "run",
        "sources",
        "approvals",
    )
    with pytest.raises(FrozenInstanceError):
        projection.run = "Changed"  # type: ignore[misc]


def test_projection_uses_explicit_default_and_temporary_fallbacks() -> None:
    state = _state(
        rows=(
            ConsoleDisplayRow("Selected conversation", "No active conversation"),
            ConsoleDisplayRow("Provider", "ready", status="ready"),
        ),
        ephemeral=True,
    )

    assert project_console_send_authority(state).where == (
        "Default › Temporary conversation · Temporary"
    )


def test_projection_marks_named_ephemeral_conversation_temporary() -> None:
    assert project_console_send_authority(_state(ephemeral=True)).where.endswith(
        "Research conversation 3 · Temporary"
    )


@pytest.mark.parametrize(
    ("prefill_rows", "scope_item_count", "expected"),
    [
        (
            (
                ConsoleDisplayRow("Prefill (pinned)", "Pinned"),
                ConsoleDisplayRow("Prefill (next send only)", "One shot"),
            ),
            4,
            "One-shot prefill · narrowed to 4 items",
        ),
        (
            (ConsoleDisplayRow("Prefill (pinned)", "Pinned"),),
            2,
            "Pinned prefill · narrowed to 2 items",
        ),
        ((), 3, "narrowed to 3 items"),
    ],
)
def test_projection_scope_precedence(prefill_rows, scope_item_count, expected) -> None:
    state = _state(rows=_state().rows + prefill_rows, scope_item_count=scope_item_count)

    assert project_console_send_authority(state).scope == expected


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"run_active": True}, "Running"),
        ({"pending_approval_count": 1, "run_active": True}, "Waiting for approval"),
        (
            {
                "rows": _state().rows
                + (ConsoleDisplayRow("Next action", "Reconnect"),),
                "pending_approval_count": 1,
            },
            "Recovery required",
        ),
        (
            {
                "rows": (
                    ConsoleDisplayRow("Provider", "missing", status="blocked"),
                    ConsoleDisplayRow("Sources", "ready", status="ready"),
                ),
                "run_active": True,
            },
            "Blocked",
        ),
        (
            {
                "rows": (
                    ConsoleDisplayRow("Provider", "ready", status="ready"),
                    ConsoleDisplayRow("RAG/source", "missing", status="blocked"),
                )
            },
            "Blocked",
        ),
    ],
)
def test_projection_run_precedence(changes, expected) -> None:
    assert project_console_send_authority(_state(**changes)).run == expected


def test_projection_uses_typed_nonzero_counts() -> None:
    projection = project_console_send_authority(
        _state(staged_source_count=2, pending_approval_count=1)
    )

    assert projection.sources == "2 staged"
    assert projection.approvals == "1 pending · action required"


def test_projection_never_invents_ready_for_incomplete_ownership() -> None:
    projection = project_console_send_authority(
        _state(rows=(ConsoleDisplayRow("Unknown", "secret"),))
    )

    assert projection.where == "Default › No active conversation"
    assert projection.scope == "Everything available"
    assert projection.run == "Inspector data incomplete"


class _SummaryHarness(App):
    def __init__(self, state: ConsoleInspectorState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield ConsoleSendAuthoritySummary(self.state)


EXPECTED_IDS = (
    "console-send-authority-heading",
    "console-send-authority-where",
    "console-send-authority-scope",
    "console-send-authority-run",
    "console-send-authority-sources",
    "console-send-authority-approvals",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [34, 80])
async def test_summary_is_one_focus_stop_with_exactly_six_fixed_rows(width) -> None:
    app = _SummaryHarness(_state())

    async with app.run_test(size=(width, 12)):
        summary = app.query_one(
            "#console-send-authority-summary", ConsoleSendAuthoritySummary
        )
        assert tuple(child.id for child in summary.children) == EXPECTED_IDS
        assert summary.can_focus
        assert all(not child.can_focus for child in summary.children)
        assert summary.region.height == 6
        for child in summary.children:
            assert child.region.height == 1
            assert str(child.styles.text_wrap) == "nowrap"
            assert str(child.styles.text_overflow) == "ellipsis"


@pytest.mark.asyncio
async def test_summary_sync_atomically_patches_rows_without_recompose() -> None:
    before = _state()
    app = _SummaryHarness(before)

    async with app.run_test(size=(80, 12)) as pilot:
        summary = app.query_one(
            "#console-send-authority-summary", ConsoleSendAuthoritySummary
        )
        identities = {child.id: child for child in summary.children}
        after = _state(
            rows=(
                ConsoleDisplayRow("Workspace", "Studio"),
                ConsoleDisplayRow("Selected conversation", "Draft B"),
                ConsoleDisplayRow("Provider", "ready", status="ready"),
            ),
            staged_source_count=3,
            pending_approval_count=2,
        )

        summary.sync_state(after)
        await pilot.pause()

        assert summary.last_state is after
        assert summary.recompose_count == 0
        assert {child.id: child for child in summary.children} == identities
        rendered = "\n".join(str(child.renderable) for child in summary.children)
        assert "Studio › Draft B" in rendered
        assert "3 staged" in rendered
        assert "2 pending · action required" in rendered
        assert "Research Lab" not in rendered


@pytest.mark.asyncio
async def test_unicode_tooltips_and_context_help_only_expose_complete_values() -> None:
    long_state = _state(
        rows=(
            ConsoleDisplayRow("Workspace", "研究所🧪" * 8),
            ConsoleDisplayRow("Selected conversation", "会話💡" * 8),
            ConsoleDisplayRow("Provider", "ready", status="ready"),
            ConsoleDisplayRow("Prefill (next send only)", "literal [bold] markup"),
        ),
        scope_item_count=1234,
    )
    app = _SummaryHarness(long_state)

    async with app.run_test(size=(34, 12)) as pilot:
        summary = app.query_one(
            "#console-send-authority-summary", ConsoleSendAuthoritySummary
        )
        where = summary.query_one("#console-send-authority-where", Static)
        scope = summary.query_one("#console-send-authority-scope", Static)
        projection = project_console_send_authority(long_state)
        assert str(where.tooltip) == projection.where
        assert str(scope.tooltip) == projection.scope
        assert summary.contextual_help_rows() == (
            ("Where", projection.where),
            ("Scope", projection.scope),
            ("Run", projection.run),
            ("Sources", projection.sources),
            ("Approvals", projection.approvals),
        )
        assert "[bold]" not in str(scope.renderable)

        await pilot.resize_terminal(160, 12)
        await pilot.pause()
        assert where.tooltip is None
        assert scope.tooltip is None

        await pilot.resize_terminal(34, 12)
        summary.sync_state(
            _state(
                rows=(
                    ConsoleDisplayRow("Workspace", "A"),
                    ConsoleDisplayRow("Selected conversation", "B"),
                    ConsoleDisplayRow("Provider", "ready", status="ready"),
                )
            )
        )
        await pilot.pause()
        assert summary.query_one("#console-send-authority-where").tooltip is None
        assert summary.query_one("#console-send-authority-scope").tooltip is None
