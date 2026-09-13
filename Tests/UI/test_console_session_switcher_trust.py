"""Trust and terminal-fit regressions for the Console Ctrl+K switcher."""

from __future__ import annotations

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
    ConsoleSwitcherChoice,
    SEARCH_DEBOUNCE_SECONDS,
)


def _row(
    key: str,
    title: str,
    *,
    native_session_id: str | None = None,
    selected: bool = False,
    run_marker: str = "",
    queued_count: int = 0,
    openable: bool = True,
) -> ConsoleConversationBrowserInputRow:
    return ConsoleConversationBrowserInputRow(
        row_key=key,
        conversation_id=None if native_session_id else key,
        native_session_id=native_session_id,
        title=title,
        scope_type="workspace",
        workspace_id="ws-1",
        workspace_label="Research Lab",
        status="active session" if selected else "workspace-thread",
        selected=selected,
        source_kind="native" if native_session_id else "persisted",
        updated_sort=f"2026-07-{20 - int(key.rsplit('-', 1)[-1]) if key.rsplit('-', 1)[-1].isdigit() else 1:02d}T10:00:00+00:00",
        run_marker=run_marker,
        queued_count=queued_count,
        openable=openable,
    )


class _SwitcherApp(ConsolidatedCSSApp):
    def __init__(
        self,
        rows: tuple[ConsoleConversationBrowserInputRow, ...],
        *,
        preferred_native_session_id: str | None = None,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.preferred_native_session_id = preferred_native_session_id
        self.result: ConsoleSwitcherChoice | None | str = "unset"

    async def on_mount(self) -> None:
        kwargs = {}
        if self.preferred_native_session_id is not None:
            kwargs["preferred_native_session_id"] = self.preferred_native_session_id
        try:
            modal = ConsoleSessionSwitcherModal(
                rows=self.rows,
                **kwargs,
            )
        except TypeError:
            modal = None
        assert modal is not None, "the switcher must accept an explicit MRU tab target"
        await self.push_screen(modal, callback=self._capture)

    def _capture(self, choice: ConsoleSwitcherChoice | None) -> None:
        self.result = choice


@pytest.mark.asyncio
async def test_immediate_enter_uses_the_exact_unsettled_query():
    """Reintroducing stale `_entries[0]` must activate the wrong native tab."""
    app = _SwitcherApp(
        (
            _row("row-1", "Release planning", native_session_id="session-a"),
            _row("row-2", "Saved migration notes"),
        )
    )

    async with app.run_test(size=(52, 20)) as pilot:
        query = app.screen.query_one("#console-switcher-query", Input)
        query.value = "migration"
        await pilot.press("enter")
        await pilot.pause()

    assert isinstance(app.result, ConsoleSwitcherChoice)
    assert app.result.entry.title == "Saved migration notes"


@pytest.mark.asyncio
async def test_f2_refuses_saved_candidate_without_renaming_another_tab():
    """Restoring the native fallback must dismiss with the wrong rename target."""
    app = _SwitcherApp(
        (
            _row("row-1", "Release planning", native_session_id="session-a"),
            _row("row-2", "Saved migration notes"),
        )
    )

    async with app.run_test(size=(90, 30)) as pilot:
        modal = app.screen
        modal.query_one("#console-switcher-result-1", Button).focus()
        await pilot.press("f2")
        await pilot.pause()

        assert app.result == "unset"
        feedback = modal.query_one("#console-switcher-feedback", Static)
        assert "saved chats cannot be renamed here" in str(feedback.renderable).lower()


@pytest.mark.asyncio
async def test_blank_enter_activates_the_mru_other_tab():
    app = _SwitcherApp(
        (
            _row(
                "row-1",
                "Current agent",
                native_session_id="session-current",
                selected=True,
            ),
            _row("row-2", "Older agent", native_session_id="session-older"),
            _row("row-3", "MRU agent", native_session_id="session-mru"),
        ),
        preferred_native_session_id="session-mru",
    )

    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.press("enter")
        await pilot.pause()

    assert isinstance(app.result, ConsoleSwitcherChoice)
    assert app.result.entry.native_session_id == "session-mru"


@pytest.mark.asyncio
async def test_small_terminal_keeps_chrome_visible_and_scrolls_candidate():
    rows = tuple(
        _row(
            f"row-{index}",
            f"Agent conversation {index}",
            native_session_id=f"session-{index}",
        )
        for index in range(1, 21)
    )
    app = _SwitcherApp(rows)

    async with app.run_test(size=(60, 18)) as pilot:
        modal = app.screen.query_one("#console-switcher-modal")
        results = app.screen.query_one("#console-switcher-results", VerticalScroll)
        cancel = app.screen.query_one("#console-switcher-cancel", Button)

        assert modal.region.x >= 0
        assert modal.region.right <= app.size.width
        assert modal.region.y >= 0
        assert modal.region.bottom <= app.size.height
        assert cancel.region.bottom <= app.size.height

        trace = []
        for _ in range(15):
            await pilot.press("down")
            trace.append(
                (
                    getattr(app.focused, "id", None),
                    app.screen._candidate_index,
                    results.scroll_y,
                )
            )
        await pilot.pause()

        focused = app.focused
        assert isinstance(focused, Button)
        assert focused.region.y >= results.content_region.y, trace
        assert focused.region.bottom <= results.content_region.bottom
        assert results.scroll_y > 0


@pytest.mark.asyncio
async def test_groups_and_text_labels_expose_operational_state():
    app = _SwitcherApp(
        (
            _row(
                "row-1",
                "Current agent",
                native_session_id="session-current",
                selected=True,
            ),
            _row(
                "row-2",
                "Release agent",
                native_session_id="session-running",
                run_marker="●",
                queued_count=2,
            ),
            _row("row-3", "Old migration notes", openable=False),
        )
    )

    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        headings = {
            str(widget.renderable)
            for widget in app.screen.query(".console-switcher-section")
        }
        labels = [str(button.label) for button in app.screen.query(Button)]

        assert headings == {"OPEN AGENT TABS", "SAVED CHATS"}
        assert any("CURRENT" in label for label in labels)
        assert any("RUNNING" in label for label in labels)
        assert any("2 QUEUED" in label for label in labels)
        assert any("UNAVAILABLE" in label for label in labels)
        assert any("▸" in label for label in labels)

        query = app.screen.query_one("#console-switcher-query", Input)
        query.value = "is:saved"
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        assert "Old migration notes" in str(
            app.screen.query_one("#console-switcher-result-0", Button).label
        )


@pytest.mark.asyncio
async def test_empty_switcher_teaches_first_agent_tab_and_search_recovery():
    app = _SwitcherApp(())

    async with app.run_test(size=(70, 22)) as pilot:
        await pilot.pause()
        empty = app.screen.query_one("#console-switcher-empty", Static)
        copy = str(empty.renderable)
        assert "Ctrl+T" in copy
        assert "agent tab" in copy
