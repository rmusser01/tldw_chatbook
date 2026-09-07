"""TASK-1900: the browser search input's mount echo must not exist.

`ConsoleWorkspaceContextTray.sync_state` re-mounts a fresh search `Input` on
every sync that changes anything -- which, while the user is typing, is every
one of them. (It used to do so on EVERY sync; TASK-15454 narrowed it to
changed states, which removes some echoes but not the ones this test is
about.) Textual's `Input._watch_value` posts
`Changed` for a constructor-set value unconditionally (the `_initial_value`
flag only positions the cursor), and that echo travels the message pump --
on a busy machine it lands AFTER the user typed a newer query. The screen's
`Changed` handler cannot tell the echo from typing, so it overwrote the
newer query with the older one and bumped the search token, discarding the
in-flight fresh search. A user typing quickly on a loaded machine had their
search silently revert.

That echo is also what made
`test_console_conversation_browser_search_ignores_stale_results` fail ~1 run
in 5 (3/3 with CPU burners alongside): the "stale result" the failure showed
was the echo re-arming the stale QUERY, not the stale search's rows winning.
"""
from __future__ import annotations

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Input

from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleBrowserSearchInput,
)


class _EchoRecorder(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.changed_values: list[str] = []

    def compose(self) -> ComposeResult:
        yield ConsoleBrowserSearchInput(
            initial_value="alpha",
            id="probe-search",
        )

    @on(Input.Changed)
    def _record(self, event: Input.Changed) -> None:
        self.changed_values.append(str(event.value))


@pytest.mark.asyncio
async def test_a_constructor_value_posts_no_changed_echo():
    """Mounting with a value is display restoration, not a user edit."""
    app = _EchoRecorder()
    async with app.run_test() as pilot:
        for _ in range(6):
            await pilot.pause()

        probe = app.query_one("#probe-search", Input)
        assert probe.value == "alpha", "the initial value must still display"
        assert app.changed_values == [], (
            "the mount echo is back: a recomposed tray would re-arm a stale "
            f"query as if the user typed it: {app.changed_values}"
        )


@pytest.mark.asyncio
async def test_typing_still_posts_changed():
    """Suppressing the echo must not eat genuine edits."""
    app = _EchoRecorder()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#probe-search", Input).focus()
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()

        assert app.changed_values, "a real keystroke no longer posts Changed"
        # Textual 8's `select_on_focus` selects the restored text, so the
        # keystroke replaces it -- "x", not "alphax". The claim under test is
        # that the edit POSTS, not what the edit does to the selection.
        assert app.changed_values[-1] == "x"


@pytest.mark.asyncio
async def test_the_tray_mounts_the_echo_free_input():
    """The fix only holds if the TRAY constructs the subclass.

    The two tests above exercise `ConsoleBrowserSearchInput` directly, so a
    regression that swaps the tray's compose back to a plain
    `Input(value=...)` would leave them green while the echo returns. This
    drives the real tray through a sync carrying a query and listens for the
    echo at the app, exactly where the screen's handler sits.
    """
    from tldw_chatbook.Workspaces.conversation_browser_state import (
        build_console_conversation_browser_state,
    )
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        ConsoleWorkspaceContextTray,
    )
    from tldw_chatbook.Workspaces.display_state import (
        ConsoleWorkspaceContextState,
    )

    browser = build_console_conversation_browser_state(
        rows=(),
        active_workspace_id=None,
        group_collapse_preferences={},
        query="alpha",
        marks_available=False,
        error_copy="",
        result_total_count=None,
    )
    state = ConsoleWorkspaceContextState(
        heading="Workspace",
        workspace_label="Default",
        authority_label="",
        sync_label="",
        runtime_label="",
        conversation_rows=(),
        conversation_empty_copy="",
        conversation_browser=browser,
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=False,
        new_conversation_recovery="",
        recovery_copy="",
    )

    class _TrayHarness(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.changed_values: list[str] = []

        def compose(self) -> ComposeResult:
            yield ConsoleWorkspaceContextTray(state, id="console-workspace-context")

        @on(Input.Changed, "#console-workspace-conversation-search")
        def _record(self, event: Input.Changed) -> None:
            self.changed_values.append(str(event.value))

    app = _TrayHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        tray = app.query_one(ConsoleWorkspaceContextTray)
        # Sync AGAIN with the same query: the recompose this schedules is the
        # production-shaped echo source (`sync_state` recomposes on every
        # sync, and the flake's trigger was a recompose, not first mount).
        tray.sync_state(state)
        for _ in range(8):
            await pilot.pause()

        search = app.query_one("#console-workspace-conversation-search", Input)
        assert search.value == "alpha", "the query text must still display"
        assert app.changed_values == [], (
            "the recomposed tray's search input echoed its restored query as "
            f"a user edit: {app.changed_values}"
        )
