"""Characterisation + boundary tests for the Console workspace cluster.

Written before the wave-2 Task 2 extraction of `ConsoleWorkspaceController`
(`tldw_chatbook/UI/Console_Modules/workspace.py`) lands, driving the resume
flow and the conversation-search debounce through REAL interactions against
the pre-move `ChatScreen` -- the same "real production coroutine, not a
rebuilt double" discipline `test_console_native_chat_flow.py`'s own resume
coverage uses. Search-token/error state is exactly where a
snapshot-vs-live binding bug would hide (see wave 1's
`ConsoleDictationController` review history), so these assert the
token/error lifecycle explicitly, not just the visible rows.

Method calls below (`console._resume_console_workspace_conversation(...)`,
`console._refresh_console_workspace_conversation_search(...)`, etc.) move to
`console._workspace.<method>(...)` once the extraction lands; the six
`_console_workspace_conversation_*` state reads/writes below are expected to
keep working completely unchanged -- `ChatScreen` keeps get/set proxy
properties for them, exactly as `ConsoleDictationController`'s cluster did in
wave 1 (see that module's docstring).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


class _InputChangedEvent:
    """Minimal stand-in for `Input.Changed` -- matches what the handler reads
    (`event.value`, `event.input.disabled`, and `event.stop()`)."""

    def __init__(self, value: str, *, input_widget=None) -> None:
        self.value = value
        self.input = input_widget

    def stop(self) -> None:
        return None


def _set_workspace_search(console, query: str) -> None:
    """Drive the real search-changed handler, mirroring a real keystroke."""
    search = console.query_one("#console-workspace-conversation-search", Input)
    search.value = query
    console.on_console_workspace_conversation_search_changed(
        _InputChangedEvent(query, input_widget=search)
    )


def _conversation_tree_payload(
    conversation_id: str,
    *,
    title: str = "Resumed chat",
    workspace_id: str | None = None,
) -> dict:
    conversation: dict = {"id": conversation_id, "title": title}
    if workspace_id is not None:
        conversation["workspace_id"] = workspace_id
    return {"conversation": conversation, "root_threads": []}


@pytest.mark.asyncio
async def test_search_debounce_mirrors_query_and_bumps_token_and_timer():
    """Typing into the rail search mirrors into workspace state and arms a timer."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        _set_workspace_search(console, "alpha")
        await pilot.pause()

        assert console._console_workspace_conversation_query == "alpha"
        # `_console_workspace_conversation_search_token` is a pure mirror of
        # `_console_conversation_browser_search_token` inside this handler
        # (see the handler's own docstring / the module docstring for this
        # test file) -- not an independently incrementing counter, so the
        # only thing worth asserting is that the mirror is faithful.
        assert (
            console._console_workspace_conversation_search_token
            == console._console_conversation_browser_search_token
        )
        assert console._console_workspace_conversation_search_timer is not None


@pytest.mark.asyncio
async def test_search_debounce_empty_query_clears_state_synchronously():
    """Clearing the search box resets workspace search state with no pending timer."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        _set_workspace_search(console, "alpha")
        await pilot.pause()
        assert console._console_workspace_conversation_query == "alpha"

        _set_workspace_search(console, "")
        await pilot.pause()

        assert console._console_workspace_conversation_query == ""
        assert console._console_workspace_conversation_search_timer is None


@pytest.mark.asyncio
async def test_search_refresh_populates_rows_from_scope_service():
    """`_refresh_console_workspace_conversation_search` fills workspace rows
    from the scope service.

    Driven directly (the same "real production coroutine, not a rebuilt
    double" pattern `test_console_native_chat_flow.py`'s resume coverage
    uses) rather than through the live search-input handler: that handler
    only mirrors THREE fields onto workspace state (see this file's module
    docstring) and drives the sibling conversation-browser's own search --
    `_refresh_console_workspace_conversation_search` itself is reached (in
    production) only via `_refresh_console_workspace_conversation_search_
    after_selection`, so this exercises it the same way the existing
    characterisation in `test_console_native_chat_flow.py` does.
    """
    app = _build_test_app()
    active_workspace = app.workspace_registry_service.get_active_workspace()
    app.chat_conversation_scope_service = SimpleNamespace(
        list_conversations=lambda **kwargs: {
            "items": [
                {
                    "id": "conv-alpha",
                    "title": "Alpha project",
                    "state": "workspace-thread",
                }
            ],
            "total": 1,
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        console._console_workspace_conversation_query = "alpha"
        token = console._console_workspace_conversation_search_token
        await console._refresh_console_workspace_conversation_search(
            active_workspace.workspace_id, "alpha", token
        )
        await pilot.pause()

        rows = console._console_workspace_conversation_search_rows
        assert any(row.conversation_id == "conv-alpha" for row in rows), rows
        assert console._console_workspace_conversation_search_error == ""


@pytest.mark.asyncio
async def test_search_refresh_ignores_stale_token():
    """A refresh whose token no longer matches current state is a no-op.

    Simulates a slow in-flight refresh (token N-1) that lands after a newer
    keystroke already bumped the token to N -- exactly the race the token
    guard exists to close.
    """
    app = _build_test_app()
    active_workspace = app.workspace_registry_service.get_active_workspace()
    app.chat_conversation_scope_service = SimpleNamespace(
        list_conversations=lambda **kwargs: {
            "items": [
                {
                    "id": "conv-late",
                    "title": "Late arrival",
                    "state": "workspace-thread",
                }
            ],
            "total": 1,
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        console._console_workspace_conversation_query = "late"
        current_token = console._console_workspace_conversation_search_token + 1
        console._console_workspace_conversation_search_token = current_token

        await console._refresh_console_workspace_conversation_search(
            active_workspace.workspace_id, "late", current_token - 1
        )

        assert console._console_workspace_conversation_search_rows == ()


@pytest.mark.asyncio
async def test_resume_workspace_conversation_restores_native_session():
    """Resuming a real persisted conversation creates a matching native session."""
    app = _build_test_app()
    active_workspace = app.workspace_registry_service.get_active_workspace()
    app.chat_conversation_scope_service = SimpleNamespace(
        get_conversation_tree=lambda conversation_id, **kwargs: (
            _conversation_tree_payload(
                conversation_id,
                title="Resumed alpha",
                workspace_id=active_workspace.workspace_id,
            )
        )
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        resumed = await console._resume_console_workspace_conversation(
            "conv-resume-1"
        )
        await pilot.pause()

        assert resumed is True
        store = console._ensure_console_chat_store()
        active_session = store.switch_session(store.active_session_id)
        assert active_session.persisted_conversation_id == "conv-resume-1"
        assert active_session.workspace_id == active_workspace.workspace_id


@pytest.mark.asyncio
async def test_resume_workspace_conversation_missing_record_returns_false():
    """A missing conversation record is reported honestly as False."""
    app = _build_test_app()
    app.chat_conversation_scope_service = SimpleNamespace(
        get_conversation_tree=lambda *args, **kwargs: {}
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        resumed = await console._resume_console_workspace_conversation(
            "conv-missing"
        )
        await pilot.pause()

        assert resumed is False


@pytest.mark.asyncio
async def test_active_workspace_id_for_conversation_search_reads_registry():
    """Falls back to the registry's active workspace when none is staged."""
    app = _build_test_app()
    active_workspace = app.workspace_registry_service.get_active_workspace()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        assert (
            console._active_console_workspace_id_for_conversation_search()
            == active_workspace.workspace_id
        )
