import asyncio
import inspect
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual import on
from textual.app import App
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)

from tldw_chatbook.Chat.chat_conversation_scope_service import ChatConversationScopeService
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.Widgets.Console import (
    ConsoleComposerBar,
    ConsoleTranscript,
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


DUMMY_OPENAI_API_KEY = "DUMMY_OPENAI_API_KEY"


def _configure_openai_missing_api_key(app) -> None:
    """Keep setup-state tests on the API-key recovery path."""
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4o"}
    app.app_config["api_settings"] = {"openai": {"api_key": ""}}


def test_console_workspace_conversation_visible_title_is_rail_safe():
    """Verify long workspace conversation titles fit in the left rail."""
    assert (
        ConsoleWorkspaceContextTray._conversation_visible_title(
            "Console UAT Workspace Chat"
        )
        == "Console UAT Works..."
    )


def test_console_workspace_conversation_title_preserves_duplicate_suffix():
    """Duplicate-title disambiguators should remain visible in rail labels."""
    title = "Chat [deadbeef]"

    assert ConsoleWorkspaceContextTray._conversation_title(title) == title
    assert ConsoleWorkspaceContextTray._conversation_visible_title(title) == title


def test_console_provider_recovery_strip_visible_handles_missing_blocker_copy():
    """Missing blocker copy should keep the recovery strip hidden."""
    assert ChatScreen._console_provider_recovery_strip_visible(object(), None) is False


def test_console_workspace_status_row_empty_value_uses_unavailable():
    """Status labels ending in a colon should not repeat the label as the value."""
    assert ConsoleWorkspaceContextTray._split_status_row("Authority: ", "Authority") == (
        "Authority",
        "unavailable",
    )


def test_console_tree_messages_follow_latest_branch_only():
    """Verify resumed transcripts do not flatten regenerated alternatives."""
    rows = ChatScreen._iter_console_tree_messages(
        [
            {
                "id": "root",
                "content": "prompt",
                "children": [
                    {
                        "id": "old-assistant",
                        "content": "old draft",
                        "children": [],
                    },
                    {
                        "id": "latest-assistant",
                        "content": "latest draft",
                        "children": [
                            {
                                "id": "followup",
                                "content": "followup",
                                "children": [],
                            }
                        ],
                    },
                ],
            }
        ]
    )

    assert [row["id"] for row in rows] == ["root", "latest-assistant", "followup"]


def test_console_workspace_conversation_search_worker_uses_dedicated_group():
    source = inspect.getsource(
        ChatScreen.on_console_workspace_conversation_search_changed
    )

    assert 'group="console-workspace-conversation-search"' in source
    assert "exclusive=True" in source


def test_console_workspace_conversation_search_clear_button_stops_pending_timer():
    source = inspect.getsource(ChatScreen.on_button_pressed)
    clear_branch = source.split(
        'if button_id == "console-workspace-conversation-search-clear":',
        1,
    )[1].split(
        'if button_id and button_id.startswith("console-workspace-conversation-")',
        1,
    )[0]

    assert "_console_workspace_conversation_search_timer.stop()" in clear_branch
    assert "_console_workspace_conversation_search_timer = None" in clear_branch


def test_console_workspace_conversation_search_selection_refresh_invalidates_token():
    source = inspect.getsource(
        ChatScreen._refresh_console_workspace_conversation_search_after_selection
    )
    active_query_branch = source.split("if not query.strip():", 1)[1]
    before_refresh = active_query_branch.split(
        "await self._refresh_console_workspace_conversation_search",
        1,
    )[0]

    assert "_console_workspace_conversation_search_token += 1" in before_refresh


class _ReadyResolutionGateway:
    async def resolve_for_send(self, selection):
        return SimpleNamespace(
            provider=selection.provider,
            base_url=selection.base_url or "",
            model=selection.explicit_model or selection.configured_model or "test-model",
            ready=True,
            visible_copy="",
        )


class SelectionCapturingGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.selections = []
        self.sent_messages = []

    async def resolve_for_send(self, selection):
        self.selections.append(selection)
        return await super().resolve_for_send(selection)

    async def stream_chat(self, resolution, messages):
        self.sent_messages.append(list(messages))
        yield "accepted"


class WaitingGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat(self, resolution, messages):
        yield "partial"
        self.started.set()
        await self.release.wait()
        yield " done"


class DelayedWaitingGateway(WaitingGateway):
    def __init__(self) -> None:
        super().__init__()
        self.validation_started = asyncio.Event()
        self.validation_release = asyncio.Event()

    async def resolve_for_send(self, selection):
        self.validation_started.set()
        await self.validation_release.wait()
        return await super().resolve_for_send(selection)


class ConsoleNavigationHarness(ConsoleHarness):
    def __init__(self, app_instance: object) -> None:
        super().__init__(app_instance)
        self.navigation_messages = []

    @on(NavigateToScreen)
    def capture_navigation(self, message: NavigateToScreen) -> None:
        self.navigation_messages.append(message)
        message.stop()


class RestoredConsoleHarness(App[None]):
    """Mount a Console ChatScreen from a previously saved state.

    Args:
        app_instance: Test application object injected into the screen.
        restored_state: Serialized screen state passed to ``ChatScreen.restore_state``.
    """

    def __init__(self, app_instance: object, restored_state: dict) -> None:
        """Initialize the restore harness with the target app and state payload.

        Args:
            app_instance: Test application object injected into the screen.
            restored_state: Serialized screen state used during mount.
        """
        super().__init__()
        self.app_instance = app_instance
        self.restored_state = restored_state

    async def on_mount(self) -> None:
        """Restore and mount a Console ChatScreen for lifecycle regression tests."""
        screen = ChatScreen(self.app_instance)
        screen.restore_state(self.restored_state)
        await self.push_screen(screen)


class BlockedGateway:
    async def resolve_for_send(self, selection):
        return SimpleNamespace(
            provider="llama_cpp",
            base_url=selection.base_url or "",
            model="test-model",
            ready=False,
            visible_copy="Provider blocked: llama.cpp unavailable.",
        )

    async def stream_chat(self, resolution, messages):
        raise AssertionError("Blocked gateway should not stream")


class CapturingGateway(_ReadyResolutionGateway):
    def __init__(self, chunks=("accepted",)) -> None:
        self.chunks = chunks
        self.sent_messages = []

    async def stream_chat(self, resolution, messages):
        self.sent_messages.append(list(messages))
        for chunk in self.chunks:
            yield chunk


class WorkspaceLinkingPersistence:
    def __init__(self, registry_service) -> None:
        self.registry_service = registry_service
        self.conversation_count = 0
        self.message_count = 0

    def create_conversation(self, **kwargs):
        self.conversation_count += 1
        conversation_id = f"persisted-conversation-{self.conversation_count}"
        workspace_id = kwargs.get("workspace_id")
        if kwargs.get("scope_type") == "workspace" and workspace_id:
            self.registry_service.link_membership(
                workspace_id,
                item_type="conversation",
                item_id=conversation_id,
                role="workspace-thread",
                title=kwargs.get("conversation_title") or "Chat 1",
            )
        return conversation_id

    def create_message(self, **kwargs):
        self.message_count += 1
        return f"persisted-message-{self.message_count}"

    def update_message_content(self, **kwargs):
        return True


class StaticConversationTreeService:
    """Return deterministic persisted trees for regression tests.

    This is a CI service double only. CDP/UAT approval evidence must use the
    running app with real persistence and live provider/API responses.
    """

    def __init__(self, trees):
        self.trees = dict(trees)
        self.calls = []

    async def get_conversation_tree(self, conversation_id: str, **kwargs):
        self.calls.append({"conversation_id": conversation_id, **kwargs})
        return self.trees.get(
            conversation_id,
            {
                "conversation": None,
                "root_threads": [],
                "pagination": {"total_root_threads": 0},
            },
        )


class SearchableConversationService(StaticConversationTreeService):
    def __init__(self, conversations: dict[str, dict]) -> None:
        super().__init__(conversations)
        self.list_calls: list[dict[str, object]] = []

    async def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        query = str(kwargs.get("query") or "").strip().lower()
        scope_type = str(kwargs.get("scope_type") or "").strip()
        workspace_id = str(kwargs.get("workspace_id") or "").strip()
        limit = int(kwargs.get("limit") or 50)
        items = []
        for conversation_id, tree in self.trees.items():
            conversation = tree.get("conversation", {})
            title = str(conversation.get("title") or "")
            conversation_workspace_id = str(conversation.get("workspace_id") or "").strip()
            conversation_scope = str(conversation.get("scope_type") or "").strip()
            if scope_type == "global":
                if conversation_scope != "global" and conversation_workspace_id:
                    continue
            elif scope_type == "workspace":
                if conversation_workspace_id != workspace_id:
                    continue
            elif workspace_id and conversation_workspace_id != workspace_id:
                continue
            if query and query not in title.lower():
                continue
            items.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "workspace_id": conversation.get("workspace_id"),
                    "scope_type": conversation.get("scope_type"),
                    "state": conversation.get("state", "active"),
                }
            )
        return {
            "items": items[:limit],
            "pagination": {
                "total": len(items),
                "limit": limit,
                "offset": 0,
            },
        }


class FailingSearchConversationService(StaticConversationTreeService):
    def __init__(self) -> None:
        super().__init__({})
        self.list_calls: list[dict[str, object]] = []

    async def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        raise RuntimeError("search failed")


class SlowSearchConversationService(StaticConversationTreeService):
    def __init__(self) -> None:
        super().__init__({})
        self.list_calls: list[dict[str, object]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        self.started.set()
        await self.release.wait()
        return {
            "items": [],
            "pagination": {
                "total": 0,
                "limit": int(kwargs.get("limit") or 50),
                "offset": int(kwargs.get("offset") or 0),
            },
        }


class SlowFirstSearchableConversationService(SearchableConversationService):
    def __init__(self, conversations: dict[str, dict]) -> None:
        super().__init__(conversations)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        if len(self.list_calls) == 1:
            self.started.set()
            await self.release.wait()
        query = str(kwargs.get("query") or "").strip().lower()
        scope_type = str(kwargs.get("scope_type") or "").strip()
        workspace_id = str(kwargs.get("workspace_id") or "").strip()
        limit = int(kwargs.get("limit") or 50)
        items = []
        for conversation_id, tree in self.trees.items():
            conversation = tree.get("conversation", {})
            title = str(conversation.get("title") or "")
            conversation_workspace_id = str(conversation.get("workspace_id") or "").strip()
            conversation_scope = str(conversation.get("scope_type") or "").strip()
            if scope_type == "global":
                if conversation_scope != "global" and conversation_workspace_id:
                    continue
            elif scope_type == "workspace":
                if conversation_workspace_id != workspace_id:
                    continue
            elif workspace_id and conversation_workspace_id != workspace_id:
                continue
            if query and query not in title.lower():
                continue
            items.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "workspace_id": conversation.get("workspace_id"),
                    "scope_type": conversation.get("scope_type"),
                    "state": conversation.get("state", "active"),
                }
            )
        return {
            "items": items[:limit],
            "pagination": {
                "total": len(items),
                "limit": limit,
                "offset": 0,
            },
        }


class SyncSearchableConversationService(SearchableConversationService):
    def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        query = str(kwargs.get("query") or "").strip().lower()
        scope_type = str(kwargs.get("scope_type") or "").strip()
        workspace_id = str(kwargs.get("workspace_id") or "").strip()
        limit = int(kwargs.get("limit") or 50)
        items = []
        for conversation_id, tree in self.trees.items():
            conversation = tree.get("conversation", {})
            title = str(conversation.get("title") or "")
            conversation_workspace_id = str(conversation.get("workspace_id") or "").strip()
            conversation_scope = str(conversation.get("scope_type") or "").strip()
            if scope_type == "global":
                if conversation_scope != "global" and conversation_workspace_id:
                    continue
            elif scope_type == "workspace":
                if conversation_workspace_id != workspace_id:
                    continue
            elif workspace_id and conversation_workspace_id != workspace_id:
                continue
            if query and query not in title.lower():
                continue
            items.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "workspace_id": conversation.get("workspace_id"),
                    "scope_type": conversation.get("scope_type"),
                    "state": conversation.get("state", "active"),
                }
            )
        return {
            "items": items[:limit],
            "pagination": {
                "total": len(items),
                "limit": limit,
                "offset": 0,
            },
        }


class NoModeSyncSearchableConversationService(SearchableConversationService):
    def list_conversations(
        self,
        *,
        query: str = "",
        scope_type: str = "",
        workspace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        self.list_calls.append(
            {
                "query": query,
                "scope_type": scope_type,
                "workspace_id": workspace_id,
                "limit": limit,
                "offset": offset,
            }
        )
        normalized_query = str(query or "").strip().lower()
        normalized_scope_type = str(scope_type or "").strip()
        normalized_workspace_id = str(workspace_id or "").strip()
        items = []
        for conversation_id, tree in self.trees.items():
            conversation = tree.get("conversation", {})
            title = str(conversation.get("title") or "")
            conversation_workspace_id = str(conversation.get("workspace_id") or "").strip()
            conversation_scope = str(conversation.get("scope_type") or "").strip()
            if normalized_scope_type == "global":
                if conversation_scope != "global" and conversation_workspace_id:
                    continue
            elif normalized_scope_type == "workspace":
                if conversation_workspace_id != normalized_workspace_id:
                    continue
            elif normalized_workspace_id and conversation_workspace_id != normalized_workspace_id:
                continue
            if normalized_query and normalized_query not in title.lower():
                continue
            items.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "workspace_id": conversation.get("workspace_id"),
                    "scope_type": conversation.get("scope_type"),
                    "state": conversation.get("state", "active"),
                }
            )
        return {
            "items": items[:limit],
            "pagination": {
                "total": len(items),
                "limit": limit,
                "offset": offset,
            },
        }


class FakeConversationLocalMarksService:
    def __init__(self, starred: tuple[str, ...] = ()) -> None:
        self.starred = set(starred)

    def star_conversation(self, conversation_id: str) -> None:
        self.starred.add(conversation_id)

    def unstar_conversation(self, conversation_id: str) -> None:
        self.starred.discard(conversation_id)

    def is_starred(self, conversation_id: str) -> bool:
        return conversation_id in self.starred

    def list_marked_conversation_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.starred))


class FailThenRecoverGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, resolution, messages):
        self.calls += 1
        if self.calls == 1:
            yield "partial"
            raise RuntimeError("llama.cpp stream failed")
        yield "recovered"


async def _wait_for_text(screen, pilot, expected: str, *, attempts: int = 80) -> None:
    for _ in range(attempts):
        if expected in _visible_text(screen):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Text not found: {expected!r}. Visible text: {_visible_text(screen)!r}")


async def _wait_for_focus(app, pilot, widget, *, attempts: int = 40) -> None:
    for _ in range(attempts):
        if getattr(app, "focused", None) is widget:
            return
        await pilot.pause(0.05)
    focused = getattr(app, "focused", None)
    raise AssertionError(
        f"Focus did not reach {getattr(widget, 'id', widget)!r}; "
        f"focused={getattr(focused, 'id', focused)!r}"
    )


async def _wait_for_active_session_change(
    store: ConsoleChatStore,
    pilot,
    previous_session_id: str | None,
    *,
    attempts: int = 40,
) -> str:
    """Wait for the Console store to activate a different session."""
    for _ in range(attempts):
        active_session_id = store.active_session_id
        if active_session_id is not None and active_session_id != previous_session_id:
            return active_session_id
        await pilot.pause(0.05)
    raise AssertionError(
        "Console active session did not change. "
        f"previous={previous_session_id!r}; active={store.active_session_id!r}"
    )


async def _wait_for_active_session(
    store: ConsoleChatStore,
    pilot,
    expected_session_id: str,
    *,
    attempts: int = 40,
) -> None:
    """Wait for the Console store to activate the expected session."""
    for _ in range(attempts):
        if store.active_session_id == expected_session_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        "Console active session did not match expected session. "
        f"expected={expected_session_id!r}; active={store.active_session_id!r}"
    )


async def _open_console_inspector_rail(console: ChatScreen, pilot) -> None:
    """Open the right rail before asserting inspector-visible content."""
    rail_state = replace(
        console._current_console_rail_state(),
        right_open=True,
    )
    console._sync_console_rail_visibility(rail_state)
    await _wait_for_selector(console, pilot, "#console-run-inspector-state")
    for _ in range(40):
        inspector = console.query_one("#console-run-inspector-state")
        if inspector.display and inspector.region.width > 0 and inspector.region.height > 0:
            return
        await pilot.pause(0.05)
    inspector = console.query_one("#console-run-inspector-state")
    raise AssertionError(
        "Console run inspector is not visible/actionable: "
        f"display={inspector.display!r} region={inspector.region!r}"
    )


async def _open_console_context_rail(console: ChatScreen, pilot) -> None:
    """Open the left rail before asserting context-visible content."""
    rail_state = replace(
        console._current_console_rail_state(),
        left_open=True,
    )
    console._sync_console_rail_visibility(rail_state)
    await _wait_for_selector(console, pilot, "#console-workspace-authority-label")
    for _ in range(40):
        label = console.query_one("#console-workspace-authority-label")
        if label.display and label.region.width > 0 and label.region.height > 0:
            return
        await pilot.pause(0.05)
    label = console.query_one("#console-workspace-authority-label")
    raise AssertionError(
        "Console workspace authority row is not visible/actionable: "
        f"display={label.display!r} region={label.region!r}"
    )


def _static_plain_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _widget_text(widget) -> str:
    if hasattr(widget, "renderable"):
        renderable = widget.renderable
        return getattr(renderable, "plain", str(renderable))
    label = getattr(widget, "label", "")
    return getattr(label, "plain", str(label))


def _console_workspace_conversation_texts(console) -> list[str]:
    rows = console.query(".console-workspace-conversation-row")
    return [_widget_text(row) for row in rows]


def _workspace_conversation_row_by_id(console, conversation_id: str):
    for row in console.query(".console-workspace-conversation-row"):
        if getattr(row, "conversation_id", None) == conversation_id:
            return row
    return None


def _workspace_conversation_row_by_key(console, row_key: str):
    for row in console.query(".console-workspace-conversation-row"):
        if getattr(row, "row_key", None) == row_key:
            return row
    return None


def _console_workspace_conversation_row_id_for_session(console, session_id: str) -> str:
    target_conversation_id = f"native:{session_id}"
    for row in console.query(".console-workspace-conversation-row"):
        if getattr(row, "conversation_id", None) == target_conversation_id:
            return str(row.id)
    rows = [
        (getattr(row, "id", ""), getattr(row, "conversation_id", None), _widget_text(row))
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation row for {target_conversation_id!r} not found. "
        f"Rows: {rows!r}"
    )


async def _click_console_workspace_conversation_for_session(
    console,
    pilot,
    store,
    session_id: str,
    *,
    attempts: int = 20,
) -> None:
    """Click a workspace conversation row once Textual hit-testing is ready."""
    row_id = _console_workspace_conversation_row_id_for_session(console, session_id)
    for _ in range(attempts):
        if await pilot.click(f"#{row_id}"):
            for _ in range(10):
                if store.active_session_id == session_id:
                    return
                await pilot.pause(0.05)
        await pilot.pause(0.05)
    rows = [
        (
            getattr(row, "id", ""),
            getattr(row, "conversation_id", None),
            getattr(row, "region", None),
            _widget_text(row),
        )
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation click did not activate {session_id!r}. "
        f"active={store.active_session_id!r}; rows={rows!r}"
    )


async def _click_console_workspace_conversation_for_id(
    console,
    pilot,
    conversation_id: str,
    *,
    attempts: int = 40,
) -> str:
    """Click a workspace conversation row by persisted conversation id."""
    for _ in range(attempts):
        for row in console.query(".console-workspace-conversation-row"):
            if getattr(row, "conversation_id", None) == conversation_id:
                row_id = str(row.id)
                await pilot.click(f"#{row_id}")
                return row_id
        await pilot.pause(0.05)
    rows = [
        (getattr(row, "id", ""), getattr(row, "conversation_id", None), _widget_text(row))
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation row for {conversation_id!r} not found. "
        f"Rows: {rows!r}"
    )


async def _click_console_workspace_conversation_for_row_key(
    console,
    pilot,
    row_key: str,
    *,
    attempts: int = 40,
) -> str:
    """Click a workspace conversation row by grouped browser row key."""
    for _ in range(attempts):
        for row in console.query(".console-workspace-conversation-row"):
            if getattr(row, "row_key", None) == row_key:
                row_id = str(row.id)
                await pilot.click(f"#{row_id}")
                return row_id
        await pilot.pause(0.05)
    rows = [
        (
            getattr(row, "id", ""),
            getattr(row, "row_key", None),
            getattr(row, "conversation_id", None),
            _widget_text(row),
        )
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation row key {row_key!r} not found. "
        f"Rows: {rows!r}"
    )


async def _wait_for_workspace_conversation_text(
    console,
    pilot,
    expected: str,
    *,
    selected: bool | None = None,
    attempts: int = 40,
) -> list[str]:
    for _ in range(attempts):
        row_texts = _console_workspace_conversation_texts(console)
        for text in row_texts:
            if expected not in text:
                continue
            if selected is None or text.startswith("> ") == selected:
                return row_texts
        await pilot.pause(0.05)
    raise AssertionError(
        f"Workspace conversation {expected!r} not found. "
        f"Rows: {_console_workspace_conversation_texts(console)!r}"
    )


async def _wait_for_console_rename_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-rename-session-modal")
            and host.screen_stack[-1].query("#console-rename-session-title")
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console rename modal did not open")


async def _wait_for_console_screen(host: ConsoleHarness, console, pilot) -> None:
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1] is console:
            await pilot.pause()
            return
        await pilot.pause(0.05)
    raise AssertionError("Console modal did not dismiss")


async def _wait_for_workspace_switcher_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-workspace-switcher-modal")
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console workspace switcher modal did not open")


def _select_llamacpp_console(console: ChatScreen) -> None:
    """Select the native llama.cpp path after mounted controls initialize."""
    app_config = console.app_instance.app_config
    api_settings = app_config.setdefault("api_settings", {})
    llama_settings = api_settings.setdefault("llama_cpp", {})
    llama_settings.setdefault("api_url", "http://127.0.0.1:9099/v1")
    llama_settings.setdefault("model", "test-model")
    console._console_control_provider = "llama_cpp"
    console._console_control_model = "test-model"
    console._sync_console_control_bar()


@pytest.mark.asyncio
async def test_console_native_generic_provider_send_renders_completed_message(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {}
    captured_kwargs = []

    def fake_chat_api_call(**_kwargs):
        captured_kwargs.append(_kwargs)
        return "generic provider response"

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        gateway = console._ensure_console_provider_gateway()
        app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "generic provider response")

        assert isinstance(gateway, ConsoleProviderGateway)
        assert captured_kwargs
        assert captured_kwargs[-1]["api_endpoint"] == "openai"
        assert captured_kwargs[-1]["api_key"] == DUMMY_OPENAI_API_KEY
        assert console._ensure_console_chat_controller().run_state.status is ConsoleRunStatus.COMPLETED
        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assistant_messages = [
            message for message in messages if message.role is ConsoleMessageRole.ASSISTANT
        ]
        assert assistant_messages[-1].status == "complete"


@pytest.mark.asyncio
async def test_console_native_send_button_click_dispatches_message(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {}
    captured_kwargs = []

    def fake_chat_api_call(**_kwargs):
        captured_kwargs.append(_kwargs)
        return "click provider response"

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("click send")

        await pilot.click("#console-send-message")
        await _wait_for_text(console, pilot, "click provider response")

        assert captured_kwargs
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_successful_send_does_not_leave_empty_send_tooltip(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}

    def fake_chat_api_call(**_kwargs):
        return "sent response"

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("send once")

        await pilot.click("#console-send-message")
        await _wait_for_text(console, pilot, "sent response")

        send_button = console.query_one("#console-send-message", Button)
        assert composer.draft_text() == ""
        assert send_button.tooltip != "Type a message before sending."


@pytest.mark.asyncio
async def test_console_native_missing_key_blocks_before_clearing_generic_draft():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "MISSING_OPENAI_KEY"}
    }
    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("preserve this")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "missing API key")

        assert composer.draft_text() == "preserve this"


@pytest.mark.asyncio
async def test_console_native_enter_on_setup_blocked_send_shows_recovery_feedback():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "MISSING_OPENAI_KEY"}
    }
    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("preserve this from keyboard")

        await pilot.press("enter")
        await pilot.pause(0.05)

        assert composer.draft_text() == "preserve this from keyboard"
        assert notifications == [
            (
                "Add API Key in Settings before sending.",
                {"severity": "warning"},
            )
        ]


@pytest.mark.asyncio
async def test_console_setup_blocked_send_adds_durable_transcript_recovery_feedback():
    """Verify setup-blocked sends leave durable transcript recovery feedback."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "MISSING_OPENAI_KEY"}
    }
    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("blocked setup draft")

        await pilot.press("enter")
        await _wait_for_text(console, pilot, "Add API Key in Settings before sending.")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert composer.draft_text() == "blocked setup draft"
        assert messages[-1].role is ConsoleMessageRole.SYSTEM
        assert messages[-1].content == "Add API Key in Settings before sending."

    assert notifications == [
        (
            "Add API Key in Settings before sending.",
            {"severity": "warning"},
        )
    ]


@pytest.mark.asyncio
async def test_console_native_blocked_send_preserves_composer_text_and_shows_recovery():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = BlockedGateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("blocked draft")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "Provider blocked")

        assert composer.draft_text() == "blocked draft"


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    (
        ("http://127.0.0.1:9099/v1/chat/completions", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/v1/models", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/v1", "http://127.0.0.1:9099"),
        ("127.0.0.1:9099", "http://127.0.0.1:9099"),
        ("127.0.0.1:9099/v1", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/completion", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/", "http://127.0.0.1:9099"),
        (None, "http://127.0.0.1:9099"),
    ),
)
def test_console_llamacpp_base_url_normalizes_openai_compatible_endpoints(raw_url, expected):
    screen = ChatScreen(_build_test_app())

    assert screen._normalize_llamacpp_base_url(raw_url) == expected


def test_console_transcript_sync_timer_polls_at_coarse_interval(monkeypatch):
    screen = ChatScreen(_build_test_app())
    captured = {}

    def fake_set_interval(interval, callback):
        captured["interval"] = interval
        captured["callback"] = callback
        return SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr(screen, "set_interval", fake_set_interval)

    screen._start_console_transcript_sync_timer()

    assert captured["interval"] >= 0.15


def test_console_transcript_fingerprint_tolerates_empty_variant_container():
    screen = ChatScreen(_build_test_app())
    message = SimpleNamespace(
        id="m1",
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        status="complete",
        turn_id="turn-1",
        persisted_message_id=None,
        variants=SimpleNamespace(selected_index=0, variants=None),
    )

    fingerprint = screen._native_console_transcript_fingerprint([message])

    assert fingerprint[1][0][-1] == (0, ())


def test_console_provider_selection_reads_local_llamacpp_configured_model():
    app = _build_test_app()
    app.chat_api_provider_value = "local_llamacpp"
    app.chat_api_model_value = "runtime-model"
    app.app_config["api_settings"] = {
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099/v1/chat/completions",
            "model": "configured-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.provider == "local_llamacpp"
    assert selection.base_url == "http://127.0.0.1:9099"
    assert selection.explicit_model == "runtime-model"
    assert selection.configured_model == "configured-model"
    assert selection.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID


def test_console_provider_selection_restores_default_workspace_when_none_active():
    app = _build_test_app()
    service = app.workspace_registry_service
    with service.db.transaction() as conn:
        conn.execute("UPDATE workspace_records SET active = 0")
    assert service.get_active_workspace() is None
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID
    assert service.get_active_workspace().workspace_id == DEFAULT_WORKSPACE_ID


def test_console_configured_llamacpp_override_wins_over_provider_api_url():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "configured-model"
    app.app_config["console"] = {
        "llama_cpp_base_url_override": "http://127.0.0.1:9099/v1",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"


def test_console_llamacpp_api_base_url_wins_over_merged_provider_api_url(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "configured-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "api_base_url": "http://127.0.0.1:9099/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"


def test_console_llamacpp_env_url_wins_over_provider_api_url(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:9099/v1")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "configured-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"


def test_console_session_settings_blank_base_url_keeps_llamacpp_fallback(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:9099/v1")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "runtime-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="settings-model",
            base_url=None,
        )
    )
    store.switch_session(session.id)
    screen._console_chat_store = store

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"
    assert selection.explicit_model == "settings-model"


def test_console_session_settings_base_url_wins_over_llamacpp_fallback(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:9099/v1")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "runtime-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="settings-model",
            base_url="http://127.0.0.1:9999/v1",
        )
    )
    store.switch_session(session.id)
    screen._console_chat_store = store

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9999"
    assert selection.explicit_model == "settings-model"


@pytest.mark.asyncio
async def test_console_stop_interrupts_stream_and_keeps_partial_message_visible():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await _wait_for_text(console, pilot, "partial")
        assert "streaming" in _visible_text(console).lower()

        console.query_one("#console-stop-generation", Button).press()
        await _wait_for_text(console, pilot, "stopped")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert messages[-1].content == "partial"
        assert messages[-1].status == "stopped"


@pytest.mark.asyncio
async def test_console_composer_stop_is_subdued_when_idle():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        stop_button = composer.query_one("#console-stop-generation", Button)

        assert stop_button.disabled is False
        assert stop_button.has_class("console-action-disabled")
        assert stop_button.has_class("console-stop-idle")
        assert not stop_button.has_class("console-stop-active")

        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await _wait_for_text(console, pilot, "partial")

        assert send_button.disabled is False
        assert send_button.has_class("console-action-disabled")
        assert send_button.has_class("console-send-blocked")
        assert not send_button.has_class("console-action-primary")
        assert stop_button.disabled is False
        assert stop_button.has_class("console-stop-active")
        assert not stop_button.has_class("console-action-disabled")
        assert not stop_button.has_class("console-stop-idle")

        stop_button.press()
        await _wait_for_text(console, pilot, "stopped")


@pytest.mark.asyncio
async def test_console_duplicate_send_during_stream_does_not_break_stop_control():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await _wait_for_text(console, pilot, "partial")

        composer.load_draft("second send")
        send_button = console.query_one("#console-send-message", Button)
        assert send_button.disabled is False
        assert send_button.has_class("console-send-blocked")
        await console.handle_console_send_message(Button.Pressed(send_button))
        await pilot.pause(0.1)
        assert console._ensure_console_chat_controller().run_state.status.value == "streaming"

        console.query_one("#console-stop-generation", Button).press()
        await _wait_for_text(console, pilot, "stopped")


@pytest.mark.asyncio
async def test_console_streaming_chunks_render_after_slow_provider_validation():
    gateway = DelayedWaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.validation_started.wait(), timeout=1)
        assert console._ensure_console_chat_controller().run_state.status is ConsoleRunStatus.VALIDATING
        console._sync_console_control_bar()
        send_button = console.query_one("#console-send-message", Button)
        stop_button = console.query_one("#console-stop-generation", Button)

        assert send_button.disabled is False
        assert send_button.has_class("console-action-disabled")
        assert send_button.has_class("console-send-blocked")
        assert not send_button.has_class("console-action-primary")
        assert stop_button.disabled is False
        assert stop_button.has_class("console-stop-idle")

        gateway.validation_release.set()
        await asyncio.wait_for(gateway.started.wait(), timeout=1)
        await _wait_for_text(console, pilot, "partial")
        gateway.release.set()
        await _wait_for_text(console, pilot, "partial done")


@pytest.mark.asyncio
async def test_console_collapsed_paste_sends_full_payload_not_visible_token():
    long_text = "x" * 80
    gateway = CapturingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_pasted_text(long_text)

        assert "Pasted Text: 80 Characters" in _visible_text(console)
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

    assert gateway.sent_messages[-1][-1]["content"] == long_text
    assert "Pasted Text: 80 Characters" not in gateway.sent_messages[-1][-1]["content"]


@pytest.mark.asyncio
async def test_console_native_send_preserves_expanded_payload_whitespace():
    gateway = CapturingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("  padded payload  ")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

    assert gateway.sent_messages[-1][-1]["content"] == "  padded payload  "


@pytest.mark.asyncio
async def test_console_configured_model_reaches_gateway_when_ui_model_is_unset():
    gateway = SelectionCapturingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "local_llamacpp"
    app.chat_api_model_value = None
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config["api_settings"] = {
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099/v1/chat/completions",
            "model": "configured-model",
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._console_control_provider = "local_llamacpp"
        console._console_control_model = None
        console._sync_console_control_bar()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

    assert gateway.selections[-1].explicit_model is None
    assert gateway.selections[-1].configured_model == "configured-model"


@pytest.mark.asyncio
async def test_console_native_send_clears_composer_after_acceptance_and_updates_store():
    """Verify accepted sends clear the composer and render compact transcript text."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("hel", "lo"))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "Assistant  hello")

        assert composer.draft_text() == ""
        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert messages[-2].content == "hello"
        assert messages[-1].content == "hello"


@pytest.mark.asyncio
async def test_console_chat_lifecycle_state_survives_screen_recreation_return():
    """Verify Console chat tabs, transcript, and draft restore after recreation."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("assistant return",))
    saved_state: dict | None = None
    first_session_id: str | None = None
    second_session_id: str | None = None

    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("typed text")
        composer.insert_pasted_text(" and pasted text")
        assert "typed text and pasted text" in _visible_text(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "assistant return")

        store = console._ensure_console_chat_store()
        first_session_id = store.active_session_id
        assert first_session_id is not None
        await pilot.click("#console-new-chat-tab")
        second_session_id = await _wait_for_active_session_change(
            store,
            pilot,
            first_session_id,
        )
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second_session_id}")
        composer.load_draft("draft before return")
        await console._sync_native_console_chat_ui()

        saved_state = console.save_state()

    assert saved_state is not None
    assert first_session_id is not None
    assert second_session_id is not None

    restored_host = RestoredConsoleHarness(app, saved_state)
    async with restored_host.run_test(size=(160, 48)) as pilot:
        console = restored_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second_session_id}")
        await _wait_for_text(console, pilot, "draft before return")

        store = console._ensure_console_chat_store()
        assert store.active_session_id == second_session_id

        await pilot.click(f"#console-session-tab-{first_session_id}")
        await _wait_for_active_session(store, pilot, first_session_id)
        await _wait_for_text(console, pilot, "typed text and pasted text")
        await _wait_for_text(console, pilot, "assistant return")


@pytest.mark.asyncio
async def test_console_send_refreshes_workspace_conversation_rail_after_persistence():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("accepted",))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        row_texts = _console_workspace_conversation_texts(console)
        assert any("Chat 1" in text for text in row_texts)
        assert len(console.query("#console-workspace-empty-conversations")) == 0
        store = console._ensure_console_chat_store()
        store.persistence = WorkspaceLinkingPersistence(app.workspace_registry_service)
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-0")

        row = console.query_one("#console-workspace-conversation-0")
        row_text = _widget_text(row)
        assert row_text.startswith("> ")
        assert "Chat 1" in row_text
        assert "\n" in row_text
        assert "Chats" in row_text
        assert "workspace-thread" not in row_text
        assert not re.search(r"\[[0-9a-f]{8}\]", row_text)
        assert len(console.query("#console-workspace-empty-conversations")) == 0


@pytest.mark.asyncio
async def test_console_send_after_workspace_switch_persists_to_selected_workspace():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("accepted",))
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-change-workspace")
        store = console._ensure_console_chat_store()
        store.persistence = WorkspaceLinkingPersistence(service)
        _select_llamacpp_console(console)
        first_session = store.ensure_session()
        store.replace_session_settings(
            first_session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
        )
        assert first_session.workspace_id == "ws-a"

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)
        assert service.get_active_workspace().workspace_id == "ws-b"

        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == "ws-b"
        assert active_session.title == "Workspace B Chat"
        assert active_session.settings.provider == "llama_cpp"
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello from b")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        workspace_a_conversations = service.list_workspace_conversations("ws-a")
        workspace_b_conversations = service.list_workspace_conversations("ws-b")
        assert workspace_a_conversations == ()
        assert [row.title for row in workspace_b_conversations] == [active_session.title]


@pytest.mark.asyncio
async def test_console_workspace_switch_refreshes_visible_session_tabs():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-change-workspace")
        store = console._ensure_console_chat_store()
        first_session = store.ensure_session()
        assert first_session.workspace_id == "ws-a"

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)

        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == "ws-b"
        await _wait_for_selector(console, pilot, f"#console-session-tab-{active_session.id}")
        assert "Workspace B Chat" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_workspace_switch_refresh_is_not_dropped_during_inflight_sync():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-change-workspace")
        store = console._ensure_console_chat_store()
        first_session = store.ensure_session()
        assert first_session.workspace_id == "ws-a"

        first_sync_blocked = asyncio.Event()
        release_first_sync = asyncio.Event()
        original_sync_tabs = console._sync_console_native_session_tabs
        blocked_once = False

        async def blocking_sync_tabs():
            nonlocal blocked_once
            await original_sync_tabs()
            if blocked_once:
                return
            blocked_once = True
            first_sync_blocked.set()
            await release_first_sync.wait()

        console._sync_console_native_session_tabs = blocking_sync_tabs
        first_sync_task = asyncio.create_task(console._sync_native_console_chat_ui())
        await first_sync_blocked.wait()

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)

        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == "ws-b"
        release_first_sync.set()
        await first_sync_task

        await _wait_for_selector(console, pilot, f"#console-session-tab-{active_session.id}")
        assert "Workspace B Chat" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_mount_uses_active_workspace_title_for_initial_session():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_text(console, pilot, "Workspace A Chat")
        assert "Workspace A" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_tab_switch_aligns_active_workspace_context():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Workspace A Chat", workspace_id="ws-a")
        second = store.create_session(title="Workspace B Chat", workspace_id="ws-b")
        service.set_active_workspace("ws-b")
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-session-tab-{first.id}")
        assert store.active_session_id == second.id
        assert service.get_active_workspace().workspace_id == "ws-b"

        await pilot.click(f"#console-session-tab-{first.id}")

        assert store.active_session_id == first.id
        assert service.get_active_workspace().workspace_id == "ws-a"
        await _wait_for_text(console, pilot, "Workspace A")


@pytest.mark.asyncio
async def test_console_unsupported_provider_block_renders_one_normalized_system_message():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="wip_provider", model="test-model"),
        )
        await console._sync_native_console_chat_ui()

        await console._submit_console_native_draft("hello")
        await _wait_for_text(console, pilot, "Provider blocked")

        messages = store.messages_for_session(store.active_session_id)
        system_messages = [message.content for message in messages if message.role is ConsoleMessageRole.SYSTEM]
        assert system_messages == [
            "Provider blocked: 'wip_provider' is not available in Console yet. "
            "Choose a supported provider."
        ]
        assert console._ensure_console_chat_controller().run_state.visible_copy == system_messages[0]


@pytest.mark.asyncio
async def test_console_add_api_key_recovery_targets_provider_settings_category():
    app = _build_test_app()
    app.app_config["api_settings"] = {"huggingface": {}}
    host = ConsoleNavigationHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="huggingface", model="meta-llama/test-model"),
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#workbench-recovery-action")

        await pilot.click("#workbench-recovery-action")

        assert len(host.navigation_messages) == 1
        message = host.navigation_messages[0]
        assert message.screen_name == "settings"
        assert message.screen_context == {
            "category": SettingsCategoryId.PROVIDERS_MODELS.value,
            "provider": "huggingface",
            "model": "meta-llama/test-model",
        }


@pytest.mark.asyncio
async def test_console_add_api_key_recovery_tolerates_missing_session_settings():
    app = _build_test_app()
    host = ConsoleNavigationHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="huggingface", model="meta-llama/test-model"),
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#workbench-recovery-action")
        console._active_console_provider_model_display = lambda: (
            "huggingface",
            "meta-llama/test-model",
            None,
        )

        await pilot.click("#workbench-recovery-action")

        assert len(host.navigation_messages) == 1
        message = host.navigation_messages[0]
        assert message.screen_context == {
            "category": SettingsCategoryId.PROVIDERS_MODELS.value,
            "provider": "huggingface",
            "model": "meta-llama/test-model",
        }


@pytest.mark.asyncio
async def test_console_assistant_message_click_exposes_selected_actions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        await pilot.click(f"#console-message-{message.id}")
        await _wait_for_selector(console, pilot, f"#console-message-action-regenerate-{message.id}")

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.selected_message_id == message.id


@pytest.mark.asyncio
async def test_console_transcript_wraps_long_message_content_without_horizontal_overflow():
    app = _build_test_app()
    host = ConsoleHarness(app)

    long_answer = " ".join(["wrapped assistant response segment"] * 180)

    async with host.run_test(size=(92, 32)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=long_answer,
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        row = console.query_one(f"#console-message-{message.id}", Static)

        assert row.region.width <= transcript.region.width
        assert transcript.virtual_size.width <= transcript.region.width
        assert row.region.height > 2


@pytest.mark.asyncio
async def test_console_selected_message_copy_action_uses_app_clipboard():
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-copy-{message.id}")

        await pilot.click(f"#console-message-action-copy-{message.id}")
        await pilot.pause()

    app.copy_to_clipboard.assert_called_once_with("answer")
    assert console._last_console_action.action_id == "copy"


@pytest.mark.asyncio
async def test_console_clicking_rendered_message_shows_action_row():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        await pilot.click(f"#console-message-{message.id}")
        await _wait_for_selector(console, pilot, f"#console-message-action-copy-{message.id}")


@pytest.mark.asyncio
async def test_console_selected_message_copy_action_works_from_keyboard():
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()
        await _wait_for_focus(console.app, pilot, transcript)
        await pilot.press("down")
        await pilot.press("enter")
        await _wait_for_selector(console, pilot, f"#console-message-action-copy-{message.id}")

        copy_selector = f"console-message-action-copy-{message.id}"
        for _ in range(16):
            focused = getattr(console.app, "focused", None)
            if getattr(focused, "id", None) == copy_selector:
                break
            await pilot.press("tab")
        else:
            raise AssertionError("Keyboard focus did not reach the selected-message Copy action")

        await pilot.press("enter")
        await pilot.pause()

    app.copy_to_clipboard.assert_called_once_with("answer")
    assert console._last_console_action.action_id == "copy"


@pytest.mark.asyncio
async def test_console_message_action_keyboard_focus_stays_inside_action_row():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-delete-{message.id}")

        transcript.focus_action(message.id, "delete")
        delete_button = console.query_one(f"#console-message-action-delete-{message.id}", Button)
        await _wait_for_focus(console.app, pilot, delete_button)

        await pilot.press("tab")
        copy_button = console.query_one(f"#console-message-action-copy-{message.id}", Button)
        await _wait_for_focus(console.app, pilot, copy_button)

        await pilot.press("tab")
        edit_button = console.query_one(f"#console-message-action-edit-{message.id}", Button)
        await _wait_for_focus(console.app, pilot, edit_button)

        await pilot.press("tab")
        save_as_button = console.query_one(f"#console-message-action-save-as-{message.id}", Button)
        await _wait_for_focus(console.app, pilot, save_as_button)

        await pilot.press("enter")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-save-as-modal")

    assert console._last_console_action.action_id == "save-as"


@pytest.mark.asyncio
async def test_console_inspector_hides_selected_message_group_without_selection():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-inspector-state")

        inspector = console.query_one("#console-run-inspector-state")
        assert "Selected Message" not in _visible_text(inspector)
        assert len(console.query("#console-inspector-selected-message-heading")) == 0


@pytest.mark.asyncio
async def test_console_setup_required_state_groups_recovery_and_action_copy():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#workbench-recovery-callout")

        recovery = console.query_one("#workbench-recovery-callout")
        recovery_text = getattr(recovery.renderable, "plain", str(recovery.renderable))
        assert "Provider setup needed" in recovery_text
        assert "Impact: Send is blocked until setup is finished." in recovery_text
        assert str(console.query_one("#workbench-recovery-action", Button).label) == "Add API Key"
        assert recovery.region.y < console.query_one("#console-native-transcript").region.y


@pytest.mark.asyncio
async def test_console_empty_transcript_teaches_setup_and_start_paths():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        empty_text = _visible_text(transcript)
        assert "Start Console" in empty_text
        assert "Add an API key" in empty_text
        assert "type in Composer" in empty_text
        assert "Attach context" in empty_text
        assert "Run Library RAG" in empty_text


@pytest.mark.asyncio
async def test_console_workspace_authority_rows_are_structured_for_scanning():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_context_rail(console, pilot)

        assert _static_plain_text(
            console.query_one("#console-workspace-authority-label", Static)
        ) == "Storage"
        assert "local" in _static_plain_text(
            console.query_one("#console-workspace-authority-value", Static)
        )
        assert _static_plain_text(
            console.query_one("#console-workspace-runtime-label", Static)
        ) == "File tools"
        assert "Off in Default workspace" in _static_plain_text(
            console.query_one("#console-workspace-runtime-value", Static)
        )
        assert _static_plain_text(
            console.query_one("#console-workspace-handoff-label", Static)
        ) == "Handoff"
        assert "Not configured" in _static_plain_text(
            console.query_one("#console-workspace-handoff-value", Static)
        )


@pytest.mark.asyncio
async def test_console_inspector_setup_state_explains_blocked_send_without_selection():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector_rail(console, pilot)

        inspector_text = _visible_text(console.query_one("#console-run-inspector-state"))
        assert "Setup" in inspector_text
        assert "Blocked impact" in inspector_text
        assert "Add API Key" in inspector_text
        assert "Selected Message" not in inspector_text


@pytest.mark.asyncio
async def test_console_composer_setup_placeholder_names_recovery_action():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer_text = _visible_text(composer)
        assert "Setup required: Add API Key before sending." in composer_text
        assert console.query_one("#console-send-message", Button).tooltip == (
            "Add API Key in Settings before sending."
        )


@pytest.mark.asyncio
async def test_console_selected_message_updates_inspector_action_guidance():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="first assistant variant",
        )
        store.add_variant(message.id, "second assistant variant")
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#console-inspector-selected-message")

        inspector_text = _visible_text(console.query_one("#console-run-inspector-state"))
        assert "Selected message: Assistant message" in inspector_text
        assert "Message actions: Copy, Edit, Save as..., Regenerate, Continue, Feedback, Delete" in inspector_text
        assert "Keyboard: Tab/Shift+Tab cycle actions; Enter activates; Esc clears selection" in inspector_text
        assert "Variants: 2 variants, showing 2/2" in inspector_text
        assert "Excerpt: second assistant variant" in inspector_text


@pytest.mark.asyncio
async def test_console_selected_message_feedback_action_records_rating():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-feedback-up-{message.id}")

        await pilot.click(f"#console-message-action-feedback-up-{message.id}")
        await pilot.pause()

    updated = store.get_message(message.id)
    assert updated.feedback == "up"
    assert console._last_console_action.action_id == "feedback-up"
    assert console._last_console_action.visible_copy == "Marked message feedback: up."


@pytest.mark.asyncio
async def test_console_selected_message_delete_action_removes_message_from_transcript():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-delete-{message.id}")

        await pilot.click(f"#console-message-action-delete-{message.id}")
        await pilot.pause()

        assert store.messages_for_session(session.id) == [message]
        assert console._last_console_action.action_id == "delete"
        assert console._last_console_action.visible_copy == "Press Delete again to remove this message."

        delete_button = console.query_one(f"#console-message-action-delete-{message.id}", Button)
        delete_button.press()
        await pilot.pause()

    assert store.messages_for_session(session.id) == []
    assert console._last_console_action.action_id == "delete"
    assert console._last_console_action.visible_copy == "Deleted message from transcript."


@pytest.mark.asyncio
async def test_console_delete_confirmation_resets_when_selection_changes():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        first_message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="first answer",
        )
        second_message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="second answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(first_message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-delete-{first_message.id}")

        await pilot.click(f"#console-message-action-delete-{first_message.id}")
        await pilot.pause()
        assert console._last_console_action.visible_copy == "Press Delete again to remove this message."

        transcript.select_message(second_message.id)
        await pilot.pause()
        transcript.select_message(first_message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-delete-{first_message.id}")

        delete_button = console.query_one(f"#console-message-action-delete-{first_message.id}", Button)
        delete_button.press()
        await pilot.pause()

    assert [message.id for message in store.messages_for_session(session.id)] == [
        first_message.id,
        second_message.id,
    ]
    assert console._last_console_action.action_id == "delete"
    assert console._last_console_action.visible_copy == "Press Delete again to remove this message."


@pytest.mark.asyncio
async def test_console_selected_message_edit_action_opens_modal_and_saves_content():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-edit-{message.id}")

        await pilot.click(f"#console-message-action-edit-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-edit-message-modal")
        edit_modal = host.screen_stack[-1]
        assert "Editing existing transcript message" in _static_plain_text(
            edit_modal.query_one("#console-edit-message-context", Static)
        )

        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        assert editor.text == "answer"
        editor.text = "edited answer"
        await pilot.click("#console-edit-message-save")
        await pilot.pause()

    assert store.get_message(message.id).content == "edited answer"
    assert console._last_console_action.action_id == "edit"
    assert console._last_console_action.visible_copy == "Edited message."


@pytest.mark.asyncio
async def test_console_selected_message_edit_action_cancel_preserves_content():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-edit-{message.id}")

        await pilot.click(f"#console-message-action-edit-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-edit-message-modal")
        edit_modal = host.screen_stack[-1]
        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "discard this"
        await pilot.click("#console-edit-message-cancel")
        await pilot.pause()

    assert store.get_message(message.id).content == "answer"


@pytest.mark.asyncio
async def test_console_selected_message_edit_action_blank_save_stays_open_with_error():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-edit-{message.id}")

        await pilot.click(f"#console-message-action-edit-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-edit-message-modal")
        edit_modal = host.screen_stack[-1]
        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "   "
        await pilot.click("#console-edit-message-save")
        await _wait_for_selector(edit_modal, pilot, "#console-edit-message-error")

        assert "cannot be blank" in _static_plain_text(
            edit_modal.query_one("#console-edit-message-error", Static)
        ).lower()
        assert store.get_message(message.id).content == "answer"


@pytest.mark.asyncio
async def test_console_sync_skips_transcript_refresh_when_messages_unchanged(monkeypatch):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        original_refresh = transcript.refresh_messages
        refresh_calls = 0
        await pilot.pause()

        async def counted_refresh():
            nonlocal refresh_calls
            refresh_calls += 1
            await original_refresh()

        monkeypatch.setattr(transcript, "refresh_messages", counted_refresh)

        await console._sync_native_console_chat_ui()
        baseline_refresh_calls = refresh_calls
        assert baseline_refresh_calls >= 1

        await console._sync_native_console_chat_ui()
        assert refresh_calls == baseline_refresh_calls

        store.add_variant(message.id, "updated answer")
        await console._sync_native_console_chat_ui()
        assert refresh_calls == baseline_refresh_calls + 1


@pytest.mark.asyncio
async def test_console_selected_message_save_as_action_opens_modal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-save-as-{message.id}")

        await pilot.click(f"#console-message-action-save-as-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-save-as-modal")

    assert console._last_console_action.action_id == "save-as"


@pytest.mark.asyncio
async def test_console_save_as_modal_labels_unwired_destinations_as_wip():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-save-as-{message.id}")

        await pilot.click(f"#console-message-action-save-as-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-save-as-modal")
        save_as_modal = host.screen_stack[-1]

        assert "Saving selected Assistant message" in _static_plain_text(
            save_as_modal.query_one("#console-save-as-context", Static)
        )
        assert "answer" in _static_plain_text(
            save_as_modal.query_one("#console-save-as-excerpt", Static)
        )
        assert _static_plain_text(save_as_modal.query_one("#console-save-as-wip-chatbook", Static)).startswith(
            "Chatbook [WIP]"
        )
        assert "WIP: save as Chatbook is not wired yet." in _static_plain_text(
            save_as_modal.query_one("#console-save-as-wip-chatbook", Static)
        )
        assert "WIP: save as Media is not wired yet." in _static_plain_text(
            save_as_modal.query_one("#console-save-as-wip-media", Static)
        )
        assert "WIP: save as Prompt is not wired yet." in _static_plain_text(
            save_as_modal.query_one("#console-save-as-wip-prompt", Static)
        )


@pytest.mark.asyncio
async def test_console_selected_message_save_as_note_creates_note_from_message():
    app = _build_test_app()
    app.notes_scope_service = SimpleNamespace(
        save_note=AsyncMock(return_value={"id": "note-1", "title": "Console message", "content": "answer"})
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-save-as-{message.id}")

        await pilot.click(f"#console-message-action-save-as-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-save-as-destination-note")
        await pilot.click("#console-save-as-destination-note")
        await pilot.pause()

    app.notes_scope_service.save_note.assert_awaited_once_with(
        scope="local_note",
        title="Console message",
        content="answer",
        note_id=None,
        version=None,
        user_id="default_user",
        workspace_id=None,
        keywords=["console"],
    )
    assert console._last_console_action.action_id == "save-as-note"
    assert console._last_console_action.visible_copy == "Saved message as Note."


@pytest.mark.asyncio
async def test_console_failed_stream_renders_inline_retry_and_recovers():
    gateway = FailThenRecoverGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")

        store = console._ensure_console_chat_store()
        failed = store.messages_for_session(store.active_session_id)[-1]
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(failed.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-retry-{failed.id}")
        retry_button = console.query_one(f"#console-message-action-retry-{failed.id}", Button)
        assert str(retry_button.label) == "Try"
        assert retry_button.tooltip == "Retry the failed response."

        await pilot.click(f"#console-message-action-retry-{failed.id}")
        await _wait_for_text(console, pilot, "recovered")

    assert store.get_message(failed.id).status == "complete"


@pytest.mark.asyncio
async def test_console_continue_action_streams_new_message_from_selected_turn():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("hel", "lo"))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        source = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="seed",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(source.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-continue-{source.id}")

        await pilot.click(f"#console-message-action-continue-{source.id}")
        await _wait_for_text(console, pilot, "hello")

        messages = store.messages_for_session(session.id)
        assert messages[-1].role is ConsoleMessageRole.ASSISTANT
        assert messages[-1].content == "hello"
        assert messages[-1].id != source.id
        assert transcript.selected_message_id is None
        assert not list(console.query(f"#console-message-actions-{source.id}"))


@pytest.mark.asyncio
async def test_console_regenerate_action_streams_selected_variant():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=("hel", "lo"))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        source = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="seed",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(source.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-regenerate-{source.id}")

        await pilot.click(f"#console-message-action-regenerate-{source.id}")
        await _wait_for_text(console, pilot, "hello")

        updated = store.get_message(source.id)
        assert updated.variants.current.content == "hello"
        assert updated.variants.can_go_previous is True


@pytest.mark.asyncio
async def test_console_regenerated_message_variant_controls_cycle_visible_content():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        source = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="seed",
        )
        store.add_variant(source.id, "updated answer")
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(source.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-action-variant-previous-{source.id}")
        await _wait_for_selector(console, pilot, f"#console-message-action-variant-next-{source.id}")

        previous_button = console.query_one(f"#console-message-action-variant-previous-{source.id}", Button)
        next_button = console.query_one(f"#console-message-action-variant-next-{source.id}", Button)
        assert previous_button.disabled is False
        assert next_button.disabled is True
        assert "updated answer" in _static_plain_text(console.query_one(f"#console-message-{source.id}", Static))

        await pilot.click(f"#console-message-action-variant-previous-{source.id}")
        await _wait_for_text(console, pilot, "seed")
        previous_button = console.query_one(f"#console-message-action-variant-previous-{source.id}", Button)
        next_button = console.query_one(f"#console-message-action-variant-next-{source.id}", Button)
        assert previous_button.disabled is True
        assert next_button.disabled is False

        await pilot.click(f"#console-message-action-variant-next-{source.id}")
        await _wait_for_text(console, pilot, "updated answer")
        previous_button = console.query_one(f"#console-message-action-variant-previous-{source.id}", Button)
        next_button = console.query_one(f"#console-message-action-variant-next-{source.id}", Button)
        assert previous_button.disabled is False
        assert next_button.disabled is True


@pytest.mark.asyncio
async def test_console_native_tab_strip_creates_and_switches_sessions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")
        assert "Chat 2" in _visible_text(console)

        await pilot.click(f"#console-session-tab-{first.id}")

        assert store.active_session_id == first.id
        assert "Chat 1" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_native_tab_switch_restores_transcript_messages():
    """Verify native tab switching restores the prior session transcript."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        store.append_message(
            first.id,
            role=ConsoleMessageRole.USER,
            content="first tab user prompt",
        )
        store.append_message(
            first.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="first tab assistant reply",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_text(console, pilot, "first tab assistant reply")

        previous = store.active_session_id
        await pilot.click("#console-new-chat-tab")
        second = await _wait_for_active_session_change(store, pilot, previous)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")
        await _wait_for_text(console, pilot, "Start Console")
        assert "first tab assistant reply" not in _visible_text(console)

        await pilot.click(f"#console-session-tab-{first.id}")

        await _wait_for_active_session(store, pilot, first.id)
        await _wait_for_text(console, pilot, "first tab user prompt")
        await _wait_for_text(console, pilot, "first tab assistant reply")


@pytest.mark.asyncio
async def test_console_workspace_conversation_switch_restores_transcript_messages():
    """Verify workspace conversation switching restores the prior transcript."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        store.append_message(
            first.id,
            role=ConsoleMessageRole.USER,
            content="workspace row user prompt",
        )
        store.append_message(
            first.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="workspace row assistant reply",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_text(console, pilot, "workspace row assistant reply")

        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        previous = store.active_session_id
        await pilot.click("#console-new-chat-tab")
        second = await _wait_for_active_session_change(store, pilot, previous)
        assert second != first.id
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        await _wait_for_text(console, pilot, "Start Console")
        assert "workspace row assistant reply" not in _visible_text(console)

        await _click_console_workspace_conversation_for_session(
            console,
            pilot,
            store,
            first.id,
        )

        await _wait_for_active_session(store, pilot, first.id)
        await _wait_for_text(console, pilot, "workspace row user prompt")
        await _wait_for_text(console, pilot, "workspace row assistant reply")


def _configure_grouped_browser_workspaces(app):
    app.app_config.setdefault("console", {}).setdefault("conversation_browser", {})[
        "collapsed_groups"
    ] = {}
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    return service


def _browser_group_toggle(console, group_id: str) -> Button:
    for button in console.query(Button):
        if getattr(button, "group_id", None) == group_id:
            return button
    groups = [
        (getattr(button, "id", None), getattr(button, "group_id", None))
        for button in console.query(Button)
        if str(getattr(button, "id", "")).startswith("console-conversation-browser")
    ]
    raise AssertionError(f"Browser group toggle {group_id!r} not found. Groups: {groups!r}")


def _browser_star_button(console, conversation_id: str) -> Button:
    for button in console.query(".console-conversation-star"):
        if getattr(button, "conversation_id", None) == conversation_id:
            return button
    stars = [
        (getattr(button, "id", None), getattr(button, "conversation_id", None))
        for button in console.query(".console-conversation-star")
    ]
    raise AssertionError(f"Star button for {conversation_id!r} not found. Stars: {stars!r}")


async def _wait_for_browser_conversation_row(console, pilot, conversation_id: str):
    for _ in range(80):
        for row in console.query(".console-workspace-conversation-row"):
            if getattr(row, "conversation_id", None) == conversation_id:
                return row
        await pilot.pause(0.05)
    rows = [
        (getattr(row, "conversation_id", None), _widget_text(row))
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(f"Browser row {conversation_id!r} not found. Rows: {rows!r}")


class _InputChangedEvent:
    def __init__(self, value: str) -> None:
        self.value = value

    def stop(self) -> None:
        return None


async def _set_console_conversation_browser_search(console, pilot, query: str) -> None:
    search = console.query_one("#console-workspace-conversation-search", Input)
    search.value = query
    console.on_console_workspace_conversation_search_changed(_InputChangedEvent(query))
    await pilot.pause(0.3)


@pytest.mark.asyncio
async def test_console_conversation_browser_lists_all_workspace_groups():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="workspace-a-chat",
        role="workspace-thread",
        title="Workspace A saved",
    )
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="workspace-b-chat",
        role="workspace-thread",
        title="Workspace B saved",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        store = console._ensure_console_chat_store()
        default_session = store.create_session(
            title="Global chat",
            workspace_id=DEFAULT_WORKSPACE_ID,
        )
        store.switch_session(default_session.id)
        await console._sync_native_console_chat_ui()

        visible_text = _visible_text(console)
        assert "Starred" in visible_text
        assert "Workspaces" in visible_text
        assert "Workspace A" in visible_text
        assert "Workspace B" in visible_text
        assert "Chats" in visible_text
        assert "Global chat" in visible_text
        assert "Storage" in visible_text
        assert "Server handoff" in visible_text
        assert len(console.query("#console-workspace-conversations-toggle")) == 0
        assert len(console.query("#console-conversation-browser-section-toggle-workspaces")) == 1
        assert any(
            getattr(button, "group_id", None) == "workspace:ws-a"
            for button in console.query(".console-workspace-conversations-toggle")
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_search_filters_all_groups():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="alpha-a",
        role="workspace-thread",
        title="Alpha in Workspace A",
    )
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="needle-b",
        role="workspace-thread",
        title="Needle in Workspace B",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await pilot.click("#console-workspace-conversation-search")
        search = console.query_one("#console-workspace-conversation-search", Input)
        search.value = "needle"
        console.on_console_workspace_conversation_search_changed(_InputChangedEvent("needle"))
        for _ in range(40):
            row_texts = _console_workspace_conversation_texts(console)
            if any(getattr(row, "conversation_id", None) == "needle-b" for row in console.query(".console-workspace-conversation-row")):
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(f"Needle row not found. Rows: {row_texts!r}")
        row_texts = _console_workspace_conversation_texts(console)

        assert any("Needle in Workspa" in text for text in row_texts)
        assert any("Workspace B" in text for text in row_texts)
        assert all("Alpha in Workspace A" not in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_browser_selecting_non_default_workspace_native_session_switches_active_workspace():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    app.app_config["console"]["conversation_browser"]["collapsed_groups"][
        "workspace:ws-b"
    ] = False
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Workspace A Chat", workspace_id="ws-a")
        second = store.create_session(title="Workspace B Chat", workspace_id="ws-b")
        store.append_message(
            second.id,
            role=ConsoleMessageRole.USER,
            content="Workspace B prompt",
        )
        store.switch_session(first.id)
        await console._sync_native_console_chat_ui()

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            f"native:{second.id}",
        )
        await _wait_for_active_session(store, pilot, second.id)
        await _wait_for_text(console, pilot, "Workspace B prompt")

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == "ws-b"
        assert store.workspace_context.active_workspace_id == "ws-b"


@pytest.mark.asyncio
async def test_console_browser_selecting_non_default_workspace_persisted_row_switches_active_workspace_before_resume():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    app.app_config["console"]["conversation_browser"]["collapsed_groups"][
        "workspace:ws-b"
    ] = False
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="persisted-ws-b",
        role="workspace-thread",
        title="Workspace B saved",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "persisted-ws-b": {
                "conversation": {
                    "id": "persisted-ws-b",
                    "title": "Workspace B saved",
                },
                "root_threads": [
                    {
                        "id": "message-ws-b",
                        "role": "user",
                        "content": "Workspace B prompt",
                    }
                ],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        store = console._ensure_console_chat_store()

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "persisted-ws-b",
        )
        await _wait_for_text(console, pilot, "Workspace B prompt")
        session = next(
            session
            for session in store.sessions()
            if session.persisted_conversation_id == "persisted-ws-b"
        )

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == "ws-b"
        assert store.workspace_context.active_workspace_id == "ws-b"
        assert session.workspace_id == "ws-b"


@pytest.mark.asyncio
async def test_console_browser_selecting_duplicate_membership_row_ignores_other_workspace_open_session():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    app.app_config["console"]["conversation_browser"]["collapsed_groups"][
        "workspace:ws-b"
    ] = False
    for workspace_id in ("ws-a", "ws-b"):
        service.link_membership(
            workspace_id,
            item_type="conversation",
            item_id="shared-open-chat",
            role="workspace-thread",
            title="Shared open chat",
        )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "shared-open-chat": {
                "conversation": {
                    "id": "shared-open-chat",
                    "title": "Shared open chat",
                },
                "root_threads": [
                    {
                        "id": "shared-open-message",
                        "role": "user",
                        "content": "Workspace B shared prompt",
                    }
                ],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        store = console._ensure_console_chat_store()
        open_ws_a = store.ensure_session(title="Shared open chat", workspace_id="ws-a")
        open_ws_a.persisted_conversation_id = "shared-open-chat"
        store.switch_session(open_ws_a.id)
        await console._sync_native_console_chat_ui()
        ws_b_row = _workspace_conversation_row_by_key(
            console,
            "workspace:ws-b:conversation:shared-open-chat",
        )

        assert ws_b_row is not None
        assert not _widget_text(ws_b_row).startswith("> ")
        assert (
            console._find_console_browser_row(
                "workspace:missing:conversation:shared-open-chat",
                conversation_id="shared-open-chat",
            )
            is None
        )

        await _click_console_workspace_conversation_for_row_key(
            console,
            pilot,
            "workspace:ws-b:conversation:shared-open-chat",
        )
        await _wait_for_text(console, pilot, "Workspace B shared prompt")
        sessions = [
            session
            for session in store.sessions()
            if session.persisted_conversation_id == "shared-open-chat"
        ]
        active_session = store.switch_session(store.active_session_id)

        assert len(sessions) == 2
        assert active_session.workspace_id == "ws-b"
        selected_shared_rows = [
            row
            for row in console.query(".console-workspace-conversation-row")
            if getattr(row, "conversation_id", None) == "shared-open-chat"
            and _widget_text(row).startswith("> ")
        ]
        assert len(selected_shared_rows) == 1
        assert getattr(selected_shared_rows[0], "native_session_id", None) == active_session.id
        selected_native_rows = [
            row
            for row in console._native_console_browser_rows("shared-open-chat")
            if row.conversation_id == "shared-open-chat" and row.selected
        ]
        assert len(selected_native_rows) == 1
        assert selected_native_rows[0].native_session_id == active_session.id
        selected_membership_rows = [
            row
            for row in console._membership_console_browser_rows("shared-open-chat")
            if row.conversation_id == "shared-open-chat" and row.selected
        ]
        assert len(selected_membership_rows) == 1
        assert selected_membership_rows[0].workspace_id == "ws-b"
        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == "ws-b"
        assert store.workspace_context.active_workspace_id == "ws-b"


@pytest.mark.asyncio
async def test_console_browser_selecting_default_native_session_switches_to_default_and_keeps_file_tools_disabled():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Workspace A Chat", workspace_id="ws-a")
        second = store.create_session(
            title="Default Chat",
            workspace_id=DEFAULT_WORKSPACE_ID,
        )
        store.append_message(
            second.id,
            role=ConsoleMessageRole.USER,
            content="Default prompt",
        )
        store.switch_session(first.id)
        await console._sync_native_console_chat_ui()

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            f"native:{second.id}",
        )
        await _wait_for_active_session(store, pilot, second.id)
        await _wait_for_text(console, pilot, "Default prompt")

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == DEFAULT_WORKSPACE_ID
        assert store.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID
        assert _static_plain_text(
            console.query_one("#console-workspace-runtime-label", Static)
        ) == "File tools"
        assert "Off in Default workspace" in _static_plain_text(
            console.query_one("#console-workspace-runtime-value", Static)
        )


@pytest.mark.asyncio
async def test_console_browser_selecting_default_persisted_row_switches_to_default_and_keeps_file_tools_disabled():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    service.link_membership(
        DEFAULT_WORKSPACE_ID,
        item_type="conversation",
        item_id="persisted-default",
        role="workspace-thread",
        title="Default saved",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "persisted-default": {
                "conversation": {
                    "id": "persisted-default",
                    "title": "Default saved",
                },
                "root_threads": [
                    {
                        "id": "message-default",
                        "role": "user",
                        "content": "Default persisted prompt",
                    }
                ],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        store = console._ensure_console_chat_store()

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "persisted-default",
        )
        await _wait_for_text(console, pilot, "Default persisted prompt")
        session = next(
            session
            for session in store.sessions()
            if session.persisted_conversation_id == "persisted-default"
        )

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == DEFAULT_WORKSPACE_ID
        assert store.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID
        assert session.workspace_id == DEFAULT_WORKSPACE_ID
        assert _static_plain_text(
            console.query_one("#console-workspace-runtime-label", Static)
        ) == "File tools"
        assert "Off in Default workspace" in _static_plain_text(
            console.query_one("#console-workspace-runtime-value", Static)
        )


@pytest.mark.asyncio
async def test_console_browser_selecting_global_persisted_row_preserves_active_workspace():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    app.chat_conversation_scope_service = SyncSearchableConversationService(
        {
            "global-persisted": {
                "conversation": {
                    "id": "global-persisted",
                    "title": "Global saved",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [
                    {
                        "id": "message-global",
                        "role": "user",
                        "content": "Global persisted prompt",
                    }
                ],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        store = console._ensure_console_chat_store()
        before = service.get_active_workspace()
        assert before is not None
        assert before.workspace_id == "ws-a"

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "global-persisted",
        )
        await _wait_for_text(console, pilot, "Global persisted prompt")
        session = next(
            session
            for session in store.sessions()
            if session.persisted_conversation_id == "global-persisted"
        )

        after = service.get_active_workspace()
        assert after is not None
        assert after.workspace_id == "ws-a"
        assert store.workspace_context.active_workspace_id == "ws-a"
        assert session.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID


@pytest.mark.asyncio
async def test_console_conversation_browser_search_counts_only_matching_local_rows():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = None
    app.local_chat_conversation_service = None
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="needle-local",
        role="workspace-thread",
        title="Needle Local Match",
    )
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="other-local",
        role="workspace-thread",
        title="Other Local Row",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await _set_console_conversation_browser_search(console, pilot, "needle")

        for _ in range(80):
            status = console.query_one(
                "#console-workspace-conversation-search-status",
                Static,
            )
            if _static_plain_text(status):
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError("Conversation browser search status did not render")

        row_texts = _console_workspace_conversation_texts(console)
        assert _static_plain_text(status) == "1 match"
        assert any("Needle Local" in text for text in row_texts)
        assert all("Other Local Row" not in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_conversation_browser_keeps_multi_workspace_memberships():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    app.app_config["console"]["conversation_browser"]["collapsed_groups"][
        "workspace:ws-b"
    ] = False
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="shared-conversation",
        role="workspace-thread",
        title="Shared Conversation",
    )
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="shared-conversation",
        role="workspace-thread",
        title="Shared Conversation",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")

        rows = [
            row
            for row in console.query(".console-workspace-conversation-row")
            if getattr(row, "conversation_id", None) == "shared-conversation"
        ]
        stars = [
            button
            for button in console.query(".console-conversation-star")
            if getattr(button, "conversation_id", None) == "shared-conversation"
        ]

        assert len(rows) == 2
        assert {
            getattr(row, "workspace_id", None)
            for row in rows
        } == {"ws-a", "ws-b"}
        assert {
            getattr(row, "row_key", None)
            for row in rows
        } == {
            "workspace:ws-a:conversation:shared-conversation",
            "workspace:ws-b:conversation:shared-conversation",
        }
        assert len(stars) == 2
        assert {getattr(button, "row_key", None) for button in stars} == {
            "workspace:ws-a:conversation:shared-conversation",
            "workspace:ws-b:conversation:shared-conversation",
        }


@pytest.mark.asyncio
async def test_console_conversation_browser_dedupes_membership_and_persisted_same_workspace():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="same-workspace-conversation",
        role="workspace-thread",
        title="Membership Title",
    )
    app.local_chat_conversation_service = SyncSearchableConversationService(
        {
            "same-workspace-conversation": {
                "conversation": {
                    "id": "same-workspace-conversation",
                    "title": "Persisted Title",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")

        rows = [
            row
            for row in console.query(".console-workspace-conversation-row")
            if (
                getattr(row, "conversation_id", None)
                == "same-workspace-conversation"
                and getattr(row, "workspace_id", None) == "ws-a"
            )
        ]
        stars = [
            button
            for button in console.query(".console-conversation-star")
            if (
                getattr(button, "conversation_id", None)
                == "same-workspace-conversation"
                and getattr(button, "row_key", None)
                == "workspace:ws-a:conversation:same-workspace-conversation"
            )
        ]

        assert len(rows) == 1
        assert getattr(rows[0], "row_key", None) == (
            "workspace:ws-a:conversation:same-workspace-conversation"
        )
        assert "Membership Title" in _widget_text(rows[0])
        assert len(stars) == 1


@pytest.mark.asyncio
async def test_console_conversation_browser_search_ignores_stale_results():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = SlowFirstSearchableConversationService(
        {
            "stale-alpha": {
                "conversation": {
                    "id": "stale-alpha",
                    "title": "Stale Alpha",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
            "fresh-beta": {
                "conversation": {
                    "id": "fresh-beta",
                    "title": "Fresh Beta",
                    "workspace_id": "ws-b",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        console._console_conversation_browser_query = "alpha"
        console._console_conversation_browser_search_token += 1
        stale_token = console._console_conversation_browser_search_token
        stale_task = asyncio.create_task(
            console._refresh_console_conversation_browser_search("alpha", stale_token)
        )
        for _ in range(40):
            if app.chat_conversation_scope_service.started.is_set():
                break
            await pilot.pause(0.05)
        assert app.chat_conversation_scope_service.started.is_set()

        console._console_conversation_browser_query = "beta"
        console._console_conversation_browser_search_token += 1
        fresh_token = console._console_conversation_browser_search_token
        app.chat_conversation_scope_service.release.set()
        await stale_task
        await console._refresh_console_conversation_browser_search("beta", fresh_token)
        row_texts = await _wait_for_workspace_conversation_text(console, pilot, "Fresh Beta")
        assert all("Stale Alpha" not in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_conversation_browser_group_collapse_persists_locally():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="collapse-chat",
        role="workspace-thread",
        title="Collapse Target",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        assert "Collapse Target" in _visible_text(console)

        _browser_group_toggle(console, "workspace:ws-a").press()
        await pilot.pause(0.1)

        assert all(
            "Collapse Target" not in text
            for text in _console_workspace_conversation_texts(console)
        )
        collapsed_groups = app.app_config["console"]["conversation_browser"][
            "collapsed_groups"
        ]
        assert collapsed_groups["workspace:ws-a"] is True

        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert all(
            "Collapse Target" not in text
            for text in _console_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_workspaces_section_collapse_persists_locally():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="section-collapse-chat",
        role="workspace-thread",
        title="Section Collapse Target",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        assert any(
            "Section Collapse" in text
            for text in _console_workspace_conversation_texts(console)
        )

        _browser_group_toggle(console, "section:workspaces").press()
        await pilot.pause(0.1)

        assert all(
            "Section Collapse" not in text
            for text in _console_workspace_conversation_texts(console)
        )
        collapsed_groups = app.app_config["console"]["conversation_browser"][
            "collapsed_groups"
        ]
        assert collapsed_groups["section:workspaces"] is True

        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert all(
            "Section Collapse" not in text
            for text in _console_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_starred_section_updates_from_row_action():
    app = _build_test_app()
    marks = FakeConversationLocalMarksService()
    app.conversation_local_marks_service = marks
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="star-target",
        role="workspace-thread",
        title="Star Target",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        _browser_star_button(console, "star-target").press()
        await pilot.pause(0.1)

        assert marks.is_starred("star-target") is True
        rows = [
            _widget_text(row)
            for row in console.query(".console-workspace-conversation-row")
            if getattr(row, "conversation_id", None) == "star-target"
        ]
        assert len(rows) >= 2
        assert any("Star Target" in row for row in rows)


@pytest.mark.asyncio
async def test_console_conversation_browser_keeps_starred_row_in_normal_group():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService(
        ("starred-normal",)
    )
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="starred-normal",
        role="workspace-thread",
        title="Starred Normal",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        rows = [
            _widget_text(row)
            for row in console.query(".console-workspace-conversation-row")
            if getattr(row, "conversation_id", None) == "starred-normal"
        ]
        assert len(rows) == 2
        assert all("Starred Normal" in row for row in rows)


@pytest.mark.asyncio
async def test_console_conversation_browser_marks_unavailable_keeps_browsing_enabled():
    app = _build_test_app()
    app.conversation_local_marks_service = None
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="marks-unavailable-chat",
        role="workspace-thread",
        title="Marks Unavailable Chat",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")

        visible_text = _visible_text(console)
        assert "Starred" in visible_text
        assert "Workspaces" in visible_text
        assert "Chats" in visible_text
        assert "Local stars unavailable" in visible_text
        assert "Marks Unavailable Chat" in visible_text
        star = _browser_star_button(console, "marks-unavailable-chat")
        assert star.disabled is True


@pytest.mark.asyncio
async def test_console_conversation_browser_default_includes_sync_persisted_rows():
    app = _build_test_app()
    app.conversation_local_marks_service = None
    service = _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = SyncSearchableConversationService(
        {
            "global-default": {
                "conversation": {
                    "id": "global-default",
                    "title": "Global persisted default",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
            "workspace-default": {
                "conversation": {
                    "id": "workspace-default",
                    "title": "Workspace A persisted default",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        global_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "global-default",
        )
        workspace_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "workspace-default",
        )

        visible_text = _visible_text(console)
        assert "Starred" in visible_text
        assert "Workspaces" in visible_text
        assert "Chats" in visible_text
        assert "Global persisted" in _widget_text(global_row)
        assert "Workspace A persi" in _widget_text(workspace_row)
        assert any(
            call.get("scope_type") == "global"
            for call in app.chat_conversation_scope_service.list_calls
        )
        assert any(
            call.get("scope_type") == "workspace" and call.get("workspace_id") == "ws-b"
            for call in app.chat_conversation_scope_service.list_calls
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_default_prefers_sync_local_service():
    app = _build_test_app()
    app.conversation_local_marks_service = None
    _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "async-scope-default": {
                "conversation": {
                    "id": "async-scope-default",
                    "title": "Async scope default should not block local rows",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
        }
    )
    app.local_chat_conversation_service = SyncSearchableConversationService(
        {
            "local-global-default": {
                "conversation": {
                    "id": "local-global-default",
                    "title": "Local global default",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
            "local-workspace-default": {
                "conversation": {
                    "id": "local-workspace-default",
                    "title": "Local Workspace A default",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")

        global_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-global-default",
        )
        workspace_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-workspace-default",
        )

        assert "Local global default" in _widget_text(global_row)
        assert "Local Workspace" in _widget_text(workspace_row)
        assert any(
            call.get("scope_type") == "global"
            for call in app.local_chat_conversation_service.list_calls
        )
        assert any(
            call.get("scope_type") == "workspace" and call.get("workspace_id") == "ws-a"
            for call in app.local_chat_conversation_service.list_calls
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_default_omits_mode_for_local_service():
    app = _build_test_app()
    app.conversation_local_marks_service = None
    _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "async-scope-default": {
                "conversation": {
                    "id": "async-scope-default",
                    "title": "Async scope fallback",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
        }
    )
    app.local_chat_conversation_service = NoModeSyncSearchableConversationService(
        {
            "local-no-mode-global": {
                "conversation": {
                    "id": "local-no-mode-global",
                    "title": "No mode local global",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
            "local-no-mode-workspace": {
                "conversation": {
                    "id": "local-no-mode-workspace",
                    "title": "No mode Workspace A",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")

        global_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-no-mode-global",
        )
        workspace_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-no-mode-workspace",
        )

        assert "No mode local global" in _widget_text(global_row)
        assert "No mode Workspace" in _widget_text(workspace_row)
        assert all(
            "mode" not in call
            for call in app.local_chat_conversation_service.list_calls
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_search_omits_mode_for_local_service():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = None
    app.local_chat_conversation_service = NoModeSyncSearchableConversationService(
        {
            "local-search-global": {
                "conversation": {
                    "id": "local-search-global",
                    "title": "Needle local global",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
            "local-search-workspace": {
                "conversation": {
                    "id": "local-search-workspace",
                    "title": "Needle local Workspace A",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await _set_console_conversation_browser_search(console, pilot, "needle")

        global_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-search-global",
        )
        workspace_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-search-workspace",
        )

        assert "Needle local global" in _widget_text(global_row)
        assert "Needle local Wor" in _widget_text(workspace_row)
        for _ in range(80):
            if console._console_conversation_browser_total is not None:
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError("Debounced persisted search did not finish")
        cached_row_ids = {
            row.conversation_id for row in console._console_conversation_browser_rows
        }
        assert console._console_conversation_browser_error == ""
        assert "local-search-global" in cached_row_ids
        assert "local-search-workspace" in cached_row_ids
        assert all(
            "mode" not in call
            for call in app.local_chat_conversation_service.list_calls
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_long_list_keeps_readiness_rows_reachable():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    for index in range(30):
        service.link_membership(
            "ws-a",
            item_type="conversation",
            item_id=f"long-chat-{index}",
            role="workspace-thread",
            title=f"Long Chat {index:02d}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        conversation_list = console.query_one("#console-workspace-conversations")
        server_label = console.query_one("#console-workspace-server-readiness-label")
        server_value = console.query_one("#console-workspace-server-readiness-value")

        assert conversation_list.region.height > 0
        assert server_label.region.y > conversation_list.region.y
        assert server_value.region.y >= server_label.region.y
        visible_text = _visible_text(console)
        assert "Storage" in visible_text
        assert "Server handoff" in visible_text


@pytest.mark.asyncio
async def test_console_new_chat_tab_appears_in_workspace_conversation_rail():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-chat-1",
        role="workspace-thread",
        title="Chat 1",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        first.persisted_conversation_id = "persisted-chat-1"
        await console._sync_native_console_chat_ui()

        assert any("Chat 1" in text for text in _console_workspace_conversation_texts(console))

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        row_texts = _console_workspace_conversation_texts(console)
        assert any("Chat 1" in text for text in row_texts)
        assert any("Chat 2" in text for text in row_texts)
        assert any(text.startswith("> ") and "Chat 2" in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_includes_all_workspace_persisted_results():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    other_workspace = service.create_workspace(
        workspace_id="ws-other-search",
        name="Other Search",
    )
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "persisted-alpha": {
                "conversation": {
                    "id": "persisted-alpha",
                    "title": "Alpha persisted conversation",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [],
            },
            "other-alpha": {
                "conversation": {
                    "id": "other-alpha",
                    "title": "Alpha other workspace",
                    "workspace_id": other_workspace.workspace_id,
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )

        await _set_console_conversation_browser_search(console, pilot, "alpha")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Alpha persisted",
            selected=False,
        )
        row_texts = _console_workspace_conversation_texts(console)
        assert any("Alpha persisted" in text for text in row_texts)
        assert any(
            getattr(row, "conversation_id", None) == "other-alpha"
            for row in console.query(".console-workspace-conversation-row")
        )
        workspace_calls = [
            call
            for call in app.chat_conversation_scope_service.list_calls
            if call.get("scope_type") == "workspace"
        ]
        assert any(
            call.get("workspace_id") == active_workspace.workspace_id
            for call in workspace_calls
        )
        assert any(
            call.get("workspace_id") == other_workspace.workspace_id
            for call in workspace_calls
        )


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_selection_keeps_query_active():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "select-alpha": {
                "conversation": {
                    "id": "select-alpha",
                    "title": "Select Alpha",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [
                    {
                        "id": "select-alpha-message",
                        "conversation_id": "select-alpha",
                        "role": "user",
                        "sender": "user",
                        "content": "selected alpha prompt",
                        "children": [],
                    }
                ],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await _set_console_conversation_browser_search(console, pilot, "alpha")
        await _wait_for_workspace_conversation_text(console, pilot, "Select Alpha", selected=False)

        await _click_console_workspace_conversation_for_id(console, pilot, "select-alpha")

        await _wait_for_text(console, pilot, "selected alpha prompt")
        search = console.query_one("#console-workspace-conversation-search", Input)
        assert search.value == "alpha"
        assert "Select Alpha" in _static_plain_text(console.query_one("#console-workspace-selected-conversation", Static))


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_blank_selection_keeps_composer_focus():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="blank-focus-chat",
        role="workspace-thread",
        title="Blank focus chat",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "blank-focus-chat": {
                "conversation": {
                    "id": "blank-focus-chat",
                    "title": "Blank focus chat",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [
                    {
                        "id": "blank-focus-message",
                        "conversation_id": "blank-focus-chat",
                        "role": "user",
                        "sender": "user",
                        "content": "blank focus prompt",
                        "children": [],
                    }
                ],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Blank focus chat",
            selected=False,
        )

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "blank-focus-chat",
        )

        await _wait_for_text(console, pilot, "blank focus prompt")
        await pilot.pause(0.2)
        search = console.query_one("#console-workspace-conversation-search", Input)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        assert console.app.focused is composer
        assert console.app.focused is not search


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_selection_invalidates_pending_worker():
    app = _build_test_app()
    app.chat_conversation_scope_service = SlowFirstSearchableConversationService({})
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Slow Alpha")
        first.title = "Slow Alpha"
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")
        second_session = store.switch_session(second)
        first.workspace_id = second_session.workspace_id
        await console._sync_native_console_chat_ui()

        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await _set_console_conversation_browser_search(console, pilot, "alpha")
        await _wait_for_workspace_conversation_text(console, pilot, "Slow Alpha", selected=False)
        for _ in range(40):
            if app.chat_conversation_scope_service.started.is_set():
                break
            await pilot.pause(0.05)
        assert app.chat_conversation_scope_service.started.is_set()
        stale_token = console._console_workspace_conversation_search_token

        await _click_console_workspace_conversation_for_session(
            console,
            pilot,
            store,
            first.id,
        )

        assert console._console_workspace_conversation_search_token > stale_token
        app.chat_conversation_scope_service.release.set()
        await pilot.pause(0.5)
        assert "Slow Alpha" in _static_plain_text(
            console.query_one("#console-workspace-selected-conversation", Static)
        )
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Slow Alpha",
            selected=True,
        )
        assert any(text.startswith("> ") and "Slow Alpha" in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_workspace_switch_clears_conversation_search_and_restores_collapse_preference():
    app = _build_test_app()
    service = app.workspace_registry_service
    workspace_a = service.get_active_workspace()
    workspace_b = service.create_workspace(workspace_id="ws-search-reset", name="Search Reset")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        await _set_console_conversation_browser_search(console, pilot, "alpha")
        _browser_group_toggle(console, "section:chats").press()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversation-search")) == 1

        service.set_active_workspace(workspace_b.workspace_id)
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversation-search")) == 1
        assert console.query_one("#console-workspace-conversation-search", Input).value == "alpha"

        service.set_active_workspace(workspace_a.workspace_id)
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversation-search")) == 1


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_shows_cap_and_empty_copy():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    conversations = {
        f"topic-{index}": {
            "conversation": {
                "id": f"topic-{index}",
                "title": f"Topic conversation {index:02d}",
                "workspace_id": active_workspace.workspace_id,
            },
            "root_threads": [],
        }
        for index in range(60)
    }
    app.chat_conversation_scope_service = SearchableConversationService(conversations)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )

        await _set_console_conversation_browser_search(console, pilot, "topic")
        await _wait_for_text(console, pilot, "60 matches")
        await _wait_for_text(console, pilot, "Showing")

        search = console.query_one("#console-workspace-conversation-search", Input)
        await _set_console_conversation_browser_search(console, pilot, "missing")
        await _wait_for_text(console, pilot, "No workspace conversations.")


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_ignores_stale_workspace_results():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.create_workspace(workspace_id="ws-stale-b", name="Stale B")
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "stale-a": {
                "conversation": {
                    "id": "stale-a",
                    "title": "Stale Alpha",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )
        console._console_workspace_conversation_query = "Alpha"
        stale_token = console._console_workspace_conversation_search_token + 1
        console._console_workspace_conversation_search_token = stale_token
        service.set_active_workspace("ws-stale-b")
        await console._refresh_console_workspace_conversation_search(
            active_workspace.workspace_id,
            "Alpha",
            stale_token,
        )
        assert "Stale Alpha" not in _visible_text(console)


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_blank_query_clears_error_cache():
    app = _build_test_app()
    app.chat_conversation_scope_service = FailingSearchConversationService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )

        await _set_console_conversation_browser_search(console, pilot, "fail")
        await _wait_for_text(
            console,
            pilot,
            "Workspace conversation search is unavailable.",
        )

        search = console.query_one("#console-workspace-conversation-search", Input)
        await _set_console_conversation_browser_search(console, pilot, "")

        assert (
            "Workspace conversation search is unavailable."
            not in _visible_text(console)
        )
        assert console._console_workspace_conversation_search_rows == ()
        assert console._console_workspace_conversation_search_total is None
        assert console._console_workspace_conversation_search_error == ""


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_shows_local_rows_before_slow_persisted_search():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="member-alpha",
        role="workspace-thread",
        title="Alpha membership conversation",
    )
    app.chat_conversation_scope_service = SlowSearchConversationService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )

        await _set_console_conversation_browser_search(console, pilot, "alpha")
        try:
            for _ in range(40):
                if app.chat_conversation_scope_service.started.is_set():
                    break
                await pilot.pause(0.05)
            assert app.chat_conversation_scope_service.started.is_set()

            assert "1 match" in _visible_text(console)
            row_texts = _console_workspace_conversation_texts(console)
            assert any("Alpha membership" in text for text in row_texts)
        finally:
            app.chat_conversation_scope_service.release.set()
            await pilot.pause(0.2)


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_filters_all_workspace_memberships():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    other_workspace = service.create_workspace(workspace_id="ws-other-search", name="Other Search")
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="member-alpha",
        role="workspace-thread",
        title="Alpha membership conversation",
    )
    service.link_membership(
        other_workspace.workspace_id,
        item_type="conversation",
        item_id="member-other-alpha",
        role="workspace-thread",
        title="Alpha other workspace",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")

        await _set_console_conversation_browser_search(console, pilot, "alpha")
        await _wait_for_text(console, pilot, "matches")
        await _wait_for_workspace_conversation_text(console, pilot, "Alpha membership", selected=False)
        row_texts = _console_workspace_conversation_texts(console)
        assert any("Alpha membership" in text for text in row_texts)
        assert any(
            getattr(row, "conversation_id", None) == "member-other-alpha"
            for row in console.query(".console-workspace-conversation-row")
        )
        assert "matches" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_uses_current_workspace_context():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-search-a", name="Search A")
    service.create_workspace(workspace_id="ws-search-b", name="Search B")
    service.set_active_workspace("ws-search-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        store = console._ensure_console_chat_store()
        assert store.workspace_context.active_workspace_id == "ws-search-a"

        service.set_active_workspace("ws-search-b")

        assert (
            console._active_console_workspace_id_for_conversation_search()
            == "ws-search-b"
        )


@pytest.mark.asyncio
async def test_console_workspace_conversation_list_reserves_two_line_rows_with_margin():
    """Verify conversation list height accounts for two-line rows plus margin."""
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    for index in range(3):
        service.link_membership(
            active_workspace.workspace_id,
            item_type="conversation",
            item_id=f"saved-chat-{index}",
            role="workspace-thread",
            title=f"Saved Chat {index}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        conversation_list = console.query_one("#console-workspace-conversations")
        assert conversation_list.styles.height.value >= 9


@pytest.mark.asyncio
async def test_console_new_chat_tab_promotes_active_native_session_in_workspace_rail():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    for index in range(5):
        service.link_membership(
            active_workspace.workspace_id,
            item_type="conversation",
            item_id=f"persisted-chat-{index}",
            role="workspace-thread",
            title=f"Older chat {index + 1}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert "Chat 2" in row_texts[0]
        assert row_texts[0].startswith("> ")


@pytest.mark.asyncio
async def test_console_workspace_new_conversation_button_is_not_under_composer():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.1)

        button = console.query_one("#console-new-chat-tab", Button)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        hit_x = button.region.x + max(0, button.region.width // 2)
        hit_y = button.region.y + max(0, button.region.height // 2)
        hit_widget, _region = console.get_widget_at(hit_x, hit_y)

        assert button.region.y + button.region.height <= composer.region.y
        assert hit_widget is button


@pytest.mark.asyncio
async def test_console_workspace_new_conversation_button_is_hit_target_in_named_workspace():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        await pilot.pause(0.1)

        button = console.query_one("#console-new-chat-tab", Button)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        hit_x = button.region.x + max(0, button.region.width // 2)
        hit_y = button.region.y + max(0, button.region.height // 2)
        hit_widget, _region = console.get_widget_at(hit_x, hit_y)

        assert button.region.y + button.region.height <= composer.region.y
        assert hit_widget is button


@pytest.mark.asyncio
async def test_console_workspace_rail_new_conversation_creates_default_workspace_session():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    assert active_workspace is not None
    assert active_workspace.workspace_id == DEFAULT_WORKSPACE_ID
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id

        active_session = next(
            session for session in store.sessions() if session.id == second
        )
        assert active_session.workspace_id == DEFAULT_WORKSPACE_ID
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert any(text.startswith("> ") and "Chat 2" in text for text in row_texts)
        assert _static_plain_text(
            console.query_one("#console-workspace-runtime-label", Static)
        ) == "File tools"
        assert "Off in Default workspace" in _static_plain_text(
            console.query_one("#console-workspace-runtime-value", Static)
        )
        assert _static_plain_text(
            console.query_one("#console-workspace-server-readiness-label", Static)
        ) == "Server handoff"
        assert "Not configured" in _static_plain_text(
            console.query_one("#console-workspace-server-readiness-value", Static)
        )
        visible_text = _visible_text(console)
        assert "Workspace conversation creation lands in a later slice" not in visible_text


@pytest.mark.asyncio
async def test_console_workspace_rail_new_conversation_stays_scoped_to_active_workspace():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_session_id = store.active_session_id

        await pilot.click("#console-new-chat-tab")
        session_id = store.active_session_id
        assert session_id is not None
        assert session_id != first_session_id
        active_session = next(
            session for session in store.sessions() if session.id == session_id
        )
        assert active_session.workspace_id == "ws-a"
        active_title = active_session.title

        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            active_title,
            selected=True,
        )
        assert any(
            text.startswith("> ") and active_title in text
            for text in row_texts
        )

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        assert service.get_active_workspace().workspace_id == "ws-b"
        assert all(
            active_title not in row_text
            for row_text in _console_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_workspace_conversation_row_switches_native_session():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 1",
            selected=False,
        )
        await _click_console_workspace_conversation_for_session(console, pilot, store, first.id)

        assert store.active_session_id == first.id
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 1",
            selected=True,
        )
        assert any(text.startswith("> ") and "Chat 1" in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_workspace_conversation_row_resumes_persisted_conversation():
    """Resume a saved workspace conversation directly from the Console rail."""
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-chat-1",
        role="workspace-thread",
        title="Saved research chat",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "persisted-chat-1": {
                "conversation": {
                    "id": "persisted-chat-1",
                    "title": "Saved research chat",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [
                    {
                        "id": "persisted-message-1",
                        "conversation_id": "persisted-chat-1",
                        "role": "user",
                        "sender": "user",
                        "content": "resume saved user prompt",
                        "children": [
                            {
                                "id": "persisted-message-2",
                                "conversation_id": "persisted-chat-1",
                                "sender": "Research Bot",
                                "content": "resume saved assistant reply",
                                "children": [],
                            }
                        ],
                    }
                ],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Saved research chat",
            selected=False,
        )

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "persisted-chat-1",
        )

        await _wait_for_text(console, pilot, "resume saved user prompt")
        await _wait_for_text(console, pilot, "resume saved assistant reply")
        store = console._ensure_console_chat_store()
        active_session = store.switch_session(store.active_session_id)
        assert active_session.persisted_conversation_id == "persisted-chat-1"
        assert active_session.title == "Saved research chat"
        assert active_session.workspace_id == active_workspace.workspace_id
        assistant_messages = [
            message
            for message in store.messages_for_session(active_session.id)
            if message.content == "resume saved assistant reply"
        ]
        assert assistant_messages[-1].role is ConsoleMessageRole.ASSISTANT
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Saved research chat",
            selected=True,
        )
        assert any(text.startswith("> ") and "Saved research chat" in text for text in row_texts)
        selected_row = _workspace_conversation_row_by_id(console, "persisted-chat-1")
        assert selected_row is not None
        selected_row_label = str(selected_row.label)
        assert "\n" in selected_row_label
        assert "Chats" in selected_row_label
        assert selected_row.has_class("console-workspace-conversation-row-selected")
        console._set_console_rail_preference(right_open=True, notify_on_failure=False)
        await pilot.pause(0.1)
        inspector_text = _visible_text(console.query_one("#console-right-rail"))
        assert "Selected Conversation" in inspector_text
        assert "Selected conversation: Saved research chat" in inspector_text
        assert "Conversation source: saved conversation" in inspector_text
        assert "Resume state: restored from persisted-chat-1" in inspector_text
        assert "Workspace: Default" in inspector_text
        assert app.chat_conversation_scope_service.calls == [
            {"conversation_id": "persisted-chat-1", "mode": "local"}
        ]


@pytest.mark.asyncio
async def test_console_workspace_conversation_resume_uses_persisted_workspace():
    """Resume into the persisted conversation workspace when it differs from active."""
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    target_workspace = service.create_workspace(
        workspace_id="ws-resume-target",
        name="Resume Target",
    )
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-cross-workspace-chat",
        role="workspace-thread",
        title="Saved cross workspace",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "persisted-cross-workspace-chat": {
                "conversation": {
                    "id": "persisted-cross-workspace-chat",
                    "title": "Saved cross workspace",
                    "workspace_id": target_workspace.workspace_id,
                },
                "root_threads": [
                    {
                        "id": "persisted-cross-message-1",
                        "conversation_id": "persisted-cross-workspace-chat",
                        "role": "user",
                        "sender": "user",
                        "content": "cross workspace prompt",
                        "children": [],
                    }
                ],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Saved cross works",
            selected=False,
        )

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "persisted-cross-workspace-chat",
        )

        await _wait_for_text(console, pilot, "cross workspace prompt")
        store = console._ensure_console_chat_store()
        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == target_workspace.workspace_id
        assert (
            store.workspace_context.active_workspace_id
            == target_workspace.workspace_id
        )
        assert (
            service.get_active_workspace().workspace_id
            == target_workspace.workspace_id
        )
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Saved cross works",
            selected=True,
        )
        assert any(text.startswith("> ") for text in row_texts)


@pytest.mark.asyncio
async def test_console_workspace_conversation_resume_uses_real_local_services(tmp_path):
    """Resume a workspace conversation through real local DB-backed services."""
    workspace_db = WorkspaceDB(tmp_path / "workspaces.db", client_id="test-client")
    chat_db = CharactersRAGDB(tmp_path / "chacha.db", client_id="test-client")
    workspace_service = LocalWorkspaceRegistryService(workspace_db)
    workspace = workspace_service.create_workspace(
        workspace_id="ws-real",
        name="Real Workspace",
    )
    workspace_service.set_active_workspace(workspace.workspace_id)

    chat_service = ChatConversationService(chat_db)
    conversation_id = chat_service.create_conversation(
        id="real-saved-chat-1",
        title="Real saved chat",
        scope_type="workspace",
        workspace_id=workspace.workspace_id,
        state="in-progress",
    )
    user_message_id = chat_db.add_message(
        {
            "id": "real-message-user-1",
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": "real service user prompt",
        }
    )
    chat_db.add_message(
        {
            "id": "real-message-assistant-1",
            "conversation_id": conversation_id,
            "parent_message_id": user_message_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "real service assistant reply",
        }
    )
    workspace_service.link_membership(
        workspace.workspace_id,
        item_type="conversation",
        item_id=conversation_id,
        role="workspace-thread",
        title="Real saved chat",
    )

    app = _build_test_app()
    app.workspace_registry_service = workspace_service
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=chat_service,
        server_service=None,
    )
    host = ConsoleHarness(app)
    saved_state = None

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Real saved chat",
            selected=False,
        )

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            conversation_id,
        )

        await _wait_for_text(console, pilot, "real service user prompt")
        await _wait_for_text(console, pilot, "real service assistant reply")
        store = console._ensure_console_chat_store()
        active_session = store.switch_session(store.active_session_id)
        assert active_session.persisted_conversation_id == conversation_id
        assert active_session.title == "Real saved chat"
        assert active_session.workspace_id == workspace.workspace_id
        assert any(
            text.startswith("> ") and "Real saved chat" in text
            for text in await _wait_for_workspace_conversation_text(
                console,
                pilot,
                "Real saved chat",
                selected=True,
            )
        )
        left_rail_text = _visible_text(console.query_one("#console-left-rail"))
        console._set_console_rail_preference(right_open=True, notify_on_failure=False)
        await pilot.pause(0.1)
        inspector_text = _visible_text(console.query_one("#console-right-rail"))
        assert "Provider:" not in left_rail_text
        assert "Model:" not in left_rail_text
        assert "Session Settings" in inspector_text
        assert "Provider:" in inspector_text
        assert "Selected conversation: Real saved chat" in inspector_text
        saved_state = console.save_state()

    restored_host = RestoredConsoleHarness(app, saved_state)
    async with restored_host.run_test(size=(160, 48)) as pilot:
        console = restored_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_text(console, pilot, "real service user prompt")
        await _wait_for_text(console, pilot, "real service assistant reply")
        store = console._ensure_console_chat_store()
        restored_session = store.switch_session(store.active_session_id)
        assert restored_session.persisted_conversation_id == conversation_id
        assert restored_session.workspace_id == workspace.workspace_id


@pytest.mark.asyncio
async def test_console_workspace_rail_keeps_active_native_session_visible_when_scope_is_global():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-chat-1",
        role="workspace-thread",
        title="Chat 1",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1", workspace_id="global")
        first.persisted_conversation_id = "persisted-chat-1"
        second = store.create_session(title="Chat 2", workspace_id="global")
        await console._sync_native_console_chat_ui()

        assert store.active_session_id == second.id
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert any("Chat 1" in text for text in row_texts), row_texts


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_keeps_selected_global_native_session():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(
            title="Global Search Chat",
            workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
        )
        session.title = "Global Search Chat"
        session.workspace_id = CONSOLE_GLOBAL_WORKSPACE_ID
        await console._sync_native_console_chat_ui()

        await _wait_for_selector(console, pilot, "#console-workspace-conversation-search")
        console.query_one("#console-workspace-conversation-search", Input).focus()
        await _set_console_conversation_browser_search(console, pilot, "global")
        await _wait_for_text(console, pilot, "1 match")
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Global Search Chat",
            selected=True,
        )

        assert any(
            text.startswith("> ") and "Global Search Chat" in text
            for text in row_texts
        )


@pytest.mark.asyncio
async def test_console_new_chat_focuses_composer_for_immediate_typing():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.press("n")
        await pilot.pause(0.1)

        assert console.app.focused is composer
        assert composer.draft_text() == "n"


@pytest.mark.asyncio
async def test_console_tab_switch_focuses_composer_for_immediate_typing():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()
        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        await pilot.click(f"#console-session-tab-{first.id}")
        assert store.active_session_id == first.id

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.press("s")
        await pilot.pause(0.1)

        assert console.app.focused is composer
        assert composer.draft_text() == "s"


@pytest.mark.asyncio
async def test_console_native_tab_strip_isolates_composer_drafts():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("first tab draft")

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        assert composer.draft_text() == ""

        composer.load_draft("second tab draft")
        await pilot.click(f"#console-session-tab-{first.id}")
        assert store.active_session_id == first.id
        assert composer.draft_text() == "first tab draft"

        await pilot.click(f"#console-session-tab-{second}")
        assert store.active_session_id == second
        assert composer.draft_text() == "second tab draft"


@pytest.mark.asyncio
async def test_console_native_tab_strip_keeps_compact_close_x():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        close_selector = f"#console-close-session-tab-{first.id}"
        await _wait_for_selector(console, pilot, close_selector)
        close_button = console.query_one(close_selector, Button)

        assert close_button.label.plain == "x"
        assert 2 <= close_button.region.width <= 4

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        await _wait_for_selector(console, pilot, f"#console-close-session-tab-{second}")
        await pilot.click(f"#console-close-session-tab-{second}")

        assert store.active_session_id == first.id
        assert second not in {session.id for session in store.sessions()}


@pytest.mark.asyncio
async def test_console_native_tab_title_has_stable_visible_label_region():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.rename_session(session.id, "Planning session with a long descriptive name")
        await console._sync_native_console_chat_ui()

        tab_selector = f"#console-session-tab-{session.id}"
        await _wait_for_selector(console, pilot, tab_selector)
        tab = console.query_one(tab_selector, Button)

        assert tab.tooltip == (
            "Active Console tab: Planning session with a long descriptive name. "
            "Click again to rename."
        )
        assert str(tab.label) == "Planning session..."
        assert tab.region.width >= 18
        assert "Planning session" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_native_active_tab_title_opens_rename_modal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        assert not list(console.query(f"#console-rename-session-tab-{session.id}"))

        await pilot.click(f"#console-session-tab-{session.id}")
        modal_screen = await _wait_for_console_rename_modal(host, pilot)

        rename_input = modal_screen.query_one("#console-rename-session-title", Input)
        assert rename_input.value == "Chat 1"
        assert getattr(console.app.focused, "id", None) == rename_input.id

        await pilot.press(*"Planning")
        modal_screen.query_one("#console-rename-session-save", Button).press()
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{session.id}")

        assert store.sessions()[0].title == "Planning"
        assert "Planning" in _visible_text(console)
        assert not list(console.query(f"#console-session-rename-input-{session.id}"))


@pytest.mark.asyncio
async def test_console_native_rename_modal_buttons_are_not_clipped():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click(f"#console-session-tab-{session.id}")
        modal_screen = await _wait_for_console_rename_modal(host, pilot)

        action_row = modal_screen.query_one("#console-rename-session-actions")
        cancel_button = modal_screen.query_one("#console-rename-session-cancel", Button)
        save_button = modal_screen.query_one("#console-rename-session-save", Button)

        assert action_row.region.height >= 3
        assert cancel_button.region.height >= 3
        assert save_button.region.height >= 3
        assert str(cancel_button.label) == "Cancel"
        assert str(save_button.label) == "Save"


@pytest.mark.asyncio
async def test_console_native_tab_rename_escape_restores_existing_title():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click(f"#console-session-tab-{session.id}")

        modal_screen = await _wait_for_console_rename_modal(host, pilot)
        rename_input = modal_screen.query_one("#console-rename-session-title", Input)
        assert rename_input.value == "Chat 1"
        await pilot.press(*"Discarded")
        await pilot.press("escape")
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{session.id}")

        assert store.sessions()[0].title == "Chat 1"
        assert "Chat 1" in _visible_text(console)
        assert not list(console.query(f"#console-session-rename-input-{session.id}"))


@pytest.mark.asyncio
async def test_console_close_tab_with_messages_shows_confirmation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
        store.create_session(title="Chat 2")
        await console._sync_native_console_chat_ui()

        close_selector = f"#console-close-session-tab-{session.id}"
        await _wait_for_selector(console, pilot, close_selector)
        await pilot.click(close_selector)

        from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

        for _ in range(20):
            await pilot.pause()
            if any(isinstance(s, ConfirmationDialog) for s in host.screen_stack):
                break

        dialog_screens = [s for s in host.screen_stack if isinstance(s, ConfirmationDialog)]
        assert len(dialog_screens) == 1, "confirmation dialog should appear for tab with messages"
        assert session.id in {s.id for s in store.sessions()}, "session not closed yet"

        await pilot.click("#confirm-button")
        for _ in range(10):
            await pilot.pause()

        assert session.id not in {s.id for s in store.sessions()}, "session closed after confirm"


def test_native_console_state_serializes_plain_string_message_role():
    """Verify saved Console messages tolerate legacy/plain-string roles."""
    message = SimpleNamespace(
        id="message-a",
        role="assistant",
        content="answer",
        turn_id=None,
        status="complete",
        persisted_message_id=None,
        feedback=None,
        variants=None,
    )

    serialized = ChatScreen._serialize_console_message(message)

    assert serialized["role"] == "assistant"
