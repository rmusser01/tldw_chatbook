"""Tests for the combined chat shell bar."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select

from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.UI.Screens.chat_screen_state import TabState
from tldw_chatbook.Widgets.Chat_Widgets.chat_shell_bar import (
    ChatShellBar,
    ChatShellContext,
    ChatShellLabelResolver,
)


@dataclass
class _ShellBarFixture:
    session_data: object
    resolver: ChatShellLabelResolver | None = None


class ShellBarTestApp(App):
    """Small app used to exercise the shell bar in isolation."""

    def __init__(self, fixture: _ShellBarFixture) -> None:
        super().__init__()
        self.fixture = fixture
        self.app_config = {
            "chat_defaults": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.3,
            }
        }

    def compose(self) -> ComposeResult:
        yield ChatShellBar(
            session_data=self.fixture.session_data,
            resolver=self.fixture.resolver,
            id="chat-shell-bar",
        )
        yield Input(id="after-shell-input")


def _focused_widget_id(app: App) -> str | None:
    for widget in app.query("Select, Input, Button"):
        if widget.has_focus:
            return widget.id
    return None


def test_chat_shell_context_defaults_from_none() -> None:
    context = ChatShellContext.from_session_data(None)

    assert context.backend_label == "Local"
    assert context.scope_label == "Global"
    assert context.assistant_label == "Assistant: General"
    assert context.session_label == "Session: New chat"


def test_chat_shell_context_supports_tab_state_and_chat_session_data() -> None:
    resolver = ChatShellLabelResolver(
        workspace_name="Research Lab",
        persona_label="Priya",
        character_label="Vox",
    )

    tab_state = TabState(
        tab_id="tab-a",
        title="A Very Long Session Name",
        runtime_backend="server",
        scope_type="workspace",
        workspace_id="ws-123",
        assistant_kind="persona",
        assistant_id="persona-9",
    )
    session_data = ChatSessionData(
        tab_id="tab-b",
        title="Chat Session",
        runtime_backend="server",
        scope_type="workspace",
        workspace_id="ws-123",
        assistant_kind="character",
        character_id=88,
        character_name="Ignored Name",
    )

    tab_context = ChatShellContext.from_tab_state(tab_state, resolver=resolver)
    session_context = ChatShellContext.from_session_data(session_data, resolver=resolver)

    assert tab_context.backend_label == "Server"
    assert tab_context.scope_label == "Workspace: Research Lab"
    assert tab_context.assistant_label == "Persona: Priya"
    assert tab_context.session_label == "Session: A Very Long Session Name"

    assert session_context.backend_label == "Server"
    assert session_context.scope_label == "Workspace: Research Lab"
    assert session_context.assistant_label == "Character: Vox"
    assert session_context.session_label == "Session: Chat Session"


def test_chat_shell_context_truncates_session_label_last() -> None:
    context = ChatShellContext(
        backend_label="Server",
        scope_label="Workspace: Research Lab",
        assistant_label="Persona: Priya",
        session_label="Session: A very long session title that should be shortened",
    )

    segments = context.prioritized_segments(80)

    assert segments[:3] == [
        "Server",
        "Workspace: Research Lab",
        "Persona: Priya",
    ]
    assert segments[-1].startswith("Session:")
    assert len(" | ".join(segments)) <= 80
    assert segments[-1] != "Session: A very long session title that should be shortened"


@pytest.mark.asyncio
async def test_chat_shell_bar_exposes_compact_control_ids() -> None:
    fixture = _ShellBarFixture(session_data=ChatSessionData(tab_id="tab-a"))
    app = ShellBarTestApp(fixture)

    with patch(
        "tldw_chatbook.Widgets.compact_model_bar.get_cli_providers_and_models",
        return_value={"openai": ["gpt-4o-mini", "gpt-4o"]},
    ):
        async with app.run_test(size=(120, 20)) as pilot:
            assert app.query_one("#compact-api-provider", Select)
            assert app.query_one("#compact-api-model", Select)
            assert app.query_one("#compact-temperature", Input)
            assert app.query_one("#compact-sidebar-toggle", Button)


@pytest.mark.asyncio
async def test_chat_shell_bar_keyboard_traversal_reaches_embedded_controls() -> None:
    fixture = _ShellBarFixture(session_data=ChatSessionData(tab_id="tab-a"))
    app = ShellBarTestApp(fixture)

    with patch(
        "tldw_chatbook.Widgets.compact_model_bar.get_cli_providers_and_models",
        return_value={"openai": ["gpt-4o-mini", "gpt-4o"]},
    ):
        async with app.run_test(size=(120, 20)) as pilot:
            observed_focus = []
            for _ in range(5):
                await pilot.press("tab")
                await pilot.pause()
                observed_focus.append(_focused_widget_id(app))

            assert "compact-api-provider" in observed_focus
            assert "compact-api-model" in observed_focus
            assert "compact-temperature" in observed_focus
            assert "compact-sidebar-toggle" in observed_focus
            assert "after-shell-input" in observed_focus
