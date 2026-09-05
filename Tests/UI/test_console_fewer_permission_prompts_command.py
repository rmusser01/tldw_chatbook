from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_command_grammar import CommandParse
from tldw_chatbook.MCP.permission_prompt_reducer import (
    PermissionPromptRecommendation,
    PermissionPromptReport,
)
from tldw_chatbook.UI.Console_Modules.wiring import build_console_commands_controller
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


class _PromptReductionService:
    def __init__(self, report: PermissionPromptReport) -> None:
        self.report = report
        self.calls = 0

    async def permission_prompt_recommendations(self) -> PermissionPromptReport:
        self.calls += 1
        return self.report


def _screen_with_service(service: object) -> tuple[ChatScreen, list[str], list[bool]]:
    screen = object.__new__(ChatScreen)
    screen.app_instance = SimpleNamespace(unified_mcp_service=service)
    messages: list[str] = []
    clears: list[bool] = []

    async def append_message(message: str) -> None:
        messages.append(message)

    screen._append_native_console_system_message = append_message
    screen._clear_console_composer_draft = lambda: clears.append(True)
    build_console_commands_controller(screen)
    return screen, messages, clears


@pytest.mark.asyncio
async def test_fewer_permission_prompts_command_renders_report_and_clears_draft():
    """Catches `/fewer-permission-prompts` parsing without dispatching a report."""
    report = PermissionPromptReport(
        recommendations=[
            PermissionPromptRecommendation(
                server_key="local:docs",
                server_label="docs",
                tool_name="search",
                approved_count=3,
                first_seen="2026-08-01T20:00:00+00:00",
                last_seen="2026-08-01T20:06:00+00:00",
                current_state="ask",
                reason="Repeatedly approved while still ask-gated.",
            )
        ],
        excluded={},
        total_records=8,
        approval_records=3,
        min_approved_count=2,
    )
    service = _PromptReductionService(report)
    screen, messages, clears = _screen_with_service(service)

    await screen._commands._console_command_fewer_permission_prompts(
        CommandParse("command", "fewer-permission-prompts", "")
    )

    assert service.calls == 1
    assert clears == [True]
    assert len(messages) == 1
    assert "MCP prompt recommendations" in messages[0]
    assert "docs / search - approved 3 times" in messages[0]
    assert "Auto Mode and bash allowlisting are deferred." in messages[0]


@pytest.mark.asyncio
async def test_fewer_permission_prompts_command_reports_missing_mcp_service():
    """Catches silently consuming the command when MCP services are unavailable."""
    screen, messages, clears = _screen_with_service(service=None)

    await screen._commands._console_command_fewer_permission_prompts(
        CommandParse("command", "fewer-permission-prompts", "")
    )

    assert clears == [True]
    assert messages == [
        "MCP prompt recommendations unavailable - MCP service is not ready."
    ]
