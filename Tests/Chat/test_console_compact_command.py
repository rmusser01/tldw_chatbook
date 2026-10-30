"""TASK-33621.4 AC#4: start compaction from /compact and the command palette.

Before this the only way to compact was Compact now at the bottom of Chat
settings ▸ Context and memory (about 20 wheel ticks down on dev): no slash
command and no palette entry existed (live 2026-10-29: Ctrl+P "compact"
listed only unrelated fuzzy matches).
"""

from __future__ import annotations

import asyncio

from tldw_chatbook.Chat.console_command_grammar import (
    CONSOLE_ACTION_COMMAND_HANDLER_ID,
    CONSOLE_ACTION_COMMANDS,
    KIND_COMMAND,
    default_console_registry,
)
from tldw_chatbook.Chat.console_command_suggestions import _COMMAND_DESCRIPTIONS


def test_compact_is_a_typed_action_command_with_a_description() -> None:
    parse = default_console_registry().parse("/compact")

    assert parse.kind == KIND_COMMAND and parse.name == "compact"
    assert ("compact", "") in CONSOLE_ACTION_COMMANDS
    by_name = {c.name: c for c in default_console_registry().commands()}
    assert by_name["compact"].handler_id == CONSOLE_ACTION_COMMAND_HANDLER_ID
    description = _COMMAND_DESCRIPTIONS["compact"]
    assert "Summarize older turns" in description
    assert "model call" in description


def test_compact_dispatches_to_the_screens_compact_action() -> None:
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    method = ChatScreen._CONSOLE_ACTION_COMMAND_TARGETS["compact"]
    assert method == "action_compact_console_context"
    assert callable(getattr(ChatScreen, method))


def test_the_palette_offers_compact_now() -> None:
    from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider

    called: list[str] = []

    class _ScreenStub:
        def __getattr__(self, name):
            if name.startswith("action_"):
                return lambda *_args: called.append(name)
            raise AttributeError(name)

    commands = ConsoleCommandProvider._commands(None, _ScreenStub())
    entry = next(
        (row for row in commands if row[0] == "Console: Compact context now"), None
    )
    assert entry is not None
    entry[1]()
    assert called == ["action_compact_console_context"]
    assert "/compact" in entry[2]


class _Controller:
    def __init__(self, result: tuple[bool, str]) -> None:
        self.result = result
        self.calls: list[str] = []
        self.store = type("Store", (), {"active_session_id": "session-1"})()

    async def compact_context_now(self, session_id: str) -> tuple[bool, str]:
        self.calls.append(session_id)
        return self.result


class _Screen:
    def __init__(self, controller: _Controller) -> None:
        self.controller = controller
        self.rows: list[tuple[str, str | None]] = []
        self.toasts: list[str] = []
        self.synced = 0

    def _ensure_console_chat_controller(self) -> _Controller:
        return self.controller

    async def _append_native_console_system_message(
        self, message: str, *, session_id: str | None = None
    ) -> None:
        self.rows.append((message, session_id))

    async def _sync_native_console_chat_ui(self) -> None:
        self.synced += 1

    def notify(self, message: str, **_kwargs) -> None:
        self.toasts.append(message)


def test_the_compact_command_runs_compact_now_and_reports_in_its_chat() -> None:
    from tldw_chatbook.UI.Console_Modules.compact_command import run_compact_command

    screen = _Screen(_Controller((True, "Conversation memory updated.")))
    assert asyncio.run(run_compact_command(screen)) is True

    assert screen.controller.calls == ["session-1"]
    assert screen.rows == [("Conversation memory updated.", "session-1")]
    assert screen.toasts and "one model call" in screen.toasts[0]
    assert screen.synced == 1


def test_a_refused_compaction_says_why_in_its_chat() -> None:
    from tldw_chatbook.UI.Console_Modules.compact_command import run_compact_command

    refusal = "There are not enough older complete turns to compact yet."
    screen = _Screen(_Controller((False, refusal)))
    assert asyncio.run(run_compact_command(screen)) is False

    assert screen.rows == [(refusal, "session-1")]
