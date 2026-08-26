"""A failed durable commit must say WHICH failure occurred (TASK-22251).

`_submit_draft_inner` wraps `commit_durable_turn` in `except Exception:` and
returns one generic sentence -- "Couldn't save the prepared turn. Retry or
cancel." -- for every way that multi-step transaction can fail: conversation
create, Library-policy write, workspace validation, checkpoint insert.

The copy is fine. Discarding the exception is not. Two distinct causes were
found behind that identical message while burning down Console tests
("Workspace registry is required for workspace conversations" and "Unknown
workspace: <id>"), and identifying each required adding a temporary `print`
inside production. A failure nobody can attribute without editing the product
is a diagnostic defect, not a user-experience one.

These tests pin both halves: the type IS recorded, and the exception's message
is NOT (it can carry conversation and workspace identifiers).

Helper classes here are named without a leading underscore, per the repo's
PascalCase rule (Qodo finding on #2102). Note this differs from the prevailing
convention in `Tests/` -- 2,448 test classes carry a leading underscore to mark
them module-private -- so the rule and the corpus disagree; this file follows
the rule.
"""

from __future__ import annotations

import pytest
from loguru import logger as loguru_logger

from Tests.console_provider_doubles import provider_resolution
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class ReadyGateway:
    """Ready provider that streams one chunk."""

    async def resolve_for_send(self, selection):
        return provider_resolution()

    async def stream_chat(self, resolution, messages, **kwargs):
        yield "reply"


class CapturedLines:
    """Collect loguru records without disturbing the ambient sinks."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._sink_id: int | None = None

    def __enter__(self) -> "CapturedLines":
        self._sink_id = loguru_logger.add(
            lambda message: self.lines.append(str(message)),
            level="TRACE",
            diagnose=False,
        )
        return self

    def __exit__(self, *exc_info) -> None:
        if self._sink_id is not None:
            try:
                loguru_logger.remove(self._sink_id)
            except ValueError:
                pass

    def matching(self, needle: str) -> list[str]:
        return [line for line in self.lines if needle in line]


#: A value that must never be logged: it stands in for the conversation and
#: workspace identifiers a real commit failure puts in its exception message.
SECRET_IDENTIFIER = "ws-private-4f2a-identifier"


class ExplodingPersistence:
    """Persistence whose durable commit fails with an identifying message."""

    db = None

    def commit_durable_turn(self, *args, **kwargs):
        raise ValueError(f"Unknown workspace: {SECRET_IDENTIFIER}")


def _controller_with_failing_commit():
    store = ConsoleChatStore(persistence=ExplodingPersistence())
    controller = ConsoleChatController(
        store=store,
        provider_gateway=ReadyGateway(),
        provider="llama_cpp",
        model="test-model",
        agent_runtime_enabled=False,
    )
    return controller, store


@pytest.mark.asyncio
async def test_a_failed_durable_commit_records_the_exception_type() -> None:
    """The refusal is attributable from logs alone, with no product edit."""
    controller, _store = _controller_with_failing_commit()

    with CapturedLines() as captured:
        result = await controller.submit_draft("capital of Japan?")

    assert result.accepted is False
    assert "Couldn't save the prepared turn" in (result.visible_copy or "")

    recorded = captured.matching("Durable turn commit failed")
    assert recorded, (
        "a swallowed durable-commit failure left no trace; the refusal is "
        "unattributable without editing production. Lines seen: "
        f"{[line[:90] for line in captured.lines][-5:]}"
    )
    assert any("ValueError" in line for line in recorded), (
        f"the exception TYPE was not recorded: {[l[:120] for l in recorded]}"
    )


@pytest.mark.asyncio
async def test_the_failure_log_does_not_leak_the_exception_message() -> None:
    """Type only. A commit exception can name a conversation or workspace."""
    controller, _store = _controller_with_failing_commit()

    with CapturedLines() as captured:
        await controller.submit_draft("capital of Japan?")

    leaked = [line for line in captured.lines if SECRET_IDENTIFIER in line]
    assert not leaked, (
        "the durable-commit failure log leaked the exception message, which "
        f"carries caller identifiers: {[line[:140] for line in leaked]}"
    )
