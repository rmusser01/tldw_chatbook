"""TASK-33621.3: automatic compaction in a live session, end to end.

Every earlier compaction test seeded its transcript through
``append_message(persist=True)``, which threads ``parent_message_id`` onto the
live node as it writes the row. A real Console send does not go that way: the
durable turn commit writes both rows and ``_hydrate_durable_turn_owner_messages``
binds only their persisted ids. The compaction fence was built from the
in-memory parent mirror, so it never matched the database and every automatic
compaction in a live session failed after a billed summarizer call -- with no
reason logged, the same generic copy, and another billed call on every later
send.

These tests therefore drive the production path only: a real ChaChaNotes DB, the
real store, the real controller ``submit_draft`` durable send, the real
``ConsoleContextRepository`` and ``ConsoleCompactionService``, and the real
provider-request preparation (token accounting). Only the network-facing
provider calls are doubled.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_library_destination import (
    resolve_console_destination,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionResult,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.cost_display import format_cost_amount
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.LLM_Calls.pricing_catalog import get_pricing_catalog

_MODEL = "gpt-test-live"
_REPLY_WORDS = 220
_SUMMARY = "The user asked numbered questions; the assistant answered each."
_SUMMARY_USAGE = ProviderUsage(
    uncached_input=1_234, output=17, provider="openai", model=_MODEL
)
_OVERRIDES = ConsoleContextPolicyOverrides(
    budget_mode=ContextBudgetMode.CUSTOM,
    custom_budget_tokens=1_800,
    compaction_mode=ContextCompactionMode.AUTOMATIC,
    summary_max_tokens=100,
)


class _LiveProviderGateway:
    """Network-facing double; request preparation stays the production one."""

    def __init__(
        self,
        *,
        summary: str = _SUMMARY,
        usage_model: str = _MODEL,
        context_window: int = 32_000,
    ) -> None:
        self._real = ConsoleProviderGateway(environ={})
        self.summary = summary
        self.usage_model = usage_model
        self.context_window = context_window
        self.stream_calls = 0
        self.auxiliary_calls = 0

    async def resolve_for_send(self, _selection: object) -> ConsoleProviderResolution:
        resolution = ConsoleProviderResolution(
            provider="openai",
            ready=True,
            execution_key="OpenAI",
            model=_MODEL,
            api_key="test-only-key",
            base_url=None,
            temperature=None,
            top_p=None,
            min_p=None,
            top_k=None,
            max_tokens=120,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            reasoning_effort=None,
            reasoning_summary=None,
            verbosity=None,
            thinking_effort=None,
            thinking_budget_tokens=None,
            streaming=False,
        )
        return replace(
            resolution, resolved_destination=resolve_console_destination(resolution)
        )

    def prepare_chat_request(self, resolution, messages, **kwargs: Any):
        kwargs.setdefault("context_window_override_tokens", self.context_window)
        return self._real.prepare_chat_request(resolution, messages, **kwargs)

    async def stream_chat(self, _resolution, _messages, **_kwargs: Any):
        self.stream_calls += 1
        yield f"answer-{self.stream_calls} " + "detail " * _REPLY_WORDS

    async def complete_auxiliary(self, _request, *, route=None):
        self.auxiliary_calls += 1
        return AuxiliaryCompletionResult(
            provider="openai",
            model=_MODEL,
            text=self.summary,
            usage=_SUMMARY_USAGE
            if self.usage_model == _MODEL
            else replace(_SUMMARY_USAGE, model=self.usage_model),
        )


def _live_controller(
    tmp_path: Path,
    *,
    gateway: _LiveProviderGateway | None = None,
    overrides: ConsoleContextPolicyOverrides = _OVERRIDES,
) -> tuple[CharactersRAGDB, ConsoleChatStore, ConsoleChatController, Any]:
    db = CharactersRAGDB(tmp_path / "live-compaction.sqlite", client_id="task33621")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(session_id="session-1", title="Chat 1")
    store.set_session_context_policy_overrides(session.id, overrides)
    live_gateway = gateway or _LiveProviderGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=live_gateway,
        provider="openai",
        model=_MODEL,
    )
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    return db, store, controller, live_gateway


def _attempt_rows(db: CharactersRAGDB) -> list[sqlite3.Row]:
    return (
        db.get_connection()
        .execute("SELECT * FROM console_auxiliary_attempts ORDER BY started_at, rowid")
        .fetchall()
    )


def _active_memory_count(db: CharactersRAGDB) -> int:
    row = (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) AS n FROM console_conversation_memories WHERE active = 1"
        )
        .fetchone()
    )
    return int(row["n"])


async def _send_until_compaction(
    controller: ConsoleChatController,
    gateway: _LiveProviderGateway,
    *,
    limit: int = 12,
) -> tuple[int, Any]:
    """Send ordinary turns until the first automatic summary call happens."""

    for index in range(limit):
        before = gateway.auxiliary_calls
        result = await controller.submit_draft(
            f"question-{index}: explain step {index} in detail.",
            session_id="session-1",
        )
        if gateway.auxiliary_calls > before:
            return index, result
    raise AssertionError("the custom budget was never crossed")


@pytest.mark.asyncio
async def test_live_automatic_compaction_commits_memory_and_the_send_replies(
    tmp_path: Path,
) -> None:
    """AC#1/#6: real durable sends cross the budget; compaction commits."""

    db, store, controller, gateway = _live_controller(tmp_path)

    index, result = await _send_until_compaction(controller, gateway)

    assert index >= 2, "compaction must need several real turns to trigger"
    assert gateway.auxiliary_calls == 1
    rows = _attempt_rows(db)
    assert [row["status"] for row in rows] == ["succeeded"]
    assert _active_memory_count(db) == 1
    # The triggering send reached the provider and got its reply.
    assert result.accepted is True
    assert gateway.stream_calls == index + 1
    assistant = [
        message
        for message in store.messages_for_session("session-1")
        if message.role is ConsoleMessageRole.ASSISTANT
    ][-1]
    assert assistant.status == "complete"
    assert assistant.content.startswith(f"answer-{index + 1} ")

    # The memory is effective in the SAME session: the next ordinary send is
    # back under the trigger and needs no second summary call.
    follow_up = await controller.submit_draft(
        "question-next: one more detail.", session_id="session-1"
    )
    assert follow_up.accepted is True
    assert gateway.auxiliary_calls == 1
    assert gateway.stream_calls == index + 2


@pytest.mark.asyncio
async def test_live_compact_now_succeeds_after_real_durable_sends(
    tmp_path: Path,
) -> None:
    """AC#1: Compact now in the same live session commits a memory."""

    db, _store, controller, gateway = _live_controller(
        tmp_path,
        overrides=replace(_OVERRIDES, compaction_mode=ContextCompactionMode.OFF),
    )
    for index in range(3):
        result = await controller.submit_draft(
            f"question-{index}: explain step {index}.", session_id="session-1"
        )
        assert result.accepted is True
    assert gateway.auxiliary_calls == 0

    succeeded, copy = await controller.compact_context_now("session-1")

    assert succeeded is True, copy
    assert gateway.auxiliary_calls == 1
    assert [row["status"] for row in _attempt_rows(db)] == ["succeeded"]
    assert [row["failure_reason"] for row in _attempt_rows(db)] == [None]
    assert _active_memory_count(db) == 1


def _system_rows(store: ConsoleChatStore) -> list[str]:
    return [
        message.content
        for message in store.messages_for_session("session-1")
        if message.role is ConsoleMessageRole.SYSTEM
    ]


@pytest.mark.asyncio
async def test_live_failed_compaction_records_reason_and_discloses_spend(
    tmp_path: Path,
) -> None:
    """AC#2/#3/#4: the reason reaches the row and log; the copy is honest."""

    priced_model = "gpt-4.1-mini"
    db, store, controller, gateway = _live_controller(
        tmp_path,
        gateway=_LiveProviderGateway(summary="", usage_model=priced_model),
    )
    records: list[str] = []
    sink = logger.add(lambda message: records.append(message.record["message"]))
    try:
        await _send_until_compaction(controller, gateway)
    finally:
        logger.remove(sink)

    rows = _attempt_rows(db)
    assert [(row["status"], row["failure_reason"]) for row in rows] == [
        ("failed", "invalid_summary_output")
    ]
    assert (
        "console_compaction_failed reason=invalid_summary_output status=failed "
        "error_type=none"
    ) in records
    assert _active_memory_count(db) == 0

    copy = _system_rows(store)[-1]
    assert copy == controller.run_state_for("session-1").visible_copy
    assert copy.startswith("Your message was not sent")
    assert "provider request was not sent" not in copy
    assert "empty, oversized or malformed summary" in copy
    # The spend of the failed call, and its cost because the model is priced.
    assert "1,234 input + 17 output tokens" in copy
    cost = get_pricing_catalog().cost_for_usage(
        replace(_SUMMARY_USAGE, model=priced_model)
    )
    assert cost is not None
    assert f"(about ${format_cost_amount(cost.total)})" in copy
    # A next step the user can act on.
    assert "raise Conversation max tokens" in copy
    assert "Omit older context" in copy
    assert "start a new chat" in copy


@pytest.mark.asyncio
async def test_live_failed_compact_now_says_nothing_changed(tmp_path: Path) -> None:
    """AC#3: a failed Compact now says so, and that nothing changed."""

    db, store, controller, gateway = _live_controller(
        tmp_path,
        gateway=_LiveProviderGateway(summary=""),
        overrides=replace(_OVERRIDES, compaction_mode=ContextCompactionMode.OFF),
    )
    for index in range(3):
        await controller.submit_draft(f"question-{index}.", session_id="session-1")
    transcript_before = [
        (message.id, message.content)
        for message in store.messages_for_session("session-1")
    ]

    succeeded, copy = await controller.compact_context_now("session-1")

    assert succeeded is False
    assert copy.startswith("Compaction failed and nothing changed: ")
    assert "not sent" not in copy
    assert "1,234 input + 17 output tokens" in copy
    assert [row["failure_reason"] for row in _attempt_rows(db)] == [
        "invalid_summary_output"
    ]
    assert _active_memory_count(db) == 0
    assert [
        (message.id, message.content)
        for message in store.messages_for_session("session-1")
    ] == transcript_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget_mode", "context_window"),
    [
        (ContextBudgetMode.CUSTOM, 32_000),
        # Automatic budget mode derives the budget from the model window
        # minus each request's own mandatory tokens, so the budget moves
        # with every send -- it must not lift the pause by itself.
        (ContextBudgetMode.AUTOMATIC, 2_400),
    ],
)
async def test_live_failure_is_not_rebilled_until_the_policy_changes(
    tmp_path: Path,
    budget_mode: ContextBudgetMode,
    context_window: int,
) -> None:
    """AC#5: Retry, Discard + a fresh send, then a policy change."""

    overrides = replace(_OVERRIDES, budget_mode=budget_mode)
    db, store, controller, gateway = _live_controller(
        tmp_path,
        gateway=_LiveProviderGateway(summary="", context_window=context_window),
        overrides=overrides,
    )
    await _send_until_compaction(controller, gateway)
    assert gateway.auxiliary_calls == 1
    assert store.dispatch_recovery_for_session("session-1") is not None

    retried = await controller.retry_dispatch_recovery("session-1")
    assert gateway.auxiliary_calls == 1
    assert "automatic compaction is paused" in retried.visible_copy
    assert "No new summary call was made." in retried.visible_copy

    streams_before = gateway.stream_calls
    await controller.submit_draft("a fresh question", session_id="session-1")
    assert gateway.auxiliary_calls == 1
    assert gateway.stream_calls == streams_before  # the message was not sent
    assert "automatic compaction is paused" in _system_rows(store)[-1]
    discarded = await controller.discard_dispatch_recovery("session-1")
    assert discarded.accepted is True, discarded.visible_copy

    # Changing the compaction policy re-arms automatic compaction: the next
    # send makes exactly ONE summary call -- never more than one per send.
    store.set_session_context_policy_overrides(
        "session-1", replace(overrides, summary_max_tokens=120)
    )
    await controller.submit_draft("after the policy change", session_id="session-1")
    assert gateway.auxiliary_calls == 2
    assert [row["failure_reason"] for row in _attempt_rows(db)] == [
        "invalid_summary_output",
        "invalid_summary_output",
    ]
