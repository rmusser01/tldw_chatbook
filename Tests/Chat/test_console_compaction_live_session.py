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
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunState
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import (
    CompactionFailureBehavior,
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

# The real controller reads settings through the guarded config loader
# (ConsoleChatController.__init__ -> get_cli_setting). Under the per-test
# sandbox that admission fails closed with
# RecoveryRequired("raw_source_selection_changed") before any test body runs,
# and the pricing catalog the copy discloses reads the same loader. These
# tests fake the provider network, never the config getters, so they keep
# the bootstrap profile like the console continuation suites in conftest.
pytestmark = pytest.mark.bootstrap_profile

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
_OMIT = replace(
    _OVERRIDES, failure_behavior=CompactionFailureBehavior.OMIT_OLDER_CONTEXT
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


#: Every database ``_live_controller`` opened in the current test.
_OPEN_DATABASES: list[CharactersRAGDB] = []


@pytest.fixture(autouse=True)
def _close_live_databases() -> Iterator[None]:
    """Close each test's databases at teardown, whether the test passed or not."""

    yield
    while _OPEN_DATABASES:
        _OPEN_DATABASES.pop().close_connection()


def _live_controller(
    tmp_path: Path,
    *,
    gateway: _LiveProviderGateway | None = None,
    overrides: ConsoleContextPolicyOverrides = _OVERRIDES,
) -> tuple[CharactersRAGDB, ConsoleChatStore, ConsoleChatController, Any]:
    db = CharactersRAGDB(tmp_path / "live-compaction.sqlite", client_id="task33621")
    _OPEN_DATABASES.append(db)
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


@pytest.mark.asyncio
async def test_live_compact_now_retries_once_after_an_automatic_failure(
    tmp_path: Path,
) -> None:
    """AC#5: the pause stops automatic attempts only; Compact now still tries."""

    db, _store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary="")
    )
    await _send_until_compaction(controller, gateway)
    discarded = await controller.discard_dispatch_recovery("session-1")
    assert discarded.accepted is True, discarded.visible_copy

    gateway.summary = _SUMMARY
    succeeded, copy = await controller.compact_context_now("session-1")

    assert succeeded is True, copy
    assert gateway.auxiliary_calls == 2
    assert [row["status"] for row in _attempt_rows(db)] == ["failed", "succeeded"]
    assert _active_memory_count(db) == 1


@pytest.mark.asyncio
async def test_live_failed_compact_now_pauses_automatic_compaction(
    tmp_path: Path,
) -> None:
    """AC#5: a billed Compact now failure is not re-billed by the next sends."""

    _db, store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary="")
    )
    for index in range(2):
        await controller.submit_draft(f"question-{index}.", session_id="session-1")
    assert gateway.auxiliary_calls == 0

    succeeded, _copy = await controller.compact_context_now("session-1")
    assert succeeded is False
    assert gateway.auxiliary_calls == 1

    for index in range(2, 12):
        streams = gateway.stream_calls
        await controller.submit_draft(f"question-{index}.", session_id="session-1")
        if gateway.stream_calls == streams:
            break
    else:
        raise AssertionError("the custom budget was never crossed")
    assert gateway.auxiliary_calls == 1
    assert "automatic compaction is paused" in _system_rows(store)[-1]


async def _cheap_auxiliary(_selection: object, main: ConsoleProviderResolution):
    return replace(main, model="gpt-aux-cheap")


@pytest.mark.asyncio
async def test_live_failed_auxiliary_compact_now_keeps_the_send_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC#5: a pause is kept per model; Compact now on another cannot lift it."""

    db, store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary="")
    )
    # [chat_defaults] auxiliary_model routes Compact now (not sends) to a
    # cheaper model; only that resolution step is replaced here.
    monkeypatch.setattr(controller, "_auxiliary_compaction_resolution", _cheap_auxiliary)
    await _send_until_compaction(controller, gateway)
    discarded = await controller.discard_dispatch_recovery("session-1")
    assert discarded.accepted is True, discarded.visible_copy

    succeeded, _copy = await controller.compact_context_now("session-1")
    assert succeeded is False
    assert gateway.auxiliary_calls == 2

    streams = gateway.stream_calls
    await controller.submit_draft("after compact now", session_id="session-1")
    assert gateway.auxiliary_calls == 2
    assert gateway.stream_calls == streams
    assert "automatic compaction is paused" in _system_rows(store)[-1]
    assert [(row["model"], row["status"]) for row in _attempt_rows(db)] == [
        (_MODEL, "failed"),
        ("gpt-aux-cheap", "failed"),
    ]


@pytest.mark.asyncio
async def test_live_failed_compact_now_under_omit_still_reports_the_failure(
    tmp_path: Path,
) -> None:
    """AC#3/#4: 'Omit older context' governs sends, never Compact now's copy."""

    db, _store, controller, gateway = _live_controller(
        tmp_path,
        gateway=_LiveProviderGateway(summary=""),
        overrides=replace(_OMIT, compaction_mode=ContextCompactionMode.OFF),
    )
    for index in range(3):
        await controller.submit_draft(f"question-{index}.", session_id="session-1")

    succeeded, copy = await controller.compact_context_now("session-1")

    assert succeeded is False
    assert gateway.auxiliary_calls == 1
    assert copy.startswith("Compaction failed and nothing changed: "), copy
    assert "1,234 input + 17 output tokens" in copy
    assert [row["failure_reason"] for row in _attempt_rows(db)] == [
        "invalid_summary_output"
    ]


@pytest.mark.asyncio
async def test_live_failed_automatic_compaction_under_omit_is_disclosed_once(
    tmp_path: Path,
) -> None:
    """AC#2/#4: a billed failure the send survives is still disclosed, once."""

    db, store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary=""), overrides=_OMIT
    )

    index, result = await _send_until_compaction(controller, gateway)

    # The message still went out, uncompacted, and got its reply.
    assert result.accepted is True
    assert gateway.stream_calls == index + 1
    notes = _system_rows(store)
    assert len(notes) == 1, notes
    assert "empty, oversized or malformed summary" in notes[0]
    assert "1,234 input + 17 output tokens" in notes[0]
    assert "Your message was sent without compacting" in notes[0]
    assert "Automatic compaction is paused" in notes[0]
    assert "provider request was not sent" not in notes[0]

    # Paused from here on: the next send goes out, is not billed, and does
    # not repeat the note.
    await controller.submit_draft("one more question", session_id="session-1")
    assert gateway.auxiliary_calls == 1
    assert gateway.stream_calls == index + 2
    assert len(_system_rows(store)) == 1
    assert [row["failure_reason"] for row in _attempt_rows(db)] == [
        "invalid_summary_output"
    ]

    # The transcript-only note sits on the live path between turns; a later
    # compaction must still commit against the durable lineage.
    gateway.summary = _SUMMARY
    store.set_session_context_policy_overrides(
        "session-1", replace(_OMIT, summary_max_tokens=120)
    )
    await controller.submit_draft("after the policy change", session_id="session-1")
    assert gateway.auxiliary_calls == 2
    assert [row["status"] for row in _attempt_rows(db)] == ["failed", "succeeded"]
    assert _active_memory_count(db) == 1


@pytest.mark.asyncio
async def test_live_recovery_fallback_keeps_only_the_preflight_blocks_own_copy(
    tmp_path: Path,
) -> None:
    """The settlement fallback keeps the compaction block's copy -- only it.

    Any other BLOCKED state that happens to be current (a trace-capture pause,
    say) must not be shown in place of the generic recovery copy.
    """

    _db, store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary="")
    )
    await _send_until_compaction(controller, gateway)
    recovery = store.dispatch_recovery_for_session("session-1")
    assert recovery is not None
    assert controller.run_state_for("session-1").visible_copy.startswith(
        "Your message was not sent"
    )

    controller._set_run_state(  # an unrelated block, e.g. trace capture
        ConsoleRunState.blocked("Retry, Send without capture, or Cancel."),
        session_id="session-1",
    )
    controller._restore_dispatch_recovery_after_settlement_failure(
        "session-1", recovery.assistant_message_id
    )

    assert controller.run_state_for("session-1").visible_copy == (
        "Response recovery failed. Try again or discard."
    )


@pytest.mark.asyncio
async def test_live_failed_micro_compaction_is_not_rebilled_by_the_next_tick(
    tmp_path: Path,
) -> None:
    """AC#5: a background micro-compaction pass honours the pause too."""

    db, _store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary="")
    )
    # Below the trigger, so a micro pass folds exactly the oldest exchange.
    for index in range(2):
        await controller.submit_draft(f"question-{index}.", session_id="session-1")
    assert gateway.auxiliary_calls == 0

    await controller.compact_context_now("session-1", micro=True)
    assert gateway.auxiliary_calls == 1  # the tick's own billed attempt
    await controller.compact_context_now("session-1", micro=True)

    assert gateway.auxiliary_calls == 1
    assert [row["failure_reason"] for row in _attempt_rows(db)] == [
        "invalid_summary_output"
    ]


def _latest_reply(store: ConsoleChatStore) -> Any:
    return [
        message
        for message in store.messages_for_session("session-1")
        if message.role is ConsoleMessageRole.ASSISTANT
    ][-1]


async def _send_until_a_send_is_held(
    controller: ConsoleChatController,
    gateway: _LiveProviderGateway,
    *,
    first: int,
    limit: int = 12,
) -> None:
    """Send ordinary turns until one is held back at the compaction preflight."""

    for index in range(first, limit):
        streams = gateway.stream_calls
        await controller.submit_draft(
            f"question-{index}: explain step {index} in detail.",
            session_id="session-1",
        )
        if gateway.stream_calls == streams:
            return
    raise AssertionError("the custom budget was never crossed")


@pytest.mark.asyncio
@pytest.mark.parametrize("follow_up", ["micro-tick", "send"])
@pytest.mark.parametrize("micro", [False, True], ids=["compact-now", "micro-tick"])
@pytest.mark.parametrize("edit", [True, False], ids=["edited", "unchanged"])
async def test_live_edit_to_the_latest_exchange_lifts_the_pause(
    tmp_path: Path, micro: bool, edit: bool, follow_up: str
) -> None:
    """AC#5: Compact now and a micro tick fence the WHOLE completed lineage.

    Neither carries an active request, so the latest exchange is history: an
    edit to it lifts their pause like an edit to any earlier turn, for the
    next micro tick and for the next send that crosses the trigger. Without
    the edit (the negative control) that automatic attempt stays paused.
    """

    db, store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary="")
    )
    for index in range(2):
        await controller.submit_draft(f"question-{index}.", session_id="session-1")
    await controller.compact_context_now("session-1", micro=micro)
    assert gateway.auxiliary_calls == 1

    if edit:
        store.update_message_content(
            _latest_reply(store).id, "An edited, much shorter answer."
        )
    if follow_up == "send":
        await _send_until_a_send_is_held(controller, gateway, first=2)
        assert ("automatic compaction is paused" in _system_rows(store)[-1]) is (
            not edit
        )
    else:
        await controller.compact_context_now("session-1", micro=True)

    assert gateway.auxiliary_calls == (2 if edit else 1)
    assert [row["failure_reason"] for row in _attempt_rows(db)] == [
        "invalid_summary_output"
    ] * gateway.auxiliary_calls


@pytest.mark.asyncio
async def test_live_compact_now_beside_an_unsent_turn_keeps_its_retry_paused(
    tmp_path: Path,
) -> None:
    """AC#5: the unsent turn in response recovery is a request, not history.

    A failed send leaves its user turn unanswered; a Compact now made then
    must fence the same history the Retry of that turn does, or the Retry
    would make one more billed summary call.
    """

    _db, store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary="")
    )
    await _send_until_compaction(controller, gateway)
    assert store.dispatch_recovery_for_session("session-1") is not None

    succeeded, _copy = await controller.compact_context_now("session-1")
    assert succeeded is False
    assert gateway.auxiliary_calls == 2  # Compact now always tries

    retried = await controller.retry_dispatch_recovery("session-1")
    assert gateway.auxiliary_calls == 2
    assert "automatic compaction is paused" in retried.visible_copy


@pytest.mark.asyncio
async def test_live_compact_now_beside_an_unsent_turn_keeps_a_fresh_send_paused(
    tmp_path: Path,
) -> None:
    """AC#5: Discard the unsent turn, then send: still the same history.

    Compact now beside the unsent turn fences the history before it, the
    same history a fresh send after Discard has, so that send stays paused.
    Were the unsent turn history, the fresh send would start a different
    request after it and make one more billed summary call.
    """

    _db, store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary="")
    )
    await _send_until_compaction(controller, gateway)
    succeeded, _copy = await controller.compact_context_now("session-1")
    assert succeeded is False
    assert gateway.auxiliary_calls == 2

    discarded = await controller.discard_dispatch_recovery("session-1")
    assert discarded.accepted is True, discarded.visible_copy
    streams = gateway.stream_calls
    await controller.submit_draft("a fresh question", session_id="session-1")

    assert gateway.auxiliary_calls == 2
    assert gateway.stream_calls == streams
    assert "automatic compaction is paused" in _system_rows(store)[-1]


@pytest.mark.asyncio
async def test_live_failed_compact_now_pauses_a_continue_of_the_latest_reply(
    tmp_path: Path,
) -> None:
    """AC#5: Continue resumes the exchange a failed Compact now just fenced.

    Under Omit older context the send that crossed the trigger went out
    uncompacted, so the latest exchange is complete. Compact now's pause
    holds that whole lineage; a Continue's request starts at the latest user
    turn, so its own history is shorter. The pause must still cover it, or
    the Continue makes one more billed summary call.
    """

    db, store, controller, gateway = _live_controller(
        tmp_path, gateway=_LiveProviderGateway(summary=""), overrides=_OMIT
    )
    await _send_until_compaction(controller, gateway)
    assert store.dispatch_recovery_for_session("session-1") is None
    succeeded, _copy = await controller.compact_context_now("session-1")
    assert succeeded is False
    assert gateway.auxiliary_calls == 2

    streams = gateway.stream_calls
    result = await controller.continue_from_message(_latest_reply(store).id)

    assert result.accepted is True, result.visible_copy
    assert gateway.auxiliary_calls == 2
    assert gateway.stream_calls == streams + 1  # Omit: it still goes out
    assert [row["failure_reason"] for row in _attempt_rows(db)] == [
        "invalid_summary_output"
    ] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("edit", [False, True], ids=["unchanged", "reply-edited"])
async def test_live_failed_compact_now_pauses_a_continue_under_stop_and_ask(
    tmp_path: Path, edit: bool
) -> None:
    """AC#5: on the default policy a paused Continue is held, not re-billed.

    The chat is built over the trigger without any send compacting, then
    automatic compaction is switched on and Compact now fails. Continuing
    the latest reply must say the pause holds and make no summary call. An
    edit to that reply (the control) lifts the pause like any other edit, so
    the Continue makes its one billed attempt and is held for that failure.
    """

    _db, store, controller, gateway = _live_controller(
        tmp_path,
        gateway=_LiveProviderGateway(summary=""),
        overrides=replace(_OVERRIDES, compaction_mode=ContextCompactionMode.OFF),
    )
    for index in range(8):
        await controller.submit_draft(
            f"question-{index}: explain step {index} in detail.",
            session_id="session-1",
        )
    store.set_session_context_policy_overrides("session-1", _OVERRIDES)
    succeeded, _copy = await controller.compact_context_now("session-1")
    assert succeeded is False
    assert gateway.auxiliary_calls == 1

    latest = _latest_reply(store)
    if edit:
        store.update_message_content(latest.id, latest.content + " (edited)")
    streams = gateway.stream_calls
    await controller.continue_from_message(latest.id)

    copy = controller.run_state_for("session-1").visible_copy
    assert gateway.stream_calls == streams
    if edit:
        assert gateway.auxiliary_calls == 2
        assert copy.startswith("Your message was not sent"), copy
        assert "automatic compaction is paused" not in copy
    else:
        assert gateway.auxiliary_calls == 1
        assert "automatic compaction is paused" in copy, copy
        assert "No new summary call was made." in copy
