from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.citation_trace_models import (
    AnswerAttempt,
    AnswerAttemptKind,
    CitationCompleteness,
    CitationOccurrence,
    CitationTrace,
    ClaimSupport,
    EvidenceRun,
    EvidenceStorageMode,
    MarkerNamespace,
    PromptEvidenceEntry,
    PromptEvidenceSet,
    StructuralValidationState,
    TraceLifecycle,
    TraceOrigin,
)
from tldw_chatbook.Chat.citation_trace_repository import ActiveCitationTraceState
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_citation_sources_modal import (
    selected_valid_evidence_ordinals,
)


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _occurrence(
    ordinal: int,
    marker_ordinal: int,
    evidence_ordinal: int | None,
    state: StructuralValidationState,
    *,
    claim_support: ClaimSupport = ClaimSupport.NOT_CHECKED,
) -> CitationOccurrence:
    raw_marker = f"[S{marker_ordinal}]"
    start = ordinal * 10
    return CitationOccurrence(
        occurrence_id=f"occurrence-{ordinal}",
        occurrence_ordinal=ordinal,
        raw_marker=raw_marker,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        evidence_ordinal=evidence_ordinal,
        marker_start=start,
        marker_end=start + len(raw_marker),
        structural_state=state,
        claim_support=claim_support,
    )


def _trace(
    *,
    selected_occurrences: tuple[CitationOccurrence, ...] | None = None,
) -> CitationTrace:
    selected_occurrences = (
        (
            _occurrence(
                1,
                2,
                2,
                StructuralValidationState.VALID,
                claim_support=ClaimSupport.UNSUPPORTED,
            ),
            _occurrence(2, 1, 1, StructuralValidationState.VALID),
            _occurrence(3, 2, 2, StructuralValidationState.VALID),
            _occurrence(4, 3, 3, StructuralValidationState.INVALID_SPAN),
            _occurrence(5, 99, None, StructuralValidationState.UNKNOWN_MARKER),
        )
        if selected_occurrences is None
        else selected_occurrences
    )
    run = EvidenceRun(
        run_id="run-1",
        request_id="request-1",
        run_ordinal=1,
        stage="retrieval",
        payload_ref="run-payload-1",
        started_at=NOW,
        ended_at=NOW,
    )
    prompt_set = PromptEvidenceSet(
        prompt_set_id="prompt-1",
        prompt_set_ordinal=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        entries=tuple(
            PromptEvidenceEntry(
                evidence_ordinal=ordinal,
                marker_ordinal=ordinal,
                run_id=run.run_id,
                snapshot_payload_ref=f"snapshot-{ordinal}",
                storage_mode=EvidenceStorageMode.EMBEDDED,
            )
            for ordinal in range(1, 4)
        ),
        created_at=NOW,
    )
    diagnostic_attempt = AnswerAttempt(
        attempt_id="attempt-diagnostic",
        attempt_ordinal=1,
        kind=AnswerAttemptKind.PIPELINE_RERUN,
        prompt_evidence_set_id=prompt_set.prompt_set_id,
        occurrences=(
            _occurrence(6, 3, 3, StructuralValidationState.VALID),
        ),
        created_at=NOW,
    )
    selected_attempt = AnswerAttempt(
        attempt_id="attempt-selected",
        attempt_ordinal=2,
        kind=AnswerAttemptKind.INITIAL,
        prompt_evidence_set_id=prompt_set.prompt_set_id,
        occurrences=selected_occurrences,
        created_at=NOW,
    )
    return CitationTrace(
        trace_id="trace-1",
        request_id="request-1",
        generation_id="generation-1",
        origin=TraceOrigin.LOCAL,
        lifecycle=TraceLifecycle.SEALED,
        completeness_at_seal=CitationCompleteness.COMPLETE,
        evidence_runs=(run,),
        prompt_evidence_sets=(prompt_set,),
        answer_attempts=(diagnostic_attempt, selected_attempt),
        selected_attempt_id=selected_attempt.attempt_id,
        policy_version="policy-1",
        created_at=NOW,
        sealed_at=NOW,
    )


def test_selected_valid_evidence_ordinals_use_only_selected_valid_occurrences() -> None:
    trace = _trace()

    assert selected_valid_evidence_ordinals(trace) == (2, 1)


def test_selected_valid_evidence_ordinals_ignore_legacy_empty_attempt() -> None:
    trace = _trace(selected_occurrences=())

    assert selected_valid_evidence_ordinals(trace) == ()


class _FakeStore:
    def __init__(
        self,
        messages: list[ConsoleChatMessage],
        *,
        session_id: str = "session-1",
    ) -> None:
        self.active_session_id = session_id
        self.messages = messages

    def messages_for_session(self, _session_id: str) -> list[ConsoleChatMessage]:
        return self.messages


class _FakeRepository:
    def __init__(
        self,
        result: object,
        *,
        verified: bool = True,
        on_lookup=None,
        db: object | None = None,
    ) -> None:
        self.result = result
        self.verified = verified
        self.on_lookup = on_lookup
        self.db = db
        self.calls: list[tuple[str, str]] = []
        self.verified_results: list[object] = []

    def get_active_trace_for_current_message(
        self,
        persisted_message_id: str,
        current_body: str,
    ) -> object:
        self.calls.append((persisted_message_id, current_body))
        if self.on_lookup is not None:
            self.on_lookup()
        return self.result

    def verify_active_trace_result(self, result: object) -> bool:
        self.verified_results.append(result)
        return self.verified


def _active_result(trace: CitationTrace) -> SimpleNamespace:
    return SimpleNamespace(
        state=ActiveCitationTraceState.ACTIVE,
        summary=SimpleNamespace(trace=trace),
    )


def _message(
    message_id: str,
    *,
    role: ConsoleMessageRole = ConsoleMessageRole.ASSISTANT,
    status: str = "complete",
    persisted_message_id: str | None = None,
    body: str = "Answer [S1].",
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        id=message_id,
        role=role,
        status=status,
        persisted_message_id=persisted_message_id,
        content=body,
    )


async def _async_noop() -> None:
    return None


def _bare_screen(
    messages: list[ConsoleChatMessage],
    repository: object | None,
    *,
    app_db: object | None = None,
) -> ChatScreen:
    screen = ChatScreen.__new__(ChatScreen)
    screen._console_chat_store = _FakeStore(messages)
    screen._console_citation_counts = {}
    screen._console_citation_input_signature = None
    screen._console_citation_request_generation = 0
    screen._last_native_transcript_refresh_key = None
    screen.app_instance = SimpleNamespace(
        citation_trace_repository=repository,
        chachanotes_db=app_db,
    )
    screen._sync_native_console_transcript_to_legacy_surface = _async_noop
    screen._sync_native_console_chat_ui = _async_noop
    return screen


@pytest.mark.asyncio
async def test_discovery_queries_only_complete_persisted_assistants_with_two_args() -> None:
    eligible = _message("assistant-ok", persisted_message_id="persisted-ok")
    messages = [
        eligible,
        _message(
            "assistant-streaming",
            status="streaming",
            persisted_message_id="persisted-streaming",
        ),
        _message(
            "assistant-pending",
            status="pending",
            persisted_message_id="persisted-pending",
        ),
        _message(
            "assistant-stopped",
            status="stopped",
            persisted_message_id="persisted-stopped",
        ),
        _message(
            "assistant-failed",
            status="failed",
            persisted_message_id="persisted-failed",
        ),
        _message(
            "user",
            role=ConsoleMessageRole.USER,
            persisted_message_id="persisted-user",
        ),
        _message(
            "system",
            role=ConsoleMessageRole.SYSTEM,
            persisted_message_id="persisted-system",
        ),
        _message(
            "tool",
            role=ConsoleMessageRole.TOOL,
            persisted_message_id="persisted-tool",
        ),
        _message("assistant-unpersisted"),
    ]
    repository = _FakeRepository(_active_result(_trace()))
    screen = _bare_screen(messages, repository)
    signature = screen._console_citation_signature(messages)
    screen._console_citation_input_signature = signature
    screen._console_citation_request_generation = 1

    await screen._discover_console_citation_counts(repository, signature, 1)

    assert repository.calls == [("persisted-ok", "Answer [S1].")]
    assert repository.verified_results == [repository.result]
    assert screen._console_citation_counts == {"assistant-ok": 2}
    assert all(type(value) is int for value in screen._console_citation_counts.values())
    assert "trace" not in repr(screen._console_citation_counts).lower()
    assert "snapshot" not in repr(screen._console_citation_counts).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "has_summary", "verified"),
    [
        (ActiveCitationTraceState.ACTIVE, True, False),
        (ActiveCitationTraceState.ACTIVE, False, True),
        (ActiveCitationTraceState.NOT_FOUND, False, True),
        (ActiveCitationTraceState.UNVERIFIABLE, False, True),
        (ActiveCitationTraceState.BODY_MISMATCH, False, True),
    ],
)
async def test_discovery_requires_active_summary_and_repository_verification(
    state: ActiveCitationTraceState,
    has_summary: bool,
    verified: bool,
) -> None:
    result = SimpleNamespace(
        state=state,
        summary=SimpleNamespace(trace=_trace()) if has_summary else None,
    )
    repository = _FakeRepository(result, verified=verified)
    messages = [_message("assistant", persisted_message_id="persisted")]
    screen = _bare_screen(messages, repository)
    signature = screen._console_citation_signature(messages)
    screen._console_citation_input_signature = signature
    screen._console_citation_request_generation = 1

    await screen._discover_console_citation_counts(repository, signature, 1)

    assert screen._console_citation_counts == {}


def test_identical_sync_signature_dispatches_only_one_exclusive_worker() -> None:
    repository = _FakeRepository(_active_result(_trace()))
    messages = [_message("assistant", persisted_message_id="persisted")]
    screen = _bare_screen(messages, repository)
    dispatched: list[tuple[object, dict[str, object]]] = []

    def capture_worker(coroutine, **kwargs):
        dispatched.append((coroutine, kwargs))
        return None

    screen.run_worker = capture_worker

    screen._sync_console_citation_count_discovery(messages)
    screen._sync_console_citation_count_discovery(messages)

    assert len(dispatched) == 1
    assert dispatched[0][1] == {
        "exclusive": True,
        "group": "console-citation-counts",
    }
    dispatched[0][0].close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_change", ["body", "persisted-id", "session", "generation"])
async def test_late_discovery_is_discarded_after_signature_or_generation_change(
    stale_change: str,
) -> None:
    message = _message("assistant", persisted_message_id="persisted")
    screen = _bare_screen([message], None)
    signature = screen._console_citation_signature([message])
    screen._console_citation_input_signature = signature
    screen._console_citation_request_generation = 1

    def make_stale() -> None:
        if stale_change == "body":
            message.content = "Edited answer [S1]."
        elif stale_change == "persisted-id":
            message.persisted_message_id = "persisted-replacement"
        elif stale_change == "session":
            screen._console_chat_store.active_session_id = "session-2"
        else:
            screen._console_citation_request_generation += 1

    repository = _FakeRepository(_active_result(_trace()), on_lookup=make_stale)

    await screen._discover_console_citation_counts(repository, signature, 1)

    assert screen._console_citation_counts == {}


def test_repository_absence_or_database_mismatch_fails_closed() -> None:
    messages = [_message("assistant", persisted_message_id="persisted")]
    for repository, app_db in (
        (None, object()),
        (_FakeRepository(_active_result(_trace()), db=object()), object()),
    ):
        screen = _bare_screen(messages, repository, app_db=app_db)
        screen._console_citation_counts = {"stale": 9}
        dispatched: list[object] = []
        screen.run_worker = lambda coroutine, **_kwargs: dispatched.append(coroutine)

        screen._sync_console_citation_count_discovery(messages)

        assert screen._console_citation_counts == {}
        assert dispatched == []
