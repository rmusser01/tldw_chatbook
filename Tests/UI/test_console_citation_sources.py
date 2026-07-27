from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from io import StringIO
from threading import Event
from types import SimpleNamespace

import pytest
from rich.console import Console as RichConsole
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, ListItem, ListView, Static

from tldw_chatbook.Chat.citation_source_locators import (
    AuthorityScope,
    CitationReadAuthorization,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    LocalCitationIdentityContext,
    local_trace_namespace,
)
from tldw_chatbook.Chat.citation_trace_models import (
    AnswerAttempt,
    AnswerAttemptKind,
    CitationCompleteness,
    CitationOccurrence,
    CitationTrace,
    ClaimSupport,
    EvidenceSnapshotPayload,
    EvidenceRun,
    EvidenceStorageMode,
    MarkerNamespace,
    PromptEvidenceEntry,
    PromptEvidenceSet,
    StructuralValidationState,
    TraceLifecycle,
    TraceOrigin,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    ActiveCitationTraceState,
    CitationHydrationResult,
    CitationHydrationState,
    CitationTraceSummary,
    GovernedCitationPayloads,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID,
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_citation_sources_modal import (
    ConsoleCitationSourceRow,
    ConsoleCitationSourcesModal,
    build_console_citation_source_rows,
    selected_valid_evidence_ordinals,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


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


IDENTITY = LocalCitationIdentityContext(
    profile_id="profile-local",
    local_authority_id="authority-local",
    fingerprint_key_id="fingerprint-key",
)


def _hydration_result(
    *,
    identities: dict[int, dict[str, object]] | None = None,
    omitted_snapshot_ordinals: frozenset[int] = frozenset(),
    snapshot_texts: dict[int, str] | None = None,
    titles: dict[int, str] | None = None,
    state: CitationHydrationState = CitationHydrationState.AUTHORIZED,
) -> CitationHydrationResult:
    trace = _trace()
    identities = identities or {
        1: {"source_kind": "notes", "source_id": "note-1"},
        2: {"source_kind": "media_db", "source_id": "media-2"},
        3: {"source_kind": "chat_history", "source_id": "conversation-3"},
    }
    snapshot_texts = snapshot_texts or {}
    titles = titles or {}
    snapshots = tuple(
        EvidenceSnapshotPayload(
            payload_id=f"snapshot-{ordinal}",
            storage_mode=EvidenceStorageMode.EMBEDDED,
            snapshot_text=snapshot_texts.get(
                ordinal,
                (
                    "[link](https://example.invalid) "
                    "[bold red]literal[/bold red] \x1b[31mANSI-looking\x1b[0m"
                    if ordinal == 2
                    else f"exact snapshot {ordinal}"
                ),
            ),
            title=titles.get(ordinal, f"[cyan]Source {ordinal}[/cyan]"),
            source_identity=identities[ordinal],
        )
        for ordinal in range(1, 4)
        if ordinal not in omitted_snapshot_ordinals
    )
    summary = CitationTraceSummary(
        namespace=local_trace_namespace(IDENTITY, trace_id=trace.trace_id),
        trace=trace,
        visibility_state="active",
    )
    return CitationHydrationResult(
        state=state,
        summary=summary,
        governed_payloads=(
            GovernedCitationPayloads(
                evidence_run_payloads=(),
                evidence_snapshot_payloads=snapshots,
                answer_attempt_payloads=(),
            )
            if state is CitationHydrationState.AUTHORIZED
            else None
        ),
    )


def test_hydrated_rows_join_selected_prompt_entries_in_first_citation_order() -> None:
    result = _hydration_result()

    rows = build_console_citation_source_rows(result)

    assert rows == (
        ConsoleCitationSourceRow(
            display_marker="[S2]",
            evidence_ordinal=2,
            title="[cyan]Source 2[/cyan]",
            snapshot_text=(
                "[link](https://example.invalid) "
                "[bold red]literal[/bold red] \x1b[31mANSI-looking\x1b[0m"
            ),
            source_kind="media_db",
            source_id="media-2",
            open_source_type="media",
        ),
        ConsoleCitationSourceRow(
            display_marker="[S1]",
            evidence_ordinal=1,
            title="[cyan]Source 1[/cyan]",
            snapshot_text="exact snapshot 1",
            source_kind="notes",
            source_id="note-1",
            open_source_type="notes",
        ),
    )


def test_hydrated_rows_return_unavailable_instead_of_partial_graph() -> None:
    result = _hydration_result(omitted_snapshot_ordinals=frozenset({1}))

    assert build_console_citation_source_rows(result) is None


@pytest.mark.parametrize(
    ("identity", "expected_kind", "expected_id", "expected_open_type"),
    [
        (
            {"source_kind": "web_content", "source_id": "web-1"},
            "web_content",
            "web-1",
            None,
        ),
        ({"source_kind": 7, "source_id": "media-2"}, None, "media-2", None),
        ({"source_kind": "media_db", "source_id": ""}, "media_db", None, None),
        (
            {"source_kind": "media_db", "source_id": "x" * 257},
            "media_db",
            None,
            None,
        ),
        (
            {
                "source_kind": "media_db",
                "source_id": " https://example.invalid/2 ",
            },
            "media_db",
            " https://example.invalid/2 ",
            "media",
        ),
        (
            {"source_kind": "media_db", "source_id": "../private\\2"},
            "media_db",
            "../private\\2",
            "media",
        ),
        (
            {"source_kind": "media_db", "source_id": "media-\n2"},
            "media_db",
            "media-\n2",
            "media",
        ),
        (
            {"source_kind": " media_db ", "source_id": "media-2"},
            " media_db ",
            "media-2",
            None,
        ),
    ],
)
def test_identity_values_are_exact_bounded_strings_and_mapping_is_static(
    identity: dict[str, object],
    expected_kind: str | None,
    expected_id: str | None,
    expected_open_type: str | None,
) -> None:
    result = _hydration_result(
        identities={
            1: {"source_kind": "notes", "source_id": "note-1"},
            2: identity,
            3: {"source_kind": "chat_history", "source_id": "conversation-3"},
        }
    )

    rows = build_console_citation_source_rows(result)

    assert rows is not None
    assert rows[0].snapshot_text.startswith("[link]")
    assert rows[0].source_kind == expected_kind
    assert rows[0].source_id == expected_id
    assert rows[0].open_source_type == expected_open_type


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
    screen._console_citation_resolved_signatures = {}
    screen._console_citation_input_signature = None
    screen._console_citation_repository_token = None
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

    assert screen._console_citation_counts == {"assistant": 0}


def test_stable_repository_and_identical_signature_dispatch_only_one_worker() -> None:
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
    app_db = object()
    stale_applied = False

    def make_stale() -> None:
        nonlocal stale_applied
        stale_applied = True
        if stale_change == "body":
            message.content = "Edited answer [S1]."
        elif stale_change == "persisted-id":
            message.persisted_message_id = "persisted-replacement"
        elif stale_change == "session":
            screen._console_chat_store.active_session_id = "session-2"
        else:
            screen._console_citation_request_generation += 1

    repository = _FakeRepository(
        _active_result(_trace()),
        on_lookup=make_stale,
        db=app_db,
    )
    screen = _bare_screen([message], repository, app_db=app_db)
    signature = screen._console_citation_signature([message])
    screen._console_citation_input_signature = signature
    screen._console_citation_request_generation = 1

    await screen._discover_console_citation_counts(repository, signature, 1)

    assert repository.calls == [("persisted", "Answer [S1].")]
    assert stale_applied
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


class _PerMessageRepository:
    def __init__(
        self,
        results: dict[str, object],
        *,
        lookup_error_id: str | None = None,
        verify_error_result: object | None = None,
    ) -> None:
        self.results = results
        self.lookup_error_id = lookup_error_id
        self.verify_error_result = verify_error_result
        self.calls: list[tuple[str, str]] = []

    def get_active_trace_for_current_message(
        self,
        persisted_message_id: str,
        current_body: str,
    ) -> object:
        self.calls.append((persisted_message_id, current_body))
        if persisted_message_id == self.lookup_error_id:
            raise RuntimeError("sensitive repository lookup failure")
        return self.results[persisted_message_id]

    def verify_active_trace_result(self, result: object) -> bool:
        if result is self.verify_error_result:
            raise RuntimeError("sensitive repository verification failure")
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize("raising_boundary", ["lookup", "verify"])
async def test_repository_error_isolated_to_message_and_other_count_still_resolves(
    raising_boundary: str,
) -> None:
    bad_result = _active_result(_trace())
    good_result = _active_result(_trace())
    repository = _PerMessageRepository(
        {
            "persisted-bad": bad_result,
            "persisted-good": good_result,
        },
        lookup_error_id=(
            "persisted-bad" if raising_boundary == "lookup" else None
        ),
        verify_error_result=bad_result if raising_boundary == "verify" else None,
    )
    messages = [
        _message("assistant-bad", persisted_message_id="persisted-bad"),
        _message("assistant-good", persisted_message_id="persisted-good"),
    ]
    screen = _bare_screen(messages, repository)
    signature = screen._console_citation_signature(messages)
    screen._console_citation_input_signature = signature
    screen._console_citation_request_generation = 1

    try:
        await screen._discover_console_citation_counts(repository, signature, 1)
    except RuntimeError as error:
        pytest.fail(f"best-effort footer discovery leaked repository error: {error}")

    assert repository.calls == [
        ("persisted-bad", "Answer [S1]."),
        ("persisted-good", "Answer [S1]."),
    ]
    assert screen._console_citation_counts == {
        "assistant-bad": 0,
        "assistant-good": 2,
    }


async def _dispatch_and_run(
    screen: ChatScreen,
    messages: list[ConsoleChatMessage],
) -> dict[str, object]:
    dispatched: list[tuple[object, dict[str, object]]] = []
    screen.run_worker = lambda coroutine, **kwargs: dispatched.append(
        (coroutine, kwargs)
    )

    screen._sync_console_citation_count_discovery(messages)

    assert len(dispatched) == 1
    coroutine, kwargs = dispatched[0]
    await coroutine
    return kwargs


@pytest.mark.asyncio
async def test_new_eligible_message_queries_only_new_entry_after_history_resolves() -> None:
    repository = _FakeRepository(_active_result(_trace()))
    messages = [
        _message("assistant-1", persisted_message_id="persisted-1"),
        _message("assistant-2", persisted_message_id="persisted-2"),
    ]
    screen = _bare_screen(messages, repository)

    await _dispatch_and_run(screen, messages)
    historical_calls = list(repository.calls)

    messages.append(_message("assistant-3", persisted_message_id="persisted-3"))
    await _dispatch_and_run(screen, messages)

    assert historical_calls == [
        ("persisted-1", "Answer [S1]."),
        ("persisted-2", "Answer [S1]."),
    ]
    assert repository.calls[len(historical_calls) :] == [
        ("persisted-3", "Answer [S1].")
    ]
    assert screen._console_citation_counts == {
        "assistant-1": 2,
        "assistant-2": 2,
        "assistant-3": 2,
    }


def _seed_resolved_counts(
    screen: ChatScreen,
    messages: list[ConsoleChatMessage],
    counts: dict[str, int],
) -> None:
    signature = screen._console_citation_signature(messages)
    screen._console_citation_input_signature = signature
    screen._console_citation_repository_token = (
        screen._console_citation_repository_readiness()[0]
    )
    screen._console_citation_counts = dict(counts)
    screen._console_citation_resolved_signatures = {
        item[0]: item for item in signature[1]
    }


def test_changed_and_removed_entries_clear_only_their_own_cached_count() -> None:
    messages = [
        _message("assistant-1", persisted_message_id="persisted-1"),
        _message("assistant-2", persisted_message_id="persisted-2"),
    ]
    repository = _FakeRepository(_active_result(_trace()))
    changed_screen = _bare_screen(messages, repository)
    _seed_resolved_counts(
        changed_screen,
        messages,
        {"assistant-1": 2, "assistant-2": 1},
    )
    changed_workers: list[object] = []
    changed_screen.run_worker = lambda coroutine, **_kwargs: changed_workers.append(
        coroutine
    )

    messages[0].content = "Changed answer [S1]."
    changed_screen._sync_console_citation_count_discovery(messages)

    assert changed_screen._console_citation_counts == {"assistant-2": 1}
    assert len(changed_workers) == 1
    changed_workers[0].close()

    removed_messages = [
        _message("assistant-1", persisted_message_id="persisted-1"),
        _message("assistant-2", persisted_message_id="persisted-2"),
    ]
    removed_screen = _bare_screen(removed_messages, repository)
    _seed_resolved_counts(
        removed_screen,
        removed_messages,
        {"assistant-1": 2, "assistant-2": 1},
    )
    removed_workers: list[object] = []
    removed_screen.run_worker = lambda coroutine, **_kwargs: removed_workers.append(
        coroutine
    )

    removed_screen._console_chat_store.messages = removed_messages[1:]
    removed_screen._sync_console_citation_count_discovery(removed_messages[1:])

    assert removed_screen._console_citation_counts == {"assistant-2": 1}
    assert removed_workers == []


@pytest.mark.asyncio
async def test_zero_result_is_cached_and_not_requeried_on_unrelated_changes() -> None:
    not_found = SimpleNamespace(
        state=ActiveCitationTraceState.NOT_FOUND,
        summary=None,
    )
    repository = _PerMessageRepository(
        {
            "persisted-uncited": not_found,
            "persisted-2": _active_result(_trace()),
            "persisted-3": _active_result(_trace()),
        }
    )
    messages = [
        _message("assistant-uncited", persisted_message_id="persisted-uncited"),
    ]
    screen = _bare_screen(messages, repository)

    await _dispatch_and_run(screen, messages)
    assert screen._console_citation_counts == {"assistant-uncited": 0}

    messages.append(_message("assistant-2", persisted_message_id="persisted-2"))
    await _dispatch_and_run(screen, messages)
    messages.append(_message("assistant-3", persisted_message_id="persisted-3"))
    await _dispatch_and_run(screen, messages)

    assert [call[0] for call in repository.calls] == [
        "persisted-uncited",
        "persisted-2",
        "persisted-3",
    ]
    transcript = ConsoleTranscript()
    transcript.set_messages(messages)
    transcript.set_citation_counts(screen._console_citation_counts)
    citation_row_ids = {
        row.message.id
        for row in transcript._transcript_rows()
        if row.kind == "citations" and row.message is not None
    }
    assert citation_row_ids == {"assistant-2", "assistant-3"}


@pytest.mark.asyncio
async def test_replacement_worker_includes_unchanged_entry_still_unresolved() -> None:
    repository = _FakeRepository(_active_result(_trace()))
    messages = [_message("assistant-1", persisted_message_id="persisted-1")]
    screen = _bare_screen(messages, repository)
    dispatched: list[object] = []
    screen.run_worker = lambda coroutine, **_kwargs: dispatched.append(coroutine)

    screen._sync_console_citation_count_discovery(messages)
    messages.append(_message("assistant-2", persisted_message_id="persisted-2"))
    screen._sync_console_citation_count_discovery(messages)

    assert len(dispatched) == 2
    dispatched[0].close()
    await dispatched[1]
    assert repository.calls == [
        ("persisted-1", "Answer [S1]."),
        ("persisted-2", "Answer [S1]."),
    ]


class _MountedTranscriptHarness(App):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


@pytest.mark.asyncio
async def test_existing_focused_sources_row_stays_mounted_for_unrelated_new_entry() -> None:
    historical = _message("assistant-1", persisted_message_id="persisted-1")
    messages = [historical]
    screen = _bare_screen(messages, _FakeRepository(_active_result(_trace())))
    _seed_resolved_counts(screen, messages, {"assistant-1": 2})
    dispatched: list[object] = []
    screen.run_worker = lambda coroutine, **_kwargs: dispatched.append(coroutine)
    app = _MountedTranscriptHarness()

    async with app.run_test() as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        transcript.set_citation_counts(screen._console_citation_counts)
        await transcript.refresh_messages()
        existing_button = transcript.query_one(
            "#console-citation-sources-assistant-1"
        )
        existing_button.focus()
        await pilot.pause()
        assert existing_button.has_focus

        messages.append(_message("assistant-2", persisted_message_id="persisted-2"))
        screen._sync_console_citation_count_discovery(messages)
        transcript.set_messages(messages)
        transcript.set_citation_counts(screen._console_citation_counts)
        await transcript.refresh_messages()
        await pilot.pause()

        matching = list(
            transcript.query("#console-citation-sources-assistant-1")
        )
        assert matching == [existing_button]
        assert existing_button.has_focus

    assert len(dispatched) == 1
    dispatched[0].close()


@pytest.mark.asyncio
async def test_inflight_result_is_discarded_after_repository_becomes_missing() -> None:
    app_db = object()
    repository = _FakeRepository(_active_result(_trace()), db=app_db)
    messages = [_message("assistant", persisted_message_id="persisted")]
    screen = _bare_screen(messages, repository, app_db=app_db)
    dispatched: list[object] = []
    screen.run_worker = lambda coroutine, **_kwargs: dispatched.append(coroutine)

    screen._sync_console_citation_count_discovery(messages)
    assert len(dispatched) == 1

    screen.app_instance.citation_trace_repository = None
    await dispatched[0]

    assert screen._console_citation_counts == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("repository_change", ["db-mismatch", "replacement"])
async def test_inflight_result_is_discarded_after_repository_identity_change(
    repository_change: str,
) -> None:
    app_db = object()
    repository = _FakeRepository(_active_result(_trace()), db=app_db)
    messages = [_message("assistant", persisted_message_id="persisted")]
    screen = _bare_screen(messages, repository, app_db=app_db)
    dispatched: list[object] = []
    screen.run_worker = lambda coroutine, **_kwargs: dispatched.append(coroutine)

    screen._sync_console_citation_count_discovery(messages)
    assert len(dispatched) == 1

    if repository_change == "db-mismatch":
        screen.app_instance.chachanotes_db = object()
    else:
        screen.app_instance.citation_trace_repository = _FakeRepository(
            _active_result(_trace()),
            db=app_db,
        )
    await dispatched[0]

    assert screen._console_citation_counts == {}


@pytest.mark.asyncio
async def test_same_transcript_discovers_when_missing_repository_later_appears() -> None:
    app_db = object()
    messages = [_message("assistant", persisted_message_id="persisted")]
    screen = _bare_screen(messages, None, app_db=app_db)
    dispatched: list[object] = []
    screen.run_worker = lambda coroutine, **_kwargs: dispatched.append(coroutine)

    screen._sync_console_citation_count_discovery(messages)
    assert dispatched == []

    repository = _FakeRepository(_active_result(_trace()), db=app_db)
    screen.app_instance.citation_trace_repository = repository
    screen._sync_console_citation_count_discovery(messages)

    assert len(dispatched) == 1
    await dispatched[0]
    assert repository.calls == [("persisted", "Answer [S1].")]
    assert screen._console_citation_counts == {"assistant": 2}


@pytest.mark.asyncio
async def test_valid_repository_replacement_invalidates_and_requeries_same_transcript() -> None:
    app_db = object()
    first_repository = _FakeRepository(_active_result(_trace()), db=app_db)
    messages = [_message("assistant", persisted_message_id="persisted")]
    screen = _bare_screen(messages, first_repository, app_db=app_db)

    await _dispatch_and_run(screen, messages)
    assert screen._console_citation_counts == {"assistant": 2}

    replacement = _FakeRepository(_active_result(_trace()), db=app_db)
    screen.app_instance.citation_trace_repository = replacement
    dispatched: list[object] = []
    screen.run_worker = lambda coroutine, **_kwargs: dispatched.append(coroutine)
    screen._sync_console_citation_count_discovery(messages)

    assert screen._console_citation_counts == {}
    assert screen._console_citation_resolved_signatures == {}
    assert len(dispatched) == 1
    await dispatched[0]
    assert replacement.calls == [("persisted", "Answer [S1].")]
    assert screen._console_citation_counts == {"assistant": 2}


class _HydrationRepository:
    def __init__(
        self,
        hydration: CitationHydrationResult,
        *,
        db: object,
        hydrate_error: bool = False,
        hydration_started: Event | None = None,
        hydration_release: Event | None = None,
    ) -> None:
        self.db = db
        self.identity_context = IDENTITY
        self.hydration = hydration
        self.hydrate_error = hydrate_error
        self.hydration_started = hydration_started
        self.hydration_release = hydration_release
        self.active = SimpleNamespace(
            state=ActiveCitationTraceState.ACTIVE,
            summary=hydration.summary,
        )
        self.events: list[str] = []
        self.lookup_calls: list[tuple[str, str]] = []
        self.hydrate_calls: list[tuple[object, CitationReadAuthorization]] = []
        self.verified_results: list[object] = []

    def get_active_trace_for_current_message(
        self,
        persisted_message_id: str,
        current_body: str,
    ) -> object:
        self.events.append("lookup")
        self.lookup_calls.append((persisted_message_id, current_body))
        return self.active

    def verify_active_trace_result(self, result: object) -> bool:
        self.events.append("verify")
        self.verified_results.append(result)
        return result is self.active

    def hydrate_trace(
        self,
        namespace: object,
        *,
        authorization: CitationReadAuthorization,
    ) -> CitationHydrationResult:
        self.events.append("hydrate")
        self.hydrate_calls.append((namespace, authorization))
        if self.hydration_started is not None:
            self.hydration_started.set()
        if self.hydration_release is not None:
            self.hydration_release.wait(timeout=5)
        if self.hydrate_error:
            raise RuntimeError("private chunk text and source identity")
        return self.hydration


class _CitationHarnessApp(App):
    def __init__(
        self,
        screen: ChatScreen,
        repository: _HydrationRepository,
        db: object,
        messages: list[ConsoleChatMessage],
        citation_counts: dict[str, int],
    ) -> None:
        super().__init__()
        self.test_screen = screen
        self.citation_trace_repository = repository
        self.chachanotes_db = db
        self.seen_navigation: list[tuple[str, dict[str, object]]] = []
        self.transcript = ConsoleTranscript(id="console-native-transcript")
        self.transcript.set_messages(messages)
        self.transcript.set_citation_counts(citation_counts)
        screen.app_instance = self

    def compose(self) -> ComposeResult:
        yield self.transcript

    async def on_mount(self) -> None:
        await self.transcript.refresh_messages()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("console-transcript-citation-sources"):
            ChatScreen.handle_console_citation_sources(self.test_screen, event)

    def on_navigate_to_screen(self, event) -> None:
        self.seen_navigation.append(
            (event.screen_name, dict(event.screen_context or {}))
        )


def _citation_harness(
    *,
    hydration: CitationHydrationResult | None = None,
    hydrate_error: bool = False,
    hydration_started: Event | None = None,
    hydration_release: Event | None = None,
) -> tuple[
    _CitationHarnessApp,
    ChatScreen,
    _HydrationRepository,
    ConsoleChatMessage,
]:
    db = object()
    message = _message(
        "assistant-1",
        persisted_message_id="persisted-1",
        body="Answer [S2] and [S1].",
    )
    repository = _HydrationRepository(
        hydration or _hydration_result(),
        db=db,
        hydrate_error=hydrate_error,
        hydration_started=hydration_started,
        hydration_release=hydration_release,
    )
    screen = ChatScreen.__new__(ChatScreen)
    Screen.__init__(screen)
    screen._console_chat_store = _FakeStore([message])
    screen._console_citation_counts = {"assistant-1": 2}
    screen._console_citation_request_generation = 1
    app = _CitationHarnessApp(
        screen,
        repository,
        db,
        [message],
        {"assistant-1": 2},
    )
    return app, screen, repository, message


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 24), (140, 50)])
async def test_footer_activation_opens_same_lazy_modal_at_narrow_and_wide_sizes(
    size: tuple[int, int],
) -> None:
    app, _screen, repository, _message = _citation_harness()

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        assert repository.hydrate_calls == []

        await pilot.click("#console-citation-sources-assistant-1")
        await pilot.pause(0.1)

        assert isinstance(app.screen, ConsoleCitationSourcesModal)
        assert len(repository.hydrate_calls) == 1
        assert len(app.screen.query("#console-citation-source-list")) == 1
        assert len(app.screen.query("#console-citation-source-detail")) == 1


@pytest.mark.asyncio
async def test_modal_authorization_and_revalidation_are_exact_and_ordered() -> None:
    app, _screen, repository, _message = _citation_harness()

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#console-citation-sources-assistant-1")
        await pilot.pause(0.1)

        assert repository.events == ["lookup", "verify", "hydrate", "verify"]
        assert repository.verified_results == [repository.active, repository.active]
        assert all(item is repository.active for item in repository.verified_results)
        namespace, authorization = repository.hydrate_calls[0]
        assert namespace is repository.active.summary.namespace
        assert authorization.model_dump() == {
            "schema_version": 1,
            "authority_scope": AuthorityScope.LOCAL_PROFILE,
            "profile_id": "profile-local",
            "authenticated_tenant_id": None,
            "governance_scope_id": "profile-local",
            "allowlisted_authority_ids": ("authority-local",),
            "view_snapshot": True,
            "view_source_identity": True,
            "resolve_current": False,
            "open_native": False,
            "open_external": False,
            "compare": False,
            "refresh_observation": False,
            "export": False,
        }


@pytest.mark.asyncio
async def test_modal_render_output_neutralizes_terminal_controls_only() -> None:
    raw_title = (
        "[cyan]\x1b]8;;https://example.invalid\x07Source"
        "\x1b]8;;\x07[/cyan]\r"
    )
    raw_chunk = (
        "[link](https://example.invalid) [bold red]literal[/bold red] Ω\n"
        "\tCSI=\x1b[31mred\x1b[0m OSC=\x1b]8;;https://example.invalid\x07link"
        "\x1b]8;;\x07 BEL=\x07 CR=\r NUL=\x00 DEL=\x7f C1=\x9b31m"
    )
    hydration = _hydration_result(
        snapshot_texts={2: raw_chunk},
        titles={2: raw_title},
    )
    app, _screen, _repository, _message = _citation_harness(hydration=hydration)

    def rich_output(renderable: object) -> str:
        stream = StringIO()
        console = RichConsole(
            file=stream,
            force_terminal=True,
            color_system=None,
            width=500,
        )
        console.print(renderable, end="")
        return stream.getvalue()

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#console-citation-sources-assistant-1")
        await pilot.pause(0.1)

        modal = app.screen
        assert isinstance(modal, ConsoleCitationSourcesModal)
        detail = modal.query_one("#console-citation-source-chunk", Static)
        title = modal.query_one("#console-citation-source-title", Static)
        source_list = modal.query_one("#console-citation-source-list", ListView)
        list_label = source_list.query_one(ListItem).query_one(Static)

        assert modal.display_rows[0].snapshot_text == raw_chunk
        assert modal.display_rows[0].title == raw_title
        assert detail.renderable.spans == []
        assert title.renderable.spans == []
        assert "\n\t" in detail.renderable.plain
        assert len(modal.query("Markdown")) == 0

        rendered_detail = rich_output(detail.renderable)
        rendered_title = rich_output(title.renderable)
        rendered_list_label = rich_output(list_label.renderable)
        aggregate = rendered_detail + rendered_title + rendered_list_label
        for control in ("\x00", "\x07", "\x0d", "\x1b", "\x7f", "\x9b"):
            assert control not in aggregate
        for visible_escape in (
            "\\x00",
            "\\x07",
            "\\x0d",
            "\\x1b",
            "\\x7f",
            "\\x9b",
        ):
            assert visible_escape in aggregate
        assert "Ω" in aggregate
        assert "[link](https://example.invalid)" in aggregate
        assert "[bold red]literal[/bold red]" in aggregate

        assert len(source_list.query(ListItem)) == 2
        source_list.index = 1
        await pilot.pause()
        assert detail.renderable.plain == "exact snapshot 1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_kind", "source_id", "expected_type"),
    [
        ("media_db", "media-exact", "media"),
        ("notes", "note-exact", "notes"),
        ("chat_history", "conversation-exact", "conversations"),
    ],
)
async def test_citation_open_source_returns_exact_library_navigation_context(
    source_kind: str,
    source_id: str,
    expected_type: str,
) -> None:
    hydration = _hydration_result(
        identities={
            1: {"source_kind": "notes", "source_id": "note-1"},
            2: {"source_kind": source_kind, "source_id": source_id},
            3: {"source_kind": "chat_history", "source_id": "conversation-3"},
        }
    )
    app, _screen, _repository, _message = _citation_harness(hydration=hydration)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#console-citation-sources-assistant-1")
        await pilot.pause(0.1)

        modal = app.screen
        assert isinstance(modal, ConsoleCitationSourcesModal)
        open_button = modal.query_one("#console-citation-source-open", Button)
        assert open_button.display is True

        await pilot.click("#console-citation-source-open")
        await pilot.pause()

        assert not isinstance(app.screen, ConsoleCitationSourcesModal)
        assert app.seen_navigation == [
            (
                "library",
                {
                    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE: expected_type,
                    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID: source_id,
                },
            )
        ]
        assert set(app.seen_navigation[0][1]) == {
            LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE,
            LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID,
        }


@pytest.mark.asyncio
async def test_citation_unsupported_source_renders_no_open_source_action() -> None:
    hydration = _hydration_result(
        identities={
            1: {"source_kind": "notes", "source_id": "note-1"},
            2: {"source_kind": "web_content", "source_id": "web-2"},
            3: {"source_kind": "chat_history", "source_id": "conversation-3"},
        }
    )
    app, _screen, _repository, _message = _citation_harness(hydration=hydration)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#console-citation-sources-assistant-1")
        await pilot.pause(0.1)

        modal = app.screen
        assert isinstance(modal, ConsoleCitationSourcesModal)
        open_button = modal.query_one("#console-citation-source-open", Button)

        assert modal.display_rows[0].open_source_type is None
        assert open_button.display is False
        assert open_button.disabled is True
        assert app.seen_navigation == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hydration",
    [
        _hydration_result(state=CitationHydrationState.PROFILE_DENIED),
        _hydration_result(omitted_snapshot_ordinals=frozenset({1})),
    ],
)
async def test_denied_or_incomplete_hydration_shows_one_unavailable_state(
    hydration: CitationHydrationResult,
) -> None:
    app, _screen, _repository, _message = _citation_harness(hydration=hydration)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#console-citation-sources-assistant-1")
        await pilot.pause(0.1)

        modal = app.screen
        assert isinstance(modal, ConsoleCitationSourcesModal)
        states = [
            widget
            for widget in modal.query(".console-citation-sources-state")
            if isinstance(widget, Static) and widget.renderable == "Sources unavailable"
        ]
        assert len(states) == 1
        assert len(modal.query_one("#console-citation-source-list", ListView).children) == 0


@pytest.mark.asyncio
async def test_hydration_exception_shows_content_free_unavailable_state() -> None:
    app, _screen, _repository, _message = _citation_harness(hydrate_error=True)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#console-citation-sources-assistant-1")
        await pilot.pause(0.1)

        modal = app.screen
        assert isinstance(modal, ConsoleCitationSourcesModal)
        state = modal.query_one("#console-citation-sources-state", Static)
        assert state.renderable == "Sources unavailable"
        assert "private" not in str(state.renderable).lower()


@pytest.mark.asyncio
async def test_dismissed_modal_discards_late_hydration() -> None:
    started = Event()
    release = Event()
    app, _screen, repository, _message = _citation_harness(
        hydration_started=started,
        hydration_release=release,
    )
    modal: ConsoleCitationSourcesModal | None = None

    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_screen = app.screen
            await pilot.click("#console-citation-sources-assistant-1")
            await pilot.pause(0.1)
            assert started.is_set()
            assert isinstance(app.screen, ConsoleCitationSourcesModal)
            modal = app.screen

            await pilot.press("escape")
            await pilot.pause()
            release.set()
            await pilot.pause(0.1)

            assert app.screen is initial_screen
            assert modal.display_rows == ()
            assert len(repository.hydrate_calls) == 1
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_change", ["body", "database"])
async def test_message_or_database_change_discards_late_hydration(
    stale_change: str,
) -> None:
    started = Event()
    release = Event()
    app, _screen, _repository, message = _citation_harness(
        hydration_started=started,
        hydration_release=release,
    )

    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#console-citation-sources-assistant-1")
            await pilot.pause(0.1)
            assert started.is_set()
            modal = app.screen
            assert isinstance(modal, ConsoleCitationSourcesModal)

            if stale_change == "body":
                message.content = "Changed body [S2]."
            else:
                app.chachanotes_db = object()
            release.set()
            await pilot.pause(0.1)

            assert modal.display_rows == ()
            assert modal.query_one(
                "#console-citation-source-chunk", Static
            ).renderable.plain == ""
    finally:
        release.set()


@pytest.mark.asyncio
async def test_message_change_during_real_list_extend_discards_mounted_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hydration_started = Event()
    hydration_release = Event()
    app, _screen, _repository, message = _citation_harness(
        hydration_started=hydration_started,
        hydration_release=hydration_release,
    )
    extend_completed = asyncio.Event()
    extend_release = asyncio.Event()
    original_extend = ListView.extend

    def gated_extend(list_view: ListView, items):
        real_mount = original_extend(list_view, items)

        async def mount_then_pause():
            await real_mount
            extend_completed.set()
            await extend_release.wait()

        return mount_then_pause()

    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#console-citation-sources-assistant-1")
            await pilot.pause(0.1)
            assert hydration_started.is_set()
            modal = app.screen
            assert isinstance(modal, ConsoleCitationSourcesModal)
            source_list = modal.query_one("#console-citation-source-list", ListView)
            monkeypatch.setattr(ListView, "extend", gated_extend)

            hydration_release.set()
            await asyncio.wait_for(extend_completed.wait(), timeout=2)
            message.content = "Changed while source rows were mounting [S2]."
            extend_release.set()
            await pilot.pause(0.1)

            assert modal.display_rows == ()
            assert len(source_list.children) == 0
            assert (
                modal.query_one(
                    "#console-citation-source-title", Static
                ).renderable.plain
                == ""
            )
            assert (
                modal.query_one(
                    "#console-citation-source-chunk", Static
                ).renderable.plain
                == ""
            )
    finally:
        hydration_release.set()
        extend_release.set()


@pytest.mark.asyncio
async def test_screen_and_transcript_do_not_cache_governed_payloads_or_chunk_body() -> None:
    app, screen, _repository, _message = _citation_harness()
    exact_chunk = _hydration_result().governed_payloads.evidence_snapshot_payloads[
        1
    ].snapshot_text

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#console-citation-sources-assistant-1")
        await pilot.pause(0.1)

        assert exact_chunk not in {
            value for value in vars(screen).values() if isinstance(value, str)
        }
        assert exact_chunk not in {
            value
            for value in vars(app.transcript).values()
            if isinstance(value, str)
        }
        assert not any(
            isinstance(value, (CitationHydrationResult, GovernedCitationPayloads))
            for value in vars(screen).values()
        )
        assert not any(
            isinstance(value, (CitationHydrationResult, GovernedCitationPayloads))
            for value in vars(app.transcript).values()
        )
