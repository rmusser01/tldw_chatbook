"""Trace-provenance recovery coverage for durable first sends."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsoleSubmitResult,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_prompt_queue_coordinator import _PromptChain
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleTraceCaptureMode,
    TraceProvenancePersistenceError,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
    preparation_actions,
)


def _force_capture_on(monkeypatch: pytest.MonkeyPatch) -> None:
    original_preparation = controller_module.ConsoleTurnPreparation

    def capture_on_preparation(**kwargs: Any):
        kwargs["capture_mode"] = ConsoleTraceCaptureMode.CAPTURE_ON
        return original_preparation(**kwargs)

    monkeypatch.setattr(
        controller_module,
        "ConsoleTurnPreparation",
        capture_on_preparation,
    )


def _record_provider_entries(
    controller: ConsoleChatController,
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    async def stream(**kwargs: Any) -> ConsoleSubmitResult:
        before_provider_dispatch = kwargs["before_provider_dispatch"]
        assert callable(before_provider_dispatch)
        await before_provider_dispatch()
        entries.append(kwargs)
        return ConsoleSubmitResult(True, True)

    monkeypatch.setattr(controller, "_stream_assistant_response", stream)
    return entries


def _fail_trace_request_once(
    failure_kind: str,
    *,
    store: ConsoleChatStore,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], None]:
    persistence = store.persistence
    assert persistence is not None
    repository = persistence.console_trace_repository
    assert repository is not None
    if failure_kind == "missing_repository":
        service_type = type(persistence)
        repository_property = service_type.console_trace_repository
        missing = True

        def restore() -> None:
            nonlocal missing
            missing = False

        monkeypatch.setattr(
            service_type,
            "console_trace_repository",
            property(
                lambda owner: (
                    None
                    if missing and owner is persistence
                    else repository_property.__get__(owner, service_type)
                )
            ),
        )
        return restore
    if failure_kind == "admission":
        original_ensure = repository.ensure_policy
        failed = False

        def fail_once(*args: Any, **kwargs: Any):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("PRIVATE-TRACE-ADMISSION-FAILURE")
            return original_ensure(*args, **kwargs)

        monkeypatch.setattr(repository, "ensure_policy", fail_once)
        return lambda: None
    if failure_kind == "build":
        original_build = controller_module.build_console_request_for_preparation
        failed = False

        def fail_once(*args: Any, **kwargs: Any):
            nonlocal failed
            if not failed:
                failed = True
                raise TraceProvenancePersistenceError()
            return original_build(*args, **kwargs)

        monkeypatch.setattr(
            controller_module,
            "build_console_request_for_preparation",
            fail_once,
        )
        return lambda: None
    raise AssertionError(f"unknown failure kind: {failure_kind}")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("missing_repository", "admission", "build"))
async def test_postcommit_trace_provenance_failure_pauses_and_retry_reuses_frozen_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)
    restore_failure = _fail_trace_request_once(
        failure_kind, store=store, monkeypatch=monkeypatch
    )

    first = await controller.submit_draft("secret durable body", session_id="session-1")

    paused = store.preparation_for_session("session-1")
    assert first.accepted is True
    assert first.provider_started is False
    assert paused is not None
    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_PROVENANCE
    assert preparation_actions(paused) == ("retry", "send_without_capture", "cancel")
    continuation = controller._durable_postcommit_continuations[paused.preparation_id]
    frozen_fingerprint = continuation.fingerprint
    frozen_messages = tuple(dict(row) for row in continuation.provider_messages)
    assert entries == []
    assert db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    checkpoint = db.get_connection().execute(
        "SELECT * FROM console_dispatch_checkpoints"
    ).fetchone()
    assert checkpoint is not None
    assert "secret durable body" not in repr(dict(checkpoint))

    restore_failure()
    retried = await controller.retry_library_preparation(paused.preparation_id)

    assert retried.accepted is True
    assert len(entries) == 1
    assert entries[0]["capture_mode_override"] is ConsoleTraceCaptureMode.CAPTURE_ON
    assert entries[0]["trace_request"] is not None
    assert continuation.fingerprint == frozen_fingerprint
    assert tuple(dict(row) for row in continuation.provider_messages) == frozen_messages


@pytest.mark.asyncio
async def test_postcommit_trace_provenance_pause_can_resume_one_shot_capture_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _db, store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)
    _fail_trace_request_once("missing_repository", store=store, monkeypatch=monkeypatch)

    first = await controller.submit_draft("secret durable body", session_id="session-1")
    paused = store.preparation_for_session("session-1")
    assert first.accepted is True
    assert paused is not None
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_PROVENANCE
    assert entries == []

    resumed = await controller.send_without_capture(paused.preparation_id)

    assert resumed.accepted is True
    assert len(entries) == 1
    assert entries[0]["capture_mode_override"] is ConsoleTraceCaptureMode.CAPTURE_OFF
    assert entries[0]["trace_request"] is None


@pytest.mark.asyncio
async def test_postcommit_trace_provenance_cancel_retires_all_durable_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)
    _fail_trace_request_once("missing_repository", store=store, monkeypatch=monkeypatch)

    first = await controller.submit_draft("secret durable body", session_id="session-1")
    paused = store.preparation_for_session("session-1")
    assert paused is not None
    continuation = controller._durable_postcommit_continuations[paused.preparation_id]
    fingerprint = continuation.fingerprint
    cancelled = controller.cancel_library_preparation(paused.preparation_id)
    repeated = controller.cancel_library_preparation(paused.preparation_id)

    assert first.provider_started is False
    assert cancelled.visible_copy == "Trace recovery canceled."
    assert repeated.accepted is False
    assert entries == []
    assert store.preparation_for_session("session-1") is None
    assert paused.preparation_id not in controller._durable_postcommit_continuations
    assert store.durable_acceptance_retired(paused.preparation_id, fingerprint)
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0
    assert db.get_message_by_id(first.user_message_id) is None
    assert db.get_message_by_id(first.assistant_message_id) is None
    restarted = ConsoleChatStore(persistence=ChatPersistenceService(db))
    assert restarted.dispatch_recovery_for_session("session-1") is None


@pytest.mark.asyncio
async def test_postcommit_trace_provenance_failure_terminates_autonomous_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _db, store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)
    _fail_trace_request_once("missing_repository", store=store, monkeypatch=monkeypatch)
    coordinator = controller.prompt_queue_coordinator
    registry = coordinator.registry
    begun = registry.begin_chain("session-1", context_epoch=0, expected_revision=0)
    admitted = registry.admit(
        "session-1",
        text="autonomous queued body",
        expected_revision=begun.snapshot.revision,
    )
    assert admitted.entry_id is not None
    coordinator._chains["session-1"] = _PromptChain()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id="session-1"
    )
    submitted: list[ConsoleSubmitResult] = []

    async def submit_queued(text: str, **kwargs: Any):
        result = await controller.submit_draft(
            text,
            session_id=kwargs["session_id"],
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id=kwargs["entry_id"],
            queue_authorization=kwargs["authorization"],
        )
        submitted.append(result)
        return result

    coordinator._submit_queued = submit_queued
    await coordinator._drain_waiting("session-1", ConsoleRunStatus.COMPLETED)

    assert len(submitted) == 1
    assert submitted[0].accepted is True
    assert submitted[0].provider_started is False
    assert submitted[0].terminal_status is ConsoleRunStatus.FAILED
    assert entries == []
    assert store.preparation_for_session("session-1") is None
    assert controller.trace_call_recovery_preparation() is None
    assert controller._durable_postcommit_continuations == {}
    assert "session-1" not in coordinator._chains
    assert registry.snapshot("session-1").total_count == 0
