"""Trace-provenance recovery coverage for durable first sends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.Chat.test_console_dispatch_recovery import _restored_store
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
from tldw_chatbook.Chat.console_endpoint_provenance import (
    EPHEMERAL_SESSION_ENDPOINT_OMITTED,
    ConsoleEndpointProvenance,
)
from tldw_chatbook.Chat.console_exchange_capture import capture_from_blob
from tldw_chatbook.Chat.console_library_destination import resolve_console_destination
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.Chat.console_prompt_queue_coordinator import _PromptChain
from tldw_chatbook.Chat.console_session_endpoint_policy import (
    ConsoleEphemeralEndpointPolicy,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    TraceProvenancePersistenceError,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
    preparation_actions,
)
from tldw_chatbook.Chat.prompt_history import PromptHistory


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


@pytest.mark.asyncio
async def test_ephemeral_vllm_real_send_omits_target_from_checkpoint_trace_and_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live adapter gets the target; no SQLite sink does."""

    target = "http://127.0.0.1:9099/v1"
    configured = "http://127.0.0.1:8000/v1"
    initial = ConsoleSessionSettings(
        provider="vllm",
        model="local-model",
        base_url=configured,
        streaming=False,
    )
    db, store, controller, _old_gateway = _controller(
        tmp_path,
        initial_settings=initial,
    )
    store.adopt_session_ephemeral_endpoint(
        "session-1",
        settings=initial,
        policy=ConsoleEphemeralEndpointPolicy(
            provider="vllm",
            model="local-model",
            base_url=target,
        ),
    )
    checkpoint_rows: list[dict[str, object]] = []
    adapter_calls: list[dict[str, object]] = []

    def adapter(**kwargs: object) -> dict[str, object]:
        adapter_calls.append(dict(kwargs))
        row = (
            db.get_connection()
            .execute(
                "SELECT frozen_authority_json, resolved_destination_json "
                "FROM console_dispatch_checkpoints"
            )
            .fetchone()
        )
        assert row is not None
        checkpoint_rows.append(dict(row))
        return {"choices": [{"message": {"content": "done"}}]}

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(db),
    )

    async def resolve(selection):
        assert selection.base_url == target
        assert (
            selection.endpoint_provenance is ConsoleEndpointProvenance.EPHEMERAL_SESSION
        )
        resolution = ConsoleProviderResolution(
            provider="vllm",
            base_url=target,
            model="local-model",
            ready=True,
            execution_key="vllm",
            endpoint_provenance=selection.endpoint_provenance,
            streaming=False,
        )
        return replace(
            resolution,
            resolved_destination=resolve_console_destination(resolution),
        )

    gateway.resolve_for_send = resolve
    controller.provider_gateway = gateway
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
    monkeypatch.setattr(
        controller,
        "_admit_capture_policy",
        lambda *_args, **_kwargs: signals,
    )
    _force_capture_on(monkeypatch)

    result = await controller.submit_draft(
        "send through the live-only endpoint",
        session_id="session-1",
    )

    assert result.accepted is True and result.provider_started is True
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_base_url"] == target
    authority = json.loads(str(checkpoint_rows[0]["frozen_authority_json"]))
    destination = json.loads(str(checkpoint_rows[0]["resolved_destination_json"]))
    assert authority["provider_intent"]["endpoint"] is None
    assert destination["endpoint_identity"] == EPHEMERAL_SESSION_ENDPOINT_OMITTED
    assert signals.exchange_captures(), signals
    persisted_message = store.get_message(str(result.assistant_message_id))
    assert persisted_message.exchanges
    persisted_message_id = persisted_message.persisted_message_id
    assert persisted_message_id is not None
    rows = db.get_message_exchanges(persisted_message_id)
    assert len(rows) == 1
    capture = capture_from_blob(rows[0]["capture_blob"])
    assert capture.endpoint is None
    assert "api_base_url" not in capture.request
    assert {"api_base_url", "endpoint"}.issubset(capture.omitted_keys)
    trace_header = (
        db.get_connection()
        .execute(
            "SELECT route_identity, endpoint_identity, generation_parameters_json, "
            "adapter_defaults_json, response_format_json, reasoning_controls_json "
            "FROM console_trace_request_headers"
        )
        .fetchone()
    )
    assert trace_header is not None
    assert trace_header["route_identity"] == ConsoleRequestRoute.FRESH.value
    assert trace_header["endpoint_identity"] == EPHEMERAL_SESSION_ENDPOINT_OMITTED
    assert target not in str(tuple(trace_header))
    component_kinds = {
        str(row[0])
        for row in db.get_connection()
        .execute("SELECT component_kind FROM console_trace_header_components")
        .fetchall()
    }
    assert "api_base_url" not in component_kinds
    assert target not in "\n".join(db.get_connection().iterdump())


@pytest.mark.asyncio
@pytest.mark.parametrize("prefill_kind", ("pinned", "one_shot"))
async def test_durable_capture_on_prefill_binds_final_vector_to_direct_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefill_kind: str,
) -> None:
    """Pinned and one-shot prefill must share one final trace/dispatch vector."""

    target = "http://127.0.0.1:9199/v1"
    configured = "http://127.0.0.1:8100/v1"
    prefill = f"{prefill_kind} response:"
    initial = ConsoleSessionSettings(
        provider="vllm",
        model="local-model",
        base_url=configured,
        streaming=False,
    )
    db, store, controller, _old_gateway = _controller(
        tmp_path,
        initial_settings=initial,
    )
    store.adopt_session_ephemeral_endpoint(
        "session-1",
        settings=initial,
        policy=ConsoleEphemeralEndpointPolicy(
            provider="vllm",
            model="local-model",
            base_url=target,
        ),
    )
    if prefill_kind == "pinned":
        store.set_session_pinned_prefill("session-1", prefill)
    else:
        store.set_session_one_shot_prefill("session-1", prefill)

    checkpoint_rows: list[dict[str, object]] = []
    adapter_calls: list[dict[str, object]] = []

    def adapter(**kwargs: object) -> dict[str, object]:
        adapter_calls.append(dict(kwargs))
        row = (
            db.get_connection()
            .execute(
                "SELECT frozen_authority_json, resolved_destination_json "
                "FROM console_dispatch_checkpoints"
            )
            .fetchone()
        )
        assert row is not None
        checkpoint_rows.append(dict(row))
        return {"choices": [{"message": {"content": "done"}}]}

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(db),
    )

    async def resolve(selection):
        resolution = ConsoleProviderResolution(
            provider="vllm",
            base_url=target,
            model="local-model",
            ready=True,
            execution_key="vllm",
            endpoint_provenance=selection.endpoint_provenance,
            streaming=False,
        )
        return replace(
            resolution,
            resolved_destination=resolve_console_destination(resolution),
        )

    gateway.resolve_for_send = resolve
    controller.provider_gateway = gateway
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
    monkeypatch.setattr(
        controller,
        "_admit_capture_policy",
        lambda *_args, **_kwargs: signals,
    )
    _force_capture_on(monkeypatch)

    result = await controller.submit_draft(
        "send one durable direct-prefill request",
        session_id="session-1",
    )

    assert result.accepted is True
    assert result.provider_started is True
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_base_url"] == target
    assert adapter_calls[0]["messages_payload"][-1] == {
        "role": "assistant",
        "content": prefill,
    }
    assistant = store.get_message(str(result.assistant_message_id))
    assert assistant.content == f"{prefill}done"
    authority = json.loads(str(checkpoint_rows[0]["frozen_authority_json"]))
    destination = json.loads(str(checkpoint_rows[0]["resolved_destination_json"]))
    assert authority["provider_intent"]["endpoint"] is None
    assert destination["endpoint_identity"] == EPHEMERAL_SESSION_ENDPOINT_OMITTED

    persisted_message_id = assistant.persisted_message_id
    assert persisted_message_id is not None
    exchange_rows = db.get_message_exchanges(persisted_message_id)
    assert len(exchange_rows) == 1
    capture = capture_from_blob(exchange_rows[0]["capture_blob"])
    assert capture.request["messages_payload"][-1] == {
        "role": "assistant",
        "content": prefill,
    }
    assert capture.endpoint is None
    assert "api_base_url" not in capture.request
    assert {"api_base_url", "endpoint"}.issubset(capture.omitted_keys)
    trace_header = (
        db.get_connection()
        .execute(
            "SELECT route_identity, endpoint_identity "
            "FROM console_trace_request_headers"
        )
        .fetchone()
    )
    assert trace_header is not None
    assert trace_header["route_identity"] == ConsoleRequestRoute.DIRECT_PREFILL.value
    assert trace_header["endpoint_identity"] == EPHEMERAL_SESSION_ENDPOINT_OMITTED
    artifacts = tuple(
        json.loads(bytes(row[0]))
        for row in db.get_connection()
        .execute("SELECT sanitized_bytes FROM console_trace_artifacts")
        .fetchall()
    )
    assert {"role": "assistant", "content": prefill} in artifacts
    assert target not in "\n".join(db.get_connection().iterdump())


@pytest.mark.asyncio
async def test_ephemeral_vllm_checkpoint_restart_refuses_endpoint_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process-only destination cannot become restart replay authority."""

    target = "http://127.0.0.1:9099/v1"
    configured = "http://127.0.0.1:8000/v1"
    initial = ConsoleSessionSettings(
        provider="vllm",
        model="local-model",
        base_url=configured,
        streaming=False,
    )
    db, store, controller, _old_gateway = _controller(
        tmp_path,
        initial_settings=initial,
    )
    store.adopt_session_ephemeral_endpoint(
        "session-1",
        settings=initial,
        policy=ConsoleEphemeralEndpointPolicy(
            provider="vllm",
            model="local-model",
            base_url=target,
        ),
    )
    adapter_calls = 0

    def adapter(**_kwargs: object) -> dict[str, object]:
        nonlocal adapter_calls
        adapter_calls += 1
        return {"choices": [{"message": {"content": "unsettled"}}]}

    gateway = ConsoleProviderGateway(chat_api_call_fn=adapter)

    async def resolve_live(selection):
        resolution = ConsoleProviderResolution(
            provider="vllm",
            base_url=selection.base_url,
            model="local-model",
            ready=True,
            execution_key="vllm",
            endpoint_provenance=selection.endpoint_provenance,
            streaming=False,
        )
        return replace(
            resolution,
            resolved_destination=resolve_console_destination(resolution),
        )

    gateway.resolve_for_send = resolve_live
    controller.provider_gateway = gateway
    monkeypatch.setattr(store, "settle_dispatch_recovery", lambda *_a, **_k: False)

    first = await controller.submit_draft("crash boundary", session_id="session-1")

    assert first.provider_started is True
    assert adapter_calls == 1
    conversation_id = str(
        db.get_connection().execute("SELECT id FROM conversations").fetchone()[0]
    )
    row = (
        db.get_connection()
        .execute(
            "SELECT frozen_authority_json, resolved_destination_json "
            "FROM console_dispatch_checkpoints"
        )
        .fetchone()
    )
    assert row is not None
    assert target not in str(tuple(row))

    restarted, restarted_session_id = _restored_store(db, conversation_id)
    restarted.replace_session_settings(restarted_session_id, initial)
    recovery = restarted.dispatch_recovery_for_session(restarted_session_id)
    assert recovery is not None and recovery.checkpoint is not None
    assert (
        recovery.checkpoint.resolved_destination.endpoint_identity
        == EPHEMERAL_SESSION_ENDPOINT_OMITTED
    )
    assert (
        recovery.checkpoint.resolved_destination.endpoint_provenance
        is ConsoleEndpointProvenance.EPHEMERAL_SESSION
    )
    assert restarted.dispatch_recovery_blocks_submission(restarted_session_id) is True

    replay_calls = 0

    def replay_adapter(**_kwargs: object) -> dict[str, object]:
        nonlocal replay_calls
        replay_calls += 1
        return {"choices": [{"message": {"content": "must not run"}}]}

    replay_gateway = ConsoleProviderGateway(chat_api_call_fn=replay_adapter)

    async def resolve_configured(_selection):
        resolution = ConsoleProviderResolution(
            provider="vllm",
            base_url=configured,
            model="local-model",
            ready=True,
            execution_key="vllm",
            streaming=False,
        )
        return replace(
            resolution,
            resolved_destination=resolve_console_destination(resolution),
        )

    replay_gateway.resolve_for_send = resolve_configured
    restarted_controller = ConsoleChatController(
        store=restarted,
        provider_gateway=replay_gateway,
        provider="vllm",
        model="local-model",
    )

    retried = await restarted_controller.retry_dispatch_recovery(restarted_session_id)

    assert retried.accepted is False
    assert "destination" in retried.visible_copy.lower()
    assert replay_calls == 0


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
