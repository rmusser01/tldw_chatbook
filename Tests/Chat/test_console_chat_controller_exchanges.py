"""Controller wiring: exchange captures attach alongside usage, and a
config kill-switch gates capture end-to-end (task-7, Console Conversation
Inspector).

Fixture/driver idioms copied mechanically from
Tests/Chat/test_console_chat_controller.py's usage-attach coverage (that
file has NO pytest fixtures at all -- every test builds `store`/`controller`
inline, mirrored here):
  * plain construction: ``ConsoleChatController(store=store,
    provider_gateway=StreamingGateway())`` and its own minimal
    ``StreamingGateway`` stub.
  * direct-call driver for ``_attach_stream_usage``:
    ``test_re_attaching_the_same_signals_is_idempotent`` /
    ``test_stop_path_usage_attach_survives_a_persistence_exception``.
  * monkeypatching a module-level import in the controller's OWN
    namespace (a from-import binds at import time, so the CONSUMER's
    namespace -- not the definition site -- is what must be patched):
    ``monkeypatch.setattr(controller_module, "is_vision_capable", ...)``.
  * swallow-and-log diagnostics capture: mirrors
    Tests/Chat/test_console_chat_store_exchanges.py's
    ``test_persist_exchanges_only_survives_a_serialization_failure``
    (a loguru sink collecting WARNING-level records).
  * per-call signals construction (``new_usage_call()`` then
    ``begin_exchange``/``close_exchange``): the real gateway call site in
    ``console_provider_gateway.py``'s ``stream_chat`` (llama.cpp branch).
  * shipped-default pin against the REAL resolved settings layer, no extra
    setup beyond the autouse config isolation: mirrors
    Tests/test_config_console_defaults.py's own
    ``test_console_sidechat_model_default_is_empty_string`` style.

Review fix round: the kill-switch must coerce ``get_cli_setting``'s RAW
string return through ``coerce_bool_setting`` (not bare ``bool()``), pinned
in both directions by the two ``test_kill_switch_string_*`` tests below.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

from Tests.console_provider_doubles import provider_resolution
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat import console_capture_policy_repository as repository_module
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_capture_policy_repository import (
    CapturePolicyWriteResult,
    CapturePolicyWriteStatus,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
from tldw_chatbook.Chat.console_library_destination import resolve_console_destination
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
from tldw_chatbook.Chat.console_trace_service import TraceCallPersistenceError
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class StreamingGateway:
    """Minimal gateway stub, mirroring test_console_chat_controller.py's own
    ``StreamingGateway`` -- this suite never drives a real send; only the
    controller's construction needs a gateway object to exist."""

    async def resolve_for_send(self, selection):
        return provider_resolution(
                   ready=True,
                   provider="llama_cpp",
                   model="test-model",
                   base_url="http://127.0.0.1:9099",
                   visible_copy="",
               )

    async def stream_chat(self, resolution, messages, **kwargs):
        for chunk in ("hel", "lo"):
            yield chunk


def _new_controller() -> ConsoleChatController:
    store = ConsoleChatStore()
    return ConsoleChatController(store=store, provider_gateway=StreamingGateway())


def _controller_with_placeholder():
    """Real store + a real assistant placeholder message, mirroring the
    setup ``test_re_attaching_the_same_signals_is_idempotent`` builds
    before driving ``_attach_stream_usage`` directly."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    return controller, store, placeholder.id


def _captured_signals() -> ConsoleProviderStreamSignals:
    """One provider call's worth of exchange capture, using the real
    per-call pattern from ``console_provider_gateway.py``'s ``stream_chat``
    (llama.cpp branch): ``new_usage_call()`` for the per-call view, then
    begin/close_exchange on it (the aggregate itself has no begin_exchange
    method -- only the per-call view does)."""
    # Explicit opt-in: the dataclass default is False (review finding I1) --
    # this helper builds a signals object with a real capture in it.
    signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
    call_signals = signals.new_usage_call()
    call_signals.begin_exchange(
        provider="p", model="m", endpoint=None, request={}, omitted_keys=()
    )
    call_signals.close_exchange()
    return signals


def test_legacy_exchange_signals_are_disabled_by_default():
    controller = _new_controller()
    signals = controller._new_run_stream_signals()
    assert signals.exchange_capture_enabled is False


def test_normalized_capture_does_not_duplicate_legacy_exchange_blobs(monkeypatch):
    controller = _new_controller()
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(
            enabled=True,
            detail=CaptureDetail.SAFE,
            generation=7,
            normalized_writes_enabled=True,
            normalized_reads_enabled=True,
            legacy_writes_enabled=False,
            pii_redaction_enabled=False,
        ),
    )

    signals = controller._new_run_stream_signals()

    assert signals.exchange_capture_enabled is False


def test_legacy_writer_can_be_reenabled_without_disabling_normalized_reads(
    monkeypatch,
):
    controller = _new_controller()
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(
            enabled=True,
            detail=CaptureDetail.SAFE,
            generation=8,
            normalized_writes_enabled=False,
            normalized_reads_enabled=True,
            legacy_writes_enabled=True,
            pii_redaction_enabled=False,
        ),
    )

    signals = controller._new_run_stream_signals()

    assert signals.exchange_capture_enabled is True


def test_capture_policy_precedence_and_wake_one_shot_exclusion(monkeypatch):
    controller = _new_controller()
    session = controller.store.ensure_session()
    snapshot = controller.capture_policy_snapshot(session.id)
    controller.store.replace_session_capture_override(
        session.id,
        CaptureDetail.SAFE,
        expected_policy_revision=snapshot.policy_revision,
    )
    snapshot = controller.capture_policy_snapshot(session.id)
    controller.set_next_capture_detail(
        session.id,
        CaptureDetail.FULL,
        expected_policy_revision=snapshot.policy_revision,
    )
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.FULL, generation=7),
    )

    wake = controller._admit_capture_policy(
        session.id, ConsoleSubmissionOrigin.AGENT_WAKE
    )
    manual = controller._admit_capture_policy(
        session.id, ConsoleSubmissionOrigin.MANUAL
    )

    assert wake.capture_detail is CaptureDetail.SAFE
    assert manual.capture_detail is CaptureDetail.FULL
    assert controller.capture_policy_snapshot(session.id).next_detail is None


def test_trace_privacy_scopes_resolve_independently_and_one_shot_is_consumed(
    monkeypatch,
) -> None:
    controller = _new_controller()
    session = controller.store.ensure_session()
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(
            enabled=True,
            detail=CaptureDetail.SAFE,
            generation=7,
            pii_redaction_enabled=False,
            viewer_profile="safe",
        ),
    )
    snapshot = controller.capture_policy_snapshot(session.id)
    controller.store.replace_session_trace_privacy_override(
        session.id,
        capture_enabled=False,
        pii_redaction_enabled=True,
        expected_policy_revision=snapshot.policy_revision,
    )
    snapshot = controller.capture_policy_snapshot(session.id)
    assert snapshot.effective_capture_enabled is False
    assert snapshot.pii_redaction_enabled is True

    result = controller.set_next_trace_privacy(
        session.id,
        capture_enabled=True,
        pii_redaction_enabled=False,
        expected_policy_revision=snapshot.policy_revision,
    )
    assert result.snapshot.effective_capture_enabled is True
    assert result.snapshot.pii_redaction_enabled is False

    signals = controller._admit_capture_policy(
        session.id, ConsoleSubmissionOrigin.MANUAL
    )

    assert signals.exchange_capture_enabled is True
    after = controller.capture_policy_snapshot(session.id)
    assert after.next_capture_enabled is None
    assert after.next_pii_redaction_enabled is None
    assert after.effective_capture_enabled is False
    assert after.pii_redaction_enabled is True


def test_frozen_next_send_capture_survives_one_shot_consumption(monkeypatch) -> None:
    controller = _new_controller()
    session = controller.store.ensure_session()
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(
            enabled=False,
            detail=CaptureDetail.SAFE,
            generation=8,
            pii_redaction_enabled=False,
            viewer_profile="safe",
        ),
    )
    snapshot = controller.capture_policy_snapshot(session.id)
    controller.set_next_trace_privacy(
        session.id,
        capture_enabled=True,
        pii_redaction_enabled=True,
        expected_policy_revision=snapshot.policy_revision,
    )
    frozen = controller.capture_policy_snapshot(session.id)
    signals = controller._admit_capture_policy(
        session.id,
        ConsoleSubmissionOrigin.MANUAL,
        frozen_capture_enabled=frozen.effective_capture_enabled,
        frozen_pii_redaction_enabled=frozen.pii_redaction_enabled,
        frozen_next_trace_privacy_revision=frozen.next_privacy_revision,
    )

    assert signals.exchange_capture_enabled is True
    assert signals.pii_redaction_enabled is True
    assert controller.capture_policy_snapshot(session.id).effective_capture_enabled is False


def test_frozen_turn_does_not_consume_a_newer_next_send_privacy_choice(
    monkeypatch,
) -> None:
    controller = _new_controller()
    session = controller.store.ensure_session()
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(
            enabled=False,
            detail=CaptureDetail.SAFE,
            generation=8,
            pii_redaction_enabled=False,
            viewer_profile="safe",
        ),
    )
    initial = controller.capture_policy_snapshot(session.id)
    first = controller.set_next_trace_privacy(
        session.id,
        capture_enabled=True,
        pii_redaction_enabled=True,
        expected_policy_revision=initial.policy_revision,
    ).snapshot
    newer = controller.set_next_trace_privacy(
        session.id,
        capture_enabled=False,
        pii_redaction_enabled=False,
        expected_policy_revision=first.policy_revision,
    ).snapshot

    signals = controller._admit_capture_policy(
        session.id,
        ConsoleSubmissionOrigin.MANUAL,
        frozen_capture_enabled=first.effective_capture_enabled,
        frozen_pii_redaction_enabled=first.pii_redaction_enabled,
        frozen_next_trace_privacy_revision=first.next_privacy_revision,
    )

    assert signals.exchange_capture_enabled is True
    assert signals.pii_redaction_enabled is True
    after = controller.capture_policy_snapshot(session.id)
    assert after.next_privacy_revision == newer.next_privacy_revision
    assert after.next_capture_enabled is False
    assert after.next_pii_redaction_enabled is False


def test_frozen_turn_without_override_does_not_consume_later_privacy_choice(
    monkeypatch,
) -> None:
    controller = _new_controller()
    session = controller.store.ensure_session()
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(
            enabled=False,
            detail=CaptureDetail.SAFE,
            generation=8,
            pii_redaction_enabled=False,
            viewer_profile="safe",
        ),
    )
    initial = controller.capture_policy_snapshot(session.id)
    armed = controller.set_next_trace_privacy(
        session.id,
        capture_enabled=True,
        pii_redaction_enabled=True,
        expected_policy_revision=initial.policy_revision,
    ).snapshot

    signals = controller._admit_capture_policy(
        session.id,
        ConsoleSubmissionOrigin.MANUAL,
        frozen_capture_enabled=False,
        frozen_pii_redaction_enabled=False,
        frozen_next_trace_privacy_revision=None,
    )

    assert signals.exchange_capture_enabled is False
    assert signals.pii_redaction_enabled is False
    after = controller.capture_policy_snapshot(session.id)
    assert after.next_privacy_revision == armed.next_privacy_revision
    assert after.next_capture_enabled is True
    assert after.next_pii_redaction_enabled is True


@pytest.mark.asyncio
async def test_temporary_capture_on_pauses_real_submit_before_gateway(
    monkeypatch,
) -> None:
    class CountingGateway(StreamingGateway):
        supports_durable_capture = True

        def __init__(self) -> None:
            self.entries = 0

        async def stream_chat(self, resolution, messages, **kwargs):
            self.entries += 1
            yield "unexpected"

    gateway = CountingGateway()
    store = ConsoleChatStore()
    session = store.create_session(
        ephemeral=True,
        settings=controller_module.ConsoleSessionSettings(
            provider="llama_cpp",
            model="test-model",
        ),
    )
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=7),
    )
    snapshot = controller.capture_policy_snapshot(session.id)
    controller.set_next_trace_privacy(
        session.id,
        capture_enabled=True,
        pii_redaction_enabled=True,
        expected_policy_revision=snapshot.policy_revision,
    )

    result = await controller.submit_draft("capture this", session_id=session.id)
    preparation = store.preparation_for_session(session.id)
    privacy = controller.capture_policy_snapshot(session.id)

    assert result.accepted is False
    assert result.provider_started is False
    assert gateway.entries == 0
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.PAUSED
    assert preparation.pause_kind is ConsolePreparationPauseKind.TEMPORARY_CAPTURE
    assert privacy.next_capture_enabled is True
    assert privacy.next_pii_redaction_enabled is True
    assert preparation.capture_mode is ConsoleTraceCaptureMode.CAPTURE_ON


@pytest.mark.asyncio
async def test_temporary_capture_on_attachment_only_pauses_before_gateway(
    monkeypatch,
) -> None:
    class CountingGateway(StreamingGateway):
        supports_durable_capture = True

        def __init__(self) -> None:
            self.entries = 0

        async def stream_chat(self, resolution, messages, **kwargs):
            self.entries += 1
            yield "unexpected"

    gateway = CountingGateway()
    store = ConsoleChatStore()
    session = store.create_session(
        ephemeral=True,
        settings=controller_module.ConsoleSessionSettings(
            provider="llama_cpp",
            model="test-model",
        ),
    )
    store.add_pending_attachment(
        session.id,
        PendingAttachment(
            file_path="/tmp/trace-attachment.png",
            display_name="trace-attachment.png",
            file_type="image",
            insert_mode="attachment",
            data=b"\x89PNG-trace",
            mime_type="image/png",
            original_size=10,
            processed_size=10,
        ),
    )
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    original_snapshot = controller.resolve_turn_configuration_snapshot
    monkeypatch.setattr(
        controller,
        "resolve_turn_configuration_snapshot",
        lambda session_id: replace(
            original_snapshot(session_id),
            capabilities={"vision": True},
        ),
    )
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=7),
    )

    result = await controller.submit_draft("", session_id=session.id)
    preparation = store.preparation_for_session(session.id)

    assert result.accepted is False
    assert result.provider_started is False
    assert gateway.entries == 0
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.PAUSED
    assert preparation.pause_kind is ConsolePreparationPauseKind.TEMPORARY_CAPTURE
    assert preparation.attachment_ids


@pytest.mark.asyncio
async def test_queued_temporary_capture_on_refuses_without_throwing_or_dispatch(
    monkeypatch,
) -> None:
    class CountingGateway(StreamingGateway):
        supports_durable_capture = True

        def __init__(self) -> None:
            self.entries = 0

        async def stream_chat(self, resolution, messages, **kwargs):
            self.entries += 1
            yield "unexpected"

    gateway = CountingGateway()
    store = ConsoleChatStore()
    session = store.create_session(
        ephemeral=True,
        settings=controller_module.ConsoleSessionSettings(
            provider="llama_cpp",
            model="test-model",
        ),
    )
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    monkeypatch.setattr(
        controller.prompt_queue_coordinator,
        "authorizes",
        lambda _authorization, _session_id: True,
    )
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=7),
    )

    result = await controller.submit_draft(
        "queued capture",
        session_id=session.id,
        origin=ConsoleSubmissionOrigin.QUEUED,
        queue_entry_id="queue-entry-1",
        queue_authorization=object(),  # type: ignore[arg-type]
    )

    assert result.accepted is False
    assert result.provider_started is False
    assert result.queue_entry_id == "queue-entry-1"
    assert "Save the chat or turn Capture Off" in result.visible_copy
    assert gateway.entries == 0
    assert store.preparation_for_session(session.id) is None
    assert store.messages_for_session(session.id) == []


@pytest.mark.asyncio
async def test_temporary_capture_cancel_removes_unsent_echo_and_preparation(
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session(
        ephemeral=True,
        settings=controller_module.ConsoleSessionSettings(
            provider="llama_cpp",
            model="test-model",
        ),
    )
    gateway = StreamingGateway()
    gateway.supports_durable_capture = True
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=7),
    )
    paused = await controller.submit_draft("cancel me", session_id=session.id)
    assert paused.preparation_id is not None

    cancelled = controller.cancel_library_preparation(paused.preparation_id)

    assert cancelled.visible_copy == "Temporary trace-captured send canceled."
    assert store.preparation_for_session(session.id) is None
    assert store.messages_for_session(session.id) == []


@pytest.mark.asyncio
async def test_save_and_send_refreshes_durable_authority_before_trace_admission(
    monkeypatch,
    tmp_path,
) -> None:
    adapter_entries = 0

    def adapter(**_kwargs):
        nonlocal adapter_entries
        adapter_entries += 1
        return {"choices": [{"message": {"content": "unexpected"}}]}

    class FailingBoundary:
        def reserve(self) -> None:
            raise TraceCallPersistenceError(
                boundary=self,
                reservation_status="not_established",
            )

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=lambda *_args: FailingBoundary(),
    )

    async def resolve_for_send(_selection):
        resolution = ConsoleProviderResolution(
            execution_key="openai",
            ready=True,
            provider="openai",
            model="test-model",
            base_url="https://api.openai.com/v1",
            visible_copy="",
            streaming=False,
        )
        return replace(
            resolution,
            resolved_destination=resolve_console_destination(resolution),
        )

    monkeypatch.setattr(gateway, "resolve_for_send", resolve_for_send)
    chat_db = CharactersRAGDB(
        tmp_path / "temporary-capture-promotion.sqlite",
        "temporary-capture-promotion",
    )
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(chat_db))
        session = store.create_session(
            ephemeral=True,
            settings=controller_module.ConsoleSessionSettings(
                provider="openai",
                model="test-model",
            ),
        )
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            provider="openai",
            model="test-model",
        )
        monkeypatch.setattr(
            controller_module,
            "runtime_capture_policy",
            lambda: SimpleNamespace(
                enabled=True,
                detail=CaptureDetail.SAFE,
                generation=7,
            ),
        )

        paused_result = await controller.submit_draft(
            "capture this",
            session_id=session.id,
        )
        paused = store.preparation_for_session(session.id)
        assert paused_result.accepted is False
        assert paused is not None
        original_attempt_id = paused.attempt_id

        result = await controller.save_and_send(paused.preparation_id)
        promoted = store.preparation_for_session(session.id)

        assert store.session_is_ephemeral(session.id) is False
        assert session.persisted_conversation_id is not None
        assert promoted is not None
        assert promoted.attempt_id != original_attempt_id
        assert promoted.execution_context.library_authority.policy.source == "durable"
        assert promoted.state is ConsoleTurnPreparationState.ACCEPTED
        assert result.accepted is True
        assert result.provider_started is False
        assert result.visible_copy == "Accepted turn is retained for recovery."
        assert adapter_entries == 0
        rows = chat_db.get_messages_for_conversation(
            session.persisted_conversation_id,
            limit=100,
        )
        assert [row["content"] for row in rows if row["sender"] == "user"] == [
            "capture this"
        ]
    finally:
        await gateway.aclose()
        chat_db.close_connection()


def test_global_full_hydration_unavailable_fails_safe_and_retries_once_per_read(
    monkeypatch,
):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    calls = []

    class Repository:
        def read(self, conversation_id):
            calls.append(conversation_id)
            return repository_module.CapturePolicyReadResult(
                repository_module.CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT,
                None,
            )

    repository = Repository()
    controller.store.capture_policy_repository = repository
    controller._capture_policy_repository = repository
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.FULL, generation=7),
    )

    first = controller.capture_policy_snapshot(session.id)
    admitted = controller._admit_capture_policy(
        session.id, ConsoleSubmissionOrigin.MANUAL
    )

    assert first.effective.detail is CaptureDetail.SAFE
    assert first.save_pending is True
    assert first.error_code == "conversation_policy_unavailable"
    assert admitted.capture_detail is CaptureDetail.SAFE
    assert calls == ["conversation-1", "conversation-1"]
    assert session.id not in controller._capture_policy_hydrated


def test_capture_policy_read_exception_never_enables_global_full(monkeypatch):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"

    class Repository:
        @staticmethod
        def read(_conversation_id):
            raise RuntimeError("SEMANTIC_POLICY_CANARY")

    repository = Repository()
    controller.store.capture_policy_repository = repository
    controller._capture_policy_repository = repository
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.FULL, generation=7),
    )

    signals = controller._admit_capture_policy(
        session.id, ConsoleSubmissionOrigin.MANUAL
    )

    assert signals.exchange_capture_enabled is True
    assert signals.capture_detail is CaptureDetail.SAFE
    assert session.id not in controller._capture_policy_hydrated


def test_admission_revision_gate_preserves_rearmed_next_slot(monkeypatch):
    controller = _new_controller()
    session = controller.store.ensure_session()
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    snapshot = controller.capture_policy_snapshot(session.id)
    controller.set_next_capture_detail(
        session.id,
        CaptureDetail.FULL,
        expected_policy_revision=snapshot.policy_revision,
    )
    original = controller.store.consume_session_next_capture_detail

    def rearm_before_consume(session_id, *, expected_next_revision):
        state = controller.store.capture_policy_state(session_id)
        controller.store.set_session_next_capture_detail(
            session_id,
            CaptureDetail.SAFE,
            expected_policy_revision=state.policy_revision,
        )
        return original(session_id, expected_next_revision=expected_next_revision)

    monkeypatch.setattr(
        controller.store, "consume_session_next_capture_detail", rearm_before_consume
    )

    signals = controller._admit_capture_policy(
        session.id, ConsoleSubmissionOrigin.MANUAL
    )

    assert signals.capture_detail is CaptureDetail.FULL
    assert controller.capture_policy_snapshot(session.id).next_detail is CaptureDetail.SAFE


def test_global_off_disarms_all_next_slots_but_keeps_conversation_detail(monkeypatch):
    controller = _new_controller()
    first = controller.store.ensure_session()
    second = controller.store.create_session(activate=False)
    controller.store.replace_session_capture_override(
        first.id,
        CaptureDetail.FULL,
        expected_policy_revision=0,
    )
    revision = controller.store.capture_policy_state(first.id).policy_revision
    controller.store.set_session_next_capture_detail(
        first.id, CaptureDetail.FULL, expected_policy_revision=revision,
    )
    revision = controller.store.capture_policy_state(first.id).policy_revision
    controller.store.set_session_next_capture_detail(
        second.id, CaptureDetail.SAFE, expected_policy_revision=revision,
    )
    revision = controller.store.capture_policy_state(first.id).policy_revision
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=3),
    )
    sibling_statuses = []

    def apply_while_sibling_attempts_mutation(**_kwargs):
        current = controller.capture_policy_snapshot(first.id)
        sibling_statuses.append(controller.set_next_capture_detail(
            first.id, CaptureDetail.SAFE,
            expected_policy_revision=current.policy_revision,
        ).status)
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        controller_module, "apply_console_capture_settings",
        apply_while_sibling_attempts_mutation,
    )
    result = controller.apply_global_capture_settings(
        enabled=False, detail=CaptureDetail.SAFE,
        expected_config_generation=3, expected_policy_revision=revision,
    )
    assert result.status is controller_module.CapturePolicyMutationStatus.APPLIED
    assert controller.store.capture_policy_state(first.id).next_detail is None
    assert controller.store.capture_policy_state(second.id).next_detail is None
    assert sibling_statuses == [controller_module.CapturePolicyMutationStatus.STALE]
    assert controller.store.capture_policy_state(
        first.id
    ).conversation_detail is CaptureDetail.FULL


@pytest.mark.asyncio
async def test_ephemeral_full_override_stages_for_later_promotion(monkeypatch):
    controller = _new_controller()
    session = controller.store.create_session(ephemeral=True)
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    before = controller.capture_policy_snapshot(session.id)

    result = await controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.FULL,
        expected_policy_revision=before.policy_revision,
    )

    assert result.status is controller_module.CapturePolicyMutationStatus.APPLIED
    assert result.snapshot.conversation_detail is CaptureDetail.FULL
    assert result.snapshot.save_pending is False


@pytest.mark.asyncio
async def test_stale_conversation_policy_never_reaches_repository():
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    calls = []

    class Repository:
        def replace(self, *args):
            calls.append(args)
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, None)

    controller._capture_policy_repository = Repository()
    result = await controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.FULL, expected_policy_revision=99,
    )

    assert result.status is controller_module.CapturePolicyMutationStatus.STALE
    assert calls == []


@pytest.mark.asyncio
async def test_conversation_policy_reservation_blocks_race_and_reconciles_cancel(
    monkeypatch,
):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    started = threading.Event()
    release = threading.Event()
    durable = {}

    class Repository:
        def replace(self, conversation_id, detail):
            started.set()
            release.wait(5)
            durable[conversation_id] = detail
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, None)

    controller._capture_policy_repository = Repository()
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    before = controller.capture_policy_snapshot(session.id)
    mutation = asyncio.create_task(controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.FULL,
        expected_policy_revision=before.policy_revision,
    ))
    assert await asyncio.to_thread(started.wait, 2)
    during = controller.capture_policy_snapshot(session.id)
    sibling = controller.set_next_capture_detail(
        session.id, CaptureDetail.SAFE,
        expected_policy_revision=during.policy_revision,
    )
    assert sibling.status is controller_module.CapturePolicyMutationStatus.STALE
    mutation.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await mutation
    assert durable["conversation-1"] is CaptureDetail.FULL
    assert controller.capture_policy_snapshot(
        session.id
    ).conversation_detail is CaptureDetail.FULL


@pytest.mark.asyncio
async def test_full_to_safe_is_safe_during_blocked_write_and_stays_safe_on_failure(
    monkeypatch,
):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    controller.store.replace_session_capture_override(
        session.id,
        CaptureDetail.FULL,
        expected_policy_revision=0,
    )
    started = threading.Event()
    release = threading.Event()

    class Repository:
        @staticmethod
        def replace(_conversation_id, _detail):
            started.set()
            release.wait(5)
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.UNAVAILABLE, None)

    controller._capture_policy_repository = Repository()
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.FULL, generation=1),
    )
    before = controller.capture_policy_snapshot(session.id)
    mutation = asyncio.create_task(
        controller.replace_conversation_capture_detail(
            session.id,
            CaptureDetail.SAFE,
            expected_policy_revision=before.policy_revision,
        )
    )
    assert await asyncio.to_thread(started.wait, 2)

    during = controller._admit_capture_policy(
        session.id, ConsoleSubmissionOrigin.MANUAL
    )
    assert during.capture_detail is CaptureDetail.SAFE
    assert controller.capture_policy_snapshot(
        session.id
    ).conversation_detail is CaptureDetail.SAFE

    release.set()
    result = await mutation
    assert result.status is controller_module.CapturePolicyMutationStatus.SAFE_SESSION_ONLY
    assert result.snapshot.conversation_detail is CaptureDetail.SAFE
    assert result.snapshot.save_pending is True


@pytest.mark.parametrize("failing_seam", ["store", "runtime", "resolver"])
def test_capture_policy_resolution_failure_disables_capture_without_consuming_one_shot(
    monkeypatch, failing_seam,
):
    controller = _new_controller()
    session = controller.store.ensure_session()
    controller.store.set_session_next_capture_detail(
        session.id,
        CaptureDetail.FULL,
        expected_policy_revision=0,
    )
    canary = "CAPTURE_POLICY_RESOLUTION_CANARY"
    if failing_seam == "store":
        monkeypatch.setattr(
            controller.store,
            "capture_policy_state",
            lambda _session_id: (_ for _ in ()).throw(RuntimeError(canary)),
        )
    elif failing_seam == "runtime":
        monkeypatch.setattr(
            controller_module,
            "runtime_capture_policy",
            lambda: (_ for _ in ()).throw(RuntimeError(canary)),
        )
    else:
        monkeypatch.setattr(
            controller_module,
            "resolve_capture_policy",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(canary)),
        )

    diagnostics: list[str] = []
    sink_id = loguru_logger.add(
        diagnostics.append,
        level="WARNING",
        format="{extra[phase]} {extra[error_type]} {message}",
    )
    try:
        signals = controller._admit_capture_policy(
            session.id, ConsoleSubmissionOrigin.MANUAL
        )
    finally:
        loguru_logger.remove(sink_id)

    assert signals.exchange_capture_enabled is False
    assert signals.capture_detail is CaptureDetail.SAFE
    assert session.next_capture_detail is CaptureDetail.FULL
    assert any("capture_policy_resolution_failed" in item for item in diagnostics)
    assert all(canary not in item for item in diagnostics)


@pytest.mark.asyncio
async def test_repeated_cancellation_still_reconciles_durable_policy(monkeypatch):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    started = threading.Event()
    release = threading.Event()
    committed = threading.Event()
    durable = {}

    class Repository:
        def replace(self, conversation_id, detail):
            started.set()
            release.wait(5)
            durable[conversation_id] = detail
            committed.set()
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, None)

    controller._capture_policy_repository = Repository()
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    before = controller.capture_policy_snapshot(session.id)
    mutation = asyncio.create_task(controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.FULL,
        expected_policy_revision=before.policy_revision,
    ))
    assert await asyncio.to_thread(started.wait, 2)

    mutation.cancel()
    await asyncio.sleep(0)
    mutation.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await mutation
    assert await asyncio.to_thread(committed.wait, 2)
    assert durable["conversation-1"] is CaptureDetail.FULL
    assert controller.capture_policy_snapshot(
        session.id
    ).conversation_detail is CaptureDetail.FULL


@pytest.mark.asyncio
async def test_caller_cancellation_precedes_repository_exception(monkeypatch):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    started = threading.Event()
    release = threading.Event()

    class Repository:
        def replace(self, conversation_id, detail):
            started.set()
            release.wait(5)
            raise RuntimeError("repository failed")

    controller._capture_policy_repository = Repository()
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    before = controller.capture_policy_snapshot(session.id)
    mutation = asyncio.create_task(controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.FULL,
        expected_policy_revision=before.policy_revision,
    ))
    assert await asyncio.to_thread(started.wait, 2)

    mutation.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await mutation

    current = controller.capture_policy_snapshot(session.id)
    follow_up = controller.set_next_capture_detail(
        session.id, CaptureDetail.SAFE,
        expected_policy_revision=current.policy_revision,
    )
    assert follow_up.status is controller_module.CapturePolicyMutationStatus.APPLIED


@pytest.mark.asyncio
async def test_cancelled_reconciliation_keeps_reservation_until_worker_settles(
    monkeypatch,
):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    started = threading.Event()
    release = threading.Event()
    committed = threading.Event()
    durable = {}
    calls = []

    class Repository:
        def replace(self, conversation_id, detail):
            calls.append(detail)
            is_first = len(calls) == 1
            if is_first:
                started.set()
                release.wait(5)
            durable[conversation_id] = detail
            if is_first:
                committed.set()
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, None)

    controller._capture_policy_repository = Repository()
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    real_create_task = asyncio.create_task
    owned_tasks = []

    def track_owned_task(coro):
        task = real_create_task(coro)
        owned_tasks.append(task)
        return task

    monkeypatch.setattr(controller_module.asyncio, "create_task", track_owned_task)
    before = controller.capture_policy_snapshot(session.id)
    mutation = real_create_task(controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.FULL,
        expected_policy_revision=before.policy_revision,
    ))
    assert await asyncio.to_thread(started.wait, 2)
    reconciliation = owned_tasks[0]
    reconciliation.cancel()
    await asyncio.sleep(0)

    during = controller.capture_policy_snapshot(session.id)
    newer = await controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.SAFE,
        expected_policy_revision=during.policy_revision,
    )
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await mutation
    assert await asyncio.to_thread(committed.wait, 2)

    assert newer.status is controller_module.CapturePolicyMutationStatus.STALE
    assert calls == [CaptureDetail.FULL]
    assert durable["conversation-1"] is CaptureDetail.FULL
    assert controller.capture_policy_snapshot(
        session.id
    ).conversation_detail is CaptureDetail.FULL
    current = controller.capture_policy_snapshot(session.id)
    follow_up = await controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.SAFE,
        expected_policy_revision=current.policy_revision,
    )
    assert follow_up.status is controller_module.CapturePolicyMutationStatus.APPLIED
    assert durable["conversation-1"] is CaptureDetail.SAFE
    assert controller.capture_policy_snapshot(
        session.id
    ).conversation_detail is CaptureDetail.SAFE


@pytest.mark.asyncio
@pytest.mark.parametrize("repository_fails", [False, True])
async def test_direct_waiter_cancellation_retains_worker_settlement(
    monkeypatch, repository_fails,
):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    started = threading.Event()
    release = threading.Event()
    durable = {}

    class Repository:
        def replace(self, conversation_id, detail):
            started.set()
            release.wait(5)
            if repository_fails:
                raise RuntimeError("repository failed")
            durable[conversation_id] = detail
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, None)

    controller._capture_policy_repository = Repository()
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    real_create_task = asyncio.create_task
    owned_tasks = []

    def track_owned_task(coro):
        task = real_create_task(coro)
        owned_tasks.append(task)
        return task

    monkeypatch.setattr(controller_module.asyncio, "create_task", track_owned_task)
    before = controller.capture_policy_snapshot(session.id)
    mutation = real_create_task(controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.FULL,
        expected_policy_revision=before.policy_revision,
    ))
    assert await asyncio.to_thread(started.wait, 2)
    owned_tasks[-1].cancel()
    await asyncio.sleep(0)

    during = controller.capture_policy_snapshot(session.id)
    newer = controller.set_next_capture_detail(
        session.id, CaptureDetail.SAFE,
        expected_policy_revision=during.policy_revision,
    )
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await mutation

    assert newer.status is controller_module.CapturePolicyMutationStatus.STALE
    assert durable == (
        {} if repository_fails else {"conversation-1": CaptureDetail.FULL}
    )
    settled = controller.capture_policy_snapshot(session.id)
    assert settled.conversation_detail is (
        CaptureDetail.SAFE if repository_fails else CaptureDetail.FULL
    )
    assert settled.save_pending is repository_fails
    current = controller.capture_policy_snapshot(session.id)
    follow_up = controller.set_next_capture_detail(
        session.id, CaptureDetail.SAFE,
        expected_policy_revision=current.policy_revision,
    )
    assert follow_up.status is controller_module.CapturePolicyMutationStatus.APPLIED


@pytest.mark.asyncio
async def test_repository_exception_without_cancellation_propagates(monkeypatch):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"

    class Repository:
        def replace(self, conversation_id, detail):
            raise RuntimeError("repository failed")

    controller._capture_policy_repository = Repository()
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    before = controller.capture_policy_snapshot(session.id)

    with pytest.raises(RuntimeError, match="repository failed"):
        await controller.replace_conversation_capture_detail(
            session.id, CaptureDetail.FULL,
            expected_policy_revision=before.policy_revision,
        )

    current = controller.capture_policy_snapshot(session.id)
    follow_up = controller.set_next_capture_detail(
        session.id, CaptureDetail.SAFE,
        expected_policy_revision=current.policy_revision,
    )
    assert follow_up.status is controller_module.CapturePolicyMutationStatus.APPLIED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin", [ConsoleSubmissionOrigin.MANUAL, ConsoleSubmissionOrigin.QUEUED]
)
async def test_admission_consumes_exact_one_shot_during_policy_reservation(
    monkeypatch, origin,
):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    before = controller.capture_policy_snapshot(session.id)
    controller.set_next_capture_detail(
        session.id, CaptureDetail.FULL,
        expected_policy_revision=before.policy_revision,
    )
    started = threading.Event()
    release = threading.Event()

    class Repository:
        def replace(self, _conversation_id, _detail):
            started.set()
            release.wait(5)
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, None)

    controller._capture_policy_repository = Repository()
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    armed = controller.capture_policy_snapshot(session.id)
    mutation = asyncio.create_task(controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.SAFE,
        expected_policy_revision=armed.policy_revision,
    ))
    assert await asyncio.to_thread(started.wait, 2)
    expected_consumed_revision = controller.store.capture_policy_state(
        session.id
    ).next_revision

    admitted = controller._admit_capture_policy(session.id, origin)
    assert admitted.capture_detail is CaptureDetail.FULL
    assert controller.capture_policy_snapshot(session.id).next_detail is None

    release.set()
    await mutation
    after = controller.capture_policy_snapshot(session.id)
    controller.set_next_capture_detail(
        session.id, CaptureDetail.SAFE,
        expected_policy_revision=after.policy_revision,
    )
    assert controller.store.consume_session_next_capture_detail(
        session.id, expected_next_revision=expected_consumed_revision,
    ) is False
    assert controller.capture_policy_snapshot(
        session.id
    ).next_detail is CaptureDetail.SAFE


@pytest.mark.asyncio
async def test_session_close_during_policy_write_releases_reservation(monkeypatch):
    controller = _new_controller()
    session = controller.store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    started = threading.Event()
    release = threading.Event()

    class Repository:
        def replace(self, _conversation_id, _detail):
            started.set()
            release.wait(5)
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, None)

    controller._capture_policy_repository = Repository()
    monkeypatch.setattr(
        controller_module, "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=1),
    )
    before = controller.capture_policy_snapshot(session.id)
    mutation = asyncio.create_task(controller.replace_conversation_capture_detail(
        session.id, CaptureDetail.FULL,
        expected_policy_revision=before.policy_revision,
    ))
    assert await asyncio.to_thread(started.wait, 2)
    controller.store.close_session(session.id)
    replacement = controller.store.ensure_session()
    release.set()

    result = await mutation
    assert result.status is controller_module.CapturePolicyMutationStatus.TARGET_MISSING
    current = controller.capture_policy_snapshot(replacement.id)
    follow_up = controller.set_next_capture_detail(
        replacement.id, CaptureDetail.SAFE,
        expected_policy_revision=current.policy_revision,
    )
    assert follow_up.status is controller_module.CapturePolicyMutationStatus.APPLIED


def test_kill_switch_disables_capture(monkeypatch):
    """Patch ``get_cli_setting`` AT THE CONTROLLER'S NAMESPACE (a from-import
    binds at import time -- patch the consumer, prove it with a call
    counter)."""
    controller = _new_controller()
    calls: list[tuple[str, str]] = []

    def fake_get_cli_setting(section, key, default=None):
        calls.append((section, key))
        if (section, key) == ("console", "exchange_capture"):
            return False
        return default

    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(
            enabled=controller_module.coerce_bool_setting(
                fake_get_cli_setting("console", "exchange_capture", True), True
            ),
            detail=CaptureDetail.SAFE,
            generation=1,
        ),
    )

    signals = controller._new_run_stream_signals()

    assert signals.exchange_capture_enabled is False
    assert ("console", "exchange_capture") in calls


def test_kill_switch_string_false_disables_capture(monkeypatch):
    """``get_cli_setting`` returns the RAW TOML value, uncoerced -- a
    hand-typed ``exchange_capture = "false"`` is a non-empty string and
    therefore truthy under bare ``bool()``, which would silently defeat
    the only escape hatch for this privacy-sensitive feature (the arc's
    sixth occurrence of this exact trap; see ``local_tools_enabled``'s own
    read in ``console_chat_controller.py`` for the first).
    ``coerce_bool_setting`` must be applied to the read."""
    controller = _new_controller()

    def fake_get_cli_setting(section, key, default=None):
        if (section, key) == ("console", "exchange_capture"):
            return "false"
        return default

    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(
            enabled=controller_module.coerce_bool_setting(
                fake_get_cli_setting("console", "exchange_capture", True), True
            ),
            detail=CaptureDetail.SAFE,
            generation=1,
        ),
    )

    signals = controller._new_run_stream_signals()

    assert signals.exchange_capture_enabled is False


def test_kill_switch_string_true_enables_capture(monkeypatch):
    """A hand-typed ``"true"`` survives the real runtime-policy coercion path."""
    controller = _new_controller()
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 614)
    monkeypatch.setattr(config_module, "_RUNTIME_CAPTURE_POLICY", None)
    monkeypatch.setattr(
        config_module,
        "_published_runtime_config_snapshot",
        lambda: config_module.RuntimeConfigSnapshot(
            614,
            {
                "console": {
                    "exchange_capture": "true",
                    "trace_legacy_writes": "true",
                }
            },
        ),
    )

    policy = config_module.runtime_capture_policy()
    signals = controller._new_run_stream_signals()

    assert policy.enabled is True
    assert policy.legacy_writes_enabled is True
    assert signals.exchange_capture_enabled is True


def test_legacy_writer_requires_both_capture_and_legacy_rollout_gate(monkeypatch):
    controller = _new_controller()
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(
            enabled=True,
            detail=CaptureDetail.SAFE,
            generation=1,
            legacy_writes_enabled=True,
        ),
    )

    signals = controller._new_run_stream_signals()

    assert signals.exchange_capture_enabled is True


def test_shipped_config_default_resolves_exchange_capture_true():
    """Make the shipped [console] default itself load-bearing: read the
    REAL resolved settings layer with no controller-side default masking
    whether the TOML key is actually present (``default=None``, not
    ``True``) -- mirrors Tests/test_config_console_defaults.py's own
    default-pin style (e.g. ``test_console_sidechat_model_default_is_
    empty_string``), which calls ``get_cli_setting`` directly with no
    extra setup beyond the autouse ``isolate_test_environment`` fixture
    (Tests/conftest.py) that already redirects XDG_CONFIG_HOME/
    TLDW_CONFIG_PATH to a per-test temp directory -- this never touches
    the user's real config (a documented incident in this repo)."""
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting

    assert real_get_cli_setting("console", "exchange_capture", None) is True


def test_attach_site_forwards_captures_to_store():
    """The usage-attach method (``_attach_stream_usage``) forwards BOTH the
    usage payload AND ``signals.exchange_captures()`` to the store from the
    SAME call -- a usage payload is recorded on the call-scoped view before
    closing the exchange, mirroring the real gateway's
    ``record_usage_payload`` + ``begin_exchange``/``close_exchange`` +
    ``close_usage_call`` sequence on one ``new_usage_call()`` view
    (``console_provider_gateway.py``'s ``stream_chat``, llama.cpp branch,
    and its ``finally`` close-out order). Driven the same way
    ``test_re_attaching_the_same_signals_is_idempotent`` drives usage."""
    controller, store, message_id = _controller_with_placeholder()
    signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
    call_signals = signals.new_usage_call()
    call_signals.record_usage_payload(
        {"prompt_tokens": 100, "completion_tokens": 20}
    )
    call_signals.begin_exchange(
        provider="p", model="m", endpoint=None, request={}, omitted_keys=()
    )
    call_signals.close_exchange()
    call_signals.close_usage_call()
    resolution = SimpleNamespace(provider="openai", model="gpt-4o")

    controller._attach_stream_usage(message_id, signals, resolution, partial=False)

    message = store.get_message(message_id)
    assert message.usage is not None
    assert message.usage.total_tokens == 120, "the usage attach must still land"
    assert message.exchanges, "the controller must forward exchange_captures() to the store"
    assert message.exchanges[0].provider == "p"


def test_attach_forwards_captures_even_without_usage():
    """Exchange capture must not be gated on a nonzero usage total -- these
    signals carry an exchange but no usage payload at all (no
    ``record_usage_payload`` call), which would otherwise make
    ``_attach_stream_usage`` return before ever reaching the exchange
    attach if it were nested inside the usage-total branch."""
    controller, store, message_id = _controller_with_placeholder()
    signals = _captured_signals()
    resolution = SimpleNamespace(provider="openai", model="gpt-4o")

    controller._attach_stream_usage(message_id, signals, resolution, partial=False)

    assert store.get_message(message_id).exchanges


def test_attach_skips_the_store_call_when_nothing_was_captured(monkeypatch):
    """No captures -> no store call at all (same shape as usage's own
    "nothing to bill" early return) -- proven via an instance-level
    monkeypatch spy, since ``ConsoleProviderStreamSignals()`` with no
    ``begin_exchange`` produces an empty ``exchange_captures()``."""
    controller, store, message_id = _controller_with_placeholder()
    calls = []
    original = store.attach_message_exchanges

    def _spy(mid, captures):
        calls.append((mid, captures))
        return original(mid, captures)

    monkeypatch.setattr(store, "attach_message_exchanges", _spy)
    signals = ConsoleProviderStreamSignals()  # no exchanges recorded

    controller._attach_stream_usage(message_id, signals, resolution=SimpleNamespace(
        provider="openai", model="gpt-4o"), partial=False)

    assert calls == []


def test_attach_never_fails_the_send(monkeypatch):
    """Store raising from ``attach_message_exchanges`` is swallowed+logged --
    same never-fail contract as ``usage_attach_failed``. Mirrors
    Tests/Chat/test_console_chat_store_exchanges.py's
    ``test_persist_exchanges_only_survives_a_serialization_failure`` for the
    loguru-sink diagnostics-capture idiom."""
    controller, store, message_id = _controller_with_placeholder()

    def _raise(mid, captures):
        raise RuntimeError("SENSITIVE CANARY")

    monkeypatch.setattr(store, "attach_message_exchanges", _raise)

    signals = _captured_signals()
    resolution = SimpleNamespace(provider="openai", model="gpt-4o")

    diagnostics: list[str] = []
    sink_id = loguru_logger.add(
        diagnostics.append,
        level="WARNING",
        format="{extra[message_id]} {extra[error_type]} {message}",
    )
    try:
        # Must not raise.
        controller._attach_stream_usage(
            message_id, signals, resolution, partial=False
        )
    finally:
        loguru_logger.remove(sink_id)

    assert any("exchange_attach_failed" in d for d in diagnostics), diagnostics
    assert any("RuntimeError" in d for d in diagnostics), diagnostics
    assert all("SENSITIVE CANARY" not in d for d in diagnostics), diagnostics
