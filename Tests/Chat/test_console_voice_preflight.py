"""Effect-free voice readiness and safe preparation failure contracts."""

import asyncio
import threading
import os
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings


def controller_for(gateway, *, persistence=None):
    store = ConsoleChatStore(persistence=persistence)
    store.create_session(
        settings=ConsoleSessionSettings(provider="deepseek", model="deepseek-chat")
    )
    return ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=False
    )


@pytest.mark.asyncio
async def test_readiness_only_resolves_selected_provider_and_returns_fenced_stamp(
    monkeypatch,
):
    calls = []

    async def resolve(selection):
        calls.append(selection)
        return SimpleNamespace(ready=True)

    controller = controller_for(SimpleNamespace(resolve_for_send=resolve))
    assert callable(getattr(controller, "validate_speculative_voice_entry", None))
    stamp = await controller.validate_speculative_voice_entry()
    assert controller.is_speculative_voice_entry_current(stamp)
    assert calls[0].provider == "deepseek"
    assert calls[0].explicit_model == "deepseek-chat"
    assert (
        controller.store.messages_for_session(controller.store.active_session_id) == []
    )
    original = controller.store.session_settings(controller.store.active_session_id)
    controller.store.replace_session_settings(
        controller.store.active_session_id, replace(original, model="other")
    )
    controller.store.replace_session_settings(
        controller.store.active_session_id, original
    )
    assert not controller.is_speculative_voice_entry_current(stamp)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["session", "provider"])
async def test_real_controller_preparation_uses_trusted_failure_category(failure):
    async def resolve(selection):
        return SimpleNamespace(ready=False, visible_copy="SECRET-credential-endpoint")

    controller = controller_for(SimpleNamespace(resolve_for_send=resolve))
    if failure == "session":
        controller.store._activate_session(None)
    with pytest.raises(Exception) as caught:
        await controller.prepare_speculative_voice_attempt(
            attempt_epoch=1, transcript="private speech", turn_id="turn"
        )
    assert getattr(caught.value, "category", None) == f"{failure}_unavailable"
    assert "SECRET" not in str(caught.value)


@pytest.mark.asyncio
async def test_readiness_rejects_config_change_during_resolution(monkeypatch):
    from tldw_chatbook import config

    generation = [config.get_runtime_config_snapshot().generation]
    monkeypatch.setattr(
        config,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(generation=generation[0]),
    )
    monkeypatch.setattr(
        config,
        "run_if_runtime_config_generation_current",
        lambda expected, action: expected == generation[0] and action(),
    )

    async def resolve(selection):
        generation[0] += 1
        return SimpleNamespace(ready=True)

    controller = controller_for(SimpleNamespace(resolve_for_send=resolve))
    assert callable(getattr(controller, "validate_speculative_voice_entry", None))
    with pytest.raises(Exception) as caught:
        await controller.validate_speculative_voice_entry()
    assert getattr(caught.value, "category", None) == "stale"


@pytest.mark.asyncio
@pytest.mark.parametrize("publications", [0, 1, 2])
async def test_readiness_retains_generation_before_initial_snapshot_wait(
    monkeypatch, publications
):
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_voice_preflight import VoicePreparationError

    initial = config.get_runtime_config_snapshot()
    entered, release = threading.Event(), threading.Event()
    probe_calls = []
    read_snapshot = config.get_runtime_config_snapshot

    def blocked_snapshot():
        entered.set()
        assert release.wait(5)
        return read_snapshot()

    async def resolve(selection):
        probe_calls.append(selection)
        return SimpleNamespace(ready=True)

    controller = controller_for(SimpleNamespace(resolve_for_send=resolve))
    monkeypatch.setattr(config, "get_runtime_config_snapshot", blocked_snapshot)
    task = asyncio.create_task(controller.validate_speculative_voice_entry())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        original = initial.values.get("dictation", {}).get("response_eagerness_ms", 700)
        for value in [original + 1, original][:publications]:
            assert config.save_setting_to_cli_config(
                "dictation", "response_eagerness_ms", value
            )
        release.set()
        if publications:
            with pytest.raises(VoicePreparationError, match="stale"):
                await task
            assert probe_calls == []
        else:
            stamp = await task
            assert controller.is_speculative_voice_entry_current(stamp)
            assert len(probe_calls) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_readiness_cancellation_propagates():
    entered = asyncio.Event()

    async def resolve(selection):
        entered.set()
        await asyncio.Future()

    controller = controller_for(SimpleNamespace(resolve_for_send=resolve))
    assert callable(getattr(controller, "validate_speculative_voice_entry", None))
    task = asyncio.create_task(controller.validate_speculative_voice_entry())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@asynccontextmanager
async def process_failure_session(monkeypatch, failure="provider"):
    from Tests.Audio.test_voice_process_core import core
    from tldw_chatbook.Audio.voice_process_entry import _ChildEffects
    from tldw_chatbook.Audio.voice_process_lifetime import LifecyclePipe
    from tldw_chatbook.Audio.voice_process_types import ControlKind
    from tldw_chatbook.Chat import console_speculative_voice_session as module
    from tldw_chatbook.Chat import console_voice_input
    from tldw_chatbook.Chat.console_voice_process import VoiceProcessSupervisor
    from tldw_chatbook.Chat.console_voice_promotion import VoicePromotionOwner
    from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor

    async def ready(_selection):
        return SimpleNamespace(ready=True)

    controller = controller_for(SimpleNamespace(resolve_for_send=ready))
    stamp = await controller.validate_speculative_voice_entry()
    notices, drafts, events, faults = [], [], [], []
    view = SimpleNamespace(
        _ensure_console_chat_controller=lambda: controller,
        _append_to_console_draft=drafts.append,
        is_mounted=True,
        _console_dictation_state="idle",
        _hands_free=SimpleNamespace(_qualified_voice_generation=1),
    )
    app = SimpleNamespace(notify=lambda message, **kw: notices.append(message))
    monkeypatch.setattr(
        console_voice_input,
        "resolve",
        lambda: SimpleNamespace(provider="faster-whisper", model="tiny", language="en"),
    )
    monkeypatch.setattr(
        module, "_persist_voice_event", lambda event, **kw: events.append((event, kw))
    )
    parent = module.create_console_speculative_voice_session(
        app_instance=app,
        view=view,
        promotion_owner=VoicePromotionOwner(lambda: object()),
        dispatch_supervisor=VoiceDispatchSupervisor(),
        process_supervisor=VoiceProcessSupervisor(),
        entry_current=lambda: controller.is_speculative_voice_entry_current(stamp),
        project_preview=lambda _: None,
        clear_preview=lambda: None,
    )

    async def refused(selection):
        if failure == "unknown":
            raise RuntimeError("SECRET-credential-endpoint")
        return SimpleNamespace(ready=False, visible_copy="SECRET-credential-endpoint")

    controller.provider_gateway.resolve_for_send = refused
    if failure == "session":
        original = parent._bindings[0]

        async def missing(**kwargs):
            owner = controller.store.active_session_id
            controller.store.active_session_id = None
            try:
                return await original(**kwargs)
            finally:
                controller.store.active_session_id = owner

        parent._bindings = (missing, *parent._bindings[1:])
    child_read, parent_write = os.pipe()
    parent_read, child_write = os.pipe()
    child = None
    parent.pipe = LifecyclePipe(
        parent_read,
        parent_write,
        generation=1,
        request_id=parent._bootstrap.header["request_id"],
        parent=True,
        consume=parent._receive,
        fault=parent._transport_fault,
    )
    parent._compose_parent()
    pipe = LifecyclePipe(
        child_read,
        child_write,
        generation=1,
        request_id=parent._bootstrap.header["request_id"],
        parent=False,
        consume=lambda record: child.receive(record),
        fault=faults.append,
    )
    audio, transport, _, scheduler = core(deferred_attempt_preparation=True)
    child = _ChildEffects(pipe, transport, lambda: audio, faults.append)
    audio._delegate_effects = child
    audio._effects._delegate = child
    await audio.coordinator.start()

    async def start():
        from Tests.Chat.test_console_speculative_voice import _speech, _revision

        frame = _speech(0, started_ns=scheduler.now_ns)
        await audio.submit(frame)
        turn = audio.coordinator.snapshot.turn_id
        revision = _revision(turn, 1, "private speech", frame.ended_ns)
        child.record_revision(revision)
        await audio.submit(revision)
        scheduler.advance_ms(audio.coordinator.response_eagerness_ms)
        await audio.coordinator.flush()
        return turn

    try:
        yield SimpleNamespace(
            parent=parent,
            child=child,
            audio=audio,
            controller=controller,
            scheduler=scheduler,
            view=view,
            start=start,
            notices=notices,
            drafts=drafts,
            events=events,
            faults=faults,
            child_pipe=pipe,
            write_fds=(parent_write, child_write),
        )
    finally:
        try:
            child.fence_all()
            await parent._effects.aclose()
            # Actual parent receipts, never fabricated child cleanup completion.
            for state in child.attempts.values():
                if not state.cleanup.done():
                    await asyncio.wait_for(asyncio.shield(state.cleanup), 1)
            await audio.fence_and_close(ControlKind.TEARDOWN)
            for task in tuple(child.tasks):
                task.cancel()
            await asyncio.gather(*tuple(child.tasks), return_exceptions=True)
        finally:
            parent.pipe.stop()
            pipe.stop()
            os.close(parent_write)
            os.close(child_write)
            joined = await asyncio.gather(parent.pipe.join(), pipe.join())
            assert all(joined)


@pytest.mark.asyncio
async def test_process_helper_closes_private_pipes_after_cleanup_error(monkeypatch):
    with pytest.raises(RuntimeError, match="fixture cleanup failure"):
        async with process_failure_session(monkeypatch) as h:

            async def failed_close():
                raise RuntimeError("fixture cleanup failure")

            monkeypatch.setattr(h.parent._effects, "aclose", failed_close)
            pipes = (h.parent.pipe, h.child_pipe)
            descriptors = (*h.write_fds, *(pipe.reader.fd for pipe in pipes))
    for pipe in pipes:
        assert not pipe.reader.alive and not pipe.writer.alive
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["session", "provider", "unknown"])
async def test_real_process_preparation_failure_recovers_without_transport_fault(
    monkeypatch, failure
):
    from Tests.Audio.test_voice_process_core import eventually
    from tldw_chatbook.Chat.console_speculative_voice import SpeculativeVoiceState

    async with process_failure_session(monkeypatch, failure) as h:
        turn = await h.start()
        await eventually(lambda: h.drafts or h.parent._failure is not None)
        assert h.parent._failure is None and h.faults == []
        assert h.audio.state is SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE
        assert h.drafts == ["private speech"]
        assert len(h.notices) == 1 and "No reply started" in h.notices[0]
        assert "SECRET" not in str(h.notices) + str(h.events)
        assert "private speech" not in str(h.notices) + str(h.events)
        category = "unexpected" if failure == "unknown" else f"{failure}_unavailable"
        assert h.events == [
            (
                "attempt_prepare_failed",
                dict(
                    status="failed",
                    error_category=category,
                    exception_type="RuntimeError"
                    if failure == "unknown"
                    else "VoicePreparationError",
                ),
            )
        ]
        h.child.preserve_draft(
            turn_id=turn, transcript="private speech", reason="provider_failed"
        )
        await asyncio.sleep(0)
        assert h.drafts == ["private speech"] and len(h.notices) == 1
        h.scheduler.advance_ms(10000)
        await h.audio.coordinator.flush()
        assert len(h.child.attempts) == 1  # No automatic retry.


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["duplicate", "unknown_epoch", "duplicate_revoke"])
async def test_process_recovery_rejects_duplicate_or_unissued_identity(
    monkeypatch, mutation
):
    from Tests.Audio.test_voice_process_core import eventually
    from tldw_chatbook.Audio.voice_process_protocol import Record

    async with process_failure_session(monkeypatch) as h:
        turn = await h.start()
        await eventually(lambda: bool(h.drafts))
        key, token, _ = h.parent._recovery_tokens[turn]

        def record(action, epoch=key.epoch, recovery_id=token):
            return Record(
                dict(
                    version=1,
                    generation=key.generation,
                    request_id=key.request_id,
                    sequence=1,
                    op="draft_recovery",
                    turn_id=turn,
                    revision=key.revision,
                    epoch=epoch,
                    action=action,
                    recovery_id=recovery_id,
                )
            )

        values = (
            [record("revoke"), record("revoke")]
            if mutation == "duplicate_revoke"
            else [record("request", key.epoch + 1, token + 1)]
            if mutation == "unknown_epoch"
            else [record("request")]
        )
        # Already-received bounded control batch; no network/device/provider.
        h.parent._pending_records.extend(values)
        h.parent._dispatch_batch()
        await asyncio.sleep(0)
        assert h.parent._failure is not None
        assert h.drafts == ["private speech"] and len(h.notices) == 1


@pytest.mark.asyncio
async def test_process_recovery_waits_for_actual_late_draft_consumption(monkeypatch):
    from Tests.Audio.test_voice_process_core import eventually

    async with process_failure_session(monkeypatch) as h:
        held, release = asyncio.Event(), asyncio.Event()
        dispatch = h.parent._dispatch_record

        async def hold_draft(record, revoked):
            if record.header["op"] == "draft":
                held.set()
                await release.wait()
            return await dispatch(record, revoked)

        h.parent._dispatch_record = hold_draft
        try:
            turn = await h.start()
            await held.wait()
            await eventually(lambda: bool(h.child.recoveries))
            assert not h.parent._recovery_tokens
            assert h.parent._effects.draft_for(turn) is None
            assert h.drafts == [] and h.notices == []
            release.set()
            await eventually(lambda: bool(h.drafts))
            assert h.drafts == ["private speech"] and len(h.notices) == 1
            assert h.parent._failure is None and h.faults == []
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "closed",
        "session_aba",
        "resumed_speech",
        "cancelled",
        "view_replaced",
        "config_aba",
        "claimed",
        "received_revoke",
    ],
)
async def test_queued_preparation_explanation_checks_currency_at_delivery(
    monkeypatch, change
):
    from Tests.Audio.test_voice_process_core import eventually
    from Tests.Chat.test_console_speculative_voice import _speech
    from tldw_chatbook.Chat.console_speculative_voice import ManualInterruption
    from tldw_chatbook import config

    async with process_failure_session(monkeypatch) as h:
        queued, release = asyncio.Event(), asyncio.Event()
        deliver = h.parent._preserve_draft

        async def queued_delivery(text, **kwargs):
            queued.set()
            await release.wait()
            return deliver(text, **kwargs)

        h.parent._preserve_draft = queued_delivery
        dispatch_batch = h.parent._dispatch_batch
        try:
            turn = await h.start()
            await asyncio.wait_for(queued.wait(), 1)
            if change == "closed":
                h.view.is_mounted = False
            elif change == "view_replaced":
                h.view._hands_free._qualified_voice_generation += 1
            elif change == "session_aba":
                session_id = h.controller.store.active_session_id
                h.controller.store.create_session()
                h.controller.store._activate_session(session_id)
            elif change == "config_aba":
                original = config.get_cli_setting(
                    "dictation", "response_eagerness_ms", 700
                )
                assert config.save_setting_to_cli_config(
                    "dictation", "response_eagerness_ms", original + 1
                )
                assert config.save_setting_to_cli_config(
                    "dictation", "response_eagerness_ms", original
                )
            elif change == "claimed":
                h.parent._claimed.add(turn)
            elif change == "received_revoke":
                h.parent._dispatch_batch = lambda: None
                await h.audio.submit(ManualInterruption())
                await eventually(
                    lambda: any(
                        record.header["op"] == "draft_recovery"
                        for record in h.parent._pending_records
                    )
                )
            else:
                event = (
                    _speech(1, started_ns=h.scheduler.now_ns)
                    if change == "resumed_speech"
                    else ManualInterruption()
                )
                await h.audio.submit(event)
                await eventually(lambda: h.parent._recovery_tokens[turn][2] is False)
                # Further local activity does not republish this revoke.
                await h.audio.submit(ManualInterruption())
            assert h.parent._effects.draft_for(turn) == "private speech"
            assert turn in h.parent._recovery_tasks
            release.set()
            await eventually(lambda: not h.parent._recovery_tasks)
            assert h.drafts == [] and h.notices == [] and h.events == []
            assert h.parent._failure is None
        finally:
            release.set()
            h.parent._dispatch_batch = dispatch_batch
            if h.parent._pending_records:
                dispatch_batch()


@pytest.mark.parametrize("typed", [False, True])
def test_failure_category_rejects_spoofed_or_mutated_categories(typed):
    from tldw_chatbook.Chat.console_voice_preflight import (
        VoicePreparationError,
        voice_failure_category,
    )

    error = (
        VoicePreparationError("provider_unavailable")
        if typed
        else RuntimeError("voice_provider_unavailable SECRET")
    )
    error.category = "SECRET-credential-endpoint" if typed else "provider_unavailable"
    assert voice_failure_category(error) == "unexpected"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["stale", "cancelled", "fenced"])
async def test_obsolete_preparation_failures_do_not_emit_diagnostics_or_drafts(
    monkeypatch, failure
):
    from Tests.Audio.test_voice_process_core import eventually
    from tldw_chatbook.Chat.console_voice_preflight import VoicePreparationError

    async with process_failure_session(monkeypatch) as h:
        entered, release = asyncio.Event(), asyncio.Event()
        cleanup_receipts = []
        send = h.parent.pipe.send

        def observe_send(op, **fields):
            if op == "cleanup":
                cleanup_receipts.append(fields)
            return send(op, **fields)

        h.parent.pipe.send = observe_send

        async def prepare(**kwargs):
            entered.set()
            await release.wait()
            if failure == "cancelled":
                raise asyncio.CancelledError()
            raise (
                VoicePreparationError("stale")
                if failure == "stale"
                else RuntimeError("SECRET")
            )

        h.parent._effects._prepare_attempt = prepare
        try:
            await h.start()
            await entered.wait()
            state = next(iter(h.child.attempts.values()))
            if failure == "fenced":
                h.child.fence_attempt(state.key.epoch)
                await eventually(lambda: h.parent._is_revoked(state.key))
            release.set()
            await eventually(lambda: state.cleanup.done())
            await asyncio.gather(*tuple(h.parent._parent_tasks))
            assert len(cleanup_receipts) == 1
            assert h.notices == [] and h.drafts == [] and h.events == []
            assert h.parent._failure is None
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize(
    "transform",
    ["none", "dictionary", "world_info", "dictionary_noop", "world_info_noop"],
)
async def test_real_deepseek_controller_prepares_without_generation_or_persistence(
    monkeypatch, tmp_path, capture, compacted, transform
):
    import httpx
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_context_compaction import (
        EffectiveMemoryKind,
        EffectiveMemoryResult,
        LegacyMemorySnapshot,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    cfg = {
        "api_settings": {
            "deepseek": {
                "api_key": "sk-hermetic-test-credential",
                "api_url": "https://api.deepseek.com",
            }
        },
        "providers": {"deepseek": ["deepseek-chat"]},
    }

    def forbid_network(request):
        raise AssertionError("Preparation must not send a generation request")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(forbid_network)
    ) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: cfg, environ={}
        )
        db = CharactersRAGDB(tmp_path / "voice-preparation.db", "voice-preparation")
        controller = controller_for(gateway, persistence=ChatPersistenceService(db))
        session = controller.store.sessions()[0]
        session.persisted_conversation_id = db.add_conversation(
            {"title": "voice preparation"}
        )
        controller._provider_config = lambda: cfg
        controller._staged_evidence_provider = lambda _: False
        transformed = transform in {"dictionary", "world_info"}
        transform_calls = []

        def dictionary_applier(conversation_id, text, frozen=None):
            transform_calls.append((conversation_id, text, frozen))
            return "dictionary expanded user" if transformed else text

        def world_info_applier(conversation_id, text, history, frozen=None):
            transform_calls.append((conversation_id, text, frozen))
            assert history == []
            return "world-info expanded user" if transformed else text

        if transform.startswith("dictionary"):
            controller._chat_dictionary_applier = dictionary_applier
        elif transform.startswith("world_info"):
            controller._world_info_applier = world_info_applier
        controller.set_next_trace_privacy(
            session.id,
            capture_enabled=capture,
            pii_redaction_enabled=True,
            expected_policy_revision=controller.store._capture_policy_revision,
        )
        snapshot = controller.capture_policy_snapshot(session.id)
        assert snapshot.error_code is None
        if compacted:
            inputs = controller.context_control_inputs(session.id)
            monkeypatch.setattr(
                controller,
                "context_control_inputs",
                lambda _session_id: (
                    *inputs[:2],
                    EffectiveMemoryResult(
                        EffectiveMemoryKind.LEGACY_PREFIX,
                        legacy=LegacyMemorySnapshot(
                            session.persisted_conversation_id,
                            "prior summary",
                            "prior-boundary",
                        ),
                    ),
                ),
            )
        prepared = await controller.prepare_speculative_voice_attempt(
            attempt_epoch=1, transcript="private speech", turn_id="turn"
        )
        try:
            assert prepared.request.resolution.provider == "deepseek"
            assert prepared.request.resolution.model == "deepseek-chat"
            expected_content = {
                "dictionary": "dictionary expanded user",
                "world_info": "world-info expanded user",
            }.get(transform, "private speech")
            assert prepared.request.prepared.messages_payload[-1] == {
                "role": "user",
                "content": expected_content,
            }
            if transform != "none":
                assert len(transform_calls) == 1
                assert transform_calls[0][:2] == (
                    session.persisted_conversation_id,
                    "private speech",
                )
                assert (
                    transform_calls[0][2]["conversation_id"]
                    == session.persisted_conversation_id
                )
            requires_authority = compacted or transformed
            assert prepared.requires_pre_dispatch_authority is requires_authority
            assert (prepared.request.provisional_trace_attempt is not None) is (
                capture and not requires_authority
            )
            assert (
                prepared.promotion_seed.next_trace_privacy_revision
                == snapshot.next_privacy_revision
            )
            assert session.next_pii_redaction_enabled is True
            assert session.next_capture_enabled is capture
            if capture and not requires_authority:
                frozen = prepared.promotion_seed.capture_policy
                assert frozen.pii_redaction_enabled is True
                assert (
                    frozen.pii_ruleset_revision_id == snapshot.pii_ruleset_revision_id
                )
                assert (
                    prepared.request.prepared.semantic.provenance.capture_policy
                    == frozen
                )
            controller.set_next_trace_privacy(
                session.id,
                capture_enabled=not capture,
                pii_redaction_enabled=False,
                expected_policy_revision=controller.store._capture_policy_revision,
            )
            assert (
                prepared.promotion_seed.next_trace_privacy_revision
                == snapshot.next_privacy_revision
            )
            if capture and not requires_authority:
                assert prepared.promotion_seed.capture_policy == frozen
            assert (
                controller.store.messages_for_session(
                    controller.store.active_session_id
                )
                == []
            )
            assert (
                db.get_connection()
                .execute("SELECT count(*) FROM messages")
                .fetchone()[0]
                == 0
            )
        finally:
            if prepared.request.provisional_trace_attempt is not None:
                gateway.abandon_provisional_voice_trace(
                    prepared.request.provisional_trace_attempt
                )
            controller._scratch_spaces.dispose()
            await gateway.aclose()
            db.close_connection()


@pytest.mark.asyncio
async def test_late_preparation_failure_for_replaced_conversation_is_silent(
    monkeypatch,
):
    from Tests.Audio.test_voice_process_core import eventually

    async with process_failure_session(monkeypatch) as h:
        entered, release = asyncio.Event(), asyncio.Event()

        async def resolve(selection):
            entered.set()
            await release.wait()
            raise RuntimeError("SECRET-credential-endpoint")

        h.controller.provider_gateway.resolve_for_send = resolve
        try:
            await h.start()
            await entered.wait()
            state = next(iter(h.child.attempts.values()))
            h.controller.store.create_session()
            release.set()
            await eventually(lambda: state.cleanup.done())
            assert h.drafts == [] and h.notices == [] and h.events == []
            assert h.parent._failure is None
        finally:
            release.set()


@pytest.mark.asyncio
async def test_deleted_readiness_owner_is_stale_without_reading_its_settings():
    async def resolve(selection):
        controller.store.close_session(controller.store.active_session_id)
        return SimpleNamespace(ready=True)

    controller = controller_for(SimpleNamespace(resolve_for_send=resolve))
    with pytest.raises(Exception) as caught:
        await controller.validate_speculative_voice_entry()
    assert getattr(caught.value, "category", None) == "stale"
