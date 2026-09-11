"""App composition through fake process endpoints; never load native drivers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_voice_promotion import _winning_case, _inline
from tldw_chatbook.Chat.console_speculative_voice_session import (
    VoicePromotionSeed,
    create_console_speculative_voice_session,
)
from tldw_chatbook.Chat.console_voice_promotion import VoicePromotionOwner
from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Chat.console_voice_trace_gateway import ProvisionalTraceRegistry
from tldw_chatbook.Chat.console_voice_trace_promotion import (
    PostDispatchTraceSurfaceComponent,
    PostDispatchTraceResponse,
)
from Tests.Chat.test_console_voice_promotion import _post_dispatch_call
from Tests.Chat.test_console_voice_capture import _eligible


def _factory(monkeypatch, store, owner, gateway):
    from tldw_chatbook.Chat import (
        console_voice_process,
        console_voice_input,
        console_runtime,
    )
    from tldw_chatbook import config

    monkeypatch.setattr(
        console_voice_input,
        "resolve",
        lambda: SimpleNamespace(provider="fixture", model="fixture", language="en"),
    )
    monkeypatch.setattr(
        config, "get_cli_setting", lambda _section, _key, default=None: default
    )
    # Qualification/source identity belongs to Task 4. This test cannot create
    # a child process: the endpoint captures only the app's bound callbacks.
    monkeypatch.setattr(
        console_voice_process,
        "bootstrap_record",
        lambda **kwargs: SimpleNamespace(header={"generation": kwargs["generation"]}),
    )

    class Endpoint:
        def __init__(self, _lease, **bindings):
            self.bindings = bindings

    monkeypatch.setattr(console_voice_process, "ConsoleVoiceProcess", Endpoint)
    supervisor = SimpleNamespace(device_lease=object(), retain=lambda _engine: None)
    monkeypatch.setattr(
        console_runtime,
        "ensure_console_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(voice_process_supervisor=supervisor),
    )
    controller = SimpleNamespace(
        store=store,
        provider_gateway=gateway,
        prepare_speculative_voice_attempt=lambda **_kwargs: None,
        submit_accepted_voice_turn=lambda *_args, **_kwargs: None,
    )
    view = SimpleNamespace(
        _ensure_console_chat_controller=lambda: controller,
        _hands_free=SimpleNamespace(_qualified_voice_generation=1),
        is_mounted=True,
        _console_dictation_state="idle",
    )
    return create_console_speculative_voice_session(
        app_instance=SimpleNamespace(),
        view=view,
        promotion_owner=owner,
        dispatch_supervisor=VoiceDispatchSupervisor(),
        project_preview=lambda _value: None,
        clear_preview=lambda: None,
        entry_current=lambda: True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [False, True])
async def test_factory_consumes_frozen_privacy_only_after_winning_claim(
    monkeypatch, capture
):
    store, session_id, context, _commit, registry, snapshot, _events = _winning_case()
    session = next(item for item in store.sessions() if item.id == session_id)
    store.set_session_next_trace_privacy(
        session_id,
        capture_enabled=capture,
        pii_redaction_enabled=True,
        expected_policy_revision=store._capture_policy_revision,
    )
    revision = session.next_privacy_revision
    seed = VoicePromotionSeed(
        context.promotion_id,
        context.attempt_id,
        context.terminal_boundary_id,
        context.origin,
        context.expected_native_leaf_id,
        context.expected_persisted_leaf_id,
        capture,
        capture_policy=FrozenTracePolicy(
            context.promotion_id, "credentials-v1", False, None
        ),
        next_trace_privacy_revision=revision,
    )
    prepared = SimpleNamespace(
        promotion_seed=seed,
        request=SimpleNamespace(
            resolution=SimpleNamespace(provider="openai", model="fixture")
        ),
    )
    if not capture:
        snapshot = replace(snapshot, trace_manifest=None, trace_envelopes=())
    owner = VoicePromotionOwner(lambda: store, sync_runner=_inline)
    engine = _factory(
        monkeypatch, store, owner, SimpleNamespace(provisional_trace_registry=registry)
    )
    assert session.next_privacy_revision == revision
    observed = []

    def claimed(_claim):
        observed.append(
            (session.next_capture_enabled, session.next_pii_redaction_enabled)
        )

    outcome = await engine.bindings["promote"](
        prepared=prepared,
        snapshot=snapshot,
        transcript=context.user_text,
        assistant_text=context.assistant_text,
        on_claim=claimed,
    )
    assert outcome._commit is not None
    assert observed == [(None, None)]
    assert session.next_privacy_revision == revision + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("import_fails", [False, True])
async def test_factory_imports_actual_trace_only_after_actual_pair_commit(
    monkeypatch, tmp_path, import_fails
):
    db = CharactersRAGDB(str(tmp_path / "factory.db"), "voice-factory")
    try:
        store, session_id, context, _commit, _registry, snapshot, _events = (
            _winning_case()
        )
        session = next(item for item in store.sessions() if item.id == session_id)
        conversation = db.add_conversation({"title": "factory voice"})
        session.persisted_conversation_id = conversation
        context = replace(
            context,
            origin=replace(context.origin, persisted_conversation_id=conversation),
        )
        store.persistence = ChatPersistenceService(db)
        frozen = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
        registry = ProvisionalTraceRegistry()
        attempt = registry.begin_attempt(
            promotion_id=context.promotion_id,
            attempt_id=context.attempt_id,
            policy=frozen,
            eligibility=_eligible(),
        )
        call = _post_dispatch_call(context.promotion_id)
        call = replace(
            call,
            request_surface=(
                PostDispatchTraceSurfaceComponent.omission(
                    node_id=call.request_surface[0].node_id,
                    component_kind="provider_message",
                    reason_code="voice_user_revision_pending",
                ),
            ),
            response=PostDispatchTraceResponse.no_response(
                "voice_assistant_revision_pending"
            ),
        )
        envelope = registry._retain_gateway_call(attempt, call)
        manifest = registry.seal_attempt(attempt, expected_call_count=1)
        snapshot = replace(
            snapshot, trace_manifest=manifest, trace_envelopes=(envelope,)
        )
        seed = VoicePromotionSeed(
            context.promotion_id,
            context.attempt_id,
            context.terminal_boundary_id,
            context.origin,
            None,
            None,
            True,
            capture_policy=frozen,
        )
        prepared = SimpleNamespace(
            promotion_seed=seed,
            request=SimpleNamespace(
                resolution=SimpleNamespace(provider="openai", model="fixture")
            ),
        )
        owner = VoicePromotionOwner(lambda: store, sync_runner=_inline)
        engine = _factory(
            monkeypatch,
            store,
            owner,
            SimpleNamespace(provisional_trace_registry=registry),
        )
        if import_fails:
            from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

            def fail_import(*_args, **_kwargs):
                assert (
                    db.get_connection()
                    .execute("SELECT count(*) FROM messages")
                    .fetchone()[0]
                    == 2
                )
                raise RuntimeError("fixture trace failure after pair")

            monkeypatch.setattr(
                ConsoleTraceService, "import_provisional_voice_trace", fail_import
            )
        assert (
            db.get_connection().execute("SELECT count(*) FROM messages").fetchone()[0]
            == 0
        )
        outcome = await engine.bindings["promote"](
            prepared=prepared,
            snapshot=snapshot,
            transcript=context.user_text,
            assistant_text=context.assistant_text,
        )
        assert outcome._commit is not None
        rows = (
            db.get_connection()
            .execute("SELECT content FROM messages ORDER BY rowid")
            .fetchall()
        )
        assert [row[0] for row in rows] == [context.user_text, context.assistant_text]
        rows = (
            db.get_connection()
            .execute("SELECT call_id, reservation_provenance FROM console_trace_calls")
            .fetchall()
        )
        assert [tuple(row) for row in rows] == (
            [] if import_fails else [(call.call_id, "post_dispatch_promoted")]
        )
        assert registry.retained_bytes == 0
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_real_saved_controller_sealed_headers_import_through_winning_claim(
    tmp_path,
):
    import json
    import httpx
    from Tests.Chat.test_console_voice_preflight import controller_for
    from Tests.Chat.test_console_voice_trace_repository import _retained_call_bytes
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        reconstruct_provider_gateway_kwargs,
    )
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService
    from tldw_chatbook.Chat.console_voice_trace_gateway import VoiceTraceImportContext

    cfg = {
        "api_settings": {
            "deepseek": {
                "api_key": "sk-fixture-credential",
                "api_url": "https://api.deepseek.com",
            }
        },
        "providers": {"deepseek": ["deepseek-chat"]},
    }

    def forbid_network(_request):
        raise AssertionError("No provider inference or network is allowed")

    class ObservedRepository(ConsoleTraceRepository):
        def import_post_dispatch_trace(self, database, request):
            self.request = request
            return super().import_post_dispatch_trace(database, request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(forbid_network)
    ) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: cfg, environ={}
        )
        db = CharactersRAGDB(
            tmp_path / "actual-voice-header.sqlite", "actual-voice-header"
        )
        controller = controller_for(gateway, persistence=ChatPersistenceService(db))
        session = controller.store.sessions()[0]
        conversation = db.add_conversation({"title": "actual voice header"})
        session.persisted_conversation_id = conversation
        controller._provider_config = lambda: cfg
        controller._staged_evidence_provider = lambda _: False
        controller.set_next_trace_privacy(
            session.id,
            capture_enabled=True,
            pii_redaction_enabled=True,
            expected_policy_revision=controller.store._capture_policy_revision,
        )
        prepared = None
        try:
            prepared = await controller.prepare_speculative_voice_attempt(
                attempt_epoch=1,
                transcript="original spoken user",
                turn_id="fixture-turn",
            )
            assert prepared.requires_pre_dispatch_authority is False
            attempt = prepared.request.provisional_trace_attempt
            assert attempt is not None
            registry = gateway.provisional_trace_registry
            boundary = registry._begin_gateway_call(
                attempt, gateway._retain_provisional_voice_trace_call
            )
            boundary.reserve()
            request = prepared.request.prepared
            resolution = prepared.request.resolution
            bundle = gateway._verify_trace_shadow(
                resolution,
                request,
                reconstruct_provider_gateway_kwargs(resolution, request),
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                trace_call_boundary=boundary,
            )
            boundary.mark_dispatch_started(bundle, request.provenance)
            boundary.mark_response_started()
            envelope = boundary.settle_response(
                usage=SimpleNamespace(
                    to_json=lambda: (
                        '{"api_key":"fixture-usage-secret","input_tokens":1}'
                    ),
                )
            )
            manifest = registry.seal_attempt(attempt, expected_call_count=1)
            user = db.add_message(
                {
                    "conversation_id": conversation,
                    "sender": "user",
                    "role": "user",
                    "content": "original spoken user",
                }
            )
            assistant = db.add_message(
                {
                    "conversation_id": conversation,
                    "parent_message_id": user,
                    "sender": "assistant",
                    "role": "assistant",
                    "content": "fixture spoken reply",
                    "assistant_generation_state": "complete",
                }
            )
            revisions = {
                row[0]: row[1]
                for row in db.get_connection().execute(
                    "SELECT source_message_id, revision_id FROM console_trace_semantic_revisions WHERE source_conversation_id = ?",
                    (conversation,),
                )
            }
            context = VoiceTraceImportContext(
                import_id=prepared.promotion_seed.promotion_id,
                conversation_id=conversation,
                user_message_id=user,
                user_revision_id=revisions[user],
                assistant_message_id=assistant,
                assistant_revision_id=revisions[assistant],
                turn_id=user,
                run_id="actual-voice-run",
                policy=prepared.promotion_seed.capture_policy,
            )
            repository = ObservedRepository()
            result = ConsoleTraceService(
                repository=repository
            ).import_provisional_voice_trace(
                db,
                registry,
                manifest,
                (envelope,),
                context,
            )
            call = repository.request.calls[0]
            assert result.call_ids == (call.call_id,)
            assert call.sealed_payload_bytes == _retained_call_bytes(call)
            assert (
                repository.request.aggregate_payload_bytes == call.sealed_payload_bytes
            )
            defaults = json.loads(call.adapter_defaults_json)
            assert "credential_source" not in defaults
            assert defaults["normalization_version"] == "canonical-json-v1"
            assert defaults["handler_projection"]
            assert json.loads(call.usage_json) == {"input_tokens": 1}
            with db.transaction() as cursor:
                stored = cursor.execute(
                    "SELECT generation_parameters_json, adapter_defaults_json, response_format_json, reasoning_controls_json "
                    "FROM console_trace_request_headers WHERE header_id = ?",
                    (repository.get_call(cursor, call.call_id).request_header_id,),
                ).fetchone()
                assert tuple(stored) == (
                    call.generation_parameters_json,
                    call.adapter_defaults_json,
                    call.response_format_json,
                    call.reasoning_controls_json,
                )
            assert registry.retained_bytes == 0
            assert db.get_message_by_id(user)["content"] == "original spoken user"
            assert db.get_message_by_id(assistant)["content"] == "fixture spoken reply"
        finally:
            if (
                prepared is not None
                and prepared.request.provisional_trace_attempt is not None
            ):
                gateway.abandon_provisional_voice_trace(
                    prepared.request.provisional_trace_attempt
                )
            controller._scratch_spaces.dispose()
            await gateway.aclose()
            db.close_connection()


def test_real_gateway_seals_dev_revision_kinds_through_winning_claim():
    from Tests.Chat.test_console_voice_capture import (
        _policy,
        _context,
        _result,
        _RecordingRepository,
    )
    from tldw_chatbook.Chat.console_chat_controller import (
        _build_speculative_voice_capture_request,
    )
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
        reconstruct_provider_gateway_kwargs,
    )
    from tldw_chatbook.Chat.console_trace_provenance import (
        ConsoleRequestRoute,
        ConsoleTraceCaptureMode,
        SavedRevisionTraceProvenance,
    )
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

    gateway = ConsoleProviderGateway(http_client=object())
    promotion_id = new_opaque_id()
    attempt = gateway.begin_provisional_voice_trace(
        promotion_id=promotion_id,
        attempt_id=new_opaque_id(),
        policy=_policy(),
        eligibility=_eligible(),
    )
    registry = gateway.provisional_trace_registry
    boundary = registry._begin_gateway_call(
        attempt, gateway._retain_provisional_voice_trace_call
    )
    boundary.reserve()
    saved = SavedRevisionTraceProvenance(new_opaque_id())
    semantic = _build_speculative_voice_capture_request(
        messages=(
            {"role": "user", "content": "saved history", "_native_message_id": "saved"},
            {"role": "user", "content": "winning current user"},
        ),
        tools=(),
        capture_policy=_policy(),
        saved_by_owner={"saved": saved},
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="openai",
        model="gpt-test",
        execution_key="openai",
        base_url="",
    )
    prepared = gateway.prepare_chat_request(
        resolution,
        semantic,
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    bundle = gateway._verify_trace_shadow(
        resolution,
        prepared,
        reconstruct_provider_gateway_kwargs(resolution, prepared),
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        trace_call_boundary=boundary,
    )
    boundary.mark_dispatch_started(bundle, prepared.provenance)
    boundary.mark_response_started()
    envelope = boundary.settle_response()
    manifest = registry.seal_attempt(attempt, expected_call_count=1)
    context = _context(promotion_id)
    repository = _RecordingRepository([_result(context, call_count=1)])
    ConsoleTraceService(repository=repository).import_provisional_voice_trace(
        object(),
        registry,
        manifest,
        (envelope,),
        context,
    )
    surface = repository.requests[0].calls[0].request_surface
    assert [
        (item.component_kind, item.reference_kind, item.revision_id) for item in surface
    ] == [
        ("message", "revision", saved.revision_id),
        ("message", "revision", context.user_revision_id),
    ]
    assert registry.retained_bytes == 0


def test_store_teardown_refuses_live_voice_lease_before_canvas_cleanup():
    store, _session_id, context, _commit, _registry, _snapshot, _events = (
        _winning_case()
    )
    closed = []
    store.canvas_promotion_participant = SimpleNamespace(
        close_runtime=lambda: closed.append("canvas")
    )
    claim = store.try_claim_voice_promotion(context)
    assert claim.lease is not None
    with pytest.raises(RuntimeError, match="Voice promotion state"):
        store.end_app_runtime()
    assert closed == []
    assert store.abort_voice_promotion(claim.lease)
    store.end_app_runtime()
    assert closed == ["canvas"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["refused", "stale_slot", "replayed_winner"])
async def test_factory_does_not_consume_foreign_or_unclaimed_next_privacy(
    monkeypatch, case
):
    store, session_id, context, _commit, registry, snapshot, events = _winning_case()
    session = next(item for item in store.sessions() if item.id == session_id)
    store.set_session_next_trace_privacy(
        session_id,
        capture_enabled=False,
        pii_redaction_enabled=True,
        expected_policy_revision=store._capture_policy_revision,
    )
    revision = session.next_privacy_revision
    seed = VoicePromotionSeed(
        context.promotion_id,
        context.attempt_id,
        context.terminal_boundary_id,
        context.origin,
        context.expected_native_leaf_id,
        context.expected_persisted_leaf_id,
        False,
        capture_policy=FrozenTracePolicy(
            context.promotion_id, "credentials-v1", False, None
        ),
        next_trace_privacy_revision=revision,
    )
    prepared = SimpleNamespace(
        promotion_seed=seed,
        request=SimpleNamespace(
            resolution=SimpleNamespace(provider="openai", model="fixture")
        ),
    )
    snapshot = replace(snapshot, trace_manifest=None, trace_envelopes=())
    engine = _factory(
        monkeypatch,
        store,
        VoicePromotionOwner(lambda: store, sync_runner=_inline),
        SimpleNamespace(provisional_trace_registry=registry),
    )
    observed = []

    async def promote(current=snapshot):
        return await engine.bindings["promote"](
            prepared=prepared,
            snapshot=current,
            transcript=context.user_text,
            assistant_text=context.assistant_text,
            on_claim=observed.append,
        )

    if case == "replayed_winner":
        assert (await promote())._commit is not None
        assert len(observed) == 1
    if case != "refused":
        store.set_session_next_trace_privacy(
            session_id,
            capture_enabled=True,
            pii_redaction_enabled=False,
            expected_policy_revision=store._capture_policy_revision,
        )
    current_revision = session.next_privacy_revision
    expected = (session.next_capture_enabled, session.next_pii_redaction_enabled)
    outcome = await promote(
        replace(snapshot, response_text="not the winning answer")
        if case == "refused"
        else snapshot
    )
    assert session.next_privacy_revision == current_revision
    assert (
        session.next_capture_enabled,
        session.next_pii_redaction_enabled,
    ) == expected
    if case == "refused":
        assert outcome._commit is None and observed == [] and events == []
    else:
        assert events == ["pair"] and len(observed) == 1
