"""Host captured managed continuation pins constrain fresh admission."""

import pytest


def test_closed_managed_envelope_roundtrips_without_granting_authority():
    from tldw_chatbook.Plugins.continuation import validate_envelope

    envelope = {
        "schema_version": 1,
        "namespace_id": "a" * 64,
        "session_id": "session",
        "run_id": "run",
        "conversation_id": "conversation",
        "message_id": "message",
        "installations": [],
        "checkpoint_digest": "b" * 64,
        "mac": "c" * 64,
    }
    assert validate_envelope(envelope) == envelope
    with pytest.raises(ValueError):
        validate_envelope(dict(envelope, unknown="ignored"))
    with pytest.raises(ValueError):
        validate_envelope(dict(envelope, mac="bad"))


from Tests.Plugins.test_revocation_persistence import (
    revocation_case as _revocation_case,
)

revocation_case = _revocation_case


@pytest.mark.asyncio
async def test_real_bound_pin_seals_body_and_constrains_fresh_admission(
    revocation_case,
):
    import asyncio
    import json
    from dataclasses import replace

    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        dump_provider_continuation_json,
        parse_provider_continuation_json,
    )

    case = revocation_case
    service = case.service
    checkpoint = parse_provider_continuation_json(json.dumps(_checkpoint()))
    await service.check_entries(case.entries["a"])
    pin = await asyncio.to_thread(
        service.capture_resume_pin,
        case.entries["a"],
        "root-a",
        "conversation",
        "message",
    )
    sealed = await asyncio.to_thread(
        service.seal_resume_checkpoint, pin, checkpoint, "conversation", "message"
    )
    assert sealed.schema_version == 2
    assert (
        parse_provider_continuation_json(dump_provider_continuation_json(sealed))
        == sealed
    )
    maximum = service.capture_maximum("a")
    constrained = await service.resume_maximum(
        maximum, sealed, "conversation", "message"
    )
    assert (await service.admit(constrained, "resume-positive"))["available_skills"]
    with pytest.raises(PermissionError):
        await service.resume_maximum(maximum, sealed, "forked-conversation", "message")
    with pytest.raises(PermissionError):
        await service.resume_maximum(
            maximum, replace(sealed, checkpoint_revision=2), "conversation", "message"
        )
    zero = await service.resume_maximum(maximum, checkpoint, "conversation", "message")
    assert not (await service.admit(zero, "legacy-zero"))["available_skills"]
    # Substituting a larger current maximum does not bypass the admission gate.
    zero["available_skills"] = maximum["available_skills"]
    assert not (await service.admit(zero, "legacy-zero-again"))["available_skills"]
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    await case.disable(RevocationTarget(case.installation, "a", False), "pin-revoke")
    with pytest.raises(PermissionError):
        await service.admit(constrained, "resume-stale")


@pytest.mark.asyncio
async def test_real_sqlite_finalizer_reseals_each_event_and_preserves_foreign_history(
    revocation_case, tmp_path
):
    import asyncio

    from Tests.Chat.test_console_provider_continuation import _active_checkpoint
    from tldw_chatbook.Agents.agent_models import (
        ContinuationEventContext,
        ToolBatchReady,
        ToolCallExecuting,
    )
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    service = revocation_case.service
    database = CharactersRAGDB(
        tmp_path / "continuation.sqlite", "managed-continuation-test"
    )
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.create_session(title="Managed continuation")
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="review", persist=True
        )
        owner = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        pin = await asyncio.to_thread(
            service.capture_resume_pin,
            revocation_case.entries["a"],
            "root-a",
            session.persisted_conversation_id,
            owner.id,
        )

        def finalize(checkpoint, conversation_id, message_id, run_id):
            assert run_id == "root-a"
            return service.seal_resume_checkpoint(
                pin, checkpoint, conversation_id, message_id
            )

        context = ContinuationEventContext(owner.id, "root-a", "primary", "persistent")
        await asyncio.to_thread(
            store.persist_provider_continuation_event,
            ToolBatchReady(context, _active_checkpoint(), None),
            checkpoint_finalizer=finalize,
        )
        initial = parse_provider_continuation_json(
            database.get_message_by_id(owner.id)["provider_continuation_json"]
        )
        assert initial.schema_version == 2
        assert (
            await service.resume_maximum(
                service.capture_maximum("a"),
                initial,
                session.persisted_conversation_id,
                owner.id,
            )
        )["available_skills"]
        await asyncio.to_thread(
            store.persist_provider_continuation_event,
            ToolCallExecuting(context, "PRIVATE-CALL-ID", 1),
            checkpoint_finalizer=finalize,
        )
        executing = parse_provider_continuation_json(
            database.get_message_by_id(owner.id)["provider_continuation_json"]
        )
        assert executing.managed_resume != initial.managed_resume
        assert executing.rounds[0].calls[0].state == "executing"
        # Decoder preserves recognized foreign bytes. Only the resume owner rejects them.
        assert (
            parse_provider_continuation_json(
                database.get_message_by_id(owner.id)["provider_continuation_json"]
            )
            == executing
        )
        with pytest.raises(PermissionError):
            await service.resume_maximum(
                service.capture_maximum("a"), initial, "import-remapped", owner.id
            )
    finally:
        database.close_connection()


def test_fleet_retention_keeps_exact_host_pin_and_refuses_missing_required_pin():
    from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator

    fleet = FleetCoordinator(max_live=2, clock=lambda: 0.0)
    missing = fleet.reserve(task="managed missing", agent=None)
    fleet.set_managed_custody(missing.handle_id, required=True)
    fleet.attach_run(missing.handle_id, "missing-run")
    fleet.finish(
        missing.handle_id,
        status="done",
        transcript=[{"role": "assistant", "content": "done"}],
    )
    assert fleet.get_retained(missing.handle_id) is None
    pinned = fleet.reserve(task="managed pinned", agent=None)
    fleet.attach_run(pinned.handle_id, "pinned-run")
    fleet.set_managed_custody(pinned.handle_id, required=True, pin="host-held-pin")
    fleet.finish(
        pinned.handle_id,
        status="done",
        transcript=[{"role": "assistant", "content": "done"}],
    )
    retained = fleet.get_retained(pinned.handle_id)
    assert retained.managed_required and retained.managed_pin == "host-held-pin"


@pytest.mark.asyncio
async def test_actual_console_fleet_producer_resumes_exact_pin_and_refuses_changed_scope(
    native_console, tmp_path
):
    import asyncio
    import threading

    from Tests.Chat.test_console_agent_bridge import (
        _fence,
        _FleetChunkGateway,
        _join_fleet_threads,
        _run,
    )
    from tldw_chatbook.Agents.agent_models import (
        SEND_TO_AGENT_TOOL_NAME,
        SPAWN_TOOL_NAME,
        WAIT_AGENTS_TOOL_NAME,
    )
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    rig = native_console
    installed = await rig.install()
    holder = {}

    def resume():
        return [
            _fence(
                SEND_TO_AGENT_TOOL_NAME,
                {"id": holder["handle"], "message": "check again"},
            )
        ]

    gateway = _FleetChunkGateway(
        [
            [_fence(SPAWN_TOOL_NAME, {"task": "managed child"})],
            [_fence(WAIT_AGENTS_TOOL_NAME, {})],
            ["first done"],
            resume,
            [_fence(WAIT_AGENTS_TOOL_NAME, {})],
            ["second done"],
            resume,
            ["refusal handled"],
        ],
        [["child done"], ["child resumed"]],
    )
    database = AgentRunsDB(tmp_path / "fleet.sqlite", client_id="managed-fleet")
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=rig.store,
        provider_gateway=gateway,
        skills_service=rig.skills,
    )

    async def turn(label):
        maximum = rig.service.capture_maximum("workspace-a")
        maximum["plugin_run_id"] = "pending:" + label
        assistant = rig.store.append_message(
            rig.session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        outcome = await asyncio.to_thread(
            _run,
            bridge,
            rig.store,
            rig.session,
            assistant.id,
            conversation_id="fleet-managed",
            skills_context=maximum,
            workspace_id="workspace-a",
            plugin_cancel_root=threading.Event().set,
        )
        await asyncio.to_thread(_join_fleet_threads)
        assert outcome.status == "done", outcome

    try:
        await turn("first")
        fleet = bridge._fleet_coordinators["fleet-managed"]
        first = next(
            handle for handle in fleet.snapshot() if handle.task == "managed child"
        )
        retained = fleet.get_retained(first.handle_id)
        assert retained and retained.managed_required and retained.managed_pin
        holder["handle"] = first.handle_id
        await turn("second")
        assert gateway.child_calls == 2
        resumed = [
            row
            for row in database.list_runs("fleet-managed")
            if row.get("resumed_from_run_id") == first.run_id
        ]
        assert len(resumed) == 1
        await rig.service.disable(
            RevocationTarget(installed.installation_id, "workspace-a", False)
        )
        await turn("third")
        assert gateway.child_calls == 2
        assert (
            len(
                [
                    row
                    for row in database.list_runs("fleet-managed")
                    if row.get("resumed_from_run_id") == first.run_id
                ]
            )
            == 1
        )
    finally:
        await asyncio.to_thread(_join_fleet_threads)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["archive", "sync"])
async def test_real_v2_whole_record_transport_preserves_host_envelope(
    revocation_case, tmp_path, chachanotes_template_db, monkeypatch, transport
):
    import asyncio

    from tldw_chatbook.Chat.provider_continuation import (
        dump_provider_continuation_json,
        parse_provider_continuation_json,
    )

    service = revocation_case.service
    if transport == "archive":
        from Tests.Chatbooks import test_provider_continuation_roundtrip as suite

        raw = suite._checkpoint_json()
    else:
        from Tests.Sync_Interop import (
            test_provider_continuation_reconciliation as suite,
        )

        raw = suite._provider_continuation_json()
    pin = await asyncio.to_thread(
        service.capture_resume_pin,
        revocation_case.entries["a"],
        "root-a",
        "transport-source-conversation",
        "transport-source-message",
    )
    checkpoint = await asyncio.to_thread(
        service.seal_resume_checkpoint,
        pin,
        parse_provider_continuation_json(raw),
        "transport-source-conversation",
        "transport-source-message",
    )
    encoded = dump_provider_continuation_json(checkpoint)
    assert encoded and "managed_resume" in encoded
    if transport == "archive":
        monkeypatch.setattr(suite, "_checkpoint_json", lambda **kwargs: encoded)
        suite.test_v2_import_remaps_complete_graph_before_attaching_private_owner(
            tmp_path, chachanotes_template_db
        )
        with pytest.raises(PermissionError):
            await service.resume_maximum(
                service.capture_maximum("a"),
                checkpoint,
                "import-remapped",
                "import-remapped-message",
            )
    else:
        monkeypatch.setattr(suite, "_provider_continuation_json", lambda: encoded)
        (tmp_path / "projection").mkdir()
        (tmp_path / "edit").mkdir()
        suite.test_barrier_requires_only_atomic_local_projection_not_remote_ack(
            tmp_path / "projection"
        )
        # Existing whole-record test asserts exact emitted payload and applies it.
        suite.test_visible_edit_keeps_checkpoint_on_its_exact_new_message_version(
            tmp_path / "edit"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_actual_console_producer_persists_v2_and_controller_resumes_it(
    native_console, tmp_path, monkeypatch, legacy
):
    import asyncio
    import threading
    from dataclasses import replace

    from Tests.Chat.test_console_agent_bridge import _run, _test_resolution
    from Tests.Chat.test_console_provider_continuation import _active_checkpoint
    from Tests.console_provider_doubles import (
        persisted_console_store,
        provider_resolution,
    )
    from tldw_chatbook.Agents.agent_models import ToolBatchReady
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_provider_gateway import (
        ProviderToolCalls,
        ProviderTurnMetadata,
    )
    from tldw_chatbook.Chat.provider_continuation import (
        ContinuationRestoreTarget,
        dump_provider_continuation_json,
        parse_provider_continuation_json,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    rig = native_console
    await rig.install()
    store = persisted_console_store(
        db_path=tmp_path / "primary.sqlite", workspace_registry=rig.registry
    )
    session = store.create_session(workspace_id="workspace-a")
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="calculate", persist=True
    )
    owner = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    stopped = threading.Event()
    persist = store.persist_provider_continuation_event

    def stop_after_first_batch(event, **kwargs):
        persist(event, **kwargs)
        if isinstance(event, ToolBatchReady):
            stopped.set()

    monkeypatch.setattr(
        store, "persist_provider_continuation_event", stop_after_first_batch
    )

    class Gateway:
        calls = 0

        def expand_provider_continuation(self, checkpoint):
            return []

        async def resolve_for_send(self, selection):
            return provider_resolution(
                ready=True,
                provider="Moonshot",
                model="kimi-k2",
                base_url="https://api.moonshot.ai/v1",
                api_mode="chat_completions",
            )

        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                call = _active_checkpoint().rounds[0].calls[0]
                yield ProviderToolCalls(
                    (
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        },
                    ),
                    ProviderTurnMetadata("tool_calls", _active_checkpoint()),
                )
            else:
                if legacy:
                    assert not rig.service._live_runs
                current = store.get_message(owner.id).provider_continuation
                final = replace(
                    current,
                    schema_version=1,
                    managed_resume=None,
                    state="complete",
                    checkpoint_revision=current.checkpoint_revision + 1,
                )
                yield "done"
                yield ProviderToolCalls((), ProviderTurnMetadata("stop", final))

    gateway = Gateway()
    runs = AgentRunsDB(tmp_path / "primary-runs.sqlite", client_id="managed-primary")
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs,
        store=store,
        provider_gateway=gateway,
        skills_service=rig.skills,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        skills_service=rig.skills,
    )
    controller.app = rig.controller.app
    maximum = rig.service.capture_maximum("workspace-a")
    maximum["plugin_run_id"] = "pending:initial-checkpoint"
    target = ContinuationRestoreTarget(
        provider="moonshot",
        model="kimi-k2",
        protocol="chat_completions",
        api_base_url="https://api.moonshot.ai/v1",
    )
    try:
        result = await asyncio.to_thread(
            _run,
            bridge,
            store,
            session,
            owner.id,
            conversation_id=session.persisted_conversation_id,
            skills_context=maximum,
            workspace_id="workspace-a",
            plugin_cancel_root=stopped.set,
            should_cancel=stopped.is_set,
            native_tools_enabled=True,
            resolution=_test_resolution(
                provider="Moonshot",
                execution_key="moonshot",
                model="kimi-k2",
                base_url="https://api.moonshot.ai/v1",
            ),
            continuation_target=target,
            expand_provider_continuation=gateway.expand_provider_continuation,
        )
        assert result.status == "cancelled", result
        checkpoint = store.get_message(owner.id).provider_continuation
        assert (
            checkpoint.schema_version == 2
            and checkpoint.rounds[0].calls[0].state == "pending"
        )
        durable = store.persistence.db.get_message_by_id(owner.id)
        assert (
            parse_provider_continuation_json(durable["provider_continuation_json"])
            == checkpoint
        )
        if legacy:
            checkpoint = replace(checkpoint, schema_version=1, managed_resume=None)
            current_owner = store.get_message(owner.id)
            version = current_owner.provider_continuation_message_version
            store.persistence.db.update_provider_continuation(
                message_id=owner.id,
                expected_message_version=version,
                provider_continuation_json=dump_provider_continuation_json(checkpoint),
            )
            live_owner = store._message_or_raise(owner.id)
            live_owner.provider_continuation = checkpoint
            live_owner.provider_continuation_message_version = version + 1
        monkeypatch.setattr(store, "persist_provider_continuation_event", persist)
        stopped.clear()
        resumed = await controller.recover_provider_continuation(
            "resume",
            owner.id,
            store.get_message(owner.id).provider_continuation_message_version,
        )
        assert resumed, controller.run_state_for(session.id).visible_copy
        assert gateway.calls == 2
        final = store.get_message(owner.id).provider_continuation
        assert (
            final.schema_version == (1 if legacy else 2) and final.state == "complete"
        )
        assert final.rounds[0].calls[0].state == "completed"
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_archived_subset_stays_narrow_under_broader_current_selection(
    tmp_path, native_package
):
    import asyncio
    import json
    import shutil
    from types import SimpleNamespace

    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService

    package = native_package()
    shutil.copytree(package / "skills/review", package / "skills/other")
    other = package / "skills/other/SKILL.md"
    other.write_text(other.read_text().replace("name: review", "name: other"))
    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: SimpleNamespace(archived=False),
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    try:
        await service.bootstrap("test passphrase")
        review = await service.review_install(
            package, selection=("skill:review", "skill:other"), workspace_id="a"
        )
        await service.commit(review, review.operation_id)
        for item in (await service.review_trust(review.installation_id),):
            await service.commit(item, item.operation_id)
        item = await service.review_activation(
            review.installation_id, workspace_id="a", intent="enabled"
        )
        await service.commit(item, item.operation_id)
        entries = (await service.admit(service.capture_maximum("a"), "parent"))[
            "available_skills"
        ]
        assert len(entries) == 2
        await asyncio.to_thread(
            service.bind_run,
            entries,
            "child",
            lambda: None,
            component_ceiling={review.installation_id: ("skill:review",)},
        )
        pin = await asyncio.to_thread(
            service.capture_resume_pin, entries, "child", "conversation", "message"
        )
        checkpoint = parse_provider_continuation_json(json.dumps(_checkpoint()))
        sealed = await asyncio.to_thread(
            service.seal_resume_checkpoint, pin, checkpoint, "conversation", "message"
        )
        maximum = await service.resume_maximum(
            service.capture_maximum("a"), sealed, "conversation", "message"
        )
        resumed = await service.admit(maximum, "resumed")
        assert [row["plugin_component_id"] for row in resumed["available_skills"]] == [
            "skill:review"
        ]
        await service.check_entries(resumed["available_skills"])
    finally:
        await asyncio.to_thread(service.complete_run, "child")
        await service.aclose()
