"""Host captured managed continuation pins constrain fresh admission."""

import pytest

from tldw_chatbook.Agents.activation import worker_guard
from tldw_chatbook.DB.base_db import run_owned_db_call

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


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
@pytest.mark.parametrize("bound", [False, True])
async def test_real_bound_pin_seals_body_and_constrains_fresh_admission(
    revocation_case,
    bound,
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
    if bound:
        checkpoint = replace(checkpoint, schema_version=3, selected_provider="deepseek")
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
    assert sealed.schema_version == (3 if bound else 2)
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
    if bound:
        with pytest.raises(PermissionError):
            await service.resume_maximum(
                maximum,
                replace(sealed, selected_provider="custom-ep:foreign"),
                "conversation",
                "message",
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
@pytest.mark.parametrize("ack_mode", [None, "valid", "missing", "stale"])
async def test_real_sqlite_finalizer_reseals_each_event_and_preserves_foreign_history(
    revocation_case, tmp_path, ack_mode
):
    import asyncio

    from Tests.Chat.test_console_provider_continuation import (
        _MOONSHOT_RUN_TARGET,
        _active_checkpoint,
    )
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
        from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter

        adapter = _StreamingModelAdapter(
            store=store,
            provider_gateway=None,
            resolution=None,
            assistant_message_id=owner.id,
            should_cancel=lambda: False,
            loop=asyncio.get_running_loop(),
            native_tools=True,
            generation_token=store.begin_generation_attempt(owner.id),
        )
        batch = ToolBatchReady(context, _active_checkpoint(), None)
        await run_owned_db_call(
            database,
            store.persist_provider_continuation_event,
            batch,
            continuation_target=_MOONSHOT_RUN_TARGET,
            checkpoint_finalizer=finalize,
        )
        initial = parse_provider_continuation_json(
            database.get_message_by_id(owner.id)["provider_continuation_json"]
        )
        assert initial.schema_version == 2
        custody = None
        if ack_mode is not None:
            if ack_mode != "missing":
                adapter._live_native_persisted_event = batch
            if ack_mode == "missing":
                with pytest.raises(ValueError, match="live_native_acknowledgment"):
                    adapter._acknowledge_live_native_continuation(batch, batch, None)
                assert adapter._live_native_custody is None
            else:
                adapter._acknowledge_live_native_continuation(batch, batch, None)
                custody = adapter._live_native_custody
                assert custody.checkpoint is batch.checkpoint
                assert custody.durable_checkpoint == initial
                assert custody.checkpoint.managed_resume is None
                assert custody.durable_checkpoint.managed_resume is not None
                assert "PRIVATE" not in repr(custody)
                with database.transaction() as cursor:
                    custody.validate(cursor, custody)
        assert (
            await service.resume_maximum(
                service.capture_maximum("a"),
                initial,
                session.persisted_conversation_id,
                owner.id,
            )
        )["available_skills"]
        await run_owned_db_call(
            database,
            store.persist_provider_continuation_event,
            ToolCallExecuting(context, "PRIVATE-CALL-ID", 1),
            continuation_target=_MOONSHOT_RUN_TARGET,
            checkpoint_finalizer=finalize,
        )
        executing = parse_provider_continuation_json(
            database.get_message_by_id(owner.id)["provider_continuation_json"]
        )
        assert executing.managed_resume != initial.managed_resume
        assert executing.rounds[0].calls[0].state == "executing"
        if custody is not None:
            with (
                database.transaction() as cursor,
                pytest.raises((ValueError, RuntimeError)),
            ):
                custody.validate(cursor, custody)
            if ack_mode == "valid":
                event = ToolCallExecuting(context, "PRIVATE-CALL-ID", 1)
                adapter._live_native_persisted_event = event
                adapter._acknowledge_live_native_continuation(event, event, None)
                refreshed = adapter._live_native_custody
                assert refreshed.durable_checkpoint == executing
                assert refreshed.checkpoint.managed_resume is None
                with database.transaction() as cursor:
                    refreshed.validate(cursor, refreshed)
                    with pytest.raises(ValueError, match="live_native_custody_expired"):
                        custody.validate(cursor, custody)
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
@pytest.mark.parametrize("broader_parent", [False, True])
async def test_actual_console_fleet_producer_resumes_exact_pin_and_refuses_changed_scope(
    native_console, tmp_path, broader_parent
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
    reviews = []

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
        [
            ["child done"],
            [_fence("calculator", {"expression": "6*7"})],
            ["child resumed"],
        ],
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
            worker_guard(bridge)(_run),
            bridge,
            rig.store,
            rig.session,
            assistant.id,
            conversation_id="fleet-managed",
            skills_context=maximum,
            workspace_id="workspace-a",
            plugin_cancel_root=threading.Event().set,
            review_tool_calls=lambda calls, run_id: (
                reviews.append((run_id, tuple(call.name for call in calls))) or {}
            ),
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
        if broader_parent:
            await rig.install()
        await turn("second")
        assert gateway.child_calls == 3
        resumed = [
            row
            for row in database.list_runs("fleet-managed")
            if row.get("resumed_from_run_id") == first.run_id
        ]
        assert len(resumed) == 1
        child = database.get_run(resumed[0]["id"])
        assert child["status"] == "done"
        assert (resumed[0]["id"], ("calculator",)) in reviews, reviews
        assert any(
            step.get("tool_name") == "calculator"
            and step.get("tool_outcome") == "success"
            for step in child["steps"]
        ), child["steps"]

        await rig.service.disable(
            RevocationTarget(installed.installation_id, "workspace-a", False)
        )
        await turn("third")
        assert gateway.child_calls == 3
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
@pytest.mark.parametrize("bound", [False, True])
async def test_real_v2_whole_record_transport_preserves_host_envelope(
    revocation_case, tmp_path, chachanotes_template_db, monkeypatch, transport, bound
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
    from dataclasses import replace

    source = parse_provider_continuation_json(raw)
    if bound:
        source = replace(source, schema_version=3, selected_provider=source.provider)
    checkpoint = await asyncio.to_thread(
        service.seal_resume_checkpoint,
        pin,
        source,
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
        ConsoleProviderGateway,
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

        # The production translator: it projects the checkpoint's rounds, so
        # the resumed request carries the native batch the checkpoint owns.
        expand_provider_continuation = staticmethod(
            ConsoleProviderGateway.expand_provider_continuation
        )

        async def resolve_for_send(self, selection):
            return provider_resolution(
                ready=True,
                provider="Moonshot",
                selected_provider="Moonshot",
                execution_key="moonshot",
                continuation_protocol="chat_completions",
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
                self.resumed = [dict(message) for message in messages]
                current = store.get_message(owner.id).provider_continuation
                final = replace(
                    current,
                    schema_version=1,
                    managed_resume=None,
                    selected_provider=None,
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
        selected_provider="Moonshot",
        model="kimi-k2",
        protocol="chat_completions",
        api_base_url="https://api.moonshot.ai/v1",
    )
    try:
        result = await asyncio.to_thread(
            worker_guard(bridge)(_run),
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
                selected_provider="Moonshot",
                continuation_protocol="chat_completions",
                model="kimi-k2",
                base_url="https://api.moonshot.ai/v1",
            ),
            continuation_target=target,
            expand_provider_continuation=gateway.expand_provider_continuation,
        )
        assert result.status == "cancelled", result
        checkpoint = store.get_message(owner.id).provider_continuation
        assert (
            checkpoint.schema_version == 3
            and checkpoint.rounds[0].calls[0].state == "pending"
        )
        durable = store.persistence.db.get_message_by_id(owner.id)
        assert (
            parse_provider_continuation_json(durable["provider_continuation_json"])
            == checkpoint
        )
        if legacy:
            checkpoint = replace(
                checkpoint,
                schema_version=1,
                managed_resume=None,
                selected_provider=None,
            )
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
        call_id = _active_checkpoint().rounds[0].calls[0].call_id
        assert [
            message["tool_calls"][0]["id"]
            for message in gateway.resumed
            if message.get("tool_calls")
        ] == [call_id]
        final = store.get_message(owner.id).provider_continuation
        assert final.schema_version == 3 and final.state == "complete"
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


@pytest.mark.asyncio
async def test_blocked_v2_admission_does_not_delay_immediate_revocation(
    revocation_case, monkeypatch
):
    import asyncio
    import json
    import threading

    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    case = revocation_case
    service = case.service
    pin = await asyncio.to_thread(
        service.capture_resume_pin,
        case.entries["a"],
        "root-a",
        "conversation",
        "message",
    )
    sealed = await asyncio.to_thread(
        service.seal_resume_checkpoint,
        pin,
        parse_provider_continuation_json(json.dumps(_checkpoint())),
        "conversation",
        "message",
    )
    maximum = await service.resume_maximum(
        service.capture_maximum("a"), sealed, "conversation", "message"
    )
    assert (await service.admit(maximum, "v2-control"))["available_skills"]
    entered, release, sealed_live = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original = service._coordinator.authority.verify_archive_pin

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(service._coordinator.authority, "verify_archive_pin", blocked)
    admitted = asyncio.create_task(service.admit(maximum, "v2-blocked"))
    request = None

    async def revoke():
        nonlocal request
        request = service.begin_disable(RevocationTarget(case.installation, "a", False))
        sealed_live.set()
        return await service.finish_revocation(request)

    def run_revoke():
        return asyncio.run(revoke())

    revocation = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        revocation = asyncio.create_task(asyncio.to_thread(run_revoke))
        assert await asyncio.to_thread(sealed_live.wait, 1), (
            "V2 storage verification blocked synchronous revocation"
        )
        assert request is not None
        await asyncio.wait_for(case.cleanup_started.wait(), 1)
        with pytest.raises(PermissionError):
            service.check_entries_live(case.entries["a"])
    finally:
        release.set()
        await asyncio.gather(admitted, return_exceptions=True)
        if revocation is not None:
            await asyncio.gather(revocation, return_exceptions=True)
    assert admitted.done() and isinstance(admitted.exception(), PermissionError)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["generation", "coverage", "fence", "path"])
async def test_authenticated_data_pin_refuses_changed_root_authority(
    revocation_case, tmp_path, change
):
    import asyncio
    import hashlib
    import json
    from uuid import uuid4

    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    case = revocation_case
    service = case.service
    creation = await service.review_data_creation(case.installation, workspace_id="a")
    await service.create_data(creation, creation.operation_id)
    root = (await service._call(service._coordinator.published_snapshot))["data_roots"][
        0
    ]

    async def publish(rows):
        # Existing F4 authenticated authority fixture, not a future F8 producer.
        # Keep activation generations fixed so only data compatibility can refuse.
        async def operation():
            c = service._coordinator
            old = c.authority.load_marker()
            result = {
                "kind": "configure",
                "installation_id": case.installation,
                "revision_digest": c.published_snapshot()["installations"][0][
                    "revision_digest"
                ],
                "result": "committed",
            }
            op = c.authority.issue_operation_id(
                old.generation + 1,
                hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                result,
                uuid4().hex + uuid4().hex,
            )
            result["operation_id"] = op
            with c.registry.transaction() as cursor:
                cursor.execute(
                    "DELETE FROM data_roots WHERE installation_id=?",
                    (case.installation,),
                )
                for row in rows:
                    cursor.execute(
                        "INSERT INTO data_roots(root_id, installation_id, workspace_id, path, generation, deletion_fenced) VALUES (?, ?, ?, ?, ?, ?)",
                        tuple(
                            row[key]
                            for key in (
                                "root_id",
                                "installation_id",
                                "workspace_id",
                                "path",
                                "generation",
                                "deletion_fenced",
                            )
                        ),
                    )
                for row in rows:
                    cursor.execute(
                        "UPDATE data_roots SET custody_json=? WHERE root_id=?",
                        (json.dumps(row["custody"]), row["root_id"]),
                    )
                snapshot = c.registry.authority_projection(operation_result=result)
                new = PluginMarker(
                    generation=old.generation + 1,
                    operation_id=op,
                    recovery_snapshot_digest=snapshot_digest(snapshot),
                )
                c.authority.prepare(snapshot, old, new)
                c.registry.write_operation(cursor, result, phase="committed")
            c.authority.certify_commit(old, new)
            c.authority.advance_marker(old, new)
            assert all(item.committed for item in await c.recover())
            service._refresh()

        await service._call(operation)

    import threading

    entries = (await service.admit(service.capture_maximum("a"), "root-data-pending"))[
        "available_skills"
    ]
    await asyncio.to_thread(
        service.bind_run, entries, "root-data", threading.Event().set
    )
    pin = await asyncio.to_thread(
        service.capture_resume_pin, entries, "root-data", "conversation", "message"
    )
    await asyncio.to_thread(service.complete_run, "root-data")
    assert json.loads(pin)["installations"][0]["data_coverage"] == "known"
    sealed = await asyncio.to_thread(
        service.seal_resume_checkpoint,
        pin,
        parse_provider_continuation_json(json.dumps(_checkpoint())),
        "conversation",
        "message",
    )
    positive = await service.resume_maximum(
        service.capture_maximum("a"), sealed, "conversation", "message"
    )
    assert (await service.admit(positive, "data-positive"))["available_skills"]
    changed = dict(root)
    if change == "generation":
        changed["generation"] += 1
    elif change == "fence":
        changed["deletion_fenced"] = True
    elif change == "path":
        changed["path"] = str(tmp_path / "another-qualified-data-root")
    await publish([] if change == "coverage" else [changed])
    with pytest.raises(PermissionError, match="authority or data changed"):
        await service.resume_maximum(
            service.capture_maximum("a"), sealed, "conversation", "message"
        )
    with pytest.raises(PermissionError, match="authority or data changed"):
        await service.admit(positive, "stale-data-positive")
