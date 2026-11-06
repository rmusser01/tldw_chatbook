"""Mixed native/phone saved-owner readiness, cleanup and publication regressions."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Web_Server.test_mobile_shared_console import (
    shared_mobile_case as _shared_fixture,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    hydrate_console_generation_settings,
    hydrate_console_session,
    load_console_conversation_tree,
)

shared_mobile_case = _shared_fixture

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.loopback_network]


async def _native_hydrate(case, conversation_id, **kwargs):
    tree = await load_console_conversation_tree(case.app, conversation_id)
    settings = hydrate_console_generation_settings(
        case.app.app_config, tree["conversation"]
    )
    return await hydrate_console_session(
        app=case.app,
        store=case.app.console_runtime.chat_store,
        conversation_id=conversation_id,
        tree=tree,
        settings=settings.settings,
        generation_durable_snapshot=settings.durable_snapshot,
        generation_metadata_status=settings.metadata_status,
        activate=False,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_remote_saved_scope_is_independent_of_active_desktop(
    shared_mobile_case, tmp_path
):
    from tldw_chatbook.Chat.console_chat_models import (
        CONSOLE_GLOBAL_WORKSPACE_ID,
        ConsoleWorkspaceContext,
    )

    case = shared_mobile_case
    registry = case.app.workspace_registry_service
    registry.create_workspace(workspace_id="native-scope", name="Native scope")
    folder = tmp_path / "scope-project"
    folder.mkdir()
    registry.add_folder_binding("native-scope", folder, allow_write=True)
    await case.service.bootstrap_host()
    store = case.app.console_runtime.chat_store
    # Bootstrap does not create the graph; the explicit native opener does.
    runtime = case.app.console_runtime
    from tldw_chatbook.Chat.console_launch_wake import _ensure_launch_runtime

    controller = _ensure_launch_runtime(case.app)
    store = controller.store
    desktop = store.create_session(workspace_id="native-scope")
    store.set_workspace_context(
        ConsoleWorkspaceContext(active_workspace_id="native-scope")
    )
    remote = await runtime.open_remote_conversation(case.conversation_id)
    assert remote.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID
    assert remote.project_instruction_state.working_folder_binding_id is None
    assert store.active_session_id == desktop.id
    saved_named = case.app.chachanotes_db.add_conversation(
        {
            "title": "Named scope",
            "scope_type": "workspace",
            "workspace_id": "native-scope",
        }
    )
    store.set_workspace_context(
        ConsoleWorkspaceContext(active_workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID)
    )
    second = await runtime.open_remote_conversation(saved_named)
    assert second.workspace_id == "native-scope"
    assert store.active_session_id == desktop.id


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_remote_postread_joins_provisional_native_readiness(
    shared_mobile_case, monkeypatch, fail
):
    import tldw_chatbook.Chat.console_conversation_hydration as hydration

    case = shared_mobile_case
    await case.service.bootstrap_host()
    from tldw_chatbook.Chat.console_launch_wake import _ensure_launch_runtime

    store = _ensure_launch_runtime(case.app).store
    read_entered, release_read = asyncio.Event(), asyncio.Event()
    policy_entered, release_policy = asyncio.Event(), asyncio.Event()
    original = hydration.load_console_conversation_tree
    original_policy = store.hydrate_session_library_policy

    async def held_read(app, identity):
        read_entered.set()
        await release_read.wait()
        return await original(app, identity)

    monkeypatch.setattr(hydration, "load_console_conversation_tree", held_read)

    async def policy(identity):
        policy_entered.set()
        await release_policy.wait()
        if fail:
            raise RuntimeError("native policy refused")
        await original_policy(identity)

    monkeypatch.setattr(store, "hydrate_session_library_policy", policy)
    remote = asyncio.create_task(
        case.app.console_runtime.open_remote_conversation(case.conversation_id)
    )
    await asyncio.wait_for(read_entered.wait(), 2)
    # The terminal already has its own tree when it calls the shared hydrator.
    tree = await original(case.app, case.conversation_id)
    terminal = asyncio.create_task(
        hydrate_console_session(
            app=case.app,
            store=store,
            conversation_id=case.conversation_id,
            tree=tree,
            settings=None,
            activate=False,
        )
    )
    # In the repaired contract the first operation already reserves before read;
    # in the defect terminal publication enters its own policy wait first.
    await asyncio.sleep(0.2)
    release_read.set()
    await asyncio.wait_for(policy_entered.wait(), 2)
    await asyncio.sleep(0.2)
    try:
        assert not remote.done(), "Remote open returned a provisional native row"
        snapshot = await case.service.snapshot()
        assert not any(row.get("session_id") for row in snapshot["chats"])
        provisional = store.sessions()[0]
        from tldw_chatbook.Chat.console_remote_commands import REVISION_FIELDS
        from tldw_chatbook.Chat.console_remote_models import ConsoleTarget

        target = ConsoleTarget(
            **case.app.served_canvas_control.host_identity.model_dump(),
            session_id=provisional.id,
            incarnation_id=provisional.incarnation_id,
        )
        payload = {
            "text": "must not enter preparation",
            "expected_revisions": {
                name: getattr(provisional, name) for name in REVISION_FIELDS
            },
        }
        ticket = await case.service.issue_command_ticket("send", target, payload)
        receipt = await case.service.submit_command(ticket["ticket"], payload)
        assert receipt.state == "refused"
        assert not case.app.console_runtime.chat_controller._active_stream_tasks
    finally:
        release_policy.set()
        results = await asyncio.wait_for(
            asyncio.gather(remote, terminal, return_exceptions=True), 5
        )
    if fail:
        assert all(isinstance(result, RuntimeError) for result in results), results
        assert not store.sessions()
    else:
        assert results[0] is results[1]
        assert store.sessions() == [results[0]]


@pytest.mark.asyncio
async def test_first_waiter_cancellation_does_not_cancel_joined_phone(
    shared_mobile_case, monkeypatch
):
    case = shared_mobile_case
    await case.service.bootstrap_host()
    from tldw_chatbook.Chat.console_launch_wake import _ensure_launch_runtime

    store = _ensure_launch_runtime(case.app).store
    entered, release = asyncio.Event(), asyncio.Event()
    original = store.hydrate_session_library_policy

    async def policy(identity):
        entered.set()
        await release.wait()
        await original(identity)

    monkeypatch.setattr(store, "hydrate_session_library_policy", policy)
    terminal = asyncio.create_task(_native_hydrate(case, case.conversation_id))
    await asyncio.wait_for(entered.wait(), 2)
    phone = asyncio.create_task(
        case.app.console_runtime.open_remote_conversation(case.conversation_id)
    )
    await asyncio.sleep(0.2)
    terminal.cancel()
    with pytest.raises(asyncio.CancelledError):
        await terminal
    release.set()
    opened = await phone
    assert store.sessions() == [opened]
    assert opened.persisted_conversation_id == case.conversation_id


@pytest.mark.asyncio
async def test_goal_requested_saved_id_is_honored_after_phone_open(shared_mobile_case):
    case = shared_mobile_case
    await case.service.bootstrap_host()
    phone = await case.app.console_runtime.open_remote_conversation(
        case.conversation_id
    )
    assert phone.id == case.conversation_id
    goal_owner = await _native_hydrate(
        case, case.conversation_id, session_id=case.conversation_id
    )
    assert goal_owner is phone
    with pytest.raises(ValueError, match="Conflicting native"):
        await _native_hydrate(
            case, case.conversation_id, session_id="conflicting-requested-id"
        )
    assert len(case.app.console_runtime.chat_store.sessions()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["supersede", "cancel", "error"])
@pytest.mark.parametrize("shared", [False, True])
async def test_terminal_presentation_cleanup_preserves_adopted_native_turn(
    shared_mobile_case, monkeypatch, fault, shared
):
    from Tests.console_provider_doubles import AgentModeGateway
    from tldw_chatbook.Chat.console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
    from tldw_chatbook.Chat.console_launch_wake import _ensure_launch_runtime
    from tldw_chatbook.Chat.console_remote_models import ConsoleTarget
    from tldw_chatbook.Chat.console_session_settings import (
        default_console_session_settings,
    )
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController

    case = shared_mobile_case
    from threading import Event

    provider_entered, release_provider = Event(), Event()

    class HeldGateway(AgentModeGateway):
        async def stream_chat(self, *args, **kwargs):
            provider_entered.set()
            assert await asyncio.to_thread(release_provider.wait, 10), (
                "Provider release timed out"
            )
            yield "shared accepted turn survives"

    case.app.console_runtime.set_provider_gateway(HeldGateway())
    async with case.app.run_test(size=(100, 32)):
        host_identity = await case.service.bootstrap_host()
        store = _ensure_launch_runtime(case.app).store
        # A raw Core row intentionally has no durable Library authority. Seed
        # this accepted-turn control through its native atomic policy owner.
        from uuid import uuid4

        from tldw_chatbook.Chat.console_library_policy import (
            ConsoleLibraryPolicyCandidate,
        )

        case.conversation_id = str(uuid4())
        defaults = store._library_policy_defaults
        store.persistence.persist_console_conversation_with_policy(
            conversation_id=case.conversation_id,
            policy_candidate=ConsoleLibraryPolicyCandidate(
                defaults.auto_retrieve, defaults.assistant_access
            ),
            conversation_kwargs={
                "conversation_title": "Native policy saved target",
                "scope_type": "global",
            },
        )
        prior = store.create_session(
            title="Unrelated desktop", workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID
        )
        entered, release = asyncio.Event(), asyncio.Event()
        current = [True]

        async def presentation(*args):
            entered.set()
            await release.wait()
            if fault == "error":
                raise RuntimeError("presentation failed")

        async def nothing(*args, **kwargs):
            pass

        monkeypatch.setattr(
            "tldw_chatbook.UI.Console_Modules.conversation_token_preparation.prepare_conversation_tokens",
            presentation,
        )
        terminal_host = SimpleNamespace(
            app_instance=case.app,
            _screen=SimpleNamespace(),
            _ensure_console_chat_store=lambda: store,
            _begin_manual_read_visit=None,
            _complete_manual_read_visit=None,
            _capture_console_draft_switch_snapshot=lambda: None,
            _console_session_settings_for_resume=lambda row: (
                hydrate_console_generation_settings(case.app.app_config, row)
            ),
            _inject_resume_agent_markers=lambda rows, target: rows,
            _resolve_console_effective_scope_state=nothing,
            _restore_console_session_after_failed_open=nothing,
        )
        terminal = asyncio.create_task(
            ConsoleWorkspaceController._resume_console_workspace_conversation(
                terminal_host, case.conversation_id, resume_if=lambda: current[0]
            )
        )
        await asyncio.wait_for(entered.wait(), 3)
        opened = None
        try:
            unpublished = next(
                s
                for s in store.sessions()
                if s.persisted_conversation_id == case.conversation_id
            )
            operation = store._saved_session_operations[case.conversation_id]
            assert operation.ready and not operation.published
            snapshot = await case.service.snapshot()
            assert not operation.published  # Metadata observation cannot adopt custody.
            assert not any(
                row.get("session_id") == unpublished.id for row in snapshot["chats"]
            )
            target = ConsoleTarget(
                **host_identity.model_dump(),
                session_id=unpublished.id,
                incarnation_id=unpublished.incarnation_id,
            )
            unsafe = {
                "text": "Must not acquire an unpublished target",
                "expected_revisions": {
                    key: getattr(unpublished, key)
                    for key in (
                        "settings_revision",
                        "generation_settings_revision",
                        "context_policy_revision",
                        "identity_revision",
                        "conversation_binding_revision",
                    )
                },
            }
            rejected_ticket = await case.service.issue_command_ticket(
                "send", target, unsafe
            )
            rejected = await case.service.submit_command(
                rejected_ticket["ticket"], unsafe
            )
            assert rejected.state == "refused"
            assert not operation.published and not provider_entered.is_set()
            if shared:
                opened = await case.app.console_runtime.open_remote_conversation(
                    case.conversation_id
                )
                store.replace_session_settings(
                    opened.id,
                    default_console_session_settings(
                        case.app.app_config, "openai", "gpt-4o-mini"
                    ),
                )
                row = next(
                    row
                    for row in (await case.service.snapshot())["chats"]
                    if row.get("session_id") == opened.id
                )
                target = ConsoleTarget(
                    **host_identity.model_dump(),
                    session_id=opened.id,
                    incarnation_id=opened.incarnation_id,
                )
                payload = {
                    "text": "Accepted while terminal presents",
                    "expected_revisions": row["expected_revisions"],
                }
                ticket = await case.service.issue_command_ticket(
                    "send", target, payload
                )
                submitted = await case.service.submit_command(ticket["ticket"], payload)
                assert submitted.state == "pending"
                assert await asyncio.to_thread(provider_entered.wait, 5), (
                    "Native provider did not enter"
                )
            if fault == "supersede":
                current[0] = False
            elif fault == "cancel":
                terminal.cancel()
            release.set()
            if fault == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await terminal
            else:
                assert await terminal is None
            if shared:
                assert next(s for s in store.sessions() if s.id == opened.id) is opened
                assert store.messages_for_session(opened.id)
                assert store.active_session_id == prior.id
            else:
                assert not any(
                    s.persisted_conversation_id == case.conversation_id
                    for s in store.sessions()
                )
        finally:
            release.set()
            release_provider.set()
            await asyncio.gather(terminal, return_exceptions=True)
        if shared:
            async with asyncio.timeout(5):
                while not any(
                    "shared accepted turn survives" in str(m.content)
                    for m in store.messages_for_session(opened.id)
                ):
                    await asyncio.sleep(0.01)
            assert store.active_session_id == prior.id


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["new_chat", "fork_chat"])
async def test_native_creator_commit_publication_is_one_phone_owner(
    shared_mobile_case, monkeypatch, tool
):
    from threading import Event

    from Tests.Chat.test_console_chat_create_integration import _approve_close_new_chat
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_launch_wake import _ensure_launch_runtime
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    case = shared_mobile_case
    committed, release = Event(), Event()
    created = []
    async with case.app.run_test(size=(100, 32)):
        await case.service.bootstrap_host()
        controller = _ensure_launch_runtime(case.app)
        store = controller.store
        source = store.create_session(
            title="Native creator source",
            settings=ConsoleSessionSettings(
                provider="llama_cpp", model="creator-model"
            ),
        )
        store.replace_session_settings(source.id, source.settings)
        await asyncio.to_thread(store.persist_session_if_needed, source.id)
        message = store.append_message(
            source.id, role=ConsoleMessageRole.USER, content="fork source"
        )
        await asyncio.to_thread(store.persist_message_if_needed, message.id)
        controller._default_session_settings = lambda: ConsoleSessionSettings(
            provider="llama_cpp",
            model="destination-model",
            system_prompt="native defaults",
        )
        if tool == "new_chat":
            payload = await asyncio.to_thread(
                _approve_close_new_chat,
                controller,
                source,
                "Native destination",
                "native opening",
            )
        else:
            payload = {
                "tool": tool,
                "session_id": source.id,
                "title": "Native destination",
                "opening_prompt": "native opening",
            }
        method = (
            "persist_console_conversation_with_policy"
            if tool == "new_chat"
            else "create_conversation"
        )
        original = getattr(store.persistence, method)

        def held_commit(*args, **kwargs):
            result = original(*args, **kwargs)
            created.append(kwargs["conversation_id"] if tool == "new_chat" else result)
            committed.set()
            assert release.wait(10), "Native publication gate timed out"
            return result

        monkeypatch.setattr(store.persistence, method, held_commit)
        native = asyncio.create_task(
            asyncio.to_thread(controller.execute_agent_chat_create, payload)
        )
        phone = None
        try:
            assert await asyncio.to_thread(committed.wait, 5)
            phone = asyncio.create_task(
                case.app.console_runtime.open_remote_conversation(created[0])
            )
            await asyncio.sleep(0.2)
            assert not phone.done(), (
                "Phone returned before creator configuration/copy publication"
            )
        finally:
            release.set()
            outcome = await asyncio.wait_for(native, 8)
            if phone is not None:
                phone_owner = await asyncio.wait_for(phone, 5)
        assert outcome["ok"], outcome
        owner = next(
            s for s in store.sessions() if s.persisted_conversation_id == created[0]
        )
        assert owner is phone_owner
        assert (
            len(
                [
                    s
                    for s in store.sessions()
                    if s.persisted_conversation_id == created[0]
                ]
            )
            == 1
        )
        assert store.active_session_id == source.id
        assert owner.draft == "native opening"
        if tool == "new_chat":
            assert owner.settings.model == "destination-model"
            assert owner.project_instruction_state.project_instructions_enabled
            assert (
                owner.project_instruction_state
                == __import__(
                    "tldw_chatbook.Chat.console_project_instructions",
                    fromlist=["ProjectInstructionControlState"],
                ).ProjectInstructionControlState.new_session()
            )
        else:
            assert owner.draft == "native opening"
            assert any(
                m.content == "fork source" for m in store.messages_for_session(owner.id)
            )
        # Close/reopen is an explicit new incarnation of the same durable owner.
        store.close_session(owner.id)
        reopened = await case.app.console_runtime.open_remote_conversation(created[0])
        assert reopened.persisted_conversation_id == owner.persisted_conversation_id
        assert reopened.incarnation_id != owner.incarnation_id
        assert (
            len(
                [
                    s
                    for s in store.sessions()
                    if s.persisted_conversation_id == created[0]
                ]
            )
            == 1
        )


@pytest.mark.asyncio
async def test_first_save_bridge_keeps_live_controls_and_joins_original_id(
    shared_mobile_case, monkeypatch
):
    from threading import Event

    from tldw_chatbook.Chat.console_launch_wake import _ensure_launch_runtime

    case = shared_mobile_case
    committed, release = Event(), Event()
    async with case.app.run_test(size=(100, 32)):
        await case.service.bootstrap_host()
        store = _ensure_launch_runtime(case.app).store
        live = store.create_session(title="Existing live first save")
        original = store.persistence.persist_console_conversation_with_policy
        identities = []

        def held_commit(**kwargs):
            result = original(**kwargs)
            identities.append(kwargs["conversation_id"])
            committed.set()
            assert release.wait(10)
            return result

        monkeypatch.setattr(
            store.persistence, "persist_console_conversation_with_policy", held_commit
        )
        saving = asyncio.create_task(
            asyncio.to_thread(store.persist_session_if_needed, live.id)
        )
        phone = None
        try:
            assert await asyncio.to_thread(committed.wait, 5)
            phone = asyncio.create_task(
                case.app.console_runtime.open_remote_conversation(identities[0])
            )
            await asyncio.sleep(0.2)
            assert not phone.done()
            assert store.saved_session_ready(
                live
            )  # Existing native target stays controllable.
            snapshot = await case.service.snapshot()
            assert any(row["session_id"] == live.id for row in snapshot["chats"])
        finally:
            release.set()
            durable = await asyncio.wait_for(saving, 5)
            if phone is not None:
                opened = await asyncio.wait_for(phone, 5)
        assert opened is live
        assert opened.id != durable  # Never renumber a live first-persist session.
        assert len(store.sessions()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("cancel_shutdown", [False, True])
async def test_detached_saved_reader_is_drained_before_native_shutdown(
    shared_mobile_case, monkeypatch, fail, cancel_shutdown
):
    from threading import Event

    from tldw_chatbook.Chat.console_launch_wake import _ensure_launch_runtime

    case = shared_mobile_case
    entered, release, physical_done = Event(), Event(), Event()
    async with case.app.run_test(size=(100, 32)):
        await case.service.bootstrap_host()
        store = _ensure_launch_runtime(case.app).store
        reader = case.app.local_chat_conversation_service
        original = reader.get_conversation_tree

        def held_reader(*args, **kwargs):
            entered.set()
            try:
                assert release.wait(10)
                if fail:
                    raise RuntimeError("physical native reader failed")
                return original(*args, **kwargs)
            finally:
                physical_done.set()

        monkeypatch.setattr(reader, "get_conversation_tree", held_reader)
        first = asyncio.create_task(
            case.app.console_runtime.open_remote_conversation(case.conversation_id)
        )
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(
            case.app.console_runtime.open_remote_conversation(case.conversation_id)
        )
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        closing = asyncio.create_task(case.app.console_runtime.dispose())
        await asyncio.sleep(0.05)
        assert not physical_done.is_set()
        assert not closing.done()
        assert not second.done()
        if cancel_shutdown:
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
        release.set()
        if cancel_shutdown:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(closing, 5)
        else:
            await asyncio.wait_for(closing, 5)
        assert case.app.console_runtime.chat_controller._shutdown_requested.is_set()
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(second, 5)
        assert physical_done.is_set()
        assert not store.sessions()
        assert not store._saved_session_operations


@pytest.mark.asyncio
async def test_durable_first_send_bridge_preserves_original_live_target(
    shared_mobile_case, monkeypatch
):
    from threading import Event

    from Tests.console_provider_doubles import AgentModeGateway
    from tldw_chatbook.Chat.console_remote_models import ConsoleTarget
    from tldw_chatbook.Chat.console_session_settings import (
        default_console_session_settings,
    )

    case = shared_mobile_case

    class FinalGateway(AgentModeGateway):
        async def stream_chat(self, *args, **kwargs):
            yield "durable first Send finished"

    case.app.console_runtime.set_provider_gateway(FinalGateway())
    committed, release = Event(), Event()
    ids = []
    async with case.app.run_test(size=(100, 32)):
        host = await case.service.bootstrap_host()
        create = {"workspace_id": "global"}
        ticket = await case.service.issue_command_ticket("create", host, create)
        receipt = await case.service.submit_command(ticket["ticket"], create)
        store = case.app.console_runtime.chat_store
        live = store._sessions[receipt.native_id]
        store.replace_session_settings(
            live.id,
            default_console_session_settings(
                case.app.app_config, "openai", "gpt-4o-mini"
            ),
        )
        original = store.persistence.commit_durable_turn

        def held_commit(**kwargs):
            result = original(**kwargs)
            ids.append(kwargs["acceptance"].conversation_id)
            committed.set()
            assert release.wait(10)
            return result

        monkeypatch.setattr(store.persistence, "commit_durable_turn", held_commit)
        row = next(
            r
            for r in (await case.service.snapshot())["chats"]
            if r["session_id"] == live.id
        )
        target = ConsoleTarget(
            **host.model_dump(), session_id=live.id, incarnation_id=live.incarnation_id
        )
        payload = {
            "text": "durable first Send",
            "expected_revisions": row["expected_revisions"],
        }
        ticket = await case.service.issue_command_ticket("send", target, payload)
        sent = await case.service.submit_command(ticket["ticket"], payload)
        assert sent.state == "pending"
        phone = None
        try:
            assert await asyncio.to_thread(committed.wait, 5)
            phone = asyncio.create_task(
                case.app.console_runtime.open_remote_conversation(ids[0])
            )
            await asyncio.sleep(0.1)
            assert not phone.done()
            assert store.saved_session_ready(live)
            # The exact already-live target remains resolvable for Stop/control,
            # even while another opener waits for committed saved publication.
            from tldw_chatbook.Chat.console_remote_commands import ConsoleRemoteCommands

            assert (
                ConsoleRemoteCommands(case.app.console_runtime, host)._resolve(target)[
                    2
                ]
                is live
            )
            current = await case.service.snapshot()
            assert any(r["session_id"] == live.id for r in current["chats"])
        finally:
            release.set()
            if phone is not None:
                opened = await asyncio.wait_for(phone, 5)
        assert opened is live
        assert live.id != live.persisted_conversation_id == ids[0]
        assert len(store.sessions()) == 1
        async with asyncio.timeout(5):
            while not any(
                m.content == "durable first Send finished"
                for m in store.messages_for_session(live.id)
            ):
                await asyncio.sleep(0.01)
        assert store.active_session_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("completion", ["lookup_exit", "after_future"])
async def test_staged_save_completion_between_lookup_and_future_allocation(
    shared_mobile_case, monkeypatch, fail, completion
):
    """Release the real producer at the old lookup/install gap, not after Open."""
    import threading

    from tldw_chatbook.Chat.console_launch_wake import _ensure_launch_runtime

    case = shared_mobile_case
    entered, release_writer, producer_done = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    ui_thread = threading.get_ident()
    identities, observed = [], []
    failure = RuntimeError("exact native first-save failure")
    async with case.app.run_test(size=(100, 32)):
        await case.service.bootstrap_host()
        store = _ensure_launch_runtime(case.app).store
        live = store.create_session(title="Native staged registration race")
        writer = store.persistence.persist_console_conversation_with_policy

        def held_writer(**kwargs):
            identities.append(kwargs["conversation_id"])
            result = None if fail else writer(**kwargs)
            entered.set()
            assert release_writer.wait(10)
            if fail:
                raise failure
            return result

        monkeypatch.setattr(
            store.persistence, "persist_console_conversation_with_policy", held_writer
        )

        def save():
            try:
                return store.persist_session_if_needed(live.id)
            finally:
                producer_done.set()

        saving = asyncio.create_task(asyncio.to_thread(save))
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
        native_lock = store._preparation_lock

        class ExitGate:
            armed = True

            def __enter__(self):
                native_lock.acquire()
                return self

            def __exit__(self, *args):
                native_lock.release()
                if self.armed and threading.get_ident() == ui_thread:
                    self.armed = False
                    operation = store._saved_session_operations.get(identities[0])
                    observed.append(operation)
                    release_writer.set()
                    assert producer_done.wait(5), (
                        "Producer must retire after releasing the native lock"
                    )

            def __getattr__(self, name):
                return getattr(native_lock, name)

        if completion == "lookup_exit":
            monkeypatch.setattr(store, "_preparation_lock", ExitGate())
        opened = asyncio.create_task(
            case.app.console_runtime.open_remote_conversation(identities[0])
        )
        if completion == "after_future":
            await asyncio.sleep(0)
            observed.append(store._saved_session_operations[identities[0]])
            assert observed[0].future is not None
            release_writer.set()
        try:
            if fail:
                with pytest.raises(
                    RuntimeError, match="exact native first-save failure"
                ) as captured:
                    await asyncio.wait_for(opened, 3)
                assert captured.value is failure
                with pytest.raises(
                    RuntimeError, match="exact native first-save failure"
                ):
                    await saving
                assert live.persisted_conversation_id is None
                assert store._sessions[live.id] is live
                assert identities[0] not in store._saved_session_operations
            else:
                assert await asyncio.wait_for(opened, 3) is live
                assert await saving == identities[0]
                assert live.id != identities[0]
                assert len(store.sessions()) == 1
                store.close_session(live.id)
                reopened = await case.app.console_runtime.open_remote_conversation(
                    identities[0]
                )
                assert (
                    reopened is not live
                    and reopened.incarnation_id != live.incarnation_id
                )
            assert live.id not in store._first_identity_reservations
            assert observed and observed[0] is not None
            if completion == "lookup_exit":
                assert (
                    observed[0].future is None
                )  # Completion preceded any observer Future.
            else:
                assert observed[0].future.done()
        finally:
            release_writer.set()
            await asyncio.gather(saving, opened, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["publish", "discard", "retire"])
async def test_durable_staged_terminal_handoff_settles_native_observer(
    tmp_path, terminal
):
    from Tests.Chat.test_console_durable_turn_acceptance import _ready_store
    from Tests.conftest import _close_database_instance

    db, _, store, preparation, acceptance = _ready_store(tmp_path, attachments=False)
    session = store._sessions[preparation.session_id]
    identity = store.staged_durable_turn_identity_for(preparation.preparation_id)
    observer = asyncio.create_task(
        store.resolve_saved_session(identity.conversation_id)
    )
    try:
        await asyncio.sleep(0)
        operation = store._saved_session_operations[identity.conversation_id]
        assert operation.future is not None and not operation.future.done()
        assert store.saved_session_ready(session)
        if terminal == "discard":
            store.discard_uncommitted_durable_preparation(preparation.preparation_id)
            message = "First turn did not commit"
        else:
            commit = await asyncio.to_thread(store.commit_durable_turn, acceptance)
            if terminal == "publish":
                store.publish_durable_turn_identity(session.id, commit)
                assert await asyncio.wait_for(observer, 2) is session
            else:
                fingerprint = store.durable_acceptance_fingerprint_for(
                    preparation.preparation_id
                )
                store.retire_durable_acceptance(preparation.preparation_id, fingerprint)
                message = "First turn retired before identity publication"
        if terminal != "publish":
            with pytest.raises(RuntimeError, match=message):
                await asyncio.wait_for(observer, 2)
            assert operation.error is not None
            assert identity.conversation_id not in store._saved_session_operations
            assert store._sessions[session.id] is session
        assert session.id not in store._first_identity_reservations
    finally:
        await asyncio.gather(observer, return_exceptions=True)
        await store.close_saved_session_operations()
        _close_database_instance(db)
        assert db.registered_connection_count() == 0


@pytest.mark.asyncio
async def test_late_first_identity_completion_cannot_settle_successor(tmp_path):
    from Tests.Chat.test_console_durable_turn_acceptance import _ready_store
    from Tests.conftest import _close_database_instance

    db, _, store, preparation, _ = _ready_store(tmp_path, attachments=False)
    session = store._sessions[preparation.session_id]
    old = store.staged_durable_turn_identity_for(preparation.preparation_id)
    first = asyncio.create_task(store.resolve_saved_session(old.conversation_id))
    second = None
    try:
        await asyncio.sleep(0)
        store.discard_uncommitted_durable_preparation(preparation.preparation_id)
        with pytest.raises(RuntimeError, match="First turn did not commit"):
            await first
        new = store.stage_durable_turn_identity(session.id, preparation.preparation_id)
        assert new is not old and new.conversation_id != old.conversation_id
        second = asyncio.create_task(store.resolve_saved_session(new.conversation_id))
        await asyncio.sleep(0)
        operation = store._saved_session_operations[new.conversation_id]
        store._finish_first_saved_owner(session, old)
        assert not second.done() and not operation.future.done()
        assert store._first_identity_reservations[session.id][1] is new
        store.discard_uncommitted_durable_preparation(preparation.preparation_id)
        with pytest.raises(RuntimeError, match="First turn did not commit"):
            await second
    finally:
        await asyncio.gather(
            first, *([second] if second is not None else []), return_exceptions=True
        )
        await store.close_saved_session_operations()
        _close_database_instance(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["incarnation_id", "persisted_conversation_id"])
async def test_character_shared_handle_rejects_changed_owner_identity(field):
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleChatStore,
        ConsoleSavedSessionCleanup,
    )
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController

    store = ConsoleChatStore()
    creator = object()
    session = store.restore_persisted_session(
        title="Unpublished",
        workspace_id=None,
        persisted_conversation_id="saved-character",
        all_nodes=[],
        cleanup_owner=creator,
    )
    handle = ConsoleSavedSessionCleanup(session, creator)
    controller = ConsoleWorkspaceController.__new__(ConsoleWorkspaceController)
    controller._chat_store_accessor = lambda: store
    setattr(session, field, "replacement-identity")
    with pytest.raises(RuntimeError, match="owner changed"):
        controller._commit_character_conversation_activation(handle)
    assert not await controller._rollback_character_conversation_activation(handle)
    assert store._sessions[session.id] is session
