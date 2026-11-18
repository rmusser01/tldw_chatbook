"""Real cold skill receipt and original MCP pending source share one turn."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, os, sys, threading
import contextvars
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()

async def exercise():
    from tldw_chatbook import app as app_source, app_service_wiring
    import tldw_chatbook.Widgets.compact_model_bar
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators

    if sys.argv[2] == 'native_managed':
        from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

        seed = LocalSkillsService(
            store_dir=app_service_wiring.default_local_skills_store_dir(app_service_wiring.get_user_data_dir()),
        )
        await seed.create_skill(
            name='task8-managed',
            content='---\nname: task8-managed\ndescription: Managed trust control\n---\nFixture body.\n',
            trust_approved=False,
        )
        assert seed._trust_service is None
    creators = OriginalPrivateAppCreators(app_source, Path(os.environ['XDG_DATA_HOME']).parent)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    runtime = app.console_runtime
    facade = app.skills_scope_service
    local_skills = app.local_skills_service
    assert vars(app)['_local_skill_trust_service'] is None
    assert vars(local_skills)['_trust_service'] is None
    calls, models, receipts, witnesses = [], [], [], []
    queued_case = sys.argv[2].startswith('queued') or sys.argv[2] in {'mounted_queued', 'mounted_new_queued', 'mounted_trust_replaced', 'mounted_policy_changed', 'mounted_queue_before'}
    replaced_case = sys.argv[2] in {'queued_replaced', 'mounted_trust_replaced'}
    policy_changed_case = sys.argv[2] == 'mounted_policy_changed'
    queue_before_case = sys.argv[2] == 'mounted_queue_before'
    mounted_case = sys.argv[2].startswith('mounted')
    first_entered, release_first = threading.Event(), threading.Event()
    claim_entered, release_claim = threading.Event(), threading.Event()
    publication_entered, release_publication = threading.Event(), threading.Event()
    changed_calls = []
    if queued_case:
        assert local_skills.plugin_service is not None
        assert vars(local_skills)['_trust_service'] is None

    def manifest():
        calls.append((current_execution_owner(), current_ordinary_tree()))
        return {'tools': []}

    # Install the fixture's original callback before any received source capture.
    app.local_mcp_control_service.manifest_provider = manifest
    if sys.argv[2] in {'mounted_queued', 'mounted_new_queued', 'mounted_trust_replaced', 'mounted_policy_changed', 'mounted_queue_before'}:
        # Existing native receiver before the tested precursor; trust stays cold.
        app.unified_mcp_service.get_kill_switch()
    store = runtime.ensure_chat_store()
    session = store.create_session(workspace_id='global')
    if sys.argv[2] in {'mounted_queued', 'mounted_trust_replaced'}:
        from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicyCandidate

        policy = session.library_policy_holder.snapshot
        conversation = store.persistence.create_conversation(conversation_title='existing queued conversation')
        inserted = store.persistence.console_library_policy_repository.insert(
            conversation, ConsoleLibraryPolicyCandidate(policy.auto_retrieve, policy.assistant_access),
        )
        assert inserted.snapshot.source == 'durable'
        store.publish_first_persisted_conversation(session.id, conversation)
        await store.hydrate_session_library_policy(session.id)
    store.set_session_draft(session.id, 'original cold skill receipt')

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            models.append(current_ordinary_tree())
            self.payloads.append(messages)
            if queued_case and len(models) == 1:
                first_entered.set()
                assert await asyncio.to_thread(release_first.wait, 15)
            yield 'original cold receipt MODEL'

    gateway = Gateway()
    controller = runtime.ensure_chat_controller(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True,
    )
    bridge = runtime.ensure_agent_bridge(
        store_factory=lambda: store, provider_gateway_factory=lambda: gateway,
        skills_service=facade,
    )
    assert bridge is not None
    controller._agent_bridge = bridge
    runtime.ensure_hook_permissions()
    if replaced_case or policy_changed_case:
        original_submit = runtime._submit_queued_turn

        async def held_submit(*args, **kwargs):
            claim_entered.set()
            assert await asyncio.to_thread(release_claim.wait, 15)
            return await original_submit(*args, **kwargs)

        controller.prompt_queue_coordinator.bind_runtime_submitter(held_submit)
    if queue_before_case:
        original_reconcile = store.reconcile_durable_turn_settings

        async def held_reconcile(session_id, commit):
            if commit.first_persist:
                publication_entered.set()
                assert await asyncio.to_thread(release_publication.wait, 15)
            return await original_reconcile(session_id, commit)

        store.reconcile_durable_turn_settings = held_reconcile
    screen = mounted = None
    if mounted_case:
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
        from tldw_chatbook.UI.Console_Modules.wiring import _admit_console_turn_to_runtime, _precapture_turn_authority

        screen = ChatScreen(app)
        mounted = await _precapture_turn_authority(screen, session.id)
        witness = preparation._mcp_capture_witness.get()
        source = witness[-5].preparation
        assert source.trust_source is not None and not calls
        if sys.argv[2] in {'mounted_lost', 'mounted_missing', 'mounted_injected'}:
            try:
                with mounted():
                    configuration = screen._session._build_console_turn_execution_context(session.id)
                    assert configuration.skill_capture_pending
                    if sys.argv[2] == 'mounted_lost':
                        original_factory = local_skills._trust_service_factory
                        local_skills._trust_service_factory = lambda: None
                        try:
                            try:
                                screen._session._build_console_turn_execution_context(session.id)
                            except RuntimeError as error:
                                assert 'source changed' in str(error)
                            else:
                                raise AssertionError('changed original trust factory was accepted')
                        finally:
                            local_skills._trust_service_factory = original_factory
                    elif sys.argv[2] == 'mounted_missing':
                        snapshot = controller.prompt_queue_registry.snapshot(session.id)
                        task = contextvars.Context().run(asyncio.create_task, controller.queue_prompt(
                            session.id, text='original cold skill receipt',
                            expected_revision=snapshot.revision, configuration=configuration,
                        ))
                        try:
                            await task
                        except RuntimeError as error:
                            assert 'Original queued capture source is unavailable' in str(error)
                        else:
                            raise AssertionError('missing private pending source was accepted')
                        assert controller.prompt_queue_registry.snapshot(session.id) == snapshot
                    else:
                        from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService

                        # Deliberately replace the captured cold source with an
                        # unpublished receiver of the native type. No receipt is
                        # manufactured and no receiver method is invoked.
                        injected = object.__new__(SkillTrustService)
                        vars(app)['_local_skill_trust_service'] = injected
                        snapshot = controller.prompt_queue_registry.snapshot(session.id)
                        try:
                            try:
                                await controller.queue_prompt(
                                    session.id, text='original cold skill receipt',
                                    expected_revision=snapshot.revision,
                                    configuration=configuration,
                                )
                            except RuntimeError:
                                pass
                            else:
                                raise AssertionError('unpublished trust receiver was adopted')
                            assert controller.prompt_queue_registry.snapshot(session.id) == snapshot
                        finally:
                            vars(app)['_local_skill_trust_service'] = None
                assert store.session_draft(session.id) == 'original cold skill receipt'
                assert not calls and not models
            finally:
                await runtime.dispose()
                creators.retire()
            assert not network_guard.blocked_attempts()
            assert not real_profile_guard.take_violations()
            print('retired and reopened')
            return
    else:
        source = preparation.capture_console_received_configuration_preparation(
        app, store, controller, session_id=session.id,
        )
    if not mounted_case:
        assert source is not None and source.trust_source is not None
        assert source.app is app and source.store is store and source.creator is controller
    assert vars(local_skills)['_trust_service'] is None
    selection = preparation.ConsoleTurnCaptureSelection(
        provider_selection=ConsoleProviderSelection(provider='llama_cpp', explicit_model='m'),
        presentation_context=None, rag_defaults={},
        tool_configuration={'agent_runtime_enabled': True, 'local_tools_enabled': False},
        skill_workspace_id=None, project_bindings_eligible=False, agent_runtime_enabled=True,
    )
    first_task = None
    if queued_case:
        store.set_session_draft(session.id, 'original first received turn')
        first_launch, first_revision, _notice = runtime.snapshot_console_staged_evidence()
        first_intent = ConsoleReceivedTurnIntent(
            'first-real-received', session.id, store.session_input_snapshot(session.id),
            selection, first_launch, first_revision, runtime._attached_generation or 0, None,
        )
        first_id = runtime.accept_received_intent(first_intent, _configuration_preparation=source)
        first_task = runtime._turn_custody[first_id].task
        expected_entry = publication_entered if queue_before_case else first_entered
        async with asyncio.timeout(10):
            while not expected_entry.is_set() and not first_task.done():
                await asyncio.sleep(0.01)
        if not expected_entry.is_set():
            release_first.set()
            release_publication.set()
            await first_task
            raise AssertionError('first original publication or MODEL not entered')
        if queue_before_case:
            provisional = vars(session).get('_console_first_library_publication')
            assert type(provisional) is preparation._FirstLibraryPublication and not provisional.verified
            assert provisional.store is store and provisional.session is session
            assert provisional.commit.first_persist and provisional.result.snapshot.source == 'durable'
        elif sys.argv[2] in {'mounted_new_queued', 'mounted_policy_changed'}:
            q1_receipt = vars(session).get('_console_first_library_publication')
            assert type(q1_receipt) is preparation._FirstLibraryPublication and q1_receipt.verified
            assert q1_receipt.store is store and q1_receipt.session is session
            assert q1_receipt.commit.first_persist and q1_receipt.result.snapshot.policy_revision == 1
        if sys.argv[2] in {'mounted_queued', 'mounted_trust_replaced'}:
            receipt = vars(local_skills).get('_console_native_skill_trust_publication')
            assert type(receipt) is preparation._NativeSkillTrustPublication
            assert receipt.source.app is app and receipt.source.local is local_skills
            assert receipt.initial_app is None and receipt.initial_local is None
            assert receipt.app_winner is vars(app)['_local_skill_trust_service']
            assert receipt.local_winner is vars(local_skills)['_trust_service'] is receipt.app_winner
        store.set_session_draft(session.id, 'original cold skill receipt')
        try:
            preparation.require_received_configuration_preparation(
                source, app, store, controller, session_id=session.id,
            )
        except BaseException:
            try:
                fresh = preparation._source_references(app, store, controller, session, _unused_trust=source.trust_source)
                print('QUEUE_SOURCE_DIFF', [name for name, a, b in zip(fresh._fields[:-1], source.refs[:-1], fresh[:-1]) if a is not b],
                      [index for index, (a, b) in enumerate(zip(source.refs.callbacks, fresh.callbacks)) if a is not b], flush=True)
            except Exception as error:
                print('QUEUE_SOURCE_DIFF_ERROR', type(error).__name__, str(error), flush=True)
            release_first.set()
            await asyncio.gather(first_task, return_exceptions=True)
            await runtime.dispose()
            creators.retire()
            raise
    launch, revision, _notice = runtime.snapshot_console_staged_evidence()
    intent = ConsoleReceivedTurnIntent(
        'original-cold-receipt', session.id, store.session_input_snapshot(session.id),
        selection, launch, revision, runtime._attached_generation or 0,
        controller.prompt_queue_registry.snapshot(session.id).revision if queued_case else None,
    )
    finish_code = preparation.finish_console_skill_catalog_owned.__code__
    library_code = type(controller)._capture_turn_library_authority.__code__
    first_library_code = preparation.finish_first_library_execution_read.__code__
    library_authorities, first_library_checks = [], []
    execution_proofs = []
    from tldw_chatbook.Chat.console_dispatch_checkpoint import ConsoleTurnLibraryAuthority
    from tldw_chatbook.Chat.console_ordinary_tree import admit_ordinary_turn
    admission_code = admit_ordinary_turn.__code__

    def profile(frame, event, arg):
        if event == 'return' and frame.f_code is finish_code and type(arg) is preparation.ConsolePreparedSkillContext:
            receipts.append(arg)
        if event == 'call' and frame.f_code is admission_code and not witnesses:
            config = frame.f_locals['configuration']
            execution_proofs.append(config.recovery_user_origin)
            witnesses.append((config.mcp_capture_pending, preparation.original_mcp_capture_matches(
                controller, session, config, pending=config.mcp_capture_pending,
            )))
        if event == 'return' and frame.f_code is library_code and type(arg) is ConsoleTurnLibraryAuthority:
            library_authorities.append(arg)
        if event == 'return' and frame.f_code is first_library_code and type(arg) is bool:
            first_library_checks.append(arg)

    sys.setprofile(profile)
    try:
        if mounted_case:
            with mounted():
                configuration = screen._session._build_console_turn_execution_context(session.id)
                assert configuration.skill_capture_pending and configuration.mcp_capture_pending
                assert screen._session._build_console_turn_execution_context(session.id) is configuration
                if sys.argv[2] in {'mounted_new_queued', 'mounted_policy_changed', 'mounted_queue_before'}:
                    captured = configuration.library_policy_maximum
                    published = session.library_policy_holder.snapshot
                    assert captured.source == 'new_session' and captured.policy_revision is None
                    assert published.source == 'durable' and published.policy_revision == 1
                    assert (captured.auto_retrieve, captured.assistant_access) == (
                        published.auto_retrieve, published.assistant_access,
                    )
                if queued_case:
                    result = await controller.queue_prompt(
                        session.id, text='original cold skill receipt',
                        expected_revision=controller.prompt_queue_registry.snapshot(session.id).revision,
                        configuration=configuration,
                    )
                    turn_id = None
                else:
                    turn_id = _admit_console_turn_to_runtime(screen, 'original cold skill receipt', session.id)
        else:
            turn_id = runtime.accept_received_intent(intent, _configuration_preparation=source)
        record = runtime._turn_custody[turn_id] if turn_id is not None else None
        try:
            if sys.argv[2] == 'mounted':
                assert record.request.configuration is configuration
                proof = record.request.recovery_user_origin
                recovery = runtime.ensure_archive_recovery()
                assert proof is not None
                assert recovery._accepted_proofs[id(proof)] == (proof, turn_id)
            if record is not None:
                result = await record.task
            if sys.argv[2] == 'mounted':
                assert len(execution_proofs) == 1 and execution_proofs[0] is proof
                assert id(proof) not in recovery._accepted_proofs
            if queued_case:
                if queue_before_case:
                    assert result.status.value == 'reroute_normal_send'
                    assert controller.prompt_queue_registry.snapshot(session.id).total_count == 0
                    assert store.session_draft(session.id) == 'original cold skill receipt'
                    assert vars(session)['_console_first_library_publication'] is provisional
                    assert not provisional.verified
                    release_first.set()
                    release_publication.set()
                    await asyncio.wait_for(first_task, 10)
                    verified = vars(session).get('_console_first_library_publication')
                    assert verified.verified and verified.commit is provisional.commit
                    assert verified.result is provisional.result
                    assert len(models) == 1
                    return
                assert result.applied, result
                assert controller.prompt_queue_registry.snapshot(session.id).waiting_count == 1
                release_first.set()
                release_publication.set()
                if replaced_case or policy_changed_case:
                    assert await asyncio.to_thread(claim_entered.wait, 10)
                    if policy_changed_case:
                        from tldw_chatbook.Chat.console_library_policy import (
                            ConsoleAutoRetrieve, ConsoleLibraryPolicyCandidate,
                        )

                        current = session.library_policy_holder.snapshot
                        next_auto = (ConsoleAutoRetrieve.AUTOMATIC if current.auto_retrieve is ConsoleAutoRetrieve.NEVER
                                     else ConsoleAutoRetrieve.NEVER)
                        store.stage_session_library_policy(session.id, ConsoleLibraryPolicyCandidate(
                            next_auto, current.assistant_access,
                        ))
                        changed = await store.save_session_library_policy(session.id)
                        assert changed.status.value == 'committed' and changed.snapshot.policy_revision == 2
                    elif sys.argv[2] == 'mounted_trust_replaced':
                        from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService

                        original_winner = vars(app)['_local_skill_trust_service']
                        vars(app)['_local_skill_trust_service'] = object.__new__(SkillTrustService)
                    else:
                        app.local_mcp_control_service.manifest_provider = lambda: changed_calls.append(True) or {'tools': []}
                    release_claim.set()
                    try:
                        try:
                            await asyncio.wait_for(first_task, 10)
                        except RuntimeError as error:
                            if not policy_changed_case:
                                assert 'Original MCP capture source is unavailable' in str(error)
                        else:
                            if not policy_changed_case:
                                raise AssertionError('changed source did not refuse')
                    finally:
                        if sys.argv[2] == 'mounted_trust_replaced':
                            vars(app)['_local_skill_trust_service'] = original_winner
                    assert len(models) == 1 and not changed_calls
                    if sys.argv[2] in {'mounted_trust_replaced', 'mounted_policy_changed'}:
                        retained = controller.prompt_queue_registry.snapshot(session.id)
                        assert retained.waiting_count == 1 and retained.claimed_count == 0
                        assert store.session_draft(session.id) == 'original cold skill receipt'
                else:
                    await asyncio.wait_for(first_task, 10)
                    async with asyncio.timeout(10):
                        while len(models) < 2 and controller.prompt_queue_registry.snapshot(session.id).mode.value != 'paused':
                            await asyncio.sleep(0.01)
                    assert len(models) == 2, controller.prompt_queue_registry.snapshot(session.id)
        except BaseException:
            print('RECEIPT_DIAGNOSTIC', len(receipts), witnesses, flush=True)
            raise
    finally:
        sys.setprofile(None)
        release_first.set()
        release_claim.set()
        release_publication.set()
        if first_task is not None:
            await asyncio.gather(first_task, return_exceptions=True)
        await runtime.dispose()
        creators.retire()
        if queue_before_case:
            assert not network_guard.blocked_attempts()
            assert not real_profile_guard.take_violations()
            print('retired and reopened')
    assert (result.applied if queued_case else result.accepted) and models and gateway.payloads
    if sys.argv[2] == 'mounted_trust_replaced':
        assert not receipts
    else:
        assert receipts and receipts[0].catalog.preparation is source
    if sys.argv[2] == 'native_managed':
        assert receipts[0].trust_refs is not None
        receipt = vars(local_skills).get('_console_native_skill_trust_publication')
        assert type(receipt) is preparation._NativeSkillTrustPublication
        assert receipt.source is source.trust_source
        assert receipt.initial_app is None and receipt.initial_local is None
        assert receipt.app_winner is vars(app)['_local_skill_trust_service']
        assert receipt.local_winner is vars(local_skills)['_trust_service'] is receipt.app_winner
    assert witnesses == [(True, not replaced_case)], witnesses
    if sys.argv[2] in {'mounted_new_queued', 'mounted_queue_before'}:
        assert first_library_checks == [True], first_library_checks
        assert library_authorities and library_authorities[-1].policy.source == 'durable'
        assert library_authorities[-1].policy.policy_revision == 1
        final_q1 = vars(session).get('_console_first_library_publication')
        assert final_q1.verified and final_q1.commit.first_persist
    if policy_changed_case:
        assert first_library_checks == [False], first_library_checks
    target = models[-1]
    if queued_case and not replaced_case and not policy_changed_case:
        assert any(tree is target and owner is tree.owner for owner, tree in calls)
    elif not replaced_case and not policy_changed_case:
        assert calls and all(owner is target.owner and tree is target for owner, tree in calls)
    assert target.footprint.opaque
    assert not network_guard.blocked_attempts()
    assert not real_profile_guard.take_violations()
    print('retired and reopened')

with user_fixture_default_owner():
    selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selected.write_text('[general]\nusers_name="task8-receipt"\n[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[chat_defaults]\nprovider="llama_cpp"\nmodel="m"\n', encoding='utf-8')
    selected.chmod(0o600)
    asyncio.run(asyncio.wait_for(exercise(), timeout=70))
"""


@pytest.mark.parametrize(
    "case",
    [
        "native",
        "native_managed",
        "queued",
        "queued_replaced",
        "mounted",
        "mounted_queued",
        "mounted_new_queued",
        "mounted_lost",
        "mounted_missing",
        "mounted_injected",
        "mounted_trust_replaced",
        "mounted_policy_changed",
        "mounted_queue_before",
    ],
)
def test_real_received_cold_skill_receipt_and_custom_pending_mcp(tmp_path, case):
    _run(tmp_path, "ordinary_receipt", case, script=_SCRIPT, timeout=70)
