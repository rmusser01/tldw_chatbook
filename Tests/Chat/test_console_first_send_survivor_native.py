"""Runtime first-send persistence waits for actual original child retirement."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
import os
import sys
import threading
import traceback
from pathlib import Path

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

os.environ['TLDW_AGENTS_MAX_LIVE_SUBAGENTS'] = '2'
os.environ['TLDW_AGENTS_AUTOWAKE_ENABLED'] = 'false'
network_guard.install()
real_profile_guard.install()

async def exercise():
    from dataclasses import replace

    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_agent_bridge import SUBAGENT_PROMPT_PREFIX
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
    from tldw_chatbook import app as app_source, config
    import tldw_chatbook.Widgets.compact_model_bar
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_library_destination import resolve_console_destination
    from tldw_chatbook.Chat.console_provider_endpoints import DEFAULT_LLAMACPP_BASE_URL
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent

    creators = OriginalPrivateAppCreators(app_source, Path(os.environ['XDG_DATA_HOME']).parent)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    runtime = app.console_runtime
    app.local_mcp_control_service.manifest_provider = lambda: {'tools': []}
    store = runtime.ensure_chat_store()
    session = store.create_session(workspace_id='global')
    store.set_session_draft(session.id, 'start actual runtime survivor')
    child_entered, child_release = threading.Event(), threading.Event()
    trees, child_owners = [], []

    class Gateway(_RecordingGateway):
        def __init__(self):
            super().__init__()
            self.parent_calls = 0

        async def resolve_for_send(self, selection):
            resolution = ConsoleProviderResolution(
                provider='llama_cpp',
                base_url=selection.base_url or DEFAULT_LLAMACPP_BASE_URL,
                model=selection.explicit_model or 'm',
                ready=True,
                readiness_key='llama_cpp',
                execution_key='llama_cpp',
            )
            return replace(resolution, resolved_destination=resolve_console_destination(resolution))

        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            system = str(messages[0].get('content', '')) if messages else ''
            if system.startswith(SUBAGENT_PROMPT_PREFIX):
                child_owners.append(current_execution_owner())
                child_entered.set()
                if not await asyncio.to_thread(child_release.wait, 15):
                    raise RuntimeError('original child timed out')
                yield 'actual child returned'
                return
            trees.append(current_ordinary_tree())
            self.parent_calls += 1
            if self.parent_calls > 1 and sys.argv[2] == 'late_nonstop':
                async with asyncio.timeout(10):
                    while not hook_entered.exists():
                        await asyncio.sleep(.01)
            yield (fence(SPAWN_TOOL_NAME, {'task': 'survive original parent'})
                   if self.parent_calls == 1 else 'actual parent returned')

    gateway = Gateway()
    controller = runtime.ensure_chat_controller(store=store, provider_gateway=gateway, agent_runtime_enabled=True)
    bridge = runtime.ensure_agent_bridge(store_factory=lambda: store, provider_gateway_factory=lambda: gateway, skills_service=app.skills_scope_service)
    controller._agent_bridge = bridge
    controller._confirm_project_instruction_dispatch = lambda notice: 'proceed'
    runtime.ensure_hook_permissions()
    hook_entered = Path(os.environ['XDG_DATA_HOME']) / 'subagent-hook-entered'
    hook_release = Path(os.environ['XDG_DATA_HOME']) / 'subagent-hook-release'
    engine = None
    if sys.argv[2] == 'late_nonstop':
        from Tests.Agents.test_hooks_v2_execution import command

        runtime.execution_capacity.set_tool_limits(2, 0)
        code = (
            'import time;from pathlib import Path;'
            f'Path({str(hook_entered)!r}).touch();'
            f'exec({f"while not Path({str(hook_release)!r}).exists(): time.sleep(0.01)"!r})'
        )
        engine = runtime.ensure_hooks_v2(
            session.id, (command(code, name='SubagentStop', timeout_seconds=30),),
            lambda *_: True,
        )
    source = preparation.capture_console_received_configuration_preparation(app, store, controller, session_id=session.id)
    assert source is not None and source.trust_source is not None
    selection = preparation.ConsoleTurnCaptureSelection(
        provider_selection=ConsoleProviderSelection(provider='llama_cpp', explicit_model='m'),
        presentation_context=None, rag_defaults={},
        tool_configuration={'agent_runtime_enabled': True, 'local_tools_enabled': False},
        skill_workspace_id=None, project_bindings_eligible=False, agent_runtime_enabled=True,
    )
    launch, revision, _ = runtime.snapshot_console_staged_evidence()
    intent = ConsoleReceivedTurnIntent(
        'fix1-native-survivor', session.id, store.session_input_snapshot(session.id),
        selection, launch, revision, runtime._attached_generation or 0, None,
    )
    before = config.current_config_identity()
    publication_entered, publication_release = threading.Event(), threading.Event()
    if sys.argv[2] == 'admission':
        original_code = config.apply_settings_mutation_to_cli_config.__code__
        def observe_publication(frame, event, _arg):
            if event == 'call' and frame.f_code is original_code:
                publication_entered.set()
                publication_release.wait(20)
        threading.setprofile_all_threads(observe_publication)
    turn_id = runtime.accept_received_intent(intent, _configuration_preparation=source)
    task = runtime._turn_custody[turn_id].task
    # Hang detectors, not speed limits: four private apps at load 75-90 need
    # 40-70 s for the whole journey, most of it before the child (TASK-41316).
    async with asyncio.timeout(60):
        while not child_entered.is_set() and not task.done():
            await asyncio.sleep(.01)
    assert child_entered.is_set(), task.result() if task.done() else 'child did not enter'
    if engine is not None:
        child_release.set()
        async with asyncio.timeout(10):
            while not hook_entered.exists():
                await asyncio.sleep(.01)
    done, _ = await asyncio.wait({task}, timeout=10)
    assert done, 'parent did not complete before original child return'
    result = task.result()
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
    assert controller.run_state_for(session.id).status is ConsoleRunStatus.COMPLETED
    assert result.accepted and trees and child_owners
    tree = trees[0]
    assert tree is not None and not tree.retired.done()
    with tree.lock:
        member = tree.members[child_owners[0]]
        assert child_owners[0].child and member.row_started
    print('FIX1_PARENT_RETURN', before, config.current_config_identity(), flush=True)
    assert config.current_config_identity() == before, 'first-send publication raced original live child'
    assert app.app_config['console']['onboarding']['first_send_completed'] is True
    if sys.argv[2] == 'source_loss':
        # Owner decision (TASK-41316, 2026-11-17): retain. The surviving child's
        # reply stays in its original runs database; its host return under the
        # changed source settles unproven and the claim stays held (TASK-40716);
        # nothing is written through the changed source, and Quit still returns.
        original_config = Path(os.environ['TLDW_CONFIG_PATH'])
        store.set_session_draft(session.id, 'preserve after selected source change')
        os.environ['TLDW_CONFIG_PATH'] = str(Path(os.environ['XDG_CONFIG_HOME']) / 'replaced.toml')
        child_release.set()
        async with asyncio.timeout(60):
            while not member.host_returned:
                await asyncio.sleep(.01)
        assert bridge._db.get_run(member.run_id)['status'] == 'done'
        assert not member.durable
        assert tree.unproven_only() and not tree.retired.done()
        assert store.session_draft(session.id) == 'preserve after selected source change'
        disposal = asyncio.ensure_future(runtime.dispose())
        await asyncio.wait({disposal}, timeout=30)
        assert disposal.done(), 'dispose waited on a turn that can only be held'
        disposal.result()
        assert runtime._first_send_task.done() and not tree.retired.done()
        assert not Path(os.environ['TLDW_CONFIG_PATH']).exists()
        assert 'first_send_completed' not in original_config.read_text()
        print('FIX1_SOURCE_RETAINED', flush=True)
        os._exit(0)
    writer_calls = []
    if sys.argv[2] == 'writer_wrapped':
        import functools
        writer = config.apply_settings_mutation_to_cli_config
        @functools.wraps(writer)
        def wrapped(mutations, **kwargs):
            writer_calls.append(mutations)
            return writer(mutations, **kwargs)
        config.apply_settings_mutation_to_cli_config = wrapped
    child_release.set()
    if engine is not None:
        async with asyncio.timeout(10):
            while not hook_entered.exists():
                await asyncio.sleep(.01)
        assert not tree.retired.done()
        assert config.current_config_identity() == before
        hook_release.touch()
    async with asyncio.timeout(12):
        while not tree.retired.done():
            await asyncio.sleep(.01)
    rows = [r for r in bridge._db.list_runs(session.persisted_conversation_id) if r['agent_kind'] == 'subagent']
    assert len(rows) == 1 and rows[0]['status'] == 'done'
    child_row = bridge._db.get_run(rows[0]['id'])
    assert child_row['steps'] and member.steps
    from tldw_chatbook.Chat.console_ordinary_tree import _json
    stored_steps = {step['index']: _json(step) for step in child_row['steps']}
    assert all(stored_steps.get(index) == value for index, value in member.steps.items())
    if engine is not None:
        await asyncio.wait_for(runtime.close_hooks_v2(session.id), timeout=10)
        assert not engine._native_commands.cleanup_pending
    assert member.host_returned and member.durable and member.drained
    assert not runtime.execution_capacity.snapshot().executions
    if sys.argv[2] == 'admission':
        async with asyncio.timeout(10):
            while not publication_entered.is_set():
                await asyncio.sleep(.01)
        assert runtime._first_send_publishing
        other = store.create_session(workspace_id='global')
        store.set_session_draft(other.id, 'preserve during original save')
        other_launch, other_revision, _ = runtime.snapshot_console_staged_evidence()
        other_intent = ConsoleReceivedTurnIntent(
            'fix1-admission', other.id, store.session_input_snapshot(other.id),
            selection, other_launch, other_revision, runtime._attached_generation or 0, None,
        )
        try:
            runtime.accept_received_intent(other_intent, _configuration_preparation=source)
        except RuntimeError as error:
            assert 'being saved' in str(error)
        else:
            raise AssertionError('intake crossed original publication fence')
        assert store.session_draft(other.id) == 'preserve during original save'
        assert 'fix1-admission' not in runtime._turn_custody
        publication_release.set()
        await asyncio.shield(runtime._first_send_task)
        threading.setprofile_all_threads(None)
        fresh = preparation.capture_console_received_configuration_preparation(
            app, store, controller, session_id=other.id,
        )
        assert fresh is not None
        other_turn = runtime.accept_received_intent(other_intent, _configuration_preparation=fresh)
        other_task = runtime._turn_custody[other_turn].task
        assert (await asyncio.wait_for(other_task, 10)).accepted
        assert controller.run_state_for(other.id).status is ConsoleRunStatus.COMPLETED
    await runtime.dispose()
    import tomllib
    persisted = tomllib.loads(Path(os.environ['TLDW_CONFIG_PATH']).read_text())
    assert persisted['console']['onboarding']['first_send_completed'] is True
    assert config.current_config_identity() != before
    if sys.argv[2] == 'writer_wrapped':
        assert writer_calls == [{'console.onboarding': {'first_send_completed': True}}]
    creators.retire()
    assert not network_guard.blocked_attempts()
    assert not real_profile_guard.take_violations()
    print('retired and reopened')

try:
    with user_fixture_default_owner():
        selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
        selected.write_text('[general]\nusers_name="fix1"\n[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[chat_defaults]\nprovider="llama_cpp"\nmodel="m"\n[tools]\nfull_agent_tools_enabled=true\n', encoding='utf-8')
        selected.chmod(0o600)
        asyncio.run(asyncio.wait_for(exercise(), timeout=120))
except BaseException:
    (Path(os.environ['XDG_DATA_HOME']) / 'subagent-hook-release').touch()
    traceback.print_exc(file=sys.__stderr__)
    sys.__stderr__.flush()
    os._exit(1)
"""


def test_runtime_first_send_keeps_original_surviving_child(tmp_path):
    _run(tmp_path, "fix1_survivor", "native", script=_SCRIPT, timeout=150)


def test_runtime_first_send_waits_for_late_subagent_stop(tmp_path):
    _run(tmp_path, "fix1_survivor", "late_nonstop", script=_SCRIPT, timeout=150)


def test_runtime_first_send_fences_new_received_intake_during_publication(tmp_path):
    _run(tmp_path, "fix1_survivor", "admission", script=_SCRIPT, timeout=150)


def test_runtime_first_send_retains_changed_original_source(tmp_path):
    import os
    import subprocess
    import sys

    for name in ("home", "config", "data"):
        (tmp_path / name).mkdir(mode=0o700)
    environment = os.environ.copy()
    environment.update(
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    log = tmp_path / "source-loss-output.log"
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", _SCRIPT, "fix1", "source_loss"],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=150,
            check=False,
        )
    text = log.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, text[-3000:]
    assert "FIX1_SOURCE_RETAINED" in text


def test_runtime_first_send_saves_through_a_wrapped_config_writer(tmp_path):
    """TASK-40719: a delegating wrapper is no reason to drop the save (ADR-356 rule 4)."""
    _run(tmp_path, "fix1_survivor", "writer_wrapped", script=_SCRIPT, timeout=150)


@pytest.mark.parametrize("failure", ["body", "cleanup", "handle"])
def test_retained_source_scope_releases_only_after_proven_unwind(failure):
    """Unit control: an uncertain ExitStack unwind cannot release the tree."""
    import threading
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_agent_sources import _original_retained_source_stack
    from tldw_chatbook.Chat.console_ordinary_tree import _OrdinaryTurn

    tree = object.__new__(_OrdinaryTurn)
    tree.lock = threading.RLock()
    tree.configuration = SimpleNamespace(scratch_space=None)
    tree.scratch_lease = None
    tree.source_scopes = 0
    tree.settled = tree.sealed = False
    local = threading.local()

    class Database:
        is_memory_db = False

    database = Database()
    database._local = local
    body_error = RuntimeError("original body")
    cleanup_error = RuntimeError("failed cleanup")

    def close():
        assert tree.source_scopes == 1
        if failure == "cleanup":
            raise cleanup_error
        if failure == "handle":
            local.conn = object()

    with (
        pytest.raises(RuntimeError) as raised,
        _original_retained_source_stack(tree, (database,)) as stack,
    ):
        stack.callback(close)
        raise body_error
    assert raised.value is (cleanup_error if failure == "cleanup" else body_error)
    assert tree.source_scopes == (0 if failure == "body" else 1)


def test_ordinary_body_error_and_global_workspace_close_refusal_keep_custody(tmp_path):
    """Actual tool caller retains a failed native return until restart."""
    import os
    import subprocess
    import sys

    setup = r"""
    import types
    from tldw_chatbook.Agents import activation as actual_activation
    from tldw_chatbook.Agents.agent_service import _call_with_timeout
    from tldw_chatbook.Agents.execution_capacity import UnprovenRetention, holds_unproven_custody
    from tldw_chatbook.DB.private_sqlite import _ordinary_connections
    from tldw_chatbook.Tools import workspace_file_roots
    actual_global = workspace_file_roots._registry_factory()
    assert actual_global.db is not app.workspace_registry_service.db
    actual_global.db.close()
    primary_error = RuntimeError('original ordinary callback body failure')
    observed = {'fault_calls': 0, 'injected': False}
    decorate_code = next(c for c in actual_activation.worker_guard.__code__.co_consts if isinstance(c, types.CodeType) and c.co_name == 'decorate')
    guard_code = next(c for c in decorate_code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'call')

    def observe_body_and_return(frame, event, argument):
        if event == 'call' and frame.f_code is _call_with_timeout.__code__ and frame.f_locals.get('tool_name') == 'calculator':
            observed['body_code'] = frame.f_locals['fn'].__code__
        elif event == 'call' and frame.f_code is guard_code:
            function = frame.f_locals['function']
            if function.__qualname__ == '_call_with_timeout.<locals>._admitted_call':
                observed['global_before_guard'] = getattr(actual_global.db._thread_local, 'conn', None)
                free = dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__)))
                observed['box'] = free['box']
        elif event == 'return' and frame.f_code is observed.get('body_code') and not observed['injected']:
            tree = current_ordinary_tree()
            assert tree is not None and getattr(argument, 'ok', False)
            assert all(actual_global.db is not database for database in tree.source.databases)
            assert observed['global_before_guard'] is None
            actual_global.get_active_workspace()
            connection = actual_global.db._thread_local.conn
            observed.update(tree=tree, connection=connection, lease=_ordinary_connections[connection], worker=threading.current_thread(), had_binding='close' in vars(connection), binding=vars(connection).get('close'))
            def refuse_actual_workspace_close():
                observed['fault_calls'] += 1
                print('F_GENUINE_GLOBAL_WORKSPACE_CLOSE_REFUSAL', flush=True)
                raise RuntimeError('original global Workspace physical close refused')
            connection.close = refuse_actual_workspace_close
            observed['injected'] = True
            print('F_GENUINE_ORDINARY_CALLBACK_RETURN', flush=True)
            raise primary_error
    threading.setprofile_all_threads(observe_body_and_return)
"""
    script = _SCRIPT
    source = "    source = preparation.capture_console_received_configuration_preparation(app, store, controller, session_id=session.id)\n"
    assert script.count(source) == 1
    script = script.replace(source, setup + source, 1)
    spawn = "fence(SPAWN_TOOL_NAME, {'task': 'survive original parent'})"
    assert script.count(spawn) == 1
    script = script.replace(spawn, "fence('calculator', {'expression': '2 + 2'})", 1)
    proof = r"""
    async with asyncio.timeout(60):
        result = await task
    assert observed['injected'] and observed['fault_calls'] > 0
    tree = observed['tree']
    retained = [item for item in runtime.execution_capacity.retained() if item.operation.owner is tree.owner and item.label == 'calculator']
    assert len(retained) == 1, 'ordinary caller mistook cleanup failure for unchanged body failure'
    retention = retained[0]
    worker = observed['worker']
    async with asyncio.timeout(10):
        while worker.is_alive() and not holds_unproven_custody(worker):
            await asyncio.sleep(.01)
    assert worker.is_alive() and holds_unproven_custody(worker)
    frame = sys._current_frames().get(worker.ident)
    held_in_actual_scope = False
    while frame is not None:
        if frame.f_code is UnprovenRetention.hold.__code__ and frame.f_locals.get('self') is retention:
            held_in_actual_scope = True
            break
        frame = frame.f_back
    del frame
    assert held_in_actual_scope
    box = observed['box']
    assert box['call_error'] is primary_error
    assert isinstance(box['error'], BaseExceptionGroup) and box['error'] is not primary_error
    assert box['error'].exceptions[0] is primary_error
    connection = observed['connection']
    assert retention in runtime.execution_capacity.retained()
    assert retention.operation.owner is tree.owner
    assert tree.owner.drain_state() == 'unproven'
    assert tree.claim.id in runtime.worktree_activity._claims
    assert not tree.retired.done() and tree.source_scopes > 0
    assert not runtime._first_send_publishing and config.current_config_identity() == before
    assert getattr(actual_global.db._thread_local, 'conn', None) is not connection  # Main-thread cache is distinct.
    assert _ordinary_connections.get(connection) is observed['lease']
    assert actual_global.db._connection_quiescence.is_registered(connection)
    # Restore the exact instance binding; no cleanup marker or lease is reset.
    if observed['had_binding']:
        connection.close = observed['binding']
    else:
        del connection.close
    assert ('close' in vars(connection)) is observed['had_binding']
    assert not observed['had_binding'] or vars(connection)['close'] is observed['binding']
    threading.setprofile_all_threads(None)
    # Quit may close a provably idle handle through its native closer. It may
    # not turn the historical unknown worker return into a proven tree drain.
    disposal = asyncio.ensure_future(runtime.dispose())
    await asyncio.wait({disposal}, timeout=30)
    assert disposal.done(), 'Quit waited on retained cleanup instead of returning'
    disposal.result()
    assert not tree.retired.done() and retention in runtime.execution_capacity.retained()
    assert not network_guard.blocked_attempts() and not real_profile_guard.take_violations()
    print('F_NATIVE_BODY_AND_WORKSPACE_RETURN_HELD', flush=True)
    os._exit(0)  # Expected restart-held custody; never a successful drain claim.
"""
    start = script.index("    # Hang detectors, not speed limits:")
    end = script.index("    writer_calls = []", start)
    script = script[:start] + proof + script[end:]
    for name in ("home", "config", "data"):
        (tmp_path / name).mkdir(mode=0o700)
    environment = os.environ.copy()
    environment.update(
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    log = tmp_path / "ordinary-cleanup-held.log"
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", script, "fix1", "native"],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=150,
            check=False,
        )
    text = log.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, text[-7000:]
    assert "F_GENUINE_ORDINARY_CALLBACK_RETURN" in text
    assert "F_GENUINE_GLOBAL_WORKSPACE_CLOSE_REFUSAL" in text
    assert "F_NATIVE_BODY_AND_WORKSPACE_RETURN_HELD" in text


@pytest.mark.parametrize("control", ["delayed-entry", "late-parent"])
def test_runtime_optional_stop_admits_before_actual_producer_return(tmp_path, control):
    script = _SCRIPT
    if control == "late-parent":
        configured = "    if sys.argv[2] == 'late_nonstop':\n"
        assert script.count(configured) == 1
        script = script.replace(
            configured, "    if sys.argv[2] in {'late_nonstop', 'native'}:\n", 1
        )
        early = "    if engine is not None:\n        child_release.set()\n"
        assert script.count(early) == 1
        script = script.replace(
            early,
            "    if engine is not None and sys.argv[2] == 'late_nonstop':\n        child_release.set()\n",
            1,
        )
    setup = r"""

    from tldw_chatbook.Agents import activation as ack_activation, scratch_admission
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Backup_Recovery.admission_runtime import execution_identity
    import time
    ack_waiting = threading.Event()
    ack = {'delayed': False}
    install_code = HookSessionLifecycle.install.__code__
    wait_code = engine._wait_notification_admissions.__code__
    deliver_code = engine._deliver.__code__
    current_code = engine._deliver_current.__code__

    def observe_native_entry(frame, event, argument):
        if frame.f_code is install_code and event == 'call':
            hook_event = frame.f_locals.get('event')
            if getattr(hook_event, 'event', None) == 'SubagentStop':
                lifecycle = frame.f_locals['self']
                ack['owner_current'] = lifecycle.checkpoints.is_current(frame.f_locals['owner'])
                ack['preheld'] = frame.f_locals['_execution'] is not None
                ack['wait_requested'] = frame.f_locals['_wait_native_admission']
                ack['producer'] = scratch_admission._current.get()
                ack['producer_local'] = ack['producer'].identity == execution_identity()
        elif frame.f_code is wait_code:
            scope = frame.f_locals['execution']
            if dict(scope._state.identity).get('event') == 'SubagentStop':
                if event == 'call':
                    ack_waiting.set()
                elif event == 'return':
                    entries = scope._state.notification_entries
                    ack['all_entered'] = bool(entries) and all(entry.done() and entry.result() is True for entry in entries)
        elif frame.f_code is deliver_code and event == 'call':
            hook_event = frame.f_locals['event']
            if hook_event.event == 'SubagentStop' and not ack['delayed']:
                ack['delayed'] = True
                scope = frame.f_locals['execution']
                # Existing producer must actually enter its bounded ACK wait;
                # the receiver cannot win by racing its lexical source return.
                assert ack_waiting.wait(min(10, max(0, scope.deadline - time.monotonic())))
                ack['producer_live_before_entry'] = ack['producer'].live
        elif frame.f_code is current_code and event == 'call':
            if frame.f_locals['event'].event == 'SubagentStop':
                own = scratch_admission._current.get()
                state = ack_activation._guard.context.get()
                ack['receiver'] = own
                ack['independent'] = own is not ack['producer'] and own.identity == execution_identity() and state.identity == execution_identity()

    threading.setprofile_all_threads(observe_native_entry)
"""
    source = "    source = preparation.capture_console_received_configuration_preparation(app, store, controller, session_id=session.id)\n"
    assert script.count(source) == 1
    script = script.replace(source, setup + source, 1)
    proof = r"""
        assert ack['delayed'] and ack['preheld'] and ack['wait_requested']
        assert ack['owner_current'] is EXPECT_CURRENT
        assert ack['producer_local'] and ack['producer_live_before_entry']
        assert ack['independent'] and ack['all_entered']
        async with asyncio.timeout(10):
            while ack['producer'].live:
                await asyncio.sleep(.01)
        assert ack['receiver'].live, 'optional command still needs its independent source scope'
        print('F_NATIVE_STOP_ACK_' + ACK_CONTROL, flush=True)
""".replace("EXPECT_CURRENT", repr(control == "delayed-entry")).replace(
        "ACK_CONTROL", repr(control)
    )
    release = "        hook_release.touch()\n"
    assert script.count(release) == 1
    script = script.replace(release, proof + release, 1)
    cleanup = "    await runtime.dispose()\n"
    assert script.count(cleanup) == 1
    script = script.replace(
        cleanup, "    threading.setprofile_all_threads(None)\n" + cleanup, 1
    )
    _run(
        tmp_path,
        "fix1_ack",
        "native" if control == "late-parent" else "late_nonstop",
        script=script,
        timeout=150,
    )
    text = (tmp_path / "restart-output.log").read_text(encoding="utf-8")
    assert "F_NATIVE_STOP_ACK_" + control in text
