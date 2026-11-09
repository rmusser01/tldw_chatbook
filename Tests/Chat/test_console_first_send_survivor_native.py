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
    async with asyncio.timeout(12):
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
        failed_child = threading.Event()
        failures = []
        original_hook = threading.excepthook
        def observe_child_failure(args):
            failures.append((args.exc_type, str(args.exc_value)))
            failed_child.set()
        threading.excepthook = observe_child_failure
        store.set_session_draft(session.id, 'preserve after selected source change')
        os.environ['TLDW_CONFIG_PATH'] = str(Path(os.environ['XDG_CONFIG_HOME']) / 'replaced.toml')
        child_release.set()
        async with asyncio.timeout(10):
            while not failed_child.is_set():
                await asyncio.sleep(.01)
        threading.excepthook = original_hook
        assert failures and failures[0][0] is RuntimeError
        assert 'original agent storage source changed' in failures[0][1]
        assert not tree.retired.done() and not runtime._first_send_task.done()
        assert store.session_draft(session.id) == 'preserve after selected source change'
        assert not Path(os.environ['TLDW_CONFIG_PATH']).exists()
        print('FIX1_SOURCE_RETAINED', flush=True)
        os._exit(0)
    producer_invoked = []
    if sys.argv[2] == 'producer_replaced':
        from tldw_chatbook.Chat import console_received_dispatch as persistence
        persistence._native_config_current = lambda: producer_invoked.append(True)
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
    if sys.argv[2] == 'producer_replaced':
        assert not persisted.get('console', {}).get('onboarding', {}).get('first_send_completed')
        assert not producer_invoked
        assert config.current_config_identity() == before
    else:
        assert persisted['console']['onboarding']['first_send_completed'] is True
        assert config.current_config_identity() != before
    creators.retire()
    assert not network_guard.blocked_attempts()
    assert not real_profile_guard.take_violations()
    print('retired and reopened')

try:
    with user_fixture_default_owner():
        selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
        selected.write_text('[general]\nusers_name="fix1"\n[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[chat_defaults]\nprovider="llama_cpp"\nmodel="m"\n', encoding='utf-8')
        selected.chmod(0o600)
        asyncio.run(asyncio.wait_for(exercise(), timeout=35))
except BaseException:
    (Path(os.environ['XDG_DATA_HOME']) / 'subagent-hook-release').touch()
    traceback.print_exc(file=sys.__stderr__)
    sys.__stderr__.flush()
    os._exit(1)
"""


def test_runtime_first_send_keeps_original_surviving_child(tmp_path):
    _run(tmp_path, "fix1_survivor", "native", script=_SCRIPT, timeout=45)


def test_runtime_first_send_waits_for_late_subagent_stop(tmp_path):
    _run(tmp_path, "fix1_survivor", "late_nonstop", script=_SCRIPT, timeout=50)


def test_runtime_first_send_fences_new_received_intake_during_publication(tmp_path):
    _run(tmp_path, "fix1_survivor", "admission", script=_SCRIPT, timeout=50)


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
            timeout=45,
            check=False,
        )
    text = log.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, text[-3000:]
    assert "FIX1_SOURCE_RETAINED" in text


def test_runtime_first_send_refuses_replaced_native_source_check(tmp_path):
    _run(tmp_path, "fix1_survivor", "producer_replaced", script=_SCRIPT, timeout=50)


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
