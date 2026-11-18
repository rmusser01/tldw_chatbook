"""Real mounted Console, original native admission, and physical retirement."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile


_SCRIPT = r"""
import asyncio
import inspect
import os
from pathlib import Path
import sys
import threading
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
os.environ['PYTHON_KEYRING_BACKEND'] = 'keyring.backends.null.Keyring'

async def until(predicate):
    async with asyncio.timeout(60):
        while not predicate():
            await asyncio.sleep(.01)

async def exercise():
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Console_Modules import skill
    from tldw_chatbook.Widgets import compact_model_bar as metadata
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Chat import console_preparation_reads

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await until(lambda: getattr(app, '_ui_ready', False))
        app.post_message(NavigateToScreen('chat'))
        await until(lambda: type(app.screen).__name__ == 'ChatScreen')
        await pilot.pause(3)
        controller = app.screen._skill
        facade = app.skills_scope_service
        local = app.local_skills_service
        owner = skill._capture_console_skill_discovery(controller)
        assert owner is not None and owner[0]()
        context = await controller._fetch_console_skill_context()
        assert context.get('available_skills'), context
        assert skill._capture_console_skill_context_read(app, facade, wiring, metadata) is not None

        # Observe/hold the ORIGINAL method after its ORIGINAL content admission.
        # No product callback or source proof is patched for this control.
        target = inspect.unwrap(LocalSkillsService.get_context).__code__
        main = threading.get_ident()
        tool_id = 4
        for outcome in ('cancel', 'source'):
            entered, release = threading.Event(), threading.Event()
            threads = []
            held_reads = []
            original_scope = app._skills_scope_service
            def hold(code, offset):
                binding = console_preparation_reads._source_binding.get()
                if binding is not None and binding.read.task is task and not entered.is_set():
                    held_reads.append(binding.read)
                    threads.append(threading.get_ident())
                    entered.set()
                    assert release.wait(10), 'native hold watchdog expired'
            sys.monitoring.use_tool_id(tool_id, 'skill-context-native-control')
            sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, hold)
            sys.monitoring.set_local_events(tool_id, target, sys.monitoring.events.PY_START)
            task = asyncio.create_task(controller._fetch_console_skill_context())
            try:
                await until(entered.is_set)
                assert threads == [threads[0]] and threads[0] != main
                assert held_reads[0] in app.console_runtime._preparation_reads
                await asyncio.sleep(.02)  # UI loop runs while native read is held.
                if outcome == 'cancel':
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done()
                    assert held_reads[0] in app.console_runtime._preparation_reads
                else:
                    app._skills_scope_service = object()
            finally:
                release.set()
                sys.monitoring.set_local_events(tool_id, target, 0)
                sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, None)
                sys.monitoring.free_tool_id(tool_id)
            try:
                if outcome == 'cancel':
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    else:
                        raise AssertionError('cancelled reader published')
                else:
                    assert await task == {}, 'stale context published'
            finally:
                app._skills_scope_service = original_scope
            assert held_reads[0] not in app.console_runtime._preparation_reads

        # Original facade with custom local collaborators keeps its UI affinity.
        original_policy = facade.policy_enforcer
        original_inputs = app._local_skills_stack_inputs
        calls = []
        class Policy:
            def require_allowed(self, *, action_id):
                calls.append(threading.get_ident())
        policy = Policy()
        facade.policy_enforcer = local.policy_enforcer = policy
        app._local_skills_stack_inputs = (policy, original_inputs[1])
        try:
            assert skill._capture_console_skill_context_read(app, facade, wiring, metadata) is None
            assert (await controller._fetch_console_skill_context()).get('available_skills')
            assert calls and set(calls) == {main}, (main, calls)
        finally:
            facade.policy_enforcer = local.policy_enforcer = original_policy
            app._local_skills_stack_inputs = original_inputs

        original_loader = local._builtin_disabled_loader
        local._builtin_disabled_loader = lambda: frozenset()
        try:
            assert skill._capture_console_skill_context_read(app, facade, wiring, metadata) is None
        finally:
            local._builtin_disabled_loader = original_loader
        trust = local._trust_service
        class CustomTrust:
            def __bool__(self):
                raise AssertionError('classification invoked a custom callback')
        local._trust_service = CustomTrust()
        try:
            assert skill._capture_console_skill_context_read(app, facade, wiring, metadata) is None
        finally:
            local._trust_service = trust
        for method in ('status_for_skill', '_status_for_skill'):
            setattr(trust, method, lambda name: None)
            try:
                assert skill._capture_console_skill_context_read(app, facade, wiring, metadata) is None
            finally:
                delattr(trust, method)

        body = inspect.unwrap(LocalSkillsService.get_context)
        original_code = body.__code__
        body.__globals__['_skill_context_control_threads'] = calls
        async def changed_context(self):
            import threading
            _skill_context_control_threads.append(threading.get_ident())
            return {'available_skills': [{'name': 'custom-context'}]}
        calls.clear()
        try:
            body.__code__ = changed_context.__code__
            assert skill._capture_console_skill_context_read(app, facade, wiring, metadata) is None
            assert await controller._fetch_console_skill_context() == {
                'available_skills': [{'name': 'custom-context', 'backend': 'local', 'record_id': 'local:skill:custom-context'}],
                'backend': 'local',
            }
            assert calls == [main]
        finally:
            body.__code__ = original_code
            del body.__globals__['_skill_context_control_threads']
    assert not network_guard.blocked_attempts()
    assert not real_profile_guard.take_violations()
    print('retired and reopened')

with user_fixture_default_owner():
    Path(os.environ['TLDW_CONFIG_PATH']).write_text(
        '[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n'
        '[chat_defaults]\nprovider="llama_cpp"\nmodel="local-model"\n'
        '[api_settings.llama_cpp]\napi_url="http://127.0.0.1:9099"\nmodel="local-model"\n',
        encoding='utf-8',
    )
    asyncio.run(exercise())
"""


def test_original_context_read_is_responsive_owned_and_custom_affine(tmp_path):
    launch = (
        "from Tests.Performance.test_console_skill_context_owned import _SCRIPT; "
        "exec(compile(_SCRIPT, '<skill-context-control>', 'exec'))"
    )
    _run(tmp_path, "skill-context", "owned", script=launch, timeout=150)
