"""Finite discovery reads preserve cancellation and publication ownership."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancel", "source", "owner", "error"])
async def test_context_read_retains_native_work_until_exit(outcome):
    from tldw_chatbook.UI.Console_Modules.skill import _read_console_skill_context

    entered, release = threading.Event(), threading.Event()
    reads = set()
    main = threading.get_ident()
    state = {"source": True, "owner": True}
    observed = []

    async def get_context(*, mode):
        observed.append((mode, threading.get_ident()))
        entered.set()
        assert release.wait(5)
        if outcome == "error":
            raise OSError("checked reader failed")
        return {"available_skills": [{"name": "checked"}]}

    source = SimpleNamespace(runtime_reads=reads)
    task = asyncio.create_task(
        _read_console_skill_context(
            get_context, source, lambda: state["source"], lambda: state["owner"]
        )
    )
    try:
        async with asyncio.timeout(3):
            while not entered.is_set():
                await asyncio.sleep(0.005)
        assert observed == [("local", observed[0][1])]
        assert observed[0][1] != main
        assert len(reads) == 1
        if outcome == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and len(reads) == 1
        elif outcome in {"source", "owner"}:
            state[outcome] = False
    finally:
        release.set()
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome in {"source", "owner"}:
        with pytest.raises(RuntimeError, match="console_skill_context_changed"):
            await task
    elif outcome == "error":
        with pytest.raises(OSError, match="checked reader failed"):
            await task
    else:
        assert await task == {"available_skills": [{"name": "checked"}]}
    assert not reads


_EAGER_DISCOVERY_SCRIPT = r"""
import asyncio, inspect, os, sys, threading
from pathlib import Path
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
    # What App.run_async installs; run_test alone does not.
    asyncio.get_running_loop().set_task_factory(asyncio.eager_task_factory)
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.app_skill_wiring import SkillServiceWiringMixin
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    main = threading.get_ident()
    threads = {'read': [], 'build': []}
    codes = {
        inspect.unwrap(LocalSkillsService.get_context).__code__: 'read',
        SkillServiceWiringMixin._build_local_skill_trust_service.__code__: 'build',
    }
    tool = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(tool, 'eager-skill-discovery')
    sys.monitoring.register_callback(
        tool, sys.monitoring.events.PY_START,
        lambda code, _offset: threads[codes[code]].append(threading.get_ident()),
    )
    for code in codes:
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
    try:
        app = TldwCli()
        async with app.run_test(size=(170, 48)):
            await until(lambda: getattr(app, '_ui_ready', False))
            if type(app.screen).__name__ != 'ChatScreen':
                app.post_message(NavigateToScreen('chat'))
            await until(lambda: type(app.screen).__name__ == 'ChatScreen')
            await until(lambda: app.screen._skill._console_skill_candidates)
    finally:
        for code in codes:
            sys.monitoring.set_local_events(tool, code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(tool)
    assert threads['read'] and threads['build'], threads
    assert main not in threads['read'], ('skill read on the UI loop', threads)
    assert main not in threads['build'], ('trust build on the UI loop', threads)
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


def test_mounted_discovery_builds_and_reads_off_the_loop_under_the_eager_factory(
    tmp_path,
):
    """TASK-38948: the real app's mount-time skill discovery leaves the loop.

    App.run_async installs asyncio's eager task factory. The trust preparation
    required no factory at all, so in every real run it refused, and the cold
    discovery built the trust service and read every skill on the UI loop,
    inside the mount. run_test installs no factory, which hid it.
    """
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    launch = (
        "from Tests.UI.test_console_skill_context_worker import "
        "_EAGER_DISCOVERY_SCRIPT; "
        "exec(compile(_EAGER_DISCOVERY_SCRIPT, '<eager-skill-discovery>', 'exec'))"
    )
    _run(tmp_path, "skill-context", "eager", script=launch, timeout=150)


_EAGER_HELD_SETUP_SCRIPT = r"""
import asyncio, inspect, os, sys, threading
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
os.environ['PYTHON_KEYRING_BACKEND'] = 'keyring.backends.null.Keyring'
case = sys.argv[2]
assert case in {'app_closing', 'runtime_dispose'}

async def until(predicate, seconds=60):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(.01)

async def exercise():
    # What App.run_async installs; run_test alone does not.
    asyncio.get_running_loop().set_task_factory(asyncio.eager_task_factory)
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.app_skill_wiring import SkillServiceWiringMixin
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    main = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    threads = {'read': [], 'build': []}
    build_code = SkillServiceWiringMixin._build_local_skill_trust_service.__code__
    codes = {inspect.unwrap(LocalSkillsService.get_context).__code__: 'read', build_code: 'build'}

    def started(code, _offset):
        threads[codes[code]].append(threading.get_ident())
        if code is build_code and threading.get_ident() != main and not entered.is_set():
            entered.set()  # Hold the off-loop trust build open.
            release.wait(30)

    tool = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(tool, 'eager-held-skill-setup')
    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
    for code in codes:
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
    app = TldwCli()
    runtime = controller = None
    try:
        async with app.run_test(size=(170, 48)):
            await until(lambda: getattr(app, '_ui_ready', False))
            if type(app.screen).__name__ != 'ChatScreen':
                app.post_message(NavigateToScreen('chat'))
            await until(lambda: type(app.screen).__name__ == 'ChatScreen')
            await until(entered.is_set)
            runtime, controller = app.console_runtime, app.screen._skill
            assert any(
                worker.group == 'console-skill-discovery' and not worker.is_finished
                for worker in app.workers
            ), 'the held build must belong to the mount-time discovery'
            control = asyncio.create_task(
                app.on_shutdown_request() if case == 'app_closing' else runtime.dispose()
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # Long enough for an unheld dispose to finish (review r13n-rev2),
            # under the 3 s worker cancellation grace, so app_closing holds.
            await asyncio.sleep(1.0)
            assert not control.done(), 'close must wait for the held build'
            release.set()
            await asyncio.wait_for(asyncio.shield(control), timeout=30)
            await asyncio.sleep(0.3)  # Room for a late discovery to publish.
            assert runtime._disposed or case == 'app_closing'
            assert controller._console_skill_candidates == (), 'closed owner published'
            assert threads['read'] == [], ('read after the owner closed', threads)
    finally:
        release.set()
        for code in codes:
            sys.monitoring.set_local_events(tool, code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(tool)
    assert threads['build'] and main not in threads['build'], threads
    assert runtime._disposed and app.console_runtime is None
    assert threads['read'] == [], ('read after the owner closed', threads)
    assert app._local_skill_trust_service is None, 'closed owner published trust'
    assert controller._console_skill_candidates == ()
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


@pytest.mark.parametrize("case", ["app_closing", "runtime_dispose"])
def test_eager_mount_discovery_closing_during_held_trust_build_publishes_nothing(
    tmp_path, case
):
    """TASK-38948 review: the eager-factory setup still retires with its owner.

    Under App.run_async's eager factory the Console's mount-time discovery now
    builds the trust service off the loop. Closing the app, or disposing the
    Console runtime, while that build is held must wait for it, then publish
    neither the trust service nor any skill candidates, and read no skills.
    """
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    launch = (
        "from Tests.UI.test_console_skill_context_worker import "
        "_EAGER_HELD_SETUP_SCRIPT; "
        "exec(compile(_EAGER_HELD_SETUP_SCRIPT, '<eager-held-skill-setup>', 'exec'))"
    )
    _run(tmp_path, "skill-context", case, script=launch, timeout=150)
