"""Exact Runtime helper controls and unchanged non-chat fixture regression."""

import pytest
from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert route == 'prepared_close_runtime'
selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
root = Path(os.environ['XDG_DATA_HOME']).absolute()
selector.write_text('[general]\nusers_name="qualifier"\n[paths]\ndata_dir="'
                    + root.as_posix() + '"\n', encoding='utf-8')
selector.chmod(0o600)

async def wait_entered(entered):
    deadline = asyncio.get_running_loop().time() + 10
    while not entered.is_set():
        assert asyncio.get_running_loop().time() < deadline, 'original policy read not reached'
        await asyncio.sleep(.01)


def main():
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    original = (ConsoleRuntime.dispose, ConsoleRuntime._watch_canvas_policy,
                ConsoleRuntime._canvas_enabled)
    codes = tuple(function.__code__ for function in original)
    globals_ = tuple(function.__globals__ for function in original)
    paths = [Path(sys.modules[ConsoleRuntime.__module__].__file__)]

    if outcome == 'fixture_retirement':
        from Tests.UI.test_console_session_tab_close import _pending_close_app
        from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
        paths += [Path(sys.modules[_pending_close_app.__module__].__file__)]
        body = _pending_close_app.__wrapped__
        body_anchor = (body.__code__, body.__globals__, body.__defaults__, body.__kwdefaults__)
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        async def fixture():
            runtime = None
            try:
                async with _pending_close_app(None, 'approval') as app:
                    runtime = app.console_runtime
                    watcher = runtime._canvas_policy_watch_task
                    assert type(runtime) is ConsoleRuntime and runtime.app is app
                    assert type(watcher) is asyncio.Task and watcher.get_loop() is asyncio.get_running_loop()
                    assert not watcher.done() and not runtime._disposed
                    # Same actual fixture route; no harness, request or source guard substituted.
                    await asyncio.sleep(0)
                assert runtime._disposed and watcher.done(), 'original non-chat fixture Runtime was not retired'
                assert runtime._canvas_policy_watch_task is None
                assert runtime._canvas_policy_read_task is None
            finally:
                if runtime is not None:
                    await ConsoleRuntime.dispose(runtime)  # Exact test-owned cleanup, after behavior assertion.
                drain_active_service_patches()
                drain_created_dirs()
        asyncio.run(fixture())
        assert (body.__code__, body.__globals__, body.__defaults__, body.__kwdefaults__) == body_anchor
    else:
        from Tests.UI import _prepared_close_runtime_owner as module
        paths += [Path(module.__file__)]
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        def fresh():
            app = SimpleNamespace(chachanotes_db=None, app_config={}, conversation_local_marks_service=None)
            runtime = ConsoleRuntime(app)
            app.console_runtime = runtime
            return app, runtime
        async def run():
            app, runtime = fresh()
            watcher = runtime._canvas_policy_watch_task
            assert type(watcher) is asyncio.Task and not watcher.done()
            owner, other, tool = None, None, None
            release, entered = threading.Event(), threading.Event()
            invalid = []
            try:
                if outcome in {'wrong_type', 'foreign', 'built'}:
                    if outcome == 'wrong_type':
                        app.console_runtime = object()
                    elif outcome == 'foreign':
                        _, other = fresh()
                        app.console_runtime = other
                    else:
                        assert runtime.ensure_chat_store() is not None
                    try:
                        module.PreparedCloseRuntimeOwner(app)
                    except RuntimeError as error:
                        assert error.args == ('prepared_close_runtime_not_fresh_owned',)
                    else:
                        raise AssertionError('non-fresh or foreign Runtime was adopted')
                    assert not runtime._disposed and (other is None or not other._disposed)
                    return
                owner = module.PreparedCloseRuntimeOwner(app)
                assert owner.runtime is runtime and owner.watcher is watcher
                assert runtime._agent_runs_db is None and runtime.chat_store is None
                if outcome == 'changed':
                    app.console_runtime = object()
                    try:
                        await owner.dispose_runtime()
                    except RuntimeError as error:
                        assert error.args == ('prepared_close_runtime_owner_changed',)
                    else:
                        raise AssertionError('changed app Runtime was not refused')
                    assert owner.dispose_task is None and not runtime._disposed
                    return
                if outcome == 'wrong_thread':
                    def foreign_thread():
                        assert threading.current_thread() is not owner.creator
                        try:
                            asyncio.run(owner.dispose_runtime())
                        except RuntimeError as error:
                            assert error.args == ('prepared_close_runtime_wrong_thread',)
                        else:
                            raise AssertionError('foreign thread was admitted')
                    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='runtime-owner-foreign') as executor:
                        await asyncio.wrap_future(executor.submit(foreign_thread))
                    assert owner.dispose_task is None and not runtime._disposed
                    return
                if outcome == 'repeated_cancel':
                    code = ConsoleRuntime._canvas_enabled.__code__
                    owner_thread = threading.current_thread()
                    held = False
                    def started(code_, offset):
                        nonlocal held
                        frame = sys._getframe(1)
                        if frame.f_locals.get('self') is not runtime or threading.current_thread() is owner_thread or held:
                            return
                        assert frame.f_code is code
                        held = True
                        entered.set()
                        if not release.wait(10):
                            invalid.append('original_read_release_timeout')
                    monitor = sys.monitoring
                    tool = next(slot for slot in range(6) if monitor.get_tool(slot) is None)
                    monitor.use_tool_id(tool, 'non-chat-runtime-owner-control')
                    monitor.register_callback(tool, monitor.events.PY_START, started)
                    monitor.set_local_events(tool, code, monitor.events.PY_START)
                    assert monitor.get_events(tool) == 0
                    await wait_entered(entered)
                    reader = runtime._canvas_policy_read_task
                    assert reader is not None and not reader.done()
                    caller = asyncio.create_task(owner.dispose_runtime())
                    await asyncio.sleep(0)
                    assert owner.dispose_task is not None
                    for _ in range(2):
                        caller.cancel()
                        await asyncio.sleep(0)
                        assert not caller.done() and not owner.dispose_task.done()
                        assert runtime._canvas_policy_read_task is reader and not reader.done()
                    assert caller.cancelling() == 2
                    release.set()
                    try:
                        await caller
                    except asyncio.CancelledError:
                        pass
                    else:
                        raise AssertionError('repeatedly cancelled awaiter completed normally')
                    assert reader.done() and owner.dispose_task.done() and not owner.dispose_task.cancelled()
                else:
                    assert outcome == 'fresh'
                    await owner.dispose_runtime()
                assert not invalid and owner.runtime_terminal and runtime._disposed and watcher.done()
                assert runtime._canvas_policy_watch_task is None and runtime._canvas_policy_read_task is None
                assert runtime._agent_runs_db is None and runtime.chat_store is None
                same_task = owner.dispose_task
                await owner.dispose_runtime()
                assert owner.dispose_task is same_task and owner.runtime_terminal
            finally:
                release.set()
                if tool is not None:
                    monitor.set_local_events(tool, code, 0)
                    monitor.register_callback(tool, monitor.events.PY_START, None)
                    monitor.free_tool_id(tool)
                    assert monitor.get_tool(tool) is None and monitor.get_events(tool) == 0
                app.console_runtime = runtime
                if owner is not None and owner.dispose_task is not None:
                    await asyncio.shield(owner.dispose_task)
                await ConsoleRuntime.dispose(runtime)
                if other is not None:
                    await ConsoleRuntime.dispose(other)
        if outcome == 'wrong_loop':
            first, second = asyncio.new_event_loop(), asyncio.new_event_loop()
            owner = None
            async def capture():
                app, runtime = fresh()
                return module.PreparedCloseRuntimeOwner(app), runtime
            async def refuse():
                try:
                    await owner.dispose_runtime()
                except RuntimeError as error:
                    assert error.args == ('prepared_close_runtime_wrong_loop',)
                else:
                    raise AssertionError('foreign loop was admitted')
                assert owner.dispose_task is None and not runtime._disposed
            try:
                owner, runtime = first.run_until_complete(capture())
                second.run_until_complete(refuse())
                first.run_until_complete(owner.dispose_runtime())
                assert owner.runtime_terminal and owner.watcher.done()
            finally:
                if owner is not None:
                    first.run_until_complete(ConsoleRuntime.dispose(runtime))
                for loop in (first, second):
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(loop.shutdown_default_executor())
                    loop.close()
        else:
            asyncio.run(run())
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == hashes
    assert tuple(function.__code__ for function in original) == codes
    assert all(function.__globals__ is namespace for function, namespace in zip(original, globals_))
    print('retired and reopened')

with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize(
    "outcome",
    [
        "fixture_retirement",
        "fresh",
        "repeated_cancel",
        "wrong_type",
        "foreign",
        "built",
        "changed",
        "wrong_thread",
        "wrong_loop",
    ],
)
def test_prepared_close_runtime_exact_owner(tmp_path, outcome):
    _run(tmp_path, "prepared_close_runtime", outcome, script=_SCRIPT)
