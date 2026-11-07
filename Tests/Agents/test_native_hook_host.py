"""Disposable hosts exercise prelaunch SIGCHLD and watcher refusal."""

import asyncio
import ctypes
import os
import signal
import subprocess
import sys

import pytest

from Tests.Agents.test_native_legacy_hooks import hook_file, native_hooks  # noqa: F401
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.skipif(
        not native_workflow_qualified(), reason="native POSIX group custody required"
    ),
]


@pytest.mark.parametrize("case", ["auto_reap", "wildcard_watcher"])
def test_original_worker_refuses_unqualified_host_and_later_admits(
    case, tmp_path, request
):
    if os.environ.get("TASK12_DISPOSABLE_HOST") != case:
        root = tmp_path / "private_profile"
        root.mkdir()
        env = dict(
            os.environ, TASK12_DISPOSABLE_HOST=case, TLDW_TEST_CONFIG_ROOT=str(root)
        )
        env.pop("TLDW_TEST_CONFIG_ROOT_OWNER", None)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_original_worker_refuses_unqualified_host_and_later_admits[{case}]",
                "-W",
                "error::RuntimeWarning",
                "-W",
                "error::pytest.PytestUnraisableExceptionWarning",
                "-o",
                f"cache_dir={tmp_path / 'cache'}",
                f"--basetemp={tmp_path / 'inner'}",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            pytest.fail("disposable host test timed out")
        assert process.returncode == 0, output
        assert "1 passed" in output
        return

    h = request.getfixturevalue("native_hooks")
    marker = h.tmp_path / "payload_entered"
    h.configure(f"import pathlib; pathlib.Path({str(marker)!r}).write_text('entered')")
    old = None
    watcher = None
    loop = None
    original_loop_init = None
    libc = None
    if case == "auto_reap":

        class Sigaction(ctypes.Structure):
            if sys.platform == "darwin":
                _fields_ = [
                    ("handler", ctypes.c_void_p),
                    ("mask", ctypes.c_uint32),
                    ("flags", ctypes.c_int),
                ]
                no_cld_wait = 32
            else:
                _fields_ = [
                    ("handler", ctypes.c_void_p),
                    ("mask", ctypes.c_ulong * 16),
                    ("flags", ctypes.c_int),
                    ("restorer", ctypes.c_void_p),
                ]
                no_cld_wait = 2

        libc = ctypes.CDLL(None, use_errno=True)
        sigaction = libc.sigaction
        sigaction.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(Sigaction),
            ctypes.POINTER(Sigaction),
        )
        sigaction.restype = ctypes.c_int
        old = Sigaction()
        assert sigaction(signal.SIGCHLD, None, ctypes.byref(old)) == 0
        changed = Sigaction.from_buffer_copy(old)
        changed.flags |= Sigaction.no_cld_wait
        assert sigaction(signal.SIGCHLD, ctypes.byref(changed), None) == 0
    elif sys.version_info[:2] == (3, 14):
        from asyncio import unix_events

        original_loop_init = unix_events._UnixSelectorEventLoop.__init__

        def unqualified_loop(self, *args, **kwargs):
            original_loop_init(self, *args, **kwargs)
            self._watcher = object()

        unix_events._UnixSelectorEventLoop.__init__ = unqualified_loop
    else:
        loop = asyncio.new_event_loop()
        watcher = asyncio.FastChildWatcher()
        asyncio.get_event_loop_policy().set_child_watcher(watcher)
        watcher.attach_loop(loop)
    try:
        refused = h.engine.fire(
            "PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path)
        )
        assert refused.blocked and not marker.exists()
        snapshot = h.runtime.execution_capacity.snapshot()
        assert not snapshot.executions and snapshot.stopping_tool_workers == 0
    finally:
        if old is not None:
            assert sigaction(signal.SIGCHLD, ctypes.byref(old), None) == 0
        if watcher is not None:
            asyncio.get_event_loop_policy().set_child_watcher(None)
            loop.close()
        if original_loop_init is not None:
            unix_events._UnixSelectorEventLoop.__init__ = original_loop_init
    accepted = h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
    assert not accepted.blocked and marker.read_text() == "entered"
    assert not h.runtime.execution_capacity.snapshot().executions


@pytest.mark.parametrize(
    "change",
    [
        "callback",
        "foreign_callback",
        "session",
        "revision",
        "workspace",
        "checker",
        "cleanup_checker",
        "target_terminal",
        "process_binding",
        "popen_poll",
        "popen_setter",
        "runtime_source_removed",
        "runtime_source_replaced",
        "engine_attached_false",
        "engine_attached_removed",
    ],
)
def test_original_target_source_replacement_never_signals_or_reaps(
    change, tmp_path, request, monkeypatch
):
    import threading

    if os.environ.get("TASK12_SOURCE_REPLACEMENT") != change:
        profile = tmp_path / "source_profile"
        profile.mkdir()
        env = dict(
            os.environ,
            TASK12_SOURCE_REPLACEMENT=change,
            TLDW_TEST_CONFIG_ROOT=str(profile),
        )
        env.pop("TLDW_TEST_CONFIG_ROOT_OWNER", None)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_original_target_source_replacement_never_signals_or_reaps[{change}]",
                "-W",
                "error::RuntimeWarning",
                "-W",
                "error::pytest.PytestUnraisableExceptionWarning",
                "-o",
                f"cache_dir={tmp_path / 'cache'}",
                f"--basetemp={tmp_path / 'inner'}",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            pytest.fail("source replacement host test timed out")
        assert process.returncode == 0, output
        assert "1 passed" in output
        return

    from Tests.Agents.test_native_legacy_hooks import (
        held_script,
        opaque_busy,
        wait_until,
    )
    from tldw_chatbook.Agents import legacy_hook_ownership, run_hooks

    h = request.getfixturevalue("native_hooks")
    controller = None
    if change == "callback":
        from Tests.Chat.test_console_chat_controller import RecordingStreamingGateway
        from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

        controller = ConsoleChatController(
            store=h.store,
            provider_gateway=RecordingStreamingGateway(),
            ensure_run_hooks=h.runtime.ensure_run_hooks,
            hook_permissions_accessor=h.runtime.ensure_hook_permissions,
        )
        h.runtime.set_chat_controller(controller)
    h.configure(held_script(h.tmp_path))
    original_provider = h.engine._cwd_provider
    original_issuer = h.runtime._legacy_hooks_source
    original_attached = h.engine._native_attached
    original_witness = h.runtime._run_hooks_construction_witness
    original_process = None
    original_checker = legacy_hook_ownership._EventSource.current_memory
    original_cleanup_checker = legacy_hook_ownership._CleanupSource.current
    original_terminal = legacy_hook_ownership._TargetCustody.process_terminal
    original_popen_poll = subprocess.Popen._internal_poll
    original_popen_setter = vars(subprocess.Popen).get("__setattr__")
    original_revision, original_workspace = (
        h.session.conversation_binding_revision,
        h.session.workspace_id,
    )
    checker_calls = []
    real_signal, real_wait = os.killpg, os.waitpid
    effects = []
    changed = threading.Event()
    child = [None]

    def group(pid, sig):
        if changed.is_set() and pid == child[0]:
            effects.append(("signal", sig))
        return real_signal(pid, sig)

    def wait(pid, flags):
        if changed.is_set() and pid == child[0]:
            effects.append(("wait", flags))
        return real_wait(pid, flags)

    monkeypatch.setattr(run_hooks.os, "killpg", group)
    monkeypatch.setattr(run_hooks.os, "waitpid", wait)
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        wait_until((h.tmp_path / "started").exists)
        child[0] = int((h.tmp_path / "started").read_text())
        event = next(iter(original_issuer._events))
        original_process = event.targets[0]._cleanup_process
        if change == "runtime_source_removed":
            del h.runtime._legacy_hooks_source
        elif change == "runtime_source_replaced":

            class ForeignSource:
                def __getattribute__(self, name):
                    checker_calls.append(name)
                    raise AssertionError("foreign source getter invoked")

                def __eq__(self, other):
                    checker_calls.append("equality")
                    raise AssertionError("foreign source equality invoked")

            h.runtime._legacy_hooks_source = ForeignSource()
        elif change == "engine_attached_false":
            h.engine._native_attached = False
        elif change == "engine_attached_removed":
            del h.engine._native_attached
        elif change == "callback":
            h.engine._cwd_provider = lambda: str(h.tmp_path)
        elif change == "foreign_callback":

            class ForeignCallback:
                def __eq__(self, other):
                    checker_calls.append(True)
                    raise AssertionError("foreign equality invoked")

            h.engine._cwd_provider = ForeignCallback()
        elif change == "session":
            assert h.store._sessions.pop(h.session.id) is h.session
        elif change == "revision":
            h.session.conversation_binding_revision += 1
        elif change == "workspace":
            h.session.workspace_id = "changed-workspace"
        elif change == "checker":
            legacy_hook_ownership._EventSource.current_memory = lambda *args, **kwargs: (
                checker_calls.append(True) or True
            )
        elif change == "cleanup_checker":
            legacy_hook_ownership._CleanupSource.current = lambda *args, **kwargs: (
                checker_calls.append(True) or True
            )
        elif change == "target_terminal":
            legacy_hook_ownership._TargetCustody.process_terminal = (
                lambda *args, **kwargs: checker_calls.append(True)
            )
        elif change == "process_binding":
            event = next(iter(h.engine._native_source._events))
            event.targets[0]._cleanup_process = object()
        elif change == "popen_poll":
            subprocess.Popen._internal_poll = lambda *args, **kwargs: (
                checker_calls.append(True)
            )
        else:
            subprocess.Popen.__setattr__ = lambda *args, **kwargs: checker_calls.append(
                True
            )
        changed.set()
        (h.tmp_path / "release").touch()
        worker.join(5)
        assert not worker.is_alive()
        assert not effects, effects
        assert not checker_calls
        assert h.runtime._run_hooks_construction_witness is original_witness
        if change in {
            "runtime_source_removed",
            "runtime_source_replaced",
            "engine_attached_false",
            "engine_attached_removed",
        }:
            assert event.targets[0]._retained_process is original_process
            assert event.targets[0].returned and not event.targets[0].proven
        assert outcomes[0].blocked
        assert h.runtime.execution_capacity.snapshot().stopping_tool_workers == 1
        assert opaque_busy(h.runtime)
    finally:
        (h.tmp_path / "release").touch()
        worker.join(5)
        h.engine._cwd_provider = original_provider
        h.runtime._legacy_hooks_source = original_issuer
        h.engine._native_attached = original_attached
        h.store._sessions[h.session.id] = h.session
        h.session.conversation_binding_revision = original_revision
        h.session.workspace_id = original_workspace
        legacy_hook_ownership._EventSource.current_memory = original_checker
        legacy_hook_ownership._CleanupSource.current = original_cleanup_checker
        legacy_hook_ownership._TargetCustody.process_terminal = original_terminal
        subprocess.Popen._internal_poll = original_popen_poll
        if original_popen_setter is None:
            if "__setattr__" in vars(subprocess.Popen):
                del subprocess.Popen.__setattr__
        else:
            subprocess.Popen.__setattr__ = original_popen_setter
        if (
            change
            in {
                "runtime_source_removed",
                "runtime_source_replaced",
                "engine_attached_false",
                "engine_attached_removed",
            }
            and original_process is not None
            and not original_process._child_created
        ):
            changed.clear()
            waited, _ = real_wait(original_process.pid, 0)
            assert waited == original_process.pid  # fixture resource cleanup only

    if controller is not None:
        event = next(iter(h.engine._native_source._events))
        process = event.targets[0]._cleanup_process
        changed.clear()
        waited, _ = real_wait(process.pid, 0)
        assert waited == process.pid  # fixture resource return; not physical proof
        revision = controller.lifecycle_impact(session_id=h.session.id).revision
        with pytest.raises(RuntimeError):
            asyncio.run(
                h.runtime.close_session(h.session.id, expected_revision=revision)
            )
        state = next(iter(controller._session_close_states.values()))
        assert state[0].session_id == h.session.id
        assert h.store._sessions[h.session.id] is h.session
        assert h.session.id in h.runtime._admission_fenced_sessions
        assert h.runtime.execution_capacity.snapshot().stopping_tool_workers == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close", [False, True], ids=["normal", "cancelled"])
async def test_real_chat_close_waits_for_original_legacy_child(
    tmp_path, request, cancel_close
):
    if os.environ.get("TASK12_SESSION_CLOSE_HOST") != "1":
        profile = tmp_path / "session_close_profile"
        profile.mkdir()
        env = dict(
            os.environ,
            TASK12_SESSION_CLOSE_HOST="1",
            TLDW_TEST_CONFIG_ROOT=str(profile),
        )
        env.pop("TLDW_TEST_CONFIG_ROOT_OWNER", None)
        process = subprocess.Popen(  # noqa: ASYNC220 -- isolated real host, bounded below
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_real_chat_close_waits_for_original_legacy_child[{'cancelled' if cancel_close else 'normal'}]",
                "-W",
                "error::RuntimeWarning",
                "-W",
                "error::pytest.PytestUnraisableExceptionWarning",
                "-o",
                f"cache_dir={tmp_path / 'cache'}",
                f"--basetemp={tmp_path / 'inner'}",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            pytest.fail("session close host test timed out")
        assert process.returncode == 0, output
        assert "1 passed" in output
        return

    from Tests.Agents.test_native_legacy_hooks import opaque_busy, wait_until
    from Tests.Chat.test_console_chat_controller import RecordingStreamingGateway
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    h = request.getfixturevalue("native_hooks")
    controller = ConsoleChatController(
        store=h.store,
        provider_gateway=RecordingStreamingGateway(),
        ensure_run_hooks=h.runtime.ensure_run_hooks,
        hook_permissions_accessor=h.runtime.ensure_hook_permissions,
    )
    h.runtime.set_chat_controller(controller)
    started, finished = h.tmp_path / "real_close_child", h.tmp_path / "real_close_done"
    h.configure(
        "import os,pathlib,time; "
        f"pathlib.Path({str(started)!r}).write_text(str(os.getpid())); "
        "time.sleep(3.5); "
        f"pathlib.Path({str(finished)!r}).write_text('done')",
        event="Stop",
        timeout=5,
    )
    h.engine.notify("Stop", session_id=h.session.id, cwd=str(h.tmp_path))
    await asyncio.to_thread(wait_until, started.exists)
    pid = int(started.read_text())
    event = next(iter(h.engine._native_source._events))
    process = event.targets[0]._cleanup_process
    assert process.pid == pid
    close_task = None
    try:
        revision = controller.lifecycle_impact(session_id=h.session.id).revision
        close_task = asyncio.create_task(
            h.runtime.close_session(h.session.id, expected_revision=revision)
        )
        if cancel_close:
            await asyncio.sleep(0.2)
            close_task.cancel()
            await asyncio.sleep(0.1)
            close_task.cancel()
            await asyncio.sleep(1.75)
        else:
            await asyncio.sleep(2.05)
        assert not close_task.done(), (
            "close removed the session before original legacy retirement"
        )
        assert h.store._sessions.get(h.session.id) is h.session
        assert not finished.exists(), (
            "original child did not outlive the normal close grace"
        )
        assert opaque_busy(h.runtime)
        await asyncio.to_thread(wait_until, finished.exists, 4)
        if cancel_close:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(close_task), 4)
        else:
            await asyncio.wait_for(asyncio.shield(close_task), 4)
        assert h.store._sessions.get(h.session.id) is None
        assert not h.runtime._legacy_hooks_source.cleanup_pending
        assert not h.runtime.execution_capacity.snapshot().executions
    finally:
        await asyncio.to_thread(wait_until, finished.exists, 4)
        if not process._child_created and process.returncode is None:
            # Fixture owns this still-unconsumed child; resource cleanup is not proof.
            waited, _ = os.waitpid(pid, 0)  # noqa: ASYNC222 -- fixture child has exited
            assert waited == pid
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await h.runtime.dispose()


@pytest.mark.parametrize("lifecycle", ["unpublish", "permissions_close"])
def test_original_cleanup_survives_expected_permission_lifecycle(request, lifecycle):
    import threading

    from Tests.Agents.test_native_legacy_hooks import (
        held_script,
        opaque_busy,
        wait_until,
    )

    h = request.getfixturevalue("native_hooks")
    h.configure(held_script(h.tmp_path))
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        wait_until((h.tmp_path / "started").exists)
        event = next(iter(h.engine._native_source._events))
        target = event.targets[0]
        if lifecycle == "unpublish":
            with h.owner._cache_lock:
                assert h.owner._published_targets
                h.owner._published_targets = ()
        else:
            h.owner.close()
        (h.tmp_path / "release").touch()
        worker.join(5)
        assert not worker.is_alive() and len(outcomes) == 1 and not outcomes[0].blocked
        assert target.returned and target.proven
        assert not h.runtime.execution_capacity.snapshot().executions
        assert not opaque_busy(h.runtime)
    finally:
        (h.tmp_path / "release").touch()
        worker.join(5)
