"""Required policy polling must leave the UI and backup coordinator runnable."""

import ast
import asyncio
import sys
import threading
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_canvas_policy_maintenance import _SCRIPT
from Tests.private_profile import private_profile_test
from Tests.Backup_Recovery.test_home_citation_retirement import _run


_POLICY_SCRIPT_GUARD = r"""
import importlib.util as __policy_imports
import os as __policy_os
import sys as __policy_sys
from pathlib import Path as __policy_Path
assert not any(name == "tldw_chatbook" or name.startswith("tldw_chatbook.") or name == "tldw_profile_core" or name.startswith("tldw_profile_core.") for name in __policy_sys.modules), "product imported before raw-script guards"
__policy_repo = __policy_Path(__SOURCE_REPO__)
assert __policy_os.environ.get("TLDW_TEST_REAL_HOME"), "original home export missing"
for __policy_name, __policy_expected in (
    ("Tests", __policy_repo / "Tests/__init__.py"),
    ("tldw_chatbook", __policy_repo / "tldw_chatbook/__init__.py"),
    ("tldw_profile_core", __policy_repo / "packages/tldw_profile_core/src/tldw_profile_core/__init__.py"),
):
    assert __policy_Path(__policy_imports.find_spec(__policy_name).origin).resolve() == __policy_expected.resolve(), __policy_name
from Tests import real_profile_guard as __policy_profile
__policy_profile.install()
assert __policy_profile._real_home() == __policy_Path(__policy_os.environ["TLDW_TEST_REAL_HOME"])
assert __policy_profile._real_home().resolve() != __policy_Path.home().resolve()
from Tests import network_guard as __policy_network
__policy_network.install()
assert not __policy_network.is_allowed() and not __policy_network.is_loopback_only()
print("policy-script-guards-installed", flush=True)
"""

_POLICY_SCRIPT_GUARD_END = r"""
for __policy_name, __policy_expected in (
    ("tldw_chatbook.Chat.console_runtime", __policy_repo / "tldw_chatbook/Chat/console_runtime.py"),
    ("tldw_chatbook.Canvas.config_policy", __policy_repo / "tldw_chatbook/Canvas/config_policy.py"),
 ):
    assert __policy_Path(__policy_sys.modules[__policy_name].__file__).resolve() == __policy_expected.resolve(), __policy_name
__policy_core = __policy_sys.modules.get("tldw_profile_core")
if __policy_core is None:
    print("policy-script-core-not-loaded", flush=True)
else:
    assert __policy_Path(__policy_core.__file__).resolve() == (__policy_repo / "packages/tldw_profile_core/src/tldw_profile_core/__init__.py").resolve()
__policy_profile_refusals = __policy_profile.take_violations()
assert not __policy_profile_refusals, __policy_profile_refusals
assert not __policy_network.blocked_attempts(), __policy_network.blocked_attempts()
print("policy-script-guards-clean", flush=True)
"""


def _protected_policy_script(script: str) -> str:
    """Protect only this module's original installed_package=None subprocesses."""
    nodes = ast.parse(script).body
    offset = 0
    if (
        nodes
        and isinstance(nodes[0], ast.Expr)
        and isinstance(nodes[0].value, ast.Constant)
        and isinstance(nodes[0].value.value, str)
    ):
        offset = nodes.pop(0).end_lineno
    for node in nodes:
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
            break
        offset = node.end_lineno
    source = Path(__file__).resolve().parents[2].as_posix()
    prefix = _POLICY_SCRIPT_GUARD.replace("__SOURCE_REPO__", repr(source))
    lines = script.splitlines(keepends=True)
    return (
        "".join(lines[:offset])
        + "\n"
        + prefix
        + "\n"
        + "".join(lines[offset:])
        + _POLICY_SCRIPT_GUARD_END
    )


_RESPONSIVE = (
    _SCRIPT.split("async def main():")[0]
    + r"""
import threading
held=threading.Event();release=threading.Event();entered=threading.Event()
watchdog_fired=[];writer_errors=[]
def writer():
 try:
  with config._config_write_lock(selected):
   held.set()
   assert release.wait(20)
 except BaseException as error:writer_errors.append(type(error).__name__)
def unblock():
 watchdog_fired.append(True);release.set()
original=runtime._canvas_enabled_reader
def read():
 entered.set()
 return original()
runtime._canvas_enabled_reader=read
worker=threading.Thread(target=writer,daemon=True)
async def main():
 worker.start();assert await asyncio.to_thread(held.wait,10)
 watchdog=threading.Timer(10 if sys.platform=='win32' else 3,unblock)
 watchdog.start()
 try:
  runtime.start_async_lifecycles()
  assert await asyncio.to_thread(entered.wait,10)
  # This callback cannot run if the watcher waits on the native config lock
  # on this event loop. Only the independent watchdog can release it then.
  release.set()
  assert not watchdog_fired,'native policy polling blocked the UI event loop'
 finally:
  release.set();watchdog.cancel()
  await asyncio.to_thread(worker.join,10)
  await runtime.dispose()
 assert not worker.is_alive() and not writer_errors
 assert not storage._raw_operations
asyncio.run(main())
print('retired and reopened')
"""
)


def test_native_policy_poll_does_not_block_ui_while_config_writer_is_active(tmp_path):
    _run(
        tmp_path,
        "responsive",
        "canvas-policy-worker",
        script=_protected_policy_script(_RESPONSIVE),
        timeout=90 if sys.platform == "win32" else 25,
    )


_RETAINED = (
    _RESPONSIVE.split("async def main():")[0]
    + r"""
# A warm native read can return while the config writer is still held.
def read():
 entered.set()
 enabled=original()
 assert release.wait(20)
 return enabled
runtime._canvas_enabled_reader=read
async def main():
 worker.start();assert await asyncio.to_thread(held.wait,10)
 watchdog=threading.Timer(15,unblock);watchdog.start()
 caller=None
 try:
  runtime.start_async_lifecycles()
  watcher=runtime._canvas_policy_watch_task
  assert await asyncio.to_thread(entered.wait,10)
  assert not watchdog_fired,'native policy polling blocked the UI event loop'
  if case=='dispose_cancel':
   caller=asyncio.create_task(runtime.dispose())
   await asyncio.sleep(.05)
   assert not caller.done(),'dispose abandoned an accepted native read'
   caller.cancel()
   try:await caller
   except asyncio.CancelledError:pass
   else:raise AssertionError('dispose cancellation was swallowed')
  else:
   runtime._canvas_maintenance_close_admission()
   if case=='watcher_cancel':
    watcher.cancel()
    try:await watcher
    except asyncio.CancelledError:pass
    else:raise AssertionError('watcher cancellation was swallowed')
   if case=='drain_cancel':
    caller=asyncio.create_task(runtime._canvas_maintenance_drain(time.monotonic()+10))
    await asyncio.sleep(0);caller.cancel()
    try:await caller
    except asyncio.CancelledError:pass
    else:raise AssertionError('drain cancellation was swallowed')
  runtime._canvas_maintenance_close_admission()
  assert not await runtime._canvas_maintenance_drain(time.monotonic()),'accepted native read was not retained'
  try:runtime._canvas_maintenance_resume()
  except RecoveryRequired as error:assert error.args==('runtime_work_not_settled',)
  else:raise AssertionError('backup resumed before native policy read completed')
 finally:
  release.set();watchdog.cancel()
  await asyncio.to_thread(worker.join,10)
  runtime._canvas_maintenance_close_admission()
  assert await runtime._canvas_maintenance_drain(time.monotonic()+10)
  if not runtime._disposed:
   runtime._canvas_maintenance_resume()
   assert runtime.canvas_enabled() and runtime._canvas_native_authority is authority
  await runtime.dispose()
 assert not worker.is_alive() and not writer_errors
 assert not storage._raw_operations
asyncio.run(main())
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "case", ["maintenance", "watcher_cancel", "drain_cancel", "dispose_cancel"]
)
def test_backup_and_disposal_retain_accepted_native_policy_read(tmp_path, case):
    _run(
        tmp_path,
        case,
        "canvas-policy-worker-retained",
        script=_protected_policy_script(_RETAINED),
        timeout=90 if sys.platform == "win32" else 25,
    )


_RESTART = (
    _RESPONSIVE.split("async def main():")[0]
    + r"""
reads=[]
def read():
 reads.append(threading.get_ident());entered.set()
 enabled=original()
 assert release.wait(20)
 return enabled
runtime._canvas_enabled_reader=read
async def main():
 worker.start();assert await asyncio.to_thread(held.wait,10)
 watchdog=threading.Timer(15,unblock);watchdog.start()
 try:
  runtime.start_async_lifecycles()
  watcher=runtime._canvas_policy_watch_task
  assert await asyncio.to_thread(entered.wait,10)
  assert not watchdog_fired,'native policy polling blocked the UI event loop'
  reader=runtime._canvas_policy_read_task
  watcher.cancel()
  try:await watcher
  except asyncio.CancelledError:pass
  else:raise AssertionError('watcher cancellation was swallowed')
  runtime.start_async_lifecycles()
  successor=runtime._canvas_policy_watch_task
  runtime.start_async_lifecycles()
  await asyncio.sleep(0)
  assert successor is not watcher and runtime._canvas_policy_watch_task is successor
  assert runtime._canvas_policy_read_task is reader and len(reads)==1
  release.set();await asyncio.to_thread(worker.join,10)
  deadline=time.monotonic()+10
  while len(reads)<2 and time.monotonic()<deadline:await asyncio.sleep(.01)
  assert len(reads)>=2,'restarted watcher stopped observing native policy'
  assert not runtime._canvas_disabled_latched
 finally:
  release.set();watchdog.cancel()
  await asyncio.to_thread(worker.join,10)
  await runtime.dispose()
 assert not worker.is_alive() and not writer_errors
 assert not storage._raw_operations
asyncio.run(main())
print('retired and reopened')
"""
)


def test_cancelled_watcher_reuses_pending_native_read_when_restarted(tmp_path):
    _run(
        tmp_path,
        "restart",
        "canvas-policy-worker-restarted",
        script=_protected_policy_script(_RESTART),
        timeout=90 if sys.platform == "win32" else 25,
    )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("held_lock", ["rebuild", "file"])
@private_profile_test
async def test_stock_canvas_policy_poll_declines_busy_config_lock(request, held_lock):
    """A stock background poll yields without joining another native config reader."""
    from types import SimpleNamespace

    from tldw_chatbook import config
    from tldw_chatbook.Canvas.config_policy import get_canvas_execution_enabled
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    path = config._get_effective_config_path()
    original = path.read_bytes()
    # This is the shipping constructor-before-loop lifecycle, on a real worker.
    runtime = await asyncio.to_thread(
        ConsoleRuntime, SimpleNamespace(chachanotes_db=None)
    )
    assert runtime._canvas_enabled_reader is get_canvas_execution_enabled
    assert runtime._canvas_policy_watch_task is None
    assert not runtime.canvas_disabled()
    held, release, expired = threading.Event(), threading.Event(), threading.Event()
    writer_errors = []
    lock = (
        config._settings_rebuild_lock()
        if held_lock == "rebuild"
        else config._config_file_lock()
    )

    def hold():
        try:
            with lock:
                held.set()
                if not release.wait(20):
                    expired.set()
        except BaseException as error:
            writer_errors.append(type(error).__name__)

    writer = threading.Thread(target=hold, name="stock-canvas-lock-holder")
    try:
        path.write_bytes(original + b"\n[unterminated\n")
        await asyncio.to_thread(config.load_settings, True)
        assert config._settings_cache_hit(path) is None
        writer.start()
        assert await asyncio.to_thread(held.wait, 5), "real holder did not enter"
        runtime.start_async_lifecycles()
        # The watcher creates its actual read Task before its worker advances.
        await asyncio.sleep(0)
        reader = runtime._canvas_policy_read_task
        assert reader is not None, "actual policy read Task was not retained"
        result = await asyncio.wait_for(asyncio.shield(reader), timeout=5)
        assert result is None, "busy poll did not report temporary unavailable"
        assert not release.is_set() and not runtime.canvas_disabled()
        if held_lock == "file":
            acquired = []

            def probe_rebuild():
                rebuild = config._settings_rebuild_lock()
                owned = rebuild.acquire(blocking=False)
                acquired.append(owned)
                if owned:
                    rebuild.release()

            probe = threading.Thread(target=probe_rebuild)
            probe.start()
            await asyncio.to_thread(probe.join, 5)
            assert not probe.is_alive() and acquired == [
                True
            ], "FILE refusal retained its partial REBUILD acquisition"
    finally:
        release.set()
        if writer.ident is not None:
            await asyncio.to_thread(writer.join, 5)
        try:
            await runtime.dispose()
        finally:
            path.write_bytes(original)
            await asyncio.to_thread(config.load_settings, True)
    assert not writer.is_alive() and not expired.is_set() and not writer_errors


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@private_profile_test
async def test_stock_canvas_poll_leaves_the_retired_read_callback_handoff(
    request, monkeypatch
):
    """A canonical helper declines until one guarded CORE attempt consumes it.

    Callback delivery is controlled here; the original UI regressions cover
    actual timer delivery, live pushes and idle convergence.
    """
    from contextlib import ExitStack

    from tldw_chatbook import config
    from tldw_chatbook.Canvas import config_policy
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.UI.Console_Modules import config_sync
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from Tests.UI.test_console_projection_reads_off_loop import (
        _UiLoopStarts,
        _fresh_display,
    )
    from Tests.UI.test_console_send_acknowledgement import (
        build,
        eager_tasks,
        ready_console,
        until,
    )

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            projection = await _fresh_display(console)
            # Attach can leave a refresh queued even when no FULL is running.
            # Drain that refresh and its coalesced replay before taking over delivery.
            async with asyncio.timeout(30):
                await until(lambda: console._console_attach_reconciled, timeout=30)
                await console.wait_for_refresh()
                await until(
                    lambda: not (
                        console._console_sync_in_progress
                        or getattr(console, "_console_control_bar_sync_scheduled", False)
                        or getattr(console, "_console_control_bar_replay_whole_sync", False)
                    ),
                    timeout=30,
                )
            runtime = console._console_runtime()
            assert type(runtime) is ConsoleRuntime
            assert (
                runtime._canvas_enabled_reader
                is config_policy._CANONICAL_EXECUTION_READER
            )
            assert not runtime.canvas_disabled()
            previous_proof = projection._display_proof
            projection.at = 0.0
            held, release, expired = (
                threading.Event(),
                threading.Event(),
                threading.Event(),
            )
            errors, queued, warmed, outcomes = [], [], [], []
            preparation_entered, begin_warm, preparation_returned, deliver = (
                asyncio.Event(),
                asyncio.Event(),
                asyncio.Event(),
                asyncio.Event(),
            )
            real_prepare = config_sync.prepare_console_presentation_sync

            def hold_rebuild():
                try:
                    with config._settings_rebuild_lock():
                        held.set()
                        if not release.wait(20):
                            expired.set()
                except BaseException as error:
                    errors.append(type(error).__name__)

            def queue_control_bar(*, delayed=False):
                queued.append(delayed)  # Manual delivery, not timer bookkeeping proof.

            async def preparation_boundary(screen):
                assert screen is console
                preparation_entered.set()
                await begin_warm.wait()
                # Current owned registration is legitimate fixture setup; the
                # real early CORE refusal and its original replay remain below.
                assert config_sync._owe_console_config_sync(screen, "core") is not None
                result = await real_prepare(screen)
                warmed.append(result)
                assert result is True
                assert projection._display_proof is not None
                assert projection._display_proof is not previous_proof
                assert projection._display_proof.value is projection.value
                assert projection._display_proof.reader is projection.read_current
                assert (
                    projection._display_proof.owner
                    == projection.key
                    == projection._key()
                )
                assert not projection.pending and not projection._reads
                core = screen._console_config_handoffs["core"]
                token = core.token
                assert token is not None and token.active and core.receipt is not None
                # No await between the original warm receipt and this probe/
                # callback. Nonblocking real locks keep unrelated producers out
                # without making the UI loop wait or changing receipt lifetime.
                with ExitStack() as owned:
                    for lock in (
                        config._settings_rebuild_lock(),
                        config._config_file_lock(),
                    ):
                        assert lock.acquire(
                            blocking=False
                        ), "fixture found a competing native config owner"
                        owned.callback(lock.release)
                    assert (
                        config_policy._poll_runtime_canvas_enabled(
                            runtime, ConsoleRuntime
                        )
                        is None
                    )
                    attempts, pushes = (
                        starts.counts["core_attempt"],
                        starts.counts["core_push"],
                    )
                    screen._run_coalesced_control_bar_sync()
                    attempted = starts.counts["core_attempt"] - attempts
                    pushed = starts.counts["core_push"] - pushes
                    assert (
                        attempted == 1
                    ), "owned callback did not make its one guarded attempt"
                    assert (
                        not token.active
                    ), "one CORE attempt retained its poll handoff"
                    assert 0 <= pushed <= 1, "one attempt duplicated the live CORE push"
                outcomes.append(
                    {
                        "actual_guarded_attempts": attempted,
                        "actual_live_pushes": pushed,
                        "saved_core_token_retired": not token.active,
                        "canonical_helper_declined": True,
                        "original_warm_true": result,
                        "new_checked_proof": True,
                        "physical_read_retired": True,
                    }
                )
                preparation_returned.set()
                await deliver.wait()
                return result

            monkeypatch.setattr(
                console, "_request_console_control_bar_sync", queue_control_bar
            )
            monkeypatch.setattr(
                config_sync, "prepare_console_presentation_sync", preparation_boundary
            )
            holder = threading.Thread(
                target=hold_rebuild, name="handoff-rebuild-holder"
            )
            worker = None
            starts = _UiLoopStarts(
                core_push=(
                    ChatScreen._sync_console_chat_core_state.__code__,
                    config_sync.sync_console_core_state.__code__,
                ),
                core_attempt=(
                    config_sync.sync_console_core_state.__code__,
                    config_sync.run_coalesced_control_bar_sync.__code__,
                ),
            )
            try:
                holder.start()
                assert await asyncio.to_thread(held.wait, 5)
                with starts:
                    worker = console.run_worker(
                        console._sync_native_console_chat_ui(),
                        exclusive=True,
                        group="console-sync",
                    )
                    await until(preparation_entered.is_set, timeout=30)
                    assert console._console_sync_in_progress
                    assert console._console_control_bar_replay_whole_sync and queued
                    assert starts.counts["core_push"] == 0
                    release.set()
                    await asyncio.to_thread(holder.join, 5)
                    assert not holder.is_alive() and not expired.is_set() and not errors
                    assert warmed == []
                    request.node.user_properties.append(
                        ("pre_warm_read_pending", projection.pending)
                    )
                    console._run_coalesced_control_bar_sync()
                    assert starts.counts == {
                        "core_push": 0,
                        "core_attempt": 0,
                    }, "cold-only owed CORE ran before original warmTrue"
                    request.node.user_properties.append(("cold_only_pushes", 0))
                    begin_warm.set()
                    await until(preparation_returned.is_set, timeout=30)
                    assert warmed == [True] and len(outcomes) == 1
                    request.node.user_properties.extend(outcomes[0].items())
            finally:
                release.set()
                begin_warm.set()
                deliver.set()
                if holder.ident is not None:
                    await asyncio.to_thread(holder.join, 5)
                if worker is not None:
                    await asyncio.wait_for(worker.wait(), timeout=30)
            assert not holder.is_alive() and not expired.is_set() and not errors


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@private_profile_test
async def test_identity_releases_unissued_read_and_accepts_unchanged_install(
    request, monkeypatch
):
    """Use an actual current read for READY busy custody; unissued has a pure test."""
    from types import CodeType

    from tldw_chatbook import config
    from tldw_chatbook.UI.Console_Modules import config_sync
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from Tests.UI.test_console_projection_reads_off_loop import _fresh_display
    from Tests.UI.test_console_send_acknowledgement import (
        build,
        eager_tasks,
        ready_console,
        until,
    )

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            projection = await _fresh_display(console)
            identity = console._sync_console_identity_surfaces
            retry_code = next(
                value
                for value in identity.__func__.__code__.co_consts
                if isinstance(value, CodeType) and value.co_name == "retry"
            )
            retry_timers = []
            set_timer = console.set_timer

            def keep_retry(timer, callback):
                if getattr(callback, "__code__", None) is retry_code:
                    timer.pause()
                    if timer not in retry_timers:
                        retry_timers.append(timer)

            def hold_identity_timer(delay, callback, *args, **kwargs):
                timer = set_timer(delay, callback, *args, **kwargs)
                keep_retry(timer, callback)
                return timer

            monkeypatch.setattr(console, "set_timer", hold_identity_timer)
            for timer in tuple(getattr(console, "_timers", ())):
                keep_retry(timer, getattr(timer, "_callback", None))
            monkeypatch.setattr(
                console, "_sync_console_identity_surfaces", lambda: None
            )
            held, release, expired = (
                threading.Event(),
                threading.Event(),
                threading.Event(),
            )
            holder_errors, observed, body_calls, observer_failed = [], [], [], []
            holder = None
            previous = sys.getprofile()
            groups = []
            run_worker = console.run_worker

            def observe_worker(coro, *args, **kwargs):
                groups.append(kwargs.get("group"))
                return run_worker(coro, *args, **kwargs)

            monkeypatch.setattr(console, "run_worker", observe_worker)
            try:
                old_value, old_key, old_proof = (
                    projection.value,
                    projection.key,
                    projection._display_proof,
                )
                logical = config_sync._owe_console_config_sync(console, "identity")
                assert logical is not None
                token = logical.begin(projection)
                assert token is not None and token.active
                projection.run(
                    lambda: None, life=0.0
                )  # Existing checked-read owner/API.
                await until(lambda: not projection.pending, timeout=30)
                assert projection.value == old_value
                assert spend._same_readiness_key(projection.key, old_key)
                assert (
                    projection._display_proof is not None
                    and projection._display_proof is not old_proof
                )
                assert not projection._reads
                assert "console-readiness-config" in groups
                assert (
                    "console-readiness-publication" not in groups
                ), "setup did not exercise unchanged installation"
                assert logical.receipt is not None and logical.eligible()

                def hold_rebuild():
                    try:
                        with config._settings_rebuild_lock():
                            held.set()
                            if not release.wait(20):
                                expired.set()
                    except BaseException as error:
                        holder_errors.append(type(error).__name__)

                def observe(frame, event, argument):
                    try:
                        if (
                            event == "return"
                            and frame.f_code
                            is config_sync._ConsoleConfigHandoff.eligible.__code__
                            and frame.f_back is not None
                            and frame.f_back.f_code is identity.__func__.__code__
                        ):
                            if len(observed) < 4:
                                observed.append(argument is True)
                            else:
                                observer_failed.append(True)
                        if (
                            event == "call"
                            and frame.f_code is identity.__wrapped__.__code__
                        ):
                            if len(body_calls) < 4:
                                body_calls.append(True)
                            else:
                                observer_failed.append(True)
                    except BaseException:
                        observer_failed.append(True)
                    if previous is not None:
                        previous(frame, event, argument)

                holder = threading.Thread(
                    target=hold_rebuild, name="ready-identity-holder"
                )
                holder.start()
                assert await asyncio.to_thread(held.wait, 5)
                assert not release.is_set() and token.active
                timers_before = tuple(retry_timers)
                sys.setprofile(observe)
                try:
                    identity()  # Original wrapped owner; no fake READY/currentness flag.
                finally:
                    sys.setprofile(previous)
                assert observed == [True] and not observer_failed, (
                    "actual READY eligibility missing",
                    observed,
                )
                assert not body_calls and not release.is_set()
                request.node.user_properties.extend(
                    [
                        ("ready_eligible_during_busy_call", True),
                        ("foreign_rebuild_still_held", True),
                        ("identity_callback_deferred", True),
                        ("actual_unchanged_checked_read_installed", True),
                    ]
                )
                assert (
                    not token.active
                ), "READY identity busy attempt retained its poll handoff"
                assert console._console_config_retries[identity.__name__] == ((), {})
                assert retry_timers
                if timers_before:
                    assert tuple(retry_timers) == timers_before
                else:
                    assert len(retry_timers) == 1
            finally:
                sys.setprofile(previous)
                release.set()
                if holder is not None and holder.ident is not None:
                    await asyncio.to_thread(holder.join, 5)
                for timer in retry_timers:
                    timer.stop()
                config_sync._abandon_console_config_handoffs(console)
            assert holder is not None and not holder.is_alive()
            assert not expired.is_set() and not holder_errors
