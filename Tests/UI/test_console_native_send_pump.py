"""Actual native preparation must not park a Console input message pump."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
import threading

import pytest
from textual.widgets import Button
from textual.worker import WorkerCancelled

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_hook_review_send_freeze import _key, _pump_runs, _until
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_workbench_contract import ConsoleHarness
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

pytestmark = pytest.mark.bootstrap_profile


class _NativeHookRead:
    """Hold only the original native config body reached by this Send owner."""

    def __init__(self, owner):
        self.owner = owner
        self.snapshot_code = type(owner).snapshot.__code__
        self.raw_code = inspect.unwrap(config._read_raw_cli_config_unlocked).__code__
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread = None
        self.observations = 0
        self.operations = ()
        self.leases = ()

    def observe(self, frame, event, arg):
        if event != "call" or frame.f_code is not self.raw_code:
            return
        caller = frame.f_back
        while caller is not None:
            if (
                caller.f_code is self.snapshot_code
                and caller.f_locals.get("self") is self.owner
            ):
                break
            caller = caller.f_back
        if caller is None or self.entered.is_set():
            return
        self.observations += 1
        self.thread = threading.current_thread()
        with storage._lock:
            self.operations = tuple(
                operation
                for operation, state in raw._states.items()
                if state.source is config and state.thread is self.thread
            )
            self.leases = tuple(
                lease
                for operation in self.operations
                for lease in raw._states[operation].leases
            )
        self.entered.set()
        assert self.release.wait(20), "native config observation was not released"

    @contextlib.contextmanager
    def installed(self):
        previous = sys.getprofile()
        previous_threads = threading.getprofile()
        threading.setprofile_all_threads(self.observe)
        try:
            yield
        finally:
            self.release.set()
            threading.setprofile_all_threads(previous_threads)
            sys.setprofile(previous)


def _await_chain(pump):
    """Return code names only from the original pump coroutine's current wait."""
    task = getattr(pump, "_task", None)
    waiting = task.get_coro() if task is not None else None
    names = []
    for _ in range(25):
        code = getattr(waiting, "cr_code", getattr(waiting, "gi_code", None))
        if code is not None:
            names.append(code.co_name)
        waiting = getattr(waiting, "cr_await", getattr(waiting, "gi_yieldfrom", None))
        if waiting is None:
            break
    return names


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["enter", "send-button"])
async def test_actual_send_keeps_both_message_pumps_free_during_native_preparation(
    route,
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        composer = console._console_composer_or_none()
        composer.load_draft("Native preparation input")
        composer.focus()
        await pilot.pause()
        owner = console._console_runtime().ensure_hook_permissions()
        assert owner.snapshot().ready
        probe = _NativeHookRead(owner)
        with probe.installed():
            try:
                if route == "enter":
                    _key(host, "enter", "\r")
                else:
                    console.query_one("#console-send-message", Button).press()
                assert await _until(
                    probe.entered.is_set, 10
                ), "Send skipped its actual native source"
                assert probe.thread is not threading.current_thread()
                assert await _pump_runs(
                    host, 0.5
                ), f"Enter parked the app input pump during native preparation: {_await_chain(host)}"
                assert await _pump_runs(
                    console, 0.5
                ), f"Send parked the Console message pump during native preparation: {_await_chain(console)}"
                _key(host, "x", "x")
                assert await _until(
                    lambda: composer.draft_text().endswith("x"), 2
                ), "driver input never reached the live composer"
                assert not probe.release.is_set()
            finally:
                # A changed revision safely refuses before provider entry. Even a
                # RED pump run must release its actual native worker and settle.
                composer.load_draft("Keep this later revision")
                probe.release.set()
                assert await _until(lambda: not console._hooks._busy, 15)
                assert composer.draft_text() == "Keep this later revision"
        assert probe.observations == 1
        assert (
            inspect.unwrap(config._read_raw_cli_config_unlocked).__code__
            is probe.raw_code
        )


@pytest.mark.asyncio
async def test_pump_send_repeated_cancellation_drains_its_actual_native_worker():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        composer = console._console_composer_or_none()
        composer.load_draft("Native cancellation input")
        composer.focus()
        await pilot.pause()
        owner = console._console_runtime().ensure_hook_permissions()
        assert owner.snapshot().ready
        probe = _NativeHookRead(owner)
        with probe.installed():
            try:
                _key(host, "enter", "\r")
                assert await _until(probe.entered.is_set, 10)
                assert (
                    probe.operations and probe.leases
                ), "the actual raw worker owns no native custody"
                workers = [
                    worker
                    for worker in console.workers
                    if worker.node is console
                    and worker.group == "console-hook-send-review"
                ]
                assert (
                    len(workers) == 1
                ), "native preparation is still owned by the input pump"
                worker = workers[0]
                worker.cancel()
                await asyncio.sleep(0)
                worker.cancel()
                await asyncio.sleep(0)
                assert (
                    not worker._task.done()
                ), "cancellation abandoned the actual native reader"
                assert console._hooks._busy
                with storage._lock:
                    assert all(
                        operation in raw._states for operation in probe.operations
                    )
                    assert all(lease in storage._live_leases for lease in probe.leases)
                probe.release.set()
                with pytest.raises(WorkerCancelled):
                    await worker.wait()
                assert not console._hooks._busy
                with storage._lock:
                    assert all(
                        operation not in raw._states for operation in probe.operations
                    )
                    assert all(
                        lease not in storage._live_leases for lease in probe.leases
                    )
                assert composer.draft_text() == "Native cancellation input"
            finally:
                composer.load_draft("Keep this later revision")
                probe.release.set()
                assert await _until(lambda: not console._hooks._busy, 15)


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["owner", "reader"])
async def test_native_ready_publication_cannot_change_its_consumed_hook_owner(
    replacement,
):
    from Tests.UI.test_console_hook_review_send_freeze import _record_dispatch
    from tldw_chatbook.Agents.hook_permissions import HookPermissions

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        composer = console._console_composer_or_none()
        composer.load_draft("Keep owner-fenced input")
        composer.focus()
        await pilot.pause()
        runtime = console._console_runtime()
        owner = runtime.ensure_hook_permissions()
        assert owner.snapshot().ready
        other = HookPermissions()
        calls = _record_dispatch(console)
        original_state = console._hooks._on_state
        published = []

        def change_after_state(snapshot):
            original_state(snapshot)
            published.append(snapshot)
            if replacement == "owner":
                with runtime._run_hooks_lock:
                    runtime._hook_permissions = other
            else:
                # A real other owner's original method, not a guarded reader fake.
                owner.snapshot = other.snapshot

        console._hooks._on_state = change_after_state
        probe = _NativeHookRead(owner)
        with probe.installed():
            try:
                _key(host, "enter", "\r")
                assert await _until(probe.entered.is_set, 10)
                assert probe.operations and probe.leases
                assert probe.thread is not threading.current_thread()
                probe.release.set()
                assert await _until(lambda: not console._hooks._busy, 15)
                assert len(published) == 1
                assert calls == [], "state publication redirected a captured ready Send"
                assert composer.draft_text() == "Keep owner-fenced input"
            finally:
                probe.release.set()
                console._hooks._on_state = original_state
                if replacement == "owner":
                    with runtime._run_hooks_lock:
                        runtime._hook_permissions = owner
                elif "snapshot" in vars(owner):
                    # A failed publication may never have installed the override.
                    del owner.snapshot
                assert await _until(lambda: not console._hooks._busy, 15)
