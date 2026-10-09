"""TASK-34563.38: original FULL tab loops must follow their current source."""

import asyncio
import contextlib
import errno
import inspect
import json
import os
import sys
import threading
import traceback

import pytest
from textual.worker import Worker
from textual.worker_manager import WorkerManager
from textual.widgets import Button

from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_poll_reconciliation import (
    _original_transition_events,
    _qualify_warm_capture_sources,
    _transition_parent,
)
from Tests.UI.test_console_received_intent_feedback import _received_console_case
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Widgets.Console.console_session_surface import ConsoleSessionSurface

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def _request_original_full(screen, origin):
    """A test-owned caller, never a substitute for the original refresh body."""
    return await screen._sync_native_console_chat_ui()


class _OriginalTabDrift:
    def __init__(self, case, surface):
        self.case, self.screen, self.surface = case, case.console, surface
        self.actor = threading.current_thread()
        self.full_code = ChatScreen._sync_native_console_chat_ui.__code__
        self.tab_code = ChatScreen._sync_console_native_session_tabs.__code__
        self.replay_code = ChatScreen._run_coalesced_control_bar_sync.__code__
        self.ensure_code = inspect.unwrap(
            type(self.screen._session)._ensure_active_console_session_settings
        ).__code__
        self.held = {}
        self.full_starts, self.full_returns = [], []
        self.publications, self.ensures, self.ensure_returns = [], [], []
        self.workers, self.leases, self.states = [], [], []
        self.pin_descriptors = []
        self.physical_closes = self.metadata_retirements = 0
        self.overlap_requested = False
        self.origin_lookups = dict(initial=0, overlap=0, unrelated=0, truncated=0)

    def _full(self, frame):
        if frame.f_code is self.full_code and frame.f_locals.get("self") is self.screen:
            return frame
        return _transition_parent(frame, self.full_code, self.screen)

    def _test_origin(self, frame):
        task = asyncio.current_task()
        request = task.get_coro() if task is not None else None
        if (
            not inspect.iscoroutine(request)
            or request.cr_code is not _request_original_full.__code__
        ):
            # An eager replay worker may retain the driver's stack ancestry,
            # but only the actual test task owns the marked request.
            self.origin_lookups["unrelated"] += 1
            return None
        parent = frame.f_back
        for _ in range(40):
            if parent is None:
                self.origin_lookups["unrelated"] += 1
                return None
            if (
                parent is request.cr_frame
                and parent.f_code is _request_original_full.__code__
                and parent.f_locals.get("screen") is self.screen
            ):
                origin = parent.f_locals["origin"]
                assert origin in {"initial", "overlap"}
                self.origin_lookups[origin] += 1
                return origin
            parent = parent.f_back
        # Ordinary app/replay calls need not have the test driver in their
        # ancestry. Never attribute a truncated lookup to a requested FULL.
        self.origin_lookups["truncated"] += 1
        return None

    def observe(self, kind, label, frame, value):
        if threading.current_thread() is not self.actor:
            return
        start, yielded, returned = (
            sys.monitoring.events.PY_START,
            sys.monitoring.events.PY_YIELD,
            sys.monitoring.events.PY_RETURN,
        )
        task = asyncio.current_task()
        if label == "full" and frame.f_locals.get("self") is self.screen:
            if kind == start:
                origin = self._test_origin(frame)
                self.full_starts.append((task, origin))
                if origin == "initial":
                    assert "full" not in self.held
                    self.held.update(full=id(frame), task=task)
            elif kind == yielded and self._test_origin(frame) == "overlap":
                self.overlap_requested |= bool(self.screen._console_sync_requested)
            elif kind == returned:
                self.full_returns.append((task, value, self.screen._console_chat_store))
            return
        full = self._full(frame)
        if label == "surface" and frame.f_locals.get("self") is self.surface:
            # Frame IDs can be reused after the issuing task finishes. The
            # retained task and frame identity jointly select this invocation.
            if task is not self.held.get("task"):
                return
            tabs = _transition_parent(frame, self.tab_code, self.screen)
            if tabs is None or full is None or id(full) != self.held.get("full"):
                return
            assert task is self.held["task"]
            if kind == yielded and "tabs" not in self.held:
                assert "visit" in full.f_locals
                assert tabs.f_locals["store"] is self.case.store
                assert frame.f_locals["active_session_id"] == self.case.session.id
                assert type(value) is asyncio.Future and not value.done()
                assert any(
                    value is waiter
                    for waiter in self.surface._session_sync_lock._waiters
                )
                self.held.update(tabs=id(tabs), future=value)
            elif kind == returned and id(tabs) == self.held.get("tabs"):
                self.publications.append(
                    (
                        tabs.f_locals["store"],
                        frame.f_locals["active_session_id"],
                        tuple(frame.f_locals["sessions"]),
                    )
                )
            return
        if label in {"ensure", "uncached"}:
            if task is not self.held.get("task"):
                return
            tabs = _transition_parent(frame, self.tab_code, self.screen)
            if tabs is None or id(tabs) != self.held.get("tabs"):
                return
            # Only the loop's direct ensure is this boundary. A coachmark or
            # another nested display helper may independently ensure settings.
            ensure = (
                frame
                if label == "ensure"
                else _transition_parent(frame, self.ensure_code, self.screen._session)
            )
            if ensure is None or ensure.f_back is not tabs:
                return
            assert task is self.held["task"]
            if label == "ensure" and kind == start:
                current = self.screen._console_chat_store
                assert current is tabs.f_locals["store"]
                session = current._sessions.get(current.active_session_id)
                assert session is None or session.settings is None
                self.ensures.append((current, session))
            elif label == "uncached" and kind == returned:
                current, session = frame.f_locals["store"], frame.f_locals["session"]
                assert current is self.screen._console_chat_store
                assert session is current._sessions[current.active_session_id]
                assert value is session.settings and value is not None
                self.ensure_returns.append((current, session, value))
            return
        if label == "worker" and kind == returned:
            immediate = (
                task is self.held.get("task")
                and full is not None
                and id(full) == self.held.get("full")
            )
            coalesced = _transition_parent(frame, self.replay_code, self.screen)
            if not immediate and coalesced is None:
                return
            if (
                type(value) is Worker
                and value.node is self.screen
                and value.group == "console-sync"
            ):
                assert (
                    inspect.iscoroutine(value._work)
                    and value._work.cr_code is self.full_code
                )
                assert type(value._task) is asyncio.Task
                self.workers.append((value, value._task, value._work))
            return
        if full is None:
            return
        if label == "acquire" and kind == returned:
            with storage._lock:
                if value in storage._live_leases:
                    self.leases.append(value)
        elif label == "raw_scope" and kind == yielded:
            state = raw._states.get(value)
            if state is not None and not any(value is op for op, _ in self.states):
                assert state.active
                self.states.append((value, state))
                # Parent pins are installed before the raw body is yielded.
                # Ordinary descriptors are a separate ledger; _retire feeds
                # each actual pinned FD through the original close method.
                for anchor, fd in state.pins.items():
                    os.fstat(fd)
                    self.pin_descriptors.append(
                        {"state": state, "anchor": anchor, "fd": fd, "closed": False}
                    )
        elif label == "close_fd" and kind == returned:
            state, fd = frame.f_locals["state"], frame.f_locals["fd"]
            if any(state is owned for _, owned in self.states):
                try:
                    os.fstat(fd)
                except OSError as error:
                    assert error.errno == errno.EBADF
                else:
                    raise AssertionError(
                        "Original descriptor close did not retire the FD"
                    )
                assert fd not in state.descriptors
                self.physical_closes += 1
                for issued in self.pin_descriptors:
                    if issued["state"] is state and issued["fd"] == fd:
                        assert state.pins.get(issued["anchor"]) == fd
                        issued["closed"] = True
        elif label == "metadata" and kind == returned:
            assert not frame.f_locals["opened"] and not frame.f_locals["failed"]
            self.metadata_retirements += 1

    def bindings(self):
        return [
            ("full", ChatScreen, "_sync_native_console_chat_ui"),
            ("tabs", ChatScreen, "_sync_console_native_session_tabs"),
            ("surface", ConsoleSessionSurface, "sync_sessions"),
            (
                "ensure",
                type(self.screen._session),
                "_ensure_active_console_session_settings",
            ),
            (
                "uncached",
                type(self.screen._session),
                "_ensure_active_console_session_settings_uncached",
            ),
            ("worker", WorkerManager, "_new_worker"),
            ("coalesced", ChatScreen, "_run_coalesced_control_bar_sync"),
            ("acquire", storage, "_acquire_storage"),
            ("raw_scope", raw, "_scope"),
            ("close_fd", raw, "_close_descriptor"),
            ("metadata", windows_files.WindowsOS, "stat_many_for_admission"),
        ]

    @contextlib.contextmanager
    def installed(self, record_property):
        try:
            with _original_transition_events(self.bindings(), self.observe):
                try:
                    yield
                except BaseException as error:
                    # Keep the body's first failure even if the shared observer's
                    # final source/monitor assertion subsequently fails too.
                    record_property(
                        "tab_boundary_first_body_failure",
                        json.dumps(
                            {
                                "type": type(error).__name__,
                                "message": str(error),
                                "frames": [
                                    {
                                        "file": row.filename,
                                        "line": row.lineno,
                                        "name": row.name,
                                    }
                                    for row in traceback.extract_tb(error.__traceback__)
                                ],
                            },
                            sort_keys=True,
                        ),
                    )
                    raise
        finally:
            record_property(
                "tab_boundary_origin_lookups",
                json.dumps(self.origin_lookups, sort_keys=True),
            )

    def assert_retired(self):
        assert self.states and self.leases and self.physical_closes
        assert self.pin_descriptors and all(
            row["closed"] for row in self.pin_descriptors
        )
        with storage._lock:
            assert all(lease not in storage._live_leases for lease in self.leases)
            for operation, state in self.states:
                assert (
                    operation not in raw._states
                    and operation not in storage._raw_operations
                )
                assert not state.active and not state.uncertain
                assert not state.descriptors and not state.pins and not state.files
                assert all(lease not in storage._live_leases for lease in state.leases)
        assert all(task.done() for _, task, _ in self.workers)


@pytest.mark.parametrize("drift", ["membership", "active-session", "rebuild-store"])
async def test_original_full_tab_loop_rechecks_live_ensure_after_surface_await(
    monkeypatch, record_property, drift
):
    observation = None
    issued_tasks = []
    async with _received_console_case(
        monkeypatch, f"tab-drift-{drift}", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        screen, runtime = case.console, case.runtime
        assert await _until(
            lambda: not screen._console_sync_in_progress
            and screen._console_session_tabs_sync_calls == 0
            and not screen._console_sync_requested
            and not getattr(screen, "_console_control_bar_replay_whole_sync", False),
            5,
        ), "Original full refresh did not settle before the tab boundary control"
        surface = screen.query_one("#console-session-surface", ConsoleSessionSurface)
        lock = surface._session_sync_lock
        assert type(lock) is asyncio.Lock and not lock.locked()
        source = config.current_config_identity()
        old_store, old_controller = case.store, case.controller
        assert (
            case.composer.draft_text()
            == old_store.session_draft(case.session.id)
            == case.draft
        )
        successor = added_session = None
        original_settings = case.session.settings
        original_baseline = case.session.canonical_settings_baseline
        original_default_generation = case.session.new_chat_default_generation
        if drift == "membership":
            assert original_settings is not None
        if drift == "active-session":
            successor = old_store.create_session(
                title="Current tab source",
                workspace_id=case.session.workspace_id,
                settings=case.session.settings,
                canonical_settings_baseline=case.session.settings,
                assistant_kind="generic",
                activate=False,
            )
            assert (
                successor.new_chat_default_generation
                == screen._session._console_new_chat_default_generation()
            )
            old_store.set_session_draft(successor.id, "Current tab source draft")
        observation = _OriginalTabDrift(case, surface)
        held_lock = False
        try:
            await lock.acquire()
            held_lock = True
            with observation.installed(record_property):
                initial = asyncio.create_task(_request_original_full(screen, "initial"))
                issued_tasks.append(initial)
                assert await _until(
                    lambda: "tabs" in observation.held, 10
                ), "Original FULL never suspended at the real Surface lock"
                assert (
                    observation.held["task"] is initial
                    and not observation.held["future"].done()
                )
                assert not screen._console_sync_requested
                if drift == "membership":
                    added_session = old_store.create_session(
                        title="Added tab source",
                        workspace_id=case.session.workspace_id,
                        settings=case.session.settings,
                        assistant_kind="generic",
                        activate=False,
                    )
                    # Membership drift does not require a settings fault. Keep
                    # the active source's actual settings and provenance intact.
                elif drift == "active-session":
                    successor.settings = None
                    old_store.switch_session(successor.id)
                else:
                    # Documented original reconstruction seams, never a fake
                    # store/controller or a partial controller.store rebind.
                    runtime.set_chat_store(None)
                    runtime.set_chat_controller(None)
                overlap = asyncio.create_task(_request_original_full(screen, "overlap"))
                issued_tasks.append(overlap)
                assert await _until(
                    lambda: observation.overlap_requested, 5
                ), "Overlapping original FULL did not retain demand"
                assert (
                    screen._console_sync_in_progress
                    and not initial.done()
                    and not overlap.done()
                )
                assert any(
                    task is overlap and origin == "overlap"
                    for task, origin in observation.full_starts
                )
                lock.release()
                held_lock = False
                assert await _until(
                    lambda: initial.done()
                    and overlap.done()
                    and observation.workers
                    and any(
                        value is True
                        and any(task is issued for _, issued, _ in observation.workers)
                        for task, value, _ in observation.full_returns
                    )
                    and not screen._console_sync_in_progress
                    and not screen._console_sync_requested
                    and not getattr(
                        screen, "_console_control_bar_replay_whole_sync", False
                    ),
                    15,
                ), "Original pending FULL replay did not complete on the current owner"
                initial.result()
                overlap.result()
                current = runtime.chat_store
                assert current is screen._console_chat_store
                assert (
                    runtime.chat_controller is not None
                    and runtime.chat_controller.store is current
                )
                assert config.current_config_identity() == source
                assert old_store.session_draft(case.session.id) == case.draft
                assert len(observation.publications) >= 2
                first, last = observation.publications[0], observation.publications[-1]
                assert first[0] is old_store and first[1] == case.session.id
                assert any(row is case.session for row in first[2])
                assert last[0] is current and last[1] == current.active_session_id
                current_sessions = tuple(current.sessions())
                assert len(current_sessions) == len(last[2])
                assert all(
                    now is published
                    for now, published in zip(current_sessions, last[2])
                )
                if drift == "membership":
                    assert observation.ensures == observation.ensure_returns == []
                    assert current is old_store
                    assert current.active_session_id == case.session.id
                    assert case.session.settings is original_settings
                    assert case.session.canonical_settings_baseline is original_baseline
                    assert (
                        case.session.new_chat_default_generation
                        == original_default_generation
                    )
                    assert added_session is not None
                    assert not any(row is added_session for row in first[2])
                    assert any(row is added_session for row in last[2])
                else:
                    assert (
                        len(observation.ensures) == len(observation.ensure_returns) == 1
                    )
                    ensured_store, ensured_session, settings = (
                        observation.ensure_returns[0]
                    )
                    assert (
                        ensured_store is current
                        and ensured_session
                        is current._sessions[current.active_session_id]
                    )
                    assert ensured_session.settings is settings
                    if drift == "rebuild-store":
                        assert (
                            current is not old_store
                            and runtime.chat_controller is not old_controller
                        )
                        assert observation.ensures[0][0] is current
                        assert observation.ensures[0][1] is None
                        assert ensured_session.canonical_settings_baseline == settings
                    else:
                        assert current is old_store
                        assert ensured_session is (
                            successor if successor is not None else case.session
                        )
                assert (
                    screen._console_visible_draft_session_id
                    == current.active_session_id
                )
                assert case.composer.draft_text() == current.session_draft(
                    current.active_session_id
                )
                if drift == "active-session":
                    assert case.composer.draft_text() == "Current tab source draft"
                assert screen.query_one(
                    f"#console-session-tab-{current.active_session_id}", Button
                ).has_class("console-session-tab-active")
                assert screen._console_session_tabs_sync_calls == 0
                assert not case.provider_calls and not runtime._turn_custody
                assert all(
                    worker._task is task and worker._work is work
                    for worker, task, work in observation.workers
                )
        finally:
            if held_lock:
                lock.release()
            for task in issued_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.wait_for(
                asyncio.gather(*issued_tasks, return_exceptions=True), 15
            )
            if runtime.chat_controller is not old_controller:
                await old_controller.shutdown()
    # The fixture has disposed the real runtime and quiesced its file-backed DB.
    assert all(task.done() for task in issued_tasks)
    observation.assert_retired()
    record_property(
        "original_tab_source_boundary",
        json.dumps(
            {
                "drift": drift,
                "same_original_tab_publications": len(observation.publications),
                "current_owner_ensures": len(observation.ensure_returns),
                "original_replay_workers": len(observation.workers),
                "issued_raw_operations": len(observation.states),
                "issued_leases": len(observation.leases),
                "physical_descriptor_closes": observation.physical_closes,
                "qualified_pin_descriptors": len(observation.pin_descriptors),
                "retired_metadata_snapshots": observation.metadata_retirements,
                "source_current": True,
                "drafts_preserved": True,
                "owned_workers_and_native_resources_retired": True,
            },
            sort_keys=True,
        ),
    )
