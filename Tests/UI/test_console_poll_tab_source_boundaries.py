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
from textual.widgets import Button
from textual.worker import Worker
from textual.worker_manager import WorkerManager

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
        self.native_census = {}
        self.warm_native_census = {}
        self.ensure_census = []
        self.drift_applied = False
        self.competing_ensure_entry = None
        self.competing_ensures = []
        self.first_surface_return_settings = None
        self.warm_loads = self.cache_hits = self.posture_checks = 0
        self.file_posture_checks = self.guarded_loads = 0
        self.origin_lookups = {
            "initial": 0,
            "overlap": 0,
            "unrelated": 0,
            "truncated": 0,
        }

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
        if label in {"acquire", "raw_scope", "close_fd", "metadata"}:
            key = (
                label,
                kind,
                threading.current_thread() is self.actor,
                self._full(frame) is not None,
            )
            self.native_census[str(key)] = self.native_census.get(str(key), 0) + 1
            parent = frame.f_back
            for _ in range(40):
                if parent is None:
                    break
                if parent.f_code is config.load_settings.__code__:
                    self.warm_native_census[str(key)] = (
                        self.warm_native_census.get(str(key), 0) + 1
                    )
                    break
                parent = parent.f_back
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
                if not self.publications:
                    current_store = self.screen._console_chat_store
                    current_session = current_store._sessions.get(current_store.active_session_id) if current_store is not None else None
                    self.first_surface_return_settings = (
                        current_store, current_session,
                        current_session.settings if current_session is not None else None,
                    )
                self.publications.append(
                    (
                        tabs.f_locals["store"],
                        frame.f_locals["active_session_id"],
                        tuple(frame.f_locals["sessions"]),
                    )
                )
            return
        if label in {"ensure", "uncached"}:
            if self.drift_applied and kind in {start, returned} and len(self.ensure_census) < 64:
                current_store = self.screen._console_chat_store
                current_session = current_store._sessions.get(current_store.active_session_id) if current_store is not None else None
                self.ensure_census.append({
                    "label": label, "kind": kind,
                    "issuing_task": task is self.held.get("task"),
                    "caller": frame.f_back.f_code.co_name,
                    "under_tabs": _transition_parent(frame, self.tab_code, self.screen) is not None,
                    "under_full": full is not None,
                    "current_settings_missing": current_session is None or current_session.settings is None,
                    "returned_current_settings": current_session is not None and value is current_session.settings,
                    "surface_held": not self.held["future"].done(),
                })
            if label == "ensure" and self.drift_applied:
                current_store = self.screen._console_chat_store
                current_session = current_store._sessions.get(current_store.active_session_id) if current_store is not None else None
                if (
                    kind == start and current_session is not None
                    and current_session.settings is None
                    and task is not self.held.get("task")
                    and frame.f_back.f_code is ChatScreen._active_console_settings_readiness_uncached.__code__
                    and self.held["future"].done() and not self.publications
                ):
                    self.competing_ensure_entry = (id(frame), task, current_store, current_session)
                elif kind == returned and self.competing_ensure_entry is not None:
                    frame_id, issued_task, issued_store, issued_session = self.competing_ensure_entry
                    if id(frame) == frame_id and task is issued_task:
                        assert current_store is issued_store and current_session is issued_session
                        assert value is not None and value is issued_session.settings
                        assert not self.publications and self.held["future"].done()
                        self.competing_ensures.append((issued_store, issued_session, value, issued_task))
                        self.competing_ensure_entry = None
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
        if label == "settings" and kind == returned:
            assert value is not None and value is config._SETTINGS_CACHE
            assert frame.f_locals["cached"] is value
            self.warm_loads += 1
        elif label == "cache_hit" and kind == returned:
            assert value is not None and value is config._SETTINGS_CACHE
            assert frame.f_locals["cached"] is value
            self.cache_hits += 1
        elif label == "posture" and kind == returned:
            assert value is True
            self.posture_checks += 1
        elif label == "file_posture" and kind == returned:
            assert type(value) is tuple and value
            self.file_posture_checks += 1
        elif label == "guarded_settings" and kind == start:
            self.guarded_loads += 1
        elif label == "acquire" and kind == returned:
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
            # Current POSIX posture returns only after its original native stat.
            # This is an observation count, not a Windows handle-retirement proof.
            if os.name != "nt":
                assert value is None or type(value) is tuple and len(value) == 5
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
            ("readiness", ChatScreen, "_active_console_settings_readiness_uncached"),
            ("worker", WorkerManager, "_new_worker"),
            ("coalesced", ChatScreen, "_run_coalesced_control_bar_sync"),
            ("settings", config, "load_settings"),
            ("cache_hit", config, "_settings_cache_hit"),
            ("posture", config, "_posture_unchanged"),
            ("file_posture", config, "_config_file_posture"),
            ("guarded_settings", config, "_load_settings_guarded"),
            ("acquire", storage, "_acquire_storage"),
            ("raw_scope", raw, "_scope"),
            ("close_fd", raw, "_close_descriptor"),
            (
                "metadata",
                windows_files._Native if os.name == "nt" else storage,
                "open_handle" if os.name == "nt" else "_posture",
            ),
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
            record_property("ensure_census", json.dumps(self.ensure_census))
            record_property("warm_native_census", json.dumps(self.warm_native_census))
            record_property(
                "tab_boundary_origin_lookups",
                json.dumps(self.origin_lookups, sort_keys=True),
            )

    def assert_retired(self):
        # TASK33265 serves warm reads before storage admission. The original
        # cold/held-read controls in test_console_tab_publication_read_barrier
        # separately calibrate real handles/leases and physical retirement.
        assert self.warm_loads > 0 and self.cache_hits >= self.warm_loads
        assert self.posture_checks == self.cache_hits
        assert self.file_posture_checks > 0
        assert self.guarded_loads == 0
        assert self.warm_native_census == {}
        # Other FULL responsibilities may issue native work, especially during
        # reconstruction. Retain their physical retirement checks when observed.
        assert all(row["closed"] for row in self.pin_descriptors)
        with storage._lock:
            assert all(lease not in storage._live_leases for lease in self.leases)
            for operation, state in self.states:
                assert operation not in raw._states
                assert operation not in storage._raw_operations
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
            lambda: (
                not screen._console_sync_in_progress
                and screen._console_session_tabs_sync_calls == 0
                and not screen._console_sync_requested
                and not getattr(screen, "_console_control_bar_replay_whole_sync", False)
            ),
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
                assert await _until(lambda: "tabs" in observation.held, 10), (
                    "Original FULL never suspended at the real Surface lock"
                )
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
                observation.drift_applied = True
                overlap = asyncio.create_task(_request_original_full(screen, "overlap"))
                issued_tasks.append(overlap)
                assert await _until(lambda: observation.overlap_requested, 5), (
                    "Overlapping original FULL did not retain demand"
                )
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
                    lambda: (
                        initial.done()
                        and overlap.done()
                        and observation.workers
                        and any(
                            value is True
                            and any(
                                task is issued for _, issued, _ in observation.workers
                            )
                            for task, value, _ in observation.full_returns
                        )
                        and not screen._console_sync_in_progress
                        and not screen._console_sync_requested
                        and not getattr(
                            screen, "_console_control_bar_replay_whole_sync", False
                        )
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
                    if drift == "active-session" and not observation.ensures:
                        assert observation.ensure_returns == []
                        assert len(observation.competing_ensures) == 1
                        ensured_store, ensured_session, settings, readiness_task = observation.competing_ensures[0]
                        assert readiness_task is not initial
                        assert all(
                            before is after for before, after in zip(
                                observation.first_surface_return_settings,
                                (ensured_store, ensured_session, settings),
                            )
                        )
                    else:
                        assert len(observation.ensures) == len(observation.ensure_returns) == 1
                        assert observation.competing_ensures == []
                        ensured_store, ensured_session, settings = observation.ensure_returns[0]
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
    record_property(
        "current_native_census", json.dumps(observation.native_census, sort_keys=True)
    )
    observation.assert_retired()
    record_property(
        "original_tab_source_boundary",
        json.dumps(
            {
                "drift": drift,
                "same_original_tab_publications": len(observation.publications),
                "current_owner_ensures": len(observation.ensure_returns),
                "competing_readiness_ensures": len(observation.competing_ensures),
                "original_replay_workers": len(observation.workers),
                "original_warm_settings_returns": observation.warm_loads,
                "original_cache_hits": observation.cache_hits,
                "original_posture_checks": observation.posture_checks,
                "original_file_posture_checks": observation.file_posture_checks,
                "original_guarded_loads": observation.guarded_loads,
                "issued_raw_operations": len(observation.states),
                "issued_leases": len(observation.leases),
                "physical_descriptor_closes": observation.physical_closes,
                "qualified_pin_descriptors": len(observation.pin_descriptors),
                "current_native_metadata_observations": observation.metadata_retirements,
                "source_current": True,
                "drafts_preserved": True,
                "owned_workers_retired_and_no_storage_admission": True,
            },
            sort_keys=True,
        ),
    )
