"""DRAFT for Tests/UI: real reusable Console credential-poll visibility.

This file is not installed or launched by its author. The coordinator must put
the reviewed bytes under Tests/UI so the original private-profile/guard fixtures
apply. No reader, permission, native implementation or timer rate is replaced.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import CodeType, FunctionType

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_screen_reuse import _boot_settled, _press_until_screen


class ReadinessCalls:
    """Count the exact installed reader's real native calls without replacing it."""

    def __init__(
        self, projection, native_code, credential_body, wrapper_code, native_kind
    ):
        reader = projection.read_current
        assert type(reader) is FunctionType
        assert reader.__qualname__ == (
            "ConsoleReadinessConfigProjection.for_screen.<locals>.read_current"
        )
        self.projection = projection
        self.reader = reader
        self.reader_code = reader.__code__
        self.native_code = native_code
        self.native_kind = native_kind
        self.run_code = type(projection).run.__code__
        self.credential_body = credential_body
        self.wrapper_code = wrapper_code
        self.phase = "visible_warm"
        self.rows = []
        self.schedules = []
        self.run_frames = {}
        self.live = {}
        self.actors = []  # Strong references prevent Thread/id reuse in this receipt.
        self.existing_actors = tuple(threading.enumerate())
        self.local = threading.local()
        self.lock = threading.Lock()
        self.active = False
        self.callback = self._observe
        self.overflow = 0
        self.restoration = None
        self.monitor = sys.monitoring
        self.tool = None
        self.tool_name = "tldw-hidden-readiness-" + str(id(self))
        self.start_callback = self._start
        self.return_callback = self._return
        self.mask = self.monitor.events.PY_START | self.monitor.events.PY_RETURN
        self.code_masks = {
            self.reader_code: self.mask,
            self.run_code: self.mask,
        }
        if self.native_code is not None:
            self.code_masks[self.native_code] = self.monitor.events.PY_START
        self.installed_codes = {}

    def _start(self, code, _offset):
        frame = sys._getframe(1)
        assert (
            frame.f_code is code
        ), "local monitoring must name the actual executing frame"
        self._observe(frame, "call", None)

    def _return(self, code, _offset, _value):
        frame = sys._getframe(1)
        assert (
            frame.f_code is code
        ), "local monitoring must name the actual returning frame"
        self._observe(frame, "return", None)

    def _observe(self, frame, event, _arg):
        if not self.active or event not in ("call", "return"):
            return
        code = frame.f_code
        if code is self.run_code and frame.f_locals.get("self") is self.projection:
            if event == "call":
                caller = frame.f_back
                function = (
                    caller.f_locals.get("function")
                    if caller is not None and caller.f_code is self.wrapper_code
                    else None
                )
                self.run_frames[id(frame)] = (
                    self.projection.pending,
                    function is self.credential_body,
                    self.phase,
                )
            else:
                entered = self.run_frames.pop(id(frame), None)
                if entered is not None and not entered[0] and self.projection.pending:
                    self.schedules.append(
                        {
                            "id": len(self.schedules) + 1,
                            "phase": entered[2],
                            "original_credential_wrapper": entered[1],
                        }
                    )
            return
        if code is self.reader_code:
            if frame.f_locals.get("projection") is not self.projection:
                return
            if event == "call":
                actor = threading.current_thread()
                with self.lock:
                    if len(self.rows) >= 128:
                        self.overflow += 1
                        return
                    row = {
                        "id": len(self.rows) + 1,
                        "phase": self.phase,
                        "thread_id": threading.get_ident(),
                        "thread_object_id": id(actor),
                        "thread_name": actor.name,
                        "actor_existed_before_monitor": any(
                            actor is held for held in self.existing_actors
                        ),
                        "entered": time.monotonic(),
                        "exited": None,
                        "native_boundary_calls": 0,
                        "schedule_id": self.schedules[-1]["id"]
                        if self.schedules
                        else None,
                        "checked_identity_reached": False,
                    }
                    self.actors.append(actor)
                    self.rows.append(row)
                    self.live[id(frame)] = row
                self.local.row = row
            else:
                with self.lock:
                    row = self.live.pop(id(frame), None)
                    if row is not None:
                        row["exited"] = time.monotonic()
                        # These are original reader locals, not copied permission.
                        row["checked_identity_reached"] = (
                            "before" in frame.f_locals and "after" in frame.f_locals
                        )
                        row["actual_operation_local_present"] = (
                            frame.f_locals.get("active") is not None
                        )
                self.local.row = None
        elif event == "call" and code is self.native_code:
            row = getattr(self.local, "row", None)
            if row is not None:
                row["native_boundary_calls"] += 1

    def start(self):
        self.tool = next(
            (number for number in (5, 4, 3) if self.monitor.get_tool(number) is None),
            None,
        )
        assert (
            self.tool is not None
        ), "code-local observer will not displace another tool"
        self.monitor.use_tool_id(self.tool, self.tool_name)
        assert self.monitor.get_events(self.tool) == 0
        assert (
            self.monitor.register_callback(
                self.tool, self.monitor.events.PY_START, self.start_callback
            )
            is None
        )
        assert (
            self.monitor.register_callback(
                self.tool, self.monitor.events.PY_RETURN, self.return_callback
            )
            is None
        )
        self.active = True
        try:
            for code, mask in self.code_masks.items():
                assert self.monitor.get_local_events(self.tool, code) == 0
                self.monitor.set_local_events(self.tool, code, mask)
                self.installed_codes[code] = mask
            assert self.monitor.get_events(self.tool) == 0
        except BaseException:
            self.stop()
            raise

    def stop(self):
        monitor, tool = self.monitor, self.tool
        assert tool is not None and monitor.get_tool(tool) == self.tool_name
        assert monitor.get_events(tool) == 0
        # Keep callbacks active until all owned events and callback slots retire.
        assert all(
            monitor.get_local_events(tool, code) == mask
            for code, mask in self.installed_codes.items()
        )
        for code in self.installed_codes:
            monitor.set_local_events(tool, code, 0)
        assert all(
            monitor.get_local_events(tool, code) == 0 for code in self.code_masks
        )
        assert (
            monitor.register_callback(tool, monitor.events.PY_START, None)
            is self.start_callback
        )
        assert (
            monitor.register_callback(tool, monitor.events.PY_RETURN, None)
            is self.return_callback
        )
        assert monitor.get_events(tool) == 0
        monitor.free_tool_id(tool)
        assert monitor.get_tool(tool) is None
        self.restoration = "local_events_callbacks_removed_tool_freed_global0"
        self.active = False

    def completed(self, phase):
        with self.lock:
            return [
                dict(row)
                for row in self.rows
                if row["phase"] == phase and row["exited"] is not None
            ]

    def receipt(self):
        with self.lock:
            return {
                "diagnostic_only": True,
                "original_reader": {
                    "module": self.reader.__module__,
                    "qualname": self.reader.__qualname__,
                    "source": self.reader_code.co_filename,
                    "line": self.reader_code.co_firstlineno,
                },
                "rows": [dict(row) for row in self.rows],
                "schedules": [dict(row) for row in self.schedules],
                "live_reader_ids": [row["id"] for row in self.live.values()],
                "overflow": self.overflow,
                "restoration": self.restoration,
                "monitor_global_events": 0,
                "selected_local_code_count": len(self.code_masks),
                "native_count_kind": self.native_kind,
                "unmatched_readers_invalidate_evidence": True,
                "guards_replaced": False,
                "callbacks_replaced": False,
                "timer_rates_changed": False,
            }


def _path_witness(value):
    """Hash path metadata without reading a file or calling a custom path object."""
    if value is None:
        return None
    if type(value) not in (str, type(Path())):
        return {"unsupported_type": type(value).__qualname__}
    text = str(value)
    return {
        "basename": Path(text).name,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _thread_witness(actor, current):
    if actor is None:
        return None
    if not isinstance(actor, threading.Thread):
        return {"unsupported_type": type(actor).__qualname__}
    return {
        "object_id": id(actor),
        "ident": actor.ident,
        "name": actor.name,
        "is_current": actor is current,
        "alive": actor.is_alive(),
    }


def _actual_lease_owners(app, storage, repositories, captured):
    """Observe actual registered links; unknown owners receive no close authority."""
    current = threading.current_thread()
    known = {
        name: inspect.getattr_static(app, name, None)
        for name in (
            "chachanotes_db",
            "media_db",
            "prompts_db",
            "notes_service",
            "prompts_service",
            "_console_chat_store",
            "_console_agent_bridge",
            "_console_chat_controller",
        )
    }
    for name in (
        "notes_service",
        "prompts_service",
        "_console_chat_store",
        "_console_agent_bridge",
        "_console_chat_controller",
    ):
        owner = known[name]
        if owner is not None:
            for field in ("db", "_db", "runs_db", "_runs_db"):
                known[name + "." + field] = inspect.getattr_static(owner, field, None)
    references = list(known.values()) + [current]
    with storage._lock:
        participants = tuple(repositories._installed_repositories)
        # These strong rows include every thread; no CURRENT-thread filtering.
        links = []
        for participant in participants:
            repository = participant.repository()
            for connection, lease in tuple(participant.connections.items()):
                links.append((participant, repository, connection, lease))
        registered_startups = set(storage._startups.values())
        ordinary = tuple(storage._live_leases - registered_startups)
        captured_leases = {item[3] for item in captured}
        rows = []
        for lease in ordinary:
            thread = inspect.getattr_static(lease, "resource_thread", None)
            path = inspect.getattr_static(lease, "resource_path", None)
            key = inspect.getattr_static(lease, "_key", None)
            hold = storage._holds.get(key) if key is not None else None
            matches = []
            for participant, repository, connection, connected_lease in links:
                if connected_lease is not lease:
                    continue
                try:
                    transaction = sqlite3.Connection.in_transaction.__get__(connection)
                    descriptor_error = None
                except (sqlite3.ProgrammingError, TypeError) as error:
                    transaction, descriptor_error = None, type(error).__name__
                matches.append(
                    {
                        "participant_object_id": id(participant),
                        "repository_object_id": id(repository)
                        if repository is not None
                        else None,
                        "connection_object_id": id(connection),
                        "participant_owner_id": participant.owner_id,
                        "participant_path": _path_witness(participant.path),
                        "repository_type": type(repository).__qualname__,
                        "exact_repository_participant": repository is not None
                        and inspect.getattr_static(
                            repository, "_maintenance_participant", None
                        )
                        is participant,
                        "exact_app_pointer_matches": [
                            name
                            for name, value in known.items()
                            if value is not None and value is repository
                        ],
                        "resource_thread": _thread_witness(thread, current),
                        "path_matches_participant": path == participant.path,
                        "connection_in_transaction": transaction,
                        "connection_descriptor_error_type": descriptor_error,
                    }
                )
                references.extend((participant, repository, connection, lease))
            key_receipt = (
                None
                if key is None
                else {
                    "pid": key[0],
                    "pid_is_current": key[0] == os.getpid(),
                    "path": _path_witness(key[1]),
                    "exact_hold_present": hold is not None,
                    "hold_object_id": id(hold) if hold is not None else None,
                    "hold_key_matches_lease": hold.key == key
                    if hold is not None
                    else None,
                    "hold_thread": _thread_witness(hold.thread, current)
                    if hold is not None
                    else None,
                    "hold_ready": hold.ready.is_set() if hold is not None else None,
                    "hold_stopping": hold.stop.is_set() if hold is not None else None,
                    "hold_error_type": type(hold.error).__name__
                    if hold is not None and hold.error is not None
                    else None,
                    "hold_count": hold.count if hold is not None else None,
                }
            )
            rows.append(
                {
                    "lease_object_id": id(lease),
                    "lease_type": type(lease).__qualname__,
                    "captured_current_thread_sql_owner": lease in captured_leases,
                    "resource_path": _path_witness(path),
                    "resource_thread": _thread_witness(thread, current),
                    "resource_close_failed": inspect.getattr_static(
                        lease, "resource_close_failed", None
                    ),
                    "key": key_receipt,
                    "registered_sql_links": matches,
                }
            )
            references.extend((lease, thread, hold))
            if hold is not None:
                references.append(hold.thread)
        return {
            "ordinary_count": len(ordinary),
            "captured_current_thread_sql_count": len(captured_leases),
            "unmatched_count": len(set(ordinary) - captured_leases),
            "rows": rows,
            "registered_participants_count": len(participants),
            "all_thread_registered_connection_count": len(links),
            "metadata_only_no_unknown_close": True,
        }, tuple(references)


async def _until(predicate, *, timeout=8):
    """Keep the original Home test's eight-second condition deadline."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert (
            asyncio.get_running_loop().time() < deadline
        ), "readiness lifecycle did not settle"
        await asyncio.sleep(0.01)


@pytest.mark.ui
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@private_profile_test
async def test_hidden_console_starts_no_credential_config_reader(request):
    """Native visible expiry -> actual Home -> same-instance visible expiry."""
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )

    assert config.save_settings_to_cli_config(
        {
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "general": {"default_tab": "home"},
            "model_catalog": {"auto_refresh_enabled": False},
        }
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    # This fresh private child has no borrowed SQL owner before the actual App.
    with storage._lock:
        assert not (storage._live_leases - set(storage._startups.values()))
        assert not storage._operations and not storage._raw_operations
    app = TldwCli()
    from Tests.UI._hidden_sql_worker_calls import OriginalNotesWorkerCalls

    sql_observation = OriginalNotesWorkerCalls(app.chachanotes_db, storage)
    sql_observation.start()
    observation = None
    original_bindings = []
    retained_modules = []
    facts = {"completed": False}
    try:
        async with app.run_test(size=(170, 48)) as pilot:
            await _boot_settled(app, pilot)
            await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
            console = app.screen
            assert str(app.chachanotes_db.db_path) != ":memory:"
            # Screen identity is visible before the deferred attach refresh ends.
            # The stock credential timer starts only after that reconciliation.
            await _until(lambda: console._console_attach_reconciled)
            assert console._console_credential_poll_timer is not None
            projection = ConsoleReadinessConfigProjection.for_screen(console)
            assert type(projection) is ConsoleReadinessConfigProjection
            assert projection.max_age == 1.0  # Observe the stock cadence unchanged.
            assert await asyncio.wait_for(projection.warm(), 8)
            await _until(lambda: not projection.pending)
            defining = inspect.getattr_static(
                ConsoleReadinessConfigProjection, "for_screen"
            )
            assert type(defining) is classmethod
            expected_reader_codes = [
                item
                for item in defining.__func__.__code__.co_consts
                if type(item) is CodeType and item.co_name == "read_current"
            ]
            assert len(expected_reader_codes) == 1
            assert projection.read_current.__code__ is expected_reader_codes[0]
            from tldw_chatbook.UI.Console_Modules import (
                console_spend_projection as spend,
            )
            from tldw_chatbook.UI.Screens import chat_screen

            assert type(console) is chat_screen.ChatScreen
            credential = inspect.getattr_static(
                chat_screen.ChatScreen, "_poll_console_credential_readiness"
            )
            assert type(credential) is FunctionType
            assert credential.__globals__ is vars(spend)
            expected_wrapper_codes = [
                item
                for item in spend.console_readiness_presentation.__code__.co_consts
                if type(item) is CodeType and item.co_name == "wrapped"
            ]
            assert len(expected_wrapper_codes) == 1
            assert credential.__code__ is expected_wrapper_codes[0]
            closures = dict(
                zip(credential.__code__.co_freevars, credential.__closure__ or ())
            )
            credential_body = closures["function"].cell_contents
            assert type(credential_body) is FunctionType
            assert credential.__dict__.get("__wrapped__") is credential_body
            assert credential_body.__globals__ is vars(chat_screen)
            assert (
                credential_body.__code__.co_qualname
                == "ChatScreen._poll_console_credential_readiness"
            )
            assert (
                Path(credential_body.__code__.co_filename).resolve()
                == Path(chat_screen.__file__).resolve()
            )

            if os.name == "nt":
                from tldw_chatbook.Utils import windows_files

                _Native = windows_files._Native

                native = inspect.getattr_static(_Native, "open_handle")
                assert type(native) is FunctionType
                assert native.__module__ == "tldw_chatbook.Utils.windows_files"
                native_code = native.__code__
                native_kind = "original_windows_Native_open_handle_calls"
                native_owner, native_name = _Native, "open_handle"
            else:
                from tldw_chatbook.Backup_Recovery import storage_admission

                native = inspect.getattr_static(storage_admission, "_observe_stamps")
                assert type(native) is FunctionType
                native_code = native.__code__
                native_kind = "original_posix_storage_observe_stamps_calls_not_descriptor_open_count"
                native_owner, native_name = storage_admission, "_observe_stamps"
            from tldw_chatbook.Backup_Recovery import (
                config_participants as participants,
                raw_participants as raw,
                storage_admission as storage,
            )

            original_bindings = [
                (owner, name, inspect.getattr_static(owner, name))
                for owner, name in (
                    (config, "load_settings"),
                    (config, "current_config_identity"),
                    (config, "load_cli_config_and_ensure_existence"),
                    (spend, "ConsoleReadinessConfigProjection"),
                    (spend, "console_readiness_presentation"),
                    (ConsoleReadinessConfigProjection, "for_screen"),
                    (ConsoleReadinessConfigProjection, "run"),
                    (chat_screen, "ChatScreen"),
                    (chat_screen.ChatScreen, "_poll_console_credential_readiness"),
                    (participants, "operation"),
                    (participants, "checked_config_identity"),
                    (raw, "_participant_identity"),
                    (storage, "acquire_storage"),
                    (native_owner, native_name),
                )
            ]
            retained_modules = [
                (module.__name__, module)
                for module in (config, spend, chat_screen, participants, raw, storage)
            ]
            observation = ReadinessCalls(
                projection,
                native_code,
                credential_body,
                credential.__code__,
                native_kind,
            )
            observation.start()

            # Only the existing 0.25-second timer supplies this post-TTL read.
            observation.phase = "visible_expiry"
            await _until(lambda: bool(observation.completed("visible_expiry")))
            await _until(lambda: not projection.pending)
            visible = observation.completed("visible_expiry")
            assert any(
                row["native_boundary_calls"] > 0 and row["checked_identity_reached"]
                for row in visible
            )
            assert any(row["actor_existed_before_monitor"] for row in visible)
            facts["visible_native_positive"] = True

            await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
            assert app.screen is not console
            assert console.is_attached and not console.is_current
            assert not console._console_attach_reconciled
            # Drain any reader admitted during navigation before the hidden window.
            await _until(lambda: not projection.pending)
            observation.phase = "hidden_home"
            await asyncio.sleep(projection.max_age + 0.5)
            hidden = [
                row
                for row in observation.receipt()["rows"]
                if row["phase"] == "hidden_home"
            ]
            facts["hidden_reader_starts"] = len(hidden)
            schedules = observation.receipt()["schedules"]
            facts["hidden_original_credential_schedules"] = sum(
                row["phase"] == "hidden_home" and row["original_credential_wrapper"]
                for row in schedules
            )
            observation.phase = "resume_navigation"
            await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
            assert app.screen is console
            await _until(
                lambda: console._console_attach_reconciled and not projection.pending
            )
            observation.phase = "resumed_expiry"
            await _until(lambda: bool(observation.completed("resumed_expiry")))
            assert any(
                row["native_boundary_calls"] > 0 and row["checked_identity_reached"]
                for row in observation.completed("resumed_expiry")
            )
            facts["same_instance_resume_native_positive"] = True
            assert not hidden, (
                "a suspended reusable Console started a credential config reader "
                f"while Home was current ({len(hidden)} actual reader calls)"
            )
            facts["completed"] = True
    finally:
        sql_observation.stop()
        sql_callers = sql_observation.receipt()
        # One bounded actual census after original teardown, without a new drain,
        # global clear, SQL-close inference or extra application deadline.
        from tldw_chatbook.Backup_Recovery import (
            raw_participants as raw,
            storage_admission as storage,
        )

        # The private App owns process-lifetime cached SQL connections. Its
        # original screen shutdown settles producers, but process exit normally
        # closes these caches. Use the original maintenance cleanup here before
        # inspecting native retirement; do not clear or exclude unknown leases.
        from tldw_chatbook.Backup_Recovery import participants as repositories

        retained_sql = []
        physically_closed_sql = 0
        ownership_before_cleanup = None
        lease_diagnostic_references = None
        try:
            with storage._lock:
                for participant in tuple(repositories._installed_repositories):
                    repository = participant.repository()
                    if repository is None:
                        continue
                    for connection, lease in participant.connections.items():
                        if lease.resource_thread is not threading.current_thread():
                            continue
                        assert lease in storage._live_leases
                        assert repository._maintenance_participant is participant
                        assert (
                            repository.db_path
                            == participant.path
                            == lease.resource_path
                        )
                        assert not connection.in_transaction
                        retained_sql.append(
                            (repository, participant, connection, lease)
                        )
            ownership_before_cleanup, lease_diagnostic_references = (
                _actual_lease_owners(app, storage, repositories, retained_sql)
            )
            requested = os.environ.get("TLDW_HIDDEN_READINESS_RESULT")
            if requested:
                # Use the already declared result path; no extra output namespace.
                Path(requested).write_text(
                    json.dumps(
                        {
                            "phase": "before_expected_sql_coverage_assertion",
                            "facts": facts,
                            "ownership_before_cleanup": ownership_before_cleanup,
                            "no_unknown_lease_closed": True,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            with storage._lock:
                assert (
                    {item[3] for item in retained_sql}
                    == (storage._live_leases - set(storage._startups.values()))
                ), "private App ordinary leases are not all covered by actual SQL owners"
            cleanup_pause = storage._begin_local_pause()
            try:
                repositories._retire_current_thread_caches(cleanup_pause)
            finally:
                cleanup_pause.resume()
            for repository, participant, connection, lease in retained_sql:
                assert repository._maintenance_participant is participant
                try:
                    sqlite3.Connection.in_transaction.__get__(connection)
                except sqlite3.ProgrammingError:
                    pass
                else:
                    raise AssertionError(
                        "private App SQL cache remains physically open"
                    )
                assert lease not in storage._live_leases
                physically_closed_sql += 1

        finally:
            cleanup_failure = sys.exception()
            with storage._lock:
                genuine_startups = set()
                for key, lease in storage._startups.items():
                    hold = storage._holds.get(key)
                    if (
                        type(lease) is storage.StorageLease
                        and lease in storage._live_leases
                        and lease._key == key
                        and key[0] == os.getpid()
                        and hold is not None
                        and hold.ready.is_set()
                        and hold.error is None
                        and not hold.stop.is_set()
                    ):
                        genuine_startups.add(lease)
                ordinary_census = {
                    "ordinary": len(storage._live_leases - genuine_startups),
                    "pending": len(storage._pending_acquisitions),
                    "core": len(storage._operations),
                    "raw": len(storage._raw_operations),
                    "raw_states": len(raw._states),
                    "retiring": len(storage._retiring_holds),
                }
                startup_census = {
                    "registered": len(storage._startups),
                    "genuine_live": len(genuine_startups),
                }
            try:
                if observation is not None:
                    observation.stop()
            finally:
                if observation is not None:
                    # run_test has already completed its original app teardown here.
                    receipt = observation.receipt()
                    receipt.update(facts)
                    receipt["private_app_sql_caches_physically_closed"] = (
                        physically_closed_sql
                    )
                    receipt["ownership_before_cleanup"] = ownership_before_cleanup
                    receipt["actual_sql_acquisition_callers"] = sql_callers
                    receipt["cleanup_failure_type"] = (
                        type(cleanup_failure).__name__
                        if cleanup_failure is not None
                        else None
                    )
                    receipt["final_census_zero_required"] = True
                    receipt["final_census_zero"] = not any(ordinary_census.values())
                    receipt["diagnostic_owner_references_retained"] = (
                        lease_diagnostic_references is not None
                    )
                    receipt["after_original_app_teardown_census"] = ordinary_census
                    receipt["intentional_startup_census"] = startup_census
                    receipt["original_bindings_unchanged"] = all(
                        inspect.getattr_static(owner, name) is original
                        for owner, name, original in original_bindings
                    ) and all(
                        sys.modules.get(name) is module
                        for name, module in retained_modules
                    )
                    receipt["original_reader_unchanged"] = (
                        projection.read_current is observation.reader
                    )
                    requested = os.environ.get("TLDW_HIDDEN_READINESS_RESULT")
                    if requested:
                        Path(requested).write_text(
                            json.dumps(receipt, indent=2), encoding="utf-8"
                        )
        if observation is not None:
            assert receipt["overflow"] == 0
            assert not receipt[
                "live_reader_ids"
            ], "reader start lacks a healthy matching return; evidence is invalid"
            assert (
                receipt["original_bindings_unchanged"]
                and receipt["original_reader_unchanged"]
            )
            assert (
                receipt["restoration"]
                == "local_events_callbacks_removed_tool_freed_global0"
            )
        assert all(count == 0 for count in ordinary_census.values()), ordinary_census
