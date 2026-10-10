"""Original legacy approval facts remain visible while fresh config is pending.

Evidence draft only. The root owns installation and real private-child execution.
"""

import asyncio
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import FunctionType
import inspect

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.UI.app_factory import (
    drain_active_service_patches,
    drain_created_dirs,
    persist_seeded_config,
)
from Tests.UI import test_console_pending_interrupt_projection as original
from Tests.UI import test_console_turn_navigation_continuity as navigation
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import (
    config_participants,
    raw_participants,
    storage_admission,
)
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector


class _OriginalColdRead(OriginalStorageUnitObserver):
    """Use the existing source records for one exact checked worker return."""

    def __init__(self, projection):
        super().__init__({}, lambda: False, lambda _unit: None)
        self.projection = projection
        self.screen = projection.screen
        self.reader = inspect.getattr_static(projection, "read_current")
        assert type(self.reader) is FunctionType
        assert self.reader.__globals__ is vars(spend)
        cells = dict(zip(self.reader.__code__.co_freevars, self.reader.__closure__))
        assert cells["projection"].cell_contents is projection
        assert cells["screen"].cell_contents is self.screen
        self.functions = (
            config_participants.checked_config_identity,
            spend.ConsoleReadinessConfigProjection.run,
            spend._checked_display_status,
            spend.run_console_config_sync,
            self.reader,
        )
        self.slots.extend(
            (
                (config_participants, "checked_config_identity", self.functions[0]),
                (spend.ConsoleReadinessConfigProjection, "run", self.functions[1]),
                (spend, "_checked_display_status", self.functions[2]),
                (spend, "run_console_config_sync", self.functions[3]),
                (projection, "read_current", self.reader),
                (projection, "screen", self.screen),
                (config, "_config_participants", config_participants),
                (storage_admission, "_lock", storage_admission._lock),
            )
        )
        for function in self.functions:
            self._pin(function)
        operation = inspect.getattr_static(config_participants, "operation")
        operation_body = inspect.getattr_static(operation, "__wrapped__")
        self._pin(operation)
        self._pin(operation_body)
        operation_cells = dict(
            zip(operation.__code__.co_freevars, operation.__closure__ or ())
        )
        assert operation_cells["func"].cell_contents is operation_body
        self.slots.extend(
            (
                (config_participants, "operation", operation),
                (operation, "__wrapped__", operation_body),
            )
        )
        # These are actual source dependencies/descriptor slots of the known
        # standard factory and reader, not arbitrary callable unwrapping.
        from tldw_chatbook.UI.Screens import chat_screen

        for owner, name in (
            (spend.ConsoleReadinessConfigProjection, "for_screen"),
            (spend.ConsoleReadinessConfigProjection, "_key"),
            (spend.ConsoleReadinessConfigProjection, "_refresh"),
            (spend, "_standard_readiness_key"),
            (config, "current_config_identity"),
            (config, "_get_effective_config_path"),
            (config, "load_settings"),
            (chat_screen, "load_settings"),
            (config_participants, "binding"),
            (raw_participants, "_check"),
            (raw_participants, "_participant_identity"),
        ):
            descriptor = inspect.getattr_static(owner, name)
            self.slots.append((owner, name, descriptor))
            function = (
                descriptor.__func__ if type(descriptor) is classmethod else descriptor
            )
            self._pin(function)
        assert chat_screen.load_settings is config.load_settings
        self.sources = {record[1]: record[-1] for record in self.modules.values()}
        self.module_files = {
            name: record[0].__file__ for name, record in self.modules.items()
        }
        self.entered, self.release = threading.Event(), threading.Event()
        self.operation, self.leases, self.thread = None, (), None
        self.expected_source = None
        self.rows = []
        self.owned_tool = False
        self.retired = False
        self.retirement = {}
        self.codes = {
            function.__code__: function.__qualname__ for function in self.functions[:4]
        }
        self.tool_name = "original-pending-config-defer"
        self.selected_events = (self.monitor.events.PY_RETURN,)
        self.enabled_codes = set()

    def _current(self, *, source_bytes=False):
        try:
            for (
                function,
                code,
                namespace,
                defaults,
                kwdefaults,
                items,
                closure,
                cells,
            ) in self.pins:
                if not (
                    function.__code__ is code
                    and function.__globals__ is namespace
                    and function.__defaults__ is defaults
                    and function.__kwdefaults__ is kwdefaults
                    and len(function.__kwdefaults__ or {}) == len(items)
                    and all(
                        (function.__kwdefaults__ or {}).get(key) is value
                        for key, value in items
                    )
                    and function.__closure__ is closure
                    and all(cell.cell_contents is value for cell, value in cells)
                ):
                    return False
            if not all(
                inspect.getattr_static(owner, name) is value
                for owner, name, value in self.slots
            ):
                return False
            for name, (
                module,
                path,
                spec,
                origin,
                loader,
                spec_loader,
                digest,
            ) in self.modules.items():
                if not (
                    sys.modules.get(name) is module
                    and module.__spec__ is spec
                    and spec.origin == origin
                    and module.__loader__ is loader
                    and spec.loader is spec_loader
                    and module.__file__ == self.module_files[name]
                ):
                    return False
                if source_bytes and not (
                    Path(module.__file__).resolve() == path
                    and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                ):
                    return False
            return True
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            return False

    def _returned(self, code, _offset, value):
        if not self.active:
            return
        frame = self._frame(code)
        if frame.f_code is not code or not self._current():
            self.invalid.append("original_body_binding_changed")
            return
        if code is self.functions[0].__code__:
            parent = frame.f_back
            if (
                self.entered.is_set()
                or parent is None
                or parent.f_code is not self.reader.__code__
                or parent.f_globals is not self.reader.__globals__
                or frame.f_locals.get("source") is not config
                or self.expected_source is None
                or value != self.expected_source
                or parent.f_locals.get("request")[2][0] != self.expected_source
            ):
                return
            active = frame.f_locals.get("active")
            with storage_admission._lock:
                state = raw_participants._states.get(active)
                assert state is not None and state.source is config
                assert state.route in {"config", "config_snapshot"}
                assert state.leases and all(
                    lease in storage_admission._live_leases for lease in state.leases
                )
                assert raw_participants._local.operation is active
                self.operation, self.leases = active, tuple(state.leases)
            self.thread = threading.current_thread()
            assert self.thread is not threading.main_thread()
            self.rows.append(
                {"site": "checked_worker_return", "live_leases": len(self.leases)}
            )
            # No selected frame or caller frame remains in this callback during
            # the held original return. Only explicit resource/actor facts do.
            del state, active, parent, frame, value
            self.entered.set()
            if not self.release.wait(10):
                self.invalid.append("held_original_reader_release_timeout")
            return
        if code is self.functions[1].__code__:
            if frame.f_locals.get("self") is self.projection:
                self.rows.append(
                    {
                        "site": "projection_run",
                        "result": value is True,
                        "cold_owner": frame.f_locals.get("current") is False,
                        "pending": self.projection.pending,
                    }
                )
        elif code is self.functions[2].__code__:
            if frame.f_locals.get("projection") is self.projection:
                self.rows.append({"site": "checked_status", "result": value})
        elif code is self.functions[3].__code__:
            if frame.f_locals.get("checked_projection") is self.projection:
                self.rows.append(
                    {
                        "site": "native_config_sync",
                        "result": value is True,
                        "entered": frame.f_locals.get("entered") is True,
                        "maintenance_paused": frame.f_locals.get("maintenance_paused")
                        is True,
                    }
                )
        if len(self.rows) > 64:
            self.invalid.append("bounded_branch_receipt_overflow")

    def start(self):
        assert self._current(source_bytes=True)
        for tool in range(5, 0, -1):
            if tool == self.monitor.DEBUGGER_ID:
                continue
            try:
                self.monitor.use_tool_id(tool, self.tool_name)
            except ValueError:
                continue
            self.tool, self.owned_tool = tool, True
            break
        assert self.tool is not None
        assert self.monitor.get_events(self.tool) == 0
        assert all(
            self.monitor.get_local_events(self.tool, code) == 0 for code in self.codes
        )
        event, callback = self.monitor.events.PY_RETURN, self._returned
        previous = self.monitor.register_callback(self.tool, event, callback)
        if previous is not None:
            self.monitor.register_callback(self.tool, event, previous)
            raise RuntimeError("monitoring_callback_not_unowned")
        self.registered[event] = callback
        for code in self.codes:
            self.monitor.set_local_events(self.tool, code, event)
            self.enabled_codes.add(code)
            assert self.monitor.get_local_events(self.tool, code) == event
        assert self.monitor.get_events(self.tool) == 0
        self.active = self.installed = True

    def stop(self):
        if not self.owned_tool:
            return
        failures = []
        source_current = self._current(source_bytes=True)
        local_zero = callbacks_none = global_zero = reclaimed = False
        for code in self.enabled_codes:
            try:
                self.monitor.set_local_events(self.tool, code, 0)
            except BaseException as error:
                failures.append("local_disable:" + type(error).__name__)
        try:
            local_zero = all(
                self.monitor.get_local_events(self.tool, code) == 0
                for code in self.codes
            )
            if not local_zero:
                failures.append("local_mask_not_zero")
        except BaseException as error:
            failures.append("local_validation:" + type(error).__name__)
        callbacks_none = True
        for event in self.selected_events:
            expected = self.registered.get(event)
            try:
                previous = self.monitor.register_callback(self.tool, event, None)
                if previous is not expected:
                    # This callback was never ours (including refused startup).
                    # Preserve it and report exclusion rather than adopting it.
                    self.monitor.register_callback(self.tool, event, previous)
                    callbacks_none = False
                    failures.append("callback_owner_changed")
                elif self.monitor.register_callback(self.tool, event, None) is not None:
                    callbacks_none = False
                    failures.append("callback_not_none")
            except BaseException as error:
                callbacks_none = False
                failures.append("callback_disable:" + type(error).__name__)
        try:
            global_zero = self.monitor.get_events(self.tool) == 0
            if not global_zero:
                failures.append("global_mask_not_zero")
        except BaseException as error:
            failures.append("global_validation:" + type(error).__name__)
        finally:
            try:
                self.monitor.set_events(self.tool, 0)
            except BaseException as error:
                failures.append("global_disable:" + type(error).__name__)
            finally:
                try:
                    self.monitor.free_tool_id(self.tool)
                    self.owned_tool = False
                except BaseException as error:
                    failures.append("tool_free:" + type(error).__name__)
        # Positive same-slot physical reclamation: name absence alone is not
        # proof that stale callbacks or local masks disappeared.
        proof_owned = False
        try:
            assert self.monitor.get_tool(self.tool) is None
            self.monitor.use_tool_id(self.tool, "pending-config-retirement-proof")
            proof_owned = True
            assert self.monitor.get_events(self.tool) == 0
            assert all(
                self.monitor.get_local_events(self.tool, code) == 0
                for code in self.codes
            )
            for event in self.selected_events:
                previous = self.monitor.register_callback(self.tool, event, None)
                if previous is not None:
                    self.monitor.register_callback(self.tool, event, previous)
                    raise RuntimeError("reclaimed_callback_not_none")
            reclaimed = True
        except BaseException as error:
            failures.append("physical_reclaim:" + type(error).__name__)
        finally:
            if proof_owned:
                try:
                    self.monitor.free_tool_id(self.tool)
                    assert self.monitor.get_tool(self.tool) is None
                except BaseException as error:
                    reclaimed = False
                    failures.append("proof_free:" + type(error).__name__)
        self.retirement = dict(
            source_current=source_current,
            local_zero=local_zero,
            callbacks_none=callbacks_none,
            global_zero=global_zero,
            physical_reclaim=reclaimed,
            failures=failures,
        )
        self.invalid.extend(failures)
        if not source_current:
            self.invalid.append("source_not_current_at_retirement")
        self.retired = (
            local_zero and callbacks_none and global_zero and reclaimed and not failures
        )
        if self.retired:
            self.active = False


class _CheckedNavigationFactory(OriginalStorageUnitObserver):
    """Derive the original fixture with only its explicit marker removal omitted."""

    def __init__(self):
        super().__init__({}, lambda: False, lambda _unit: None)
        helpers = (
            navigation._build_navigation_app,
            navigation._build_test_app,
            navigation._attach_real_dbs,
            navigation._configure_native_ready_console,
            persist_seeded_config,
        )
        for function in helpers:
            self._pin(function)
        self.slots = [
            (navigation, "_build_navigation_app", helpers[0]),
            (navigation, "_build_test_app", helpers[1]),
            (navigation, "_attach_real_dbs", helpers[2]),
            (navigation, "_configure_native_ready_console", helpers[3]),
            (navigation, "_TwoChunkGateway", navigation._TwoChunkGateway),
        ]
        self.module_files = {
            name: record[0].__file__ for name, record in self.modules.items()
        }
        self.source_hashes = {
            str(record[1]): record[-1] for record in self.modules.values()
        }
        function = helpers[0]
        path = Path(function.__code__.co_filename).resolve()
        tree = ast.parse(path.read_bytes(), filename=str(path))
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == function.__name__
            and node.lineno == function.__code__.co_firstlineno
        )
        assert not node.decorator_list
        derived = copy.deepcopy(node)
        removals = [
            node
            for node in derived.body
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "pop"
                and isinstance(node.value.func.value, ast.Attribute)
                and node.value.func.value.attr == "app_config"
                and isinstance(node.value.func.value.value, ast.Name)
                and node.value.func.value.value.id == "app"
                and len(node.value.args) == 2
                and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value == "logging"
                and isinstance(node.value.args[1], ast.Constant)
                and node.value.args[1].value is None
                and not node.value.keywords
            )
        ]
        assert len(removals) == 1
        derived.body.remove(removals[0])
        expected = copy.deepcopy(node)
        expected.body = [
            item for item in expected.body if ast.dump(item) != ast.dump(removals[0])
        ]
        assert ast.dump(expected, include_attributes=False) == ast.dump(
            derived, include_attributes=False
        )
        scope = dict(function.__globals__)
        exec(
            compile(
                ast.Module(body=[derived], type_ignores=[]),
                str(path),
                "exec",
                dont_inherit=True,
            ),
            scope,
        )
        self.derived = scope[function.__name__]
        self.namespace = function.__globals__
        self.helper_slots = {
            name: scope[name]
            for name in (
                "_build_test_app",
                "_attach_real_dbs",
                "_configure_native_ready_console",
                "_TwoChunkGateway",
            )
        }
        self.facts = {"only_logging_removal_omitted_ast": True, "marker_removals": 1}

    def current(self):
        return _OriginalColdRead._current(self, source_bytes=True) and all(
            self.namespace.get(name) is original
            for name, original in self.helper_slots.items()
        )

    def build(self, tmp_path):
        assert self.current()
        assert config.save_setting_to_cli_config("splash_screen", "enabled", False)
        app, gateway = self.derived(tmp_path)
        assert self.current()
        # Original supported persistence keeps the real fresh read equivalent
        # to the original provider/default/agent test setup. No marker/cache
        # restoration is performed and no authorizing callback is replaced.
        persist_seeded_config(app, "chat_defaults", "api_settings.llama_cpp", "console")
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        assert ChatScreen._console_config_snapshot_is_disk_loaded(app.app_config)
        assert ChatScreen._console_config_snapshot_is_disk_loaded(
            config.load_settings()
        )
        gateway.cached_context_window = (
            lambda settings: original.resolve_context_window(
                settings.provider, settings.model or ""
            )
        )
        self.facts.update(
            stock_app_markers=True,
            actual_loaded_markers=True,
            supported_provider_defaults_persisted=True,
        )
        assert self.current()
        return app


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@private_profile_test
async def test_original_direct_control_pending_facts_paint_before_cold_config_retires(
    request, tmp_path
):
    """Keep the actual card and registry count live through a genuine cold owner."""
    factory = _CheckedNavigationFactory()
    app = factory.build(tmp_path)
    workers, violations = [], []
    gate = None
    branch_facts = []
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            controller = None
            try:
                console, controller, store, session_id = await original._seed_console(
                    app, pilot
                )
                original._start_live_turn(console, controller, store, session_id)
                projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
                assert await projection.warm()
                console._sync_console_rail_and_controls()
                await pilot.pause()
                inspector = console.query_one(
                    "#console-run-inspector-state", ConsoleRunInspector
                )
                assert inspector.state.pending_approval_count == 0
                gate = _OriginalColdRead(projection)
                gate.start()
                before = config.current_config_identity()
                # Actual public write invalidates the checked owner. No forced clock,
                # private cache field or production callback is replaced.
                assert config.save_setting_to_cli_config(
                    "splash_screen", "enabled", False
                )
                assert config.current_config_identity() != before
                gate.expected_source = config.current_config_identity()
                sync_result = console._sync_console_rail_and_controls()
                await original._wait(pilot, gate.entered.is_set)
                assert projection.pending and gate.operation in raw_participants._states
                assert all(
                    lease in storage_admission._live_leases for lease in gate.leases
                )
                assert any(
                    row["site"] == "projection_run" and row["cold_owner"]
                    for row in gate.rows
                ), "the original cold-owner branch was not reached"
                assert console._console_pending_approval_count() == 0
                assert inspector.state.pending_approval_count == 0
                branch_facts.append(
                    {
                        "phase": "cold_before_arm",
                        "in_memory_count": 0,
                        "inspector_count": inspector.state.pending_approval_count,
                        "sync_result": sync_result,
                    }
                )
                worker, result = original._arm(
                    controller, None, call=original._risk_row()
                )
                workers.append(worker)
                await original._wait(
                    pilot,
                    lambda: (
                        bool(list(console.query("#chat-approval-card")))
                        and console.query_one("#chat-approval-card").display
                        and bool(
                            console.query_one("#chat-approval-card")._batch_round_id
                        )
                    ),
                )
                round_id = console.query_one("#chat-approval-card")._batch_round_id
                assert (
                    round_id in controller._pending_approval_rounds
                    and worker.is_alive()
                )
                assert controller.pending_round_count(session_id) == 0
                assert (
                    console._task_resume_state.pending_approval["round_id"] == round_id
                )
                assert console._console_pending_approval_count() == 1

                sync_result = console._sync_console_control_bar()
                await pilot.pause()
                assert gate.entered.is_set() and projection.pending
                branch_facts.append({"sync_result": sync_result, "in_memory_count": 1})
                if inspector.state.pending_approval_count != 1:
                    violations.append(
                        "live legacy count1 deferred behind fresh configuration"
                    )
                rendered = "\n".join(
                    str(row.render()) for row in inspector.query(Static)
                )
                if "Approvals: 1 pending" not in rendered:
                    violations.append(
                        "mounted approval row did not paint current count"
                    )
                if not inspector.state.has_pending_approval:
                    violations.append("live Review affordance state remained false")

                assert projection.pending, "the held original checked reader retired too early"
                assert all(
                    lease in storage_admission._live_leases for lease in gate.leases
                )
                controller.resolve_pending_approval(
                    {"builtin__write_file": "deny"}, round_id=round_id
                )
                # Resolution retains its original fresh authority reads. Release
                # the test-held config lock before requiring worker completion.
                gate.release.set()
                await asyncio.wait_for(projection._settled.wait(), 10)
                await original._finish_worker(pilot, worker)
                assert result["decisions"] == {"builtin__write_file": "deny"}
                assert console._console_pending_approval_count() == 0
                console._sync_console_rail_and_controls()
                await pilot.pause()
                if (
                    inspector.state.pending_approval_count != 0
                    or inspector.state.has_pending_approval
                ):
                    violations.append(
                        "settled live count0 stale after fresh publication"
                    )
                assert gate.operation not in raw_participants._states
                assert all(
                    lease not in storage_admission._live_leases for lease in gate.leases
                )
                await pilot.pause()
                assert inspector.state.pending_approval_count == 0
                assert not gate.invalid
            finally:
                if gate is not None:
                    gate.release.set()
                    if projection.pending:
                        await asyncio.wait_for(projection._settled.wait(), 10)
                if workers:
                    await original._stop_workers(controller, workers, pilot)
    finally:
        if gate is not None:
            gate.release.set()
        try:
            drain_created_dirs()
            drain_active_service_patches()
        finally:
            if gate is not None:
                try:
                    gate.stop()
                finally:
                    (tmp_path / "pending-cold-config-branch.json").write_text(
                        json.dumps(
                            {
                                "rows": gate.rows,
                                "fixture": factory.facts,
                                "fixture_source_current": factory.current(),
                                "fixture_source_hashes": factory.source_hashes,
                                "invalid": gate.invalid,
                                "branches": branch_facts,
                                "violations": violations,
                                "monitor_retired": gate.retired,
                                "retirement": gate.retirement,
                                "held_raw_retired": gate.operation
                                not in raw_participants._states,
                                "held_leases_retired": all(
                                    lease not in storage_admission._live_leases
                                    for lease in gate.leases
                                ),
                                "source_hashes": {
                                    str(path): digest
                                    for path, digest in gate.sources.items()
                                },
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
    assert gate is not None and gate.retired and not gate.active
    assert not gate.invalid, gate.invalid
    assert not violations, violations
