"""Exact constructor owners for test_console_agent_controller only.

Keep the original live return ABI. This test utility changes no production
constructor, shutdown API, participant registration or global drain.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sqlite3
import sys
import threading
from types import FunctionType, ModuleType
from typing import Any

import pytest

from Tests.Performance.console_storage_unit_observer import (
    OriginalStorageUnitObserver,
    _nested,
    _shape,
)


class FixtureCallFailure:
    """Record this test's original call failure through pytest's report hook."""

    def __init__(self, node: Any) -> None:
        self.node, self.error = node, None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
        if item is self.node and call.when == "call" and call.excinfo is not None:
            self.error = call.excinfo.value
        yield


async def finish_owner_retirement(
    owners: Any, body_error: BaseException | None
) -> bool:
    """Keep an original call error primary; successful calls expose refusal."""
    try:
        await owners.retire()
    except BaseException:
        if body_error is None:
            raise
        body_error.add_note("Agent fixture cleanup could not prove owned retirement")
        return False
    return True


class AgentFixtureOwners(OriginalStorageUnitObserver):
    """Retain original factory births, not a later mutable App-field lookup."""

    def __init__(self, module: Any, *, pytest_config: Any = None) -> None:
        super().__init__({}, False, lambda _name: None)
        self.test_module = module
        self.pytest_config = pytest_config
        self.pytest_rewrite_context = None
        self.creator = threading.current_thread()
        self.owner_loop = asyncio.get_running_loop()
        self.roles: dict[Any, str] = {}
        self.constructors: dict[str, tuple[Any, Any, Any, Any]] = {}
        self.births: dict[int, tuple[Any, str, Any]] = {}
        self.runtime_births: dict[int, tuple[Any, Any, Any]] = {}
        self.runtime_watchers: dict[int, Any] = {}
        self.apps: list[tuple[Any, Any, Any]] = []
        self.hosts: list[tuple[Any, Any]] = []
        self.workers: list[tuple[Any, Any, Any, Any, Any]] = []
        self.databases: list[tuple[Any, Any, Any, Any, Any, Any]] = []
        self.notes_births: dict[int, tuple[Any, Any, Any, Any, Any, Any, Any]] = {}
        self.notes_databases: list[tuple[Any, Any, Any, Any, Any, Any, Any]] = []
        self.runtime_databases: dict[int, tuple[Any, Any, Any, Any, Any, Any]] = {}
        self.birth_errors: list[str] = []
        self.retirement_task: asyncio.Task[None] | None = None
        self.source_receipt: dict[str, Any] | None = None

    def _pin(self, function):
        if type(function) is not FunctionType or function.__globals__ is not vars(
            self.test_module
        ):
            return super()._pin(function)
        from _pytest.assertion.rewrite import AssertionRewritingHook

        loader = self.test_module.__loader__
        if type(loader) is not AssertionRewritingHook:
            assert type(loader) is SourceFileLoader
            return super()._pin(function)
        return self._pin_original_pytest_controller(function, loader)

    def _pin_original_pytest_controller(self, function, loader):
        """Compare only this exact module against its original active rewrite."""
        from _pytest import assertion
        from _pytest.assertion import rewrite
        from _pytest.config import Config

        module = self.test_module
        path = Path(module.__file__).resolve()
        config = self.pytest_config
        assert module.__name__ == "Tests.UI.test_console_agent_controller"
        assert path.name == "test_console_agent_controller.py"
        assert type(module) is ModuleType and sys.modules[module.__name__] is module
        assert (
            module.__spec__.loader is loader
            and module.__spec__.origin == module.__file__
        )
        assert (
            type(loader) is rewrite.AssertionRewritingHook and loader in sys.meta_path
        )
        assert type(config) is Config and loader.config is config
        assert Path(loader._rewritten_names[module.__name__]).resolve() == path
        state = config.stash[rewrite.assertstate_key]
        assert type(state) is assertion.AssertionState
        assert state.mode == "rewrite" and state.hook is loader
        assert function.__globals__ is vars(module)
        assert Path(function.__code__.co_filename).resolve() == path
        data = path.read_bytes()

        context = getattr(self, "pytest_rewrite_context", None)
        if context is None:
            slots = [
                (rewrite, "AssertionRewritingHook", rewrite.AssertionRewritingHook),
                (rewrite, "AssertionRewriter", rewrite.AssertionRewriter),
                (rewrite, "_rewrite_test", rewrite._rewrite_test),
                (rewrite, "rewrite_asserts", rewrite.rewrite_asserts),
                (rewrite, "assertstate_key", rewrite.assertstate_key),
                (assertion, "AssertionState", assertion.AssertionState),
                (
                    assertion.AssertionState,
                    "__init__",
                    assertion.AssertionState.__init__,
                ),
                (Config, "getini", inspect.getattr_static(Config, "getini")),
                (Config, "getoption", inspect.getattr_static(Config, "getoption")),
            ]
            for owner, name in (
                (rewrite.AssertionRewritingHook, "__init__"),
                (rewrite.AssertionRewritingHook, "find_spec"),
                (rewrite.AssertionRewritingHook, "exec_module"),
            ):
                slots.append((owner, name, inspect.getattr_static(owner, name)))
            # Every defining rewriter method remains original. No generic AST
            # transform or custom visitor may qualify a changed test function.
            for name, value in vars(rewrite.AssertionRewriter).items():
                if type(value) is FunctionType:
                    slots.append((rewrite.AssertionRewriter, name, value))
            for owner, name, value in slots:
                if type(value) is FunctionType:
                    defining = (
                        vars(owner)
                        if type(owner) is ModuleType
                        else vars(sys.modules[owner.__module__])
                    )
                    qualname = (
                        name
                        if type(owner) is ModuleType
                        else owner.__qualname__ + "." + name
                    )
                    assert (
                        value.__globals__ is defining and value.__qualname__ == qualname
                    )
                    super()._pin(value)
                self.slots.append((owner, name, value))
            enabled = config.getini("enable_assertion_pass_hook")
            assert type(enabled) is bool  # noqa: E721 - exact recorded rewrite option.
            context = (loader, config, state, path, enabled)
            self.pytest_rewrite_context = context
        assert all(
            context[index] is value
            for index, value in enumerate((loader, config, state))
        )
        assert context[3] == path
        assert config.getini("enable_assertion_pass_hook") is context[4]
        assert all(
            inspect.getattr_static(owner, name) is value
            for owner, name, value in self.slots
        )

        # Use pytest's actual original source reader/AST rewriter/compiler with
        # its real active config. Compile only; never execute the resulting code.
        _stat, compiled = rewrite._rewrite_test(path, config)
        assert path.read_bytes() == data
        expected = _nested(compiled, function.__code__)
        assert expected is not None and _shape(expected) == _shape(function.__code__)
        closure = function.__closure__
        self.pins.append(
            (
                function,
                function.__code__,
                function.__globals__,
                function.__defaults__,
                function.__kwdefaults__,
                tuple((function.__kwdefaults__ or {}).items()),
                closure,
                tuple((cell, cell.cell_contents) for cell in closure or ()),
            )
        )
        self.modules[module.__name__] = (
            module,
            path,
            module.__spec__,
            module.__spec__.origin,
            module.__loader__,
            module.__spec__.loader,
            hashlib.sha256(data).hexdigest(),
        )
        return function.__code__

    def pytest_rewrite_current(self):
        """Retain loader/config/state and selected rewrite option through births."""
        context = getattr(self, "pytest_rewrite_context", None)
        if context is None:
            return True
        try:
            from _pytest import assertion
            from _pytest.assertion import rewrite

            loader, config, state, path, enabled = context
            module = self.test_module
            return (
                type(loader) is rewrite.AssertionRewritingHook
                and loader in sys.meta_path
                and module.__loader__ is loader
                and module.__spec__.loader is loader
                and self.pytest_config is config
                and loader.config is config
                and config.stash[rewrite.assertstate_key] is state
                and type(state) is assertion.AssertionState
                and state.mode == "rewrite"
                and state.hook is loader
                and Path(loader._rewritten_names[module.__name__]).resolve() == path
                and config.getini("enable_assertion_pass_hook") is enabled
            )
        except BaseException:
            return False

    def close(self) -> dict[str, Any]:
        """Retire original monitoring and report this utility's deliberate custody."""
        receipt = super().close()
        receipt["original_source_current"] &= self.pytest_rewrite_current()
        receipt["complete"] &= receipt["original_source_current"]
        receipt["frames_arguments_results_tasks_retained"] = bool(
            self.apps
            or self.hosts
            or self.workers
            or self.databases
            or self.runtime_births
            or self.runtime_databases
            or self.notes_births
            or self.notes_databases
        )
        receipt["frames_retained"] = False
        receipt["birth_errors"] = tuple(self.birth_errors)
        receipt["unfinished_constructor_spans"] = len(self.births)
        receipt["unfinished_factory_births"] = len(self.runtime_births) + len(
            self.notes_births
        )
        receipt["complete"] &= not (
            self.birth_errors or self.births or self.runtime_births or self.notes_births
        )
        return receipt

    def sources_current(self) -> bool:
        """Check the existing source pins without retiring or repinning them."""
        try:
            if not self.pytest_rewrite_current():
                return False
            for (
                function,
                code,
                defining,
                defaults,
                kwdefaults,
                items,
                closure,
                cells,
            ) in self.pins:
                if not (
                    function.__code__ is code
                    and function.__globals__ is defining
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
            if any(
                inspect.getattr_static(owner, name) is not value
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
                    and Path(module.__file__).resolve() == path
                    and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                ):
                    return False
            return True
        except BaseException:
            return False

    def _constructor_current(self, role: str) -> bool:
        owner, init, new, call = self.constructors[role]
        return (
            type(init) is FunctionType
            and self.roles.get(init.__code__) == role
            and inspect.getattr_static(owner, "__init__") is init
            and inspect.getattr_static(owner, "__new__") is new is object.__new__
            and inspect.getattr_static(type(owner), "__call__") is call is type.__call__
        )

    def install_births(self) -> None:
        """Observe original normal constructor completion with local events only.

        A preinstalled custom route remains callable and is excluded. A partial
        monitoring install is physically retired before its error propagates.
        """
        from Tests.UI import app_factory
        from Tests.UI import test_console_fleet_wake_wiring as attachment_module
        from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
            ConsoleHarness,
        )
        from textual.worker import Worker
        from textual.worker_manager import WorkerManager
        from tldw_chatbook import app as app_module
        from tldw_chatbook.Chat import console_runtime as runtime_module
        from tldw_chatbook.DB import AgentRuns_DB as database_module
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
        from tldw_chatbook.DB.base_db import SQLiteConnectionQuiescenceRegistry
        from tldw_chatbook.Backup_Recovery import participants

        self.app_type, self.runtime_type = (
            app_module.TldwCli,
            runtime_module.ConsoleRuntime,
        )
        self.database_type, self.host_type = database_module.AgentRunsDB, ConsoleHarness
        self.notes_type = CharactersRAGDB
        self.attachment_module = attachment_module
        self.attachment_factory = attachment_module._attach_real_dbs
        self.attachment_code = self._pin(self.attachment_factory)
        self.attachment_qualified = attachment_module.CharactersRAGDB is CharactersRAGDB
        self.slots.extend(
            (
                (attachment_module, "_attach_real_dbs", self.attachment_factory),
                (
                    attachment_module,
                    "CharactersRAGDB",
                    attachment_module.CharactersRAGDB,
                ),
            )
        )
        self.worker_type, self.manager_type = Worker, WorkerManager
        self.factory = app_factory._build_test_app
        self.bridge_factory = self.test_module._bridge_over
        factory_source_qualified = (
            type(self.factory) is FunctionType
            and self.factory.__globals__ is vars(app_factory)
            and self.factory.__qualname__ == "_build_test_app"
        )
        try:
            self.factory_code = (
                self._pin(self.factory) if factory_source_qualified else None
            )
        except (AssertionError, OSError, ValueError):
            self.factory_code, factory_source_qualified = None, False
        self.bridge_qualified = (
            self.test_module.AgentRunsDB is self.database_type
            and type(self.bridge_factory) is FunctionType
            and self.bridge_factory.__globals__ is vars(self.test_module)
            and self.bridge_factory.__qualname__ == "_bridge_over"
        )
        try:
            self.bridge_code = (
                self._pin(self.bridge_factory) if self.bridge_qualified else None
            )
        except (AssertionError, OSError, ValueError):
            self.bridge_code, self.bridge_qualified = None, False
        self.slots.extend(
            (
                (app_factory, "_build_test_app", self.factory),
                (self.test_module, "_bridge_over", self.bridge_factory),
                (self.test_module, "AgentRunsDB", self.test_module.AgentRunsDB),
                (self.test_module, "ConsoleHarness", self.test_module.ConsoleHarness),
                (app_factory, "TldwCli", app_factory.TldwCli),
                (app_module, "ConsoleRuntime", app_module.ConsoleRuntime),
            )
        )
        current_factory = self.test_module._build_test_app
        if current_factory is not self.factory:
            # The unchanged imported Fleet fixture has the only supported wrapper.
            cells = (
                dict(
                    zip(
                        current_factory.__code__.co_freevars,
                        current_factory.__closure__ or (),
                    )
                )
                if type(current_factory) is FunctionType
                else {}
            )
            if (
                type(current_factory) is not FunctionType
                or current_factory.__qualname__
                != "_real_fleet_recovery_database.<locals>.build_with_db"
                or "build" not in cells
                or cells["build"].cell_contents is not self.factory
            ):
                self.factory_qualified = False
            else:
                self._pin(current_factory)
                self.factory_qualified = True
        else:
            self.factory_qualified = True
        self.factory_qualified &= factory_source_qualified and (
            app_factory.TldwCli is self.app_type
            and app_module.ConsoleRuntime is self.runtime_type
        )
        self.slots.append((self.test_module, "_build_test_app", current_factory))
        self.sync_codes = {
            self._pin(getattr(self.test_module, name))
            for name in (
                "test_agent_bridge_is_built_from_the_sibling_run_store_and_memoized",
                "test_agent_bridge_is_absent_without_a_durable_run_store",
            )
        }
        self.test_codes = {
            self._pin(function)
            for name, function in vars(self.test_module).items()
            if name.startswith("test_") and type(function) is FunctionType
        }
        self.direct_database_code = self._pin(
            self.test_module.test_drilldown_header_names_the_resumed_from_run
        )
        self.receipt_code = self._pin(self.runtime_type.ensure_activity_receipt_service)
        self.slots.append(
            (
                self.runtime_type,
                "ensure_activity_receipt_service",
                self.runtime_type.ensure_activity_receipt_service,
            )
        )
        for owner in (
            self.app_type,
            self.runtime_type,
            self.database_type,
            self.notes_type,
            self.host_type,
        ):
            init = inspect.getattr_static(owner, "__init__")
            new = inspect.getattr_static(owner, "__new__")
            call = inspect.getattr_static(type(owner), "__call__")
            self.constructors[owner.__name__] = (owner, init, new, call)
            self.slots.extend(
                (
                    (owner, "__init__", init),
                    (owner, "__new__", new),
                    (type(owner), "__call__", call),
                )
            )
            if (
                new is not object.__new__
                or call is not type.__call__
                or type(init) is not FunctionType
                or init.__qualname__ != owner.__qualname__ + ".__init__"
                or init.__globals__ is not vars(sys.modules[owner.__module__])
            ):
                continue
            try:
                code = self._pin(init)
            except (AssertionError, OSError, ValueError):
                continue
            self.roles[code] = owner.__name__
        self.app_code = getattr(self.constructors["TldwCli"][1], "__code__", None)
        if self.factory_code is not None:
            self.roles[self.factory_code] = "factory"
        self.roles[self.attachment_code] = "attachment"
        self.worker_start = Worker._start
        if (
            type(self.worker_start) is FunctionType
            and self.worker_start.__qualname__ == "Worker._start"
            and self.worker_start.__globals__ is vars(sys.modules[Worker.__module__])
        ):
            self.roles[self._pin(self.worker_start)] = "worker"
        self.dispose, self.close_database = (
            self.runtime_type.dispose,
            self.database_type.close,
        )
        self.close_notes = CharactersRAGDB.close_connection
        self.cancel, self.wait = Worker.cancel, Worker.wait
        self.api_qualified = {}
        for owner, name, function in (
            (Worker, "_start", self.worker_start),
            (Worker, "cancel", self.cancel),
            (Worker, "wait", self.wait),
            (self.runtime_type, "dispose", self.dispose),
            (self.database_type, "close", self.close_database),
            (self.notes_type, "close_connection", self.close_notes),
            (
                self.notes_type,
                "_close_connection_handle",
                self.notes_type._close_connection_handle,
            ),
            (
                SQLiteConnectionQuiescenceRegistry,
                "unregister",
                SQLiteConnectionQuiescenceRegistry.unregister,
            ),
            (
                SQLiteConnectionQuiescenceRegistry,
                "is_registered",
                SQLiteConnectionQuiescenceRegistry.is_registered,
            ),
            (
                self.runtime_type,
                "_start_canvas_policy_watcher",
                self.runtime_type._start_canvas_policy_watcher,
            ),
            (
                self.runtime_type,
                "_watch_canvas_policy",
                self.runtime_type._watch_canvas_policy,
            ),
            (
                participants,
                "_repository_participant",
                participants._repository_participant,
            ),
        ):
            defining = (
                vars(owner)
                if type(owner) is ModuleType
                else vars(sys.modules[owner.__module__])
            )
            qualname = (
                name if type(owner) is ModuleType else owner.__qualname__ + "." + name
            )
            qualified = (
                type(function) is FunctionType
                and function.__globals__ is defining
                and function.__qualname__ == qualname
            )
            if qualified:
                try:
                    self._pin(function)
                except (AssertionError, OSError, ValueError):
                    qualified = False
            self.api_qualified[(owner, name)] = qualified
            self.slots.append((owner, name, function))
        self.runtime_api_qualified = all(
            value
            for (owner, _name), value in self.api_qualified.items()
            if owner is self.runtime_type
        )
        self.database_api_qualified = self.api_qualified[(self.database_type, "close")]
        self.host_api_qualified = all(
            value
            for (owner, _name), value in self.api_qualified.items()
            if owner is Worker
        )
        self.notes_api_qualified = all(
            value
            for (owner, _name), value in self.api_qualified.items()
            if owner in (self.notes_type, SQLiteConnectionQuiescenceRegistry)
        )
        for name, descriptor in vars(AgentFixtureOwners).items():
            function = (
                descriptor.__func__ if type(descriptor) is staticmethod else descriptor
            )
            if type(function) is FunctionType:
                self._pin(function)
                self.slots.append((AgentFixtureOwners, name, descriptor))
        self.slots.append((asyncio, "Task", asyncio.Task))
        try:
            for tool in range(5, 0, -1):
                if tool == self.monitor.DEBUGGER_ID:
                    continue
                try:
                    self.monitor.use_tool_id(tool, "agent-fixture-birth")
                except ValueError:
                    continue
                self.tool = tool
                break
            if self.tool is None:
                raise RuntimeError("agent_fixture_birth_monitor_unavailable")
            if self.monitor.get_events(self.tool) != 0:
                raise RuntimeError("agent_fixture_global_events_changed")
            for event, callback in (
                (self.monitor.events.PY_START, self._birth_start),
                (self.monitor.events.PY_RETURN, self._birth_return),
            ):
                previous = self.monitor.register_callback(self.tool, event, callback)
                if previous is not None:
                    self.monitor.register_callback(self.tool, event, previous)
                    raise RuntimeError("agent_fixture_callback_not_unowned")
                self.registered[event] = callback
            self.active = True
            for code, role in self.roles.items():
                self.codes[code] = role
                if self.monitor.get_local_events(self.tool, code) != 0:
                    raise RuntimeError("agent_fixture_local_events_changed")
                self.monitor.set_local_events(
                    self.tool,
                    code,
                    self.monitor.events.PY_START | self.monitor.events.PY_RETURN,
                )
            self.installed = True
        except BaseException:
            self.source_receipt = self.close()
            raise

    def _scope(self, frame: Any, role: str) -> bool:
        parent = frame.f_back
        if parent is None:
            return False
        if role == "TldwCli":
            return self.factory_qualified and parent.f_code is self.factory_code
        if role == "ConsoleRuntime":
            return (
                self.factory_qualified
                and self._constructor_current("TldwCli")
                and self.runtime_api_qualified
                and self.database_api_qualified
                and self._constructor_current("AgentRunsDB")
                and parent.f_code is self.app_code
                and parent.f_back is not None
                and parent.f_back.f_code is self.factory_code
            )
        if role == "AgentRunsDB":
            return self.database_api_qualified and (
                (
                    self.bridge_qualified
                    and parent.f_code in (self.bridge_code, self.direct_database_code)
                )
                or (
                    parent.f_code is self.receipt_code
                    and any(
                        parent.f_locals.get("self") is runtime
                        for _app, runtime, _loop in self.apps
                    )
                )
            )
        if role == "CharactersRAGDB":
            return (
                self.attachment_qualified
                and self.notes_api_qualified
                and parent.f_code is self.attachment_code
                and parent.f_back is not None
                and parent.f_back.f_code in self.sync_codes
                and parent.f_back.f_globals is vars(self.test_module)
                and any(
                    parent.f_locals.get("app") is app
                    for app, _runtime, _loop in self.apps
                )
            )
        if role == "attachment":
            return (
                self.attachment_qualified
                and self.notes_api_qualified
                and parent.f_code in self.sync_codes
                and parent.f_globals is vars(self.test_module)
                and any(
                    frame.f_locals.get("app") is app
                    for app, _runtime, _loop in self.apps
                )
                and self._constructor_current("CharactersRAGDB")
            )
        if role == "ConsoleHarness":
            return (
                parent.f_globals is vars(self.test_module)
                and self.host_api_qualified
                and any(
                    frame.f_locals.get("app_instance") is app
                    for app, _runtime, _loop in self.apps
                )
                and parent.f_code in self.test_codes
            )
        if role == "factory":
            return self.factory_qualified
        if role == "worker":
            return any(
                frame.f_locals.get("app") is host for host, _manager in self.hosts
            )
        return False

    def _birth_start(self, code: Any, _offset: int) -> None:
        frame = actor = None
        try:
            if not self.active or code not in self.roles:
                return
            frame = sys._getframe(1)
            role = self.roles[code]
            if frame.f_code is not code or not self._scope(frame, role):
                return
            if not self.sources_current():
                self.birth_errors.append("source_changed_at_birth")
                return
            if role in self.constructors and not self._constructor_current(role):
                return
            actor = frame.f_locals.get("self")
            if (
                role in self.constructors
                and type(actor) is not self.constructors[role][0]
            ):
                return
            if role != "AgentRunsDB" and threading.current_thread() is not self.creator:
                self.birth_errors.append("birth_wrong_thread")
                return
            self.births[id(frame)] = (actor, role, threading.current_thread())
        except BaseException:
            self.birth_errors.append("birth_start_invalid")
        finally:
            del frame, actor

    def _database_birth(
        self, actor: Any, thread: Any
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        values = vars(actor)
        participant = values.get("_maintenance_participant")
        if (
            values.get("is_memory_db")
            or participant is None
            or participant.repository() is not actor
            or participant.path != values.get("db_path")
        ):
            raise RuntimeError("agent_fixture_database_birth_invalid")
        cache = values["_thread_local"]
        connection = getattr(cache, "conn", None)
        return actor, cache, values["db_path"], participant, thread, connection

    def _birth_return(self, code: Any, _offset: int, value: Any) -> None:
        frame = actor = parent = None
        try:
            if not self.active or code not in self.roles:
                return
            frame = sys._getframe(1)
            captured = self.births.pop(id(frame), None)
            if captured is None:
                return
            actor, role, thread = captured
            if (
                frame.f_code is not code
                or not self.sources_current()
                or thread is not threading.current_thread()
                or (
                    role in self.constructors
                    and (value is not None or not self._constructor_current(role))
                )
            ):
                raise RuntimeError("agent_fixture_birth_source_changed")
            parent = frame.f_back
            if role == "ConsoleRuntime":
                app = frame.f_locals.get("app")
                if (
                    parent.f_locals.get("self") is not app
                    or actor.app is not app
                    or actor._disposed
                    or actor.chat_store is not None
                    or actor.chat_controller is not None
                    or actor._agent_runs_db is not None
                ):
                    raise RuntimeError("agent_fixture_runtime_not_fresh")
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is None:
                    caller = parent.f_back.f_back
                    # The dependency's original wrapper intervenes before the test.
                    if caller is not None and caller.f_globals is not vars(
                        self.test_module
                    ):
                        caller = caller.f_back
                    if (
                        caller is None
                        or caller.f_code not in self.sync_codes
                        or actor._canvas_policy_watch_task is not None
                        or actor._canvas_policy_read_task is not None
                    ):
                        raise RuntimeError("agent_fixture_sync_runtime_unqualified")
                elif loop is not self.owner_loop:
                    raise RuntimeError("agent_fixture_runtime_wrong_loop")
                self.runtime_births[id(app)] = (app, actor, loop)
                self.runtime_watchers[id(actor)] = actor._canvas_policy_watch_task
            elif role == "factory":
                owned = self.runtime_births.pop(id(value), None)
                if owned is not None:
                    if (
                        type(value) is not self.app_type
                        or vars(value).get("console_runtime") is not owned[1]
                    ):
                        raise RuntimeError("agent_fixture_factory_runtime_changed")
                    self.apps.append(owned)
            elif role == "ConsoleHarness":
                manager = vars(actor).get("_workers")
                if (
                    type(manager) is not self.manager_type
                    or vars(manager).get("_app") is not actor
                ):
                    raise RuntimeError("agent_fixture_host_owner_changed")
                self.hosts.append((actor, manager))
            elif role == "AgentRunsDB":
                if actor.is_memory_db:
                    return
                database = self._database_birth(actor, thread)
                if parent.f_code is self.receipt_code:
                    self.runtime_databases[id(parent.f_locals["self"])] = database
                else:
                    self.databases.append(database)
            elif role == "CharactersRAGDB":
                if actor.is_memory_db:
                    return
                values = vars(actor)
                participant = values.get("_maintenance_participant")
                if (
                    participant is None
                    or participant.repository() is not actor
                    or participant.path != values.get("db_path")
                ):
                    raise RuntimeError("agent_fixture_attachment_birth_invalid")
                cache = values["_local"]
                self.notes_births[id(parent)] = (
                    actor,
                    cache,
                    values["db_path"],
                    participant,
                    thread,
                    getattr(cache, "conn", None),
                    values["_connection_quiescence"],
                )
            elif role == "attachment":
                captured_notes = self.notes_births.pop(id(frame), None)
                if captured_notes is None:
                    raise RuntimeError(
                        "agent_fixture_attachment_constructor_unqualified"
                    )
                app = frame.f_locals["app"]
                if (
                    frame.f_locals.get("db") is not captured_notes[0]
                    or vars(app).get("chachanotes_db") is not captured_notes[0]
                    or vars(app).get("conversation_local_marks_service") is not value
                    or vars(value).get("db") is not captured_notes[0]
                ):
                    raise RuntimeError("agent_fixture_attachment_return_changed")
                self.notes_databases.append(captured_notes)
            elif role == "worker":
                values = vars(actor)
                task, host = values.get("_task"), frame.f_locals.get("app")
                if (
                    type(actor) is not self.worker_type
                    or type(task) is not asyncio.Task
                    or task.get_loop() is not self.owner_loop
                ):
                    raise RuntimeError("agent_fixture_worker_owner_changed")
                if not any(
                    issued[0] is actor and issued[3] is task for issued in self.workers
                ):
                    self.workers.append(
                        (actor, values.get("_node"), values.get("_work"), task, host)
                    )
        except BaseException:
            self.birth_errors.append("birth_return_invalid")
        finally:
            del frame, actor, parent, value

    async def retire(self) -> None:
        """Propagate repeated cancellation only after exact owned settlement."""
        if (
            threading.current_thread() is not self.creator
            or asyncio.get_running_loop() is not self.owner_loop
        ):
            raise RuntimeError("agent_fixture_retirement_wrong_owner")
        if self.retirement_task is None or (
            self.retirement_task.done() and self.retirement_task.exception() is not None
        ):
            self.retirement_task = asyncio.Task(self._retire(), loop=self.owner_loop)
        cancellation = None
        while not self.retirement_task.done():
            try:
                await asyncio.shield(self.retirement_task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        self.retirement_task.result()
        if cancellation is not None:
            raise cancellation

    @staticmethod
    def _physically_closed(connection: Any) -> bool:
        if connection is None:
            return True
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError as error:
            return "closed" in str(error).lower()
        return False

    def _retire_database(
        self, captured: Any, *, close: bool, require_retired: bool = True
    ) -> None:
        from tldw_chatbook.Backup_Recovery import (
            participants,
            storage_admission as storage,
        )
        from tldw_chatbook.DB import private_sqlite

        database, cache, path, participant, thread, birth_connection = captured
        if (
            type(database) is not self.database_type
            or vars(database).get("_thread_local") is not cache
            or vars(database).get("_maintenance_participant") is not participant
            or database.db_path != path
            or participant is not participants._repository_participant(database)
        ):
            raise RuntimeError("agent_fixture_database_owner_changed")
        connection = getattr(cache, "conn", None)
        with storage._lock:
            if (
                storage._pause is not None
                or participant.retiring_threads
                or any(op.participant is participant for op in storage._operations)
                or any(
                    getattr(attempt.operation, "participant", None) is participant
                    for attempt in storage._pending_acquisitions
                )
                or any(
                    lease.resource_thread is not self.creator
                    for lease in participant.connections.values()
                )
                or any(conn is not connection for conn in participant.connections)
            ):
                raise RuntimeError("agent_fixture_database_still_active")
            lease = participant.connections.get(connection)
            if connection is not None and (
                lease is None
                or lease not in storage._live_leases
                or lease.resource_path != path
                or lease.resource_participant is not participant
                or lease.resource_thread is not self.creator
            ):
                raise RuntimeError("agent_fixture_database_lease_changed")
        if (
            connection is not None
            and not self._physically_closed(connection)
            and sqlite3.Connection.in_transaction.__get__(connection)
        ):
            raise RuntimeError("agent_fixture_database_transaction_live")
        if not require_retired:
            if birth_connection is not connection and not self._physically_closed(
                birth_connection
            ):
                raise RuntimeError("agent_fixture_database_still_active")
            return
        if close:
            if thread is not self.creator:
                raise RuntimeError("agent_fixture_database_wrong_creator")
            self.close_database(database)
        if not self._physically_closed(connection) or not self._physically_closed(
            birth_connection
        ):
            raise RuntimeError("agent_fixture_database_not_closed")
        with storage._lock:
            if (
                getattr(cache, "conn", None) is not None
                or participant.connections
                or participant.retiring_threads
                or (lease is not None and lease in storage._live_leases)
                or any(
                    getattr(item, "resource_participant", None) is participant
                    for item in storage._live_leases
                )
                or connection in private_sqlite._ordinary_connections
                or birth_connection in private_sqlite._ordinary_connections
            ):
                raise RuntimeError("agent_fixture_database_not_retired")

    async def _retire(self) -> None:
        if self.source_receipt is None:
            self.source_receipt = self.close()
        if (
            self.birth_errors
            or self.births
            or self.runtime_births
            or self.notes_births
            or not self.source_receipt["original_source_current"]
            or not self.source_receipt["hooks_retired_before_inactive"]
            or not self.sources_current()
        ):
            raise RuntimeError("agent_fixture_birth_provenance_invalid")
        for host, manager in self.hosts:
            if (
                vars(host).get("_workers") is not manager
                or vars(manager).get("_app") is not host
            ):
                raise RuntimeError("agent_fixture_host_owner_changed")
        for worker, node, work, task, _host in self.workers:
            values = vars(worker)
            if (
                values.get("_node") is not node
                or values.get("_work") is not work
                or values.get("_task") is not task
            ):
                raise RuntimeError("agent_fixture_worker_owner_changed")
            if not task.done():
                self.cancel(worker)
        await asyncio.gather(
            *(self.wait(worker) for worker, *_rest in self.workers),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(task for _worker, _node, _work, task, _host in self.workers),
            return_exceptions=True,
        )
        if any(not task.done() for _worker, _node, _work, task, _host in self.workers):
            raise RuntimeError("agent_fixture_worker_not_retired")
        for app, runtime, loop in self.apps:
            if not self.sources_current():
                raise RuntimeError("agent_fixture_retirement_source_changed")
            if (
                runtime.app is not app
                or runtime._canvas_policy_watch_task
                is not self.runtime_watchers.get(id(runtime))
                or (loop is not None and loop is not self.owner_loop)
                or any(
                    type(task) is not asyncio.Task
                    or task.get_loop() is not self.owner_loop
                    for task in (
                        runtime._canvas_policy_watch_task,
                        runtime._canvas_policy_read_task,
                    )
                    if task is not None
                )
            ):
                raise RuntimeError("agent_fixture_runtime_owner_changed")
            database = self.runtime_databases.get(id(runtime))
            if runtime._agent_runs_db is not None and (
                database is None or runtime._agent_runs_db is not database[0]
            ):
                raise RuntimeError("agent_fixture_runtime_database_borrowed")
            before_connection = None
            if database is not None:
                self._retire_database(database, close=False, require_retired=False)
                before_connection = getattr(database[1], "conn", None)
            watcher = runtime._canvas_policy_watch_task
            await self.dispose(runtime)
            if (
                not runtime._disposed
                or runtime._canvas_policy_watch_task is not None
                or runtime._canvas_policy_read_task is not None
            ):
                raise RuntimeError("agent_fixture_runtime_not_retired")
            if watcher is not None and not watcher.done():
                raise RuntimeError("agent_fixture_runtime_watcher_not_retired")
            if not self._physically_closed(before_connection):
                raise RuntimeError("agent_fixture_runtime_database_not_closed")
            if database is not None:
                self._retire_database(database, close=False)
        for database in self.databases:
            if not self.sources_current():
                raise RuntimeError("agent_fixture_retirement_source_changed")
            self._retire_database(database, close=True)
        for database in self.notes_databases:
            if not self.sources_current():
                raise RuntimeError("agent_fixture_retirement_source_changed")
            self._retire_notes_database(database)

    def _retire_notes_database(self, captured: Any) -> None:
        """Close only the original explicit attachment, never its whole registry."""
        from tldw_chatbook.Backup_Recovery import (
            participants,
            storage_admission as storage,
        )
        from tldw_chatbook.DB import private_sqlite

        database, cache, path, participant, thread, birth_connection, registry = (
            captured
        )
        if (
            thread is not self.creator
            or type(database) is not self.notes_type
            or vars(database).get("_local") is not cache
            or vars(database).get("_maintenance_participant") is not participant
            or vars(database).get("_connection_quiescence") is not registry
            or database.db_path != path
            or participant is not participants._repository_participant(database)
        ):
            raise RuntimeError("agent_fixture_attachment_owner_changed")
        connection = getattr(cache, "conn", None)
        with registry._condition:
            if (
                registry._active_uses
                or registry._active_acquisitions
                or registry._quiescence_token is not None
            ):
                raise RuntimeError("agent_fixture_attachment_still_active")
            if (
                connection is not None
                and registry._connections.get(id(connection)) is not connection
            ):
                raise RuntimeError("agent_fixture_attachment_registration_changed")
        with storage._lock:
            if (
                storage._pause is not None
                or participant.retiring_threads
                or any(op.participant is participant for op in storage._operations)
                or any(
                    getattr(attempt.operation, "participant", None) is participant
                    for attempt in storage._pending_acquisitions
                )
                or any(
                    item is not connection or lease.resource_thread is not self.creator
                    for item, lease in participant.connections.items()
                )
            ):
                raise RuntimeError("agent_fixture_attachment_still_active")
            lease = participant.connections.get(connection)
            if connection is not None and (
                lease is None
                or lease not in storage._live_leases
                or lease.resource_path != path
                or lease.resource_participant is not participant
            ):
                raise RuntimeError("agent_fixture_attachment_lease_changed")
        if (
            connection is not None
            and not self._physically_closed(connection)
            and sqlite3.Connection.in_transaction.__get__(connection)
        ):
            raise RuntimeError("agent_fixture_attachment_transaction_live")
        self.close_notes(database)
        if not self._physically_closed(connection) or not self._physically_closed(
            birth_connection
        ):
            raise RuntimeError("agent_fixture_attachment_not_closed")
        with registry._condition:
            if (
                connection is not None
                and registry._connections.get(id(connection)) is connection
                or birth_connection is not None
                and registry._connections.get(id(birth_connection)) is birth_connection
            ):
                raise RuntimeError("agent_fixture_attachment_not_retired")
        with storage._lock:
            if (
                getattr(cache, "conn", None) is not None
                or participant.connections
                or participant.retiring_threads
                or any(
                    getattr(item, "resource_participant", None) is participant
                    for item in storage._live_leases
                )
                or connection in private_sqlite._ordinary_connections
                or birth_connection in private_sqlite._ordinary_connections
            ):
                raise RuntimeError("agent_fixture_attachment_not_retired")
