"""Opt-in exact fixture/native-owner witness; no application imports here."""

import ast
import hashlib
import inspect
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import AsyncGeneratorType, CodeType, FunctionType

import pytest


TESTS = "Tests.UI.test_console_session_tab_close"
TARGET = "test_session_close_pending_race_and_fleet_journeys"


def _shape(code):
    return (
        code.co_name,
        code.co_qualname,
        code.co_firstlineno,
        code.co_code,
        code.co_flags,
        code.co_stacksize,
        code.co_exceptiontable,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(
            _shape(item) if type(item) is CodeType else item for item in code.co_consts
        ),
    )


def _source_code(module, qualname, *, config=None):
    source = Path(module.__file__)
    if config is None:
        code = compile(source.read_bytes(), str(source), "exec", dont_inherit=True)
    else:
        from _pytest.assertion import rewrite

        loader = module.__spec__.loader
        assert type(loader) is rewrite.AssertionRewritingHook
        assert module.__loader__ is loader and loader.config is config
        assert Path(module.__spec__.origin).resolve() == source.resolve()
        assert loader._rewritten_names[module.__name__].resolve() == source.resolve()
        _, code = rewrite._rewrite_test(source, config)
    for name in (part for part in qualname.split(".") if part != "<locals>"):
        code = next(
            item
            for item in code.co_consts
            if type(item) is CodeType and item.co_name == name
        )
    return code


def _physically_closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


class OriginalPreparedFleetDBLifetime:
    """Observe only the exact yielded fixture owner and known request chain."""

    def __init__(self, config):
        self.config = config
        self.bindings, self.modules, self.codes = [], {}, {}
        self.owners, self.rows, self.references = {}, [], []
        self.lock = threading.Lock()
        self.tool, self.active, self.restored = None, False, False
        self.installed, self.callbacks, self.overflow = [], {}, 0
        self.monitor = sys.monitoring
        tests = sys.modules[TESTS]
        self.runs_type = self.runs_module = self.storage = None
        self.runs_bound = False
        self.fixtures, self.runtime_starts = {}, {}
        self.runtime_type = self.runtime_module = None
        self.runtime_bound = False
        self.request_stages, self.stage_lines, self.stage_frames = [], {}, set()
        self.stage_bound, self.stage_aliases, self.stage_origins = False, [], {}
        self.characters_origin_bound = False
        self.characters_origin_gaps = {
            "caller_code_capacity": 0,
            "operation_capacity": 0,
            "acquisition_capacity": 0,
            "row_capacity": 0,
            "live_operation_capacity": 0,
            "live_acquisition_capacity": 0,
            "operation_caller_source_gap": 0,
            "acquisition_caller_source_gap": 0,
        }
        (
            self.characters_origins,
            self.characters_operations,
            self.characters_acquisitions,
        ) = [], {}, {}
        self.characters_origin_codes, self.characters_origin_aliases = {}, []
        self.characters_operation_code = self.characters_core_code = None
        self.characters_acquire_code = self.characters_acquire_close_code = None
        self.cleanup_bound = False
        self.cleanup_type = self.cleanup_module = None
        self.cleanup_refusal_lines = {}
        self.origin_pins = {}
        pending = inspect.getattr_static(tests, "_pending_close_app")
        self.fixture_body_code = pending.__wrapped__.__code__
        self.async_fixture = inspect.isasyncgenfunction(pending.__wrapped__)
        self.async_manager_type = None
        definitions = (
            (tests, tests, "_pending_close_app", pending, "fixture_wrapper", 0, None),
            (
                tests,
                pending,
                "__wrapped__",
                pending.__wrapped__,
                "fixture_yield",
                0 if self.async_fixture else self.monitor.events.PY_YIELD,
                config,
            ),
            (
                tests,
                tests,
                "_arm_pending_round",
                tests._arm_pending_round,
                "arm",
                0,
                config,
            ),
            (
                sys.modules["tempfile"],
                sys.modules["tempfile"].TemporaryDirectory,
                "cleanup",
                sys.modules["tempfile"].TemporaryDirectory.cleanup,
                "cleanup_start",
                self.monitor.events.PY_START,
                None,
            ),
            (
                sys.modules["shutil"],
                sys.modules["shutil"],
                "rmtree",
                sys.modules["shutil"].rmtree,
                "delete_start",
                self.monitor.events.PY_START,
                None,
            ),
        )
        if self.async_fixture:
            contextlib = sys.modules["contextlib"]
            self.async_manager_type = inspect.getattr_static(
                contextlib, "_AsyncGeneratorContextManager"
            )
            definitions += (
                (
                    contextlib,
                    self.async_manager_type,
                    "__aenter__",
                    inspect.getattr_static(self.async_manager_type, "__aenter__"),
                    "fixture_async_enter_return",
                    self.monitor.events.PY_RETURN,
                    None,
                ),
            )
        for definition in definitions:
            self._bind_definition(definition)
        self.request_code = next(
            code
            for code in tests._arm_pending_round.__code__.co_consts
            if type(code) is CodeType and code.co_name == "request"
        )

    def _bind_definition(self, definition):
        source_module, owner, name, function, label, mask, rewrite_config = definition
        assert type(function) is FunctionType
        defining = sys.modules[function.__globals__["__name__"]]
        assert defining.__dict__ is function.__globals__
        # Exact wrappers must have the original defining body and exact
        # original closure cell/content, rather than a weak __wrapped__ alias.
        expected = _source_code(
            defining,
            function.__code__.co_qualname,
            config=rewrite_config if defining is sys.modules[TESTS] else None,
        )
        assert _shape(function.__code__) == _shape(expected)
        cells = tuple(function.__closure__ or ())
        contents = tuple(cell.cell_contents for cell in cells)
        if label in {
            "fixture_wrapper",
            "held_wrapper",
            "stage_wrapper",
            "characters_origin_wrapper",
        }:
            assert len(cells) == 1 and contents[0] is function.__wrapped__
        elif label == "stage_getter_body":
            assert function.__code__.co_freevars == ("__class__",)
            assert len(cells) == 1
            assert contents[0] is inspect.getattr_static(source_module, "AgentRunsDB")
        else:
            assert not cells
        if label.startswith("stage_"):
            # Exact selected defaults are literal booleans/None in original source.
            node = ast.parse(Path(defining.__file__).read_bytes())
            for part in (
                item
                for item in function.__code__.co_qualname.split(".")
                if item != "<locals>"
            ):
                node = next(
                    item
                    for item in node.body
                    if isinstance(
                        item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    and item.name == part
                )
            expected_defaults = tuple(
                ast.literal_eval(item) for item in node.args.defaults
            )
            assert tuple(function.__defaults__ or ()) == expected_defaults
            expected_kw = {
                arg.arg: ast.literal_eval(item)
                for arg, item in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if item is not None
            }
            assert (function.__kwdefaults__ or {}) == expected_kw
            for module in (source_module, defining):
                spec = module.__spec__
                assert (
                    spec is not None
                    and Path(spec.origin).resolve() == Path(module.__file__).resolve()
                )
                self.stage_origins[module.__name__] = (
                    module,
                    module.__file__,
                    spec,
                    spec.origin,
                    spec.loader,
                    module.__loader__,
                )
        self.bindings.append(
            (
                owner,
                name,
                function,
                function.__code__,
                function.__globals__,
                function.__defaults__,
                function.__kwdefaults__,
                cells,
                contents,
            )
        )
        for module in (source_module, defining):
            path = Path(module.__file__)
            self.modules[module.__name__] = (
                module,
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            if module.__name__ not in self.origin_pins:
                spec = module.__spec__
                assert (
                    spec is not None and Path(spec.origin).resolve() == path.resolve()
                )
                self.origin_pins[module.__name__] = (
                    module.__file__,
                    spec,
                    spec.origin,
                    spec.loader,
                    module.__loader__,
                )
        if mask:
            self.codes[function.__code__] = (label, mask)
        if mask and self.active:
            code = function.__code__
            assert self.monitor.get_local_events(self.tool, code) == 0
            self.monitor.set_local_events(self.tool, code, mask)
            self.installed.append(code)

    def _bind_runs(self):
        if self.runs_bound:
            return
        # The exact original surviving-child fixture has created this DB before
        # yielding. Only bind its now-loaded stock class; never force an import.
        module = sys.modules.get("tldw_chatbook.DB.AgentRuns_DB")
        storage = sys.modules.get("tldw_chatbook.Backup_Recovery.storage_admission")
        assert (
            module is not None and storage is not None
        ), "required yielded DB source not loaded"
        runs_type = inspect.getattr_static(module, "AgentRunsDB")
        held = inspect.getattr_static(runs_type, "_held_connection")
        definitions = (
            (module, runs_type, "_held_connection", held, "held_wrapper", 0, None),
            (
                module,
                held,
                "__wrapped__",
                held.__wrapped__,
                "held_return",
                self.monitor.events.PY_START
                | self.monitor.events.PY_RETURN
                | self.monitor.events.LINE,
                None,
            ),
            (
                module,
                runs_type,
                "close",
                runs_type.close,
                "close_return",
                self.monitor.events.PY_RETURN,
                None,
            ),
        )
        for definition in definitions:
            self._bind_definition(definition)
        self.runs_module, self.runs_type, self.storage = module, runs_type, storage
        self.runs_bound = True
        self._bind_request_stages()

    def _bind_request_stages(self):
        if self.stage_bound:
            return
        names = (
            "tldw_chatbook.DB.base_db",
            "tldw_chatbook.DB.private_sqlite",
            "tldw_chatbook.Backup_Recovery.storage_admission",
        )
        base, private, storage = (sys.modules.get(name) for name in names)
        assert base is not None and private is not None and storage is not None
        base_type = inspect.getattr_static(base, "BaseDB")
        getter = inspect.getattr_static(self.runs_type, "_get_connection")
        connector = inspect.getattr_static(private, "_connect_registered_sqlite")
        operation = inspect.getattr_static(storage, "_repository_operation")
        mask = (
            self.monitor.events.PY_START
            | self.monitor.events.PY_RETURN
            | self.monitor.events.LINE
        )
        assert base.connect_private_sqlite is private.connect_private_sqlite
        self.stage_aliases.extend(
            (
                (base, "BaseDB", base_type),
                (base, "connect_private_sqlite", private.connect_private_sqlite),
            )
        )
        definitions = (
            (
                self.runs_module,
                self.runs_type,
                "get_run",
                self.runs_type.get_run,
                "stage_get_run",
                mask,
                None,
            ),
            (
                self.runs_module,
                self.runs_type,
                "_get_connection",
                getter,
                "stage_wrapper",
                mask,
                None,
            ),
            (
                self.runs_module,
                getter,
                "__wrapped__",
                getter.__wrapped__,
                "stage_getter_body",
                mask,
                None,
            ),
            (
                base,
                base_type,
                "_get_connection",
                base_type._get_connection,
                "stage_base_getter",
                mask,
                None,
            ),
            (
                private,
                private,
                "connect_private_sqlite",
                private.connect_private_sqlite,
                "stage_private_connect",
                mask,
                None,
            ),
            (
                private,
                private,
                "_connect_registered_sqlite",
                connector,
                "stage_wrapper",
                mask,
                None,
            ),
            (
                private,
                connector,
                "__wrapped__",
                connector.__wrapped__,
                "stage_private_body",
                mask,
                None,
            ),
            (
                storage,
                storage,
                "_repository_operation",
                operation,
                "stage_wrapper",
                0,
                None,
            ),
            (
                storage,
                operation,
                "__wrapped__",
                operation.__wrapped__,
                "stage_repository_operation",
                mask,
                None,
            ),
        )
        for definition in definitions:
            self._bind_definition(definition)
        self.stage_bound = True

    def _stage_entry(self, frame):
        caller, entry, request = frame, None, None
        for _ in range(32):
            if caller is None:
                break
            if entry is None:
                owner = caller.f_locals.get("self", caller.f_locals.get("repository"))
                candidate = self.owners.get(id(owner))
                if candidate is not None and candidate["owner"] is owner:
                    entry = candidate
                participant = caller.f_locals.get("participant")
                if entry is None and participant is not None:
                    entry = next(
                        (
                            item
                            for item in self.owners.values()
                            if item["participant"] is participant
                        ),
                        None,
                    )
            if caller.f_code is self.request_code:
                request = caller
                break
            caller = caller.f_back
        if (
            request is None
            or entry is None
            or request.f_globals is not sys.modules[TESTS].__dict__
        ):
            return None
        controller = request.f_locals["controller"]
        if (
            controller.app is not entry["app"]
            or request.f_locals.get("kind") != "chat_create"
        ):
            return None
        bridge = controller._agent_bridge
        actual_db = getattr(bridge, "runs_db", None) or getattr(
            bridge, "agent_runs_db", None
        )
        if actual_db is not entry["owner"]:
            return None
        return entry, request, controller

    def _stage(self, frame, phase, *, value=None):
        if not self.stage_bound:
            return
        label = self.codes[frame.f_code][0]
        if label != "held_return" and not label.startswith("stage_"):
            return
        selected = self._stage_entry(frame)
        if selected is None:
            return
        entry, request, controller = selected
        thread = threading.current_thread()
        key = (id(frame), id(thread))
        if key not in self.stage_frames:
            assert len(self.stage_frames) < 64
            self.stage_frames.add(key)
            self.references.extend((frame, request, controller, thread))
        stamp = time.monotonic_ns()
        line_key = (*key, frame.f_lineno)
        if phase == "line" and line_key in self.stage_lines:
            row = self.stage_lines[line_key]
            row["line_revisits"] += 1
            row["last_observer_monotonic_ns"] = stamp
            return
        if len(self.request_stages) >= 512:
            self.overflow += 1
            return
        row = {
            "stage_ordinal": len(self.request_stages),
            "observer_monotonic_ns": stamp,
            "phase": phase,
            "source_kind": label,
            "source_qualname": frame.f_code.co_qualname,
            "source_line": frame.f_lineno,
            "frame_object": id(frame),
            "request_frame_object": id(request),
            "thread_object": id(thread),
            "owner_object": id(entry["owner"]),
            "controller_object": id(controller),
        }
        body = frame.f_locals.get("function")
        if type(body) is FunctionType:
            row["wrapped_body_qualname"] = body.__code__.co_qualname
        if phase == "return":
            row["actual_return_type"] = type(value).__name__
        if phase == "line":
            row["line_revisits"] = 0
            row["last_observer_monotonic_ns"] = stamp
            self.stage_lines[line_key] = row
        self.request_stages.append(row)

    def _bind_characters_origins(self):
        if self.characters_origin_bound:
            return
        participants = sys.modules.get("tldw_chatbook.Backup_Recovery.participants")
        storage = sys.modules.get("tldw_chatbook.Backup_Recovery.storage_admission")
        characters = sys.modules.get("tldw_chatbook.DB.ChaChaNotes_DB")
        assert (
            participants is not None
            and storage is self.storage
            and characters is not None
        )
        core = inspect.getattr_static(participants, "_core_operation")
        operation = inspect.getattr_static(storage, "_repository_operation")
        acquisition_type = inspect.getattr_static(storage, "_Acquisition")
        assert inspect.getattr_static(characters, "_core_operation") is core
        self.characters_origin_aliases.extend(
            (
                (characters, "_core_operation", core),
                (storage, "_Acquisition", acquisition_type),
            )
        )
        mask = (
            self.monitor.events.PY_START
            | self.monitor.events.PY_RETURN
            | self.monitor.events.LINE
        )
        for definition in (
            (
                participants,
                participants,
                "_core_operation",
                core,
                "characters_origin_wrapper",
                0,
                None,
            ),
            (
                participants,
                core,
                "__wrapped__",
                core.__wrapped__,
                "characters_core",
                mask,
                None,
            ),
            (
                storage,
                storage,
                "_repository_operation",
                operation,
                "characters_origin_wrapper",
                0,
                None,
            ),
            (
                storage,
                operation,
                "__wrapped__",
                operation.__wrapped__,
                "characters_operation",
                0,
                None,
            ),
            (
                storage,
                acquisition_type,
                "__init__",
                acquisition_type.__init__,
                "characters_acquire",
                self.monitor.events.PY_RETURN,
                None,
            ),
            (
                storage,
                acquisition_type,
                "close",
                acquisition_type.close,
                "characters_acquire_close",
                self.monitor.events.PY_START | self.monitor.events.PY_RETURN,
                None,
            ),
        ):
            self._bind_definition(definition)
        # The existing request-stage binder installs this exact repository body
        # once when the actual Runs owner is captured. Do not double-install it.
        self.characters_core_code = core.__wrapped__.__code__
        self.characters_operation_code = operation.__wrapped__.__code__
        self.characters_acquire_code = acquisition_type.__init__.__code__
        self.characters_acquire_close_code = acquisition_type.close.__code__
        self.characters_origin_bound = True

    def _characters_fixture(self, participant):
        return next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.get("characters_origin_window", False)
                and fixture["chars"]._maintenance_participant is participant
            ),
            None,
        )

    def _characters_source_chain(self, frame):
        chain, caller = [], frame
        for _ in range(24):
            if caller is None:
                break
            namespace, code = caller.f_globals, caller.f_code
            name = namespace.get("__name__")
            module = sys.modules.get(name)
            if (
                module is not None
                and module.__dict__ is namespace
                and (
                    name.startswith("tldw_chatbook.")
                    or name.startswith("Tests.")
                    or name
                    in {"contextlib", "concurrent.futures.thread", "asyncio.events"}
                )
            ):
                if code not in self.characters_origin_codes:
                    if len(self.characters_origin_codes) >= 64:
                        self.characters_origin_gaps["caller_code_capacity"] += 1
                        return None
                    rewrite = sys.modules.get("_pytest.assertion.rewrite")
                    config = (
                        self.config
                        if rewrite is not None
                        and type(module.__spec__.loader)
                        is rewrite.AssertionRewritingHook
                        else None
                    )
                    expected = _source_code(module, code.co_qualname, config=config)
                    assert _shape(code) == _shape(expected)
                    path, spec = Path(module.__file__), module.__spec__
                    assert (
                        spec is not None
                        and Path(spec.origin).resolve() == path.resolve()
                    )
                    self.modules[name] = (
                        module,
                        path,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    self.origin_pins[name] = (
                        module.__file__,
                        spec,
                        spec.origin,
                        spec.loader,
                        module.__loader__,
                    )
                    self.characters_origin_codes[code] = (
                        module,
                        namespace,
                        len(self.characters_origin_codes),
                    )
                    self.references.extend((code, namespace))
                module, original_globals, code_index = self.characters_origin_codes[
                    code
                ]
                assert (
                    caller.f_globals is original_globals
                    and sys.modules.get(module.__name__) is module
                )
                chain.append(
                    {
                        "code_index": code_index,
                        "source_module": module.__name__,
                        "source_qualname": code.co_qualname,
                        "source_line": caller.f_lineno,
                    }
                )
            caller = caller.f_back
        return chain

    def _characters_origin(self, frame, phase):
        if not self.characters_origin_bound:
            return
        code, local = frame.f_code, frame.f_locals
        operation = (
            local.get("operation") if code is self.characters_operation_code else None
        )
        attempt = (
            local.get("self")
            if code
            in {self.characters_acquire_code, self.characters_acquire_close_code}
            else None
        )
        if attempt is not None:
            operation = getattr(attempt, "operation", None)
            if operation is None:
                # Initial acquire clears the ambient operation. Its exact original
                # repository-operation caller still supplies the issued identity.
                caller = frame.f_back
                for _ in range(24):
                    if caller is None:
                        break
                    if caller.f_code is self.characters_operation_code:
                        operation = caller.f_locals.get("operation")
                        break
                    caller = caller.f_back
        if operation is None:
            return
        key = id(operation)
        known_operation = self.characters_operations.get(key)
        if operation not in self.storage._operations and (
            known_operation is None or known_operation[0] is not operation
        ):
            return  # Original metadata is incomplete before _operations.add.
        participant = getattr(operation, "participant", None)
        fixture = self._characters_fixture(participant)
        if fixture is None:
            return
        assert participant.repository() is fixture["chars"]
        assert participant.path == fixture["directory"] / "chats.sqlite"
        assert operation.thread is threading.current_thread()
        key = id(operation)
        if key not in self.characters_operations:
            # LINE after the original add observes a real installed operation;
            # no manufactured operation or additional repository call is issued.
            if operation not in self.storage._operations:
                return
            if len(self.characters_operations) >= 64:
                self.characters_origin_gaps["operation_capacity"] += 1
                return
            chain = self._characters_source_chain(frame)
            if chain is None:
                self.characters_origin_gaps["operation_caller_source_gap"] += 1
                return
            self.characters_operations[key] = (operation, fixture, chain)
            self.references.extend(
                (operation, operation.thread, operation.task, participant)
            )
            self._characters_append(
                {
                    "kind": "original_characters_operation_admitted",
                    "operation_object": key,
                    "participant_object": id(participant),
                    "owner_object": id(fixture["chars"]),
                    "runtime_object": id(fixture["scope"].runtime),
                    "thread_object": id(operation.thread),
                    "task_object": id(operation.task) if operation.task else None,
                    "source_chain": chain,
                }
            )
        else:
            assert self.characters_operations[key][0] is operation
        if attempt is not None:
            assert attempt.thread is operation.thread and attempt.task is operation.task
            acquire_key = id(attempt)
            if acquire_key not in self.characters_acquisitions:
                if len(self.characters_acquisitions) >= 128:
                    self.characters_origin_gaps["acquisition_capacity"] += 1
                    return
                chain = self._characters_source_chain(frame)
                if chain is None:
                    self.characters_origin_gaps["acquisition_caller_source_gap"] += 1
                    return
                self.characters_acquisitions[acquire_key] = (attempt, operation, chain)
                self.references.extend((attempt, attempt.thread, attempt.task))
                self._characters_append(
                    {
                        "kind": "original_characters_acquisition_observed",
                        "operation_object": key,
                        "acquisition_object": acquire_key,
                        "thread_object": id(attempt.thread),
                        "task_object": id(attempt.task) if attempt.task else None,
                        "actual_acquisition_operation_matches": attempt.operation
                        is operation,
                        "pending": attempt in self.storage._pending_acquisitions,
                        "source_chain": chain,
                    }
                )
            if code is self.characters_acquire_close_code:
                self._characters_append(
                    {
                        "kind": "original_characters_acquisition_close",
                        "phase": phase,
                        "operation_object": key,
                        "acquisition_object": acquire_key,
                        "pending": attempt in self.storage._pending_acquisitions,
                    }
                )
        elif code is self.characters_operation_code and phase == "return":
            self._characters_append(
                {
                    "kind": "original_characters_operation_return",
                    "operation_object": key,
                    "still_counted": operation in self.storage._operations,
                }
            )

    def _characters_append(self, row):
        # These finite callbacks run with the GIL. Do not add a second lock order
        # while the original generator may itself hold storage._changed.
        if len(self.characters_origins) >= 384:
            self.characters_origin_gaps["row_capacity"] += 1
            return
        row["origin_ordinal"] = len(self.characters_origins)
        row["observer_monotonic_ns"] = time.monotonic_ns()
        self.characters_origins.append(row)

    def _characters_live_state(self, fixture, participant):
        # Called only inside the pre-existing exact-declaration storage lock.
        operations = tuple(
            item for item in self.storage._operations if item.participant is participant
        )
        if len(operations) > 64:
            self.characters_origin_gaps["live_operation_capacity"] += 1
            operations = operations[:64]
        result = []
        for operation in operations:
            known = self.characters_operations.get(id(operation))
            pending = tuple(
                item
                for item in self.storage._pending_acquisitions
                if item.operation is operation
                or (
                    id(item) in self.characters_acquisitions
                    and self.characters_acquisitions[id(item)][1] is operation
                )
            )
            if len(pending) > 16:
                self.characters_origin_gaps["live_acquisition_capacity"] += 1
                pending = pending[:16]
            result.append(
                {
                    "operation_object": id(operation),
                    "origin_observed": known is not None,
                    "thread_object": id(operation.thread),
                    "thread_alive": operation.thread.is_alive(),
                    "task_object": id(operation.task) if operation.task else None,
                    "operation_lease_present": operation.lease is not None,
                    "pending_acquisitions": [
                        {
                            "acquisition_object": id(item),
                            "origin_observed": id(item) in self.characters_acquisitions,
                            "actual_acquisition_operation_matches": item.operation
                            is operation,
                            "thread_object": id(item.thread),
                            "task_object": id(item.task) if item.task else None,
                            "initializing_root_present": item.initializing_root
                            is not None,
                        }
                        for item in pending
                    ],
                }
            )
        return result

    def _bind_runtime(self):
        if self.runtime_bound:
            return
        module = sys.modules.get("tldw_chatbook.Chat.console_runtime")
        assert module is not None, "required fixture runtime defining source not loaded"
        runtime_type = inspect.getattr_static(module, "ConsoleRuntime")
        function = inspect.getattr_static(runtime_type, "ensure_agent_bridge")
        self._bind_definition(
            (
                module,
                runtime_type,
                "ensure_agent_bridge",
                function,
                "runtime_bridge",
                self.monitor.events.PY_START | self.monitor.events.PY_RETURN,
                None,
            )
        )
        self.runtime_module, self.runtime_type = module, runtime_type
        self.runtime_bound = True

    def _own_runs(self, fixture, runs, *, origin, runtime=None):
        self._bind_runs()
        assert type(runs) is self.runs_type
        expected = fixture["directory"] / (
            "runs.sqlite" if origin == "fixture_explicit" else "agent_runs.db"
        )
        assert runs.db_path == expected
        if id(runs) in self.owners:
            assert self.owners[id(runs)]["owner"] is runs
            return
        assert len(self.owners) < 8
        participant = runs._maintenance_participant
        assert participant.repository() is runs and participant.path == expected
        with self.storage._lock:
            creators = tuple(
                (connection, lease)
                for connection, lease in participant.connections.items()
                if lease.resource_thread is fixture["creator"]
            )
        assert len(creators) <= 4
        self.references.extend(item for pair in creators for item in pair)
        entry = dict(
            creator_handles=creators,
            owner=runs,
            participant=participant,
            directory=fixture["directory"],
            creator=fixture["creator"],
            app=fixture["app"],
            chars=fixture["chars"],
            runtime=runtime,
            origin=origin,
            workers={},
            deleted=False,
        )
        self.owners[id(runs)] = entry
        self.references.extend(
            (runs, participant, fixture["app"], fixture["chars"], runtime)
        )
        self._append(
            {
                "kind": "exact_fixture_runs_owner",
                "origin": origin,
                "owner_object": id(runs),
                "app_object": id(fixture["app"]),
                "chars_object": id(fixture["chars"]),
                "path_basename": expected.name,
                "new_since_original_fixture_yield": runs
                is not fixture["runtime_runs_at_yield"],
            }
        )

    def _runtime_return(self, frame, value):
        runtime = frame.f_locals["self"]
        key = (id(threading.current_thread()), id(frame))
        fixture = self.runtime_starts.pop(key, None)
        if fixture is None:
            return
        assert type(runtime) is self.runtime_type and runtime._app is fixture["app"]
        app = fixture["app"]
        assert app.chachanotes_db is fixture["chars"]
        runs = runtime._agent_runs_db
        if value is None or runs is None:
            self._append(
                {
                    "kind": "original_runtime_bridge_return_unavailable",
                    "app_object": id(app),
                }
            )
            return
        assert runtime._agent_bridge is value
        assert (
            runs is not fixture["runtime_runs_at_yield"]
        ), "pre-existing or borrowed runtime DB is not newly owned"
        self._own_runs(fixture, runs, origin="runtime_sibling", runtime=runtime)

    def _bind_cleanup(self):
        if self.cleanup_bound:
            return
        module = sys.modules.get("Tests.UI._prepared_close_owned_resources")
        assert module is not None
        storage = sys.modules.get("tldw_chatbook.Backup_Recovery.storage_admission")
        assert storage is not None
        if self.storage is None:
            self.storage = storage
        assert self.storage is storage
        storage_path = Path(storage.__file__)
        self.modules[storage.__name__] = (
            storage,
            storage_path,
            hashlib.sha256(storage_path.read_bytes()).hexdigest(),
        )
        spec = storage.__spec__
        assert (
            spec is not None and Path(spec.origin).resolve() == storage_path.resolve()
        )
        self.origin_pins[storage.__name__] = (
            storage.__file__,
            spec,
            spec.origin,
            spec.loader,
            storage.__loader__,
        )
        helper_type = inspect.getattr_static(module, "PreparedCloseOwnedResources")
        function = inspect.getattr_static(helper_type, "close_creators")
        runtime_dispose = inspect.getattr_static(self.runtime_type, "dispose")
        for definition in (
            (
                module,
                helper_type,
                "close_creators",
                function,
                "owned_cleanup",
                self.monitor.events.PY_START
                | self.monitor.events.PY_RETURN
                | self.monitor.events.LINE,
                None,
            ),
            (
                self.runtime_module,
                self.runtime_type,
                "dispose",
                runtime_dispose,
                "owned_runtime_dispose",
                self.monitor.events.PY_START | self.monitor.events.PY_RETURN,
                None,
            ),
        ):
            self._bind_definition(definition)
        tree = ast.parse(Path(module.__file__).read_bytes())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "PreparedCloseOwnedResources"
        )
        body = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "close_creators"
        )
        for node in ast.walk(body):
            if (
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and isinstance(node.exc.func, ast.Name)
                and node.exc.func.id == "RuntimeError"
            ):
                assert len(node.exc.args) == 1
                reason = ast.literal_eval(node.exc.args[0])
                assert type(reason) is str and reason.startswith("prepared_close_")  # noqa: E721 - accept only builtin static refusal strings
                self.cleanup_refusal_lines[node.lineno] = reason
        self.cleanup_module, self.cleanup_type = module, helper_type
        self.cleanup_bound = True

    def _cleanup_frame(self, frame, phase, *, reason=None):
        label = self.codes[frame.f_code][0]
        current = frame.f_locals["self"]
        if label == "owned_runtime_dispose":
            fixture = self.fixtures.get(id(current._app))
            if fixture is None or fixture["scope"].runtime is not current:
                return
            scope = fixture["scope"]
        else:
            fixture = self.fixtures.get(id(current.app))
            if fixture is None or fixture["scope"] is not current:
                return
            scope = current
        assert type(scope) is self.cleanup_type
        declarations = tuple(scope.declarations)
        assert len(declarations) <= 3
        states = []
        with self.storage._lock:
            for database, expected_type, expected_path in declarations:
                assert (
                    type(database) is expected_type
                    and database.db_path == expected_path
                )
                participant = database._maintenance_participant
                assert (
                    participant.repository() is database
                    and participant.path == expected_path
                )
                connections = tuple(participant.connections.items())
                assert len(connections) <= 24
                physical = []
                for connection, lease in connections:
                    self.references.extend((connection, lease, lease.resource_thread))
                    physical.append(
                        {
                            "connection_object": id(connection),
                            "lease_object": id(lease),
                            "thread_object": id(lease.resource_thread),
                            "creator_thread": lease.resource_thread
                            is fixture["creator"],
                            "thread_alive": lease.resource_thread.is_alive(),
                            "physically_closed": _physically_closed(connection),
                        }
                    )
                states.append(
                    {
                        "owner_object": id(database),
                        "source_type": expected_type.__name__,
                        "path_basename": expected_path.name,
                        "admission_closed": participant.closed,
                        "connections": physical,
                        "operations": sum(
                            operation.participant is participant
                            for operation in self.storage._operations
                        ),
                        "pending_acquisitions": sum(
                            getattr(attempt.operation, "participant", None)
                            is participant
                            for attempt in self.storage._pending_acquisitions
                        ),
                        "retiring_threads": len(participant.retiring_threads),
                        "exact_path_leases": sum(
                            lease.resource_path == expected_path
                            for lease in self.storage._live_leases
                        ),
                        "exact_characters_operations": (
                            self._characters_live_state(fixture, participant)
                            if database is fixture["chars"]
                            else []
                        ),
                    }
                )
        old_handles = []
        for entry in self.owners.values():
            if entry["app"] is not fixture["app"]:
                continue
            for connection, lease in entry["creator_handles"]:
                old_handles.append(
                    {
                        "role": "original_creator",
                        "owner_object": id(entry["owner"]),
                        "connection_object": id(connection),
                        "lease_object": id(lease),
                        "physically_closed": _physically_closed(connection),
                    }
                )
            for connection, lease, thread in entry["workers"].values():
                old_handles.append(
                    {
                        "role": "original_request_worker",
                        "owner_object": id(entry["owner"]),
                        "connection_object": id(connection),
                        "lease_object": id(lease),
                        "physically_closed": _physically_closed(connection),
                    }
                )
        runtime = scope.runtime
        self._append(
            {
                "kind": label,
                "phase": phase,
                "line": frame.f_lineno,
                "scope_object": id(scope),
                "runtime_object": id(runtime),
                "app_object": id(scope.app),
                "current_thread_object": id(threading.current_thread()),
                "same_creator_thread": threading.current_thread() is fixture["creator"],
                "runtime_disposed": runtime._disposed,
                "runtime_terminal": scope.runtime_terminal,
                "canvas_watcher_present": runtime._canvas_policy_watch_task is not None,
                "canvas_read_present": runtime._canvas_policy_read_task is not None,
                "maintenance_task_present": runtime._legacy_trace_maintenance_task
                is not None,
                "maintenance_task_done": (
                    None
                    if runtime._legacy_trace_maintenance_task is None
                    else runtime._legacy_trace_maintenance_task.done()
                ),
                "static_refusal_reason": reason,
                "declared_native_states": states,
                "retained_original_handles": old_handles,
            }
        )

    def _line(self, code, line):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        self._characters_origin(frame, "line")
        if self.codes[code][0].startswith("characters_"):
            return
        with self.lock:
            self._stage(frame, "line")
            if (
                self.codes[code][0] == "owned_cleanup"
                and line in self.cleanup_refusal_lines
            ):
                self._cleanup_frame(
                    frame,
                    "before_original_refusal",
                    reason=self.cleanup_refusal_lines[line],
                )

    def _current(self):
        if (
            self.async_fixture
            and inspect.getattr_static(
                sys.modules["contextlib"], "_AsyncGeneratorContextManager"
            )
            is not self.async_manager_type
        ):
            return False
        stage_current = all(
            sys.modules.get(name) is module
            and module.__file__ == filename
            and module.__spec__ is spec
            and spec.origin == origin
            and spec.loader is loader
            and module.__loader__ is module_loader
            for name, (
                module,
                filename,
                spec,
                origin,
                loader,
                module_loader,
            ) in self.stage_origins.items()
        ) and all(
            inspect.getattr_static(owner, name) is value
            for owner, name, value in self.stage_aliases
        )
        runs_alias_current = (
            not self.runs_bound
            or inspect.getattr_static(self.runs_module, "AgentRunsDB") is self.runs_type
        )
        runtime_alias_current = (
            not self.runtime_bound
            or inspect.getattr_static(self.runtime_module, "ConsoleRuntime")
            is self.runtime_type
        )
        cleanup_alias_current = (
            not self.cleanup_bound
            or inspect.getattr_static(
                self.cleanup_module, "PreparedCloseOwnedResources"
            )
            is self.cleanup_type
        )
        characters_alias_current = all(
            inspect.getattr_static(owner, name) is value
            for owner, name, value in self.characters_origin_aliases
        )
        return (
            characters_alias_current
            and stage_current
            and cleanup_alias_current
            and runs_alias_current
            and runtime_alias_current
            and all(
                inspect.getattr_static(owner, name) is function
                and function.__code__ is code
                and function.__globals__ is namespace
                and function.__defaults__ is defaults
                and function.__kwdefaults__ is kwdefaults
                and tuple(function.__closure__ or ()) == cells
                and all(
                    cell.cell_contents is content
                    for cell, content in zip(cells, contents)
                )
                for owner, name, function, code, namespace, defaults, kwdefaults, cells, contents in self.bindings
            )
            and all(
                sys.modules.get(name) is module
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                and module.__file__ == self.origin_pins[name][0]
                and module.__spec__ is self.origin_pins[name][1]
                and module.__spec__.origin == self.origin_pins[name][2]
                and module.__spec__.loader is self.origin_pins[name][3]
                and module.__loader__ is self.origin_pins[name][4]
                for name, (module, path, digest) in self.modules.items()
            )
        )

    def _append(self, row):
        if len(self.rows) >= 256:
            self.overflow += 1
        else:
            row["sequence"] = len(self.rows)
            self.rows.append(row)

    def _yield(self, code, offset, value):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        self._fixture_admitted(frame, value)

    def _fixture_admitted(self, frame, value):
        local = frame.f_locals
        if local["kind"] != "chat_create":
            return
        app, chars, directory = local["app"], local["db"], Path(local["directory"])
        assert value is app and app.chachanotes_db is chars
        tests = sys.modules[TESTS]
        assert type(chars) is inspect.getattr_static(tests, "CharactersRAGDB")
        assert chars.db_path == directory / "chats.sqlite"
        self._bind_runtime()
        self._bind_cleanup()
        scope = local["owner"]
        assert type(scope) is self.cleanup_type and scope.app is app
        assert scope.characters_db is chars and scope.directory == directory
        runtime = getattr(app, "console_runtime", None)
        before = getattr(runtime, "_agent_runs_db", None)
        with self.lock:
            assert len(self.fixtures) < 8 and id(app) not in self.fixtures
            fixture = dict(
                app=app,
                chars=chars,
                directory=directory,
                creator=threading.current_thread(),
                runtime_runs_at_yield=before,
                scope=scope,
            )
            self.fixtures[id(app)] = fixture
            self._bind_characters_origins()
            # This is the original yielded app, before the original caller can
            # construct/mount its harness. Observe only its declared Characters.
            fixture["characters_origin_window"] = True
            self.references.extend(
                (app, chars, runtime, before, threading.current_thread())
            )
            self._append(
                {
                    "kind": "fixture_yield",
                    "app_object": id(app),
                    "chars_object": id(chars),
                    "explicit_runs_present": local.get("runs") is not None,
                    "runtime_runs_present_at_yield": before is not None,
                    "creator_thread_object": id(threading.current_thread()),
                }
            )
            runs = local.get("runs")
            if runs is not None:
                assert local["surviving_child"] and app._pending_close_runs is runs
                self._own_runs(fixture, runs, origin="fixture_explicit")

    def _return(self, code, offset, value):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        self._characters_origin(frame, "return")
        if self.codes[code][0].startswith("characters_"):
            return
        with self.lock:
            self._stage(frame, "return", value=value)
        if self.codes[code][0].startswith("stage_"):
            return
        if self.codes[code][0] in {"owned_cleanup", "owned_runtime_dispose"}:
            with self.lock:
                self._cleanup_frame(frame, "return")
            return
        if self.codes[code][0] == "fixture_async_enter_return":
            manager = frame.f_locals["self"]
            if type(manager) is not self.async_manager_type:
                return
            generator = manager.gen
            if (
                type(generator) is not AsyncGeneratorType
                or generator.ag_code is not self.fixture_body_code
            ):
                return
            original_frame = generator.ag_frame
            assert original_frame is not None
            assert original_frame.f_code is self.fixture_body_code
            assert original_frame.f_globals is sys.modules[TESTS].__dict__
            self.references.extend((manager, generator, original_frame))
            self._fixture_admitted(original_frame, value)
            return
        if self.codes[code][0] == "runtime_bridge":
            with self.lock:
                self._runtime_return(frame, value)
            return
        owner = frame.f_locals.get("self")
        with self.lock:
            entry = self.owners.get(id(owner))
            if entry is None or entry["owner"] is not owner:
                return
            label = self.codes[code][0]
            thread = threading.current_thread()
            if label == "close_return":
                self._append(
                    {
                        "kind": label,
                        "owner_object": id(owner),
                        "thread_object": id(thread),
                    }
                )
                return
            if thread is entry["creator"]:
                return
            caller = frame.f_back
            for _ in range(12):
                if caller is None or caller.f_code is self.request_code:
                    break
                caller = caller.f_back
            if caller is None or caller.f_code is not self.request_code:
                return  # Other worker/database chains confer no fixture authority.
            controller = caller.f_locals["controller"]
            assert controller.app is entry["app"]
            if entry["origin"] == "fixture_explicit":
                assert controller.app._pending_close_runs is owner
            else:
                assert controller.app.console_runtime is entry["runtime"]
                assert entry["runtime"]._agent_runs_db is owner
            with self.storage._lock:
                lease = entry["participant"].connections[value]
                assert (
                    lease.resource_thread is thread
                    and lease in self.storage._live_leases
                )
            key = id(value)
            if key not in entry["workers"]:
                assert len(entry["workers"]) < 24
                entry["workers"][key] = (value, lease, thread)
                self.references.extend((value, lease, thread, caller, controller))
                self._append(
                    {
                        "kind": "exact_request_worker_native_handle",
                        "owner_object": id(owner),
                        "thread_object": id(thread),
                        "connection_object": key,
                        "physically_closed": _physically_closed(value),
                    }
                )

    def _start(self, code, offset):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        self._characters_origin(frame, "start")
        if self.codes[code][0].startswith("characters_"):
            return
        label = self.codes[code][0]
        with self.lock:
            self._stage(frame, "start")
        if label == "held_return" or label.startswith("stage_"):
            return
        if label in {"owned_cleanup", "owned_runtime_dispose"}:
            with self.lock:
                self._cleanup_frame(frame, "start")
            return
        if label == "runtime_bridge":
            runtime = frame.f_locals["self"]
            fixture = self.fixtures.get(id(runtime._app))
            if fixture is None or fixture["app"] is not runtime._app:
                return
            assert type(runtime) is self.runtime_type
            assert runtime._app.chachanotes_db is fixture["chars"]
            with self.lock:
                assert len(self.runtime_starts) < 16
                key = (id(threading.current_thread()), id(frame))
                assert key not in self.runtime_starts
                self.runtime_starts[key] = fixture
                self.references.extend((runtime, threading.current_thread()))
            return
        path = (
            Path(frame.f_locals["self"].name)
            if label == "cleanup_start"
            else Path(frame.f_locals["path"])
        )
        with self.lock:
            entries = [
                entry for entry in self.owners.values() if entry["directory"] == path
            ]
            for entry in entries:
                if label == "delete_start" and entry["deleted"]:
                    continue
                with self.storage._lock:
                    participant = entry["participant"]
                    connections = tuple(participant.connections.items())
                    assert len(connections) <= 24
                    live_path_leases = sum(
                        lease.resource_path == participant.path
                        for lease in self.storage._live_leases
                    )
                    operations = sum(
                        operation.participant is participant
                        for operation in self.storage._operations
                    )
                    physical = [
                        dict(
                            connection_object=id(conn),
                            lease_object=id(lease),
                            thread_object=id(lease.resource_thread),
                            creator_thread=lease.resource_thread is entry["creator"],
                            matched_request_worker=id(conn) in entry["workers"],
                            physically_closed=_physically_closed(conn),
                        )
                        for conn, lease in connections
                    ]
                native = tuple(item[0] for item in entry["workers"].values())
                for conn, lease in connections:
                    self.references.extend((conn, lease, lease.resource_thread))
                self._append(
                    {
                        "kind": label,
                        "owner_object": id(entry["owner"]),
                        "origin": entry["origin"],
                        "exact_request_worker_handles": len(native),
                        "worker_handles_physically_closed": [
                            _physically_closed(item) for item in native
                        ],
                        "participant_native_handles": len(connections),
                        "participant_native_physical_states": physical,
                        "live_exact_path_leases": live_path_leases,
                        "exact_owner_operations": operations,
                    }
                )
                if label == "delete_start":
                    entry["deleted"] = True

    def start(self):
        assert self._current()
        self.tool = next(
            slot
            for slot in range(5, 0, -1)
            if slot != self.monitor.DEBUGGER_ID and self.monitor.get_tool(slot) is None
        )
        self.monitor.use_tool_id(self.tool, "tldw-finite-prepared-fleet-db")
        assert self.monitor.get_events(self.tool) == 0
        try:
            for event, callback in (
                (self.monitor.events.PY_START, self._start),
                (self.monitor.events.PY_RETURN, self._return),
                (self.monitor.events.PY_YIELD, self._yield),
                (self.monitor.events.LINE, self._line),
            ):
                assert (
                    self.monitor.register_callback(self.tool, event, callback) is None
                )
                self.callbacks[event] = callback
            self.active = True
            for code, (_, mask) in self.codes.items():
                assert self.monitor.get_local_events(self.tool, code) == 0
                self.monitor.set_local_events(self.tool, code, mask)
                self.installed.append(code)
        except BaseException:
            self.stop()
            raise

    def stop(self):
        assert self.monitor.get_events(self.tool) == 0
        for code in self.installed:
            assert self.monitor.get_local_events(self.tool, code) == self.codes[code][1]
            self.monitor.set_local_events(self.tool, code, 0)
        for event, callback in self.callbacks.items():
            assert self.monitor.register_callback(self.tool, event, None) is callback
        self.active = False
        self.monitor.free_tool_id(self.tool)
        self.restored = self.monitor.get_tool(self.tool) is None

    def receipt(self):
        return {
            "diagnostic_only": True,
            "source_and_original_body_qualified": self.runs_bound and self._current(),
            "required_yielded_db_source_coverage_complete": self.runs_bound,
            "exact_original_chat_create_fixtures": len(self.fixtures),
            "exact_new_owned_runs_databases": len(self.owners),
            "runtime_bridge_source_bound": self.runtime_bound,
            "unmatched_original_runtime_returns": len(self.runtime_starts),
            "pytest_rewritten_fixture_body_qualified": True,
            "exact_wrapper_body_globals_defaults_closure_qualified": True,
            "global_events": 0,
            "local_hooks_retired": self.restored,
            "overflow": self.overflow,
            "source_hashes": {
                name: digest for name, (_, _, digest) in self.modules.items()
            },
            "events": self.rows,
            "request_stage_defining_sources_and_closures_current": self.stage_bound
            and self._current(),
            "exact_request_get_run_start_covered": any(
                row["source_kind"] == "stage_get_run" and row["phase"] == "start"
                for row in self.request_stages
            ),
            "exact_request_private_connector_start_covered": any(
                row["source_kind"] == "stage_private_body" and row["phase"] == "start"
                for row in self.request_stages
            ),
            "request_database_stages": self.request_stages,
            "characters_origin_defining_sources_current": self.characters_origin_bound
            and self._current(),
            "characters_repository_hook_coverage_complete": self.stage_bound
            and self.characters_operation_code in self.installed,
            "characters_original_caller_codes_qualified": len(
                self.characters_origin_codes
            ),
            "characters_disposal_origins": self.characters_origins,
            "characters_origins_armed_before_mount": bool(self.fixtures)
            and all(
                fixture.get("characters_origin_window", False)
                for fixture in self.fixtures.values()
            ),
            "characters_origin_gap_counts": self.characters_origin_gaps,
            "characters_origin_capacity_complete": not any(
                self.characters_origin_gaps.values()
            ),
            "characters_terminal_live_origins_complete": all(
                operation["origin_observed"]
                and all(
                    attempt["origin_observed"]
                    for attempt in operation["pending_acquisitions"]
                )
                for row in self.rows
                if row["kind"] in {"owned_runtime_dispose", "owned_cleanup"}
                and row["phase"] in {"return", "before_original_refusal"}
                for state in row["declared_native_states"]
                for operation in state["exact_characters_operations"]
            ),
            "characters_origin_limits": "Only the exact declared fixture Characters participant from its actual "
            "fixture yield, before original harness mount. Actual installed operation/acquisition identities and original "
            "executing caller CodeTypes/globals/source; wrapper body/closure bindings retained. "
            "No extra DB reads, close, census, task scheduling, wait, guard or deadline. "
            "The synchronous worker chain does not by itself identify an absent async submitter.",
            "request_stage_limits": "Only exact original request ancestry and declared DB. No new reads, "
            "native/API replacement, initialization, guards, waits or deadlines. Actual original "
            "START/RETURN and deduplicated LINE boundaries; revisits retain last timestamp. "
            "Spans overlap and must not be summed; diagnostics add overhead.",
            "exact_cleanup_helper_body_and_alias_current": self.cleanup_bound
            and self._current(),
            "owned_runtime_dispose_return_covered": any(
                row["kind"] == "owned_runtime_dispose" and row["phase"] == "return"
                for row in self.rows
            ),
            "owned_creator_finalization_start_covered": any(
                row["kind"] == "owned_cleanup" and row["phase"] == "start"
                for row in self.rows
            ),
            "cleanup_observer_limits": "Only exact declared helper/runtime and retained creator/request-worker handles. "
            "An original static refusal LINE observes exact native physical state, admission/operations/leases; "
            "no authority DB read, producer close, wait, deadline or original effect is replaced.",
            "limits": "Only yielded exact fixture owners and actual original request-worker chains. "
            "Missing actual worker acquisition/delete boundary is incomplete. No guards, waits, "
            "requests, database APIs, deletion methods or startup owners replaced. Diagnostic adds overhead.",
        }


@pytest.fixture(autouse=True)
def original_prepared_fleet_db_lifetime(request):
    from Tests.private_profile import is_private_profile_child

    if (
        os.environ.get("TLDW_TEST_PREPARED_FLEET_DB_LIFETIME") != "1"
        or request.module.__name__ != TESTS
        or request.node.name != TARGET
        or not is_private_profile_child(request)
    ):
        yield
        return
    observer = OriginalPreparedFleetDBLifetime(request.config)
    output = Path(
        os.environ.get("TLDW_TEST_PREPARED_FLEET_DB_RECEIPT")
        or str(
            Path(os.environ["TLDW_TEST_CONFIG_ROOT"]).parent
            / "prepared-fleet-db-lifetime.json"
        )
    )
    try:
        observer.start()
        yield
    finally:
        if observer.active:
            observer.stop()
        output.write_text(json.dumps(observer.receipt(), indent=2), encoding="utf-8")
        assert (
            observer.runs_bound
        ), "required original yielded AgentRunsDB source never bound"
