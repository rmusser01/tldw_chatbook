"""Evidence-only held original receipt-schema callback on the actual startup route.

No production imports, callable replacement, SQL observer/query, global events,
profile hook, foreign resource close, or startup-budget claim. Parent launches.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.machinery
import inspect
from pathlib import Path
import sqlite3
import sys
import threading
import time
from types import CodeType, FunctionType, MethodType, ModuleType
import weakref


SPECS = (
    ("tldw_chatbook.app", "TldwCli.__init__", "app_ctor"),
    ("tldw_chatbook.app", "TldwCli.on_mount", "app_mount"),
    ("tldw_chatbook.app", "TldwCli._push_initial_screen", "initial_screen"),
    ("tldw_chatbook.UI.Screens.chat_screen", "ChatScreen.compose_content", "compose"),
    (
        "tldw_chatbook.UI.Console_Modules.context_spend",
        "ConsoleContextSpendController._build_console_cost_state",
        "cost",
    ),
    (
        "tldw_chatbook.UI.Screens.chat_screen",
        "ChatScreen._ensure_console_chat_controller",
        "controller",
    ),
    (
        "tldw_chatbook.UI.Screens.chat_screen",
        "ChatScreen._sync_console_chat_core_state",
        "core",
    ),
    ("tldw_chatbook.UI.Screens.chat_screen", "ChatScreen.on_key", "key"),
    (
        "tldw_chatbook.Widgets.Console.console_composer_bar",
        "ConsoleComposerBar.insert_text",
        "insert",
    ),
    (
        "tldw_chatbook.Chat.console_runtime",
        "ConsoleRuntime.ensure_agent_bridge",
        "bridge",
    ),
    (
        "tldw_chatbook.Chat.console_runtime",
        "ConsoleRuntime.ensure_activity_receipt_service",
        "receipts",
    ),
    (
        "tldw_chatbook.UI.Console_Modules.agent",
        "ConsoleAgentController._console_agent_section_lines",
        "agent_section",
    ),
    (
        "tldw_chatbook.UI.Console_Modules.agent",
        "ConsoleAgentController._ensure_console_agent_bridge",
        "agent_bridge",
    ),
    (
        "tldw_chatbook.Chat.console_runtime",
        "ConsoleRuntime._prepare_initial_activity_receipts",
        "receipt_preparation",
    ),
    ("tldw_chatbook.DB.AgentRuns_DB", "AgentRunsDB._initialize_schema", "schema"),
    ("tldw_chatbook.Utils.windows_files", "_Native.open_handle", "native_open"),
)


def shape(code):
    return (
        code.co_name,
        code.co_qualname,
        code.co_firstlineno,
        code.co_code,
        code.co_exceptiontable,
        code.co_stacksize,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_flags,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(
            shape(value) if type(value) is CodeType else value
            for value in code.co_consts
        ),
    )


def nested(code, qualname):
    if code.co_qualname == qualname:
        return code
    for value in code.co_consts:
        if type(value) is CodeType:
            found = nested(value, qualname)
            if found is not None:
                return found
    return None


def frozen(value):
    if type(value) in (str, bytes, int, float, bool, type(None)):  # noqa: E721 -- exact primitive types only.
        return type(value), value
    if type(value) in (tuple, list):  # noqa: E721 -- compare exact container types without custom iteration.
        return type(value), tuple(frozen(item) for item in value)
    if type(value) is dict:  # noqa: E721 -- exact metadata dictionary, never custom iteration.
        return tuple((frozen(key), frozen(item)) for key, item in value.items())
    return type(value), id(value)


def physical_open(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return False
    return True


class StartupReceiptNativeGate:
    """Hold one existing schema SQL entry; probe the same original event loop."""

    def __init__(self, repo, owner_loop, *, hold_native=True, _specs=SPECS):
        self.repo, self.loop = Path(repo).absolute(), owner_loop
        self.hold_native = hold_native
        self.main_ident, self.specs = threading.get_ident(), tuple(_specs)
        self.monitor, self.tool, self.active = sys.monitoring, None, False
        self.masks, self.callbacks, self.codes = {}, {}, {}
        self.module_records, self.method_records, self.sources = {}, {}, {}
        self.module_codes, self.schema_attempts = {}, []
        self.invalid, self.rows, self.overflow = [], [], 0
        self.counts, self.states = {}, {}
        self.native_counts = {}
        self.receiver_entries = set()
        self.app = None
        self.entered, self.release, self.progress = (
            threading.Event() for _ in range(3)
        )
        self.hold = None
        self.held_connection = self.held_database = None
        self.native_source_records = {}
        self.native_method_records = []
        self.native_class_records = []
        self.held_native_lease = self.held_native_participant = None
        self.schema_line = None
        self.controller_thread = None
        self.initial_task = None
        self.preparation_actor = None
        self.preparation_thread_record = None
        self.preparation_thread_binding = None
        self.loader = inspect.getattr_static(
            importlib.machinery.SourceFileLoader, "get_code"
        )
        assert type(self.loader) is FunctionType
        self.loader_record = self._function_record(self.loader)
        self.loader_exec = inspect.getattr_static(
            importlib.machinery.SourceFileLoader, "exec_module"
        )
        assert type(self.loader_exec) is FunctionType
        self.loader_exec_record = self._function_record(self.loader_exec)

    @staticmethod
    def _function_record(function):
        return (
            function,
            function.__code__,
            function.__globals__,
            function.__defaults__,
            frozen(function.__defaults__),
            function.__kwdefaults__,
            frozen(function.__kwdefaults__),
            function.__closure__,
            ()
            if function.__closure__ is None
            else tuple(cell.cell_contents for cell in function.__closure__),
            getattr(function, "__wrapped__", None),
        )

    @staticmethod
    def _function_current(record):
        (
            function,
            code,
            namespace,
            defaults,
            defaults_value,
            kw,
            kw_value,
            closure,
            cells,
            wrapped,
        ) = record
        return (
            function.__code__ is code
            and function.__globals__ is namespace
            and function.__defaults__ is defaults
            and frozen(defaults) == defaults_value
            and function.__kwdefaults__ is kw
            and frozen(kw) == kw_value
            and function.__closure__ is closure
            and (
                closure is None
                or all(
                    cell.cell_contents is value
                    for cell, value in zip(closure, cells, strict=True)
                )
            )
            and getattr(function, "__wrapped__", None) is wrapped
        )

    def _issue(self, reason):
        if len(self.invalid) < 24:
            self.invalid.append(reason)
        else:
            self.overflow += 1

    def _enable(self, code, mask):
        if code in self.masks:
            assert self.masks[code] == mask
            return
        assert self.monitor.get_local_events(self.tool, code) == 0
        self.masks[code] = mask  # Ledger precedes possibly failing installation.
        self.monitor.set_local_events(self.tool, code, mask)

    def _publish(self, loader, module_code):
        if (
            type(loader) is not importlib.machinery.SourceFileLoader
            or type(module_code) is not CodeType
        ):
            return
        name = loader.name
        wanted = [spec for spec in self.specs if spec[0] == name]
        if not wanted:
            return
        path = self.repo / (name.replace(".", "/") + ".py")
        assert Path(module_code.co_filename).absolute() == path
        raw = path.read_bytes()
        expected = compile(
            raw,
            module_code.co_filename,
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
        assert shape(module_code) == shape(expected)
        self.sources[name] = (path, hashlib.sha256(raw).hexdigest())
        self.module_codes[name] = module_code
        for spec in wanted:
            code = nested(module_code, spec[1])
            if spec[2] == "receipt_preparation" and code is None:
                # The original pre-repair synchronous route remains valid.
                continue
            assert code is not None and code not in self.codes
            self.codes[code] = spec
            mask = self.monitor.events.PY_START | self.monitor.events.PY_RETURN
            if code.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE):
                mask |= self.monitor.events.PY_YIELD | self.monitor.events.PY_RESUME
            if spec[2] == "native_open":
                mask = self.monitor.events.PY_START
            if spec[2] == "schema":
                tree = ast.parse(raw)
                klass = next(
                    node
                    for node in tree.body
                    if type(node) is ast.ClassDef and node.name == "AgentRunsDB"
                )
                method = next(
                    node
                    for node in klass.body
                    if type(node) is ast.FunctionDef
                    and node.name == "_initialize_schema"
                )
                # The method also contains later migration executescripts. Pin
                # its original first connection/first schema SQL statement.
                statement = method.body[0]
                assert type(statement) is ast.With and len(statement.items) == 1
                context = statement.items[0]
                assert type(context.context_expr) is ast.Call
                assert (
                    type(context.context_expr.func) is ast.Attribute
                    and context.context_expr.func.attr == "connection"
                )
                assert (
                    type(context.context_expr.func.value) is ast.Name
                    and context.context_expr.func.value.id == "self"
                )
                assert (
                    type(context.optional_vars) is ast.Name
                    and context.optional_vars.id == "conn"
                )
                assert type(statement.body[0]) is ast.Expr
                call = statement.body[0].value
                assert (
                    type(call) is ast.Call
                    and type(call.func) is ast.Attribute
                    and call.func.attr == "executescript"
                )
                assert (
                    type(call.func.value) is ast.Name and call.func.value.id == "conn"
                )
                self.schema_line = call.lineno
                mask |= self.monitor.events.LINE
            self._enable(code, mask)

    def _bind(self, code, frame=None):
        name, member, label = self.codes[code]
        if code in self.method_records and frame is None:
            return
        module = sys.modules.get(name)
        assert type(module) is ModuleType
        if frame is not None:
            assert frame.f_globals is vars(module) and code in self.method_records
            _, class_name, owner, method_name, function, record = self.method_records[
                code
            ]
            assert vars(module).get(class_name) is owner
            assert inspect.getattr_static(owner, method_name) is function
            assert self._function_current(record)
            receiver = frame.f_locals.get("self")
            assert type(receiver) is owner
            self.receiver_entries.add(code)
            if label == "app_ctor":
                assert self.app is None
                self.app = weakref.ref(receiver)
            return
        spec = module.__spec__
        assert (
            spec is not None
            and Path(module.__file__).absolute() == self.sources[name][0]
        )
        assert Path(spec.origin).absolute() == self.sources[name][0]
        self.module_records[name] = (
            module,
            module.__dict__,
            module.__file__,
            spec,
            spec.origin,
            spec.loader,
            module.__loader__,
        )
        class_name, method_name = member.split(".")
        owner = vars(module)[class_name]
        function = inspect.getattr_static(owner, method_name)
        assert type(function) is FunctionType and function.__globals__ is vars(module)
        assert function.__code__ is code
        if code.co_freevars:
            assert code.co_freevars == ("__class__",)
            assert function.__closure__ is not None and len(function.__closure__) == 1
            assert function.__closure__[0].cell_contents is owner
        else:
            assert function.__closure__ is None
        self.method_records[code] = (
            module,
            class_name,
            owner,
            method_name,
            function,
            self._function_record(function),
        )

    def _on_start(self, code, offset):
        if not self.active or code is self.loader.__code__:
            return
        if code in self.receiver_entries and self.codes[code][2] == "native_open":
            ident = threading.get_ident()
            self.counts["native_open"] = self.counts.get("native_open", 0) + 1
            self.native_counts[ident] = self.native_counts.get(ident, 0) + 1
            return
        try:
            frame = sys._getframe(1)
            assert frame.f_code is code
            self._bind(code, frame)
            label = self.codes[code][2]
            ident = threading.get_ident()
            if label == "initial_screen":
                assert (
                    ident == self.main_ident and asyncio.get_running_loop() is self.loop
                )
                task = asyncio.current_task()
                assert task is not None
                assert self.initial_task is None or self.initial_task is task
                self.initial_task = task
            self.counts[label] = self.counts.get(label, 0) + 1
            if label == "native_open":
                self.native_counts[ident] = self.native_counts.get(ident, 0) + 1
                return
            assert len(self.states) < 512
            key = (ident, id(frame))
            assert key not in self.states
            self.states[key] = (
                label,
                time.perf_counter(),
                self.native_counts.get(ident, 0),
                ident == self.main_ident,
            )
        except Exception as error:
            self._issue("start:" + type(error).__name__)

    def _on_return(self, code, offset, value):
        if not self.active:
            return
        try:
            frame = sys._getframe(1)
            assert frame.f_code is code
            if code is self.loader.__code__:
                self._publish(frame.f_locals.get("self"), value)
            elif code is self.loader_exec.__code__:
                # Pin the actual defining function/class/closure immediately at
                # original module publication, before any first-use replacement.
                loader, module = (
                    frame.f_locals.get("self"),
                    frame.f_locals.get("module"),
                )
                if (
                    type(loader) is importlib.machinery.SourceFileLoader
                    and type(module) is ModuleType
                ):
                    for selected, spec in tuple(self.codes.items()):
                        if spec[0] == module.__name__:
                            assert sys.modules.get(spec[0]) is module
                            self._bind(selected)
            else:
                self._finish(frame, "normal_return")
        except Exception as error:
            self._issue("return:" + type(error).__name__)

    def _finish(self, frame, kind):
        state = self.states.pop((threading.get_ident(), id(frame)), None)
        if state is None:
            self._issue("finish_without_segment")
            return
        label, start, native_before, main = state
        row = dict(
            label=label,
            start=start,
            end=time.perf_counter(),
            main_thread=main,
            native_entries=self.native_counts.get(threading.get_ident(), 0)
            - native_before,
            edge=kind,
        )
        if len(self.rows) < 512:
            self.rows.append(row)
        else:
            self.overflow += 1

    def _capture_preparation(self, frame):
        """Retain only the child issued by the observed original startup task."""
        self._bind(frame.f_code, frame)
        runtime = frame.f_locals["self"]
        app = None if self.app is None else self.app()
        assert app is not None and frame.f_locals["app"] is app and runtime._app is app
        owner_task = frame.f_locals["owner_task"]
        assert owner_task is self.initial_task is asyncio.current_task()
        assert frame.f_locals["owner_loop"] is asyncio.get_running_loop() is self.loop
        assert frame.f_locals["owner_thread"] is threading.current_thread()
        assert threading.get_ident() == self.main_ident
        pending = frame.f_locals["pending"]
        assert type(pending) is asyncio.Task and pending.get_loop() is self.loop
        initialize = frame.f_locals["initialize"]
        current = frame.f_locals["source_current"]
        assert type(initialize) is FunctionType and type(current) is FunctionType
        assert initialize.__code__ is nested(
            frame.f_code, frame.f_code.co_qualname + ".<locals>.initialize"
        )
        assert current.__code__ is nested(
            frame.f_code, frame.f_code.co_qualname + ".<locals>.source_current"
        )
        assert initialize.__globals__ is current.__globals__ is frame.f_globals
        reader = frame.f_locals["reader"]
        assert type(reader) is MethodType and reader.__self__ is runtime
        _, _, owner, _, function, reader_record = next(
            record
            for code, record in self.method_records.items()
            if self.codes[code][2] == "receipts"
        )
        assert reader.__func__ is function and self._function_current(reader_record)
        scope = frame.f_locals["scope"]
        anchor = frame.f_locals["anchor"]
        assert frame.f_globals["_INITIAL_ACTIVITY_RECEIPT_READERS"] is anchor
        assert anchor[0] is owner and anchor[2] is scope
        assert frame.f_globals["_INITIAL_ACTIVITY_RECEIPT_SCOPE"] is scope
        child = pending.get_coro()
        thread_function = asyncio.to_thread
        assert child.cr_code is thread_function.__code__
        module = sys.modules[thread_function.__module__]
        path = Path(module.__file__).absolute()
        raw = path.read_bytes()
        expected = nested(
            compile(
                raw,
                thread_function.__code__.co_filename,
                "exec",
                dont_inherit=True,
                optimize=sys.flags.optimize,
            ),
            "to_thread",
        )
        assert thread_function.__globals__ is vars(module) and shape(expected) == shape(
            thread_function.__code__
        )
        spec = module.__spec__
        assert spec is not None and Path(spec.origin).absolute() == path
        self.preparation_thread_record = (
            thread_function,
            self._function_record(thread_function),
        )
        self.preparation_thread_binding = (
            module,
            module.__dict__,
            module.__file__,
            spec,
            spec.origin,
            spec.loader,
            module.__loader__,
            path,
            hashlib.sha256(raw).hexdigest(),
        )
        actor = (
            runtime,
            app,
            owner_task,
            pending,
            initialize,
            self._function_record(initialize),
            current,
            self._function_record(current),
            reader,
            scope,
            anchor,
            owner,
        )
        assert self.preparation_actor is None or self.preparation_actor[3] is pending
        self.preparation_actor = actor

    def _qualify_preparation_worker(self, runtime, app, initialize_frame, attempt):
        assert self.preparation_actor is not None and initialize_frame is not None
        (
            selected_runtime,
            selected_app,
            owner_task,
            pending,
            initialize,
            initialize_record,
            current,
            current_record,
            reader,
            scope,
            anchor,
            owner,
        ) = self.preparation_actor
        assert runtime is selected_runtime and app is selected_app
        assert owner_task is self.initial_task and owner_task.get_loop() is self.loop
        assert pending.get_loop() is self.loop and not pending.done()
        assert initialize_frame.f_code is initialize.__code__
        assert initialize_frame.f_globals is initialize.__globals__
        assert self._function_current(initialize_record) and self._function_current(
            current_record
        )
        namespace = initialize.__globals__
        assert namespace["ConsoleRuntime"] is owner
        assert (
            namespace["_INITIAL_ACTIVITY_RECEIPT_READERS"] is anchor
            and anchor[2] is scope
        )
        assert namespace["_INITIAL_ACTIVITY_RECEIPT_SCOPE"] is scope
        assert reader.__self__ is runtime and reader.__func__ is inspect.getattr_static(
            owner, "ensure_activity_receipt_service"
        )
        proof = getattr(scope, "proof", None)
        assert (
            type(proof) is tuple
            and len(proof) == 2
            and proof[0] is runtime
            and proof[1] is current
        )
        child = pending.get_coro()
        assert child.cr_code is self.preparation_thread_record[0].__code__
        assert (
            child.cr_frame is not None and child.cr_frame.f_locals["func"] is initialize
        )
        assert self._function_current(self.preparation_thread_record[1])
        attempt["startup_worker_source_qualified"] = True
        attempt["issued_by_actual_initial_task"] = True
        attempt["original_owner_loop_is_actual_UI_loop"] = True

    def _preparation_current(self):
        if self.preparation_actor is None:
            return True
        (
            runtime,
            app,
            owner_task,
            pending,
            initialize,
            initialize_record,
            current,
            current_record,
            reader,
            scope,
            anchor,
            owner,
        ) = self.preparation_actor
        (
            module,
            namespace,
            filename,
            spec,
            origin,
            loader,
            module_loader,
            path,
            digest,
        ) = self.preparation_thread_binding
        function, record = self.preparation_thread_record
        return (
            asyncio.to_thread is function
            and self._function_current(record)
            and self._function_current(initialize_record)
            and self._function_current(current_record)
            and initialize.__globals__["ConsoleRuntime"] is owner
            and initialize.__globals__["_INITIAL_ACTIVITY_RECEIPT_READERS"] is anchor
            and initialize.__globals__["_INITIAL_ACTIVITY_RECEIPT_SCOPE"] is scope
            and reader.__self__ is runtime
            and reader.__func__
            is inspect.getattr_static(owner, "ensure_activity_receipt_service")
            and sys.modules.get(module.__name__) is module
            and module.__dict__ is namespace
            and module.__file__ == filename
            and module.__spec__ is spec
            and spec.origin == origin
            and spec.loader is loader
            and module.__loader__ is module_loader
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        )

    def _on_yield(self, code, offset, value):
        if self.active:
            try:
                frame = sys._getframe(1)
                assert frame.f_code is code
                if (
                    self.codes[code][2] == "receipt_preparation"
                    and "pending" in frame.f_locals
                ):
                    self._capture_preparation(frame)
                self._finish(frame, "yield")
            except Exception as error:
                self._issue("yield:" + type(error).__name__)

    def _on_resume(self, code, offset):
        if self.active:
            try:
                frame = sys._getframe(1)
                assert frame.f_code is code
                ident = threading.get_ident()
                key = ident, id(frame)
                assert key not in self.states and len(self.states) < 512
                self.states[key] = (
                    self.codes[code][2],
                    time.perf_counter(),
                    self.native_counts.get(ident, 0),
                    ident == self.main_ident,
                )
            except Exception as error:
                self._issue("resume:" + type(error).__name__)

    def _qualify_admitted_native(self, database, connection, attempt):
        # No import, acquisition, query or enumeration: only the reached exact
        # admitted native object and its original defining wrapper/custody.
        attempt["native_qualifier_step"] = "loaded_original_modules"
        private_name = "tldw_chatbook.DB.private_sqlite"
        storage_name = "tldw_chatbook.Backup_Recovery.storage_admission"
        bootstrap_name = "tldw_chatbook.Backup_Recovery.bootstrap"
        private, storage = sys.modules.get(private_name), sys.modules.get(storage_name)
        bootstrap = sys.modules.get(bootstrap_name)
        assert type(private) is ModuleType and type(storage) is ModuleType
        assert type(bootstrap) is ModuleType
        owner = type(connection)
        attempt["connection_type"] = {
            "module": owner.__module__,
            "qualname": owner.__qualname__,
            "mro": [(base.__module__, base.__qualname__) for base in owner.__mro__],
        }
        attempt["native_qualifier_step"] = "exact_admitted_native_class"
        assert owner.__mro__ == (owner, sqlite3.Connection, object)
        assert owner.__module__ == private_name
        assert (
            owner.__qualname__
            == "_with_storage_admission.<locals>.admitted.<locals>.AdmittedConnection"
        )
        attempt["native_qualifier_step"] = "native_module_source_origin"
        for module in (private, storage, bootstrap):
            name = module.__name__
            if name not in self.native_source_records:
                path = self.repo / (name.replace(".", "/") + ".py")
                assert Path(module.__file__).absolute() == path
                spec = module.__spec__
                assert spec is not None and Path(spec.origin).absolute() == path
                raw = path.read_bytes()
                self.native_source_records[name] = (
                    module,
                    module.__dict__,
                    module.__file__,
                    spec,
                    spec.origin,
                    spec.loader,
                    module.__loader__,
                    path,
                    hashlib.sha256(raw).hexdigest(),
                )
        attempt["native_qualifier_step"] = "original_builder_identity"
        builder = vars(private)["_with_storage_admission"]
        assert type(builder) is FunctionType and builder.__globals__ is vars(private)
        expected = nested(
            compile(
                self.native_source_records[private_name][-2].read_bytes(),
                builder.__code__.co_filename,
                "exec",
                dont_inherit=True,
                optimize=sys.flags.optimize,
            ),
            "_with_storage_admission",
        )
        attempt["native_qualifier_step"] = "original_builder_source_shape"
        assert shape(builder.__code__) == shape(expected)
        attempt["native_qualifier_step"] = "original_admitted_close_identity"
        close = inspect.getattr_static(owner, "close")
        assert type(close) is FunctionType and close.__globals__ is vars(private)
        attempt["native_qualifier_step"] = "original_nested_close_code"
        assert close.__code__ is nested(builder.__code__, owner.__qualname__ + ".close")
        attempt["native_qualifier_step"] = "original_close_defaults"
        assert close.__defaults__ is None and close.__kwdefaults__ is None
        attempt["native_qualifier_step"] = "original_close_closure"
        cells = dict(
            zip(
                close.__code__.co_freevars,
                (cell.cell_contents for cell in close.__closure__),
            )
        )
        attempt["close_freevars"] = list(close.__code__.co_freevars)
        assert close.__code__.co_freevars == (
            "RecoveryRequired",
            "__class__",
            "capture_lease",
            "constructing",
            "lease",
        )
        attempt["native_qualifier_step"] = "original_closing_exception_class"
        recovery = vars(bootstrap)["RecoveryRequired"]
        assert cells["RecoveryRequired"] is recovery and type(recovery) is type
        assert (
            recovery.__module__ == bootstrap_name
            and recovery.__qualname__ == "RecoveryRequired"
        )
        assert recovery.__bases__ == (RuntimeError,)
        assert recovery.__mro__ == (
            recovery,
            RuntimeError,
            Exception,
            BaseException,
            object,
        )
        class_tree = ast.parse(
            self.native_source_records[bootstrap_name][-2].read_bytes()
        )
        class_node = next(
            node
            for node in class_tree.body
            if type(node) is ast.ClassDef and node.name == "RecoveryRequired"
        )
        assert (
            not class_node.decorator_list
            and not class_node.keywords
            and not class_node.type_params
        )
        assert len(class_node.bases) == 1 and type(class_node.bases[0]) is ast.Name
        assert class_node.bases[0].id == "RuntimeError"
        assert len(class_node.body) == 1 and type(class_node.body[0]) is ast.Expr
        assert type(class_node.body[0].value) is ast.Constant
        assert type(class_node.body[0].value.value) is str  # noqa: E721 -- exact declared docstring, no custom values.
        assert set(vars(recovery)) == {"__module__", "__doc__", "__weakref__"}
        assert recovery.__doc__ == class_node.body[0].value.value
        assert vars(recovery)["__weakref__"].__objclass__ is recovery
        self.native_class_records = [
            (
                bootstrap,
                "RecoveryRequired",
                recovery,
                recovery.__bases__,
                recovery.__mro__,
                tuple(vars(recovery).items()),
            )
        ]
        attempt["original_closing_exception_class_qualified"] = True
        assert cells["__class__"] is owner and cells["capture_lease"] is False
        assert cells["constructing"] is False
        attempt["native_qualifier_step"] = "exact_native_lease_custody"
        with storage._lock:
            lease = vars(private)["_ordinary_connections"].get(connection)
            participant = database._maintenance_participant
            attempt["native_custody"] = {
                "ordinary_lease_present": lease is not None,
                "closure_lease_same": cells["lease"] is lease,
                "lease_live": lease in storage._live_leases,
                "registered_participant_same": getattr(
                    lease, "resource_participant", None
                )
                is participant,
                "participant_repository_same": participant.repository() is database,
                "registered_connection_lease_same": participant.connections.get(
                    connection
                )
                is lease,
                "lease_path_same": getattr(lease, "resource_path", None)
                == database.db_path,
                "lease_thread_same": getattr(lease, "resource_thread", None)
                is threading.current_thread(),
                "lease_close_failed": getattr(lease, "resource_close_failed", None)
                is True,
            }
            assert cells["lease"] is lease and lease in storage._live_leases
            assert lease.resource_participant is participant
            assert participant.repository() is database
            assert participant.connections.get(connection) is lease
            assert lease.resource_path == database.db_path
            assert lease.resource_thread is threading.current_thread()
            assert not lease.resource_close_failed
        attempt["native_qualifier_step"] = "unchanged_native_descriptor_live"
        assert physical_open(connection)
        self.native_method_records = [
            (
                private,
                "_with_storage_admission",
                builder,
                self._function_record(builder),
            ),
            (owner, "close", close, self._function_record(close)),
        ]
        self.held_native_lease, self.held_native_participant = lease, participant
        attempt["original_admitted_native_qualified"] = True
        attempt["exact_lease_live"] = True
        return True

    def _native_sources_current(self):
        return (
            all(
                sys.modules.get(name) is module
                and module.__dict__ is namespace
                and module.__file__ == filename
                and module.__spec__ is spec
                and spec.origin == origin
                and spec.loader is loader
                and module.__loader__ is module_loader
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                for name, (
                    module,
                    namespace,
                    filename,
                    spec,
                    origin,
                    loader,
                    module_loader,
                    path,
                    digest,
                ) in self.native_source_records.items()
            )
            and all(
                inspect.getattr_static(owner, name) is function
                and self._function_current(record)
                for owner, name, function, record in self.native_method_records
            )
            and all(
                vars(module).get(name) is owner
                and owner.__bases__ is bases
                and owner.__mro__ is mro
                and tuple(vars(owner)) == tuple(key for key, _ in attributes)
                and all(vars(owner).get(key) is value for key, value in attributes)
                for module, name, owner, bases, mro, attributes in self.native_class_records
            )
        )

    def held_native_retired(self):
        if self.held_connection is None or self.held_native_lease is None:
            return False
        storage = sys.modules.get("tldw_chatbook.Backup_Recovery.storage_admission")
        private = sys.modules.get("tldw_chatbook.DB.private_sqlite")
        assert type(storage) is ModuleType and type(private) is ModuleType
        with storage._lock:
            return (
                not physical_open(self.held_connection)
                and self.held_native_lease not in storage._live_leases
                and self.held_connection not in self.held_native_participant.connections
                and vars(private)["_ordinary_connections"].get(self.held_connection)
                is None
                and not self.held_native_lease.resource_close_failed
            )

    def _on_line(self, code, lineno):
        if not self.active or self.hold is not None or lineno != self.schema_line:
            return
        attempt = {
            "stage": "selected_schema_entry",
            "known_ancestry": [],
            "original_parent_codes": [],
            "scope": "unqualified",
        }
        if len(self.schema_attempts) >= 16:
            self.overflow += 1
            self._issue("schema_attempt_overflow")
            return
        self.schema_attempts.append(attempt)
        try:
            frame = sys._getframe(1)
            assert frame.f_code is code and self.codes[code][2] == "schema"
            database, connection = frame.f_locals["self"], frame.f_locals["conn"]
            attempt["native_sqlite_connection"] = type(connection) is sqlite3.Connection
            ancestors, runtime, initialize_frame = [], None, None
            parent = frame.f_back
            attempt["stage"] = "original_ancestry"
            for _ in range(96):
                if parent is None:
                    break
                if (
                    self.preparation_actor is not None
                    and parent.f_code is self.preparation_actor[4].__code__
                ):
                    assert initialize_frame is None
                    initialize_frame = parent
                if parent.f_code in self.codes:
                    spec = self.codes[parent.f_code]
                    assert parent.f_globals is vars(sys.modules[spec[0]])
                    ancestors.append(spec[2])
                    if spec[2] == "receipts":
                        runtime = parent.f_locals["self"]
                name = parent.f_globals.get("__name__")
                if name in self.module_codes:
                    expected = nested(
                        self.module_codes[name], parent.f_code.co_qualname
                    )
                    assert expected is parent.f_code
                    assert parent.f_globals is vars(sys.modules[name])
                    if len(attempt["original_parent_codes"]) < 24:
                        attempt["original_parent_codes"].append(
                            {
                                "module": name,
                                "qualname": parent.f_code.co_qualname,
                                "first_line": parent.f_code.co_firstlineno,
                                "code_from_original_loader": True,
                            }
                        )
                    else:
                        raise AssertionError("parent code bound")
                parent = parent.f_back
            else:
                raise AssertionError("known ancestry bound")
            attempt["known_ancestry"] = ancestors
            if "receipts" not in ancestors:
                attempt.update(scope="outside_scope", stage="unrelated_schema_declined")
                return
            attempt["scope"] = "console_receipts"
            attempt["stage"] = "main_or_worker_route"
            main_thread = threading.get_ident() == self.main_ident
            if main_thread:
                assert {"bridge", "compose"} <= set(ancestors)
            else:
                assert self.counts.get("initial_screen", 0) > 0
                assert "compose" not in ancestors and "cost" not in ancestors
            attempt["stage"] = "same_original_app_runtime_database"
            app = None if self.app is None else self.app()
            assert app is not None and runtime._app is app
            if not main_thread:
                self._qualify_preparation_worker(
                    runtime, app, initialize_frame, attempt
                )
            assert (
                Path(database.db_path)
                == Path(app.chachanotes_db.db_path).parent / "agent_runs.db"
            )
            attempt["stage"] = "original_native_connection_live"
            assert self._qualify_admitted_native(database, connection, attempt)
            try:
                shared_loop = asyncio.get_running_loop() is self.loop
            except RuntimeError:
                shared_loop = False
            assert not main_thread or shared_loop
            self.held_database, self.held_connection = database, connection
            self.hold = dict(
                ancestry=ancestors,
                native_connection_actor=id(connection),
                database_actor=id(database),
                runtime_actor=id(runtime),
                shared_actual_loop=shared_loop,
                main_thread=main_thread,
                physically_open_at_hold=True,
                entered=time.perf_counter(),
                hold_enabled=self.hold_native,
            )
            attempt["stage"] = "original_native_held"
            self.entered.set()
            if not self.hold_native:
                return
            self.controller_thread = threading.Thread(
                target=self._control, name="exact-startup-receipt-gate-controller"
            )
            self.controller_thread.start()
            assert self.release.wait(
                10
            ), "held native observer controller did not release"
            self.hold["released"] = time.perf_counter()
            attempt["stage"] = "original_native_released"
        except Exception as error:
            traceback = error.__traceback__
            for _ in range(8):
                if traceback is None:
                    break
                if (
                    traceback.tb_frame.f_code
                    is StartupReceiptNativeGate._qualify_admitted_native.__code__
                ):
                    attempt["native_qualification_failure_line"] = traceback.tb_lineno
                    break
                traceback = traceback.tb_next
            self._issue("line:" + attempt["stage"] + ":" + type(error).__name__)
            self.entered.set()
            self.release.set()

    def _probe_progress(self):
        if not self.release.is_set():
            self.progress.set()

    def _control(self):
        if not self.entered.wait(10):
            self._issue("original_native_stage_not_reached")
            self.release.set()
            return
        try:
            self.loop.call_soon_threadsafe(self._probe_progress)
            self.progress.wait(0.1)
        finally:
            self.release.set()

    def start(self):
        self.tool = next(
            (number for number in (5, 4, 3) if self.monitor.get_tool(number) is None),
            None,
        )
        assert self.tool is not None
        self.monitor.use_tool_id(self.tool, "startup-original-receipt-native-gate")
        try:
            assert self.monitor.get_events(self.tool) == 0
            events = self.monitor.events
            for event, callback in (
                (events.PY_START, self._on_start),
                (events.PY_RETURN, self._on_return),
                (events.PY_YIELD, self._on_yield),
                (events.PY_RESUME, self._on_resume),
                (events.LINE, self._on_line),
            ):
                assert (
                    self.monitor.register_callback(self.tool, event, callback) is None
                )
                self.callbacks[event] = callback
            self.active = True
            self._enable(self.loader.__code__, events.PY_RETURN)
            self._enable(self.loader_exec.__code__, events.PY_RETURN)
        except BaseException:
            self.stop()
            raise

    def current(self):
        return (
            self._native_sources_current()
            and self._preparation_current()
            and inspect.getattr_static(importlib.machinery.SourceFileLoader, "get_code")
            is self.loader
            and self._function_current(self.loader_record)
            and inspect.getattr_static(
                importlib.machinery.SourceFileLoader, "exec_module"
            )
            is self.loader_exec
            and self._function_current(self.loader_exec_record)
            and all(
                sys.modules.get(name) is module
                and module.__name__ == name
                and module.__dict__ is namespace
                and module.__file__ == filename
                and module.__spec__ is spec
                and spec.origin == origin
                and spec.loader is loader
                and module.__loader__ is module_loader
                for name, (
                    module,
                    namespace,
                    filename,
                    spec,
                    origin,
                    loader,
                    module_loader,
                ) in self.module_records.items()
            )
            and all(
                vars(module).get(class_name) is owner
                and inspect.getattr_static(owner, method_name) is function
                and self._function_current(record)
                for module, class_name, owner, method_name, function, record in self.method_records.values()
            )
            and all(
                hashlib.sha256(path.read_bytes()).hexdigest() == digest
                for path, digest in self.sources.values()
            )
        )

    def stop(self):
        self.release.set()
        if self.controller_thread is not None:
            self.controller_thread.join(10)
            if self.controller_thread.is_alive():
                self._issue("known_controller_not_retired")
        global_zero = self.tool is None or self.monitor.get_events(self.tool) == 0
        masks_current, callbacks_current = True, True
        try:
            if self.tool is not None:
                for code, mask in self.masks.items():
                    masks_current &= (
                        self.monitor.get_local_events(self.tool, code) == mask
                    )
                    self.monitor.set_local_events(self.tool, code, 0)
                for event, callback in self.callbacks.items():
                    callbacks_current &= (
                        self.monitor.register_callback(self.tool, event, None)
                        is callback
                    )
        finally:
            self.active = False
            if self.tool is not None:
                self.monitor.free_tool_id(self.tool)
        return dict(
            diagnostic_only=True,
            hold=self.hold,
            schema_attempts=self.schema_attempts,
            exact_held_native_and_lease_retired=self.held_native_retired(),
            ui_progress_while_original_native_held=self.progress.is_set(),
            original_source_current=self.current(),
            global_events_zero=global_zero,
            local_masks_owned=masks_current,
            callbacks_owned=callbacks_current,
            local_masks_retired=all(
                self.monitor.get_local_events(self.tool, code) == 0
                for code in self.masks
            ),
            tool_retired=self.tool is None or self.monitor.get_tool(self.tool) is None,
            unmatched_segments=len(self.states),
            invalid=self.invalid,
            overflow=self.overflow,
            selected_starts=self.counts,
            segments=self.rows,
            source_hashes={name: digest for name, (_, digest) in self.sources.items()},
            counts_are_inclusive_and_overlap=True,
            held_duration_is_not_product_timing=self.hold_native,
            passive_durations_do_not_qualify_unobserved_startup_budgets=True,
        )
