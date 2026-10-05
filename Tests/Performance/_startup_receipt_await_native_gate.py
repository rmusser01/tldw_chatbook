"""Post-repair leaf controls: exact Runtime/native SQL, no App launch."""

import ast
import asyncio
import hashlib
import inspect
from pathlib import Path
import sys
import threading
from types import CoroutineType, FunctionType
from Tests.Performance._startup_receipt_native_gate import (
    StartupReceiptNativeGate,
    nested,
    shape,
)


# Exact v5 qualifier body; additive startup ancestry outside this method is allowed.
_QUALIFIER_V5_AST_SHA256 = (
    "80a20e843573c139c8870751a7f55cb7aaddb93425b3f35e641abadc887eeee7"
)


class ReceiptAwaitNativeGate(StartupReceiptNativeGate):
    def __init__(self, repo, loop, runtime, database):
        super().__init__(repo, loop, _specs=())
        self.runtime, self.database = runtime, database
        self.reader_returned = threading.Event()
        self.rows = []
        self.control_records = []
        self.control_modules = {}
        self.queued_callback_qualified = False
        self.queue_records = []
        self.stopped_receipt = None
        original_module = sys.modules[StartupReceiptNativeGate.__module__]
        original_path = self.repo / "Tests/Performance/_startup_receipt_native_gate.py"
        assert Path(original_module.__file__).absolute() == original_path
        original_raw = original_path.read_bytes()
        original_tree = ast.parse(original_raw)
        original_owner = next(
            node
            for node in original_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "StartupReceiptNativeGate"
        )
        original_qualifier = next(
            node
            for node in original_owner.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_qualify_admitted_native"
        )
        assert (
            hashlib.sha256(
                ast.dump(original_qualifier, include_attributes=False).encode()
            ).hexdigest()
            == _QUALIFIER_V5_AST_SHA256
        )
        original_spec = original_module.__spec__
        assert (
            original_spec is not None
            and Path(original_spec.origin).absolute() == original_path
        )
        self.original_binding = (
            original_module,
            original_module.__dict__,
            original_module.__file__,
            original_spec,
            original_spec.origin,
            original_spec.loader,
            original_module.__loader__,
            original_path,
            hashlib.sha256(original_raw).hexdigest(),
            StartupReceiptNativeGate,
        )
        self.original_records = []
        original_code = compile(
            original_raw,
            str(original_path),
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
        for member in (
            "__init__",
            "_qualify_admitted_native",
            "_native_sources_current",
            "held_native_retired",
        ):
            function = inspect.getattr_static(StartupReceiptNativeGate, member)
            assert (
                type(function) is FunctionType
                and function.__globals__ is original_module.__dict__
            )
            assert shape(function.__code__) == shape(
                nested(original_code, function.__code__.co_qualname)
            )
            self.original_records.append(
                (member, function, self._function_record(function))
            )
        self.thread_function = asyncio.to_thread
        self.thread_record = self._function_record(self.thread_function)
        thread_module = sys.modules[self.thread_function.__module__]
        thread_path = Path(thread_module.__file__).absolute()
        thread_raw = thread_path.read_bytes()
        thread_spec = thread_module.__spec__
        assert (
            thread_spec is not None
            and Path(thread_spec.origin).absolute() == thread_path
        )
        assert shape(self.thread_function.__code__) == shape(
            nested(
                compile(
                    thread_raw,
                    self.thread_function.__code__.co_filename,
                    "exec",
                    dont_inherit=True,
                    optimize=sys.flags.optimize,
                ),
                "to_thread",
            )
        )
        self.thread_binding = (
            thread_module,
            thread_module.__dict__,
            thread_module.__file__,
            thread_spec,
            thread_spec.origin,
            thread_spec.loader,
            thread_module.__loader__,
            thread_path,
            hashlib.sha256(thread_raw).hexdigest(),
        )
        names = (
            ("tldw_chatbook.Chat.console_runtime", "ConsoleRuntime", "__init__"),
            ("tldw_chatbook.Chat.console_runtime", "ConsoleRuntime", "dispose"),
            (
                "tldw_chatbook.Chat.console_runtime",
                "ConsoleRuntime",
                "ensure_activity_receipt_service",
            ),
            (
                "tldw_chatbook.Chat.console_runtime",
                "ConsoleRuntime",
                "_prepare_initial_activity_receipts",
            ),
            ("tldw_chatbook.DB.AgentRuns_DB", "AgentRunsDB", "_initialize_schema"),
        )
        for module_name, class_name, name in names:
            module = sys.modules[module_name]
            owner = vars(module)[class_name]
            spec = module.__spec__
            assert spec is not None
            self.control_modules[module_name] = (
                module,
                module.__dict__,
                module.__file__,
                spec,
                spec.origin,
                spec.loader,
                module.__loader__,
            )
            function = inspect.getattr_static(owner, name)
            assert type(function) is FunctionType and function.__globals__ is vars(
                module
            )
            path = Path(module.__file__).absolute()
            assert path == self.repo / (module_name.replace(".", "/") + ".py")
            raw = path.read_bytes()
            original = nested(
                compile(
                    raw,
                    function.__code__.co_filename,
                    "exec",
                    dont_inherit=True,
                    optimize=sys.flags.optimize,
                ),
                function.__code__.co_qualname,
            )
            assert shape(original) == shape(function.__code__)
            self.control_records.append(
                (
                    module,
                    class_name,
                    owner,
                    name,
                    function,
                    self._function_record(function),
                    path,
                    hashlib.sha256(raw).hexdigest(),
                )
            )
            if name == "ensure_activity_receipt_service":
                self.reader_code = function.__code__
                self.reader_function = function
            if name == "_prepare_initial_activity_receipts":
                self.prepare_code = function.__code__
                self.initialize_code = nested(
                    function.__code__,
                    function.__code__.co_qualname + ".<locals>.initialize",
                )
                self.source_current_code = nested(
                    function.__code__,
                    function.__code__.co_qualname + ".<locals>.source_current",
                )
                assert (
                    self.initialize_code is not None
                    and self.source_current_code is not None
                )
            if name == "_initialize_schema":
                self.schema_code = function.__code__
                tree = ast.parse(raw)
                klass = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == class_name
                )
                method = next(
                    n
                    for n in klass.body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )
                statement = method.body[0]
                assert isinstance(statement, ast.With) and len(statement.items) == 1
                context = statement.items[0]
                assert (
                    isinstance(context.context_expr, ast.Call)
                    and context.context_expr.func.attr == "connection"
                )
                assert (
                    context.context_expr.func.value.id == "self"
                    and context.optional_vars.id == "conn"
                )
                call = statement.body[0].value
                assert isinstance(call, ast.Call) and call.func.attr == "executescript"
                self.schema_line = call.lineno

    def exact_callback_queued(self, task):
        """Inspect only the exact owned helper task and its stock to_thread child."""
        coroutine = task.get_coro()
        if (
            type(coroutine) is not CoroutineType
            or coroutine.cr_code is not self.prepare_code
        ):
            return False
        frame = coroutine.cr_frame
        if frame is None or frame.f_locals.get("self") is not self.runtime:
            return False
        pending = frame.f_locals.get("pending")
        if (
            type(pending) is not asyncio.Task
            or pending.done()
            or pending.get_loop() is not self.loop
        ):
            return False
        child = pending.get_coro()
        if (
            type(child) is not CoroutineType
            or child.cr_code is not self.thread_function.__code__
            or child.cr_await is None
        ):
            return False
        child_frame = child.cr_frame
        if child_frame is None:
            return False
        function = child_frame.f_locals.get("func")
        if (
            type(function) is not FunctionType
            or function.__code__ is not self.initialize_code
        ):
            return False
        namespace = sys.modules["tldw_chatbook.Chat.console_runtime"].__dict__
        assert function.__globals__ is namespace and frame.f_globals is namespace
        cells = dict(
            zip(
                function.__code__.co_freevars,
                (cell.cell_contents for cell in function.__closure__),
            )
        )
        assert cells["self"] is self.runtime
        reader = cells["reader"]
        assert (
            reader.__self__ is self.runtime and reader.__func__ is self.reader_function
        )
        current = cells["source_current"]
        assert (
            type(current) is FunctionType
            and current.__code__ is self.source_current_code
        )
        assert current.__globals__ is namespace
        self.queue_records = [
            (function, self._function_record(function)),
            (current, self._function_record(current)),
        ]
        self.queued_callback_qualified = True
        return True

    def start(self):
        super().start()
        try:
            self._enable(self.schema_code, self.monitor.events.LINE)
            self._enable(self.reader_code, self.monitor.events.PY_RETURN)
        except BaseException:
            self.stop()
            raise

    def _on_line(self, code, line):
        if (
            not self.active
            or code is not self.schema_code
            or line != self.schema_line
            or self.hold is not None
        ):
            return
        attempt = {"scope": "exact_leaf_receipt_callback"}
        self.schema_attempts.append(attempt)
        try:
            frame = sys._getframe(1)
            assert frame.f_code is code
            database, connection = frame.f_locals["self"], frame.f_locals["conn"]
            assert (
                Path(database.db_path)
                == Path(self.database.db_path).parent / "agent_runs.db"
            )
            parent = frame.f_back
            matched = False
            for _ in range(96):
                if parent is None:
                    break
                if parent.f_code is self.reader_code:
                    assert parent.f_locals["self"] is self.runtime
                    assert (
                        parent.f_globals
                        is sys.modules["tldw_chatbook.Chat.console_runtime"].__dict__
                    )
                    matched = True
                    break
                parent = parent.f_back
            assert matched
            assert self.runtime._app.chachanotes_db is self.database
            assert self._qualify_admitted_native(database, connection, attempt)
            self.held_database, self.held_connection = database, connection
            self.hold = attempt
            self.entered.set()
            assert self.release.wait(10), "exact original SQL release expired"
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
            self._issue("leaf_line:" + type(error).__name__)
            self.release.set()
            self.entered.set()

    def _on_return(self, code, offset, value):
        if not self.active or code is not getattr(self, "reader_code", None):
            return
        frame = sys._getframe(1)
        if frame.f_locals.get("self") is self.runtime:
            self.reader_returned.set()

    def stop(self):
        # A failed partial start may already have retired the exact tool.
        if self.stopped_receipt is None:
            self.stopped_receipt = super().stop()
        return self.stopped_receipt

    def current(self):
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
            owner,
        ) = self.original_binding
        (
            thread_module,
            thread_namespace,
            thread_filename,
            thread_spec,
            thread_origin,
            thread_loader,
            thread_module_loader,
            thread_path,
            thread_digest,
        ) = self.thread_binding
        return (
            super().current()
            and sys.modules.get(module.__name__) is module
            and module.__dict__ is namespace
            and module.__file__ == filename
            and module.__spec__ is spec
            and spec.origin == origin
            and spec.loader is loader
            and module.__loader__ is module_loader
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
            and vars(module).get("StartupReceiptNativeGate") is owner
            and all(
                inspect.getattr_static(owner, name) is function
                and self._function_current(record)
                for name, function, record in self.original_records
            )
            and asyncio.to_thread is self.thread_function
            and self._function_current(self.thread_record)
            and sys.modules.get(thread_module.__name__) is thread_module
            and thread_module.__dict__ is thread_namespace
            and thread_module.__file__ == thread_filename
            and thread_module.__spec__ is thread_spec
            and thread_spec.origin == thread_origin
            and thread_spec.loader is thread_loader
            and thread_module.__loader__ is thread_module_loader
            and hashlib.sha256(thread_path.read_bytes()).hexdigest() == thread_digest
            and all(self._function_current(record) for _, record in self.queue_records)
            and all(
                sys.modules.get(name) is module
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
                ) in self.control_modules.items()
            )
            and all(
                vars(module).get(class_name) is owner
                and inspect.getattr_static(owner, name) is function
                and self._function_current(record)
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                for module, class_name, owner, name, function, record, path, digest in self.control_records
            )
        )
