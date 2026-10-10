"""Opt-in original-body exception metadata for the mounted resend regression."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
from types import CodeType

import pytest

TARGET = "Tests/UI/test_console_turn_resend_ui.py::test_console_resend_click_re_runs_a_failed_turn_in_place"
OUTPUT = "TLDW_RESEND_TRACE_FAILURE_RECEIPT"
state = None


def shape(code):
    return (
        code.co_code,
        code.co_exceptiontable,
        code.co_stacksize,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(shape(x) if type(x) is CodeType else x for x in code.co_consts),
    )


def nested(code, name):
    for part in name.split("."):
        if part == "<locals>":
            continue
        code = next(
            x for x in code.co_consts if type(x) is CodeType and x.co_name == part
        )
    return code


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global state
    if not os.environ.get(OUTPUT) or item.nodeid != TARGET:
        return
    from tldw_chatbook.Chat import console_chat_controller as module

    cls = module.ConsoleChatController
    methods = tuple(
        inspect.getattr_static(cls, name)
        for name in (
            "_build_durable_trace_request",
            "_handle_durable_trace_provenance_failure",
        )
    )
    from tldw_chatbook.Chat import console_trace_errors as errors

    constructor = errors.TraceCallPersistenceError.__init__
    error_path = Path(errors.__file__).resolve()
    error_bytes = error_path.read_bytes()
    error_code = compile(error_bytes, str(error_path), "exec", dont_inherit=True)
    assert constructor.__globals__ is errors.__dict__
    assert shape(constructor.__code__) == shape(
        nested(error_code, constructor.__qualname__)
    )
    path = Path(module.__file__).resolve()
    data = path.read_bytes()
    compiled = compile(data, str(path), "exec", dont_inherit=True)
    for method in methods:
        assert method.__globals__ is module.__dict__
        assert shape(method.__code__) == shape(nested(compiled, method.__qualname__))
    rows = []
    monitor = sys.monitoring
    tool = next(i for i in range(6) if monitor.get_tool(i) is None)
    monitor.use_tool_id(tool, "tldw-original-resend-trace-exception")
    assert monitor.get_events(tool) == 0

    def record_error(frame, exception):
        owner = frame.f_locals["self"]
        db = owner.store.persistence.db
        chain = []
        current = exception
        for _ in range(4):
            if current is None:
                break
            sites = []
            tb = current.__traceback__
            for _ in range(18):
                if tb is None:
                    break
                f = tb.tb_frame
                name = f.f_globals.get("__name__", "")
                if type(name) is str and name.startswith("tldw_chatbook."):  # noqa: E721 - require the exact declared source owner
                    sites.append([name, f.f_code.co_qualname, tb.tb_lineno])
                tb = tb.tb_next
            chain.append({"type": type(current).__name__, "sites": sites})
            current = current.__cause__ or current.__context__
        assert len(rows) < 16
        rows.append(
            {
                "event": "original_handler_error",
                "thread_ident": threading.get_ident(),
                "is_memory_db": db.is_memory_db,
                "exception_chain": chain,
            }
        )

    def on_start(code, offset):
        frame = sys._getframe(1)
        assert frame.f_code is code
        if code is methods[1].__code__:
            assert frame.f_globals is module.__dict__
            record_error(frame, frame.f_locals["error"])
            return
        assert code is constructor.__code__ and frame.f_globals is errors.__dict__
        sites = []
        parent = frame.f_back
        for _ in range(8):
            if parent is None:
                break
            name = parent.f_globals.get("__name__", "")
            if type(name) is str and name.startswith("tldw_chatbook."):  # noqa: E721 - require the exact declared source owner
                original_module = sys.modules[name]
                data = Path(original_module.__file__).read_bytes()
                compiled = compile(
                    data, parent.f_code.co_filename, "exec", dont_inherit=True
                )
                assert parent.f_globals is original_module.__dict__
                assert shape(parent.f_code) == shape(
                    nested(compiled, parent.f_code.co_qualname)
                )
                sites.append(
                    [
                        name,
                        parent.f_code.co_qualname,
                        parent.f_lineno,
                        hashlib.sha256(data).hexdigest(),
                    ]
                )
            parent = parent.f_back
        assert len(rows) < 16
        rows.append(
            {
                "event": "original_trace_error_constructor",
                "sites": sites,
                "thread_ident": threading.get_ident(),
            }
        )

    monitor.register_callback(tool, monitor.events.PY_START, on_start)
    monitor.set_local_events(tool, methods[1].__code__, monitor.events.PY_START)
    monitor.set_local_events(tool, constructor.__code__, monitor.events.PY_START)
    state = (
        monitor,
        tool,
        module,
        cls,
        methods,
        tuple(x.__code__ for x in methods) + (constructor.__code__,),
        path,
        hashlib.sha256(data).hexdigest(),
        rows,
    )


def pytest_sessionfinish(session, exitstatus):
    if state is None:
        return
    monitor, tool, module, cls, methods, codes, path, digest, rows = state
    try:
        for code in codes:
            monitor.set_local_events(tool, code, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        assert monitor.get_events(tool) == 0
    finally:
        monitor.free_tool_id(tool)
    current = hashlib.sha256(path.read_bytes()).hexdigest() == digest and all(
        inspect.getattr_static(cls, method.__name__) is method
        and method.__code__ is code
        and method.__globals__ is module.__dict__
        for method, code in zip(methods, codes[:2], strict=True)
    )
    output = Path(os.environ[OUTPUT])
    assert not output.exists()
    output.write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "node": TARGET,
                "rows": rows,
                "original_exitstatus": int(exitstatus),
                "source_current": current,
                "global_events": 0,
                "hooks_retired": monitor.get_tool(tool) is None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert current
