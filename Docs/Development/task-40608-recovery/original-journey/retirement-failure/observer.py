"""Disposable, failure-only observation of two unchanged original warning sites."""

import ast
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from types import CodeType, FunctionType, MethodType

SITES = {
    "_failure": (
        "tldw_chatbook/Chat/console_execution_worker.py", "_BackendService",
        "Console backend {} {} failed ({}); frames={}",
    ),
    "_settle": (
        "tldw_chatbook/Chat/console_execution_retirement.py", "ConsoleTurnRetirement",
        "Console turn retirement unconfirmed; last_check={} effect_facts={} error_type={} reads={} trees={}",
    ),
    "observe": (
        "tldw_chatbook/Chat/console_execution_retirement.py", "_ReadResources",
        "Console received native read settled; elapsed={} thread_cpu={}",
    ),
}
OPS = frozenset({
    "ReceiveTurn", "ReadProjection", "ReadDecisionHead", "ReadRunHistory",
    "UpdateInputs", "CommitPlanAck", "ArchiveSendCheckpointAck", "OpenSession",
    "ResolveDecision", "ProjectDecisionPresentation", "PermissionAction",
    "FenceSource", "ApplySourceRevision", "RefreshConfiguration",
    "SetWorkspaceBindingAccess", "CloseIntake", "Drain", "Shutdown", "CancelTurn",
})
STAGES = frozenset({"admission", "applied", "terminal", "retirement"})
CHECK = re.compile(r"(?:definitions|task|logical_closed|qualified_reads|source_closed|read_[0-9]+|effect_[0-9]+_[a-z_]+)\Z")
TYPES = frozenset({
    "RuntimeError", "ValueError", "TypeError", "OSError", "PermissionError",
    "FileNotFoundError", "CancelledError", "TimeoutError", "AssertionError",
    "RecoveryRequired", "ConsoleProtocolError", "ConsoleDisplayPhaseRefused",
    "ConsoleBackendViewSuperseded", "ConsoleBackendReadinessChanged",
})
EFFECTS = frozenset({
    "received_preparation_read", "archive_read", "durable_commit", "permission",
    "provider_dispatch", "tool_dispatch", "send_selection", "session_settings",
    "saved_open", "transcript_marker_read", "display_readiness",
    "workspace_subagent_counts", "inspector_scope_summary", "index_composition",
    "index_operation", "commit_reservation", "terminal", "postcommit", "trace",
})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def codes(code):
    yield code
    for value in code.co_consts:
        if type(value) is CodeType:
            yield from codes(value)


def checked_site(root, sources, name):
    relative, owner, message = SITES[name]
    path = root / relative
    data = path.read_bytes()
    if digest(data) != sources[relative]:
        raise RuntimeError("Failure observer source changed")
    tree = ast.parse(data)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    method = next(node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    warnings = [node for node in ast.walk(method) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "warning"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "logger"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == message]
    if len(warnings) != 1:
        raise RuntimeError("Original warning site changed")
    return data, warnings[0].lineno


def code_location(root, sources, code, line, verified):
    """Return only a location in an exact pinned original code object."""
    try:
        path = Path(code.co_filename).resolve(strict=True)
        relative = path.relative_to(root).as_posix()
    except (OSError, ValueError):
        return "other"
    if relative not in sources:
        return "other"
    if code not in verified:
        data = path.read_bytes()
        if digest(data) != sources[relative]:
            return "other"
        expected = next((item for item in codes(compile(data, code.co_filename, "exec", dont_inherit=True))
                         if item.co_qualname == code.co_qualname and item == code), None)
        if expected is None:
            return "other"
        verified[code] = (expected.co_qualname, frozenset(line for _, _, line in expected.co_lines()))
    name, lines = verified[code]
    if line not in lines or name != code.co_qualname:
        return "other"
    return f"{name}:{line}"


def trace_location(root, sources, trace, verified):
    return code_location(root, sources, trace.tb_frame.f_code, trace.tb_lineno, verified)


def read_facts(observation, root, sources, verified):
    """Only retained metadata and in-memory Future state; no resource getters."""
    values = vars(observation)
    read, invocation = values.get("read"), values.get("invocation")
    originals = vars(invocation).get("reads", ())
    index = next((i for i, value in enumerate(originals) if value is observation), None)
    callback = read.callback
    if type(callback) is MethodType:
        callback = callback.__func__
    location = "other"
    if type(callback) is FunctionType:
        location = code_location(root, sources, callback.__code__, callback.__code__.co_firstlineno, verified)
    def future_state(value):
        if type(value) not in {asyncio.Future, asyncio.Task}:
            return None
        return {"done": value.done(), "cancelled": value.cancelled()}
    def boolean(value):
        return value if type(value) is bool else None
    return {
        "read_index": index, "callback": location,
        "creator_type": type(read.creator).__name__ if type(read.creator).__name__ in {"ConsoleRuntime", "ConsoleChatController"} else "other",
        "entered": boolean(values.get("entered")), "closed": boolean(values.get("closed")), "invalid": boolean(values.get("invalid")),
        "callback_returned": boolean(read.callback_returned),
        "handoff": future_state(read.retired), "producer": future_state(read._producer),
        "resources": {key: len(value) for key, value in values["observations"].items()
                      if key in {"acquisition", "lease", "raw", "descriptor", "file"} and type(value) is list},
        "connections": len(values["connections"]),
    }


@contextmanager
def failure_facts(root, settings):
    """Observe original synchronous logs; never format arbitrary log messages."""
    if settings.get("retirement_failure_diagnostic") is not True:
        raise RuntimeError("Explicit nonqualifying diagnostic required")
    root = Path(root).resolve(strict=True)
    sources = settings["sources"]
    sites = {name: checked_site(root, sources, name) for name in SITES}
    from loguru import logger

    fd = os.open(Path(os.environ["TLDW_TEST_LAB_OUTPUT"]) / f"retirement-failure-{os.getpid()}.jsonl",
                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    started, count = time.monotonic(), 0
    qualified, trace_codes = {}, {}
    slot = monitored_code = monitored_class = monitored_method = None
    pending = {}
    observed_returns = 0

    def write(value):
        value["elapsed_seconds"] = time.monotonic() - started
        payload = json.dumps(value, separators=(",", ":")).encode() + b"\n"
        while payload:
            size = os.write(fd, payload)
            if size <= 0:
                raise OSError("Diagnostic write unavailable")
            payload = payload[size:]
        os.fsync(fd)  # Failure-only: retain the fact before original retirement.

    def original(frame, name):
        relative, owner, _ = SITES[name]
        data, _line = sites[name]
        if Path(frame.f_code.co_filename).resolve() != root / relative:
            return False
        cls = frame.f_globals.get(owner)
        if type(cls) is not type:
            return False
        method = vars(cls).get(name)
        if name == "_failure":
            if type(method) is not staticmethod:
                return False
            method = method.__func__
        elif name == "observe":
            if type(frame.f_locals.get("self")) is not cls or type(method) is not FunctionType or type(getattr(method, "__wrapped__", None)) is not FunctionType:
                return False
            method = method.__wrapped__
        elif type(frame.f_locals.get("self")) is not cls:
            return False
        if type(method) is not FunctionType or method.__code__ is not frame.f_code or method.__globals__ is not frame.f_globals:
            return False
        if name not in qualified:
            expected = next(code for code in codes(compile(data, frame.f_code.co_filename, "exec", dont_inherit=True))
                            if code.co_qualname == f"{owner}.{name}")
            if expected != frame.f_code or expected.co_qualname != frame.f_code.co_qualname:
                return False
            qualified[name] = (cls, method, frame.f_code)
        return qualified[name] == (cls, method, frame.f_code)

    def register_confirmed(frame):
        nonlocal slot, monitored_code, monitored_class, monitored_method
        if slot is not None:
            return
        cls = frame.f_globals["_ReadResources"]
        method = vars(cls).get("confirmed")
        data = sites["observe"][0]
        expected = next(code for code in codes(compile(data, frame.f_code.co_filename, "exec", dont_inherit=True))
                        if code.co_qualname == "_ReadResources.confirmed")
        if type(method) is not FunctionType or method.__code__ != expected or method.__globals__ is not frame.f_globals:
            raise RuntimeError("Original confirmed source changed")
        tree = ast.parse(data)
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_ReadResources")
        function = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "confirmed")
        assignments = sorted((node for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AugAssign)) and any(isinstance(target, ast.Name) and target.id == "valid"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))), key=lambda node: node.lineno)
        labels = ("acquisition", "file", "raw", "descriptor", "lease", "shared_hold", "connection")
        if len(assignments) != len(labels):
            raise RuntimeError("Original resource predicates changed")
        groups = {line: label for node, label in zip(assignments, labels, strict=True)
                  for line in range(node.lineno, node.end_lineno + 1)}
        monitored_class, monitored_method, monitored_code = cls, method, method.__code__
        monitoring = sys.monitoring
        slot = next(value for value in range(6) if value != monitoring.DEBUGGER_ID and monitoring.get_tool(value) is None)
        monitoring.use_tool_id(slot, "task62-failure-predicate")

        def observe(_code, *args):
            nonlocal observed_returns, count
            actual = sys._getframe(1)
            try:
                if (actual.f_code is not monitored_code or type(actual.f_locals.get("self")) is not monitored_class
                    or vars(monitored_class).get("confirmed") is not monitored_method
                    or monitored_method.__code__ is not monitored_code):
                    return
                invocation = vars(actual.f_locals["self"]).get("invocation")
                if type(invocation) is not actual.f_globals.get("ConsoleTurnRetirement"):
                    return
                if len(args) == 1 and args[0] == function.body[0].lineno:
                    # A prior call may have raised instead of returning. Never
                    # carry its predicate state into a recycled frame identity.
                    pending.pop(id(actual), None)
                if id(actual) not in pending and len(pending) >= 256:
                    monitoring.set_local_events(slot, monitored_code, 0)
                    write({"event": "predicate_cap_reached"})
                    return
                state = pending.setdefault(id(actual), {"group": None, "first_false_group": None})
                if actual.f_locals.get("valid") is False and state["first_false_group"] is None:
                    state["first_false_group"] = state["group"]
                if len(args) == 1:  # LINE carries the original line number.
                    state["group"] = groups.get(args[0], state["group"])
                    return
                pending.pop(id(actual), None)
                observed_returns += 1
                if args[-1] is False and count < 32:
                    facts = read_facts(actual.f_locals["self"], root, settings["failure_trace_sources"], trace_codes)
                    result = {"event": "read_unconfirmed", "return_line": actual.f_lineno,
                              "first_false_group": state["first_false_group"], "read": facts}
                    closed = actual.f_locals.get("closed")
                    result["last_connection_closed"] = closed if type(closed) is bool else None
                    write(result)
                    count += 1
                if observed_returns >= 256:
                    monitoring.set_local_events(slot, monitored_code, 0)
                    write({"event": "predicate_cap_reached"})
            except Exception:
                pending.pop(id(actual), None)
                try:
                    write({"event": "predicate_observation_unavailable"})
                except Exception:
                    pass
        monitoring.register_callback(slot, monitoring.events.LINE, observe)
        monitoring.register_callback(slot, monitoring.events.PY_RETURN, observe)
        monitoring.set_local_events(slot, monitored_code, monitoring.events.LINE | monitoring.events.PY_RETURN)
        write({"event": "predicate_registered", "received_turns_only": True, "return_cap": 256})

    def sink(message):
        nonlocal count
        name = message.record["function"]
        if name not in SITES or count >= 32 or message.record["line"] != sites[name][1]:
            return
        frame = sys._getframe(1)
        try:
            while frame is not None and frame.f_code.co_name != name:
                frame = frame.f_back
            if frame is None or not original(frame, name):
                write({"event": "source_refused"})
                count += 1
                return
            if name == "observe":
                register_confirmed(frame)
                return
            values = frame.f_locals
            if name == "_failure":
                operation, stage, error = (values.get(key) for key in ("operation", "stage", "error"))
                if type(operation) is not str or operation not in OPS or type(stage) is not str or stage not in STAGES or not isinstance(error, BaseException):
                    return
                # Never trust the emitted strings. Qualify each actual traceback
                # code object against a separately pinned original source; unknown
                # or forged code/path/name becomes the fixed literal "other".
                locations, trace = [], error.__traceback__
                while trace is not None and len(locations) < 48:
                    locations.append(trace_location(root, settings["failure_trace_sources"], trace, trace_codes))
                    trace = trace.tb_next
                result = {"event": "backend_failure", "operation": operation, "stage": stage,
                          "error_type": type(error).__name__ if type(error).__name__ in TYPES else "other",
                          "locations": locations[-16:], "locations_truncated": trace is not None}
            else:
                check, effect, error_type = (values.get(key) for key in ("last_check", "last_effect", "error_type"))
                if values.get("confirmed") is not False:
                    return
                if check is not None and (type(check) is not tuple or len(check) != 2 or type(check[0]) is not str or not CHECK.fullmatch(check[0]) or type(check[1]) is not bool):
                    return
                if effect is not None and (type(effect) is not tuple or len(effect) != 3 or any(item is not None and type(item) is not bool for item in effect)):
                    return
                if check is not None and check[0].startswith("effect_"):
                    prefix, number, kind = check[0].split("_", 2)
                    check = (f"{prefix}_{number}_{kind if kind in EFFECTS else 'other'}", check[1])
                owner = values["self"]
                reads, trees = vars(owner).get("reads"), vars(owner).get("trees")
                if type(reads) is not list or type(trees) is not list:
                    return
                result = {"event": "retirement_unconfirmed", "last_check": check,
                          "effect_facts": effect, "error_type": error_type if error_type in TYPES or error_type is None else "other",
                          "reads": len(reads), "trees": len(trees)}
                if check is not None and re.fullmatch(r"read_[0-9]+", check[0]):
                    index = int(check[0][5:])
                    if index < len(reads):
                        result["read"] = read_facts(reads[index], root, settings["failure_trace_sources"], trace_codes)
            write(result)
            count += 1
        except Exception:
            count += 1
            try:
                write({"event": "observation_unavailable"})
            except Exception:
                pass  # Diagnostic failure must not replace the original outcome.
        finally:
            del frame

    token = None
    try:
        write({"event": "installed", "qualifying": False,
               "source_sha256": {relative: sources[relative] for relative, _, _ in SITES.values()}})
        token = logger.add(sink, level="WARNING", enqueue=False, catch=True, backtrace=False, diagnose=False)
        yield
    finally:
        if slot is not None:
            sys.monitoring.set_local_events(slot, monitored_code, 0)
            sys.monitoring.register_callback(slot, sys.monitoring.events.LINE, None)
            sys.monitoring.register_callback(slot, sys.monitoring.events.PY_RETURN, None)
            sys.monitoring.free_tool_id(slot)
        if token is not None:
            logger.remove(token)
        os.close(fd)
