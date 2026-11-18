"""Disposable, failure-only observation of two unchanged original warning sites."""

import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from types import CodeType, FunctionType

SITES = {
    "_failure": (
        "tldw_chatbook/Chat/console_execution_worker.py", "_BackendService",
        "Console backend {} {} failed ({}); frames={}",
    ),
    "_settle": (
        "tldw_chatbook/Chat/console_execution_retirement.py", "ConsoleTurnRetirement",
        "Console turn retirement unconfirmed; last_check={} effect_facts={} error_type={} reads={} trees={}",
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


def trace_location(root, sources, trace, verified):
    """Return only a location in an exact pinned original code object."""
    code = trace.tb_frame.f_code
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
    if trace.tb_lineno not in lines:
        return "other"
    return f"{name}:{trace.tb_lineno}"


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
        if token is not None:
            logger.remove(token)
        os.close(fd)
