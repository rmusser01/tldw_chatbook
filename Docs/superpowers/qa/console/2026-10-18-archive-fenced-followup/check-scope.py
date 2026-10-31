"""Check only Task5 source boundaries and existing static debt."""

import ast
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BASE = "be3bd821b2936d15ea9c44f1b806289fa2a90326"
QA = Path(__file__).resolve().parent
CONTROLLER = "tldw_chatbook/Chat/console_chat_controller.py"
TEST = "Tests/Chat/test_console_archive_recovery_followup.py"
RUNTIME = "tldw_chatbook/Agents/agent_runtime.py"
summary = {"base": BASE, "interpreter": sys.version, "static": {}, "unchanged": {}}
for name in [CONTROLLER, RUNTIME, TEST]:
    signatures = []
    for phase in ["baseline", "current"]:
        source = (
            subprocess.check_output(["git", "show", f"{BASE}:{name}"])
            if phase == "baseline"
            else Path(name).read_bytes()
        )
        argv = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--output-format=json",
            "--stdin-filename",
            name,
            "-",
        ]
        result = subprocess.run(
            argv,
            input=source,
            stdout=subprocess.PIPE,
            check=False,
            stderr=subprocess.STDOUT,
        )
        label = f"{Path(name).stem}-{phase}"
        (QA / f"{label}-ruff.json").write_bytes(result.stdout)
        issues = json.loads(result.stdout)
        lines = source.decode().splitlines()
        signatures.append(
            collections.Counter(
                (
                    item["code"],
                    item["message"],
                    lines[item["location"]["row"] - 1].strip(),
                )
                for item in issues
            )
        )
        fmt_argv = [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "--stdin-filename",
            name,
            "-",
        ]
        fmt = subprocess.run(
            fmt_argv,
            input=source,
            stdout=subprocess.PIPE,
            check=False,
            stderr=subprocess.STDOUT,
        )
        (QA / f"{label}-format.log").write_bytes(fmt.stdout)
        summary["static"][label] = {
            "ruff_argv": argv,
            "ruff_exit": result.returncode,
            "diagnostics": len(issues),
            "format_argv": fmt_argv,
            "format_exit": fmt.returncode,
            "source_sha256": hashlib.sha256(source).hexdigest(),
        }
    assert signatures[0] == signatures[1], name
    summary["static"][name] = {"same_ruff_signatures": True}
source = Path(CONTROLLER).read_text()
start = source.index(
    "                if (\n                    not killed",
    source.index("# Admission owns the rider"),
)
end = source.index("                ):\n", start) + len("                ):\n")
snippet = (
    "class Scope:\n    def check(self):\n        if True:\n            if True:\n"
    + source[start:end]
    + "                    pass\n"
)
argv = [sys.executable, "-m", "ruff", "format", "--stdin-filename", CONTROLLER, "-"]
formatted = subprocess.run(
    argv,
    input=snippet.encode(),
    capture_output=True,
    check=False,
)
assert formatted.returncode == 0 and formatted.stdout == snippet.encode(), (
    formatted.stdout.decode()
)
summary["changed_predicate_formatted"] = True
runtime = Path(RUNTIME).read_text()
baseline_runtime = subprocess.check_output(
    ["git", "show", f"{BASE}:{RUNTIME}"]
).decode()
old = "                    result = deps.request_conversation_restore(dict(call.args), call_id=call.call_id)"
new = "                    result = deps.request_conversation_restore(\n                        dict(call.args), call_id=current_call_correlation\n                    )"
assert runtime == baseline_runtime.replace(old, new)
wrapper = (
    "class Scope:\n    def check(self):\n        if True:\n            if True:\n                if True:\n"
    + new
    + "\n"
)
formatted = subprocess.run(
    [sys.executable, "-m", "ruff", "format", "--stdin-filename", RUNTIME, "-"],
    input=wrapper.encode(),
    capture_output=True,
    check=False,
)
assert formatted.returncode == 0 and formatted.stdout == wrapper.encode(), (
    formatted.stdout.decode()
)
summary["runtime_only_dispatch_argument_changed"] = True
summary["changed_dispatch_formatted"] = True
paths = [
    "tldw_chatbook/Chat/console_provider_gateway.py",
    "tldw_chatbook/Chat/console_agent_bridge.py",
    "tldw_chatbook/Agents/native_tools.py",
    "tldw_chatbook/Chat/console_runtime.py",
    "tldw_chatbook/Chat/console_trace_runtime.py",
    "tldw_chatbook/Chat/console_trace_service.py",
    "tldw_chatbook/Chat/console_semantic_revision.py",
    "tldw_chatbook/Chat/archive_recovery/runtime.py",
]
for name in paths:
    baseline = subprocess.check_output(["git", "show", f"{BASE}:{name}"])
    current = Path(name).read_bytes()
    assert baseline == current, name
    summary["unchanged"][name] = hashlib.sha256(current).hexdigest()
service = Path("tldw_chatbook/Chat/console_trace_service.py").read_text()
summary["strict_service_guards"] = {}
for name in [
    "_validate_completed_tool_turn",
    "_completed_tool_turn_parent",
    "prepare_current_surface_delta",
    "persist_request",
    "_validate_replacement_projection",
]:
    node = next(
        node
        for node in ast.walk(ast.parse(service))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    body = ast.get_source_segment(service, node).encode()
    summary["strict_service_guards"][name] = hashlib.sha256(body).hexdigest()
(QA / "scope-static-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
