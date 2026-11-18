"""Finite optional-stack policy tests; no real C timer, App or product import."""

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import asyncio

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", required=True, type=Path)
parser.add_argument("--output-root", required=True, type=Path)
parser.add_argument("--dependencies", type=Path)
args = parser.parse_args()
output = args.output_root.resolve()
if not output.is_relative_to(args.source_root.resolve() / ".superpowers"):
    raise RuntimeError("Pure outputs must stay beneath selectedsource/.superpowers")
output.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "native_series_method", here / "native_series_method.py"
)
method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(method)
tree = ast.parse((here / "native_product_child.py").read_text(encoding="utf-8"))
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
command = next(
    n for n in main.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "command"
)
watch = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StageWatchdog"
)
results = {}


class Timer:
    def __init__(self):
        self.calls = []

    def dump_traceback_later(self, *arguments, **kwargs):
        self.calls.append(("arm", arguments, kwargs))

    def cancel_dump_traceback_later(self):
        self.calls.append(("cancel",))


async def stage_case(enabled, raises=False):
    with tempfile.TemporaryDirectory(dir=output) as directory:
        timer = Timer()
        namespace = {
            "Path": Path,
            "faulthandler": timer,
            "time": __import__("time"),
            "write_json": lambda path, value: path.write_text(
                json.dumps(value), encoding="utf-8"
            ),
        }
        exec(
            compile(ast.Module(body=[watch], type_ignores=[]), "actual-child", "exec"),
            namespace,
        )
        watchdog = namespace["StageWatchdog"](Path(directory), lambda: None)
        future = asyncio.get_running_loop().create_future()
        calls = []
        identity = object()
        original = RuntimeError("original controlled stage exception")

        class Fixture:
            worker = future

            def new_correlation(self):
                return "a" * 32

            async def stage(self, *values):
                calls.append(values)
                if raises:
                    raise original
                return identity

            def snapshot(self):
                return identity

        namespace.update(
            fixture=Fixture(),
            round_limit=1,
            diagnostic_stack_enabled=enabled,
            arguments=SimpleNamespace(
                negative_controls=False, shape="single", columns=120, rows=40
            ),
            stage_watchdog=watchdog,
            common_observer=None,
            observer_native_state=lambda: {},
            state={"negative_results": None},
            time=__import__("time"),
        )
        factory = ast.FunctionDef(
            name="factory",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="stage_count", ctx=ast.Store())],
                    value=ast.Constant(value=0),
                ),
                copy.deepcopy(command),
                ast.Return(value=ast.Name(id="command", ctx=ast.Load())),
            ],
            decorator_list=[],
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
                "actual-child",
                "exec",
            ),
            namespace,
        )
        caught = None
        returned = None
        try:
            returned = await namespace["factory"]()(
                {"command": "stage", "shape": "single"}
            )
        except BaseException as error:
            caught = error
        future.set_result(None)
        await asyncio.sleep(0)
        arms = [row for row in timer.calls if row[0] == "arm"]
        return {
            "calls_once_unchanged": calls == [("single", "a" * 32)],
            "return_preserved": returned is not None
            and returned["snapshot"] is identity,
            "error_preserved": caught is original,
            "arm_count": len(arms),
            "one20s_repeatFalse": len(arms) == 1
            and arms[0][1] == (20,)
            and arms[0][2]["repeat"] is False,
            "all_timer_calls": len(timer.calls),
        }


for label, enabled, raises in (
    ("default_off", False, False),
    ("opt_in", True, False),
    ("off_original_error", False, True),
):
    actual = asyncio.run(stage_case(enabled, raises))
    expected_arms = 1 if enabled else 0
    results[label] = {
        "actual": actual,
        "pass": actual["calls_once_unchanged"]
        and actual["arm_count"] == expected_arms
        and (actual["error_preserved"] if raises else actual["return_preserved"])
        and (
            actual["one20s_repeatFalse"] if enabled else actual["all_timer_calls"] == 0
        ),
    }
authorization = {
    "qualified_method": True,
    "rounds": 41,
    "feedback_root_reviewed": True,
    "terminal_root_reviewed": True,
}
for label, selection, allowed in (
    (
        "qualification_debug_refused",
        {"method_mode": "qualification", "diagnostic_stack": True},
        False,
    ),
    (
        "qualification_off_allowed",
        {"method_mode": "qualification", "diagnostic_stack": False},
        True,
    ),
    (
        "validation_badflag_refused",
        {"method_mode": "validation", "diagnostic_stack": "true"},
        False,
    ),
):
    try:
        actual = method.runtime_round_limit(selection, authorization)
    except RuntimeError:
        actual = None
    results[label] = {"actual_limit": actual, "pass": (actual is not None) == allowed}
bind = getattr(method, "bind_diagnostic_stack", None)
binding = True
if bind is None:
    binding = False
else:
    for selected, passed, allowed in (
        (False, False, True),
        (True, True, True),
        (False, True, False),
        (True, False, False),
        (False, None, False),
    ):
        try:
            value = bind({"diagnostic_stack": selected}, passed)
            accepted = value is selected
        except RuntimeError:
            accepted = False
        binding = binding and accepted == allowed
results["explicit_selected_child_flag_binding"] = {"pass": binding}
# Default and explicit CLI parsing executes only the actual parser AST prefix.
seeder_tree = ast.parse((here / "prepare_native_run.py").read_text(encoding="utf-8"))
seeder = next(
    n for n in seeder_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
)
prefix = []
for node in seeder.body:
    prefix.append(copy.deepcopy(node))
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "args" for t in node.targets
    ):
        break
base = [
    "qa",
    "--source-root",
    "synthetic-source",
    "--freeze",
    "synthetic-freeze",
    "--shared-dependencies",
    "synthetic-dependencies",
    "--original-home",
    "synthetic-original",
    "--output-root",
    "synthetic-output",
    "--shape",
    "single",
]
for label, extra, want in (
    ("seeder_default_off", [], False),
    ("seeder_opt_in", ["--diagnostic-stack"], True),
):
    old_argv = sys.argv
    sys.argv = base + extra
    try:
        namespace = {"argparse": argparse, "Path": Path}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])),
                "actual-seeder",
                "exec",
            ),
            namespace,
        )
        actual = getattr(namespace["args"], "diagnostic_stack", None)
    except SystemExit:
        actual = None
    finally:
        sys.argv = old_argv
    results[label] = {"actual_flag": actual, "pass": actual is want}


# All QA C calls are confined to the retained optional class, never startup/attach/status.
def c_calls(node):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "faulthandler"
    ]


results["no_C_calls_outside_optional_class"] = {
    "pass": len(c_calls(tree)) == len(c_calls(watch)) == 2
}
# Execute only the actual child-argv AST nodes; no process is started.
parent = ast.parse((here / "native_product_observer.py").read_text(encoding="utf-8"))
parent_main = next(
    n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == "main"
)
argv_nodes = [
    n
    for n in parent_main.body
    if (
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "child" for t in n.targets)
    )
    or (
        isinstance(n, ast.AugAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "child"
    )
]
passed = True
for flag in (False, True):
    namespace = {
        "sys": SimpleNamespace(executable="controlled-python"),
        "run": Path("controlled-owned-run"),
        "nonce": "a" * 32,
        "selected": {
            "columns": 120,
            "rows": 40,
            "shape": "single",
            "negative_controls": False,
            "diagnostic_stack": flag,
        },
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=argv_nodes, type_ignores=[])),
            "actual-parent",
            "exec",
        ),
        namespace,
    )
    expected = "--diagnostic-stack" if flag else "--no-diagnostic-stack"
    passed = passed and namespace["child"][-1] == expected
results["actual_parent_passes_both_explicit_flag_values"] = {"pass": passed}
assignment = next(
    n
    for n in ast.walk(seeder)
    if isinstance(n, ast.Assign)
    and any(
        isinstance(t, ast.Subscript)
        and isinstance(t.value, ast.Name)
        and t.value.id == "selection"
        and isinstance(t.slice, ast.Constant)
        and t.slice.value == "diagnostic_stack"
        for t in n.targets
    )
)
passed = True
for flag in (False, True):
    namespace = {"selection": {}, "args": SimpleNamespace(diagnostic_stack=flag)}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
            "actual-seeder",
            "exec",
        ),
        namespace,
    )
    passed = passed and namespace["selection"]["diagnostic_stack"] is flag
results["actual_selection_records_explicit_flag"] = {"pass": passed}
failures = [name for name, value in results.items() if not value["pass"]]
receipt = {
    "status": "GREEN" if not failures else "RED",
    "qualified": False,
    "product_samples": 0,
    "results": results,
    "failures": failures,
    "scope": "Actual private command/class/parser AST plus pure mode policy with controlled timers/tasks/files only; no real C timer, App, native API, product import, external/model traffic or crash causality.",
}
(output / "diagnostic-policy-control-results.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)
print(json.dumps(receipt, indent=2))
assert not failures, failures
assert not any(
    name.startswith(("tldw_chatbook", "tldw_profile_core", "textual", "ctypes"))
    for name in sys.modules
)
