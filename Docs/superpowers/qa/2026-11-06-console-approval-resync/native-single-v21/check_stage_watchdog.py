"""Finite pure controls for the private stage watchdog; no App or native launch."""

import ast
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--source-root", required=True, type=Path)
_parser.add_argument("--output-root", required=True, type=Path)
_parser.add_argument("--dependencies", type=Path)
_args = _parser.parse_args()
OUTPUT = _args.output_root.resolve()
if not OUTPUT.is_relative_to(_args.source_root.resolve() / ".superpowers"):
    raise RuntimeError(
        "Pure control outputs must stay inside selectedsource/.superpowers"
    )
OUTPUT.mkdir(parents=True, exist_ok=True)
if _args.dependencies is not None:
    sys.path.insert(0, str(_args.dependencies.resolve()))
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "native_product_child.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
main = next(
    node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "main"
)
command = next(
    node
    for node in main.body
    if isinstance(node, ast.AsyncFunctionDef) and node.name == "command"
)
watchdog_class = next(
    (
        node
        for node in TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "StageWatchdog"
    ),
    None,
)


class Timer:
    def __init__(self):
        self.calls = []
        self.arm_error = None
        self.cancel_error = None

    def dump_traceback_later(self, *args, **kwargs):
        self.calls.append(("arm", args, kwargs))
        if self.arm_error:
            raise self.arm_error

    def cancel_dump_traceback_later(self):
        self.calls.append(("cancel",))
        if self.cancel_error:
            raise self.cancel_error


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


results = {}


async def run():
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        timer = Timer()
        failures = []
        space = {
            "Path": Path,
            "faulthandler": timer,
            "time": __import__("time"),
            "write_json": write_json,
        }
        if watchdog_class is None:

            class Unavailable:
                def __init__(self, *args):
                    pass

            watchdog = Unavailable(directory, lambda: failures.append(True))
        else:
            exec(
                compile(
                    ast.Module(body=[watchdog_class], type_ignores=[]),
                    str(SOURCE),
                    "exec",
                ),
                space,
            )
            watchdog = space["StageWatchdog"](directory, lambda: failures.append(True))
        delegate = []
        identity = object()
        original_error = RuntimeError("controlled original error")
        future = asyncio.get_running_loop().create_future()

        class Fixture:
            worker = future
            current = {}

            def new_correlation(self):
                return "0" * 32

            async def stage(self, *args):
                delegate.append(args)
                if self.error:
                    raise self.error
                return identity

            def snapshot(self):
                return identity

            async def collect(self):
                delegate.append(("collect",))
                return identity

            async def close(self):
                delegate.append(("close",))

            error = None

        fixture = Fixture()
        namespace = dict(
            round_limit=1,
            diagnostic_stack_enabled=True,
            fixture=fixture,
            arguments=SimpleNamespace(
                negative_controls=False, shape="single", columns=120, rows=40
            ),
            stage_watchdog=watchdog,
            common_observer=None,
            card_events=None,
            sink=SimpleNamespace(health=lambda: {}),
            observer_native_state=lambda: identity,
            time=__import__("time"),
            state={"negative_results": None},
            app=SimpleNamespace(exit=lambda: delegate.append(("app.exit",))),
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
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
        actual = namespace["factory"]()
        returned = await actual({"command": "stage", "shape": "single"})
        assert (
            len(timer.calls) == 1
            and timer.calls[0][0] == "arm"
            and timer.calls[0][1] == (20,)
            and timer.calls[0][2]["repeat"] is False
        ), "Missing original-stage one-shot20s watchdog"
        assert not timer.calls[0][2][
            "file"
        ].closed, "Timer closed/cancelled merely because stage scheduled its task"
        assert (
            delegate == [("single", "0" * 32)] and returned["snapshot"] is identity
        ), "Original stage arguments/call count/return changed"
        results["actual_stage_arms_once_and_stays_armed_after_scheduling"] = True
        future.set_result(identity)
        await asyncio.sleep(0)
        assert [row[0] for row in timer.calls] == ["arm", "cancel"]
        assert timer.calls[0][2]["file"].closed
        health = json.loads((directory / "native-stage-watchdog.json").read_text())
        assert [row["event"] for row in health["events"]] == ["armed", "cancelled"]
        assert health["events"][1]["reason"] == "original_task_done" and all(
            type(row["timestamp_ns"]) is int  # noqa: E721 - Exact timestamp type rejects bool.
            for row in health["events"]
        )
        assert not health["observer_failure"]
        results["original_task_completion_cancels_with_bounded_timestamp_receipt"] = (
            True
        )
        for reason in ("collect", "exit", "app_finally"):
            with tempfile.TemporaryDirectory() as other:
                t = Timer()
                space["faulthandler"] = t
                w = space["StageWatchdog"](Path(other), lambda: failures.append(True))
                w.arm()
                w.cancel(reason)
                w.cancel(reason)
                w.arm()
                assert [row[0] for row in t.calls] == [
                    "arm",
                    "cancel",
                ], "More than one timer or repeated cancellation"
                assert t.calls[0][2]["file"].closed
        results["collect_exit_finally_cancel_idempotently_and_never_rearm"] = True
        for kind in ("collect", "exit"):
            delegate.clear()
            recorder = SimpleNamespace(
                cancel=lambda reason: delegate.append(("cancel", reason))
            )
            namespace["stage_watchdog"] = recorder
            await actual({"command": kind})
            expected = (
                [("cancel", "collect"), ("collect",)]
                if kind == "collect"
                else [("cancel", "exit"), ("close",), ("app.exit",)]
            )
            assert (
                delegate == expected
            ), "Cancellation changed original collect/exit sequence"
        results["actual_collect_exit_delegate_once_after_cancel"] = True
        delegate.clear()
        namespace["stage_watchdog"] = space["StageWatchdog"](
            directory / "unused", lambda: failures.append(True)
        )
        fixture.error = original_error
        actual = namespace["factory"]()
        caught = None
        try:
            await actual({"command": "stage", "shape": "single"})
        except BaseException as error:
            caught = error
        assert caught is original_error and delegate == [("single", "0" * 32)]
        assert failures, "Unwritable observer path must flag failure"
        results[
            "watchdog_setup_failure_preserves_original_stage_exception_and_once_call"
        ] = True
        for failure_point in ("arm", "cancel", "record"):
            with tempfile.TemporaryDirectory() as other:
                t = Timer()
                space["faulthandler"] = t
                flag = []
                if failure_point == "arm":
                    t.arm_error = OSError("controlled timer failure")
                w = space["StageWatchdog"](Path(other), lambda: flag.append(True))
                w.arm()
                if failure_point == "cancel":
                    t.cancel_error = OSError("controlled cancel failure")
                if failure_point == "record":
                    space["write_json"] = lambda *_: (_ for _ in ()).throw(
                        OSError("controlled metadata failure")
                    )
                w.cancel("app_finally")
                assert flag, "Observer failure not marked"
                if failure_point == "cancel":
                    assert not t.calls[0][2][
                        "file"
                    ].closed, "Unsafe close while timer cancellation unproven"
                    t.cancel_error = None
                    w.cancel("app_finally")
                space["write_json"] = write_json
        results["arm_cancel_metadata_failures_are_observation_only"] = True
        # app.run finally must cancel even when the original app body raises.
        app_try = next(
            node
            for node in main.body
            if isinstance(node, ast.Try)
            and any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "run"
                for item in node.body
                for sub in ast.walk(item)
            )
        )
        first = copy.deepcopy(app_try.finalbody[0])
        log = []
        error = ValueError("controlled app error")
        namespace = {
            "stage_watchdog": SimpleNamespace(cancel=lambda reason: log.append(reason)),
            "error": error,
        }
        bounded = ast.Try(
            body=[ast.Raise(exc=ast.Name(id="error", ctx=ast.Load()))],
            handlers=[],
            orelse=[],
            finalbody=[first],
        )
        caught = None
        try:
            exec(
                compile(
                    ast.fix_missing_locations(
                        ast.Module(body=[bounded], type_ignores=[])
                    ),
                    str(SOURCE),
                    "exec",
                ),
                namespace,
            )
        except BaseException as actual_error:
            caught = actual_error
        assert caught is error and log == ["app_finally"]
        results["actual_app_finally_cancels_without_masking_original_error"] = True


error = None
try:
    asyncio.run(run())
except BaseException as caught:
    error = {"type": type(caught).__name__, "reason": str(caught)}
facts = {
    "qualified": False,
    "product_samples": 0,
    "status": "GREEN" if error is None else "RED",
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "results": results,
    "error": error,
    "scope": "Actual private child class/command/finally AST, controlled timer/files/asyncio task only; no App, product imports, Win32, GUI, product tool execution, external/service network or real timer; stock asyncio may use its internal self-pipe.",
}
(OUTPUT / "stage-watchdog-control-results.json").write_text(
    json.dumps(facts, indent=2), encoding="utf-8"
)
print(json.dumps(facts, indent=2))
assert error is None, error
assert not any(
    name.startswith(("tldw_chatbook", "tldw_profile_core", "textual", "ctypes"))
    for name in sys.modules
)
