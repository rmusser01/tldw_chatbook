"""Pure actual-task snapshot controls; no product import, App, socket or GUI."""

import argparse
import ast
import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", required=True, type=Path)
parser.add_argument("--output-root", required=True, type=Path)
args = parser.parse_args()
source = args.source_root.resolve()
output = args.output_root.resolve()
if not output.is_relative_to(source / ".superpowers"):
    raise RuntimeError("Pure outputs must stay beneath selectedsource/.superpowers")
output.mkdir(parents=True, exist_ok=True)
workload = Path(__file__).resolve().with_name("approval_fixture_workload_v7.py")
tree = ast.parse(workload.read_text(encoding="utf-8"))
owner = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "ApprovalFixtureDriver"
)
snapshot = next(
    n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "snapshot"
)
helper = [
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_completed_task_diagnostic"
]
namespace = {"Path": Path, "sys": SimpleNamespace(modules={})}
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(body=helper + [snapshot], type_ignores=[])
        ),
        "actual-snapshot",
        "exec",
    ),
    namespace,
)


class RecoveryRequired(RuntimeError):
    def __str__(self):
        raise AssertionError("Exception text must never be inspected")

    def __repr__(self):
        raise AssertionError("Exception repr must never be inspected")


class OtherRecoveryRequired(RecoveryRequired):
    pass


class StringSubclass(str):
    pass


bootstrap = ModuleType("controlled_resident_bootstrap")
bootstrap.RecoveryRequired = RecoveryRequired
resident_guard = ModuleType("controlled_resident_guard")
resident_guard.REPO = source
resident = namespace["sys"].modules
resident["tldw_chatbook.Backup_Recovery.bootstrap"] = bootstrap
resident["qa_process_guard"] = resident_guard
results = {}


async def completed(error=None, value=None, depth=0, filename=None):
    code = "async def raise_actual(error, depth):\n    if depth:\n        return await raise_actual(error, depth-1)\n    if error is not None:\n        raise error\n    return value\n"
    scope = {"value": value}
    exec(
        compile(
            code,
            filename or str(source / "tldw_chatbook" / "controlled_trace.py"),
            "exec",
        ),
        scope,
    )
    task = asyncio.create_task(scope["raise_actual"](error, depth))
    await asyncio.wait({task})
    return task


def read(task, expected_failure=False):
    failures = []
    self = SimpleNamespace(
        _lock=threading.Lock(),
        current=None,
        worker=task,
        controller=None,
        app=SimpleNamespace(
            _driver=None,
            size=SimpleNamespace(width=120, height=40),
            theme="textual-dark",
        ),
        _visible_send_lifecycle=lambda: None,
        _observation_failed=lambda *items: failures.append(items),
    )
    state = namespace["snapshot"](self)["submission_task"]
    assert bool(failures) is expected_failure
    if failures:
        assert failures[0][0] == "completed_task_diagnostic"
    return state


def check(name, value):
    results[name] = bool(value)


async def main():
    error = RecoveryRequired(
        "raw_source_selection_changed", "MUST_NOT_EXPORT_SECOND_ARG"
    )
    task = await completed(error)
    state = read(task)
    diagnostic = state.get("completed_diagnostic", {})
    check(
        "actual_failed_task_canonical_safe_identifier",
        diagnostic.get("reason_code") == "raw_source_selection_changed",
    )
    check(
        "source_relative_code_frames",
        diagnostic.get("traceback")
        == [
            {
                "file": "tldw_chatbook/controlled_trace.py",
                "function": "raise_actual",
                "line": 5,
                "source": "selected_source",
            }
        ],
    )
    check(
        "original_failed_task_unchanged",
        task.exception() is error
        and task.done()
        and not task.cancelled()
        and state["result"] is None,
    )
    check("no_exception_text_or_other_args", "MUST_NOT_EXPORT" not in json.dumps(state))
    for label, reason in [
        ("uppercase", "Bad"),
        ("digit_first", "1bad"),
        ("slash", "bad/path"),
        ("hyphen", "bad-code"),
        ("unicode", "bad_é"),
        ("empty", ""),
        ("too_long", "a" * 97),
        ("subclass", StringSubclass("valid")),
        ("not_string", 17),
    ]:
        d = read(await completed(RecoveryRequired(reason))).get(
            "completed_diagnostic", {}
        )
        check(
            "unsafe_reason_" + label,
            d.get("reason_code") is None
            and d.get("reason_status") == "unsafe_or_missing",
        )
    d = read(await completed(RecoveryRequired())).get("completed_diagnostic", {})
    check(
        "missing_reason",
        d.get("reason_code") is None and d.get("reason_status") == "unsafe_or_missing",
    )
    for label, error in [
        ("subclass", OtherRecoveryRequired("valid")),
        ("foreign_name", type("RecoveryRequired", (RuntimeError,), {})("valid")),
    ]:
        d = read(await completed(error)).get("completed_diagnostic", {})
        check(
            "noncanonical_" + label,
            d.get("reason_code") is None
            and d.get("reason_status") == "noncanonical_exception",
        )
    for label, reason in [
        ("one", "a"),
        ("max", "a" * 96),
        ("digits_underscore", "a_09"),
    ]:
        check(
            "safe_" + label,
            read(await completed(RecoveryRequired(reason)))
            .get("completed_diagnostic", {})
            .get("reason_code")
            == reason,
        )
    d = read(
        await completed(
            RecoveryRequired("valid"),
            filename=str(source.parent / "PRIVATE_OUTSIDE.py"),
        )
    ).get("completed_diagnostic", {})
    check(
        "outside_source_honest_and_hidden",
        d.get("traceback")
        == [
            {
                "file": None,
                "function": None,
                "line": 5,
                "source": "outside_selected_source",
            }
        ]
        and "PRIVATE_OUTSIDE" not in json.dumps(d),
    )
    d = read(await completed(RecoveryRequired("valid"), filename="<opaque>")).get(
        "completed_diagnostic", {}
    )
    check(
        "relative_filename_unknown",
        d.get("traceback")
        == [{"file": None, "function": None, "line": 5, "source": "unknown_filename"}],
    )
    d = read(await completed(RecoveryRequired("valid"), depth=16)).get(
        "completed_diagnostic", {}
    )
    check(
        "bounded_innermost_and_actual_raise_site",
        len(d.get("traceback", [])) == 12
        and d.get("traceback_truncated") is True
        and d.get("traceback_order") == "innermost_observed_outer_to_inner"
        and d["traceback"][-1]["line"] == 5
        and d.get("raise_site_included") is True
        and d.get("traceback_walk_limited") is False
        and d.get("traceback_frames_walked") == 17,
    )
    d = read(await completed(RecoveryRequired("valid"), depth=11)).get(
        "completed_diagnostic", {}
    )
    check(
        "exact12_not_truncated",
        len(d.get("traceback", [])) == 12 and d.get("traceback_truncated") is False,
    )
    d = read(await completed(RecoveryRequired("valid"), depth=70)).get(
        "completed_diagnostic", {}
    )
    check(
        "walk64_limit_honest_missing_raise_site",
        len(d.get("traceback", [])) == 12
        and d.get("traceback_truncated") is True
        and d.get("traceback_walk_limited") is True
        and d.get("traceback_frames_walked") == 64
        and d.get("raise_site_included") is False
        and all(frame["line"] == 3 for frame in d["traceback"]),
    )
    resident.pop("tldw_chatbook.Backup_Recovery.bootstrap")
    d = read(await completed(RecoveryRequired("valid"))).get("completed_diagnostic", {})
    check(
        "bootstrap_not_imported_when_missing",
        d.get("reason_code") is None
        and d.get("reason_status") == "canonical_class_unavailable"
        and "tldw_chatbook.Backup_Recovery.bootstrap" not in resident,
    )
    resident["tldw_chatbook.Backup_Recovery.bootstrap"] = bootstrap
    resident.pop("qa_process_guard")
    d = read(await completed(RecoveryRequired("valid"))).get("completed_diagnostic", {})
    check(
        "source_root_unavailable_is_honest",
        d.get("traceback")
        == [
            {
                "file": None,
                "function": None,
                "line": 5,
                "source": "source_root_unavailable",
            }
        ],
    )
    resident["qa_process_guard"] = resident_guard
    value = SimpleNamespace(
        accepted=True,
        should_clear_draft=True,
        session_closed=False,
        provider_started=False,
        terminal_status=None,
        blocker=None,
    )
    task = await completed(value=value)
    state = read(task)
    check(
        "success_unchanged_no_failure_diagnostic",
        task.result() is value
        and state["result"]["accepted"] is True
        and "completed_diagnostic" not in state,
    )
    task = asyncio.create_task(asyncio.sleep(60))
    task.cancel()
    await asyncio.wait({task})
    state = read(task)
    check(
        "cancelled_unchanged_no_exception_access",
        state["done"] is True
        and state["cancelled"] is True
        and state["exception_type"] is None
        and "completed_diagnostic" not in state,
    )
    actual_diagnostic = namespace.get("_completed_task_diagnostic")
    if actual_diagnostic is not None:
        failure = RuntimeError("observer-only")

        def refuse_observation(exception):
            raise failure

        namespace["_completed_task_diagnostic"] = refuse_observation
        error = RecoveryRequired("valid")
        task = await completed(error)
        state = read(task, expected_failure=True)
        check(
            "diagnostic_failure_does_not_replace_original_task_error",
            task.exception() is error
            and state["completed_diagnostic"] == {"status": "unavailable"}
            and state["exception_type"].endswith(".RecoveryRequired"),
        )
        namespace["_completed_task_diagnostic"] = actual_diagnostic
    check(
        "no_actual_product_imports",
        not any(name.startswith("tldw_chatbook") for name in sys.modules),
    )


asyncio.run(main())
receipt = {
    "scope": "Pure extracted actual completed-task snapshot with real asyncio Tasks; controlled resident metadata only; no product imports/App/network/GUI/timing qualification",
    "workload_SHA256": hashlib.sha256(workload.read_bytes()).hexdigest(),
    "results": results,
    "passed": sum(results.values()),
    "failed": sum(not v for v in results.values()),
    "qualified": False,
    "product_samples": 0,
}
(output / "completed-task-diagnostic-controls.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt))
raise SystemExit(0 if all(results.values()) else 1)
