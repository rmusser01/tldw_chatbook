"""Controlled clocks execute ONLY actual private preparation-loop source."""

from pathlib import Path
import hashlib
import json
import sys
import ast

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
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
from native_rehearsal_observation import check_scope  # noqa: E402 - Selected-source/profile/dependency guard runs first.

SOURCE = HERE / "native_product_observer.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
function = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef) and node.name == "run_round"
)
index = next(
    index
    for index, node in enumerate(function.body)
    if isinstance(node, ast.While) and "wait_started" in ast.unparse(node.test)
)
names = {"wait_started", "candidate", "last_source_ns"}
prefix = [
    node
    for node in function.body[:index]
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id in names for target in node.targets
    )
]
actual = ast.fix_missing_locations(
    ast.Module(body=prefix + [function.body[index]], type_ignores=[])
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def perf_counter_ns(self):
        return round(self.now * 1000000000)

    def sleep(self, seconds):
        self.now += seconds


def run_case(steps, after_scope=None):
    clock = Clock()
    events = []
    index = 0

    def poll():
        nonlocal index
        if index >= len(steps):
            raise RuntimeError("CONTROL_no_more_steps")
        now, stamp, ready = steps[index]
        index += 1
        clock.now = now
        events.append(
            {
                "stage": "real_stage_" + str(index),
                "timestamp_ns": stamp,
                "correlation": "controlled",
            }
        )
        return {
            "snapshot": {
                "current": {"correlation": "controlled"},
                "theme": "textual-dark",
                "size": [120, 40],
                "inspect_open": False,
                "controls": [
                    {
                        "id": "approval-fast-approve-12-0",
                        "disabled": False,
                        "display": True,
                        "fully_visible": True,
                    }
                ],
                "card_display": ready,
                "card_identity": ["controlled-round", 1],
                "information_ready": ready,
                "card_region": {"fully_visible": True},
            },
            "native": {"font": [8, 16]},
            "observer_health": {
                "loss": False,
                "write_failure": False,
                "observer_failure": False,
            },
        }

    def actual_scope(value):
        check_scope(value)
        if after_scope is not None:
            clock.now = after_scope

    namespace = {
        "event_start": 0,
        "expected_correlation": "controlled",
        "time": clock,
        "poll_events": poll,
        "event_rows": events,
        "result": {},
        "eligible_scope": actual_scope,
    }
    reason = None
    try:
        exec(compile(actual, str(SOURCE), "exec"), namespace)
    except RuntimeError as error:
        reason = str(error)
    return {
        "accepted": namespace.get("candidate") is not None,
        "reason": reason,
        "poll_calls": index,
        "elapsed": clock.now - 100,
    }


def steps(times, last_ready=True):
    return [
        (value, round(value * 1e9), last_ready and index == len(times) - 1)
        for index, value in enumerate(times)
    ]


cases = {
    "within_bounds": (steps([101]), True),
    "owner_phase_exact60": (steps([160]), True),
    "timely_actual_progress_to_total299": (steps([150, 200, 250, 300, 350, 399]), True),
    "late_normal_candidate_total301": (steps([150, 200, 250, 300, 350, 401]), False),
    "late_normal_candidate_prior_owner_phase61": (steps([161]), False),
    "late_poll_with_fresh_returned_stage_masks_prior_phase": (
        [(161, 161000000000, True)],
        False,
    ),
    "future_source_timestamp_cannot_extend_phase": ([(150, 210000000000, True)], False),
    "nonfinite_source_timestamp_cannot_mask_phase": (
        [(150, float("nan"), True)],
        False,
    ),
}
results = {
    name: dict(run_case(values), expected_accepted=expected)
    for name, (values, expected) in cases.items()
}
results["late_during_eligibility_total301"] = dict(
    run_case(steps([150, 200, 250, 300, 350, 399]), 401), expected_accepted=False
)
results["late_during_eligibility_current_phase72"] = dict(
    run_case(steps([150]), 222), expected_accepted=False
)
failures = [
    name
    for name, value in results.items()
    if value["accepted"] != value["expected_accepted"]
]
facts = {
    "scope": "Actual private parent loop extracted as source; synthetic clocks/event metadata only, no App, product import, native API, network or UI/input.",
    "qualified": False,
    "product_samples": 0,
    "status": "GREEN" if not failures else "RED",
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "results": results,
    "failures": failures,
}
(OUTPUT / "preparation-clock-control-results.json").write_text(
    json.dumps(facts, indent=2), encoding="utf-8"
)
print(json.dumps(facts, indent=2))
assert not failures, (
    "Actual preparation-loop deadline/clock edge reproduced: " + ", ".join(failures)
)
assert not any(
    name.startswith(("tldw_chatbook", "tldw_profile_core", "textual", "ctypes"))
    for name in sys.modules
)
