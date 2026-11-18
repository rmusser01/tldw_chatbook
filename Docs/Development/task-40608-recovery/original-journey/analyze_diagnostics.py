"""Read finished foreground/approval diagnostics; never launch a workload.

Intervals are observations on one process-local clock. They are neither
qualification timings nor proof of permission, physical retirement or paint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


PREFIXES = {"TOOL_FOREGROUND": "foreground", "APPROVAL_MILESTONE": "approval"}
CALLS = {f"uat-call-{n}": ("readme", n) for n in range(5)} | {
    f"uat-dev-{n}": ("edit", n) for n in range(5)
}
LOCATION = ("owner", "function", "line", "producer", "thread")
STATE = (
    "visit",
    "card_generation",
    "same_generation",
    "information_ready",
    "approval_phase",
    "already_submitted",
    "original_trace_started",
    "pending",
    "returned_true",
    "reused",
    "qualified_rendered_fast_approve",
    "qualified_delivered_button_region_cells",
    "original_fixed_label_match",
    "original_match_cells_emitted",
    "eligible_visible_button",
    "active_console",
    "card_belongs_to_console",
    "button_belongs_to_card",
    "fully_within_visible_clip",
    "button_present",
    "button_mounted",
    "button_disabled_known",
    "button_disabled",
    "button_closing",
    "geometry_known",
    "visible_map_known",
    "button_in_visible_map",
    "outside_viewport",
    "fully_clipped",
    "front_layer_overlap",
    "update_present",
    "update_cells_known",
    "supported_compositor_update",
    "update_has_button_region_cells",
    "prior_frame_seen",
    "prior_eligible_seen",
    "prior_cells_seen",
    "prior_label_match_seen",
    "prior_qualified_rendered_seen",
    "prior_qualified_region_cells_seen",
)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def scope_relation(left, right):
    known = False
    missing = False
    for name in ("visit", "card_generation"):
        a, b = left.get(name), right.get(name)
        if a is None or b is None:
            missing = True
        elif a != b:
            return "different"
        else:
            known = True
    return "same" if known and not missing else "partly_known" if known else "unknown"


def interval(left, right):
    scope = scope_relation(left, right)
    delta = right["wall"] - left["wall"]
    return {
        "from": left["stage"],
        "to": right["stage"],
        "scope": scope,
        "seconds": delta if delta >= 0 and scope != "different" else None,
        "reason": "scope_changed"
        if scope == "different"
        else "order_reversed"
        if delta < 0
        else None,
        "from_wall": left["wall"],
        "to_wall": right["wall"],
    }


def union_seconds(intervals):
    end = None
    total = 0.0
    for start, finish in sorted(intervals):
        if finish < start:
            continue
        if end is None or start >= end:
            total += finish - start
            end = finish
        elif finish > end:
            total += finish - end
            end = finish
    return total


def approval(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("event") != "milestone" or not number(row.get("wall")):
            continue
        key = row.get("journey"), row.get("fixture_call_ordinal")
        if (
            key[0] not in {"readme", "edit"}
            or type(key[1]) is not int
            or not 0 <= key[1] < 5
        ):
            continue
        stage = row.get("stage")
        if type(stage) is not str:
            continue
        groups[key].append(
            {
                name: row[name]
                for name in ("stage", "wall", *LOCATION, *STATE)
                if name in row
            }
        )
    result = []
    for (journey, call), events in sorted(groups.items()):
        events.sort(key=lambda row: row["wall"])

        def first(stage, _events=events, **required):
            return next(
                (
                    row
                    for row in _events
                    if row["stage"] == stage
                    and all(row.get(key) is value for key, value in required.items())
                ),
                None,
            )

        points = {
            "policy_pending": first("policy_gate_return", pending=True),
            "request": first("approval_request_start"),
            "registered": first("round_register_return", returned_true=True),
            "published": first("pending_publish_return", returned_true=True),
            "marshal": first("pending_marshal_start"),
            "ui_delivery": min(
                (
                    row
                    for row in events
                    if row["stage"] in {"ui_projection_start", "ui_approval_start"}
                ),
                key=lambda row: row["wall"],
                default=None,
            ),
            "card": first("card_batch_return")
            or first("card_reuse_return", reused=True),
            "controls_ready": first(
                "controls_ready_return", same_generation=True, information_ready=True
            ),
            "qualified_frame": first(
                "original_label_probe", qualified_rendered_fast_approve=True
            ),
            "pilot_observed": first("approval_observed"),
            "clicked": first("approval_click"),
            "request_settled": first("approval_request_return"),
            "review_settled": first("local_review_return"),
        }
        edges = []
        names = list(points)
        for left_name, right_name in zip(names, names[1:], strict=False):
            left, right = points[left_name], points[right_name]
            value = (
                interval(left, right)
                if left and right
                else {
                    "seconds": None,
                    "reason": "missing_observation",
                    "scope": "unknown",
                }
            )
            edges.append(
                {"from_boundary": left_name, "to_boundary": right_name, **value}
            )
        result.append(
            {
                "journey": journey,
                "call_ordinal": call,
                "events": events,
                "adjacent_observed_intervals": [
                    interval(a, b) for a, b in zip(events, events[1:], strict=False)
                ],
                "boundary_intervals": edges,
                "original_painted_approval_observed": first("painted_approval")
                is not None,
                "qualified_frame_observed": points["qualified_frame"] is not None,
                "prior_qualified_frame_at_click": (points["clicked"] or {}).get(
                    "prior_qualified_rendered_seen"
                ),
            }
        )
    return result


def foreground(rows):
    starts = {
        row["call"]: row
        for row in rows
        if row.get("event") == "start" and type(row.get("call")) is int
    }
    ends = {
        row["call"]: row
        for row in rows
        if row.get("event") in {"return", "unwind"} and type(row.get("call")) is int
    }
    storage_start = {
        row["admission"]: row for row in rows if row.get("event") == "storage_start"
    }
    storage_end = {
        row["admission"]: row
        for row in rows
        if row.get("event") in {"storage_return", "storage_unwind"}
    }
    calls = []
    for invocation, row in starts.items():
        end = ends.get(invocation, {})
        if not number(row.get("wall")):
            continue
        context = row.get("context", {})
        correlation = "entry_context"
        key = CALLS.get(context.get("fixture_call"))
        if key is None:
            context = end.get("current_context", {})
            key = CALLS.get(context.get("fixture_call"))
            correlation = "observed_exit_context"
        if key is None:
            continue
        finish = end.get("wall")
        elapsed = (
            finish - row["wall"] if number(finish) and finish >= row["wall"] else None
        )
        children = [
            (child["wall"], ends[n]["wall"])
            for n, child in starts.items()
            if child.get("parent_call") == invocation
            and n in ends
            and number(child.get("wall"))
            and number(ends[n].get("wall"))
            and elapsed is not None
            and row["wall"] <= child["wall"] <= ends[n]["wall"] <= finish
        ]
        covered = union_seconds(children)
        calls.append(
            {
                "journey": key[0],
                "call_ordinal": key[1],
                "invocation": invocation,
                "parent_invocation": row.get("parent_call"),
                "correlation": correlation,
                "phase": context.get("phase"),
                "operation": context.get("operation"),
                **{name: row.get(name) for name in LOCATION},
                "entry_wall": row["wall"],
                "return_wall": finish,
                "outcome": end.get("event"),
                "elapsed_seconds": elapsed,
                "active_thread_cpu_seconds": end.get("active_thread_cpu"),
                "observed_direct_child_union_seconds": covered,
                "outside_observed_direct_children_seconds": None
                if elapsed is None
                else elapsed - covered,
            }
        )
    calls_by_id = {row["invocation"]: row for row in calls}

    def storage_scope(row, end):
        context = row.get("context", {})
        key = CALLS.get(context.get("fixture_call"))
        if key is not None:
            return key, context.get("phase"), "entry_context"
        # The worker's original handover may establish the fixture call after
        # admission. Associate only its retained invocation, never a nearby thread.
        parent = calls_by_id.get(row.get("call"))
        if (
            parent is not None
            and parent["thread"] == row.get("thread")
            and number(row.get("wall"))
            and number(end.get("wall"))
            and number(parent["return_wall"])
            and parent["entry_wall"]
            <= row["wall"]
            <= end["wall"]
            <= parent["return_wall"]
        ):
            return (
                (parent["journey"], parent["call_ordinal"]),
                parent["phase"],
                "retained_invocation_" + parent["correlation"],
            )
        return None, None, None

    stacks = defaultdict(list)
    unfinished_storage = 0
    for invocation, row in storage_start.items():
        end = storage_end.get(invocation, {})
        key, phase, correlation = storage_scope(row, end)
        if key is None:
            continue
        if not number(end.get("elapsed_wall")):
            unfinished_storage += 1
            continue
        stack = tuple(
            (item.get("owner"), item.get("function"), item.get("line"))
            for item in row.get("stack", ())
        )
        group = (*key, row.get("thread"), phase, correlation, stack)
        stacks[group].append(end["elapsed_wall"])
    stack_rows = [
        {
            "journey": key[0],
            "call_ordinal": key[1],
            "thread": key[2],
            "phase": key[3],
            "correlation": key[4],
            "stack": [
                {"owner": owner, "function": function, "line": line}
                for owner, function, line in key[5]
            ],
            "count": len(values),
            "inclusive_thread_sum_seconds": sum(values),
            "max_seconds": max(values),
        }
        for key, values in stacks.items()
    ]
    boundaries = []
    storage_slices = []
    for key in sorted({(row["journey"], row["call_ordinal"]) for row in calls}):
        selected = [
            row for row in calls if (row["journey"], row["call_ordinal"]) == key
        ]
        executors = [row for row in selected if row["producer"] == "workspace_execute"]
        builders = [row for row in selected if row["producer"] == "request_build"]
        start = next(
            (
                row
                for row in rows
                if row.get("event") == "milestone"
                and row.get("phase") == "tool_execution_started"
                and CALLS.get(row.get("fixture_call")) == key
                and number(row.get("wall"))
            ),
            None,
        )
        before = None
        if start is not None and len(executors) == 1:
            delta = executors[0]["entry_wall"] - start["wall"]
            before = delta if delta >= 0 else None
        boundaries.append(
            {
                "journey": key[0],
                "call_ordinal": key[1],
                "execution_start_producer": None
                if start is None
                else start.get("producer"),
                "execution_start_wall": None if start is None else start["wall"],
                "executor_invocations": len(executors),
                "request_builder_invocations": len(builders),
                "observed_before_single_executor_seconds": before,
                "executor_elapsed_seconds": [
                    row["elapsed_seconds"] for row in executors
                ],
                "request_builder_elapsed_seconds": [
                    row["elapsed_seconds"] for row in builders
                ],
            }
        )
        if start is None or len(executors) != 1 or len(builders) != 1:
            continue
        executor, builder = executors[0], builders[0]
        workers = [
            row
            for row in selected
            if row["producer"] == "tool_worker" and row["thread"] == executor["thread"]
        ]
        points = [
            start["wall"],
            executor["entry_wall"],
            builder["entry_wall"],
            builder["return_wall"],
            executor["return_wall"],
        ]
        labels = [
            "before_executor",
            "executor_before_build",
            "request_build",
            "executor_after_build",
        ]
        if len(workers) == 1:
            points.append(workers[0]["return_wall"])
            labels.append("after_executor_to_worker_return")
        if not all(number(value) for value in points) or points != sorted(points):
            continue
        for label, left, right in zip(labels, points, points[1:], strict=False):
            slices = defaultdict(list)
            for admission, row in storage_start.items():
                end = storage_end.get(admission, {})
                if (
                    storage_scope(row, end)[0] != key
                    or row.get("thread") != executor["thread"]
                    or not number(row.get("wall"))
                    or not number(end.get("wall"))
                ):
                    continue
                begin, finish = max(left, row["wall"]), min(right, end["wall"])
                if begin >= finish:
                    continue
                stack = tuple(
                    (item.get("owner"), item.get("function"), item.get("line"))
                    for item in row.get("stack", ())
                )
                slices[stack].append((begin, finish, left <= row["wall"] < right))
            storage_slices.append(
                {
                    "journey": key[0],
                    "call_ordinal": key[1],
                    "thread": executor["thread"],
                    "boundary": label,
                    "entry_wall": left,
                    "return_wall": right,
                    "elapsed_seconds": right - left,
                    "stacks": [
                        {
                            "stack": [
                                {"owner": owner, "function": function, "line": line}
                                for owner, function, line in stack
                            ],
                            "admissions_touching_boundary": len(values),
                            "admissions_started_in_boundary": sum(
                                value[2] for value in values
                            ),
                            "clipped_sum_seconds": sum(b - a for a, b, _ in values),
                            "clipped_union_seconds": union_seconds(
                                [(a, b) for a, b, _ in values]
                            ),
                        }
                        for stack, values in slices.items()
                    ],
                }
            )
    return {
        "calls": calls,
        "call_boundaries": boundaries,
        "admissions_by_exact_stack_and_phase": stack_rows,
        "unmatched_correlated_admissions": unfinished_storage,
        "executor_thread_admissions_by_boundary": storage_slices,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def analyze(root):
    metadata = read_json(root / "run-metadata.json")
    if not metadata.get("finishedUtc") or type(metadata.get("exitCode")) is not int:
        raise ValueError("The native runner has not recorded completion")
    timings = (
        read_json(root / "timings.json") if (root / "timings.json").is_file() else {}
    )
    result = {
        "qualifying": False,
        "native_exit_code": metadata["exitCode"],
        "original_status": timings.get("status"),
        "original_source_verified": timings.get("source_verified"),
        "same_run_original_total_seconds": timings.get("original_duration_seconds"),
        "journeys": {
            name: {
                key: row.get(key)
                for key in (
                    "completed_s",
                    "original_adjusted_s",
                    "common_adjusted_reply_seen_s",
                    "approval_wait_s",
                )
            }
            for name, row in timings.get("turns", {}).items()
        },
        "streams": [],
        "limitations": [
            "Diagnostic overhead is included; this is not a qualifying pair.",
            "No interval subtracts timestamps from different processes or streams.",
            "Unknown early scope remains call-correlated observation, not presentation authority.",
            "Nested/thread storage totals are inclusive; do not add them to critical-path wall time.",
            "A Python return never proves permission, success, or physical retirement.",
        ],
    }
    for path in sorted((root / "lab-observer").glob("*.jsonl")):
        grouped = defaultdict(list)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            prefix, separator, body = line.partition(" ")
            if separator and prefix in PREFIXES:
                grouped[prefix].append(json.loads(body))
        for prefix, rows in grouped.items():
            installed = [row for row in rows if row.get("event") == "installed"]
            summaries = [row for row in rows if row.get("event") == "summary"]
            summary = summaries[-1] if summaries else {}
            complete = len(installed) == len(summaries) == 1 and not any(
                summary.get(name)
                for name in (
                    "errors",
                    "source_refused",
                    "unfinished",
                    "unfinished_admissions",
                    "approval_unfinished",
                    "truncated",
                )
            )
            if len(installed) != 1:
                raise ValueError("A diagnostic stream must have one clock origin")
            result["streams"].append(
                {
                    "file": path.name,
                    "kind": PREFIXES[prefix],
                    "role": installed[0].get("process_role"),
                    "complete_observation": complete,
                    "summary": {
                        name: summary.get(name)
                        for name in (
                            "errors",
                            "source_refused",
                            "unfinished",
                            "unfinished_admissions",
                            "approval_unfinished",
                            "truncated",
                        )
                    },
                    "analysis": approval(rows)
                    if prefix == "APPROVAL_MILESTONE"
                    else foreground(rows),
                }
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_directory.resolve(strict=True))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
