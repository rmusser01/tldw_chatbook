"""Small arithmetic/association checks; no product or native worker is loaded."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "diagnostic_analysis", Path(__file__).with_name("analyze_diagnostics.py")
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class DiagnosticAnalysisTests(unittest.TestCase):
    def approval_event(self, stage, wall, **extra):
        return {
            "event": "milestone",
            "stage": stage,
            "wall": wall,
            "journey": "edit",
            "fixture_call_ordinal": 3,
            **extra,
        }

    def test_strong_paint_is_distinct_and_resolution_stays_after_click(self):
        rows = [
            self.approval_event("policy_gate_return", 1, pending=True),
            self.approval_event("approval_request_start", 2),
            self.approval_event("painted_approval", 3),
            self.approval_event(
                "original_label_probe", 4, qualified_rendered_fast_approve=False
            ),
            self.approval_event("approval_click", 5),
            self.approval_event("approval_request_return", 8),
            self.approval_event("local_review_return", 9),
        ]
        [result] = analysis.approval(rows)
        self.assertTrue(result["original_painted_approval_observed"])
        self.assertFalse(result["qualified_frame_observed"])
        edges = {
            (row["from_boundary"], row["to_boundary"]): row
            for row in result["boundary_intervals"]
        }
        self.assertEqual(edges["clicked", "request_settled"]["seconds"], 3)
        self.assertIsNone(edges["qualified_frame", "pilot_observed"]["seconds"])

    def test_scope_change_does_not_form_a_causal_interval(self):
        rows = [
            self.approval_event(
                "controls_ready_return",
                1,
                same_generation=True,
                information_ready=True,
                card_generation=4,
            ),
            self.approval_event(
                "original_label_probe",
                2,
                qualified_rendered_fast_approve=True,
                card_generation=5,
            ),
        ]
        [result] = analysis.approval(rows)
        [edge] = result["adjacent_observed_intervals"]
        self.assertIsNone(edge["seconds"])
        self.assertEqual(edge["reason"], "scope_changed")

    def test_local_ui_delivery_and_missing_paint_reasons_are_retained(self):
        rows = [
            self.approval_event("pending_marshal_start", 1),
            self.approval_event("ui_approval_start", 2),
            self.approval_event(
                "approval_observed",
                4,
                active_console=True,
                visible_map_known=False,
                prior_cells_seen=False,
            ),
        ]
        [result] = analysis.approval(rows)
        edge = next(
            row
            for row in result["boundary_intervals"]
            if row["from_boundary"] == "marshal"
        )
        self.assertEqual(edge["to"], "ui_approval_start")
        self.assertEqual(edge["seconds"], 1)
        self.assertFalse(result["events"][-1]["visible_map_known"])
        self.assertFalse(result["events"][-1]["prior_cells_seen"])

    def test_nested_children_use_union_and_storage_keeps_phase(self):
        context = {"fixture_call": "uat-dev-3", "phase": "tool_execution_started"}
        starts = [
            {
                "event": "start",
                "call": n,
                "wall": start,
                "parent_call": parent,
                "producer": producer,
                "context": context,
                "thread": 1,
            }
            for n, start, parent, producer in (
                (1, 10, None, "workspace_execute"),
                (2, 11, 1, "request_build"),
                (3, 12, 1, "other"),
            )
        ]
        ends = [
            {"event": "return", "call": n, "wall": finish}
            for n, finish in ((1, 20), (2, 15), (3, 17))
        ]
        rows = [
            *starts,
            *ends,
            {
                "event": "milestone",
                "phase": "tool_execution_started",
                "fixture_call": "uat-dev-3",
                "wall": 2,
            },
            {
                "event": "storage_start",
                "admission": 1,
                "call": 2,
                "wall": 12,
                "thread": 1,
                "context": context,
                "stack": [{"owner": "module", "function": "read", "line": 10}],
            },
            {
                "event": "storage_return",
                "admission": 1,
                "elapsed_wall": 0.5,
                "wall": 12.5,
            },
            {
                "event": "storage_start",
                "admission": 2,
                "wall": 14,
                "thread": 1,
                "context": context,
                "stack": [],
            },
            {"event": "storage_return", "admission": 2, "elapsed_wall": 2, "wall": 16},
            {
                "event": "storage_start",
                "admission": 3,
                "call": 2,
                "wall": 13,
                "thread": 1,
                "context": {},
                "stack": [],
            },
            {
                "event": "storage_return",
                "admission": 3,
                "elapsed_wall": 0.25,
                "wall": 13.25,
            },
        ]
        result = analysis.foreground(rows)
        outer = result["calls"][0]
        self.assertEqual(outer["observed_direct_child_union_seconds"], 6)
        self.assertEqual(outer["outside_observed_direct_children_seconds"], 4)
        self.assertEqual(
            result["call_boundaries"][0]["observed_before_single_executor_seconds"], 8
        )
        self.assertEqual(
            result["admissions_by_exact_stack_and_phase"][0][
                "inclusive_thread_sum_seconds"
            ],
            0.5,
        )
        slices = {
            row["boundary"]: row
            for row in result["executor_thread_admissions_by_boundary"]
        }
        crossed = slices["request_build"]["stacks"][1]
        self.assertEqual(crossed["clipped_sum_seconds"], 1.25)
        self.assertEqual(crossed["admissions_started_in_boundary"], 2)
        self.assertEqual(
            result["admissions_by_exact_stack_and_phase"][-1]["correlation"],
            "retained_invocation_entry_context",
        )
        after = slices["executor_after_build"]["stacks"][0]
        self.assertEqual(after["clipped_sum_seconds"], 1)
        self.assertEqual(after["admissions_started_in_boundary"], 0)

    def test_streams_never_subtract_different_process_clocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run-metadata.json").write_text(
                json.dumps({"finishedUtc": "recorded", "exitCode": 0})
            )
            (root / "timings.json").write_text(
                json.dumps({"status": "passed", "original_duration_seconds": 17})
            )
            (root / "lab-observer").mkdir()
            for role, stamp in (("ui", 1000), ("backend", 1)):
                rows = [
                    {"event": "installed", "process_role": role},
                    self.approval_event("approval_request_start", stamp),
                    {"event": "summary", "errors": 0},
                ]
                (root / "lab-observer" / f"{role}.jsonl").write_text(
                    "\n".join("APPROVAL_MILESTONE " + json.dumps(row) for row in rows)
                )
            result = analysis.analyze(root)
            self.assertEqual(len(result["streams"]), 2)
            for stream in result["streams"]:
                self.assertFalse(stream["analysis"][0]["adjacent_observed_intervals"])
            self.assertFalse(result["qualifying"])
            self.assertEqual(result["same_run_original_total_seconds"], 17)
            (root / "run-metadata.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "not recorded completion"):
                analysis.analyze(root)


if __name__ == "__main__":
    unittest.main()
