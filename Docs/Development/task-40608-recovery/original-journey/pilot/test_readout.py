"""Small arithmetic controls; no app, profile, network or Windows process."""

import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pilot_audit_test", HERE / "audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ReadoutTests(unittest.TestCase):
    def test_union_does_not_sum_nested_or_overlapping_waits(self):
        self.assertEqual(AUDIT.READOUT.union([(2, 4), (1, 8), (7, 10), (12, 13)]), 10)
        self.assertEqual(AUDIT.READOUT.union([]), 0)

    def test_phase_clips_waits_and_excludes_partial_header_cpu(self):
        def barrier(start, end, count):
            return {"start_ns": start, "end_ns": end, "scheduled": count,
                    "children_at_return": count}

        result = AUDIT.READOUT.phase({
            "barriers": [barrier(5, 15, 2), barrier(12, 18, 3), barrier(25, 35, 5)],
            "headers": [[5, 12, 4], [12, 18, 6], [27, 35, 8]],
        }, 10, 30)
        self.assertEqual(result["pilot_calls_started"], 2)
        self.assertEqual(result["pilot_calls_fully_inside"], 1)
        self.assertEqual(result["pilot_queued_callbacks"], 8)
        self.assertAlmostEqual(result["pilot_wait_overlap_union_seconds"], 13e-9)
        self.assertEqual(result["header_calls_started"], 2)
        self.assertEqual(result["header_calls_fully_inside"], 1)
        self.assertAlmostEqual(result["header_complete_thread_cpu_seconds"], 6e-9)

    def test_subtree_counts_stay_separate(self):
        groups = AUDIT.compact_groups([
            {"groups": {"parent": [12, 7], "child": [4, 4]}},
            {"groups": {"parent": [10, 5], "child": [3, 3]}},
        ])
        self.assertEqual(groups["parent"], {"samples": 2, "nodes": [10, 12], "hidden": [5, 7]})
        self.assertEqual(groups["child"]["hidden"], [3, 4])

    def test_published_subphases_reconcile_to_each_original_window(self):
        data = json.loads((HERE / "g72p-result.json").read_bytes())
        self.assertTrue(data["source_pins_match"])
        self.assertTrue(data["original_guards_complete"])
        self.assertTrue(data["diagnostic_complete"])
        self.assertFalse(data["qualifying_pair"])
        for turn in ("readme", "edit"):
            rows = [row for row in data["phases"] if row["turn"] == turn]
            whole = next(row for row in rows if row["phase"] == "send_to_reply")
            parts = [row for row in rows if row["phase"] != "send_to_reply"]
            for field in ("pilot_calls_started", "pilot_queued_callbacks", "header_calls_started"):
                self.assertEqual(whole[field], sum(row[field] for row in parts))
            self.assertAlmostEqual(whole["wall_seconds"], sum(row["wall_seconds"] for row in parts))
            self.assertAlmostEqual(whole["pilot_wait_overlap_union_seconds"],
                                   sum(row["pilot_wait_overlap_union_seconds"] for row in parts))


if __name__ == "__main__":
    unittest.main()
