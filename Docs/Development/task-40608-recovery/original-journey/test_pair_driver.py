"""Pure report arithmetic/ledger checks; no app, pytest, or native worker launch."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("pair_driver", Path(__file__).with_name("pair_driver.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def ledger(baseline, candidate):
    rows = []
    for pair, side in driver.ORDER:
        value = (baseline if side == "baseline" else candidate)[pair - 1]
        start = {"pair": pair, "side": side, "status": "started", "clock_ns": len(rows) + 1}
        end = {**start, "status": "finished", "clock_ns": len(rows) + 2,
               "eligible": True, "code": 0, "observation": {"times": {
                   journey: {"application_s": value, "common_reply_seen_application_s": value}
                   for journey in driver.JOURNEYS}}}
        rows.extend((start, end))
    return rows


class PairArithmeticTests(unittest.TestCase):
    def evaluate(self, rows, tail=""):
        with tempfile.TemporaryDirectory(prefix="task40608-pair-arithmetic-") as name:
            root = Path(name)
            self.assertTrue(root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
            (root / "order.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows) + tail, encoding="utf-8")
            return driver.report(root)

    def test_median_gate_does_not_add_a_per_pair_percentage_gate(self):
        result = self.evaluate(ledger([6.0, 20.0, 20.0], [4.9] * 3))
        self.assertTrue(result["qualification_passed"])
        self.assertLess(result["journeys"]["readme"]["per_pair_reduction_pct_informational"][0], 30)

    def test_every_candidate_uses_a_strict_unrounded_five_second_limit(self):
        for values, expected in (([5.0, 4.9, 4.9], False), ([4.999999999] * 3, True)):
            with self.subTest(values=values):
                self.assertEqual(self.evaluate(ledger([20.0] * 3, values))["qualification_passed"], expected)

    def test_subfive_candidate_still_requires_thirty_percent_median_reduction(self):
        self.assertFalse(self.evaluate(ledger([5.0] * 3, [4.0] * 3))["qualification_passed"])

    def test_incomplete_failed_overlapping_and_truncated_ledgers_cannot_qualify(self):
        good = ledger([20.0] * 3, [4.0] * 3)
        failed, overlap = copy.deepcopy(good), copy.deepcopy(good)
        failed[-1]["eligible"] = False
        overlap[2]["clock_ns"] = overlap[1]["clock_ns"] - 1
        for rows, tail in ((good[:-1], ""), (good[:-1], '{"status":'), (failed, ""), (overlap, "")):
            with self.subTest(tail=tail, rows=len(rows)):
                self.assertFalse(self.evaluate(rows, tail)["complete_eligible_sequence"])

    def attempts(self, *cohorts):
        with tempfile.TemporaryDirectory(prefix="task40608-pair-recovery-") as name:
            root = Path(name)
            self.assertTrue(root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
            for number, rows in enumerate(cohorts, 1):
                attempt = root / f"attempt-{number:04d}"
                attempt.mkdir()
                (attempt / "order.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            return driver.report(root)

    def test_resume_retains_completed_pair_and_retries_interrupted_pair_after_clock_reset(self):
        good = ledger([20.0] * 3, [4.0] * 3)
        resumed = copy.deepcopy(good[4:])
        for row in resumed:
            row["clock_ns"] -= 4  # New driver/boot, no cross-attempt clock comparison.
        result = self.attempts(good[:6], resumed)
        self.assertTrue(result["qualification_passed"])
        self.assertEqual(result["pair_attempts"][1], "attempt-0001/order.jsonl")
        self.assertEqual(result["pair_attempts"][2], "attempt-0002/order.jsonl")

    def test_half_pairs_from_separate_attempts_cannot_be_combined(self):
        good = ledger([20.0] * 3, [4.0] * 3)
        self.assertEqual(self.attempts(good[:2], good[2:4])["completed_pairs"], [])

    def test_completed_slow_pair_is_not_replaced_by_a_later_faster_retry(self):
        slow = ledger([20.0] * 3, [5.1] * 3)
        fast = ledger([20.0] * 3, [4.0] * 3)
        result = self.attempts(slow[:4], fast)
        self.assertFalse(result["qualification_passed"])
        self.assertEqual(result["pair_attempts"][1], "attempt-0001/order.jsonl")

    def test_pair_source_hashes_cannot_change_between_samples(self):
        good = ledger([20.0] * 3, [4.0] * 3)
        good[2]["input_hashes"] = {"candidate_manifest": "changed"}
        self.assertNotIn(1, self.attempts(good)["completed_pairs"])

    def test_resume_rejects_new_configuration_pins_before_creating_an_attempt(self):
        with tempfile.TemporaryDirectory(prefix="task40608-resume-pins-") as name:
            root = Path(name)
            self.assertTrue(root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
            batch = root / "q1"
            batch.mkdir()
            driver.durable(batch / "inputs.json", {"files": {"candidate": "old"}})
            with self.assertRaisesRegex(RuntimeError, "different source/configuration pins"):
                driver.prepare_attempt(batch, root / "active.json", "q1",
                                       {"files": {"candidate": "new"}}, root / "config.json", True)
            self.assertEqual(list(batch.glob("attempt-*")), [])

    def test_resume_preserves_completed_pair_and_torn_marker_in_fresh_attempt(self):
        with tempfile.TemporaryDirectory(prefix="task40608-resume-marker-") as name:
            root = Path(name)
            self.assertTrue(root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
            batch = root / "q1"
            batch.mkdir()
            settings = {"files": {}}
            driver.durable(batch / "inputs.json", settings)
            rows = ledger([20.0] * 3, [4.0] * 3)[:6]
            for index, row in enumerate(rows):
                row["input_hashes" if index % 2 == 0 else "source_hashes_after"] = {}
                driver.durable(batch / "order.jsonl", row, "a")
            marker = root / "active.json"
            marker.write_bytes(b"\x00")
            attempt, prior = driver.prepare_attempt(batch, marker, "q1", settings,
                                                    root / "config.json", True)
            self.assertEqual(prior["completed_pairs"], [1])
            self.assertEqual(prior["pending_pairs"], [2, 3])
            self.assertEqual(attempt.name, "attempt-0001")
            self.assertEqual((attempt / "previous-active.bin").read_bytes(), b"\x00")
            self.assertTrue(driver.read(marker)["resumed"])


if __name__ == "__main__":
    unittest.main()
