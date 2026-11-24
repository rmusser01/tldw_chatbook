"""Mutate retained measurements without opening a file player or capture device."""

import asyncio
from copy import deepcopy
from contextlib import ExitStack
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import json
from pathlib import Path
import unittest

from controlled_instrument import separation_check
from export_controlled_receipts import project
from grade_controlled_audio import (
    grade_archive,
    grade_control,
    grade_matrix,
    grade_transfer,
)

ARCHIVE = (
    Path(__file__).resolve().parents[3]
    / "Docs/Testing/evidence/watchlist-article-audio"
)


def receipt(name):
    return json.loads(
        (ARCHIVE / f"controlled-adapter-{name}.json").read_text(encoding="utf-8")
    )


class ControlledEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controls = {r["case"]: r for r in receipt("native-controls-4")["runs"]}
        cls.transfers = receipt("transfers-3rounds")["runs"]

    def reject_mutation(self, grade, original, keys, invalid):
        changed = deepcopy(original)
        leaf = changed
        for key in keys[:-1]:
            leaf = leaf[key]
        leaf[keys[-1]] = invalid
        with (
            self.subTest(keys=keys, invalid=invalid),
            self.assertRaises((ValueError, TypeError, KeyError)),
        ):
            grade(changed)

    def test_archive_has_seven_waveform_passes_and_fifteen_signed_gaps(self):
        result = grade_archive(ARCHIVE)
        self.assertEqual(
            sum(v == "natural_waveform_pass" for v in result["controls"].values()), 7
        )
        self.assertEqual(
            (result["transfers"], result["matched_pairs"], result["quiet_old"]),
            (18, 15, 3),
        )
        self.assertAlmostEqual(result["minimum_gap_s"], 0.0783167)

    def test_missing_tail_cannot_pass_on_clean_natural_exit(self):
        self.reject_mutation(
            grade_control,
            self.controls["natural_500ms"],
            ("waveform_match", "end_source_s"),
            0.43,
        )

    def test_missing_early_and_late_correlation_or_drift(self):
        for key, value in (
            ("early_score", 0.699),
            ("tail_score", 0.699),
            ("alignment_drift_s", -0.021),
        ):
            self.reject_mutation(
                grade_control,
                self.controls["natural_500ms"],
                ("waveform_match", key),
                value,
            )

    def test_wrong_terminal_unknown_custody_and_stale_generation(self):
        r = self.controls["active_stop"]
        for path, value in (
            (("completion", "kind"), "natural"),
            (("stop_observation", "generation"), 2),
            (("final_retirement_confirmed",), None),
            (("completion", "evidence", "retired_at_s"), 1.0),
        ):
            self.reject_mutation(grade_control, r, path, value)

    def test_capture_gaps_and_missing_quiet_observer(self):
        r = self.controls["natural_500ms"]
        for path, value in (
            (("capture_reaped",), False),
            (("capture_error",), "capture_failed"),
            (("capture_metrics", "discontinuity_count"), 1),
            (("capture_metrics", "timestamp_error_count"), 1),
            (("capture_metrics", "poll_gap_max_s"), 0.1001),
            (("waveform_diagnostics", "packet_clock_error_max_s"), 0.00501),
            (("post_reap_quiet", "evidence"), "monitor_gap"),
        ):
            self.reject_mutation(grade_control, r, path, value)

    def test_saved_resume_requires_exact_source_bookmark(self):
        stopped = self.controls["active_stop"]
        resume = self.controls["resume_saved_bookmark"]
        self.reject_mutation(
            lambda row: grade_control(row, stopped),
            resume,
            ("start_frame",),
            resume["start_frame"] + 1,
        )
        self.reject_mutation(
            lambda row: grade_control(row, stopped),
            resume,
            ("completion", "bookmark", "sha256"),
            "0" * 64,
        )

    def test_one_frame_never_qualifies_waveform_and_exact_eof_never_launches(self):
        self.assertEqual(
            grade_control(self.controls["near_eof_one_frame"]),
            "natural_waveform_unqualified",
        )
        self.reject_mutation(
            grade_control,
            self.controls["exact_eof"],
            ("completion", "evidence", "launched_at_s"),
            1.0,
        )

    def test_padding_stop_must_precede_reap_and_reach_physical_padding(self):
        r = receipt("padding-stop-5")["runs"][0]
        self.reject_mutation(
            grade_control,
            r,
            ("completion", "evidence", "stop_at_s"),
            r["completion"]["evidence"]["reaped_at_s"] + 0.01,
        )
        self.reject_mutation(
            grade_control,
            r,
            ("completion", "evidence", "last_physical_frame"),
            r["content_frames"] - 1,
        )

    def test_exit_zero_needs_uncertain_and_independent_shortfall(self):
        r = receipt("early-zero-exit")
        self.reject_mutation(grade_control, r, ("completion", "kind"), "natural")
        self.reject_mutation(grade_control, r, ("waveform_match", "end_source_s"), 2.5)
        self.reject_mutation(
            grade_control, r, ("fault", "duplicate_handle_closed"), False
        )

    def test_programmed_stereo_mixture_keeps_real_overlap_and_gap(self):
        overlap, gap = separation_check()
        self.assertAlmostEqual(overlap, -0.2, delta=0.02)
        self.assertAlmostEqual(gap, 0.2, delta=0.02)

    def test_overlap_and_marginal_separation_fail_despite_quiet_isolation_windows(self):
        for gap in (-0.05, 0.039):
            r = deepcopy(self.transfers[0])
            r["new_match"]["capture_onset_qpc_s"] = (
                r["old_match"]["capture_end_qpc_s"] + gap
            )
            r["new_match"]["capture_end_qpc_s"] = (
                r["new_match"]["capture_onset_qpc_s"]
                + r["new_match"]["capture_end_s"]
                - r["new_match"]["capture_onset_s"]
            )
            r["old_end_to_successor_onset_s"] = gap
            with self.assertRaisesRegex(ValueError, "overlap"):
                grade_transfer(r)

    def test_crosstalk_unknown_retirement_and_stale_lease(self):
        for path, value in (
            (("left_channel_during_new_only", "rms"), 0.00021),
            (("right_channel_during_old_only", "rms"), 0.00021),
            (("new_retirement_confirmed",), False),
            (("old_retirement_confirmed",), None),
            (("old_lease_revoked",), False),
            (("new_completion", "generation"), 1),
        ):
            self.reject_mutation(grade_transfer, self.transfers[0], path, value)

    def test_startup_no_match_requires_quiet_observed_old_channel(self):
        self.reject_mutation(
            grade_transfer, self.transfers[1], ("old_channel_monitor", "rms"), 0.001
        )
        self.reject_mutation(
            grade_transfer,
            self.transfers[1],
            ("old_channel_monitor", "packet_frames"),
            0,
        )

    def test_missing_matrix_cell_cannot_hide_behind_duplicate_pass(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            grade_matrix(self.transfers[:-1] + [self.transfers[0]])

    def test_export_excludes_private_or_unknown_values(self):
        self.assertEqual(
            project({"pid": 99, "stderr": "private", "unknown": [1, 2]}, "runs"), {}
        )
        with self.assertRaises(ValueError):
            project({"reason": "private"}, "runs")
        with self.assertRaises(ValueError):
            project({"packet_frames": [0, 1, 2]}, "runs")

    def test_matching_reconstructs_diagnostics_and_requires_real_packets(self):
        row = self.controls["natural_2500ms"]
        self.reject_mutation(
            grade_control,
            row,
            ("waveform_diagnostics",),
            receipt("early-zero-exit")["waveform_diagnostics"],
        )
        self.reject_mutation(grade_control, row, ("capture_metrics", "packet_count"), 0)
        self.reject_mutation(
            grade_control, row, ("waveform_diagnostics", "active_frames"), 0
        )

    def test_measured_edges_stay_in_run_lifetime_and_calibration_bound(self):
        for seconds in (-20, 20):
            for original, grade, keys in (
                (self.controls["natural_120ms"], grade_control, ("waveform_match",)),
                (self.transfers[0], grade_transfer, ("old_match", "new_match")),
            ):
                r = deepcopy(original)
                for key in keys:
                    for edge in ("capture_onset_qpc_s", "capture_end_qpc_s"):
                        r[key][edge] += seconds
                with (
                    self.subTest(seconds=seconds, keys=keys),
                    self.assertRaises(ValueError),
                ):
                    grade(r)
        self.reject_mutation(
            grade_control,
            self.controls["natural_120ms"],
            ("capture_configuration", "qpc_calibration_bracket_s"),
            0.004,
        )

    def raw_cli(self, kind, data):
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "receipt.json"
            target.write_text(json.dumps(data), encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("grade_controlled_audio.py")),
                    "--kind",
                    kind,
                    str(target),
                ],
                capture_output=True,
                text=True,
            ).returncode

    def test_raw_controls_cli_requires_schema_manifest_and_fixture_proof(self):
        original = receipt("canonical-controls-6")
        self.assertEqual(self.raw_cli("controls", original), 0)
        for change in ("schema", "name", "digest", "mode", "fixture", "identity"):
            data = deepcopy(original)
            if change == "schema":
                data["schema_version"] = 1
            elif change in ("name", "digest"):
                files = data["source_file_sha256"]
                key = next(iter(files))
                if change == "name":
                    files["unexpected.py"] = files.pop(key)
                else:
                    files[key] = "bad"
            elif change == "mode":
                del data["runs"][0]["fixture_native_mode_at_admission"]
            elif change == "identity":
                del next(
                    r for r in data["runs"] if r["case"] == "resume_saved_bookmark"
                )["same_retained_file"]
            else:
                for r in data["runs"]:
                    for key in (
                        "fixture",
                        "fixture_native_mode_at_admission",
                        "fixture_native_mode_after_retirement",
                        "same_retained_file",
                    ):
                        r.pop(key, None)
            with self.subTest(change=change):
                self.assertNotEqual(self.raw_cli("controls", data), 0)

    def test_raw_transfer_cli_requires_both_fixtures(self):
        data = receipt("canonical-controls-6")
        fixtures = {r["content_frames"]: r["fixture"] for r in data["runs"]}
        data["runs"] = deepcopy(self.transfers)
        # Exercise the schema-2 boundary with labelled synthetic fixture facts.
        for row in data["runs"]:
            for role, count in (("old", 120000), ("new", 24000)):
                info = dict(fixtures[count])
                info["sha256"] = row[role + "_completion"].get("bookmark", info)[
                    "sha256"
                ]
                row[role + "_fixture"] = info
                row[role + "_fixture_mode_after_retirement"] = "0o600"
        self.assertEqual(self.raw_cli("transfers", data), 0)
        for role in ("old", "new"):
            for suffix in ("_fixture", "_fixture_mode_after_retirement"):
                changed = deepcopy(data)
                del changed["runs"][0][role + suffix]
                self.assertNotEqual(self.raw_cli("transfers", changed), 0)

    def test_raw_and_historical_negative_entry_refuse_natural(self):
        data = receipt("canonical-controls-6")
        ordinary = data.pop("runs")[0]
        self.assertNotEqual(self.raw_cli("negative", {**data, **ordinary}), 0)
        old_reader = Path.read_text

        def substituted(path, *args, **kwargs):
            if path.name == "controlled-adapter-early-zero-exit.json":
                return json.dumps(ordinary)
            return old_reader(path, *args, **kwargs)

        with (
            patch.object(Path, "read_text", substituted),
            self.assertRaises((KeyError, ValueError)),
        ):
            grade_archive(ARCHIVE)

    def test_negative_driver_persists_errors_and_fails_unconfirmed_custody(self):
        import windows_early_exit as driver

        for change, status in (("valid", 0), ("fault", 1), ("custody", 1)):
            with self.subTest(change=change), ExitStack() as stack:
                row = receipt("early-zero-exit")
                row["fixture"] = next(
                    r["fixture"]
                    for r in receipt("canonical-controls-6")["runs"]
                    if r["case"] == "natural_2500ms"
                )
                fault = row.pop("fault")
                if change == "fault":
                    fault = {"error": "OSError"}
                elif change == "custody":
                    row["final_retirement_confirmed"] = False
                stack.enter_context(
                    patch.object(
                        sys, "argv", ["test", "--play", "--output-dir", "unused"]
                    )
                )
                stack.enter_context(
                    patch.object(
                        driver.ci, "new_output", return_value=(Path("unused"), {})
                    )
                )
                fixtures = stack.enter_context(patch.object(driver.ci, "Fixtures"))
                stack.enter_context(patch.object(driver.ci, "make_fixture"))
                stack.enter_context(
                    patch.object(
                        driver.instrument, "control", new=AsyncMock(return_value=row)
                    )
                )
                stack.enter_context(patch.object(driver, "FAULT", fault))
                save = stack.enter_context(patch.object(driver.ci, "save"))
                self.assertEqual(asyncio.run(driver.main()), status)
                saved = save.call_args.args[1]
                self.assertEqual(saved["fault"], fault)
                self.assertEqual(
                    saved["final_retirement_confirmed"], change != "custody"
                )
                self.assertEqual(
                    fixtures.return_value.remove.call_count, int(change == "valid")
                )

    def test_transfer_stat_and_retirement_errors_do_not_skip_other_cleanup(self):
        import windows_output_transfers as driver

        async def exercise():
            old = SimpleNamespace(retire=AsyncMock(side_effect=OSError("private")))
            new = SimpleNamespace(retire=AsyncMock(return_value=True))
            capture = Mock(
                start=AsyncMock(),
                finish=AsyncMock(return_value=True),
                quiet=Mock(return_value={"status": "pass"}),
                ready_at=0,
                config={},
                metrics={},
                error=None,
            )
            host = Mock(acquire=AsyncMock(return_value=Mock(revoked=True)), owner=None)
            fed = []

            async def feeding(role):
                fed.append(role)

            async def start(run, *_):
                return feeding("old" if run is old else "new")

            with ExitStack() as stack:

                def replace(obj, name, **kw):
                    return stack.enter_context(patch.object(obj, name, **kw))

                replace(
                    driver.ci,
                    "make_fixture",
                    return_value=(SimpleNamespace(path=Path("unused")), [], b""),
                )
                replace(driver.ci, "fixture_receipt", return_value={})
                mode = replace(
                    driver.ci, "fixture_mode", side_effect=OSError("private")
                )
                replace(driver, "prepared", new=AsyncMock(side_effect=[old, new]))
                replace(driver, "Capture", return_value=capture)
                replace(driver, "AudioOutputOwnership", return_value=host)
                replace(driver, "start_run", new=start)
                replace(driver, "result_of", new=AsyncMock(return_value={}))
                row = await driver.transfer(
                    Path("unused"), "controlled", "controlled", "startup", 1
                )
                self.assertIs(row["old_retirement_confirmed"], False)
                self.assertIs(row["new_retirement_confirmed"], True)
                self.assertEqual(mode.call_count, 2)
                self.assertEqual(fed, ["old", "new"])
                capture.finish.assert_awaited_once()
                self.assertIs(row["capture_reaped"], True)
                self.assertTrue(row["error"])
                self.assertNotIn("private", json.dumps(row))

        asyncio.run(exercise())

    def test_transfer_stop_deadline_is_independent_of_successor_launch(self):
        row = deepcopy(self.transfers[4])  # PCM Stop, then controlled successor
        delta = (
            row["acquire_requested_at_s"] + 1.51 - row["old_match"]["capture_end_qpc_s"]
        )
        for side in ("old_match", "new_match"):
            for edge in ("capture_onset_qpc_s", "capture_end_qpc_s"):
                row[side][edge] += delta
        for key in (
            "launched_at_s",
            "reaped_at_s",
            "retired_at_s",
            "last_observed_at_s",
        ):
            row["new_completion"]["evidence"][key] += delta
        with self.assertRaises(ValueError):
            grade_transfer(row)

    def test_raw_negative_requires_promoted_fixture_and_uncertain_tail(self):
        metadata = receipt("canonical-controls-6")
        source = next(r for r in metadata.pop("runs") if r["case"] == "natural_2500ms")
        row = {**metadata, **receipt("early-zero-exit"), "fixture": source["fixture"]}
        self.assertEqual(self.raw_cli("negative", row), 0)
        for key in (
            "fixture",
            "source_file_sha256",
            "fixture_native_mode_after_retirement",
        ):
            changed = dict(row)
            del changed[key]
            self.assertNotEqual(self.raw_cli("negative", changed), 0)
