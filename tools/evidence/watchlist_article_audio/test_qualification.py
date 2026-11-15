"""Real parser and numerical measurement checks; no audio-device access."""

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import wave

import numpy as np
from scipy.signal import resample_poly

from qualification import (
    ProgressObservation,
    StatsReader,
    _write_wav,
    artifact_receipt,
    bookmark_alignment,
    marker_signal,
    measure_match,
    parse_stats_position,
    native_result,
    observed_playing,
    resolve_ffplay,
    run_matrix,
    saved_bookmark_frame,
    select_fresh,
    stop_is_due,
)


def stats(value):
    # ffplay's stderr record format, independent of the parser implementation.
    return (
        f"  {value} M-A:  0.000 fd=   0 aq=    8KB vq=    0KB sq=    0B f=0/0".encode()
    )


class ParserTests(unittest.TestCase):
    def test_valid_ffplay_clock(self):
        self.assertEqual(parse_stats_position(stats("12.34")), 12.34)
        self.assertEqual(
            parse_stats_position(b"  0.05 A-V: nan fd=0 aq=8KB vq=0KB sq=0B f=0/0"),
            0.05,
        )

    def test_rejects_nonfinite_negative_and_arbitrary_logs(self):
        for record in [
            stats("nan"),
            stats("inf"),
            stats("-0.20"),
            b"12.34 log",
            b"warning " + stats("2.0"),
        ]:
            with self.subTest(record=record):
                self.assertIsNone(parse_stats_position(record))

    def test_split_crlf_records(self):
        reader = StatsReader()
        record = stats("1.25")
        self.assertEqual(reader.feed(record[:12], generation=7, observed_at_s=1), [])
        self.assertEqual(
            reader.feed(
                record[12:] + b"\r\n" + stats("1.50") + b"\r",
                generation=7,
                observed_at_s=2,
            ),
            [ProgressObservation(1.25, 2, 7), ProgressObservation(1.5, 2, 7)],
        )

    def test_oversize_record_is_discarded_until_boundary(self):
        reader = StatsReader(max_record_bytes=128)
        self.assertEqual(reader.feed(b"x" * 1000000, generation=1, observed_at_s=1), [])
        self.assertLessEqual(reader.retained_bytes, 128)
        self.assertEqual(
            reader.feed(stats("9.0") + b"\r", generation=1, observed_at_s=1), []
        )
        self.assertEqual(
            reader.feed(stats("2.0") + b"\r", generation=1, observed_at_s=2),
            [ProgressObservation(2, 2, 1)],
        )

    def test_partial_retired_generation_cannot_complete_new_record(self):
        reader = StatsReader()
        reader.feed(stats("8.0")[:20], generation=1, observed_at_s=1)
        self.assertEqual(
            reader.feed(stats("8.0")[20:] + b"\r", generation=2, observed_at_s=2), []
        )

    def test_fresh_selection_rejects_stale_future_and_retired(self):
        observations = [
            ProgressObservation(1, 9.8, 1),
            ProgressObservation(2, 9.0, 2),
            ProgressObservation(3, 10.1, 2),
        ]
        self.assertIsNone(select_fresh(observations, generation=2, now_s=10))
        observations.append(ProgressObservation(4, 9.9, 2))
        selected = select_fresh(observations, generation=2, now_s=10)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.position_s, 4)


class MeasurementTests(unittest.TestCase):
    def test_full_near_eof_slices_match_and_truncated_slices_cannot_pass_tail(self):
        source = marker_signal(2.5)
        for start_frame, expected_start in [(113280, 2.36), (117120, 2.44)]:
            with self.subTest(remaining=2.5 - expected_start):
                full = np.concatenate(
                    [np.zeros(4800), source[start_frame:], np.zeros(12000)]
                )
                match = measure_match(full, source)
                self.assertIsNotNone(match)
                self.assertAlmostEqual(
                    match["start_source_s"], expected_start, delta=0.01
                )
                self.assertAlmostEqual(match["end_source_s"], 2.5, delta=0.01)
                truncated = np.concatenate(
                    [np.zeros(4800), source[start_frame:118560], np.zeros(12000)]
                )
                match = measure_match(truncated, source)
                if match is not None:
                    self.assertGreater(2.5 - match["end_source_s"], 0.020)

    def test_padding_preserves_original_pcm_and_adds_exactly_12000_zero_frames(self):
        source = marker_signal(0.120)
        with tempfile.TemporaryDirectory() as scratch:
            original = Path(scratch) / "original.wav"
            padded = Path(scratch) / "padded.wav"
            _write_wav(original, source)
            _write_wav(padded, source, tail_padding_ms=250)
            with wave.open(str(original), "rb") as wav:
                original_pcm = wav.readframes(5760)
            with wave.open(str(padded), "rb") as wav:
                self.assertEqual(wav.getnframes(), 17760)
                self.assertEqual(wav.getnchannels(), 2)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), 48000)
                self.assertEqual(wav.readframes(5760), original_pcm)
                self.assertEqual(wav.readframes(12000), b"\x00" * 48000)
            geometry = artifact_receipt(padded, 5760)
            self.assertEqual(geometry["content_frames"], 5760)
            self.assertEqual(geometry["file_frames"], 17760)
            self.assertEqual(geometry["padding_frames"], 12000)
            self.assertEqual(geometry["content_duration_s"], 0.120)
            self.assertEqual(geometry["file_duration_s"], 0.370)

    def test_tail_correlator_distinguishes_full_and_truncated_content(self):
        source = marker_signal(2.5, seed=41177)
        full = np.concatenate([np.zeros(4800), source, np.zeros(4800)])
        truncated = np.concatenate([np.zeros(4800), source[:117600], np.zeros(4800)])
        complete = measure_match(full, source)
        incomplete = measure_match(truncated, source)
        self.assertIsNotNone(complete)
        self.assertIsNotNone(incomplete)
        self.assertAlmostEqual(complete["end_source_s"], 2.5, delta=0.01)
        self.assertAlmostEqual(incomplete["end_source_s"], 2.45, delta=0.01)
        self.assertGreater(2.5 - incomplete["end_source_s"], 0.04)

    def test_pcm16_stereo_wav_preserves_matchable_marker(self):
        source = marker_signal(0.5)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "synthetic.wav"
            _write_wav(path, source)
            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 2)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), 48000)
                pcm = np.frombuffer(wav.readframes(24000), dtype="<i2").reshape(-1, 2)
            match = measure_match(pcm / 32768, source)
            self.assertIsNotNone(match)
            self.assertAlmostEqual(match["start_source_s"], 0, delta=0.01)
            self.assertAlmostEqual(match["end_source_s"], 0.5, delta=0.01)

    def test_invalid_source_cannot_claim_a_match(self):
        source = marker_signal(1)
        captured = source.copy()
        source[0] = np.nan
        with warnings.catch_warnings(record=True) as emitted:
            self.assertIsNone(measure_match(captured, source))
        self.assertEqual(len(emitted), 0)
        self.assertIsNone(measure_match(np.ones(5), captured))

    def test_delayed_resampled_capture_recovers_offset_and_tail(self):
        source = marker_signal(3)
        # Independent known truth: capture contains media [1.230, 2.530] seconds.
        excerpt = source[59040:121440]
        captured = np.concatenate(
            [np.zeros(8820), resample_poly(excerpt, 147, 160), np.zeros(4410)]
        )
        result = measure_match(captured, source, capture_rate=44100)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["start_source_s"], 1.230, delta=0.010)
        self.assertAlmostEqual(result["end_source_s"], 2.530, delta=0.010)
        self.assertAlmostEqual(result["capture_onset_s"], 0.200, delta=0.010)

    def test_short_marker_tail_is_measured(self):
        source = marker_signal(0.120)
        captured = np.concatenate([np.zeros(4800), source, np.zeros(4800)])
        result = measure_match(captured, source)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["end_source_s"], 0.110)

    def test_silence_and_unrelated_capture_do_not_match(self):
        source = marker_signal(2)
        self.assertIsNone(measure_match(np.zeros(48000), source))
        unrelated = np.random.default_rng(123).normal(0, 0.01, 48000)
        self.assertIsNone(measure_match(unrelated, source))
        self.assertIsNone(measure_match(marker_signal(1, seed=22), source))


class CliTests(unittest.TestCase):
    def test_padded_repeat_options_in_no_capture_receipt(self):
        with tempfile.TemporaryDirectory() as scratch:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("qualification.py")),
                    "--output-dir",
                    scratch,
                    "--tail-padding-ms",
                    "250",
                    "--repeat",
                    "3",
                    "--ffplay",
                    "must-not-open",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads((Path(scratch) / "qualification.json").read_text())
            self.assertEqual(receipt["tail_padding_ms"], 250)
            self.assertEqual(receipt["repeat"], 3)
            self.assertEqual(receipt["run_count"], 0)
            self.assertEqual(receipt["full_gate"], "unqualified")

    def test_invalid_padding_and_repeat_refuse_before_creating_output(self):
        for option, value in [
            ("--tail-padding-ms", "1"),
            ("--tail-padding-ms", "-250"),
            ("--repeat", "0"),
            ("--repeat", "4"),
        ]:
            with (
                self.subTest(option=option, value=value),
                tempfile.TemporaryDirectory() as scratch,
            ):
                destination = Path(scratch) / "must-not-create"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("qualification.py")),
                        "--output-dir",
                        str(destination),
                        "--capture-loopback",
                        option,
                        value,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(destination.exists())

    def test_no_capture_only_checks_fixtures_and_never_launches_player(self):
        with tempfile.TemporaryDirectory() as scratch:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("qualification.py")),
                    "--output-dir",
                    scratch,
                    "--ffplay",
                    "missing-player-must-not-be-launched",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(completed.stderr, "")
            receipt = json.loads((Path(scratch) / "qualification.json").read_text())
            self.assertEqual(receipt["fixture"]["result"], "pass")
            self.assertEqual(receipt["fixture"].get("parser_result"), "pass")
            self.assertGreater(receipt["clock"]["resolution_s"], 0)
            self.assertEqual(receipt["run_count"], 0)
            self.assertEqual(receipt["full_gate"], "unqualified")
            self.assertEqual(set(receipt["native_platforms"].values()), {"unqualified"})
            self.assertNotIn("executable", receipt)


class PhaseTests(unittest.TestCase):
    def test_padded_stop_countdown_waits_for_playing_instead_of_launch(self):
        self.assertFalse(
            stop_is_due(
                now_s=20,
                launched_at_s=0,
                playing_at_s=None,
                delay_s=0.7,
                wait_for_playing=True,
            )
        )
        self.assertFalse(
            stop_is_due(
                now_s=20,
                launched_at_s=0,
                playing_at_s=19.5,
                delay_s=0.7,
                wait_for_playing=True,
            )
        )
        self.assertTrue(
            stop_is_due(
                now_s=20.3,
                launched_at_s=0,
                playing_at_s=19.5,
                delay_s=0.7,
                wait_for_playing=True,
            )
        )
        self.assertTrue(
            stop_is_due(
                now_s=1,
                launched_at_s=0,
                playing_at_s=None,
                delay_s=0.7,
                wait_for_playing=False,
            )
        )

    def test_playing_requires_recent_postlaunch_source_matched_loopback(self):
        source = marker_signal(2.5)
        blocks = [(10.2, np.repeat(source[4800:14400, None], 2, axis=1))]
        self.assertIsNotNone(
            observed_playing(blocks, source, launched_at_s=10, now_s=10.3)
        )
        self.assertIsNone(observed_playing(blocks, source, launched_at_s=10, now_s=11))
        self.assertIsNone(
            observed_playing(blocks, source, launched_at_s=10.5, now_s=10.6)
        )
        self.assertIsNone(
            observed_playing(
                [(10.2, np.zeros((9600, 2)))], source, launched_at_s=10, now_s=10.3
            )
        )


class ResolutionTests(unittest.TestCase):
    def test_explicit_default_and_custom_chocolatey_shims_are_refused(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            custom = base / "alternate-install"
            with patch.dict(
                os.environ, {"ProgramData": scratch, "ChocolateyInstall": str(custom)}
            ):
                for folder in [base / "chocolatey" / "bin", custom / "bin"]:
                    folder.mkdir(parents=True)
                    shim = folder / "ffplay.exe"
                    shim.write_bytes(b"fixture only: must never execute")
                    # Exercise normalization before the shim-location check.
                    explicit = folder / ".." / "bin" / "ffplay.exe"
                    with self.subTest(folder=folder), self.assertRaises(ValueError):
                        resolve_ffplay(str(explicit))

    def test_explicit_direct_payload_file_remains_accepted(self):
        with tempfile.TemporaryDirectory() as scratch:
            payload = Path(scratch) / "chocolatey" / "lib" / "ffmpeg" / "ffplay.exe"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"fixture only: must never execute")
            self.assertEqual(resolve_ffplay(str(payload)), payload.resolve())


class BookmarkTests(unittest.TestCase):
    def test_bookmark_must_align_with_independently_matched_stopped_end(self):
        self.assertEqual(bookmark_alignment(24000, 0.749)["bookmark_alignment"], "pass")
        self.assertAlmostEqual(
            bookmark_alignment(24000, 0.749)["bookmark_end_error_s"], -0.249
        )
        self.assertEqual(bookmark_alignment(24000, 0.751)["bookmark_alignment"], "fail")
        self.assertEqual(
            bookmark_alignment(None, 0.75)["bookmark_alignment"], "unqualified"
        )

    def test_fresh_current_generation_bookmark_becomes_one_content_frame(self):
        values = [ProgressObservation(1.25, 9.9, 4), ProgressObservation(2.7, 9.95, 3)]
        self.assertEqual(
            saved_bookmark_frame(values, generation=4, now_s=10, content_frames=120000),
            60000,
        )
        values.append(ProgressObservation(2.7, 9.99, 4))
        self.assertIsNone(
            saved_bookmark_frame(values, generation=4, now_s=10, content_frames=120000)
        )


class MatrixTests(unittest.TestCase):
    def test_censored_stop_missing_latency_is_unqualified_but_measured_overrun_fails(
        self,
    ):
        censored = {"case": "stop_playing", "stop_after_launch_s": 0.7}
        self.assertEqual(native_result([censored]), "unqualified")
        self.assertEqual(
            native_result([dict(censored, stop_process_latency_s=1.6)]), "fail"
        )

    def test_missing_bookmark_logical_eof_or_unreaped_stop_cannot_launch_resume(self):
        for changed in [
            {"saved_bookmark_frame": None, "bookmark": "unqualified"},
            {"saved_bookmark_frame": 120000},
            {"capture_reaped": False},
        ]:
            calls = []

            def capture_boundary(*args, calls=calls, changed=changed, **kwargs):
                calls.append(kwargs["offset"])
                result = {
                    "launched": True,
                    "process_exit": "pass",
                    "endpoint_loopback": "pass",
                    "post_stop_silence": "pass",
                    "post_stop_latency": "pass",
                    "player_reaped": True,
                    "capture_reaped": True,
                    "saved_bookmark_frame": 19200,
                    "bookmark": "pass",
                    "bookmark_alignment": "pass",
                    "stop_process_latency_s": 0.01,
                }
                result.update(changed)
                return result

            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as scratch,
                patch("qualification.run_case", side_effect=capture_boundary),
            ):
                runs = run_matrix(
                    Path("never-executed"), Path(scratch), tail_padding_ms=250, repeat=1
                )
            self.assertEqual(len(calls), 6)
            self.assertEqual(len(runs), 7)
            self.assertFalse(runs[-1]["launched"])
            self.assertEqual(
                native_result(runs),
                "fail" if changed.get("capture_reaped") is False else "unqualified",
            )

    def test_three_rounds_resume_exact_stopped_artifact_and_saved_frame(self):
        calls = []

        def capture_boundary(
            executable, wav, source, *, offset, generation, stop_after, padded=False
        ):
            calls.append(
                (
                    wav,
                    id(source),
                    hashlib.sha256(wav.read_bytes()).hexdigest(),
                    offset,
                    stop_after,
                )
            )
            return {
                "launched": True,
                "process_exit": "pass",
                "endpoint_loopback": "pass",
                "post_stop_silence": "pass",
                "post_stop_latency": "pass",
                "player_reaped": True,
                "capture_reaped": True,
                "saved_bookmark_frame": 19200,
                "bookmark": "pass",
                "bookmark_alignment": "pass",
                "stop_process_latency_s": 0.01,
            }

        with (
            tempfile.TemporaryDirectory() as scratch,
            patch("qualification.run_case", side_effect=capture_boundary),
        ):
            runs = run_matrix(
                Path("never-executed"), Path(scratch), tail_padding_ms=250, repeat=3
            )
        self.assertEqual(len(calls), 21)
        self.assertEqual(len(runs), 21)
        for index in (5, 12, 19):
            stopped, resumed = calls[index : index + 2]
            self.assertEqual(stopped[:3], resumed[:3])
            self.assertEqual(resumed[3], 0.4)
            self.assertEqual(
                runs[index]["artifact_sha256"], runs[index + 1]["artifact_sha256"]
            )
            self.assertEqual(
                runs[index + 1]["resume_from_generation"], runs[index]["generation"]
            )
        self.assertEqual({run["round"] for run in runs}, {1, 2, 3})
        self.assertEqual(len({str(call[0]) for call in calls}), 3)

    def test_native_gate_cannot_average_away_unmatched_or_unreaped_case(self):
        run = {
            "case": "near_eof_60ms",
            "stop_after_launch_s": None,
            "duration_s": 2.5,
            "requested_start_s": 2.44,
            "process_exit": "pass",
            "endpoint_loopback": "pass",
            "seek": "pass",
            "tail_coverage": "pass",
            "player_reaped": True,
            "capture_reaped": True,
        }
        self.assertEqual(native_result([run]), "pass")
        self.assertEqual(
            native_result([run, dict(run, endpoint_loopback="unqualified")]),
            "unqualified",
        )
        self.assertEqual(native_result([run, dict(run, tail_coverage="fail")]), "fail")
        self.assertEqual(native_result([run, dict(run, capture_reaped=False)]), "fail")

    def test_retired_stale_or_future_callback_cannot_supply_bookmark(self):
        values = [
            ProgressObservation(1.0, 9.9, 3),
            ProgressObservation(2.0, 9.0, 4),
            ProgressObservation(1.5, 10.1, 4),
        ]
        self.assertIsNone(
            saved_bookmark_frame(values, generation=4, now_s=10, content_frames=120000)
        )


if __name__ == "__main__":
    unittest.main()
