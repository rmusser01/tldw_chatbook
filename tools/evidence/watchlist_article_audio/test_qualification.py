"""Real parser and numerical measurement checks; no audio-device access."""

import json
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
    marker_signal,
    measure_match,
    parse_stats_position,
    resolve_ffplay,
    select_fresh,
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


if __name__ == "__main__":
    unittest.main()
