"""Device-free checks for the WASAPI timeline and CFFI buffer ownership."""

import unittest

from cffi import FFI
import numpy as np

from wasapi_packets import (
    BUFFER_EMPTY,
    DISCONTINUITY,
    SILENT,
    TIMESTAMP_ERROR,
    Packet,
    content_span,
    quiet_window,
    read_packet,
)
from qualification import _match_packets, marker_signal


def packet(frame, start_s, flags=0):
    value = 0.0 if flags & SILENT else 0.01
    return Packet(frame, start_s, flags, np.full((480, 2), value, dtype=np.float32))


class PacketTests(unittest.TestCase):
    def test_content_uses_real_packet_positions_across_idle_monitor_intervals(self):
        metrics = {}
        span = content_span(
            [packet(0, 10.0, SILENT), packet(6000, 10.125), packet(6480, 10.135)],
            metrics,
        )
        self.assertIsNotNone(span)
        data, start = span
        self.assertEqual(data.shape, (960, 2))
        self.assertAlmostEqual(start, 10.125)
        self.assertEqual(metrics["content_frames"], 960)

    def test_gaps_and_timestamp_errors_cannot_be_spliced_between_marker_packets(self):
        for later, reason in [
            (packet(960, 10.02), "device_frame_gap"),
            (packet(480, 10.13), "packet_clock_gap"),
            (packet(480, 10.01, TIMESTAMP_ERROR), "packet_timestamp_error"),
            (packet(480, 10.01, DISCONTINUITY), "packet_discontinuity"),
        ]:
            with self.subTest(reason=reason):
                metrics = {}
                self.assertIsNone(content_span([packet(0, 10.0), later], metrics))
                self.assertEqual(metrics.get("timeline_stage"), reason)

    def test_no_packet_quiet_requires_continuous_drain_observations(self):
        complete = quiet_window([], [10.0, 10.05, 10.1, 10.15, 10.2], 10.01, 10.19)
        self.assertEqual(complete.get("status"), "pass")
        self.assertEqual(complete["evidence"], "monitored_no_packets")
        self.assertIsNone(complete["rms"])
        missing = quiet_window([], [10.0, 10.2], 10.01, 10.19)
        self.assertEqual(missing["status"], "unqualified")

    def test_quiet_only_measures_frames_inside_window(self):
        result = quiet_window(
            [packet(0, 10.00), packet(480, 10.01, SILENT)], [10.0, 10.03], 10.01, 10.02
        )
        self.assertEqual(result.get("status"), "pass")
        self.assertEqual(result["packet_frames"], 480)
        self.assertEqual(result["rms"], 0)

    def test_invalid_timestamp_or_pcm_cannot_be_quiet(self):
        for invalid in [
            packet(0, 0.0, TIMESTAMP_ERROR),
            Packet(0, 10.01, 0, np.full((480, 2), np.nan)),
        ]:
            with self.subTest(flags=invalid.flags):
                result = quiet_window([invalid], [10.0, 10.05], 10.01, 10.02)
                self.assertEqual(result["status"], "unqualified")

    def test_quiet_requires_post_window_drain_and_local_buffer_bound(self):
        self.assertEqual(
            quiet_window([], [10.0, 10.05, 10.10], 10.01, 10.09, drain_delay_s=0.02)[
                "status"
            ],
            "unqualified",
        )
        self.assertEqual(
            quiet_window([], [10.0, 10.05, 10.10], 10.01, 10.09, max_gap_s=0.02)[
                "status"
            ],
            "unqualified",
        )

    def test_packet_source_match_uses_native_qpc_origin(self):
        source = marker_signal(0.5)
        frames = np.repeat(np.asarray(source, dtype=np.float32)[:, None], 2, axis=1)
        packets = [
            Packet(5000 + frame, 10.0 + frame / 48000, 0, frames[frame : frame + 480])
            for frame in range(0, len(frames), 480)
        ]
        match, metrics = _match_packets(packets, source)
        self.assertIsNotNone(match)
        self.assertAlmostEqual(match["capture_onset_qpc_s"], 10.0, delta=0.004)
        self.assertAlmostEqual(match["capture_end_qpc_s"], 10.5, delta=0.004)
        self.assertIsNone(_match_packets(packets[:10] + packets[11:], source)[0])
        self.assertEqual(metrics["timeline_stage"], "contiguous")

    def test_silent_null_packet_is_released_without_dereference(self):
        ffi = FFI()
        ffi.cdef(
            "typedef unsigned char BYTE; typedef uint32_t UINT32; typedef uint32_t DWORD; typedef uint64_t UINT64;"
        )
        releases = []

        class Client:
            def GetBuffer(self, pointer, data, frames, flags, device_pos, qpc):
                frames[0], flags[0], device_pos[0], qpc[0] = 480, SILENT, 960, 100000000
                return 0

            def ReleaseBuffer(self, pointer, frames):
                releases.append(frames)
                return 0

        entry = type("Entry", (), {"lpVtbl": Client()})()
        result = read_packet(
            ffi,
            lambda hr: self.assertEqual(hr, 0),
            [entry],
            expected_frames=480,
            capacity=4800,
        )
        self.assertEqual(releases, [480])
        self.assertEqual(result.frame, 960)
        self.assertEqual(result.start_s, 10.0)
        self.assertFalse(result.data.any())
        with self.assertRaisesRegex(RuntimeError, "invalid_native_packet_size"):
            read_packet(
                ffi,
                lambda hr: self.assertEqual(hr, 0),
                [entry],
                expected_frames=481,
                capacity=4800,
            )
        self.assertEqual(releases, [480, 480])
        entry.lpVtbl.GetBuffer = lambda *args: BUFFER_EMPTY
        self.assertIsNone(
            read_packet(
                ffi,
                lambda hr: self.fail("empty result is not an error"),
                [entry],
                expected_frames=480,
                capacity=4800,
            )
        )
        self.assertEqual(releases, [480, 480])
