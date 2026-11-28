"""Device-free checks for Linux native capture and receipt boundaries."""

import io
import json
import os
import sys
from pathlib import Path
import time
import tempfile
import threading
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cffi import FFI
import numpy as np

import qualification as q
from pulse_fragments import Route


class FragmentTests(unittest.TestCase):
    def test_native_frame_copy_survives_drop_and_rejects_bad_custody(self):
        ffi = FFI()
        for fault in (
            None,
            "hole",
            "bad_size",
            "oversize",
            "nonfinite",
            "gap",
            "corrupt",
            "after_corrupt",
            "wrong_drop",
            "drop_error",
            "unaligned",
        ):
            with self.subTest(fault=fault):
                self.check_fragment_fault(ffi, fault)

    def check_fragment_fault(self, ffi, fault):
        from pulse_fragments import read_fragment

        storage = ffi.new("float[]", [0.25, -0.25, 0.5, -0.5])
        timing = SimpleNamespace(
            read_index=88, read_index_corrupt=int(fault == "corrupt")
        )
        if fault == "unaligned":
            timing.read_index = 89
        if fault == "nonfinite":
            storage[0] = float("nan")

        def peek(stream, pointer, size):
            pointer[0] = ffi.NULL if fault == "hole" else storage
            size[0] = {"bad_size": 15, "oversize": 80}.get(fault, 16)
            return 0

        def drop(stream):
            storage[0] = 0
            timing.read_index += 8 if fault == "wrong_drop" else 16
            timing.read_index_corrupt = int(fault == "after_corrupt")
            return -1 if fault == "drop_error" else 0

        api = SimpleNamespace(
            pa_stream_get_timing_info=lambda stream: timing,
            pa_stream_peek=peek,
            pa_stream_drop=drop,
        )
        if fault:
            with self.assertRaises(RuntimeError):
                read_fragment(ffi, api, None, 16, 16, 12 if fault == "gap" else 11)
        else:
            fragment = read_fragment(ffi, api, None, 16, 16, 11)
            self.assertEqual(fragment.frame, 11)
            np.testing.assert_equal(fragment.data, [[0.25, -0.25], [0.5, -0.5]])

    def test_monitor_requires_both_directions_and_exact_names_and_indexes(self):
        from pulse_fragments import verify_route

        sink = (b"private sink", 3, b"different monitor name", 8)
        source = (b"different monitor name", 8, b"private sink", 3)
        self.assertEqual(
            verify_route(b"private sink", sink, source),
            Route(b"private sink", 3, b"different monitor name", 8),
        )
        for bad in [
            (b"different", 8, b"private sink", 3),
            (source[0], 7, source[2], 3),
            (source[0], 8, b"other", 3),
            (source[0], 8, source[2], 4),
        ]:
            with self.assertRaises(RuntimeError):
                verify_route(b"private sink", sink, bad)
        with self.assertRaises(RuntimeError):
            verify_route(b"other default", sink, source)

    def test_stream_rejects_move_suspend_wrong_format_and_unavailable_buffer(self):
        ffi = FFI()
        for fault in (
            None,
            "moved",
            "suspended",
            "failed",
            "rate",
            "channels",
            "format",
            "attributes",
            "same_name_replacement",
        ):
            with self.subTest(fault=fault):
                self.check_geometry(ffi, fault)

    def check_geometry(self, ffi, fault):
        from pulse_fragments import stream_geometry

        api = SimpleNamespace(
            PA_STREAM_READY=2,
            PA_SAMPLE_FLOAT32LE=5,
            pa_stream_get_device_index=lambda _: 9
            if fault == "same_name_replacement"
            else 8,
            pa_stream_get_state=lambda _: 3 if fault == "failed" else 2,
            pa_stream_get_device_name=lambda _: ffi.new(
                "char[]", b"mic" if fault == "moved" else b"mon"
            ),
            pa_stream_is_suspended=lambda _: int(fault == "suspended"),
            pa_stream_get_sample_spec=lambda _: SimpleNamespace(
                rate=44100 if fault == "rate" else 48000,
                channels=1 if fault == "channels" else 2,
                format=3 if fault == "format" else 5,
            ),
            pa_stream_get_buffer_attr=lambda _: ffi.NULL
            if fault == "attributes"
            else SimpleNamespace(maxlength=4194304, fragsize=38400),
        )
        with self.subTest(fault=fault):
            if fault:
                with self.assertRaises(RuntimeError):
                    stream_geometry(ffi, api, None, Route(b"sink", 3, b"mon", 8))
            else:
                self.assertEqual(
                    stream_geometry(ffi, api, None, Route(b"sink", 3, b"mon", 8)),
                    (4194304, 38400),
                )


class PulseCaseTests(unittest.TestCase):
    def run_fixture(
        self,
        stop_after=None,
        *,
        baseline=True,
        complete=True,
        padded=False,
        clipped=False,
        fragment_frames=4800,
        playback_route="verified",
    ):
        from pulse_fragments import Fragment

        source = q.marker_signal(5.0 if clipped else 0.5)
        pcm = np.repeat(source[:, None], 2, axis=1).astype("<f4")
        silence = np.zeros((fragment_frames, 2), dtype="<f4")
        start = time.perf_counter()
        process = Mock(stderr=io.BytesIO(), returncode=0)
        player_env = {}
        launched = None

        class Receiver:
            count = 0
            next_frame = 900
            ready = False
            finished = False

            def poll(self, timeout=None):
                if not self.ready:
                    return True
                if finish.is_set():
                    return complete and not self.finished
                return self.count < int(
                    (time.perf_counter() - start) * 120000 / fragment_frames
                )

            def recv(self):
                if not self.ready:
                    self.ready = True
                    return (
                        "ready",
                        start,
                        {"maxlength_bytes": 4194304, "fragsize_bytes": 38400},
                        "PRIVATE_SINK",
                    )
                if finish.is_set():
                    self.finished = True
                    return ("pulse_complete", playback_route)
                index = self.count
                self.count += 1
                data = silence
                if launched is not None:
                    delta = index - launched
                    if delta * fragment_frames < len(pcm) and (
                        clipped or not process.terminate.called
                    ):
                        data = pcm[
                            delta * fragment_frames : (delta + 1) * fragment_frames
                        ]
                elif baseline == "one" and index > 0:
                    return ("unobserved",)
                elif not baseline:
                    return ("unobserved",)
                fragment = Fragment(self.next_frame, data)
                self.next_frame += len(data)
                return ("fragment", time.perf_counter(), fragment, {})

            def close(self):
                pass

        receiver = Receiver()

        def popen(*args, env, **kwargs):
            nonlocal launched
            launched = receiver.count
            player_env.update(env)
            return process

        def poll():
            return (
                0
                if process.terminate.called
                or (receiver.count - launched) * fragment_frames >= 28800
                else None
            )

        process.poll.side_effect = poll
        context = Mock()

        finish = threading.Event()
        context.Event.return_value = finish
        context.Pipe.return_value = (receiver, Mock())
        context.Process.return_value.is_alive.return_value = False
        with (
            patch("qualification.mp.get_context", return_value=context),
            patch("qualification.platform.system", return_value="Linux"),
            patch("qualification.subprocess.Popen", side_effect=popen),
        ):
            result = q.run_case(
                Path("never"),
                Path("never.wav"),
                source,
                offset=0,
                generation=1,
                stop_after=stop_after,
                padded=padded,
                capture_clock="pulse-fragment",
            )
        self.assertEqual(player_env.get("PULSE_SINK"), "PRIVATE_SINK")
        self.assertEqual(player_env.get("SDL_AUDIODRIVER"), "pulseaudio")
        self.assertNotIn("PRIVATE_SINK", json.dumps(result))
        self.assertNotEqual(os.environ.get("PULSE_SINK"), "PRIVATE_SINK")
        return result

    def test_content_passes_with_absolute_clock_unqualified(self):
        run = self.run_fixture()
        self.assertEqual(
            (run["seek"], run["tail_coverage"], run["endpoint_loopback"]),
            ("pass", "pass", "pass"),
        )
        self.assertEqual(run["telemetry"], "unqualified")
        self.assertEqual(run["telemetry_reason"], "pulse_snapshot_age_unqualified")
        self.assertEqual(q.native_result([dict(run, case="start")]), "unqualified")
        short = dict(run, case="short_120ms", duration_s=0.12)
        self.assertEqual(q.native_result([short]), "unqualified")
        self.assertEqual(q.native_result([dict(short, tail_coverage="fail")]), "fail")

    def test_startup_stop_keeps_process_and_sample_results_separate(self):
        run = self.run_fixture(stop_after=0.02)
        self.assertEqual(run["process_exit"], "pass")
        self.assertLessEqual(run["stop_process_latency_s"], 1.5)
        self.assertEqual(run["engine_quiet_samples"]["status"], "pass")
        self.assertEqual(run["post_stop_latency"], "unqualified")
        self.assertEqual(
            run["post_stop_latency_reason"], "pulse_snapshot_age_unqualified"
        )

    def test_no_prelaunch_pcm_cannot_qualify_matched_audio(self):
        run = self.run_fixture(baseline=False)
        self.assertEqual(run["reason"], "baseline_unobserved")
        self.assertEqual(run["endpoint_loopback"], "unqualified")

    def test_missing_capture_completion_cannot_qualify_matched_audio(self):
        run = self.run_fixture(complete=False)
        self.assertEqual(run["endpoint_loopback"], "unqualified")
        self.assertEqual(run["reason"], "capture_monitor_ended_early")

    def test_short_prelaunch_coverage_cannot_qualify_later_content(self):
        run = self.run_fixture(baseline="one")
        self.assertEqual(run["reason"], "baseline_unobserved")
        self.assertEqual(run["baseline_monitor"]["frames"], 4800)

    def test_playing_stop_requires_content_match_and_measures_stopped_source(self):
        run = self.run_fixture(stop_after=0.015, padded=True)
        self.assertGreater(run["diagnostics"]["live_matcher_calls"], 0)
        self.assertGreater(run["playing_source_position_s"], 0)
        self.assertEqual(run["endpoint_loopback"], "pass")
        self.assertGreater(run["missing_tail_s"], 0.02)
        self.assertEqual(run["post_stop_latency"], "unqualified")

    def test_capture_ending_inside_marker_cannot_grade_natural_tail(self):
        row = self.run_fixture(clipped=True)
        self.assertEqual(row["seek"], "pass")
        self.assertEqual(row["tail_coverage"], "unqualified")
        self.assertNotIn("missing_tail_s", row)

    def test_capture_ending_inside_marker_cannot_grade_stopped_source(self):
        with patch("qualification.saved_bookmark_frame", return_value=4800):
            row = self.run_fixture(stop_after=0.015, clipped=True, padded=True)
        self.assertEqual(row["seek"], "pass")
        self.assertEqual(row["bookmark_alignment"], "unqualified")
        self.assertNotIn("missing_tail_s", row)
        self.assertNotIn("bookmark_end_error_s", row)

    def test_small_pulse_fragments_can_trigger_playing_stop(self):
        row = self.run_fixture(stop_after=0.015, padded=True, fragment_frames=256)
        self.assertIn("stop_process_latency_s", row)
        self.assertEqual(row["endpoint_loopback"], "pass")

    def test_moved_or_unavailable_player_route_censors_content_with_intact_monitor(
        self,
    ):
        for route in ("pulse_player_route_changed", "pulse_player_route_unavailable"):
            with self.subTest(route=route):
                row = self.run_fixture(playback_route=route)
                self.assertEqual(row["process_exit"], "pass")
                self.assertEqual(row["endpoint_loopback"], "unqualified")
                self.assertEqual(row["tail_coverage"], "unqualified")
                self.assertEqual(row["reason"], route)

    def test_unattributed_source_match_preserves_observed_and_censored_boundaries(self):
        route = "pulse_player_route_unavailable"
        for clipped in (False, True):
            with self.subTest(clipped=clipped):
                row = self.run_fixture(clipped=clipped, playback_route=route)
                self.assertEqual(row["reason"], route)
                self.assertEqual(
                    (row["endpoint_loopback"], row["seek"], row["tail_coverage"]),
                    ("unqualified",) * 3,
                )
                self.assertNotIn("measurement", row)
                self.assertNotIn("missing_tail_s", row)
                info = row["diagnostics"].get("unattributed_source_match", {})
                self.assertEqual(info.get("attribution"), "unattributed")
                self.assertEqual(
                    info["end_boundary"], "censored" if clipped else "observed"
                )
                self.assertAlmostEqual(info["start_source_s"], 0, delta=0.02)
                self.assertGreater(info["end_source_s"], 0.4)
                self.assertGreaterEqual(
                    min(info["early_score"], info["tail_score"]), 0.7
                )
                self.assertAlmostEqual(info["alignment_drift_s"], 0, delta=0.02)
        stopped = self.run_fixture(stop_after=0.015, padded=True, playback_route=route)
        self.assertEqual(stopped["endpoint_loopback"], "unqualified")
        self.assertNotIn(stopped.get("bookmark_alignment"), ("pass", "fail"))
        self.assertNotIn("bookmark_end_error_s", stopped)
        self.assertGreater(
            stopped["diagnostics"]["unattributed_source_match"]["end_source_s"], 0
        )
        incomplete = self.run_fixture(complete=False, playback_route=route)
        self.assertNotIn("unattributed_source_match", incomplete["diagnostics"])

    def test_native_cli_refuses_pulse_on_other_platforms_before_opening_player(self):
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "no-output"
            argv = [
                "qualification.py",
                "--output-dir",
                str(target),
                "--capture-loopback",
                "--capture-clock",
                "pulse-fragment",
                "--ffplay",
                "invalid",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch("qualification.platform.system", return_value="Windows"),
                patch.object(sys, "stderr", io.StringIO()),
                patch("builtins.print"),
                self.assertRaises(SystemExit) as refused,
            ):
                q.main()
            self.assertEqual(refused.exception.code, 2)
            self.assertFalse(target.exists())

    def test_padded_matrix_keeps_21_rows_with_three_censored_resumes(self):
        def capture(*args, capture_clock, **kwargs):
            self.assertEqual(capture_clock, "pulse-fragment")
            return dict(
                launched=True,
                process_exit="pass",
                player_reaped=True,
                capture_reaped=True,
                endpoint_loopback="pass",
                bookmark="pass",
                bookmark_alignment="pass",
                saved_bookmark_frame=24000,
                post_stop_silence="unqualified",
                post_stop_latency="unqualified",
                stop_process_latency_s=0.05,
            )

        with (
            tempfile.TemporaryDirectory() as scratch,
            patch("qualification.run_case", side_effect=capture),
        ):
            rows = q.run_matrix(
                Path("never"),
                Path(scratch),
                tail_padding_ms=250,
                repeat=3,
                capture_clock="pulse-fragment",
            )
        self.assertEqual(len(rows), 21)
        self.assertEqual(
            [row["case"] for row in rows if not row["launched"]], ["resume_saved"] * 3
        )
        self.assertNotIn("stop_startup", [row["case"] for row in rows])
        self.assertEqual(q.native_result(rows), "unqualified")


class CaptureStartupTests(unittest.TestCase):
    def test_wait_for_initial_activation_is_bounded_and_refuses_invalid_state(self):
        from pulse_fragments import wait_for_active

        pulse = SimpleNamespace(_lock_mainloop=nullcontext)
        api = Mock(PA_STREAM_READY=2)
        api.pa_stream_get_state.return_value = 2
        api.pa_stream_is_suspended.side_effect = [1, 0]
        wait_for_active(api, pulse, None)
        for state in (1, -1):
            api.pa_stream_is_suspended.side_effect = None
            api.pa_stream_is_suspended.return_value = state
            with (
                patch("pulse_fragments.time.monotonic", side_effect=[0, 2]),
                self.assertRaises(RuntimeError),
            ):
                wait_for_active(api, pulse, None)

    def test_native_operation_times_out_cancels_and_releases(self):
        from pulse_fragments import _operation

        ffi = FFI()
        api = Mock(PA_OPERATION_DONE=1, PA_OPERATION_RUNNING=0)
        api.pa_operation_get_state.return_value = 0
        operation = ffi.new("char *")
        with (
            patch("pulse_fragments.time.monotonic", side_effect=[0, 2]),
            self.assertRaisesRegex(RuntimeError, "timeout"),
        ):
            _operation(
                ffi, api, SimpleNamespace(_lock_mainloop=nullcontext), lambda: operation
            )
        api.pa_operation_cancel.assert_called_once_with(operation)
        api.pa_operation_unref.assert_called_once_with(operation)

    def test_finishing_drain_keeps_peeked_frames_and_negative_latency(self):
        from queue import Queue
        from pulse_fragments import _drain

        ffi = FFI()
        ffi.cdef(
            "typedef struct pa_stream pa_stream; typedef unsigned long long pa_usec_t; typedef void (*pa_stream_success_cb_t)(pa_stream *,int,void*);"
        )
        stream = ffi.cast("pa_stream *", 1)
        op = ffi.new("char *")
        api = Mock(
            PA_STREAM_READY=2,
            PA_SAMPLE_FLOAT32LE=5,
            PA_OPERATION_DONE=1,
            PA_OPERATION_RUNNING=0,
        )
        api.pa_stream_get_state.return_value = 2
        api.pa_stream_is_suspended.return_value = 0
        api.pa_stream_get_device_index.return_value = 8
        api.pa_stream_get_device_name.return_value = ffi.new("char[]", b"monitor")
        api.pa_stream_get_sample_spec.return_value = SimpleNamespace(
            format=5, rate=48000, channels=2
        )
        api.pa_stream_get_buffer_attr.return_value = SimpleNamespace(
            maxlength=4194304, fragsize=38400
        )
        api.pa_stream_readable_size.side_effect = [16, 16, 0]
        api.pa_operation_get_state.return_value = 1
        timing = SimpleNamespace(read_index=88, read_index_corrupt=False)
        api.pa_stream_get_timing_info.return_value = timing
        storage = ffi.new("float[]", [1, 2, 3, 4])

        def peek(stream, pointer, size):
            pointer[0] = storage
            size[0] = 16
            return 0

        def drop(stream):
            timing.read_index += 16
            return 0

        def latency(stream, usec, negative):
            usec[0] = 12000
            negative[0] = 1
            return 0

        def update(stream, callback, user):
            callback(stream, 1, user)
            return op

        api.pa_stream_peek.side_effect = peek
        api.pa_stream_drop.side_effect = drop
        api.pa_stream_get_latency.side_effect = latency
        api.pa_stream_update_timing_info.side_effect = update
        finish = threading.Event()
        finish.set()
        messages = Queue()
        _drain(
            ffi,
            api,
            SimpleNamespace(_lock_mainloop=nullcontext),
            stream,
            Route(b"private", 3, b"monitor", 8),
            1,
            4800,
            messages,
            finish,
        )
        ready = messages.get_nowait()
        self.assertEqual(ready[2]["maxlength_bytes"], 4194304)
        self.assertEqual(ready[2]["fragsize_bytes"], 38400)
        self.assertNotIn("private", json.dumps(ready[2]))
        message = messages.get_nowait()
        self.assertEqual(message[2].frame, 11)
        self.assertAlmostEqual(message[3]["signed_latency_min_s"], -0.012)


class PlayerRouteTests(unittest.TestCase):
    def test_unassigned_new_then_overlapping_expected_links_is_unavailable(self):
        from pulse_fragments import InputEvent, PlayerRouteHistory, NEW, CHANGE, REMOVE

        def history(sinks, overlap=None):
            result = PlayerRouteHistory()
            result.events = [
                InputEvent(
                    NEW if n == 0 else CHANGE, 77, b"1234", sink, True, n == overlap
                )
                for n, sink in enumerate(sinks)
            ]
            result.events.append(InputEvent(REMOVE, 77))
            return result.verdict(1234, 54)

        self.assertEqual(
            history([0xFFFFFFFF, 54, 54, 54], overlap=1),
            "pulse_player_route_unavailable",
        )
        self.assertEqual(history([0xFFFFFFFF, 54, 54]), "verified")
        self.assertEqual(history([0xFFFFFFFF]), "pulse_player_route_unavailable")
        self.assertEqual(
            history([0xFFFFFFFF, 62, 54], overlap=1), "pulse_player_route_changed"
        )
        self.assertNotEqual(history([54, 0xFFFFFFFF, 54]), "verified")

    def test_native_notifications_track_owned_input_through_removal(self):
        from pulse_fragments import watch_player_route

        class FFIStub:
            NULL = None
            callback = staticmethod(lambda signature: lambda function: function)
            string = staticmethod(lambda value: value)

        pulse = SimpleNamespace(context=object(), _lock_mainloop=nullcontext)
        for moved in (False, True):
            with self.subTest(moved=moved):
                observed = threading.Event()
                api = Mock(PA_OPERATION_DONE=1)
                api.pa_operation_get_state.return_value = 1
                api.pa_context_subscribe.side_effect = (
                    lambda context, mask, callback, user: (
                        callback(context, 1, user),
                        object(),
                    )[1]
                )
                info = SimpleNamespace(
                    index=77, sink=3, proplist={"application.process.id": b"1234"}
                )
                api.pa_proplist_gets.side_effect = lambda props, key: props.get(
                    key.decode()
                )

                def query(context, index, callback, user, info=info, observed=observed):
                    callback(context, info, 0, user)
                    callback(context, None, 1, user)
                    observed.set()
                    return object()

                api.pa_context_get_sink_input_info.side_effect = query
                with watch_player_route(FFIStub(), api, pulse) as history:
                    callback = api.pa_context_set_subscribe_callback.call_args.args[1]
                    callback(pulse.context, 2, 77, None)
                    self.assertTrue(observed.wait(1))
                    observed.clear()
                    if moved:
                        info.sink = 4
                    callback(pulse.context, 0x12, 77, None)
                    self.assertTrue(observed.wait(1))
                    callback(pulse.context, 0x22, 77, None)
                self.assertEqual(
                    history.verdict(1234, 3),
                    "pulse_player_route_changed" if moved else "verified",
                )
                self.assertEqual(
                    history.verdict(4321, 3), "pulse_player_route_unavailable"
                )
