"""Pure scalar/source controls; does not import or execute App."""

import asyncio
import gc
import importlib.util
import sys
import threading
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "startup_timing_witness_candidate", HERE / "console_startup_timing_witness.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ScalarControls(unittest.TestCase):
    def test_overlap_is_recorded_as_individual_intervals_not_added_wall(self):
        ledger = MODULE._ScalarSpans()
        ledger.event("start", (1, 10, 100), "notes", 1.0, False)
        ledger.event("start", (2, 11, 200), "media", 2.0, False)
        ledger.event("return", (1, 10, 100), "notes", 4.0, False)
        ledger.event("return", (2, 11, 200), "media", 5.0, False)
        rows = ledger.finish(6.0)["spans"]
        self.assertEqual(
            [(r["started"], r["finished"]) for r in rows], [(1.0, 4.0), (2.0, 5.0)]
        )
        self.assertFalse(
            any(
                "exclusive" in key or "total_worker_wall" in key
                for r in rows
                for key in r
            )
        )

    def test_async_active_segments_exclude_suspend_but_wall_is_inclusive(self):
        ledger = MODULE._ScalarSpans()
        key = (1, 10, 100)
        for event, time in (
            ("start", 1.0),
            ("yield", 2.0),
            ("resume", 8.0),
            ("return", 9.0),
        ):
            ledger.event(event, key, "push", time, True)
        receipt = ledger.finish(10.0)
        self.assertEqual(len(receipt["spans"]), 1)
        row = receipt["spans"][0]
        self.assertEqual(row["inclusive_elapsed"], 8.0)
        self.assertEqual(row["active_elapsed"], 2.0)
        self.assertEqual(row["segments"], [[1.0, 2.0], [8.0, 9.0]])

    def test_recursion_has_distinct_frames_and_reuse_never_finishes_old_span(self):
        ledger = MODULE._ScalarSpans()
        a, b = (1, 10, 100), (1, 10, 101)
        ledger.event("start", a, "recurse", 1.0, False)
        ledger.event("start", b, "recurse", 2.0, False)
        ledger.event("return", b, "recurse", 3.0, False)
        ledger.event("start", a, "recurse", 4.0, False)
        ledger.event("return", a, "recurse", 5.0, False)
        receipt = ledger.finish(6.0)
        self.assertEqual(len(receipt["spans"]), 2)
        self.assertEqual(receipt["gaps"][0]["reason"], "frame_id_reused_without_return")
        self.assertFalse(receipt["complete"])

    def test_exception_and_unknown_resume_are_explicit_gaps(self):
        ledger = MODULE._ScalarSpans()
        ledger.event("start", (1, 10, 100), "raises", 1.0, False)
        ledger.event("resume", (1, 11, 200), "unknown", 2.0, True)
        receipt = ledger.finish(3.0)
        self.assertEqual(receipt["spans"], [])
        self.assertEqual(
            {r["reason"] for r in receipt["gaps"]},
            {"resume_without_start", "no_observed_normal_return"},
        )

    def test_initializer_summary_uses_union_and_last_completion_not_sum(self):
        rows = [{"label": "__init__", "started": 0.0, "finished": 10.0}]
        rows += [
            {"label": label, "started": begin, "finished": end}
            for label, begin, end in zip(
                sorted(MODULE._INITIALIZERS), (1.0, 2.0, 3.0, 6.0), (4.0, 5.0, 4.0, 7.0)
            )
        ]
        regions = [
            {"label": "constructor_parallel_join", "started": 2.0, "finished": 8.0}
        ]
        result = MODULE._initializer_summary(rows, regions)
        self.assertEqual(result["last_observed_normal_completion"], 7.0)
        self.assertEqual(
            result["worker_body_union_intervals"], [[1.0, 5.0], [6.0, 7.0]]
        )
        self.assertEqual(
            result["constructor_intervals_without_selected_worker_body"],
            [[0.0, 1.0], [5.0, 6.0], [7.0, 10.0]],
        )
        self.assertIsNone(MODULE._initializer_summary(rows[:-1], regions))


class TimedMixin:
    def _timed_init_task(self, function, *args):
        return function(*args)


class Tiny(TimedMixin):
    def __init__(self, payload=None):
        pass

    def _init_notes_service(self, depth=0, payload=None):
        if depth:
            return self._timed_init_task(self._init_notes_service, depth - 1, payload)
        return payload

    def _init_media_db(self, entered=None, release=None):
        if entered is not None:
            entered.set()
            if not release.wait(2):
                raise TimeoutError("bounded original tiny callback")

    def _init_providers_models(self):
        pass

    def _init_prompts_service(self):
        raise ValueError("original tiny failure")

    def on_mount(self):
        pass

    async def _push_initial_screen(self, payload=None):
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return payload

    async def _post_mount_setup(self):
        await asyncio.sleep(0)


class Payload:
    pass


class ActualLocalControls(unittest.TestCase):
    def witness(self):
        witness = MODULE.StartupTimingWitness(HERE)
        witness._specs = tuple(
            (__name__, "Tiny", name, Path(__file__).name, name)
            for name in MODULE._MEMBERS
        )
        witness._caller_spec = (
            __name__,
            "TimedMixin",
            "_timed_init_task",
            Path(__file__).name,
        )
        witness._regions = ()
        return witness

    def test_actual_sync_async_recursion_and_no_receiver_payload_retention(self):
        witness = self.witness()
        receiver = Tiny()
        payload = Payload()
        receiver_ref, payload_ref = weakref.ref(receiver), weakref.ref(payload)
        witness.start()
        try:
            Tiny()
            self.assertIs(
                receiver._timed_init_task(receiver._init_notes_service, 2, payload),
                payload,
            )
            self.assertIs(asyncio.run(receiver._push_initial_screen(payload)), payload)
            asyncio.run(receiver._post_mount_setup())
            receiver.on_mount()
        finally:
            receipt = witness.stop()
        self.assertTrue(receipt["monitoring_tool_freed"])
        self.assertTrue(receipt["monitoring_masks_cleared_while_active"])
        self.assertTrue(receipt["monitoring_global_zero"])
        self.assertFalse(receipt["budget_acceptance_eligible"])
        self.assertIsNone(receipt["budgets_pass"])
        self.assertEqual(
            sum(row["label"] == "_init_notes_service" for row in receipt["spans"]), 3
        )
        push = next(
            row for row in receipt["spans"] if row["label"] == "_push_initial_screen"
        )
        self.assertGreaterEqual(len(push["segments"]), 3)
        del receiver, payload
        gc.collect()
        self.assertIsNone(receiver_ref())
        self.assertIsNone(payload_ref())

    def test_two_actual_threads_overlap_and_exception_is_incomplete(self):
        witness = self.witness()
        receiver = Tiny()
        entered = [threading.Event(), threading.Event()]
        release = threading.Event()
        workers = [
            threading.Thread(
                target=receiver._timed_init_task,
                args=(receiver._init_media_db, entered[i], release),
            )
            for i in range(2)
        ]
        witness.start()
        try:
            Tiny()
            for worker in workers:
                worker.start()
            self.assertTrue(all(event.wait(1) for event in entered))
            release.set()
            for worker in workers:
                worker.join(2)
            with self.assertRaises(ValueError):
                receiver._timed_init_task(receiver._init_prompts_service)
        finally:
            release.set()
            for worker in workers:
                if worker.ident is not None:
                    worker.join(2)
            receipt = witness.stop()
        rows = [row for row in receipt["spans"] if row["label"] == "_init_media_db"]
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["thread"], rows[1]["thread"])
        self.assertLess(
            max(row["started"] for row in rows), min(row["finished"] for row in rows)
        )
        self.assertIn("no_observed_normal_return", receipt["gap_counts"])
        self.assertFalse(receipt["complete"])
        self.assertTrue(receipt["monitoring_tool_freed"])

    def test_class_slot_drift_invalidates_but_retires_owned_observer(self):
        witness = self.witness()
        witness.start()
        original = globals()["Tiny"]
        try:
            globals()["Tiny"] = type("Foreign", (), {})
            receipt = witness.stop()
        finally:
            globals()["Tiny"] = original
        self.assertFalse(receipt["source_current"])
        self.assertFalse(receipt["complete"])
        self.assertTrue(receipt["monitoring_tool_freed"])

    def test_source_read_failure_during_stop_does_not_skip_retirement(self):
        witness = self.witness()
        witness.start()
        with patch.object(
            Path, "read_bytes", side_effect=OSError("controlled source read failure")
        ):
            receipt = witness.stop()
        self.assertFalse(receipt["source_current"])
        self.assertTrue(receipt["monitoring_tool_freed"])
        self.assertTrue(receipt["monitoring_masks_cleared_while_active"])

    def test_controlled_source_bytes_drift_is_refused_even_if_code_is_same(self):
        witness = self.witness()
        witness.start()
        original_read = Path.read_bytes

        def drifted_read(path):
            return original_read(path) + b"\n# controlled source-byte drift\n"

        with patch.object(Path, "read_bytes", new=drifted_read):
            receipt = witness.stop()
        self.assertFalse(receipt["source_current"])
        self.assertFalse(receipt["complete"])
        self.assertTrue(receipt["monitoring_tool_freed"])

    def test_occupied_tools_and_debugger_are_untouched(self):
        monitor = sys.monitoring
        owned = []
        try:
            for number in (5, 4, 3, monitor.DEBUGGER_ID):
                if monitor.get_tool(number) is None:
                    monitor.use_tool_id(number, "foreign-control-" + str(number))
                    owned.append(number)
            before = {
                number: monitor.get_tool(number)
                for number in (5, 4, 3, monitor.DEBUGGER_ID)
            }
            with self.assertRaisesRegex(ValueError, "occupied"):
                self.witness().start()
            self.assertEqual(
                before, {number: monitor.get_tool(number) for number in before}
            )
        finally:
            for number in owned:
                monitor.free_tool_id(number)

    def test_foreign_callback_is_preserved_without_claiming_retirement(self):
        witness = self.witness()
        witness.start()
        number = witness.tool

        def foreign(*args):
            pass

        monitor = sys.monitoring
        monitor.register_callback(number, monitor.events.PY_RETURN, foreign)
        try:
            receipt = witness.stop()
            self.assertFalse(receipt["monitoring_callbacks_owned"])
            self.assertFalse(receipt["monitoring_tool_freed"])
            self.assertFalse(receipt["complete"])
            self.assertIs(
                monitor.register_callback(number, monitor.events.PY_RETURN, None),
                foreign,
            )
        finally:
            for code in witness.codes:
                monitor.set_local_events(number, code, 0)
            for event in witness.callbacks:
                monitor.register_callback(number, event, None)
            monitor.free_tool_id(number)

    def test_wrong_canonical_module_path_is_refused_before_install(self):
        original = globals()["__file__"]
        witness = self.witness()
        try:
            globals()["__file__"] = str(HERE / "foreign.py")
            with self.assertRaisesRegex(ValueError, "module_source"):
                witness.start()
        finally:
            globals()["__file__"] = original
            if witness.active:
                witness.stop()

    def test_changed_spec_loader_is_refused_at_retirement(self):
        from importlib.machinery import ModuleSpec

        original = globals().get("__spec__")
        loader = globals()["__loader__"]
        spec = ModuleSpec(__name__, loader, origin=__file__)
        globals()["__spec__"] = spec
        witness = self.witness()
        try:
            witness.start()
            spec.loader = object()
            receipt = witness.stop()
        finally:
            globals()["__spec__"] = original
        self.assertFalse(receipt["source_current"])
        self.assertTrue(receipt["monitoring_tool_freed"])

    def test_partial_local_install_failure_retires_actual_installed_mask(self):
        witness = self.witness()
        monitor = sys.monitoring

        class PartialInstall:
            def __getattr__(self, name):
                return getattr(monitor, name)

            def set_local_events(self, number, code, mask):
                monitor.set_local_events(number, code, mask)
                if mask:
                    raise RuntimeError("controlled post-install failure")

        witness.monitor = PartialInstall()
        try:
            with self.assertRaisesRegex(RuntimeError, "post-install"):
                witness.start()
            self.assertTrue(
                all(
                    monitor.get_local_events(witness.tool, code) == 0
                    for code in witness.codes
                )
            )
            self.assertIsNone(monitor.get_tool(witness.tool))
        finally:
            # Retire only this control's exact issued tool and masks on RED.
            if monitor.get_tool(witness.tool) is None:
                monitor.use_tool_id(witness.tool, witness.tool_name)
            if monitor.get_tool(witness.tool) == witness.tool_name:
                for code in witness.codes:
                    monitor.set_local_events(witness.tool, code, 0)
                for event in witness.callbacks:
                    monitor.register_callback(witness.tool, event, None)
                monitor.free_tool_id(witness.tool)

    def test_partial_callback_install_retires_only_actual_owned_callback(self):
        monitor = sys.monitoring
        for after_call in (False, True):
            with self.subTest(after_call=after_call):
                witness = self.witness()

                class PartialCallback:
                    failed = False

                    def __getattr__(self, name):
                        return getattr(monitor, name)

                    def register_callback(self, number, event, callback):
                        if callback is not None and not self.failed:
                            self.failed = True
                            if after_call:
                                monitor.register_callback(number, event, callback)
                            raise RuntimeError("controlled callback install failure")
                        return monitor.register_callback(number, event, callback)

                witness.monitor = PartialCallback()
                try:
                    with self.assertRaisesRegex(RuntimeError, "callback install"):
                        witness.start()
                    # The actual C API retains callbacks after freeing a tool.
                    for event in witness.callbacks:
                        self.assertIsNone(
                            monitor.register_callback(witness.tool, event, None)
                        )
                    self.assertIsNone(monitor.get_tool(witness.tool))
                finally:
                    if monitor.get_tool(witness.tool) is None:
                        monitor.use_tool_id(witness.tool, witness.tool_name)
                    if monitor.get_tool(witness.tool) == witness.tool_name:
                        for event in witness.callbacks:
                            monitor.register_callback(witness.tool, event, None)
                        monitor.free_tool_id(witness.tool)

    def test_code_namespace_and_callback_drift_do_not_qualify(self):
        from types import FunctionType

        original = Tiny.on_mount
        for replacement in (
            Tiny._init_providers_models,
            FunctionType(original.__code__, {}),
        ):
            try:
                Tiny.on_mount = replacement
                with self.assertRaises(ValueError):
                    self.witness().start()
            finally:
                Tiny.on_mount = original
        witness = self.witness()
        witness.start()
        original_record = MODULE.StartupTimingWitness._record
        try:
            MODULE.StartupTimingWitness._record = lambda *args: None
            receipt = witness.stop()
        finally:
            MODULE.StartupTimingWitness._record = original_record
        self.assertFalse(receipt["source_current"])
        self.assertTrue(receipt["monitoring_tool_freed"])

    def test_callback_replacement_before_start_cannot_borrow_observer_slot(self):
        witness = self.witness()
        original = MODULE.StartupTimingWitness._return_event
        MODULE.StartupTimingWitness._return_event = Tiny.on_mount
        try:
            with self.assertRaises(ValueError):
                witness.start()
        finally:
            MODULE.StartupTimingWitness._return_event = original
            if witness.active:
                witness.stop()

    def test_custom_initializer_caller_does_not_borrow_stock_label(self):
        witness = self.witness()
        receiver = Tiny()
        witness.start()
        try:
            receiver._init_notes_service()
        finally:
            receipt = witness.stop()
        self.assertFalse(
            any(row["label"] == "_init_notes_service" for row in receipt["spans"])
        )
        self.assertIn("initializer_caller_not_original", receipt["issues"])
        self.assertFalse(receipt["complete"])

    def test_missing_bodies_and_overflow_never_become_complete(self):
        witness = self.witness()
        witness.start()
        try:
            Tiny()
        finally:
            receipt = witness.stop()
        self.assertEqual(len(receipt["missing_bodies"]), 7)
        self.assertFalse(receipt["complete"])
        ledger = MODULE._ScalarSpans()
        for number in range(70):
            ledger.event("start", (0, 10, number), "bounded", 1.0, False)
        self.assertEqual(len(ledger.active), 64)
        for number in range(3000):
            ledger.gap("bounded", "controlled_gap")
        self.assertEqual(len(ledger.gaps), 32)
        self.assertFalse(ledger.finish(2.0)["complete"])

    def test_actual_inherited_caller_override_is_refused_and_drift_retires(self):
        def replacement(*args):
            pass

        Tiny._timed_init_task = replacement
        try:
            with self.assertRaisesRegex(ValueError, "inherited_initializer"):
                self.witness().start()
        finally:
            del Tiny._timed_init_task
        witness = self.witness()
        witness.start()
        Tiny._timed_init_task = replacement
        try:
            receipt = witness.stop()
        finally:
            del Tiny._timed_init_task
        self.assertFalse(receipt["source_current"])
        self.assertTrue(receipt["monitoring_tool_freed"])

    def test_changed_local_mask_is_preserved_and_not_complete(self):
        witness = self.witness()
        witness.start()
        monitor = sys.monitoring
        code = Tiny.on_mount.__code__
        foreign_mask = monitor.events.PY_START
        monitor.set_local_events(witness.tool, code, foreign_mask)
        try:
            receipt = witness.stop()
            self.assertFalse(receipt["monitoring_masks_owned"])
            self.assertFalse(receipt["complete"])
            self.assertEqual(monitor.get_local_events(witness.tool, code), foreign_mask)
        finally:
            for selected in witness.codes:
                monitor.set_local_events(witness.tool, selected, 0)
            for event in witness.callbacks:
                monitor.register_callback(witness.tool, event, None)
            monitor.free_tool_id(witness.tool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
