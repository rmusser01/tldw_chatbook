"""Code-local controls for one actual issued CompactModelBar callback."""

import asyncio
import inspect
import threading
import time

from Tests.Performance._compact_original_config_gate import OriginalCompactModelReadGate


class CompactSetupControlGate(OriginalCompactModelReadGate):
    def __init__(self, widget, loop, selected_path, action, *, queued=False):
        super().__init__(widget, loop, selected_path, "compose")
        self.action = action
        self.queued = queued
        self.action_task = None
        self.mount_task = None
        self.entry_request = None
        self.reader_started = 0
        self.actual_entry_facts = None
        self.populate_started = self.host_sync_started = 0
        self.populate_code = self.sync_code = None

    async def _act(self):
        try:
            await self.action(self)
        except BaseException as error:
            self.invalid.append("compact_control:" + type(error).__name__)
        finally:
            self.progress.set()

    def _probe(self):
        self.loop_probe_called_at = time.monotonic()
        self.loop_probe_called_after_release = self.release.is_set()
        if not self.release.is_set():
            self.action_task = asyncio.create_task(self._act())

    def _start(self, code, offset):
        if self.active and code in (self.populate_code, self.sync_code):
            frame = self._frame(code)
            target = (
                frame.f_locals.get("self")
                if code is self.populate_code
                else frame.f_locals.get("widget")
            )
            if target is self.widget:
                assert frame.f_globals is self.worker_namespace
                if code is self.populate_code:
                    self.populate_started += 1
                else:
                    self.host_sync_started += 1
        if self.active and code is self.mount_code:
            frame = self._frame(code)
            if frame.f_locals.get("self") is self.widget:
                assert frame.f_globals is self.mount_namespace
                assert threading.current_thread() is self.main_thread
                assert asyncio.get_running_loop() is self.loop
                self.mount_task = asyncio.current_task()
        if self.active and code is self.reader_code:
            frame = self._frame(code)
            if self._worker_ancestry(frame) is not None:
                self.reader_started += 1
        if (
            self.active
            and self.queued
            and code is self.worker_code
            and self.entry_request is None
        ):
            try:
                frame = self._frame(code)
                assert frame.f_globals is self.worker_namespace
                request = frame.f_locals.get("request")
                if (
                    type(request) is not self.setup_type
                    or request.widget is not self.widget
                ):
                    return
                assert request.app is self.widget.app_instance
                assert request.loop is self.loop and request.thread is self.main_thread
                assert request.reader is self.config.get_cli_providers_and_models
                assert threading.current_thread() is not self.main_thread
                self._qualify_issued_read(request, frame)
                self._edit_before_reader(request)
                self.entry_request = request
                self.actual_entry_facts = {
                    "exact_issued_Widget_App_reader_Task_and_private_producer": True,
                    "original_callback_before_first_body_instruction": True,
                    "actual_worker_Thread": True,
                }
                self.entered.set()
                assert self.release.wait(10)
            except BaseException as error:
                self.invalid.append("compact_entry_control:" + type(error).__name__)
                self.entered.set()
                self.release.set()
            return
        super()._start(code, offset)

    def install(self):
        super().install()
        from textual.app import App
        from tldw_chatbook.Widgets import compact_model_bar

        assert self.worker_code is not None
        prune = inspect.getattr_static(App, "_prune")
        self._pin(prune)
        self.slots.append((App, "_prune", prune))
        self.populate_code = self._pin(
            inspect.getattr_static(
                compact_model_bar.CompactModelBar, "_populate_initial_defaults"
            )
        )
        self.sync_code = self._pin(compact_model_bar._request_default_sync)
        self.slots.append(
            (
                compact_model_bar,
                "_request_default_sync",
                compact_model_bar._request_default_sync,
            )
        )
        self.slots.append(
            (
                compact_model_bar.CompactModelBar,
                "_populate_initial_defaults",
                compact_model_bar.CompactModelBar._populate_initial_defaults,
            )
        )
        for code in (self.populate_code, self.sync_code):
            self.codes[code] = "actual_default_publication_or_host_sync"
            self.monitor.set_local_events(
                self.tool,
                code,
                self.monitor.events.PY_START | self.monitor.events.PY_RETURN,
            )
        self.codes[self.reader_code] = "selected_original_public_model_reader"
        self.monitor.set_local_events(
            self.tool,
            self.reader_code,
            self.monitor.events.PY_START | self.monitor.events.PY_RETURN,
        )

    def close(self):
        result = super().close()
        result.update(
            actual_selected_public_reader_START_count=self.reader_started,
            queued_original_worker_entry_facts=self.actual_entry_facts,
            control_UI_Task_retired=self.action_task is None or self.action_task.done(),
            captured_mount_Task_retired=self.mount_task is None
            or self.mount_task.done(),
            queued_callback_Task_retired=self._physically_retired(self.entry_request),
            actual_target_default_population_START_count=self.populate_started,
            actual_target_host_sync_START_count=self.host_sync_started,
            frames_arguments_results_tasks_retained=True,
            exact_known_actor_Tasks_retained_only_for_finite_custody=True,
            arbitrary_frames_arguments_results_collected=False,
        )
        return result
