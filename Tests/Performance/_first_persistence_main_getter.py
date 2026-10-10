"""Add one original public-getter START to the unchanged cold-read gate."""

import asyncio
import contextlib
import inspect
import threading

from Tests.UI.test_console_pending_facts_cold_readiness import _OriginalColdRead
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import raw_participants, storage_admission
from tldw_chatbook.Backup_Recovery import config_participants
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost
from types import FunctionType
from tldw_chatbook import emergency_stop
from tldw_chatbook.Widgets import compact_model_bar


class AttributedOriginalColdRead(_OriginalColdRead):
    def __init__(self, projection):
        super().__init__(projection)
        self.loop = asyncio.get_running_loop()
        self.main_actor = threading.current_thread()
        self.getter = inspect.getattr_static(config, "get_cli_providers_and_models")
        alias = inspect.getattr_static(
            compact_model_bar, "get_cli_providers_and_models"
        )
        assert alias is self.getter
        self.sync = inspect.getattr_static(
            compact_model_bar.CompactModelBar, "sync_from_sidebar"
        )
        for function in (self.getter, self.sync):
            self._pin(function)
        self.slots.extend(
            (
                (config, "get_cli_providers_and_models", self.getter),
                (compact_model_bar, "get_cli_providers_and_models", self.getter),
                (compact_model_bar.CompactModelBar, "sync_from_sidebar", self.sync),
            )
        )
        self.codes[self.getter.__code__] = "original_public_models_getter"
        operation = inspect.getattr_static(config_participants, "operation")
        self.operation_body = inspect.getattr_static(operation, "__wrapped__")
        self.operation_code = self.operation_body.__code__
        self.codes[self.operation_code] = "original_config_operation_START"
        self.snapshot = inspect.getattr_static(config, "get_runtime_config_snapshot")
        assert (
            inspect.getattr_static(controller_module, "get_runtime_config_snapshot")
            is self.snapshot
        )
        self.controller = self.screen._console_chat_controller
        assert type(self.controller) is ConsoleChatController
        self.host = vars(self.controller).get("_interrupt_host")
        assert type(self.host) is InterruptRoundHost
        self.registry = vars(self.host).get("registries")
        assert type(self.registry) is dict  # noqa: E721 - no custom registry dispatch.
        self.approvals = self.registry.get("approval")
        assert type(self.approvals) is dict  # noqa: E721 - no custom registry dispatch.
        self.snapshot_getter = vars(self.host).get(
            "read_global_get_runtime_config_snapshot"
        )
        assert type(
            self.snapshot_getter
        ) is FunctionType and self.snapshot_getter.__globals__ is vars(
            controller_module
        )
        self._pin(self.snapshot_getter)
        guarded = inspect.getattr_static(config, "_get_runtime_config_snapshot_guarded")
        guarded_body = inspect.getattr_static(guarded, "__wrapped__")
        self._pin(guarded)
        self._pin(guarded_body)
        self.slots.extend(
            (
                (config, "_get_runtime_config_snapshot_guarded", guarded),
                (guarded, "__wrapped__", guarded_body),
                (controller_module, "get_runtime_config_snapshot", self.snapshot),
                (
                    self.host,
                    "read_global_get_runtime_config_snapshot",
                    self.snapshot_getter,
                ),
            )
        )
        for owner, name in (
            (config, "get_runtime_config_snapshot"),
            (InterruptRoundHost, "_maybe_fire_permission_summary"),
            (InterruptRoundHost, "project_pending_decision_for_active_session"),
            (ConsoleChatController, "_maybe_fire_permission_summary"),
        ):
            function = inspect.getattr_static(owner, name)
            self._pin(function)
            self.slots.append((owner, name, function))
        self.host_summary = inspect.getattr_static(
            InterruptRoundHost, "_maybe_fire_permission_summary"
        )
        # Exact expected ancestor/callers; unknowns stay explicitly unqualified.
        for owner, name in (
            (contextlib._GeneratorContextManager, "__enter__"),
            (spend, "run_console_config_sync"),
            (config, "load_settings"),
        ):
            function = inspect.getattr_static(owner, name)
            self._pin(function)
            self.slots.append((owner, name, function))
        for owner, name in (
            (ConsoleChatController, "_global_context_policy_overrides"),
            (ConsoleChatController, "_compaction_admission_check"),
            (ConsoleChatController, "_apply_conversation_memory_preflight"),
            (emergency_stop, "default_emergency_stop_path"),
        ):
            function = inspect.getattr_static(owner, name)
            self._pin(function)
            self.slots.append((owner, name, function))
        self.caller_codes = {record[1]: record[2] for record in self.pins}
        # Keep original gate's complete module-file maps valid for new pins.
        self.sources = {record[1]: record[-1] for record in self.modules.values()}
        self.module_files = {
            name: record[0].__file__ for name, record in self.modules.items()
        }
        self.main_getter_rows = 0

    def _started(self, code, offset):
        if (
            not self.active
            or code not in (self.getter.__code__, self.operation_code)
            or not self.entered.is_set()
            or self.release.is_set()
        ):
            return
        if threading.current_thread() is not self.main_actor:
            return
        try:
            assert self._current()
            assert self.main_actor is threading.main_thread()
            assert asyncio.get_running_loop() is self.loop
            frame = self._frame(code)
            defining = (
                self.getter.__globals__
                if code is self.getter.__code__
                else self.operation_body.__globals__
            )
            assert frame.f_code is code and frame.f_globals is defining
            if code is self.operation_code:
                assert frame.f_locals.get("source") is config
            parent = frame.f_back
            chain = []
            cursor = parent
            summary = None
            for _ in range(6):
                if cursor is None:
                    break
                namespace = self.caller_codes.get(cursor.f_code)
                qualified = namespace is not None and cursor.f_globals is namespace
                if cursor.f_code is self.host_summary.__code__:
                    assert qualified and cursor.f_locals.get("self") is self.host
                    assert vars(self.controller).get("_interrupt_host") is self.host
                    assert (
                        vars(self.host).get("registries") is self.registry
                        and self.registry.get("approval") is self.approvals
                    )
                    round_id = cursor.f_locals.get("round_id")
                    assert type(round_id) is str and bool(round_id)  # noqa: E721 - no custom coercion.
                    state = self.approvals.get(round_id)
                    assert state is None or type(state) is dict  # noqa: E721 - no custom state dispatch.
                    summary = {
                        "exact_Host_Controller_registry": True,
                        "round_id_actor": id(round_id),
                        "state_present": state is not None,
                        "state_actor": None if state is None else id(state),
                        "summary_fired_literal_true": state is not None
                        and state.get("summary_fired") is True,
                    }
                chain.append(
                    {
                        "function": cursor.f_code.co_qualname,
                        "line": cursor.f_lineno,
                        "module": cursor.f_globals.get("__name__", ""),
                        "exact_selected_original": qualified,
                    }
                )
                cursor = cursor.f_back
            with storage_admission._lock:
                state = raw_participants._states.get(self.operation)
                assert state is not None and state.source is config
                assert len(state.leases) == len(self.leases) == 2
                assert all(
                    current is held for current, held in zip(state.leases, self.leases)
                )
                assert all(
                    lease in storage_admission._live_leases for lease in self.leases
                )
                assert (
                    state.thread is self.thread and self.thread is not self.main_actor
                )
            if self.main_getter_rows >= 8:
                if len(self.invalid) < 12:
                    self.invalid.append("bounded_main_getter_attribution_overflow")
                return
            self.main_getter_rows += 1
            self.rows.append(
                {
                    "site": "main_original_config_operation_START"
                    if code is self.operation_code
                    else "main_public_models_getter_START",
                    "caller_chain": chain,
                    "permission_summary_state": summary,
                    "selected_original_function": self.operation_body.__qualname__
                    if code is self.operation_code
                    else self.getter.__qualname__,
                    "current_source_bindings": True,
                    "same_current_loop_and_MainThread": True,
                    "held_original_worker_actor": True,
                    "held_raw_operation_live": True,
                    "held_raw_leases_live": 2,
                }
            )
            # No frame, caller, widget or source state is retained by a row.
            del state, cursor, parent, frame
        except Exception as error:
            if len(self.invalid) < 12:
                self.invalid.append("main_getter_START:" + type(error).__name__)

    def start(self):
        super().start()
        event = self.monitor.events.PY_START
        # Original stop owns both event registrations and their local masks.
        self.selected_events = (self.monitor.events.PY_RETURN, event)
        callback = self._started
        previous = self.monitor.register_callback(self.tool, event, callback)
        if previous is not None:
            self.monitor.register_callback(self.tool, event, previous)
            raise RuntimeError("main_getter_callback_not_unowned")
        self.registered[event] = callback
        mask = self.monitor.events.PY_RETURN | event
        for code in (self.getter.__code__, self.operation_code):
            self.monitor.set_local_events(self.tool, code, mask)
            self.enabled_codes.add(code)
            assert self.monitor.get_local_events(self.tool, code) == mask
        assert self.monitor.get_events(self.tool) == 0
