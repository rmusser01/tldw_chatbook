"""Opt-in bounded original-function wall spans for Console Send attribution."""

import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import CodeType

import pytest

from Tests.Performance.console_shared_composition_counter import CompositionCounter


class SendSpanObserver(CompositionCounter):
    """Reuse local-event custody; retain only names, timestamps and numeric IDs."""

    def __init__(self):
        from tldw_chatbook.Chat import console_chat_controller as controller_source
        from tldw_chatbook.Chat import console_agent_bridge as bridge_source
        from tldw_chatbook.app import TldwCli
        from tldw_chatbook.Agents import run_log as run_log_source
        from tldw_chatbook.Agents.agent_service import AgentService
        from tldw_chatbook.Backup_Recovery.admission_runtime import (
            RecoveryAdmissionGuard,
        )
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.Personal_Context.context_service import ProfileContextService
        from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
        from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
        from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
        from tldw_chatbook.UI.Console_Modules.hooks import ConsoleHooksController
        from tldw_chatbook.UI.Console_Modules.prompt_queue import (
            ConsolePromptQueueUIController,
        )
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        self.monitor = sys.monitoring
        self.tool = None
        self.active = False
        self.selected = {}
        self.bindings = []
        self.files = {}
        self.rows = []
        self.overflow = 0
        self.admission_code = inspect.unwrap(RecoveryAdmissionGuard.execution).__code__
        self.admission_starts = {}
        self.admission_entries = []
        self.callbacks = (self.on_start, self.on_return, self.on_yield)
        self.events = (
            self.monitor.events.PY_START,
            self.monitor.events.PY_RETURN,
            self.monitor.events.PY_YIELD,
        )
        self.mask = self.events[0] | self.events[1]
        targets = (
            (
                AgentService,
                (
                    "run_turn",
                    "_run_owned",
                    "_run_one",
                    "_append_run_lifecycle",
                    "_make_call_model",
                ),
            ),
            (AgentRunsDB, ("create_run", "insert_steps_at_indices")),
            (run_log_source, ("bind_scoped_log_writer",)),
            (RecoveryAdmissionGuard, ("execution",)),
            (bridge_source._StreamingModelAdapter, ("_chat_call_impl",)),
            (bridge_source.ConsoleAgentTraceRequestFactory, ("build",)),
            (bridge_source, ("console_run_budget",)),
            (TldwCli, ("get_personal_context_service",)),
            (controller_source, ("_compose_profile_tool_provider",)),
            (bridge_source.ConsoleAgentBridge, ("run_reply",)),
            (
                bridge_source,
                (
                    "build_console_first_request_plan",
                    "_console_first_request_runtime_context",
                ),
            ),
            (ProfileContextService, ("build_snapshot",)),
            (
                ChatScreen,
                (
                    "_send_console_message_from_visible_action",
                    "_dispatch_console_draft_send",
                ),
            ),
            (
                ConsoleHooksController,
                ("dispatch", "_prepare_in_worker", "_continue_in_worker"),
            ),
            (
                ConsolePromptQueueUIController,
                ("dispatch", "_capture_configuration_for_dispatch"),
            ),
            (ConsoleRuntime, ("prepare_hooks_v2",)),
            (
                ConsoleChatController,
                (
                    "resume_durable_postcommit",
                    "hook_admission_reason",
                    "_stream_assistant_response_inner",
                    "_compose_agent_request_providers",
                    "_capture_and_resolve_turn_execution_context",
                    "capture_turn_configuration_snapshot",
                    "_durable_context_snapshots",
                    "_build_durable_trace_request",
                    "_admit_capture_policy",
                    "_apply_conversation_memory_preflight",
                    "_personal_context_service",
                    "_personal_context_builder",
                    "_run_maintenance_agent_call",
                    "_run_agent_reply",
                    "_run_direct_provider_reply",
                ),
            ),
            (
                ConsoleChatStore,
                (
                    "reconcile_durable_turn_settings",
                    "reconcile_durable_turn_roleplay_context",
                    "commit_durable_turn",
                ),
            ),
            (ConsoleProviderGateway, ("prepare_chat_request", "stream_chat")),
        )
        effect_names = {
            "publish_identity_and_settings",
            "publish_owners",
            "clear_staged_input",
            "project_workspace",
            "queue_acknowledgement",
            "accepted_hook",
            "prompt_history",
            "publish_preparation",
            "enter_provider_dispatch",
            "transition_checkpoint",
        }
        nested_targets = {
            (AgentService, "_run_one"): {
                "observe_trace_step",
                "observe_context_assembled",
            },
            (AgentService, "_make_call_model"): {"call_model"},
            (bridge_source._StreamingModelAdapter, "_chat_call_impl"): {"_consume"},
        }
        for owner, names in targets:
            for name in names:
                descriptor = inspect.getattr_static(owner, name)
                function = inspect.unwrap(descriptor)
                module = sys.modules[function.__module__]
                path = Path(module.__file__).resolve()
                assert Path(function.__code__.co_filename).resolve() == path
                self.bindings.append(
                    (owner, name, descriptor, function, function.__code__, module)
                )
                self.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                code = function.__code__
                assert code not in self.selected
                self.selected[code] = owner.__name__ + "." + name
                nested_names = (
                    effect_names
                    if name == "resume_durable_postcommit"
                    else nested_targets.get((owner, name), set())
                )
                selected_names = set()
                for child in code.co_consts:
                    if type(child) is CodeType and child.co_name in nested_names:
                        assert child not in self.selected
                        selected_names.add(child.co_name)
                        prefix = (
                            "postcommit"
                            if name == "resume_durable_postcommit"
                            else self.selected[code]
                        )
                        self.selected[child] = prefix + "." + child.co_name
                assert selected_names == nested_names

    def start(self):
        super().start()
        try:
            self.monitor.set_local_events(
                self.tool, self.admission_code, self.mask | self.events[2]
            )
        except BaseException:
            self.monitor.set_local_events(self.tool, self.admission_code, self.mask)
            super().stop()
            raise

    def stop(self):
        assert self.monitor.get_local_events(self.tool, self.admission_code) == (
            self.mask | self.events[2]
        )
        self.monitor.set_local_events(self.tool, self.admission_code, self.mask)
        super().stop()

    def _record(self, kind, code):
        if len(self.rows) >= 4096:
            self.overflow += 1
            return
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        index = len(self.rows)
        self.rows.append(
            (
                kind,
                self.selected[code],
                time.perf_counter(),
                threading.get_ident(),
                id(task) if task is not None else 0,
            )
        )
        return index

    def on_start(self, code, offset):
        index = self._record("start", code)
        if code is self.admission_code and index is not None:
            frame = sys._getframe(1)
            assert frame.f_code is code
            self.admission_starts[id(frame)] = index

    def on_return(self, code, offset, value):
        self._record("return", code)
        if code is self.admission_code:
            frame = sys._getframe(1)
            assert frame.f_code is code
            self.admission_starts.pop(id(frame), None)

    def on_yield(self, code, offset, value):
        assert code is self.admission_code
        frame = sys._getframe(1)
        assert frame.f_code is code
        start = self.admission_starts.pop(id(frame), None)
        index = self._record("yield", code)
        if start is not None and index is not None:
            self.admission_entries.append(
                (start, index, self.rows[index][2] - self.rows[start][2])
            )

    def receipt(self):
        return dict(
            diagnostic_only=True,
            timing_acceptance=False,
            original_bindings_and_sources_current=self.bindings_current(),
            source_sha256=self.files,
            selected_code_count=len(self.selected),
            global_events=0,
            monitoring_retired=not self.active
            and self.monitor.get_tool(self.tool) is None,
            overflow=self.overflow,
            event_limit=4096,
            events=self.rows,
            yield_code_count=1,
            yield_code_label=self.selected[self.admission_code],
            admission_entry_pairs=self.admission_entries,
            admission_pair_fields=(
                "start_event_index",
                "first_yield_event_index",
                "entry_seconds",
            ),
            unfinished_admission_entries=len(self.admission_starts),
            no_frame_receiver_argument_or_return_objects_retained=True,
        )


@pytest.fixture(autouse=True)
def original_send_spans(request):
    requested = os.environ.get("TLDW_SEND_SPANS_RESULT")
    if not requested:
        yield
        return
    from Tests.private_profile import is_private_profile_child

    if not is_private_profile_child(request):
        yield
        return
    target = Path(requested).resolve()
    assert target.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve())
    observer = SendSpanObserver()
    observer.start()
    try:
        yield
    finally:
        observer.stop()
        target.write_text(json.dumps(observer.receipt(), indent=2), encoding="utf-8")
