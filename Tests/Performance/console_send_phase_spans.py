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
        self.entry_starts = {self.admission_code: self.admission_starts}
        self.entry_pairs = {self.admission_code: self.admission_entries}
        self.callbacks = (self.on_start, self.on_return, self.on_yield)
        self.events = (
            self.monitor.events.PY_START,
            self.monitor.events.PY_RETURN,
            self.monitor.events.PY_YIELD,
        )
        self.mask = self.events[0] | self.events[1]
        self.preparation_detail = os.environ.get("TLDW_SEND_PREPARATION_DETAIL") == "1"
        self.hook_current_code = self.raw_check_code = self.receive_code = None
        self.sensitive_bundle_code = None
        self.source_prefix = Path(__file__).resolve().parents[2].as_posix() + "/"
        self.hook_entries = []
        self.hook_active = {}
        self.raw_active = {}
        self.detail_overflow = self.raw_unmatched = self.hook_unmatched = 0
        self.ancestor_overflow = 0
        self.receive_returns = []
        self.sensitive_bundle_returns = []
        self.context_snapshot_code = self.context_versions_code = None
        self.context_purpose_codes = {}
        self.context_reads = []
        self.context_active = {}
        self.context_overflow = self.context_unmatched = 0
        self.context_ancestor_overflow = 0
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
        if self.preparation_detail:
            from tldw_chatbook import config
            from tldw_chatbook.Agents import hook_permissions
            from tldw_chatbook.Backup_Recovery import (
                config_participants,
                raw_participants,
            )
            from tldw_chatbook.Chat import (
                console_configuration_capture,
                console_configuration_preparation,
            )
            from tldw_chatbook.Chat.chat_persistence_service import (
                ChatPersistenceService,
            )
            from tldw_chatbook.Chat.console_context_compaction import (
                ConsoleCompactionPreflight,
            )
            from tldw_chatbook.MCP import console_snapshot, console_tool_preparation
            from tldw_chatbook.MCP.unified_control_plane_service import (
                UnifiedMCPControlPlaneService,
            )
            from tldw_chatbook.UI.Console_Modules import wiring
            from tldw_chatbook.Utils import private_paths, sensitive_paths

            targets += (
                (hook_permissions.HookPermissions, ("_current", "_read_state")),
                (hook_permissions, ("default_hook_permissions_path",)),
                (config, ("get_user_data_dir", "locked_hooks_config_snapshot")),
                (config_participants, ("verified_user_data_directory",)),
                (raw_participants, ("_check", "_scope")),
                (wiring, ("receive_console_visible_intent",)),
                (sensitive_paths, ("_stock_sensitive_config_bundle",)),
                (private_paths, ("create_private_text",)),
                (ChatPersistenceService, ("get_message_versions",)),
                (
                    console_configuration_preparation,
                    ("capture_console_turn_configuration_owned",),
                ),
                (
                    console_snapshot,
                    ("capture_console_definition_maximum", "_checked_read"),
                ),
                (
                    console_snapshot._CapturedSources,
                    ("permission_call", "read_permission_payload"),
                ),
                (console_tool_preparation, ("prepare_console_tools",)),
                (UnifiedMCPControlPlaneService, ("local_external_catalog",)),
                (ConsoleChatController, ("_compose_local_provider",)),
                (
                    console_configuration_capture,
                    ("capture_console_turn_configuration",),
                ),
            )
            for generator in (
                raw_participants._scope,
                config.locked_hooks_config_snapshot,
            ):
                code = inspect.unwrap(generator).__code__
                self.entry_starts[code] = {}
                self.entry_pairs[code] = []
            self.hook_current_code = inspect.unwrap(
                hook_permissions.HookPermissions._current
            ).__code__
            self.raw_check_code = inspect.unwrap(raw_participants._check).__code__
            self.receive_code = inspect.unwrap(
                wiring.receive_console_visible_intent
            ).__code__
            self.sensitive_bundle_code = inspect.unwrap(
                sensitive_paths._stock_sensitive_config_bundle
            ).__code__
            self.context_snapshot_code = inspect.unwrap(
                ConsoleChatController._durable_context_snapshots
            ).__code__
            self.context_versions_code = inspect.unwrap(
                ChatPersistenceService.get_message_versions
            ).__code__
            # These are original-code ancestry anchors, not extra event targets.
            purpose_sources = (
                (
                    ConsoleChatController,
                    "context_control_presentation_inputs",
                    "read",
                    "presentation",
                ),
                (
                    ConsoleChatController,
                    "_compaction_admission_check",
                    None,
                    "preaccept_assessment",
                ),
                (
                    ConsoleChatController,
                    "_stream_assistant_response_inner",
                    None,
                    "dispatch_preflight",
                ),
                (
                    ConsoleCompactionPreflight,
                    "compact_context_now",
                    None,
                    "manual_or_micro_compaction",
                ),
                (ConsoleChatController, "build_context_snapshot", None, "preview"),
                (
                    ConsoleChatController,
                    "_manual_summary_planning",
                    None,
                    "manual_summary",
                ),
                (
                    ConsoleChatController,
                    "_compaction_admission",
                    None,
                    "compaction_revalidation",
                ),
                (
                    ConsoleChatController,
                    "_hooks_for_compaction",
                    "memory_current",
                    "compaction_memory_revalidation",
                ),
                (
                    ConsoleChatController,
                    "_summarize_manual",
                    "current_admission",
                    "manual_summary_revalidation",
                ),
            )
            for owner, name, child_name, purpose in purpose_sources:
                descriptor = inspect.getattr_static(owner, name)
                function = inspect.unwrap(descriptor)
                module = sys.modules[function.__module__]
                path = Path(module.__file__).resolve()
                assert function.__globals__ is vars(module)
                assert Path(function.__code__.co_filename).resolve() == path
                self.bindings.append(
                    (owner, name, descriptor, function, function.__code__, module)
                )
                self.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                code = function.__code__
                if child_name is not None:
                    children = [
                        child
                        for child in code.co_consts
                        if type(child) is CodeType and child.co_name == child_name
                    ]
                    assert len(children) == 1
                    code = children[0]
                self.context_purpose_codes[code] = purpose
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
        if self.preparation_detail:
            nested_targets.update(
                {
                    (
                        console_configuration_preparation,
                        "capture_console_turn_configuration_owned",
                    ): {"capture"},
                    (console_snapshot, "capture_console_definition_maximum"): {
                        "read_sources"
                    },
                }
            )
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
            for code in self.entry_starts:
                self.monitor.set_local_events(
                    self.tool, code, self.mask | self.events[2]
                )
        except BaseException:
            for code in self.entry_starts:
                self.monitor.set_local_events(self.tool, code, self.mask)
            super().stop()
            raise

    def stop(self):
        for code in self.entry_starts:
            assert self.monitor.get_local_events(self.tool, code) == (
                self.mask | self.events[2]
            )
            self.monitor.set_local_events(self.tool, code, self.mask)
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

    def _hook_ancestor(self, frame):
        for _ in range(64):
            frame = frame.f_back
            if frame is None:
                return None
            if frame.f_code is self.hook_current_code:
                return id(frame), threading.get_ident()
        self.ancestor_overflow += 1
        return None

    def _raw_caller(self, frame):
        caller = frame.f_back
        if caller is None:
            return "<unknown>", "<unknown>", 0
        filename = caller.f_code.co_filename.replace("\\", "/")
        filename = (
            filename[len(self.source_prefix) :]
            if filename.casefold().startswith(self.source_prefix.casefold())
            else "<outside-repository>"
        )
        return filename, caller.f_code.co_qualname, caller.f_lineno

    def _context_ancestry(self, frame):
        for _ in range(64):
            frame = frame.f_back
            if frame is None:
                return "other", None
            if frame.f_code is self.context_snapshot_code:
                parent = self.context_active.get((id(frame), threading.get_ident()))
                if parent is not None:
                    return self.context_reads[parent]["purpose"], parent
            purpose = self.context_purpose_codes.get(frame.f_code)
            if purpose is not None:
                if purpose == "preaccept_assessment":
                    purpose = (
                        "preaccept_changed_request"
                        if frame.f_locals.get("after_side_effects") is True
                        else "preaccept_early"
                    )
                elif purpose == "manual_or_micro_compaction":
                    purpose = (
                        "micro_compaction"
                        if frame.f_locals.get("micro") is True
                        else "manual_compaction"
                    )
                return purpose, None
        self.context_ancestor_overflow += 1
        return "other", None

    def _context_start(self, frame, index):
        key = id(frame), threading.get_ident()
        if key in self.context_active:
            self.context_unmatched += 1
            self.context_active.pop(key)
        if len(self.context_active) >= 128 or len(self.context_reads) >= 512:
            self.context_overflow += 1
            return
        purpose, parent = self._context_ancestry(frame)
        count = None
        if frame.f_code is self.context_versions_code:
            ids = frame.f_locals.get("message_ids")
            # Do not invoke arbitrary __len__, consume iterators or copy IDs.
            if type(ids) in (list, tuple):
                count = len(ids)
            if parent is not None:
                self.context_reads[parent]["requested_id_count"] = count
        filename, caller, line = self._raw_caller(frame)
        self.context_active[key] = len(self.context_reads)
        self.context_reads.append(
            dict(
                frame_id=key[0],
                start_event_index=index,
                return_event_index=None,
                purpose=purpose,
                source_file=filename,
                caller_qualname=caller,
                source_line=line,
                snapshot_start_event_index=(
                    self.context_reads[parent]["start_event_index"]
                    if parent is not None
                    else None
                ),
                requested_id_count=count,
            )
        )

    def on_start(self, code, offset):
        if code is self.raw_check_code:
            frame = sys._getframe(1)
            assert frame.f_code is code
            current = self._hook_ancestor(frame)
            entry = self.hook_active.get(current)
            if entry is None:
                return
            key = id(frame), threading.get_ident()
            if key in self.raw_active:
                self.raw_unmatched += 1
                self.raw_active.pop(key)
            if len(self.raw_active) >= 128:
                self.detail_overflow += 1
                return
            summary = self.hook_entries[entry]
            caller = self._raw_caller(frame)
            callers = summary["raw_callers"]
            if caller not in callers:
                if len(callers) >= 64:
                    summary["raw_caller_overflow"] += 1
                    caller = None
                else:
                    callers[caller] = dict(started=0, returned=0, inclusive_seconds=0.0)
            if caller is not None:
                callers[caller]["started"] += 1
            self.raw_active[key] = entry, time.perf_counter(), caller
            summary["raw_started"] += 1
            return
        index = self._record("start", code)
        if (
            code in (self.context_snapshot_code, self.context_versions_code)
            and index is not None
        ):
            frame = sys._getframe(1)
            assert frame.f_code is code
            self._context_start(frame, index)
        if code is self.hook_current_code and index is not None:
            frame = sys._getframe(1)
            assert frame.f_code is code
            key = id(frame), threading.get_ident()
            if key in self.hook_active:
                self.hook_unmatched += 1
                self.hook_active.pop(key)
            if len(self.hook_active) >= 128 or len(self.hook_entries) >= 512:
                self.detail_overflow += 1
            else:
                self.hook_active[key] = len(self.hook_entries)
                self.hook_entries.append(
                    dict(
                        frame_id=key[0],
                        thread_id=key[1],
                        start_event_index=index,
                        return_event_index=None,
                        raw_started=0,
                        raw_returned=0,
                        raw_inclusive_seconds=0.0,
                        raw_callers={},
                        raw_caller_overflow=0,
                    )
                )
        if code in self.entry_starts and index is not None:
            frame = sys._getframe(1)
            assert frame.f_code is code
            self.entry_starts[code][id(frame)] = index

    def on_return(self, code, offset, value):
        if code is self.raw_check_code:
            frame = sys._getframe(1)
            assert frame.f_code is code
            current = self._hook_ancestor(frame)
            started = self.raw_active.pop((id(frame), threading.get_ident()), None)
            if started is not None:
                entry, timestamp, caller = started
                if self.hook_active.get(current) == entry:
                    elapsed = time.perf_counter() - timestamp
                    summary = self.hook_entries[entry]
                    summary["raw_returned"] += 1
                    summary["raw_inclusive_seconds"] += elapsed
                    if caller is not None:
                        summary["raw_callers"][caller]["returned"] += 1
                        summary["raw_callers"][caller]["inclusive_seconds"] += elapsed
                else:
                    self.raw_unmatched += 1
            return
        index = self._record("return", code)
        if code in (self.context_snapshot_code, self.context_versions_code):
            frame = sys._getframe(1)
            assert frame.f_code is code
            entry = self.context_active.pop((id(frame), threading.get_ident()), None)
            if entry is not None:
                self.context_reads[entry]["return_event_index"] = index
            else:
                self.context_unmatched += 1
        if code is self.hook_current_code:
            frame = sys._getframe(1)
            assert frame.f_code is code
            entry = self.hook_active.pop((id(frame), threading.get_ident()), None)
            if entry is not None:
                self.hook_entries[entry]["return_event_index"] = index
        elif code is self.receive_code and index is not None:
            outcome = (
                "none"
                if value is None
                else ("accepted" if len(value) else "refused")
                if type(value) is str  # noqa: E721 -- avoid custom result callbacks.
                else "unexpected"
            )
            self.receive_returns.append(
                (index, self._line(code, offset), offset, outcome)
            )
        elif code is self.sensitive_bundle_code and index is not None:
            self.sensitive_bundle_returns.append(
                (index, self._line(code, offset), offset, value is None)
            )
        if code in self.entry_starts:
            frame = sys._getframe(1)
            assert frame.f_code is code
            self.entry_starts[code].pop(id(frame), None)

    def on_yield(self, code, offset, value):
        assert code in self.entry_starts
        frame = sys._getframe(1)
        assert frame.f_code is code
        start = self.entry_starts[code].pop(id(frame), None)
        index = self._record("yield", code)
        if start is not None and index is not None:
            self.entry_pairs[code].append(
                (start, index, self.rows[index][2] - self.rows[start][2])
            )

    def receipt(self):
        result = dict(
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
            yield_code_count=len(self.entry_starts),
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
        if self.preparation_detail:
            result["preparation_detail"] = dict(
                enabled=True,
                context_reads=self.context_reads,
                context_read_limit=512,
                context_active_slot_limit=128,
                context_ancestor_walk_limit=64,
                context_overflow=self.context_overflow,
                context_ancestor_overflow=self.context_ancestor_overflow,
                unmatched_context_starts=self.context_unmatched,
                unfinished_context_reads=len(self.context_active),
                context_requested_count_is_list_or_tuple_length_not_sql_count=True,
                context_snapshot_count_copied_from_original_version_reader=True,
                context_event_times_are_inclusive_not_additive=True,
                generator_entry_pairs={
                    self.selected[code]: pairs
                    for code, pairs in self.entry_pairs.items()
                    if code is not self.admission_code
                },
                unfinished_generator_entries={
                    self.selected[code]: len(starts)
                    for code, starts in self.entry_starts.items()
                    if code is not self.admission_code
                },
                generator_entry_pair_fields=(
                    "start_event_index",
                    "first_yield_event_index",
                    "entry_seconds",
                ),
                generator_start_return_spans_are_full_context_lifetimes=True,
                private_text_span_is_unwrapped_body_elapsed=True,
                raw_checks_scoped_to_original_hook_current=True,
                raw_checks_are_aggregated_not_event_rows=True,
                hook_entries=[
                    dict(
                        entry,
                        raw_callers=[
                            dict(
                                source_file=filename,
                                caller_qualname=qualname,
                                source_line=line,
                                **counts,
                            )
                            for (filename, qualname, line), counts in entry[
                                "raw_callers"
                            ].items()
                        ],
                    )
                    for entry in self.hook_entries
                ],
                raw_caller_limit_per_entry=64,
                raw_caller_source_files="repository-relative; external paths omitted",
                hook_entry_limit=512,
                active_slot_limit=128,
                ancestor_walk_limit=64,
                overflow=self.detail_overflow,
                ancestor_overflow=self.ancestor_overflow,
                unmatched_raw_starts=self.raw_unmatched,
                unmatched_hook_starts=self.hook_unmatched,
                unfinished_raw_checks=len(self.raw_active),
                unfinished_hook_entries=len(self.hook_active),
                raw_elapsed_is_inclusive_not_additive=True,
                exception_unwinds_are_unfinished_not_completed=True,
                sensitive_bundle_returns=self.sensitive_bundle_returns,
                sensitive_bundle_return_fields=(
                    "event_index",
                    "source_line",
                    "bytecode_offset",
                    "returned_none",
                ),
                receive_returns=self.receive_returns,
                receive_return_fields=(
                    "event_index",
                    "source_line",
                    "bytecode_offset",
                    "outcome",
                ),
            )
        return result


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
