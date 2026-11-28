"""Disposable body-free spans of original Send functions; no callback replacement."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

NAMES = {'standard_console_catalog_sources', '_personal_context_service', 'run_catalog_fingerprint', '_reserve_trace_call', 'standard_local_catalog_sources', 'create_chain', 'run_reply', '_console_first_request_runtime_context', 'resume_durable_postcommit', 'prepare_console_tools', 'standard_console_sources', '_compose_mcp_provider', 'plugin_service', '_run_agent_reply', 'resolve_log_root', '_run_maintenance_agent_call', 'build_console_first_request_plan', '_run_durable_postcommit_effect', 'transition_checkpoint', '_controller_factory_current', '_compose_local_provider', 'compose_run_mcp_provider', '_console_preparation_pipeline_current', '_compose_agent_request_providers', 'capture_run_tool_read'}

EXTRA = {
    'Chat/console_preparation_reads.py': {'run_preparation_read', 'run_preparation_read.<locals>.invoke'},
    'app.py': {'TldwCli._preimport_screens', 'TldwCli._pause_between_preimports'},
    'MCP/console_snapshot.py': {'_CapturedSources.require_current', '_CapturedSources.permission_owner', '_CapturedSources.read_permission_payload', '_CapturedSources.permission_call', '_CapturedSources.log_owner', '_CapturedLocalCatalog.read_bundle', '_checked_read', '_owned_worker'},
    'MCP/console_tool_preparation.py': {'_resolver_current', '_shared_normalization_eligible', '_schema_json', 'preparation_pipeline_current'},
    'MCP/unified_control_plane_service.py': {'UnifiedMCPControlPlaneService.local_external_catalog', '_resolve_tool_states_from_payload', 'UnifiedMCPControlPlaneService._audit_downgrade_if_fresh'},
    'MCP/local_control_service.py': {'LocalMCPControlService.get_inventory', 'LocalMCPControlService._project_external_catalog'},
    'MCP/local_store.py': {'LocalMCPStore.get_catalog_bundle', 'LocalMCPStore.load'},
    'MCP/permission_store.py': {'MCPPermissionStore.load', 'MCPPermissionStore.mark_config_changed'},
}

EXTRA.update({
    "DB/private_sqlite_process.py": {"HelperAdmission.reserve", "HelperReservation._claim", "HelperReservation._reaped", "HelperLease.start", "HelperLease.request", "HelperLease._exchange", "HelperLease.close"},
    "Backup_Recovery/storage_admission.py": {"_Acquisition.__init__", "_Acquisition.close", "acquire_storage", "_acquire_storage", "_reuse_evidence", "_Evidence.observe"},
})
EXTRA["MCP/permission_store.py"].add("MCPPermissionStore.catalog_fingerprint")
EXTRA["MCP/local_store.py"].add("LocalMCPStore.catalog_fingerprint")

# Additional body-free attribution only; no callback or application source replacement.
EXTRA['UI/Console_Modules/character_context.py'] = {'_stock_character_refresh_readers', 'ConsoleCharacterContextController._read_database_scope_metadata_pair', 'ConsoleCharacterContextController._await_database_work', 'ConsoleCharacterContextController._read_refresh_scope_sync', 'ConsoleCharacterContextController._read_database_scope_metadata', 'ConsoleCharacterContextController.refresh_presentation_if_scope_changed', 'ConsoleCharacterContextController._scope_is_current', 'ConsoleCharacterContextController.refresh_if_scope_changed', 'ConsoleCharacterContextController.refresh', 'ConsoleCharacterContextController._capture_scope', 'ConsoleCharacterContextController._refresh_recent_batch', 'ConsoleCharacterContextController._operation_scope_is_current', 'ConsoleCharacterContextController._load_recent_sync', 'ConsoleCharacterContextController._read_recent_scope_sync', 'ConsoleCharacterContextController._mark_checked', 'ConsoleCharacterContextController._checked_recently'}
EXTRA['UI/Screens/chat_screen.py'] = {'ChatScreen._sync_native_console_chat_ui'}


# Metadata-only source/lease attribution for the permission-read investigation.
for _file, _names in {'MCP/console_snapshot.py': {'_source_identity'}, 'MCP/permission_store.py': {'_load_in_owned_scope', '_console_owned_load_current', '_capture_console_owned_load', 'MCPPermissionStore._load_locked', 'MCPPermissionStore.mutation_fence'}, 'MCP/recovery_activation.py': {'observed', '_require_store_write', 'readable', 'selected_path'}, 'Backup_Recovery/raw_participants.py': {'_scope', '_mcp_observation', '_selection'}, 'Backup_Recovery/mcp_source_participants.py': {'binding', 'selection', 'members', 'preflight', 'canonical_path', '_selected'}, 'Backup_Recovery/generation_witnesses.py': {'_paired_witnesses', '_witnesses'}, 'Backup_Recovery/storage_admission.py': {'_derived_reuse', '_metadata_evidence', '_execution_selection_for', 'StorageLease.execution_context', '_derived_before'}}.items():
    EXTRA.setdefault(_file, set()).update(_names)

EXTRA['Backup_Recovery/storage_admission.py'].update({'_posture', '_content', '_Evidence.__init__', '_selector_evidence', '_path_evidence', '_mount_read_only', 'StorageLease.execution_scope', '_ordinary_hold'})
EXTRA['Backup_Recovery/qualification.py'] = {'native_identity', '_qualified_identity'}


# New focused boundaries are metadata only; original functions are never replaced.
HOOK_PHASES = {
 'Chat/console_chat_controller.py': {'ConsoleChatController._hook_admission_reason'},
 'Chat/console_automatic_submission.py': {'hook_admission_reason'},
 'Agents/hook_permissions.py': {'HookPermissions.snapshot', 'HookPermissions._current', 'HookPermissions._bind_store_lock.<locals>._store_lock', 'HookPermissions._read_state', 'HookPermissions._make_snapshot', 'HookPermissions._reconcile'},
 'config.py': {'locked_hooks_config_snapshot', '_config_write_lock', '_read_raw_cli_config_unlocked', '_hooks_config_snapshot', '_bind_config_interprocess_lock.<locals>._config_interprocess_lock'},
 'Backup_Recovery/config_participants.py': {'operation', 'companion_guard'},
 'Backup_Recovery/raw_participants.py': {'_raw_holds_pause_requested', '_scope', '_retire'},
 'Backup_Recovery/admission.py': {'Admission.pause_requested', 'Admission._directory', 'Admission._lock'},
}
for filename, names in HOOK_PHASES.items():
    EXTRA.setdefault(filename, set()).update(names)
HOOK_FUNCTIONS = set().union(*HOOK_PHASES.values())
HOOK_LINES = {'ConsoleChatController._hook_admission_reason', 'HookPermissions._current', 'locked_hooks_config_snapshot', '_config_write_lock', '_bind_config_interprocess_lock.<locals>._config_interprocess_lock', 'operation', 'companion_guard', '_scope', '_retire', '_raw_holds_pause_requested'}

# Focused original UI/first-send boundaries, no application callback replacement.
EXTRA['UI/Screens/chat_screen.py'].update({'ChatScreen._sync_console_workspace_context', 'ChatScreen._sync_console_rail_and_controls', 'ChatScreen._sync_console_control_bar', 'ChatScreen._sync_console_control_bar_under_config', 'ChatScreen._sync_console_mode_bar', 'ChatScreen._sync_native_console_transcript', 'ChatScreen._sync_console_native_session_tabs'})
EXTRA['UI/Console_Modules/retrieval.py'] = {'ConsoleRetrievalController._warm_console_effective_scope_cache_if_stale', 'ConsoleRetrievalController._refresh_active_dictionaries_summary_if_scope_changed', 'ConsoleRetrievalController._refresh_active_world_books_summary_if_scope_changed'}

# Distinguish executor start, fresh worker admission and the original body.
EXTRA['Agents/activation.py'] = {'worker_guard.<locals>.decorate.<locals>.call', '_sources', '_worker_databases'}
EXTRA['Backup_Recovery/admission_runtime.py'] = {'RecoveryAdmissionGuard.execution', 'RecoveryAdmissionGuard.captured_sources'}
EXTRA['Chat/console_chat_controller.py'] = {'ConsoleChatController._run_owned_chat_db_operation'}
EXTRA['Chat/console_automatic_submission.py'] = {'run_owned_chat_operation'}

UI_LINES = {'ChatScreen._sync_native_console_chat_ui', 'ChatScreen._sync_native_console_transcript'}
EXTRA['UI/Screens/chat_screen.py'].update({'ChatScreen._sync_console_native_session_tabs', 'ChatScreen._current_console_run_status_value', 'ChatScreen._ensure_console_chat_store', 'ChatScreen._sync_console_transcript_guidance', 'ChatScreen._sync_console_rail_and_controls', 'ChatScreen._native_console_transcript_fingerprint', 'ChatScreen._request_console_control_bar_sync', 'ChatScreen._current_console_attach_visit', 'ChatScreen._sync_native_console_chat_ui', 'ChatScreen._sync_console_chat_core_state', 'ChatScreen._sync_console_rail_visibility_if_changed', 'ChatScreen._sync_console_workspace_context', 'ChatScreen._current_console_rail_state', 'ChatScreen._sync_console_live_work_readiness_rows', 'ChatScreen.query_one', 'ChatScreen._prep_console_images', 'ChatScreen._ensure_console_chat_controller', 'ChatScreen._dispatch_console_rail_preference_prune', 'ChatScreen._record_ui_worker_finished', 'ChatScreen._rendered_console_terminal_receipts', 'ChatScreen.call_after_refresh', 'ChatScreen._console_presentation_context', 'ChatScreen._record_ui_worker_started', 'ChatScreen._recent_console_image_messages', 'ChatScreen._dispatch_active_console_roleplay_refresh', 'ChatScreen._ensure_console_image_view', 'ChatScreen._console_transcript_region_or_none', 'ChatScreen._show_model_thinking', 'ChatScreen.run_worker', 'ChatScreen._sync_native_console_transcript', 'ChatScreen._sync_console_mode_bar', 'ChatScreen._sync_console_settings_recovery_surfaces', 'ChatScreen._sync_console_citation_count_discovery'})
EXTRA['Widgets/Console/console_transcript.py'] = {'ConsoleTranscript.set_messages', 'ConsoleTranscript.set_presentation_context', 'ConsoleTranscript.set_model_thinking_visible', 'ConsoleTranscript.set_citation_counts', 'ConsoleTranscript.set_memory_banner_presentation', 'ConsoleTranscript.set_change_review_provider_factory', 'ConsoleTranscript.refresh_messages', 'ConsoleTranscript._reconcile_rows'}


EXTRA['Backup_Recovery/dictionary_source_job.py'] = {'_DictionaryJob.__init__', '_DictionaryJob.run', '_DictionaryJob._work', '_DictionaryJob._complete'}
EXTRA['Character_Chat/chat_dictionary_scope_service.py'] = {'ChatDictionaryScopeService.summarize_active_dictionaries', 'ChatDictionaryScopeService._invoke', 'ChatDictionaryScopeService._call_backend', 'ChatDictionaryScopeService._enforce_policy'}
EXTRA['UI/Console_Modules/retrieval.py'].add('_run_dictionary_summary_off_thread')
EXTRA['runtime_policy/enforcement.py'] = {'ServicePolicyEnforcer.require_allowed', 'ServicePolicyEnforcer.current_state'}


NAMES.update({'accept_received_intent', 'submit_draft'})
EXTRA['Chat/console_chat_controller.py'].add('ConsoleChatController._run_durable_db_call')
EXTRA['Chat/console_preparation_reads.py'].add('_preparation_handoffs')
EXTRA['DB/base_db.py'] = {'run_owned_db_call.<locals>.invoke', 'operation_owned_connection'}


from importlib.util import find_spec

TEXTUAL_ROOT = Path(find_spec('textual').origin).parent

NAMES.update({'_controller_accept_durable_turn', 'durable_acceptance_fingerprint_for', 'validate_durable_acceptance_fingerprint', '_build_durable_trace_request', '_admit_capture_policy', 'durable_postcommit_effects_for', '_build_trace_preparation', '_verify_trace_preparation_recovery', 'load_for_console', '_materialize_write_plan', 'load_runtime_config'})
UI_LINES.update({'_controller_accept_durable_turn', 'ConsoleProviderGateway._reserve_trace_call'})
FRAMEWORK = {
    'screen.py': {'Screen._refresh_layout', 'Screen._compositor_refresh'},
    '_compositor.py': {'Compositor.reflow', 'Compositor.reflow_visible', 'Compositor.render_update'},
    'css/stylesheet.py': {'Stylesheet.update', 'Stylesheet.update_nodes', 'Stylesheet.apply'},
}

class Observer:
    def __init__(self, root, output):
        self.root, self.output = Path(root), Path(output)
        self.rows, self.codes = [], set()
        self.overflow = False
        self.active = False
        self.permission_thread = None
        self.tool = next(i for i in (3, 4, 5) if sys.monitoring.get_tool(i) is None)
        self.callbacks = {}

    def selected(self, code):
        prefix = str(self.root / "tldw_chatbook") + "/"
        external = str(TEXTUAL_ROOT) + "/"
        if code.co_filename.startswith(external):
            return code.co_qualname in FRAMEWORK.get(code.co_filename[len(external):], set())
        return (code.co_filename.endswith("/pathlib.py") and code.co_qualname == "Path.resolve" or code.co_filename == "<frozen posixpath>" and code.co_name in {"realpath", "_joinrealpath"}) or code.co_filename.startswith(prefix) and (
            code.co_name in NAMES or code.co_name == "<module>"
            or code.co_qualname in EXTRA.get(code.co_filename[len(prefix):], set())
        )

    def record(self, code, event, current_line=None, callee=None):
        filename = Path(code.co_filename).name if event == "line" else ""
        if filename == "console_native_commit.py" and current_line == 950:
            self.active = True
            event = "durable_commit_succeeded"
        if not self.active:
            return
        if filename == "console_provider_gateway.py" and current_line == 3622:
            self.active = False
            event = "trace_reservation_entered"
        if event == 'start' and code.co_qualname == 'ConsoleChatController._hook_admission_reason':
            self.permission_thread = threading.get_ident()
        if code.co_qualname in {'Path.resolve', 'realpath', '_joinrealpath', '_posture', '_content', '_Evidence.__init__', '_selector_evidence', '_path_evidence', '_mount_read_only', 'native_identity', '_qualified_identity'} and threading.get_ident() != self.permission_thread:
            return
        if code.co_qualname in HOOK_FUNCTIONS and threading.get_ident() != self.permission_thread:
            return
        if event == 'line' and code.co_qualname in HOOK_LINES and threading.get_ident() != self.permission_thread:
            return
        if len(self.rows) >= 32768:
            self.overflow = True
            return
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        self.rows.append({"event": event, "ns": time.monotonic_ns(), "cpu_ns": time.thread_time_ns(), "handoff_seconds": sys.getswitchinterval(), "thread": threading.get_ident(), "task": id(task) if task is not None else None, "function": code.co_qualname if callee is None else code.co_qualname + " -> native:" + callee, "file": code.co_filename[len(str(self.root)) + 1:] if code.co_filename.startswith(str(self.root) + "/") else "<textual>/" + code.co_filename[len(str(TEXTUAL_ROOT)) + 1:] if code.co_filename.startswith(str(TEXTUAL_ROOT) + "/") else "<stdlib>/" + code.co_filename.rsplit("/", 1)[-1], "line": code.co_firstlineno, "at": current_line})

    def start(self):
        monitor = sys.monitoring
        events = monitor.events
        monitor.use_tool_id(self.tool, "console-postsave-spans")
        def started(code, offset):
            if not self.selected(code):
                return monitor.DISABLE
            if code not in self.codes:
                self.codes.add(code)
                mask = events.PY_RETURN | events.PY_YIELD | events.PY_RESUME
                if code.co_qualname in HOOK_LINES or code.co_qualname in {'_posture', '_content', '_mount_read_only', 'Path.resolve', '_joinrealpath', 'StorageLease.execution_context', 'StorageLease.execution_scope', '_reuse_evidence', '_derived_reuse', '_ordinary_hold', '_Acquisition.__init__', '_Acquisition.close'}:
                    mask |= events.CALL | events.C_RETURN | events.C_RAISE
                if code.co_qualname in HOOK_LINES | UI_LINES or code.co_name in {"prepare_console_tools", "run_preparation_read", "invoke", "refresh_presentation_if_scope_changed", "refresh_if_scope_changed"}:
                    mask |= events.LINE
                monitor.set_local_events(self.tool, code, mask)
            self.record(code, "start")
        def line(code, lineno):
            self.record(code, "line", lineno)
        def returned(code, offset, value):
            self.record(code, "return")
        def yielded(code, offset, value):
            self.record(code, "yield")
        def resumed(code, offset):
            self.record(code, "resume")
        def thrown(code, offset, error):
            if code in self.codes:
                self.record(code, "throw")
        def unwound(code, offset, error):
            if code in self.codes:
                self.record(code, "unwind")
        def native_event(code, event, callee):
            name = getattr(callee, '__qualname__', '')
            if name in {'stat', 'lstat', 'statvfs', 'RLock.__enter__', 'RLock.__exit__', 'RLock.acquire', 'RLock.release', 'lock.acquire', 'lock.release', 'flock', 'lockf'} and threading.get_ident() == self.permission_thread:
                self.record(code, event, callee=name)
        def native_start(code, offset, callee, unused_argument):
            native_event(code, 'start', callee)
        def native_return(code, offset, callee, unused_argument):
            native_event(code, 'return', callee)
        def native_raise(code, offset, callee, unused_argument):
            native_event(code, 'unwind', callee)
        self.callbacks = {events.CALL: native_start, events.C_RETURN: native_return, events.C_RAISE: native_raise, events.LINE: line, events.PY_START: started, events.PY_RETURN: returned, events.PY_YIELD: yielded, events.PY_RESUME: resumed, events.PY_THROW: thrown, events.PY_UNWIND: unwound}
        for event, callback in self.callbacks.items():
            monitor.register_callback(self.tool, event, callback)
        monitor.set_events(self.tool, events.PY_START)

    def close(self):
        monitor = sys.monitoring
        monitor.set_events(self.tool, 0)
        for code in self.codes:
            monitor.set_local_events(self.tool, code, 0)
        for event in self.callbacks:
            monitor.register_callback(self.tool, event, None)
        monitor.free_tool_id(self.tool)
        self.output.write_text(json.dumps({"rows": self.rows, "overflow": self.overflow, "tool_id": self.tool, "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scope": "Focused hook-config phase diagnostic, metadata only: original _posture/_content and stdlib Path.resolve/realpath boundaries, no callback replacement, no payload/arguments/locals; post-save-only original function/import spans, global exception events disabled; spans include scheduling; no arguments/locals/content or native paint; timing baseline excludes this observer; LINE enabled only for prepare_console_tools, finite preparation-read submit/invoke functions, and the two selected Character display/freshness dispatch functions; narrow native helper/storage/fingerprint boundaries; hot descriptor/file/cursor/check selections omitted; 32768 event ceiling"}, indent=2) + "\n")
        assert not self.overflow, "profile overflow; do not qualify truncated data"
