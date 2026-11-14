"""Private custody of the original direct RawCLI and model shell callers."""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Event, Lock

from tldw_chatbook.app_services import (
    read_app_config,
    read_chat_database,
    read_console_runtime,
    read_workspace_registry,
)
from tldw_chatbook.Agents.activation import (
    captured_sources,
    raw_cli_storage_execution,
    selected_sources,
)
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.Tools.raw_cli_executor import (
    _current_call,
    bind_model_raw_cli as bind_model_raw_cli,
    capture_model_raw_cli as capture_model_raw_cli,
    current_raw_cli_call as current_raw_cli_call,
)
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

from .console_raw_cli import NativeRawCliSource as NativeRawCliSource


class _UserRawCliCall:
    def __init__(self, issuer, session_id, invocation_id):
        from tldw_chatbook import config
        from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths

        selected = bootstrap.effective_config_path()
        cached = config._CONFIG_CACHE if config._CONFIG_CACHE_SOURCE == selected else {}
        self.log_root = profile_paths.user_data_dir(cached or {})
        self.issuer, self.runtime, self.app, self.raw = (
            issuer,
            issuer.runtime,
            issuer.app,
            issuer.raw,
        )
        self.invocation_id = invocation_id
        self._store = self.runtime._chat_store
        self.session = self._store._sessions[session_id]
        self.identity = self._identity()
        self.conversation_id = self.session.persisted_conversation_id
        self.persistence = self._store.persistence
        self.database = getattr(self.persistence, "db", None)
        self.registry = read_workspace_registry(self.app)
        self.registry_db = getattr(self.registry, "db", None)
        self.registry_revision = getattr(self.registry, "_mutation_generation", None)
        self.app_config = read_app_config(self.app)
        self.config_generation = config.get_runtime_config_generation()
        self._db = self.runtime._agent_runs_db
        self.receipts = self.runtime._activity_receipts
        self.capacity, self.activity = (
            self.runtime._execution_capacity,
            self.runtime._worktree_activity,
        )
        self.scratch = self.runtime.scratch_spaces
        self.snapshot = self.scratch_lease = self.chain = None
        self.owner = self.operation = self.claim = self.request = None
        self.run_id = None
        self.required_proven = self.physical_proven = True
        self.host_proven = True
        self.retired = Event()
        self._start_lock = Lock()
        self._started = self._cancelled = self._retiring = self._settling = False
        self._operation_finished = Event()
        self._storage_stack = None
        self._refuse_unstarted = None
        self.selection_sources = selected_sources(self)
        self.sources = tuple(
            dict.fromkeys(
                (*captured_sources(self), ("agents.history", self.log_root))
            )
        )

    def _identity(self):
        return (
            self.session.incarnation_id,
            self.session.conversation_binding_revision,
            self.session.workspace_id,
            self.session.identity_revision,
            self.session.settings_revision,
            self.session.generation_settings_revision,
            self.session.ephemeral,
            self.session.project_instruction_state.working_folder_binding_id,
        )

    def check_memory(self):
        from tldw_chatbook import config

        r = self.runtime
        if (
            self._cancelled
            or r._disposed
            or r._app is not self.app
            or read_console_runtime(self.app) is not r
            or self.app.raw_cli_runtime is not self.raw
            or r._chat_store is not self._store
            or self._store._sessions.get(self.session.id) is not self.session
            or self._identity() != self.identity
            or self.session.persisted_conversation_id != self.conversation_id
            or self.session.id in r._admission_fenced_sessions
            or self._store.persistence is not self.persistence
            or getattr(self.persistence, "db", None) is not self.database
            or read_chat_database(self.app) is not self.database
            or read_app_config(self.app) is not self.app_config
            or config.get_runtime_config_generation() != self.config_generation
            or read_workspace_registry(self.app)
            is not self.registry
            or getattr(self.registry, "db", None) is not self.registry_db
            or getattr(self.registry, "_mutation_generation", None)
            != self.registry_revision
            or self._db is not None
            and r._agent_runs_db is not self._db
            or self.receipts is not None
            and r._activity_receipts is not self.receipts
            or self.capacity is not None
            and r._execution_capacity is not self.capacity
            or self.activity is not None
            and r._worktree_activity is not self.activity
            or r.scratch_spaces is not self.scratch
        ):
            raise RuntimeError("original RawCLI source changed")

    def check(self):
        self.check_memory()
        if selected_sources(self) != self.selection_sources:
            raise RuntimeError("original RawCLI storage selection changed")
        if (
            self.chain is not None
            and capture_directory_chain(self.chain.canonical_root) != self.chain
        ):
            raise RuntimeError("original RawCLI directory changed")
        if self.snapshot is not None and not self.scratch.is_live(self.snapshot):
            raise RuntimeError("original RawCLI scratch changed")
        self.check_memory()

    @contextmanager
    def activate(self):
        with ExitStack() as stack:
            stack.enter_context(raw_cli_storage_execution(self))
            if self.owner is not None:
                stack.enter_context(self.owner.activate())
            for db in dict.fromkeys((self._db, self.database, self.registry_db)):
                if db is not None:
                    stack.enter_context(operation_owned_connection(db))
            previous_stack, self._storage_stack = self._storage_stack, stack
            token = _current_call.set(self) if self.request is not None else None
            try:
                yield
            finally:
                self._storage_stack = previous_stack
                if token is not None:
                    _current_call.reset(token)

    def start(self):
        with self._start_lock:
            if self._cancelled:
                return False
            self._started = True
            return True

    def prevent_start(self):
        with self._start_lock:
            self._cancelled = True
            return not self._started

    def cancel_unstarted(self):
        if self.prevent_start() and not self.retired.is_set():
            try:
                if self._refuse_unstarted is not None:
                    self._refuse_unstarted()
            finally:
                self.finish()
        elif self.request is not None:
            self.raw.cancel(self.request.invocation_id)

    def root(self):
        self.check()
        selected = self.identity[-1]
        if selected:
            binding = (
                self.registry.get_runtime_binding(selected) if self.registry else None
            )
            if (
                binding is None
                or str(binding.workspace_id) != str(self.session.workspace_id)
                or getattr(binding.status, "value", binding.status) != "ready"
                or getattr(binding.binding_kind, "value", binding.binding_kind)
                != "local-filesystem"
            ):
                raise RuntimeError("original RawCLI folder unavailable")
            root = Path(binding.locator)
        else:
            self.snapshot = self.scratch.snapshot(self.session.id)
            self.scratch_lease = self.scratch.lease(self.snapshot)
            root = self.scratch_lease.__enter__()
        self.chain = capture_directory_chain(root)
        self.check()
        return root

    def database_for_run(self):
        self.check()
        if self._db is None:
            receipts = self.runtime.ensure_activity_receipt_service()
            self._db = getattr(receipts, "_runs_db", None)
            self.receipts = receipts
            if self._db is None:
                raise RuntimeError("original RawCLI storage unavailable")
            self.selection_sources = selected_sources(self)
            self.sources = tuple(
                dict.fromkeys((*self.sources, *self.selection_sources))
            )
            self._storage_stack.enter_context(
                raw_cli_storage_execution(self)
            )
            self._storage_stack.enter_context(operation_owned_connection(self._db))
        self.check()
        return self._db

    def persisted(self, conversation_id):
        # First persistence alone preserves the store's conversation binding revision.
        if (
            self.conversation_id is None
            and self.session.persisted_conversation_id == conversation_id
        ):
            self.conversation_id = conversation_id
        self.check()

    def created(self, run_id):
        if self.run_id is not None and self.run_id != run_id:
            raise RuntimeError("original RawCLI run changed")
        self.run_id = run_id
        self.owner.bind_run(run_id)
        self.required_proven = False

    def matches_row(self, row):
        return (
            row is not None
            and row["id"] == self.run_id
            and row["conversation_id"] == self.conversation_id
            and row["agent_kind"] == "local_command"
            and row["assistant_message_id"] == self.request.transcript_anchor_id
            and row["task"] == "Local command"
            and row["parent_run_id"] is None
            and row["result"] is None
        )

    def observe_terminal(self, status, step=None):
        try:
            row = self._db.get_run(self.run_id)
            return bool(
                self.matches_row(row)
                and row["status"] == status
                and (
                    step is None
                    or any(
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        == json.dumps(step, sort_keys=True, separators=(",", ":"))
                        for value in row["steps"]
                    )
                )
            )
        except Exception:  # noqa: BLE001 -- unreadable original outcome is unresolved
            return False

    def deliver(self, marshal, callback, *args, refusal=False):
        def accept():
            try:
                self.check_memory()
            except Exception:  # noqa: BLE001 -- stale source cannot receive output
                if refusal and not self.runtime._disposed:
                    self.runtime.raw_cli_refusal_stash_bank.setdefault(
                        self.session.id, []
                    ).append(args[1])
                return  # Original presentation is conclusively stale; do not reselect.
            return callback(*args)

        self.host_proven = False
        try:
            result = marshal(accept)
        except Exception:
            try:
                self.check_memory()
            except Exception:  # noqa: BLE001 -- conclusive source loss discards delivery
                self.host_proven = True
            raise
        self.host_proven = True
        return result

    def retain_unproven(self):
        if not (self.required_proven and self.physical_proven and self.host_proven):
            self.operation.mark_stopping()
            if not self.physical_proven:
                self.owner.mark_cleanup_unproven()
            # No native retry/reaper exists; preserve this worker's original scopes.
            Event().wait()

    def finish_operation(self):
        """Join the original finish callback before leaving its storage scopes."""
        with self._start_lock:
            first = not self._retiring
            self._retiring = True
        if not first:
            self._operation_finished.wait()
            return
        self.retain_unproven()
        self.operation.finish()
        self._operation_finished.set()

    def finish(self):
        if not self._operation_finished.is_set():
            # Conclusively never-started/refused calls have no worker activation.
            with self.activate():
                self.finish_operation()
        with self._start_lock:
            if self._settling:
                return
            self._settling = True
        if self.scratch_lease is not None:
            self.scratch_lease.__exit__(None, None, None)
            self.scratch_lease = None
        self.owner.finish_root()
        self.claim.settle(durable=True)
        with self.runtime._execution_capacity_lock:
            # Lost original membership cannot prove release of its metadata.
            if self in self.issuer.pending:
                self.issuer.pending.remove(self)
                self.runtime._first_send_sources -= 1
        self.retired.set()
        self.runtime._wake_first_send()
