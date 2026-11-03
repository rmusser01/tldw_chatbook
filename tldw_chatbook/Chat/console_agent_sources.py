"""Original storage selection for one Console submit and its native agent tree."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path

from tldw_chatbook.Agents.activation import _guard, _sources, execution
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

_issuing = ContextVar("console_agent_sources", default=None)


@contextmanager
def issuing_agent_sources(controller: object) -> Iterator[None]:
    """Keep live provenance outside copied configuration and prompt identities."""
    with ExitStack() as stack:
        holder = [controller, asyncio.current_task(), stack, None]
        token = _issuing.set(holder)
        try:
            yield
        finally:
            _issuing.reset(token)


def current_agent_sources(controller: object | None = None) -> _AgentSources | None:
    holder = _issuing.get()
    if holder is None or controller is not None and holder[0] is not controller:
        return None
    return holder[3]


async def prepare_agent_sources(
    controller: object, session: object, configuration: object
) -> None:
    """Select the original source tuple before the first SessionStart effect."""
    from .console_agent_bridge import ConsoleAgentBridge
    from .console_preparation_reads import run_preparation_read
    from .console_workflow_runs import native_workflow_qualified

    holder = _issuing.get()
    runtime = getattr(controller, "_hooks_v2_runtime", None)
    bridge = controller._agent_bridge
    if (
        runtime is None
        or type(bridge) is not ConsoleAgentBridge
        or not native_workflow_qualified()
        or not controller._agent_runtime_enabled
        or session.assistant_kind == "character"
    ):
        return
    if holder is None or holder[:2] != [controller, asyncio.current_task()]:
        raise RuntimeError("original agent source issuer unavailable")
    if holder[3] is not None:
        raise RuntimeError("agent storage already selected")
    source = _AgentSources(controller, runtime, bridge, session, configuration)
    from .console_received_turn import received_turn_claim_for

    source.received_claim = received_turn_claim_for(controller.store, session.id)
    await run_preparation_read(
        source.prepare,
        creator=controller,
        session_id=session.id,
        reads=controller._preparation_reads,
        observers=(runtime._preparation_reads,),
        require_current=source.check_memory,
    )
    source.check_memory()
    holder[3] = source
    holder[2].enter_context(source.activate())


class _AgentSources:
    def __init__(self, controller, runtime, bridge, session, configuration):
        from tldw_chatbook import config

        self.controller, self.runtime, self.bridge = controller, runtime, bridge
        self.app, self.store, self.session = controller.app, controller.store, session
        self.app_config = getattr(self.app, "app_config", None)
        self.configuration = configuration
        self.config_identity = config.current_config_identity()
        self.persistence = self.store.persistence
        self.registry = getattr(self.app, "workspace_registry_service", None)
        self.databases = (
            bridge._db,
            getattr(self.persistence, "db", None),
            getattr(self.registry, "db", None),
        )
        self.database_paths = tuple(
            getattr(db, "db_path", None) for db in self.databases
        )
        self.mcp = getattr(self.app, "unified_mcp_service", None)
        self.permission = getattr(self.mcp, "_permission_store", None)
        self.permission_path = getattr(self.permission, "path", None)
        self.prospective_mcp_store = self.prospective_mcp_identity = None
        self.prospective_permission_path = None
        self.native_permission_publication = None
        if self.mcp is not None and self.permission is None:
            from tldw_chatbook.MCP.console_snapshot import _source_identity
            from tldw_chatbook.MCP.local_store import LocalMCPStore
            from tldw_chatbook.MCP.unified_control_plane_service import (
                UnifiedMCPControlPlaneService,
            )

            local = getattr(self.mcp, "local_service", None)
            store = getattr(local, "store", None)
            if (
                type(self.mcp) is UnifiedMCPControlPlaneService
                and type(store) is LocalMCPStore
            ):
                self.prospective_mcp_store = store
                self.prospective_mcp_identity = _source_identity(store)
                self.prospective_permission_path = Path(
                    getattr(store, "_recovery_original_path", store.path)
                ).with_name("mcp_permissions.json")
        self.scratch_manager = controller._scratch_spaces
        self.scratch = configuration.scratch_space
        self.workspace_id = session.workspace_id
        self.incarnation = session.incarnation_id
        self.sources = tuple(
            dict.fromkeys((*_guard.captured_sources(None), *_sources(bridge)))
        )
        workspace_path = self.database_paths[2]
        if workspace_path is not None and not self.databases[2].is_memory_db:
            from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path

            self.sources = tuple(
                dict.fromkeys(
                    (
                        *self.sources,
                        ("db.workspaces", lexical_path(workspace_path)),
                    )
                )
            )
        self.root = self.root_identity = None
        self.writer = self.writer_state = self.bound = None
        self.first_send_source = None

    def _check_original_sources(self):
        from tldw_chatbook import config

        self._check_original_bindings()
        if config.current_config_identity() != self.config_identity:
            raise RuntimeError("original agent storage source changed")

    def _check_original_bindings(self):
        tree = getattr(self, "ordinary_tree", None)
        retained = tree is not None and tree.has_admitted_members()
        if (
            self.controller._hooks_v2_runtime is not self.runtime
            or self.controller._agent_bridge is not self.bridge
            or self.runtime._agent_bridge is not self.bridge
            or self.runtime._chat_controller is not self.controller
            or self.runtime._chat_store is not self.store
            or self.controller.app is not self.app
            or self.controller.store is not self.store
            or self.store._sessions.get(self.session.id) is not self.session
            or self.session.incarnation_id != self.incarnation
            or self.session.workspace_id != self.workspace_id
            or (self.controller._disposed or self.runtime._disposed)
            and not retained
            or self.store.persistence is not self.persistence
            or self.bridge._db is not self.databases[0]
            or self.persistence.db is not self.databases[1]
            or getattr(self.app, "workspace_registry_service", None)
            is not self.registry
            or getattr(self.registry, "db", None) is not self.databases[2]
            or tuple(getattr(db, "db_path", None) for db in self.databases)
            != self.database_paths
            or getattr(self.app, "app_config", None) is not self.app_config
            or getattr(self.app, "unified_mcp_service", None) is not self.mcp
            or getattr(self.mcp, "_permission_store", None) is not self.permission
            or getattr(self.permission, "path", None) != self.permission_path
            or self.controller._scratch_spaces is not self.scratch_manager
        ):
            raise RuntimeError("original agent storage source changed")

    def check_memory(self):
        self._check_original_sources()
        if self.writer is not None:
            self._check_writer(self.writer)

    def check_retirement(self, event):
        """Check only the retained original, fully returned Stop event."""
        from tldw_chatbook.Agents.hooks_v2.native_commands import (
            _CommandCustody,
            _CommandEvent,
        )

        from .console_first_send_retirement import first_send_receipt_current

        tree = getattr(self, "ordinary_tree", None)
        receipt = getattr(self, "first_send_receipt", None)
        if (
            type(event) is not _CommandCustody
            or type(event.source) is not _CommandEvent
            or event.source.agent_sources is not self
            or event.source.ordinary_tree is not tree
            or event.source.event.event != "Stop"
            or event.source.issuer.runtime is not self.runtime
            or event.claim is not None
            or tree is None
            or tree.source is not self
            or not first_send_receipt_current(self, receipt)
        ):
            raise RuntimeError("original Stop retirement proof unavailable")
        with event._lock:
            if (
                event.retired
                or event.failed_write
                or not event.host_returned
                or not all(
                    target.event is event
                    and target.returned
                    and target.proven
                    and target.operation.owner is event.owner
                    for target in event.targets
                )
            ):
                raise RuntimeError("original Stop physical return unavailable")
        with tree.lock:
            member = tree.members.get(event.owner)
            if (
                member is None
                or not tree.sealed
                or not member.host_returned
                or not member.durable
                or tree.settled
                or tree.configuration.scratch_space is not None
                and tree.scratch_lease is None
            ):
                raise RuntimeError("original Stop member proof unavailable")
        self._check_original_bindings()
        if self.writer is not None:
            self._check_writer(self.writer)
        self.check_storage()
        if not first_send_receipt_current(self, receipt):
            raise RuntimeError("original Stop publication changed")

    @contextmanager
    def retain_retirement(self, event):
        self.check_retirement(event)
        tree = self.ordinary_tree
        with tree.lock:
            if (
                tree.scratch_lease is None
                and tree.configuration.scratch_space is not None
            ):
                raise RuntimeError("original ordinary scratch lease retired")
            tree.source_scopes += 1
        self.check_retirement(event)
        # An exception at the yield skips release, retaining unproven custody.
        yield
        with tree.lock:
            tree.source_scopes -= 1
        tree._retire_scratch()

    @contextmanager
    def retirement_journal(self, event):
        """Expose retained DB context only to the checked native journal."""
        self.check_retirement(event)
        with ExitStack() as stack:
            stack.enter_context(execution(sources=self.sources))
            for db in dict.fromkeys(db for db in self.databases if db is not None):
                stack.enter_context(operation_owned_connection(db))
            yield

    def retirement_journal_callback(self, event, record):
        """Recognize the callback actually issued by the native worktree owner."""
        from types import CodeType

        from tldw_chatbook.Agents.agent_service import (
            _NATIVE_CONSOLE_WORKTREE_ADMISSION,
            AgentService,
        )
        from tldw_chatbook.DB.agent_worktrees import _NATIVE_CONSOLE_DRAIN_REPOSITORY

        from .console_ordinary_tree import _function_unchanged

        if type(record) is not tuple or len(record) != 11:
            raise RuntimeError("original worktree journal unavailable")
        (
            callback,
            callback_record,
            connection,
            connection_record,
            generator_record,
            service,
            repository,
            db,
            run_id,
            owner,
            tree,
        ) = record
        function, code, defaults, kwdefaults, closure = (
            _NATIVE_CONSOLE_WORKTREE_ADMISSION
        )
        (
            repo_class,
            repo_method,
            repo_code,
            repo_defaults,
            repo_kwdefaults,
            repo_closure,
        ) = _NATIVE_CONSOLE_DRAIN_REPOSITORY
        nested = {
            item.co_name: item for item in code.co_consts if type(item) is CodeType
        }
        callback_cells = dict(
            zip(
                callback.__code__.co_freevars,
                (cell.cell_contents for cell in callback.__closure__ or ()),
            )
        )
        if (
            inspect.getattr_static(AgentService, "_admit_agent_worktree")
            is not function
            or function.__code__ is not code
            or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not kwdefaults
            or function.__closure__ is not closure
            or type(service) is not AgentService
            or service.db is not db
            or getattr(service, "_ordinary_tree", None) is not tree
            or tree is not self.ordinary_tree
            or owner is not event.owner
            or db is not self.databases[0]
            or type(repository) is not repo_class
            or repository._db is not db
            or inspect.getattr_static(repo_class, "mark_writer_finished")
            is not repo_method
            or repo_method.__code__ is not repo_code
            or repo_method.__defaults__ is not repo_defaults
            or repo_method.__kwdefaults__ is not repo_kwdefaults
            or repo_method.__closure__ is not repo_closure
            or not _function_unchanged(callback_record)
            or not _function_unchanged(connection_record)
            or not _function_unchanged(generator_record)
            or callback.__code__ is not nested.get("persist_actual_drain")
            or getattr(connection, "__wrapped__", None).__code__
            is not nested.get("callback_connection")
            or callback_cells.get("callback_connection") is not connection
            or callback_cells.get("repository") is not repository
            or callback_cells.get("execution_owner") is not owner
            or callback_cells.get("child_run_id") != run_id
        ):
            raise RuntimeError("original worktree journal source changed")
        with tree.lock:
            member = tree.members.get(owner)
            if member is None or member.run_id != run_id or not member.durable:
                raise RuntimeError("original worktree journal member changed")
        return callback

    def _allowed_roots(self):
        from tldw_chatbook.Tools.workspace_file_roots import (
            allowed_file_roots,
            run_workspace,
        )

        authority = self.configuration.project_authority
        bindings = authority.options if authority is not None else ()
        workspace_id = (
            authority.workspace_id
            if authority is not None
            else self.configuration.provider_selection.workspace_context.active_workspace_id
        )
        with run_workspace(
            workspace_id,
            read_binding_ids=tuple(b.binding_id for b in bindings),
            write_binding_ids=tuple(b.binding_id for b in bindings if b.allow_write),
            binding_authority=bindings,
        ):
            return allowed_file_roots(
                write=True,
                sandbox_root=self.scratch.root,
                _selection=(self.registry, workspace_id),
            )

    def prepare(self):
        self.check_memory()
        if self.scratch is None:
            raise RuntimeError("original agent scratch source unavailable")
        with self.activate(), self.scratch_manager.lease(self.scratch):
            roots = self._allowed_roots()
            self.root = roots[1] if len(roots) > 1 else roots[0]
            self.root_identity = capture_directory_chain(self.root)
            extra = [("agents.history", self.root)]
            if self.permission_path is not None:
                extra.append(("mcp.permissions", Path(self.permission_path)))
            elif self.prospective_permission_path is not None:
                extra.append(("mcp.permissions", self.prospective_permission_path))
            self.sources = tuple(dict.fromkeys((*self.sources, *extra)))
        self.check_memory()
        # This is optional, private retirement evidence only.  An unavailable
        # or non-native view never weakens the normal source checks.
        from . import console_first_send_retirement as retirement

        try:
            fence, code, defaults, kwdefaults, closure = (
                retirement._NATIVE_STOP_PRODUCER_FENCE
            )
            if (
                retirement._stop_producers_current is fence
                and fence.__code__ is code
                and fence.__defaults__ is defaults
                and fence.__kwdefaults__ is kwdefaults
                and fence.__closure__ is closure
                and fence()
            ):
                captured = retirement.capture_first_send_source(self)
                if fence():
                    self.first_send_source = captured
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
            self.first_send_source = None
        self.check_memory()

    @contextmanager
    def activate(self, *, _initial_refusal=None):
        try:
            self.check_memory()
        except RuntimeError:
            if type(_initial_refusal) is list and not _initial_refusal:
                _initial_refusal.append(self)
            raise
        entry_error = None
        try:
            with ExitStack() as stack:
                from .console_ordinary_tree import bind_ordinary_tree

                stack.enter_context(
                    bind_ordinary_tree(getattr(self, "ordinary_tree", None))
                )
                stack.enter_context(execution(sources=self.sources))
                for db in dict.fromkeys(db for db in self.databases if db is not None):
                    stack.enter_context(operation_owned_connection(db))
                if self.root_identity is not None:
                    tree = getattr(self, "ordinary_tree", None)
                    stack.enter_context(
                        tree.retain_source_scope()
                        if tree is not None
                        else self.scratch_manager.lease(self.scratch)
                    )
                    self.check_storage()
                try:
                    self.check_memory()
                except RuntimeError as error:
                    entry_error = error
                    raise
                yield
        except BaseException as error:
            if (
                error is entry_error
                and type(_initial_refusal) is list
                and not _initial_refusal
            ):
                _initial_refusal.append(self)
            raise

    def bind_writer(self, writer):
        from tldw_chatbook.Agents.run_log import _RUN_LOG_PATH_WRITER_ORIGINAL

        if self.writer is not None:
            raise RuntimeError("original agent writer already selected")
        self.writer = writer
        self.writer_state = (
            _RUN_LOG_PATH_WRITER_ORIGINAL,
            writer._dir_name,
            writer._access_scope,
            writer._on_bound,
        )
        writer._console_storage_source = self
        self.writer_root(writer)

    def check_storage(self):
        if (
            capture_directory_chain(self.root) != self.root_identity
            or self.root not in self._allowed_roots()
        ):
            raise RuntimeError("original agent log root changed")
        if (
            self.bound is not None
            and capture_directory_chain(self.bound[1]) != self.bound[2]
        ):
            raise RuntimeError("original agent log binding changed")

    def _check_writer(self, writer):
        record, directory, scope, on_bound = self.writer_state
        writer_class, methods = record
        if (
            writer is not self.writer
            or inspect.getattr_static(writer, "_console_storage_source", None)
            is not self
            or type(writer) is not writer_class
            or writer._explicit_root != self.root
            or writer._dir_name != directory
            or writer._access_scope is not scope
            or writer._on_bound is not on_bound
            or not all(
                inspect.getattr_static(writer, name, None) is callback
                and callback.__code__ is code
                and callback.__defaults__ is defaults
                for name, callback, code, defaults in methods
            )
            or writer.log_dir != (self.bound[1] if self.bound is not None else None)
        ):
            raise RuntimeError("original agent log writer changed")

    def writer_root(self, writer):
        self._check_original_sources()
        self._check_writer(writer)
        return self.root

    def writer_bound(self, writer, run_id, root, directory):
        self.writer_root(writer)
        expected = (
            self.root
            / (
                self.writer_state[1]
                if self.writer_state[1].startswith(".")
                else "." + self.writer_state[1]
            )
            / run_id
        )
        if self.bound is not None or root != self.root or directory != expected:
            raise RuntimeError("original agent log binding changed")
        identity = capture_directory_chain(directory)
        if identity.canonical_root != directory:
            raise RuntimeError("original agent log binding changed")
        self.bound = (run_id, directory, identity)


def console_writer_source(writer: object) -> _AgentSources | None:
    """Recognize only a writer bound by the original Console submit source."""
    source = inspect.getattr_static(writer, "_console_storage_source", None)
    if source is None:
        return None
    if type(source) is not _AgentSources or source.writer is not writer:
        raise RuntimeError("original agent writer source changed")
    source.writer_root(writer)
    return source


from .console_ordinary_tree import _function_record as _record_source_function

_NATIVE_STOP_SOURCE_METHODS = tuple(
    (
        name,
        _record_source_function(function),
        _record_source_function(function.__wrapped__)
        if hasattr(function, "__wrapped__")
        else None,
    )
    for name in (
        "check_memory",
        "_check_original_sources",
        "_check_original_bindings",
        "check_retirement",
        "retain_retirement",
        "retirement_journal",
        "retirement_journal_callback",
        "activate",
        "check_storage",
        "_check_writer",
        "_allowed_roots",
    )
    for function in (inspect.getattr_static(_AgentSources, name),)
)
