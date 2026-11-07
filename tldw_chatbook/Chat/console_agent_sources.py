"""Original storage selection for one Console submit and its native agent tree."""

from __future__ import annotations

# Capture the original tuple before other module initialization.
# isort: split
from tldw_chatbook.Chat.console_ordinary_tree import (
    _NATIVE_CURRENTNESS_SOURCE as _ORIGINAL_CURRENTNESS_SOURCE,
)

# isort: split

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
        from types import ModuleType as _currentness_module_type

        from tldw_chatbook.Chat import (
            console_chat_controller as _currentness_controller,
        )
        from tldw_chatbook.Chat import console_runtime as _currentness_runtime

        _currentness_source = _ORIGINAL_CURRENTNESS_SOURCE
        if (
            type(_currentness_runtime) is not _currentness_module_type
            or type(_currentness_controller) is not _currentness_module_type
            or _currentness_source is not vars(_currentness_runtime).get("_ORIGINAL_CURRENTNESS_SOURCE")
            or _currentness_source is not vars(_currentness_controller).get("_ORIGINAL_CURRENTNESS_SOURCE")
            or _currentness_source[6].modules.get("tldw_chatbook.Chat.console_runtime") is not _currentness_runtime
            or _currentness_source[6].modules.get("tldw_chatbook.Chat.console_chat_controller") is not _currentness_controller
        ):
            raise RuntimeError("original native currentness consumer changed")
        _currentness_record = _currentness_source[4][-1][1]
        _currentness_reader, _currentness_code, _currentness_defaults, _currentness_kwdefaults, _currentness_entries, _currentness_closure, _currentness_cells = _currentness_record
        if (
            _currentness_reader.__code__ is not _currentness_code
            or _currentness_reader.__defaults__ is not _currentness_defaults
            or _currentness_reader.__kwdefaults__ is not _currentness_kwdefaults
            or tuple((_currentness_kwdefaults or {}).items()) != _currentness_entries
            or _currentness_reader.__closure__ is not _currentness_closure
            or any(cell.cell_contents is not value for cell, value in _currentness_cells)
        ):
            raise RuntimeError("original native currentness dispatcher changed")

        prepared = _currentness_reader(_currentness_source, 'workflow', controller)
        if prepared is not None:
            return prepared.source
        return None
    return holder[3]


async def prepare_agent_sources(
    controller: object, session: object, configuration: object
) -> None:
    """Select the original source tuple before the first SessionStart effect."""
    from types import ModuleType as _currentness_module_type

    from tldw_chatbook.Chat import console_chat_controller as _currentness_controller
    from tldw_chatbook.Chat import console_runtime as _currentness_runtime

    from .console_agent_bridge import ConsoleAgentBridge
    from .console_preparation_reads import run_preparation_read

    _currentness_source = _ORIGINAL_CURRENTNESS_SOURCE
    if (
        type(_currentness_runtime) is not _currentness_module_type
        or type(_currentness_controller) is not _currentness_module_type
        or _currentness_source is not vars(_currentness_runtime).get("_ORIGINAL_CURRENTNESS_SOURCE")
        or _currentness_source is not vars(_currentness_controller).get("_ORIGINAL_CURRENTNESS_SOURCE")
        or _currentness_source[6].modules.get("tldw_chatbook.Chat.console_runtime") is not _currentness_runtime
        or _currentness_source[6].modules.get("tldw_chatbook.Chat.console_chat_controller") is not _currentness_controller
    ):
        raise RuntimeError("original native currentness consumer changed")
    _currentness_record = _currentness_source[4][-1][1]
    _currentness_reader, _currentness_code, _currentness_defaults, _currentness_kwdefaults, _currentness_entries, _currentness_closure, _currentness_cells = _currentness_record
    if (
        _currentness_reader.__code__ is not _currentness_code
        or _currentness_reader.__defaults__ is not _currentness_defaults
        or _currentness_reader.__kwdefaults__ is not _currentness_kwdefaults
        or tuple((_currentness_kwdefaults or {}).items()) != _currentness_entries
        or _currentness_reader.__closure__ is not _currentness_closure
        or any(cell.cell_contents is not value for cell, value in _currentness_cells)
    ):
        raise RuntimeError("original native currentness dispatcher changed")

    holder = _issuing.get()
    runtime = getattr(controller, "_hooks_v2_runtime", None)
    bridge = controller._agent_bridge
    if (
        runtime is None
        or type(bridge) is not ConsoleAgentBridge
        or not _currentness_reader(_currentness_source, 'platform')
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


@contextmanager
def _retained_source_stack(tree, databases):
    """Retain the original tree until native scopes and handles return."""
    locals_and_handles = []
    for database in dict.fromkeys(db for db in databases if db is not None):
        if getattr(database, "is_memory_db", False):
            continue
        namespace = vars(database)
        local = namespace.get("_local") or namespace.get("_thread_local")
        if local is not None:
            locals_and_handles.append((local, getattr(local, "conn", None)))
    cleanup = [False]
    body_error = None
    with tree.retain_source_scope(_cleanup_state=cleanup):
        try:
            with ExitStack() as stack:
                try:
                    yield stack
                except BaseException as error:
                    body_error = error
                    raise
        except BaseException as error:
            if error is body_error and all(
                getattr(local, "conn", None) is original
                for local, original in locals_and_handles
            ):
                cleanup[0] = True
            raise
        else:
            if not all(
                getattr(local, "conn", None) is original
                for local, original in locals_and_handles
            ):
                raise RuntimeError("original source database handle did not return")
            cleanup[0] = True


class _AgentSources:
    def __init__(self, controller, runtime, bridge, session, configuration):
        from tldw_chatbook import config

        self.controller, self.runtime, self.bridge = controller, runtime, bridge
        self.app, self.store, self.session = controller.app, controller.store, session
        app = self.app
        try:
            self.app_config = app.app_config
        except AttributeError:
            self.app_config = None
        self.configuration = configuration
        self.config_identity = config.current_config_identity()
        self.persistence = self.store.persistence
        app = self.app
        try:
            self.registry = app.workspace_registry_service
        except AttributeError:
            self.registry = None
        self.databases = (
            bridge._db,
            getattr(self.persistence, "db", None),
            getattr(self.registry, "db", None),
        )
        self.database_paths = tuple(
            getattr(db, "db_path", None) for db in self.databases
        )
        app = self.app
        try:
            self.mcp = app.unified_mcp_service
        except AttributeError:
            self.mcp = None
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

    def _check_original_sources(self):
        from tldw_chatbook import config

        self._check_original_bindings()
        if config.current_config_identity() != self.config_identity:
            raise RuntimeError("original agent storage source changed")

    def _check_original_bindings(self):
        from tldw_chatbook.Agents import agent_request_execution
        from . import (
            console_agent_reply, console_streaming_request, console_project_dispatch,
            console_trace_maintenance_runtime, console_review_delivery, console_plugin_context,
        )
        from .console_ordinary_tree import _function_unchanged

        for module in (
            agent_request_execution, console_agent_reply, console_streaming_request,
            console_project_dispatch, console_trace_maintenance_runtime,
            console_review_delivery, console_plugin_context,
        ):
            if any(
                vars(module).get(name) is not record[0] or not _function_unchanged(record)
                for name, record in module._NATIVE_WORKFLOW_RECEIVERS
            ):
                raise PermissionError("original extracted native receiver changed")
        tree = getattr(self, "ordinary_tree", None)
        retained = tree is not None and tree.has_admitted_members()
        prepared = getattr(self, "workflow_preparation", None)
        retained = retained or (
            prepared is not None
            and prepared.source is self
            and prepared in self.runtime._workflow_preparations
            and prepared._source_scope is not None
        )
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
        ):
            raise RuntimeError("original agent storage source changed")
        app = self.app
        try:
            registry = app.workspace_registry_service
        except AttributeError:
            registry = None
        if (
            registry is not self.registry
            or getattr(self.registry, "db", None) is not self.databases[2]
            or tuple(getattr(db, "db_path", None) for db in self.databases)
            != self.database_paths
        ):
            raise RuntimeError("original agent storage source changed")
        app = self.app
        try:
            app_config = app.app_config
        except AttributeError:
            app_config = None
        if app_config is not self.app_config:
            raise RuntimeError("original agent storage source changed")
        app = self.app
        try:
            mcp = app.unified_mcp_service
        except AttributeError:
            mcp = None
        if (
            mcp is not self.mcp
            or getattr(self.mcp, "_permission_store", None) is not self.permission
            or getattr(self.permission, "path", None) != self.permission_path
            or self.controller._scratch_spaces is not self.scratch_manager
        ):
            raise RuntimeError("original agent storage source changed")

    def check_memory(self):
        self._check_original_sources()
        if self.writer is not None:
            self._check_writer(self.writer)

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

    @contextmanager
    def activate(self):
        self.check_memory()
        tree = getattr(self, "ordinary_tree", None)
        scope = (
            _original_retained_source_stack(tree, self.databases)
            if tree is not None and self.root_identity is not None
            else ExitStack()
        )
        with scope as stack:
            from .console_ordinary_tree import bind_ordinary_tree

            stack.enter_context(bind_ordinary_tree(tree))
            stack.enter_context(execution(sources=self.sources))
            for db in dict.fromkeys(db for db in self.databases if db is not None):
                stack.enter_context(operation_owned_connection(db))
            if self.root_identity is not None:
                if tree is None:
                    stack.enter_context(self.scratch_manager.lease(self.scratch))
                self.check_storage()
            self.check_memory()
            yield

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
from .console_ordinary_tree import _function_unchanged, _OrdinaryTurn

_NATIVE_RETAINED_SOURCE_STACK = (
    _record_source_function(_retained_source_stack),
    _record_source_function(_retained_source_stack.__wrapped__),
    _record_source_function(_OrdinaryTurn.retain_source_scope),
    _record_source_function(_OrdinaryTurn.retain_source_scope.__wrapped__),
)


def _original_retained_source_stack(tree, databases):
    records = _NATIVE_RETAINED_SOURCE_STACK
    if (
        type(tree) is not _OrdinaryTurn
        or globals().get("_retained_source_stack") is not records[0][0]
        or getattr(_retained_source_stack, "__wrapped__", None) is not records[1][0]
        or inspect.getattr_static(tree, "retain_source_scope", None)
        is not records[2][0]
        or inspect.getattr_static(_OrdinaryTurn, "retain_source_scope", None)
        is not records[2][0]
        or getattr(records[2][0], "__wrapped__", None) is not records[3][0]
        or not all(_function_unchanged(record) for record in records)
    ):
        raise RuntimeError("original source retention changed")
    return _retained_source_stack(tree, databases)
