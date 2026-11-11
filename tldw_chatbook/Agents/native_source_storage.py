"""Original storage selection for one Console submit and its native agent tree."""

from __future__ import annotations

import inspect
from contextlib import ExitStack, contextmanager
from pathlib import Path

from tldw_chatbook.app_services import (
    read_app_config,
    read_mcp_service,
    read_workspace_registry,
)
from .activation import _sources, captured_sources, execution
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain


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
        self.app_config = read_app_config(self.app)
        self.configuration = configuration
        self.config_identity = config.current_config_identity()
        self.persistence = self.store.persistence
        self.registry = read_workspace_registry(self.app)
        self.databases = (
            bridge._db,
            getattr(self.persistence, "db", None),
            getattr(self.registry, "db", None),
        )
        self.database_paths = tuple(
            getattr(db, "db_path", None) for db in self.databases
        )
        self.mcp = read_mcp_service(self.app)
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
            dict.fromkeys((*captured_sources(None), *_sources(bridge)))
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
            or read_workspace_registry(self.app)
            is not self.registry
            or getattr(self.registry, "db", None) is not self.databases[2]
            or tuple(getattr(db, "db_path", None) for db in self.databases)
            != self.database_paths
            or read_app_config(self.app) is not self.app_config
            or read_mcp_service(self.app) is not self.mcp
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
            original_retained_source_stack(tree, self.databases)
            if tree is not None and self.root_identity is not None
            else ExitStack()
        )
        with scope as stack:
            from .execution_capacity import bind_ordinary_tree

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
        from tldw_chatbook.Agents.run_log import RunLogWriter

        if self.writer is not None:
            raise RuntimeError("original agent writer already selected")
        self.writer = writer
        self.writer_state = (
            RunLogWriter,
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
        writer_class, directory, scope, on_bound = self.writer_state
        if (
            writer is not self.writer
            or inspect.getattr_static(writer, "_console_storage_source", None)
            is not self
            or type(writer) is not writer_class
            or writer._explicit_root != self.root
            or writer._dir_name != directory
            or writer._access_scope is not scope
            or writer._on_bound is not on_bound
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


NativeAgentSources = _AgentSources
STORAGE_SOURCE_CLASS = _AgentSources


def console_writer_source(writer: object) -> _AgentSources | None:
    """Recognize only a writer bound by the original Console submit source."""
    source = inspect.getattr_static(writer, "_console_storage_source", None)
    if source is None:
        return None
    if type(source) is not _AgentSources or source.writer is not writer:
        raise RuntimeError("original agent writer source changed")
    source.writer_root(writer)
    return source


def original_retained_source_stack(tree, databases):
    """Retain the actual originating tree through native storage scope return."""
    from .ordinary_tree import require_original_ordinary_turn

    require_original_ordinary_turn(tree)
    return _retained_source_stack(tree, databases)
