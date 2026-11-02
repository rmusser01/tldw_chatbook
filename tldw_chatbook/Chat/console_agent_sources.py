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
            or self.controller._disposed
            or self.runtime._disposed
            or self.store.persistence is not self.persistence
            or self.bridge._db is not self.databases[0]
            or self.persistence.db is not self.databases[1]
            or getattr(self.app, "workspace_registry_service", None)
            is not self.registry
            or getattr(self.registry, "db", None) is not self.databases[2]
            or tuple(getattr(db, "db_path", None) for db in self.databases)
            != self.database_paths
            or getattr(self.app, "app_config", None) is not self.app_config
            or config.current_config_identity() != self.config_identity
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
            self.sources = tuple(dict.fromkeys((*self.sources, *extra)))
        self.check_memory()

    @contextmanager
    def activate(self):
        self.check_memory()
        with ExitStack() as stack:
            stack.enter_context(execution(sources=self.sources))
            for db in dict.fromkeys(db for db in self.databases if db is not None):
                stack.enter_context(operation_owned_connection(db))
            if self.root_identity is not None:
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
