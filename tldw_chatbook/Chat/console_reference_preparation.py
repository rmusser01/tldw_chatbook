"""Original explicit-reference authority and finite worker/host custody."""

from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import ExitStack, asynccontextmanager, contextmanager, nullcontext
from pathlib import Path

from tldw_chatbook.Agents.activation import _guard, execution
from tldw_chatbook.Agents.execution_capacity import CapacityRefused, WorkOrigin
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.Tools import workspace_file_roots as roots_api
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Utils.sensitive_paths import (
    merge_sensitive_context,
    resolve_sensitive_context,
)
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint, WorktreeScope

from .console_preparation_reads import (
    _READS_LOCK,
    ConsolePreparationRead,
    _observe_locked,
    _retire,
    run_preparation_read,
)


def _run_scope():
    return (
        roots_api.current_run_workspace_id(),
        roots_api._RUN_WORKSPACE_READ_BINDING_IDS.get(),
        roots_api._RUN_WORKSPACE_BINDING_AUTHORITY.get(),
        roots_api.current_run_sandbox_root(),
    )


def _binding_selection(registry, workspace):
    if registry is None or workspace is None:
        return ()
    return tuple(
        (
            b.binding_id,
            b.workspace_id,
            b.binding_kind,
            b.locator,
            b.label,
            {k: v for k, v in b.metadata.items() if k != "exclusions"},
        )
        for b in registry.list_folder_bindings(workspace)
    )


class ReferenceAuthority:
    """One frozen existing reference selection; live changes can only refuse."""

    def __init__(self) -> None:
        from tldw_chatbook import config
        from tldw_chatbook.Tools.file_operation_tools import _sandbox_config_path

        self.scope = _run_scope()
        snapshot = config.get_runtime_config_snapshot()
        self.config_generation = snapshot.generation
        self.config_path = config._get_effective_config_path()
        self.sandbox = self.scope[3] or _sandbox_config_path(snapshot.values)
        self.factory = roots_api._registry_factory
        self.registry = self.factory()
        self.database = getattr(self.registry, "db", None)
        with (
            operation_owned_connection(self.database)
            if self.database is not None
            else nullcontext()
        ):
            self._capture()

    def _capture(self):
        self.workspace = self.scope[0]
        if self.workspace is None:
            active = self.registry.get_active_workspace()
            self.workspace = active.workspace_id if active is not None else None
        self.selection = (self.registry, self.workspace)
        self.bindings = _binding_selection(*self.selection)
        self.registry_generation = getattr(self.registry, "_mutation_generation", None)
        sandbox = self.sandbox.resolve()
        self.roots = tuple(
            dict.fromkeys(
                root
                for root in roots_api.allowed_file_roots(
                    write=False, sandbox_root=sandbox, _selection=self.selection
                )
                if root.is_dir()
            )
        )
        self.chains = tuple(capture_directory_chain(root) for root in self.roots)
        if tuple(chain.canonical_root for chain in self.chains) != self.roots:
            raise RuntimeError("reference root changed")
        self.launch_cwd = roots_api.get_launch_cwd()
        self.cwd_chain = capture_directory_chain(Path(self.launch_cwd))
        self.cwd = self.cwd_chain.canonical_root
        self.executable = shutil.which("git")
        self.executable_identity = self._executable_identity()
        self.environment = {
            k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")
        }
        self.environment["GIT_TERMINAL_PROMPT"] = "0"
        self.context = resolve_sensitive_context()
        self.exclusions = roots_api.current_folder_binding_exclusions(
            _selection=self.selection
        )
        self.context = self._merge(self.exclusions)
        self.check()

    def _merge(self, paths):
        return merge_sensitive_context(
            self.context,
            extra_files=tuple(p for p in paths if p.is_file()),
            extra_dirs=tuple(p for p in paths if not p.is_file()),
        )

    def _executable_identity(self):
        if self.executable is None:
            return None
        value = os.stat(self.executable)
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
        )

    def check_memory(self, *, registry: bool = True) -> None:
        from tldw_chatbook import config

        if (
            roots_api._registry_factory is not self.factory
            or _run_scope() != self.scope
            or config.get_runtime_config_generation() != self.config_generation
            or getattr(self.registry, "db", None) is not self.database
            or registry
            and getattr(self.registry, "_mutation_generation", None)
            != self.registry_generation
        ):
            raise RuntimeError("original reference selection changed")

    def check(self) -> None:
        """Revalidate original selection with its database handle owned on this worker."""
        with (
            operation_owned_connection(self.database)
            if self.database is not None
            else nullcontext()
        ):
            self._check()

    def _check(self):
        from tldw_chatbook import config

        self.check_memory(registry=False)
        generation = getattr(self.registry, "_mutation_generation", None)
        if self.factory() is not self.registry:
            raise RuntimeError("reference registry source changed")
        if config._get_effective_config_path() != self.config_path:
            raise RuntimeError("reference configuration changed")
        if self.scope[0] is None:
            active = self.registry.get_active_workspace()
            if (active.workspace_id if active is not None else None) != self.workspace:
                raise RuntimeError("reference workspace selection changed")
        if _binding_selection(*self.selection) != self.bindings:
            raise RuntimeError("reference folder selection changed")
        live_roots = tuple(
            dict.fromkeys(
                root
                for root in roots_api.allowed_file_roots(
                    write=False,
                    sandbox_root=self.sandbox.resolve(),
                    _selection=self.selection,
                )
                if root.is_dir()
            )
        )
        if live_roots != self.roots or any(
            capture_directory_chain(root) != chain
            for root, chain in zip(self.roots, self.chains, strict=True)
        ):
            raise RuntimeError("reference root identity changed")
        if (
            roots_api.get_launch_cwd() != self.launch_cwd
            or capture_directory_chain(Path(self.launch_cwd)) != self.cwd_chain
        ):
            raise RuntimeError("reference launch directory changed")
        live = roots_api.current_folder_binding_exclusions(_selection=self.selection)
        if not set(live).issubset(self.exclusions):
            # An already-read body cannot be filtered retroactively at publication.
            raise RuntimeError("reference exclusions narrowed during expansion")
        if getattr(self.registry, "_mutation_generation", None) != generation:
            raise RuntimeError("reference selection changed during validation")
        self.registry_generation = generation
        self.check_memory()

    def check_git(self) -> None:
        self.check()
        if self._executable_identity() != self.executable_identity:
            raise RuntimeError("reference Git executable changed")


def _session_identity(session):
    return tuple(
        getattr(session, key, None)
        for key in (
            "incarnation_id",
            "conversation_binding_revision",
            "persisted_conversation_id",
            "workspace_id",
            "identity_revision",
            "settings_revision",
            "generation_settings_revision",
            "ephemeral",
        )
    )


class _ReferenceRead:
    required_writes = ()

    def __init__(self, controller, session, origin, text, candidates):
        from . import console_references as refs
        from .console_workflow_runs import native_workflow_qualified

        self.controller, self.store, self.session = (
            controller,
            controller.store,
            session,
        )
        self.runtime = getattr(controller, "_hooks_v2_runtime", None)
        self.app = getattr(controller, "app", None)
        self.app_runtime = getattr(self.app, "console_runtime", None)
        self.app_config = getattr(self.app, "app_config", None)
        self.persistence = getattr(self.store, "persistence", None)
        self.database = getattr(self.persistence, "db", None)
        self.registry = getattr(self.app, "workspace_registry_service", None)
        self.registry_db = getattr(self.registry, "db", None)
        self.identity = _session_identity(session)
        self.sources = _guard.captured_sources(None)
        self.native = self.runtime is not None and native_workflow_qualified()
        self.capacity = getattr(self.runtime, "_execution_capacity", None)
        self.activity = getattr(self.runtime, "_worktree_activity", None)
        self.origin = WorkOrigin.AUTOMATIC if origin.automatic else WorkOrigin.MANUAL
        self.text, self.authority = text, None
        self.candidates = candidates
        self.owner = self.operation = self.claim = None
        self.callbacks = refs.reference_callbacks()
        self.opaque = self.callbacks != refs._STOCK_REFERENCE_CALLBACKS or any(
            c.token.lower() in refs._SPECIAL_TOKENS for c in candidates
        )

    def check_memory(self):
        from . import console_references as refs

        c, r = self.controller, self.runtime
        if (
            c.store is not self.store
            or getattr(c, "app", None) is not self.app
            or getattr(c, "_hooks_v2_runtime", None) is not r
            or getattr(self.app, "console_runtime", None) is not self.app_runtime
            or self.store._sessions.get(self.session.id) is not self.session
            or _session_identity(self.session) != self.identity
            or c._disposed
            or c._shutdown_requested.is_set()
            or getattr(self.app, "app_config", None) is not self.app_config
            or getattr(self.store, "persistence", None) is not self.persistence
            or getattr(self.persistence, "db", None) is not self.database
            or getattr(self.app, "workspace_registry_service", None)
            is not self.registry
            or getattr(self.registry, "db", None) is not self.registry_db
            or refs.reference_callbacks() != self.callbacks
            or r is not None
            and (
                r._disposed
                or self.session.id in r._admission_fenced_sessions
                or r._chat_controller is not c
                or r._chat_store is not self.store
                or r._app is not self.app
                or self.capacity is not None
                and r._execution_capacity is not self.capacity
                or self.activity is not None
                and r._worktree_activity is not self.activity
            )
        ):
            raise RuntimeError("original reference source changed")
        if self.authority is not None:
            self.authority.check_memory()

    def check(self):
        self.check_memory()
        if _guard.captured_sources(None) != self.sources:
            raise RuntimeError("reference activation source changed")
        if self.authority is not None:
            self.authority.check()
        self.check_memory()

    @contextmanager
    def activate(self):
        with ExitStack() as stack:
            stack.enter_context(execution(sources=self.sources))
            if self.owner is not None:
                stack.enter_context(self.owner.activate())
            for db in dict.fromkeys(
                (
                    self.database,
                    self.registry_db,
                    getattr(self.authority, "database", None),
                )
            ):
                if db is not None:
                    stack.enter_context(operation_owned_connection(db))
            yield

    def prepare(self):
        with self.activate():
            self.check()
            self.authority = ReferenceAuthority()
            self.check()
            if not self.native or not self.authority.roots:
                return
            self.capacity = self.capacity or self.runtime.execution_capacity
            self.activity = self.activity or self.runtime.worktree_activity
            self.check()
            footprint = (
                WorktreeFootprint(opaque=True)
                if self.opaque
                else WorktreeFootprint(
                    reads=tuple(
                        WorktreeScope(c.canonical_root, True, c)
                        for c in self.authority.chains
                    )
                )
            )
            with self.activity.prelaunch(footprint=footprint) as claim:
                if claim is None:
                    raise CapacityRefused("reference_activity_busy")
                owner = self.capacity.begin_execution(
                    origin=self.origin,
                    conversation_id=self.session.persisted_conversation_id
                    or self.session.id,
                )
                try:
                    operation = owner.reserve_tool()
                    try:
                        self.check_memory()
                        claim.bind_owner(owner)
                    except BaseException:
                        operation.finish()
                        raise
                except BaseException:
                    owner.finish_root()
                    raise
                self.owner, self.operation, self.claim = owner, operation, claim

    def expand(self):
        from . import console_references as refs

        with self.activate():
            self.check()
            if not self.authority.roots:
                return refs.ReferenceExpansion(
                    self.text,
                    [
                        refs.ReferenceRecord(
                            c.raw,
                            "refused",
                            False,
                            "no allowed workspace roots are configured",
                        )
                        for c in self.candidates
                    ],
                )
            result = refs.expand_references(
                self.text,
                resolve=refs.build_console_reference_resolver(authority=self.authority),
                git_runner=refs.build_console_git_runner(
                    authority=self.authority, producer=self if self.native else None
                ),
            )
            self.check()
            return result

    def finish(self):
        if self.operation is not None:
            # Availability listeners inside finish must return before root/claim release.
            with self.activate():
                self.operation.finish()
                self.owner.finish_root()
                self.claim.settle(durable=True)


@asynccontextmanager
async def prepare_reference_expansion(controller, session, origin, text, candidates):
    """Retain the original worker and native operation through host assignment."""
    request = _ReferenceRead(controller, session, origin, text, candidates)
    read = ConsolePreparationRead(
        controller,
        request.expand,
        request,
        asyncio.current_task(),
        session.id,
        asyncio.get_running_loop().create_future(),
    )
    with _READS_LOCK:
        _observe_locked(read, controller._preparation_reads)
        if request.runtime is not None:
            _observe_locked(read, request.runtime._preparation_reads)
    local_reads = set()

    async def run(callback, *, check=True):
        return await run_preparation_read(
            callback,
            creator=controller,
            session_id=session.id,
            reads=local_reads,
            require_current=request.check_memory if check else lambda: None,
            source=request,
        )

    try:
        await run(request.prepare)
        result = await run(request.expand)
        request.check_memory()
        yield result
        request.check_memory()
    finally:
        try:
            await run(request.finish, check=False)
        finally:
            _retire(read)
