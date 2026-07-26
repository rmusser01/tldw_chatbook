"""Run-bound workspace folder roots for the agent file tools.

Spec: Docs/superpowers/specs/2026-07-26-settings-workspaces-category-design.md §3.
The provider (`BuiltinToolProvider.invoke`) binds the run's workspace via
``run_workspace``; the file tools ask ``allowed_file_roots`` at call time.
Reads and writes are both confined to sandbox+roots (deliberate Codex
divergence, ADR-028) and stored binding status is never trusted — existence
is re-checked here on every call.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import threading
from typing import Iterator

from loguru import logger

_RUN_WORKSPACE_ID: ContextVar[str | None] = ContextVar(
    "tldw_run_workspace_id", default=None
)

#: Process-wide cache for the default registry service (see
#: ``_default_registry_factory``). Reset to ``None`` by tests that need a
#: fresh instance.
_default_registry_instance = None
_default_registry_lock = threading.Lock()


def _default_registry_factory():
    """Build, cache, and return the process-wide default workspace registry.

    Constructing a ``WorkspaceDB`` runs schema initialization (and logs) on
    every call, which is wasteful on the hot path of every tool
    invocation. This factory memoizes that construction at module scope:
    the first call builds the service and caches it in
    ``_default_registry_instance``; subsequent calls return the cached
    instance without touching the database again.

    Thread safety: the cached ``LocalWorkspaceRegistryService`` instance is
    shared across threads/calls, but ``WorkspaceDB`` opens a fresh
    ``sqlite3`` connection per operation (see ``WorkspaceDB.connection`` /
    ``WorkspaceDB.transaction``) rather than holding one open, so sharing
    the service object does not share any live connection state and is
    safe under concurrent tool calls.

    Returns:
        The cached (or newly constructed) ``LocalWorkspaceRegistryService``
        backed by the default workspaces database.
    """
    global _default_registry_instance
    if _default_registry_instance is not None:
        return _default_registry_instance
    with _default_registry_lock:
        if _default_registry_instance is None:
            from tldw_chatbook.config import get_workspaces_db_path
            from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
            from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

            _default_registry_instance = LocalWorkspaceRegistryService(
                WorkspaceDB(get_workspaces_db_path(), client_id="file-tools")
            )
    return _default_registry_instance


#: Test seam: monkeypatch with a factory returning a prepared registry.
_registry_factory = _default_registry_factory


@contextmanager
def run_workspace(workspace_id: str | None) -> Iterator[None]:
    """Bind the current run's workspace for the duration of a tool call.

    Sets a context-local workspace id that ``allowed_file_roots`` and
    ``current_run_workspace_id`` read for the lifetime of the ``with``
    block, then restores whatever was bound before (or unbound), so nested
    or sequential runs never leak each other's workspace binding.

    Args:
        workspace_id: Identifier of the workspace to bind for the run, or
            ``None`` to explicitly bind "no workspace" (``allowed_file_roots``
            then falls back to the active workspace).

    Yields:
        None. The wrapped block executes with ``workspace_id`` bound as the
        current run's workspace.
    """
    token = _RUN_WORKSPACE_ID.set(workspace_id)
    try:
        yield
    finally:
        _RUN_WORKSPACE_ID.reset(token)


def current_run_workspace_id() -> str | None:
    """Return the workspace id bound by the current ``run_workspace`` scope.

    Returns:
        The workspace id most recently bound via ``run_workspace`` for the
        current run/task, or ``None`` if no run has bound one.
    """
    return _RUN_WORKSPACE_ID.get()


def allowed_file_roots(*, write: bool, sandbox_root: Path) -> tuple[Path, ...]:
    """Sandbox root plus the run's workspace folder roots, existing-only.

    Fail-safe: any registry failure degrades to sandbox-only rather than
    widening access. Folder bindings are re-checked against the filesystem
    on every call rather than trusting stored status: a bound folder that
    has been deleted is dropped, and so is one whose path no longer
    resolves to itself -- for example because it was replaced by a symlink
    or the target of a mount after binding, which would otherwise silently
    widen the sandboxed root at enforcement time (ADR-028).

    Args:
        write: Whether the caller needs write access. When True, only
            folder bindings whose access metadata is ``"rw"`` are included;
            read-only bindings are omitted entirely from the result.
        sandbox_root: The tool's own sandbox root; always included first,
            regardless of ``write``.

    Returns:
        A tuple of existing, non-symlinked directories the current run may
        operate on: ``sandbox_root`` followed by zero or more bound
        workspace folders in binding order. Falls back to
        ``(sandbox_root,)`` alone if the workspace registry is unavailable,
        raises, or no workspace is bound.
    """
    roots: list[Path] = [sandbox_root]
    try:
        registry = _registry_factory()
        workspace_id = current_run_workspace_id()
        if workspace_id is None:
            active = registry.get_active_workspace()
            workspace_id = active.workspace_id if active is not None else None
        if workspace_id is None:
            return tuple(roots)
        for binding in registry.list_folder_bindings(workspace_id):
            if write and str(binding.metadata.get("access", "ro")) != "rw":
                continue
            folder = Path(binding.locator)
            if not folder.is_dir():
                continue
            if folder.is_symlink() or folder.resolve() != folder:
                logger.warning(
                    "Workspace folder root {!r} no longer resolves to its "
                    "bound path (symlink or mount drift); excluding from "
                    "allowed roots",
                    binding.locator,
                )
                continue
            roots.append(folder)
    except Exception:
        logger.opt(exception=True).warning(
            "Workspace folder roots unavailable; file tools confined to sandbox"
        )
        return (sandbox_root,)
    return tuple(roots)
