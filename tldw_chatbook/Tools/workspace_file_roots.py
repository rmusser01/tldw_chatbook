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
from typing import Iterator

from loguru import logger

_RUN_WORKSPACE_ID: ContextVar[str | None] = ContextVar(
    "tldw_run_workspace_id", default=None
)


def _default_registry_factory():
    from tldw_chatbook.config import get_workspaces_db_path
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    return LocalWorkspaceRegistryService(
        WorkspaceDB(get_workspaces_db_path(), client_id="file-tools")
    )


#: Test seam: monkeypatch with a factory returning a prepared registry.
_registry_factory = _default_registry_factory


@contextmanager
def run_workspace(workspace_id: str | None) -> Iterator[None]:
    """Bind the current run's workspace for the duration of a tool call."""
    token = _RUN_WORKSPACE_ID.set(workspace_id)
    try:
        yield
    finally:
        _RUN_WORKSPACE_ID.reset(token)


def current_run_workspace_id() -> str | None:
    return _RUN_WORKSPACE_ID.get()


def allowed_file_roots(*, write: bool, sandbox_root: Path) -> tuple[Path, ...]:
    """Sandbox root plus the run's workspace folder roots, existing-only.

    Fail-safe: any registry failure degrades to sandbox-only rather than
    widening access.
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
            if folder.is_dir():
                roots.append(folder)
    except Exception:
        logger.opt(exception=True).warning(
            "Workspace folder roots unavailable; file tools confined to sandbox"
        )
        return (sandbox_root,)
    return tuple(roots)
