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
import json
import os
from pathlib import Path
import threading
from typing import Iterator

from loguru import logger

_RUN_WORKSPACE_ID: ContextVar[str | None] = ContextVar(
    "tldw_run_workspace_id", default=None
)
_RUN_FILE_SANDBOX_ROOT: ContextVar[Path | None] = ContextVar(
    "tldw_run_file_sandbox_root", default=None
)

#: The directory the app process was launched from, captured once at boot by
#: ``set_launch_cwd``. The workspace-context note (``workspace_context_note``)
#: expresses a workspace's folder roots relative to this so an agent is never
#: handed an absolute host path. ``None`` until boot records it, at which point
#: ``get_launch_cwd`` returns the captured value; before that it degrades to the
#: live process cwd.
_LAUNCH_CWD: str | None = None


def set_launch_cwd(path: str | os.PathLike[str] | None = None) -> None:
    """Record the app's launch directory once, at boot (first write wins).

    Intended to be called once, from single-threaded process startup. A later
    (sequential) re-entrant boot -- or a test that constructs a second app --
    is ignored, so the recorded launch location cannot move out from under an
    in-flight run. The set-once check is not internally locked; it relies on
    boot being single-threaded rather than guarding concurrent first calls.

    Args:
        path: Directory to record as the launch location; defaults to the
            current process working directory. Stored as an absolute path.
    """
    global _LAUNCH_CWD
    if _LAUNCH_CWD is not None:
        return
    _LAUNCH_CWD = os.path.abspath(str(path) if path is not None else os.getcwd())


def get_launch_cwd() -> str:
    """Return the recorded launch directory, or the live cwd if unset.

    Returns:
        The absolute directory recorded by ``set_launch_cwd``, or -- when boot
        never recorded one (e.g. in tests or a headless import) -- the current
        process working directory.
    """
    if _LAUNCH_CWD is not None:
        return _LAUNCH_CWD
    return os.path.abspath(os.getcwd())


#: Fixed scaffolding for the workspace-context note. Kept as module-level
#: constants (rather than the internal-prompt registry) because the note is
#: assembled from live per-run values around them; moving the wording into the
#: registry is a possible follow-up. Mirrors ``agent_service``'s own
#: ``RUN_LOG_PROMPT_SECTION`` precedent for a conditionally-appended section.
_NOTE_HEADER = "Note: This session is NOT running in the default workspace."
_NOTE_UNAVAILABLE = _NOTE_HEADER + " (Workspace details are currently unavailable.)"
_NOTE_NO_ROOTS = (
    "This workspace has no filesystem roots bound; file tools are limited to "
    "the app sandbox."
)


def _relativize_root(folder: Path, launch: Path) -> tuple[str, bool]:
    """Return ``(display, is_outside)`` for one root relative to ``launch``.

    In-tree roots render as their relative subpath; a root equal to the launch
    directory renders as ``"."``; anything outside the launch tree (including a
    different drive on Windows, where ``relpath`` raises) renders as its leaf
    folder name only -- never a ``../..`` traversal chain -- so the note never
    reveals how the host's directories sit above the launch point.
    """
    try:
        rel = os.path.relpath(folder, launch)
    except ValueError:
        return folder.name, True
    if rel == os.curdir:
        return ".", False
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return folder.name, True
    return rel.replace(os.sep, "/"), False


def workspace_context_note(
    workspace_id: str | None,
    *,
    launch_cwd: str | os.PathLike[str] | None = None,
    registry=None,
) -> str:
    """Build the agent system-prompt note for a non-default workspace.

    Returns an empty string for the default workspace (or when no workspace is
    bound), so the common case adds nothing to the prompt. For a non-default
    workspace it names the workspace and lists its filesystem roots expressed
    *relative to the launch directory* -- absolute host paths are never
    emitted. Roots are filtered exactly as ``allowed_file_roots`` filters them
    (existing, non-symlink, non-drifted), so the note reflects what the file
    tools will actually honor rather than what is merely configured.

    Args:
        workspace_id: The run's workspace id, or ``None`` for none.
        launch_cwd: Directory to relativize roots against; defaults to
            ``get_launch_cwd()``.
        registry: Workspace registry to read from; defaults to the shared
            process registry ``allowed_file_roots`` uses.

    Returns:
        The note text, or ``""`` when no note applies. On any registry failure
        (or an unknown workspace id) it degrades to a one-line note that still
        tells the agent it is not in the default workspace.
    """
    from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

    if not workspace_id or workspace_id == DEFAULT_WORKSPACE_ID:
        return ""
    launch = Path(
        str(launch_cwd) if launch_cwd is not None else get_launch_cwd()
    ).resolve()
    if registry is None:
        try:
            registry = _registry_factory()
        except Exception:
            return _NOTE_UNAVAILABLE
    try:
        record = registry.get_workspace(workspace_id)
        if record is None:
            return _NOTE_UNAVAILABLE
        name = " ".join(str(record.name).split())[:120] or workspace_id
        root_lines: list[str] = []
        for binding in registry.list_folder_bindings(workspace_id):
            folder = Path(binding.locator)
            if not folder.is_dir():
                continue
            if folder.is_symlink() or folder.resolve() != folder:
                continue
            display, outside = _relativize_root(folder, launch)
            # Collapse whitespace in the rendered path exactly as the workspace
            # name is collapsed above: a bound folder whose leaf name contains
            # a newline (legal on POSIX) would otherwise splice a fake prompt
            # section into the note the agent reads as instructions.
            display = " ".join(display.split())
            read_only = str(binding.metadata.get("access", "ro")) != "rw"
            tags: list[str] = []
            if outside:
                tags.append("outside the launch directory")
            if read_only:
                tags.append("read-only")
            suffix = f" ({', '.join(tags)})" if tags else ""
            root_lines.append(f"  - {display}{suffix}")
    except Exception:
        logger.opt(exception=True).debug("workspace_context_note: registry unavailable")
        return _NOTE_UNAVAILABLE
    launch_label = f"{launch.name}/" if launch.name else (launch.anchor or "/")
    lines = [
        _NOTE_HEADER,
        # Render the (user-controlled) workspace name as a JSON string literal:
        # it delimits the value as data and escapes embedded quotes/backslashes/
        # control chars, so a crafted name cannot break out of the quoted field
        # to add instruction-like text. Belt-and-suspenders with the
        # whitespace-collapse above; ``ensure_ascii=False`` keeps unicode names
        # readable.
        f"Active workspace: {json.dumps(name, ensure_ascii=False)}",
        f"Launched from: {launch_label}",
    ]
    if root_lines:
        lines.append("Workspace file roots (relative to the launch directory):")
        lines.extend(root_lines)
    else:
        lines.append(_NOTE_NO_ROOTS)
    return "\n".join(lines)


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
    shared across threads/calls. ``WorkspaceDB`` (task-3011) holds one
    ``sqlite3`` connection per THREAD rather than opening a fresh one per
    operation -- see ``WorkspaceDB.connection`` / ``WorkspaceDB.transaction``
    -- so sharing the service object shares no connection ACROSS threads
    (each thread gets and keeps its own), which is exactly what makes it
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


def folder_binding_roots(workspace_id: str | None) -> tuple[Path, ...]:
    """Return a workspace's bound folder roots (all access levels).

    TASK-1971: the Agent Change Review tracker's root list. Unlike
    ``allowed_file_roots`` this includes READ-ONLY bindings (a script can
    write into an ro root -- the tools cannot, but tracking is about what
    happened on disk, not what tools were permitted) and never appends the
    sandbox root (app-managed scratch; retained script outputs live there
    deliberately and would be pure review noise).

    Args:
        workspace_id: The run's workspace, or ``None`` for none.

    Returns:
        Existing, resolved root directories; empty when the workspace has
        no usable bindings or the registry is unavailable.
    """
    from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

    if not workspace_id or workspace_id == DEFAULT_WORKSPACE_ID:
        return ()
    # TASK-1979: this function exists solely as the change-review tracker's
    # root source, so the enable gates live HERE — one choke point, read
    # fresh per turn, no restart needed.
    from tldw_chatbook.Workspaces.change_bounds import (
        change_review_enabled_globally,
    )

    if not change_review_enabled_globally():
        return ()
    roots: list[Path] = []
    try:
        registry = _registry_factory()
        if not registry.change_review_enabled(workspace_id):
            return ()
        for binding in registry.list_folder_bindings(workspace_id):
            folder = Path(binding.locator)
            if not folder.is_dir():
                continue
            if folder.is_symlink() or folder.resolve() != folder:
                # Same drift exclusion `allowed_file_roots` applies: a
                # binding that no longer resolves to its bound path is
                # stale config, and tracking its TARGET could snapshot an
                # unintended (potentially huge) tree.
                logger.warning(
                    "folder_binding_roots: excluding drifted root {!r}",
                    binding.locator,
                )
                continue
            roots.append(folder)
    except Exception:
        logger.opt(exception=True).debug("folder_binding_roots: registry unavailable")
        return ()
    return tuple(roots)


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


@contextmanager
def run_file_sandbox(root: Path | None) -> Iterator[None]:
    """Bind one run's private file-tool sandbox without changing global config."""

    resolved = Path(root).resolve() if root is not None else None
    token = _RUN_FILE_SANDBOX_ROOT.set(resolved)
    try:
        yield
    finally:
        _RUN_FILE_SANDBOX_ROOT.reset(token)


def current_run_sandbox_root() -> Path | None:
    """Return the private sandbox root bound to the current run, if any."""

    return _RUN_FILE_SANDBOX_ROOT.get()


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
        workspace_id = current_run_workspace_id()
        from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

        if workspace_id == DEFAULT_WORKSPACE_ID:
            return tuple(roots)
        registry = _registry_factory()
        if workspace_id is None:
            active = registry.get_active_workspace()
            workspace_id = active.workspace_id if active is not None else None
        if workspace_id is None or workspace_id == DEFAULT_WORKSPACE_ID:
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
