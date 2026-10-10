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
import stat
import threading
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from loguru import logger

from tldw_chatbook.Tools.remote_root_types import LocalRoot, RemoteRoot

if TYPE_CHECKING:
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

_RUN_WORKSPACE_ID: ContextVar[str | None] = ContextVar(
    "tldw_run_workspace_id", default=None
)
_RUN_WORKSPACE_READ_BINDING_IDS: ContextVar[frozenset[str] | None] = ContextVar(
    "tldw_run_workspace_read_binding_ids", default=None
)
_RUN_WORKSPACE_WRITE_BINDING_IDS: ContextVar[frozenset[str] | None] = ContextVar(
    "tldw_run_workspace_write_binding_ids", default=None
)
_RUN_WORKSPACE_BINDING_AUTHORITY: ContextVar[tuple[Any, ...] | None] = ContextVar(
    "tldw_run_workspace_binding_authority", default=None
)
_INHERIT_BINDING_MAXIMUM = object()
_RUN_FILE_SANDBOX_ROOT: ContextVar[Path | None] = ContextVar(
    "tldw_run_file_sandbox_root", default=None
)

#: The directory the app process was launched from, captured once at boot by
#: ``set_launch_cwd`` (Console ``@`` references resolve against it). ``None``
#: until boot records it, at which point ``get_launch_cwd`` returns the
#: captured value; before that it degrades to the live process cwd.
#:
#: TASK-33940.1: the workspace-context note deliberately no longer uses it. The
#: note once listed folders relative to this directory, a coordinate system no
#: file tool resolves (fs_* paths are relative to the bound folder, the
#: read_file family's to private scratch), so agents asked for paths that
#: could not exist.
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
    "this chat's private scratch space."
)
#: Local-folder section header (TASK-33940.1). It states the addressing rule in
#: the tools' own terms -- the alias plus a path relative to that folder --
#: because that is the only form the fs_*, git_* and virtual_cli tools resolve.
_NOTE_LOCAL_HEADER = (
    "Workspace folders (reachable with the fs_*, git_* and virtual_cli tools — "
    'pass root_alias "<alias>" and a path relative to that folder; "." is the '
    "folder itself):"
)
_NOTE_NO_PATH_TOOLS = (
    "This workspace has bound folders, but no fs_*, git_* or virtual_cli "
    "tools are available in this run."
)
#: Remote-root section header (Phase 4a, spec "Model-facing surface"):
#: the alias->URI mapping the model addresses with root_alias, plus the
#: fs_*-only rule (git tools are not yet supported on SSH bindings).
_NOTE_REMOTE_HEADER = (
    "Remote workspace roots (SSH bindings; reachable with the fs_* tools "
    'only — pass root_alias "<alias>" and a path relative to that root):'
)
#: The explicit degradation line (spec: "Degraded semantics" — the note
#: says which remote bindings composition excluded this run).
_NOTE_REMOTE_UNREACHABLE = "remote binding unreachable — excluded this run"


def _iter_valid_folder_bindings(
    bindings: Iterable[WorkspaceRuntimeBinding],
) -> Iterator[tuple[WorkspaceRuntimeBinding, Path]]:
    """Yield existing folder bindings whose stored path has not drifted."""
    for binding in bindings:
        kind = (
            getattr(getattr(binding, "binding_kind", None), "value", None)
            or str(getattr(binding, "binding_kind", ""))
        )
        if kind == "ssh-filesystem":
            # Phase 4a: an ssh binding's root lives on the REMOTE host;
            # it is never a family-B laptop root. Skipping by KIND (not
            # by the existence check below) also closes the collision
            # where ``Path("ssh://host/path")`` -- a RELATIVE laptop
            # path -- happens to exist under the process cwd and would
            # otherwise admit a "remote" binding as a local root.
            continue
        folder = Path(binding.locator)
        if not folder.is_dir():
            continue
        if folder.is_symlink() or folder.resolve() != folder:
            logger.warning(
                "Workspace folder binding excluded because its path no longer "
                "resolves to itself (symlink or mount drift)"
            )
            continue
        yield binding, folder


def _list_remote_bindings(registry: Any, workspace_id: str) -> tuple[Any, ...]:
    """The workspace's ssh-filesystem binding rows (service or fallback).

    The real service exposes ``list_ssh_bindings``; fake registries and
    older shapes fall back to filtering ``list_runtime_bindings`` by
    kind. Any failure degrades to "no remote rows" (fail-safe: the note
    simply lists no remote roots).
    """
    try:
        listing = getattr(registry, "list_ssh_bindings", None)
        if callable(listing):
            return tuple(listing(workspace_id))
        return tuple(
            binding
            for binding in registry.list_runtime_bindings(workspace_id)
            if (
                (
                    getattr(
                        getattr(binding, "binding_kind", None), "value", None
                    )
                    or str(getattr(binding, "binding_kind", ""))
                )
                == "ssh-filesystem"
            )
        )
    except Exception:  # noqa: BLE001 - remote rows are note-only extras
        logger.opt(exception=True).debug(
            "workspace_context_note: remote bindings unavailable"
        )
        return ()


def _remote_note_lines(
    registry: Any,
    workspace_id: str,
    *,
    authority_by_id: dict[str, Any] | None,
    status_cache: Any,
    path_tool_aliases: frozenset[str] | None = None,
) -> tuple[list[str], list[str], int]:
    """Render the remote-root lines: admitted aliases and dropped ones.

    Admitted (READY / STALE_IDENTITY / cold-optimistic) remote bindings
    render as the alias -> display-URI mapping with ``[ssh]`` and the
    ro/rw tag; BLOCKED/MISSING bindings render the explicit degradation
    line instead. Everything is whitespace-collapsed exactly like the
    local display paths: a crafted locator cannot splice a fake prompt
    section into the note. When ``path_tool_aliases`` is given, a usable
    binding the run's fs_* tools did not admit is left out: the note never
    offers an alias the tools would reject; such bindings are still counted
    (third return value) so a remote-only workspace is never described as
    having no roots (Qodo #3 on PR #2975).
    """
    from tldw_chatbook.Tools.remote_root_types import RemoteRoot, display_uri

    admitted: list[str] = []
    dropped: list[str] = []
    unaddressable = 0
    for binding in _list_remote_bindings(registry, workspace_id):
        binding_id = str(getattr(binding, "binding_id", ""))
        if not binding_id:
            continue
        if authority_by_id is not None and binding_id not in authority_by_id:
            continue
        try:
            from tldw_chatbook.Tools.remote_binding_locator import (
                locator_string,
                parse_remote_locator,
            )

            parsed = parse_remote_locator(str(binding.locator))
            descriptor = RemoteRoot(
                alias=binding_id,
                canonical_locator=locator_string(parsed),
                root=parsed.path,
                binding_id=binding_id,
            )
            uri = display_uri(descriptor)
        except Exception:  # noqa: BLE001 - unrenderable row stays out of the note
            logger.opt(exception=True).debug(
                "workspace_context_note: remote binding locator unparseable"
            )
            continue
        frozen = (
            authority_by_id.get(binding_id) if authority_by_id is not None else None
        )
        read_only = (
            not bool(frozen.allow_write)
            if frozen is not None
            else str(
                (getattr(binding, "metadata", None) or {}).get("access", "ro")
            )
            != "rw"
        )
        state: str | None = None
        try:
            cached = status_cache.status(binding_id)
            state = str(getattr(cached, "state", cached))
        except Exception:  # noqa: BLE001 - unreadable cache drops the row
            state = None
        alias = " ".join(binding_id.split())
        uri = " ".join(uri.split())
        if state in {"BLOCKED", "MISSING"} or state is None:
            dropped.append(f"  - {alias}: {_NOTE_REMOTE_UNREACHABLE}")
        elif path_tool_aliases is not None and binding_id not in path_tool_aliases:
            unaddressable += 1
        else:
            admitted.append(
                f"  - {alias} → {uri} [ssh, {'ro' if read_only else 'rw'}]"
            )
    return admitted, dropped, unaddressable


def workspace_context_note(
    workspace_id: str | None,
    *,
    registry=None,
    binding_authority: Iterable[Any] | None = None,
    status_cache: Any = None,
    path_tool_aliases: Iterable[str] | None = None,
    scratch_relative_tools: Iterable[str] = (),
) -> str:
    """Build the agent system-prompt note for a non-default workspace.

    Returns an empty string for the default workspace (or when no workspace is
    bound), so the common case adds nothing to the prompt. For a non-default
    workspace it names the workspace and tells the agent how to reach each
    bound folder in the terms its tools resolve (TASK-33940.1): the
    ``root_alias`` the fs_*/git_* tools accept, the folder's own name, and its
    access, with paths relative to that folder. Absolute host paths are never
    emitted. Roots are filtered exactly as ``allowed_file_roots`` filters them
    (existing, non-symlink, non-drifted), so the note reflects what the file
    tools will actually honor rather than what is merely configured.

    Args:
        workspace_id: The run's workspace id, or ``None`` for none.
        registry: Workspace registry to read from; defaults to the shared
            process registry ``allowed_file_roots`` uses.
        binding_authority: Optional frozen run authority — when present,
            only bindings it names render (local and remote alike).
        status_cache: Optional :class:`RemoteBindingStatusCache` backing
            the remote-root lines; ``None`` resolves the process-wide
            singleton (test seam for injection).
        path_tool_aliases: The ``root_alias`` values this run's fs_*/git_*
            tools advertise. Only folders they admit are offered; an empty
            collection means the run has no such tools. ``None`` (no tool
            information) offers every authorized folder under its binding id,
            which is the alias Console runs admit it by.
        scratch_relative_tools: Names of this run's tools whose relative
            paths resolve in the chat's private scratch space (read_file,
            list_directory, write_file); the note says so when any are given.

    Returns:
        The note text, or ``""`` when no note applies. On any registry failure
        (or an unknown workspace id) it degrades to a one-line note that still
        tells the agent it is not in the default workspace.
    """
    from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

    if not workspace_id or workspace_id == DEFAULT_WORKSPACE_ID:
        return ""
    aliases = (
        frozenset(str(alias) for alias in path_tool_aliases)
        if path_tool_aliases is not None
        else None
    )
    if registry is None:
        try:
            registry = _registry_factory()
        except Exception:
            return _NOTE_UNAVAILABLE
    unaddressable_folders = 0
    try:
        record = registry.get_workspace(workspace_id)
        if record is None:
            return _NOTE_UNAVAILABLE
        name = " ".join(str(record.name).split())[:120] or workspace_id
        root_lines: list[str] = []
        authority_by_id = (
            {
                str(getattr(item, "binding_id", "")): item
                for item in binding_authority
            }
            if binding_authority is not None
            else None
        )
        for binding, folder in _iter_valid_folder_bindings(
            registry.list_folder_bindings(workspace_id)
        ):
            binding_id = str(getattr(binding, "binding_id", ""))
            frozen = (
                authority_by_id.get(binding_id)
                if authority_by_id is not None
                else None
            )
            if authority_by_id is not None and frozen is None:
                continue
            if frozen is not None and not _binding_matches_frozen_authority(
                folder, frozen
            ):
                continue
            if aliases is not None and binding_id not in aliases:
                # Authorized, but no path tool this run admits it: offering
                # its alias would only earn a "root_alias does not name a
                # root admitted for this run" refusal.
                unaddressable_folders += 1
                continue
            # Collapse whitespace in the rendered alias and folder name exactly
            # as the workspace name is collapsed above: a bound folder whose
            # leaf name contains a newline (legal on POSIX) would otherwise
            # splice a fake prompt section into the note the agent reads as
            # instructions.
            alias = " ".join(binding_id.split())
            label = " ".join(folder.name.split()) or alias
            read_only = (
                not bool(frozen.allow_write)
                if frozen is not None
                else str(binding.metadata.get("access", "ro")) != "rw"
            )
            access = "read-only" if read_only else "read-write"
            root_lines.append(f"  - {alias} → {label} [{access}]")
    except Exception:
        logger.opt(exception=True).debug("workspace_context_note: registry unavailable")
        return _NOTE_UNAVAILABLE
    lines = [
        _NOTE_HEADER,
        # Render the (user-controlled) workspace name as a JSON string literal:
        # it delimits the value as data and escapes embedded quotes/backslashes/
        # control chars, so a crafted name cannot break out of the quoted field
        # to add instruction-like text. Belt-and-suspenders with the
        # whitespace-collapse above; ``ensure_ascii=False`` keeps unicode names
        # readable.
        f"Active workspace: {json.dumps(name, ensure_ascii=False)}",
    ]
    # Remote-root lines (Phase 4a): alias -> URI mappings, the fs_*-only
    # rule, and the explicit degradation line for excluded bindings. The
    # singleton resolution failure degrades to no remote lines (the note
    # never fails because the cache is unavailable).
    remote_admitted: list[str] = []
    remote_dropped: list[str] = []
    remote_unaddressable = 0
    if status_cache is None:
        try:
            from tldw_chatbook.Tools.remote_binding_status import (
                get_remote_binding_status_cache,
            )

            status_cache = get_remote_binding_status_cache()
        except Exception:  # noqa: BLE001 - note-only extra, fail-soft
            status_cache = None
    if status_cache is not None:
        remote_admitted, remote_dropped, remote_unaddressable = _remote_note_lines(
            registry,
            workspace_id,
            authority_by_id=authority_by_id,
            status_cache=status_cache,
            path_tool_aliases=aliases,
        )
    if root_lines:
        lines.append(_NOTE_LOCAL_HEADER)
        lines.extend(root_lines)
    if remote_admitted or remote_dropped:
        lines.append(_NOTE_REMOTE_HEADER)
        lines.extend(remote_admitted)
        lines.extend(remote_dropped)
    has_folders = bool(root_lines or remote_admitted or remote_dropped)
    if not has_folders and (unaddressable_folders or remote_unaddressable):
        lines.append(_NOTE_NO_PATH_TOOLS)
    elif not has_folders:
        lines.append(_NOTE_NO_ROOTS)
        return "\n".join(lines)
    scratch_tools = sorted({str(tool) for tool in scratch_relative_tools})
    if scratch_tools:
        lines.append(
            f"Relative paths in {', '.join(scratch_tools)} resolve inside this "
            "chat's private scratch space, not in the workspace folders."
        )
    return "\n".join(lines)


def frozen_workspace_roots(
    workspace_id: str | None,
    binding_authority: Iterable[Any],
    *,
    registry=None,
) -> tuple[Path, ...]:
    """Return exact admitted roots that remain live without retargeting."""
    if not workspace_id:
        return ()
    try:
        binding_authority = tuple(binding_authority)
        if not binding_authority:
            return ()
        registry = registry or _registry_factory()
        live = {
            str(getattr(item, "binding_id", "")): item
            for item in registry.list_folder_bindings(workspace_id)
        }
        roots: list[Path] = []
        for frozen in binding_authority:
            binding = live.get(str(getattr(frozen, "binding_id", "")))
            if binding is None:
                continue
            root = Path(binding.locator)
            if (
                root.is_dir()
                and not root.is_symlink()
                and root.resolve() == root
                and _binding_matches_frozen_authority(root, frozen)
            ):
                roots.append(root)
        return tuple(roots)
    except Exception:
        logger.opt(exception=True).debug("Frozen workspace roots unavailable")
        return ()


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
        for _binding, folder in _iter_valid_folder_bindings(
            registry.list_folder_bindings(workspace_id)
        ):
            roots.append(folder)
    except Exception:
        logger.opt(exception=True).debug("folder_binding_roots: registry unavailable")
        return ()
    return tuple(roots)


@contextmanager
def run_workspace(
    workspace_id: str | None,
    *,
    read_binding_ids: Iterable[str] | None | object = _INHERIT_BINDING_MAXIMUM,
    write_binding_ids: Iterable[str] | None | object = _INHERIT_BINDING_MAXIMUM,
    binding_authority: Iterable[Any] | None | object = _INHERIT_BINDING_MAXIMUM,
) -> Iterator[None]:
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
    read_token = (
        None
        if read_binding_ids is _INHERIT_BINDING_MAXIMUM
        else _RUN_WORKSPACE_READ_BINDING_IDS.set(
            None
            if read_binding_ids is None
            else frozenset(str(value) for value in read_binding_ids)
        )
    )
    write_token = (
        None
        if write_binding_ids is _INHERIT_BINDING_MAXIMUM
        else _RUN_WORKSPACE_WRITE_BINDING_IDS.set(
            None
            if write_binding_ids is None
            else frozenset(str(value) for value in write_binding_ids)
        )
    )
    authority_token = (
        None
        if binding_authority is _INHERIT_BINDING_MAXIMUM
        else _RUN_WORKSPACE_BINDING_AUTHORITY.set(
            None if binding_authority is None else tuple(binding_authority)
        )
    )
    try:
        yield
    finally:
        if authority_token is not None:
            _RUN_WORKSPACE_BINDING_AUTHORITY.reset(authority_token)
        if write_token is not None:
            _RUN_WORKSPACE_WRITE_BINDING_IDS.reset(write_token)
        if read_token is not None:
            _RUN_WORKSPACE_READ_BINDING_IDS.reset(read_token)
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
    """Bind one run's private file-tool sandbox without changing global config.

    Args:
        root: Private sandbox root for the current run, or ``None`` to clear
            an inherited binding within the scope.

    Yields:
        None. The wrapped block executes with ``root`` as its sandbox binding.
    """

    resolved = Path(root).resolve() if root is not None else None
    token = _RUN_FILE_SANDBOX_ROOT.set(resolved)
    try:
        yield
    finally:
        _RUN_FILE_SANDBOX_ROOT.reset(token)


def current_run_sandbox_root() -> Path | None:
    """Return the private sandbox root bound to the current run, if any.

    Returns:
        The resolved sandbox root for the current run, or ``None`` when no
        sandbox is bound.
    """

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
        maximum_binding_ids = (
            _RUN_WORKSPACE_WRITE_BINDING_IDS.get()
            if write
            else _RUN_WORKSPACE_READ_BINDING_IDS.get()
        )
        frozen_authority = _RUN_WORKSPACE_BINDING_AUTHORITY.get()
        authority_by_id = (
            {
                str(getattr(item, "binding_id", "")): item
                for item in frozen_authority
            }
            if frozen_authority is not None
            else None
        )
        bindings = (
            binding
            for binding in registry.list_folder_bindings(workspace_id)
            if not write or str(binding.metadata.get("access", "ro")) == "rw"
        )
        for binding, folder in _iter_valid_folder_bindings(bindings):
            binding_id = str(getattr(binding, "binding_id", ""))
            if (
                maximum_binding_ids is not None
                and binding_id not in maximum_binding_ids
            ):
                continue
            frozen = (
                authority_by_id.get(binding_id)
                if authority_by_id is not None
                else None
            )
            if authority_by_id is not None and frozen is None:
                continue
            if write and frozen is not None and not bool(frozen.allow_write):
                continue
            if frozen is not None and not _binding_matches_frozen_authority(
                folder, frozen
            ):
                continue
            roots.append(folder)
    except Exception:
        logger.opt(exception=True).warning(
            "Workspace folder roots unavailable; file tools confined to sandbox"
        )
        return (sandbox_root,)
    return tuple(roots)


def current_folder_binding_exclusions() -> tuple[Path, ...]:
    """Absolute user-exclusion paths for the current run's folder bindings.

    Family-2 injection point (spec 2026-09-20): the builtin file tools fold
    these into their per-call sensitive context, mirroring what
    ``WorkspaceToolExecutor._call_context`` and the local provider's preflight
    already do for the other file-tool families. Enumerates exactly the
    bindings ``allowed_file_roots`` would admit for the current run workspace
    on the READ side (the superset: a read-admitted binding's exclusions apply
    to its writes too), so a binding dropped for path drift, narrowed run
    scopes, or a frozen-authority mismatch also drops its exclusions here.
    Mirrors ``allowed_file_roots`` exactly: with no run workspace bound, the
    ACTIVE workspace's bindings (and their exclusions) still apply.

    Exclusions frozen into the run's ``ConsoleProjectBindingSnapshot``
    authority are UNIONED with the live entries per admitted binding: a
    mid-run registry removal stays enforced for the rest of the run
    (family-1 high-water parity -- run authority never expands mid-run),
    while a mid-run addition applies on the next call.

    Returns:
        Resolved exclusion paths from the admitted bindings; ``()`` when no
        workspace resolves (no run workspace and no active workspace), for
        the default workspace, when no admitted binding carries exclusions,
        or when the registry is unavailable (consistent with
        ``allowed_file_roots`` degrading to sandbox-only there, which
        already makes every bound path unreachable). An individual entry
        that cannot be resolved (e.g. replaced by a symlink loop mid-run)
        is skipped with a warning while the resolvable rest stays enforced.
    """
    paths: list[Path] = []
    try:
        workspace_id = current_run_workspace_id()
        from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

        registry = _registry_factory()
        if workspace_id is None:
            # Fallback parity (Finding A, PR #2767): ``allowed_file_roots``
            # admits the ACTIVE workspace's bindings when no run workspace
            # is bound; the exclusions here must follow the same fallback or
            # the builtin tools reach those folders with ZERO exclusions.
            active = registry.get_active_workspace()
            workspace_id = active.workspace_id if active is not None else None
        if not workspace_id or workspace_id == DEFAULT_WORKSPACE_ID:
            return ()
        maximum_binding_ids = _RUN_WORKSPACE_READ_BINDING_IDS.get()
        frozen_authority = _RUN_WORKSPACE_BINDING_AUTHORITY.get()
        authority_by_id = (
            {
                str(getattr(item, "binding_id", "")): item
                for item in frozen_authority
            }
            if frozen_authority is not None
            else None
        )
        from tldw_chatbook.Workspaces.registry_service import (
            binding_exclusion_entries,
        )

        for binding, folder in _iter_valid_folder_bindings(
            registry.list_folder_bindings(workspace_id)
        ):
            binding_id = str(getattr(binding, "binding_id", ""))
            if (
                maximum_binding_ids is not None
                and binding_id not in maximum_binding_ids
            ):
                continue
            frozen = (
                authority_by_id.get(binding_id)
                if authority_by_id is not None
                else None
            )
            if authority_by_id is not None and frozen is None:
                continue
            if frozen is not None and not _binding_matches_frozen_authority(
                folder, frozen
            ):
                continue
            # Live entries first, then the run's frozen exclusion rels that
            # are no longer live (union; deduped). Frozen rels keep mid-run
            # removals enforced for this run exactly as family-1's
            # high-water provider does; live rels make additions apply on
            # the next call.
            rels = [entry.path for entry in binding_exclusion_entries(binding)]
            if frozen is not None:
                seen = set(rels)
                for rel in tuple(getattr(frozen, "exclusions", ()) or ()):
                    if isinstance(rel, str) and rel and rel not in seen:
                        seen.add(rel)
                        rels.append(rel)
            for rel in rels:
                try:
                    paths.append((folder / rel).resolve(strict=False))
                except Exception:  # noqa: BLE001 - per-entry isolation (final
                    # review Finding 2c): skip the unresolvable entry at
                    # warning, keep the resolvable rest. Collapsing to () here
                    # would fail OPEN (every exclusion dropped at once).
                    logger.warning(
                        "Workspace binding exclusion could not be resolved; "
                        "skipped while keeping the remaining exclusions"
                    )
    except Exception:
        logger.opt(exception=True).warning(
            "Workspace binding exclusions unavailable; treating as none"
        )
        return ()
    return tuple(paths)


def _binding_matches_frozen_authority(folder: Path, frozen: Any) -> bool:
    """Return whether one live binding is still the exact admitted root."""
    # Phase 3a (task 15) type boundary: this check lstats every path
    # component on the LAPTOP, so it admits LOCAL roots only. A RemoteRoot
    # in the frozen authority is a composition bug (remote authority is
    # validated client-side -- registry row + status cache -- inside the
    # executor, never here). The raise sits OUTSIDE the try below on
    # purpose: the except clause would otherwise swallow it into a silent
    # authority mismatch and just drop the binding.
    raw_root = getattr(frozen, "root", None)
    if isinstance(raw_root, RemoteRoot):
        raise TypeError(
            "remote root reached laptop-disk path: frozen-authority lstat check"
        )
    if raw_root is None:
        return False
    try:
        expected_root = (
            raw_root.path
            if isinstance(raw_root, LocalRoot)
            else Path(raw_root)
        )
        if folder != expected_root:
            return False
        from tldw_chatbook.Chat.console_project_instructions import (
            fingerprint_canonical_locator,
        )

        if fingerprint_canonical_locator(str(folder)) != str(
            frozen.locator_fingerprint
        ):
            return False
        identities: list[tuple[str, int, int, int]] = []
        for component in (*reversed(folder.parents), folder):
            value = os.lstat(component)
            if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
                return False
            identities.append(
                (str(component), value.st_dev, value.st_ino, value.st_mode)
            )
        return tuple(identities) == tuple(frozen.root_identity)
    except (AttributeError, OSError, TypeError, ValueError):
        return False


# Defining callbacks for the optional finite legacy run-log probe only.
_RUN_LOG_PROBE_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    (
        *(
            (globals(), name, globals()[name])
            for name in (
                "_default_registry_factory",
                "allowed_file_roots",
                "current_run_workspace_id",
                "current_run_sandbox_root",
                "_iter_valid_folder_bindings",
                "_binding_matches_frozen_authority",
            )
        ),
        (globals(), "_registry_factory", _default_registry_factory),
        (globals(), "_default_registry_lock", _default_registry_lock),
        *(
            (globals(), name, globals()[name])
            for name in (
                "_RUN_WORKSPACE_ID",
                "_RUN_WORKSPACE_READ_BINDING_IDS",
                "_RUN_WORKSPACE_WRITE_BINDING_IDS",
                "_RUN_WORKSPACE_BINDING_AUTHORITY",
                "_RUN_FILE_SANDBOX_ROOT",
            )
        ),
    ),
    tuple(
        (
            function,
            function.__code__,
            function.__globals__,
            function.__defaults__,
            function.__kwdefaults__,
            tuple((function.__kwdefaults__ or {}).items()),
            function.__closure__,
            tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
            vars(function).get("__wrapped__"),
        )
        for _owner, _name, descriptor in (
            *(
                (globals(), name, globals()[name])
                for name in (
                    "_default_registry_factory",
                    "allowed_file_roots",
                    "current_run_workspace_id",
                    "current_run_sandbox_root",
                    "_iter_valid_folder_bindings",
                    "_binding_matches_frozen_authority",
                )
            ),
            (globals(), "_registry_factory", _default_registry_factory),
            (globals(), "_default_registry_lock", _default_registry_lock),
            *(
                (globals(), name, globals()[name])
                for name in (
                    "_RUN_WORKSPACE_ID",
                    "_RUN_WORKSPACE_READ_BINDING_IDS",
                    "_RUN_WORKSPACE_WRITE_BINDING_IDS",
                    "_RUN_WORKSPACE_BINDING_AUTHORITY",
                    "_RUN_FILE_SANDBOX_ROOT",
                )
            ),
        )
        if callable(descriptor) or isinstance(descriptor, (staticmethod, classmethod))
        for outer in (
            descriptor.__func__
            if isinstance(descriptor, (staticmethod, classmethod))
            else descriptor,
        )
        if hasattr(outer, "__code__")
        for function in (
            outer,
            *((outer.__wrapped__,) if hasattr(outer, "__wrapped__") else ()),
            *(
                (outer.__wrapped__.__wrapped__,)
                if hasattr(outer, "__wrapped__")
                and hasattr(outer.__wrapped__, "__wrapped__")
                else ()
            ),
        )
    ),
)
