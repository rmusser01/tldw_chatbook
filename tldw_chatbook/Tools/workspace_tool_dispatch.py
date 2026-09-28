"""Dispatch closed workspace operations against one retained root pin.

Part of the pinned worker's stdlib-only import closure (Phase 0c): nothing
here may import the parent's pydantic protocol module. Request frames
arrive already decoded/validated; dispatch consumes the attribute surface
``_PinnedOperationRequest`` describes (the parent's
``WorkspaceToolRequest`` dataclass and the worker's decoded-request view
both satisfy it).
"""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from tldw_chatbook.Tools.git_tool_impls import (
    git_blame,
    git_branches,
    git_diff,
    git_log,
    git_status,
)
from tldw_chatbook.Tools.local_tool_impls import (
    MAX_GLOB_RESULTS,
    MAX_GREP_RESULTS,
    MAX_LIST_ENTRIES,
    _glob_relative_files,
    _grep_relative_files,
    _edit_relative_file,
    _list_relative_directory,
    _read_relative_file,
    _relative_target_is_safe,
    _stat_relative_path,
    _write_relative_file,
)
from tldw_chatbook.Tools.patch_tool_impls import (
    FilesystemPatchError,
    parse_patch_targets,
    patch_validated_files,
)
from tldw_chatbook.Tools.workspace_root_pin import (
    PinnedWorkspaceRoot,
    WorkspaceRootPinError,
)
from tldw_chatbook.Tools.workspace_wire_decode import (
    WireDecodeError,
    validate_glob_pattern,
)
from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion


class WorkspaceToolDispatchError(RuntimeError):
    """A fixed-code refusal from the pinned worker dispatcher."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _PinnedOperationRequest(Protocol):
    """The attribute surface dispatch consumes from one admitted request."""

    operation: str
    arguments: dict[str, Any]


#: Catch-all exclusion for a pinned root that ITSELF sits inside a
#: denylisted entry: an empty 'subtree' prefix matches every relative
#: path in the exclusion matcher (``parts[:0] == ()`` is vacuously
#: true), so every operation on the binding refuses — the whole root is
#: a denylisted subtree (review fix, Important 1).
_DENYLISTED_ROOT_CATCH_ALL = (SensitiveExclusion("subtree", ""),)


def _remote_home_denylist_exclusions() -> tuple[SensitiveExclusion, ...]:
    """Map the worker-side remote-home denylist into root-relative space.

    ``REMOTE_SENSITIVE_PATHS`` (``Tools/remote_sensitive_paths.py``) is
    the set of REMOTE-HOME-relative subtrees the pinned worker must never
    expose regardless of which root a binding pins. This module is shared
    by the LOCAL pinned worker and the flattened remote bundle; the
    denylist NAME exists only in the bundle's flat namespace (Task 8
    embeds the data below the dispatch section), so the live local worker
    resolves it to ``()`` through ``globals().get`` — byte-identical
    local behavior — while the bundle picks the tuple up at call time.

    Mapping rule (review fix, Important 1 — the general form): each
    entry is joined onto the resolved home directory, then re-expressed
    relative to the pinned root (``Path('.')`` after the root pin's
    chdir) via ``relative_to`` — which is exactly ``relpath(home/entry,
    root)`` and holds for EVERY root relationship:

    - root BELOW home (the common case, e.g. ``~/projects``): entries
      map to plain subtrees under the root when they are under it, and
      to nothing when they are not (the worker is root-confined, so an
      entry outside the root is unreachable);
    - root AT or ABOVE home (e.g. ``ssh://host/`` pinning ``/``): every
      entry lies under the root and maps to a deep subtree refusal —
      the pre-review mapping RAISED here (``root.relative_to(home)`` is
      inverted for these roots) and silently voided the entire denylist;
    - root INSIDE a denylisted entry (e.g. a binding pinned at
      ``~/.ssh`` itself): the whole root is denylisted — every operation
      refuses via :data:`_DENYLISTED_ROOT_CATCH_ALL`.

    An unresolvable home degrades to ``()``: the request-carried
    exclusions still enforce, and a missing ``HOME`` must not crash
    unrelated operations.
    """
    entries = globals().get("REMOTE_SENSITIVE_PATHS", ())
    if not entries:
        return ()
    try:
        home = Path.home().resolve()
        root = Path(".").resolve()
    except (RuntimeError, OSError):
        return ()
    root_under_home: PurePosixPath | None = None
    try:
        root_under_home = PurePosixPath(root.relative_to(home).as_posix())
    except ValueError:
        root_under_home = None  # root at/above home (or disjoint)
    if root_under_home is not None:
        for entry in entries:
            entry_parts = PurePosixPath(entry).parts
            if root_under_home.parts[: len(entry_parts)] == entry_parts:
                return _DENYLISTED_ROOT_CATCH_ALL
    mapped: list[SensitiveExclusion] = []
    for entry in entries:
        try:
            target = (home / entry).resolve()
            relative = target.relative_to(root)
        except (OSError, ValueError):
            continue  # entry outside the pinned root: unreachable
        mapped.append(SensitiveExclusion("subtree", relative.as_posix()))
    return tuple(mapped)


def execute_pinned_operation(
    request: _PinnedOperationRequest,
    root: PinnedWorkspaceRoot,
) -> str:
    """Execute one supported request relative to ``root`` or refuse it."""
    if request.operation == "stat_path":
        # ``stat_path``'s wire schema carries no exclusions field, so the
        # serialized binding exclusions are refused PARENT-side (the
        # remote builder's lexical admission); the worker-side denylist
        # applies here (ADR-174: excluded/sensitive paths are fully
        # invisible — a stat must never confirm existence).
        relative = _request_relative_path(request, root)
        denylist = _remote_home_denylist_exclusions()
        if denylist and not _relative_target_is_safe(
            relative, Path("."), denylist, is_directory=True
        ):
            raise WorkspaceToolDispatchError(
                "invalid_request", "workspace path is invalid"
            )
        return _stat_relative_path(relative)
    if request.operation == "fs_write":
        return _write_relative_file(
            _request_mutation_path(request, root),
            request.arguments["content"],
            workspace=Path("."),
            display_path=request.arguments["path"],
            dry_run=request.arguments.get("dry_run", False),
            expected_sha256=request.arguments.get("expected_sha256"),
            expected_absent=request.arguments.get("expected_absent", False),
            content_stamps=True,
        )
    if request.operation == "fs_edit":
        return _edit_relative_file(
            _request_mutation_path(request, root),
            request.arguments["old_string"],
            request.arguments["new_string"],
            workspace=Path("."),
            replace_all=request.arguments.get("replace_all", False),
            display_path=request.arguments["path"],
            content_stamps=True,
        )
    if request.operation == "fs_patch":
        return _patch_request(request, root)
    if request.operation in {
        "git_status",
        "git_diff",
        "git_log",
        "git_blame",
        "git_branches",
    }:
        return _git_request(request)
    if request.operation not in {"fs_list", "fs_read", "fs_glob", "fs_grep"}:
        raise WorkspaceToolDispatchError(
            "unsupported_operation", "workspace operation is not implemented"
        )
    exclusions = _request_exclusions(request, "sensitive_exclusions")
    if request.operation == "fs_list":
        return _list_relative_directory(
            _request_relative_path(request, root),
            workspace=Path("."),
            max_entries=MAX_LIST_ENTRIES,
            sensitive_exclusions=exclusions,
        )
    if request.operation == "fs_read":
        return _read_relative_file(
            _request_relative_path(request, root),
            workspace=Path("."),
            offset=request.arguments.get("offset", 1),
            limit=request.arguments.get("limit"),
            sensitive_exclusions=exclusions,
            content_stamps=True,
        )
    if request.operation == "fs_glob":
        try:
            pattern = validate_glob_pattern(request.arguments["pattern"])
        except WireDecodeError:
            raise WorkspaceToolDispatchError(
                "invalid_request", "workspace glob pattern is invalid"
            ) from None
        return _glob_relative_files(
            pattern,
            workspace=Path("."),
            max_results=request.arguments.get("max_results", MAX_GLOB_RESULTS),
            sensitive_exclusions=exclusions,
            validate_targets=True,
        )
    if request.operation == "fs_grep":
        return _grep_relative_files(
            request.arguments["pattern"],
            workspace=Path("."),
            mode=request.arguments.get("mode", "content"),
            max_results=request.arguments.get("max_results", MAX_GREP_RESULTS),
            sensitive_exclusions=_request_exclusions(request, "content_exclusions"),
        )


def _git_request(request: _PinnedOperationRequest) -> str:
    """Run one closed read-only Git operation beneath the retained root."""
    discovered = shutil.which("git")
    if discovered is None:
        raise WorkspaceToolDispatchError(
            "tool_failure", "git is not available on this system"
        )
    executable = Path(discovered).resolve()
    exclusions = _request_exclusions(request, "sensitive_exclusions")
    execution = {
        "executable": executable,
        "own_process_group": False,
        "sensitive_exclusions": exclusions,
    }
    arguments = request.arguments
    if request.operation == "git_status":
        return git_status(Path("."), arguments.get("path", "."), **execution)
    if request.operation == "git_diff":
        return git_diff(
            Path("."),
            staged=arguments.get("staged", False),
            commit_range=arguments.get("commit_range"),
            path=arguments.get("path"),
            stat=arguments.get("stat", False),
            **execution,
        )
    if request.operation == "git_log":
        return git_log(
            Path("."),
            count=arguments.get("count", 20),
            path=arguments.get("path"),
            **execution,
        )
    if request.operation == "git_blame":
        return git_blame(
            Path("."),
            arguments["path"],
            start_line=arguments.get("start_line"),
            end_line=arguments.get("end_line"),
            **execution,
        )
    return git_branches(Path("."), **execution)


def _request_relative_path(
    request: _PinnedOperationRequest, root: PinnedWorkspaceRoot
) -> Path:
    """Return one request path validated as lexical root-relative text."""
    try:
        return root.relative_path(request.arguments["path"])
    except WorkspaceRootPinError:
        raise WorkspaceToolDispatchError(
            "invalid_request",
            "workspace operation path is invalid",
        ) from None


def _request_exclusions(
    request: _PinnedOperationRequest, field: str
) -> tuple[SensitiveExclusion, ...]:
    """Decode the parent's fixed bounded exclusions without filesystem discovery.

    Task 17: the worker-side remote-home denylist joins the request's
    serialized exclusions here — the single decode point every read,
    write, patch, and git consumer passes through — so ``REMOTE_SENSITIVE_PATHS``
    is enforced on every operation even when the parent serialized none
    (the local pinned worker's denylist contribution is ``()``, keeping
    its behavior byte-identical).
    """
    return tuple(
        SensitiveExclusion(item["kind"], item["value"])
        for item in request.arguments[field]
    ) + _remote_home_denylist_exclusions()


def _request_mutation_path(
    request: _PinnedOperationRequest, root: PinnedWorkspaceRoot
) -> Path:
    """Validate a mutation target's live lexical and resolved location."""
    relative = _request_relative_path(request, root)
    if not _relative_target_is_safe(
        relative,
        Path("."),
        _request_exclusions(request, "sensitive_exclusions"),
        is_directory=False,
    ):
        raise WorkspaceToolDispatchError(
            "invalid_request", "workspace mutation target is invalid"
        )
    return relative


def _patch_request(request: _PinnedOperationRequest, root: PinnedWorkspaceRoot) -> str:
    """Reparse a bounded patch and require its exact parent-admitted targets."""
    try:
        plans = parse_patch_targets(request.arguments["diff"])
        parsed_paths = tuple(
            root.relative_path(plan.new_path)
            for plan in plans
            if plan.new_path is not None
        )
        parsed_targets = tuple(path.as_posix() for path in parsed_paths)
        requested_targets = tuple(
            root.relative_path(target).as_posix()
            for target in request.arguments.get("targets", ())
        )
    except (FilesystemPatchError, WorkspaceRootPinError, TypeError):
        raise WorkspaceToolDispatchError(
            "invalid_request", "workspace patch request is invalid"
        ) from None
    if len(parsed_targets) != len(plans) or parsed_targets != requested_targets:
        raise WorkspaceToolDispatchError(
            "invalid_request", "workspace patch targets changed after admission"
        )
    exclusions = _request_exclusions(request, "sensitive_exclusions")
    if not all(
        _relative_target_is_safe(
            relative, Path("."), exclusions, is_directory=False
        )
        for relative in parsed_paths
    ):
        raise WorkspaceToolDispatchError(
            "invalid_request", "workspace patch target is invalid"
        )
    return patch_validated_files(
        plans,
        root=root,
        dry_run=request.arguments.get("dry_run", False),
        content_stamps=True,
    )


__all__ = ["WorkspaceToolDispatchError", "execute_pinned_operation"]
