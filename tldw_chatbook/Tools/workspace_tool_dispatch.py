"""Dispatch closed workspace operations against one retained root pin."""

from __future__ import annotations

import shutil
from pathlib import Path

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
from tldw_chatbook.Tools.workspace_tool_protocol import (
    WorkspaceProtocolError,
    WorkspaceToolRequest,
    validate_glob_pattern,
)
from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion


class WorkspaceToolDispatchError(RuntimeError):
    """A fixed-code refusal from the pinned worker dispatcher."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def execute_pinned_operation(
    request: WorkspaceToolRequest,
    root: PinnedWorkspaceRoot,
) -> str:
    """Execute one supported request relative to ``root`` or refuse it."""
    if request.operation == "stat_path":
        return _stat_relative_path(_request_relative_path(request, root))
    if request.operation == "fs_write":
        return _write_relative_file(
            _request_mutation_path(request, root),
            request.arguments["content"],
            workspace=Path("."),
            display_path=request.arguments["path"],
        )
    if request.operation == "fs_edit":
        return _edit_relative_file(
            _request_mutation_path(request, root),
            request.arguments["old_string"],
            request.arguments["new_string"],
            workspace=Path("."),
            replace_all=request.arguments.get("replace_all", False),
            display_path=request.arguments["path"],
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
        )
    if request.operation == "fs_glob":
        try:
            pattern = validate_glob_pattern(request.arguments["pattern"])
        except WorkspaceProtocolError:
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


def _git_request(request: WorkspaceToolRequest) -> str:
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
    request: WorkspaceToolRequest, root: PinnedWorkspaceRoot
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
    request: WorkspaceToolRequest, field: str
) -> tuple[SensitiveExclusion, ...]:
    """Decode the parent's fixed bounded exclusions without filesystem discovery."""
    return tuple(
        SensitiveExclusion(item["kind"], item["value"])
        for item in request.arguments[field]
    )


def _request_mutation_path(
    request: WorkspaceToolRequest, root: PinnedWorkspaceRoot
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


def _patch_request(request: WorkspaceToolRequest, root: PinnedWorkspaceRoot) -> str:
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
    )


__all__ = ["WorkspaceToolDispatchError", "execute_pinned_operation"]
