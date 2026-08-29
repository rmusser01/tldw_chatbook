"""Dispatch closed workspace operations against one retained root pin."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Tools.local_tool_impls import (
    MAX_GLOB_RESULTS,
    MAX_GREP_RESULTS,
    MAX_LIST_ENTRIES,
    _glob_relative_files,
    _grep_relative_files,
    _edit_relative_file,
    _list_relative_directory,
    _read_relative_file,
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
            _request_relative_path(request, root),
            request.arguments["content"],
            workspace=Path("."),
            display_path=request.arguments["path"],
        )
    if request.operation == "fs_edit":
        return _edit_relative_file(
            _request_relative_path(request, root),
            request.arguments["old_string"],
            request.arguments["new_string"],
            workspace=Path("."),
            replace_all=request.arguments.get("replace_all", False),
            display_path=request.arguments["path"],
        )
    if request.operation == "fs_patch":
        return _patch_request(request, root)
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


def _patch_request(request: WorkspaceToolRequest, root: PinnedWorkspaceRoot) -> str:
    """Reparse a bounded patch and require its exact parent-admitted targets."""
    try:
        plans = parse_patch_targets(request.arguments["diff"])
        parsed_targets = tuple(
            root.relative_path(plan.new_path).as_posix()
            for plan in plans
            if plan.new_path is not None
        )
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
    return patch_validated_files(
        plans,
        root=root,
        dry_run=request.arguments.get("dry_run", False),
    )


__all__ = ["WorkspaceToolDispatchError", "execute_pinned_operation"]
