"""Dispatch closed workspace operations against one retained root pin."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Tools.local_tool_impls import (
    MAX_GLOB_RESULTS,
    MAX_GREP_RESULTS,
    MAX_LIST_ENTRIES,
    _glob_relative_files,
    _grep_relative_files,
    _list_relative_directory,
    _read_relative_file,
    _stat_relative_path,
)
from tldw_chatbook.Tools.workspace_root_pin import (
    PinnedWorkspaceRoot,
    WorkspaceRootPinError,
)
from tldw_chatbook.Tools.workspace_tool_protocol import WorkspaceToolRequest
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

    exclusions = _request_exclusions(request, "sensitive_exclusions")
    if request.operation == "fs_list":
        return _list_relative_directory(
            _request_relative_path(request, root),
            workspace=Path("."),
            max_entries=MAX_LIST_ENTRIES,
            sensitive_exclusions=exclusions,
            validate_target=False,
        )
    if request.operation == "fs_read":
        return _read_relative_file(
            _request_relative_path(request, root),
            workspace=Path("."),
            offset=request.arguments.get("offset", 1),
            limit=request.arguments.get("limit"),
            validate_target=False,
        )
    if request.operation == "fs_glob":
        return _glob_relative_files(
            request.arguments["pattern"],
            workspace=Path("."),
            max_results=request.arguments.get("max_results", MAX_GLOB_RESULTS),
            sensitive_exclusions=exclusions,
        )
    if request.operation == "fs_grep":
        return _grep_relative_files(
            request.arguments["pattern"],
            workspace=Path("."),
            mode=request.arguments.get("mode", "content"),
            max_results=request.arguments.get("max_results", MAX_GREP_RESULTS),
            sensitive_exclusions=_request_exclusions(request, "content_exclusions"),
            validate_symlink_targets=False,
        )
    raise WorkspaceToolDispatchError(
        "unsupported_operation",
        "workspace operation is not implemented",
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


__all__ = ["WorkspaceToolDispatchError", "execute_pinned_operation"]
