"""Dispatch closed workspace operations against one retained root pin."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Tools.local_tool_impls import (
    MAX_GLOB_RESULTS,
    MAX_GREP_RESULTS,
    MAX_LIST_ENTRIES,
    _glob_relative_files,
    _grep_relative_files,
    _is_relative_sensitive_path,
    _list_relative_directory,
    _read_relative_file,
    _stat_relative_path,
)
from tldw_chatbook.Tools.workspace_root_pin import (
    PinnedWorkspaceRoot,
    WorkspaceRootPinError,
)
from tldw_chatbook.Tools.workspace_tool_protocol import WorkspaceToolRequest
from tldw_chatbook.Utils.sensitive_paths import (
    SensitiveExclusion,
    sensitive_exclusions_under,
)


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

    exclusions = sensitive_exclusions_under(Path("."))
    if request.operation == "fs_list":
        return _list_relative_directory(
            _read_relative_path(request, root, exclusions),
            workspace=Path("."),
            max_entries=MAX_LIST_ENTRIES,
            sensitive_exclusions=exclusions,
        )
    if request.operation == "fs_read":
        return _read_relative_file(
            _read_relative_path(request, root, exclusions),
            workspace=Path("."),
            offset=request.arguments.get("offset", 1),
            limit=request.arguments.get("limit"),
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
            sensitive_exclusions=exclusions,
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


def _read_relative_path(
    request: WorkspaceToolRequest,
    root: PinnedWorkspaceRoot,
    exclusions: tuple[SensitiveExclusion, ...],
) -> Path:
    """Return an admitted relative read path without exposing protected names."""
    relative = _request_relative_path(request, root)
    if _is_relative_sensitive_path(relative, exclusions, is_directory=relative.is_dir()):
        raise WorkspaceToolDispatchError(
            "invalid_request",
            "workspace operation path is invalid",
        )
    return relative


__all__ = ["WorkspaceToolDispatchError", "execute_pinned_operation"]
