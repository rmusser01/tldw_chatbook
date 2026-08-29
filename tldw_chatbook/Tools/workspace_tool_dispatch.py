"""Dispatch closed workspace operations against one retained root pin."""

from __future__ import annotations

from tldw_chatbook.Tools.local_tool_impls import _stat_relative_path
from tldw_chatbook.Tools.workspace_root_pin import PinnedWorkspaceRoot
from tldw_chatbook.Tools.workspace_tool_protocol import WorkspaceToolRequest


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
    if request.operation != "stat_path":
        raise WorkspaceToolDispatchError(
            "unsupported_operation",
            "workspace operation is not implemented",
        )
    relative = root.relative_path(request.arguments["path"])
    return _stat_relative_path(relative)


__all__ = ["WorkspaceToolDispatchError", "execute_pinned_operation"]
