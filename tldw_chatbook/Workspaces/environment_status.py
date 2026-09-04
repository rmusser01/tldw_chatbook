"""Impure gatherers for the Console Environment panel.

Everything here runs on a worker thread. Functions never raise: failures
map to availability enums (spec: absence is silent, errors keep last
good data with a stale marker).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from loguru import logger

from tldw_chatbook.Chat.console_environment_state import (
    EnvSourceAvailability,
    GitEnvState,
)
from tldw_chatbook.Workspaces.git_workspace import (
    GitWorkspaceError,
    GitWorkspaceInfo,
    detect_git_workspace,
    linked_worktree_name,
    working_tree_status,
    _run_user_git,
)


def _head_short(root: Path) -> str:
    try:
        result = _run_user_git(root, "rev-parse", "--short", "HEAD")
    except GitWorkspaceError:
        return ""
    return result.stdout.strip()


def gather_git_env(root: Path, *, previous: GitEnvState | None = None) -> GitEnvState:
    """Assemble the git tier of the Environment snapshot. Never raises."""
    info = detect_git_workspace(root)
    if not isinstance(info, GitWorkspaceInfo):
        return GitEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE)
    try:
        status = working_tree_status(root, info)
    except GitWorkspaceError as exc:
        logger.debug("environment_status: working_tree_status failed: {}", exc)
        if previous is not None and previous.availability is EnvSourceAvailability.OK:
            return replace(previous, stale=True)
        return GitEnvState(availability=EnvSourceAvailability.ERROR)
    files = status.files
    return GitEnvState(
        availability=EnvSourceAvailability.OK,
        root=str(status.root),
        branch=info.branch,
        detached=info.detached,
        unborn=info.unborn,
        head_short=_head_short(root) if info.detached else "",
        upstream=info.upstream,
        ahead=info.ahead,
        behind=info.behind,
        adds=sum(f.adds for f in files),
        dels=sum(f.dels for f in files),
        files=files,
        worktree_name=linked_worktree_name(root),
        stale=False,
    )
