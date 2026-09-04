"""Impure gatherers for the Console Environment panel.

Everything here runs on a worker thread. Functions never raise: failures
map to availability enums (spec: absence is silent, errors keep last
good data with a stale marker).
"""
from __future__ import annotations

import json
import os
import re as _re
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml
from loguru import logger

from tldw_chatbook.Chat.console_environment_state import (
    BacklogTaskEntry,
    BranchTaskState,
    EnvSourceAvailability,
    GitEnvState,
    PrCheck,
    PrEnvState,
    TasksEnvState,
    branch_task_id,
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


_GH_EXECUTABLE = "gh"
_GH_TIMEOUT_SECONDS = 5.0
_PR_JSON_FIELDS = (
    "number,title,state,isDraft,url,additions,deletions,mergedAt,statusCheckRollup"
)


@dataclass(frozen=True)
class GhResult:
    returncode: int
    stdout: str
    stderr: str


def run_gh(root: Path, args: list[str], *, timeout: float = _GH_TIMEOUT_SECONDS) -> GhResult | None:
    """Run gh non-interactively in ``root``. ``None`` == binary missing."""
    env = dict(os.environ)
    env.update({"GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1", "NO_COLOR": "1"})
    try:
        completed = subprocess.run(
            [_GH_EXECUTABLE, *args],
            cwd=str(root), env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return None
    except (subprocess.TimeoutExpired, OSError) as exc:
        return GhResult(returncode=124, stdout="", stderr=str(exc))
    return GhResult(completed.returncode, completed.stdout, completed.stderr)


GhRunner = Callable[[Path, list[str]], GhResult | None]


def _parse_check(entry: dict) -> PrCheck | None:
    typename = entry.get("__typename", "")
    if typename == "CheckRun":
        status = (entry.get("status") or "").upper()
        conclusion_raw = (entry.get("conclusion") or "").upper()
        if status != "COMPLETED":
            conclusion = "pending"
        elif conclusion_raw in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
            conclusion = "success"
        else:
            conclusion = "failure"
        return PrCheck(name=str(entry.get("name") or "?"), conclusion=conclusion,
                       details_url=str(entry.get("detailsUrl") or ""))
    if typename == "StatusContext":
        state = (entry.get("state") or "").upper()
        conclusion = ("success" if state == "SUCCESS"
                      else "pending" if state in {"PENDING", "EXPECTED"} else "failure")
        return PrCheck(name=str(entry.get("context") or "?"), conclusion=conclusion,
                       details_url=str(entry.get("targetUrl") or ""))
    return None


def _parse_merged_at(raw: object) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def gather_pr_env(
    root: Path,
    branch: str | None,
    *,
    runner: GhRunner = run_gh,
    previous: PrEnvState | None = None,
) -> PrEnvState:
    """Fetch PR + checks for ``branch`` via gh. Never raises."""
    if not branch:
        return PrEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE)
    result = runner(root, ["pr", "view", branch, "--json", _PR_JSON_FIELDS])
    if result is None:
        return PrEnvState(availability=EnvSourceAvailability.MISSING_TOOL)
    if result.returncode != 0:
        if "no pull requests found" in result.stderr.lower():
            return PrEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE)
        logger.debug("environment_status: gh failed rc={} err={}",
                     result.returncode, result.stderr[:200])
        if previous is not None and previous.availability is EnvSourceAvailability.OK:
            return replace(previous, stale=True)
        return PrEnvState(availability=EnvSourceAvailability.ERROR)
    try:
        payload = json.loads(result.stdout)
        checks = tuple(
            check for entry in (payload.get("statusCheckRollup") or [])
            if isinstance(entry, dict) and (check := _parse_check(entry)) is not None
        )
        return PrEnvState(
            availability=EnvSourceAvailability.OK,
            number=int(payload.get("number") or 0),
            title=str(payload.get("title") or ""),
            state=str(payload.get("state") or ""),
            is_draft=bool(payload.get("isDraft")),
            url=str(payload.get("url") or ""),
            adds=int(payload.get("additions") or 0),
            dels=int(payload.get("deletions") or 0),
            merged_at=_parse_merged_at(payload.get("mergedAt")),
            checks=checks,
        )
    except (ValueError, TypeError) as exc:
        logger.debug("environment_status: gh JSON parse failed: {}", exc)
        return PrEnvState(availability=EnvSourceAvailability.ERROR)


_TASK_FILENAME_RE = _re.compile(r"^task-(\d+(?:\.\d+)*) - (.+)\.md$")
_FRONT_MATTER_RE = _re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", _re.DOTALL)
_AC_DONE_RE = _re.compile(r"^- \[x\]", _re.MULTILINE | _re.IGNORECASE)
_AC_OPEN_RE = _re.compile(r"^- \[ \]", _re.MULTILINE)


class BacklogTaskScanner:
    """Scans <workspace>/backlog/tasks/ with an instance-scoped (mtime, size) cache."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[tuple[float, int], str]] = {}

    def _parse_status(self, path: Path) -> str:
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:4096]
        except OSError:
            return ""
        match = _FRONT_MATTER_RE.match(head)
        if not match:
            return ""
        try:
            meta = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return ""
        if not isinstance(meta, dict):
            return ""
        return str(meta.get("status") or "").strip()

    def _status_for(self, path: Path) -> str:
        try:
            stat = path.stat()
        except OSError:
            return ""
        signature = (stat.st_mtime, stat.st_size)
        cached = self._cache.get(str(path))
        if cached is not None and cached[0] == signature:
            return cached[1]
        status = self._parse_status(path)
        self._cache[str(path)] = (signature, status)
        return status

    @staticmethod
    def _ac_progress(path: Path) -> tuple[int, int]:
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return (0, 0)
        done = len(_AC_DONE_RE.findall(body))
        open_count = len(_AC_OPEN_RE.findall(body))
        return (done, done + open_count)

    def scan(self, workspace_root: Path, branch: str | None) -> TasksEnvState:
        tasks_dir = workspace_root / "backlog" / "tasks"
        if not tasks_dir.is_dir():
            return TasksEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE)
        wanted_id = branch_task_id(branch)
        entries: list[BacklogTaskEntry] = []
        branch_task: BranchTaskState | None = None
        in_progress = todo = 0
        try:
            listing = sorted(tasks_dir.iterdir())
        except OSError:
            return TasksEnvState(availability=EnvSourceAvailability.ERROR)
        for path in listing:
            match = _TASK_FILENAME_RE.match(path.name)
            if not match:
                continue
            task_id, title = match.group(1), match.group(2)
            status = self._status_for(path)
            if not status:
                continue
            if status == "In Progress":
                in_progress += 1
            elif status == "To Do":
                todo += 1
            if status != "Done" or task_id == wanted_id:
                entries.append(BacklogTaskEntry(task_id=task_id, title=title, status=status))
            if wanted_id is not None and task_id == wanted_id:
                done, total = self._ac_progress(path)
                branch_task = BranchTaskState(
                    task_id=task_id, title=title, status=status,
                    ac_done=done, ac_total=total, path=str(path),
                )
        return TasksEnvState(
            availability=EnvSourceAvailability.OK,
            branch_task=branch_task,
            in_progress=in_progress,
            todo=todo,
            entries=tuple(entries),
        )
