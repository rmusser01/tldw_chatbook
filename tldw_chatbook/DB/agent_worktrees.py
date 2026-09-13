"""Durable structural records for app-created agent worktrees."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES

if TYPE_CHECKING:
    from .AgentRuns_DB import AgentRunsDB


AGENT_WORKTREES_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_worktrees (
    run_id TEXT PRIMARY KEY REFERENCES agent_runs(id),
    workspace_id TEXT NOT NULL,
    binding_id TEXT NOT NULL,
    locator_fingerprint TEXT NOT NULL,
    repo_root TEXT NOT NULL,
    repo_identity TEXT NOT NULL,
    git_common_dir TEXT NOT NULL,
    git_common_identity TEXT NOT NULL,
    child_path TEXT NOT NULL,
    child_identity TEXT NOT NULL,
    branch TEXT NOT NULL,
    base_sha TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    writer_state TEXT NOT NULL DEFAULT 'held'
        CHECK(writer_state IN ('held', 'drained', 'uncertain')),
    mutation_state TEXT NOT NULL DEFAULT 'unresolved'
        CHECK(mutation_state IN ('unresolved', 'applying', 'merging', 'discarding',
            'applied', 'merged', 'discarded_cleanup_pending', 'uncertain')),
    operation_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_worktrees_scope
    ON agent_worktrees(workspace_id, binding_id, run_id);
"""

_ACTIONS = {"apply": "applying", "merge": "merging", "discard": "discarding"}
_COMPLETIONS = {
    "applying": "applied",
    "merging": "merged",
    "discarding": "discarded_cleanup_pending",
}
_TEXT_ID = re.compile(r"[^\x00-\x1f\x7f]{1,256}\Z")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_INVALID_BRANCH_CHARS = frozenset(" ~^:?*[\\")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _TEXT_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a non-empty bounded identifier")
    return value


def _path(name: str, value: object) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute() or "\x00" in value:
        raise ValueError(f"{name} must be an absolute path")
    return value


def _branch(value: object) -> str:
    value = _identifier("branch", value)
    components = value.split("/")
    if (
        value.startswith(("-", "/"))
        or value.endswith("/")
        or value == "@"
        or ".." in value
        or "@{" in value
        or any(character in _INVALID_BRANCH_CHARS for character in value)
        or any(
            not component
            or component.startswith(".")
            or component.endswith((".", ".lock"))
            for component in components
        )
    ):
        raise ValueError("branch must be a valid bounded Git branch name")
    return value


def _identity(name: str, value: object) -> tuple[tuple[str, int, int, int], ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(
            not isinstance(component, tuple)
            or len(component) != 4
            or not isinstance(component[0], str)
            or not Path(component[0]).is_absolute()
            or "\x00" in component[0]
            or any(type(part) is not int or part < 0 for part in component[1:])
            for component in value
        )
    ):
        raise ValueError(
            f"{name} must be a non-empty (path, device, inode, mode) chain"
        )
    return value


def _encode_identity(value: tuple[tuple[str, int, int, int], ...]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _decode_identity(value: str) -> tuple[tuple[str, int, int, int], ...]:
    return tuple(tuple(component) for component in json.loads(value))


class AgentWorktreeRepository:
    """Read and mutate worktree records through a borrowed AgentRunsDB."""

    def __init__(self, db: AgentRunsDB) -> None:
        self._db = db

    def record_created(
        self,
        *,
        run_id: str,
        workspace_id: str,
        binding_id: str,
        locator_fingerprint: str,
        repo_root: str,
        repo_identity: tuple,
        git_common_dir: str,
        git_common_identity: tuple,
        child_path: str,
        child_identity: tuple,
        branch: str,
        base_sha: str,
        execution_id: str,
    ) -> None:
        run_id = _identifier("run_id", run_id)
        workspace_id = _identifier("workspace_id", workspace_id)
        binding_id = _identifier("binding_id", binding_id)
        execution_id = _identifier("execution_id", execution_id)
        branch = _branch(branch)
        if _FINGERPRINT.fullmatch(locator_fingerprint) is None:
            raise ValueError("locator_fingerprint must be a lowercase SHA-256")
        if _SHA.fullmatch(base_sha) is None:
            raise ValueError("base_sha must be a lowercase Git object id")
        repo_identity = _identity("repo_identity", repo_identity)
        git_common_identity = _identity("git_common_identity", git_common_identity)
        child_identity = _identity("child_identity", child_identity)
        stamp = _now()
        try:
            with self._db.transaction() as connection:
                cursor = connection.execute(
                    """INSERT INTO agent_worktrees
                       (run_id, workspace_id, binding_id, locator_fingerprint,
                        repo_root, repo_identity, git_common_dir,
                        git_common_identity, child_path, child_identity,
                        branch, base_sha, execution_id,
                        writer_state, mutation_state, operation_id, created_at, updated_at)
                       SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              'held', 'unresolved', NULL, ?, ?
                       WHERE EXISTS (SELECT 1 FROM agent_runs WHERE id = ?)""",
                    (
                        run_id,
                        workspace_id,
                        binding_id,
                        locator_fingerprint,
                        _path("repo_root", repo_root),
                        _encode_identity(repo_identity),
                        _path("git_common_dir", git_common_dir),
                        _encode_identity(git_common_identity),
                        _path("child_path", child_path),
                        _encode_identity(child_identity),
                        branch,
                        base_sha,
                        execution_id,
                        stamp,
                        stamp,
                        run_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("run does not exist")
        except sqlite3.IntegrityError as exc:
            raise ValueError("worktree ownership already recorded") from exc

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for name in ("repo_identity", "git_common_identity", "child_identity"):
            result[name] = _decode_identity(result[name])
        return result

    @staticmethod
    def _select() -> str:
        return """SELECT worktree.*, run.conversation_id, run.parent_run_id,
                         run.status AS run_status
                  FROM agent_worktrees AS worktree
                  JOIN agent_runs AS run ON run.id = worktree.run_id"""

    def get_for_conversation(self, run_id: str, conversation_id: str) -> dict | None:
        _identifier("run_id", run_id)
        _identifier("conversation_id", conversation_id)
        with self._db.connection() as connection:
            row = connection.execute(
                self._select() + " WHERE worktree.run_id=? AND run.conversation_id=?",
                (run_id, conversation_id),
            ).fetchone()
        return self._row(row)

    def list_for_conversation(
        self,
        conversation_id: str,
        *,
        workspace_id: str,
        binding_id: str,
        limit: int = 50,
        after_run_id: str | None = None,
    ) -> list[dict]:
        conversation_id = _identifier("conversation_id", conversation_id)
        workspace_id = _identifier("workspace_id", workspace_id)
        binding_id = _identifier("binding_id", binding_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 through 100")
        if after_run_id is not None:
            after_run_id = _identifier("after_run_id", after_run_id)
        with self._db.connection() as connection:
            rows = connection.execute(
                self._select()
                + """ WHERE run.conversation_id=? AND worktree.workspace_id=?
                           AND worktree.binding_id=? AND worktree.run_id>?
                      ORDER BY worktree.run_id ASC LIMIT ?""",
                (conversation_id, workspace_id, binding_id, after_run_id or "", limit),
            ).fetchall()
        return [self._row(row) for row in rows]

    def mark_writer_finished(
        self, run_id: str, execution_id: str, *, cleanup_proven: bool
    ) -> bool:
        run_id = _identifier("run_id", run_id)
        execution_id = _identifier("execution_id", execution_id)
        if type(cleanup_proven) is not bool:
            raise TypeError("cleanup_proven must be a bool")
        stamp = _now()
        with self._db.transaction() as connection:
            if cleanup_proven:
                cursor = connection.execute(
                    """UPDATE agent_worktrees SET writer_state='drained', updated_at=?
                       WHERE run_id=? AND execution_id=? AND writer_state='held'""",
                    (stamp, run_id, execution_id),
                )
            else:
                cursor = connection.execute(
                    """UPDATE agent_worktrees SET writer_state='uncertain', updated_at=?
                       WHERE run_id=? AND execution_id=?
                         AND writer_state IN ('held', 'drained')""",
                    (stamp, run_id, execution_id),
                )
        return cursor.rowcount == 1

    def claim(
        self,
        run_id: str,
        conversation_id: str,
        *,
        operation_id: str,
        action: str,
    ) -> bool:
        run_id = _identifier("run_id", run_id)
        conversation_id = _identifier("conversation_id", conversation_id)
        operation_id = _identifier("operation_id", operation_id)
        if action not in _ACTIONS:
            raise ValueError("action must be apply, merge, or discard")
        terminal = sorted(TERMINAL_RUN_STATUSES)
        placeholders = ",".join("?" for _ in terminal)
        with self._db.transaction() as connection:
            cursor = connection.execute(
                f"""UPDATE agent_worktrees SET mutation_state=?, operation_id=?, updated_at=?
                    WHERE run_id=? AND writer_state='drained'
                      AND mutation_state='unresolved' AND operation_id IS NULL
                      AND EXISTS (SELECT 1 FROM agent_runs AS run
                                  WHERE run.id=agent_worktrees.run_id
                                    AND run.conversation_id=?
                                    AND run.status IN ({placeholders}))""",
                (
                    _ACTIONS[action],
                    operation_id,
                    _now(),
                    run_id,
                    conversation_id,
                    *terminal,
                ),
            )
        return cursor.rowcount == 1

    def finish_operation(self, run_id: str, operation_id: str, *, state: str) -> bool:
        run_id = _identifier("run_id", run_id)
        operation_id = _identifier("operation_id", operation_id)
        expected = next(
            (
                inflight
                for inflight, success in _COMPLETIONS.items()
                if success == state
            ),
            None,
        )
        if state == "uncertain":
            inflight = tuple(_COMPLETIONS)
        elif expected is not None:
            inflight = (expected,)
        else:
            raise ValueError("state is not a valid operation completion")
        placeholders = ",".join("?" for _ in inflight)
        with self._db.transaction() as connection:
            cursor = connection.execute(
                f"""UPDATE agent_worktrees SET mutation_state=?, operation_id=NULL,
                           updated_at=?
                      WHERE run_id=? AND operation_id=?
                        AND mutation_state IN ({placeholders})""",
                (state, _now(), run_id, operation_id, *inflight),
            )
        return cursor.rowcount == 1
