"""Revert engine for Agent Change Review (TASK-1974).

Restores files to their turn-baseline (**B**) state, with the guards the
spec demands:

* **Refusal during active runs** — the per-root lock serializes *git*
  operations, but the agent's own file tools do not take it; reverting under
  a writing agent would interleave clobbers. The caller injects the
  ``run_active`` probe and the refusal happens BEFORE any file is touched.
* **User-edit guard** — :func:`preflight_revert` names, per path, the files
  whose disk state differs from the turn's end (**E**): the user (or a later
  turn) changed them, and the confirm dialog must list them BY NAME before
  anything is overwritten.
* **Un-create is a guarded delete** — ``checkout B -- path`` errors on a
  path absent from B; a created file is removed explicitly, and only when it
  is genuinely absent at B.
* **Per-path outcomes** — one failing path reports itself and the rest
  complete; a partial failure is never silent.
* **History stays true** — every revert takes a fresh snapshot and updates
  the row's ``reverted`` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Callable, Sequence

from loguru import logger

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Workspaces.change_tracking import (
    ChangeTrackingError,
    ShadowRepoService,
)


class RevertRefusedError(ChangeTrackingError):
    """The revert was refused before touching anything (active run)."""


@dataclass(frozen=True)
class RevertOutcome:
    """One path's revert result.

    Attributes:
        path: Root-relative path.
        ok: Whether the restore landed.
        error: Failure copy when ``ok`` is False.
    """

    path: str
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class RevertPreflight:
    """What the confirm dialog needs to say before a revert.

    Attributes:
        edited_since: Paths whose CURRENT disk state differs from the
            turn's E snapshot — the user (or a later turn) changed them
            after this turn, and reverting will overwrite that work.
        disk_state: Stable content fingerprints for every path the revert
            may overwrite. A caller can compare a fresh preflight with the
            confirmation-time snapshot and refuse a stale confirmation.
    """

    edited_since: list[str] = field(default_factory=list)
    disk_state: dict[str, str] = field(default_factory=dict)


def _unsafe_reason(path: str) -> str | None:
    """Lexical traversal guard for caller-supplied paths.

    Git-relative paths — the only kind a turn's change set contains — are
    never absolute and never carry a ``..`` segment, so any such request
    is refused before ANY disk operation touches it.

    Args:
        path: The requested root-relative path.

    Returns:
        The refusal copy, or ``None`` when the path is lexically safe.
    """
    parts = Path(path)
    if parts.is_absolute() or ".." in parts.parts:
        return "absolute or traversal path refused"
    return None


def preflight_revert(
    service: ShadowRepoService,
    row: dict,
    paths: Sequence[str],
) -> RevertPreflight:
    """Compare each path's disk state against the turn's E snapshot.

    A rename's revert ALSO restores ``old_path`` from B, so its disk state
    is compared too — the requested (new) path alone is not the full
    overwrite set. Lexically unsafe paths are skipped: they will never be
    reverted, so they cannot overwrite anything.

    Args:
        service: Shadow-repo service.
        row: The turn's ``change_snapshots`` row.
        paths: Root-relative paths the user asked to revert.

    Returns:
        The preflight report; a diff/read failure on a path conservatively
        counts it as edited (better a spurious warning than a silent
        overwrite).
    """
    repo = service.repo_for_root(row["root"])
    root = Path(str(row["root"]))
    end = str(row["end_sha"])
    changed = {
        c.path: c
        for c in repo.changed_files(str(row["baseline_sha"]), str(row["end_sha"]))
    }
    edited: list[str] = []
    disk_state: dict[str, str] = {}

    def _check(path: str) -> None:
        if path in edited:
            return
        try:
            at_end = repo.file_bytes(end, path)
            target = root / path
            on_disk = target.read_bytes() if target.is_file() else None
            disk_state[path] = (
                "missing"
                if on_disk is None
                else f"sha256:{sha256(on_disk).hexdigest()}"
            )
            if at_end != on_disk:
                edited.append(path)
        except (OSError, ChangeTrackingError):
            disk_state[path] = "unreadable"
            edited.append(path)

    for path in paths:
        if _unsafe_reason(path) is not None:
            continue
        _check(path)
        change = changed.get(path)
        if change is not None and change.status == "R" and change.old_path:
            _check(change.old_path)
    return RevertPreflight(edited_since=edited, disk_state=disk_state)


def revert_paths(
    service: ShadowRepoService,
    db: AgentRunsDB,
    row: dict,
    paths: Sequence[str],
    *,
    run_active: Callable[[], bool],
) -> list[RevertOutcome]:
    """Restore ``paths`` to the turn's baseline state.

    Args:
        service: Shadow-repo service.
        db: Runs database (the row's ``reverted`` field is updated).
        row: The turn's ``change_snapshots`` row.
        paths: Root-relative paths to restore.
        run_active: Probe for an active run on this root's workspace; True
            refuses the whole revert before any file is touched.

    Returns:
        One :class:`RevertOutcome` per requested path, in order.

    Raises:
        RevertRefusedError: A run is active — finish or stop it first.
    """
    if run_active():
        raise RevertRefusedError(
            "a run is active on this workspace — finish or stop the run first"
        )
    repo = service.repo_for_root(row["root"])
    root = Path(str(row["root"]))
    baseline = str(row["baseline_sha"])
    end = str(row["end_sha"])
    changed = {c.path: c for c in repo.changed_files(baseline, end)}

    outcomes: list[RevertOutcome] = []
    reverted: list[str] = []
    for path in paths:
        unsafe = _unsafe_reason(path)
        if unsafe is not None:
            outcomes.append(RevertOutcome(path=path, ok=False, error=unsafe))
            continue
        change = changed.get(path)
        if change is None:
            outcomes.append(
                RevertOutcome(
                    path=path,
                    ok=False,
                    error="not part of this turn's changes",
                )
            )
            continue
        try:
            if change.status == "A":
                _guarded_uncreate(repo, root, baseline, path)
            elif change.status == "R" and change.old_path:
                repo.restore_paths(baseline, [change.old_path])
                _guarded_uncreate(repo, root, baseline, path)
            else:
                # M, D, and the verbatim rare letters all restore from B.
                repo.restore_paths(baseline, [path])
            outcomes.append(RevertOutcome(path=path, ok=True))
            reverted.append(path)
        except (ChangeTrackingError, OSError) as exc:
            outcomes.append(
                RevertOutcome(path=path, ok=False, error=str(exc)[:300])
            )

    if reverted:
        try:
            repo.snapshot("revert")
        except ChangeTrackingError:
            logger.opt(exception=True).warning(
                "change_review: post-revert snapshot failed"
            )
        try:
            db.update_change_snapshot_reverted(int(row["id"]), reverted)
        except Exception:  # noqa: BLE001 -- bookkeeping must not undo the revert
            logger.opt(exception=True).warning(
                "change_review: could not record reverted paths"
            )
    return outcomes


def _guarded_uncreate(repo, root: Path, baseline: str, path: str) -> None:
    """Remove a turn-created file — explicitly, and only if absent at B.

    ``checkout B -- path`` errors on a B-absent path, so un-create is a
    delete; the guard makes it impossible to delete something the baseline
    actually contained (that case restores instead). A directory now
    squatting the path is removed only when empty — a non-empty one raises
    (→ an honest per-path failure), never an rmtree of the user's data.

    Args:
        repo: The root's shadow repo.
        root: The root directory.
        baseline: The B snapshot sha.
        path: Root-relative path to un-create.

    Raises:
        OSError: A non-empty directory occupies the path.
    """
    if repo.file_bytes(baseline, path) is not None:
        repo.restore_paths(baseline, [path])
        return
    target = root / path
    if target.is_dir() and not target.is_symlink():
        target.rmdir()
    elif target.is_file() or target.is_symlink():
        target.unlink()
