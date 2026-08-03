"""Snapshot retention + shadow-repo GC for Agent Change Review (TASK-1975).

Retention has three moves, run on the app's existing maintenance path:

1. **Row prune** — ``change_snapshots`` rows older than ``retention_days``
   are deleted; their history rows disappear from the Review screen's turn
   selector.
2. **Repo shrink** — every snapshot is an ancestor of the shadow repo's
   HEAD, so ordinary ``gc`` can never collect referenced history; instead,
   a repo whose root has NO remaining rows is RESET (its git dir removed —
   the next snapshot re-initializes from scratch), and a repo that still
   backs rows gets ``reflog expire`` + ``gc --prune=now`` to collect the
   genuinely unreachable (superseded stage blobs, revert staging).
3. **Orphan GC** — a shadow repo whose ROOT no longer exists is removed
   once the repo directory itself ages past retention (a fresh orphan may
   still be re-bound; an old one is dead weight).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

from tldw_chatbook.Workspaces.change_bounds import (
    DEFAULT_RETENTION_DAYS,
    change_review_setting,
)
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService


@dataclass(frozen=True)
class PruneReport:
    """What one retention pass did.

    Attributes:
        rows_pruned: ``change_snapshots`` rows deleted.
        repos_reset: Shadow repos removed because no rows reference their
            root anymore (recreated on next use).
        repos_gcd: Live repos that received reflog-expire + gc.
        orphans_removed: Repo dirs removed because their root vanished and
            the repo aged past retention.
    """

    rows_pruned: int = 0
    repos_reset: int = 0
    repos_gcd: int = 0
    orphans_removed: int = 0


def _repo_root(git_dir: Path) -> str | None:
    """Read a shadow repo's bound root from its pinned ``core.worktree``."""
    try:
        proc = subprocess.run(
            ["git", "--git-dir", str(git_dir), "config", "core.worktree"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def prune_change_history(
    db,
    service: ShadowRepoService,
    *,
    now: datetime | None = None,
    retention_days: int | None = None,
) -> PruneReport:
    """Run one retention pass. Best-effort per repo; never raises.

    Args:
        db: The ``AgentRunsDB`` holding ``change_snapshots`` rows.
        service: Shadow-repo service whose data dir is swept.
        now: Injected clock for tests; defaults to UTC now.
        retention_days: Override; ``None`` reads the flat
            ``[change_review]`` knob (default 30).

    Returns:
        A :class:`PruneReport` of what happened.
    """
    if retention_days is None:
        retention_days = change_review_setting(
            "retention_days", DEFAULT_RETENTION_DAYS
        )
    if retention_days <= 0:
        return PruneReport()
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=retention_days)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    rows_pruned = 0
    try:
        rows_pruned = db.delete_change_snapshots_older_than(cutoff_iso)
    except Exception:  # noqa: BLE001 -- retention must never crash the app
        logger.opt(exception=True).warning(
            "change_review: retention row-prune failed"
        )
    try:
        live_roots = db.roots_with_change_snapshots()
    except Exception:  # noqa: BLE001
        logger.opt(exception=True).warning(
            "change_review: could not list live roots; skipping repo sweep"
        )
        return PruneReport(rows_pruned=rows_pruned)

    repos_reset = 0
    repos_gcd = 0
    orphans_removed = 0
    data_dir = getattr(service, "_data_dir", None)
    if data_dir is None or not Path(data_dir).is_dir():
        return PruneReport(rows_pruned=rows_pruned)

    # Layout: <data_dir>/<root-hash>/git (plus hooks/ and the lockdir) --
    # each CONTAINER dir is one root's shadow state; removal takes the
    # whole container so hooks and stale locks go with it.
    for container in sorted(Path(data_dir).iterdir()):
        git_dir = container / "git"
        if not container.is_dir() or not (git_dir / "HEAD").exists():
            continue
        try:
            root = _repo_root(git_dir)
            if root is None or not Path(root).is_dir():
                # Orphan: the bound root vanished. Old orphans are dead
                # weight; fresh ones may still be re-bound.
                age = now.timestamp() - git_dir.stat().st_mtime
                if age > retention_days * 86400:
                    shutil.rmtree(container, ignore_errors=True)
                    orphans_removed += 1
                continue
            if root not in live_roots:
                # No rows left for this root — nothing the Review screen
                # can show needs these objects. Reset; next snapshot
                # re-initializes. This is the shrink move: ancestry keeps
                # every snapshot reachable, so gc alone cannot collect it.
                shutil.rmtree(container, ignore_errors=True)
                repos_reset += 1
                continue
            repo = service.repo_for_root(root)
            with repo._locked():  # noqa: SLF001 -- retention is a peer op
                repo._run(  # noqa: SLF001
                    "reflog", "expire", "--expire=now", "--all", check=False
                )
                repo._run(  # noqa: SLF001
                    "gc", "--prune=now", "--quiet", check=False
                )
            repos_gcd += 1
        except Exception:  # noqa: BLE001 -- one bad repo must not stop the sweep
            logger.opt(exception=True).warning(
                f"change_review: retention sweep failed for {git_dir}"
            )
    report = PruneReport(
        rows_pruned=rows_pruned,
        repos_reset=repos_reset,
        repos_gcd=repos_gcd,
        orphans_removed=orphans_removed,
    )
    logger.info(
        "change_review: retention pass "
        f"rows={report.rows_pruned} reset={report.repos_reset} "
        f"gc={report.repos_gcd} orphans={report.orphans_removed}"
    )
    return report


def run_retention_for_app(
    db_path: Path | str,
    *,
    service: ShadowRepoService | None = None,
) -> PruneReport | None:
    """One maintenance-path retention pass for the production layout.

    Args:
        db_path: The ChaChaNotes DB path — the runs DB lives beside it as
            ``agent_runs.db`` (the same siting rule the Console bridge
            uses).
        service: Injection seam for tests; defaults to the app's shadow
            service.

    Returns:
        The pass report, or ``None`` when git is unavailable or the pass
        could not run (never raises).
    """
    try:
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        if service is None:
            service = ShadowRepoService()
        if not service.available:
            return None
        runs_db = AgentRunsDB(Path(db_path).parent / "agent_runs.db")
        return prune_change_history(runs_db, service)
    except Exception:  # noqa: BLE001 -- maintenance must never crash the app
        logger.opt(exception=True).warning(
            "change_review: retention pass failed"
        )
        return None
