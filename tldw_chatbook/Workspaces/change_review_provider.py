"""Textual-free data source for the Agent Change Review screen (TASK-36944.7).

``AgentRunsChangeReviewProvider`` and the turn/window helpers it shares with
``UI/Screens/change_review_screen.py`` live here so non-UI callers (the
Console agent bridge) can use them without importing Textual.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Sequence

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Workspaces.change_tracking import (
    ChangedFile,
    ChangeTrackingError,
    ShadowRepoService,
)

#: Default per-file diff display cap; the Settings surface is TASK-1979,
#: but the flat section name is the spec's (dotted sections drop defaults).
DEFAULT_DIFF_DISPLAY_MAX_LINES = 2000


#: User-facing provenance for the three persisted window kinds. Keep this
#: local to Review: importing the Console bridge for three storage literals
#: would invert the display/runtime dependency.
_WINDOW_KIND_LABELS = {
    "turn": "Turn",
    "turn_concurrent_subagent": "Turn with sub-agent overlap",
    "subagent_post_turn": "Sub-agent after turn",
}


def _snapshot_root_key(row: dict) -> str:
    """Return the card-compatible comparison key for a snapshot root."""
    root = str(row.get("root") or "")
    return os.path.normcase(os.path.normpath(root)) if root else ""


def _window_labels(rows: "Sequence[dict]") -> dict[int, str]:
    """Label rows only where one canonical root owns multiple windows.

    Known kinds use concise provenance. Repeated or unknown kinds gain a
    stable ordinal so every affected row remains distinguishable.

    Args:
        rows: One run's snapshot rows in storage order.

    Returns:
        Row-object identities mapped to their visible window labels.
    """
    by_root: dict[str, list[dict]] = {}
    for row in rows:
        by_root.setdefault(_snapshot_root_key(row), []).append(row)

    labels: dict[int, str] = {}
    for root_rows in by_root.values():
        if len(root_rows) < 2:
            continue
        base_labels = [
            _WINDOW_KIND_LABELS.get(str(row.get("kind") or ""), "Change window")
            for row in root_rows
        ]
        totals = {base: base_labels.count(base) for base in set(base_labels)}
        seen: dict[str, int] = {}
        for row, base in zip(root_rows, base_labels):
            seen[base] = seen.get(base, 0) + 1
            labels[id(row)] = f"{base} {seen[base]}" if totals[base] > 1 else base
    return labels


def _file_count_noun(count: int, *, repeated_root: bool) -> str:
    """Name an aggregate without calling repeated paths unique files.

    TASK-31665 AC#6: the non-repeated-root branch returned the plural
    unconditionally, so a one-file turn read "1 files" in the header and
    the turn selector while the Inspect rail's own row for the same fact
    read "1 file". The repeated-root branch already pluralized correctly;
    this makes the two branches agree.
    """
    if repeated_root:
        return "file change" if count == 1 else "file changes"
    return "file" if count == 1 else "files"


@dataclass(frozen=True)
class ReviewTurn:
    """One reviewable turn: an agent run's snapshot rows.

    Attributes:
        run_id: The agent run.
        label: Selector copy (run timestamp + totals).
        rows: The run's ``change_snapshots`` rows (one per root).
    """

    run_id: str
    label: str
    rows: tuple[dict, ...]


class AgentRunsChangeReviewProvider:
    """Concrete data source over (AgentRunsDB, ShadowRepoService).

    Used by production AND by tests — the fixture-invented-shapes trap has
    bitten this repo four separate times, so the tests drive this real
    provider against a real database and real git.
    """

    def __init__(
        self,
        *,
        db: AgentRunsDB,
        service: ShadowRepoService,
        conversation_id: str,
        diff_display_max_lines: int | None = None,
        run_active: "Callable[[], bool] | None" = None,
    ) -> None:
        """Create a provider over one conversation's recorded turns.

        Args:
            db: The runs database holding ``change_snapshots`` rows.
            service: Shadow-repo service for diff content.
            conversation_id: Conversation whose turns are reviewable.
            diff_display_max_lines: Per-file render cap; ``None`` reads the
                flat ``[change_review]`` config section (default 2000).
        """
        self._db = db
        self._service = service
        self._conversation_id = conversation_id
        #: TASK-1974: probe for an active run on this conversation's
        #: workspace. The revert engine refuses while one is live -- the
        #: per-root lock covers git ops, not the agent's own file tools.
        self.run_active = run_active if run_active is not None else (lambda: False)
        self.run_active_for_root: Callable[[str], bool] = lambda _root: (
            self.run_active()
        )
        if diff_display_max_lines is None:
            diff_display_max_lines = self._configured_cap()
        # Review finding: an explicit 0/negative/non-int cap would defeat
        # the windowing guarantee (negative slicing renders almost the full
        # diff) or crash mid-render. Same floor as the configured path.
        try:
            self.diff_display_max_lines = max(50, int(diff_display_max_lines))
        except (TypeError, ValueError):
            self.diff_display_max_lines = DEFAULT_DIFF_DISPLAY_MAX_LINES

    @staticmethod
    def _configured_cap() -> int:
        try:
            from tldw_chatbook.config import get_cli_setting

            value = get_cli_setting(
                "change_review",
                "diff_display_max_lines",
                DEFAULT_DIFF_DISPLAY_MAX_LINES,
            )
            return max(50, int(value))
        except Exception:  # noqa: BLE001 -- a bad config never breaks review
            return DEFAULT_DIFF_DISPLAY_MAX_LINES

    @staticmethod
    def git_actions_enabled() -> bool:
        """Whether the git-modes kill switch (TASK-16801 arc B) is on.

        Reads the flat ``[change_review] git_actions`` config key,
        default True -- the feature ships ON. This is the ONE gate that
        makes the whole `current` mode (pseudo-entry, detection, commit/
        push/PR) disappear from the screen: :meth:`detect_git` returns
        ``{}`` when this is False, and Task 6's screen offers the
        pseudo-entry only when that dict is non-empty.

        Same guard shape as :meth:`_configured_cap`: reading and
        coercing the config value is wrapped in one broad
        ``except Exception`` that falls back to the default -- a bad or
        garbage config value must never break Change Review, and it
        must never silently disable a feature that shipped ON either.

        Returns:
            True unless config explicitly disables git actions.
        """
        try:
            from tldw_chatbook.config import (
                coerce_bool_setting,
                get_cli_setting,
            )

            # coerce_bool_setting now returns the default for a None value
            # (TASK-32808.4), so the former None workaround is gone. It also
            # handles hand-edited spellings like "false" that bool(value)
            # got wrong (Qodo #2, PR #1914).
            return coerce_bool_setting(
                get_cli_setting("change_review", "git_actions", True), True
            )
        except Exception:  # noqa: BLE001 -- a bad config never breaks review
            return True

    def turns(self) -> list[ReviewTurn]:
        """Reviewable turns, NEWEST first (the screen opens on the latest).

        Returns:
            One :class:`ReviewTurn` per run that has snapshot rows.
        """
        by_run: dict[str, list[dict]] = {}
        order: list[str] = []
        for row in self._db.change_snapshots_for_conversation(self._conversation_id):
            run_id = str(row["run_id"])
            if run_id not in by_run:
                by_run[run_id] = []
                order.append(run_id)
            by_run[run_id].append(row)
        return [
            self._build_review_turn(run_id, by_run[run_id])
            for run_id in reversed(order)
        ]

    def turn_for_run(self, run_id: str) -> "ReviewTurn | None":
        """One run's :class:`ReviewTurn`, read run-scoped (no history scan).

        The turn file card renders exactly one run per card, and a
        transcript can hold many cards -- resolving each through
        :meth:`turns` would scan and group the whole conversation's
        snapshot history per card (Qodo, PR #1728). This reads only the
        run's own rows and builds the identical ``ReviewTurn`` the scan
        would have produced.

        Args:
            run_id: The agent run to load.

        Returns:
            The run's ``ReviewTurn``, or ``None`` when it has no
            snapshot rows.
        """
        rows = self._db.change_snapshots_for_run_review(run_id)
        if not rows:
            return None
        return self._build_review_turn(str(run_id), rows)

    @staticmethod
    def _build_review_turn(run_id: str, rows: list[dict]) -> ReviewTurn:
        """Assemble a :class:`ReviewTurn` from a run's snapshot rows.

        Shared by :meth:`turns` and :meth:`turn_for_run` so the two paths
        can never disagree about label or row order.

        Args:
            run_id: The agent run the rows belong to.
            rows: The run's ``change_snapshots`` rows, oldest first.

        Returns:
            The assembled turn.
        """
        files = sum(int(r["files_changed"] or 0) for r in rows)
        adds = sum(int(r["adds"] or 0) for r in rows)
        dels = sum(int(r["dels"] or 0) for r in rows)
        stamp = str(rows[0].get("created_at", ""))[:19].replace("T", " ")
        repeated_root = bool(_window_labels(rows))
        return ReviewTurn(
            run_id=run_id,
            label=(
                f"{stamp} · {files} "
                f"{_file_count_noun(files, repeated_root=repeated_root)} "
                f"+{adds} −{dels}"
            ),
            rows=tuple(rows),
        )

    def tool_touched_relpaths(self, row: dict) -> "set[str] | None":
        """Root-relative paths the run's recorded WRITE tools touched.

        TASK-1978: the badge derivation. Uses the SAME extractor the
        force-add carve-out runs over the SAME persisted step shape, so
        the badge and the carve-out can never disagree about provenance.

        Args:
            row: One ``change_snapshots`` row.

        Returns:
            The touched set, or ``None`` when the run has no recorded
            steps (older data) — callers must then render NO badges
            rather than badging everything.
        """
        run = self._db.get_run(str(row.get("run_id") or ""))
        steps = (run or {}).get("steps") or []
        if not steps:
            return None
        from pathlib import Path as _P

        from tldw_chatbook.Workspaces.change_turn_tracker import (
            ChangeTurnTracker,
        )

        root = _P(str(row["root"]))
        rel: set[str] = set()
        for raw in ChangeTurnTracker.tool_touched_paths(steps):
            try:
                rel.add(_P(raw).expanduser().resolve().relative_to(root).as_posix())
            except (ValueError, OSError):
                continue
        return rel

    def snapshots_pruned(self, row: dict) -> bool:
        """Whether a row's snapshots no longer exist (retention reset).

        Args:
            row: One ``change_snapshots`` row.

        Returns:
            True when either recorded sha is gone from the shadow repo —
            the row's history was pruned by retention (TASK-1975).
        """
        try:
            repo = self._service.repo_for_root(row["root"])
            return not (
                repo.has_snapshot(str(row.get("baseline_sha") or ""))
                and repo.has_snapshot(str(row.get("end_sha") or ""))
            )
        except ChangeTrackingError:
            # The root itself vanished — the diff is equally unrenderable;
            # the generic unavailable copy handles it.
            return False

    def changed_files(self, row: dict) -> list[ChangedFile]:
        """A snapshot row's changed files (empty for tracking-error rows).

        Args:
            row: One ``change_snapshots`` row.

        Returns:
            The row's :class:`ChangedFile` list from the shadow repo.
        """
        if row.get("tracking_error") or not row.get("end_sha"):
            return []
        repo = self._service.repo_for_root(row["root"])
        return repo.changed_files(str(row["baseline_sha"]), str(row["end_sha"]))

    def preflight_revert(self, row: dict, paths: list[str]) -> RevertPreflight:
        """The confirm dialog's data: which paths were edited after E.

        Args:
            row: One ``change_snapshots`` row.
            paths: Paths the user asked to revert.

        Returns:
            The engine's :class:`RevertPreflight`.
        """
        from tldw_chatbook.Workspaces.change_revert import preflight_revert

        return preflight_revert(self._service, row, paths)

    def revert(self, row: dict, paths: list[str]) -> list[RevertOutcome]:
        """Restore ``paths`` to the turn's baseline.

        Args:
            row: One ``change_snapshots`` row.
            paths: Paths to restore.

        Returns:
            The engine's per-path :class:`RevertOutcome` list.

        Raises:
            RevertRefusedError: A run is active on this workspace.
        """
        from tldw_chatbook.Workspaces.change_revert import revert_paths

        return revert_paths(
            self._service,
            self._db,
            row,
            paths,
            run_active=lambda: self.run_active_for_root(str(row["root"])),
        )

    def diff_text(self, row: dict, path: str) -> str:
        """One file's unified diff for a snapshot row.

        Args:
            row: One ``change_snapshots`` row.
            path: Root-relative changed path.

        Returns:
            Unified diff text.
        """
        repo = self._service.repo_for_root(row["root"])
        return repo.diff_text(str(row["baseline_sha"]), str(row["end_sha"]), path)

    def add_change_note(
        self,
        *,
        run_id: str,
        root: str,
        path: str,
        hunk_index: int,
        hunk_header: str,
        hunk_excerpt: str,
        note: str,
        snapshot_id: int | None = None,
        anchor_kind: str = "hunk",
        diff_line_index: int | None = None,
        diff_line_text: str | None = None,
    ) -> int:
        """Record a user-authored note anchored to a turn's diff.

        Thin delegate onto :meth:`AgentRunsDB.add_change_note` (TASK-16800
        spec §1/§3, anchor kinds extended by TASK-18060 Task 1 / review-rail
        spec §4) -- the turn file card and the Review screen's comment
        affordances write notes through the provider exactly like every
        other change-review read/write, never touching the database
        directly.

        Args:
            run_id: The agent run whose diff this note is anchored to.
            root: Canonical root path of the changed file.
            path: The changed file's path (root-relative).
            hunk_index: 0-based index of the hunk over the file's full diff,
                or ``-1`` for a ``"file"`` note's sentinel.
            hunk_header: The hunk's ``"@@ -a,b +c,d @@ ..."`` line, verbatim,
                or ``""`` for a ``"file"`` note's sentinel.
            hunk_excerpt: The hunk body captured at note time (already
                capped/elided by the caller), or ``""`` for a ``"file"``
                note.
            note: The user's note text.
            snapshot_id: The owning ``change_snapshots`` row's own DB
                ``id`` (Qodo #6, PR #1779 fix round) -- disambiguates
                which of two same-run/root/path windows this note's hunk
                came from. ``None`` when the caller has no snapshot row
                to anchor to.
            anchor_kind: ``"hunk"`` (default), ``"file"``, or
                ``"diff_line"`` (TASK-18060 Task 1). The default keeps
                every existing caller of this delegate byte-compatible.
            diff_line_index: 0-based index over the file's full diff text,
                required for ``"diff_line"`` notes and ``None`` otherwise.
            diff_line_text: The anchored line, captured verbatim at
                note-creation time, required for ``"diff_line"`` notes and
                ``None`` otherwise.

        Returns:
            The newly created note's row id.
        """
        return self._db.add_change_note(
            run_id=run_id,
            root=root,
            path=path,
            hunk_index=hunk_index,
            hunk_header=hunk_header,
            hunk_excerpt=hunk_excerpt,
            note=note,
            snapshot_id=snapshot_id,
            anchor_kind=anchor_kind,
            diff_line_index=diff_line_index,
            diff_line_text=diff_line_text,
        )

    def delete_change_note(self, note_id: int) -> bool:
        """Delete a pending (undelivered) note.

        Thin delegate onto :meth:`AgentRunsDB.delete_change_note`.

        Args:
            note_id: The note's row id.

        Returns:
            True if a pending note was deleted; False if the note does
            not exist or has already been delivered.
        """
        return self._db.delete_change_note(note_id)

    def notes_for_run(self, run_id: str) -> list[dict]:
        """Return a run's change notes, oldest first.

        Thin delegate onto :meth:`AgentRunsDB.notes_for_run`.

        Args:
            run_id: The agent run id.

        Returns:
            One dict per note row (all columns), oldest first.
        """
        return self._db.notes_for_run(run_id)

    # -- Git modes (TASK-16801 arc B) -----------------------------------
    #
    # Thin wrappers over `Workspaces/git_workspace.py` -- no logic here
    # beyond the kill-switch read, root resolution/dedupe, and straight
    # delegation. Worker dispatch is the SCREEN's job (Task 6); every
    # method below is synchronous.
    #
    # CONTRACT (binding for every caller -- Tasks 6-8): the wrapped
    # engine functions have DELIBERATELY ASYMMETRIC error postures,
    # preserved here exactly:
    #   - `detect_git` NEVER raises (each entry is Info | Refusal | None).
    #   - `current_status` / `current_diff_text` / `untracked_preview`
    #     RAISE `GitWorkspaceError` on a git failure.
    #   - `commit_selected` RAISES: `CommitRefusedError` (active run) and
    #     `GitWorkspaceError` (empty files / blank message / git
    #     failure). Do not wrap this in try/except here -- let it
    #     propagate; the screen catches it.
    #   - `push_current` RAISES `GitWorkspaceError` for detached HEAD and
    #     no-remote; otherwise RETURNS a `PushResult` whose `state` can
    #     be `"failed"` -- a failed push is a returned value, not an
    #     exception.
    #   - `pr_url` NEVER raises -- it returns `str` or
    #     `GitWorkspaceRefusal`; callers use `isinstance`, never
    #     try/except.

    def detect_git(
        self, roots: "Sequence[str]"
    ) -> "dict[str, GitWorkspaceInfo | GitWorkspaceRefusal | None]":
        """Detect real git repositories at ``roots``, keyed by resolved root.

        Never raises -- delegates to
        :func:`~tldw_chatbook.Workspaces.git_workspace.detect_git_workspace`,
        which itself never raises.

        Args:
            roots: Candidate workspace roots, any spelling (relative,
                symlinked, trailing slash, ...).

        Returns:
            ``{}`` when :meth:`git_actions_enabled` is False -- this
            single check is what makes the whole `current` mode vanish
            from the screen. Otherwise one entry per DISTINCT resolved
            root (``str(Path(root).resolve())``), so two spellings of
            the same directory dedupe to one detection call and one key.
            Both :class:`~tldw_chatbook.Workspaces.git_workspace.GitWorkspaceInfo.root`
            and :class:`~tldw_chatbook.Workspaces.git_workspace.CurrentRootStatus.root`
            are always resolved paths -- keying by the resolved spelling
            here means a caller that looks a root up by ITS resolved
            spelling can never get a silent miss because this dict used
            the raw input spelling instead (the exact bug class Task 2's
            engine layer was fixed against, by construction).
        """
        if not self.git_actions_enabled():
            return {}

        from pathlib import Path

        from tldw_chatbook.Workspaces.git_workspace import detect_git_workspace

        result: "dict[str, GitWorkspaceInfo | GitWorkspaceRefusal | None]" = {}
        for raw_root in roots:
            resolved_key = str(Path(raw_root).resolve())
            if resolved_key in result:
                continue
            result[resolved_key] = detect_git_workspace(Path(raw_root))
        return result

    def current_status(self, root: str) -> "CurrentRootStatus":
        """The real working tree's status at ``root``, freshly detected.

        Re-detects (fresh, not cached) before reading status -- spec §4:
        the `current` mode's worker re-detects on every load, since the
        repo's branch/upstream/ahead-behind can move between reloads.

        Args:
            root: Workspace root, any spelling.

        Returns:
            The root's :class:`~tldw_chatbook.Workspaces.git_workspace.CurrentRootStatus`.

        Raises:
            GitWorkspaceError: ``root`` is no longer a detectable git
                repository (removed, or now refused -- e.g. it moved
                inside another repo since the mode was offered), or the
                underlying git invocation failed.
        """
        from pathlib import Path

        from tldw_chatbook.Workspaces.git_workspace import (
            GitWorkspaceError,
            GitWorkspaceInfo,
            detect_git_workspace,
            working_tree_status,
        )

        path = Path(root)
        info = detect_git_workspace(path)
        if not isinstance(info, GitWorkspaceInfo):
            reason = info.reason if info is not None else "not a git repository"
            raise GitWorkspaceError(
                f"git workspace detection failed for {root}: {reason}"
            )
        return working_tree_status(path, info)

    def current_diff_text(self, root: str, change: "ChangedFile") -> str:
        """One TRACKED file's unified diff, working tree vs HEAD.

        Tracked-only: an untracked ``change`` (``change.path in
        status.untracked``) must be routed by the SCREEN through
        :meth:`untracked_preview` instead -- this method always calls
        :func:`~tldw_chatbook.Workspaces.git_workspace.working_tree_diff`,
        which is a fatal git error against an unborn HEAD or an
        untracked path.

        Args:
            root: Workspace root.
            change: The tracked changed file to diff.

        Returns:
            Unified diff text.

        Raises:
            GitWorkspaceError: The git invocation failed.
        """
        from pathlib import Path

        from tldw_chatbook.Workspaces.git_workspace import working_tree_diff

        return working_tree_diff(Path(root), change.path)

    def untracked_preview(self, root: str, path: str) -> str:
        """A bounded preview of one untracked file, capped at the screen's limit.

        Args:
            root: Workspace root.
            path: Root-relative path of the untracked file.

        Returns:
            The preview text (see
            :func:`~tldw_chatbook.Workspaces.git_workspace.untracked_preview`
            for the exact rendering rules). Never raises -- I/O errors
            render as an honest one-line message instead.
        """
        from pathlib import Path

        from tldw_chatbook.Workspaces.git_workspace import (
            untracked_preview as _untracked_preview,
        )

        return _untracked_preview(Path(root), path, self.diff_display_max_lines)

    def commit_selected(
        self,
        root: str,
        files: "Sequence[str]",
        message: str,
        new_branch: "str | None",
    ) -> "CommitResult":
        """Commit exactly ``files`` at ``root``, threading the run-active probe.

        Passes :attr:`run_active` through as the keyword-only
        ``run_active=`` argument -- exactly how :meth:`revert` threads it
        into ``revert_paths``.

        Args:
            root: Workspace root (must be the repo toplevel).
            files: Root-relative paths to stage and commit.
            message: The commit message.
            new_branch: When set, create and check out this branch
                before committing. ``None``/empty commits to the current
                branch.

        Returns:
            The engine's :class:`~tldw_chatbook.Workspaces.git_workspace.CommitResult`.

        Raises:
            CommitRefusedError: A run is active on this workspace --
                this method does NOT catch it; the caller (screen) must.
            GitWorkspaceError: ``files`` is empty, ``message`` is blank,
                or a git step failed outside the returned per-step
                outcomes (the engine still raises for these two
                preconditions rather than returning a silent no-op).
        """
        from pathlib import Path

        from tldw_chatbook.Workspaces.git_workspace import (
            commit_selected as _commit_selected,
        )

        return _commit_selected(
            Path(root),
            files,
            message,
            new_branch,
            run_active=lambda: self.run_active_for_root(root),
        )

    def push_current(
        self, root: str, info: "GitWorkspaceInfo", remote: "str | None"
    ) -> "PushResult":
        """Push the current branch at ``root``.

        NOT gated on :attr:`run_active` -- push only ships already-
        committed state; the working tree is untouched (spec §6
        states this explicitly, in contrast to commit).

        Args:
            root: Workspace root (must be the repo toplevel).
            info: The root's detected
                :class:`~tldw_chatbook.Workspaces.git_workspace.GitWorkspaceInfo`.
            remote: Explicit target remote name, or ``None`` to derive one.

        Returns:
            The engine's :class:`~tldw_chatbook.Workspaces.git_workspace.PushResult`.
            **A failed push is a RETURNED result** (``state ==
            "failed"``), never an exception -- callers must check
            ``result.state``, not wrap this call in try/except for the
            ordinary rejected-push case.

        Raises:
            GitWorkspaceError: HEAD is detached, or no remote could be
                resolved. These are precondition failures, not push
                outcomes, so they raise rather than returning a
                ``PushResult``.
        """
        from pathlib import Path

        from tldw_chatbook.Workspaces.git_workspace import (
            push_current as _push_current,
        )

        return _push_current(Path(root), info, remote)

    def pr_url(
        self, root: str, info: "GitWorkspaceInfo"
    ) -> "str | GitWorkspaceRefusal":
        """Build the compare/merge-request URL for the current branch.

        Args:
            root: Workspace root (used only for the codeberg/Gitea-family
                local default-branch lookup).
            info: The root's detected
                :class:`~tldw_chatbook.Workspaces.git_workspace.GitWorkspaceInfo`.

        Returns:
            The compare URL, or a
            :class:`~tldw_chatbook.Workspaces.git_workspace.GitWorkspaceRefusal`
            naming why one can't be built. **Never raises** -- callers
            must use ``isinstance(result, GitWorkspaceRefusal)``, never
            try/except.
        """
        from pathlib import Path

        from tldw_chatbook.Workspaces.git_workspace import pr_compare_url

        return pr_compare_url(Path(root), info)
