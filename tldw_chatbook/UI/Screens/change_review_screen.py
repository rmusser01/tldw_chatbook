"""The Agent Change Review screen (TASK-1973).

A PUSHED overlay (``push_screen`` from the Console, ``Esc`` returns) — not a
``BaseAppScreen`` tab: the spec deliberately chose an overlay so diffs get
full real estate without a new tab route.

Layout: header (turn selector, totals), honesty banners (tracking errors),
left changed-file tree grouped Added/Modified/Deleted/Renamed (rare git
letters bucket as Other — TASK-1970's verbatim-status contract), right diff
pane. The diff pane mounts ONLY the focused file — a 50k-line generated
file must not freeze the screen — and renders as a Rich ``Text`` assembled
line by line with diff coloring, never markup-parsed: file content is data
(the transcript's literal-backslash lesson).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from tldw_chatbook.Workspaces.change_revert import (
        RevertOutcome,
        RevertPreflight,
    )
    from tldw_chatbook.Workspaces.git_workspace import (
        CommitResult,
        CurrentRootStatus,
        GitWorkspaceInfo,
        GitWorkspaceRefusal,
        PushResult,
    )

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.geometry import Region
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Checkbox, Input, Select, Static, Tree

from tldw_chatbook.Chat.console_display_state import (
    ConversationFileEntry,
    DiffHunk,
    conversation_file_summary,
    hunk_excerpt,
    split_unified_diff,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph
from tldw_chatbook.Workspaces.change_tracking import (
    ChangedFile,
    ChangeTrackingError,
    ShadowRepoService,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

#: Default per-file diff display cap; the Settings surface is TASK-1979,
#: but the flat section name is the spec's (dotted sections drop defaults).
DEFAULT_DIFF_DISPLAY_MAX_LINES = 2000

#: How many nested-repo paths the disclosure banner names before "+N more"
#: (TASK-1976; Qodo #1254 asked for the limit to be discoverable).
NESTED_BANNER_NAMED_LIMIT = 5

#: TASK-16801 arc B (spec §4): the turn ``Select``'s ONE pseudo-entry value,
#: standing for "the real working tree" rather than a recorded turn. Run ids
#: are UUIDs, so this literal can never collide with one.
CURRENT_MODE_SENTINEL = "__git_current__"

#: The pseudo-entry's label stem; the branch state is appended per repo.
CURRENT_MODE_LABEL = "Working tree (current)"

#: Spec §4.1's row-consumers table, verbatim. Revert is snapshot-anchored
#: (it restores a turn's baseline) and notes anchor to ``change_snapshots``
#: rows -- neither has any meaning against the real working tree, so both
#: refuse with copy instead of acting on the pseudo row.
CURRENT_MODE_REVERT_REFUSAL = "revert works on recorded turns — select a turn"
CURRENT_MODE_COMMENT_REFUSAL = "comments attach to recorded turns"

#: TASK-16801 arc B (spec §5): commit/push/PR share ONE worker group,
#: deliberately NOT ``"change-review-current"`` (the status-read group). An
#: exclusive re-dispatch cancels the worker TASK but a queued
#: ``call_from_thread`` still lands, so a mutation and a status read sharing
#: a group could cancel each other mid-flight (Task 6's carry-forward).
GIT_ACTION_WORKER_GROUP = "change-review-git-action"

#: The run-active refusal shown BEFORE any modal (spec §5 step 1). The same
#: sentence :class:`~tldw_chatbook.Workspaces.git_workspace.CommitRefusedError`
#: carries, so the screen's early refusal and the engine's own last-moment
#: one can never read differently to the user.
COMMIT_RUN_ACTIVE_REFUSAL = (
    "a run is active on this workspace — finish or stop the run first"
)

#: Why the `Commit…` button is disabled, as tooltip AND notify copy -- spec
#: §8: every disabled action carries its reason, never a dead control.
COMMIT_CLEAN_TREE_REASON = "working tree clean — nothing to commit"
COMMIT_BUSY_REASON = "a git action is already running"
COMMIT_TURN_MODE_REFUSAL = (
    "commit works on the working tree — select “Working tree (current)”"
)

#: The same sentence, under the name push/PR read it by (T8). One literal,
#: so the three git affordances can never disagree about what "busy" means.
GIT_BUSY_REASON = COMMIT_BUSY_REASON

#: TASK-16801 arc B (spec §6): push/PR refusal + tooltip copy.
#:
#: The two push reasons are the ENGINE's own sentences
#: (``git_workspace.push_current`` raises ``GitWorkspaceError`` with exactly
#: these strings), reused here on purpose: the disabled tooltip a user reads
#: before pressing and the error they would read after must be one literal,
#: never two that can drift apart.
PUSH_NO_REMOTE_REASON = "no git remote configured"
PUSH_DETACHED_REASON = "no branch checked out"
#: A branch whose NAME begins with ``"-"`` cannot be pushed from here: git
#: would read it as an option in ``push -u <remote> <branch>`` (T8
#: re-review -- ``--mirror`` deletes remote refs, ``--all`` publishes every
#: local branch). Same literal as the engine's refusal, for the same reason
#: the two above are.
PUSH_OPTION_BRANCH_REASON = "unsupported branch name"
#: Likewise ``pr_compare_url``'s own no-upstream refusal.
PR_NO_UPSTREAM_REASON = "push the branch first"
#: Neither action has a repository to act on at all (detection found none,
#: or the only detected root has since gone).
GIT_NO_REPO_REASON = "no git repository detected"
PUSH_TURN_MODE_REFUSAL = (
    "push works on the working tree — select “Working tree (current)”"
)
PR_TURN_MODE_REFUSAL = (
    "opening a PR works on the working tree — select “Working tree (current)”"
)

#: The footer's key legend, per mode. In `current` mode the snapshot-only
#: keys stop being advertised and the line says WHY (T6 re-review finding
#: (a): the affordances must present as unavailable rather than looking live
#: and failing on press).
FOOTER_TURN_MODE = (
    "j/k files · Enter diff · c comment line · C comment file · Esc back"
)
FOOTER_CURRENT_MODE = (
    "j/k files · Enter diff · g commit · p push · P open PR · Esc back · "
    "revert (u/U) and comments (c/C) need a recorded turn"
)

#: Group headings in display order. "Other" carries the rare git letters
#: (T typechange, C copy) that pass through verbatim rather than being
#: coerced — see ``ChangedFile.status``.
_GROUPS: tuple[tuple[str, str], ...] = (
    ("A", "Added"),
    ("M", "Modified"),
    ("D", "Deleted"),
    ("R", "Renamed"),
)
_OTHER_GROUP = "Other"


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
            from tldw_chatbook.config import get_cli_setting

            value = get_cli_setting("change_review", "git_actions", True)
            return bool(value)
        except Exception:  # noqa: BLE001 -- a bad config never breaks review
            return True

    def turns(self) -> list[ReviewTurn]:
        """Reviewable turns, NEWEST first (the screen opens on the latest).

        Returns:
            One :class:`ReviewTurn` per run that has snapshot rows.
        """
        by_run: dict[str, list[dict]] = {}
        order: list[str] = []
        for row in self._db.change_snapshots_for_conversation(
            self._conversation_id
        ):
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
        return ReviewTurn(
            run_id=run_id,
            label=f"{stamp} · {files} files +{adds} −{dels}",
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
                rel.add(
                    _P(raw)
                    .expanduser()
                    .resolve()
                    .relative_to(root)
                    .as_posix()
                )
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

    def preflight_revert(
        self, row: dict, paths: list[str]
    ) -> RevertPreflight:
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
            self._service, self._db, row, paths, run_active=self.run_active
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

    def conversation_changed_files(
        self,
        row_cache: "dict[int, list[ChangedFile]] | None" = None,
    ) -> "tuple[list[ConversationFileEntry], int]":
        """Cross-turn latest-state summary of the WHOLE conversation.

        TASK-18060 Task 2 (review-rail spec §1): reads every clean
        ``change_snapshots`` row for :attr:`_conversation_id`, calls
        :meth:`changed_files` on each (one shadow-repo diff -- a git
        subprocess pair -- PER ROW), joins per-file note counts in one
        query, and delegates assembly to
        :func:`conversation_file_summary`.

        A row is "clean" the same way :meth:`changed_files` itself already
        guards: ``tracking_error`` falsy AND ``end_sha`` truthy. A clean
        row can still raise :class:`ChangeTrackingError` when retention
        pruned its snapshots out from under it (:meth:`snapshots_pruned`)
        -- that row is skipped and counted in ``pruned_rows`` rather than
        failing the whole summary (spec §1's honest "history pruned for N
        turns" tail line; the Review screen's own ``_load_turn`` uses the
        identical per-row try/except posture).

        **NEVER call this on the UI thread** -- unlike this screen's own
        synchronous diff-on-focus reads, this walks the conversation's
        ENTIRE snapshot history and can run many git subprocesses in one
        call. Callers (the Inspector rail's cached-summary worker, §2)
        must run it off-thread (``asyncio.to_thread`` or a worker) and
        land the result via ``call_from_thread``.

        Args:
            row_cache: Optional per-row git-diff memo, keyed by the
                owning ``change_snapshots`` row's own DB ``id`` (spec §2's
                stated per-row memo, fix round). A row already present is
                reused verbatim instead of re-running :meth:`changed_files`
                (a git subprocess pair) for it; a row not yet present is
                computed and then stored into this SAME dict, in place --
                so a caller that reuses one dict across repeated calls
                (the rail's cached-summary worker) only pays git cost for
                turns it has not diffed before (measured: ~18ms per
                row-pair: a degenerate hundred-turn conversation's
                recompute cost is otherwise quadratic across the
                conversation's lifetime, ~900ms already by turn 50).
                ``None`` (the default -- every pre-fix-round caller,
                including :class:`ConsoleTurnFileCard`'s per-turn reads
                via other methods) computes every row fresh each call,
                byte-identical to behavior before this parameter existed.
                A row that raises :class:`ChangeTrackingError` (pruned) is
                NEVER cached -- there is nothing valid to store, and a
                pruned row is rare enough that re-probing it each call is
                an acceptable, disclosed cost.

        Returns:
            ``(entries, pruned_rows)`` -- the cross-turn summary and how
            many otherwise-clean rows were skipped because retention
            pruned their snapshots.
        """
        rows = self._db.change_snapshots_for_conversation(self._conversation_id)
        clean_rows = [
            row
            for row in rows
            if not row.get("tracking_error") and row.get("end_sha")
        ]
        rows_with_files: list[tuple[dict, list[ChangedFile]]] = []
        pruned_rows = 0
        for row in clean_rows:
            row_id = row.get("id")
            cached = (
                row_cache.get(row_id)
                if row_cache is not None and row_id is not None
                else None
            )
            if cached is not None:
                rows_with_files.append((row, cached))
                continue
            try:
                files = self.changed_files(row)
            except ChangeTrackingError:
                pruned_rows += 1
                continue
            if row_cache is not None and row_id is not None:
                row_cache[row_id] = files
            rows_with_files.append((row, files))
        note_counts = self._db.change_note_counts_for_conversation(
            self._conversation_id
        )
        entries = conversation_file_summary(rows_with_files, note_counts)
        return entries, pruned_rows

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
            reason = (
                info.reason if info is not None else "not a git repository"
            )
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
            Path(root), files, message, new_branch, run_active=self.run_active
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


#: Cursor line background (TASK-18060 Task 6, review-rail spec §3) — an
#: explicit Rich style, never markup: the cursor is applied by appending a
#: STYLE onto the (already-data) line text, exactly like the existing
#: +/-/@@ diff coloring right below it, so the same "content is data, never
#: markup-parsed" discipline covers it.
_CURSOR_LINE_STYLE = "on grey37"

#: Note text cap (TASK-18060 Task 7, review-rail spec §3) — mirrors
#: ``ConsoleTurnFileCard.NOTE_MAX_LENGTH`` (TASK-16800 spec §1) so the
#: Review screen's comment inputs enforce the identical boundary. Kept as
#: a local constant rather than importing the card's private module-level
#: name, to avoid a cross-module private import between the two comment-
#: creation surfaces; both derive from the same spec value.
NOTE_MAX_LENGTH = 2000

#: TASK-18060 Task 7: the notes strip's glyphs, routed through
#: ``resolve_glyph`` for terminal-fallback safety — same vocabulary as
#: ``ConsoleTurnFileCard``'s ``_GLYPH_NOTE``/``_GLYPH_DELETE``.
_GLYPH_NOTE = "✎"
_GLYPH_DELETE = "✕"
#: TASK-18060 Task 7 fix round (spec §3 Note display): the inline marker
#: appended to a diff line carrying a `diff_line` note. Deliberately the
#: SAME "●" the run-marker vocabulary already maps to `"[*]"` in ASCII
#: mode (`Widgets/glyph_fallback.py`'s character-keyed table) -- reusing
#: it (rather than minting a new mapping) is what the reviewer asked for
#: ("via resolve_glyph for ●"), and the fallback still reads sensibly as
#: a generic marker glyph ahead of the literal word "comment".
_GLYPH_LINE_COMMENT_MARKER = "●"


def _validate_note_text(raw: str) -> "str | None":
    """Strip and bound-check a comment before it reaches the DB.

    Mirrors ``console_turn_file_card._validate_note_text`` (TASK-16800
    spec §1) verbatim: the strip-then-empty check runs first because
    ``input_validation.validate_text_input`` treats an empty string as
    valid (it only rejects *oversized*/dangerous text).

    Args:
        raw: The raw ``Input.value`` text.

    Returns:
        The stripped comment text, or ``None`` when it is empty or fails
        validation (over ``NOTE_MAX_LENGTH`` chars, or a dangerous
        HTML/script pattern).
    """
    text = (raw or "").strip()
    if not text:
        return None
    if not validate_text_input(text, max_length=NOTE_MAX_LENGTH):
        return None
    return text


def _note_kind_label(note: dict) -> str:
    """Render one note's kind as the strip's short label (spec §3).

    Args:
        note: A ``change_notes`` row dict.

    Returns:
        ``"file"``, ``"line <index>"``, or ``"hunk"`` (the default for
        any row whose ``anchor_kind`` is missing or unrecognized — every
        pre-TASK-18060 row reads as ``"hunk"`` truthfully, spec §4).
    """
    kind = str(note.get("anchor_kind") or "hunk")
    if kind == "diff_line":
        index = note.get("diff_line_index")
        return f"line {index}" if index is not None else "line"
    if kind == "file":
        return "file"
    return "hunk"


def _note_matches_leaf(note: dict, snapshot_id: "int | None") -> bool:
    """Whether a note belongs to the CURRENTLY focused leaf's snapshot row.

    Mirrors ``console_turn_file_card._note_matches_snapshot`` (Qodo #6,
    PR #1779 fix round): a note carrying a non-null ``snapshot_id`` must
    equal the focused leaf's own ``change_snapshots`` row id to match — a
    run can hold two windows on the same root+path, and this disambiguates
    which window's notes belong under which leaf. A legacy note
    (``snapshot_id`` is ``None``) matches by root+path alone, the same
    fallback the card uses.

    Args:
        note: A note record (DB row dict).
        snapshot_id: The focused leaf's owning ``change_snapshots`` row id
            (``row.get("id")``), or ``None`` when unavailable.

    Returns:
        Whether this note belongs under the focused leaf.
    """
    note_snapshot_id = note.get("snapshot_id")
    if note_snapshot_id is None:
        return True
    return note_snapshot_id == snapshot_id


def _head_label(info: "GitWorkspaceInfo") -> str:
    """One repository's HEAD state as display copy.

    Args:
        info: A detected repository.

    Returns:
        The branch name; ``"detached HEAD"`` when HEAD is not on a branch;
        ``"<branch> (no commits yet)"`` on an unborn branch (spec §4 --
        a fresh ``git init`` is a plausible Console workspace and must
        read as a state, not as an error).
    """
    if info.detached or not info.branch:
        return "detached HEAD"
    if info.unborn:
        return f"{info.branch} (no commits yet)"
    return info.branch


def _land_on_ui(app, callback: Callable, *args) -> None:
    """Hand a worker result to the UI thread, tolerating ONLY teardown.

    ``RuntimeError`` and nothing else, deliberately (the Task 6 re-review's
    binding ruling, mirrored here for the git-action workers): Textual
    signals teardown as ``RuntimeError("App is not running")`` (both raise
    sites in ``App.call_from_thread`` are ``RuntimeError``, and a closed
    loop reports one too), while ``call_from_thread`` ALSO re-raises
    whatever the landing callback itself raised. The commit landings do
    real work (notify, affordance refresh, a full current-mode reload), so
    a bare ``except Exception`` would downgrade a genuine bug in them to
    one debug line whose text would be an outright lie, instead of the loud
    ``WorkerFailed`` traceback Textual gives it.

    Args:
        app: The running app (captured at dispatch, never re-read off the
            worker thread).
        callback: The UI-thread callable to run.
        *args: Its positional arguments.
    """
    try:
        app.call_from_thread(callback, *args)
    except RuntimeError:
        logger.debug(
            "change_review: git-action landing skipped -- the app is no "
            "longer accepting callbacks"
        )


def _commit_warnings(info: "GitWorkspaceInfo") -> list[str]:
    """The commit modal's WARNINGS (spec §5 step 3) -- these never block.

    Args:
        info: The freshly detected repository state the commit will run
            against.

    Returns:
        Zero or more warning lines. Detached HEAD and main/master are
        mutually exclusive by construction (a detached HEAD has no branch),
        so at most one is ever produced today.
    """
    if info.detached or not info.branch:
        return ["⚠ detached HEAD — this commit will not be on any branch"]
    if info.branch in ("main", "master"):
        return [f"⚠ committing directly to {info.branch}"]
    return []


def _push_refusal_for_info(info: "GitWorkspaceInfo | None") -> "str | None":
    """Why ``info``'s repository cannot be pushed, or ``None`` when it can.

    Spec §6's two preconditions, in the engine's own words (see
    :data:`PUSH_DETACHED_REASON` / :data:`PUSH_NO_REMOTE_REASON`) so the
    tooltip a user reads before pressing and the error they would read
    after can never say different things.

    Args:
        info: A detected repository, or ``None`` when detection has no
            entry for the root at all.

    Returns:
        The refusal copy, or ``None`` when a push is possible.
    """
    if info is None:
        return GIT_NO_REPO_REASON
    if info.detached or not info.branch:
        return PUSH_DETACHED_REASON
    # T8 re-review: the engine refuses an option-shaped branch outright (it
    # is a ref-destruction vector -- `--mirror` deletes remote refs). That
    # refusal is the SECURITY guard and must never be removed on the
    # strength of this one; this is the spec §8 half, so the control
    # presents as unavailable-with-a-reason instead of looking live and
    # erroring on press. Both read the same literal, from the same `info`,
    # so they cannot drift or open a TOCTOU window.
    if info.branch.startswith("-"):
        return PUSH_OPTION_BRANCH_REASON
    if not info.remotes:
        return PUSH_NO_REMOTE_REASON
    return None


def _pr_refusal_for_info(info: "GitWorkspaceInfo | None") -> "str | None":
    """Why ``info``'s branch has no PR page yet — the CHEAP half only.

    Deliberately does not judge the HOST: that answer costs a remote-URL
    parse and, for the Gitea family, a git invocation, so it is computed
    once per load off-thread (``provider.pr_url``) and cached instead --
    see ``ChangeReviewScreen._pr_refusal``. This half is pure attribute
    reads and is safe to call on every affordance refresh.

    Args:
        info: A detected repository, or ``None``.

    Returns:
        The refusal copy, or ``None`` when the branch has an upstream.
    """
    if info is None:
        return GIT_NO_REPO_REASON
    if info.upstream is None:
        return PR_NO_UPSTREAM_REASON
    return None


def _combined_refusal(reasons: "Sequence[tuple[str, str]]") -> str:
    """Fold per-root refusals into ONE honest line.

    With a single candidate root there is exactly one truth to tell, so it
    is told verbatim. With several, naming each root's own reason is the
    only honest option -- picking one root's reason and presenting it as
    the whole answer would misdescribe the others.

    Args:
        reasons: ``(root, reason)`` pairs, one per candidate root.

    Returns:
        The refusal copy for the affordance's tooltip and its notify.
    """
    if not reasons:
        return GIT_NO_REPO_REASON
    if len(reasons) == 1:
        return reasons[0][1]
    from pathlib import Path as _P

    detail = "; ".join(f"{_P(root).name}: {reason}" for root, reason in reasons)
    return f"no repository here can be used — {detail}"


def _commit_entries(
    files: "Sequence[ChangedFile]",
) -> list[tuple[str, tuple[str, ...]]]:
    """One checklist row per changed file: ``(label, pathspec)``.

    A RENAME carries BOTH paths in its ONE row's pathspec. Verified against
    real git (T7), because the alternative is worse in a way that is not
    obvious:

    - both paths ⇒ ``git commit -m … -- <new> <old>`` records the WHOLE
      rename (HEAD loses the old path, gains the new one);
    - new path only ⇒ the commit ADDS the new path while leaving the old
      one in HEAD, and a staged deletion behind in the index -- a commit
      that does not match what the checkbox promised, and one the user may
      well push before noticing.

    **Known limitation, deliberately loud (see the T7 report).** It is not
    a rename-only problem -- the trigger is ANY selected path that exists
    in neither the worktree NOR the index, because the engine's
    ``git add -A -- <paths>`` step shares this one pathspec with the
    commit and exits fatal ("pathspec … did not match any files"). Two
    shapes reach it, and BOTH are ordinary terminal gestures rather than
    anything an agent had to do:

    1. a rename already RECORDED IN THE INDEX (``git mv``, porcelain
       ``R``) -- the old path is gone from both;
    2. a STAGED DELETION (``git rm <path>``, or ``rm`` followed by
       ``git add``, porcelain ``D`` in the index column) -- the deleted
       path is likewise in neither.

    In both cases the commit is refused with git's own message and
    NOTHING is staged or committed; the user's staged work is untouched,
    unchecking that one row lets the rest commit, and the change is
    committable from a terminal. Splitting the engine's add/commit
    pathspecs is the fix, and it belongs in ``git_workspace.py``, not
    here. UNSTAGED versions of both are unaffected: git reports them as
    worktree-column changes whose paths are still in the index, which
    stage and commit exactly as expected.

    Args:
        files: The fresh status read's changed files, in porcelain order.

    Returns:
        ``(display label, paths)`` pairs in the same order.
    """
    entries: list[tuple[str, tuple[str, ...]]] = []
    for change in files:
        if change.status == "R" and change.old_path:
            entries.append(
                (
                    f"{change.old_path} → {change.path}",
                    (change.path, change.old_path),
                )
            )
        else:
            entries.append((change.path, (change.path,)))
    return entries


def _root_summary_line(
    info: "GitWorkspaceInfo", *, name_root: bool
) -> str:
    """The `current` mode's per-root header line (spec §4).

    Args:
        info: The root's detected repository state.
        name_root: Whether to prefix the root's directory name -- done
            only when more than one repository is listed, matching the
            leaf labels' own multi-root rule.

    Returns:
        e.g. ``"feat/x ↑2 ↓0 → origin/feat/x"``, or ``"main · no
        upstream"`` when the branch has none (ahead/behind are
        meaningless then, so they are not rendered as a misleading 0/0).
    """
    head = _head_label(info)
    if info.upstream:
        summary = f"{head} ↑{info.ahead} ↓{info.behind} → {info.upstream}"
    else:
        summary = f"{head} · no upstream"
    if name_root:
        return f"{info.root.name}: {summary}"
    return summary


def _hunk_containing_line(
    hunks: "list[DiffHunk]", line_index: int
) -> "tuple[int, DiffHunk | None]":
    """Find the hunk covering ``line_index`` over the file's FULL diff text.

    TASK-18060 Task 7 (review-rail spec §3/§4): a ``diff_line`` note's
    hunk fields are ALSO populated with the hunk the cursor's line falls
    in — ``split_unified_diff`` segments a diff into ``DiffHunk``s but
    does not carry each hunk's absolute line offset, so this reconstructs
    it by walking the hunks in order: hunk *k* occupies its header line
    (one line) plus its body lines, contiguously, right after the shared
    file prelude — ``split_unified_diff``'s own segmentation guarantees no
    gaps between one hunk's last body line and the next hunk's header.

    Args:
        hunks: ``split_unified_diff``'s output for the file's full diff.
        line_index: 0-based index over the SAME full diff text (the
            screen's own ``_cursor_line``, which under the display cap
            equals the full-diff line index — the cap truncates only the
            tail).

    Returns:
        ``(hunk_index, hunk)`` for the hunk covering ``line_index``. A
        line in the shared prelude (before any hunk header) degrades to
        the first hunk rather than reporting "no hunk" — a line comment
        always has SOME hunk context when the diff has hunks at all. A
        line past every hunk's tracked span (should not happen for a real
        cursor line) degrades to the last hunk. ``(-1, None)`` only when
        ``hunks`` itself is empty.
    """
    if not hunks:
        return -1, None
    if len(hunks) == 1 and not hunks[0].header:
        # The fallback shape (binary/rename-no-change, split_unified_
        # diff's own docstring): no real hunk headers at all — every line
        # belongs to the one synthetic hunk.
        return 0, hunks[0]
    prelude = hunks[0].file_prelude
    offset = len(prelude.splitlines()) if prelude else 0
    if line_index < offset:
        return 0, hunks[0]
    for index, hunk in enumerate(hunks):
        span = 1 + len(hunk.body_lines)  # the header line + its body
        if offset <= line_index < offset + span:
            return index, hunk
        offset += span
    return len(hunks) - 1, hunks[-1]


class ChangeReviewDiffPane(VerticalScroll):
    """The diff viewport: hosts the review line cursor, reclaiming keys.

    A BINDINGS-only approach is provably wrong here for the same reason it
    was wrong for ``ConsoleTurnFileCard``'s note input (that class's own
    ``on_key`` docstring traces the root cause against Textual's real
    dispatch, and is the precedent this mirrors): a non-priority binding
    (this pane's own ``ScrollableContainer.BINDINGS`` up/down-scroll, and
    ``ChangeReviewScreen``'s own ``escape -> dismiss_screen`` binding one
    level up) is resolved by ``App._on_key`` ONLY once the raw ``Key``
    MESSAGE has bubbled all the way to the App completely UNSTOPPED. This
    pane sits directly on the bubble path from the focused widget (itself,
    while the pane is focused — ``ChangeReviewScreen.action_focus_diff``
    calls ``.focus()`` on it) up to the App, so reclaiming a key HERE — via
    ``event.stop()`` before it can bubble further — wins the race and
    replaces the DEFAULT behavior for that key while this pane is focused,
    without touching the pane's or the screen's ``BINDINGS`` at all.

    ONLY up/down/``c``/escape are reclaimed (spec §3's named hazard):

    - up/down move the review cursor one rendered line (clamped; see
      ``ChangeReviewScreen._move_diff_cursor``/``_render_diff``) instead of
      scrolling the pane directly — the cursor's own re-render scrolls it
      into view, so the net effect still tracks the keypress.
    - ``c`` opens the inline line-comment ``Input`` (TASK-18060 Task 7,
      review-rail spec §3), anchored to the cursor's current diff line —
      mounted as a SIBLING of this pane (under ``#change-review-right``),
      never as this pane's own descendant, so once it is focused the key
      bubble no longer passes through this pane at all and this handler
      stops seeing keys (``ChangeReviewScreen`` reclaims Enter/Escape for
      it instead — see that screen's own ``on_key``).
    - escape moves focus to the changed-file tree — a DELIBERATE shadow of
      the screen's own ``escape -> dismiss_screen`` binding while the pane
      is focused (spec §3's explicit UX ruling): Esc-Esc is pane -> tree ->
      dismiss, never a single Esc dismissing straight out of the diff pane.

      Page-up/page-down/home/end are intentionally NOT in this list — they
      keep the pane's native ``ScrollableContainer`` scrolling untouched.
    """

    async def on_key(self, event: Key) -> None:
        """Reclaim up/down/``c``/escape while this pane is focused.

        Every branch below both ``stop()``s (block the bubble, so no
        non-priority binding anywhere on the ancestor chain — including
        this pane's OWN native up/down scroll bindings — ever sees the raw
        key) and ``prevent_default()``s (mirrors the card precedent) the
        event, in one whole-handler ``try/except`` so a failure here
        degrades to a swallowed keypress rather than propagating out of an
        ``on_*`` handler (which Textual would otherwise hand to
        ``app._handle_exception()`` and exit the whole app).

        Args:
            event: The bubbling key event, dispatched here because this
                pane is the focused widget (or an ancestor of it).
        """
        try:
            screen = self.screen
            if event.key == "up":
                event.stop()
                event.prevent_default()
                screen._move_diff_cursor(-1)  # noqa: SLF001 -- same module
            elif event.key == "down":
                event.stop()
                event.prevent_default()
                screen._move_diff_cursor(1)  # noqa: SLF001 -- same module
            elif event.key == "c":
                event.stop()
                event.prevent_default()
                await screen._open_comment_input("diff_line")  # noqa: SLF001 -- same module
            elif event.key == "escape":
                event.stop()
                event.prevent_default()
                screen.query_one("#change-review-tree", Tree).focus()
        except Exception:
            logger.opt(exception=True).warning(
                "change_review: diff pane key handling failed"
            )


class ChangeReviewScreen(Screen):
    """Changed-file tree + windowed diff viewer for one conversation."""

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back"),
        Binding("j", "next_file", "Next file"),
        Binding("k", "previous_file", "Previous file"),
        Binding("enter", "focus_diff", "View diff", show=False),
        Binding("u", "revert_file", "Revert file"),
        Binding("U", "undo_all", "Undo all", show=False),
        # TASK-18060 Task 7 (review-rail spec §3): a whole-file comment,
        # reachable from either the tree or the diff pane — it always
        # acts on `self._focused_leaf` (the file whose diff is showing),
        # not on which widget currently holds keyboard focus.
        Binding("C", "comment_file", "Comment file", show=False),
        # TASK-16801 arc B (spec §5): the confirmed, file-picked commit. A
        # plain letter (nothing else on this screen or in the diff pane
        # claims `g`), and it refuses with copy outside `current` mode
        # rather than being conditionally unbound -- see
        # `_refresh_mode_affordances` for why `check_action` is NOT used.
        Binding("g", "git_commit", "Commit…", show=False),
        # TASK-16801 arc B (spec §6): push and the PR compare link. Same
        # posture as `g` -- plain letters nothing else on this screen or in
        # the diff pane claims (the pane reclaims only up/down/`c`/escape),
        # and they refuse with copy outside `current` mode rather than
        # being conditionally unbound.
        Binding("p", "git_push", "Push…", show=False),
        Binding("P", "git_pr", "Open PR", show=False),
    ]

    def __init__(
        self,
        provider: AgentRunsChangeReviewProvider,
        initial_run_id: str | None = None,
        initial_path: str | None = None,
        initial_snapshot_id: int | None = None,
        workspace_roots: "Sequence[str] | None" = None,
    ) -> None:
        """Args are stored; all loading happens in ``on_mount``.

        Args:
            provider: The conversation's turn/diff data source.
            initial_run_id: Turn to open on, or ``None`` for the latest.
                Constructor state rather than a post-push ``select_turn``
                call: the opener's ``call_after_refresh`` fired before this
                screen had composed (NoMatches on the Select) -- the test
                that pinned the opener caught it.
            initial_path: File to focus once the initial turn loads, or
                ``None`` for the turn's first leaf (today's default).
                TASK-18060 Task 3 (review-rail spec §3) -- the rail's
                click-through recipe. Same constructor-state rationale as
                ``initial_run_id``: consumed exactly once by ``_load_turn``'s
                tail, then cleared, so a later turn switch reverts to the
                first leaf. An unmatched path degrades to the first leaf
                rather than an empty pane.
            initial_snapshot_id: When given alongside ``initial_path``,
                prefer the leaf whose owning ``change_snapshots`` row id
                equals this -- disambiguates two windows of the SAME run
                that cover the same path (spec §2's same-run
                turn/subagent-post-turn overlap). ``None`` matches the
                first leaf whose path matches, same as today.
            workspace_roots: The conversation's LIVE workspace roots
                (TASK-16801 arc B, spec §4), unioned with the roots of the
                recorded snapshot rows to form the `current` mode's
                detection candidates. ``None`` (every legacy caller) keeps
                today's behavior exactly -- candidates are then the
                recorded rows' roots alone.
        """
        super().__init__()
        self._provider = provider
        self._initial_run_id = initial_run_id
        self._initial_path = initial_path
        self._initial_snapshot_id = initial_snapshot_id
        self._workspace_roots: tuple[str, ...] = tuple(
            str(root) for root in (workspace_roots or ()) if str(root)
        )
        self._turns: list[ReviewTurn] = []
        self._active_turn: ReviewTurn | None = None
        #: Flattened (row, ChangedFile) leaves in tree order, for j/k.
        self._leaves: list[tuple[dict, ChangedFile]] = []
        self._focused_leaf: int = -1
        #: TASK-18060 Task 6 (review-rail spec §3): the diff pane's line
        #: cursor, an index over the file's RENDERED lines (which, under
        #: the display cap, equal the full-diff line indices — the cap
        #: truncates only the tail; see ``_render_diff``). Reset to 0 every
        #: time the focused file changes (``_focus_leaf``); clamped into
        #: range inside ``_render_diff`` itself against that render's own
        #: line count, never against a stale one.
        self._cursor_line: int = 0
        #: TASK-18060 Task 7 fix round (review rework #1): the focused
        #: leaf's ``diff_line`` note indices, over the FULL diff text —
        #: spec §3's inline "● comment" marker. Recomputed alongside the
        #: notes strip (``_focus_leaf``, and after every save/delete via
        #: ``_refresh_notes_ui_for_focused_leaf``) and consulted READ-ONLY
        #: inside ``_render_diff`` (including on every cursor-move
        #: re-render) so a marker never costs a DB query per keypress.
        self._marked_diff_lines: "set[int]" = set()
        #: TASK-18060 final-review fix round (Fix 2): the memoized diff
        #: text for the CURRENTLY focused leaf — see ``_diff_text_for``.
        #: Keyed by ``(generation, id(row), path)`` so cursor movement
        #: never re-spawns the ``provider.diff_text`` git subprocess pair;
        #: ``_diff_cache_generation`` is bumped by ``_load_turn`` (every
        #: turn (re)load, including a post-revert reload) so a revert of
        #: the SAME turn still forces a fresh read despite the row objects
        #: and path being byte-identical to what was cached before it.
        self._diff_cache_generation: int = 0
        self._diff_cache_key: "tuple[int, int, str] | None" = None
        self._diff_cache_text: "str | None" = None
        self._diff_cache_error: "ChangeTrackingError | None" = None
        # -- `current` mode state (TASK-16801 arc B, spec §4) -------------
        #: The turn Select's options as this screen last set them, blank
        #: entry excluded -- the read seam ``turn_select_options`` serves
        #: this rather than reaching into Textual's private ``_options``.
        self._select_options: list[tuple[str, str]] = []
        #: Detected repositories, keyed by RESOLVED root string (the same
        #: key ``provider.detect_git`` uses, and the same spelling
        #: ``CurrentRootStatus.root`` carries -- so a lookup can never
        #: silently miss). Populated by the detection worker's landing and
        #: refreshed by every current-mode load's own fresh detection.
        self._current_infos: "dict[str, GitWorkspaceInfo]" = {}
        #: Per-root untracked paths from the last current-mode load. What
        #: routes a leaf's diff through ``untracked_preview`` instead of
        #: ``current_diff_text`` (which is a fatal git error on an
        #: untracked path or an unborn HEAD).
        self._current_untracked: "dict[str, frozenset[str]]" = {}
        #: TASK-16801 arc B (spec §6): the last load's ``provider.pr_url``
        #: answer per root -- a URL, or a
        #: ``GitWorkspaceRefusal`` naming why there is none. Computed IN
        #: THE LOAD WORKER, never on the UI thread: answering it costs a
        #: remote-URL parse and, for the Gitea family, a git invocation,
        #: and ``_refresh_mode_affordances`` runs on every busy flip (this
        #: repo has already shipped one "git subprocess per keypress"
        #: regression). Consulted READ-ONLY by the PR button's tooltip;
        #: the press path always re-reads rather than opening this URL,
        #: which can be as stale as the last load.
        self._current_pr_results: "dict[str, object]" = {}
        #: Identity token for the in-flight current-mode load. Captured
        #: before dispatch and re-checked in the landing -- see
        #: ``_land_current_mode``.
        self._current_load_token: "object | None" = None
        #: Whether repo detection has settled (landed, or been skipped
        #: because the kill switch is off / there are no candidate roots).
        #: Read by tests to wait on the mode's availability.
        self._git_detection_settled: bool = False
        #: Banner lines contributed by the active view (turn or current).
        self._turn_banner_lines: list[str] = []
        #: Banner lines for roots that failed BETWEEN detection and status
        #: within one current-mode load (per-root degradation).
        self._current_root_errors: list[str] = []
        #: Banner lines for detection refusals (root inside a repository).
        #: Live truth about the workspace, so they survive turn switches.
        self._git_refusal_banners: list[str] = []
        #: TASK-16801 arc B (spec §5): a git ACTION (the commit preflight or
        #: the commit itself) is in flight. Every git affordance is disabled
        #: while it is True, so nothing can be double-dispatched. Task 8's
        #: push/PR buttons consume the same flag.
        self._git_busy: bool = False
        #: Dispatch identity for the in-flight git action -- the same
        #: token-guard shape as ``_current_load_token``, in its OWN worker
        #: group (``GIT_ACTION_WORKER_GROUP``) so a mutation and a
        #: current-mode status read can never cancel each other.
        self._git_action_token: "object | None" = None

    # -- compose -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Header + banner + (tree | diff pane) + footer.

        Returns:
            The screen's widget tree.
        """
        with Vertical(id="change-review-screen"):
            with Horizontal(id="change-review-header"):
                yield Static(
                    "Change review", id="change-review-title", markup=False
                )
                yield Select(
                    [],
                    id="change-review-turn-select",
                    allow_blank=True,
                    prompt="No turns",
                )
                yield Static("", id="change-review-totals", markup=False)
                # TASK-18060 Task 7 (review-rail spec §3): the "Comment
                # file" affordance named alongside footer key `C` — a
                # small button near the totals.
                yield Button(
                    "Comment file",
                    id="change-review-comment-file-btn",
                    classes="change-review-comment-file-btn",
                    compact=True,
                )
                # TASK-16801 arc B (spec §5): the commit affordance. Hidden
                # until `_refresh_mode_affordances` decides otherwise --
                # the screen OPENS on a recorded turn, where committing is
                # meaningless, so it must never flash into view first.
                commit_button = Button(
                    "Commit…",
                    id="change-review-git-commit-btn",
                    classes="change-review-git-commit-btn",
                    compact=True,
                )
                commit_button.display = False
                yield commit_button
                # TASK-16801 arc B (spec §6): push and PR. Hidden on the
                # same terms as commit -- but unlike commit these stay
                # OFFERED on a clean tree, because unpushed commits are
                # exactly the state they exist for.
                push_button = Button(
                    "Push…",
                    id="change-review-git-push-btn",
                    classes="change-review-git-push-btn",
                    compact=True,
                )
                push_button.display = False
                yield push_button
                pr_button = Button(
                    "Open PR",
                    id="change-review-git-pr-btn",
                    classes="change-review-git-pr-btn",
                    compact=True,
                )
                pr_button.display = False
                yield pr_button
            yield Static(
                "",
                id="change-review-banner",
                classes="change-review-banner",
                markup=False,
            )
            with Horizontal(id="change-review-body"):
                yield Tree("Changes", id="change-review-tree")
                # TASK-18060 Task 7: the inline comment `Input` and notes
                # strip are mounted as SIBLINGS of the diff pane (never as
                # its descendants) — while either is focused, key events
                # bubble past this `Vertical` straight to the screen
                # without ever passing through `ChangeReviewDiffPane`, so
                # the pane's own up/down/`c`/escape reclaim never
                # intercepts them (see `ChangeReviewDiffPane`'s docstring).
                with Vertical(id="change-review-right"):
                    with ChangeReviewDiffPane(id="change-review-diff"):
                        yield Static(
                            "",
                            id="change-review-diff-content",
                            classes="change-review-diff-body",
                            markup=False,
                        )
                    yield Vertical(
                        id="change-review-notes-strip",
                        classes="change-review-notes-strip",
                    )
            yield Static(
                FOOTER_TURN_MODE,
                id="change-review-footer",
                markup=False,
            )

    def on_mount(self) -> None:
        """Defer the initial load until this screen's children exist.

        ``on_mount`` can fire before composed children are queryable when
        the screen is pushed onto a LIVE app (the standalone harness passed
        by timing luck; the opener wiring test caught NoMatches on the
        Select). Same deferral pattern as ``ChatApprovalCard.on_mount``.
        """
        self.call_after_refresh(self._initialize_turns)

    def _initialize_turns(self) -> None:
        """Load turn history and open on the requested (or latest) turn."""
        try:
            select = self.query_one("#change-review-turn-select", Select)
        except Exception:  # noqa: BLE001 -- screen dismissed before refresh
            return
        self._turns = self._provider.turns()
        self._select_options = [
            (turn.label, turn.run_id) for turn in self._turns
        ]
        select.set_options(list(self._select_options))
        if self._turns:
            wanted = self._initial_run_id
            if wanted and not any(t.run_id == wanted for t in self._turns):
                wanted = None
            # Setting the value posts Select.Changed; the handler is the ONE
            # loader (review finding: value-set + direct _load_turn loaded
            # every turn twice -- doubled git work per open).
            select.value = wanted or self._turns[0].run_id
        else:
            self._show_empty("No file changes recorded for this conversation.")
        self._refresh_mode_affordances()
        # TASK-16801 arc B: the `current` mode is OFFERED (never opened on)
        # -- so detection runs AFTER the turn view is already up, keeping
        # this open path byte-compatible, and off-thread because probing a
        # real repository spawns git subprocesses.
        self._dispatch_git_detection()

    # -- `current` mode: detection (TASK-16801 arc B, spec §4) ------------

    def turn_select_options(self) -> list[tuple[str, str]]:
        """The turn selector's options as set by this screen (test seam).

        Returns:
            ``(label, value)`` pairs in display order, blank entry
            excluded. The `current` pseudo-entry (when offered) is first.
        """
        return list(self._select_options)

    @property
    def git_detection_settled(self) -> bool:
        """Whether repo detection has finished (or was skipped).

        Returns:
            True once the detection worker's result has landed, or as soon
            as detection was skipped entirely (kill switch off, or no
            candidate roots).
        """
        return self._git_detection_settled

    def _candidate_git_roots(self) -> list[str]:
        """Roots to probe: the recorded rows' roots ∪ the live workspace roots.

        Returns:
            Distinct root strings in a stable order (row roots first, in
            turn order). ``provider.detect_git`` dedupes further by
            RESOLVED spelling, so two spellings of one directory still
            cost one probe.
        """
        roots: list[str] = []
        seen: set[str] = set()
        for turn in self._turns:
            for row in turn.rows:
                root = str(row.get("root") or "")
                if root and root not in seen:
                    seen.add(root)
                    roots.append(root)
        for root in self._workspace_roots:
            if root and root not in seen:
                seen.add(root)
                roots.append(root)
        return roots

    def _dispatch_git_detection(self) -> None:
        """Probe the candidate roots for real repositories, off-thread.

        Skipped entirely (no worker, no git) when the kill switch is off or
        there are no candidate roots -- spec §8's "off ⇒ zero behavior
        change". Any failure degrades to "no mode offered"; detection is
        an affordance, never a precondition for reviewing turns.
        """
        enabled = getattr(self._provider, "git_actions_enabled", None)
        detect = getattr(self._provider, "detect_git", None)
        try:
            if not callable(enabled) or not callable(detect) or not enabled():
                self._git_detection_settled = True
                return
        except Exception:  # noqa: BLE001 -- a bad config never breaks review
            logger.opt(exception=True).warning(
                "change_review: git kill-switch read failed; mode not offered"
            )
            self._git_detection_settled = True
            return
        candidates = self._candidate_git_roots()
        if not candidates:
            self._git_detection_settled = True
            return
        app = self.app

        def _detect() -> None:
            try:
                detected = detect(candidates)
            except Exception:  # noqa: BLE001 -- never kill the worker
                logger.opt(exception=True).warning(
                    "change_review: git detection failed; mode not offered"
                )
                detected = {}
            app.call_from_thread(self._land_git_detection, detected)

        self.run_worker(
            _detect,
            thread=True,
            exclusive=True,
            group="change-review-git-detect",
        )

    def _land_git_detection(self, detected: dict) -> None:
        """Apply detection: refusal copy, then the pseudo-entry (if any).

        Args:
            detected: ``provider.detect_git``'s result -- one
                ``GitWorkspaceInfo | GitWorkspaceRefusal | None`` per
                resolved root.
        """
        from tldw_chatbook.Workspaces.git_workspace import (
            GitWorkspaceInfo as _Info,
            GitWorkspaceRefusal as _Refusal,
        )

        self._git_detection_settled = True
        infos: "dict[str, GitWorkspaceInfo]" = {}
        refusals: list[str] = []
        for root, result in (detected or {}).items():
            if isinstance(result, _Info):
                infos[str(result.root)] = result
            elif isinstance(result, _Refusal):
                refusals.append(
                    f"git actions unavailable for {root}: {result.reason}"
                )
        self._current_infos = infos
        self._git_refusal_banners = refusals
        if refusals:
            self._update_banner()
        if not infos:
            return
        try:
            select = self.query_one("#change-review-turn-select", Select)
        except Exception:  # noqa: BLE001 -- screen dismissed before landing
            return
        # Prepend, preserving the turn the screen already opened on. A
        # `set_options` call resets the value to blank and posts a
        # `Select.Changed` for the restore too -- `_on_turn_changed`'s
        # already-loaded guard is what keeps that from re-running the
        # turn's git work.
        previous = select.value
        self._select_options = [
            (self._current_mode_label(infos), CURRENT_MODE_SENTINEL),
            *((turn.label, turn.run_id) for turn in self._turns),
        ]
        select.set_options(list(self._select_options))
        if previous != Select.BLANK and any(
            value == previous for _label, value in self._select_options
        ):
            select.value = previous

    @staticmethod
    def _current_mode_label(infos: "dict[str, GitWorkspaceInfo]") -> str:
        """The pseudo-entry's label for the detected repositories.

        Args:
            infos: Detected repositories, keyed by resolved root.

        Returns:
            ``"Working tree (current) — <branch>"`` for one repository
            (``detached HEAD`` when HEAD is detached, ``<branch> (no
            commits yet)`` on an unborn branch); a repository count when
            more than one root was detected, since no single branch names
            them all.
        """
        if len(infos) != 1:
            return f"{CURRENT_MODE_LABEL} — {len(infos)} repositories"
        info = next(iter(infos.values()))
        return f"{CURRENT_MODE_LABEL} — {_head_label(info)}"

    # -- turn loading ------------------------------------------------------

    def select_turn(self, run_id: str) -> None:
        """Switch the view to ``run_id``'s turn (also the Select's handler).

        Args:
            run_id: A turn from the provider's history.
        """
        for turn in self._turns:
            if turn.run_id == run_id:
                select = self.query_one("#change-review-turn-select", Select)
                if select.value == run_id:
                    return  # already loaded; a same-value set posts nothing
                select.value = run_id  # Select.Changed performs the load
                return

    def _close_any_open_comment_input(self) -> None:
        """Unmount any open comment ``Input`` and neuter its pending capture.

        TASK-18060 final-review fix round (Fix 1b): the comment ``Input``
        is mounted as a SIBLING of the diff pane (never its descendant),
        so nothing ``_load_turn`` rebuilds ever removes it on its own —
        switching turns via the Select left an open input mounted,
        still bound (via ``leaf_row``/``leaf_change``) to the leaf it was
        opened against on the PRIOR turn. Left alone, a later Enter on it
        would fire ``_save_comment_input`` with that stale capture while
        ``self._active_turn`` has already moved on. Clearing the captured
        attributes makes any in-flight submit a guaranteed no-op
        (``_save_comment_input``'s own ``row is None`` guard) even before
        ``remove()``'s async detach has actually taken the widget out of
        the DOM.
        """
        try:
            for note_input in list(self.query(".change-review-comment-input")):
                note_input.leaf_row = None
                note_input.leaf_change = None
                note_input.anchor_kind = None
                note_input.cursor_line = None
                note_input.remove()
        except Exception:
            logger.opt(exception=True).warning(
                "change_review: comment input cleanup on turn load failed"
            )

    @on(Select.Changed, "#change-review-turn-select")
    def _on_turn_changed(self, event: Select.Changed) -> None:
        if event.value == CURRENT_MODE_SENTINEL:
            # TASK-16801 arc B: the real working tree gets its own
            # worker-backed load; the snapshot path below is untouched.
            #
            # HAZARD for later work (T7/T8): this branch has NO
            # already-loaded guard, and it cannot get the one below --
            # current mode holds no `_active_turn` to compare against. It
            # is safe today only because nothing rebuilds the Select's
            # options after `_land_git_detection` runs once. If you ever
            # rebuild them while this mode is selected (e.g. to relabel the
            # entry after a commit changes the branch), `set_options`
            # blanks the value and the restore posts a `Changed` naming the
            # SENTINEL -- landing right here and re-dispatching a whole
            # status read. Suppress it at the source (don't rebuild while
            # `_current_mode_active()`), or give this branch its own guard.
            self._load_current_mode()
            return
        if isinstance(event.value, str) and event.value:
            if (
                self._active_turn is not None
                and self._active_turn.run_id == event.value
            ):
                # Already loaded. Reached when `set_options` (the
                # pseudo-entry prepend) resets the value to blank and this
                # screen restores it -- reloading here would re-run the
                # turn's whole git diff for no visible change and would
                # throw away the user's focused leaf.
                return
            for turn in self._turns:
                if turn.run_id == event.value:
                    self._load_turn(turn)
                    return

    def _load_turn(self, turn: ReviewTurn) -> None:
        # TASK-18060 final-review fix round (Fix 1b + Fix 2): both must
        # happen before anything below rebuilds the tree/leaves. Fix 1b: a
        # comment input opened against the PRIOR turn's leaf is a SIBLING
        # of the diff pane, not a descendant of anything rebuilt here --
        # left mounted, a later Enter on it would save a note whose
        # row/change was captured against a turn this screen has already
        # left. Fix 2: bumping the generation invalidates the per-leaf
        # diff-text memo so a revert-reload of THIS SAME turn (identical
        # row objects, identical path) refetches instead of serving
        # pre-revert content.
        self._close_any_open_comment_input()
        self._diff_cache_generation += 1
        self._diff_cache_key = None
        self._diff_cache_text = None
        self._diff_cache_error = None
        self._active_turn = turn
        multi_root = len(turn.rows) > 1
        tree = self.query_one("#change-review-tree", Tree)
        tree.clear()
        tree.root.expand()
        self._leaves = []
        banners: list[str] = []
        grouped: dict[str, list[tuple[dict, ChangedFile]]] = {}
        for row in turn.rows:
            error = str(row.get("tracking_error") or "")
            if error:
                banners.append(f"⚠ tracking failed for {row['root']}: {error}")
                continue
            nested_raw = row.get("nested_repos") or "[]"
            try:
                import json as _json

                nested = [str(p) for p in _json.loads(nested_raw)]
            except (ValueError, TypeError):
                nested = []
            if nested:
                # TASK-1976: name the holes — changes inside these repos
                # are not tracked at all.
                limit = NESTED_BANNER_NAMED_LIMIT
                shown = ", ".join(nested[:limit]) + (
                    f" (+{len(nested) - limit} more)"
                    if len(nested) > limit
                    else ""
                )
                plural = "ies" if len(nested) != 1 else "y"
                banners.append(
                    f"⚠ {len(nested)} nested repositor{plural} inside "
                    f"{row['root']} not tracked: {shown}"
                )
            oversize = int(row.get("untracked_oversize") or 0)
            if oversize:
                # TASK-1975 AC#2: cost bounds are honest, not silent.
                plural = "s" if oversize != 1 else ""
                banners.append(
                    f"⚠ {oversize} oversized file{plural} untracked for "
                    f"{row['root']} (over the size cap)"
                )
            try:
                for change in self._provider.changed_files(row):
                    grouped.setdefault(change.status, []).append((row, change))
            except ChangeTrackingError as exc:
                if self._provider.snapshots_pruned(row):
                    banners.append(
                        f"history for {row['root']} was pruned by retention"
                    )
                else:
                    banners.append(
                        f"⚠ diff unavailable for {row['root']}: {exc}"
                    )

        touched_by_row: dict[int, "set[str] | None"] = {}

        def _badged(row: dict, change: ChangedFile) -> bool:
            rid = id(row)
            if rid not in touched_by_row:
                try:
                    touched_by_row[rid] = self._provider.tool_touched_relpaths(
                        row
                    )
                except Exception:  # noqa: BLE001 -- a badge must never break review
                    logger.opt(exception=True).warning(
                        "change_review: badge derivation failed for "
                        f"{row.get('root')!r}; rendering without badges"
                    )
                    touched_by_row[rid] = None
            touched = touched_by_row[rid]
            if touched is None:
                return False
            # Qodo #1262: no file tool can DELETE or RENAME — those rows
            # badge regardless of path membership (a write_file-created
            # path later script-deleted must not launder the deletion).
            if change.status in ("D", "R"):
                return True
            return change.path not in touched

        self._populate_tree(tree, grouped, multi_root, _badged)

        self._turn_banner_lines = banners
        # A turn view carries no current-mode per-root failures.
        self._current_root_errors = []
        self._update_banner()
        # TASK-16801 arc B: a snapshot turn is showing -- commit goes away,
        # the snapshot-only affordances come back.
        self._refresh_mode_affordances()

        totals = self.query_one("#change-review-totals", Static)
        adds = sum(int(r["adds"] or 0) for r in turn.rows)
        dels = sum(int(r["dels"] or 0) for r in turn.rows)
        totals.update(f"{len(self._leaves)} files  +{adds} −{dels}")

        # TASK-18060 Task 3: the initials are constructor state consumed
        # exactly ONCE -- cleared here, UNCONDITIONALLY, so a later turn
        # switch (this same method, via select_turn/Select.Changed) reverts
        # to the first leaf like today. This must happen before the
        # `self._leaves` branch below: a zero-leaf initial turn (a
        # tracking-error row, or one whose snapshots were pruned) must not
        # leave the initials sitting around to hijack focus on the NEXT
        # `_load_turn` call once a later turn happens to contain a
        # same-named path (reviewer catch on the first cut of this task).
        initial_path = self._initial_path
        initial_snapshot_id = self._initial_snapshot_id
        self._initial_path = None
        self._initial_snapshot_id = None

        if self._leaves:
            # A path that doesn't exist in this turn (stale rail cache, a
            # revert since) degrades to the first leaf rather than an empty
            # pane -- select_file itself stays a no-op on an unmatched path
            # so its legacy behavior (external callers with no fallback
            # expectation) is untouched.
            if initial_path is not None and any(
                change.path == initial_path for _row, change in self._leaves
            ):
                self.select_file(initial_path, snapshot_id=initial_snapshot_id)
            else:
                self._focus_leaf(0)
        else:
            self._show_empty("No file changes in this turn.")
            # TASK-18060 Task 7: `_focus_leaf` is the strip's usual refresh
            # choke point, but a zero-leaf turn never reaches it — clear
            # (hide) any stale strip content from a previously focused
            # turn explicitly.
            self._refresh_notes_strip()

    def _populate_tree(
        self,
        tree: Tree,
        grouped: "dict[str, list[tuple[dict, ChangedFile]]]",
        multi_root: bool,
        badge: "Callable[[dict, ChangedFile], bool]",
    ) -> None:
        """Fill the changed-file tree and ``self._leaves`` from ``grouped``.

        The ONE grouping/labeling path, shared by the snapshot turn view
        and the `current` working-tree view (TASK-16801 arc B) so the two
        can never drift on group order, the "Other" bucket, or the leaf
        index each node carries.

        Args:
            tree: The (already cleared) changed-file tree.
            grouped: Entries bucketed by ``ChangedFile.status``.
            multi_root: Whether leaf labels should name their root.
            badge: Per-leaf "changed outside direct file tools" predicate
                (TASK-1978). Always False in `current` mode -- provenance
                is a property of a recorded run, not of the working tree.
        """
        known = {code for code, _label in _GROUPS}
        for code, label in _GROUPS:
            entries = grouped.get(code, [])
            if not entries:
                continue
            branch = tree.root.add(f"{label} ({len(entries)})", expand=True)
            for row, change in entries:
                # TASK-2032: the node carries its leaf index so a MOUSE
                # selection can load the diff (j/k was the only loader).
                branch.add_leaf(
                    self._leaf_label(
                        row, change, multi_root, badge=badge(row, change)
                    ),
                    data=len(self._leaves),
                )
                self._leaves.append((row, change))
        other = [
            entry
            for code, entries in grouped.items()
            if code not in known
            for entry in entries
        ]
        if other:
            branch = tree.root.add(f"{_OTHER_GROUP} ({len(other)})", expand=True)
            for row, change in other:
                # TASK-2032: the node carries its leaf index so a MOUSE
                # selection can load the diff (j/k was the only loader).
                branch.add_leaf(
                    self._leaf_label(
                        row, change, multi_root, badge=badge(row, change)
                    ),
                    data=len(self._leaves),
                )
                self._leaves.append((row, change))

    def _update_banner(self) -> None:
        """Render the honesty banner from every live source, deduped.

        Three sources, in display order: the active view's own lines
        (tracking errors, nested repos, oversize files, or `current`
        mode's per-root headers), that load's per-root git failures, and
        the standing detection refusals (a workspace inside a repository
        -- live truth about the workspace, so it survives turn switches
        and is the "why unavailable" copy spec §8 requires).
        """
        try:
            banner = self.query_one("#change-review-banner", Static)
        except Exception:  # noqa: BLE001 -- screen dismissed before refresh
            return
        lines: list[str] = []
        seen: set[str] = set()
        for line in (
            *self._turn_banner_lines,
            *self._current_root_errors,
            *self._git_refusal_banners,
        ):
            if line and line not in seen:
                seen.add(line)
                lines.append(line)
        banner.update("\n".join(lines))
        banner.display = bool(lines)

    # -- `current` mode: load and land (TASK-16801 arc B, spec §4) --------

    def _current_mode_active(self) -> bool:
        """Whether the screen is showing the REAL working tree.

        The single predicate spec §4.1's row-consumers table gates on --
        the turn ``Select``'s value IS the mode, so there is no second
        flag that can disagree with what the user sees selected.

        Returns:
            True while the pseudo-entry is the selected option.
        """
        try:
            select = self.query_one("#change-review-turn-select", Select)
        except Exception:  # noqa: BLE001 -- screen dismissed / not composed
            return False
        return select.value == CURRENT_MODE_SENTINEL

    def _commit_target_root(self) -> "str | None":
        """The root a commit would act on (spec §6: the focused leaf's).

        Returns:
            The focused leaf's RESOLVED root string, or ``None`` when there
            is no `current`-mode leaf to take one from.

        Note:
            ``None`` here means the tree is EMPTY (``_focus_leaf`` always
            focuses leaf 0 when any leaf exists), i.e. every root is clean
            or unreadable -- which is exactly when commit is disabled. Spec
            §6's ">1 detected root and no focused leaf ⇒ root ``Select``"
            fallback is therefore unreachable for COMMIT, and no root
            selector is built here. It stays relevant for Task 8's
            push/PR, which ARE offered on a clean tree (unpushed commits
            are the whole point right after committing).
        """
        if not self._leaves or self._focused_leaf < 0:
            return None
        row, _change = self._leaves[self._focused_leaf]
        if row.get("kind") != "git_current":
            return None
        return str(row.get("root") or "") or None

    def _git_target_roots(self) -> list[str]:
        """The roots a PUSH or a PR may act on (spec §6's targeting rule).

        Returns:
            ``[focused leaf's root]`` when a `current`-mode leaf is
            focused; otherwise EVERY detected root, in detection order.
            The second case is what makes spec §6's ">1 detected root and
            no focused leaf" branch reachable at all -- unlike commit
            (``_commit_target_root``), push and PR are offered on a CLEAN
            tree, which is precisely the state with no leaf to target
            from. The ambiguity is resolved by the modal's root ``Select``,
            never guessed at here.
        """
        focused = self._commit_target_root()
        if focused is not None:
            return [focused]
        return list(self._current_infos)

    def _push_refusal(self) -> "str | None":
        """Why the `Push…` affordance is unavailable, or ``None``.

        Returns:
            The refusal copy. ``None`` as soon as ANY candidate root can be
            pushed -- with several roots the modal's ``Select`` offers only
            the ones that can, so one unusable sibling must not disable the
            control (spec §6, AC #2: a reason, never a dead control, but
            also never a needless one).
        """
        roots = self._git_target_roots()
        reasons: list[tuple[str, str]] = []
        for root in roots:
            reason = _push_refusal_for_info(self._current_infos.get(root))
            if reason is None:
                return None
            reasons.append((root, reason))
        return _combined_refusal(reasons)

    def _pr_refusal(self) -> "str | None":
        """Why the `Open PR` affordance is unavailable, or ``None``.

        Two layers, cheap first: no upstream is answerable from detection
        alone (:func:`_pr_refusal_for_info`), while "this host has no PR
        page we can link" is the CACHED answer from the last load's
        off-thread ``provider.pr_url`` call. Before a load has landed the
        cache is empty and only the cheap layer applies -- so the button
        can be live for a moment against an unsupported host, and pressing
        it then reports the host refusal honestly rather than opening
        anything (``_land_pr_url``).

        Returns:
            The refusal copy, or ``None`` when a PR link is available.
        """
        from tldw_chatbook.Workspaces.git_workspace import (
            GitWorkspaceRefusal as _Refusal,
        )

        roots = self._git_target_roots()
        reasons: list[tuple[str, str]] = []
        for root in roots:
            reason = _pr_refusal_for_info(self._current_infos.get(root))
            if reason is None:
                cached = self._current_pr_results.get(root)
                if isinstance(cached, _Refusal):
                    reason = cached.reason
            if reason is None:
                return None
            reasons.append((root, reason))
        return _combined_refusal(reasons)

    def _refresh_mode_affordances(self) -> None:
        """Make every mode-scoped control LOOK like what it will actually do.

        Spec §5 (commit is offered only in `current` mode, disabled with a
        reason when there is nothing to commit or a git action is already
        running) and §8 (a disabled action carries its reason as copy,
        never a dead control). The Task 6 re-review's finding (a) adds the
        other half: in `current` mode the SNAPSHOT-only affordances must
        present as unavailable instead of looking live and failing on
        press.

        Textual's ``check_action`` is the canonical way to dim a binding,
        and it is deliberately NOT used here: it makes the key a SILENT
        no-op, which is precisely the failure mode Task 6 fixed for
        ``action_undo_all`` (a gate below the `_active_turn is None` early
        return left `U` silently dead). The refusals stay live and audible;
        what changes is the PRESENTATION -- the disabled button plus a
        mode-aware footer legend that stops advertising the keys it cannot
        honor and says why.
        """
        try:
            commit_btn = self.query_one("#change-review-git-commit-btn", Button)
            push_btn = self.query_one("#change-review-git-push-btn", Button)
            pr_btn = self.query_one("#change-review-git-pr-btn", Button)
            comment_btn = self.query_one(
                "#change-review-comment-file-btn", Button
            )
            footer = self.query_one("#change-review-footer", Static)
        except Exception:  # noqa: BLE001 -- screen dismissed / not composed
            return
        current = self._current_mode_active()
        comment_btn.disabled = current
        comment_btn.tooltip = CURRENT_MODE_COMMENT_REFUSAL if current else None
        footer.update(FOOTER_CURRENT_MODE if current else FOOTER_TURN_MODE)
        commit_btn.display = current
        push_btn.display = current
        pr_btn.display = current
        if not current:
            for button in (commit_btn, push_btn, pr_btn):
                button.disabled = False
                button.tooltip = None
            return
        if self._git_busy:
            commit_btn.disabled = True
            commit_btn.tooltip = GIT_BUSY_REASON
        elif self._commit_target_root() is None:
            commit_btn.disabled = True
            commit_btn.tooltip = COMMIT_CLEAN_TREE_REASON
        else:
            commit_btn.disabled = False
            commit_btn.tooltip = None
        # Spec §6: push and PR are NOT gated on the tree having changes --
        # a clean tree with unpushed commits is their whole reason to exist.
        for button, refusal in (
            (push_btn, self._push_refusal),
            (pr_btn, self._pr_refusal),
        ):
            if self._git_busy:
                button.disabled = True
                button.tooltip = GIT_BUSY_REASON
                continue
            reason = refusal()
            button.disabled = reason is not None
            button.tooltip = reason

    def _set_git_busy(self, busy: bool) -> None:
        """Flip the git-action busy flag and re-render what it gates.

        Args:
            busy: Whether a git action (preflight or commit) is in flight.
        """
        self._git_busy = busy
        self._refresh_mode_affordances()

    def _load_current_mode(self) -> None:
        """Read the real working tree off-thread and land it.

        Deliberately NOT ``_load_turn``'s synchronous posture (spec §4):
        a status scan of a large cold repository would stall the UI
        thread. The snapshot path keeps its existing synchronous posture
        unchanged; only this mode is worker-backed.
        """
        self._close_any_open_comment_input()
        # A reload must refetch: the working tree moves under the view,
        # and the pseudo rows/paths are byte-identical between loads.
        self._diff_cache_generation += 1
        self._diff_cache_key = None
        self._diff_cache_text = None
        self._diff_cache_error = None
        # No recorded turn is in view: this is what makes every
        # notes/comment path (which anchors to `change_snapshots` rows)
        # inert here, on top of the explicit gates.
        self._active_turn = None
        self._leaves = []
        self._focused_leaf = -1
        self._marked_diff_lines = set()
        self._current_untracked = {}
        self._current_pr_results = {}
        self._current_root_errors = []
        self._turn_banner_lines = []
        self._update_banner()
        try:
            tree = self.query_one("#change-review-tree", Tree)
            tree.clear()
            tree.root.expand()
            self.query_one("#change-review-totals", Static).update("")
        except Exception:  # noqa: BLE001 -- screen dismissed before load
            return
        self._refresh_notes_strip()
        self._show_empty("Loading working tree…")
        # Nothing is listed yet, so commit has no target: this renders it
        # disabled-with-a-reason for the duration of the read.
        self._refresh_mode_affordances()

        token = self._current_load_token = object()
        roots = list(self._current_infos)
        provider = self._provider
        app = self.app

        def _read_working_trees() -> None:
            from tldw_chatbook.Workspaces.git_workspace import GitWorkspaceError

            def _land(callback, *args) -> None:
                """Hand a result back to the UI thread, tolerating teardown.

                ``call_from_thread`` raises once the app is shutting down
                -- which a status read can easily outlive, since it is a
                git subprocess. Unhandled, that surfaces as a logged
                ``WorkerFailed``; worse, raising out of a PER-ROOT landing
                mid-loop would abort the roots after it and quietly break
                the per-root isolation this loop exists to guarantee.

                ``RuntimeError`` ONLY, deliberately (re-review round 2):
                Textual signals teardown as
                ``RuntimeError("App is not running")`` (``app.py``; a
                closed loop reports ``RuntimeError`` too), while
                ``call_from_thread`` ALSO re-raises whatever the landing
                callback itself raised -- and the landings do real work
                (tree queries, ``_populate_tree``, banner math). A bare
                ``except Exception`` here would downgrade a genuine bug in
                them to one debug line whose text ("app is no longer
                accepting callbacks") would be an outright lie, instead of
                the loud ``WorkerFailed`` traceback Textual gives it.
                ``CancelledError`` is a ``BaseException`` and was never
                caught here in either form.
                """
                try:
                    app.call_from_thread(callback, *args)
                except RuntimeError:
                    logger.debug(
                        "change_review: current-mode landing skipped -- "
                        "the app is no longer accepting callbacks"
                    )

            statuses: list["CurrentRootStatus"] = []
            #: TASK-16801 T8 (spec §6): the PR link's availability per root,
            #: answered HERE (off-thread) so the affordance refresh never
            #: has to spawn git to decide whether to enable a button.
            pr_results: "dict[str, object]" = {}
            for root in roots:
                # ONE try/except PER ROOT, never one around the batch: a
                # root that vanished (or moved inside another repository)
                # between detection and this read must degrade alone --
                # aborting here would blank the other roots' changes too,
                # and letting it propagate would kill the worker.
                try:
                    statuses.append(provider.current_status(root))
                except GitWorkspaceError as exc:
                    _land(self._land_current_root_failure, token, root, str(exc))
                    continue
                except Exception as exc:  # noqa: BLE001 -- worker must live
                    logger.opt(exception=True).warning(
                        f"change_review: working-tree status failed for {root!r}"
                    )
                    _land(self._land_current_root_failure, token, root, str(exc))
                    continue
                # Its OWN try/except, never folded into the status read's:
                # `pr_url` is contractually non-raising, so a raise here is
                # a BUG in it -- and a bug in the PR link must not cost the
                # user the working-tree read they actually asked for.
                try:
                    pr_results[str(statuses[-1].root)] = provider.pr_url(
                        str(statuses[-1].root), statuses[-1].info
                    )
                except Exception:  # noqa: BLE001 -- the read must survive
                    logger.opt(exception=True).warning(
                        f"change_review: PR link probe failed for {root!r}"
                    )
            # Its OWN landing, deliberately, rather than a third argument to
            # `_land_current_mode`: that landing's two-argument shape is
            # pinned by T6's swallow/don't-swallow tests, and this is the
            # same per-concern landing shape `_land_current_root_failure`
            # already uses. `call_from_thread` preserves order, so the cache
            # is in place before the render below refreshes the affordances.
            _land(self._land_current_pr_links, token, pr_results)
            _land(self._land_current_mode, token, statuses)

        self.run_worker(
            _read_working_trees,
            thread=True,
            exclusive=True,
            group="change-review-current",
        )

    def _current_load_is_live(self, token: object) -> bool:
        """Whether a landing from ``token``'s dispatch may still apply.

        Two independent ways a landing goes stale, both real: a NEWER
        current-mode load was dispatched (token superseded), or the user
        moved the ``Select`` back to a recorded turn while this read was
        in flight. Textual's exclusive-worker group cancels the prior
        worker TASK, but a ``call_from_thread`` callback it had already
        queued still runs -- without this check those working-tree rows
        land inside a turn view (the ``chat_screen``
        ``_land_console_changed_files`` precedent).

        Args:
            token: The identity captured at dispatch time.

        Returns:
            True when this load is still the live one.
        """
        if token is not self._current_load_token:
            logger.debug(
                "change_review: dropping a superseded current-mode landing"
            )
            return False
        if not self._current_mode_active():
            logger.debug(
                "change_review: dropping a current-mode landing -- the "
                "selector moved back to a recorded turn"
            )
            return False
        return True

    def _land_current_root_failure(
        self, token: object, root: str, message: str
    ) -> None:
        """Degrade ONE root to an honest banner line and keep going.

        Args:
            token: The dispatch identity of the load that failed.
            root: The root whose status read failed.
            message: The engine's error text (already excerpt-capped).
        """
        if not self._current_load_is_live(token):
            return
        from pathlib import Path as _P

        line = f"⚠ working tree unavailable for {_P(root).name}: {message}"
        if line not in self._current_root_errors:
            self._current_root_errors.append(line)
        self._update_banner()

    def _land_current_pr_links(
        self, token: object, pr_results: "dict[str, object]"
    ) -> None:
        """Adopt the load's per-root PR-link answers (TASK-16801 T8).

        Args:
            token: The dispatch identity of the load that produced them.
            pr_results: Root → URL or
                :class:`~tldw_chatbook.Workspaces.git_workspace.GitWorkspaceRefusal`,
                as computed off-thread by the load worker.
        """
        if not self._current_load_is_live(token):
            return
        self._current_pr_results = dict(pr_results or {})

    def _land_current_mode(
        self, token: object, statuses: "Sequence[CurrentRootStatus]"
    ) -> None:
        """Render one current-mode read: pseudo rows, tree, header, totals.

        Args:
            token: The dispatch identity captured by ``_load_current_mode``.
            statuses: One :class:`CurrentRootStatus` per root that could be
                read (a root that failed is already on the banner).
        """
        if not self._current_load_is_live(token):
            return
        try:
            tree = self.query_one("#change-review-tree", Tree)
            totals = self.query_one("#change-review-totals", Static)
        except Exception:  # noqa: BLE001 -- screen dismissed before landing
            return
        tree.clear()
        tree.root.expand()
        self._leaves = []
        self._focused_leaf = -1
        self._marked_diff_lines = set()
        self._current_untracked = {}
        multi_root = len(statuses) > 1
        grouped: "dict[str, list[tuple[dict, ChangedFile]]]" = {}
        headers: list[str] = []
        adds = 0
        dels = 0
        for status in statuses:
            root_key = str(status.root)
            self._current_infos[root_key] = status.info
            self._current_untracked[root_key] = status.untracked
            headers.append(_root_summary_line(status.info, name_root=multi_root))
            # ONE pseudo row per root (spec §4's pinned shape) -- its
            # IDENTITY is half the diff memo's key, so it must be created
            # once here, never per leaf.
            row = {"root": root_key, "kind": "git_current", "id": -1}
            for change in status.files:
                grouped.setdefault(change.status, []).append((row, change))
                adds += int(change.adds or 0)
                dels += int(change.dels or 0)
        self._populate_tree(tree, grouped, multi_root, lambda _row, _c: False)
        self._turn_banner_lines = headers
        self._update_banner()
        totals.update(f"{len(self._leaves)} files  +{adds} −{dels}")
        if self._leaves:
            self._focus_leaf(0)
        else:
            # A clean tree still ENTERS the mode (spec §4): commit is
            # meaningless but unpushed commits are exactly the case for
            # push/PR right after committing.
            #
            # But "working tree clean" is a positive claim ABOUT THE USER'S
            # REPOSITORY, and `statuses` is ALSO empty when every root's
            # read failed -- printing it there would have the pane asserting
            # something false directly beneath a banner saying the tree
            # could not be read (spec §8: the two surfaces must agree).
            self._show_empty(
                "working tree clean"
                if statuses
                else "working tree unavailable — see above"
            )
            self._refresh_notes_strip()
        # After the leaves exist (or provably don't): commit's target root
        # comes from the focused leaf, so this must run AFTER `_focus_leaf`.
        self._refresh_mode_affordances()

    # -- commit (TASK-16801 arc B, spec §5) --------------------------------

    @on(Button.Pressed, "#change-review-git-commit-btn")
    def _on_commit_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_git_commit()

    def action_git_commit(self) -> None:
        """`g` / `Commit…`: confirm and commit the real working tree.

        Spec §5's order is load-bearing and preserved literally: the
        run-active refusal comes FIRST (a notify, never a modal), and the
        modal is only reached through a FRESH status read -- the rendered
        view can be arbitrarily stale, and a commit must list what git
        will actually see.
        """
        if not self._current_mode_active():
            self.notify(COMMIT_TURN_MODE_REFUSAL, severity="warning")
            return
        if self._git_busy:
            self.notify(COMMIT_BUSY_REASON, severity="warning")
            return
        root = self._commit_target_root()
        if root is None:
            self.notify(COMMIT_CLEAN_TREE_REASON, severity="warning")
            return
        try:
            refused = bool(self._provider.run_active())
        except Exception:  # noqa: BLE001 -- a broken probe must not block work
            logger.opt(exception=True).warning(
                "change_review: run_active probe failed; deferring to the "
                "engine's own refusal"
            )
            refused = False
        if refused:
            # Spec §5 step 1. The engine re-checks immediately before it
            # runs anything (an injected probe), so this early exit is the
            # UX half of the guard, never the only one.
            self.notify(COMMIT_RUN_ACTIVE_REFUSAL, severity="warning")
            return
        self._dispatch_commit_preflight(root)

    def _dispatch_commit_preflight(self, root: str) -> None:
        """Read ``root``'s working tree FRESH, then open the commit modal.

        Args:
            root: The resolved root the commit will act on.
        """
        token = self._git_action_token = object()
        self._set_git_busy(True)
        provider = self._provider
        app = self.app

        def _preflight() -> None:
            from tldw_chatbook.Workspaces.git_workspace import GitWorkspaceError

            # ONE root, so ONE try/except -- never a batch guard (the T5
            # review's binding instruction): `current_status` re-detects and
            # raises `GitWorkspaceError` when the root is no longer a usable
            # repository, and `_current_infos` is never pruned, so a cached
            # info can name a repository that has since moved or vanished.
            # Git's own error is what the user sees, not our stale belief.
            try:
                status = provider.current_status(root)
            except GitWorkspaceError as exc:
                _land_on_ui(
                    app, self._land_commit_preflight_failure, token, root, str(exc)
                )
                return
            except Exception as exc:  # noqa: BLE001 -- the worker must live
                logger.opt(exception=True).warning(
                    f"change_review: commit preflight failed for {root!r}"
                )
                _land_on_ui(
                    app, self._land_commit_preflight_failure, token, root, str(exc)
                )
                return
            _land_on_ui(app, self._land_commit_preflight, token, status)

        self.run_worker(
            _preflight,
            thread=True,
            exclusive=True,
            group=GIT_ACTION_WORKER_GROUP,
        )

    def _git_action_is_live(self, token: object) -> bool:
        """Whether a landing from ``token``'s git-action dispatch still applies.

        Deliberately narrower than :meth:`_current_load_is_live`: it checks
        the dispatch token ONLY. A commit that already ran has MUTATED the
        user's repository, so its outcome must be reported even if the user
        moved the selector back to a recorded turn while it was in flight
        -- silence there would be the dishonest option. The mode is
        re-checked separately by the callers that need it (opening a modal
        over a turn view, and reloading the current-mode view).

        Args:
            token: The identity captured at dispatch time.

        Returns:
            True when this is still the live git action.
        """
        if token is not self._git_action_token:
            logger.debug("change_review: dropping a superseded git-action landing")
            return False
        return True

    def _land_commit_preflight_failure(
        self, token: object, root: str, message: str
    ) -> None:
        """Report a fresh-read failure honestly instead of opening a modal.

        Args:
            token: The dispatch identity of the failed preflight.
            root: The root whose status read failed.
            message: The engine's (already excerpt-capped) error text.
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        from pathlib import Path as _P

        self.notify(
            f"Could not read the working tree at {_P(root).name}: {message}",
            severity="error",
        )

    def _land_commit_preflight(
        self, token: object, status: "CurrentRootStatus"
    ) -> None:
        """Open the commit modal over the FRESH read (spec §5 steps 2-3).

        Args:
            token: The dispatch identity of this preflight.
            status: The fresh :class:`CurrentRootStatus` for the target root.
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        if not self._current_mode_active():
            # The user left the mode while the fresh read ran; a commit
            # modal over a recorded-turn view would be a lie about context.
            return
        entries = _commit_entries(status.files)
        if not entries:
            # The tree went clean between the button and the read.
            self.notify(COMMIT_CLEAN_TREE_REASON, severity="warning")
            return
        root = str(status.root)

        def _apply(result: "dict | None") -> None:
            if not result:
                return
            self._dispatch_commit(result)

        self.app.push_screen(
            ChangeGitCommitModal(root=root, info=status.info, entries=entries),
            callback=_apply,
        )

    def _dispatch_commit(self, request: dict) -> None:
        """Run the confirmed commit off-thread.

        Args:
            request: The modal's result --
                ``{"message", "new_branch", "files", "root", "file_count"}``.
                ``files`` is the flat PATHSPEC (a rename row contributes
                two paths); ``file_count`` is what the user actually
                checked, and is what the outcome copy must count.
        """
        root = str(request["root"])
        files = list(request["files"])
        message = str(request["message"])
        new_branch = request["new_branch"]
        file_count = int(request.get("file_count") or len(files))
        token = self._git_action_token = object()
        self._set_git_busy(True)
        provider = self._provider
        app = self.app

        def _commit() -> None:
            from tldw_chatbook.Workspaces.git_workspace import (
                CommitRefusedError,
                GitWorkspaceError,
            )

            # ASYMMETRIC by contract (the T5 seam comment): `commit_selected`
            # RAISES for a refusal and for the two preconditions, but a
            # failing git STEP comes back as a returned `CommitResult`. Both
            # shapes are handled; neither is wrapped as the other.
            try:
                result = provider.commit_selected(root, files, message, new_branch)
            except CommitRefusedError as exc:
                _land_on_ui(app, self._land_commit_refused, token, str(exc))
                return
            except GitWorkspaceError as exc:
                _land_on_ui(app, self._land_commit_refused, token, str(exc))
                return
            except Exception as exc:  # noqa: BLE001 -- the worker must live
                logger.opt(exception=True).warning(
                    f"change_review: commit failed for {root!r}"
                )
                _land_on_ui(app, self._land_commit_refused, token, str(exc))
                return
            _land_on_ui(app, self._land_commit_result, token, result, file_count)

        self.run_worker(
            _commit,
            thread=True,
            exclusive=True,
            group=GIT_ACTION_WORKER_GROUP,
        )

    def _land_commit_refused(self, token: object, message: str) -> None:
        """A commit that never ran: report the reason, change nothing.

        Args:
            token: The dispatch identity of the refused commit.
            message: The refusal text (the engine's own sentence).
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        self.notify(message, severity="warning")

    def _land_commit_result(
        self, token: object, result: "CommitResult", file_count: int
    ) -> None:
        """Report the commit's outcome, then reload the working tree.

        Failure copy names the blocking STEP plus git's own excerpt (spec
        §5 step 5) -- never a rolled-up "git failed". ``outcomes[-1]`` IS
        the blocking step by the engine's contract: a guard that passes
        appends nothing, and every action step appends win or lose, so the
        last row is always the one that stopped the run.

        Args:
            token: The dispatch identity of this commit.
            result: The engine's :class:`CommitResult`.
            file_count: How many paths were sent to the engine.
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        last = result.outcomes[-1] if result.outcomes else None
        if last is not None and not last.ok:
            detail = last.detail or "git reported no detail"
            self.notify(
                f"Commit failed at {last.step}: {detail}", severity="error"
            )
        elif result.short_sha:
            self.notify(f"Committed {file_count} file(s) as {result.short_sha}")
        else:
            # The commit landed but its sha could not be resolved -- say
            # both halves rather than claiming success or failure alone.
            self.notify(
                f"Committed {file_count} file(s) — could not resolve the new "
                "commit's sha",
                severity="warning",
            )
        # Reload on EVERY outcome, not just success: a failure can still
        # have moved the repository (a created branch, a staged index), and
        # the view must show disk truth either way. `_load_current_mode`
        # bumps the diff-cache generation, so no pre-commit diff survives.
        if self._current_mode_active():
            self._load_current_mode()

    # -- push and PR (TASK-16801 arc B, spec §6) ---------------------------

    @on(Button.Pressed, "#change-review-git-push-btn")
    def _on_push_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_git_push()

    @on(Button.Pressed, "#change-review-git-pr-btn")
    def _on_pr_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_git_pr()

    def action_git_push(self) -> None:
        """`p` / `Push…`: confirm and push the current branch.

        **Deliberately NOT gated on ``provider.run_active()``**, and the
        contrast with :meth:`action_git_commit` is the point (spec §6 states
        it explicitly): a push ships state that is ALREADY committed and
        leaves the working tree untouched, so an agent run still editing
        that tree is not endangered by it -- whereas a commit, which writes
        the tree the run is mid-edit on, is refused.
        """
        if not self._current_mode_active():
            self.notify(PUSH_TURN_MODE_REFUSAL, severity="warning")
            return
        if self._git_busy:
            self.notify(GIT_BUSY_REASON, severity="warning")
            return
        refusal = self._push_refusal()
        if refusal is not None:
            self.notify(refusal, severity="warning")
            return
        self._dispatch_git_target_preflight("push")

    def action_git_pr(self) -> None:
        """`P` / `Open PR`: open the branch's compare/merge-request page."""
        if not self._current_mode_active():
            self.notify(PR_TURN_MODE_REFUSAL, severity="warning")
            return
        if self._git_busy:
            self.notify(GIT_BUSY_REASON, severity="warning")
            return
        refusal = self._pr_refusal()
        if refusal is not None:
            self.notify(refusal, severity="warning")
            return
        self._dispatch_git_target_preflight("pr")

    def _dispatch_git_target_preflight(self, action: str) -> None:
        """Re-detect the candidate roots FRESH, then pick a target.

        ``_current_infos`` is never pruned, so a cached entry can name a
        repository that has since moved, lost its remote, or had its
        upstream deleted -- and a push modal built on that would describe a
        push that cannot happen. Detection is the right primitive here (not
        ``current_status``): push and PR need branch/upstream/remotes, never
        the file list, so this stays a cheap probe.

        Args:
            action: ``"push"`` or ``"pr"`` -- which landing to take.
        """
        roots = self._git_target_roots()
        token = self._git_action_token = object()
        self._set_git_busy(True)
        provider = self._provider
        app = self.app

        def _preflight() -> None:
            try:
                detected = provider.detect_git(roots)
            except Exception as exc:  # noqa: BLE001 -- the worker must live
                logger.opt(exception=True).warning(
                    f"change_review: {action} preflight failed for {roots!r}"
                )
                _land_on_ui(app, self._land_git_target_failure, token, str(exc))
                return
            _land_on_ui(
                app,
                self._land_git_target_preflight,
                token,
                action,
                detected,
                tuple(roots),
            )

        self.run_worker(
            _preflight,
            thread=True,
            exclusive=True,
            group=GIT_ACTION_WORKER_GROUP,
        )

    def _land_git_target_failure(self, token: object, message: str) -> None:
        """A preflight that never produced a target: say so, do nothing.

        Args:
            token: The dispatch identity of the failed preflight.
            message: The failure text.
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        self.notify(f"Could not read the repository: {message}", severity="error")

    def _land_git_target_preflight(
        self,
        token: object,
        action: str,
        detected: dict,
        roots: "Sequence[str]",
    ) -> None:
        """Choose the target root (asking when ambiguous) and proceed.

        Spec §6's multi-root rule: with more than one usable root and no
        focused leaf, the action's modal carries a root ``Select`` -- and
        the modal ALWAYS names the root it will act on, whether it asked or
        not.

        Args:
            token: The dispatch identity of this preflight.
            action: ``"push"`` or ``"pr"``.
            detected: ``provider.detect_git``'s fresh result.
            roots: The candidate roots, in the order they were probed.
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        if not self._current_mode_active():
            # The user left the mode while the fresh read ran; acting on a
            # repository the view no longer shows would be a context lie.
            return
        from tldw_chatbook.Workspaces.git_workspace import (
            GitWorkspaceInfo as _Info,
            GitWorkspaceRefusal as _Refusal,
        )

        judge = _push_refusal_for_info if action == "push" else _pr_refusal_for_info
        entries: "list[tuple[str, GitWorkspaceInfo]]" = []
        refused: list[tuple[str, str]] = []
        for root in roots:
            result = (detected or {}).get(root)
            if isinstance(result, _Info):
                reason = judge(result)
                if reason is None:
                    entries.append((root, result))
                else:
                    refused.append((root, reason))
            elif isinstance(result, _Refusal):
                refused.append((root, result.reason))
            else:
                refused.append((root, GIT_NO_REPO_REASON))
        if not entries:
            # Every candidate went away or became unusable between the
            # affordance refresh and this read -- git's live answer wins
            # over what the button believed a moment ago.
            self.notify(_combined_refusal(refused), severity="warning")
            self._refresh_mode_affordances()
            return
        if action == "pr" and len(entries) == 1:
            # Nothing to confirm: opening a page is not a mutation, so the
            # single-repository case goes straight through.
            self._dispatch_pr(entries[0][0], entries[0][1])
            return

        def _apply(result: "dict | None") -> None:
            if not result:
                return
            if result.get("action") == "pr":
                self._dispatch_pr(str(result["root"]), result["info"])
            else:
                self._dispatch_push(result)

        self.app.push_screen(
            ChangeGitPushModal(action=action, entries=entries), callback=_apply
        )

    def _dispatch_push(self, request: dict) -> None:
        """Run the confirmed push off-thread.

        Args:
            request: The modal's result -- ``{"action", "root", "remote",
                "info"}``. ``remote`` is ``None`` when the branch already
                has an upstream (the engine then pushes to
                ``info.upstream_remote``; spec §6 forbids refspec games).
        """
        root = str(request["root"])
        remote = request["remote"]
        info = request["info"]
        # Defense in depth (T8 review): only a remote DETECTION actually
        # reported may be forwarded. The engine carries the real guard
        # against option-shaped names (a repository can ship a remote
        # literally named `--force`, and git reads it as the flag), because
        # such a name IS a detected remote and would pass this check; this
        # one covers the other direction -- a caller naming a remote no
        # detection ever saw.
        known_remotes = {name for name, _url in info.remotes}
        if remote is not None and remote not in known_remotes:
            logger.warning(
                f"change_review: refusing to push to an undetected remote "
                f"{remote!r} at {root!r}"
            )
            self.notify(
                f"Push refused: {remote!r} is not one of this repository's "
                "configured remotes",
                severity="error",
            )
            return
        token = self._git_action_token = object()
        self._set_git_busy(True)
        provider = self._provider
        app = self.app

        def _push() -> None:
            from tldw_chatbook.Workspaces.git_workspace import GitWorkspaceError

            # ASYMMETRIC by contract (the T5 seam comment), and deliberately
            # NOT wrapped symmetrically: `push_current` RAISES only for the
            # two preconditions (detached HEAD, no remote), while a push git
            # actually REJECTED comes back as a returned `PushResult` whose
            # `state` is "failed". Catching the returned case as an
            # exception -- or reporting the raised case as a push outcome --
            # would each misdescribe what happened to the user's remote.
            try:
                result = provider.push_current(root, info, remote)
            except GitWorkspaceError as exc:
                _land_on_ui(app, self._land_push_refused, token, str(exc))
                return
            except Exception as exc:  # noqa: BLE001 -- the worker must live
                logger.opt(exception=True).warning(
                    f"change_review: push failed for {root!r}"
                )
                # A DIFFERENT landing from the typed refusal above (T8
                # review): anything reaching here is a bug, not one of the
                # engine's honest preconditions, and reporting a raw
                # `str(exc)` at warning level would be indistinguishable
                # from "no git remote configured".
                _land_on_ui(app, self._land_push_error, token, str(exc))
                return
            _land_on_ui(app, self._land_push_result, token, result, info, remote)

        self.run_worker(
            _push,
            thread=True,
            exclusive=True,
            group=GIT_ACTION_WORKER_GROUP,
        )

    def _land_push_refused(self, token: object, message: str) -> None:
        """A push the engine DECLINED to run: report its reason verbatim.

        Only ``GitWorkspaceError`` reaches here -- the engine's own honest
        preconditions ("no git remote configured", "no branch checked out",
        "unsupported remote name"). An unexpected exception goes to
        :meth:`_land_push_error` instead, so the two can never read alike.

        Args:
            token: The dispatch identity of the refused push.
            message: The engine's own sentence.
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        self.notify(message, severity="warning")

    def _land_push_error(self, token: object, message: str) -> None:
        """A push that hit an UNEXPECTED error: say so, and say it loudly.

        Args:
            token: The dispatch identity of the failed push.
            message: The exception's text, prefixed so it can never be
                mistaken for one of the engine's refusals.
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        self.notify(f"Push could not run: {message}", severity="error")

    def _land_push_result(
        self,
        token: object,
        result: "PushResult",
        info: "GitWorkspaceInfo",
        remote: "str | None",
    ) -> None:
        """Report the push's outcome, then re-read detection.

        A FAILED push is a returned result, not an exception, and its
        ``detail`` is git's own stderr excerpt -- surfaced VERBATIM (the
        engine has already appended the credential-helper hint when the
        excerpt looks like one; re-mapping it here would either duplicate
        or contradict that).

        Args:
            token: The dispatch identity of this push.
            result: The engine's :class:`PushResult`.
            info: The fresh info the push ran against (names the branch).
            remote: The explicit remote, or ``None`` when the upstream's
                own remote was used.
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        branch = info.branch or "HEAD"
        target = remote or info.upstream_remote or "the remote"
        if result.state == "pushed":
            self.notify(f"Pushed {branch} to {target}")
        elif result.state == "up_to_date":
            self.notify(f"{target} is already up to date — nothing to push")
        else:
            detail = result.detail or "git reported no detail"
            self.notify(f"Push failed: {detail}", severity="error")
        # Re-read on EVERY outcome (spec §6: ahead/behind is re-read after a
        # push): a first `-u` push sets an upstream, and even a rejected one
        # means the header's ahead/behind is now describing a remote whose
        # state we have just learned is not what we thought.
        if self._current_mode_active():
            self._load_current_mode()

    def _dispatch_pr(self, root: str, info: "GitWorkspaceInfo") -> None:
        """Build the compare URL off-thread, then open it.

        Args:
            root: The resolved root whose branch gets the PR page.
            info: That root's FRESH detection.
        """
        token = self._git_action_token = object()
        self._set_git_busy(True)
        provider = self._provider
        app = self.app

        def _build() -> None:
            # `pr_url` NEVER raises by contract -- it returns a URL or a
            # typed refusal, so `isinstance` is the check, never try/except.
            # The guard below is for a BUG in it, not for its normal
            # refusal path.
            try:
                outcome = provider.pr_url(root, info)
            except Exception as exc:  # noqa: BLE001 -- the worker must live
                logger.opt(exception=True).warning(
                    f"change_review: PR link failed for {root!r}"
                )
                _land_on_ui(app, self._land_git_target_failure, token, str(exc))
                return
            _land_on_ui(app, self._land_pr_url, token, outcome)

        self.run_worker(
            _build,
            thread=True,
            exclusive=True,
            group=GIT_ACTION_WORKER_GROUP,
        )

    def _land_pr_url(self, token: object, outcome: object) -> None:
        """Open the compare URL, or say why there is none.

        ``self.app.open_url`` and never ``webbrowser.open`` (spec §6):
        ``webbrowser`` can write to the terminal's stdout, which corrupts a
        running TUI.

        Args:
            token: The dispatch identity of this PR build.
            outcome: ``provider.pr_url``'s result -- a URL string or a
                :class:`~tldw_chatbook.Workspaces.git_workspace.GitWorkspaceRefusal`.
        """
        if not self._git_action_is_live(token):
            return
        self._set_git_busy(False)
        from tldw_chatbook.Workspaces.git_workspace import (
            GitWorkspaceRefusal as _Refusal,
        )

        if isinstance(outcome, _Refusal):
            self.notify(outcome.reason, severity="warning")
            return
        url = str(outcome)
        try:
            self.app.open_url(url)
        except Exception:  # noqa: BLE001 -- never raise out of a landing
            logger.opt(exception=True).warning("change_review: open_url failed")
            # The URL itself is the recovery: a user who can read it can
            # paste it, which a bare "could not open a browser" would deny.
            self.notify(f"Could not open a browser — {url}", severity="error")
            return
        self.notify(f"Opening {url}")

    @staticmethod
    def _leaf_label(
        row: dict,
        change: ChangedFile,
        multi_root: bool,
        badge: bool = False,
    ) -> Text:
        """Build one leaf label as a PLAIN rich Text.

        Tree labels are markup-PARSED when given as strings — "[binary]"
        silently vanished as a tag, and a FILENAME containing brackets
        would corrupt the same way (caught by this screen's own group
        test). A ``Text`` instance is rendered verbatim: labels are data.
        """
        parts = [change.path]
        if change.status == "R" and change.old_path:
            parts = [f"{change.old_path} → {change.path}"]
        if change.binary:
            parts.append("(binary)")
        else:
            parts.append(f"+{change.adds} −{change.dels}")
        if multi_root:
            from pathlib import Path as _P

            parts.append(f"· {_P(str(row['root'])).name}")
        label = Text("  ".join(parts))
        if badge:
            # TASK-1978: exact spec copy — 'outside direct file tools',
            # never 'not by the agent' (script writes are agent work too,
            # and badge absence is not proof of tool provenance). Dim,
            # monochrome.
            label.append(
                "  ⚠ changed outside direct file tools", style="dim"
            )
        return label

    def _show_empty(self, copy: str) -> None:
        self.query_one("#change-review-diff-content", Static).update(copy)

    # -- file focus / diff pane -------------------------------------------

    def select_file(self, path: str, snapshot_id: int | None = None) -> None:
        """Focus the leaf matching ``path``, preferring an exact snapshot.

        Args:
            path: Root-relative path from the active turn.
            snapshot_id: When given, prefer the leaf whose OWNING
                ``change_snapshots`` row id equals this (TASK-18060 Task 3)
                -- a run can hold two windows (its own turn window and a
                surviving sub-agent's post-turn window) covering the SAME
                path, and path-only matching can only ever reach the
                first-recorded one. Falls back to the first path match when
                no leaf's row id matches -- legacy callers that pass no
                snapshot id keep today's behavior exactly.
        """
        if snapshot_id is not None:
            for index, (row, change) in enumerate(self._leaves):
                if change.path == path and row.get("id") == snapshot_id:
                    self._focus_leaf(index)
                    return
        for index, (_row, change) in enumerate(self._leaves):
            if change.path == path:
                self._focus_leaf(index)
                return

    def _focus_leaf(self, index: int) -> None:
        if not self._leaves:
            return
        self._focused_leaf = max(0, min(index, len(self._leaves) - 1))
        # TASK-18060 Task 6: the line cursor is per-file — every leaf focus
        # (a fresh selection, or j/k switching files) starts back at the
        # top rather than carrying over an unrelated file's line index.
        self._cursor_line = 0
        row, change = self._leaves[self._focused_leaf]
        # TASK-18060 Task 7 fix round: ONE synchronous notes read, shared by
        # both the inline diff-line marker (spec §3's "● comment") and the
        # strip below — `_render_diff` needs the marker set computed BEFORE
        # it renders, so this must happen ahead of that call.
        leaf_notes = self._notes_for_leaf(row, change)
        self._marked_diff_lines = self._marked_diff_line_indices(leaf_notes)
        self._render_diff(row, change)
        # TASK-18060 Task 7 (review-rail spec §3): the strip's one refresh
        # choke point — every leaf focus (a fresh selection, j/k, or a
        # mouse click) shows exactly the newly-focused file's notes.
        self._refresh_notes_strip(leaf_notes)

    def _move_diff_cursor(self, delta: int) -> None:
        """Move the diff-pane line cursor by ``delta`` and re-render.

        TASK-18060 Task 6: called from ``ChangeReviewDiffPane.on_key``'s
        up/down reclaim. The lower bound (0) is clamped HERE; the upper
        bound (the file's own rendered-line count, excluding the
        truncation tail) is clamped inside ``_render_diff`` against that
        render's own line count — so a downward move that overshoots past
        the cap settles on the last real line rather than the tail.

        Args:
            delta: ``-1`` for up, ``1`` for down.
        """
        if not self._leaves or self._focused_leaf < 0:
            return
        self._cursor_line = max(0, self._cursor_line + delta)
        row, change = self._leaves[self._focused_leaf]
        self._render_diff(row, change)

    def _scroll_cursor_into_view(self, line: int) -> None:
        """Scroll the diff pane so the cursor's rendered line is visible.

        TASK-18060 Task 6 (spec §3): the diff pane is one flat ``Static``,
        line-per-line, so the cursor's line index IS its y-offset inside
        the pane's virtual content — ``Region(0, line, 1, 1)`` names it
        directly. Deferred via ``call_after_refresh`` rather than called
        synchronously right after ``content.update()``: the pane's virtual
        size (needed to compute how far to scroll) is only current once
        Textual has laid the just-updated ``Static`` out, which happens on
        the next refresh, not synchronously inside this call.

        Args:
            line: The cursor's 0-based rendered-line index to scroll to.
        """
        try:
            pane = self.query_one("#change-review-diff", ChangeReviewDiffPane)
        except Exception:  # noqa: BLE001 -- screen dismissed before refresh
            return

        def _scroll() -> None:
            try:
                pane.scroll_to_region(
                    Region(0, line, 1, 1), animate=False, x_axis=False
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "change_review: cursor scroll-into-view failed"
                )

        self.call_after_refresh(_scroll)

    @on(Tree.NodeSelected, "#change-review-tree")
    def _on_tree_node_selected(self, event: "Tree.NodeSelected") -> None:
        """Load the selected file's diff — the mouse-click path (TASK-2032).

        Group nodes carry no index and are ignored (their click just
        expands/collapses).
        """
        index = getattr(event.node, "data", None)
        if isinstance(index, int):
            self._focus_leaf(index)

    def action_next_file(self) -> None:
        self._focus_leaf(self._focused_leaf + 1)

    def action_previous_file(self) -> None:
        self._focus_leaf(self._focused_leaf - 1)

    def action_focus_diff(self) -> None:
        self.query_one("#change-review-diff", VerticalScroll).focus()

    # -- comment creation + notes strip (TASK-18060 Task 7, spec §3) ------

    async def action_comment_file(self) -> None:
        """`C`: open a whole-file comment on the focused leaf.

        The current-mode refusal (spec §4.1) lives at the top of
        ``_open_comment_input`` -- the ONE choke point both this action,
        the header button, and the diff pane's ``c`` reclaim funnel
        through, so every comment path refuses with the same copy exactly
        once per attempt.
        """
        await self._open_comment_input("file")

    @on(Button.Pressed, "#change-review-comment-file-btn")
    async def _on_comment_file_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._open_comment_input("file")

    async def _open_comment_input(self, kind: str) -> None:
        """Mount the inline comment ``Input`` below the diff pane.

        Args:
            kind: ``"diff_line"`` (the cursor's current line) or
                ``"file"`` (the whole focused file). A ``"diff_line"``
                request degrades to a no-op when the focused file has no
                cursor to anchor to — a binary render (spec §3's "no
                cursor there") or a diff the provider cannot produce
                (a pruned/tracking-error row). ``"file"`` always works,
                including on those same renders.
        """
        try:
            # TASK-16801 arc B (spec §4.1): notes anchor to
            # `change_snapshots` rows -- the pseudo row's `id=-1` must
            # never reach the notes DB. Gated at the TOP, ahead of the
            # focus checks, so `C`, the button, and the pane's `c` all
            # refuse with copy rather than no-oping silently.
            if self._current_mode_active():
                self.notify(CURRENT_MODE_COMMENT_REFUSAL, severity="warning")
                return
            if not self._leaves or self._focused_leaf < 0:
                return
            row, change = self._leaves[self._focused_leaf]
            if kind == "diff_line":
                if change.binary:
                    return
                try:
                    self._diff_text_for(row, change)
                except ChangeTrackingError:
                    return
            container = self.query_one("#change-review-right", Vertical)
            existing = list(container.query(".change-review-comment-input"))
            if existing:
                # Already open (either kind) -- focus it rather than
                # mounting a second input, same "already open" precedent
                # as the turn file card's hunk note input.
                existing[0].focus()
                return
            note_input = Input(
                classes="change-review-comment-input",
                placeholder=(
                    "Add a comment on this line…"
                    if kind == "diff_line"
                    else "Add a comment on this file…"
                ),
                max_length=NOTE_MAX_LENGTH,
            )
            note_input.anchor_kind = kind
            note_input.leaf_row = row
            note_input.leaf_change = change
            note_input.cursor_line = self._cursor_line if kind == "diff_line" else None
            strip = self.query_one("#change-review-notes-strip", Vertical)
            await container.mount(note_input, before=strip)
            note_input.focus()
        except Exception:
            logger.opt(exception=True).warning(
                "change_review: comment input open failed"
            )

    async def on_key(self, event: Key) -> None:
        """Reclaim Enter/Escape while the comment ``Input`` is focused.

        Same raw-Key belt-and-braces precedent as
        ``ConsoleTurnFileCard.on_key`` (see that class's own docstring for
        the traced Textual dispatch this mirrors): the comment ``Input``
        is mounted as a SIBLING of the diff pane, so its bubble path
        reaches this screen directly (never through
        ``ChangeReviewDiffPane.on_key``) — Escape here must NOT fall
        through to this screen's own ``escape -> dismiss_screen`` binding.
        """
        try:
            focused = self.app.focused
            if focused is None or not focused.has_class(
                "change-review-comment-input"
            ):
                return
            if event.key == "enter":
                event.stop()
                event.prevent_default()
                await self._save_comment_input(focused)
            elif event.key == "escape":
                event.stop()
                event.prevent_default()
                await self._cancel_comment_input()
            elif event.key in ("up", "down"):
                event.stop()
                event.prevent_default()
        except Exception:
            logger.opt(exception=True).warning(
                "change_review: comment input key handling failed"
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the comment input: save off-thread."""
        if not event.input.has_class("change-review-comment-input"):
            return
        event.stop()
        await self._save_comment_input(event.input)

    async def _cancel_comment_input(self) -> None:
        """Unmount the comment input without saving; refocus the pane.

        Spec §3's Escape contract: cancels WITHOUT dismissing the screen
        or moving focus to the tree — focus returns to the diff pane.
        """
        try:
            focused = self.app.focused
            if focused is not None and focused.has_class(
                "change-review-comment-input"
            ):
                await focused.remove()
            self.query_one("#change-review-diff", ChangeReviewDiffPane).focus()
        except Exception:
            logger.opt(exception=True).warning(
                "change_review: comment input cancel failed"
            )

    async def _save_comment_input(self, note_input: Input) -> None:
        """Validate, persist off-thread, and refresh the notes strip.

        A raising ``provider.add_change_note`` (or any other failure) is
        swallowed here — the input stays mounted with the user's text
        intact so nothing is lost, and a warning is logged (the screen's
        own "no exception escapes an `on_*` handler" rule, same as the
        turn file card's).

        Args:
            note_input: The submitted comment ``Input``.
        """
        try:
            if not note_input.is_mounted or self._active_turn is None:
                return
            text = _validate_note_text(note_input.value)
            if text is None:
                return
            row = getattr(note_input, "leaf_row", None)
            change = getattr(note_input, "leaf_change", None)
            kind = getattr(note_input, "anchor_kind", None)
            if row is None or change is None or kind is None:
                return
            # TASK-18060 final-review fix round (Fix 1a): the run this note
            # belongs to is the CAPTURED leaf's own row -- read at
            # input-OPEN time alongside `change`/`cursor_line` -- never
            # `self._active_turn`, which is read at SAVE time and can have
            # moved on to a different turn if the Select was switched while
            # this input sat open (Fix 1b closes that input on switch, but
            # this keeps the WRITE itself self-consistent regardless: the
            # note's run_id always matches the same row its path/snapshot/
            # excerpt came from).
            run_id = str(row["run_id"])
            snapshot_id = row.get("id")

            if kind == "diff_line":
                cursor_line = getattr(note_input, "cursor_line", None)
                if cursor_line is None:
                    return
                try:
                    diff_text = self._diff_text_for(row, change)
                except ChangeTrackingError:
                    self.notify(
                        "Diff unavailable — comment not saved",
                        severity="warning",
                    )
                    return
                lines = diff_text.splitlines()
                if not (0 <= cursor_line < len(lines)):
                    # Fix 4(a) (final review): mirror the ChangeTrackingError
                    # branch above -- a shrunk diff (the file changed under
                    # the open input, e.g. a revert or a later turn) must
                    # tell the user the comment was NOT saved rather than
                    # silently leaving the input mounted with no feedback.
                    self.notify(
                        "That diff line no longer exists — comment not saved",
                        severity="warning",
                    )
                    return
                diff_line_text = lines[cursor_line]
                hunks = split_unified_diff(diff_text)
                hunk_idx, hunk = _hunk_containing_line(hunks, cursor_line)
                write_kwargs = dict(
                    anchor_kind="diff_line",
                    hunk_index=hunk_idx,
                    hunk_header=hunk.header if hunk is not None else "",
                    hunk_excerpt=hunk_excerpt(hunk) if hunk is not None else "",
                    diff_line_index=cursor_line,
                    diff_line_text=diff_line_text,
                )
            else:
                write_kwargs = dict(
                    anchor_kind="file",
                    hunk_index=-1,
                    hunk_header="",
                    hunk_excerpt="",
                    diff_line_index=None,
                    diff_line_text=None,
                )

            def _write() -> int:
                return self._provider.add_change_note(
                    run_id=run_id,
                    root=row["root"],
                    path=change.path,
                    note=text,
                    snapshot_id=snapshot_id,
                    **write_kwargs,
                )

            try:
                await asyncio.to_thread(_write)
            except Exception:
                logger.opt(exception=True).warning(
                    "change_review: comment save failed"
                )
                self.notify("Could not save comment", severity="warning")
                return
            if note_input.is_mounted:
                await note_input.remove()
            try:
                self.query_one(
                    "#change-review-diff", ChangeReviewDiffPane
                ).focus()
            except Exception:  # noqa: BLE001 -- focus-return is cosmetic
                pass
            self._refresh_notes_ui_for_focused_leaf()
        except Exception:
            logger.opt(exception=True).warning(
                "change_review: comment save failed"
            )

    async def _delete_review_note(self, button: Button) -> None:
        """Delete a pending note off-thread and refresh the strip.

        Args:
            button: The pressed ``✕`` button, carrying a ``note_id``
                attribute set at mount time.
        """
        try:
            note_id = getattr(button, "note_id", None)
            if note_id is None:
                return
            delete_change_note = getattr(
                self._provider, "delete_change_note", None
            )
            if not callable(delete_change_note):
                return

            def _delete() -> bool:
                return delete_change_note(note_id)

            deleted = await asyncio.to_thread(_delete)
            if not deleted:
                # Delivered behind the screen's back (same live-view
                # honesty rule as the turn file card): don't silently
                # no-op a press that looks actionable.
                self.notify(
                    "Note already sent — no longer deletable",
                    severity="warning",
                )
                return
            self._refresh_notes_ui_for_focused_leaf()
        except Exception:
            logger.opt(exception=True).warning(
                "change_review: note delete failed"
            )

    @on(Button.Pressed, ".change-review-note-delete")
    async def _on_note_delete_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self._delete_review_note(event.button)

    def _notes_for_leaf(self, row: dict, change: ChangedFile) -> list[dict]:
        """All notes (any kind) anchored to ``(row, change)``, filtered.

        TASK-18060 Task 7 fix round: the ONE synchronous ``notes_for_run``
        read + root/path/snapshot filter, shared by the notes strip
        (``_refresh_notes_strip``) and the diff pane's inline "● comment"
        marker (``_marked_diff_line_indices``) — a single file focus or
        note mutation costs exactly one query, not two. Spec §3's posture
        correction: synchronous, matching this screen's existing
        diff-load posture (only the comment WRITE paths run off-thread).

        Args:
            row: The focused leaf's owning ``change_snapshots`` row.
            change: The focused leaf's ``ChangedFile``.

        Returns:
            The leaf's notes, oldest first (``notes_for_run``'s own
            order); empty when the read itself fails (logged, never
            raised — a note-load failure must never break the pane).
        """
        # TASK-16801 arc B (spec §4.1): the notes strip and the inline
        # marker set are skipped ENTIRELY in current mode -- there is no
        # snapshot id to query by, so the read must not happen at all
        # (`_active_turn` is already None there; this is the explicit,
        # named guard the row-consumers table asks for).
        if self._current_mode_active():
            return []
        if self._active_turn is None:
            return []
        try:
            notes = self._provider.notes_for_run(self._active_turn.run_id)
        except Exception:
            logger.opt(exception=True).warning(
                "change_review: notes load failed"
            )
            return []
        return [
            note
            for note in notes
            if note.get("root") == row.get("root")
            and note.get("path") == change.path
            and _note_matches_leaf(note, row.get("id"))
        ]

    @staticmethod
    def _marked_diff_line_indices(notes: list[dict]) -> "set[int]":
        """``diff_line`` note indices from an already-filtered leaf list.

        Pure (no I/O) — ``_render_diff`` re-renders on every cursor move,
        so the marker set is computed once per file-focus/note-mutation
        (``_focus_leaf``/``_refresh_notes_ui_for_focused_leaf``) and
        merely CONSULTED here, keeping the render itself O(lines).

        Args:
            notes: ``_notes_for_leaf``'s output.

        Returns:
            The set of 0-based full-diff line indices carrying a
            ``diff_line`` note.
        """
        marked: set[int] = set()
        for note in notes:
            if note.get("anchor_kind") != "diff_line":
                continue
            index = note.get("diff_line_index")
            if index is not None:
                marked.add(int(index))
        return marked

    def _refresh_notes_ui_for_focused_leaf(self) -> None:
        """The ONE choke point after any note mutation (save/delete).

        Recomputes the marker set and re-renders the diff (so a new/
        removed inline "● comment" marker shows immediately) AND
        refreshes the strip — both fed by a single ``_notes_for_leaf``
        read. A save/delete when nothing is focused degrades to just
        clearing the strip (mirrors ``_refresh_notes_strip``'s own
        no-leaves guard).
        """
        if not self._leaves or self._focused_leaf < 0:
            self._refresh_notes_strip()
            return
        row, change = self._leaves[self._focused_leaf]
        leaf_notes = self._notes_for_leaf(row, change)
        self._marked_diff_lines = self._marked_diff_line_indices(leaf_notes)
        self._render_diff(row, change)
        self._refresh_notes_strip(leaf_notes)

    def _refresh_notes_strip(self, notes: "list[dict] | None" = None) -> None:
        """Repopulate the focused file's notes strip.

        Args:
            notes: The focused leaf's already-filtered notes
                (``_notes_for_leaf``'s output), when the caller already
                has them (``_focus_leaf``/
                ``_refresh_notes_ui_for_focused_leaf``) — avoids a second
                query. ``None`` (callers with no notes in hand) fetches
                fresh.
        """
        try:
            strip = self.query_one("#change-review-notes-strip", Vertical)
        except Exception:  # noqa: BLE001 -- screen dismissed before refresh
            return
        try:
            strip.remove_children()
        except Exception:
            logger.opt(exception=True).warning(
                "change_review: notes strip clear failed"
            )
            return
        if (
            not self._leaves
            or self._focused_leaf < 0
            or self._active_turn is None
        ):
            strip.display = False
            return
        if notes is None:
            row, change = self._leaves[self._focused_leaf]
            notes = self._notes_for_leaf(row, change)
        strip.display = bool(notes)
        for note in notes:
            try:
                strip.mount(self._build_note_row(note))
            except Exception:
                logger.opt(exception=True).warning(
                    "change_review: note row mount failed"
                )

    @staticmethod
    def _build_note_row(note: dict) -> Horizontal:
        """Render one note as a ``.change-review-note-row`` row.

        Args:
            note: A ``change_notes`` row dict (at minimum ``id``,
                ``note``, ``anchor_kind``, ``diff_line_index``,
                ``delivered_at``).

        Returns:
            A row with the note's kind label + text, plus a ``✕`` delete
            button while ``delivered_at`` is null — delivered notes
            render a ``sent`` marker instead and carry no delete
            affordance (they are record).
        """
        delivered = note.get("delivered_at") is not None
        label = Text()
        label.append(f"{resolve_glyph(_GLYPH_NOTE)} ", style="dim")
        label.append(_note_kind_label(note), style="bold")
        label.append(" · ")
        label.append(str(note.get("note", "")))
        if delivered:
            label.append("  · sent", style="dim")
        children: list = [
            Static(label, classes="change-review-note-text")
        ]
        if not delivered:
            delete_btn = Button(
                Text(resolve_glyph(_GLYPH_DELETE)),
                classes="change-review-note-delete",
                compact=True,
            )
            delete_btn.note_id = int(note["id"])
            delete_btn.active_effect_duration = 0
            children.append(delete_btn)
        return Horizontal(*children, classes="change-review-note-row")

    def action_revert_file(self) -> None:
        """Revert the focused file (confirmed)."""
        # TASK-16801 arc B (spec §4.1): revert restores a TURN's baseline;
        # the working tree has none. Gated at the top so the refusal is
        # visible copy rather than a silently dead key.
        if self._current_mode_active():
            self.notify(CURRENT_MODE_REVERT_REFUSAL, severity="warning")
            return
        if not self._leaves or self._focused_leaf < 0:
            return
        row, change = self._leaves[self._focused_leaf]
        self._confirm_and_revert(row, [change.path], f"Revert {change.path}?")

    def action_undo_all(self) -> None:
        """Revert every file in the active turn, per root (confirmed)."""
        # TASK-16801 arc B (spec §4.1): same gate as `action_revert_file`,
        # and it must come BEFORE the `_active_turn is None` check below --
        # current mode holds no active turn, so without this the key would
        # be silently dead instead of refusing with copy.
        if self._current_mode_active():
            self.notify(CURRENT_MODE_REVERT_REFUSAL, severity="warning")
            return
        if self._active_turn is None or not self._leaves:
            return
        by_row: dict[int, tuple[dict, list[str]]] = {}
        for row, change in self._leaves:
            key = id(row)
            by_row.setdefault(key, (row, []))[1].append(change.path)
        total = len(self._leaves)
        # Multi-root Undo-all: one confirm covering everything, then the
        # engine runs per root row.
        rows_paths = list(by_row.values())
        all_edited: list[str] = []
        for row, paths in rows_paths:
            all_edited.extend(self._provider.preflight_revert(row, paths).edited_since)

        def _apply(confirmed: bool | None) -> None:
            if not confirmed:
                return
            outcomes = []
            for row, paths in rows_paths:
                outcomes.extend(self._run_revert(row, paths))
            self._report_outcomes(outcomes)

        self.app.push_screen(
            ChangeRevertConfirmModal(
                f"Undo all {total} files from this turn?", all_edited
            ),
            callback=_apply,
        )

    def _confirm_and_revert(self, row: dict, paths: list[str], summary: str) -> None:
        edited = self._provider.preflight_revert(row, paths).edited_since

        def _apply(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self._report_outcomes(self._run_revert(row, paths))

        self.app.push_screen(
            ChangeRevertConfirmModal(summary, edited), callback=_apply
        )

    def _run_revert(self, row: dict, paths: list[str]) -> list:
        from tldw_chatbook.Workspaces.change_revert import RevertRefusedError

        try:
            return self._provider.revert(row, paths)
        except RevertRefusedError as exc:
            self.notify(str(exc), severity="warning")
            return []

    def _report_outcomes(self, outcomes: list) -> None:
        """Per-path honesty: failures are named, never rolled up silently."""
        failed = [o for o in outcomes if not o.ok]
        if failed:
            names = ", ".join(f"{o.path} ({o.error})" for o in failed[:5])
            self.notify(
                f"{len(failed)} file(s) could not be reverted: {names}",
                severity="error",
            )
        elif outcomes:
            self.notify(f"Reverted {len(outcomes)} file(s).")
        if self._active_turn is not None:
            # Reload from disk truth -- the turn's diff no longer matches.
            self._load_turn(self._active_turn)

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)

    def diff_pane_text(self) -> str:
        """The diff pane's current plain text (test/observability seam)."""
        content = self.query_one("#change-review-diff-content", Static)
        renderable = content.renderable
        if isinstance(renderable, Text):
            return renderable.plain
        return str(renderable)

    def _diff_text_for(self, row: dict, change: ChangedFile) -> str:
        """Fetch (once per focused leaf) and memoize ``change``'s diff text.

        TASK-18060 final-review fix round (Fix 2): the SOLE path every
        diff-line consumer reads through — ``_render_diff``'s render,
        ``_open_comment_input``'s diff-line availability probe, and
        ``_save_comment_input``'s line-text lookup. Pre-fix each ran its
        OWN ``provider.diff_text`` call (a git subprocess pair): cursor
        movement alone re-ran it on EVERY keypress (``_move_diff_cursor``
        -> ``_render_diff``, 40 arrow presses == 40 synchronous subprocess
        spawns on the UI thread), and a single ``c``+Enter line-comment
        flow ran it three separate times (render, open-probe, save).

        Cached per ``(generation, id(row), change.path)`` — ``_load_turn``
        bumps ``_diff_cache_generation`` on every turn (re)load, including
        the post-revert reload in ``_report_outcomes``, so a revert of the
        SAME turn (identical row objects, identical path) still forces a
        fresh read rather than serving pre-revert content still sitting in
        the cache; a genuine file-focus change is already disambiguated by
        ``change.path`` differing within one generation, and a STALE
        capture (a comment input opened against an earlier leaf) resolves
        correctly too since its own ``(row, path)`` pair still computes the
        right key.

        Args:
            row: The leaf's owning ``change_snapshots`` row.
            change: The leaf's ``ChangedFile``.

        Returns:
            The file's unified diff text.

        Raises:
            ChangeTrackingError: Propagated from the provider on a cache
                miss — and itself cached, so a transient failure does not
                re-spawn a doomed subprocess on every subsequent keypress
                either; a later ``_load_turn`` (a fresh generation) is what
                gives it another chance.
        """
        key = (self._diff_cache_generation, id(row), change.path)
        if self._diff_cache_key != key:
            self._diff_cache_key = key
            self._diff_cache_text = None
            self._diff_cache_error = None
            try:
                if row.get("kind") == "git_current":
                    self._diff_cache_text = self._current_diff_text(row, change)
                else:
                    self._diff_cache_text = self._provider.diff_text(
                        row, change.path
                    )
            except ChangeTrackingError as exc:
                self._diff_cache_error = exc
        if self._diff_cache_error is not None:
            raise self._diff_cache_error
        assert self._diff_cache_text is not None
        return self._diff_cache_text

    def _current_diff_text(self, row: dict, change: ChangedFile) -> str:
        """One `current`-mode leaf's diff text (TASK-16801 arc B, spec §4).

        Routes on untracked-ness, which is why
        :attr:`CurrentRootStatus.untracked` is carried separately from the
        collapsed status letter: ``git diff HEAD`` is a FATAL error against
        an untracked path and against an unborn HEAD (spec §2 probe 4),
        where EVERY file is untracked and the whole tree therefore renders
        through the synthesized preview.

        Args:
            row: The leaf's pseudo row (``kind == "git_current"``).
            change: The leaf's :class:`ChangedFile`.

        Returns:
            The tracked file's unified diff, or the untracked file's
            bounded preview.

        Raises:
            ChangeTrackingError: Translated from the engine's
                ``GitWorkspaceError`` so this mode's failures render
                through the SAME "diff unavailable: …" path every other
                leaf already uses -- ``_render_diff`` and the comment
                paths catch exactly one error type, and current mode must
                not become the one place that can crash them.
        """
        from tldw_chatbook.Workspaces.git_workspace import GitWorkspaceError

        root = str(row.get("root") or "")
        if change.path in self._current_untracked.get(root, frozenset()):
            return self._provider.untracked_preview(root, change.path)
        try:
            return self._provider.current_diff_text(root, change)
        except GitWorkspaceError as exc:
            raise ChangeTrackingError(str(exc)) from exc

    def _render_diff(self, row: dict, change: ChangedFile) -> None:
        content = self.query_one("#change-review-diff-content", Static)
        if change.binary:
            content.update(
                Text(
                    f"{change.path}\nBinary file changed.",
                    no_wrap=False,
                )
            )
            return
        try:
            diff = self._diff_text_for(row, change)
        except ChangeTrackingError as exc:
            content.update(Text(f"diff unavailable: {exc}"))
            return
        cap = self._provider.diff_display_max_lines
        lines = diff.splitlines()
        hidden = max(0, len(lines) - cap)
        # TASK-18060 Task 6 (spec §3): the cursor only ranges over REAL
        # rendered lines, never the truncation tail line appended below —
        # clamped here against THIS render's own count so a cursor left
        # over from a longer file (or a downward move that overshot) never
        # points past what is actually on screen.
        rendered_count = min(len(lines), cap)
        self._cursor_line = (
            max(0, min(self._cursor_line, rendered_count - 1))
            if rendered_count > 0
            else 0
        )
        # TASK-18060 Task 6 follow-up (review catch): `_scroll_cursor_
        # into_view`'s `Region(0, line, 1, 1)` target assumes the cursor's
        # logical line index IS its rendered row -- true only when this
        # `Text` never wraps. Without that, a long line ahead of the cursor
        # consumes several extra visual rows under word-wrapping, silently
        # drifting every row after it and scrolling to the WRONG target.
        # `no_wrap=True` is set here for correctness when this `Text` is
        # consumed by a plain Rich `Console` (tests, `diff_pane_text()`),
        # but it is NOT what stops Textual's own Static from wrapping --
        # empirically, Textual 8.x converts a `rich.text.Text` into its own
        # `Content` type (`textual.visual.visualize`/`Content.from_rich_
        # text`), which discards this flag entirely and instead reads a
        # `text-wrap` CSS rule at render time. The actual fix is
        # `.change-review-diff-body`'s CSS (`text-wrap: nowrap` +
        # `width: auto` -- see `_change_review.tcss` for why BOTH are
        # required); `#change-review-diff`'s `overflow-x: auto` then makes
        # a long, now-unwrapped line horizontally scrollable instead of
        # clipped.
        text = Text(no_wrap=True)
        for index, line in enumerate(lines[:cap]):
            # Plain-string appends with explicit styles: content is DATA;
            # nothing here is ever markup-parsed.
            if line.startswith("+") and not line.startswith("+++"):
                style = "green"
            elif line.startswith("-") and not line.startswith("---"):
                style = "red"
            elif line.startswith("@@"):
                style = "cyan"
            else:
                style = ""
            cursor_here = index == self._cursor_line
            if cursor_here:
                style = f"{style} {_CURSOR_LINE_STYLE}".strip()
            if style:
                text.append(line, style=style)
            else:
                text.append(line)
            if index in self._marked_diff_lines:
                # TASK-18060 Task 7 fix round (spec §3 Note display): a
                # line carrying a `diff_line` note gets a dim "● comment"
                # marker APPENDED to the same logical line -- never a new
                # line, so the row==index invariant the Task 6 follow-up
                # fix established (`_scroll_cursor_into_view`) is
                # untouched (this only widens the line; wrapping stays
                # off). Composes with the cursor highlight: when this IS
                # the cursor's own line, the marker also carries the
                # cursor's background so the highlighted band doesn't
                # visibly break partway through.
                marker_style = (
                    f"dim {_CURSOR_LINE_STYLE}" if cursor_here else "dim"
                )
                text.append(
                    f" {resolve_glyph(_GLYPH_LINE_COMMENT_MARKER)} comment",
                    style=marker_style,
                )
            text.append("\n")
        if hidden:
            text.append(
                f"… diff truncated — {hidden} more lines", style="yellow"
            )
        content.update(text)
        if rendered_count > 0:
            self._scroll_cursor_into_view(self._cursor_line)


class ChangeRevertConfirmModal(SafeModalDismissMixin, ModalScreen[bool]):
    """Confirm a revert, naming user-edited files BY NAME (TASK-1974).

    The list is the guard's whole point: files whose disk state differs from
    the turn's end were changed by the user (or a later turn) after this
    turn, and reverting will overwrite that work -- the dialog must say
    exactly which files, not "some files changed".
    """

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#change-revert-confirm"

    def __init__(self, summary: str, edited_since: list[str]) -> None:
        super().__init__()
        self._summary = summary
        self._edited_since = edited_since

    def compose(self) -> ComposeResult:
        """Summary + the named-files warning + Revert/Cancel."""
        with Vertical(id="change-revert-confirm"):
            yield Static(self._summary, markup=False)
            if self._edited_since:
                names = "\n".join(f"  • {p}" for p in self._edited_since)
                yield Static(
                    "⚠ Changed since this turn (reverting overwrites "
                    f"that work):\n{names}",
                    id="change-revert-edited-warning",
                    markup=False,
                )
            with Horizontal(id="change-revert-buttons"):
                yield Button("Revert", id="change-revert-yes", variant="error")
                yield Button("Cancel", id="change-revert-no")

    @on(Button.Pressed, "#change-revert-yes")
    def _confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#change-revert-no")
    async def _cancel_button(self) -> None:
        await self.request_safe_cancel(source="button")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(False)


class ChangeGitCommitModal(SafeModalDismissMixin, ModalScreen["dict | None"]):
    """Confirm a file-picked commit into the user's REAL repository (spec §5).

    Same modal discipline as :class:`ChangeRevertConfirmModal` (the
    ``SafeModalDismissMixin``, escape-cancels, one content container), with
    the commit-specific payload: a checklist of the files the FRESH status
    read found (all pre-checked -- unchecking excludes), a required
    message, an optional "create branch first" name, the branch the commit
    will land on, and WARNINGS that never block (detached HEAD, committing
    straight to main/master).

    Dismisses with ``{"message", "new_branch", "files", "root"}`` on
    confirm, or ``None`` on any cancellation (escape, the Cancel button, a
    backdrop click) -- the mixin's default cancel result.
    """

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#change-git-commit"

    def __init__(
        self,
        *,
        root: str,
        info: "GitWorkspaceInfo",
        entries: "Sequence[tuple[str, tuple[str, ...]]]",
    ) -> None:
        """Args are stored; the checklist is built in ``compose``.

        Args:
            root: The resolved repository root this commit acts on. Named
                in full in the dialog (spec §6: the modal always says which
                repository it will touch).
            info: The FRESH detection this modal was opened over -- the
                branch line and the warnings both read from it, so they can
                never describe a state the preflight did not just observe.
            entries: ``(label, pathspec)`` pairs from
                :func:`_commit_entries`. A rename's pathspec carries BOTH
                paths under one checkbox.
        """
        super().__init__()
        self._root = root
        self._info = info
        self._entries = list(entries)

    def compose(self) -> ComposeResult:
        """Target + warnings + checklist + message/branch inputs + buttons.

        Returns:
            The modal's widget tree.
        """
        with Vertical(id="change-git-commit"):
            # Rich `Text`, never a plain string: a repository path or a
            # branch name can contain brackets, and `Static`'s markup would
            # eat them as a tag (this screen's own leaf-label scar).
            yield Static(
                Text(f"Commit in {self._root}  ·  {_head_label(self._info)}"),
                id="change-git-commit-target",
                markup=False,
            )
            warnings = _commit_warnings(self._info)
            if warnings:
                yield Static(
                    Text("\n".join(warnings)),
                    id="change-git-commit-warnings",
                    markup=False,
                )
            with VerticalScroll(id="change-git-commit-files"):
                for label, paths in self._entries:
                    # `Checkbox` labels are markup-PARSED exactly like
                    # `Button`'s: `Checkbox("a[b].txt")` renders "a.txt"
                    # with a bold span (verified). A `Text` is verbatim.
                    box = Checkbox(
                        Text(label), value=True, classes="change-git-commit-file"
                    )
                    box.file_paths = paths
                    yield box
            yield Input(
                placeholder="Commit message (required)",
                id="change-git-commit-message",
            )
            yield Input(
                placeholder="Create branch first (optional)",
                id="change-git-commit-branch",
            )
            yield Static("", id="change-git-commit-error", markup=False)
            with Horizontal(id="change-git-commit-buttons"):
                yield Button(
                    "Commit", id="change-git-commit-yes", variant="primary"
                )
                yield Button("Cancel", id="change-git-commit-no")

    def on_mount(self) -> None:
        """Focus the required field, keeping the mixin's own mount work.

        ``super().on_mount()`` is mandatory, not politeness: Textual
        resolves ``on_mount`` by ordinary attribute lookup, so defining one
        here SHADOWS :class:`SafeModalDismissMixin`'s -- which is what
        records the mount generation and the opener's focus for the
        restore-on-dismiss contract.
        """
        super().on_mount()
        try:
            self.query_one("#change-git-commit-message", Input).focus()
        except Exception:  # noqa: BLE001 -- focus is never load-bearing
            logger.opt(exception=True).warning(
                "change_review: commit modal focus failed"
            )

    def _show_error(self, message: str) -> None:
        """Render an inline validation error without dismissing.

        Args:
            message: Why the submit was refused.
        """
        try:
            self.query_one("#change-git-commit-error", Static).update(
                Text(message)
            )
        except Exception:  # noqa: BLE001 -- never raise out of a handler
            logger.opt(exception=True).warning(
                "change_review: commit modal error render failed"
            )

    def _submit(self) -> None:
        """Validate the form and dismiss with the commit request."""
        try:
            message = self.query_one(
                "#change-git-commit-message", Input
            ).value.strip()
            if not message:
                self._show_error("a commit message is required")
                return
            selected = [box for box in self.query(Checkbox) if box.value]
            files = [
                path
                for box in selected
                for path in getattr(box, "file_paths", ())
            ]
            if not files:
                self._show_error("select at least one file to commit")
                return
            branch = self.query_one(
                "#change-git-commit-branch", Input
            ).value.strip()
            self.dismiss(
                {
                    "message": message,
                    "new_branch": branch or None,
                    "files": files,
                    "root": self._root,
                    # Review fix round 1: the success copy counts what the
                    # USER checked, not pathspec elements -- one row can
                    # carry two paths (a rename), and "Committed 2 file(s)"
                    # for a single checked rename is a small lie about a
                    # write to their repository. Additive key; the four the
                    # plan pinned are unchanged.
                    "file_count": len(selected),
                }
            )
        except Exception:  # noqa: BLE001 -- never raise out of a handler
            logger.opt(exception=True).warning(
                "change_review: commit modal submit failed"
            )

    @on(Button.Pressed, "#change-git-commit-yes")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#change-git-commit-message")
    def _submit_from_message(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    @on(Button.Pressed, "#change-git-commit-no")
    async def _cancel_button(self) -> None:
        await self.request_safe_cancel(source="button")


class ChangeGitPushModal(SafeModalDismissMixin, ModalScreen["dict | None"]):
    """Confirm WHICH repository (and remote) a push -- or a PR link -- targets.

    Spec §6's multi-root rule gives push and PR the same shape, so they
    share one dialog rather than two near-identical ones: it always NAMES
    the root it will act on, carries a root ``Select`` when more than one
    detected repository qualifies and no leaf is focused (the clean-tree
    case commit can never reach), and -- for a push with no upstream and
    more than one remote -- a remote ``Select`` instead of a guess.

    The ``action`` flag is what the two uses differ by: ``"push"`` confirms
    a write to a remote, ``"pr"`` only disambiguates which repository's
    compare page to open (a single-repository PR skips this dialog
    entirely, since opening a page is not a mutation).

    Dismisses with ``{"action", "root", "remote", "info"}`` on confirm, or
    ``None`` on any cancellation (escape, Cancel, a backdrop click) -- the
    mixin's default cancel result.
    """

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#change-git-push"

    def __init__(
        self,
        *,
        action: str,
        entries: "Sequence[tuple[str, GitWorkspaceInfo]]",
    ) -> None:
        """Args are stored; the selectors are built in ``compose``.

        Args:
            action: ``"push"`` or ``"pr"``.
            entries: ``(resolved root, FRESH detection)`` pairs -- already
                filtered to the roots this action can actually use, so the
                dialog never offers a target that would fail. Must be
                non-empty.

        Raises:
            ValueError: ``entries`` is empty (a dialog with no target is a
                programming error, not a user-facing state).
        """
        super().__init__()
        if not entries:
            raise ValueError("ChangeGitPushModal needs at least one target root")
        self._action = action
        self._entries = list(entries)
        self._infos = {root: info for root, info in self._entries}
        self._selected_root = self._entries[0][0]

    # -- copy ---------------------------------------------------------------

    def _title(self, root: str) -> str:
        """The dialog's first line -- it always names the repository.

        Args:
            root: The root currently targeted.

        Returns:
            Title copy for :data:`root`.
        """
        if self._action == "pr":
            return f"Open a pull request for {root}"
        return f"Push from {root}"

    def _remote_names(self, root: str) -> list[str]:
        """The remote NAMES configured at ``root``.

        Args:
            root: The root currently targeted.

        Returns:
            Remote names in ``git remote -v`` order.
        """
        return [name for name, _url in self._infos[root].remotes]

    def _needs_remote_choice(self, root: str) -> bool:
        """Whether the user must pick a remote for ``root``.

        Args:
            root: The root currently targeted.

        Returns:
            True only for a PUSH of a branch with no upstream on a
            repository with more than one remote -- an upstream already
            names its remote (spec §6: ``push <upstream_remote>``), and a
            PR never pushes anything.
        """
        if self._action != "push":
            return False
        info = self._infos[root]
        return info.upstream is None and len(info.remotes) > 1

    def _summary_line(self, root: str) -> str:
        """The branch/upstream/remote state this action will act on.

        Args:
            root: The root currently targeted.

        Returns:
            One line of copy naming the branch and what happens to it.
        """
        info = self._infos[root]
        head = _head_label(info)
        if self._action == "pr":
            return f"{head} → {info.upstream}"
        if info.upstream is not None:
            target = info.upstream_remote or "the upstream's remote"
            return f"{head} ↑{info.ahead} → {info.upstream}  ·  pushes to {target}"
        if self._needs_remote_choice(root):
            return f"{head}  ·  no upstream — choose a remote to track"
        names = self._remote_names(root)
        target = names[0] if names else "the remote"
        return f"{head}  ·  no upstream — will push -u to {target}"

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Target + (root Select) + summary + (remote Select) + buttons.

        Returns:
            The modal's widget tree.
        """
        root = self._selected_root
        with Vertical(id="change-git-push"):
            # Rich `Text`, never a plain string: a repository path or a
            # branch name can contain brackets, and `Static`'s markup would
            # eat them as a tag (this screen's own leaf-label scar).
            yield Static(
                Text(self._title(root)), id="change-git-push-target", markup=False
            )
            if len(self._entries) > 1:
                from pathlib import Path as _P

                yield Select(
                    [(f"{_P(r).name} — {r}", r) for r, _info in self._entries],
                    id="change-git-push-root",
                    allow_blank=False,
                    value=root,
                )
            yield Static(
                Text(self._summary_line(root)),
                id="change-git-push-summary",
                markup=False,
            )
            names = self._remote_names(root)
            remote_select = Select(
                [(name, name) for name in names],
                id="change-git-push-remote",
                # `allow_blank=True` on purpose: `set_options` blanks the
                # value on every root switch, and a non-blankable Select
                # raises on exactly that.
                allow_blank=True,
                value=names[0] if names else Select.BLANK,
            )
            remote_select.display = self._needs_remote_choice(root)
            yield remote_select
            with Horizontal(id="change-git-push-buttons"):
                yield Button(
                    "Open PR" if self._action == "pr" else "Push",
                    id="change-git-push-yes",
                    variant="primary",
                )
                yield Button("Cancel", id="change-git-push-no")

    def on_mount(self) -> None:
        """Focus the confirm button, keeping the mixin's own mount work.

        ``super().on_mount()`` is mandatory, not politeness: Textual
        resolves ``on_mount`` by ordinary attribute lookup, so defining one
        here SHADOWS :class:`SafeModalDismissMixin`'s -- which is what
        records the mount generation and the opener's focus for the
        restore-on-dismiss contract.
        """
        super().on_mount()
        try:
            self.query_one("#change-git-push-yes", Button).focus()
        except Exception:  # noqa: BLE001 -- focus is never load-bearing
            logger.opt(exception=True).warning(
                "change_review: push modal focus failed"
            )

    @on(Select.Changed, "#change-git-push-root")
    def _on_root_changed(self, event: Select.Changed) -> None:
        """Re-render the dialog for the newly chosen repository.

        Scoped to the ROOT selector by id: the remote selector's own
        ``set_options`` below posts a ``Select.Changed`` too, and an
        unscoped handler would re-enter this on it.

        Args:
            event: The selector's change event.
        """
        event.stop()
        value = event.value
        if not isinstance(value, str) or value not in self._infos:
            return
        self._apply_root(value)

    def _apply_root(self, root: str) -> None:
        """Point the dialog's copy and remote choice at ``root``.

        Args:
            root: The newly selected repository root.
        """
        try:
            self._selected_root = root
            self.query_one("#change-git-push-target", Static).update(
                Text(self._title(root))
            )
            self.query_one("#change-git-push-summary", Static).update(
                Text(self._summary_line(root))
            )
            names = self._remote_names(root)
            remote_select = self.query_one("#change-git-push-remote", Select)
            remote_select.set_options([(name, name) for name in names])
            remote_select.value = names[0] if names else Select.BLANK
            remote_select.display = self._needs_remote_choice(root)
        except Exception:  # noqa: BLE001 -- never raise out of a handler
            logger.opt(exception=True).warning(
                "change_review: push modal root switch failed"
            )

    def _resolve_remote(self) -> "str | None":
        """The remote to hand the engine, or ``None`` to let it derive one.

        Returns:
            ``None`` when the branch has an upstream -- spec §6 pushes to
            ``info.upstream_remote``, and the engine reads that from
            detection's ``%(upstream:remotename)`` rather than by splitting
            the upstream string (remote names may contain ``"/"``).
            Otherwise the chosen remote, or the sole one.
        """
        info = self._infos[self._selected_root]
        if info.upstream is not None:
            return None
        names = self._remote_names(self._selected_root)
        if self._needs_remote_choice(self._selected_root):
            value = self.query_one("#change-git-push-remote", Select).value
            if isinstance(value, str) and value in names:
                return value
        return names[0] if names else None

    def _submit(self) -> None:
        """Dismiss with the confirmed target."""
        try:
            root = self._selected_root
            self.dismiss(
                {
                    "action": self._action,
                    "root": root,
                    "remote": self._resolve_remote(),
                    # The FRESH detection this dialog was built over, so the
                    # worker pushes what the user was just shown rather than
                    # re-deriving it from the screen's never-pruned cache.
                    "info": self._infos[root],
                }
            )
        except Exception:  # noqa: BLE001 -- never raise out of a handler
            logger.opt(exception=True).warning(
                "change_review: push modal submit failed"
            )

    @on(Button.Pressed, "#change-git-push-yes")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Button.Pressed, "#change-git-push-no")
    async def _cancel_button(self) -> None:
        await self.request_safe_cancel(source="button")
