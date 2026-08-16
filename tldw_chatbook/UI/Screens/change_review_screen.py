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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from tldw_chatbook.Workspaces.change_revert import (
        RevertOutcome,
        RevertPreflight,
    )

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Select, Static, Tree

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
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


class ChangeReviewScreen(Screen):
    """Changed-file tree + windowed diff viewer for one conversation."""

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back"),
        Binding("j", "next_file", "Next file"),
        Binding("k", "previous_file", "Previous file"),
        Binding("enter", "focus_diff", "View diff", show=False),
        Binding("u", "revert_file", "Revert file"),
        Binding("U", "undo_all", "Undo all", show=False),
    ]

    def __init__(
        self,
        provider: AgentRunsChangeReviewProvider,
        initial_run_id: str | None = None,
    ) -> None:
        """Args are stored; all loading happens in ``on_mount``.

        Args:
            provider: The conversation's turn/diff data source.
            initial_run_id: Turn to open on, or ``None`` for the latest.
                Constructor state rather than a post-push ``select_turn``
                call: the opener's ``call_after_refresh`` fired before this
                screen had composed (NoMatches on the Select) -- the test
                that pinned the opener caught it.
        """
        super().__init__()
        self._provider = provider
        self._initial_run_id = initial_run_id
        self._turns: list[ReviewTurn] = []
        self._active_turn: ReviewTurn | None = None
        #: Flattened (row, ChangedFile) leaves in tree order, for j/k.
        self._leaves: list[tuple[dict, ChangedFile]] = []
        self._focused_leaf: int = -1

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
            yield Static(
                "",
                id="change-review-banner",
                classes="change-review-banner",
                markup=False,
            )
            with Horizontal(id="change-review-body"):
                yield Tree("Changes", id="change-review-tree")
                with VerticalScroll(id="change-review-diff"):
                    yield Static(
                        "",
                        id="change-review-diff-content",
                        classes="change-review-diff-body",
                        markup=False,
                    )
            yield Static(
                "j/k files · Enter diff · Esc back",
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
        select.set_options(
            (turn.label, turn.run_id) for turn in self._turns
        )
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

    @on(Select.Changed, "#change-review-turn-select")
    def _on_turn_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, str) and event.value:
            for turn in self._turns:
                if turn.run_id == event.value:
                    self._load_turn(turn)
                    return

    def _load_turn(self, turn: ReviewTurn) -> None:
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
                        row, change, multi_root, badge=_badged(row, change)
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
                        row, change, multi_root, badge=_badged(row, change)
                    ),
                    data=len(self._leaves),
                )
                self._leaves.append((row, change))

        banner = self.query_one("#change-review-banner", Static)
        banner.update("\n".join(banners))
        banner.display = bool(banners)

        totals = self.query_one("#change-review-totals", Static)
        adds = sum(int(r["adds"] or 0) for r in turn.rows)
        dels = sum(int(r["dels"] or 0) for r in turn.rows)
        totals.update(f"{len(self._leaves)} files  +{adds} −{dels}")

        if self._leaves:
            self._focus_leaf(0)
        else:
            self._show_empty("No file changes in this turn.")

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

    def select_file(self, path: str) -> None:
        """Focus the first leaf whose change path matches ``path``.

        Args:
            path: Root-relative path from the active turn.
        """
        for index, (_row, change) in enumerate(self._leaves):
            if change.path == path:
                self._focus_leaf(index)
                return

    def _focus_leaf(self, index: int) -> None:
        if not self._leaves:
            return
        self._focused_leaf = max(0, min(index, len(self._leaves) - 1))
        row, change = self._leaves[self._focused_leaf]
        self._render_diff(row, change)

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

    def action_revert_file(self) -> None:
        """Revert the focused file (confirmed)."""
        if not self._leaves or self._focused_leaf < 0:
            return
        row, change = self._leaves[self._focused_leaf]
        self._confirm_and_revert(row, [change.path], f"Revert {change.path}?")

    def action_undo_all(self) -> None:
        """Revert every file in the active turn, per root (confirmed)."""
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
            diff = self._provider.diff_text(row, change.path)
        except ChangeTrackingError as exc:
            content.update(Text(f"diff unavailable: {exc}"))
            return
        cap = self._provider.diff_display_max_lines
        lines = diff.splitlines()
        hidden = max(0, len(lines) - cap)
        text = Text()
        for line in lines[:cap]:
            # Plain-string appends with explicit styles: content is DATA;
            # nothing here is ever markup-parsed.
            if line.startswith("+") and not line.startswith("+++"):
                text.append(line, style="green")
            elif line.startswith("-") and not line.startswith("---"):
                text.append(line, style="red")
            elif line.startswith("@@"):
                text.append(line, style="cyan")
            else:
                text.append(line)
            text.append("\n")
        if hidden:
            text.append(
                f"… diff truncated — {hidden} more lines", style="yellow"
            )
        content.update(text)


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
