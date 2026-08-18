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
from textual.events import Key
from textual.geometry import Region
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Select, Static, Tree

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
    ]

    def __init__(
        self,
        provider: AgentRunsChangeReviewProvider,
        initial_run_id: str | None = None,
        initial_path: str | None = None,
        initial_snapshot_id: int | None = None,
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
        """
        super().__init__()
        self._provider = provider
        self._initial_run_id = initial_run_id
        self._initial_path = initial_path
        self._initial_snapshot_id = initial_snapshot_id
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
                "j/k files · Enter diff · c comment line · C comment file "
                "· Esc back",
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
        """`C`: open a whole-file comment on the focused leaf."""
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
            if not self._leaves or self._focused_leaf < 0:
                return
            row, change = self._leaves[self._focused_leaf]
            if kind == "diff_line":
                if change.binary:
                    return
                try:
                    self._provider.diff_text(row, change.path)
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
            run_id = self._active_turn.run_id
            snapshot_id = row.get("id")

            if kind == "diff_line":
                cursor_line = getattr(note_input, "cursor_line", None)
                if cursor_line is None:
                    return
                try:
                    diff_text = self._provider.diff_text(row, change.path)
                except ChangeTrackingError:
                    self.notify(
                        "Diff unavailable — comment not saved",
                        severity="warning",
                    )
                    return
                lines = diff_text.splitlines()
                if not (0 <= cursor_line < len(lines)):
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
