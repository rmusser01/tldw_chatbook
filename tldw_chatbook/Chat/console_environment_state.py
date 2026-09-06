"""Pure display state for the Console Environment panel (Inspect rail).

No I/O here: gatherers live in ``Workspaces/environment_status.py`` and
projections consume only these frozen dataclasses. Follows the
``console_display_state.py`` convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from rich.cells import cell_len

from tldw_chatbook.Chat import console_rail_state
from tldw_chatbook.Workspaces.change_tracking import ChangedFile

# Preserve callers of the original display-state exports without making the
# lightweight rail identifiers depend on this first-open projection module.
ENVIRONMENT_SECTION_ID = console_rail_state.ENVIRONMENT_SECTION_ID
TASKS_SECTION_ID = console_rail_state.TASKS_SECTION_ID


class EnvSourceAvailability(str, Enum):
    """Why a tier's data is (or is not) there.

    ``NOT_APPLICABLE`` used to carry two incompatible meanings -- "we
    looked, and this is not a repository" AND "nobody has looked yet" --
    and the projection rendered both as the flat assertion "No git
    workspace". TASK-31660 splits the second meaning out (``PENDING``) and
    adds the third state the panel could not express at all (``UNBOUND``:
    the conversation's workspace binds no folder, so there is nothing to
    look at). ``NOT_APPLICABLE``'s meaning and rendering are unchanged.
    """

    OK = "ok"
    #: Checked; the bound root is genuinely not a git repository.
    NOT_APPLICABLE = "not_applicable"
    MISSING_TOOL = "missing_tool"
    ERROR = "error"
    #: No folder is bound to this conversation's workspace at all. A
    #: negative about the BINDING, never a negative about the contents.
    UNBOUND = "unbound"
    #: No gatherer has answered yet. Renders as progress, never as a claim.
    PENDING = "pending"
    #: The workspace root could not be DETERMINED, and has stayed that way
    #: (TASK-31665 AC#11). Distinct from ``UNBOUND`` ("determined: nothing
    #: is bound") and from ``PENDING`` ("a gatherer is on its way"): here
    #: nothing is on its way, because the accessor chain itself has no
    #: answer -- no chat controller, or no active session. Landed only
    #: after the condition PERSISTS, so a transient blip still leaves the
    #: previous paint alone.
    UNKNOWN = "unknown"


class ExecTargetKind(str, Enum):
    LOCAL = "local"
    REMOTE_TLDW_SERVER = "remote_tldw_server"


@dataclass(frozen=True)
class GitEnvState:
    availability: EnvSourceAvailability = EnvSourceAvailability.PENDING
    root: str = ""
    branch: str | None = None
    detached: bool = False
    unborn: bool = False
    head_short: str = ""
    upstream: str | None = None
    ahead: int = 0
    behind: int = 0
    adds: int = 0
    dels: int = 0
    files: tuple[ChangedFile, ...] = ()
    worktree_name: str | None = None
    stale: bool = False

    @property
    def dirty(self) -> bool:
        return bool(self.files)


@dataclass(frozen=True)
class PrCheck:
    name: str
    conclusion: str  # "success" | "failure" | "pending" (normalized)
    details_url: str = ""


@dataclass(frozen=True)
class PrEnvState:
    availability: EnvSourceAvailability = EnvSourceAvailability.PENDING
    number: int = 0
    title: str = ""
    state: str = ""  # "OPEN" | "MERGED" | "CLOSED"
    is_draft: bool = False
    url: str = ""
    adds: int = 0
    dels: int = 0
    merged_at: datetime | None = None
    checks: tuple[PrCheck, ...] = ()
    stale: bool = False

    @property
    def failing_checks(self) -> tuple[PrCheck, ...]:
        return tuple(c for c in self.checks if c.conclusion == "failure")

    @property
    def pending_checks(self) -> tuple[PrCheck, ...]:
        return tuple(c for c in self.checks if c.conclusion == "pending")

    @property
    def passing_count(self) -> int:
        return sum(1 for c in self.checks if c.conclusion == "success")


@dataclass(frozen=True)
class BacklogTaskEntry:
    task_id: str
    title: str
    status: str  # "To Do" | "In Progress" | "Done" | other verbatim


@dataclass(frozen=True)
class BranchTaskState:
    task_id: str
    title: str
    status: str
    ac_done: int = 0
    ac_total: int = 0
    path: str = ""


@dataclass(frozen=True)
class TasksEnvState:
    availability: EnvSourceAvailability = EnvSourceAvailability.PENDING
    branch_task: BranchTaskState | None = None
    in_progress: int = 0
    todo: int = 0
    entries: tuple[BacklogTaskEntry, ...] = ()
    scanning: bool = False


@dataclass(frozen=True)
class ExecTargetState:
    kind: ExecTargetKind = ExecTargetKind.LOCAL


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """The panel's whole state.

    Every tier defaults to ``PENDING``: a freshly constructed snapshot is
    what the panel shows before ANY gatherer has answered, and the honest
    description of that moment is "checking", not "there is no git
    workspace here" (TASK-31660).
    """

    git: GitEnvState = field(default_factory=GitEnvState)
    target: ExecTargetState = field(default_factory=ExecTargetState)
    pr: PrEnvState = field(default_factory=PrEnvState)
    tasks: TasksEnvState = field(default_factory=TasksEnvState)


def unbound_snapshot(target: ExecTargetState | None = None) -> EnvironmentSnapshot:
    """Build the snapshot for "no folder is bound to this workspace".

    One factory so the three tiers can never disagree about it, and so the
    controller cannot accidentally keep the PREVIOUS root's ``pr``/``tasks``
    while marking only ``git`` unbound -- which is exactly the shape of the
    reported P0 (another repository's PR and "Commit or push · N files"
    left painted after a workspace switch).

    Args:
        target: Exec target to preserve; the execution destination is a
            property of the session, not of the workspace binding, so it
            survives an unbind. Defaults to a fresh ``ExecTargetState``.

    Returns:
        A snapshot whose git, pr, and tasks tiers are all ``UNBOUND`` and
        carry no data.
    """
    return EnvironmentSnapshot(
        git=GitEnvState(availability=EnvSourceAvailability.UNBOUND),
        target=target if target is not None else ExecTargetState(),
        pr=PrEnvState(availability=EnvSourceAvailability.UNBOUND),
        tasks=TasksEnvState(availability=EnvSourceAvailability.UNBOUND),
    )


def unknown_snapshot(target: ExecTargetState | None = None) -> EnvironmentSnapshot:
    """Build the snapshot for "the workspace root could not be determined".

    TASK-31665 AC#11. Mirrors ``unbound_snapshot`` exactly in shape -- all
    three tiers carry the SAME availability and no data, so no tier can be
    left describing a root nobody could name. Kept a separate factory
    rather than a parameter so a caller cannot land "unknown" for one tier
    and "unbound" for another.

    Args:
        target: Exec target to preserve; the execution destination is a
            session property, not a workspace one.

    Returns:
        A snapshot whose git, pr, and tasks tiers are all ``UNKNOWN``.
    """
    return EnvironmentSnapshot(
        git=GitEnvState(availability=EnvSourceAvailability.UNKNOWN),
        target=target if target is not None else ExecTargetState(),
        pr=PrEnvState(availability=EnvSourceAvailability.UNKNOWN),
        tasks=TasksEnvState(availability=EnvSourceAvailability.UNKNOWN),
    )


_BRANCH_TASK_RE = re.compile(r"task-(\d+(?:\.\d+)*)")


def branch_task_id(branch: str | None) -> str | None:
    """Extract a backlog task id (subtasks included) from a branch name."""
    if not branch:
        return None
    match = _BRANCH_TASK_RE.search(branch)
    return match.group(1) if match else None


def compact_count(n: int) -> str:
    """Humanize a line count: exact with separators below 100k, compact above."""
    if n < 100_000:
        return f"{n:,}"
    if n < 1_000_000:
        return f"{round(n / 1_000)}k"
    return f"{n / 1_000_000:.1f}M"


def signed_change_counts(adds: int, dels: int) -> str:
    return f"+{compact_count(adds)} −{compact_count(dels)}"


def relative_age(then: datetime | None, now: datetime) -> str:
    """Coarse '5m ago' / '3h ago' / '6d ago' bucket; '' for None."""
    if then is None:
        return ""
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


from tldw_chatbook.Widgets.Console.console_inspector_section import (
    RAIL_CONTENT_WIDTH_MIN,
    ROW_INDENT_COLUMNS,
    SECTION_TOGGLE_WIDTH,
    SINGLE_LINE_ROW_BUDGET,
    ConsoleInspectorSectionState,
    InspectorSectionRow,
)

#: Columns an ``indent=1`` expansion child gets for its own text
#: (TASK-31665 AC#3). Every projection below that ellipsizes an indented
#: row must budget against THIS, not ``SINGLE_LINE_ROW_BUDGET`` -- an
#: indent the fitter does not know about is exactly the silent truncation
#: `_with_expand_marker`'s docstring already warns about.
_CHILD_ROW_BUDGET = SINGLE_LINE_ROW_BUDGET - ROW_INDENT_COLUMNS

ENV_ROW_CHANGES = "env-changes"
ENV_ROW_ERROR = "env-error"
ENV_ROW_EMPTY = "env-empty"
ENV_ROW_PENDING = "env-pending"
ENV_ROW_UNBOUND = "env-unbound"
ENV_ROW_UNBOUND_NOTE = "env-unbound-note"
ENV_ROW_UNKNOWN = "env-unknown"
ENV_ROW_UNKNOWN_NOTE = "env-unknown-note"
ENV_ROW_LOCAL = "env-local"
ENV_ROW_BRANCH = "env-branch"
ENV_ROW_COMMIT_PUSH = "env-commit-push"
ENV_ROW_PR = "env-pr"
ENV_ROW_CHECKS = "env-checks"
ENV_ROW_PR_OPEN = "env-pr-open"
ENV_ROW_PR_ADD = "env-pr-add"
ENV_ROW_CHECKS_FIX = "env-checks-fix"
ENV_FILE_ROW_PREFIX = "env-file-"

EXPANDABLE_ENV_ROWS = frozenset(
    {ENV_ROW_CHANGES, ENV_ROW_LOCAL, ENV_ROW_BRANCH, ENV_ROW_PR, ENV_ROW_CHECKS}
)

_MAX_FILE_ROWS = 12

# Column budgets for the two section header summaries.
#
# The section header is `title (1fr) + summary (auto) + toggle (3)` on ONE
# line (`console_inspector_section.py::compose`). The summary Static is
# `width: auto`, so an unbudgeted summary takes whatever it wants and the
# 1fr title is starved to nothing -- any branch name over ~33 characters
# (routine here: `feat/console-inspector-environment`) squeezed the title
# AND the collapse chevron off the header entirely.
#
# The budget is DERIVED from the rail's measured content width rather than
# chosen (TASK-31662 / TASK-31629 #12). The earlier "~34 columns at every
# terminal size" reading was wrong in the direction that matters: probing
# the real Console on 2026-09-05 puts the section box at 30 columns at
# 80x24 and 36 at 200x50, so 34 was a width no supported terminal
# produces, and an 18-column summary left "Environment" painting as
# "Environm…" at the small end. Each budget below is
# `RAIL_CONTENT_WIDTH_MIN - len(title) - toggle - 1` (one column keeping
# the title and summary apart), so the title always paints in full.
#
# Truncation lives HERE, in the pure projection, not in the widget -- this
# arc's rule, so it is testable without a running app.
ENV_SUMMARY_BUDGET = (
    RAIL_CONTENT_WIDTH_MIN - len("Environment") - SECTION_TOGGLE_WIDTH - 1
)
#: Same derivation for the Tasks section, whose title is four columns
#: shorter (TASK-31629 #13: "task-31450 · In Progress" is 24 columns and
#: left three for the 5-column "Tasks" title).
TASKS_SUMMARY_BUDGET = (
    RAIL_CONTENT_WIDTH_MIN - len("Tasks") - SECTION_TOGGLE_WIDTH - 1
)

# TASK-31660 copy, REWORDED by TASK-31664 AC#5: `git.availability is
# UNBOUND` does not mean ONLY "no folder is bound" -- the identical
# `workspace_roots == ()` also lands when Change Review's consent is not
# ENABLED for an otherwise-bound folder (the common default), when the
# consent service is absent or raises, or when every bound root got
# skipped. "No folder is bound…" was a confident, specific, and WRONG
# claim in the first three of those.
#
# Investigated whether the true cause can be told apart cheaply at the
# seam this panel reads (`ChatScreen._console_environment_root` ->
# `resolve_turn_execution_context(...).workspace_roots`): it cannot, for
# the common cases. `ChangeReviewConsentService.admit_turn` returns the
# SAME empty `ChangeReviewAdmission()` -- no ready roots, no
# `skipped_roots` entries -- whether consent is off, the capability is
# unavailable, or nothing is bound at all; the one sub-case that DOES
# leave a distinguishing trace (`skipped_roots` non-empty: a bound root
# that is still preparing/failed) is the least common of the four and is
# not currently plumbed up to this projection. Shipping the AC-sanctioned
# cause-agnostic fallback: name the one thing that is ALWAYS true (changes
# are not tracked here) and point at both remediation steps -- bind AND
# enable -- rather than asserting a cause that is wrong most of the time.
ENV_PENDING_TEXT = "Checking workspace…"
ENV_UNBOUND_TEXT = "Changes aren't tracked for this workspace."
ENV_UNBOUND_NOTE_TEXT = (
    "Bind a folder and enable Change Review in Settings ▸ Workspaces — "
    "this is not a report that nothing changed."
)

# TASK-31665 AC#11. `ENV_PENDING_TEXT` promises motion ("Checking…"), and
# a root the accessor chain cannot determine has none: nothing is
# dispatched, so nothing will ever land, and the panel sat on that promise
# with an inert Refresh for the life of the screen. This copy stops
# promising and says what is actually true. Deliberately NOT the UNBOUND
# copy: "changes aren't tracked for this workspace" asserts something about
# a workspace that has not been identified.
#
# Round-1 review (I2): the first cut read "No active chat session —
# workspace not determined", which NAMES a cause the panel cannot know.
# `UNKNOWN_ROOT` has (at least) two sources -- the chat controller not
# being built yet / having no active session, AND a swallowed exception
# inside the roots accessor (`review_selection.py`'s bare except), which
# happens with a perfectly live session. Naming the first was simply false
# in the second case, the exact dishonest-empty-state class TASK-31664 AC#5
# had just corrected for the UNBOUND copy. Say only the half that is always
# true, and phrase the remedy as OPTIONS rather than as a diagnosis.
ENV_UNKNOWN_TEXT = "Workspace not determined."
ENV_UNKNOWN_NOTE_TEXT = (
    "Open a chat in a Workspace, or press Refresh to retry."
)


def _git_status_class(stale: bool) -> str:
    return "blocked" if stale else ""


# TASK-31664: trailing-marker convention. Enter on a rail row used to have
# FIVE outcome classes that read identically -- expand-in-place, navigate
# to another surface (in-app or the OS browser), insert text into the
# composer draft, and no-op. `▸`/`▾` reuse the section header's own
# collapse/expand chevron vocabulary (`GLYPH_COLLAPSED`/`GLYPH_EXPANDED` in
# `console_inspector_section.py`), so a row's own expand affordance reads
# in the same language the section-level chevron already does. "…" mirrors
# Change Review's own `Commit…`/`Push…`/`Narrow…` precedent for "activating
# this opens something else": in-app navigation and leaving to the OS
# browser are the SAME promise from the row's point of view (this isn't
# staying here), so both share the one marker. "+ " marks an
# insert-into-composer action and shares nothing with the other two, so it
# can never be mistaken for navigation. Inert rows (file rows, task
# entries, "Local instance ✓", every expanded detail row) carry none of
# these -- their absence IS the fourth, inert state.
ENV_MARKER_EXPAND_COLLAPSED = "▸"
ENV_MARKER_EXPAND_OPEN = "▾"
ENV_MARKER_OPENS_SURFACE = "…"
ENV_MARKER_COMPOSER_INSERT = "+ "


def _with_expand_marker(
    label: str,
    row_id: str,
    expanded: frozenset[str],
    *,
    budget: int = SINGLE_LINE_ROW_BUDGET,
) -> str:
    """Append the row-level expand/collapse chevron, ellipsizing FIRST.

    The branch name is the only unbounded label among the expand-in-place
    rows; ellipsizing here and appending the marker AFTER means the
    terminal's own CSS `text-overflow: ellipsis` never gets a chance to
    swallow the marker along with an overflowing label -- a marker that
    can silently disappear on a long branch name is worse than none.

    Args:
        label: The row's own text, pre-marker.
        row_id: This row's stable id, checked against ``expanded``.
        expanded: Currently expanded row ids for this section.
        budget: Columns available to the row's own text.

    Returns:
        ``label`` (ellipsized if needed) plus a trailing space and the
        chevron matching this row's current expand state.
    """
    marker = ENV_MARKER_EXPAND_OPEN if row_id in expanded else ENV_MARKER_EXPAND_COLLAPSED
    room = budget - cell_len(marker) - 1  # M4: cells -- see `_ellipsize`
    if room <= 0:
        return marker
    return f"{_ellipsize(label, room)} {marker}"


def _with_surface_marker(label: str, *, budget: int = SINGLE_LINE_ROW_BUDGET) -> str:
    """Append the "opens another surface" marker (in-app nav or the browser).

    No separating space -- matches Change Review's own `Commit…`/`Push…`/
    `Narrow…` precedent, which butts the ellipsis directly against the verb
    (round-1 review: the space here didn't match that precedent).
    """
    room = budget - len(ENV_MARKER_OPENS_SURFACE)
    if room <= 0:
        return ENV_MARKER_OPENS_SURFACE
    return f"{_ellipsize(label, room)}{ENV_MARKER_OPENS_SURFACE}"


def _with_insert_marker(label: str) -> str:
    """Prefix the composer-insert marker onto a fixed, short row label."""
    return f"{ENV_MARKER_COMPOSER_INSERT}{label}"


def _with_stale_marker(secondary_text: str, stale: bool) -> str:
    """Give "stale" a text carrier alongside its color (TASK-31664 AC#4).

    Color alone left "stale" indistinguishable from an error: it painted
    in the identical hue ($ds-status-blocked aliases $ds-status-error).
    """
    if not stale:
        return secondary_text
    if not secondary_text:
        return "(stale)"
    return f"{secondary_text} (stale)"


def _head_within_cells(text: str, limit: int) -> str:
    """Longest head of ``text`` whose RENDERED width fits ``limit`` columns.

    TASK-31665 final review M4: the counterpart to measuring with
    ``cell_len`` -- a budget in columns cannot be spent by slicing on
    character INDEX. ``text[:limit]`` on a CJK path or an emoji-bearing
    branch name yields up to twice ``limit`` columns, which is the same
    over-run the ``len``-based measurement produced, just moved one step
    later. Stops before the first character that would not fit whole (a
    2-cell character is never split into a half column).
    """
    if limit <= 0:
        return ""
    if cell_len(text) <= limit:
        return text
    used = 0
    head: list[str] = []
    for char in text:
        width = cell_len(char)
        if used + width > limit:
            break
        head.append(char)
        used += width
    return "".join(head)


def _ellipsize(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` columns, marking the cut with a trailing "…".

    Head-anchored (keeps the start, drops the tail) because this repo's
    branch names lead with the identifying fragment -- ``feat/task-31450-…``
    -- so the head is what tells the branches apart.

    TASK-31665 final review M4: measured (and sliced) in terminal CELLS,
    not characters. ``row_fits_one_line`` decides a row's SHAPE with
    ``rich.cells.cell_len``; this function decides what that shape gets
    FILLED with, and the two disagreeing is what let a double-width label
    be called a fit and then overflow. Branch names and changed-file paths
    are user data, so the wide case is reachable.

    Args:
        text: Text to fit.
        limit: Maximum column count; ``<= 0`` yields ``""``.

    Returns:
        ``text`` unchanged when it already fits, else its head plus "…"
        (the ellipsis is one cell), together at most ``limit`` columns.
    """
    if limit <= 0:
        return ""
    if cell_len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return _head_within_cells(text, limit - 1) + "…"


def environment_summary(git: GitEnvState, *, budget: int = ENV_SUMMARY_BUDGET) -> str:
    """Build the Environment header summary, fitted to ``budget`` columns.

    The signed ± counts are the priority half -- they are the number the
    user is scanning for and they are already compacted
    (``compact_count``), so they are never truncated. Whatever the counts
    leave over is the branch fragment's budget; when that is too small to
    say anything (under two columns, i.e. not even one character plus the
    ellipsis) the branch is dropped and the counts stand alone.

    Args:
        git: The git tier state to describe.
        budget: Column budget for the whole summary.

    Returns:
        A summary string of at most ``budget`` columns (or exactly the
        counts, when the counts alone already exceed it).
    """
    counts = signed_change_counts(git.adds, git.dels)
    # M4: cells, not characters -- see `_ellipsize`. `counts` is ASCII
    # today, but measuring the two halves of one budget by two different
    # rulers is the bug, not the width of this particular half.
    room = budget - cell_len(counts) - 1  # -1 for the separating space
    if room < 2:
        return counts
    return f"{_ellipsize(_branch_primary(git), room)} {counts}"


def _branch_primary(git: GitEnvState) -> str:
    if git.detached:
        return f"detached @ {git.head_short or 'HEAD'}"
    if git.unborn:
        return f"{git.branch or '?'} (no commits yet)"
    return git.branch or "?"


def _branch_secondary(git: GitEnvState) -> str:
    parts: list[str] = []
    if git.ahead:
        parts.append(f"↑{git.ahead}")
    if git.behind:
        parts.append(f"↓{git.behind}")
    if git.worktree_name:
        parts.append(f"wt:{git.worktree_name}")
    return " ".join(parts)


def project_environment_section(
    snapshot: EnvironmentSnapshot,
    expanded: frozenset[str],
    *,
    now: datetime,
) -> ConsoleInspectorSectionState:
    git = snapshot.git
    if git.availability is EnvSourceAvailability.PENDING:
        # Nobody has looked yet. Every other branch below is an ANSWER, and
        # rendering this one as an answer is the cold-start defect: a git
        # worktree was told "No git workspace" for the ~20s until the first
        # gatherer landed. Counts, Commit-or-push, PR/checks and the Tasks
        # card are all suppressed by returning here -- there is nothing yet
        # to suppress that was ever measured against the current root.
        return ConsoleInspectorSectionState(
            rows=(InspectorSectionRow(
                row_id=ENV_ROW_PENDING, primary_text=ENV_PENDING_TEXT,
            ),),
            summary="",
        )
    if git.availability is EnvSourceAvailability.UNKNOWN:
        # TASK-31665 AC#11: a root that stayed undetermined. Says so, and
        # names the gesture that changes it, instead of sitting forever on
        # PENDING's "Checking workspace…" with nothing checking.
        return ConsoleInspectorSectionState(
            rows=(
                InspectorSectionRow(
                    row_id=ENV_ROW_UNKNOWN, primary_text=ENV_UNKNOWN_TEXT,
                ),
                InspectorSectionRow(
                    row_id=ENV_ROW_UNKNOWN_NOTE, primary_text=ENV_UNKNOWN_NOTE_TEXT,
                ),
            ),
            summary="",
        )
    if git.availability is EnvSourceAvailability.UNBOUND:
        # No folder is bound, so nothing about a repository can be asserted
        # -- including, emphatically, "clean". Returning before the counts
        # is what makes the panel drop the PREVIOUS root's branch, files,
        # and "Commit or push · N files" offer on a workspace switch.
        return ConsoleInspectorSectionState(
            rows=(
                InspectorSectionRow(
                    row_id=ENV_ROW_UNBOUND, primary_text=ENV_UNBOUND_TEXT,
                ),
                InspectorSectionRow(
                    row_id=ENV_ROW_UNBOUND_NOTE, primary_text=ENV_UNBOUND_NOTE_TEXT,
                ),
            ),
            summary="",
        )
    if git.availability is EnvSourceAvailability.ERROR:
        # ERROR is NOT "there is nothing here" -- it is "we could not look".
        # Rendering it as the NOT_APPLICABLE empty state told a user whose
        # git call timed out (or whose tier had backed off after 3 failures)
        # that their repository was not a git workspace, with no hint that
        # the Refresh slot would revive it.
        return ConsoleInspectorSectionState(
            rows=(InspectorSectionRow(
                row_id=ENV_ROW_ERROR,
                primary_text="Environment unavailable — Refresh to retry",
                status="blocked",
            ),),
            summary="",
        )
    if git.availability is not EnvSourceAvailability.OK:
        return ConsoleInspectorSectionState(
            rows=(InspectorSectionRow(row_id=ENV_ROW_EMPTY, primary_text="No git workspace"),),
            summary="",
        )
    status = _git_status_class(git.stale)
    rows: list[InspectorSectionRow] = []

    rows.append(InspectorSectionRow(
        row_id=ENV_ROW_CHANGES,
        primary_text=_with_expand_marker("Changes", ENV_ROW_CHANGES, expanded),
        secondary_text=_with_stale_marker(
            signed_change_counts(git.adds, git.dels), git.stale
        ),
        status=status, clickable=True,
    ))
    if ENV_ROW_CHANGES in expanded:
        for index, change in enumerate(git.files[:_MAX_FILE_ROWS]):
            rows.append(InspectorSectionRow(
                row_id=f"{ENV_FILE_ROW_PREFIX}{index}",
                primary_text=f"{change.status} {change.path}",
                secondary_text=signed_change_counts(change.adds, change.dels),
                indent=1,
            ))
        if len(git.files) > _MAX_FILE_ROWS:
            rows.append(InspectorSectionRow(
                row_id="env-file-more",
                primary_text=f"… {len(git.files) - _MAX_FILE_ROWS} more — Review opens all",
                indent=1,
            ))
        rows.append(InspectorSectionRow(
            row_id="env-changes-review",
            primary_text=_with_surface_marker(
                "Review in Change Review", budget=_CHILD_ROW_BUDGET
            ),
            clickable=True,
            indent=1,
        ))

    rows.append(InspectorSectionRow(
        row_id=ENV_ROW_LOCAL,
        primary_text=_with_expand_marker("Local", ENV_ROW_LOCAL, expanded),
        clickable=True,
    ))
    if ENV_ROW_LOCAL in expanded:
        rows.append(InspectorSectionRow(
            row_id="env-local-current", primary_text="Local instance ✓",
            indent=1,
        ))
        rows.append(InspectorSectionRow(
            row_id="env-local-remote",
            primary_text="Remote tldw_server — not configured",
            indent=1,
        ))

    rows.append(InspectorSectionRow(
        row_id=ENV_ROW_BRANCH,
        primary_text=_with_expand_marker(
            _branch_primary(git), ENV_ROW_BRANCH, expanded
        ),
        secondary_text=_with_stale_marker(_branch_secondary(git), git.stale),
        status=status, clickable=True,
    ))
    if ENV_ROW_BRANCH in expanded:
        rows.append(InspectorSectionRow(
            row_id="env-branch-detail",
            indent=1,
            primary_text=git.branch or _branch_primary(git),
            secondary_text=(
                f"upstream {git.upstream} (↑↓ vs last fetch)"
                if git.upstream else "no upstream"
            ),
        ))
        if git.worktree_name:
            rows.append(InspectorSectionRow(
                row_id="env-branch-worktree",
                primary_text=f"worktree {git.worktree_name}",
                secondary_text=git.root,
                indent=1,
            ))

    if git.dirty or git.ahead:
        # TASK-31664 AC#2: "Commit or push" performed navigation to Change
        # Review but omitted the "…" Change Review's own destination uses
        # (`Commit…`/`Push…`). Both variants navigate there, so both carry
        # the marker; the dirty variant's rename bakes it in right after
        # the verb phrase (matching this AC's own example), rather than at
        # the very end after "· N files" -- `_with_surface_marker` is not
        # used here because it would double the ellipsis on the dirty
        # label.
        if git.dirty:
            count = len(git.files)
            label = f"Review & commit… · {count} file" + ("s" if count != 1 else "")
        else:
            label = f"Push ↑{git.ahead}…"
        rows.append(InspectorSectionRow(
            row_id=ENV_ROW_COMMIT_PUSH, primary_text=label, clickable=True,
        ))

    pr = snapshot.pr
    if pr.availability is EnvSourceAvailability.OK and pr.number:
        state_label = "Draft" if (pr.is_draft and pr.state == "OPEN") else pr.state.capitalize()
        secondary = ""
        if pr.state == "MERGED" and pr.merged_at is not None:
            secondary = f"Merged {relative_age(pr.merged_at, now)}"
        rows.append(InspectorSectionRow(
            row_id=ENV_ROW_PR,
            primary_text=_with_expand_marker(
                f"PR #{pr.number} · {state_label}", ENV_ROW_PR, expanded
            ),
            secondary_text=_with_stale_marker(secondary, pr.stale),
            status="blocked" if pr.stale else "",
            clickable=True,
        ))
        if ENV_ROW_PR in expanded:
            rows.append(InspectorSectionRow(
                row_id="env-pr-title", primary_text=pr.title,
                secondary_text=signed_change_counts(pr.adds, pr.dels),
                indent=1,
            ))
            rows.append(InspectorSectionRow(
                row_id=ENV_ROW_PR_OPEN,
                primary_text=_with_surface_marker(
                    "Open in browser", budget=_CHILD_ROW_BUDGET
                ),
                clickable=True,
                indent=1,
            ))
            rows.append(InspectorSectionRow(
                row_id=ENV_ROW_PR_ADD,
                primary_text=_with_insert_marker("Add to chat"),
                clickable=True,
                indent=1,
            ))
        if pr.checks:
            failing = len(pr.failing_checks)
            pending = len(pr.pending_checks)
            if failing:
                checks_primary = f"{failing} failing check" + ("s" if failing != 1 else "")
                checks_status = "error"
            elif pending:
                checks_primary = f"{pending} pending check" + ("s" if pending != 1 else "")
                checks_status = "running"
            else:
                checks_primary = f"{pr.passing_count} checks passed"
                checks_status = "done"
            rows.append(InspectorSectionRow(
                row_id=ENV_ROW_CHECKS,
                primary_text=_with_expand_marker(
                    checks_primary, ENV_ROW_CHECKS, expanded
                ),
                secondary_text=(
                    f"{pr.passing_count} passed · {pending} pending" if failing else ""
                ),
                status=checks_status, clickable=True,
            ))
            if ENV_ROW_CHECKS in expanded:
                for index, check in enumerate(pr.failing_checks):
                    rows.append(InspectorSectionRow(
                        row_id=f"env-check-{index}", primary_text=check.name,
                        status="error",
                        indent=1,
                    ))
                if failing:
                    rows.append(InspectorSectionRow(
                        row_id=ENV_ROW_CHECKS_FIX,
                        primary_text=_with_insert_marker(
                            "Fix — add failure summary to chat"
                        ),
                        clickable=True,
                        indent=1,
                    ))

    return ConsoleInspectorSectionState(
        rows=tuple(rows), summary=environment_summary(git)
    )


def pr_summary_text(pr: PrEnvState) -> str:
    """Composer-insert payload for the PR 'Add to chat' action."""
    lines = [f"PR #{pr.number}: {pr.title} [{pr.state}]", pr.url]
    if pr.failing_checks:
        lines.append("Failing checks: " + ", ".join(c.name for c in pr.failing_checks))
    return "\n".join(line for line in lines if line)


def failing_checks_text(pr: PrEnvState) -> str:
    """Composer-insert payload for the failing-checks 'Fix' action."""
    lines = [f"CI is failing on PR #{pr.number} — please investigate and fix:"]
    for check in pr.failing_checks:
        suffix = f" — {check.details_url}" if check.details_url else ""
        lines.append(f"- {check.name}{suffix}")
    return "\n".join(lines)


TASKS_ROW_HEAD = "task-head"
TASKS_ROW_ADD = "task-add"
TASKS_ENTRY_ROW_PREFIX = "task-entry-"
MAX_TASK_LIST_ROWS = 30

_STATUS_ROW_CLASS = {"In Progress": "running", "Done": "done"}


def tasks_count_summary(
    in_progress: int, todo: int, *, budget: int = TASKS_SUMMARY_BUDGET
) -> str:
    """Tasks header summary for a branch with no task of its own.

    Keeps BOTH counts, in the compact form, on the COLLAPSED header --
    TASK-31665 AC#6 round-1 ruling. The first cut of this function adopted
    the backlog's canonical words here too ("3 in progress · 12 to do", 24
    columns against a 21-column budget at the narrowest rail) and therefore
    had to drop the to-do count. That mitigation was FALSE: the expansion it
    pointed at caps at ``MAX_TASK_LIST_ROWS`` (30) and a real backlog holds
    651 entries, so the ~586 to-do tasks it dropped were visible NOWHERE in
    Console.

    The ruling: AC#6's vocabulary unification is about ROWS, where the
    canonical "In Progress"/"To Do" stay. The critique's complaint was the
    same fact stated in two vocabularies ADJACENT -- this header and the
    duplicate counts row directly beneath it -- and TASK-31662 deleted that
    row, so the adjacency the complaint was about is already gone. A
    collapsed one-line summary may use compact forms; it is a teaser, not a
    statement of record.

    Args:
        in_progress: Number of "In Progress" tasks in the backlog.
        todo: Number of "To Do" tasks in the backlog.
        budget: Columns the header summary may take.

    Returns:
        ``"N doing · M todo"``, ellipsized only if a pathological count
        makes even that overflow (19 columns at four-digit counts, against
        a 21-column budget, so in practice never).
    """
    return _ellipsize(f"{in_progress} doing · {todo} todo", budget)


def project_tasks_section(
    snapshot: EnvironmentSnapshot,
    expanded: frozenset[str],
) -> ConsoleInspectorSectionState:
    tasks = snapshot.tasks
    if tasks.availability is not EnvSourceAvailability.OK:
        # TASK-31660: PENDING and UNBOUND land here too, and hiding the card
        # is the SAME choice the Environment section makes for them -- say
        # nothing rather than assert something. The difference is only that
        # the Environment section is a permanently mounted header (with the
        # Refresh tail), so it needs a visible row to say "checking"/"not
        # bound"; the Tasks card has no header of its own to keep honest, so
        # its non-assertion is simply its absence.
        return ConsoleInspectorSectionState(rows=(), summary="")
    if tasks.scanning and not tasks.entries and tasks.branch_task is None:
        return ConsoleInspectorSectionState(
            rows=(InspectorSectionRow(row_id="task-scanning",
                                      primary_text="Scanning backlog…"),),
            summary="",
        )
    rows: list[InspectorSectionRow] = []
    if tasks.branch_task is not None:
        bt = tasks.branch_task
        ac = f"{bt.ac_done}/{bt.ac_total} ACs · " if bt.ac_total else ""
        rows.append(InspectorSectionRow(
            row_id=TASKS_ROW_HEAD,
            primary_text=_with_expand_marker(
                f"task-{bt.task_id} · {bt.status}", TASKS_ROW_HEAD, expanded
            ),
            secondary_text=f"{ac}{bt.title}",
            status=_STATUS_ROW_CLASS.get(bt.status, ""),
            clickable=True,
        ))
    else:
        # TASK-31662 AC#4: this row used to read "3 in progress · 12 to do"
        # directly under a header reading "3 doing · 12 todo" -- one fact,
        # twice, in two vocabularies. The counts belong to the header (it
        # is what a COLLAPSED section shows); the row is the handle onto
        # the list, so it says what the list holds. It stays a row rather
        # than being dropped because it is the only expand/collapse gesture
        # for the entry list, and because a Tasks section with no rows
        # hides itself entirely (`right_rail.py`) -- which is also what
        # `test_poll_landing_that_hides_the_tasks_section_falls_back_to_
        # environment_toggle` measures.
        count = len(tasks.entries)
        rows.append(InspectorSectionRow(
            row_id=TASKS_ROW_HEAD,
            primary_text=_with_expand_marker("Backlog", TASKS_ROW_HEAD, expanded),
            secondary_text=f"{count} task" + ("s" if count != 1 else ""),
            clickable=True,
        ))
    if TASKS_ROW_HEAD in expanded:
        ordered = sorted(
            tasks.entries,
            key=lambda e: (0 if e.status == "In Progress" else 1, e.task_id),
        )
        for index, entry in enumerate(ordered[:MAX_TASK_LIST_ROWS]):
            rows.append(InspectorSectionRow(
                row_id=f"{TASKS_ENTRY_ROW_PREFIX}{index}",
                primary_text=f"task-{entry.task_id} · {entry.title}",
                secondary_text=entry.status,
                status=_STATUS_ROW_CLASS.get(entry.status, ""),
                indent=1,
            ))
        if len(tasks.entries) > MAX_TASK_LIST_ROWS:
            rows.append(InspectorSectionRow(
                row_id="task-entry-more",
                primary_text=f"… {len(tasks.entries) - MAX_TASK_LIST_ROWS} more",
                indent=1,
            ))
        if tasks.branch_task is not None:
            rows.append(InspectorSectionRow(
                row_id=TASKS_ROW_ADD,
                primary_text=_with_insert_marker("Add task to chat"),
                clickable=True,
                indent=1,
            ))
    summary = (
        f"task-{tasks.branch_task.task_id} · {tasks.branch_task.status}"
        if tasks.branch_task
        else tasks_count_summary(tasks.in_progress, tasks.todo)
    )
    return ConsoleInspectorSectionState(
        rows=tuple(rows), summary=_ellipsize(summary, TASKS_SUMMARY_BUDGET)
    )
