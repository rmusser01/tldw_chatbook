"""Console shell right rail — the Inspector region.

Extracted verbatim out of ``ChatScreen.compose_content`` (wave-1 console
decomposition, task 4): the subtree that used to live inside
``with self._frame_console_region(right_rail):``. Its header is one full-width
collapse Button (``#console-inspector-rail-collapse``), followed by the staged-
Context tray, retrieval-Scope row, run inspector, and live-work card.

**Naming**: shell ids use the ``console-inspector-rail-*`` family
(``console-inspector-rail-collapse``, ``console-inspector-rail-body``) plus
ids that already carry their own distinct names (``console-staged-context-
tray``, the retrieval-scope row's ``ROW_ID``, ``console-run-inspector*``,
``console-settings-summary``). No id in this block belongs to the
``console-context-rail-*`` family — those all live on the LEFT rail (task 3,
``ConsoleLeftRail``) and its handle. The DOM says "Inspector," so this class
is ``ConsoleInspectorRail``, not the plan's placeholder guess.

Deliberately NOT included: ``right_handle`` (``ConsoleRailHandle``, id
``console-inspector-rail-handle``), the collapsed 11-column form shown when
this rail is closed. It stays a direct sibling of this widget under
``#console-workspace-grid`` in ``chat_screen.py``, exactly as ``left_handle``
stayed a sibling of ``ConsoleLeftRail`` in task 3 — the identical shape:
today these are two ``Horizontal``-arranged siblings with independent width
rules, one fixed (11 columns for the handle, vs. 13 for the left rail's
handle) and one fractional (this rail is ``4fr`` vs. the left rail's
``3fr``). Folding both under one parent would force that parent to switch
its own width between those two unit kinds depending on open/closed state —
the same class of layout risk task 3's report flags, and this extraction
does not take it either. "Same nesting" for the ids this widget DOES own is
preserved with zero structural change: this class reuses
``id="console-right-rail"`` as its own root, so it sits in the DOM exactly
where the old ``Vertical`` sat.

Unlike the left rail, nothing in this block is machinery that could move
with it: there are no per-section toggle headers here (the whole rail is
one collapse/expand unit, driven by the already narrowly-``@on``-decorated
``ChatScreen.on_console_inspector_rail_collapse``/``_open``, which reach
``_set_console_rail_preference`` and therefore the left rail too — they stay
on the screen unchanged, same as the left rail's own collapse/open handlers
in task 3). The screen-side sync methods that touch this rail's ids
(``_sync_console_staged_context_tray``, ``_sync_console_retrieval_scope_row``,
``_sync_console_settings_summary``, ``_apply_console_live_work_card_swap``)
all either call screen-owned business-state builders internally or also
reach the LEFT rail's DOM (``_sync_console_settings_summary`` updates the
left rail's Model-section value rows too) or the control bar's status chips
(``_sync_console_retrieval_scope_row``) — so, per task 3's own rule (reach
beyond the region -> stays on the screen), none of them move. They keep
working unmodified: screen-side ``query_one`` into this rail's ids crosses
the compound-widget boundary transparently, proven live in task 3's review.
"""

from __future__ import annotations

from collections.abc import Callable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button

from ...Chat.console_display_state import (
    ConsoleInspectorState,
    ConsoleRetrievalScopeState,
    ConsoleStagedContextState,
)
from ...Chat.console_session_settings import ConsoleSettingsSummaryState
from ...Widgets.Console import (
    ConsoleChangedFilesSection,
    ConsoleChangedFilesState,
    ConsoleRetrievalScopeRow,
    ConsoleRunInspector,
    ConsoleSettingsSummary,
    ConsoleStagedContextTray,
)
from ...Widgets.Console.console_retrieval_scope_row import (
    ROW_ID as CONSOLE_RETRIEVAL_SCOPE_ROW_ID,
)
from .frame import frame_console_region


class ConsoleInspectorRail(Vertical):
    """The Console shell's right rail: staged Context, Scope, and the run Inspector.

    All display data is supplied at construction, computed by the screen
    exactly as it was computed inline before this extraction (staged
    launch-context resolution, retrieval scope resolution, the run inspector
    state, session settings summary, and the live-work card build all remain
    screen-owned concerns — this widget only renders the results). Nothing
    here reaches ``app_instance``.
    """

    def __init__(
        self,
        *,
        staged_context_state: ConsoleStagedContextState,
        retrieval_scope_state: ConsoleRetrievalScopeState,
        inspector_state: ConsoleInspectorState,
        changed_files_state: ConsoleChangedFilesState,
        settings_summary_state: ConsoleSettingsSummaryState,
        live_work_card_builder: Callable[[], Widget],
        **kwargs,
    ) -> None:
        """Create the right rail from pre-computed display data.

        Args:
            staged_context_state: Staged-sources display state for the
                Context tray (``ChatScreen._build_console_staged_context_state``).
            retrieval_scope_state: Retrieval scope display state for the
                Scope row (``ChatScreen._build_console_retrieval_scope_state``),
                the same snapshot the status-pills strip's scope chip renders
                from -- computed once by the screen, shared verbatim.
            inspector_state: Run inspector rows/state
                (``ChatScreen._build_console_inspector_state``).
            changed_files_state: Cross-turn "Changed files" section display
                state (``ChatScreen._build_console_changed_files_state``,
                TASK-18060 Task 5, review-rail spec §2) -- reads only the
                screen's cached summary, never the DB/git. Mounted between
                the Scope row and the run inspector; renders nothing when
                empty (config OFF or no changed-file history).
            settings_summary_state: Console session settings summary
                (``ChatScreen._build_console_settings_summary_state``),
                parsed independently here from the left rail's own copy --
                matches the pre-extraction inline compose, which built this
                twice (once per rail) rather than sharing one value.
            live_work_card_builder: Zero-arg callable that builds the
                pending-launch status card or source-readiness card
                (``ChatScreen._build_console_live_work_status_card``/
                ``_build_console_live_work_source_readiness_card``) -- these
                reach ``self.app_instance`` and
                ``self._console_library_rag_query``, so the screen still
                owns the build logic; only the CALL is deferred to this
                rail's own ``compose()``. A builder, not a pre-built
                instance, is the point: ``_apply_console_live_work_card_swap``
                removes and remounts this same card by id without going
                through this rail at all, so a stored widget INSTANCE here
                would go stale the moment anything recomposes this rail
                (see the design spec's region-widget rule) -- re-yielding a
                widget Textual already removed from the DOM. Calling the
                builder fresh on every ``compose()`` (matching
                ``ConsoleDictationController``'s late-binding constructor
                rule -- see ``dictation.py``'s module docstring) always
                mounts a brand-new instance instead.
            kwargs: Forwarded to ``Vertical``.
        """
        super().__init__(
            id="console-right-rail",
            classes="console-region destination-workbench-pane",
            **kwargs,
        )
        self._staged_context_state = staged_context_state
        self._retrieval_scope_state = retrieval_scope_state
        self._inspector_state = inspector_state
        self._changed_files_state = changed_files_state
        self._settings_summary_state = settings_summary_state
        self._live_work_card_builder = live_work_card_builder

    def compose(self) -> ComposeResult:
        """Compose the rail header, staged-context tray, scope row, and run inspector.

        Returns:
            The rail-header row followed by the scrollable body: the staged
            Context tray, the retrieval-Scope row, the run inspector plus
            session settings summary, and the live-work status card, in
            mount order.
        """
        right_rail_header = Horizontal(classes="console-rail-header")
        right_rail_header.styles.height = 1
        right_rail_header.styles.min_height = 1
        right_rail_header.styles.max_height = 1
        with right_rail_header:
            collapse_button = Button(
                "Inspect|--------->",
                id="console-inspector-rail-collapse",
                classes="console-rail-collapse-button",
                compact=True,
            )
            collapse_button.tooltip = "Collapse Inspector rail"
            collapse_button.styles.width = "100%"
            collapse_button.styles.min_width = 0
            collapse_button.styles.max_width = "100%"
            collapse_button.styles.text_align = "left"
            collapse_button.styles.content_align = ("left", "middle")
            yield collapse_button

        with VerticalScroll(
            id="console-inspector-rail-body",
            classes="console-inspector-rail-body",
        ):
            # Context (staged sources) section -- moved here from the left
            # rail (task-400). Pinned to the TOP of the Inspector body so it
            # is visible without scrolling and reads above the run
            # inspector's Source Readiness section (splitting the
            # monolithic ConsoleRunInspector at that boundary would have
            # meant reworking its TASK-259 in-place-update fingerprinting
            # for a placement change). Same pure display-state seam as
            # before: no DB reads on compose/recompose.
            staged_context_state = self._staged_context_state
            staged_context_tray = ConsoleStagedContextTray(
                staged_context_state,
                id="console-staged-context-tray",
                classes="console-inspector-context-section",
            )
            staged_context_tray.styles.width = "100%"
            staged_context_tray.styles.min_width = 0
            staged_context_tray.styles.height = "auto"
            staged_context_tray.styles.min_height = (
                3 if staged_context_state.is_empty else 4
            )
            staged_context_tray.styles.max_height = (
                6 if staged_context_state.is_empty else 10
            )
            # `ChatScreen._staged_context_frame_variant` is a `@staticmethod`
            # returning "quiet" unconditionally (mirrors task 3's inlining
            # of `_workspace_context_frame_variant`) -- inlined as a literal
            # here rather than passed as data.
            yield frame_console_region(staged_context_tray, variant="quiet")

            # task-9: Retrieval scope row -- a sibling of the Sources tray
            # above (never a row inside it or inside ConsoleRunInspector
            # below: design spec section 4 keeps the staged-vs-scope
            # mechanism boundary visible). Renders purely from session
            # state -- no DB reads on compose/recompose.
            retrieval_scope_row = ConsoleRetrievalScopeRow(
                self._retrieval_scope_state,
                id=CONSOLE_RETRIEVAL_SCOPE_ROW_ID,
                classes="console-inspector-context-section",
            )
            retrieval_scope_row.styles.width = "100%"
            retrieval_scope_row.styles.min_width = 0
            retrieval_scope_row.styles.height = "auto"
            yield frame_console_region(retrieval_scope_row, variant="quiet")

            # TASK-18060 Task 5 (review-rail spec §2): cross-turn
            # "Changed files" section -- a sibling of the Scope row above,
            # between it and the run inspector below. Renders purely from
            # precomputed state (never a DB/git read on compose/recompose);
            # the widget itself suppresses its own display when the state
            # is empty (config OFF or no changed-file history), so nothing
            # here needs to conditionally skip mounting it.
            changed_files_section = ConsoleChangedFilesSection(
                self._changed_files_state,
                id="console-changed-files-section",
            )
            # Same margin/padding rhythm as the Sources tray and Scope row
            # above (`.console-inspector-context-section`) -- the widget's
            # own fixed `console-changed-files-section` class is untouched,
            # this is additive, matching how `frame_console_region` below
            # also adds a class rather than replacing the widget's own.
            changed_files_section.add_class("console-inspector-context-section")
            changed_files_section.styles.width = "100%"
            changed_files_section.styles.min_width = 0
            yield frame_console_region(changed_files_section, variant="quiet")

            with Vertical(id="console-run-inspector"):
                yield ConsoleRunInspector(
                    self._inspector_state,
                    id="console-run-inspector-state",
                )
                settings_summary = ConsoleSettingsSummary(
                    self._settings_summary_state,
                    id="console-settings-summary",
                    classes=(
                        "console-inspector-session-settings console-settings-summary"
                    ),
                )
                settings_summary.styles.width = "100%"
                settings_summary.styles.min_width = 0
                yield settings_summary

            yield self._live_work_card_builder()
