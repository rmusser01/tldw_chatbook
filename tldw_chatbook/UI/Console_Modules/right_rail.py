"""Console shell right rail — the Inspector region.

Extracted verbatim out of ``ChatScreen.compose_content`` (wave-1 console
decomposition, task 4): the subtree that used to live inside
``with self._frame_console_region(right_rail):``. Its header is one full-width
collapse Button (``#console-inspector-rail-collapse``), followed by the staged-
Context tray, retrieval-Scope row, run inspector, and live-work card. task-9
prepended two more scrollable-body sections ahead of the tray: Environment
(``#console-environment-section``) and Tasks (``#console-tasks-section``).
task-10 then moved the Agent rail's fleet mini-section
(``#console-agent-section-subagents``) here from the left rail, directly
below Tasks -- the left rail's Agent section keeps its status/steps/
drilldown controls; only the sub-agent list moved.

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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import os

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.events import DescendantBlur, DescendantFocus, Key, Resize
from textual.geometry import Size
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Static, TextArea

from ...Chat.console_display_state import (
    ConsoleInspectorState,
    ConsoleProjectInstructionState,
    ConsoleRetrievalScopeState,
    ConsoleStagedContextState,
)
from ...Chat.console_environment_state import ENVIRONMENT_SECTION_ID, TASKS_SECTION_ID
from ...Chat.console_glyphs import GLYPH_COLLAPSE_RIGHT
from ...Widgets.glyph_fallback import resolve_glyph
from ...Chat.console_session_settings import ConsoleSettingsSummaryState
from ...Chat.console_live_work import PENDING_LAUNCH_CARD_ID
from ...Chat.console_library_activity_buffer import LibraryActivityFlushResult
from ...Chat.library_activity import LibraryActivityRecord, LibraryActivityView
from ...Constants import (
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID,
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE,
)
from ..Navigation.main_navigation import NavigateToScreen
from ...Widgets.Console import (
    ConsoleBoundedSection,
    InspectorOwnershipPolicy,
    ConsoleProjectInstructionStatusRow,
    ConsoleRetrievalScopeRow,
    ConsoleRunInspector,
    ConsoleSendAuthoritySummary,
    ConsoleSettingsSummary,
    ConsoleStagedContextTray,
    ConsoleStagedSourceOpenRequested,
)
from ...Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionState,
)
from ...Widgets.Console.console_retrieval_scope_row import (
    ROW_ID as CONSOLE_RETRIEVAL_SCOPE_ROW_ID,
)
from .agent import CONSOLE_AGENT_FLEET_SECTION_ID
from .frame import frame_console_region
from .rail_section_layout import outer_hint_required


INSPECTOR_OUTER_HINT = "▼ more sections — scroll"
INSPECTOR_OUTER_HINT_ID = "console-inspector-outer-scroll-hint"
INSPECTOR_SCROLL_OWNER_CLASS = "console-inspector-scroll-owner"


@dataclass(frozen=True)
class _FocusRecoveryIncident:
    """Semantic focus snapshot that remains valid across widget replacement."""

    target_id: str | None
    target_index: int | None
    remaining_reconcile_passes: int = 8


class ConsoleSelectedTurnActivity(Vertical):
    """Render cited-source and Library-operation review for one selected turn."""

    class RetryRequested(Message):
        """Request that the screen/controller retry retained activity writes."""

    def __init__(
        self,
        view: LibraryActivityView,
        *,
        citation_count: int = 0,
        flush_result: LibraryActivityFlushResult | None = None,
        on_reconcile: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(id="console-selected-turn")
        self._view = view
        self._citation_count = max(0, citation_count)
        self._flush_result = flush_result
        self._on_reconcile = on_reconcile
        self.display = view.selected_turn_id is not None
        self.styles.height = "auto"
        self.styles.min_height = 0
        self.styles.width = "100%"
        self.styles.min_width = 0

    def sync_state(
        self,
        view: LibraryActivityView,
        *,
        citation_count: int,
        flush_result: LibraryActivityFlushResult | None,
    ) -> None:
        """Replace the pure selected-turn projection when it changed."""

        normalized_count = max(0, citation_count)
        self.display = view.selected_turn_id is not None
        if (
            view == self._view
            and normalized_count == self._citation_count
            and flush_result == self._flush_result
        ):
            return
        self._view = view
        self._citation_count = normalized_count
        self._flush_result = flush_result
        self.refresh(recompose=True)
        if self._on_reconcile is not None:
            self.call_after_refresh(self._on_reconcile)

    @staticmethod
    def _time_copy(record: LibraryActivityRecord) -> str:
        if record.occurred_at is None:
            return "time unavailable"
        try:
            occurred_at = datetime.fromtimestamp(
                record.occurred_at, tz=timezone.utc
            )
        except (OSError, OverflowError, ValueError):
            return "time unavailable"
        return occurred_at.strftime("%H:%M:%S UTC")

    @classmethod
    def _action_widgets(
        cls, record: LibraryActivityRecord, index: int
    ) -> tuple[Widget, ...]:
        event = record.event
        mode = "Direct" if event.library_provider == "direct" else "RAG"
        result_word = "result" if event.result_count == 1 else "results"
        widgets: list[Widget] = [
            Static(
                (
                    f"{event.operation} · {event.actor_kind} · {mode} · "
                    f"{event.status} · {event.result_count} {result_word} · "
                    f"{cls._time_copy(record)}"
                ),
                id=f"console-library-activity-action-{index}",
                classes="console-library-activity-action",
                markup=False,
            )
        ]
        for ref_index, ref in enumerate(event.source_refs):
            widgets.append(
                Static(
                    f"{ref.source_type} · {ref.title} · {ref.source_id}",
                    id=f"console-library-activity-ref-{index}-{ref_index}",
                    classes="console-library-activity-source-ref",
                    markup=False,
                )
            )
        if event.error_summary:
            widgets.append(
                Static(
                    event.error_summary,
                    id=f"console-library-activity-error-{index}",
                    classes="console-library-activity-error",
                    markup=False,
                )
            )
        return tuple(widgets)

    def compose(self) -> ComposeResult:
        yield Static(
            "Selected turn",
            id="console-selected-turn-heading",
            classes="console-inspector-group-heading destination-section",
            markup=False,
        )
        yield Static(
            f"Cited sources ({self._citation_count})",
            id="console-selected-turn-cited-sources",
            classes="console-selected-turn-subsection",
            markup=False,
        )
        with Vertical(id="console-selected-turn-library-activity"):
            yield Static(
                f"Library activity ({len(self._view.actions)} actions)",
                id="console-selected-turn-library-activity-heading",
                classes="console-selected-turn-subsection",
                markup=False,
            )
            body: list[Widget] = []
            if not self._view.actions:
                body.append(
                    Static(
                        "No Library activity for this turn.",
                        id="console-library-activity-empty",
                        markup=False,
                    )
                )
            else:
                for index, record in enumerate(self._view.actions):
                    body.extend(self._action_widgets(record, index))
            if self._view.corrupt_row_count:
                body.append(
                    Static(
                        "Some Library activity could not be displayed.",
                        id="console-library-activity-corrupt",
                        markup=False,
                    )
                )
            if self._flush_result is not None and self._flush_result.warning:
                body.append(
                    Static(
                        self._flush_result.warning,
                        id="console-library-activity-save-warning",
                        markup=False,
                    )
                )
                body.append(
                    Button(
                        "Retry",
                        id="console-library-activity-retry",
                        compact=True,
                    )
                )
            yield ConsoleBoundedSection(*body, section_id="library-activity")

    @on(Button.Pressed, "#console-library-activity-retry")
    def _request_retry(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.RetryRequested())


class _InspectorOuterBody(VerticalScroll):
    """Inspector scroller that separates scroll notices from geometry notices.

    Two distinct owner notifications, because they cost two very different
    things (TASK-21117):

    * ``on_scrolled`` -- the offset moved and nothing else. The only outer
      state a scroll can change is the fold copy, so the owner repaints that
      and stops.
    * ``on_geometry_changed`` -- committed size or virtual-size change
      (``Resize``, or a virtual-size-only change whose ``Resize`` does not
      bubble). This is the one that can change what the fold REQUIRES, so it
      still schedules the full outer reconcile.
    """

    def __init__(
        self,
        *,
        on_geometry_changed: Callable[[], None],
        on_scrolled: Callable[[], None],
    ) -> None:
        super().__init__(
            id="console-inspector-rail-body",
            classes="console-inspector-rail-body",
            can_focus=True,
        )
        self._on_geometry_changed = on_geometry_changed
        self._on_scrolled = on_scrolled

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if old_value != new_value:
            self._on_scrolled()

    def on_resize(self, _event: Resize) -> None:
        self._on_geometry_changed()

    def _size_updated(
        self,
        size: Size,
        virtual_size: Size,
        container_size: Size,
        layout: bool = True,
    ) -> bool:
        """Invalidate on committed size or virtual-size changes.

        Textual is exact-pinned in this project; this narrow override covers
        virtual-size-only changes whose ``Resize`` event does not bubble.
        """

        previous_size = self.size
        previous_virtual_size = self.virtual_size
        updated = super()._size_updated(size, virtual_size, container_size, layout)
        if self.is_mounted and (
            self.size != previous_size or self.virtual_size != previous_virtual_size
        ):
            self._on_geometry_changed()
        return updated


def _resolve_inspector_ownership_policy() -> InspectorOwnershipPolicy:
    """Resolve the opt-in strict policy at the production composition boundary."""
    if os.environ.get("TLDW_CONSOLE_STRICT_INSPECTOR_OWNERSHIP") == "1":
        return InspectorOwnershipPolicy.STRICT
    return InspectorOwnershipPolicy.RESILIENT


class ConsoleInspectorRail(Vertical):
    """The Console shell's right rail: staged Context, Scope, and the run Inspector.

    All display data is supplied at construction, computed by the screen
    exactly as it was computed inline before this extraction (staged
    launch-context resolution, retrieval scope resolution, the run inspector
    state, session settings summary, and the live-work card build all remain
    screen-owned concerns — this widget only renders the results). Nothing
    here reaches ``app_instance``.

    **Outer-fold invalidation triggers** (TASK-21117). Every source that can
    invalidate the outer fold routes explicitly to one of two paths; nothing
    reaches the fold by any other route:

    * ``on_mount`` — full (owner demand)
    * rail ``Resize`` — full (owner demand)
    * section-widget state sync: the staged-context tray, changed-files
      section, run inspector, and settings summary each call the
      ``on_reconcile`` callback this rail hands them at compose time when a new
      display state lands — full (owner demand)
    * the screen's live-work card swap (``chat_screen.py``, the one external
      ``request_outer_reconcile`` caller) — full (owner demand)
    * focus-recovery scheduling/retry — full (owner demand)
    * body ``Resize`` — full (geometry only)
    * body ``_size_updated``: any committed size or virtual-size change. This
      is the route a ``ConsoleBoundedSection`` collapse/expand or content
      growth actually takes — geometry only, *not* owner demand;
      ``ConsoleBoundedSection`` has no ``on_reconcile`` of its own — full
      (geometry only)
    * hint display-toggle continuation — full (geometry only)
    * body ``scroll_y`` change: wheel, keys, ``scroll_to``, reveal — pure
      scroll

    The full path is ``_request_outer_reconcile`` — two ``ConsoleBoundedSection``
    sweeps, ``refresh(layout=True)`` on the whole rail, then a second
    ``call_after_refresh`` hop for focus recovery and fold measurement. The
    pure-scroll path is ``_handle_outer_scrolled``: an offset clamp and, only
    when it actually changes, the fold copy. Scroll position is not an input to
    ``outer_hint_required``, so a scroll frame cannot change the fold itself.
    """

    BUNDLED_CSS = """
    ConsoleInspectorRail .console-inspector-scroll-owner,
    ConsoleInspectorRail .console-inspector-scroll-owner .console-rail-section-title {
        text-style: bold underline;
    }
    """

    def __init__(
        self,
        *,
        staged_context_state: ConsoleStagedContextState,
        retrieval_scope_state: ConsoleRetrievalScopeState,
        inspector_state: ConsoleInspectorState,
        project_instruction_state: ConsoleProjectInstructionState,
        settings_summary_state: ConsoleSettingsSummaryState,
        live_work_card_builder: Callable[[], Widget],
        # TASK-24611: a zero-arg BUILDER, not a widget instance, for the same
        # reason the live-work card is one -- a region must not store a child
        # the screen may remove and replace outside its own `compose()`, or a
        # later recompose re-yields a widget Textual has already removed.
        library_search_builder: Callable[[], Widget] | None = None,
        library_activity_view: LibraryActivityView | None = None,
        library_activity_citation_count: int = 0,
        library_activity_flush_result: LibraryActivityFlushResult | None = None,
        library_activity_retry: Callable[[], Awaitable[None]] | None = None,
        ownership_policy: InspectorOwnershipPolicy | None = None,
        inspector_more_open: bool = False,
        environment_section_state: ConsoleInspectorSectionState | None = None,
        tasks_section_state: ConsoleInspectorSectionState | None = None,
        environment_open: bool = True,
        tasks_open: bool = True,
        agent_fleet_section_state: ConsoleInspectorSectionState | None = None,
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
            project_instruction_state: Content-free project-instruction status
                rendered above Sources.
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
            library_search_builder: Zero-arg callable that builds the Library
                search controls (``ChatScreen._build_console_library_search_region``),
                yielded directly below the staged-context tray so retrieval's
                inputs sit with the sources they filter (TASK-24611). Omitted
                (``None``) by callers that render no Library search -- the
                rail then yields nothing there rather than an empty holder.
                A BUILDER, not a widget instance, for the same reason as
                ``live_work_card_builder`` above: a region must never store a
                child the screen may remove and replace outside this rail's
                own ``compose()``, or a later recompose re-yields a widget
                Textual has already removed from the DOM.
            library_activity_view: Pure selected-turn activity projection.
            library_activity_citation_count: Cited-source count for the same
                selected turn.
            library_activity_flush_result: Current store-owned persistence
                state for retained Library activity.
            library_activity_retry: Store-owned retained-write retry callback.
            ownership_policy: Optional explicit Inspector ownership policy.
                Production resolves the strict opt-in environment flag at
                this composition boundary when omitted.
            inspector_more_open: Whether the run inspector's "More" toggle
                starts expanded.
            environment_section_state: Rows/summary for the Environment
                section (task-9), mounted at the top of the Inspector body.
                ``None`` renders the empty-state projection (no rows), which
                hides the section (``styles.display = "none"``).
            tasks_section_state: Rows/summary for the Tasks section
                (task-9), mounted directly below Environment. Same
                empty-hides-the-section behavior as above.
            environment_open: Initial collapse state for the Environment
                section.
            tasks_open: Initial collapse state for the Tasks section.
            agent_fleet_section_state: Rows + header summary (task-10,
                moved here from the left rail's Agent section) for the
                ``ConsoleInspectorSection`` that renders the conversation's
                own sub-agent fleet -- computed by
                ``ConsoleAgentController._console_agent_fleet_section_
                state``. Empty rows/summary hides the section entirely
                (nothing to show, or a sub-agent drill-down is active and
                the left rail's Agent status/steps Statics already carry
                that one child's own detail -- state 3, unchanged). Mounted
                directly after the Tasks section, always collapsed
                (``open=False``) on first paint, matching the left rail's
                prior behavior.
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
        self._project_instruction_state = project_instruction_state
        self._settings_summary_state = settings_summary_state
        self._live_work_card_builder = live_work_card_builder
        self._library_search_builder = library_search_builder
        self._library_activity_view = library_activity_view or LibraryActivityView(
            selected_turn_id=None,
            actions=(),
        )
        self._library_activity_citation_count = max(
            0, library_activity_citation_count
        )
        self._library_activity_flush_result = library_activity_flush_result
        self._library_activity_retry = library_activity_retry
        self._ownership_policy = (
            ownership_policy or _resolve_inspector_ownership_policy()
        )
        self._inspector_more_open = inspector_more_open
        self._environment_section_state = environment_section_state or (
            ConsoleInspectorSectionState(rows=(), summary="")
        )
        self._tasks_section_state = tasks_section_state or (
            ConsoleInspectorSectionState(rows=(), summary="")
        )
        self._environment_open = environment_open
        self._tasks_open = tasks_open
        self._agent_fleet_section_state = agent_fleet_section_state or (
            ConsoleInspectorSectionState(rows=(), summary="")
        )
        self._reported_unknown_fingerprints: set[tuple[str, ...]] = set()
        self._outer_reconcile_scheduled = False
        self._outer_reconcile_dirty = False
        self._outer_reconcile_owner_demand = False
        self._outer_owner_reconcile_count = 0
        self._library_activity_focus_pending = False
        self._inspector_focus_active = False
        self._navigation_generation = 0
        #: TASK-24704: the boundary `n`/`p` last moved to, kept ONLY while the
        #: outer scroller holds focus because navigation parked it there (a
        #: section with no focusable control of its own). Cleared by any other
        #: focus change, so a deliberately focused outer body keeps its
        #: documented "outside the list" meaning.
        self._last_boundary_index: int | None = None
        self._section_focus_history: dict[str, tuple[Widget, tuple[Widget, ...]]] = {}
        self._pending_focus_recoveries: dict[str, _FocusRecoveryIncident] = {}

    def on_mount(self) -> None:
        """Schedule the first owner pass after descendant layout settles."""

        self.request_outer_reconcile()

    def on_unmount(self) -> None:
        """Discard pending generations when the Inspector leaves the DOM."""

        self._library_activity_focus_pending = False
        self._clear_outer_reconcile_state()

    def sync_library_activity(
        self,
        view: LibraryActivityView,
        *,
        citation_count: int,
        flush_result: LibraryActivityFlushResult | None,
    ) -> None:
        """Keep the rail owner and mounted selected-turn child in sync."""

        self._library_activity_view = view
        self._library_activity_citation_count = max(0, citation_count)
        self._library_activity_flush_result = flush_result
        try:
            selected_turn = self.query_one(
                "#console-selected-turn", ConsoleSelectedTurnActivity
            )
        except (NoMatches, QueryError):
            return
        selected_turn.sync_state(
            view,
            citation_count=citation_count,
            flush_result=flush_result,
        )

    def request_library_activity_focus(self) -> None:
        """Focus activity after the selected-turn and outer rail settle."""

        self._library_activity_focus_pending = True
        self.request_outer_reconcile()

    @on(ConsoleSelectedTurnActivity.RetryRequested)
    def retry_library_activity(
        self, event: ConsoleSelectedTurnActivity.RetryRequested
    ) -> None:
        """Keep Retry inside the Inspector region and delegate persistence."""
        event.stop()
        if self._library_activity_retry is not None:
            self.run_worker(
                self._library_activity_retry(),
                group="console-library-activity-retry",
                exclusive=True,
            )

    @on(ConsoleStagedSourceOpenRequested)
    def open_staged_source(self, event: ConsoleStagedSourceOpenRequested) -> None:
        """Navigate from a staged-source detail action into Library."""
        event.stop()
        self.app.post_message(
            NavigateToScreen(
                "library",
                {
                    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE: event.source_type,
                    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID: event.source_id,
                },
            )
        )

    def request_outer_reconcile(self) -> None:
        """Coalesce Inspector owner invalidation behind local section demand."""

        self._request_outer_reconcile(owner_demand=True)

    def _request_outer_geometry_reconcile(self) -> None:
        """Reconcile geometry without creating a second logical owner pass."""

        self._request_outer_reconcile(owner_demand=False)

    def _handle_outer_scrolled(self) -> None:
        """Repaint the fold copy for a pure scroll; never re-lay out the rail.

        A scroll offset cannot change what the outer fold *requires*:
        ``outer_hint_required`` reads content demand against the viewport, and
        a scroll moves neither. Every input that does move them has its own
        trigger into the full reconcile (see the class docstring's trigger
        table), so the pure-scroll path re-clamps the offset and repaints the
        hint copy -- no ``refresh(layout=True)`` on the rail, no refold chain,
        no focus-recovery pass (TASK-21117).

        When a geometry generation is already scheduled it owns this frame's
        fold *and* copy, so this path stands down rather than painting copy
        that pass is about to recompute.
        """

        if not self.is_mounted or not self.is_attached or self._pruning:
            return
        if self._outer_reconcile_scheduled:
            return
        self._update_outer_hint(clamp=True)

    def _request_outer_reconcile(self, *, owner_demand: bool) -> None:
        """Schedule one generation while preserving its semantic owner demand."""

        if not self.is_mounted or not self.is_attached or self._pruning:
            return
        self._outer_reconcile_owner_demand |= owner_demand
        if self._outer_reconcile_scheduled:
            self._outer_reconcile_dirty = True
            return
        self._outer_reconcile_scheduled = True
        self._outer_reconcile_dirty = False
        self.call_after_refresh(self._run_scheduled_outer_reconcile)

    def _clear_outer_reconcile_state(self) -> None:
        """Clear scheduler state without completing a logical owner pass."""

        self._outer_reconcile_scheduled = False
        self._outer_reconcile_dirty = False
        self._outer_reconcile_owner_demand = False

    def _run_scheduled_outer_reconcile(self) -> None:
        """Reconcile the outer fold only after every local body has settled."""

        if not self.is_mounted or not self.is_attached or self._pruning:
            self._clear_outer_reconcile_state()
            return
        if any(
            section._reconcile_scheduled
            for section in self.query(ConsoleBoundedSection)
        ):
            self.call_after_refresh(self._run_scheduled_outer_reconcile)
            return
        # A local pass may have changed a fixed-height owner in this refresh.
        # Let Textual commit that physical geometry before measuring the outer
        # body; descendant Resize is not guaranteed for such changes.
        self.refresh(layout=True)
        self.call_after_refresh(self._finish_scheduled_outer_reconcile)

    def _finish_scheduled_outer_reconcile(self) -> None:
        """Measure after the settled local state has completed one layout pass."""

        if not self.is_mounted or not self.is_attached or self._pruning:
            self._clear_outer_reconcile_state()
            return
        if any(
            section._reconcile_scheduled
            for section in self.query(ConsoleBoundedSection)
        ):
            self.call_after_refresh(self._run_scheduled_outer_reconcile)
            return
        if self._outer_reconcile_dirty:
            self._outer_reconcile_dirty = False
            self.call_after_refresh(self._run_scheduled_outer_reconcile)
            return
        owner_demand = self._outer_reconcile_owner_demand
        self._outer_reconcile_owner_demand = False
        self._outer_reconcile_scheduled = False
        self._install_focus_recovery_callbacks()
        self._reconcile_focus_recovery_state()
        if not self._reconcile_outer_fold():
            self._outer_reconcile_owner_demand |= owner_demand
            return
        if owner_demand:
            self._outer_owner_reconcile_count += 1
        if self._library_activity_focus_pending:
            self._library_activity_focus_pending = False
            self._focus_library_activity()

    def _focus_library_activity(self) -> None:
        """Focus and reveal the settled selected-turn activity heading."""

        try:
            heading = self.query_one(
                "#console-selected-turn-library-activity-heading", Static
            )
            body = self.query_one("#console-inspector-rail-body", VerticalScroll)
        except (NoMatches, QueryError):
            return
        heading.can_focus = True
        self.screen.set_focus(heading, scroll_visible=False)
        body.scroll_to_widget(
            heading,
            animate=False,
            immediate=True,
            force=True,
            top=True,
        )

    def _reconcile_outer_fold(self) -> bool:
        """Apply the counterfactual outer-hint predicate from laid-out geometry."""

        try:
            body = self.query_one("#console-inspector-rail-body", VerticalScroll)
            hint = self.query_one(f"#{INSPECTOR_OUTER_HINT_ID}", Static)
        except (NoMatches, QueryError):
            return True

        desired_rows = max(
            (
                child.virtual_region_with_margin.bottom
                for child in body.children
                if child.display
            ),
            default=0,
        )
        hint_rows = hint.region.height if hint.display else 0
        viewport_without_hint = body.content_region.height + hint_rows
        if viewport_without_hint <= 0:
            return True
        required = outer_hint_required(desired_rows, viewport_without_hint)
        body.scroll_y = min(body.scroll_y, max(0, body.max_scroll_y))
        if hint.display is not required:
            # Clear the copy before changing layout, then measure/clamp again on
            # the next refresh using the new actual body height.
            hint.update("")
            hint.display = required
            self._request_outer_geometry_reconcile()
            return False

        self._update_outer_hint()
        return True

    def _update_outer_hint(self, *, clamp: bool = False) -> None:
        """Paint copy only while actual outer content remains below.

        Args:
            clamp: Re-clamp the outer offset, for symmetry with the full
                reconcile, which clamps before it measures. Defensive rather
                than load-bearing: ``Widget.validate_scroll_y`` already clamps
                every assignment, and the one case that can leave an offset
                past the end — the viewport shrinking with no scroll
                assignment — arrives on the geometry path and is clamped by
                ``_reconcile_outer_fold``.
        """

        try:
            body = self.query_one("#console-inspector-rail-body", VerticalScroll)
            hint = self.query_one(f"#{INSPECTOR_OUTER_HINT_ID}", Static)
        except (NoMatches, QueryError):
            return
        if clamp:
            body.scroll_y = min(body.scroll_y, max(0, body.max_scroll_y))
        if not hint.display:
            self._paint_outer_hint(hint, "")
            return
        self._paint_outer_hint(
            hint,
            INSPECTOR_OUTER_HINT
            if body.max_scroll_y > 0 and body.scroll_y < body.max_scroll_y
            else "",
        )

    @staticmethod
    def _paint_outer_hint(hint: Static, text: str) -> None:
        """Write the fold copy only when it actually changes.

        Most scroll frames leave the copy identical, and ``Static.update``
        costs a repaint each time. The unchanged check reads the live widget's
        own content rather than a shadow copy held on the rail: the fold path
        writes this same widget directly when it toggles the hint's display, so
        a cached string would need an invalidation hook at every writer -- the
        stale-one-frame trap TASK-21115's review found.

        ``layout=False`` because this slot's height is pinned to exactly one
        row at compose time, so its copy can never resize it; the default
        ``Static.update`` would schedule a view layout for a one-row repaint
        (measured: two extra screen layout passes per wheel gesture). Changing
        the hint's DISPLAY still goes through the full reconcile, which lays
        out normally.
        """

        current = hint.content
        if isinstance(current, str) and current == text:
            return
        hint.update(text, layout=False)

    def on_resize(self, _event: Resize) -> None:
        """Recompute fixed-child overflow on terminal grow and shrink."""

        self.request_outer_reconcile()

    @staticmethod
    def _is_visible(widget: Widget) -> bool:
        return widget.display and all(
            not isinstance(ancestor, Widget) or ancestor.display
            for ancestor in widget.ancestors
        )

    @classmethod
    def _is_enabled_focus_target(cls, widget: Widget) -> bool:
        return bool(
            widget.is_mounted
            and widget.focusable
            and cls._is_visible(widget)
            and not getattr(widget, "disabled", False)
        )

    def inspector_active(self, focused: Widget | None = None) -> bool:
        """Return whether live focus belongs to this mounted Inspector rail."""

        target = self.app.focused if focused is None else focused
        return target is self or (
            isinstance(target, Widget) and self in target.ancestors
        )

    def _mounted_boundaries(
        self,
    ) -> tuple[tuple[ConsoleBoundedSection, Widget], ...]:
        """Return visible direct boundaries as body and external header."""

        boundaries: list[tuple[ConsoleBoundedSection, Widget]] = []
        for section in self.query(ConsoleBoundedSection):
            if not self._is_visible(section):
                continue
            parent = section.parent
            if not isinstance(parent, Widget):
                continue
            siblings = list(parent.children)
            try:
                index = siblings.index(section)
            except ValueError:
                continue
            if index == 0:
                continue
            header = siblings[index - 1]
            boundaries.append((section, header))
        return tuple(boundaries)

    def _boundary_index_for_target(
        self,
        target: Widget,
        boundaries: tuple[tuple[ConsoleBoundedSection, Widget], ...],
    ) -> int | None:
        for index, (section, header) in enumerate(boundaries):
            if (
                target is section
                or section in target.ancestors
                or target is header
                or header in target.ancestors
            ):
                return index
        return None

    def _positional_boundary_index(
        self,
        target: Widget,
        boundaries: tuple[tuple[ConsoleBoundedSection, Widget], ...],
        direction: int,
    ) -> int | None:
        """Resolve compact non-boundary descendants by mounted DOM position."""

        if target in (
            self,
            self.query_one("#console-inspector-rail-body"),
            self.query_one("#console-inspector-rail-collapse"),
        ):
            return 0 if direction > 0 else len(boundaries) - 1
        order = [self, *self.query("*")]
        try:
            target_position = order.index(target)
        except ValueError:
            return 0 if direction > 0 else len(boundaries) - 1
        candidates = []
        for index, (_section, header) in enumerate(boundaries):
            try:
                header_position = order.index(header)
            except ValueError:
                continue
            if (direction > 0 and header_position > target_position) or (
                direction < 0 and header_position < target_position
            ):
                candidates.append((header_position, index))
        if not candidates:
            return None
        return min(candidates)[1] if direction > 0 else max(candidates)[1]

    def on_key(self, event: Key) -> None:
        """Handle bubbling n/p section navigation only at the rail boundary.

        Moves focus to the next (``n``) or previous (``p``) boundary section.
        A boundary whose section has no focusable control parks focus on the
        outer scroller and reveals its header instead, and the index it
        parked at is remembered in ``_last_boundary_index`` so a repeated
        press continues rather than restarting -- see ``on_descendant_focus``
        for when that memory is discarded.

        Both keys are consumed whenever the rail is active, including at a
        no-wrap edge: they are rail-local commands here, and letting the
        printable key bubble would reach ``ChatScreen``'s global hands-free
        barge-in hook.

        Args:
            event: The bubbling key event. Ignored unless its ``key`` is
                ``n``/``p``, the Inspector is active, and focus is not inside
                a text-entry widget (where the letters must type).
        """

        if event.key not in ("n", "p") or not self.inspector_active():
            return
        focused = self.app.focused
        if isinstance(focused, (Input, TextArea)):
            return
        if not isinstance(focused, Widget):
            return
        # Once n/p is a rail-local command, consume it even when navigation
        # reaches a no-wrap edge. Otherwise the same printable key bubbles to
        # ChatScreen's global hands-free/realtime barge-in hook.
        event.stop()
        event.prevent_default()
        direction = 1 if event.key == "n" else -1
        boundaries = self._mounted_boundaries()
        if not boundaries:
            return
        anchored = self._boundary_index_for_target(focused, boundaries)
        if anchored is None:
            # TASK-24704 (Qodo #6). A section whose content is all `Static`
            # -- `Run`, `Source Readiness`, and now `Changes` when its only
            # action is disabled -- has no focusable control, so
            # `_focus_boundary` parks focus on the OUTER SCROLLER and reveals
            # the header. `_positional_boundary_index` then treats the outer
            # body as "before the first boundary" and answers 0, so the next
            # `n` returned to the top and navigation could never get past
            # such a section: measured as six consecutive `n` presses all
            # landing on `#console-inspector-rail-body`.
            #
            # Continue from where navigation actually was instead. Only for
            # the outer body, and only when navigation has run: entering the
            # rail cold still starts at boundary 0.
            if (
                self._last_boundary_index is not None
                and focused is self.query_one("#console-inspector-rail-body")
            ):
                target_index = self._last_boundary_index + direction
                if target_index < 0 or target_index >= len(boundaries):
                    return
            else:
                target_index = self._positional_boundary_index(
                    focused, boundaries, direction
                )
        else:
            target_index = anchored + direction
            if target_index < 0 or target_index >= len(boundaries):
                return
        if target_index is None:
            return
        self._navigation_generation += 1
        self._focus_boundary(
            boundaries[target_index],
            generation=self._navigation_generation,
            index=target_index,
        )

    def _focus_boundary(
        self,
        boundary: tuple[ConsoleBoundedSection, Widget],
        *,
        generation: int,
        index: int | None = None,
    ) -> None:
        section, header = boundary
        try:
            outer = self.query_one("#console-inspector-rail-body", VerticalScroll)
        except (NoMatches, QueryError):
            return
        target: Widget = outer
        if section.viewport.can_focus and self._is_visible(section.viewport):
            target = section.viewport
        else:
            boundary_targets = (
                header,
                *header.query("*"),
                *section.viewport.query("*"),
            )
            for widget in boundary_targets:
                if isinstance(widget, Widget) and self._is_enabled_focus_target(widget):
                    target = widget
                    break
        # TASK-24704: the anchor exists ONLY for the case it was added for --
        # navigation parked focus on the outer scroller because this section
        # has no focusable control. When focus lands on a real control the
        # anchor is dropped, so a later deliberate `outer.focus()` keeps the
        # outer body's documented "outside the list, wrap to the far end"
        # meaning. Decided synchronously here rather than in
        # `on_descendant_focus`, because Textual delivers that event after
        # this call returns -- a flag set around `focus()` is already reset by
        # the time the handler reads it.
        self._last_boundary_index = index if target is outer else None
        target.focus()
        # Focusing a descendant may make Textual reveal that control and push
        # its external heading off the top edge. Header visibility is the
        # navigation contract, so apply it after the focus target is settled.
        outer.scroll_to_widget(
            header,
            animate=False,
            immediate=True,
            force=True,
            top=True,
        )
        self.call_after_refresh(
            self._reveal_boundary_header,
            outer,
            header,
            target,
            generation,
            2,
        )

    def _reveal_boundary_header(
        self,
        outer: VerticalScroll,
        header: Widget,
        target: Widget,
        generation: int,
        remaining_passes: int,
    ) -> None:
        """Repeat a generation-guarded reveal after focus/layout commits."""

        if remaining_passes <= 0 or not self._header_reveal_is_current(
            outer, header, target, generation
        ):
            return
        outer.scroll_to(
            y=max(
                0,
                outer.scroll_y + header.region.y - outer.content_region.y,
            ),
            animate=False,
            immediate=True,
            force=True,
        )
        if remaining_passes > 1:
            self.call_after_refresh(
                self._reveal_boundary_header,
                outer,
                header,
                target,
                generation,
                remaining_passes - 1,
            )

    def _header_reveal_is_current(
        self,
        outer: VerticalScroll,
        header: Widget,
        target: Widget,
        generation: int,
    ) -> bool:
        """Return whether a delayed reveal still belongs to the active navigation."""

        return (
            generation == self._navigation_generation
            and outer.is_mounted
            and header.is_mounted
            and target.is_mounted
            and self.app.focused is target
        )

    def _body_controls(self, section: ConsoleBoundedSection) -> tuple[Widget, ...]:
        return tuple(
            widget
            for widget in section.viewport.query("*")
            if isinstance(widget, Widget) and self._is_enabled_focus_target(widget)
        )

    def _install_focus_recovery_callbacks(self) -> None:
        for section in self.query(ConsoleBoundedSection):
            section._on_focus_recovery = lambda owned=section: (
                self.recover_section_focus(owned)
            )

    def recover_section_focus(self, section: ConsoleBoundedSection) -> None:
        """Recover next, previous, external header, outer body, then collapse."""

        focused = self.app.focused
        if isinstance(focused, Widget) and self._is_enabled_focus_target(focused):
            if (
                focused is not section.viewport
                and section.viewport not in focused.ancestors
            ):
                if not self.inspector_active(focused):
                    self._section_focus_history.pop(section.section_id, None)
                return
        previous, controls = self._section_focus_history.get(
            section.section_id, (section.viewport, self._body_controls(section))
        )
        if not previous.is_attached:
            self._schedule_removed_focus_recovery(
                section.section_id, previous, controls
            )
            self.request_outer_reconcile()
            return
        incident = self._focus_recovery_incident(previous, controls)
        self._section_focus_history.pop(section.section_id, None)
        self._recover_focus_incident(section.section_id, incident)

    @staticmethod
    def _stable_widget_id(widget: Widget) -> str | None:
        """Return a usable semantic identity for a replaceable focus target."""

        return widget.id or None

    def _focus_recovery_incident(
        self,
        previous: Widget,
        old_controls: tuple[Widget, ...],
    ) -> _FocusRecoveryIncident:
        """Convert widget history into a detached-reference-free snapshot."""

        return _FocusRecoveryIncident(
            target_id=self._stable_widget_id(previous),
            target_index=(
                old_controls.index(previous) if previous in old_controls else None
            ),
        )

    def _recover_focus_incident(
        self,
        section_id: str,
        incident: _FocusRecoveryIncident,
    ) -> None:
        """Recover one semantic incident against the current mounted boundary."""

        boundaries = self._mounted_boundaries()
        boundary = next(
            (item for item in boundaries if item[0].section_id == section_id),
            None,
        )
        if boundary is not None:
            section, header = boundary
            current_controls = self._body_controls(section)
            controls_by_id = {
                stable_id: control
                for control in current_controls
                if (stable_id := self._stable_widget_id(control)) is not None
            }
            candidate_ids: list[str | None] = [incident.target_id]
            if incident.target_index is not None:
                index = min(incident.target_index, len(current_controls))
                candidate_ids.extend(
                    self._stable_widget_id(control)
                    for control in (
                        current_controls[index:]
                        + tuple(reversed(current_controls[:index]))
                    )
                )
            else:
                candidate_ids.extend(
                    self._stable_widget_id(control) for control in current_controls
                )

            seen_ids: set[str] = set()
            for candidate_id in candidate_ids:
                if candidate_id is None or candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate_id)
                candidate = controls_by_id.get(candidate_id)
                if candidate is None:
                    continue
                if self._is_enabled_focus_target(candidate):
                    self._section_focus_history[section_id] = (
                        candidate,
                        current_controls,
                    )
                    candidate.focus()
                    return

            for candidate in (header, *header.query("*")):
                if isinstance(candidate, Widget) and self._is_enabled_focus_target(
                    candidate
                ):
                    candidate.focus()
                    return
        for selector in (
            "#console-inspector-rail-body",
            "#console-inspector-rail-collapse",
        ):
            try:
                candidate = self.query_one(selector, Widget)
            except (NoMatches, QueryError):
                continue
            if self._is_enabled_focus_target(candidate):
                candidate.focus()
                return

    def _reconcile_focus_recovery_state(self) -> None:
        """Map retained focus history onto the current mounted boundaries."""

        boundaries = {
            section.section_id: section
            for section, _header in self._mounted_boundaries()
        }
        for section_id, (previous, old_controls) in tuple(
            self._section_focus_history.items()
        ):
            section = boundaries.get(section_id)
            if (
                section is not None
                and previous.is_attached
                and (
                    previous is section.viewport
                    or section.viewport in previous.ancestors
                )
            ):
                self._section_focus_history[section_id] = (
                    previous,
                    self._body_controls(section),
                )
                continue

            focused = self.app.focused
            if (
                isinstance(focused, Widget)
                and self._is_enabled_focus_target(focused)
                and not self.inspector_active(focused)
            ):
                self._section_focus_history.pop(section_id, None)
                continue
            self._schedule_removed_focus_recovery(section_id, previous, old_controls)

        focused = self.app.focused
        outside_focus_is_valid = bool(
            isinstance(focused, Widget)
            and self._is_enabled_focus_target(focused)
            and not self.inspector_active(focused)
        )
        for section_id, incident in tuple(self._pending_focus_recoveries.items()):
            if outside_focus_is_valid:
                self._pending_focus_recoveries.pop(section_id, None)
                continue
            section = boundaries.get(section_id)
            boundary_unsettled = section is None or not self._body_controls(section)
            if boundary_unsettled and incident.remaining_reconcile_passes > 0:
                self._pending_focus_recoveries[section_id] = _FocusRecoveryIncident(
                    target_id=incident.target_id,
                    target_index=incident.target_index,
                    remaining_reconcile_passes=(
                        incident.remaining_reconcile_passes - 1
                    ),
                )
                self.request_outer_reconcile()
                continue
            self._pending_focus_recoveries.pop(section_id, None)
            self._recover_focus_incident(section_id, incident)

    def _recover_keyed_boundary_focus(self, section_id: str | None) -> None:
        """Focus a captured boundary after Textual commits recompose fallback."""

        focused = self.app.focused
        if (
            isinstance(focused, Widget)
            and self._is_enabled_focus_target(focused)
            and not self.inspector_active(focused)
        ):
            return
        self._recover_focus_incident(
            section_id or "",
            _FocusRecoveryIncident(target_id=None, target_index=None),
        )

    def _paint_scroll_owner(self, focused: Widget | None) -> None:
        active_header: Widget | None = None
        outer_active = bool(
            focused is not None and focused.id == "console-inspector-rail-body"
        )
        if focused is not None:
            boundaries = self._mounted_boundaries()
            index = self._boundary_index_for_target(focused, boundaries)
            if index is not None:
                section, header = boundaries[index]
                if focused is section.viewport or section.viewport in focused.ancestors:
                    active_header = header
        headers = [header for _section, header in self._mounted_boundaries()]
        try:
            collapse = self.query_one("#console-inspector-rail-collapse", Button)
        except (NoMatches, QueryError):
            collapse = None
        for header in (*headers, collapse):
            if header is None:
                continue
            active = header is active_header or (header is collapse and outer_active)
            header.set_class(active, INSPECTOR_SCROLL_OWNER_CLASS)

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """React to focus landing anywhere inside this rail.

        Does two things. It ends `n`/`p` navigation state once focus leaves
        the parked outer scroller, and it recovers the previous local focus
        order when a focused child was removed by a recompose rather than by
        the user.

        Args:
            event: The focus event. Its ``widget`` is the descendant that
                just gained focus, and is the only signal used -- deciding
                from a flag set around ``focus()`` does not work, because
                Textual delivers this message asynchronously and the flag is
                already reset by the time it arrives.
        """

        target = event.widget
        # TASK-24704 (Qodo #5, second round): the boundary anchor is only
        # meaningful while the outer scroller holds focus BECAUSE navigation
        # parked it there. Any focus landing anywhere else ends that state --
        # including an ordinary Tab away. Without this, Tabbing off the
        # scroller and back left the anchor set, so the next `n`/`p`
        # continued from stale section history instead of the outer body's
        # documented first/last-boundary behaviour.
        #
        # Driven by the focus EVENT, not by a flag set around `focus()`: a
        # flag is already reset by the time Textual delivers this message,
        # which is how an earlier attempt at the same guard failed. A park
        # focuses the scroller itself, so "focused something else" is exactly
        # the signal that navigation is over.
        try:
            outer_body = self.query_one("#console-inspector-rail-body")
        except (NoMatches, QueryError):
            outer_body = None
        if outer_body is None or target is not outer_body:
            self._last_boundary_index = None
        # Removing a focused child can make Textual choose the next app-wide
        # focusable before the section's post-refresh recovery runs. Preserve
        # the old local order when that incidental target is still in this
        # Inspector; an intentional, valid focus outside the rail is left alone.
        for section_id, (previous, controls) in tuple(
            self._section_focus_history.items()
        ):
            if (
                previous is not None
                and previous is not target
                and not previous.is_attached
            ):
                self._schedule_removed_focus_recovery(section_id, previous, controls)
                break
        if not self._pending_focus_recoveries:
            self._section_focus_history.clear()
        for section, _header in self._mounted_boundaries():
            if target is section.viewport or section.viewport in target.ancestors:
                incident = self._pending_focus_recoveries.get(section.section_id)
                if (
                    incident is not None
                    and self._stable_widget_id(target) != incident.target_id
                ):
                    break
                self._pending_focus_recoveries.pop(section.section_id, None)
                self._section_focus_history[section.section_id] = (
                    target,
                    self._body_controls(section),
                )
                break
        self._paint_scroll_owner(target)
        self._set_inspector_focus_active(True)

    def _recover_removed_focus_snapshot(
        self,
        section_id: str,
        incident: _FocusRecoveryIncident,
    ) -> None:
        """Recover after Textual's incidental focus without losing old order."""

        if self._pending_focus_recoveries.get(section_id) != incident:
            return
        focused = self.app.focused
        if (
            isinstance(focused, Widget)
            and self._is_enabled_focus_target(focused)
            and not self.inspector_active(focused)
        ):
            self._pending_focus_recoveries.pop(section_id, None)
            return
        self.request_outer_reconcile()

    def _schedule_removed_focus_recovery(
        self,
        section_id: str,
        removed_target: Widget,
        old_controls: tuple[Widget, ...],
    ) -> None:
        """Schedule at most one recovery for a detached focus incident."""

        if section_id in self._pending_focus_recoveries:
            return
        incident = self._focus_recovery_incident(removed_target, old_controls)
        self._section_focus_history.pop(section_id, None)
        self._pending_focus_recoveries[section_id] = incident
        self.call_after_refresh(
            self._recover_removed_focus_snapshot,
            section_id,
            incident,
        )

    def on_focus(self) -> None:
        """Refresh contextual shortcuts when the rail root itself is focused."""

        self._set_inspector_focus_active(True)

    def on_blur(self) -> None:
        """Defer root-focus exit truth until replacement focus is committed."""

        self.call_after_refresh(self._finish_descendant_blur)

    def on_descendant_blur(self, _event: DescendantBlur) -> None:
        self.call_after_refresh(self._finish_descendant_blur)

    def _finish_descendant_blur(self) -> None:
        active = self.inspector_active()
        if not active:
            self._paint_scroll_owner(None)
            self._section_focus_history.clear()
            self._pending_focus_recoveries.clear()
        self._set_inspector_focus_active(active)

    def _set_inspector_focus_active(self, active: bool) -> None:
        if active == self._inspector_focus_active:
            return
        self._inspector_focus_active = active
        refresh_footer = getattr(
            self.screen, "_register_console_footer_shortcuts", None
        )
        if callable(refresh_footer):
            refresh_footer()

    def compose(self) -> ComposeResult:
        """Compose the rail header, staged-context tray, scope row, and run inspector.

        Returns:
            The rail-header row followed by the scrollable body: the staged
            Context tray, the retrieval-Scope row, the run inspector plus
            session settings summary, and the live-work status card, in
            mount order.
        """
        right_rail_header = Horizontal(
            id="console-inspector-rail-header", classes="console-rail-header"
        )
        right_rail_header.styles.height = 1
        right_rail_header.styles.min_height = 1
        right_rail_header.styles.max_height = 1
        with right_rail_header:
            # TASK-23195 follow-up: mirrors the Context rail's header. Both
            # read as a name plus one resolved glyph, and both put the glyph
            # on the edge ADJACENT to the transcript pointing outward, so the
            # affordance says which way the rail leaves. The former
            # "Inspect|--------->" was hard-coded ASCII art that bypassed the
            # `ascii_glyphs` fallback every other Console glyph routes
            # through, and spent most of the rail's width on the arrow.
            collapse_button = Button(
                f"{resolve_glyph(GLYPH_COLLAPSE_RIGHT)} Inspect",
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

        project_instruction_row = ConsoleProjectInstructionStatusRow(
            self._project_instruction_state
        )
        project_instruction_row.styles.width = "100%"
        project_instruction_row.styles.height = 1
        yield project_instruction_row

        yield ConsoleSendAuthoritySummary(self._inspector_state)

        with _InspectorOuterBody(
            on_geometry_changed=self._request_outer_geometry_reconcile,
            on_scrolled=self._handle_outer_scrolled,
        ):
            # task-9: Environment and Tasks sections -- mounted FIRST, ahead
            # of the staged-context tray, per the redesign spec's rail
            # ordering. Pure display state supplied by the screen (a
            # projection of the controller's current snapshot, or the
            # empty-state projection of a default `EnvironmentSnapshot()`
            # pre-wiring); this rail never gathers environment data itself.
            # Each section hides itself (`styles.display = "none"`) when its
            # projection has no rows -- the fleet pattern this rail already
            # uses for the live-work header Statics above.
            environment_section = ConsoleInspectorSection(
                title="Environment",
                section_id=ENVIRONMENT_SECTION_ID,
                rows=self._environment_section_state.rows,
                summary=self._environment_section_state.summary,
                collapsible=True,
                open=self._environment_open,
                view_all_label="Refresh",
                id="console-environment-section",
            )
            environment_section.styles.display = (
                "block" if self._environment_section_state.rows else "none"
            )
            yield environment_section

            tasks_section = ConsoleInspectorSection(
                title="Tasks",
                section_id=TASKS_SECTION_ID,
                rows=self._tasks_section_state.rows,
                summary=self._tasks_section_state.summary,
                collapsible=True,
                open=self._tasks_open,
                id="console-tasks-section",
            )
            tasks_section.styles.display = (
                "block" if self._tasks_section_state.rows else "none"
            )
            yield tasks_section

            # task-10: the Agent rail's fleet mini-section, moved here from
            # the left rail's Agent section (it stays ordinary content --
            # its own `ConsoleInspectorSection` header/body handle its own
            # collapse; no bounded viewport wraps it). Same unconditional-
            # mount-hide-when-empty pattern as Environment/Tasks above; the
            # widget id, section id, and default-collapsed `open=False`
            # are all unchanged from the left rail so the sync path
            # (`ChatScreen._sync_console_agent_section`) and existing tests
            # keep querying by the same id without change.
            fleet_section = ConsoleInspectorSection(
                title="Agents",
                section_id=CONSOLE_AGENT_FLEET_SECTION_ID,
                rows=self._agent_fleet_section_state.rows,
                summary=self._agent_fleet_section_state.summary,
                collapsible=True,
                open=False,
                id="console-agent-section-subagents",
            )
            fleet_section.styles.display = (
                "block" if self._agent_fleet_section_state.rows else "none"
            )
            yield fleet_section

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
                on_reconcile=self.request_outer_reconcile,
                id="console-staged-context-tray",
                classes="console-inspector-context-section",
            )
            staged_context_tray.styles.width = "100%"
            staged_context_tray.styles.min_width = 0
            staged_context_tray.styles.height = "auto"
            # `ChatScreen._staged_context_frame_variant` is a `@staticmethod`
            # returning "quiet" unconditionally (mirrors task 3's inlining
            # of `_workspace_context_frame_variant`) -- inlined as a literal
            # here rather than passed as data.
            yield frame_console_region(staged_context_tray, variant="quiet")

            # TASK-24611: the Library search controls, directly beneath the
            # tray whose empty state says "Stage sources from Library." They
            # used to be the first children of the live-work readiness card
            # at the BOTTOM of the rail -- the one control useful to someone
            # with nothing staged, ~25 rows below the sentence telling them
            # to go and stage something, under a heading naming a status
            # inventory. Placed here rather than swapping whole sections so
            # the readiness card keeps the bottom anchor task-400 chose for
            # it, and run state keeps its place above the fold.
            if self._library_search_builder is not None:
                yield self._library_search_builder()

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

            with Vertical(id="console-run-inspector"):
                yield ConsoleRunInspector(
                    self._inspector_state,
                    ownership_policy=self._ownership_policy,
                    reported_unknown_fingerprints=(self._reported_unknown_fingerprints),
                    on_reconcile=self.request_outer_reconcile,
                    on_more_focus_removed=self._recover_keyed_boundary_focus,
                    more_open=self._inspector_more_open,
                    id="console-run-inspector-state",
                )
                yield ConsoleSelectedTurnActivity(
                    self._library_activity_view,
                    citation_count=self._library_activity_citation_count,
                    flush_result=self._library_activity_flush_result,
                    on_reconcile=self.request_outer_reconcile,
                )
                settings_summary = ConsoleSettingsSummary(
                    self._settings_summary_state,
                    on_reconcile=self.request_outer_reconcile,
                    id="console-settings-summary",
                    classes=(
                        "console-inspector-session-settings console-settings-summary"
                    ),
                )
                settings_summary.styles.width = "100%"
                settings_summary.styles.min_width = 0
                yield settings_summary

            live_work_card = self._live_work_card_builder()
            pending_visible = live_work_card.id == PENDING_LAUNCH_CARD_ID
            with Vertical(id="console-live-work-section"):
                with Vertical(id="console-live-work-header"):
                    pending_header = Static(
                        "Pending Console launch",
                        id="console-live-work-status-badge",
                        classes="ds-status-badge console-live-work-status-badge",
                    )
                    pending_header.display = pending_visible
                    yield pending_header
                    readiness_header = Static(
                        "Live work sources",
                        id="console-live-work-source-readiness-title",
                        classes=(
                            "ds-status-badge console-live-work-source-readiness-title"
                        ),
                    )
                    readiness_header.display = not pending_visible
                    yield readiness_header
                yield ConsoleBoundedSection(
                    live_work_card,
                    section_id="live-work",
                )
        outer_hint = Static(
            "",
            id=INSPECTOR_OUTER_HINT_ID,
            classes="console-inspector-outer-scroll-hint",
            markup=False,
        )
        outer_hint.can_focus = False
        outer_hint.display = False
        outer_hint.styles.height = 1
        outer_hint.styles.min_height = 1
        outer_hint.styles.max_height = 1
        yield outer_hint
