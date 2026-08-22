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
import os

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.events import DescendantBlur, DescendantFocus, Key, Resize
from textual.geometry import Size
from textual.widget import Widget
from textual.widgets import Button, Input, Static, TextArea

from ...Chat.console_display_state import (
    ConsoleInspectorState,
    ConsoleProjectInstructionState,
    ConsoleRetrievalScopeState,
    ConsoleStagedContextState,
)
from ...Chat.console_session_settings import ConsoleSettingsSummaryState
from ...Chat.console_live_work import PENDING_LAUNCH_CARD_ID
from ...Widgets.Console import (
    ConsoleChangedFilesSection,
    ConsoleChangedFilesState,
    ConsoleBoundedSection,
    InspectorOwnershipPolicy,
    ConsoleProjectInstructionStatusRow,
    ConsoleRetrievalScopeRow,
    ConsoleRunInspector,
    ConsoleSettingsSummary,
    ConsoleStagedContextTray,
)
from ...Widgets.Console.console_retrieval_scope_row import (
    ROW_ID as CONSOLE_RETRIEVAL_SCOPE_ROW_ID,
)
from .frame import frame_console_region
from .rail_section_layout import outer_hint_required


INSPECTOR_OUTER_HINT = "▼ more sections — scroll"
INSPECTOR_OUTER_HINT_ID = "console-inspector-outer-scroll-hint"
INSPECTOR_SCROLL_OWNER_CLASS = "console-inspector-scroll-owner"


class _InspectorOuterBody(VerticalScroll):
    """Inspector scroller that invalidates its owner on scroll and resize."""

    def __init__(self, *, on_geometry_changed: Callable[[], None]) -> None:
        super().__init__(
            id="console-inspector-rail-body",
            classes="console-inspector-rail-body",
            can_focus=True,
        )
        self._on_geometry_changed = on_geometry_changed

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if old_value != new_value:
            self._on_geometry_changed()

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
    """

    DEFAULT_CSS = """
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
        changed_files_state: ConsoleChangedFilesState,
        project_instruction_state: ConsoleProjectInstructionState,
        settings_summary_state: ConsoleSettingsSummaryState,
        live_work_card_builder: Callable[[], Widget],
        ownership_policy: InspectorOwnershipPolicy | None = None,
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
            ownership_policy: Optional explicit Inspector ownership policy.
                Production resolves the strict opt-in environment flag at
                this composition boundary when omitted.
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
        self._project_instruction_state = project_instruction_state
        self._settings_summary_state = settings_summary_state
        self._live_work_card_builder = live_work_card_builder
        self._ownership_policy = (
            ownership_policy or _resolve_inspector_ownership_policy()
        )
        self._reported_unknown_fingerprints: set[tuple[str, ...]] = set()
        self._outer_reconcile_scheduled = False
        self._outer_reconcile_dirty = False
        self._outer_reconcile_count = 0
        self._inspector_focus_active = False
        self._navigation_generation = 0
        self._section_focus_history: dict[str, tuple[Widget, tuple[Widget, ...]]] = {}
        self._pending_focus_recoveries: set[Widget] = set()

    def on_mount(self) -> None:
        """Schedule the first owner pass after descendant layout settles."""

        self.request_outer_reconcile()

    def request_outer_reconcile(self) -> None:
        """Coalesce Inspector owner invalidation behind local section demand."""

        if not self.is_mounted:
            return
        if self._outer_reconcile_scheduled:
            self._outer_reconcile_dirty = True
            return
        self._outer_reconcile_scheduled = True
        self._outer_reconcile_dirty = False
        self.call_after_refresh(self._run_scheduled_outer_reconcile)

    def _run_scheduled_outer_reconcile(self) -> None:
        """Reconcile the outer fold only after every local body has settled."""

        if not self.is_mounted:
            self._outer_reconcile_scheduled = False
            self._outer_reconcile_dirty = False
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

        if not self.is_mounted:
            self._outer_reconcile_scheduled = False
            self._outer_reconcile_dirty = False
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
        self._outer_reconcile_scheduled = False
        self._install_focus_recovery_callbacks()
        if not self._reconcile_outer_fold():
            return
        self._outer_reconcile_count += 1

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
            self.request_outer_reconcile()
            return False

        self._update_outer_hint()
        return True

    def _update_outer_hint(self) -> None:
        """Paint copy only while actual outer content remains below."""

        try:
            body = self.query_one("#console-inspector-rail-body", VerticalScroll)
            hint = self.query_one(f"#{INSPECTOR_OUTER_HINT_ID}", Static)
        except (NoMatches, QueryError):
            return
        if not hint.display:
            hint.update("")
            return
        hint.update(
            INSPECTOR_OUTER_HINT
            if body.max_scroll_y > 0 and body.scroll_y < body.max_scroll_y
            else ""
        )

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
    ) -> tuple[tuple[ConsoleBoundedSection, Widget, Widget], ...]:
        """Return visible direct boundaries as body, header, owning root."""

        boundaries: list[tuple[ConsoleBoundedSection, Widget, Widget]] = []
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
            boundaries.append((section, header, parent))
        return tuple(boundaries)

    def _boundary_index_for_target(
        self,
        target: Widget,
        boundaries: tuple[tuple[ConsoleBoundedSection, Widget, Widget], ...],
    ) -> int | None:
        for index, (section, header, _root) in enumerate(boundaries):
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
        boundaries: tuple[tuple[ConsoleBoundedSection, Widget, Widget], ...],
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
        for index, (_section, header, _root) in enumerate(boundaries):
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
        """Handle bubbling n/p section navigation only at the rail boundary."""

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
            boundaries[target_index], generation=self._navigation_generation
        )

    def _focus_boundary(
        self,
        boundary: tuple[ConsoleBoundedSection, Widget, Widget],
        *,
        generation: int,
    ) -> None:
        section, header, _root = boundary
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
        )

    def _reveal_boundary_header(
        self,
        outer: VerticalScroll,
        header: Widget,
        target: Widget,
        generation: int,
    ) -> None:
        """Repeat the reveal after Textual's focus scroll has committed."""

        if not self._header_reveal_is_current(outer, header, target, generation):
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
        self.call_after_refresh(
            self._finish_boundary_header_reveal,
            outer,
            header,
            target,
            generation,
        )

    def _finish_boundary_header_reveal(
        self,
        outer: VerticalScroll,
        header: Widget,
        target: Widget,
        generation: int,
    ) -> None:
        """Reveal once more after late focus-layout work, if navigation is current."""

        if not self._header_reveal_is_current(outer, header, target, generation):
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

    def recover_section_focus(
        self,
        section: ConsoleBoundedSection,
        *,
        replace_incidental_inside_rail: bool = False,
    ) -> None:
        """Recover next, previous, external header, outer body, then collapse."""

        focused = self.app.focused
        if isinstance(focused, Widget) and self._is_enabled_focus_target(focused):
            if (
                focused is not section.viewport
                and section.viewport not in focused.ancestors
            ):
                if not (
                    replace_incidental_inside_rail
                    and (focused is self or self in focused.ancestors)
                ):
                    return
        previous, controls = self._section_focus_history.get(
            section.section_id, (section.viewport, self._body_controls(section))
        )
        if previous in controls:
            index = controls.index(previous)
            candidates = controls[index + 1 :] + tuple(reversed(controls[:index]))
        else:
            candidates = controls
        for candidate in candidates:
            if self._is_enabled_focus_target(candidate):
                self._section_focus_history[section.section_id] = (
                    candidate,
                    self._body_controls(section),
                )
                candidate.focus()
                return
        self._section_focus_history.pop(section.section_id, None)
        boundaries = self._mounted_boundaries()
        boundary = next((item for item in boundaries if item[0] is section), None)
        if boundary is not None:
            for candidate in boundary[1].query("*"):
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

    def _paint_scroll_owner(self, focused: Widget | None) -> None:
        active_header: Widget | None = None
        outer_active = bool(
            focused is not None and focused.id == "console-inspector-rail-body"
        )
        if focused is not None:
            boundaries = self._mounted_boundaries()
            index = self._boundary_index_for_target(focused, boundaries)
            if index is not None:
                section, header, _root = boundaries[index]
                if focused is section.viewport or section.viewport in focused.ancestors:
                    active_header = header
        headers = [header for _section, header, _root in self._mounted_boundaries()]
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
        target = event.widget
        # Removing a focused child can make Textual choose the next app-wide
        # focusable before the section's post-refresh recovery runs. Preserve
        # the old local order when that incidental target is still in this
        # Inspector; an intentional, valid focus outside the rail is left alone.
        for section, _header, _root in self._mounted_boundaries():
            previous, controls = self._section_focus_history.get(
                section.section_id, (None, ())
            )
            if (
                previous is not None
                and previous is not target
                and previous in controls
                and not previous.is_attached
            ):
                self._schedule_removed_focus_recovery(section, previous)
                break
        for section, _header, _root in self._mounted_boundaries():
            if target is section.viewport or section.viewport in target.ancestors:
                previous, controls = self._section_focus_history.get(
                    section.section_id, (None, ())
                )
                if (
                    previous is not None
                    and previous is not target
                    and previous in controls
                    and not previous.is_attached
                ):
                    self._schedule_removed_focus_recovery(section, previous)
                else:
                    self._section_focus_history[section.section_id] = (
                        target,
                        self._body_controls(section),
                    )
                break
        self._paint_scroll_owner(target)
        self._set_inspector_focus_active(True)

    def _recover_removed_focus_snapshot(
        self,
        section: ConsoleBoundedSection,
        removed_target: Widget,
    ) -> None:
        """Recover after Textual's incidental focus without losing old order."""

        self._pending_focus_recoveries.discard(removed_target)
        current, _controls = self._section_focus_history.get(
            section.section_id, (None, ())
        )
        if current is removed_target and not removed_target.is_attached:
            self.recover_section_focus(
                section,
                replace_incidental_inside_rail=True,
            )

    def _schedule_removed_focus_recovery(
        self,
        section: ConsoleBoundedSection,
        removed_target: Widget,
    ) -> None:
        """Schedule at most one recovery for a detached focus incident."""

        if removed_target in self._pending_focus_recoveries:
            return
        self._pending_focus_recoveries.add(removed_target)
        self.call_after_refresh(
            self._recover_removed_focus_snapshot,
            section,
            removed_target,
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

        with _InspectorOuterBody(on_geometry_changed=self.request_outer_reconcile):
            project_instruction_row = ConsoleProjectInstructionStatusRow(
                self._project_instruction_state
            )
            project_instruction_row.styles.width = "100%"
            project_instruction_row.styles.height = 1
            yield project_instruction_row

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
                on_reconcile=self.request_outer_reconcile,
            )
            # Same margin/padding rhythm as the Sources tray and Scope row
            # above (`.console-inspector-context-section`) -- the widget's
            # own fixed `console-changed-files-section` class is untouched,
            # this is additive, matching how `frame_console_region` below
            # also adds a class rather than replacing the widget's own.
            # Width/min-width live in `_agentic_terminal.tcss`'s
            # `#console-changed-files-section` rule (TASK-19192), matching
            # the sibling id-rule convention -- not inline here.
            changed_files_section.add_class("console-inspector-context-section")
            yield frame_console_region(changed_files_section, variant="quiet")

            with Vertical(id="console-run-inspector"):
                yield ConsoleRunInspector(
                    self._inspector_state,
                    ownership_policy=self._ownership_policy,
                    reported_unknown_fingerprints=(self._reported_unknown_fingerprints),
                    on_reconcile=self.request_outer_reconcile,
                    id="console-run-inspector-state",
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
