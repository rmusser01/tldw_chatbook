"""Console shell left rail — persistent Console context sections.

Extracted verbatim out of ``ChatScreen.compose_content`` (wave-1 console
decomposition, task 3): the subtree that used to live inside
``with self._frame_console_region(left_rail):`` — the rail header, the
pinned fleet-summary line, and the scrollable context sections. The former
mixed Session section is now three peer sections (Sessions, Workspaces, and
Conversations), followed by Model, Agent, Details, and Character.

Deliberately NOT included: ``left_handle`` (``ConsoleRailHandle``, id
``console-context-rail-handle``), the collapsed 13-column form shown when
this rail is closed. It stays a direct sibling of this widget under
``#console-workspace-grid`` in ``chat_screen.py``, exactly as it was a
sibling of the old plain ``Vertical`` this widget replaces. The two are
mutually exclusive (only one is ever visible), and today that exclusivity
is expressed as two ``Horizontal``-arranged siblings with independent
width rules — one fixed (13 columns), one fractional (``3fr``). Folding
both under one parent would require that parent to switch its own width
between those two unit kinds depending on open/closed state, which is a
real layout risk this extraction does not take (see the design spec's
"Migration safety" section on this exact class of defect). This class
reuses ``id="console-left-rail"`` as its own root, so it sits in the DOM
exactly where the old ``Vertical`` sat. Direct domain body IDs remain
stable descendants inside their presentation-only bounded wrappers.

Also NOT moved: the rail-state machinery whose body reaches into the
right (Inspector) rail or into app-wide persisted config —
``_console_rail_state_config``, ``_console_rail_available_columns``,
``_toggle_console_rail_section``, and the system-line text builder
(``_console_rail_system_line_state``, which depends on session-settings
resolution, not rail DOM). Those stay on ``ChatScreen``; the screen still
computes their outputs and passes them in as data. What DID move —
because its body touches only this rail's own ids —
is the section open/close DOM sync (``sync_sections`` /
``apply_section_open``, formerly ``_sync_console_rail_sections`` /
``_apply_console_rail_section_open``), and the section-toggle button
press is now caught here and re-posted as a typed ``SectionToggled``
message rather than matched by id prefix in the screen's
``on_button_pressed``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.events import DescendantBlur, DescendantFocus, MouseDown, MouseUp
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from ...Chat.console_rail_state import CONSOLE_RAIL_SECTION_IDS, ConsoleRailState
from ...Chat.console_session_settings import (
    ConsoleSettingsSummaryState,
    _summary_row_value,
)
from ...Widgets.Console import ConsoleBoundedSection, ConsoleWorkspaceContextTray
from ...Widgets.Console.console_agent_steering_bar import (
    STEERING_BAR_ID,
    ConsoleAgentSteeringBar,
    ConsoleAgentSteeringState,
)
from ...Widgets.Console.console_image_viewer_modal import ClickableAvatarBox
from ...Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionState,
)
from ...Widgets.Console.console_workspace_details import ConsoleWorkspaceDetailsTray
from ...Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
    DestinationRailSectionHeader,
)
from ...Workspaces.display_state import ConsoleWorkspaceContextState
from .agent import CONSOLE_AGENT_CANCEL_ALL_ID, CONSOLE_AGENT_FLEET_SECTION_ID
from .frame import frame_console_region
from .rail_section_layout import (
    ContextAllocationResult,
    ContextSectionDemand,
    allocate_context_sections,
    fallback_active_section,
    outer_hint_required,
)


OUTER_SECTION_SCROLL_HINT = "▼ more sections — scroll"


@dataclass(frozen=True, slots=True)
class ContextSectionDescriptor:
    """Stable direct Context-section identity in allocator/DOM order."""

    section_id: str
    title: str


CONTEXT_SECTION_DESCRIPTORS = (
    ContextSectionDescriptor("session", "Sessions"),
    ContextSectionDescriptor("workspace", "Workspaces"),
    ContextSectionDescriptor("conversations", "Conversations"),
    ContextSectionDescriptor("model", "Model"),
    ContextSectionDescriptor("agent", "Agent"),
    ContextSectionDescriptor("details", "Details"),
    ContextSectionDescriptor("character", "Character"),
)


class _ContextOuterBody(VerticalScroll):
    """Context scroll owner with direct resize/scroll invalidation seams."""

    def __init__(self, *, owner: "ConsoleLeftRail", **kwargs) -> None:
        super().__init__(**kwargs)
        self._owner = owner

    def on_resize(self) -> None:
        self._owner.request_allocation_reconcile()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if old_value != new_value:
            self._owner._update_outer_hint()


class _ContextBoundedSection(ConsoleBoundedSection):
    """Direct Context body that invalidates its allocator when demand changes."""

    def __init__(
        self,
        *content: Widget,
        section_id: str,
        owner: "ConsoleLeftRail",
    ) -> None:
        self._allocation_owner = owner
        super().__init__(
            *content,
            section_id=section_id,
            on_focus_recovery=lambda: owner.recover_section_focus(section_id),
        )

    def _run_scheduled_reconcile(self) -> None:
        previous_demand = self.desired_content_lines
        super()._run_scheduled_reconcile()
        if self.desired_content_lines != previous_demand:
            self._allocation_owner.request_allocation_reconcile()


class ConsoleLeftRail(Vertical):
    """The Console shell's left rail and its persistent context sections.

    All display data is supplied at construction, computed by the screen
    exactly as it was computed inline before this extraction (session
    settings resolution, the agent bridge, character avatar rendering, and
    Console rail preference persistence all remain screen-owned concerns —
    this widget only renders the results). Nothing here reaches
    ``app_instance``.
    """

    class SectionToggled(Message):
        """A rail section's toggle button was pressed by the user.

        The screen still owns the actual open/closed decision
        (``ChatScreen._toggle_console_rail_section``, which also persists
        the preference and can touch the Inspector rail's auto-open rule)
        — this message only reports that this rail's own toggle button
        fired, so the screen never has to match a button id prefix out of
        an unrelated ``on_button_pressed`` if-chain.
        """

        def __init__(self, section_id: str, opened: bool) -> None:
            """Record which section toggled and this rail's last-synced open state for it.

            Args:
                section_id: The toggled section's id (e.g. ``"session"``).
                opened: The mounted header's intended next open state. The
                    screen persists this exact gesture after transient
                    activation, avoiding a same-tick state-sync race.
            """
            self.section_id = section_id
            self.opened = opened
            super().__init__()

    class ReactionPickerRequested(Message):
        """The Character section asked its owning screen to open reactions."""

    def __init__(
        self,
        *,
        rail_state: ConsoleRailState,
        workspace_context_state: ConsoleWorkspaceContextState,
        settings_summary_state: ConsoleSettingsSummaryState,
        system_line_text: str,
        system_line_dim: bool,
        fleet_line: str,
        agent_status_line: str,
        agent_steps_text: str,
        agent_fleet_section_state: ConsoleInspectorSectionState,
        agent_drilldown_active: bool,
        agent_full_log_available: bool,
        agent_steering_state: ConsoleAgentSteeringState | None = None,
        agent_cancel_all_visible: bool = False,
        show_character_section: bool,
        character_avatar_widget_builder: Callable[[], Widget] | None,
        character_avatar_name: str,
        manual_reaction_label: str | None = None,
        **kwargs,
    ) -> None:
        """Create the left rail from pre-computed display data.

        Args:
            rail_state: Effective Console rail state (open flags, labels).
            workspace_context_state: Shared workspace/conversation display
                state, projected into the Sessions, Workspaces, Conversations,
                and Details trays.
            settings_summary_state: Console session settings summary, parsed
                here into the Model section's provider/model/temperature/
                max-tokens/readiness rows (pure formatting, moved verbatim
                from the old inline compose).
            system_line_text: The Model section's ``System: <preview>``
                line text.
            system_line_dim: Whether that line renders in its dim/unset
                style.
            fleet_line: The pinned fleet-summary line text; empty hides it.
            agent_status_line: Agent section status text.
            agent_steps_text: Agent section step list text.
            agent_fleet_section_state: Rows + header summary (PR2b Task 4,
                spec §7 states 1/2) for the ``ConsoleInspectorSection`` that
                renders the conversation's own sub-agent fleet -- computed
                by ``ConsoleAgentController._console_agent_fleet_section_
                state``. Empty rows/summary hides the section entirely
                (nothing to show, or a sub-agent drill-down is active and
                the status/steps Statics already carry that one child's
                own detail -- state 3, unchanged).
            agent_drilldown_active: Whether a sub-agent drill-down is
                active, driving the "Back" button's visibility.
            agent_full_log_available: Whether the "View full log" button
                should be visible.
            agent_steering_state: The drill-in steering bar's state (PR3b
                Task 3), computed by ``ConsoleAgentController._console_
                agent_steering_state`` -- visible only while drilled into
                a LIVE child. Passed at construction for the same reason
                ``agent_drilldown_active`` is: a rail recompose while
                drilled in must paint the bar correctly immediately,
                without waiting for the next equality-guarded sync tick.
                ``None`` (bare test constructions) means hidden.
            agent_cancel_all_visible: Whether the "Cancel all agents"
                button paints (PR3b Task 5) -- true only while the
                conversation has a LIVE child, computed by
                ``ConsoleAgentController._console_agent_cancel_all_
                visible``. Passed at construction for the same
                recompose-mid-state reason as ``agent_steering_state``;
                the default keeps bare test constructions valid (hidden).
            show_character_section: Whether the Character section is
                composed at all (config-gated; matches
                ``resolve_show_character_avatar``).
            character_avatar_widget_builder: Zero-arg callable that builds
                the avatar widget, when ``show_character_section`` is True
                (``None`` otherwise). The screen still owns
                ``_build_character_avatar_widget`` and the spec it reads
                (``self._active_character_avatar``); only the CALL is
                deferred to this rail's own ``compose()``. A builder, not a
                pre-built instance, is the point:
                ``_render_character_avatar_into_section`` replaces this
                widget by re-querying ``#console-character-avatar`` and
                remounting directly, without going through this rail at
                all, so a stored widget INSTANCE here would go stale the
                moment anything recomposes this rail (see the design spec's
                region-widget rule) -- re-yielding a widget Textual already
                removed from the DOM. Calling the builder fresh on every
                ``compose()`` (matching ``ConsoleDictationController``'s
                late-binding constructor rule -- see ``dictation.py``'s
                module docstring) always mounts a brand-new instance built
                from the CURRENT ``self._active_character_avatar`` instead.
            character_avatar_name: Character name label text, when
                ``show_character_section`` is True.
            manual_reaction_label: Active session-local manual reaction label,
                or ``None`` while operational reactions remain automatic.
            kwargs: Forwarded to ``Vertical``.
        """
        super().__init__(
            id="console-left-rail",
            classes="console-region destination-workbench-pane",
            **kwargs,
        )
        self._rail_state = rail_state
        self._workspace_context_state = workspace_context_state
        self._settings_summary_state = settings_summary_state
        self._system_line_text = system_line_text
        self._system_line_dim = system_line_dim
        self._fleet_line = fleet_line
        self._agent_status_line = agent_status_line
        self._agent_steps_text = agent_steps_text
        self._agent_fleet_section_state = agent_fleet_section_state
        self._agent_drilldown_active = agent_drilldown_active
        self._agent_full_log_available = agent_full_log_available
        self._agent_steering_state = agent_steering_state
        self._agent_cancel_all_visible = agent_cancel_all_visible
        self._show_character_section = show_character_section
        self._character_avatar_widget_builder = character_avatar_widget_builder
        self._character_avatar_name = character_avatar_name
        self._manual_reaction_label = str(manual_reaction_label or "").strip()
        self._active_section_id: str | None = None
        self._allocation_reconcile_scheduled = False
        self._last_allocation_state: (
            tuple[
                ContextAllocationResult,
                bool,
                tuple[ConsoleBoundedSection, ...],
                str | None,
            ]
            | None
        ) = None
        self._active_transition_generation = 0
        self._pending_active_reveal_generation: int | None = None
        self._active_reveal_token = 0
        self._no_room_section_ids: frozenset[str] = frozenset()
        self._outer_hint_exists = False
        self._outer_hint_text = ""
        self._section_focus_history: dict[str, tuple[Widget, tuple[Widget, ...]]] = {}
        self._pointer_activation_pending: str | None = None
        self._pointer_activation_waits_for_button = False
        self._pointer_activation_target: Widget | None = None
        self._pointer_activation_generation = 0

    @staticmethod
    def _section_header(
        section_id: str,
        is_open: bool,
    ) -> DestinationRailSectionHeader:
        """Build a direct header from the stable descriptor title source."""

        descriptor = next(
            item
            for item in CONTEXT_SECTION_DESCRIPTORS
            if item.section_id == section_id
        )
        return DestinationRailSectionHeader(
            descriptor.title,
            section_id=section_id,
            open=is_open,
            id=f"console-rail-section-header-{section_id}",
        )

    @staticmethod
    def _section_body(
        section_id: str,
        is_open: bool,
        *children: Widget,
        classes: str = "",
    ) -> Vertical:
        """Build the preserved domain body as bounded presentation content."""

        body_classes = "console-rail-section-body"
        if classes:
            body_classes = f"{body_classes} {classes}"
        body = Vertical(
            *children,
            id=f"console-rail-section-body-{section_id}",
            classes=body_classes,
        )
        body.styles.height = "auto"
        # Retire the legacy descendant body's independent 20% scroll owner
        # inline; Task 8 owns the stylesheet cleanup itself.
        # A viewport-relative ceiling defeats the legacy 20% selector without
        # obscuring the only distinction the allocator needs above its 20-row
        # cap: whether physical demand still exceeds 20 rows.
        body.styles.max_height = "100vh"
        body.styles.overflow_y = "hidden"
        body.styles.margin_bottom = 0
        if not is_open:
            body.styles.display = "none"
        return body

    def on_mount(self) -> None:
        """Allocate from the first complete mounted geometry snapshot."""

        self.request_allocation_reconcile()

    def on_resize(self) -> None:
        """Reallocate when the outer Context region changes height."""

        self.request_allocation_reconcile()

    def _mounted_descriptors(self) -> tuple[ContextSectionDescriptor, ...]:
        """Return only direct sections currently mounted, in stable DOM order."""

        mounted: list[ContextSectionDescriptor] = []
        for descriptor in CONTEXT_SECTION_DESCRIPTORS:
            try:
                self.query_one(
                    f"#console-bounded-section-{descriptor.section_id}",
                    ConsoleBoundedSection,
                )
            except (NoMatches, QueryError):
                continue
            mounted.append(descriptor)
        return tuple(mounted)

    def activate_section(
        self,
        section_id: str,
        *,
        request_reconcile: bool = True,
    ) -> None:
        """Prioritize one mounted direct Context section without persistence."""

        if section_id not in {
            descriptor.section_id for descriptor in self._mounted_descriptors()
        }:
            return
        if section_id != self._active_section_id:
            self._active_section_id = section_id
            self._active_transition_generation += 1
            self._pending_active_reveal_generation = self._active_transition_generation
            self._active_reveal_token += 1
        if request_reconcile:
            self.request_allocation_reconcile()

    @staticmethod
    def _is_enabled_focus_target(widget: Widget) -> bool:
        """Return whether ``widget`` is a usable focus-recovery destination."""

        return bool(
            widget.is_mounted
            and widget.focusable
            and widget.display
            and not getattr(widget, "disabled", False)
            and all(
                not isinstance(ancestor, Widget) or ancestor.display
                for ancestor in widget.ancestors
            )
        )

    def _section_for_owned_target(self, target: Widget | None) -> str | None:
        """Resolve an activatable direct section from one focus/pointer target."""

        if target is None:
            return None

        node: Widget | None = target
        bounded: ConsoleBoundedSection | None = None
        while node is not None and node is not self:
            if getattr(node, "disabled", False):
                return None
            if isinstance(node, DestinationRailSectionHeader):
                return node.section_id
            if isinstance(node, ConsoleBoundedSection):
                bounded = node
                break
            parent = node.parent
            node = parent if isinstance(parent, Widget) else None
        if bounded is None:
            return None
        return bounded.section_id

    def _focusable_body_controls(self, section_id: str) -> tuple[Widget, ...]:
        """Return enabled body descendants in the same order as Textual Tab."""

        try:
            bounded = self.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            )
        except (NoMatches, QueryError):
            return ()
        return tuple(
            widget
            for widget in bounded.viewport.query("*")
            if isinstance(widget, Widget) and self._is_enabled_focus_target(widget)
        )

    def _record_section_focus(self, section_id: str, target: Widget) -> None:
        self._section_focus_history[section_id] = (
            target,
            self._focusable_body_controls(section_id),
        )

    def recover_section_focus(self, section_id: str) -> None:
        """Recover invalidated local focus next, previous, header, then rail toggle."""

        try:
            bounded = self.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            )
        except (NoMatches, QueryError):
            bounded = None
        focused = self.app.focused
        if focused is not None and self._is_enabled_focus_target(focused):
            owned = bool(
                bounded is not None
                and (
                    focused is bounded.viewport or bounded.viewport in focused.ancestors
                )
            )
            if not owned:
                return

        previous_target, previous_controls = self._section_focus_history.get(
            section_id,
            (bounded.viewport if bounded is not None else self, ()),
        )
        if previous_target in previous_controls:
            previous_index = previous_controls.index(previous_target)
            ordered_candidates = previous_controls[previous_index + 1 :] + tuple(
                reversed(previous_controls[:previous_index])
            )
        else:
            ordered_candidates = previous_controls
        for candidate in ordered_candidates:
            if self._is_enabled_focus_target(candidate):
                self._record_section_focus(section_id, candidate)
                candidate.focus()
                return

        for selector in (
            f"#{RAIL_SECTION_TOGGLE_PREFIX}{section_id}",
            "#console-context-rail-collapse",
        ):
            try:
                candidate = self.query_one(selector, Button)
            except (NoMatches, QueryError):
                continue
            if self._is_enabled_focus_target(candidate):
                self._record_section_focus(section_id, candidate)
                candidate.focus()
                return

    def _paint_scroll_focus_owner(
        self,
        *,
        section_id: str | None,
        outer_active: bool,
    ) -> None:
        """Apply dimension-stable non-color ownership cues without stylesheet edits."""

        for descriptor in self._mounted_descriptors():
            try:
                title = self.query_one(
                    f"#console-rail-section-title-{descriptor.section_id}", Static
                )
            except (NoMatches, QueryError):
                continue
            title.styles.text_style = (
                "bold underline" if descriptor.section_id == section_id else "bold"
            )
        try:
            collapse = self.query_one("#console-context-rail-collapse", Button)
        except (NoMatches, QueryError):
            return
        collapse.styles.text_style = "underline" if outer_active else "none"

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Activate owned keyboard targets and paint the current scroll owner."""

        target = event.widget
        section_id = self._section_for_owned_target(target)
        if section_id is not None:
            previous_target, previous_controls = self._section_focus_history.get(
                section_id,
                (None, ()),
            )
            removed_target_snapshot = bool(
                previous_target is not None
                and previous_target is not target
                and previous_target in previous_controls
                and not previous_target.is_attached
            )
            if removed_target_snapshot:
                self.call_after_refresh(
                    self._recover_removed_focus_snapshot,
                    section_id,
                    previous_target,
                )
            else:
                self._record_section_focus(section_id, target)
            self.activate_section(section_id, request_reconcile=False)
            self.call_after_refresh(
                self._finish_focus_activation,
                section_id,
                target,
            )
        outer_active = target.id == "console-left-rail-body"
        self._paint_scroll_focus_owner(
            section_id=section_id,
            outer_active=outer_active,
        )

    def _recover_removed_focus_snapshot(
        self,
        section_id: str,
        removed_target: Widget,
    ) -> None:
        """Recover from Textual's incidental focus without replacing its snapshot."""

        current_target, _controls = self._section_focus_history.get(
            section_id,
            (None, ()),
        )
        if current_target is removed_target and not removed_target.is_attached:
            self.recover_section_focus(section_id)

    def _finish_focus_activation(self, section_id: str, target: Widget) -> None:
        """Reconcile keyboard focus unless a pointer press still owns the target."""

        if (
            self._pointer_activation_pending == section_id
            and self.app.focused is target
        ):
            return
        self.request_allocation_reconcile()

    def on_descendant_blur(self, _event: DescendantBlur) -> None:
        """Clear transient underlines when keyboard focus leaves this rail."""

        self.call_after_refresh(self._clear_focus_owner_if_focus_left)

    def _clear_focus_owner_if_focus_left(self) -> None:
        """Clear cues only after Textual has committed the replacement focus."""

        focused = self.app.focused
        if focused is self or (
            isinstance(focused, Widget) and self in focused.ancestors
        ):
            return
        self._section_focus_history.clear()
        self._paint_scroll_focus_owner(section_id=None, outer_active=False)

    def on_mouse_down(self, event: MouseDown) -> None:
        """Record an owned press without reallocating before its native action."""

        target = event.widget
        section_id = self._section_for_owned_target(target)
        if section_id is None:
            return
        self._pointer_activation_generation += 1
        self._pointer_activation_pending = section_id
        self._pointer_activation_target = target
        self._pointer_activation_waits_for_button = isinstance(target, Button) or any(
            isinstance(ancestor, DestinationRailSectionHeader)
            for ancestor in target.ancestors
        )
        self.activate_section(section_id, request_reconcile=False)

    def on_mouse_up(self, event: MouseUp) -> None:
        """Defer canceled-press cleanup until a native button action can win."""

        if (
            self._pointer_activation_waits_for_button
            and event.widget is self._pointer_activation_target
        ):
            return
        generation = self._pointer_activation_generation
        self.call_after_refresh(
            self._finish_pointer_mouse_up,
            generation,
        )

    def _finish_pointer_mouse_up(self, generation: int) -> None:
        """Flush only the still-current press after its Click/Pressed lifecycle."""

        if generation == self._pointer_activation_generation:
            self._flush_pointer_activation()

    def _flush_pointer_activation(self) -> None:
        """Commit allocation after the pressed control has retained its target."""

        pending = self._pointer_activation_pending
        self._pointer_activation_generation += 1
        self._pointer_activation_pending = None
        self._pointer_activation_waits_for_button = False
        self._pointer_activation_target = None
        if pending is not None:
            self.request_allocation_reconcile()

    def request_allocation_reconcile(self) -> None:
        """Coalesce one post-refresh, all-sibling allocation reconciliation."""

        if not self.is_mounted or self._allocation_reconcile_scheduled:
            return
        self._allocation_reconcile_scheduled = True
        self.call_after_refresh(self._prepare_allocation_reconcile)

    def _prepare_allocation_reconcile(self) -> None:
        """Let every bounded body measure demand before the owner snapshots."""

        if not self.is_mounted:
            self._allocation_reconcile_scheduled = False
            return
        for descriptor in self._mounted_descriptors():
            try:
                section = self.query_one(
                    f"#console-bounded-section-{descriptor.section_id}",
                    ConsoleBoundedSection,
                )
            except (NoMatches, QueryError):
                continue
            section.request_reconcile()
        self.call_after_refresh(self._run_allocation_reconcile)

    def _run_allocation_reconcile(self) -> None:
        """Snapshot once, call the pure policy once, and commit one full state."""

        try:
            descriptors = self._mounted_descriptors()
            if not descriptors:
                return
            viewport_height = self._snapshot_outer_viewport_height()
            if viewport_height <= 0:
                return
            header_chrome_height = self._measure_visible_header_chrome_height(
                descriptors
            )
            sections: list[ConsoleBoundedSection] = []
            demands: list[ContextSectionDemand] = []
            for descriptor in descriptors:
                bounded = self.query_one(
                    f"#console-bounded-section-{descriptor.section_id}",
                    ConsoleBoundedSection,
                )
                header = self.query_one(
                    f"#console-rail-section-header-{descriptor.section_id}",
                    DestinationRailSectionHeader,
                )
                body = bounded.query_one(
                    f"#console-rail-section-body-{descriptor.section_id}"
                )
                sections.append(bounded)
                demands.append(
                    ContextSectionDemand(
                        section_id=descriptor.section_id,
                        desired_content_rows=bounded.desired_content_lines,
                        is_open=bool(header.open and body.display),
                    )
                )

            active_section_id = self._active_section_id
            if active_section_id is not None:
                demands_by_id = {demand.section_id: demand for demand in demands}
                fallback_demands = tuple(
                    demands_by_id.get(
                        descriptor.section_id,
                        ContextSectionDemand(
                            section_id=descriptor.section_id,
                            desired_content_rows=0,
                            is_open=False,
                        ),
                    )
                    for descriptor in CONTEXT_SECTION_DESCRIPTORS
                )
                active_section_id = fallback_active_section(
                    fallback_demands,
                    active_section_id,
                )
                if active_section_id != self._active_section_id:
                    self._active_reveal_token += 1
                    self._pending_active_reveal_generation = None
                self._active_section_id = active_section_id
            result = allocate_context_sections(
                viewport_height=viewport_height,
                header_chrome_height=header_chrome_height,
                sections=tuple(demands),
                active_section_id=active_section_id,
            )
            desired_outer_rows = header_chrome_height + sum(
                allocation.allocated_content_rows + int(allocation.hint_required)
                for allocation in result.allocations
            )
            needs_outer_hint = outer_hint_required(
                desired_outer_rows,
                viewport_height,
            )
            self._apply_allocation_state(
                descriptors,
                sections,
                result,
                needs_outer_hint=needs_outer_hint,
            )
        except (NoMatches, QueryError):
            # Recompose may briefly remove one member of the complete snapshot.
            return
        finally:
            self._allocation_reconcile_scheduled = False

    def _snapshot_outer_viewport_height(self) -> int:
        """Return the no-hint Context viewport height for counterfactual policy."""

        outer = self.query_one("#console-left-rail-body", VerticalScroll)
        height = outer.content_region.height
        try:
            hint = self.query_one("#console-left-rail-outer-hint", Static)
        except (NoMatches, QueryError):
            return height
        if hint.display:
            height += max(1, hint.outer_size.height)
        return height

    def _measure_visible_header_chrome_height(
        self,
        descriptors: tuple[ContextSectionDescriptor, ...],
    ) -> int:
        """Measure fixed header demand from the constrained Textual box model.

        ``Widget._get_box_model`` is intentionally isolated here. Textual 8.2.8
        is exactly pinned by this project, and its ``constrain_width`` path is
        the only source that includes the rail's actual available width while
        avoiding the already-compressed arranged height.
        """

        outer = self.query_one("#console-left-rail-body", VerticalScroll)
        container_size = outer.content_region.size
        viewport_size = self.screen.size
        width_fraction = Fraction(max(0, container_size.width))
        height_fraction = Fraction(max(0, container_size.height))
        total = 0
        for descriptor in descriptors:
            header = self.query_one(
                f"#console-rail-section-header-{descriptor.section_id}",
                DestinationRailSectionHeader,
            )
            box_model = header._get_box_model(
                container_size,
                viewport_size,
                width_fraction,
                height_fraction,
                constrain_width=True,
                greedy=False,
            )
            total += ceil(box_model.height) + header.styles.margin.height
        return total

    def _apply_allocation_state(
        self,
        descriptors: tuple[ContextSectionDescriptor, ...],
        sections: list[ConsoleBoundedSection],
        result: ContextAllocationResult,
        *,
        needs_outer_hint: bool,
    ) -> None:
        """Equality-guard and synchronously apply a complete owner snapshot."""

        no_room_ids = frozenset(
            allocation.section_id
            for allocation in result.allocations
            if allocation.no_room
        )
        presentation_changed = False
        for descriptor in descriptors:
            header = self.query_one(
                f"#console-rail-section-header-{descriptor.section_id}",
                DestinationRailSectionHeader,
            )
            title = header.query_one(
                f"#console-rail-section-title-{descriptor.section_id}", Static
            )
            toggle = header.query_one(
                f"#{RAIL_SECTION_TOGGLE_PREFIX}{descriptor.section_id}", Button
            )
            constrained = descriptor.section_id in no_room_ids
            base_title = header.title
            presentation_changed |= self._present_header_title(
                title,
                base_title=base_title,
                constrained=constrained,
            )
            if constrained:
                if str(toggle.label) != "[>]":
                    toggle.label = "[>]"
                toggle.tooltip = f"Prioritize {base_title}"
            else:
                header.sync_open(header.open)

        if presentation_changed:
            # The first paint installs dimension-stable one-line title chrome.
            # Measure once more after Textual has laid it out; equality guards
            # make the resulting fixed point a no-op rather than a refresh loop.
            self.call_after_refresh(self.request_allocation_reconcile)

        complete_state = (
            result,
            needs_outer_hint,
            tuple(sections),
            self._active_section_id,
        )
        if complete_state == self._last_allocation_state:
            self._update_outer_hint()
            return

        previous_result = (
            self._last_allocation_state[0]
            if self._last_allocation_state is not None
            else None
        )
        entering_fallback = result.uses_outer_scroll and (
            previous_result is None or not previous_result.uses_outer_scroll
        )
        if previous_result is not None and (
            previous_result.uses_outer_scroll != result.uses_outer_scroll
        ):
            self._active_reveal_token += 1

        for bounded, allocation in zip(sections, result.allocations):
            bounded.set_allocation(allocation.allocated_content_rows)
            bounded.styles.height = allocation.allocated_content_rows + int(
                allocation.hint_required
            )

        outer = self.query_one("#console-left-rail-body", VerticalScroll)
        target_overflow = "auto" if result.uses_outer_scroll else "hidden"
        if str(outer.styles.overflow_y) != target_overflow:
            outer.styles.overflow_y = target_overflow
        outer.can_focus = result.uses_outer_scroll
        if not result.uses_outer_scroll and outer.scroll_y != 0:
            outer.scroll_y = 0

        hint = self.query_one("#console-left-rail-outer-hint", Static)
        if hint.display is not needs_outer_hint:
            hint.display = needs_outer_hint
        self._outer_hint_exists = needs_outer_hint
        self._no_room_section_ids = no_room_ids
        self._last_allocation_state = complete_state
        self._update_outer_hint()

        activation_pending = self._pending_active_reveal_generation is not None
        if (
            result.uses_outer_scroll
            and self._active_section_id is not None
            and (entering_fallback or activation_pending)
        ):
            self._pending_active_reveal_generation = None
            self._queue_active_reveal(self._active_section_id)
        elif not result.uses_outer_scroll:
            self._pending_active_reveal_generation = None

    @staticmethod
    def _present_header_title(
        title: Static,
        *,
        base_title: str,
        constrained: bool,
    ) -> bool:
        """Paint one-line chrome while retaining the complete canonical title."""

        changed = False
        if not getattr(title, "_console_context_one_line", False):
            title.styles.height = 1
            title.styles.min_height = 1
            title.styles.max_height = 1
            title.styles.text_wrap = "nowrap"
            title.styles.text_overflow = "clip"
            title._console_context_one_line = True
            changed = True

        title.tooltip = base_title
        target_title = base_title
        if constrained:
            target_title = ConsoleLeftRail._title_with_no_room_suffix(
                base_title,
                title.content_region.width,
            )
        if str(title.renderable) != target_title:
            title.update(target_title)
        return changed

    @staticmethod
    def _title_with_no_room_suffix(base_title: str, width: int) -> str:
        """Fit the base into ``width`` while preserving the exact status suffix."""

        suffix = " · no room"
        full_title = f"{base_title}{suffix}"
        if width <= 0 or cell_len(full_title) <= width:
            return full_title

        base_budget = width - cell_len(suffix)
        if base_budget <= 0:
            # At widths smaller than the suffix itself, keeping the complete
            # status string is the only honest representation; one-line clipping
            # remains dimensionally stable until space becomes available.
            return suffix
        if cell_len(base_title) <= base_budget:
            return full_title

        ellipsis = "…"
        prefix_budget = max(0, base_budget - cell_len(ellipsis))
        prefix = ""
        for character in base_title:
            candidate = f"{prefix}{character}"
            if cell_len(candidate) > prefix_budget:
                break
            prefix = candidate
        return f"{prefix}{ellipsis}{suffix}"

    def _queue_active_reveal(self, section_id: str) -> None:
        """Queue one reveal guarded by the current mode/activation token."""

        self._active_reveal_token += 1
        token = self._active_reveal_token
        self.call_after_refresh(self._schedule_active_reveal, token, section_id)

    def _schedule_active_reveal(self, token: int, section_id: str) -> None:
        """Wait through bounded-body layout before revealing outer content."""

        if not self._active_reveal_is_current(token, section_id):
            return
        self.call_after_refresh(self._stage_active_reveal, token, section_id)

    def _stage_active_reveal(self, token: int, section_id: str) -> None:
        """Wait through the outer-scroll reflow before committing its offset."""

        if not self._active_reveal_is_current(token, section_id):
            return
        self.call_after_refresh(self._reveal_active_section, token, section_id)

    def _active_reveal_is_current(self, token: int, section_id: str) -> bool:
        """Return whether a delayed reveal still matches active fallback state."""

        return bool(
            self.is_attached
            and token == self._active_reveal_token
            and section_id == self._active_section_id
            and self._last_allocation_state is not None
            and self._last_allocation_state[0].uses_outer_scroll
        )

    def _reveal_active_section(self, token: int, section_id: str) -> None:
        """Reveal the active header and its first content row in fallback mode."""

        if not self._active_reveal_is_current(token, section_id):
            return
        try:
            outer = self.query_one("#console-left-rail-body", VerticalScroll)
            header = self.query_one(
                f"#console-rail-section-header-{section_id}",
                DestinationRailSectionHeader,
            )
        except (NoMatches, QueryError):
            return
        outer.scroll_to(
            y=max(0, header.virtual_region.y),
            animate=False,
            immediate=True,
        )
        self._update_outer_hint()
        self.call_after_refresh(self._confirm_active_reveal, token, section_id)

    def _confirm_active_reveal(self, token: int, section_id: str) -> None:
        """Commit the transition reveal against the final outer virtual height."""

        if not self._active_reveal_is_current(token, section_id):
            return
        try:
            outer = self.query_one("#console-left-rail-body", VerticalScroll)
            header = self.query_one(
                f"#console-rail-section-header-{section_id}",
                DestinationRailSectionHeader,
            )
        except (NoMatches, QueryError):
            return
        outer.scroll_to(
            y=max(0, header.virtual_region.y),
            animate=False,
            immediate=True,
        )
        self._update_outer_hint()

    def _update_outer_hint(self) -> None:
        """Keep the pinned outer slot blank at end and exact before end."""

        try:
            outer = self.query_one("#console-left-rail-body", VerticalScroll)
            hint = self.query_one("#console-left-rail-outer-hint", Static)
        except (NoMatches, QueryError):
            return
        before_end = outer.max_scroll_y <= 0 or outer.scroll_y < outer.max_scroll_y
        text = (
            OUTER_SECTION_SCROLL_HINT
            if self._outer_hint_exists and hint.display and before_end
            else ""
        )
        if text == self._outer_hint_text and str(hint.renderable) == text:
            return
        self._outer_hint_text = text
        hint.update(text)

    def sync_workspace_context(self, state: ConsoleWorkspaceContextState) -> None:
        """Push one context snapshot into every scoped rail projection.

        Args:
            state: Shared Console workspace snapshot to project into the
                Sessions, Workspaces, Conversations, and Details trays.

        Returns:
            None.
        """
        for selector in (
            "#console-session-context",
            "#console-workspaces-context",
            "#console-workspace-context",
        ):
            try:
                tray = self.query_one(selector, ConsoleWorkspaceContextTray)
            except (NoMatches, QueryError):
                continue
            tray.sync_state(state)
            tray._console_workspace_context_synced = True
        try:
            details_tray = self.query_one(
                "#console-workspace-details", ConsoleWorkspaceDetailsTray
            )
        except (NoMatches, QueryError):
            pass
        else:
            details_tray.sync_state(state)
        self.request_allocation_reconcile()

    def compose(self) -> ComposeResult:
        """Compose the rail header, pinned fleet line, and context sections.

        Returns:
            The rail-header row, the pinned fleet-summary line, and the
            scrollable Sessions/Workspaces/Conversations/Model/Agent/Details/
            Character section widgets, in mount order.
        """
        rail_state = self._rail_state
        workspace_context_state = self._workspace_context_state

        left_rail_header = Horizontal(classes="console-rail-header")
        left_rail_header.styles.height = 1
        left_rail_header.styles.min_height = 1
        left_rail_header.styles.max_height = 1
        with left_rail_header:
            collapse_button = Button(
                "<---------|Context",
                id="console-context-rail-collapse",
                classes="console-rail-collapse-button",
                compact=True,
            )
            collapse_button.tooltip = "Collapse Console context rail"
            collapse_button.styles.width = "100%"
            collapse_button.styles.min_width = 0
            collapse_button.styles.max_width = "100%"
            collapse_button.styles.text_align = "right"
            collapse_button.styles.content_align = ("right", "middle")
            yield collapse_button

        # TASK-1140 (UAT F1, fix round 1): the fleet summary -- "N other
        # agents running, M waiting for approval." (parallel-agents spec
        # §6, PA-T8) -- is pinned HERE, a plain sibling of `left_rail_header`
        # inside this non-scrolling rail, deliberately OUTSIDE
        # `#console-left-rail-body` (the `VerticalScroll` every section
        # below shares) so it is painted regardless of scroll position,
        # section open/collapsed state, or step-content length. Present but
        # display:none when empty (mirrors the recovery Static in the Model
        # section) so a targeted sync never needs to mount/unmount it.
        fleet_summary = Static(
            self._fleet_line,
            id="console-agent-fleet-summary",
            classes="console-agent-section-fleet-summary",
            markup=False,
        )
        fleet_summary.styles.height = "auto"
        fleet_summary.styles.display = "block" if self._fleet_line else "none"
        yield fleet_summary

        with _ContextOuterBody(
            owner=self,
            id="console-left-rail-body",
            classes="console-left-rail-body",
        ):
            # TASK-14810: the former Session body mixed three distinct jobs.
            # Keep the live session first, then expose workspace context and
            # durable conversation browsing as peer disclosure sections.
            yield self._section_header(
                "session",
                rail_state.session_open,
            )
            session_context_tray = ConsoleWorkspaceContextTray(
                workspace_context_state,
                show_heading=False,
                content="session",
                id="console-session-context",
                classes="console-left-rail-section",
            )
            session_context_tray.styles.width = "100%"
            session_context_tray.styles.min_width = 0
            session_body = self._section_body(
                "session",
                rail_state.session_open,
                frame_console_region(session_context_tray, variant="quiet"),
            )
            yield _ContextBoundedSection(
                session_body,
                section_id="session",
                owner=self,
            )

            yield self._section_header(
                "workspace",
                rail_state.workspace_open,
            )
            workspace_context_tray = ConsoleWorkspaceContextTray(
                workspace_context_state,
                show_heading=False,
                content="workspace",
                id="console-workspaces-context",
                classes="console-left-rail-section",
            )
            workspace_context_tray.styles.width = "100%"
            workspace_context_tray.styles.min_width = 0
            workspace_body = self._section_body(
                "workspace",
                rail_state.workspace_open,
                frame_console_region(workspace_context_tray, variant="quiet"),
            )
            yield _ContextBoundedSection(
                workspace_body,
                section_id="workspace",
                owner=self,
            )

            yield self._section_header(
                "conversations",
                rail_state.conversations_open,
            )
            conversation_context_tray = ConsoleWorkspaceContextTray(
                workspace_context_state,
                show_heading=False,
                content="conversations",
                # Keep the long-standing id on the conversation browser:
                # search/resume tests and screen helpers use it as the
                # grouped browser's stable synchronization seam.
                id="console-workspace-context",
                classes="console-left-rail-section",
            )
            conversation_context_tray.styles.width = "100%"
            conversation_context_tray.styles.min_width = 0
            conversations_body = self._section_body(
                "conversations",
                rail_state.conversations_open,
                frame_console_region(
                    conversation_context_tray,
                    variant="quiet",
                ),
            )
            yield _ContextBoundedSection(
                conversations_body,
                section_id="conversations",
                owner=self,
            )

            # Model (provider/model readout lines plus a
            # Configure shortcut into the Console session settings).
            yield self._section_header(
                "model",
                rail_state.model_open,
            )
            summary_state = self._settings_summary_state
            provider_value = _summary_row_value(summary_state.provider_row) or "—"
            model_value = _summary_row_value(summary_state.model_row) or "—"
            temperature_match = re.search(
                r"T ([\d.]+)", summary_state.sampling_row or ""
            )
            temperature_value = temperature_match.group(1) if temperature_match else "—"
            max_tokens_match = re.search(
                r"max_tokens (\d+)", summary_state.sampling_row or ""
            )
            max_tokens_value = max_tokens_match.group(1) if max_tokens_match else "—"

            model_rows = (
                Horizontal(
                    Static(
                        "Provider",
                        classes="console-model-section-label",
                        markup=False,
                    ),
                    Static(
                        provider_value,
                        classes="console-model-section-value",
                        markup=False,
                    ),
                    id="console-model-section-provider",
                    classes="console-model-section-line",
                ),
                Horizontal(
                    Static(
                        "Model",
                        classes="console-model-section-label",
                        markup=False,
                    ),
                    Static(
                        model_value,
                        classes="console-model-section-value",
                        markup=False,
                    ),
                    id="console-model-section-model",
                    classes="console-model-section-line",
                ),
                Horizontal(
                    Static(
                        "Temperature",
                        classes="console-model-section-label",
                        markup=False,
                    ),
                    Static(
                        temperature_value,
                        classes="console-model-section-value",
                        markup=False,
                    ),
                    id="console-model-section-temperature",
                    classes="console-model-section-line",
                ),
                Horizontal(
                    Static(
                        "Max tokens",
                        classes="console-model-section-label",
                        markup=False,
                    ),
                    Static(
                        max_tokens_value,
                        classes="console-model-section-value",
                        markup=False,
                    ),
                    id="console-model-section-max-tokens",
                    classes="console-model-section-line",
                ),
            )
            readiness = (summary_state.readiness_label or "").strip()
            recovery = Static(
                readiness or "",
                id="console-model-section-recovery",
                classes="console-model-section-recovery",
                markup=False,
            )
            recovery.styles.display = "none"

            system_line = Static(
                self._system_line_text,
                id="console-rail-system-line",
                markup=False,
            )
            # Same one-row clipping hazard as the model line above
            # (task-186): nowrap + ellipsis so a long system prompt
            # truncates visibly instead of word-wrapping onto a hidden
            # second row.
            system_line.styles.text_wrap = "nowrap"
            system_line.styles.text_overflow = "ellipsis"
            system_line.set_class(self._system_line_dim, "console-rail-system-line-dim")
            configure = Button(
                "Configure",
                id="console-model-section-configure",
                classes="console-workspace-action",
                compact=True,
            )
            configure.tooltip = "Configure Console session settings"
            model_body = self._section_body(
                "model",
                rail_state.model_open,
                *model_rows,
                recovery,
                system_line,
                configure,
            )
            yield _ContextBoundedSection(
                model_body,
                section_id="model",
                owner=self,
            )

            # Agent (run inspector -- the watch-and-drill surface
            # for the live/most-recent agent run and its historical
            # sub-agent runs).
            yield self._section_header(
                "agent",
                rail_state.agent_open,
            )
            agent_status = Static(
                self._agent_status_line,
                id="console-agent-section-status",
                classes="console-agent-section-line",
                markup=False,
            )
            agent_steps = Static(
                self._agent_steps_text,
                id="console-agent-section-steps",
                classes="console-agent-section-steps",
                markup=False,
            )
            # This nested fleet subsection remains ordinary Agent content; only
            # the direct Agent wrapper below owns a bounded viewport.
            fleet_section = ConsoleInspectorSection(
                title="Sub-agents",
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
            cancel_all_button = Button(
                "Cancel all agents",
                id=CONSOLE_AGENT_CANCEL_ALL_ID,
                classes="console-workspace-action",
                compact=True,
            )
            cancel_all_button.tooltip = (
                "Stop every running sub-agent of this conversation. "
                "Each is cancelled cooperatively and any pending "
                "approval cards are withdrawn."
            )
            if not self._agent_cancel_all_visible:
                cancel_all_button.styles.display = "none"
            steering_bar = ConsoleAgentSteeringBar(
                self._agent_steering_state,
                id=STEERING_BAR_ID,
            )
            back_button = Button(
                "Back",
                id="console-agent-drilldown-back",
                classes="console-workspace-action console-agent-drilldown-back",
                compact=True,
            )
            back_button.tooltip = "Return to the live agent run view"
            if not self._agent_drilldown_active:
                back_button.styles.display = "none"
            full_log_button = Button(
                "View full log",
                id="console-agent-view-full-log",
                classes=("console-workspace-action console-agent-view-full-log"),
                compact=True,
            )
            full_log_button.tooltip = (
                "Open the full, untruncated run log for this run -- "
                "what the model actually saw, before the Console's "
                "display cap trimmed it."
            )
            if not self._agent_full_log_available:
                full_log_button.styles.display = "none"
            agent_body = self._section_body(
                "agent",
                rail_state.agent_open,
                agent_status,
                agent_steps,
                fleet_section,
                cancel_all_button,
                steering_bar,
                back_button,
                full_log_button,
                classes="console-agent-section",
            )
            yield _ContextBoundedSection(
                agent_body,
                section_id="agent",
                owner=self,
            )

            # Details (storage, sync, handoff plumbing).
            yield self._section_header(
                "details",
                rail_state.details_open,
            )
            details_tray = ConsoleWorkspaceDetailsTray(
                workspace_context_state,
                id="console-workspace-details",
                classes="console-left-rail-section",
            )
            details_tray.styles.width = "100%"
            details_tray.styles.min_width = 0
            details_body = self._section_body(
                "details",
                rail_state.details_open,
                details_tray,
            )
            yield _ContextBoundedSection(
                details_body,
                section_id="details",
                owner=self,
            )

            # Character (avatar of the active character).
            if self._show_character_section:
                yield self._section_header(
                    "character",
                    rail_state.character_open,
                )
                avatar_children = ()
                if self._character_avatar_widget_builder is not None:
                    avatar_children = (self._character_avatar_widget_builder(),)
                avatar_holder = ClickableAvatarBox(
                    *avatar_children,
                    id="console-character-avatar",
                )
                # task-1661: hug the image instead of claiming the rail.
                avatar_holder.styles.width = "auto"
                avatar_holder.styles.height = "auto"
                character_name = Static(
                    self._character_avatar_name,
                    id="console-character-name",
                    markup=False,
                )
                reaction_state = Static(
                    (
                        f"Reaction: {self._manual_reaction_label} (manual)"
                        if self._manual_reaction_label
                        else "Reaction: Automatic"
                    ),
                    id="console-character-reaction-state",
                    markup=False,
                )
                reaction_state.styles.text_wrap = "nowrap"
                reaction_state.styles.text_overflow = "ellipsis"
                reaction_button = Button(
                    "Reaction…",
                    id="console-character-reaction-open",
                    classes="console-workspace-action",
                    compact=True,
                )
                reaction_button.tooltip = "Choose or clear a reaction"
                character_body = self._section_body(
                    "character",
                    rail_state.character_open,
                    avatar_holder,
                    character_name,
                    reaction_state,
                    reaction_button,
                )
                yield _ContextBoundedSection(
                    character_body,
                    section_id="character",
                    owner=self,
                )

        outer_hint = Static(
            "",
            id="console-left-rail-outer-hint",
            classes="console-left-rail-outer-hint",
            markup=False,
        )
        outer_hint.can_focus = False
        outer_hint.styles.height = 1
        outer_hint.styles.min_height = 1
        outer_hint.styles.max_height = 1
        outer_hint.styles.display = "none"
        yield outer_hint

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Catch this rail's own section-toggle buttons; let everything else bubble.

        Every other button inside this rail's subtree (Configure, the
        agent drill-down Back/View-full-log buttons, the rail collapse
        button) has a handler whose body reaches beyond this rail -- the
        Console settings modal, the agent run-log viewer, a screen-wide
        chat-UI sync, or Console rail preference persistence that also
        drives the Inspector rail. Those stay on ``ChatScreen`` and keep
        working unchanged: this method does not stop or otherwise touch
        events it does not recognize, so they bubble to the screen exactly
        as they did before this rail existed as its own widget.

        Args:
            event: The button-press event, as delivered to any Textual
                ``on_button_pressed`` handler; only ``event.button.id`` is
                consulted here.
        """
        button_id = event.button.id or ""
        self._flush_pointer_activation()
        owned_section_id = self._section_for_owned_target(event.button)
        if owned_section_id is not None:
            self.activate_section(owned_section_id)
        if button_id == "console-character-reaction-open":
            event.stop()
            self.post_message(self.ReactionPickerRequested())
            return
        if not button_id.startswith(RAIL_SECTION_TOGGLE_PREFIX):
            return
        event.stop()
        section_id = button_id.removeprefix(RAIL_SECTION_TOGGLE_PREFIX)
        was_constrained = section_id in self._no_room_section_ids
        if was_constrained:
            return
        try:
            header = self.query_one(
                f"#console-rail-section-header-{section_id}",
                DestinationRailSectionHeader,
            )
        except (NoMatches, QueryError):
            opened = True
        else:
            opened = not header.open
        self.post_message(self.SectionToggled(section_id=section_id, opened=opened))

    def sync_sections(self, rail_state: ConsoleRailState) -> None:
        """Apply section open flags to section bodies and headers.

        Moved verbatim from ``ChatScreen._sync_console_rail_sections``:
        stored section preferences are scoped per workspace/conversation,
        so a runtime scope switch (for example resuming a saved
        conversation after a relaunch) can change the effective flags
        without a recompose.

        Args:
            rail_state: The effective Console rail state to sync every
                section's body/header to, one ``apply_section_open`` call
                per id in ``CONSOLE_RAIL_SECTION_IDS``.
        """
        for section_id in CONSOLE_RAIL_SECTION_IDS:
            section_open = bool(getattr(rail_state, f"{section_id}_open", True))
            self.apply_section_open(section_id, section_open)
        self._rail_state = rail_state
        self.request_allocation_reconcile()

    def apply_section_open(self, section_id: str, section_open: bool) -> None:
        """Sync one section's body display and header glyph to an open state.

        Moved verbatim from ``ChatScreen._apply_console_rail_section_open``.

        Args:
            section_id: The section's id (e.g. ``"session"``), used to
                locate its body and header by the ``console-rail-section-
                body-<id>`` / ``console-rail-section-header-<id>`` ids.
            section_open: Whether that section should render open (body
                shown, header glyph synced) or closed.
        """
        try:
            body = self.query_one(f"#console-rail-section-body-{section_id}")
            header = self.query_one(
                f"#console-rail-section-header-{section_id}",
                DestinationRailSectionHeader,
            )
        except (NoMatches, QueryError):
            return
        body.styles.display = "block" if section_open else "none"
        header.sync_open(section_open)
        if not section_open and self._active_section_id == section_id:
            self.recover_section_focus(section_id)
        self.request_allocation_reconcile()
