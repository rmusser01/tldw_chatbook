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

import asyncio
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from loguru import logger
from rich.cells import cell_len
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.events import DescendantBlur, DescendantFocus, MouseDown, MouseUp
from textual.message import Message
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import Button, Static

from ...Chat.console_glyphs import GLYPH_COLLAPSE_LEFT
from ...Chat.console_rail_state import CONSOLE_RAIL_SECTION_IDS, ConsoleRailState
from ...Chat.console_chat_store import (
    ConsoleSettingsComponent,
    ConsoleSettingsPolicyFailureLabel,
    ConsoleSettingsPersistenceFailure,
)
from ...Chat.console_settings_apply import ConsoleSettingsAction
from ...Chat.console_settings_defaults import (
    ConsoleDefaultDurabilityState,
    ConsoleDefaultSavePhase,
    parse_console_endpoint_preview,
)
from ...Widgets.glyph_fallback import resolve_glyph
from ...Chat.console_session_settings import ConsoleSettingsSummaryState
from ...Widgets.Console import (
    ConsoleBoundedSection,
    ConsoleWorkspaceContextTray,
    ConsoleWorkspaceTree,
    WorkspaceTreeContextChanged,
    WorkspaceTreeExpansionChanged,
    WorkspaceTreeFocusRecoveryRequested,
)
from ...Widgets.Console.console_agent_steering_bar import (
    STEERING_BAR_ID,
    ConsoleAgentSteeringBar,
    ConsoleAgentSteeringState,
)
from ...Widgets.Console.console_image_viewer_modal import ClickableAvatarBox
from ...Widgets.Console.console_workspace_details import ConsoleWorkspaceDetailsTray
from ...Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
    DestinationRailSectionHeader,
)
from ...Workspaces.conversation_browser_state import (
    console_rail_section_height_budget,
)
from ...Workspaces.display_state import ConsoleWorkspaceContextState
from .agent import CONSOLE_AGENT_CANCEL_ALL_ID
from .frame import frame_console_region
from .rail_section_layout import (
    ContextSectionDemand,
    fallback_active_section,
    outer_hint_required,
)


OUTER_SECTION_SCROLL_HINT = "▼ more sections — scroll"
CONSOLE_RETRY_GENERATION_SETTINGS_ID = "console-retry-generation-settings"
CONSOLE_RETRY_CONTEXT_SETTINGS_ID = "console-retry-context-settings"
CONSOLE_RETRY_DEFAULT_SAVE_ID = "console-retry-default-save"
CONSOLE_DISCARD_DEFAULT_RETRY_ID = "console-discard-default-retry"
CONSOLE_REFRESH_RUNNING_APP_ID = "console-refresh-running-app"
CONSOLE_DISMISS_DEFAULT_REFRESH_ID = "console-dismiss-default-refresh"
#: The two peer list sections whose bounded-section ceilings grow to fill
#: the rail (half the measured viewport each, via
#: `console_rail_section_height_budget`). Every other section keeps the
#: historical fixed ceiling.
_ADAPTIVE_BUDGET_SECTION_IDS = frozenset({"workspace", "conversations"})

CharacterAvatarBox = tuple[int, int]
CharacterAvatarWidgetBuilder = Callable[..., Widget]
CharacterAvatarFitBox = Callable[[int, int], CharacterAvatarBox | None]
#: Loop-side factory returning a thread-safe zero-arg render job for a box
#: (TASK-22221). ``None`` from either call means "nothing to prerender".
CharacterAvatarPrerenderJob = Callable[
    [CharacterAvatarBox], Callable[[], object] | None
]


@dataclass(frozen=True, slots=True)
class ContextSectionDescriptor:
    """Stable direct Context-section identity and rendered-content ceiling."""

    section_id: str
    title: str
    max_content_lines: int


@dataclass(frozen=True, slots=True)
class _ContextFocusRecoveryIncident:
    """Stable local-focus identity retained across one DOM mutation."""

    target_id: str | None
    target_index: int | None


CONTEXT_SECTION_DESCRIPTORS = (
    # TASK-23199 retired "Sessions": a header plus one row naming the active
    # chat, which the Conversations browser already shows as a selected row
    # marked "active session".
    ContextSectionDescriptor("workspace", "Workspaces", 20),
    ContextSectionDescriptor("conversations", "Conversations", 20),
    ContextSectionDescriptor("model", "Model", 15),
    ContextSectionDescriptor("agent", "Agent", 15),
    ContextSectionDescriptor("details", "Details", 15),
    ContextSectionDescriptor("character", "Character", 35),
)


class _ContextOuterBody(VerticalScroll):
    """Context scroll owner with direct resize/scroll invalidation seams."""

    def __init__(self, *, owner: "ConsoleLeftRail", **kwargs) -> None:
        super().__init__(**kwargs)
        self._owner = owner

    def on_resize(self) -> None:
        self._owner.note_character_avatar_viewport_size(
            self.content_size.width,
            self.content_size.height,
        )
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
        native_scroll_owner: ConsoleWorkspaceTree | None = None,
    ) -> None:
        self._allocation_owner = owner
        descriptor = next(
            item
            for item in CONTEXT_SECTION_DESCRIPTORS
            if item.section_id == section_id
        )
        super().__init__(
            *content,
            section_id=section_id,
            max_content_lines=descriptor.max_content_lines,
            on_focus_recovery=lambda: owner.recover_section_focus(section_id),
            native_scroll_owner=native_scroll_owner,
        )

    def _run_scheduled_reconcile(self) -> None:
        scoped = self._reconcile_scoped
        previous_demand = self.desired_content_lines
        super()._run_scheduled_reconcile()
        if scoped:
            # TASK-22203: a scoped pass owns only this section's local
            # geometry (the workspace action row's one-row display flip); its
            # demand delta is self-absorbed within the current allocation and
            # must not fan the cursor move out into the rail-wide allocation
            # pipeline. Any unscoped request coalescing into the same pass
            # already demoted ``_reconcile_scoped`` before this ran.
            return
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

    # TASK-23198 (audit S4-A): the rail declared no bindings at all, so a
    # user who wanted every section shut had to click seven disclosure
    # toggles. These fire only while focus is inside the rail, so they
    # cannot shadow anything the composer or transcript owns.
    #
    # NOT included: a Tab-escape binding. Tab is scoped to the focused
    # Console region on purpose (ChatScreen.action_focus_next, TASK-2154.11)
    # -- unscoped, a Tab tour crossed all fifteen app-nav buttons mid-session
    # -- and F6 already moves focus out in one press with "F6 next pane"
    # permanently advertised in the footer. That satisfies WCAG 2.1.2, whose
    # requirement is a keyboard way out plus advice, not Tab specifically.
    BINDINGS = [
        Binding(
            "ctrl+shift+left",
            "collapse_all_sections",
            "Collapse all",
            show=True,
        ),
        Binding(
            "ctrl+shift+right",
            "expand_all_sections",
            "Expand all",
            show=True,
        ),
    ]

    def action_collapse_all_sections(self) -> None:
        """Close every Context section that is currently open."""
        self._set_all_sections_open(False)

    def action_expand_all_sections(self) -> None:
        """Open every Context section that is currently closed."""
        self._set_all_sections_open(True)

    def _set_all_sections_open(self, opened: bool) -> None:
        """Post one SectionToggled per section that must change.

        Routed through the same message the disclosure buttons post, so the
        screen's persistence and layout reconciliation run exactly as they do
        for a click -- no second path to keep in sync.

        Args:
            opened: Target open state for every mounted section.
        """
        for descriptor in self._mounted_descriptors():
            try:
                header = self.query_one(
                    f"#console-rail-section-header-{descriptor.section_id}",
                    DestinationRailSectionHeader,
                )
            except (NoMatches, QueryError):
                continue
            if bool(header.open) is opened:
                continue
            self.post_message(
                self.SectionToggled(
                    section_id=descriptor.section_id, opened=opened
                )
            )


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

    class TerminalRequested(Message):
        """The pinned Context action asked its owning screen for Terminal."""

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
        agent_drilldown_active: bool,
        agent_full_log_available: bool,
        agent_steering_state: ConsoleAgentSteeringState | None = None,
        agent_cancel_all_visible: bool = False,
        show_character_section: bool,
        character_avatar_widget_builder: CharacterAvatarWidgetBuilder | None,
        character_avatar_name: str,
        character_avatar_fit_box: CharacterAvatarFitBox | None = None,
        character_avatar_prerender_job: CharacterAvatarPrerenderJob | None = None,
        workspace_tree_expanded_ids: frozenset[str] | None = None,
        workspace_tree_expansion_preferences_changed: (
            Callable[[frozenset[str]], None] | None
        ) = None,
        manual_reaction_label: str | None = None,
        settings_session_id: str | None = None,
        settings_persistence_failures: Mapping[
            ConsoleSettingsComponent,
            ConsoleSettingsPersistenceFailure,
        ]
        | None = None,
        default_durability_state: ConsoleDefaultDurabilityState | None = None,
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
            character_avatar_widget_builder: Callable that builds the avatar
                widget for an optional fitted cell box, when
                ``show_character_section`` is True (``None`` otherwise). The
                screen still owns
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
            character_avatar_fit_box: Late-binding source-aware contain fitter.
                It receives the mounted body's measured image budget and returns
                the scale-down-only cell box, ``(0, 0)`` when no image rows
                remain, or ``None`` when no valid image is available.
            character_avatar_prerender_job: Late-binding loop-side factory
                (TASK-22221). Given the target box it returns a thread-safe
                zero-arg job that renders the avatar's pixels, or ``None``
                when this box has no pixel leg worth moving off the loop.
                The rail runs that job in a worker thread and hands the
                result to the widget builder, which uses it only if it still
                matches.
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
        self._agent_drilldown_active = agent_drilldown_active
        self._agent_full_log_available = agent_full_log_available
        self._agent_steering_state = agent_steering_state
        self._agent_cancel_all_visible = agent_cancel_all_visible
        self._show_character_section = show_character_section
        self._character_avatar_widget_builder = character_avatar_widget_builder
        self._character_avatar_name = character_avatar_name
        self._character_avatar_fit_box = character_avatar_fit_box
        self._character_avatar_prerender_job = character_avatar_prerender_job
        self._character_avatar_box: CharacterAvatarBox | None = None
        self._character_avatar_fit_generation = 0
        self._character_avatar_geometry_epoch = 0
        self._character_avatar_viewport_size: tuple[int, int] | None = None
        self._character_avatar_fit_signature: tuple[int, int, int] | None = None
        self._character_avatar_fit_result: CharacterAvatarBox | None = None
        self._character_avatar_followup_pending = False
        self._character_avatar_suppressed_epoch: int | None = None
        self._character_avatar_mount_lock = asyncio.Lock()
        self._workspace_tree_expanded_ids = workspace_tree_expanded_ids
        self._workspace_tree_expansion_preferences_changed = (
            workspace_tree_expansion_preferences_changed
        )
        self._manual_reaction_label = str(manual_reaction_label or "").strip()
        self._settings_session_id = settings_session_id
        self._settings_persistence_failures = dict(
            settings_persistence_failures or {}
        )
        self._default_durability_state = (
            default_durability_state or ConsoleDefaultDurabilityState()
        )
        self._active_section_id: str | None = None
        self._active_reveal_generation = 0
        self._pending_active_reveal: tuple[int, str, str | None, bool] | None = None
        self._allocation_reconcile_scheduled = False
        self._last_allocation_state: (
            tuple[bool, int, tuple[ConsoleBoundedSection, ...]] | None
        ) = None
        self._outer_hint_exists = False
        self._outer_hint_text = ""
        self._workspace_tree_reflow_check_scheduled = False
        self._workspace_tree_reflow_state: tuple[object, ...] | None = None
        self._section_focus_history: dict[str, tuple[Widget, tuple[Widget, ...]]] = {}
        self._pending_focus_recoveries: dict[str, _ContextFocusRecoveryIncident] = {}
        self._pointer_activation_pending: str | None = None
        self._pointer_activation_waits_for_button = False
        self._pointer_activation_target: Widget | None = None
        self._pointer_activation_generation = 0

    @property
    def character_avatar_box(self) -> CharacterAvatarBox | None:
        """Latest equality-guarded fitted image box, if a valid image exists."""

        return self._character_avatar_box

    def invalidate_character_avatar_geometry(self) -> None:
        """Start a new source/viewport epoch for the Character contain fit."""

        self._character_avatar_geometry_epoch += 1
        self._character_avatar_fit_signature = None
        self._character_avatar_followup_pending = False
        self._character_avatar_suppressed_epoch = None

    def note_character_avatar_viewport_size(self, width: int, height: int) -> None:
        """Start a geometry epoch only for a genuinely new outer viewport."""

        viewport_size = (max(0, int(width)), max(0, int(height)))
        if viewport_size == self._character_avatar_viewport_size:
            return
        self._character_avatar_viewport_size = viewport_size
        self.invalidate_character_avatar_geometry()

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
        header = DestinationRailSectionHeader(
            descriptor.title,
            section_id=section_id,
            open=is_open,
            id=f"console-rail-section-header-{section_id}",
        )
        # Keep the disclosure chrome structurally above its bounded body even
        # in lightweight hosts that mount the Console screen without the app
        # stylesheet.  The production CSS declares the same two constraints;
        # owning them here prevents a late allocator pass from collapsing the
        # default ``Horizontal`` 1fr height to zero and leaving the visible
        # toggle painted over by the first body row.
        header.styles.height = "auto"
        header.styles.min_height = 2
        return header

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

        self.sync_model_recovery(
            session_id=self._settings_session_id,
            failures=self._settings_persistence_failures,
            default_state=self._default_durability_state,
        )
        self.request_allocation_reconcile()
        self.call_after_refresh(self._request_initial_workspace_tree_pages)

    @staticmethod
    def _default_recovery_copy(state: ConsoleDefaultDurabilityState) -> str:
        """Return a credential-free exact-intent recovery summary."""

        intent = state.recovery_intent
        phase = state.failure_phase
        if intent is None or phase is None:
            return ""
        action = (
            "Make default for new chats"
            if intent.action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
            else "Save as model default"
        )
        scope = ", ".join(sorted(intent.field_mask)) or "none"
        phase_copy = (
            "Not written to disk"
            if phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
            else "Saved on disk; running app refresh failed"
        )
        endpoint = intent.endpoint_patch
        preview = (
            parse_console_endpoint_preview(endpoint.value)
            if endpoint is not None and endpoint.dirty and endpoint.checked
            else None
        )
        endpoint_copy = (
            f" · {preview.authority} · {preview.network_classification}"
            if preview is not None
            else ""
        )
        return (
            f"{phase_copy} · {action} · {intent.provider_config_key}/"
            f"{intent.literal_model_id} · fields: {scope}{endpoint_copy}"
        )

    def sync_model_recovery(
        self,
        *,
        session_id: str | None,
        failures: Mapping[
            ConsoleSettingsComponent,
            ConsoleSettingsPersistenceFailure,
        ],
        default_state: ConsoleDefaultDurabilityState,
    ) -> None:
        """Synchronize revision-bound conversation and app recovery rows."""

        self._settings_session_id = session_id
        self._settings_persistence_failures = dict(failures)
        self._default_durability_state = default_state
        generation = failures.get(ConsoleSettingsComponent.GENERATION_SETTINGS)
        context = failures.get(ConsoleSettingsComponent.CONTEXT_POLICY)
        default_copy = self._default_recovery_copy(default_state)
        has_warning = generation is not None or context is not None or bool(default_copy)

        try:
            title = self.query_one("#console-rail-section-title-model", Static)
            title.update("Model ⚠" if has_warning else "Model")
            generation_group = self.query_one("#console-generation-recovery-row")
            context_group = self.query_one("#console-context-recovery-row")
            default_group = self.query_one("#console-default-recovery-row")
            generation_button = self.query_one(
                f"#{CONSOLE_RETRY_GENERATION_SETTINGS_ID}", Button
            )
            context_button = self.query_one(
                f"#{CONSOLE_RETRY_CONTEXT_SETTINGS_ID}", Button
            )
            retry_default = self.query_one(
                f"#{CONSOLE_RETRY_DEFAULT_SAVE_ID}", Button
            )
            discard_default = self.query_one(
                f"#{CONSOLE_DISCARD_DEFAULT_RETRY_ID}", Button
            )
            refresh_default = self.query_one(
                f"#{CONSOLE_REFRESH_RUNNING_APP_ID}", Button
            )
            dismiss_default = self.query_one(
                f"#{CONSOLE_DISMISS_DEFAULT_REFRESH_ID}", Button
            )
        except (NoMatches, QueryError):
            return

        generation_group.styles.display = "block" if generation is not None else "none"
        context_group.styles.display = "block" if context is not None else "none"
        default_group.styles.display = "block" if default_copy else "none"
        if generation is not None:
            generation_button.console_settings_session_id = session_id
            generation_button.console_settings_revision = generation.revision
        if context is not None:
            context_button.console_settings_session_id = session_id
            context_button.console_settings_revision = context.revision
            policy_label = context.policy_failure_label
            assert isinstance(policy_label, ConsoleSettingsPolicyFailureLabel)
            self.query_one("#console-context-recovery-copy", Static).update(
                f"Not saved: {policy_label.value}"
            )
        generation_button.disabled = generation is None
        context_button.disabled = context is None

        self.query_one("#console-default-recovery-copy", Static).update(default_copy)
        generation_token = default_state.newest_intent_generation
        for button in (
            retry_default,
            discard_default,
            refresh_default,
            dismiss_default,
        ):
            button.console_default_intent_generation = generation_token
        before_replace = (
            default_state.failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
        )
        cache_failure = (
            default_state.failure_phase is ConsoleDefaultSavePhase.CACHE_PUBLICATION
        )
        retry_default.styles.display = "block" if before_replace else "none"
        discard_default.styles.display = "block" if before_replace else "none"
        refresh_default.styles.display = "block" if cache_failure else "none"
        dismiss_default.styles.display = "block" if cache_failure else "none"
        self.request_allocation_reconcile()

    def _request_initial_workspace_tree_pages(self) -> None:
        """Route persisted/default-expanded nodes through the page loader once."""

        try:
            tree = self.query_one("#console-workspace-tree", ConsoleWorkspaceTree)
        except (NoMatches, QueryError):
            return
        for workspace_id in sorted(tree.preferred_expanded_workspace_ids):
            self.post_message(
                WorkspaceTreeExpansionChanged(workspace_id, expanded=True)
            )

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
        reveal_target: Widget | None = None,
        deliberate_reveal: bool = True,
    ) -> None:
        """Activate and deliberately reveal one section without persistence."""

        if section_id not in {
            descriptor.section_id for descriptor in self._mounted_descriptors()
        }:
            return
        self._active_section_id = section_id
        self._active_reveal_generation += 1
        self._pending_active_reveal = (
            (
                self._active_reveal_generation,
                section_id,
                reveal_target.id if reveal_target is not None else None,
                reveal_target is not None,
            )
            if deliberate_reveal
            else None
        )
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
        """Return enabled body descendants in the same order as Textual Tab.

        TASK-22228 (item 3): the subtree walk is ``walk_children(Widget)``,
        not ``query("*")``. Textual builds a ``query`` from exactly that
        walk and then runs the parsed universal selector through ``match()``
        for every node it visited (``DOMQuery.nodes``), so the selector
        round-trip is pure overhead for a filter that admits everything --
        measured at 74.1 us against 2.2 us for the same 16-node Model body
        on the mounted Console. This runs once or twice per focus change in
        the rail, so the walk is the whole cost of that path; the resulting
        list is identical (same nodes, same depth-first order) by
        construction.
        """

        try:
            bounded = self.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            )
        except (NoMatches, QueryError):
            return ()
        controls = tuple(
            widget
            for widget in bounded.viewport.walk_children(Widget)
            if self._is_enabled_focus_target(widget)
        )
        if bounded.native_scroll_owner is not None and self._is_enabled_focus_target(
            bounded.viewport
        ):
            return (bounded.viewport, *controls)
        return controls

    def _record_section_focus(self, section_id: str, target: Widget) -> None:
        self._section_focus_history[section_id] = (
            target,
            self._focusable_body_controls(section_id),
        )

    @staticmethod
    def _stable_focus_id(widget: Widget) -> str | None:
        return widget.id or None

    def _focus_recovery_incident(
        self,
        previous: Widget,
        controls: tuple[Widget, ...],
    ) -> _ContextFocusRecoveryIncident:
        return _ContextFocusRecoveryIncident(
            target_id=self._stable_focus_id(previous),
            target_index=controls.index(previous) if previous in controls else None,
        )

    def _ensure_focus_recovery(
        self,
        section_id: str,
    ) -> _ContextFocusRecoveryIncident | None:
        """Freeze and schedule one semantic recovery incident per section."""

        pending = self._pending_focus_recoveries.get(section_id)
        if pending is not None:
            self._section_focus_history.pop(section_id, None)
            return pending
        history = self._section_focus_history.pop(section_id, None)
        if history is None:
            return None
        incident = self._focus_recovery_incident(*history)
        self._pending_focus_recoveries[section_id] = incident
        self.call_after_refresh(
            self._recover_pending_focus,
            section_id,
            incident,
        )
        return incident

    def _focus_is_valid_outside_rail(self, focused: Widget | None) -> bool:
        return bool(
            focused is not None
            and self._is_enabled_focus_target(focused)
            and focused is not self
            and self not in focused.ancestors
        )

    def _recover_pending_focus(
        self,
        section_id: str,
        incident: _ContextFocusRecoveryIncident,
    ) -> None:
        """Resolve one current incident against the section's current DOM."""

        if self._pending_focus_recoveries.get(section_id) is not incident:
            return
        if not self.is_attached:
            self._pending_focus_recoveries.pop(section_id, None)
            return
        if self._focus_is_valid_outside_rail(self.app.focused):
            self._pending_focus_recoveries.pop(section_id, None)
            self._section_focus_history.pop(section_id, None)
            return
        try:
            bounded = self.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            )
        except (NoMatches, QueryError):
            self._pending_focus_recoveries.pop(section_id, None)
            return

        controls = self._focusable_body_controls(section_id)
        candidates: list[Widget] = []
        if incident.target_id is not None:
            candidates.extend(
                control
                for control in controls
                if self._stable_focus_id(control) == incident.target_id
            )
        if incident.target_index is None:
            candidates.extend(controls)
        else:
            index = min(incident.target_index, len(controls))
            candidates.extend(controls[index:])
            candidates.extend(reversed(controls[:index]))

        seen: set[Widget] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if self._is_enabled_focus_target(candidate):
                self._commit_focus_recovery(
                    section_id,
                    incident,
                    bounded,
                    candidate,
                    controls,
                )
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
                self._commit_focus_recovery(
                    section_id,
                    incident,
                    bounded,
                    candidate,
                    controls,
                )
                return

        self._pending_focus_recoveries.pop(section_id, None)
        self._section_focus_history.pop(section_id, None)
        bounded._acknowledge_focus_recovery(None)

    def _commit_focus_recovery(
        self,
        section_id: str,
        incident: _ContextFocusRecoveryIncident,
        bounded: ConsoleBoundedSection,
        candidate: Widget,
        controls: tuple[Widget, ...],
    ) -> None:
        """Synchronously select and acknowledge one current recovery target."""

        if self._pending_focus_recoveries.get(section_id) is not incident:
            return
        self._pending_focus_recoveries.pop(section_id, None)
        self._section_focus_history[section_id] = (candidate, controls)
        self.screen.set_focus(candidate)
        bounded._acknowledge_focus_recovery(candidate)

    def _section_is_outer_visible(self, section_id: str) -> bool:
        """Return whether a section's header and first body rows are visible."""

        try:
            outer = self.query_one("#console-left-rail-body", VerticalScroll)
            header = self.query_one(
                f"#console-rail-section-header-{section_id}",
                DestinationRailSectionHeader,
            )
            bounded = self.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            )
        except (NoMatches, QueryError):
            return False
        return header.region.overlaps(
            outer.content_region
        ) and bounded.viewport.region.overlaps(outer.content_region)

    def recover_section_focus(self, section_id: str) -> None:
        """Coalesce one bounded invalidation into semantic local recovery."""

        if self._focus_is_valid_outside_rail(self.app.focused):
            self._pending_focus_recoveries.pop(section_id, None)
            self._section_focus_history.pop(section_id, None)
            return
        history = self._section_focus_history.get(section_id)
        if (
            history is not None
            and self.app.focused is history[0]
            and self._is_enabled_focus_target(history[0])
        ):
            return
        self._ensure_focus_recovery(section_id)

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
                self._ensure_focus_recovery(section_id)
            elif section_id not in self._pending_focus_recoveries:
                self._record_section_focus(section_id, target)
            # The Tree captured a stable key before focus-triggered reflow, so
            # unlike ordinary controls this press is safe to reveal immediately.
            stable_tree_press = (
                isinstance(target, ConsoleWorkspaceTree)
                and target._pressed_node_key is not None
            )
            self.activate_section(
                section_id,
                request_reconcile=stable_tree_press,
                deliberate_reveal=stable_tree_press,
            )
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

    def _finish_focus_activation(self, section_id: str, target: Widget) -> None:
        """Reconcile keyboard focus unless a pointer press still owns the target."""

        if (
            self._pointer_activation_pending == section_id
            and self.app.focused is target
        ):
            return
        if (
            self.app.focused is target
            and target.is_mounted
            and not self._section_is_outer_visible(section_id)
        ):
            self.activate_section(section_id, reveal_target=target)
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
        self._pending_focus_recoveries.clear()
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
        # Preserve the same early-reveal contract when MouseDown reaches the
        # rail before Textual's descendant-focus notification.
        stable_tree_press = (
            isinstance(target, ConsoleWorkspaceTree)
            and target._pressed_node_key is not None
        )
        self.activate_section(
            section_id,
            request_reconcile=stable_tree_press,
            deliberate_reveal=stable_tree_press,
        )

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
        target = self._pointer_activation_target
        self._pointer_activation_generation += 1
        self._pointer_activation_pending = None
        self._pointer_activation_waits_for_button = False
        self._pointer_activation_target = None
        if pending is not None:
            self.activate_section(pending, reveal_target=target)

    def request_allocation_reconcile(self) -> None:
        """Coalesce one post-refresh local-then-outer reconciliation."""

        if not self.is_mounted or self._allocation_reconcile_scheduled:
            return
        self._allocation_reconcile_scheduled = True
        self.call_after_refresh(self._prepare_allocation_reconcile)

    def _prepare_allocation_reconcile(self) -> None:
        """Let every bounded body commit its own ceiling before outer geometry."""

        if not self.is_mounted:
            self._allocation_reconcile_scheduled = False
            return
        # The two peer list sections grow to fill the rail: each gets half
        # the measured viewport as its ceiling (see
        # `console_rail_section_height_budget`). Applied in the PREPARE pass
        # so the sections commit their geometry at the new ceiling within
        # this same refresh cycle -- the run pass below then snapshots
        # settled totals. An unmeasured viewport (early mount) leaves the
        # ceilings untouched; the next resize/reconcile converges.
        adaptive_budget: int | None = None
        viewport_height = self._snapshot_outer_viewport_height()
        if viewport_height > 0:
            adaptive_budget = console_rail_section_height_budget(viewport_height)
        for descriptor in self._mounted_descriptors():
            try:
                section = self.query_one(
                    f"#console-bounded-section-{descriptor.section_id}",
                    ConsoleBoundedSection,
                )
            except (NoMatches, QueryError):
                continue
            if (
                adaptive_budget is not None
                and descriptor.section_id in _ADAPTIVE_BUDGET_SECTION_IDS
                and section.max_content_lines != adaptive_budget
            ):
                section.max_content_lines = adaptive_budget
            section.set_allocation(None)
            if section.native_scroll_owner is None:
                section.styles.height = "auto"
            section.request_reconcile()
        self.call_after_refresh(self._run_allocation_reconcile)

    def _run_allocation_reconcile(self) -> None:
        """Snapshot committed complete sections and reconcile the outer owner."""

        try:
            self._reconcile_character_avatar_geometry()
            descriptors = self._mounted_descriptors()
            if not descriptors:
                return
            viewport_height = self._snapshot_outer_viewport_height()
            if viewport_height <= 0:
                return
            sections: list[ConsoleBoundedSection] = []
            demands: list[ContextSectionDemand] = []
            presentation_changed = False
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
                title = header.query_one(
                    f"#console-rail-section-title-{descriptor.section_id}", Static
                )
                presentation_changed |= self._present_header_title(
                    title,
                    base_title=header.title,
                )
                header.sync_open(header.open)
                sections.append(bounded)
                demands.append(
                    ContextSectionDemand(
                        section_id=descriptor.section_id,
                        desired_content_rows=max(
                            bounded.desired_content_lines,
                            int(
                                bounded.native_scroll_owner is not None and body.display
                            ),
                        ),
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
                    self._active_reveal_generation += 1
                    self._pending_active_reveal = None
                self._active_section_id = active_section_id

            if presentation_changed:
                self.call_after_refresh(self.request_allocation_reconcile)

            outer = self.query_one("#console-left-rail-body", VerticalScroll)
            desired_outer_rows = self._measure_outer_content_height(outer)
            needs_outer_hint = outer_hint_required(
                desired_outer_rows,
                viewport_height,
            )
            self._apply_allocation_state(
                sections,
                desired_outer_rows=desired_outer_rows,
                needs_outer_hint=needs_outer_hint,
            )
            self._queue_pending_active_reveal()
            self._refresh_workspace_tree_after_reflow()
        except (NoMatches, QueryError):
            # Recompose may briefly remove one member of the complete snapshot.
            return
        finally:
            self._allocation_reconcile_scheduled = False

    def _reconcile_character_avatar_geometry(self) -> None:
        """Measure controls and schedule one unequal Character image replacement."""

        builder = self._character_avatar_widget_builder
        fitter = self._character_avatar_fit_box
        if builder is None or fitter is None:
            return
        try:
            body = self.query_one("#console-rail-section-body-character", Vertical)
            frame = self.query_one("#console-character-avatar-frame", Horizontal)
            holder = self.query_one("#console-character-avatar", ClickableAvatarBox)
        except (NoMatches, QueryError):
            return
        if not body.display or not body.is_mounted or not holder.is_mounted:
            return

        complete_rows = max(0, body.virtual_region_with_margin.height)
        image_rows = max(0, frame.virtual_region_with_margin.height)
        measured_non_image_rows = max(0, complete_rows - image_rows)
        available_rows = max(0, 35 - measured_non_image_rows)
        available_cols = max(0, body.content_region.width)
        is_followup = self._character_avatar_followup_pending
        self._character_avatar_followup_pending = False
        fit_signature = (
            self._character_avatar_geometry_epoch,
            available_cols,
            available_rows,
        )
        if fit_signature == self._character_avatar_fit_signature:
            target_box = self._character_avatar_fit_result
        else:
            target_box = fitter(available_cols, available_rows)
            self._character_avatar_fit_signature = fit_signature
            self._character_avatar_fit_result = target_box
        if target_box == self._character_avatar_box:
            return
        if is_followup:
            # A replacement gets one local-to-outer settle pass. If scrollbar
            # feedback changes the measured box again, wait for a real source
            # or viewport epoch instead of entering a fixed-point loop.
            self._character_avatar_suppressed_epoch = (
                self._character_avatar_geometry_epoch
            )
            return
        if (
            self._character_avatar_suppressed_epoch
            == self._character_avatar_geometry_epoch
        ):
            return

        self._character_avatar_box = target_box
        self._character_avatar_fit_generation += 1
        generation = self._character_avatar_fit_generation
        self.run_worker(
            self._replace_character_avatar_for_geometry(generation, target_box),
            group="console-character-avatar-fit",
            exclusive=True,
        )

    async def _replace_character_avatar_for_geometry(
        self,
        generation: int,
        target_box: CharacterAvatarBox | None,
    ) -> None:
        """Apply one still-current fitted box and request its sole follow-up."""

        builder = self._character_avatar_widget_builder
        if builder is None:
            return

        def is_current() -> bool:
            return bool(
                generation == self._character_avatar_fit_generation and self.is_mounted
            )

        # The pixels are rendered off the loop, so the viewport can move
        # again while we are suspended here. No fence is needed at this line:
        # `replace_character_avatar_widget` re-checks `is_current()` as the
        # FIRST thing it does inside the mount lock, before it unmounts
        # anything -- so a superseded pass neither paints a stale-size avatar
        # nor blanks the live one. Both properties are pinned by
        # Tests/UI/test_console_avatar_geometry_offloop.py.
        prerendered = await self._prerender_character_avatar(generation, target_box)
        replaced = await self.replace_character_avatar_widget(
            lambda: builder(target_box, prerendered=prerendered),
            is_current=is_current,
        )
        if not replaced:
            return
        if generation != self._character_avatar_fit_generation:
            return
        self._character_avatar_followup_pending = True
        try:
            bounded = self.query_one(
                "#console-bounded-section-character", ConsoleBoundedSection
            )
        except (NoMatches, QueryError):
            return
        bounded.request_reconcile()
        self.request_allocation_reconcile()

    async def _prerender_character_avatar(
        self,
        generation: int,
        target_box: CharacterAvatarBox | None,
    ) -> object | None:
        """Render this box's avatar pixels in a worker thread (TASK-22221).

        The pixel leg is the expensive half of a viewport change (measured
        6.29 ms median for a 1024px card, 187.2 ms across a 37-step resize
        drag) and it is pure -- no DOM, no shared state -- so it belongs off
        the event loop. The loop-side factory snapshots the live spec and
        colour mode BEFORE the hop; only immutable values cross the boundary.

        Args:
            generation: The fit generation this replacement belongs to.
            target_box: The cell box being fitted, if any.

        Returns:
            An opaque prerender token for the widget builder, or ``None``
            when there is nothing to prerender or the render failed. Staleness
            is the CALLER's check, not this one's -- see
            ``_replace_character_avatar_for_geometry``. Never raises: this runs
            inside a Textual worker whose default ``exit_on_error`` would take
            the app down with it.
        """

        factory = self._character_avatar_prerender_job
        if factory is None or not target_box or target_box == (0, 0):
            return None
        try:
            job = factory(target_box)
        except Exception:
            logger.opt(exception=True).debug("avatar: prerender job build failed")
            return None
        if job is None:
            return None
        try:
            prerendered = await asyncio.to_thread(job)
        except Exception:
            # Fail soft: the builder rebuilds inline, exactly as before.
            logger.opt(exception=True).debug("avatar: off-loop prerender failed")
            return None
        return prerendered

    async def replace_character_avatar_widget(
        self,
        widget_builder: Callable[[], Widget],
        *,
        is_current: Callable[[], bool],
    ) -> bool:
        """Serialize one freshness-fenced replacement of the avatar child."""

        async with self._character_avatar_mount_lock:
            if not is_current() or not self.is_mounted:
                return False
            try:
                holder = self.query_one("#console-character-avatar", ClickableAvatarBox)
            except (NoMatches, QueryError):
                return False
            await holder.remove_children()
            if not is_current() or not holder.is_mounted:
                return False
            widget = widget_builder()
            if not is_current():
                return False
            await holder.mount(widget)
            if is_current():
                return True
            if widget.parent is holder:
                await widget.remove()
            return False

    def _snapshot_outer_viewport_height(self) -> int:
        """Return the no-hint Context viewport height for counterfactual policy."""

        # TASK-25715 finding 2 (captured traceback, 2026-09-01): a deferred
        # _prepare_allocation_reconcile invoked after this rail's removal
        # reached this query with `is_mounted` still True and crashed on
        # NoMatches. A rail with no body has no viewport; report it as such
        # instead of raising out of a stale callback.
        try:
            outer = self.query_one("#console-left-rail-body", VerticalScroll)
        except (NoMatches, QueryError):
            return 0
        height = outer.content_region.height
        try:
            hint = self.query_one("#console-left-rail-outer-hint", Static)
        except (NoMatches, QueryError):
            return height
        if hint.display:
            height += max(1, hint.outer_size.height)
        return height

    @staticmethod
    def _measure_outer_content_height(outer: VerticalScroll) -> int:
        """Return the committed bottom edge of complete visible sections."""

        return max(
            (
                child.virtual_region_with_margin.bottom
                for child in outer.children
                if child.display
            ),
            default=0,
        )

    def _apply_allocation_state(
        self,
        sections: list[ConsoleBoundedSection],
        *,
        desired_outer_rows: int,
        needs_outer_hint: bool,
    ) -> None:
        """Equality-guard and synchronously apply complete outer geometry."""

        complete_state = (
            needs_outer_hint,
            desired_outer_rows,
            tuple(sections),
        )
        if complete_state == self._last_allocation_state:
            self._update_outer_hint()
            return

        outer = self.query_one("#console-left-rail-body", VerticalScroll)
        if str(outer.styles.overflow_y) != "auto":
            outer.styles.overflow_y = "auto"
        outer.can_focus = needs_outer_hint
        if not needs_outer_hint and outer.scroll_y != 0:
            outer.scroll_y = 0

        hint = self.query_one("#console-left-rail-outer-hint", Static)
        if hint.display is not needs_outer_hint:
            hint.display = needs_outer_hint
        self._outer_hint_exists = needs_outer_hint
        self._last_allocation_state = complete_state
        self._update_outer_hint()

    def _refresh_workspace_tree_after_reflow(self) -> None:
        """Schedule one settled check for whether the reflow moved tree rows.

        Deferred, not immediate: this is called from the END of an allocation
        pass, whose own style writes have not been laid out yet, so reading
        the tree's geometry here would compare the PREVIOUS layout against
        itself and miss the move this very pass causes. Coalesced, so the
        several legs that reach it in one frame settle once.
        """

        if self._workspace_tree_reflow_check_scheduled or not self.is_mounted:
            return
        self._workspace_tree_reflow_check_scheduled = True
        self.call_after_refresh(self._settle_workspace_tree_after_reflow)

    @staticmethod
    def _workspace_tree_reflow_signature(
        tree: ConsoleWorkspaceTree,
    ) -> tuple[object, ...] | None:
        """Return the cheap on-screen geometry identity of the tree's rows.

        Hover is pointer-anchored: it is stale exactly when the row under a
        stationary pointer changed, which for a rail reflow means the tree
        moved, resized, or scrolled. Content changes are NOT this leg's
        business -- ``sync_projection`` re-checks hover identity itself, and
        local scrolling is caught by ``watch_scroll_y``.

        Args:
            tree: The mounted workspace tree.

        Returns:
            A comparable signature, or ``None`` when the geometry cannot be
            read (the tree is not in the compositor yet), which is treated as
            "assume it moved" so the leg fails toward clearing.
        """

        try:
            return (tree.region, tree.scroll_offset)
        except Exception:
            return None

    def _settle_workspace_tree_after_reflow(self) -> None:
        """Clear stale hover + recompute truncation only if the tree moved."""

        self._workspace_tree_reflow_check_scheduled = False
        try:
            tree = self.query_one("#console-workspace-tree", ConsoleWorkspaceTree)
        except (NoMatches, QueryError):
            self._workspace_tree_reflow_state = None
            return
        signature = self._workspace_tree_reflow_signature(tree)
        if signature is not None and signature == self._workspace_tree_reflow_state:
            # The reconcile settled the rail back to the same geometry: the
            # row under the pointer is the row that was under it before, so
            # the hover highlight and its tooltip are still correct.
            return
        self._workspace_tree_reflow_state = signature
        if tree.hover_line >= 0:
            tree.hover_line = -1
        tree._update_tooltip()

    @staticmethod
    def _present_header_title(
        title: Static,
        *,
        base_title: str,
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
        if str(title.renderable) != base_title:
            title.update(base_title)
        return changed

    def _hidden_section_titles(self, outer: VerticalScroll) -> tuple[str, ...]:
        """Return the titles of sections whose header is below the fold.

        TASK-23195: the hint used to say only "more sections - scroll", which
        a user cannot act on -- it reports that something is hidden without
        saying whether it is the thing they want. Naming the sections costs
        the same single row.

        Args:
            outer: The rail's scroll owner, whose visible band bounds the fold.

        Returns:
            Section titles in DOM order, empty when everything is visible.
        """
        bottom = outer.region.y + outer.size.height
        below: list[str] = []
        for descriptor in self._mounted_descriptors():
            try:
                header = self.query_one(
                    f"#console-rail-section-header-{descriptor.section_id}",
                    DestinationRailSectionHeader,
                )
            except (NoMatches, QueryError):
                continue
            # Qodo review, PR #2233: strictly BELOW the visible band. An
            # earlier version treated anything outside it as hidden, so once
            # the user scrolled down the "▼" hint could name a section that
            # only scrolling UP reveals -- pointing the wrong way.
            if header.display and header.region.y >= bottom:
                below.append(descriptor.title)
        return tuple(below)

    def _outer_hint_copy(self, outer: VerticalScroll) -> str:
        """Compose the overflow hint, naming what is below the fold."""
        hidden = self._hidden_section_titles(outer)
        if not hidden:
            return OUTER_SECTION_SCROLL_HINT
        width = max(0, outer.size.width - 1)

        # Name as many as fit and count the rest. A 27-column rail cannot
        # hold "Agent · Details · Character", but it can hold "Agent · +2" --
        # and the first name is the actionable part, because it tells the
        # user what is immediately below the fold.
        for count in range(len(hidden), 0, -1):
            shown = " · ".join(hidden[:count])
            remainder = len(hidden) - count
            candidate = f"▼ {shown}" + (f" · +{remainder}" if remainder else "")
            if not width or cell_len(candidate) <= width:
                return candidate
        return f"▼ {len(hidden)} more — scroll"

    def _update_outer_hint(self) -> None:
        """Keep the pinned outer slot blank at end and exact before end."""

        try:
            outer = self.query_one("#console-left-rail-body", VerticalScroll)
            hint = self.query_one("#console-left-rail-outer-hint", Static)
        except (NoMatches, QueryError):
            return
        before_end = outer.max_scroll_y <= 0 or outer.scroll_y < outer.max_scroll_y
        text = (
            self._outer_hint_copy(outer)
            if self._outer_hint_exists and hint.display and before_end
            else ""
        )
        if text == self._outer_hint_text and str(hint.renderable) == text:
            return
        self._outer_hint_text = text
        hint.update(text)

    def _queue_pending_active_reveal(self) -> None:
        """Queue one deliberate reveal after complete geometry is committed."""

        pending = self._pending_active_reveal
        self._pending_active_reveal = None
        if pending is None:
            return
        generation, section_id, target_id, target_required = pending
        self.call_after_refresh(
            self._reveal_active_section,
            generation,
            section_id,
            target_id,
            target_required,
        )

    def _queue_active_reveal(
        self,
        section_id: str,
        target: Widget | None,
        *,
        generation: int | None = None,
    ) -> None:
        """Queue one bounded reveal guarded by current activation intent."""

        if generation is None:
            self._active_reveal_generation += 1
            generation = self._active_reveal_generation
        self.call_after_refresh(
            self._reveal_active_section,
            generation,
            section_id,
            target.id if target is not None else None,
            target is not None,
        )

    def _active_reveal_is_current(
        self,
        generation: int,
        section_id: str,
        target_id: str | None,
        target_required: bool,
    ) -> bool:
        """Reject delayed reveals after newer intent, focus change, or unmount."""

        if (
            not self.is_attached
            or generation != self._active_reveal_generation
            or section_id != self._active_section_id
        ):
            return False
        if target_required:
            if target_id is None:
                return False
            try:
                target = self.query_one(f"#{target_id}", Widget)
            except (NoMatches, QueryError):
                return False
            if not target.is_mounted or self.app.focused is not target:
                return False
        return True

    def _reveal_active_section(
        self,
        generation: int,
        section_id: str,
        target_id: str | None,
        target_required: bool,
    ) -> None:
        """Physically reveal the active header and first complete body rows."""

        if not self._active_reveal_is_current(
            generation,
            section_id,
            target_id,
            target_required,
        ):
            return
        try:
            outer = self.query_one("#console-left-rail-body", VerticalScroll)
            header = self.query_one(
                f"#console-rail-section-header-{section_id}",
                DestinationRailSectionHeader,
            )
            bounded = self.query_one(
                f"#console-bounded-section-{section_id}",
                ConsoleBoundedSection,
            )
        except (NoMatches, QueryError):
            return
        if not (outer.is_mounted and header.is_mounted and bounded.display):
            return
        outer.scroll_to(
            y=max(0, outer.scroll_y + header.region.y - outer.content_region.y),
            animate=False,
            immediate=True,
            force=True,
        )
        self._update_outer_hint()
        self._refresh_workspace_tree_after_reflow()

    def sync_workspace_context(self, state: ConsoleWorkspaceContextState) -> None:
        """Push one context snapshot into every scoped rail projection.

        Args:
            state: Shared Console workspace snapshot to project into the
                Workspaces, Conversations, and Details trays. (TASK-23199
                retired the Sessions tray; its active-chat summary moved
                into the Conversations projection.)

        Returns:
            None.
        """
        for selector in (
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
        try:
            tree = self.query_one("#console-workspace-tree", ConsoleWorkspaceTree)
        except (NoMatches, QueryError):
            pass
        else:
            tree.star_enabled = state.workspace_marks_available
            preferred = tree.preferred_expanded_workspace_ids
            if not tree.workspace_nodes and self._workspace_tree_expanded_ids is None:
                preferred = frozenset(
                    workspace.workspace_id for workspace in state.workspace_tree
                )
            query_active = bool(state.workspace_query.strip())
            if query_active and not tree.search_active:
                tree.set_search_active(True)
            tree.sync_projection(
                state.workspace_tree,
                expanded_workspace_ids=preferred,
            )
            if query_active:
                tree.set_search_active(
                    True,
                    forced_workspace_ids={
                        workspace.workspace_id for workspace in state.workspace_tree
                    },
                )
            elif tree.search_active:
                tree.set_search_active(False)
            try:
                workspace_tray = self.query_one(
                    "#console-workspaces-context", ConsoleWorkspaceContextTray
                )
            except (NoMatches, QueryError):
                pass
            else:
                cursor = tree.cursor_node
                workspace_tray.sync_workspace_tree_context(
                    cursor.data
                    if cursor is not None and cursor is not tree.root
                    else None
                )
            try:
                bounded = self.query_one(
                    "#console-bounded-section-workspace", ConsoleBoundedSection
                )
            except (NoMatches, QueryError):
                pass
            else:
                bounded.request_reconcile()
        self.request_allocation_reconcile()

    def compose(self) -> ComposeResult:
        """Compose the rail header, pinned fleet line, and context sections.

        Returns:
            The rail-header row, the pinned fleet-summary line, and the
            scrollable Workspaces/Conversations/Model/Agent/Details/Character
            section widgets, in mount order. (TASK-23199 retired the Sessions
            section; the active chat it named is shown by the Conversations
            browser on a selected row.)
        """
        rail_state = self._rail_state
        workspace_context_state = self._workspace_context_state

        left_rail_header = Horizontal(classes="console-rail-header")
        left_rail_header.styles.height = 1
        left_rail_header.styles.min_height = 1
        left_rail_header.styles.max_height = 1
        with left_rail_header:
            # TASK-23195: this row stays ONE full-width collapse target --
            # that large click area is deliberate (a previous task pinned
            # clicking anywhere along it, and the Inspector mirrors it). What
            # changed is the label. It used to read "<---------|Context":
            # hard-coded ASCII art that spent 18 of the rail's 27 columns on a
            # decorative arrow, buried the rail's only occurrence of its own
            # name inside the control that destroys it, and bypassed the
            # `ascii_glyphs` fallback every other Console glyph routes
            # through. It now reads "Context <glyph>", so the rail is named
            # and the affordance still resolves for ASCII terminals.
            collapse_button = Button(
                f"Context {resolve_glyph(GLYPH_COLLAPSE_LEFT)}",
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

        yield Button(
            "Terminal",
            id="console-terminal-open",
            classes="console-pinned-terminal-action",
            compact=True,
        )

        with _ContextOuterBody(
            owner=self,
            id="console-left-rail-body",
            classes="console-left-rail-body",
        ):
            # TASK-14810 split one mixed Session body into three peer
            # sections; TASK-23199 then retired the Sessions one, because
            # what it showed -- the active chat's name -- the Conversations
            # browser below already shows on a selected row marked "active
            # session". Workspace context and conversation browsing remain.
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
            workspace_context_tray.styles.height = "auto"
            workspace_context_tray.styles.max_height = 12
            workspace_context_tray.styles.overflow_y = "hidden"
            workspace_tree = ConsoleWorkspaceTree(id="console-workspace-tree")
            workspace_tree.star_enabled = (
                workspace_context_state.workspace_marks_available
            )
            workspace_tree.expansion_preferences_changed = (
                self._workspace_tree_expansion_preferences_changed
            )
            initial_expanded = self._workspace_tree_expanded_ids
            if initial_expanded is None:
                initial_expanded = frozenset(
                    workspace.workspace_id
                    for workspace in workspace_context_state.workspace_tree
                )
            workspace_tree.sync_projection(
                workspace_context_state.workspace_tree,
                expanded_workspace_ids=initial_expanded,
            )
            if workspace_context_state.workspace_query.strip():
                workspace_tree.set_search_active(
                    True,
                    forced_workspace_ids={
                        workspace.workspace_id
                        for workspace in workspace_context_state.workspace_tree
                    },
                )
            workspace_body = self._section_body(
                "workspace",
                rail_state.workspace_open,
                frame_console_region(workspace_context_tray, variant="quiet"),
            )
            workspace_body.styles.max_height = 12
            yield _ContextBoundedSection(
                workspace_body,
                section_id="workspace",
                owner=self,
                native_scroll_owner=workspace_tree,
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
            # TASK-23196: provider_row/model_row are deliberately NOT read
            # here any more; the status bar owns those two values.
            temperature_match = re.search(
                r"T ([\d.]+)", summary_state.sampling_row or ""
            )
            temperature_value = temperature_match.group(1) if temperature_match else "—"
            max_tokens_match = re.search(
                r"max_tokens (\d+)", summary_state.sampling_row or ""
            )
            max_tokens_value = max_tokens_match.group(1) if max_tokens_match else "—"

            # TASK-23196: the Provider and Model rows that stood here were
            # the third simultaneous rendering of the same two values -- the
            # persistent status bar and the Inspector's run recipe both
            # already show them, and the status bar carries both at every
            # width where this rail is shown at all (below 100 columns the
            # rail force-collapses). This was the copy that cost scarce
            # vertical space, so it is the copy that went. What remains is
            # what is NOT duplicated: the sampling parameters, the
            # system-prompt row, and Configure.
            model_rows = (
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
            generation_recovery = Vertical(
                Static(
                    "Not saved: generation settings",
                    markup=False,
                ),
                Button(
                    "Retry save",
                    id=CONSOLE_RETRY_GENERATION_SETTINGS_ID,
                    classes="console-workspace-action",
                    compact=True,
                ),
                id="console-generation-recovery-row",
            )
            context_recovery = Vertical(
                Static(
                    "Not saved: context settings",
                    id="console-context-recovery-copy",
                    markup=False,
                ),
                Button(
                    "Retry save",
                    id=CONSOLE_RETRY_CONTEXT_SETTINGS_ID,
                    classes="console-workspace-action",
                    compact=True,
                ),
                id="console-context-recovery-row",
            )
            default_recovery = Vertical(
                Static("", id="console-default-recovery-copy", markup=False),
                Button(
                    "Retry default save",
                    id=CONSOLE_RETRY_DEFAULT_SAVE_ID,
                    classes="console-workspace-action",
                    compact=True,
                ),
                Button(
                    "Discard retry",
                    id=CONSOLE_DISCARD_DEFAULT_RETRY_ID,
                    classes="console-workspace-action",
                    compact=True,
                ),
                Button(
                    "Refresh running app",
                    id=CONSOLE_REFRESH_RUNNING_APP_ID,
                    classes="console-workspace-action",
                    compact=True,
                ),
                Button(
                    "Dismiss",
                    id=CONSOLE_DISMISS_DEFAULT_REFRESH_ID,
                    classes="console-workspace-action",
                    compact=True,
                ),
                id="console-default-recovery-row",
            )
            generation_recovery.styles.display = "none"
            context_recovery.styles.display = "none"
            default_recovery.styles.display = "none"

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
                generation_recovery,
                context_recovery,
                default_recovery,
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
            # sub-agent runs). task-10: the fleet mini-section listing
            # those sub-agent runs moved to the right (Inspector) rail
            # (`#console-agent-section-subagents`, right_rail.py); this
            # section keeps the status/steps Statics, the drilldown
            # controls, and the steering bar.
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
                    avatar_children = (
                        self._character_avatar_widget_builder(
                            self._character_avatar_box
                        ),
                    )
                avatar_holder = ClickableAvatarBox(
                    *avatar_children,
                    id="console-character-avatar",
                )
                # task-1661: hug the image instead of claiming the rail.
                avatar_holder.styles.width = "auto"
                avatar_holder.styles.height = "auto"
                avatar_frame = Horizontal(
                    avatar_holder,
                    id="console-character-avatar-frame",
                )
                avatar_frame.styles.width = "100%"
                avatar_frame.styles.height = "auto"
                avatar_frame.styles.align_horizontal = "center"
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
                    avatar_frame,
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
        if button_id == "console-terminal-open":
            event.stop()
            self.post_message(self.TerminalRequested())
            return
        if not button_id.startswith(RAIL_SECTION_TOGGLE_PREFIX):
            return
        event.stop()
        self.screen.set_focus(event.button)
        section_id = button_id.removeprefix(RAIL_SECTION_TOGGLE_PREFIX)
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
        self.call_after_refresh(self._restore_section_toggle_focus, button_id)

    def _restore_section_toggle_focus(self, button_id: str) -> None:
        """Retain pointer focus after the toggle's layout reconciliation."""

        focused = self.app.focused
        if focused is not self and getattr(focused, "id", None) != button_id:
            return
        try:
            button = self.query_one(f"#{button_id}", Button)
        except (NoMatches, QueryError):
            return
        if self._is_enabled_focus_target(button):
            self.screen.set_focus(button)

    def on_workspace_tree_context_changed(
        self, event: WorkspaceTreeContextChanged
    ) -> None:
        """Project the native cursor into the visible contextual action."""

        event.stop()
        try:
            tray = self.query_one(
                "#console-workspaces-context", ConsoleWorkspaceContextTray
            )
        except (NoMatches, QueryError):
            return
        tray.sync_workspace_tree_context(event.data)

    def on_workspace_tree_focus_recovery_requested(
        self, event: WorkspaceTreeFocusRecoveryRequested
    ) -> None:
        """Return focus to the Workspaces disclosure when its Tree empties."""

        event.stop()
        try:
            disclosure = self.query_one(
                f"#{RAIL_SECTION_TOGGLE_PREFIX}workspace",
                Button,
            )
        except (NoMatches, QueryError):
            return
        disclosure.focus()

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

        Preserves the extracted screen behavior while hiding the bounded owner
        with its body so native local-scroll state survives collapse/reopen.

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
            bounded = self.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            )
        except (NoMatches, QueryError):
            return
        # Hide the same bounded owner along with its body. Its reconcile path
        # deliberately preserves scroll state while hidden, so reopening can
        # clamp the prior local offset against the newly measured geometry.
        if section_open:
            body.styles.display = "block"
            bounded.set_presented(True)
        else:
            bounded.set_presented(False)
            body.styles.display = "none"
        header.sync_open(section_open)
        if not section_open and self._active_section_id == section_id:
            self.recover_section_focus(section_id)
        elif section_open:
            bounded.request_reconcile()
        self.request_allocation_reconcile()
