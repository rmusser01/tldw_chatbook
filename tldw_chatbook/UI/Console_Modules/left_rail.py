"""Console shell left rail — session/model/agent/details/character region.

Extracted verbatim out of ``ChatScreen.compose_content`` (wave-1 console
decomposition, task 3): the subtree that used to live inside
``with self._frame_console_region(left_rail):`` — the rail header, the
pinned fleet-summary line, and the five scrollable sections (Session,
Model, Agent, Details, Character). Ids and nesting inside this subtree are
preserved exactly.

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
"Migration safety" section on this exact class of defect). "Same nesting"
for the ids this widget DOES own is preserved with zero structural
change: this class reuses ``id="console-left-rail"`` as its own root, so
it sits in the DOM exactly where the old ``Vertical`` sat.

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

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from ...Chat.console_glyphs import GLYPH_COLLAPSE_LEFT
from ...Chat.console_rail_state import CONSOLE_RAIL_SECTION_IDS, ConsoleRailState
from ...Chat.console_session_settings import (
    ConsoleSettingsSummaryState,
    _summary_row_value,
)
from ...Widgets.Console import ConsoleWorkspaceContextTray
from ...Widgets.Console.console_image_viewer_modal import ClickableAvatarBox
from ...Widgets.Console.console_workspace_details import ConsoleWorkspaceDetailsTray
from ...Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
    DestinationRailSectionHeader,
)
from ...Workspaces.display_state import ConsoleWorkspaceContextState
from .frame import frame_console_region


class ConsoleLeftRail(Vertical):
    """The Console shell's left rail: Session/Model/Agent/Details/Character.

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

        Args:
            section_id: The toggled section's id (e.g. ``"session"``).
            opened: Whether the section appears open in this rail's own
                last-synced header state (informational; the screen
                recomputes the authoritative next state itself).
        """

        def __init__(self, section_id: str, opened: bool) -> None:
            self.section_id = section_id
            self.opened = opened
            super().__init__()

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
        agent_subagents_text: str,
        agent_drilldown_active: bool,
        agent_full_log_available: bool,
        show_character_section: bool,
        character_avatar_widget: Widget | None,
        character_avatar_name: str,
        **kwargs,
    ) -> None:
        """Create the left rail from pre-computed display data.

        Args:
            rail_state: Effective Console rail state (open flags, labels).
            workspace_context_state: Shared workspace/conversation display
                state, reused for both the Session tray and the Details tray
                exactly as the pre-extraction inline compose did.
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
            agent_subagents_text: Agent section sub-agent list text.
            agent_drilldown_active: Whether a sub-agent drill-down is
                active, driving the "Back" button's visibility.
            agent_full_log_available: Whether the "View full log" button
                should be visible.
            show_character_section: Whether the Character section is
                composed at all (config-gated; matches
                ``resolve_show_character_avatar``).
            character_avatar_widget: Pre-built avatar widget, when
                ``show_character_section`` is True.
            character_avatar_name: Character name label text, when
                ``show_character_section`` is True.
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
        self._agent_subagents_text = agent_subagents_text
        self._agent_drilldown_active = agent_drilldown_active
        self._agent_full_log_available = agent_full_log_available
        self._show_character_section = show_character_section
        self._character_avatar_widget = character_avatar_widget
        self._character_avatar_name = character_avatar_name

    def compose(self) -> ComposeResult:
        """Compose the rail header, pinned fleet line, and the five sections."""
        rail_state = self._rail_state
        workspace_context_state = self._workspace_context_state

        left_rail_header = Horizontal(classes="console-rail-header")
        left_rail_header.styles.height = 1
        left_rail_header.styles.min_height = 1
        left_rail_header.styles.max_height = 1
        with left_rail_header:
            # The "Context" (staged sources) section moved into the
            # Inspector rail (task-400); this title now only names the
            # rail itself.
            rail_label = Static(
                "Console context",
                id="console-context-rail-title",
                classes="console-rail-title",
            )
            rail_label.styles.width = "1fr"
            yield rail_label
            collapse_button = Button(
                GLYPH_COLLAPSE_LEFT,
                id="console-context-rail-collapse",
                classes="console-rail-collapse-button",
                compact=True,
            )
            collapse_button.tooltip = "Collapse Console context rail"
            collapse_button.styles.width = 3
            collapse_button.styles.min_width = 3
            collapse_button.styles.max_width = 3
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

        with VerticalScroll(
            id="console-left-rail-body",
            classes="console-left-rail-body",
        ):
            # Section 1: Session (workspace + conversations).
            yield DestinationRailSectionHeader(
                "Session",
                section_id="session",
                open=rail_state.session_open,
                id="console-rail-section-header-session",
            )
            session_body = Vertical(
                id="console-rail-section-body-session",
                classes="console-rail-section-body",
            )
            session_body.styles.height = "auto"
            if not rail_state.session_open:
                session_body.styles.display = "none"
            with session_body:
                workspace_context_tray = ConsoleWorkspaceContextTray(
                    workspace_context_state,
                    show_heading=False,
                    id="console-workspace-context",
                    classes="console-left-rail-section",
                )
                workspace_context_tray.styles.width = "100%"
                workspace_context_tray.styles.min_width = 0
                # Always "quiet": mirrors
                # `ChatScreen._workspace_context_frame_variant`, which keeps
                # this tray visually nested inside the rail's own frame
                # rather than drawing a second border.
                yield frame_console_region(workspace_context_tray, variant="quiet")

            # Section 2: Model (provider/model readout lines plus a
            # Configure shortcut into the Console session settings).
            yield DestinationRailSectionHeader(
                "Model",
                section_id="model",
                open=rail_state.model_open,
                id="console-rail-section-header-model",
            )
            model_body = Vertical(
                id="console-rail-section-body-model",
                classes="console-rail-section-body",
            )
            model_body.styles.height = "auto"
            if not rail_state.model_open:
                model_body.styles.display = "none"
            with model_body:
                summary_state = self._settings_summary_state
                provider_value = _summary_row_value(summary_state.provider_row) or "—"
                model_value = _summary_row_value(summary_state.model_row) or "—"
                temperature_match = re.search(
                    r"T ([\d.]+)", summary_state.sampling_row or ""
                )
                temperature_value = (
                    temperature_match.group(1) if temperature_match else "—"
                )
                max_tokens_match = re.search(
                    r"max_tokens (\d+)", summary_state.sampling_row or ""
                )
                max_tokens_value = (
                    max_tokens_match.group(1) if max_tokens_match else "—"
                )

                with Horizontal(
                    id="console-model-section-provider",
                    classes="console-model-section-line",
                ):
                    yield Static(
                        "Provider",
                        classes="console-model-section-label",
                        markup=False,
                    )
                    yield Static(
                        provider_value,
                        classes="console-model-section-value",
                        markup=False,
                    )
                with Horizontal(
                    id="console-model-section-model",
                    classes="console-model-section-line",
                ):
                    yield Static(
                        "Model",
                        classes="console-model-section-label",
                        markup=False,
                    )
                    yield Static(
                        model_value,
                        classes="console-model-section-value",
                        markup=False,
                    )
                with Horizontal(
                    id="console-model-section-temperature",
                    classes="console-model-section-line",
                ):
                    yield Static(
                        "Temperature",
                        classes="console-model-section-label",
                        markup=False,
                    )
                    yield Static(
                        temperature_value,
                        classes="console-model-section-value",
                        markup=False,
                    )
                with Horizontal(
                    id="console-model-section-max-tokens",
                    classes="console-model-section-line",
                ):
                    yield Static(
                        "Max tokens",
                        classes="console-model-section-label",
                        markup=False,
                    )
                    yield Static(
                        max_tokens_value,
                        classes="console-model-section-value",
                        markup=False,
                    )

                readiness = (summary_state.readiness_label or "").strip()
                recovery = Static(
                    readiness or "",
                    id="console-model-section-recovery",
                    classes="console-model-section-recovery",
                    markup=False,
                )
                recovery.styles.display = "none"
                yield recovery

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
                system_line.set_class(
                    self._system_line_dim, "console-rail-system-line-dim"
                )
                yield system_line
                configure = Button(
                    "Configure",
                    id="console-model-section-configure",
                    classes="console-workspace-action",
                    compact=True,
                )
                configure.tooltip = "Configure Console session settings"
                yield configure

            # Section 3: Agent (run inspector -- the watch-and-drill surface
            # for the live/most-recent agent run and its historical
            # sub-agent runs).
            yield DestinationRailSectionHeader(
                "Agent",
                section_id="agent",
                open=rail_state.agent_open,
                id="console-rail-section-header-agent",
            )
            agent_body = Vertical(
                id="console-rail-section-body-agent",
                classes="console-rail-section-body console-agent-section",
            )
            agent_body.styles.height = "auto"
            if not rail_state.agent_open:
                agent_body.styles.display = "none"
            with agent_body:
                yield Static(
                    self._agent_status_line,
                    id="console-agent-section-status",
                    classes="console-agent-section-line",
                    markup=False,
                )
                yield Static(
                    self._agent_steps_text,
                    id="console-agent-section-steps",
                    classes="console-agent-section-steps",
                    markup=False,
                )
                yield Static(
                    self._agent_subagents_text,
                    id="console-agent-section-subagents",
                    classes="console-agent-section-subagents",
                    markup=False,
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
                yield back_button
                # TASK-870 (AC#6/#7): the full-run-log affordance -- present
                # only while a run log actually exists for whatever run this
                # section is currently showing.
                full_log_button = Button(
                    "View full log",
                    id="console-agent-view-full-log",
                    classes=(
                        "console-workspace-action console-agent-view-full-log"
                    ),
                    compact=True,
                )
                full_log_button.tooltip = (
                    "Open the full, untruncated run log for this run -- "
                    "what the model actually saw, before the Console's "
                    "display cap trimmed it."
                )
                if not self._agent_full_log_available:
                    full_log_button.styles.display = "none"
                yield full_log_button

            # Section 4: Details (storage, sync, handoff plumbing).
            yield DestinationRailSectionHeader(
                "Details",
                section_id="details",
                open=rail_state.details_open,
                id="console-rail-section-header-details",
            )
            details_body = Vertical(
                id="console-rail-section-body-details",
                classes="console-rail-section-body",
            )
            details_body.styles.height = "auto"
            if not rail_state.details_open:
                details_body.styles.display = "none"
            with details_body:
                details_tray = ConsoleWorkspaceDetailsTray(
                    workspace_context_state,
                    id="console-workspace-details",
                    classes="console-left-rail-section",
                )
                details_tray.styles.width = "100%"
                details_tray.styles.min_width = 0
                yield details_tray

            # Section 5: Character (avatar of the active character).
            if self._show_character_section:
                yield DestinationRailSectionHeader(
                    "Character",
                    section_id="character",
                    open=rail_state.character_open,
                    id="console-rail-section-header-character",
                )
                character_body = Vertical(
                    id="console-rail-section-body-character",
                    classes="console-rail-section-body",
                )
                character_body.styles.height = "auto"
                if not rail_state.character_open:
                    character_body.styles.display = "none"
                with character_body:
                    avatar_holder = ClickableAvatarBox(id="console-character-avatar")
                    # task-1661: Container defaults to width/height 1fr, so
                    # the holder claimed the entire rail section -- the
                    # portrait sat in the corner of a tall empty box with
                    # the name pushed to the bottom. Hug the image instead.
                    avatar_holder.styles.width = "auto"
                    avatar_holder.styles.height = "auto"
                    with avatar_holder:
                        if self._character_avatar_widget is not None:
                            yield self._character_avatar_widget
                    yield Static(
                        self._character_avatar_name,
                        id="console-character-name",
                    )

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
        """
        button_id = event.button.id or ""
        if not button_id.startswith(RAIL_SECTION_TOGGLE_PREFIX):
            return
        event.stop()
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

    def sync_sections(self, rail_state: ConsoleRailState) -> None:
        """Apply section open flags to section bodies and headers.

        Moved verbatim from ``ChatScreen._sync_console_rail_sections``:
        stored section preferences are scoped per workspace/conversation,
        so a runtime scope switch (for example resuming a saved
        conversation after a relaunch) can change the effective flags
        without a recompose.
        """
        for section_id in CONSOLE_RAIL_SECTION_IDS:
            section_open = bool(getattr(rail_state, f"{section_id}_open", True))
            self.apply_section_open(section_id, section_open)

    def apply_section_open(self, section_id: str, section_open: bool) -> None:
        """Sync one section's body display and header glyph to an open state.

        Moved verbatim from ``ChatScreen._apply_console_rail_section_open``.
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
