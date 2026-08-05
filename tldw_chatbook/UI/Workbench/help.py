"""Contextual help for visible Workbench actions."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchAction


@dataclass(frozen=True)
class WorkbenchHelpState:
    """Plain-text help content for the current Workbench route."""

    route_id: str
    title: str
    actions: tuple[WorkbenchAction, ...] = ()
    shortcuts: tuple[tuple[str, str], ...] = ()
    #: TASK-362: grouped keyboard map (group name -> (key, label) pairs). When
    #: present it replaces the flat ``shortcuts`` list so the help can surface the
    #: full vocabulary (panes, transcript, composer, modals), not just the handful
    #: of top-level bindings.
    shortcut_groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    #: TASK-1232: an optional free-text section (e.g. Console's "Agents"
    #: primer) rendered between Actions and Shortcuts. Generic rather than
    #: Console-specific so any other route can reuse it the same way.
    notes_heading: str = ""
    notes: tuple[str, ...] = ()

    def render_text(self) -> str:
        """Render visible actions and explicit shortcuts as plain text."""
        lines = [self.title]
        visible_actions = tuple(
            action for action in self.actions if not action.disabled
        )
        if visible_actions:
            lines.append("Actions:")
            lines.extend(f"- {action.label}" for action in visible_actions)
        if self.notes:
            lines.append(f"{self.notes_heading}:" if self.notes_heading else "Notes:")
            lines.extend(f"- {note}" for note in self.notes)
        if self.shortcut_groups:
            lines.append("Shortcuts:")
            for group_name, group_shortcuts in self.shortcut_groups:
                lines.append(f"  {group_name}:")
                lines.extend(
                    f"    {key}: {label}" for key, label in group_shortcuts
                )
        elif self.shortcuts:
            lines.append("Shortcuts:")
            lines.extend(f"- {key}: {label}" for key, label in self.shortcuts)
        return "\n".join(lines)


class WorkbenchHelpPanel(ModalScreen[None]):
    """Modal panel showing contextual Workbench help.

    Fleet-UX expert review F2 follow-up (task-1232 round 1, Critical): before
    this fix ``#workbench-help-panel`` was a plain ``Vertical`` with NO CSS
    anywhere, so it inherited Textual's ``Vertical`` defaults
    (``height: 1fr``, ``overflow: hidden hidden``) -- the panel silently
    filled and then HARD-CLIPPED at the screen edge with no scrollbar, so
    any help content past the fold (Console's new "Agents" section among
    it) was simply unreachable at ordinary terminal sizes. The body now
    lives in its own ``VerticalScroll`` (bounded height + a visible
    scrollbar via CSS -- see ``css/components/_workbench.tcss``) while the
    Close button stays outside it, pinned at the bottom of the bounded
    panel (mirrors ``ConsoleScopePickerModal``'s scroll-body-plus-fixed-
    footer structure).
    """

    BINDINGS = [Binding("escape", "dismiss", "Close", show=False)]

    # KEEP IN SYNC with the live bundle source
    # css/components/_workbench.tcss (the "task-1232 round 1" block): this
    # DEFAULT_CSS is a structural SUBSET using only built-in Textual theme
    # variables ($primary/$surface), so the panel is correctly bounded and
    # scrollable even in a stylesheet-less test harness that never loads the
    # app's CSS bundle (mirrors AppFooterStatus.DEFAULT_CSS's own rationale
    # -- WorkbenchHelpPanel is shared infra invoked from many screens'
    # lightweight test harnesses, not just Console's). The bundle's richer
    # `$ds-*`-token rule wins by origin in production for color/border
    # polish; this copy only has to get sizing/scrolling right.
    DEFAULT_CSS = """
    WorkbenchHelpPanel {
        align: center middle;
    }

    WorkbenchHelpPanel #workbench-help-panel {
        width: 76;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }

    WorkbenchHelpPanel #workbench-help-scroll {
        height: 1fr;
        min-height: 3;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-size: 1 1;
    }

    WorkbenchHelpPanel #workbench-help-body {
        width: 100%;
        height: auto;
    }

    WorkbenchHelpPanel #workbench-help-close {
        width: auto;
        height: auto;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, state: WorkbenchHelpState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="workbench-help-panel", classes="workbench-help-panel"):
            with VerticalScroll(id="workbench-help-scroll"):
                yield Static(self.state.render_text(), id="workbench-help-body")
            yield Button("Close", id="workbench-help-close", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss the panel from its close button."""
        if event.button.id == "workbench-help-close":
            event.stop()
            self.dismiss(None)
