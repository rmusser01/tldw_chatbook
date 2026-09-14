"""Design-system pattern gallery (ADR-161). Command-palette entry only.

Renders every canonical component family from ``css/patterns.json`` with the
production bundle's own rules: the gallery composes the canonical classes and
contributes no family styling of its own (``features/_pattern_gallery.tcss``
is layout glue only). The dark+light SVG snapshots in
``Tests/UI/snapshots/pattern_gallery/`` pin this rendering (spec 3.4).
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
    TextArea,
)

#: This gallery's own source path. The governance test (ADR-161 task 6)
#: greps it for every canonical class declared in ``css/patterns.json``.
GALLERY_SOURCE = Path(__file__)


class PatternGalleryScreen(Screen):
    """Live preview of every canonical component family (ADR-161)."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="pg-root"):
            yield Label(
                "Pattern Gallery — canonical component families (ADR-161)",
                classes="section-title",
            )
            # forms
            yield Label("Forms & fields", classes="section-header")
            with Container(classes="pg-frame-forms"):
                with Container(classes="form-col"):
                    yield Label("Model", classes="form-label")
                    yield Input(placeholder="gpt-4o", classes="form-input")
                    yield Label("System prompt", classes="form-label")
                    yield TextArea(classes="form-textarea")
                    yield Select(
                        [("option-a", "a"), ("option-b", "b")], classes="form-select"
                    )
                    yield Checkbox("Stream response", classes="form-checkbox")
                    with Horizontal(classes="form-row"):
                        yield Label("Temperature", classes="form-label")
                        yield Input(placeholder="0.7", classes="form-input")
                    # ADR-161 task 7: the settings grid-row label variant --
                    # fixed 24-col column with panel background; a documented
                    # variant of form-label, NOT a merge candidate (probe).
                    with Horizontal(classes="form-row"):
                        yield Static(
                            "Palette limit", classes="settings-input-label"
                        )
                        yield Static("24", classes="settings-compact-input")
                    yield Label(
                        "Generation parameters", classes="form-section-title"
                    )
                    with Collapsible(
                        title="Advanced", classes="form-section-collapsible"
                    ):
                        yield Label("Top-p", classes="form-label")
                        yield Input(placeholder="1.0", classes="form-input")
                    with Horizontal(classes="form-actions"):
                        yield Button("Save", classes="form-button")
                        yield Button("Reset", classes="form-button")
            # buttons
            yield Label("Buttons & action rows", classes="section-header")
            with Container(classes="button-group button-group-left"):
                yield Button("Primary", classes="form-button")
                yield Button("Disabled", disabled=True)
            with Container(classes="button-group button-group-center"):
                yield Button("Action", classes="action-button")
                yield Button("Primary action", classes="action-button primary")
            with Container(classes="button-group button-group-right"):
                yield Button("Right-aligned", classes="action-button")
            yield Button("Sidebar toggle", classes="sidebar-toggle")
            # lists
            yield Label("Lists & tables", classes="section-header")
            with Container(classes="pg-frame-list"):
                yield ListView(
                    ListItem(Label("ready — succeeded row")),
                    ListItem(Label("running — active row")),
                    ListItem(Label("blocked — failed row")),
                )
            # dialogs (structure preview, not a pushed modal)
            yield Label("Dialogs & modals", classes="section-header")
            with Container(classes="pg-dialog-frame"):
                yield Label("Confirm deletion", classes="dialog-title")
                yield Static("This cannot be undone.", classes="help-text")
                with Container(
                    classes="dialog-buttons button-group button-group-right"
                ):
                    yield Button("Cancel")
                    yield Button("Delete")
            # status
            yield Label("Status, empty & loading", classes="section-header")
            with Container(classes="pg-frame-status"):
                with Container(classes="status-area"):
                    yield Label("ready", classes="status-label")
                    yield Label("running", classes="status-label")
                    yield Label("approval-required", classes="status-label")
            # navigation
            yield Label("Navigation & sidebars", classes="section-header")
            with Container(classes="pg-nav-frame"):
                with Container(classes="sidebar"):
                    yield Label("Sidebar header", classes="sidebar-header")
                    yield Button("Nav item", classes="sidebar-button")
                    yield Button("Toggle pane", classes="sidebar-toggle")
                    yield ListView(
                        ListItem(Label("alpha")),
                        ListItem(Label("beta")),
                        classes="sidebar-listview",
                    )
                    with Collapsible(
                        title="Section", classes="sidebar-section-collapsible"
                    ):
                        yield Button("Nested destination", classes="nav-button")
            # messages
            yield Label("Chat & message rendering", classes="section-header")
            yield Label("Transcript preview", classes="subsection-title")
            yield Static(
                "user: speaks with $ds-chat-user-accent", classes="pg-msg-user"
            )
            yield Static(
                "assistant: speaks with the chat accent family",
                classes="pg-msg-assistant",
            )
            # ds-primitives
            yield Label("ds-primitives", classes="section-header")
            yield Static("Destination header", classes="ds-destination-header")
            with Container(classes="ds-toolbar"):
                yield Button("Tool A")
                yield Button("Tool B")
            with Container(classes="ds-info-callout"):
                yield Label("Info callout — informational surface")
            with Container(classes="ds-approval-card"):
                yield Label("Approval card — approval-required surface")
            with Container(classes="ds-panel"):
                yield Label("Panel content", classes="ds-field-row")


class PatternGalleryProvider(Provider):
    """Command-palette entry that opens the pattern gallery."""

    COMMANDS = (
        (
            "Design System: Pattern Gallery",
            "open_pattern_gallery",
            "Browse every canonical component pattern live",
        ),
    )

    async def discover(self) -> Hits:
        for text, _id, help_text in self.COMMANDS:
            yield Hit(
                1.0,
                text,
                lambda: self.app.push_screen(PatternGalleryScreen()),
                help=help_text,
            )

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for text, _id, help_text in self.COMMANDS:
            if (score := matcher.match(text)) > 0:
                yield Hit(
                    score,
                    matcher.highlight(text),
                    lambda: self.app.push_screen(PatternGalleryScreen()),
                    help=help_text,
                )
