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
                    # ADR-161 task 9: the read-only detail-line variant --
                    # panel fill, primary text, one-row minimum; a documented
                    # variant of form-row (probe), consumed by Settings and
                    # the STTS speech settings pane.
                    yield Static(
                        "Saved profile: default", classes="settings-detail-row"
                    )
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
                # ADR-161 task 8a: the per-line status entries (wizard
                # progress lists) -- canonical states completed/active/error.
                yield Label("validated", classes="status-item completed")
                yield Label("exporting conversations", classes="status-item active")
                yield Label("finalize failed", classes="status-item error")
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
            # ADR-161 task 8c: the shared bubble grammar (header band,
            # body, action strip) -- same classes ChatMessage (Console
            # transcripts) and ChatMessageEnhanced (Chat window) compose.
            with Container(classes="pg-msg-frame"):
                yield Label("user", classes="message-header")
                yield Static("message body", classes="message-text")
                with Horizontal(classes="message-actions"):
                    yield Button("Copy", classes="action-button")
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
            # Additional override utilities are exercised below in bounded samples.
            # sizing utilities (ADR-161 task 12): the one-dimensional
            # companions. Fixed-value rows render directly; the share-based
            # heights (h-full/h-fill) need definite-height parents, so each
            # gets its own bounded frame instead of the auto strip.
            yield Label("Sizing & box-model utilities", classes="section-header")
            with Container(classes="pg-frame-sizing"):
                yield Static("w-full", classes="w-full")
                yield Static("w-fill", classes="w-fill")
                yield Static("w-auto", classes="w-auto")
                yield Static("w-0", classes="w-0")
                yield Static("h-1", classes="h-1")
                yield Static("h-2", classes="h-2")
                yield Static("h-3", classes="h-3")
                yield Static("h-0", classes="h-0")
                yield Static("h-auto", classes="h-auto")
                yield Static("p-0", classes="p-0")
                yield Static("m-0", classes="m-0")
                yield Static("mt-1", classes="mt-1")
                yield Static("mb-0", classes="mb-0")
                yield Static("border-none", classes="border-none")
            with Container(classes="pg-frame-sizing-share"):
                yield Static("h-full", classes="h-full")
            with Container(classes="pg-frame-sizing-share"):
                yield Static("h-fill", classes="h-fill")

            # Bounded cells show each explicit override, including finite sizes.
            for offset in range(0, len(EXTENDED_UTILITIES), 4):
                with Horizontal(classes="pg-utility-row"):
                    for utility in EXTENDED_UTILITIES[offset:offset + 4]:
                        with Container(classes="pg-utility-cell"):
                            yield Label(utility)
                            yield Static("Sample", classes=utility)


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


EXTENDED_UTILITIES = (
    "w-7",
    'w-1',
    'w-3',
    'w-4',
    'w-5',
    'w-9',
    'w-11',
    'w-12',
    'w-13',
    'w-17',
    'w-18',
    'w-20',
    'w-21',
    'w-23',
    'w-24',
    'w-28',
    'w-3fr',
    'w-4fr',
    'w-13fr',
    'h-4',
    'h-5',
    'h-6',
    'h-7',
    'h-8',
    'h-9',
    'p-inline-1',
    'm-left-2',
    'm-left-3',
    'ds-text-error',
    'ds-text-warning',
    'ds-text-ready',
    'ds-text-primary',
    'ds-text-muted',
    'w-6',
    'w-8',
    'w-10',
    'w-16',
    'h-10',
    'p-left-1',
)
