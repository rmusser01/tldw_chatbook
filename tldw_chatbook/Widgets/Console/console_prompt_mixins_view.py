"""Mix-in controls within the existing Prompt Workbench."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Checkbox, Input, Select, Static

from ...Chat.console_prompt_mixins import MixinPreferences, MixinSourceRef
from ...Prompt_Management.prompt_artifact_models import LiteralMixinDefinition
from ..Prompts.prompt_mixin_editor import PromptMixinEditor


@dataclass(frozen=True)
class MixinWorkbenchIntent:
    """An exact rendered chat/revision action, never an ambient active chat."""

    session_id: str
    incarnation_id: str
    revision: int
    action: Literal["add", "toggle", "up", "down", "remove", "inspect", "retry"]
    source: MixinSourceRef | None = None


@dataclass(frozen=True)
class MixinWorkbenchProjection:
    session_id: str
    incarnation_id: str
    preferences: MixinPreferences = field(repr=False)
    candidates: tuple[tuple[str, MixinSourceRef], ...] = field(default=(), repr=False)
    status: str = ""
    unsaved: bool = False
    unavailable: frozenset[MixinSourceRef] = frozenset()
    updated: frozenset[MixinSourceRef] = frozenset()
    page: int = 1
    total_pages: int = 1
    query: str = field(default="", repr=False)


class ConsolePromptMixinsView(Vertical):
    """Render owner projections and emit typed preference intents."""

    class IntentRequested(Message):
        def __init__(self, intent: MixinWorkbenchIntent) -> None:
            self.intent = intent
            super().__init__()

    class BrowseRequested(Message):
        pass

    class RefreshRequested(Message):
        def __init__(self, query: str, page: int) -> None:
            self.query, self.page = query, page
            super().__init__()

    class SaveRequested(Message):
        pass

    class PreviewRequested(Message):
        pass

    def __init__(self, projection: MixinWorkbenchProjection, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.projection = projection
        self._ready = False
        self.editor_record = None
        self.editor_initial = None

    def compose(self) -> ComposeResult:
        projection = self.projection
        enabled = sum(row.enabled for row in projection.preferences.entries)
        yield Static(
            f"Mix-ins · {enabled} enabled", id="console-mixins-count", markup=False
        )
        yield Static(
            "Before / your message / After. Changes apply to future sends.",
            markup=False,
        )
        with Horizontal(classes="console-mixins-actions"):
            yield Button("Browse Prompts / Recipes", id="console-mixins-browse")
            yield Button("Preview next send", id="console-mixins-preview")
        status = Static(projection.status, id="console-mixins-status", markup=False)
        status.display = bool(projection.status)
        yield status
        retry = Button("Retry saving choices", id="console-mixins-retry")
        retry.display = projection.unsaved
        yield retry
        with VerticalScroll(id="console-mixins-scroll"):
            for placement in ("before", "after"):
                yield Static(placement.title(), classes="section-title")
                grouped = [
                    (i, row)
                    for i, row in enumerate(projection.preferences.entries)
                    if row.placement == placement
                ]
                if not grouped:
                    yield Static("None selected", classes="status-label", markup=False)
                for position, (index, row) in enumerate(grouped):
                    with Vertical(classes="console-mixin-row"):
                        suffix = (
                            " · unavailable"
                            if row.source in projection.unavailable
                            else " · updated since last send"
                            if row.source in projection.updated
                            else ""
                        )
                        yield Checkbox(
                            Text(row.name + suffix),
                            value=row.enabled,
                            id=f"console-mixin-enabled-{index}",
                        )
                        with Horizontal(classes="console-mixins-actions"):
                            yield Button(
                                "Inspect / Edit", id=f"console-mixin-inspect-{index}"
                            )
                            yield Button(
                                "Move up",
                                id=f"console-mixin-up-{index}",
                                disabled=position == 0,
                            )
                            yield Button(
                                "Move down",
                                id=f"console-mixin-down-{index}",
                                disabled=position == len(grouped) - 1,
                            )
                            yield Button("Remove", id=f"console-mixin-remove-{index}")
            yield Static("Add from Local Library", classes="section-title")
            yield Static(
                "Server mix-ins cannot be enabled: revision snapshots are unsupported.",
                classes="status-label",
                markup=False,
            )
            yield Input(
                value=projection.query,
                placeholder="Search mix-ins",
                id="console-mixins-query",
            )
            with Horizontal(classes="console-mixins-actions"):
                yield Select(
                    tuple(
                        (Text(name), str(i))
                        for i, (name, _) in enumerate(projection.candidates)
                    ),
                    prompt="Choose a mix-in",
                    id="console-mixins-library",
                )
                yield Button(
                    "Add", id="console-mixins-add", disabled=not projection.candidates
                )
                yield Button("Search", id="console-mixins-search")
            with Horizontal(classes="console-mixins-actions"):
                yield Button(
                    "Previous",
                    id="console-mixins-previous",
                    disabled=projection.page <= 1,
                )
                yield Static(
                    f"Page {projection.page} of {max(1, projection.total_pages)}",
                    classes="w-fill",
                    markup=False,
                )
                yield Button(
                    "Next",
                    id="console-mixins-next",
                    disabled=projection.page >= projection.total_pages,
                )
            yield Vertical(id="console-mixins-editor", classes="h-auto")
            yield Static(
                "", id="console-mixins-preview-content", classes="h-auto", markup=False
            )

    def on_mount(self) -> None:
        self._ready = True

    @property
    def dirty(self) -> bool:
        editors = self.query(PromptMixinEditor)
        editor = editors.first() if editors else None
        return editor is not None and editor.read_draft() != self.editor_initial

    def intent(
        self, action: str, source: MixinSourceRef | None = None
    ) -> MixinWorkbenchIntent:
        p = self.projection
        return MixinWorkbenchIntent(
            p.session_id, p.incarnation_id, p.preferences.revision, action, source
        )

    @on(Checkbox.Changed)
    def _toggle(self, event: Checkbox.Changed) -> None:
        event.stop()
        if self._ready:
            index = int(event.checkbox.id.rsplit("-", 1)[1])
            row = self.projection.preferences.entries[index]
            if event.value != row.enabled:
                self.post_message(
                    self.IntentRequested(self.intent("toggle", row.source))
                )

    @on(Button.Pressed)
    def _button(self, event: Button.Pressed) -> None:
        event.stop()
        identifier = event.button.id or ""
        if identifier.startswith("console-mixin-"):
            action, index = identifier.removeprefix("console-mixin-").rsplit("-", 1)
            row = self.projection.preferences.entries[int(index)]
            self.post_message(self.IntentRequested(self.intent(action, row.source)))
        elif identifier == "console-mixins-add":
            value = self.query_one("#console-mixins-library", Select).value
            if value is not Select.BLANK:
                self.post_message(
                    self.IntentRequested(
                        self.intent("add", self.projection.candidates[int(value)][1])
                    )
                )
        elif identifier == "console-mixins-browse":
            self.post_message(self.BrowseRequested())
        elif identifier == "console-mixins-retry":
            self.post_message(self.IntentRequested(self.intent("retry")))
        elif identifier == "console-mixins-save":
            self.post_message(self.SaveRequested())
        elif identifier == "console-mixins-preview":
            self.post_message(self.PreviewRequested())
        elif identifier in {
            "console-mixins-search",
            "console-mixins-previous",
            "console-mixins-next",
        }:
            page = (
                self.projection.page + (1 if identifier.endswith("-next") else -1)
                if not identifier.endswith("-search")
                else 1
            )
            self.post_message(
                self.RefreshRequested(
                    self.query_one("#console-mixins-query", Input).value, page
                )
            )

    @on(PromptMixinEditor.Changed)
    def _editor_changed(self) -> None:
        status = self.query_one("#console-mixins-status", Static)
        if self.dirty:
            status.display = True
            status.update(
                "Library edit unsaved. Save Library changes, or close and discard the edit."
            )

    async def show_editor(
        self,
        record: Mapping[str, Any],
        definition: LiteralMixinDefinition,
        *,
        is_current: Callable[[], bool],
    ) -> None:
        if not is_current():
            return
        area = self.query_one("#console-mixins-editor", Vertical)
        await area.remove_children()
        if not is_current():
            return
        self.editor_record = record
        editor = PromptMixinEditor(record["name"], definition)
        self.editor_initial = editor.draft
        await area.mount(
            editor, Button("Save Library changes", id="console-mixins-save")
        )
        if not is_current():
            return
        editor.query_one("#mixin-name", Input).focus()
        editor.scroll_visible(animate=False, immediate=True)
