"""Path-free Persona Visual authoring section for the Persona editor."""

from __future__ import annotations

from typing import Any, Literal

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

from ...Persona_Visual.authoring import (
    PersonaVisualDraftInventory,
    PersonaVisualDraftRow,
)


class _PersonaVisualStateRequested(Message):
    """Base for a path-free action targeting one visible state."""

    def __init__(self, state: str) -> None:
        self.state = state
        super().__init__()


class PersonaVisualPreviewRequested(_PersonaVisualStateRequested):
    """Ask the screen to resolve only the selected draft state."""


class PersonaVisualReplaceRequested(_PersonaVisualStateRequested):
    """Ask the screen to stage a replacement for the selected state."""


class PersonaVisualClearRequested(_PersonaVisualStateRequested):
    """Ask the screen to clear the selected state in its isolated draft."""


class PersonaVisualAddCustomRequested(Message):
    """Ask the screen to collect and stage one safe custom state."""


class PersonaVisualImportRequested(Message):
    """Ask the screen to import one Persona Visual archive for review."""


class PersonaVisualSaveRequested(Message):
    """Ask the screen to publish its current isolated draft exactly once."""


class PersonaVisualCancelRequested(Message):
    """Ask the screen to cancel work and discard only its isolated draft."""


PersonaVisualAvailability = Literal[
    "available", "loading", "server", "unavailable", "unsaved"
]
PersonaVisualBusyState = Literal["importing", "preparing", "previewing", "saving"]


class PersonasPersonaVisualPackWidget(Vertical):
    """Browse one path-free draft inventory and emit typed authoring actions."""

    BUNDLED_CSS = """
    PersonasPersonaVisualPackWidget {
        width: 100%;
        height: 27;
        min-height: 20;
        margin-top: 1;
        padding: 1;
        border: round $surface-lighten-1;
        background: $panel;
    }

    PersonasPersonaVisualPackWidget .persona-visual-copy {
        width: 100%;
        height: auto;
    }

    PersonasPersonaVisualPackWidget #personas-persona-visual-notice,
    PersonasPersonaVisualPackWidget #personas-persona-visual-status {
        color: $text-muted;
    }

    PersonasPersonaVisualPackWidget #personas-persona-visual-body {
        width: 100%;
        height: 1fr;
        min-height: 8;
        margin-top: 1;
    }

    PersonasPersonaVisualPackWidget #personas-persona-visual-results {
        width: 2fr;
        height: 100%;
        min-width: 20;
    }

    PersonasPersonaVisualPackWidget #personas-persona-visual-preview-host {
        width: 1fr;
        height: 100%;
        min-width: 18;
        margin-left: 1;
        padding: 1;
        background: $surface;
        content-align: center middle;
    }

    PersonasPersonaVisualPackWidget #personas-persona-visual-actions {
        width: 100%;
        height: 6;
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 0 1;
        margin-top: 1;
    }

    PersonasPersonaVisualPackWidget #personas-persona-visual-actions Button {
        width: 1fr;
        height: 3;
        min-width: 0;
        border: none;
    }

    PersonasPersonaVisualPackWidget #personas-persona-visual-actions Button:focus {
        outline: heavy $accent;
    }

    PersonasPersonaVisualPackWidget.-narrow {
        height: 31;
    }

    PersonasPersonaVisualPackWidget.-narrow #personas-persona-visual-body {
        height: 12;
    }

    PersonasPersonaVisualPackWidget.-narrow #personas-persona-visual-preview-host {
        min-width: 14;
        padding: 0 1;
    }"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "personas-persona-visual-pack")
        super().__init__(**kwargs)
        self._availability: PersonaVisualAvailability = "unsaved"
        self._busy: PersonaVisualBusyState | None = None
        self._inventory: PersonaVisualDraftInventory | None = None
        self._rows: tuple[PersonaVisualDraftRow, ...] = ()
        self._selected: PersonaVisualDraftRow | None = None
        self._dirty = False
        self._preview_content: Widget | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Persona Visual",
            id="personas-persona-visual-title",
            classes="persona-visual-copy",
        )
        yield Static(
            "Save Persona first to author a visual pack.",
            id="personas-persona-visual-notice",
            classes="persona-visual-copy",
            markup=False,
        )
        yield Static(
            "No draft loaded",
            id="personas-persona-visual-status",
            classes="persona-visual-copy",
            markup=False,
        )
        with Horizontal(id="personas-persona-visual-body"):
            yield OptionList(id="personas-persona-visual-results")
            with Container(id="personas-persona-visual-preview-host"):
                yield Static(
                    "Select a state to preview.",
                    id="personas-persona-visual-preview",
                    markup=False,
                )
        with Horizontal(id="personas-persona-visual-actions"):
            yield Button(
                "Replace…",
                id="personas-persona-visual-replace",
                classes="console-action-secondary",
            )
            yield Button(
                "Clear",
                id="personas-persona-visual-clear",
                classes="console-action-subdued",
            )
            yield Button(
                "Add Custom State",
                id="personas-persona-visual-add-custom",
                classes="console-action-secondary",
            )
            yield Button(
                "Import Pack…",
                id="personas-persona-visual-import",
                classes="console-action-secondary",
            )
            yield Button(
                "Save Pack",
                id="personas-persona-visual-save",
                classes="console-action-primary",
            )
            yield Button(
                "Cancel Draft",
                id="personas-persona-visual-cancel",
                classes="console-action-subdued",
            )

    def on_mount(self) -> None:
        self._sync_narrow()
        self._sync_copy_and_controls()

    def on_resize(self, event: events.Resize) -> None:
        self._sync_narrow(event.size.width)

    def _sync_narrow(self, width: int | None = None) -> None:
        width = self.size.width if width is None else width
        self.set_class(width < 96, "-narrow")

    @property
    def selected_state(self) -> str | None:
        """Return the selected path-free state key."""

        return self._selected.state if self._selected is not None else None

    @property
    def availability(self) -> PersonaVisualAvailability:
        """Return the current local/server authoring availability."""

        return self._availability

    def set_availability(self, availability: PersonaVisualAvailability) -> None:
        """Show one honest authoring eligibility state."""

        if availability not in {
            "available",
            "loading",
            "server",
            "unavailable",
            "unsaved",
        }:
            raise ValueError("persona_visual_availability_invalid")
        self._availability = availability
        if availability != "available":
            self._inventory = None
            self._rows = ()
            self._selected = None
            self._dirty = False
            self._busy = None
            if self.is_mounted:
                self.query_one(
                    "#personas-persona-visual-results", OptionList
                ).clear_options()
                self._replace_preview("Select a state to preview.")
        if self.is_mounted:
            self._sync_copy_and_controls()

    def show_inventory(
        self,
        inventory: PersonaVisualDraftInventory,
        *,
        dirty: bool,
    ) -> None:
        """Render one path-free draft inventory without loading image bytes."""

        if (
            type(inventory) is not PersonaVisualDraftInventory
            or type(dirty) is not bool
        ):
            raise ValueError("persona_visual_inventory_invalid")
        self._availability = "available"
        self._inventory = inventory
        self._rows = inventory.rows
        self._dirty = dirty
        self._busy = None
        options = self.query_one("#personas-persona-visual-results", OptionList)
        options.clear_options()
        options.add_options(
            Option(
                Text(f"{row.label}  {'configured' if row.configured else 'empty'}"),
                id=f"state-{index}",
            )
            for index, row in enumerate(self._rows)
        )
        if self._rows:
            options.highlighted = 0
            self._select_index(0)
        else:
            self._selected = None
            self._replace_preview("No states available.")
        self._sync_copy_and_controls()

    def set_busy(self, busy: PersonaVisualBusyState | None) -> None:
        """Expose distinct cancellable preparation and atomic Save states."""

        if busy not in {None, "importing", "preparing", "previewing", "saving"}:
            raise ValueError("persona_visual_busy_state_invalid")
        self._busy = busy
        self._sync_copy_and_controls()

    def set_preview(self, renderable: object, *, state: str) -> Widget | None:
        """Mount a decoded selected preview and return its weak-targetable widget."""

        if self.selected_state != state:
            return None
        return self._replace_preview(renderable)

    def set_preview_unavailable(self, *, state: str) -> Widget | None:
        """Show a fixed failure only while the failed state remains selected."""

        if self.selected_state != state:
            return None
        return self._replace_preview("Preview unavailable.")

    def _replace_preview(self, renderable: object) -> Widget:
        host = self.query_one("#personas-persona-visual-preview-host", Container)
        status = self.query_one("#personas-persona-visual-preview", Static)
        if self._preview_content is not None:
            self._preview_content.remove()
            self._preview_content = None
        if not isinstance(renderable, Widget):
            status.display = True
            status.update(renderable)
            return status
        status.display = False
        self._preview_content = renderable
        host.mount(renderable)
        return renderable

    def _select_index(self, index: int) -> None:
        if not 0 <= index < len(self._rows):
            return
        row = self._rows[index]
        changed = row != self._selected
        self._selected = row
        if changed:
            self._replace_preview("Loading selected preview…")
            self.post_message(PersonaVisualPreviewRequested(row.state))
        self._sync_copy_and_controls()

    def _sync_copy_and_controls(self) -> None:
        if not self.is_mounted:
            return
        notice = {
            "available": "Local Persona — active visuals stay unchanged until Save Pack.",
            "loading": "Loading Persona Visual draft…",
            "server": "Save Local Copy first to author a Persona Visual pack.",
            "unavailable": "Persona Visual is unavailable for this Persona.",
            "unsaved": "Save Persona first to author a visual pack.",
        }[self._availability]
        self.query_one("#personas-persona-visual-notice", Static).update(notice)
        status = self._status_copy()
        self.query_one("#personas-persona-visual-status", Static).update(status)
        available = self._availability == "available"
        selected = self._selected is not None
        idle = self._busy is None
        self.query_one("#personas-persona-visual-replace", Button).disabled = not (
            available and selected and idle
        )
        self.query_one("#personas-persona-visual-clear", Button).disabled = not (
            available and selected and idle
        )
        for action in ("add-custom", "import"):
            self.query_one(
                f"#personas-persona-visual-{action}", Button
            ).disabled = not (available and idle)
        activatable = self._inventory is not None and self._inventory.activatable
        self.query_one("#personas-persona-visual-save", Button).disabled = not (
            available and idle and self._dirty and activatable
        )
        cancel_allowed = available and (
            self._dirty or self._busy in {"importing", "preparing", "previewing"}
        )
        self.query_one("#personas-persona-visual-cancel", Button).disabled = not (
            cancel_allowed and self._busy != "saving"
        )

    def _status_copy(self) -> str:
        if self._busy is not None:
            return {
                "importing": "Importing pack for review… Cancel Draft is available.",
                "preparing": "Preparing draft… Cancel Draft is available.",
                "previewing": "Loading selected preview… Cancel Draft is available.",
                "saving": "Saving pack… publication cannot be cancelled.",
            }[self._busy]
        if self._availability != "available" or self._inventory is None:
            return "No draft loaded"
        if self._dirty:
            return "Draft — unsaved changes"
        readiness = "complete" if self._inventory.activatable else "incomplete"
        suffix = "asset" if self._inventory.asset_count == 1 else "assets"
        return f"Ready — {self._inventory.asset_count} {suffix} • {readiness}"

    @on(OptionList.OptionHighlighted, "#personas-persona-visual-results")
    def _option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index is not None:
            self._select_index(event.option_index)

    def _post_selected(self, message_type: type[_PersonaVisualStateRequested]) -> None:
        if self._selected is not None:
            self.post_message(message_type(self._selected.state))

    @on(Button.Pressed, "#personas-persona-visual-replace")
    def _replace_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_selected(PersonaVisualReplaceRequested)

    @on(Button.Pressed, "#personas-persona-visual-clear")
    def _clear_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_selected(PersonaVisualClearRequested)

    @on(Button.Pressed, "#personas-persona-visual-add-custom")
    def _add_custom_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaVisualAddCustomRequested())

    @on(Button.Pressed, "#personas-persona-visual-import")
    def _import_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaVisualImportRequested())

    @on(Button.Pressed, "#personas-persona-visual-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaVisualSaveRequested())

    @on(Button.Pressed, "#personas-persona-visual-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaVisualCancelRequested())


__all__ = [
    "PersonaVisualAddCustomRequested",
    "PersonaVisualCancelRequested",
    "PersonaVisualClearRequested",
    "PersonaVisualImportRequested",
    "PersonaVisualPreviewRequested",
    "PersonaVisualReplaceRequested",
    "PersonaVisualSaveRequested",
    "PersonasPersonaVisualPackWidget",
]
