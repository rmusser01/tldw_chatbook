"""Explicit save, file selection, and recovery confirmation dialogs."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, OptionList, Static
from textual.widgets.option_list import Option


class LabDialog(ModalScreen[dict | None]):
    """Small task-local form; file targets always originate in user input."""

    BUNDLED_CSS = """
    LabDialog { align: center middle; background: $background 70%; }
    LabDialog > VerticalScroll { width: 70; max-width: 96%; height: auto; max-height: 90%; padding: 1 2; background: $surface; border: solid $primary; }
    LabDialog Static { height: auto; }
    LabDialog Input { width: 100%; }
    LabDialog Horizontal { height: auto; }
    LabDialog Button { width: auto; min-width: 12; }
    """
    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", show=False)]

    def action_cancel(self) -> None:
        self.dismiss(None)

    def __init__(
        self,
        title: str,
        explanation: str,
        *,
        fields: dict[str, tuple[str, str]] | None = None,
        checks: dict[str, str] | None = None,
        accept: str = "Continue",
        accept_disabled: bool = False,
        checked: frozenset[str] = frozenset(),
        on_edit: Callable[[str, str], None] | None = None,
    ):
        super().__init__()
        self.title_text, self.explanation = title, explanation
        self.fields, self.checks = fields or {}, checks or {}
        self.accept_text, self.on_edit = accept, on_edit
        self.accept_disabled = accept_disabled
        self.checked = checked

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(self.title_text, markup=False)
            yield Static(self.explanation, markup=False)
            for key, (label, value) in self.fields.items():
                yield Static(label, markup=False)
                yield Input(value=value, id=f"dialog-{key}")
            for key, label in self.checks.items():
                yield Checkbox(label, value=key in self.checked, id=f"dialog-{key}")
            with Horizontal():
                yield Button(
                    self.accept_text,
                    id="dialog-accept",
                    variant="primary",
                    disabled=self.accept_disabled,
                )
                yield Button("Cancel", id="dialog-cancel")

    @on(Input.Changed)
    def changed(self, event: Input.Changed) -> None:
        event.stop()
        if self.on_edit is not None:
            self.on_edit((event.input.id or "").removeprefix("dialog-"), event.value)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "dialog-cancel":
            self.dismiss(None)
        elif event.button.id == "dialog-accept":
            self.dismiss(
                {
                    **{
                        key: self.query_one(f"#dialog-{key}", Input).value
                        for key in self.fields
                    },
                    **{
                        key: self.query_one(f"#dialog-{key}", Checkbox).value
                        for key in self.checks
                    },
                }
            )


class TemplateDialog(LabDialog):
    """Search decorated local saved records without hiding invalid entries."""

    def __init__(self, records: list[dict]):
        super().__init__(
            "Saved local templates",
            "Choose a detached draft. Built-ins can be saved as new; invalid and reserved entries stay visible.",
            fields={"search": ("Search names / tags", "")},
        )
        self.records = records
        self.visible_records = records

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(self.title_text, markup=False)
            yield Static(self.explanation, markup=False)
            yield Input(placeholder="Search names / tags", id="dialog-search")
            yield OptionList(id="dialog-records")
            yield Button("Cancel", id="dialog-cancel")

    def on_mount(self) -> None:
        self._filter("")

    def _filter(self, query: str) -> None:
        from rich.text import Text

        self.visible_records = [
            record
            for record in self.records
            if query.casefold()
            in (
                str(record.get("name", "")) + " " + " ".join(record.get("tags") or [])
            ).casefold()
        ]
        options = []
        for index, record in enumerate(self.visible_records):
            flags = []
            if record.get("is_builtin"):
                flags.append("Built-in · save as new")
            if record.get("name_reserved"):
                flags.append("Reserved name")
            if record.get("template_valid") is False:
                flags.append(
                    "Invalid: "
                    + "; ".join(record.get("template_validation_errors", []))
                )
            options.append(
                Option(
                    Text(
                        str(record.get("name", ""))
                        + (" · " + " · ".join(flags) if flags else "")
                    ),
                    id=str(index),
                )
            )
        self.query_one(OptionList).clear_options().add_options(options)

    @on(Input.Changed, "#dialog-search")
    def search_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._filter(event.value)

    @on(OptionList.OptionSelected)
    def selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self.visible_records[event.option_index])
