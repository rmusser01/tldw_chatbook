"""Library region and safe compact selection/confirmation overlays."""

from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Workflows.catalog import discover


class WorkflowButton(Button):
    """Button-local Space activation, never a printable screen shortcut."""

    BINDINGS: ClassVar = [("space", "press", "Activate")]


def compact_button(label: str, identifier: str, **kwargs) -> Button:
    """One-row action with focus background; no outline painted over its label."""
    button = WorkflowButton(Text(label), id=identifier, tooltip=label, **kwargs)
    button.add_class("workflow-compact")
    return button


class ChoiceModal(ModalScreen[str | None]):
    """Escape/Cancel never chooses or mutates; callers restore their own opener."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        title: str,
        choices: tuple[tuple[str, str], ...],
        *,
        detail: str = "",
        text_input: bool = False,
        scroll_detail: bool = False,
    ):
        super().__init__()
        self.title_text, self.choices, self.detail = title, choices, detail
        self.text_input = text_input
        self.scroll_detail = scroll_detail
        self.set_class(scroll_detail, "has-scroll-detail")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.title_text)
            if self.scroll_detail:
                with VerticalScroll(id="workflow-dialog-detail"):
                    yield Static(self.detail, markup=False)
            else:
                yield Static(self.detail, markup=False, classes="choice-copy")
            if self.text_input:
                yield Input(placeholder="Name", id="workflow-dialog-name")
            yield OptionList(
                *(Option(Text(label), id=key) for label, key in self.choices),
                id="workflow-dialog-choices",
            )
            yield compact_button("Cancel · Esc", "workflow-dialog-cancel")

    def on_mount(self):
        self.query_one(Input if self.text_input else OptionList).focus()

    def action_cancel(self):
        self.dismiss(None)

    @on(Button.Pressed, "#workflow-dialog-cancel")
    def cancel(self, event):
        event.stop()
        self.action_cancel()

    @on(OptionList.OptionSelected)
    def choose(self, event):
        event.stop()
        self.select_option(event.option)

    def select_option(self, option):
        self.dismiss(self.query_one(Input).value if self.text_input else option.id)

    @on(Input.Submitted)
    def named(self, event):
        if self.text_input and event.value.strip():
            event.stop()
            self.dismiss(event.value)


class StepChooser(ChoiceModal):
    """Searchable task groups; Show all exposes unavailable inventory honestly."""

    def __init__(self):
        super().__init__("Add step · Local subsets", ())
        self.show_all = False

    def compose(self):
        with Vertical():
            yield Label(self.title_text)
            yield Input(
                placeholder="Search actions or canonical types",
                id="workflow-type-search",
            )
            yield compact_button("Show all", "workflow-show-all")
            yield OptionList(id="workflow-dialog-choices")
            yield Static(
                "Select an action to review its requirements.",
                id="workflow-type-detail",
                markup=False,
                classes="choice-copy",
            )
            with Horizontal(classes="workflow-dialog-actions"):
                yield compact_button(
                    "Insert step", "workflow-insert-confirm", disabled=True
                )
                yield compact_button("Cancel · Esc", "workflow-dialog-cancel")

    def on_mount(self):
        self.fill()
        self.add_class("workflow-step-chooser")
        self.query_one(Input).focus()

    def fill(self, query=""):
        entries = discover(show_all=self.show_all, query=query)
        listing = self.query_one(OptionList)
        listing.clear_options()
        listing.add_options(
            Option(
                f"{item.family} · {item.label} ({item.step_type})\n{item.example} · "
                + (
                    "Local subset; setup checked before run"
                    if item.available
                    else "Unavailable · " + item.disposition
                ),
                id=item.step_type,
                disabled=not item.available,
            )
            for item in entries
        )
        listing.highlighted = 0 if entries else None
        self.query_one("#workflow-insert-confirm", Button).disabled = True

    @on(Input.Changed, "#workflow-type-search")
    def search(self, event):
        self.fill(event.value)

    @on(Button.Pressed, "#workflow-show-all")
    def all_types(self, event):
        event.stop()
        self.show_all = not self.show_all
        event.button.label = "Supported subsets" if self.show_all else "Show all"
        self.fill(self.query_one(Input).value)

    def select_option(self, option):
        self.query_one("#workflow-type-detail", Static).update(
            str(option.prompt)
            + "\nInserts a draft step. Bind file/model/actor/Notes resources before execution."
        )
        self.query_one("#workflow-insert-confirm", Button).disabled = False
        self.query_one("#workflow-insert-confirm", Button).focus()

    @on(Button.Pressed, "#workflow-insert-confirm")
    def insert(self, event):
        event.stop()
        listing = self.query_one(OptionList)
        if listing.highlighted is not None:
            self.dismiss(listing.get_option_at_index(listing.highlighted).id)


class WorkflowLibrary(Vertical):
    """Exact document identities, filtered in memory after the owner loads them."""

    class Selected(Message):
        def __init__(self, workflow_id: str, revision_id: str):
            super().__init__()
            self.workflow_id, self.revision_id = workflow_id, revision_id

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows: tuple[tuple[str, str, str], ...] = ()

    def compose(self):
        yield Label("Workflow library", id="workflow-library-heading")
        yield Input(placeholder="Search workflows", id="workflow-library-search")
        yield OptionList(id="workflow-library-list")
        yield compact_button("New workflow", "workflow-new")

    def show_rows(self, rows):
        self.rows = tuple(rows)
        self.filter(self.query_one(Input).value)

    def filter(self, query):
        listing = self.query_one(OptionList)
        listing.clear_options()
        listing.add_options(
            Option(Text(label + "\nLocal · saved revision"), id=wid)
            for label, wid, rid in self.rows
            if query.casefold() in label.casefold()
        )
        listing.highlighted = 0 if listing.option_count else None

    @on(Input.Changed)
    def search(self, event):
        event.stop()
        self.filter(event.value)

    @on(OptionList.OptionSelected)
    def select(self, event):
        event.stop()
        row = next(row for row in self.rows if row[1] == event.option.id)
        self.post_message(self.Selected(row[1], row[2]))
