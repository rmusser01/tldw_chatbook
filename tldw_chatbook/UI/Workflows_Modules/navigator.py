"""Ordered selection, distinct from execution highlighting."""

from rich.text import Text
from textual import on
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from .controller import step_label
from .library import compact_button


class WorkflowNavigator(Vertical):
    class Selected(Message):
        def __init__(self, section: str):
            super().__init__()
            self.section = section

    def compose(self):
        yield Label("Step navigator", id="workflow-navigator-heading")
        yield OptionList(id="workflow-navigation-list")
        yield compact_button("Add step", "workflow-add-step")

    def show_document(self, document: dict, selected: str, issues=()):
        listing = self.query_one(OptionList)
        listing.clear_options()
        options = [
            ("Overview", "overview"),
            ("Run inputs", "inputs"),
            ("Requirements", "requirements"),
            ("Versions", "versions"),
        ]
        for index, step in enumerate(document["steps"]):
            mark = (
                " !"
                if any(
                    issue.pointer.startswith(f"/steps/{index}/")
                    or issue.pointer == f"/steps/{index}"
                    for issue in issues
                )
                else ""
            )
            options.append(
                (f"{index + 1:02} {step_label(step)}{mark}", "step:" + step["id"])
            )
        listing.add_options(
            Option(Text(("> " if key == selected else "  ") + label), id=key)
            for label, key in options
        )
        listing.highlighted = next(
            (i for i, (_, key) in enumerate(options) if key == selected), 0
        )

    @on(OptionList.OptionSelected)
    def select(self, event):
        event.stop()
        self.post_message(self.Selected(event.option.id))
