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

    def show_document(self, document: dict, selected: str, issues=(), *, rebuild=True):
        listing = self.query_one(OptionList)
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
        prompts = [
            Option(Text(("> " if key == selected else "  ") + label), id=key)
            for label, key in options
        ]
        if not rebuild:
            for option in prompts:
                if listing.get_option(option.id).prompt != option.prompt:
                    listing.replace_option_prompt(option.id, option.prompt)
            return
        listing.clear_options()
        listing.add_options(prompts)
        listing.highlighted = next(
            (i for i, (_, key) in enumerate(options) if key == selected), 0
        )

    @on(OptionList.OptionSelected)
    def select(self, event):
        event.stop()
        self.post_message(self.Selected(event.option.id))
