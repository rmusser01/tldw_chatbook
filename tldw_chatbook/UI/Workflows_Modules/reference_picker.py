"""Canonical typed references and explicitly historical fixed-value review."""

from textual import on
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Workflows.catalog import output_types
from tldw_chatbook.Workflows.expressions import IDENTIFIER

from .controller import step_label
from .library import ChoiceModal, compact_button


def reference_choices(
    document: dict, step_id: str, value_type: str, source: str
) -> tuple[tuple[str, str], ...]:
    """Offer declared compatible types; unverified schemas stay explicitly labeled."""
    choices = []

    def add(label, path, declared):
        if declared not in (value_type, "unverified"):
            return
        if not all(IDENTIFIER.fullmatch(part) for part in path.split(".")):
            return
        suffix = (
            "unverified — runtime validation required"
            if declared == "unverified"
            else declared
        )
        choices.append((f"{label} · {path} · {suffix}", "{{ " + path + " }}"))

    if source == "inputs":
        schema = (
            document.get("metadata", {})
            .get("tldw_workflow", {})
            .get("input_schema", {})
        )
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if isinstance(properties, dict):
            for name, spec in properties.items():
                declared = (
                    spec.get("type", "unverified")
                    if isinstance(spec, dict)
                    else "unverified"
                )
                add("Workflow input", "inputs." + name, declared)
    else:
        for step in document["steps"]:
            if step["id"] == step_id:
                break
            for path, declared in output_types(step["type"]):
                add(step_label(step), step["id"] + "." + path, declared)
    return tuple(choices)


class ReferencePicker(ChoiceModal):
    """Return a canonical expression or None; Fixed value returns to manual entry."""

    def __init__(self, document: dict, step_id: str, value_type: str = "string"):
        super().__init__("Choose a value source", ())
        self.document, self.step_id, self.value_type = document, step_id, value_type
        self.source = "inputs"

    def compose(self):
        with Vertical():
            yield Label("Choose a value source")
            with Horizontal(classes="workflow-dialog-actions"):
                yield compact_button("Fixed value", "workflow-reference-fixed")
                yield compact_button("Workflow input", "workflow-reference-input")
                yield compact_button("Earlier step output", "workflow-reference-step")
            yield Static(
                "Only declared compatible types are verified. No historical result is live data.",
                markup=False,
                classes="choice-copy",
            )
            yield OptionList(id="workflow-dialog-choices")
            yield compact_button("Cancel · Esc", "workflow-dialog-cancel")

    def on_mount(self):
        self.fill()
        self.query_one(OptionList).focus()

    def fill(self):
        listing = self.query_one(OptionList)
        listing.clear_options()
        listing.add_options(
            Option(label, id=expression)
            for label, expression in reference_choices(
                self.document, self.step_id, self.value_type, self.source
            )
        )
        listing.highlighted = 0 if listing.option_count else None

    @on(Button.Pressed)
    def source_selected(self, event):
        identifier = event.button.id
        if identifier == "workflow-reference-fixed":
            event.stop()
            self.dismiss(None)
        elif identifier in ("workflow-reference-input", "workflow-reference-step"):
            event.stop()
            self.source = "inputs" if identifier.endswith("input") else "steps"
            self.fill()
            self.query_one(OptionList).focus()
