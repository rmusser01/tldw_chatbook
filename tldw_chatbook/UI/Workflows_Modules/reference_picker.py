"""Canonical typed references and explicitly historical fixed-value review."""

from textual import on
from textual.app import ComposeResult
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
    """List compatible references, labeling unknown output types as unverified.

    Args:
        document: Projected workflow with optional
            ``metadata.tldw_workflow.input_schema.properties`` and ordered
            ``steps`` containing string ``id`` and ``type`` values.
        step_id: Consumer step ID. Output choices stop before this step; if it
            is absent, all steps are considered. Ignored for workflow inputs.
        value_type: Required declared type, such as ``string`` or ``array``.
            References declared ``unverified`` are also offered with a warning.
        source: ``inputs`` selects workflow inputs; any other value selects
            earlier step outputs (the picker uses ``steps``).

    Returns:
        Ordered ``(label, expression)`` pairs using canonical ``{{ path }}``
        expressions. Incompatible types and invalid identifier paths are
        omitted. Missing input properties or unknown output contracts yield
        no choices for those sources; values are not resolved or validated.

    Raises:
        KeyError: For step outputs, the document lacks ``steps`` or a visited
            step lacks ``id`` or a required ``type``.
    """
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
    """Choose a canonical reference expression without resolving its value.

    Selecting a reference dismisses with its expression. Fixed value and
    cancellation dismiss with None so the caller can retain manual entry.

    Attributes:
        document: Projected workflow supplying inputs and ordered steps.
        step_id: Consumer whose preceding steps may supply outputs.
        value_type: Required declared type; unverified references remain labeled.
        source: Active source, either ``inputs`` or ``steps``.
    """

    def __init__(
        self, document: dict, step_id: str, value_type: str = "string"
    ) -> None:
        """Initialize the picker with workflow inputs as the first source.

        Args:
            document: Projected workflow accepted by ``reference_choices``.
            step_id: Consumer step ID used to restrict earlier-step outputs.
            value_type: Declared type required by the destination field.
        """
        super().__init__("Choose a value source", ())
        self.document, self.step_id, self.value_type = document, step_id, value_type
        self.source = "inputs"

    def compose(self) -> ComposeResult:
        """Build source controls, reference choices and cancellation.

        Yields:
            Widgets for choosing a value source and a canonical expression.
        """
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

    def on_mount(self) -> None:
        """Populate the initial input references and focus their list.

        Returns:
            None.
        """
        self.fill()
        self.query_one(OptionList).focus()

    def fill(self) -> None:
        """Replace choices for the current source and highlight the first one.

        Reads ``document``, ``step_id``, ``value_type`` and ``source`` through
        ``reference_choices``. An empty result clears the highlight.

        Returns:
            None.

        Raises:
            NoMatches: The picker's OptionList has not been composed.
            KeyError: The document lacks keys required for step references.
            DuplicateID: The document produces duplicate reference expressions.
        """
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
    def source_selected(self, event: Button.Pressed) -> None:
        """Switch reference sources or dismiss for manual fixed-value entry.

        Args:
            event: Source-button press. Unrelated buttons remain unhandled.

        Returns:
            None.
        """
        identifier = event.button.id
        if identifier == "workflow-reference-fixed":
            event.stop()
            self.dismiss(None)
        elif identifier in ("workflow-reference-input", "workflow-reference-step"):
            event.stop()
            self.source = "inputs" if identifier.endswith("input") else "steps"
            self.fill()
            self.query_one(OptionList).focus()
