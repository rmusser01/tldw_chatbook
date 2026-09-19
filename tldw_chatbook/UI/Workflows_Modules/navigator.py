"""Ordered selection, distinct from execution highlighting."""

from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from .controller import step_label
from .library import compact_button

if TYPE_CHECKING:
    from tldw_chatbook.Workflows.models import Issue


class WorkflowNavigator(Vertical):
    """Show document sections and ordered steps, posting selection requests.

    The document owner supplies selection and validation issues. This widget
    only renders that state; selecting a row posts ``Selected`` to its owner.
    """

    class Selected(Message):
        """Request navigation to a document section or a stable step ID.

        Attributes:
            section: Section key or ``step:<step_id>`` from the selected row.
        """

        def __init__(self, section: str) -> None:
            """Initialize a navigation request.

            Args:
                section: Section key or ``step:<step_id>`` to select.
            """
            super().__init__()
            self.section = section

    def compose(self) -> ComposeResult:
        """Build the navigator's heading, list and add-step control.

        Yields:
            Widgets for document navigation and requesting a new step.
        """
        yield Label("Step navigator", id="workflow-navigator-heading")
        yield OptionList(id="workflow-navigation-list")
        yield compact_button("Add step", "workflow-add-step")

    def show_document(
        self,
        document: dict,
        selected: str,
        issues: tuple["Issue", ...] = (),
        *,
        rebuild: bool = True,
    ) -> None:
        """Refresh section and step labels from an already projected document.

        Args:
            document: Workflow mapping with ordered ``steps`` containing unique
                string ``id`` values and a ``type`` used to derive each label.
            selected: Section key or ``step:<step_id>`` to mark as selected.
                A rebuild highlights the first row if this key is absent.
            issues: Validation issues whose step pointers add warning markers.
            rebuild: Replace rows and reset highlighting when true. False
                updates prompts in place, retaining highlight and scroll state;
                the existing rows must have the same IDs and order.

        Returns:
            None.

        Raises:
            NoMatches: The navigator's OptionList has not been composed.
            KeyError: The document lacks ``steps`` or a step lacks ``id`` or
                ``type``.
            DuplicateID: A rebuild encounters duplicate step IDs.
            OptionDoesNotExist: An incremental refresh needs an absent row ID.
        """
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
    def select(self, event: OptionList.OptionSelected) -> None:
        """Stop the row event and post a section-selection request.

        Args:
            event: Selection of a navigator option with a section or step ID.

        Returns:
            None.
        """
        event.stop()
        self.post_message(self.Selected(event.option.id))
