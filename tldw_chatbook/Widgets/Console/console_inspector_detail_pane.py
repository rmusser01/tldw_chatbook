"""Modal-local section/detail navigation; no persistence or disclosure policy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from .console_inspector_presentation import InspectorSection


class ConsoleInspectorDetailPane(Vertical):
    """One list and one reader, retained while switching narrow/wide layouts.

    Args:
        sections: Initial metadata-only section descriptors.
        **kwargs: Standard Textual container arguments.
    """

    WIDE_LAYOUT_MIN_WIDTH = 90

    class SectionSelected(Message):
        """Notify the owning modal that a section was selected.

        Args:
            pane: Detail pane originating the selection.
            key: Selected section's stable identifier.
        """

        def __init__(self, pane: ConsoleInspectorDetailPane, key: str) -> None:
            """Initialize the selection message.

            Args:
                pane: Detail pane originating the selection.
                key: Selected section's stable identifier.
            """
            super().__init__()
            self.pane = pane
            self.key = key

    def __init__(
        self, sections: Iterable[InspectorSection] = (), **kwargs: Any
    ) -> None:
        """Initialize the retained list/detail state.

        Args:
            sections: Initial metadata-only descriptors.
            **kwargs: Standard Textual container arguments.
        """
        super().__init__(**kwargs)
        self.sections = tuple(sections)
        self.selected_key = None
        self._detail_open = False

    @property
    def detail_open(self) -> bool:
        """Return whether narrow-mode navigation has opened the reader.

        Returns:
            True when detail is open, even if a wide layout shows both panes.
        """
        return self._detail_open

    def compose(self) -> ComposeResult:
        """Build the section selector and read-only detail reader.

        Returns:
            Child widgets for the retained list/detail layout.
        """
        yield Button(
            "‹ Back to sections", classes="inspector-detail-back", compact=True
        )
        with Horizontal(classes="inspector-detail-body"):
            yield OptionList(*self._options(), classes="inspector-section-list")
            with Vertical(classes="inspector-detail-reader"):
                yield Static(
                    "Select a section", classes="inspector-detail-title", markup=False
                )
                yield TextArea(
                    "", read_only=True, soft_wrap=True, classes="inspector-detail-text"
                )

    def _options(self) -> Iterable[Option]:
        for section in self.sections:
            count = f" ({section.item_count})" if section.item_count else ""
            group = {
                "Current conversation": "Current",
                "Next send preview": "Preview",
            }.get(section.group, section.group)
            yield Option(Text(f"{group} · {section.label}{count}"), id=section.key)

    def set_sections(self, sections: Iterable[InspectorSection]) -> None:
        """Replace descriptors while retaining an existing valid selection.

        Args:
            sections: Metadata-only descriptors in display order.
        """
        self.sections = tuple(sections)
        if not self.is_mounted:
            return
        choices = self.query_one(OptionList)
        selected = self.selected_key
        with self.prevent(OptionList.OptionHighlighted):
            choices.clear_options()
            choices.add_options(list(self._options()))
            if selected in [row.key for row in self.sections]:
                choices.highlighted = next(
                    i for i, row in enumerate(self.sections) if row.key == selected
                )

    def set_detail(self, key: str, title: str, text: str) -> None:
        """Update the mounted reader without resetting unchanged text.

        Args:
            key: Section identifier whose content was loaded.
            title: Literal reader heading.
            text: Already authorized, bounded detail content.
        """
        self.selected_key = key
        self.query_one(".inspector-detail-title", Static).update(title)
        reader = self.query_one(TextArea)
        if reader.text != text:
            reader.load_text(text)

    def clear_detail(self, reason: str = "Select a section") -> None:
        """Clear the mounted reader and its selection.

        Args:
            reason: Literal heading explaining why the reader is empty.
        """
        self.selected_key = None
        self.query_one(".inspector-detail-title", Static).update(reason)
        self.query_one(TextArea).load_text("")

    def select(self, key: str, *, open_detail: bool = True) -> None:
        """Select an existing section and request its content from the owner.

        Args:
            key: Section identifier; unknown identifiers are ignored.
            open_detail: Whether narrow mode should switch to the reader.
        """
        choices = self.query_one(OptionList)
        index = next((i for i, row in enumerate(self.sections) if row.key == key), None)
        if index is None:
            return
        self.selected_key = key
        with self.prevent(OptionList.OptionHighlighted):
            choices.highlighted = index
        self._detail_open = open_detail
        self._apply_layout()
        if open_detail and self.size.width < self.WIDE_LAYOUT_MIN_WIDTH:
            self.query_one(TextArea).focus()
        self.post_message(self.SectionSelected(self, key))

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.select(event.option.id)

    @on(OptionList.OptionHighlighted)
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        # In narrow mode arrows browse, Enter opens detail without losing the list.
        if self.size.width >= self.WIDE_LAYOUT_MIN_WIDTH:
            self.select(event.option.id, open_detail=False)

    @on(Button.Pressed, ".inspector-detail-back")
    def _back(self, event: Button.Pressed) -> None:
        event.stop()
        self._detail_open = False
        self._apply_layout()
        self.query_one(OptionList).focus()

    def on_mount(self) -> None:
        """Apply the initial layout after Textual mounts the children."""
        self._apply_layout()

    def on_resize(self) -> None:
        """Reapply the layout while preserving selection and reader contents."""
        self._apply_layout()

    def _apply_layout(self) -> None:
        if not self.is_mounted:
            return
        narrow = self.size.width < self.WIDE_LAYOUT_MIN_WIDTH
        self.set_class(narrow, "inspector-detail-narrow")
        self.query_one(".inspector-section-list").display = not (
            narrow and self._detail_open
        )
        self.query_one(".inspector-detail-reader").display = (
            not narrow or self._detail_open
        )
        self.query_one(".inspector-detail-back").display = narrow and self._detail_open
