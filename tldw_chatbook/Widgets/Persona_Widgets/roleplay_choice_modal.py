"""Roleplay's small choice modal: the New chooser and "Import as:" (frame slice B2b).

Spec 1.4.6. One literal ``OptionList`` of choices: Enter (or a click) returns
the picked id, Escape returns None (spec 4.5, rung 0). Every string it paints
is plain text (``Content``, ``markup=False``), because a title can quote a
file name and a row can quote a reason (R33).

The screen imports this module lazily, inside the worker that pushes the
modal, so the Roleplay route pre-imports nothing new (spec 5.10). Its rules
live in the lazy Roleplay sheet (``css/features/_roleplay.tcss``), which is
always loaded here: the modal is only pushed from Roleplay.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


def chooser_keys() -> str:
    """The modal's one keyboard line (the separator follows the glyph mode)."""
    return f"enter picks {resolve_glyph('·')} esc cancels"


@dataclass(frozen=True)
class ChoiceOption:
    """One choice.

    Attributes:
        id: What the modal returns when this row is picked.
        label: The row's words.
        hint: A dim note after the label, or None.
        enabled: False paints the row disabled; it cannot be picked.
        reason: Why it is disabled (painted instead of the hint).
    """

    id: str
    label: str
    hint: str | None = None
    enabled: bool = True
    reason: str | None = None


def choice_prompt(option: ChoiceOption) -> Content:
    """The row as painted: the label, then the hint (or the reason) dimmed."""
    note = option.hint if option.enabled else option.reason
    if not note:
        return Content(option.label)
    return Content.assemble(option.label, (f"  {note}", "dim"))


class RoleplayChoiceModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Pick one of a few choices; dismisses with its id, or None on Escape.

    Args:
        title: The heading, painted literally.
        options: The rows, in order.
        preselect: The id of the row highlighted on open (an enabled one).
    """

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#roleplay-items-chooser-box"

    def __init__(
        self, title: str, options: Sequence[ChoiceOption], *, preselect: str
    ) -> None:
        super().__init__(classes="roleplay-items-chooser")
        self._title = title
        self._options = tuple(options)
        self._preselect = preselect

    def compose(self) -> ComposeResult:
        with Vertical(id="roleplay-items-chooser-box"):
            yield Static(self._title, id="roleplay-items-chooser-title", markup=False)
            yield OptionList(
                *(
                    Option(
                        choice_prompt(option), id=option.id, disabled=not option.enabled
                    )
                    for option in self._options
                ),
                id="roleplay-items-chooser-options",
                markup=False,
            )
            yield Static(
                chooser_keys(), classes="roleplay-items-chooser-keys", markup=False
            )

    def on_mount(self) -> None:
        """Open on the preselected row, with the list focused."""
        options = self.query_one("#roleplay-items-chooser-options", OptionList)
        for index, option in enumerate(self._options):
            if option.id == self._preselect and option.enabled:
                options.highlighted = index
                break
        options.focus()

    @on(OptionList.OptionSelected, "#roleplay-items-chooser-options")
    def _picked(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss_safe_once(event.option.id)


__all__ = ["ChoiceOption", "RoleplayChoiceModal", "choice_prompt", "chooser_keys"]
