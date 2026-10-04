"""A picker of known values with an 'Other…' escape, for the Voice step.

TASK-34100.8 (voice-speech-06). Voice and Format under Advanced were bare
text fields: a user had to know that OpenAI says ``shimmer`` and pocket-tts
says ``alba``. Each is now a Select of the selected service's own values plus
"Other…", which reveals a free-text field. That text field keeps the old id
(``#setup-voice-voice`` / ``#setup-voice-format``) and stays the single source
of the value, so everything that reads or resumes the field is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Select

OTHER_VALUE = "__other__"
OTHER_LABEL = "Other…"


class VoiceOptionPicker(Vertical):
    """A Select of known values over the free-text Input that holds the value."""

    DEFAULT_CSS = """
    VoiceOptionPicker {
        height: auto;
    }
    """

    def __init__(
        self,
        *,
        input_id: str,
        value: str,
        options: Sequence[str],
        label: str,
    ) -> None:
        super().__init__(classes="setup-voice-picker", id=f"{input_id}-picker")
        self._input_id = input_id
        self._value = value
        self._options = tuple(options)
        self._label = label

    def compose(self) -> ComposeResult:
        yield Select(
            self._select_options(),
            value=self._select_value(self._value),
            allow_blank=False,
            id=f"{self._input_id}-select",
        )
        field = Input(
            value=self._value,
            id=self._input_id,
            placeholder=f"Type a {self._label.lower()} name",
        )
        field.display = self._value not in self._options
        yield field

    def _select_options(self) -> list[tuple[str, str]]:
        return [(option, option) for option in self._options] + [
            (OTHER_LABEL, OTHER_VALUE)
        ]

    def _select_value(self, value: str) -> str:
        return value if value in self._options else OTHER_VALUE

    def set_options(self, options: Sequence[str]) -> None:
        """Offer another service's values, keeping the current text."""
        self._options = tuple(options)
        select = self.query_one(Select)
        with select.prevent(Select.Changed):
            select.set_options(self._select_options())
        self.sync()

    def sync(self) -> None:
        """Point the Select at the text field's value (after a programmatic set)."""
        field = self.query_one(f"#{self._input_id}", Input)
        select = self.query_one(Select)
        with select.prevent(Select.Changed):
            select.value = self._select_value(field.value.strip())
        field.display = field.value.strip() not in self._options

    @on(Select.Changed)
    def _on_pick(self, event: Select.Changed) -> None:
        event.stop()
        field = self.query_one(f"#{self._input_id}", Input)
        if event.value == OTHER_VALUE:
            field.display = True
            field.focus()
            return
        field.display = False
        if isinstance(event.value, str) and field.value != event.value:
            field.value = event.value


__all__ = ["OTHER_LABEL", "OTHER_VALUE", "VoiceOptionPicker"]
