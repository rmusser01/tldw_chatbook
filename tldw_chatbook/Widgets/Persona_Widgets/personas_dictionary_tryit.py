"""Try-it substitution preview for the Roleplay Dictionaries mode."""

from __future__ import annotations

import difflib
import re

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Static, TextArea

_TOKEN_RE = re.compile(r"(\s+)")


def word_diff(original: str, processed: str) -> tuple[Text, Text]:
    """Word-level diff: removals styled 'strike dim' in the original, additions
    'bold underline' in the processed text. Theme-safe (no colors).

    Args:
        original: The sample text before dictionary substitution.
        processed: The sample text after dictionary substitution.

    Returns:
        A ``(original, processed)`` pair of Rich ``Text`` objects with the
        changed spans styled for display; unchanged spans carry no style.
    """
    left = Text()
    right = Text()
    a = _TOKEN_RE.split(original)
    b = _TOKEN_RE.split(processed)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        a_chunk = "".join(a[a0:a1])
        b_chunk = "".join(b[b0:b1])
        if op == "equal":
            left.append(a_chunk)
            right.append(b_chunk)
        elif op == "delete":
            left.append(a_chunk, style="strike dim")
        elif op == "insert":
            right.append(b_chunk, style="bold underline")
        else:  # replace
            left.append(a_chunk, style="strike dim")
            right.append(b_chunk, style="bold underline")
    return left, right


class DictionaryTryItRunRequested(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class PersonasDictionaryTryItWidget(Vertical):
    """Sample text in, highlighted before/after diff out. Owns no I/O."""

    BINDINGS = [
        Binding("ctrl+enter", "run_preview", "Run preview", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    PersonasDictionaryTryItWidget {
        height: auto;
        max-height: 60%;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-sample {
        height: 4;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-original,
    PersonasDictionaryTryItWidget #personas-dict-tryit-processed {
        height: auto;
        max-height: 8;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-status {
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Try it — substitution preview", markup=False)
        yield TextArea(id="personas-dict-tryit-sample")
        yield Button(
            "Run preview",
            id="personas-dict-tryit-run",
            classes="console-action-secondary",
            disabled=True,
            tooltip="Run the selected dictionary against the sample text (Ctrl+Enter).",
        )
        yield Static("", id="personas-dict-tryit-status", markup=False)
        yield Static("", id="personas-dict-tryit-original")
        yield Static("", id="personas-dict-tryit-processed")

    def set_ready(self, ready: bool, hint: str = "") -> None:
        self.query_one("#personas-dict-tryit-run", Button).disabled = not ready
        if hint:
            self.query_one("#personas-dict-tryit-status", Static).update(hint)

    def sample_text(self) -> str:
        return self.query_one("#personas-dict-tryit-sample", TextArea).text

    def render_result(self, original: str, processed: str) -> None:
        left, right = word_diff(original, processed)
        status = self.query_one("#personas-dict-tryit-status", Static)
        if original == processed:
            status.update("No differences - no entry changed the sample.")
        else:
            status.update("Changed spans highlighted below.")
        self.query_one("#personas-dict-tryit-original", Static).update(left)
        self.query_one("#personas-dict-tryit-processed", Static).update(right)

    def show_error(self, message: str) -> None:
        self.query_one("#personas-dict-tryit-status", Static).update(message)

    def _post_run(self) -> None:
        text = self.sample_text()
        if not text.strip():
            self.show_error("Type some sample text first.")
            return
        self.post_message(DictionaryTryItRunRequested(text))

    def action_run_preview(self) -> None:
        if not self.query_one("#personas-dict-tryit-run", Button).disabled:
            self._post_run()

    @on(Button.Pressed, "#personas-dict-tryit-run")
    def _run_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_run()


__all__ = [
    "DictionaryTryItRunRequested",
    "PersonasDictionaryTryItWidget",
    "word_diff",
]
