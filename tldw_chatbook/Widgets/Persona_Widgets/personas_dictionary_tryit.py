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

_SKIP_REASON_COPY = {
    "skipped:group_scoring": "skipped: lost group scoring",
    "skipped:probability": "skipped: probability roll — re-running may differ",
    "skipped:timed_effects": "skipped: cooldown or delay",
    "skipped:token_budget": "skipped: token budget",
    "no_replacement": "no replacement — text changed by an earlier entry",
}


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
    PersonasDictionaryTryItWidget #personas-dict-tryit-summary {
        height: 1;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-details {
        height: auto;
        max-height: 12;
        overflow-y: auto;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-fired,
    PersonasDictionaryTryItWidget #personas-dict-tryit-nearmiss {
        height: auto;
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
        yield Static("", id="personas-dict-tryit-summary", markup=False)
        with Vertical(id="personas-dict-tryit-details"):
            yield Static("", id="personas-dict-tryit-fired")
            yield Static("", id="personas-dict-tryit-nearmiss")

    def set_ready(self, ready: bool, hint: str = "") -> None:
        self.query_one("#personas-dict-tryit-run", Button).disabled = not ready
        if hint:
            self.query_one("#personas-dict-tryit-status", Static).update(hint)

    def sample_text(self) -> str:
        return self.query_one("#personas-dict-tryit-sample", TextArea).text

    def render_result(
        self, original: str, processed: str, diagnostics: dict | None = None
    ) -> None:
        """Renders the word-diff and, when available, the diagnostics story.

        Args:
            original: The sample text before substitution.
            processed: The sample text after substitution.
            diagnostics: The service's diagnostics dict, or None to degrade
                to the diff-only view with an "unavailable" note.
        """
        left, right = word_diff(original, processed)
        status = self.query_one("#personas-dict-tryit-status", Static)
        if original == processed:
            status.update("No differences - no entry changed the sample.")
        else:
            status.update("Changed spans highlighted below.")
        self.query_one("#personas-dict-tryit-original", Static).update(left)
        self.query_one("#personas-dict-tryit-processed", Static).update(right)
        self._render_diagnostics(diagnostics)

    def _render_diagnostics(self, diagnostics: dict | None) -> None:
        summary = self.query_one("#personas-dict-tryit-summary", Static)
        fired_area = self.query_one("#personas-dict-tryit-fired", Static)
        nearmiss_area = self.query_one("#personas-dict-tryit-nearmiss", Static)
        if not isinstance(diagnostics, dict):
            summary.update(Text("diagnostics unavailable", style="dim"))
            fired_area.update("")
            nearmiss_area.update("")
            return
        # Each section guards independently: render what parses, skip what doesn't.
        try:
            line = (
                f"{int(diagnostics.get('fired') or 0)} fired · "
                f"{int(diagnostics.get('skipped') or 0)} skipped · "
                f"{int(diagnostics.get('tokens_used') or 0)}"
                f"/{int(diagnostics.get('token_budget') or 0)} tokens"
            )
            text = Text(line)
            if diagnostics.get("budget_exceeded"):
                text.append(" · over budget", style="bold")
            summary.update(text)
        except Exception:
            summary.update(Text("diagnostics unavailable", style="dim"))
        records = diagnostics.get("entries")
        records = records if isinstance(records, list) else []
        try:
            fired = sorted(
                (r for r in records if r.get("status") == "fired"),
                key=lambda r: (r.get("applied_order") is None, r.get("applied_order") or 0),
            )
            fired_text = Text()
            for record in fired:
                fired_text.append(
                    f"{record.get('pattern')} → {record.get('content_preview')}"
                    f" · ×{int(record.get('replacements') or 0)}"
                    f" · {int(record.get('token_cost') or 0)} tok\n"
                )
            fired_area.update(fired_text)
        except Exception:
            fired_area.update("")
        try:
            misses = sorted(
                (r for r in records if r.get("status") != "fired"),
                key=lambda r: int(r.get("input_index") or 0),
            )
            miss_text = Text()
            for record in misses:
                reason = _SKIP_REASON_COPY.get(str(record.get("status")), str(record.get("status")))
                miss_text.append(f"{record.get('pattern')} — {reason}\n", style="dim")
            nearmiss_area.update(miss_text)
        except Exception:
            nearmiss_area.update("")

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
