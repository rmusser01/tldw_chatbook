"""Try-it injection preview for the Roleplay Lore/world-book mode."""

from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static, Switch, TextArea

_POSITION_ORDER = ("before_char", "at_start", "at_end", "after_char")
_POSITION_LABELS = {
    "before_char": "Before character",
    "at_start": "At start",
    "at_end": "At end",
    "after_char": "After character",
}


class LoreTryItRunRequested(Message):
    def __init__(self, text: str, pull_history: bool) -> None:
        super().__init__()
        self.text = text
        self.pull_history = pull_history


class PersonasLoreTryItWidget(Vertical):
    """Sample text in, injection-by-position preview + diagnostics out.

    Owns no I/O — the screen runs the scan and calls ``render_result``.
    """

    BINDINGS = [
        Binding("ctrl+enter", "run_preview", "Run preview", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    PersonasLoreTryItWidget {
        height: auto;
        max-height: 60%;
    }
    PersonasLoreTryItWidget #personas-lore-tryit-sample {
        height: 4;
    }
    PersonasLoreTryItWidget #personas-lore-tryit-pull-history-row {
        height: auto;
    }
    PersonasLoreTryItWidget #personas-lore-tryit-status {
        height: 1;
    }
    PersonasLoreTryItWidget #personas-lore-tryit-injections {
        height: auto;
        max-height: 10;
        overflow-y: auto;
    }
    PersonasLoreTryItWidget #personas-lore-tryit-summary {
        height: 1;
    }
    PersonasLoreTryItWidget #personas-lore-tryit-details {
        height: auto;
        max-height: 12;
        overflow-y: auto;
    }
    PersonasLoreTryItWidget #personas-lore-tryit-fired,
    PersonasLoreTryItWidget #personas-lore-tryit-nearmiss {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Try it — injection preview", markup=False)
        yield TextArea(id="personas-lore-tryit-sample")
        with Horizontal(id="personas-lore-tryit-pull-history-row"):
            # Disabled in P2a: the Personas screen has no conversation-history
            # source yet, so this preview scans the sample text only. Recent-turn
            # scanning arrives with the later Lore send-integration cycle.
            yield Switch(
                id="personas-lore-tryit-pull-history",
                disabled=True,
                tooltip="Scanning recent conversation turns arrives in a later Lore cycle.",
            )
            yield Static("Include recent turns (soon)", markup=False)
        yield Button(
            "Run preview",
            id="personas-lore-tryit-run",
            classes="console-action-secondary",
            disabled=True,
            tooltip="Run the world book scan against the sample text (Ctrl+Enter).",
        )
        yield Static("", id="personas-lore-tryit-status", markup=False)
        yield Static("", id="personas-lore-tryit-injections")
        yield Static("", id="personas-lore-tryit-summary", markup=False)
        with Vertical(id="personas-lore-tryit-details"):
            yield Static("", id="personas-lore-tryit-fired")
            yield Static("", id="personas-lore-tryit-nearmiss")

    def set_ready(self, ready: bool, hint: str = "") -> None:
        self.query_one("#personas-lore-tryit-run", Button).disabled = not ready
        if hint:
            self.query_one("#personas-lore-tryit-status", Static).update(hint)

    def sample_text(self) -> str:
        return self.query_one("#personas-lore-tryit-sample", TextArea).text

    def pull_history(self) -> bool:
        return bool(self.query_one("#personas-lore-tryit-pull-history", Switch).value)

    def render_result(self, injections: dict, diagnostics: dict | None = None) -> None:
        """Renders the injection-by-position preview and, when available, the
        diagnostics story.

        Args:
            injections: Mapping of position -> list of injected content
                strings, as produced by ``world_info_processor``.
            diagnostics: The scan service's diagnostics dict, or None to
                degrade to an "unavailable" note in the diagnostics story.
        """
        injections = injections if isinstance(injections, dict) else {}
        preview = Text()
        any_content = False
        for position in _POSITION_ORDER:
            lines = injections.get(position) or []
            if not lines:
                continue
            any_content = True
            preview.append(f"{_POSITION_LABELS[position]}:\n", style="bold")
            for line in lines:
                preview.append(f"  {line}\n")
        status = self.query_one("#personas-lore-tryit-status", Static)
        if any_content:
            status.update("Injected content shown below, grouped by position.")
        else:
            status.update("No entries fired - nothing was injected.")
        self.query_one("#personas-lore-tryit-injections", Static).update(preview)
        self._render_diagnostics(diagnostics)

    def _render_diagnostics(self, diagnostics: dict | None) -> None:
        summary = self.query_one("#personas-lore-tryit-summary", Static)
        fired_area = self.query_one("#personas-lore-tryit-fired", Static)
        nearmiss_area = self.query_one("#personas-lore-tryit-nearmiss", Static)
        if not isinstance(diagnostics, dict):
            summary.update(Text("diagnostics unavailable", style="dim"))
            fired_area.update("")
            nearmiss_area.update("")
            return
        # Each section guards independently: render what parses, skip what doesn't.
        try:
            line = (
                f"{int(diagnostics.get('fired') or 0)} fired · "
                f"{int(diagnostics.get('skipped') or 0)} near-miss · "
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
                key=lambda r: (r.get("injection_order") is None, r.get("injection_order") or 0),
            )
            fired_text = Text()
            for record in fired:
                keys_str = ", ".join(str(k) for k in (record.get("keys") or []))
                fired_text.append(
                    f"{keys_str} → {record.get('content_preview')}"
                    f" · {int(record.get('token_cost') or 0)} tok\n"
                )
            fired_area.update(fired_text)
        except Exception:
            fired_area.update("")
        try:
            misses = [r for r in records if r.get("status") != "fired"]
            miss_text = Text()
            for record in misses:
                keys_str = ", ".join(str(k) for k in (record.get("keys") or []))
                miss_text.append(
                    f"{keys_str} — {record.get('activation_reason')}\n", style="dim"
                )
            nearmiss_area.update(miss_text)
        except Exception:
            nearmiss_area.update("")

    def show_error(self, message: str) -> None:
        self.query_one("#personas-lore-tryit-status", Static).update(message)

    def _post_run(self) -> None:
        text = self.sample_text()
        if not text.strip():
            self.show_error("Type some sample text first.")
            return
        self.post_message(LoreTryItRunRequested(text, self.pull_history()))

    def action_run_preview(self) -> None:
        if not self.query_one("#personas-lore-tryit-run", Button).disabled:
            self._post_run()

    @on(Button.Pressed, "#personas-lore-tryit-run")
    def _run_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_run()


__all__ = [
    "LoreTryItRunRequested",
    "PersonasLoreTryItWidget",
]
