"""Source-mode controls and keyboard-accessible syntax diagnostics."""

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Collapsible, OptionList, Select, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Document_Editing.models import CheckResult, DocumentRequest

MODES = (
    ("Default", None),
    ("Text", "text"),
    ("Markdown", "markdown"),
    ("YAML", "yaml"),
    ("JSON", "json"),
    ("JSONL", "jsonl"),
)


class DocumentDiagnostics(Vertical):
    """Publish actions; checking, formatting and saving belong to the host."""

    class Action(Message):
        def __init__(
            self,
            control: "DocumentDiagnostics",
            action: str,
            value: str | int | None = None,
        ) -> None:
            super().__init__()
            self._control = control
            self.action = action
            self.value = value

        @property
        def control(self) -> "DocumentDiagnostics":
            return self._control

    def __init__(self, *, temporary: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temporary = temporary
        self._mode: str | None = None
        self._presented = False

    def compose(self) -> ComposeResult:
        with Horizontal(classes="document-actions ds-toolbar"):
            yield Select(
                MODES,
                value=None,
                allow_blank=False,
                compact=True,
                classes="document-language",
            )
            yield Button("Check", name="check", classes="document-check", compact=True)
            yield Button(
                "Format document",
                name="format",
                classes="document-format",
                compact=True,
                disabled=True,
            )
        yield Static(
            (
                "Mode is temporary for this open file"
                if self.temporary
                else "Mode is saved with the note"
            ),
            classes="document-language-scope",
            markup=False,
        )
        yield Static("Pending", classes="document-check-status", markup=False)
        with Collapsible(title="Problems", collapsed=True, classes="document-problems"):
            yield OptionList(classes="document-problem-list", markup=False)
            yield Static("", classes="document-truncation", markup=False)

    def present(
        self,
        request: DocumentRequest | None,
        result: CheckResult | None,
        status: str,
        editable: bool,
    ) -> None:
        """Patch status without moving focus or remounting the editor."""
        if not self.is_attached or not self.query(Select):
            return
        language = request.language if request else self._mode
        selector = self.query_one(Select)
        if language != self._mode:
            self._mode = language
            options: list[tuple[Text | str, str | None]] = list(MODES)
            if language is not None and language not in {value for _, value in options}:
                options.append((Text(f"Unknown: {language}"), language))
            with selector.prevent(Select.Changed):
                selector.set_options(options)
                selector.value = language
        if result is not None and not status:
            if result.state == "checked":
                errors = sum(d.severity == "error" for d in result.diagnostics)
                warnings = len(result.diagnostics) - errors
                counts = []
                if errors:
                    counts.append(f"{errors} error{'s' if errors != 1 else ''}")
                if warnings:
                    counts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
                status = (
                    " · ".join(counts)
                    if counts
                    else ("No syntax errors" if result.complete else "Check incomplete")
                )
                if not result.complete and counts:
                    status += " shown · incomplete"
            else:
                status = {
                    "unsupported": "Unsupported mode or excerpt",
                    "resource_limited": "Check resource limit reached",
                    "failed": "Check failed — try Check again",
                }[result.state]
        mode = (language or "Default").upper()
        allowed = bool(result and result.format_allowed and editable)
        reason = (
            ""
            if allowed
            else (
                " · Format unavailable: read-only"
                if not editable
                else (
                    " · Format blocked: diagnostics"
                    if result and result.diagnostics
                    else " · Format needs a complete check"
                )
            )
        )
        self.query_one(".document-check-status", Static).update(
            f"{mode} · {status}{reason}"
        )
        self.query_one(".document-format", Button).disabled = not allowed
        problems = self.query_one(".document-problem-list", OptionList)
        problems.clear_options()
        if request and result:
            for diagnostic in result.diagnostics:
                prefix = request.text[: diagnostic.start]
                line = prefix.count("\n") + 1
                column = len(prefix.rsplit("\n", 1)[-1].removesuffix("\r")) + 1
                problems.add_option(
                    Option(
                        f"{diagnostic.severity} {line}:{column} [{diagnostic.code}] {diagnostic.message}"
                    )
                )
        truncated = bool(result and result.truncated)
        self.query_one(".document-truncation", Static).update(
            "Further diagnostics omitted; check is incomplete." if truncated else ""
        )
        self._presented = True
        self.query_one(Collapsible).title = (
            f'Problems ({len(result.diagnostics) if result else 0}{"+" if truncated else ""})'
        )

    @on(Select.Changed)
    def _language_changed(self, event: Select.Changed) -> None:
        event.stop()
        if (
            self._presented
            and event.value != Select.BLANK
            and event.value == event.select.value
            and self.is_mounted
        ):
            self.post_message(
                self.Action(
                    self,
                    "language",
                    None if event.value is None else str(event.value),
                )
            )

    @on(Button.Pressed)
    def _action(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Action(self, event.button.name or ""))

    @on(OptionList.OptionSelected, ".document-problem-list")
    def _navigate(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.query_one(Collapsible).collapsed = True
        self.post_message(self.Action(self, "navigate", event.option_index))
