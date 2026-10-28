"""Describe, review, and request a client copy of retained diagnostics (ADR-252)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Collapsible, Label, Static, TextArea

from tldw_chatbook.Utils.bug_report import (
    MAX_AUTHORED_BYTES,
    BugReportDraft,
    BugReportOptions,
    prepare_bug_report_copy,
    render_bug_report,
)
from tldw_chatbook.Utils.bug_report_capture import BugReportCollector
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.confirmation_dialog import (
    ConfirmationDialog,
    confirm_quit_discarding_edits,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Utils.bug_report_capture import BugReportContext


class _ReportForm(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)
    description: str = Field(repr=False)
    steps: str = Field(default="", repr=False)
    expected: str = Field(default="", repr=False)
    actual: str = Field(default="", repr=False)

    @model_validator(mode="after")
    def check_text(self):
        fields = (self.description, self.steps, self.expected, self.actual)
        # Fenced plain text may legitimately describe HTML/script bugs.
        if not all(
            validate_text_input(text, max_length=MAX_AUTHORED_BYTES, allow_html=True)
            for text in fields
        ):
            raise ValueError("Report fields exceed the text limit")
        if not self.description.strip():
            raise ValueError("A description is required")
        if sum(len(text.encode("utf-8")) for text in fields) > MAX_AUTHORED_BYTES:
            raise ValueError("Report fields exceed the byte limit")
        return self


class BugReportDialog(SafeModalDismissMixin, ModalScreen[None]):
    """One memory-only report with frozen evidence and editable review text."""

    SAFE_MODAL_CONTENT = "#bug-report-dialog"
    AUTO_FOCUS = "#bug-report-description"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    BUNDLED_CSS = """
    BugReportDialog { align: center middle; }
    #bug-report-dialog {
        width: $ds-inspector-modal-width;
        max-width: $ds-percent-95;
        height: $ds-inspector-modal-height;
        max-height: $ds-percent-95;
        border: round $ds-grid-line;
        background: $ds-surface-panel;
        padding: $ds-space-inset;
    }
    #bug-report-describe, #bug-report-review-stage { height: $ds-height-fill; }
    #bug-report-describe { scrollbar-gutter: stable; }
    BugReportDialog .form-textarea { height: $ds-textarea-min-height; }
    #bug-report-review {
        height: $ds-height-fill;
        max-height: $ds-height-full;
        margin: $ds-space-0;
    }
    #bug-report-inventory, #bug-report-selection, #bug-report-disclosure {
        height: $ds-height-auto;
        margin-bottom: $ds-space-stack;
    }
    #bug-report-status {
        height: $ds-height-auto;
        max-height: $ds-size-3;
        margin: $ds-space-0;
        padding: $ds-space-0 $ds-space-inline;
    }
    #bug-report-status.-error { color: $ds-status-error-readable; }
    #bug-report-actions { dock: bottom; }
    """

    def __init__(self, app_instance: TldwCli, context: BugReportContext) -> None:
        super().__init__()
        self._collector = BugReportCollector(app_instance, context)
        self._report_context = context
        self._capture = None
        self._ready = asyncio.Event()
        self._generating = False
        self._copying = False
        self._reviewing = False
        self._generated_text = ""
        self._review_revision = 0
        self._close_prompt = False

    def compose(self) -> ComposeResult:
        with Vertical(id="bug-report-dialog"):
            yield Static("Report a bug", classes="dialog-title", markup=False)
            with VerticalScroll(id="bug-report-describe"):
                yield Label("Description (required)", classes="form-label")
                yield TextArea(id="bug-report-description", classes="form-textarea")
                with Collapsible(
                    title="Steps, expected and actual behavior",
                    collapsed=True,
                    classes="form-section-collapsible",
                ):
                    for name, label in (
                        ("steps", "Steps to reproduce"),
                        ("expected", "Expected behavior"),
                        ("actual", "Actual behavior"),
                    ):
                        yield Label(label, classes="form-label")
                        yield TextArea(id=f"bug-report-{name}", classes="form-textarea")
                selected = self._report_context.conversation_label
                yield Static(
                    f"Selected Console conversation: {selected}. Open another conversation in Console before starting a report to change this selection."
                    if selected
                    else "Console trace unavailable: no selected local durable conversation. You can still describe the issue and include session logs.",
                    id="bug-report-selection",
                    markup=False,
                )
                yield Checkbox(
                    "Application-wide session logs and exceptions",
                    value=True,
                    id="bug-report-logs",
                    classes="form-checkbox",
                )
                yield Checkbox(
                    "Console events, timing and errors",
                    value=True,
                    id="bug-report-trace",
                    classes="form-checkbox",
                )
                yield Checkbox(
                    "Include redacted tool inputs/results (off by default)",
                    value=False,
                    disabled=not self._report_context.conversation_id,
                    id="bug-report-tools",
                    classes="form-checkbox",
                )
                yield Static(
                    "Tool results can contain file, document and other-conversation excerpts. Tool content requires a selected Console trace. Review before sharing; redaction can miss private details. Nothing is uploaded.",
                    id="bug-report-disclosure",
                    markup=False,
                )
                yield Static(
                    "Collecting retained evidence…",
                    id="bug-report-inventory",
                    markup=False,
                )
            with Vertical(id="bug-report-review-stage"):
                yield Static(
                    "Edit or remove anything before copying. Share the report manually.",
                    markup=False,
                )
                yield TextArea(id="bug-report-review", classes="form-textarea")
            yield Static(
                "Describe what went wrong.",
                id="bug-report-status",
                classes="status-area",
                markup=False,
            )
            with Horizontal(
                id="bug-report-actions",
                classes="dialog-buttons button-group button-group-right",
            ):
                yield Button("Back", id="bug-report-back")
                yield Button("Generate", id="bug-report-generate", variant="primary")
                yield Button("Copy report", id="bug-report-copy", variant="primary")
                yield Button("Close", id="bug-report-close")

    def on_mount(self) -> None:
        self._set_stage(False)
        self.run_worker(
            self._collect(),
            group="bug-report-collect",
            exclusive=True,
            exit_on_error=False,
        )
        self.request_description_focus()

    def on_unmount(self) -> None:
        self._collector.cancel()

    def request_description_focus(self) -> None:
        """Focus the current report without replacing its text or collecting again."""
        if self.is_mounted:
            self.query_one(
                "#bug-report-review" if self._reviewing else "#bug-report-description",
                TextArea,
            ).focus()

    def _set_stage(self, reviewing: bool) -> None:
        self._reviewing = reviewing
        self.query_one("#bug-report-describe").display = not reviewing
        self.query_one("#bug-report-review-stage").display = reviewing
        self.query_one("#bug-report-generate").display = not reviewing
        for name in ("back", "copy"):
            self.query_one(f"#bug-report-{name}").display = reviewing
        self.request_description_focus()

    def _status(self, text: str, *, error: bool = False) -> None:
        if self.is_mounted:
            status = self.query_one("#bug-report-status", Static)
            status.set_class(error, "-error")
            status.update(text)

    async def _collect(self) -> None:
        try:
            self._capture = await self._collector.collect()
            if self.is_mounted:
                evidence = self._capture.evidence
                notes = (
                    f"Frozen excerpt: {len(evidence.logs)} log entries; {len(evidence.trace)} Console facts. ",
                    *evidence.inventory,
                )
                self.query_one("#bug-report-inventory", Static).update("\n".join(notes))
        except Exception:
            self._status(
                "Evidence unavailable. You can still generate a description-only report.",
                error=True,
            )
        finally:
            self._ready.set()

    @on(Checkbox.Changed, "#bug-report-trace")
    def _trace_changed(self, event: Checkbox.Changed) -> None:
        tools = self.query_one("#bug-report-tools", Checkbox)
        tools.disabled = not event.value or not self._report_context.conversation_id
        if tools.disabled:
            tools.value = False

    @on(TextArea.Changed, "#bug-report-review")
    def _review_changed(self) -> None:
        self._review_revision += 1

    @on(Button.Pressed, "#bug-report-generate")
    def _generate_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._start_generate()

    def _start_generate(self, discard_confirmed: bool = False) -> None:
        if self._generating or not self.is_mounted or self.app.screen is not self:
            return
        review = self.query_one("#bug-report-review", TextArea)
        if (
            review.text
            and review.text != self._generated_text
            and not discard_confirmed
        ):

            def confirmed(choice):
                if choice:
                    self.call_after_refresh(self._start_generate, True)

            self.app.push_screen(
                ConfirmationDialog(
                    title="Replace review edits?",
                    message="Generating again replaces your manual review edits.",
                    confirm_label="Replace edits",
                    cancel_label="Keep edits",
                ),
                confirmed,
            )
            return
        values = {
            name: self.query_one(f"#bug-report-{name}", TextArea).text
            for name in ("description", "steps", "expected", "actual")
        }
        options = BugReportOptions(
            *(
                self.query_one(f"#bug-report-{name}", Checkbox).value
                for name in ("logs", "trace", "tools")
            )
        )
        self._generating = True
        self.query_one("#bug-report-generate").disabled = True
        self._status("Preparing report…")
        self.run_worker(
            self._generate(values, options),
            group="bug-report-generate",
            exclusive=True,
            exit_on_error=False,
        )

    async def _generate(
        self, values: dict[str, str], options: BugReportOptions
    ) -> None:
        try:
            form = await asyncio.to_thread(_ReportForm.model_validate, values)
            await self._ready.wait()
            from tldw_chatbook.Utils.bug_report import BugReportEvidence

            capture = self._capture
            if capture is not None and options.include_tool_content:
                capture = await self._collector.collect_tools(capture)
            evidence = capture.evidence if capture is not None else BugReportEvidence()
            if not self._collector.is_current():
                evidence = replace(
                    evidence,
                    trace=(),
                    tools=(),
                    inventory=evidence.inventory
                    + (
                        "Selection changed; Console evidence excluded. Close and reopen to select it again.",
                    ),
                )
            text = await asyncio.to_thread(
                render_bug_report,
                BugReportDraft(**form.model_dump()),
                evidence,
                options,
            )
            if self.is_mounted and self.app.screen is self:
                current_values = {
                    name: self.query_one(f"#bug-report-{name}", TextArea).text
                    for name in values
                }
                current_options = BugReportOptions(
                    *(
                        self.query_one(f"#bug-report-{name}", Checkbox).value
                        for name in ("logs", "trace", "tools")
                    )
                )
                if current_values != values or current_options != options:
                    self._status(
                        "Description or evidence choices changed. Generate again to use them."
                    )
                    return
                self._generated_text = text
                self.query_one("#bug-report-review", TextArea).load_text(text)
                self._set_stage(True)
                self._status(
                    f"Review before sharing — {len(text.encode('utf-8')):,} / 60,000 bytes."
                )
        except (ValidationError, ValueError):
            self._status(
                "Enter a description; all four fields together must fit 8,000 UTF-8 bytes.",
                error=True,
            )
            self.request_description_focus()
        except Exception:
            self._status(
                "Report preparation unavailable. Your description is preserved.",
                error=True,
            )
        finally:
            self._generating = False
            if self.is_mounted:
                self.query_one("#bug-report-generate").disabled = False

    @on(Button.Pressed, "#bug-report-back")
    def _back(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._copying:
            self._set_stage(False)

    @on(Button.Pressed, "#bug-report-copy")
    def _copy_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._copying:
            return
        self._copying = True
        self.query_one("#bug-report-copy").disabled = True
        review = self.query_one("#bug-report-review", TextArea)
        self.run_worker(
            self._copy(review.text, self._review_revision),
            group="bug-report-copy",
            exclusive=True,
            exit_on_error=False,
        )

    async def _copy(self, candidate: str, revision: int) -> None:
        try:
            checked = await asyncio.to_thread(prepare_bug_report_copy, candidate)
            if not self.is_mounted or self.app.screen is not self:
                return
            review = self.query_one("#bug-report-review", TextArea)
            if review.text != candidate or self._review_revision != revision:
                self._status(
                    "Report changed during checking. Review it and press Copy report again."
                )
                return
            if not self._collector.is_current():
                self._status(
                    "Selection changed. Close and reopen reporting before copying.",
                    error=True,
                )
                return
            if checked.state == "review_required":
                review.load_text(checked.text)
                self._status(checked.message)
            elif checked.state != "ready":
                self._status(checked.message, error=True)
            else:
                try:
                    self.app.copy_to_clipboard(checked.text)
                except Exception:
                    self._status(
                        "Clipboard unavailable. Select and copy the review text manually.",
                        error=True,
                    )
                    review.focus()
                else:
                    self._status(
                        "Copy requested. Paste the report into your issue, message or paste site to check delivery."
                    )
        finally:
            self._copying = False
            if self.is_mounted:
                self.query_one("#bug-report-copy").disabled = False

    def _dirty(self) -> bool:
        return any(
            self.query_one(f"#bug-report-{name}", TextArea).text
            for name in ("description", "steps", "expected", "actual", "review")
        )

    @on(Button.Pressed, "#bug-report-close")
    async def _close(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._close_prompt:
            return
        if not self._dirty():
            self.dismiss_safe_once(None)
            return
        self._close_prompt = True

        def confirmed(choice):
            self._close_prompt = False
            if choice and self.is_mounted:
                self.call_after_refresh(self.dismiss_safe_once_when_on_top, None)

        self.app.push_screen(
            ConfirmationDialog(
                title="Discard bug report?",
                message="Your description and review edits are kept only in this dialog.",
                confirm_label="Discard report",
                cancel_label="Keep editing",
            ),
            confirmed,
        )

    async def confirm_quit(self) -> bool:
        """Use the same discard decision when the app's quit walk reaches us."""
        return not self._dirty() or await confirm_quit_discarding_edits(
            self, "Quitting discards this memory-only bug report."
        )
