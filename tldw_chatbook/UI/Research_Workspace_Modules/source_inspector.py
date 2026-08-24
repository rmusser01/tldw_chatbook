"""Honest source status, preview, and device-only annotation modal."""

from __future__ import annotations

from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea

from ...Research_Workspace import (
    ResearchSourcePreview,
    ResearchSourceSummary,
    SourceReadiness,
)


@dataclass(frozen=True, slots=True)
class ResearchSourceAnnotationDraft:
    source_id: str
    quote: str
    note: str


class ResearchSourceInspectorModal(ModalScreen[ResearchSourceAnnotationDraft | None]):
    """Display normalized owner facts without inventing unavailable progress."""

    BINDINGS = []

    def __init__(
        self,
        source: ResearchSourceSummary,
        *,
        readiness: SourceReadiness | None,
        preview: ResearchSourcePreview | None,
    ) -> None:
        super().__init__(id="research-source-inspector-modal")
        self.source = source
        self.readiness = readiness
        self.preview = preview

    def compose(self) -> ComposeResult:
        readiness = self.readiness
        state = (
            readiness.state.value.replace("_", " ").title()
            if readiness
            else "Unavailable"
        )
        reason = (
            readiness.detail
            if readiness and readiness.detail
            else "Owner did not report a reason."
        )
        next_action = (
            readiness.next_action
            if readiness and readiness.next_action
            else "Refresh / Recheck status."
        )
        truth = self.source.ref.data_source.value.title()
        with Vertical(id="research-source-inspector-dialog"):
            with Horizontal(id="research-source-inspector-heading"):
                yield Static(self.source.title, id="research-source-inspector-title")
                yield Button(
                    "Close", id="research-source-inspector-close", compact=True
                )
            with VerticalScroll(id="research-source-inspector-body"):
                yield Static(
                    f"Lifecycle: {state}", id="research-source-status-lifecycle"
                )
                yield Static(f"Reason: {reason}", id="research-source-status-reason")
                yield Static(
                    f"Source of truth: {truth} owner", id="research-source-status-truth"
                )
                yield Static(
                    "Progress: owner did not report a percentage",
                    id="research-source-status-progress",
                )
                yield Static(
                    f"Retry eligible: {'Yes' if readiness and readiness.retry_eligible else 'No'}",
                    id="research-source-status-retry",
                )
                yield Static(
                    f"Stale: {'Yes' if readiness and readiness.stale else 'No'}",
                    id="research-source-status-stale",
                )
                yield Static(
                    f"Readiness: {state}", id="research-source-status-readiness"
                )
                yield Static(
                    f"Association ID: {self.source.source_id} · Catalog ID: {self.source.catalog_item_id or 'missing'}",
                    id="research-source-status-identifiers",
                )
                yield Static(
                    f"Next action: {next_action}",
                    id="research-source-status-next-action",
                )
                preview_text = (
                    self.preview.text
                    if self.preview is not None and self.preview.text
                    else "Preview unavailable. Refresh or open the canonical Library/Media item."
                )
                yield Static(
                    preview_text, id="research-source-preview-text", markup=False
                )
                yield Static(
                    "Annotation · Device-only", id="research-source-annotation-label"
                )
                yield Input(
                    placeholder="Quoted text (optional)",
                    id="research-source-annotation-quote",
                )
                yield TextArea(id="research-source-annotation-note")
                yield Button(
                    "Save annotation",
                    id="research-source-annotation-save",
                    variant="primary",
                )

    @on(Button.Pressed, "#research-source-inspector-close")
    def close(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#research-source-annotation-save")
    def save(self) -> None:
        quote = self.query_one("#research-source-annotation-quote", Input).value.strip()
        note = self.query_one("#research-source-annotation-note", TextArea).text.strip()
        if not quote and not note:
            self.notify("Enter quoted text or a note.", severity="warning")
            return
        self.dismiss(ResearchSourceAnnotationDraft(self.source.source_id, quote, note))
