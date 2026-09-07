"""Honest source status, preview, and device-only annotation modal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static, TextArea

from ...Research_Workspace import (
    ResearchSourcePreview,
    ResearchSourceSummary,
    SourceReadiness,
)
from ...Widgets.modal_dismissal import SafeModalDismissMixin
from ...Research_Workspace.overlay_store import ResearchSourceAnnotation


@dataclass(frozen=True, slots=True)
class ResearchSourceAnnotationDraft:
    source_id: str
    quote: str
    note: str
    action: Literal["create", "update", "delete", "recheck"] = "create"
    annotation_id: str = ""


class ResearchSourceInspectorModal(
    SafeModalDismissMixin, ModalScreen[ResearchSourceAnnotationDraft | None]
):
    """Display normalized owner facts without inventing unavailable progress."""

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-source-inspector-dialog"

    def __init__(
        self,
        source: ResearchSourceSummary,
        *,
        readiness: SourceReadiness | None,
        preview: ResearchSourcePreview | None,
        annotations: tuple[ResearchSourceAnnotation, ...] = (),
    ) -> None:
        super().__init__(id="research-source-inspector-modal")
        self.source = source
        self.readiness = readiness
        self.preview = preview
        self.annotations = tuple(
            annotation
            for annotation in annotations
            if annotation.source_id == source.source_id
        )
        self._selected_annotation_id = ""

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
                yield Button(
                    "Refresh / Recheck status",
                    id="research-source-status-recheck",
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
                yield Select(
                    tuple(
                        (
                            annotation.note or annotation.quote,
                            annotation.annotation_id,
                        )
                        for annotation in self.annotations
                    ),
                    allow_blank=True,
                    id="research-source-annotation-list",
                )
                with Horizontal(id="research-source-annotation-actions"):
                    yield Button(
                        "New annotation", id="research-source-annotation-new"
                    )
                    yield Button(
                        "Delete annotation",
                        id="research-source-annotation-delete",
                        disabled=True,
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

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Escape is the same non-mutating outcome as Close."""

        del source
        self.dismiss_safe_once(None)

    @on(Button.Pressed, "#research-source-status-recheck")
    def recheck(self) -> None:
        """Return an explicit refresh request to the owner screen."""

        self.dismiss(
            ResearchSourceAnnotationDraft(
                self.source.source_id, "", "", action="recheck"
            )
        )

    @on(Button.Pressed, "#research-source-annotation-save")
    def save(self) -> None:
        quote = self.query_one("#research-source-annotation-quote", Input).value.strip()
        note = self.query_one("#research-source-annotation-note", TextArea).text.strip()
        if not quote and not note:
            self.notify("Enter quoted text or a note.", severity="warning")
            return
        self.dismiss(
            ResearchSourceAnnotationDraft(
                self.source.source_id,
                quote,
                note,
                action="update" if self._selected_annotation_id else "create",
                annotation_id=self._selected_annotation_id,
            )
        )

    @on(Select.Changed, "#research-source-annotation-list")
    def select_annotation(self, event: Select.Changed) -> None:
        annotation_id = str(event.value or "")
        annotation = next(
            (
                item
                for item in self.annotations
                if item.annotation_id == annotation_id
            ),
            None,
        )
        self._selected_annotation_id = annotation_id if annotation is not None else ""
        self.query_one("#research-source-annotation-delete", Button).disabled = (
            annotation is None
        )
        if annotation is None:
            return
        self.query_one("#research-source-annotation-quote", Input).value = annotation.quote
        self.query_one("#research-source-annotation-note", TextArea).text = annotation.note

    @on(Button.Pressed, "#research-source-annotation-new")
    def new_annotation(self) -> None:
        self._selected_annotation_id = ""
        self.query_one("#research-source-annotation-list", Select).clear()
        self.query_one("#research-source-annotation-quote", Input).value = ""
        self.query_one("#research-source-annotation-note", TextArea).text = ""
        self.query_one("#research-source-annotation-delete", Button).disabled = True

    @on(Button.Pressed, "#research-source-annotation-delete")
    def delete_annotation(self) -> None:
        if not self._selected_annotation_id:
            return
        self.dismiss(
            ResearchSourceAnnotationDraft(
                self.source.source_id,
                "",
                "",
                action="delete",
                annotation_id=self._selected_annotation_id,
            )
        )
