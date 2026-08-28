"""Bulk source-entry modal for the Watchlists inspection surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Select, Static, TextArea

from ...Utils.input_validation import sanitize_string, validate_text_input, validate_url
from ...Widgets.prune_safe_select import PruneSafeSelect


@dataclass(frozen=True)
class BulkSourceRequestRow:
    """One validated row in a bulk source request."""

    input_index: int
    payload: dict[str, Any]


class BulkSourcesCreateRequested(Message):
    """Ask the owning screen to persist one exact validated source batch."""

    def __init__(
        self,
        modal: "BulkSourcesModal",
        rows: tuple[BulkSourceRequestRow, ...],
    ) -> None:
        super().__init__()
        self.modal = modal
        self.rows = rows


class OpenBulkSourcesRequested(Message):
    """Ask the owning Watchlists screen to open the bulk modal."""


class BulkSourcesContinueRequested(Message):
    """Continue with source IDs without mutating collection membership."""

    def __init__(
        self,
        modal: "BulkSourcesModal",
        source_ids: tuple[str, ...],
        destination: str,
    ) -> None:
        super().__init__()
        self.modal = modal
        self.source_ids = source_ids
        self.destination = destination


@dataclass(frozen=True)
class _DisplayRow:
    input_index: int
    url: str
    outcome: str
    detail: str
    source_id: str | None = None


class BulkSourcesModal(ModalScreen[None]):
    """Validate and display ordered outcomes for up to 50 source URLs."""

    BINDINGS = [("escape", "cancel", "Cancel")]
    MAX_SOURCES = 50

    def __init__(
        self,
        *,
        source_types: Sequence[str] = ("rss", "atom", "url"),
        message_target: Any | None = None,
    ) -> None:
        super().__init__()
        self._source_types = tuple(str(value) for value in source_types)
        self._message_target = message_target
        self._validation_rows: list[_DisplayRow] = []
        self._submitted_rows: tuple[BulkSourceRequestRow, ...] = ()
        self._successful_ids: tuple[str, ...] = ()
        self._batch_posted = False
        self._continue_posted = False

    def compose(self) -> ComposeResult:
        with Vertical(id="bulk-sources-dialog", classes="opml-dialog"):
            yield Static("Add several sources", classes="dialog-title")
            yield Static("One URL per line · up to 50", id="bulk-sources-draft-label")
            yield TextArea("", id="bulk-sources-draft")
            with Horizontal(id="bulk-sources-options"):
                yield Static("Type", classes="bulk-sources-field-label")
                yield PruneSafeSelect(
                    [(self._type_label(value), value) for value in self._source_types],
                    value=self._source_types[0],
                    id="bulk-sources-type",
                    allow_blank=False,
                    compact=True,
                )
                yield Static("Tags", classes="bulk-sources-field-label")
                yield Input(
                    placeholder="Optional, comma separated",
                    id="bulk-sources-tags",
                    compact=True,
                )
                yield Static("Next", classes="bulk-sources-field-label")
                yield PruneSafeSelect(
                    [
                        ("All Sources", "all_sources"),
                        ("Create Watchlist next", "create_watchlist"),
                    ],
                    value="all_sources",
                    id="bulk-sources-destination",
                    allow_blank=False,
                    compact=True,
                )
            yield Static("Ready to validate.", id="bulk-sources-status")
            with VerticalScroll(id="bulk-sources-results-region"):
                table = DataTable(id="bulk-sources-results")
                table.add_columns("URL", "Result", "Details")
                yield table
            with Horizontal(id="bulk-sources-actions", classes="dialog-buttons"):
                yield Button(
                    "Validate and create",
                    id="bulk-sources-create",
                    variant="success",
                )
                yield Button("Cancel", id="bulk-sources-cancel")
            with Horizontal(id="bulk-sources-decisions", classes="dialog-buttons"):
                yield Button(
                    "Continue with successful sources",
                    id="bulk-sources-continue",
                    variant="primary",
                    disabled=True,
                )
                yield Button("Return to draft", id="bulk-sources-return")

    def on_mount(self) -> None:
        self.query_one("#bulk-sources-decisions").display = False
        self.query_one("#bulk-sources-draft", TextArea).focus()

    @staticmethod
    def _type_label(value: str) -> str:
        return {"rss": "RSS", "atom": "Atom", "url": "Web page"}.get(
            value, value.title()
        )

    def action_cancel(self) -> None:
        """Dismiss without writing or clearing the draft first."""
        if self._batch_posted:
            self._show_status("Creating sources… Keep this dialog open for results.")
            return
        if self.query_one("#bulk-sources-decisions").display:
            self._return_to_draft()
            return
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = str(event.button.id)
        if button_id == "bulk-sources-create":
            self._submit()
        elif button_id == "bulk-sources-return":
            self._return_to_draft()
        elif button_id == "bulk-sources-continue":
            self._continue()
        elif button_id == "bulk-sources-cancel":
            self.action_cancel()

    def _submit(self) -> None:
        if self._batch_posted:
            return
        draft = self.query_one("#bulk-sources-draft", TextArea).text
        urls = [line.strip() for line in draft.splitlines() if line.strip()]
        if not urls:
            self._show_status("Enter at least one URL, one per line.", error=True)
            return
        if len(urls) > self.MAX_SOURCES:
            self._show_status(
                "Enter no more than 50 nonblank URLs; the draft was preserved.",
                error=True,
            )
            return

        source_type = str(self.query_one("#bulk-sources-type", Select).value)
        tags = self._tags()
        self._validation_rows = []
        valid: list[BulkSourceRequestRow] = []
        for input_index, url in enumerate(urls):
            if not validate_url(url):
                self._validation_rows.append(
                    _DisplayRow(
                        input_index,
                        url,
                        "Invalid",
                        "Use an absolute http(s) URL.",
                    )
                )
                continue
            valid.append(
                BulkSourceRequestRow(
                    input_index=input_index,
                    payload={
                        "name": url,
                        "url": url,
                        "source_type": source_type,
                        "active": True,
                        "tags": tags,
                    },
                )
            )

        self._submitted_rows = tuple(valid)
        self._successful_ids = ()
        self._continue_posted = False
        self.query_one("#bulk-sources-continue", Button).disabled = True
        self._render_rows(self._validation_rows)
        self.query_one("#bulk-sources-decisions").display = False
        if not valid:
            self._show_status("No valid URLs were found. Return to the draft.", error=True)
            return
        self._batch_posted = True
        self.query_one("#bulk-sources-create", Button).disabled = True
        self.query_one("#bulk-sources-cancel", Button).disabled = True
        self._show_status(f"Creating {len(valid)} validated source(s)…")
        self._post_to_owner(BulkSourcesCreateRequested(self, self._submitted_rows))

    def _tags(self) -> list[str]:
        raw = self.query_one("#bulk-sources-tags", Input).value
        tags: list[str] = []
        for value in raw.split(","):
            cleaned = sanitize_string(value.strip(), max_length=100)
            if cleaned and validate_text_input(cleaned, max_length=100):
                tags.append(cleaned)
        return tags

    def apply_results(self, results: Sequence[Mapping[str, Any]]) -> None:
        """Merge exact-batch outcomes with invalid rows in original order."""
        rows = list(self._validation_rows)
        successful: list[str] = []
        results_by_index: dict[int, Mapping[str, Any]] = {}
        for result in results:
            try:
                batch_index = int(result.get("input_index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= batch_index < len(self._submitted_rows):
                results_by_index.setdefault(batch_index, result)

        for batch_index, request_row in enumerate(self._submitted_rows):
            result = results_by_index.get(batch_index, {})
            source = result.get("source")
            source_row = dict(source) if isinstance(source, Mapping) else {}
            source_id = self._canonical_source_id(source_row)
            outcome = str(result.get("outcome") or "invalid").casefold()
            succeeded = source_id is not None and outcome in {"created", "existing"}
            if succeeded:
                label = "Created" if outcome == "created" else "Existing"
                detail = "Ready for the next step."
                if source_id not in successful:
                    successful.append(source_id)
            else:
                label = "Invalid"
                detail = "Creation did not return a recognized source."
            rows.append(
                _DisplayRow(
                    request_row.input_index,
                    str(request_row.payload["url"]),
                    label,
                    detail,
                    source_id,
                )
            )
        self._successful_ids = tuple(successful)
        self._batch_posted = False
        self._continue_posted = False
        self._render_rows(rows)
        self._show_status(
            f"{len(successful)} source(s) are ready. Choose how to continue."
        )
        self.query_one("#bulk-sources-actions").display = False
        self.query_one("#bulk-sources-decisions").display = True
        self.query_one("#bulk-sources-continue", Button).disabled = not successful

    @staticmethod
    def _canonical_source_id(source: Mapping[str, Any]) -> str | None:
        """Return one valid local source identity from a result projection."""
        candidate = source.get("id")
        if not isinstance(candidate, str):
            return None
        prefix = "local:subscription:"
        if not candidate.startswith(prefix):
            return None
        suffix = candidate.removeprefix(prefix)
        return candidate if suffix.isdigit() and int(suffix) > 0 else None

    def show_write_failure(self, message: str) -> None:
        """Keep the draft and expose one bounded recovery message."""
        self._batch_posted = False
        self.query_one("#bulk-sources-create", Button).disabled = False
        self.query_one("#bulk-sources-cancel", Button).disabled = False
        self.query_one("#bulk-sources-actions").display = True
        self._show_status(message, error=True)
        self.query_one("#bulk-sources-decisions").display = False
        self.query_one("#bulk-sources-continue", Button).disabled = True

    def _render_rows(self, rows: Sequence[_DisplayRow]) -> None:
        table = self.query_one("#bulk-sources-results", DataTable)
        table.clear()
        for row in sorted(rows, key=lambda item: item.input_index):
            table.add_row(row.url, row.outcome, row.detail)

    def _show_status(self, message: str, *, error: bool = False) -> None:
        status = self.query_one("#bulk-sources-status", Static)
        status.update(message)
        status.set_class(error, "is-error")

    def _return_to_draft(self) -> None:
        self._successful_ids = ()
        self._batch_posted = False
        self._continue_posted = False
        self.query_one("#bulk-sources-create", Button).disabled = False
        self.query_one("#bulk-sources-cancel", Button).disabled = False
        self.query_one("#bulk-sources-actions").display = True
        self.query_one("#bulk-sources-decisions").display = False
        self.query_one("#bulk-sources-continue", Button).disabled = True
        self._show_status("Draft restored. Edit URLs, then validate again.")
        self.query_one("#bulk-sources-draft", TextArea).focus()

    def _continue(self) -> None:
        if not self._successful_ids or self._continue_posted:
            return
        self._continue_posted = True
        self.query_one("#bulk-sources-continue", Button).disabled = True
        destination = str(
            self.query_one("#bulk-sources-destination", Select).value
        )
        self._post_to_owner(
            BulkSourcesContinueRequested(
                self,
                self._successful_ids,
                destination,
            )
        )

    def _post_to_owner(self, message: Message) -> None:
        target = self._message_target or self.app
        target.post_message(message)
