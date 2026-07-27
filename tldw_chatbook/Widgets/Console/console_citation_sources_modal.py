"""Lazy, literal Console citation-source inspection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, ListItem, ListView, Static

from tldw_chatbook.Chat.citation_source_locators import (
    AuthorityScope,
    CitationReadAuthorization,
)
from tldw_chatbook.Chat.citation_trace_models import (
    EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX,
    CitationTrace,
    StructuralValidationState,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    ActiveCitationTraceState,
    CitationHydrationResult,
    CitationHydrationState,
)
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID,
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE,
)


OPEN_SOURCE_TYPES = {
    "media_db": "media",
    "notes": "notes",
    "chat_history": "conversations",
}


@dataclass(frozen=True, slots=True)
class ConsoleCitationSourceRow:
    """Literal display data for one selected cited evidence entry."""

    display_marker: str
    evidence_ordinal: int
    title: str
    snapshot_text: str
    source_kind: str | None
    source_id: str | None
    open_source_type: str | None


def selected_valid_evidence_ordinals(trace: CitationTrace) -> tuple[int, ...]:
    """Return selected-attempt valid evidence ordinals in first-citation order.

    Args:
        trace: Citation trace containing the selected answer attempt.

    Returns:
        Deduplicated valid evidence ordinals in first-citation order.
    """

    selected_attempt = next(
        (
            attempt
            for attempt in trace.answer_attempts
            if attempt.attempt_id == trace.selected_attempt_id
        ),
        None,
    )
    if selected_attempt is None:
        return ()

    seen: set[int] = set()
    ordinals: list[int] = []
    for occurrence in selected_attempt.occurrences:
        evidence_ordinal = occurrence.evidence_ordinal
        if (
            occurrence.structural_state is not StructuralValidationState.VALID
            or type(evidence_ordinal) is not int
            or evidence_ordinal in seen
        ):
            continue
        seen.add(evidence_ordinal)
        ordinals.append(evidence_ordinal)
    return tuple(ordinals)


def _safe_source_identifier(value: object) -> str | None:
    """Return an inert bounded identifier without coercing untrusted JSON."""

    if type(value) is not str or not value:
        return None
    if len(value.encode("utf-8")) > EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX:
        return None
    return value


def _render_literal_text(value: str) -> str:
    """Make terminal controls visible while preserving ordinary text layout."""

    return "".join(
        character
        if character in "\n\t"
        or not (ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F)
        else f"\\x{ord(character):02x}"
        for character in value
    )


def build_console_citation_source_rows(
    hydration: CitationHydrationResult,
) -> tuple[ConsoleCitationSourceRow, ...] | None:
    """Join one authorized hydration into complete cited rows.

    ``None`` is the single unavailable outcome. A missing selected attempt,
    prompt set, prompt entry, snapshot payload, or exact snapshot body rejects
    the whole graph so the caller can never render a misleading partial list.

    Args:
        hydration: Authorized, all-or-nothing citation hydration result.

    Returns:
        Complete cited-source rows, or ``None`` when any required data is
        unavailable or inconsistent.
    """

    if (
        not isinstance(hydration, CitationHydrationResult)
        or hydration.state is not CitationHydrationState.AUTHORIZED
        or hydration.summary is None
        or hydration.governed_payloads is None
    ):
        return None

    trace = hydration.summary.trace
    selected_attempts = [
        attempt
        for attempt in trace.answer_attempts
        if attempt.attempt_id == trace.selected_attempt_id
    ]
    if len(selected_attempts) != 1:
        return None
    selected_attempt = selected_attempts[0]
    prompt_sets = [
        prompt_set
        for prompt_set in trace.prompt_evidence_sets
        if prompt_set.prompt_set_id == selected_attempt.prompt_evidence_set_id
    ]
    if len(prompt_sets) != 1:
        return None
    prompt_set = prompt_sets[0]

    evidence_ordinals = selected_valid_evidence_ordinals(trace)
    if not evidence_ordinals:
        return None

    entries_by_ordinal: dict[int, Any] = {}
    for entry in prompt_set.entries:
        if entry.evidence_ordinal in entries_by_ordinal:
            return None
        entries_by_ordinal[entry.evidence_ordinal] = entry

    payloads_by_id: dict[str, Any] = {}
    for payload in hydration.governed_payloads.evidence_snapshot_payloads:
        if payload.payload_id in payloads_by_id:
            return None
        payloads_by_id[payload.payload_id] = payload

    rows: list[ConsoleCitationSourceRow] = []
    for evidence_ordinal in evidence_ordinals:
        entry = entries_by_ordinal.get(evidence_ordinal)
        if entry is None:
            return None
        payload = payloads_by_id.get(entry.snapshot_payload_ref)
        if payload is None or type(payload.snapshot_text) is not str:
            return None

        display_marker = f"[S{entry.marker_ordinal}]"
        title = (
            payload.title
            if type(payload.title) is str and payload.title
            else f"Source {display_marker}"
        )
        identity = payload.source_identity
        if not isinstance(identity, dict):
            source_kind = None
            source_id = None
        else:
            source_kind = _safe_source_identifier(identity.get("source_kind"))
            source_id = _safe_source_identifier(identity.get("source_id"))
        open_source_type = (
            OPEN_SOURCE_TYPES.get(source_kind)
            if source_kind is not None and source_id is not None
            else None
        )
        rows.append(
            ConsoleCitationSourceRow(
                display_marker=display_marker,
                evidence_ordinal=evidence_ordinal,
                title=title,
                snapshot_text=payload.snapshot_text,
                source_kind=source_kind,
                source_id=source_id,
                open_source_type=open_source_type,
            )
        )
    return tuple(rows)


class ConsoleCitationSourcesModal(ModalScreen[dict[str, str] | None]):
    """Show exact cited snapshots after lazy authorized hydration."""

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(
        self,
        *,
        native_message_id: str,
        persisted_message_id: str,
        current_body: str,
        repository: Any,
        request_is_current: Callable[[], bool],
    ) -> None:
        super().__init__()
        self._native_message_id = native_message_id
        self._persisted_message_id = persisted_message_id
        self._current_body = current_body
        self._repository = repository
        self._request_is_current_callback = request_is_current
        self._request_generation = 0
        self._worker_started = False
        self.display_rows: tuple[ConsoleCitationSourceRow, ...] = ()

    def compose(self) -> ComposeResult:
        with Vertical(id="console-citation-sources-modal"):
            yield Static("Sources", classes="console-modal-header", markup=False)
            yield Static(
                "Loading sources…",
                id="console-citation-sources-state",
                classes="console-citation-sources-state",
                markup=False,
            )
            with Horizontal(id="console-citation-sources-body"):
                yield ListView(id="console-citation-source-list")
                with ScrollableContainer(id="console-citation-source-detail"):
                    yield Static(
                        Text(),
                        id="console-citation-source-marker",
                        markup=False,
                    )
                    yield Static(
                        Text(),
                        id="console-citation-source-title",
                        markup=False,
                    )
                    yield Static(
                        Text(),
                        id="console-citation-source-chunk",
                        markup=False,
                    )
                    open_button = Button(
                        "Open in Library",
                        id="console-citation-source-open",
                        disabled=True,
                    )
                    open_button.display = False
                    yield open_button
            yield Button("Close", id="console-citation-sources-close")

    def on_mount(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        self._request_generation += 1
        generation = self._request_generation
        self.run_worker(
            self._load_sources(generation),
            exclusive=True,
            group="console-citation-sources",
        )

    def on_unmount(self) -> None:
        self._request_generation += 1

    def action_dismiss(self) -> None:
        self._request_generation += 1
        self.dismiss(None)

    @on(Button.Pressed, "#console-citation-sources-close")
    def _close(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_dismiss()

    def _hydrate_rows(self) -> tuple[ConsoleCitationSourceRow, ...] | None:
        repository = self._repository
        active = repository.get_active_trace_for_current_message(
            self._persisted_message_id,
            self._current_body,
        )
        summary = getattr(active, "summary", None)
        if (
            getattr(active, "state", None) is not ActiveCitationTraceState.ACTIVE
            or summary is None
            or not repository.verify_active_trace_result(active)
        ):
            return None

        identity = repository.identity_context
        authorization = CitationReadAuthorization(
            authority_scope=AuthorityScope.LOCAL_PROFILE,
            profile_id=identity.profile_id,
            governance_scope_id=identity.profile_id,
            allowlisted_authority_ids=(identity.local_authority_id,),
            view_snapshot=True,
            view_source_identity=True,
        )
        hydration = repository.hydrate_trace(
            summary.namespace,
            authorization=authorization,
        )
        if not repository.verify_active_trace_result(active):
            return None
        return build_console_citation_source_rows(hydration)

    def _request_is_current(self, generation: int) -> bool:
        if generation != self._request_generation or not self.is_mounted:
            return False
        try:
            return bool(self._request_is_current_callback())
        except Exception:
            return False

    async def _load_sources(self, generation: int) -> None:
        try:
            rows = await asyncio.to_thread(self._hydrate_rows)
        except Exception:
            rows = None
        if not self._request_is_current(generation):
            return
        if not rows:
            await self._show_unavailable()
            return
        await self._show_rows(rows, generation)

    async def _show_unavailable(self) -> None:
        self.display_rows = ()
        source_list = self.query_one("#console-citation-source-list", ListView)
        await source_list.clear()
        self.query_one("#console-citation-sources-state", Static).update(
            "Sources unavailable"
        )
        self._update_detail(None)

    async def _show_rows(
        self,
        rows: tuple[ConsoleCitationSourceRow, ...],
        generation: int,
    ) -> None:
        self.display_rows = ()
        source_list = self.query_one("#console-citation-source-list", ListView)
        await source_list.clear()
        if not self._request_is_current(generation):
            await self._discard_rows(source_list)
            return
        items: list[ListItem] = []
        for index, row in enumerate(rows):
            label = Text()
            label.append(row.display_marker)
            label.append(" ")
            label.append(_render_literal_text(row.title))
            item = ListItem(Static(label, markup=False))
            item.citation_row_index = index
            items.append(item)
        await source_list.extend(items)
        if not self._request_is_current(generation):
            await self._discard_rows(source_list)
            return
        self.display_rows = rows
        self.query_one("#console-citation-sources-state", Static).update("")
        source_list.index = 0
        self._update_detail(rows[0])

    async def _discard_rows(self, source_list: ListView) -> None:
        """Remove any governed text mounted by a request that became stale."""

        self.display_rows = ()
        if not self.is_mounted:
            return
        await source_list.clear()
        if self.is_mounted:
            self._update_detail(None)

    def _update_detail(self, row: ConsoleCitationSourceRow | None) -> None:
        marker = self.query_one("#console-citation-source-marker", Static)
        title = self.query_one("#console-citation-source-title", Static)
        chunk = self.query_one("#console-citation-source-chunk", Static)
        open_button = self.query_one("#console-citation-source-open", Button)
        can_open = (
            row is not None
            and type(row.open_source_type) is str
            and type(row.source_id) is str
        )
        open_button.display = can_open
        open_button.disabled = not can_open
        if row is None:
            marker.update(Text())
            title.update(Text())
            chunk.update(Text())
            return
        marker.update(Text(row.display_marker))
        title.update(Text(_render_literal_text(row.title)))
        chunk.update(Text(_render_literal_text(row.snapshot_text)))

    def _show_item(self, item: ListItem | None) -> None:
        if item is None:
            return
        index = getattr(item, "citation_row_index", None)
        if type(index) is not int or not 0 <= index < len(self.display_rows):
            return
        self._update_detail(self.display_rows[index])

    @on(ListView.Highlighted, "#console-citation-source-list")
    def _source_highlighted(self, event: ListView.Highlighted) -> None:
        self._show_item(event.item)

    @on(ListView.Selected, "#console-citation-source-list")
    def _source_selected(self, event: ListView.Selected) -> None:
        self._show_item(event.item)

    @on(Button.Pressed, "#console-citation-source-open")
    def _open_source(self, event: Button.Pressed) -> None:
        """Return the selected supported source identity to Console."""

        event.stop()
        source_list = self.query_one("#console-citation-source-list", ListView)
        index = source_list.index
        if type(index) is not int or not 0 <= index < len(self.display_rows):
            return
        row = self.display_rows[index]
        if type(row.open_source_type) is not str or type(row.source_id) is not str:
            return
        self._request_generation += 1
        self.dismiss(
            {
                LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE: row.open_source_type,
                LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID: row.source_id,
            }
        )


__all__ = [
    "ConsoleCitationSourceRow",
    "ConsoleCitationSourcesModal",
    "OPEN_SOURCE_TYPES",
    "build_console_citation_source_rows",
    "selected_valid_evidence_ordinals",
]
