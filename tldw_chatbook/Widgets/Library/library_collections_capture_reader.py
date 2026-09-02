"""Render-only panes for the Collections capture reader.

Direction contract:
- Thesis: Collections is a calm capture-and-reading workbench, not a folder manager.
- Story: choose a capture scope, traverse the reading list, then read or annotate.
- Form: the established dense Library rail + Items + permanent Work topology.

The widgets in this module own pixels only.  Authority, requests, mutations, and
late-result fencing remain in ``LibraryCollectionsCaptureController``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.collections_capture_models import (
    CapabilityState,
    CaptureCapabilities,
    CaptureHighlight,
    CaptureIdentity,
    CaptureSummary,
    SavedCaptureSearch,
)
from tldw_chatbook.Library.library_shell_state import library_disabled_action_label
from tldw_chatbook.UI.Library_Modules.library_collections_capture_controller import (
    CollectionsCaptureControllerState,
)
CollectionsReaderMode = Literal["read", "highlights", "notes", "info"]
CollectionsScope = Literal[
    "all",
    "saved",
    "reading",
    "read",
    "archived",
    "favorites",
]

_BUILT_IN_SCOPES: tuple[tuple[CollectionsScope, str], ...] = (
    ("all", "All Captures"),
    ("saved", "Saved"),
    ("reading", "Reading"),
    ("read", "Read"),
    ("archived", "Archived"),
    ("favorites", "Favorites"),
)


def _widget_id(value: str) -> str:
    """Return a stable Textual-id fragment without leaking source data."""
    fragment = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return fragment[:48] or "item"


def _reason_copy(reason: str | None) -> str:
    """Convert bounded service reasons into compact visible UI copy."""
    return (reason or "availability_unknown").replace("_", " ")


def _row_title(title: str, budget: int = 120) -> str:
    """Return compact literal row text without pre-escaping ``Content``."""
    readable = " ".join(str(title).split())
    if len(readable) <= budget:
        return readable
    return f"{readable[: max(1, budget - 1)].rstrip()}…"


class LibraryCollectionsItemButton(Button):
    """Capture row button carrying its opaque authority-qualified identity."""

    def __init__(
        self,
        label: Content,
        *,
        capture_identity: CaptureIdentity,
        **kwargs: Any,
    ) -> None:
        super().__init__(label, **kwargs)
        self.capture_identity = capture_identity


class LibraryCollectionsArchiveUndoButton(Button):
    """Undo action carrying the archived capture's stable identity."""

    def __init__(
        self,
        *args: Any,
        capture_identity: CaptureIdentity,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.capture_identity = capture_identity


class LibraryCollectionsHighlightButton(Button):
    """Highlight action carrying its source-owned identifier."""

    def __init__(self, *args: Any, highlight_id: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.highlight_id = highlight_id


class LibraryCollectionsNoteLinkButton(Button):
    """Linked-Note action carrying its source-owned link identifier."""

    def __init__(self, *args: Any, link_id: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.link_id = link_id


@dataclass(frozen=True)
class CollectionsCaptureReaderPresentation:
    """Complete immutable render input for the three Collections panes."""

    state: CollectionsCaptureControllerState
    capabilities: CaptureCapabilities | None = None
    saved_searches: tuple[SavedCaptureSearch, ...] = ()
    saved_searches_total: int = 0
    active_scope: str = "all"
    authority_label: str = "Local"
    mode: CollectionsReaderMode = "read"
    highlights: tuple[CaptureHighlight, ...] = ()
    quick_capture_open: bool = False
    quick_capture_url: str = ""
    quick_capture_title: str = ""
    quick_capture_tags: str = ""
    quick_capture_note: str = ""
    save_outcome_unknown: bool = False
    confirming_save_retry: bool = False
    quick_capture_saving: bool = False
    filters_open: bool = False
    more_open: bool = False
    confirming_hard_delete: bool = False
    legacy_recovery_rows: int = 0
    legacy_recovery_open: bool = False
    legacy_recovery_lines: tuple[str, ...] = ()
    action_status: str = ""
    action_content: str = ""

    def capability(self, action: str) -> tuple[bool, str]:
        """Return enabled state and a truthful reason for one action."""
        if self.capabilities is None:
            return False, "Availability has not been checked."
        capability = self.capabilities.for_action(action)
        if capability.state is CapabilityState.SUPPORTED:
            return True, ""
        if capability.state is CapabilityState.UNKNOWN:
            return False, "Availability has not been checked."
        return False, _reason_copy(capability.reason)


class LibraryCollectionsScopeRows(Vertical):
    """Contextual capture scopes mounted directly beneath Collections."""

    def __init__(
        self,
        presentation: CollectionsCaptureReaderPresentation,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.presentation = presentation
        self.styles.height = "auto"

    def compose(self) -> ComposeResult:
        """Render bounded built-ins, saved searches, and continuation."""
        active_total = self.presentation.state.exact_total
        for key, title in _BUILT_IN_SCOPES:
            selected = self.presentation.active_scope == key
            total = f" ({active_total})" if selected and active_total is not None else ""
            yield Button(
                f"{'▸' if selected else ' '} {title}{total}",
                id=f"library-collections-scope-{key}",
                classes="library-collections-scope-row",
                compact=True,
            )
        for search in self.presentation.saved_searches:
            selected = self.presentation.active_scope == f"search:{search.search_id}"
            total = f" ({active_total})" if selected and active_total is not None else ""
            yield Button(
                Content(
                    f"{'▸' if selected else ' '} "
                    f"{_row_title(search.name)}{total}"
                ),
                id=(
                    "library-collections-saved-search-"
                    f"{_widget_id(search.search_id)}"
                ),
                classes="library-collections-scope-row library-collections-saved-search-row",
                compact=True,
            )
        if self.presentation.saved_searches_total > len(
            self.presentation.saved_searches
        ):
            yield Button(
                "More saved searches…",
                id="library-collections-more-saved-searches",
                compact=True,
            )


def _capture_row_label(
    item: CaptureSummary,
    *,
    selected: bool,
    loaded: bool,
    loading: bool,
) -> Content:
    """Build a query-free, two-line capture row label."""
    title = _row_title(item.title or item.domain or "Untitled capture")
    relationship = (
        "Selected · loading  "
        if selected and loading
        else "Loaded in Reader  "
        if loaded
        else ""
    )
    date = (item.published_at or item.created_at or item.updated_at)[:10]
    markers = [item.status.title()]
    if item.favorite:
        markers.append("Favorite")
    if item.processing_state in {"failed", "interrupted"}:
        markers.append(f"Extraction {item.processing_state}")
    secondary = " · ".join(part for part in (item.domain, date, *markers) if part)
    return Content(
        f"{'▸' if selected else ' '} {relationship}{title}\n    {secondary}"
    )


class LibraryCollectionsItemsPane(Vertical):
    """Compact capture list, scope controls, paging, and recoverable states."""

    def __init__(
        self,
        presentation: CollectionsCaptureReaderPresentation,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.presentation = presentation
        self.styles.width = "1fr"
        self.styles.min_width = 0

    def compose(self) -> ComposeResult:
        """Render the reading-list surface without exposing URL queries."""
        state = self.presentation.state
        capture_enabled, capture_reason = self.presentation.capability("capture")
        with Horizontal(classes="ds-toolbar", id="library-collections-items-toolbar"):
            yield Button(
                library_disabled_action_label("Quick Capture", not capture_enabled),
                id="library-collections-quick-capture",
                compact=True,
                disabled=not capture_enabled,
                tooltip=capture_reason or "Save a URL to this reading list.",
            )
            yield Button(
                "Filters",
                id="library-collections-filters",
                compact=True,
            )
        with Horizontal(classes="ds-toolbar", id="library-collections-sort-toolbar"):
            sort = state.requested_scope.sort if state.requested_scope else "saved_desc"
            yield Button(
                f"Sort: {sort.replace('_', ' ')}",
                id="library-collections-sort",
                compact=True,
            )
        if self.presentation.quick_capture_open:
            with Vertical(id="library-collections-quick-capture-form"):
                yield Input(
                    value=self.presentation.quick_capture_url,
                    placeholder="https://example.com/article",
                    id="library-collections-capture-url",
                )
                yield Input(
                    value=self.presentation.quick_capture_title,
                    placeholder="Title (optional)",
                    id="library-collections-capture-title",
                )
                yield Input(
                    value=self.presentation.quick_capture_tags,
                    placeholder="Tags, comma separated (optional)",
                    id="library-collections-capture-tags",
                )
                yield TextArea(
                    self.presentation.quick_capture_note,
                    id="library-collections-capture-note",
                )
                if self.presentation.save_outcome_unknown:
                    yield Static(
                        "Save outcome unknown. Refresh before retrying.",
                        id="library-collections-capture-unknown",
                        markup=False,
                    )
                    yield Button(
                        "Refresh capture list",
                        id="library-collections-capture-refresh",
                        compact=True,
                        disabled=self.presentation.quick_capture_saving,
                    )
                if self.presentation.confirming_save_retry:
                    yield Static(
                        "Retrying against the current Server may reapply Saved status "
                        "and clear Favorite on an existing canonical URL.",
                        id="library-collections-capture-retry-warning",
                        markup=False,
                    )
                with Horizontal(classes="ds-toolbar"):
                    yield Button(
                        (
                            "Saving…"
                            if self.presentation.quick_capture_saving
                            else "Retry anyway"
                            if self.presentation.confirming_save_retry
                            else "Retry save…"
                            if self.presentation.save_outcome_unknown
                            else "Save capture"
                        ),
                        id=(
                            "library-collections-capture-retry-confirm"
                            if self.presentation.confirming_save_retry
                            else "library-collections-capture-save"
                        ),
                        compact=True,
                        disabled=self.presentation.quick_capture_saving,
                    )
                    yield Button(
                        "Back" if self.presentation.confirming_save_retry else "Cancel",
                        id=(
                            "library-collections-capture-retry-back"
                            if self.presentation.confirming_save_retry
                            else "library-collections-capture-cancel"
                        ),
                        compact=True,
                        disabled=self.presentation.quick_capture_saving,
                    )
        if self.presentation.filters_open:
            request = state.requested_scope
            with Vertical(id="library-collections-filters-form"):
                yield Input(
                    value=(request.domain or "") if request is not None else "",
                    placeholder="Domain",
                    id="library-collections-filter-domain",
                )
                yield Input(
                    value=", ".join(request.tags) if request is not None else "",
                    placeholder="Tags, comma separated",
                    id="library-collections-filter-tags",
                )
                yield Input(
                    value=(request.date_from or "") if request is not None else "",
                    placeholder="From date (YYYY-MM-DD)",
                    id="library-collections-filter-date-from",
                )
                yield Input(
                    value=(request.date_to or "") if request is not None else "",
                    placeholder="To date (YYYY-MM-DD)",
                    id="library-collections-filter-date-to",
                )
                with Horizontal(classes="ds-toolbar"):
                    yield Button(
                        "Apply filters",
                        id="library-collections-filters-apply",
                        compact=True,
                    )
                    yield Button(
                        "Clear",
                        id="library-collections-filters-clear",
                        compact=True,
                    )
        yield Input(
            value=state.requested_scope.search if state.requested_scope else "",
            placeholder="Filter captures",
            id="library-collections-filter",
        )

        if state.page_stale:
            yield Static(
                "Showing the last good page. Refresh failed; totals and page actions are paused.",
                id="library-collections-page-stale",
                classes="destination-purpose",
                markup=False,
            )
            yield Button("Retry", id="library-collections-page-retry", compact=True)
        elif state.page_error:
            yield Static(
                f"Captures could not be loaded: {_reason_copy(state.page_error)}.",
                id="library-collections-page-error",
                classes="destination-purpose",
                markup=False,
            )
            yield Button("Retry", id="library-collections-page-retry", compact=True)
        elif state.page_loading and state.page is None:
            yield Static(
                "Loading captures…",
                id="library-collections-page-loading",
                classes="destination-purpose",
                markup=False,
            )

        page = state.page
        if page is None or not page.items:
            yield Static(
                "No captures match this scope. Clear filters or save a URL with Quick Capture.",
                id="library-collections-items-empty",
                classes="destination-purpose",
                markup=False,
            )
        else:
            loaded_identity = (
                state.loaded_detail.capture.identity
                if state.loaded_detail is not None
                else None
            )
            rows = VerticalScroll(id="library-collections-items-scroll")
            rows.styles.height = "1fr"
            with rows:
                for index, item in enumerate(page.items):
                    selected = item.identity == state.selected_identity
                    loaded = item.identity == loaded_identity
                    button = LibraryCollectionsItemButton(
                        _capture_row_label(
                            item,
                            selected=selected,
                            loaded=loaded,
                            loading=selected and state.detail_loading,
                        ),
                        id=f"library-collections-row-{index}",
                        classes="library-collections-item-row",
                        compact=True,
                        capture_identity=item.identity,
                    )
                    button.styles.height = 2
                    button.styles.min_height = 2
                    yield button

        current_page = (
            page.applied.page
            if page is not None
            else state.requested_scope.page
            if state.requested_scope is not None
            else 1
        )
        if state.exact_total is None:
            range_copy = f"Page {current_page} · total unavailable"
        else:
            start = 0 if state.exact_total == 0 else (current_page - 1) * 20 + 1
            stop = min(current_page * 20, state.exact_total)
            range_copy = f"{start}–{stop} of {state.exact_total}"
        yield Static(
            range_copy,
            id="library-collections-page-range",
            markup=False,
        )
        has_previous = state.paging_enabled and current_page > 1
        has_next = (
            state.paging_enabled
            and state.exact_total is not None
            and current_page * 20 < state.exact_total
        )
        with Horizontal(classes="ds-toolbar", id="library-collections-page-toolbar"):
            yield Button(
                "Previous",
                id="library-collections-page-previous",
                compact=True,
                disabled=not has_previous,
                tooltip=(
                    "Load the previous page."
                    if has_previous
                    else "No current previous page is available."
                ),
            )
            yield Button(
                "Next",
                id="library-collections-page-next",
                compact=True,
                disabled=not has_next,
                tooltip=(
                    "Load the next page."
                    if has_next
                    else "No current next page is available."
                ),
            )


def _action_button(
    presentation: CollectionsCaptureReaderPresentation,
    action: str,
    label: str,
    *,
    identity_required: bool = True,
    button_id: str,
) -> Button:
    """Build a capability- and identity-gated action with visible reason."""
    supported, reason = presentation.capability(action)
    if (
        supported
        and identity_required
        and not presentation.state.identity_actions_enabled
    ):
        supported = False
        reason = "Wait until the selected capture is loaded and current."
    return Button(
        library_disabled_action_label(label, not supported),
        id=button_id,
        compact=True,
        disabled=not supported,
        tooltip=reason or label,
    )


def _open_original_button(
    presentation: CollectionsCaptureReaderPresentation,
) -> Button:
    """Build the identity-gated Open Original action in either toolbar."""
    enabled = presentation.state.identity_actions_enabled
    return Button(
        "Open Original",
        id="library-collections-open-original",
        compact=True,
        disabled=not enabled,
        tooltip=(
            "Open the capture's original URL."
            if enabled
            else "Wait until the selected capture is loaded and current."
        ),
    )


class LibraryCollectionsWorkPane(VerticalScroll):
    """Permanent reading-first Work region for one loaded capture."""

    def __init__(
        self,
        presentation: CollectionsCaptureReaderPresentation,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.presentation = presentation
        self.styles.width = "1fr"
        self.styles.min_width = 0

    def compose(self) -> ComposeResult:
        """Render truthful identity, actions, one active mode, and recovery."""
        state = self.presentation.state
        retained_copy = state.retained_reader_copy
        if retained_copy:
            yield Static(
                retained_copy,
                id="library-collections-reader-loading",
                classes="destination-purpose",
                markup=False,
            )
        if state.detail_error:
            yield Static(
                f"Capture could not be loaded: {_reason_copy(state.detail_error)}.",
                id="library-collections-reader-error",
                classes="destination-purpose",
                markup=False,
            )
            yield Button("Retry", id="library-collections-reader-retry", compact=True)
        if self.presentation.action_status:
            yield Static(
                self.presentation.action_status,
                id="library-collections-action-status",
                classes="destination-purpose",
                markup=False,
            )
        if self.presentation.action_content:
            yield Static(
                self.presentation.action_content,
                id="library-collections-action-content",
                markup=False,
            )
        if state.visible_archive_receipts:
            receipt = state.visible_archive_receipts[0]
            with Horizontal(
                classes="ds-toolbar",
                id="library-collections-archive-receipt",
            ):
                yield Static(
                    f"Moved to Archive · was {receipt.previous_status.title()}.",
                    markup=False,
                )
                yield LibraryCollectionsArchiveUndoButton(
                    "Undo",
                    capture_identity=receipt.identity,
                    id="library-collections-archive-undo",
                    compact=True,
                )

        resolved = state.loaded_detail
        if resolved is None:
            yield Static(
                "Select a capture to read it here.",
                id="library-collections-reader-empty",
                classes="destination-purpose",
                markup=False,
            )
            if self.presentation.legacy_recovery_rows:
                supported, reason = self.presentation.capability("legacy_recovery")
                yield Button(
                    f"Legacy Collections data… ({self.presentation.legacy_recovery_rows})",
                    id="library-collections-legacy-recovery",
                    compact=True,
                    disabled=not supported,
                    tooltip=reason or "Inspect and export preserved legacy data.",
                )
            if self.presentation.legacy_recovery_open:
                yield from self._compose_legacy_recovery()
            return
        capture = resolved.capture
        yield Static(
            f"{self.presentation.authority_label} Collections · {capture.domain or 'Saved capture'}",
            id="library-collections-reader-identity",
            markup=False,
        )
        yield Static(
            capture.title or capture.domain or "Untitled capture",
            id="library-collections-reader-title",
            markup=False,
        )
        byline_parts = [capture.byline or capture.published_at or ""]
        if capture.word_count is not None:
            minutes = max(1, round(capture.word_count / 220))
            byline_parts.append(f"{minutes} min read")
        byline_parts.extend((capture.status.title(), self.presentation.authority_label))
        yield Static(
            " · ".join(part for part in byline_parts if part),
            id="library-collections-reader-byline",
            markup=False,
        )

        with Horizontal(classes="ds-toolbar", id="library-collections-primary-toolbar"):
            yield _action_button(
                self.presentation,
                "update",
                "Mark Read",
                button_id="library-collections-mark-read",
            )
            yield _action_button(
                self.presentation,
                "update",
                "Favorite",
                button_id="library-collections-favorite",
            )
            yield _action_button(
                self.presentation,
                "archive",
                "Move to Archive",
                button_id="library-collections-archive",
            )
        with Horizontal(
            classes="ds-toolbar",
            id="library-collections-secondary-toolbar",
        ):
            yield _open_original_button(self.presentation)
            yield Button(
                "More",
                id="library-collections-more",
                compact=True,
            )

        with Horizontal(classes="ds-toolbar", id="library-collections-mode-toolbar"):
            for mode in ("read", "highlights", "notes", "info"):
                label = f"✓ {mode.title()}" if self.presentation.mode == mode else mode.title()
                yield Button(
                    label,
                    id=f"library-collections-mode-{mode}",
                    compact=True,
                )

        if self.presentation.more_open:
            yield from self._compose_more()
        if self.presentation.confirming_hard_delete:
            yield Static(
                f"Permanently delete “{capture.title or capture.domain or 'this capture'}”, "
                "its highlights, and its managed offline copy? This cannot be undone.",
                id="library-collections-hard-delete-copy",
                markup=False,
            )
            with Horizontal(classes="ds-toolbar"):
                yield _action_button(
                    self.presentation,
                    "hard_delete",
                    "Delete permanently",
                    button_id="library-collections-hard-delete-confirm",
                )
                yield Button(
                    "Cancel",
                    id="library-collections-hard-delete-cancel",
                    compact=True,
                )

        if self.presentation.mode == "read":
            yield Static(
                capture.text_content or "No readable content is stored for this capture.",
                id="library-collections-read-body",
                markup=False,
            )
        elif self.presentation.mode == "highlights":
            yield from self._compose_highlights()
        elif self.presentation.mode == "notes":
            yield from self._compose_notes()
        else:
            yield from self._compose_info()

    def _compose_more(self) -> ComposeResult:
        """Render lower-frequency actions without hiding capability reasons."""
        yield _action_button(
            self.presentation,
            "summarize",
            "Summarize",
            button_id="library-collections-summarize",
        )
        yield _action_button(
            self.presentation,
            "listen",
            "Listen",
            button_id="library-collections-listen",
        )
        yield _action_button(
            self.presentation,
            "offline_copy",
            "Save Offline Copy",
            button_id="library-collections-save-offline",
        )
        yield _action_button(
            self.presentation,
            "retry_extraction",
            "Retry Extraction",
            button_id="library-collections-retry-extraction",
        )
        yield _action_button(
            self.presentation,
            "hard_delete",
            "Delete Permanently…",
            button_id="library-collections-hard-delete",
        )
        legacy_supported, legacy_reason = self.presentation.capability("legacy_recovery")
        if self.presentation.legacy_recovery_rows:
            yield Button(
                f"Legacy Collections data… ({self.presentation.legacy_recovery_rows})",
                id="library-collections-legacy-recovery",
                compact=True,
                disabled=not legacy_supported,
                tooltip=legacy_reason or "Inspect and export preserved legacy data.",
            )
        if self.presentation.legacy_recovery_open:
            yield from self._compose_legacy_recovery()

    def _compose_legacy_recovery(self) -> ComposeResult:
        """Render the bounded inspector and complete-export action."""
        yield Static(
            "Legacy Collections · read-only recovery",
            id="library-collections-legacy-recovery-heading",
            markup=False,
        )
        yield Static(
            "\n".join(self.presentation.legacy_recovery_lines)
            or "No legacy records are available.",
            id="library-collections-legacy-recovery-content",
            markup=False,
        )
        with Horizontal(classes="ds-toolbar"):
            yield Button(
                "Export complete JSON…",
                id="library-collections-legacy-recovery-export",
                compact=True,
            )
            yield Button(
                "Close inspector",
                id="library-collections-legacy-recovery-close",
                compact=True,
            )

    def _compose_highlights(self) -> ComposeResult:
        """Render active and detached capture-owned highlights."""
        supported, reason = self.presentation.capability("highlights")
        enabled = supported and self.presentation.state.identity_actions_enabled
        yield TextArea(
            "",
            id="library-collections-highlight-quote",
            disabled=not enabled,
            tooltip=reason or "Quote to keep with this capture.",
        )
        yield Input(
            placeholder="Highlight note (optional)",
            id="library-collections-highlight-note",
            disabled=not enabled,
        )
        yield Button(
            library_disabled_action_label("Add highlight", not enabled),
            id="library-collections-highlight-save",
            compact=True,
            disabled=not enabled,
            tooltip=reason or "Save this highlight.",
        )
        if not self.presentation.highlights:
            yield Static(
                "No highlights for this capture.",
                id="library-collections-highlights-empty",
                markup=False,
            )
            return
        for highlight in self.presentation.highlights:
            state = "Detached · reattach needed" if highlight.detached else "Active"
            yield Static(
                f"{state}\n{highlight.quote}"
                + (f"\nNote: {highlight.note}" if highlight.note else ""),
                id=f"library-collections-highlight-{_widget_id(highlight.highlight_id)}",
                markup=False,
            )
            yield LibraryCollectionsHighlightButton(
                "Delete highlight",
                id=f"library-collections-highlight-delete-{_widget_id(highlight.highlight_id)}",
                classes="library-collections-highlight-delete",
                compact=True,
                disabled=not enabled,
                highlight_id=highlight.highlight_id,
            )

    def _compose_notes(self) -> ComposeResult:
        """Keep the capture note and Linked Notes visibly distinct."""
        resolved = self.presentation.state.loaded_detail
        assert resolved is not None
        linked_enabled, linked_reason = self.presentation.capability("linked_notes")
        linked_enabled = (
            linked_enabled and self.presentation.state.identity_actions_enabled
        )
        if not self.presentation.state.identity_actions_enabled:
            linked_reason = "Wait until the selected capture is loaded and current."
        yield Static("Capture note", id="library-collections-freeform-note-heading", markup=False)
        yield TextArea(
            resolved.capture.freeform_note or "",
            id="library-collections-freeform-note",
            disabled=(
                not self.presentation.capability("update")[0]
                or not self.presentation.state.identity_actions_enabled
            ),
        )
        yield _action_button(
            self.presentation,
            "update",
            "Save capture note",
            button_id="library-collections-freeform-note-save",
        )
        yield Static(
            "Linked Notes",
            id="library-collections-linked-notes-heading",
            markup=False,
        )
        if not resolved.note_links:
            yield Static(
                "No Notes are linked to this capture.",
                id="library-collections-linked-notes-empty",
                markup=False,
            )
        for link, availability in resolved.note_links:
            status = (
                "Available"
                if availability.state == "available"
                else f"Unavailable: {_reason_copy(availability.reason)}"
            )
            yield Static(
                f"Note {link.note_reference.note_id} · {status}",
                id=f"library-collections-linked-note-{_widget_id(link.link_id)}",
                markup=False,
            )
            yield LibraryCollectionsNoteLinkButton(
                "Unlink",
                id=f"library-collections-linked-note-unlink-{_widget_id(link.link_id)}",
                classes="library-collections-linked-note-unlink",
                compact=True,
                disabled=not linked_enabled,
                link_id=link.link_id,
            )
        yield Input(
            placeholder="Note ID",
            id="library-collections-linked-note-id",
            disabled=not linked_enabled,
            tooltip=linked_reason or "Link a Note by its exact ID.",
        )
        yield Button(
            library_disabled_action_label("Link Note", not linked_enabled),
            id="library-collections-linked-note-save",
            compact=True,
            disabled=not linked_enabled,
            tooltip=linked_reason or "Link this capture to the Note.",
        )

    def _compose_info(self) -> ComposeResult:
        """Render capture metadata and external-reference provenance."""
        resolved = self.presentation.state.loaded_detail
        assert resolved is not None
        capture = resolved.capture
        yield Static(
            "\n".join(
                (
                    f"Canonical URL: {capture.canonical_url}",
                    f"Submitted URL: {capture.submitted_url}",
                    f"Tags: {', '.join(capture.tags) or 'None'}",
                    f"Status: {capture.status}",
                    f"Extraction: {capture.processing_state}",
                    f"Words: {capture.word_count if capture.word_count is not None else 'Unknown'}",
                    f"Authority: {self.presentation.authority_label}",
                )
            ),
            id="library-collections-info-body",
            markup=False,
        )
        if capture.media_reference is not None:
            availability = resolved.media
            status = (
                "Available"
                if availability is not None and availability.state == "available"
                else f"Unavailable: {_reason_copy(availability.reason if availability else None)}"
            )
            yield Static(
                f"Backing Media {capture.media_reference.item_id} · {status}",
                id="library-collections-media-provenance",
                markup=False,
            )


__all__ = [
    "CollectionsCaptureReaderPresentation",
    "CollectionsReaderMode",
    "LibraryCollectionsHighlightButton",
    "LibraryCollectionsItemsPane",
    "LibraryCollectionsScopeRows",
    "LibraryCollectionsNoteLinkButton",
    "LibraryCollectionsWorkPane",
]
