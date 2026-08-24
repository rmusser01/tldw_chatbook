"""Compose-once Quick Notes workbench for Research Studio."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Markdown, Select, Static, TextArea

from ...Research_Workspace import (
    QualifiedWorkspaceRef,
    ResearchCapability,
    ResearchNotePage,
    ResearchNoteSaveRequest,
    ResearchQuickNote,
)


@dataclass(frozen=True, slots=True)
class _EditorSnapshot:
    title: str
    content: str
    tags: tuple[str, ...]
    message_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


class ResearchQuickNotesSection(Vertical):
    """A bounded canonical Notes editor subordinate to Studio."""

    class SearchRequested(Message):
        def __init__(self, ref: QualifiedWorkspaceRef, query: str) -> None:
            super().__init__()
            self.ref = ref
            self.query = query

    class PageRequested(Message):
        def __init__(self, ref: QualifiedWorkspaceRef, delta: int) -> None:
            super().__init__()
            self.ref = ref
            self.delta = delta

    class LoadRequested(Message):
        def __init__(self, ref: QualifiedWorkspaceRef, note_id: str) -> None:
            super().__init__()
            self.ref = ref
            self.note_id = note_id

    class NewRequested(Message):
        def __init__(self, ref: QualifiedWorkspaceRef) -> None:
            super().__init__()
            self.ref = ref

    class SaveRequested(Message):
        def __init__(
            self, ref: QualifiedWorkspaceRef, request: ResearchNoteSaveRequest
        ) -> None:
            super().__init__()
            self.ref = ref
            self.request = request

    class DeleteRequested(Message):
        def __init__(
            self,
            ref: QualifiedWorkspaceRef,
            note_id: str,
            expected_version: int,
        ) -> None:
            super().__init__()
            self.ref = ref
            self.note_id = note_id
            self.expected_version = expected_version

    class DownloadRequested(Message):
        def __init__(self, title: str, content: str, tags: tuple[str, ...]) -> None:
            super().__init__()
            self.title = title
            self.content = content
            self.tags = tags

    class CaptureSourcesRequested(Message):
        def __init__(self, ref: QualifiedWorkspaceRef) -> None:
            super().__init__()
            self.ref = ref

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.editor_ref: QualifiedWorkspaceRef | None = None
        self._note_id: str | None = None
        self._version: int | None = None
        self._message_ids: tuple[str, ...] = ()
        self._source_ids: tuple[str, ...] = ()
        self._baseline = _EditorSnapshot("", "", (), (), ())
        self._undo: _EditorSnapshot | None = None
        self._page_offset = 0
        self._page_limit = 20
        self._page_has_more = False
        self._selected_note_id = ""
        self._list_available = False
        self._get_available = False
        self._save_available = False
        self._delete_available = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="research-quick-note-heading-row"):
            yield Static("Quick Notes", id="research-quick-notes-heading")
            yield Static("0 notes", id="research-quick-note-count", markup=False)
            yield Button("Load", id="research-quick-note-load", disabled=True)
            yield Button("New", id="research-quick-note-new", disabled=True)
        with Horizontal(id="research-quick-note-search-row"):
            yield Input(
                placeholder="Search workspace notes",
                id="research-quick-note-search",
                disabled=True,
            )
            yield Button(
                "Search", id="research-quick-note-search-submit", disabled=True
            )
        yield Select(
            (),
            prompt="Choose a workspace note",
            allow_blank=True,
            id="research-quick-note-list",
        )
        with Horizontal(id="research-quick-note-page-row"):
            yield Button("Previous", id="research-quick-note-prev", disabled=True)
            yield Static("Page 1 · no notes", id="research-quick-note-page")
            yield Button("Next", id="research-quick-note-next", disabled=True)
        yield Input(placeholder="Title", id="research-quick-note-title")
        yield Input(placeholder="Tags (comma-separated)", id="research-quick-note-tags")
        with Horizontal(id="research-quick-note-provenance-row"):
            yield Button(
                "Capture selected sources", id="research-quick-note-capture-sources"
            )
            yield Button(
                "Capture message",
                id="research-quick-note-capture-message",
                disabled=True,
                tooltip=(
                    "Message capture becomes available when grounded Chat exposes "
                    "a canonical message owner."
                ),
            )
        yield Static(
            "No captured provenance",
            id="research-quick-note-provenance",
            markup=False,
        )
        with Horizontal(id="research-quick-note-mode-row"):
            yield Button(
                "Edit", id="research-quick-note-edit-mode", classes="is-active"
            )
            yield Button("Markdown Preview", id="research-quick-note-preview-mode")
        yield TextArea(id="research-quick-note-body")
        preview = Markdown("", id="research-quick-note-preview")
        preview.display = False
        yield preview
        with Horizontal(id="research-quick-note-actions"):
            yield Button(
                "Save",
                id="research-quick-note-save",
                variant="primary",
                disabled=True,
            )
            yield Button("Delete", id="research-quick-note-delete", disabled=True)
            yield Button("Download .md", id="research-quick-note-download")
            yield Button("Clear", id="research-quick-note-clear")
            yield Button("Undo", id="research-quick-note-undo", disabled=True)
        yield Static(
            "Select a workspace to use Quick Notes.",
            id="research-quick-note-status",
            markup=False,
        )
        yield Static(
            "",
            id="research-quick-note-owner-limits",
            markup=False,
        )

    @property
    def is_dirty(self) -> bool:
        return self._snapshot() != self._baseline

    @property
    def has_nonempty_dirty_draft(self) -> bool:
        snapshot = self._snapshot()
        return self.is_dirty and bool(
            self._note_id is not None
            or snapshot.title
            or snapshot.content
            or snapshot.tags
            or snapshot.message_ids
            or snapshot.source_ids
        )

    def sync_workspace(self, ref: QualifiedWorkspaceRef | None) -> None:
        """Reset editor identity when its exact canonical owner changes."""

        if ref == self.editor_ref:
            return
        self.editor_ref = ref
        self._selected_note_id = ""
        self._page_offset = 0
        self._page_has_more = False
        self._list_available = False
        self._get_available = False
        self._save_available = False
        self._delete_available = False
        self._reset_editor()
        selector = self.query_one("#research-quick-note-list", Select)
        with selector.prevent(Select.Changed):
            selector.set_options(())
            selector.value = Select.NULL
        self.query_one("#research-quick-note-load", Button).disabled = True
        self.query_one("#research-quick-note-count", Static).update("0 notes")
        self.query_one("#research-quick-note-page", Static).update("Page 1 · no notes")
        self.query_one("#research-quick-note-prev", Button).disabled = True
        self.query_one("#research-quick-note-next", Button).disabled = True
        self._apply_capability_states()
        self.query_one("#research-quick-note-status", Static).update(
            "New note · Unsaved"
            if ref is not None
            else "Select a workspace to use Quick Notes."
        )

    def sync_page(self, page: ResearchNotePage) -> None:
        self._page_offset = page.offset
        self._page_limit = page.limit
        self._page_has_more = page.has_more
        options = tuple((note.title or "Untitled", note.note_id) for note in page.items)
        selector = self.query_one("#research-quick-note-list", Select)
        with selector.prevent(Select.Changed):
            selector.set_options(options)
            selector.value = (
                self._selected_note_id
                if self._selected_note_id
                and any(value == self._selected_note_id for _, value in options)
                else Select.NULL
            )
        count = "?" if page.total is None else str(page.total)
        self.query_one("#research-quick-note-count", Static).update(f"{count} notes")
        page_number = page.offset // page.limit + 1
        self.query_one("#research-quick-note-page", Static).update(
            f"Page {page_number} · {len(page.items)} shown"
        )
        self._apply_capability_states()

    def sync_note(self, note: ResearchQuickNote) -> None:
        self.editor_ref = note.ref
        self._note_id = note.note_id
        self._selected_note_id = note.note_id
        self._version = note.version
        self._message_ids = note.message_ids
        self._source_ids = note.source_ids
        self._set_fields(note.title, note.content, note.tags)
        self._baseline = self._snapshot()
        self._undo = None
        self._sync_editor_state(f"Saved · version {note.version}")

    def new_draft(self) -> None:
        self._reset_editor()
        self._sync_editor_state("New note · Unsaved")
        self.query_one("#research-quick-note-title", Input).focus()

    def sync_capabilities(self, capabilities: Mapping[str, ResearchCapability]) -> None:
        limitations: list[str] = []
        attributes = {
            "list_notes": "_list_available",
            "get_note": "_get_available",
            "save_note": "_save_available",
            "delete_note": "_delete_available",
        }
        tooltips: dict[str, str | None] = {}
        for capability_name, attribute in attributes.items():
            capability = capabilities.get(capability_name)
            if capability is None:
                continue
            unavailable = not capability.available
            setattr(self, attribute, not unavailable)
            if unavailable:
                reason = " ".join(
                    part
                    for part in (
                        capability.user_message,
                        capability.recovery_action,
                    )
                    if part
                )
                tooltips[capability_name] = reason
                limitations.append(reason)
            else:
                tooltips[capability_name] = None
        controls = {
            "list_notes": (
                "research-quick-note-search",
                "research-quick-note-search-submit",
                "research-quick-note-prev",
                "research-quick-note-next",
            ),
            "get_note": ("research-quick-note-load",),
            "save_note": (
                "research-quick-note-new",
                "research-quick-note-save",
            ),
            "delete_note": ("research-quick-note-delete",),
        }
        for capability_name, reason in tooltips.items():
            for control_id in controls[capability_name]:
                self.query_one(f"#{control_id}").tooltip = reason
        self._apply_capability_states()
        self.query_one("#research-quick-note-owner-limits", Static).update(
            " ".join(dict.fromkeys(limitations))
        )

    def _apply_capability_states(self) -> None:
        self.query_one(
            "#research-quick-note-search", Input
        ).disabled = not self._list_available
        self.query_one(
            "#research-quick-note-search-submit", Button
        ).disabled = not self._list_available
        self.query_one("#research-quick-note-prev", Button).disabled = (
            not self._list_available or self._page_offset == 0
        )
        self.query_one("#research-quick-note-next", Button).disabled = (
            not self._list_available or not self._page_has_more
        )
        self.query_one("#research-quick-note-load", Button).disabled = (
            not self._get_available or not self._selected_note_id
        )
        self.query_one(
            "#research-quick-note-new", Button
        ).disabled = not self._save_available
        self.query_one(
            "#research-quick-note-save", Button
        ).disabled = not self._save_available
        self.query_one("#research-quick-note-delete", Button).disabled = (
            not self._delete_available or self._note_id is None
        )

    def set_source_provenance(self, source_ids: tuple[str, ...]) -> None:
        self._source_ids = tuple(dict.fromkeys(source_ids))[:100]
        self._sync_dirty_status()

    def set_message_provenance(self, message_ids: tuple[str, ...]) -> None:
        self._message_ids = tuple(dict.fromkeys(message_ids))[:20]
        self._sync_dirty_status()

    def capture_save_request(
        self,
    ) -> tuple[QualifiedWorkspaceRef, ResearchNoteSaveRequest]:
        if self.editor_ref is None:
            raise ValueError("Select a Research workspace first.")
        snapshot = self._snapshot()
        return self.editor_ref, ResearchNoteSaveRequest(
            note_id=self._note_id,
            title=snapshot.title,
            content=snapshot.content,
            tags=snapshot.tags,
            expected_version=self._version if self._note_id is not None else None,
            message_ids=snapshot.message_ids,
            source_ids=snapshot.source_ids,
        )

    def discard_for_switch(self) -> None:
        """Drop the mounted draft only after the user explicitly allows switching."""

        self._baseline = self._snapshot()

    def show_recovery(self, message: str) -> None:
        self.query_one("#research-quick-note-status", Static).update(message)

    def _snapshot(self) -> _EditorSnapshot:
        if not self.is_mounted:
            return self._baseline
        tags = tuple(
            dict.fromkeys(
                part.strip()
                for part in self.query_one(
                    "#research-quick-note-tags", Input
                ).value.split(",")
                if part.strip()
            )
        )
        return _EditorSnapshot(
            self.query_one("#research-quick-note-title", Input).value.strip(),
            self.query_one("#research-quick-note-body", TextArea).text,
            tags,
            self._message_ids,
            self._source_ids,
        )

    def _set_fields(self, title: str, content: str, tags: tuple[str, ...]) -> None:
        title_input = self.query_one("#research-quick-note-title", Input)
        tags_input = self.query_one("#research-quick-note-tags", Input)
        body = self.query_one("#research-quick-note-body", TextArea)
        with (
            title_input.prevent(Input.Changed),
            tags_input.prevent(Input.Changed),
            body.prevent(TextArea.Changed),
        ):
            title_input.value = title
            tags_input.value = ", ".join(tags)
            body.load_text(content)
        self.query_one("#research-quick-note-preview", Markdown).update(content)
        self._sync_provenance()

    def _reset_editor(self) -> None:
        self._note_id = None
        self._version = None
        self._message_ids = ()
        self._source_ids = ()
        if self.is_mounted:
            self._set_fields("", "", ())
            self.query_one("#research-quick-note-delete", Button).disabled = True
        self._baseline = _EditorSnapshot("", "", (), (), ())
        self._undo = None
        if self.is_mounted:
            self.query_one("#research-quick-note-undo", Button).disabled = True

    def _sync_provenance(self) -> None:
        parts = []
        if self._message_ids:
            parts.append(f"{len(self._message_ids)} message(s)")
        if self._source_ids:
            parts.append(f"{len(self._source_ids)} source(s)")
        self.query_one("#research-quick-note-provenance", Static).update(
            "Captured: " + " · ".join(parts) if parts else "No captured provenance"
        )

    def _sync_dirty_status(self) -> None:
        self._sync_provenance()
        self.query_one("#research-quick-note-status", Static).update(
            "Unsaved changes" if self.is_dirty else "Saved"
        )

    def _sync_editor_state(self, status: str) -> None:
        self.query_one("#research-quick-note-status", Static).update(status)
        self.query_one("#research-quick-note-undo", Button).disabled = (
            self._undo is None
        )
        self._apply_capability_states()
        self._sync_provenance()

    @on(Input.Changed, "#research-quick-note-title")
    @on(Input.Changed, "#research-quick-note-tags")
    @on(TextArea.Changed, "#research-quick-note-body")
    def editor_changed(self) -> None:
        self._sync_dirty_status()

    @on(Select.Changed, "#research-quick-note-list")
    def select_note(self, event: Select.Changed) -> None:
        self._selected_note_id = "" if event.value is Select.NULL else str(event.value)
        self._apply_capability_states()

    @on(Button.Pressed, "#research-quick-note-load")
    def request_load(self) -> None:
        if (
            self._get_available
            and self.editor_ref is not None
            and self._selected_note_id
        ):
            self.post_message(
                self.LoadRequested(self.editor_ref, self._selected_note_id)
            )

    @on(Button.Pressed, "#research-quick-note-new")
    def create_new(self) -> None:
        if self._save_available and self.editor_ref is not None:
            self.post_message(self.NewRequested(self.editor_ref))

    @on(Button.Pressed, "#research-quick-note-search-submit")
    def request_search(self) -> None:
        if not self._list_available or self.editor_ref is None:
            return
        self.post_message(
            self.SearchRequested(
                self.editor_ref,
                self.query_one("#research-quick-note-search", Input).value.strip(),
            )
        )

    @on(Input.Submitted, "#research-quick-note-search")
    def submit_search(self) -> None:
        self.request_search()

    @on(Button.Pressed, "#research-quick-note-prev")
    def previous_page(self) -> None:
        if (
            self._list_available
            and self.editor_ref is not None
            and self._page_offset > 0
        ):
            self.post_message(self.PageRequested(self.editor_ref, -1))

    @on(Button.Pressed, "#research-quick-note-next")
    def next_page(self) -> None:
        if self._list_available and self.editor_ref is not None and self._page_has_more:
            self.post_message(self.PageRequested(self.editor_ref, 1))

    @on(Button.Pressed, "#research-quick-note-edit-mode")
    def show_edit(self) -> None:
        self.query_one("#research-quick-note-body", TextArea).display = True
        self.query_one("#research-quick-note-preview", Markdown).display = False
        self.query_one("#research-quick-note-edit-mode", Button).set_class(
            True, "is-active"
        )
        self.query_one("#research-quick-note-preview-mode", Button).set_class(
            False, "is-active"
        )

    @on(Button.Pressed, "#research-quick-note-preview-mode")
    def show_preview(self) -> None:
        self.query_one("#research-quick-note-preview", Markdown).update(
            self.query_one("#research-quick-note-body", TextArea).text
        )
        self.query_one("#research-quick-note-body", TextArea).display = False
        self.query_one("#research-quick-note-preview", Markdown).display = True
        self.query_one("#research-quick-note-edit-mode", Button).set_class(
            False, "is-active"
        )
        self.query_one("#research-quick-note-preview-mode", Button).set_class(
            True, "is-active"
        )

    @on(Button.Pressed, "#research-quick-note-save")
    def request_save(self) -> None:
        if not self._save_available:
            return
        try:
            ref, request = self.capture_save_request()
        except (TypeError, ValueError) as exc:
            self.show_recovery(str(exc))
            return
        self.post_message(self.SaveRequested(ref, request))

    @on(Button.Pressed, "#research-quick-note-delete")
    def request_delete(self) -> None:
        if (
            self._delete_available
            and self.editor_ref is not None
            and self._note_id is not None
            and self._version is not None
        ):
            self.post_message(
                self.DeleteRequested(self.editor_ref, self._note_id, self._version)
            )

    @on(Button.Pressed, "#research-quick-note-download")
    def request_download(self) -> None:
        snapshot = self._snapshot()
        self.post_message(
            self.DownloadRequested(snapshot.title, snapshot.content, snapshot.tags)
        )

    @on(Button.Pressed, "#research-quick-note-clear")
    def clear_editor(self) -> None:
        self._undo = self._snapshot()
        self._message_ids = ()
        self._source_ids = ()
        self._set_fields("", "", ())
        self._sync_editor_state("Unsaved changes · Clear can be undone")

    @on(Button.Pressed, "#research-quick-note-undo")
    def undo_clear(self) -> None:
        if self._undo is None:
            return
        snapshot = self._undo
        self._undo = None
        self._message_ids = snapshot.message_ids
        self._source_ids = snapshot.source_ids
        self._set_fields(snapshot.title, snapshot.content, snapshot.tags)
        self._sync_editor_state("Unsaved changes")

    @on(Button.Pressed, "#research-quick-note-capture-sources")
    def capture_sources(self) -> None:
        if self.editor_ref is not None:
            self.post_message(self.CaptureSourcesRequested(self.editor_ref))
