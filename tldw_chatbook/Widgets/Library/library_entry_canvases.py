"""Retained owners for Library landing and Study handoff entry surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_CANVAS_KIND_NOTES_CREATE,
    LIBRARY_ROW_BROWSE_SEARCH,
    LIBRARY_ROW_CREATE_NOTE,
    LIBRARY_ROW_INGEST_MEDIA,
)
from tldw_chatbook.Library.library_rail_state import LibraryLifecycle
from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback


@dataclass(frozen=True)
class LibraryLandingRecentItem:
    """Display data and stable dispatch identity for one landing recent row."""

    source_type: str
    record_id: str
    title: str
    source_label: str


@dataclass(frozen=True)
class LibraryLandingCanvasState:
    """Complete display snapshot for the retained Library landing canvas."""

    purpose: str
    counts_line: str
    recent_items: tuple[LibraryLandingRecentItem, ...] = ()
    lifecycle: LibraryLifecycle = LibraryLifecycle.EXPANDED
    lifecycle_status: str = ""
    show_retry: bool = False
    show_explore: bool = False


@dataclass(frozen=True)
class LibraryStudyHandoffCanvasState:
    """Complete display snapshot for one retained Study handoff canvas."""

    header: str
    purpose: str
    context: str
    owner: str
    recovery: str
    blocked: bool
    button_label: str
    button_id: str
    action_label: str


class _RetainedSyncCallback(PostRecomposeCallback):
    """Complete a targeted in-place sync through the mixin's single slot."""

    def _complete_targeted_sync(self) -> None:
        callback: Callable[[], None] | None = self._post_recompose_callback
        self._post_recompose_callback = None
        if callback is not None:
            callback()


class LibraryLandingCanvas(_RetainedSyncCallback, Vertical):
    """Retain the landing actions while counts and recent rows change."""

    def __init__(self, state: LibraryLandingCanvasState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self._deferred_sync_guard: Callable[[], bool] | None = None
        self._deferred_sync_serial = 0
        self.styles.width = "1fr"
        self.styles.min_width = 0

    def set_deferred_sync_guard(self, guard: Callable[[], bool] | None) -> None:
        """Set the route/generation predicate for the next deferred row sync."""
        self._deferred_sync_guard = guard

    def _deferred_sync_is_current(self, serial: int) -> bool:
        """Return whether deferred work still owns this attached canvas."""
        return (
            serial == self._deferred_sync_serial
            and self.is_attached
            and not getattr(self, "_pruning", False)
            and (
                self._deferred_sync_guard is None or self._deferred_sync_guard()
            )
        )

    def _discard_stale_deferred_sync(self, serial: int) -> None:
        """Clear only the callback owned by the latest now-stale sync."""
        if serial == self._deferred_sync_serial:
            self.queue_after_recompose(None)

    @staticmethod
    def _action_button(
        label: str,
        tooltip: str,
        row_id: str,
        target_id: str,
        button_id: str,
    ) -> Button:
        action = Button(
            label,
            id=button_id,
            classes="library-hub-action console-action-subdued",
            compact=True,
            tooltip=tooltip,
        )
        action.row_id = row_id
        action.target_kind = "canvas"
        action.target_id = target_id
        return action

    @staticmethod
    def _recent_button(item: LibraryLandingRecentItem) -> Button:
        recent = Button(
            f"{item.source_label} · {escape_markup(item.title)}",
            id=f"library-hub-recent-{item.source_type}",
            classes="library-hub-recent console-action-subdued",
            compact=True,
            tooltip=f"Open {item.source_label.lower()}: {item.title}",
        )
        recent.source_type = item.source_type
        recent.record_id = item.record_id
        return recent

    @staticmethod
    def _is_get_started(state: LibraryLandingCanvasState) -> bool:
        return state.lifecycle in (
            LibraryLifecycle.UNKNOWN,
            LibraryLifecycle.STARTER,
        )

    @classmethod
    def _widget_set_key(cls, state: LibraryLandingCanvasState) -> tuple[bool, bool, bool]:
        return (cls._is_get_started(state), state.show_retry, state.show_explore)

    @staticmethod
    def _status(value: str, widget_id: str) -> Static:
        status = Static(value, id=widget_id, classes="library-hub-meta", markup=False)
        status.display = bool(value)
        return status

    def compose(self) -> ComposeResult:
        get_started = self._is_get_started(self.state)
        if get_started:
            yield Static(
                "Get started",
                id="library-hub-heading",
                classes="destination-section",
                markup=False,
            )
        yield Static(
            self.state.purpose,
            id="library-canvas-landing",
            classes="destination-purpose",
            markup=False,
        )
        yield self._status(
            self.state.lifecycle_status,
            "library-hub-lifecycle-status",
        )
        if get_started:
            orientation = Static(
                "1 Add · 2 Find · 3 Use",
                id="library-hub-orientation",
                classes="library-hub-meta",
                markup=False,
            )
            orientation.display = self.state.lifecycle is LibraryLifecycle.STARTER
            yield orientation
        else:
            yield Static(
                self.state.counts_line,
                id="library-hub-counts",
                classes="library-hub-meta",
                markup=False,
            )
        with Horizontal(id="library-hub-actions", classes="ds-toolbar"):
            yield self._action_button(
                "Import…",
                "Add files, links, and transcripts to your Library.",
                LIBRARY_ROW_INGEST_MEDIA,
                "ingest-media",
                "library-hub-action-import",
            )
            if not get_started:
                yield self._action_button(
                    "Search",
                    "Search everything in the Library.",
                    LIBRARY_ROW_BROWSE_SEARCH,
                    "search",
                    "library-hub-action-search",
                )
            yield self._action_button(
                "New note",
                "Create a new note.",
                LIBRARY_ROW_CREATE_NOTE,
                LIBRARY_CANVAS_KIND_NOTES_CREATE,
                "library-hub-action-new-note",
            )
            if self.state.show_explore:
                yield Button("Explore all tools", id="library-hub-explore-all")
        if self.state.show_retry:
            yield Button("Retry source check", id="library-hub-retry-evidence")
        if not get_started:
            recents = Vertical(id="library-hub-recents")
            recents.styles.height = "auto"
            with recents:
                for item in self.state.recent_items:
                    yield self._recent_button(item)

    def sync_state(self, state: LibraryLandingCanvasState) -> None:
        """Patch stable fields, recomposing only when the widget set changes."""
        previous_key = self._widget_set_key(self.state)
        self.state = state
        self._deferred_sync_serial += 1
        if self._widget_set_key(state) != previous_key:
            self.refresh(recompose=True)
            return
        purpose = self.query_one("#library-canvas-landing", Static)
        purpose.update(state.purpose)
        lifecycle_status = self.query_one("#library-hub-lifecycle-status", Static)
        lifecycle_status.update(state.lifecycle_status)
        lifecycle_status.display = bool(state.lifecycle_status)
        if self._is_get_started(state):
            orientation = self.query_one("#library-hub-orientation", Static)
            orientation.display = state.lifecycle is LibraryLifecycle.STARTER
            self._complete_targeted_sync()
            return
        self.query_one("#library-hub-counts", Static).update(state.counts_line)
        self.call_later(self._replace_recent_rows, self._deferred_sync_serial)

    async def _replace_recent_rows(self, serial: int | None = None) -> None:
        """Converge queued replacements on the latest state after each await."""
        serial = self._deferred_sync_serial if serial is None else serial
        if not self._deferred_sync_is_current(serial):
            self._discard_stale_deferred_sync(serial)
            return
        recents = self.query_one("#library-hub-recents", Vertical)
        if not self._deferred_sync_is_current(serial):
            self._discard_stale_deferred_sync(serial)
            return
        await recents.remove_children()
        if not self._deferred_sync_is_current(serial):
            self._discard_stale_deferred_sync(serial)
            return
        recent_items = self.state.recent_items
        if recent_items:
            await recents.mount(*(self._recent_button(item) for item in recent_items))
        if not self._deferred_sync_is_current(serial):
            self._discard_stale_deferred_sync(serial)
            return
        if recent_items != self.state.recent_items:
            self.call_later(self._replace_recent_rows, serial)
            return
        self._complete_targeted_sync()


class LibraryStudyHandoffCanvas(_RetainedSyncCallback, Vertical):
    """Retain a Study handoff action while source readiness changes."""

    def __init__(self, state: LibraryStudyHandoffCanvasState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.styles.width = "1fr"
        self.styles.min_width = 0

    def compose(self) -> ComposeResult:
        yield Static(
            self.state.header,
            id="library-active-mode-title",
            classes="destination-section",
        )
        yield Static(self.state.purpose, id="library-study-handoff-purpose")
        context = Static(self.state.context, id="library-study-handoff-context")
        context.display = bool(self.state.context)
        yield context
        yield Static(self.state.owner, id="library-study-handoff-owner")
        recovery = Static(self.state.recovery, id="library-study-handoff-recovery")
        recovery.set_class(self.state.blocked, "ds-recovery-callout")
        recovery.set_class(self.state.blocked, "is-blocked")
        yield recovery
        toolbar = Horizontal(id="library-study-handoff-actions", classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                self.state.button_label,
                id=self.state.button_id,
                classes="library-canvas-action console-action-primary",
                compact=True,
                tooltip=(
                    f"Open {self.state.action_label} with the current Library "
                    "source snapshot, or globally when none is available."
                ),
            )

    def sync_state(self, state: LibraryStudyHandoffCanvasState) -> None:
        """Patch handoff copy and blocked styling without replacing the action."""
        self.state = state
        self.query_one("#library-active-mode-title", Static).update(state.header)
        self.query_one("#library-study-handoff-purpose", Static).update(state.purpose)
        context = self.query_one("#library-study-handoff-context", Static)
        context.update(state.context)
        context.display = bool(state.context)
        self.query_one("#library-study-handoff-owner", Static).update(state.owner)
        recovery = self.query_one("#library-study-handoff-recovery", Static)
        recovery.update(state.recovery)
        recovery.set_class(state.blocked, "ds-recovery-callout")
        recovery.set_class(state.blocked, "is-blocked")
        self._complete_targeted_sync()
