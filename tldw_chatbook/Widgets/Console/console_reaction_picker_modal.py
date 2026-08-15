"""Metadata-only Console reaction picker with lazy preview requests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from rich.markup import escape as escape_markup
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.message_pump import MessagePump
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

FILTER_INPUT_ID = "console-reaction-picker-filter"
RESULTS_CONTAINER_ID = "console-reaction-picker-results"
PREVIEW_ID = "console-reaction-picker-preview"
ROW_CLASS = "console-reaction-picker-row"
ROW_HIGHLIGHTED_CLASS = "console-reaction-picker-row-highlighted"

FILTER_DEBOUNCE_SECONDS = 0.2
PREVIEW_DEBOUNCE_SECONDS = 0.2


@dataclass(frozen=True, slots=True)
class ReactionOption:
    """One selectable reaction's display and decode metadata."""

    expression_key: str
    display_label: str
    content_type: str
    is_animated: bool


class ReactionPreviewRequested(Message):
    """Ask the Console owner to lazily load one highlighted reaction."""

    def __init__(
        self, option: ReactionOption, picker: ConsoleReactionPickerModal | None = None
    ) -> None:
        self.option = option
        self.picker = picker
        super().__init__()


class ReactionSelected(Message):
    """Report an explicit session-local reaction selection."""

    def __init__(self, option: ReactionOption) -> None:
        self.option = option
        super().__init__()


class ReactionCleared(Message):
    """Ask the Console owner to return to automatic reactions."""


def filter_reaction_options(
    options: Sequence[ReactionOption], query: str
) -> tuple[ReactionOption, ...]:
    """Return options whose display label or expression key contains ``query``."""

    needle = str(query or "").strip().casefold()
    if not needle:
        return tuple(options)
    return tuple(
        option
        for option in options
        if needle in option.display_label.casefold()
        or needle in option.expression_key.casefold()
    )


class ConsoleReactionPickerModal(SafeModalDismissMixin, ModalScreen[None]):
    """Filter reaction metadata and request at most one highlighted preview."""

    DEFAULT_CSS = """
    ConsoleReactionPickerModal {
        align: center middle;
    }

    #console-reaction-picker-modal {
        width: 76;
        max-width: 100%;
        height: 30;
        max-height: 100%;
        border: tall $accent;
        background: $panel;
        padding: 1 2;
    }

    #console-reaction-picker-filter {
        width: 100%;
        height: 3;
    }

    #console-reaction-picker-count {
        width: 100%;
        height: 1;
        color: $text-muted;
    }

    #console-reaction-picker-body {
        width: 100%;
        height: 1fr;
        min-height: 4;
    }

    #console-reaction-picker-results {
        width: 2fr;
        height: 100%;
        min-width: 0;
        background: $surface-darken-1;
    }

    .console-reaction-picker-row {
        width: 100%;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
        margin: 0;
        background: $surface;
        color: $text;
        content-align: left middle;
        text-align: left;
    }

    .console-reaction-picker-row-highlighted {
        background: $accent 25%;
        color: $text;
        text-style: bold;
    }

    #console-reaction-picker-preview {
        width: 1fr;
        height: 100%;
        min-width: 0;
        margin-left: 1;
        padding: 1;
        background: $surface;
    }

    #console-reaction-picker-preview-label {
        width: 100%;
        height: auto;
        text-style: bold;
    }

    #console-reaction-picker-preview-meta {
        width: 100%;
        height: auto;
        color: $text-muted;
    }

    #console-reaction-picker-preview-image {
        width: 100%;
        height: 1fr;
        content-align: center middle;
    }

    #console-reaction-picker-actions {
        width: 100%;
        height: 3;
        margin-top: 1;
    }

    #console-reaction-picker-actions Button {
        width: 1fr;
        height: 3;
        border: none;
        margin: 0 1 0 0;
    }

    #console-reaction-picker-select {
        background: $primary;
        color: $text;
    }

    #console-reaction-picker-clear,
    #console-reaction-picker-cancel {
        background: $surface;
        color: $text;
    }

    #console-reaction-picker-actions Button:focus {
        outline: heavy $accent;
    }

    ConsoleReactionPickerModal.-narrow #console-reaction-picker-modal {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }

    ConsoleReactionPickerModal.-narrow #console-reaction-picker-preview {
        display: none;
    }

    ConsoleReactionPickerModal.-narrow #console-reaction-picker-results {
        width: 100%;
    }
    """

    BINDINGS: ClassVar = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#console-reaction-picker-modal"

    def __init__(
        self,
        *,
        options: Sequence[ReactionOption],
        message_target: MessagePump | None = None,
        preview_callback: Callable[[ReactionOption, ConsoleReactionPickerModal], None]
        | None = None,
        preview_cancel_callback: Callable[[ConsoleReactionPickerModal], None]
        | None = None,
        selection_callback: Callable[[ReactionOption], None] | None = None,
        clear_callback: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the picker from metadata and an optional owning message pump."""

        super().__init__(**kwargs)
        self._options = tuple(options)
        self._message_target = message_target
        self._preview_callback = preview_callback
        self._preview_cancel_callback = preview_cancel_callback
        self._preview_owner_cancelled = False
        self._selection_callback = selection_callback
        self._clear_callback = clear_callback
        self._filtered: tuple[ReactionOption, ...] = ()
        self._highlighted_index = 0
        self._row_ids: list[str] = []
        self._last_preview_key: str | None = None
        self._filter_debounce_timer: Timer | None = None
        self._pending_filter_query: str | None = None
        self._pending_filter_token: int | None = None
        self._filter_generation = 0
        self._filter_apply_task: asyncio.Task[None] | None = None
        self._preview_debounce_timer: Timer | None = None
        self._pending_preview_key: str | None = None
        self._preview_generation = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="console-reaction-picker-modal"):
            yield Static("Choose reaction", classes="console-modal-header")
            yield Input(
                placeholder="Filter reactions…",
                id=FILTER_INPUT_ID,
            )
            yield Static(
                "0 / 0 reactions",
                id="console-reaction-picker-count",
                markup=False,
            )
            with Horizontal(id="console-reaction-picker-body"):
                yield VerticalScroll(id=RESULTS_CONTAINER_ID, can_focus=False)
                with Vertical(id=PREVIEW_ID):
                    yield Static(
                        "Highlight a reaction",
                        id="console-reaction-picker-preview-label",
                        markup=False,
                    )
                    yield Static(
                        "Preview loads only for the highlighted reaction.",
                        id="console-reaction-picker-preview-meta",
                        markup=False,
                    )
                    yield Static(
                        "Loading preview…",
                        id="console-reaction-picker-preview-image",
                        markup=False,
                    )
            with Horizontal(id="console-reaction-picker-actions"):
                yield Button(
                    "Use reaction",
                    id="console-reaction-picker-select",
                    compact=True,
                )
                yield Button(
                    "Clear",
                    id="console-reaction-picker-clear",
                    compact=True,
                )
                yield Button(
                    "Cancel",
                    id="console-reaction-picker-cancel",
                    compact=True,
                )

    async def on_mount(self) -> None:  # type: ignore[override]
        self._set_responsive(self.app.size.width, self.app.size.height)
        await self._apply_filter("")
        self._focus_filter()

    def on_unmount(self) -> None:
        self._cancel_pending_updates()

    def on_resize(self, event: Any) -> None:
        self._set_responsive(event.size.width, event.size.height)

    def _set_responsive(self, width: int, height: int) -> None:
        self.set_class(width <= 80 or height <= 24, "-narrow")

    def _focus_filter(self) -> None:
        try:
            self.query_one(f"#{FILTER_INPUT_ID}", Input).focus()
        except (NoMatches, QueryError):
            return

    @on(Input.Changed, f"#{FILTER_INPUT_ID}")
    def _filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._cancel_filter_debounce()
        self._cancel_preview_debounce()
        self._filter_generation += 1
        token = self._filter_generation
        query = event.value
        self._pending_filter_query = query
        self._pending_filter_token = token
        self._filter_debounce_timer = self.set_timer(
            FILTER_DEBOUNCE_SECONDS,
            lambda: self._settle_filter_timer(token=token),
        )

    @on(Input.Submitted, f"#{FILTER_INPUT_ID}")
    async def _filter_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        await self._flush_pending_filter()
        self._select_highlighted()

    async def on_key(self, event: events.Key) -> None:
        if event.key == "down":
            event.stop()
            await self._flush_pending_filter()
            self._move_highlight(1)
        elif event.key == "up":
            event.stop()
            await self._flush_pending_filter()
            self._move_highlight(-1)

    def _cancel_filter_debounce(self) -> None:
        if self._filter_debounce_timer is not None:
            self._filter_debounce_timer.stop()
            self._filter_debounce_timer = None

    async def _settle_filter_timer(self, *, token: int) -> None:
        if not self.is_mounted or token != self._filter_generation:
            return
        self._filter_debounce_timer = None
        await self._drain_filter_updates(expected_token=token)

    def _claim_pending_filter(self) -> tuple[int, str] | None:
        query = self._pending_filter_query
        token = self._pending_filter_token
        if (
            query is None
            or token is None
            or token != self._filter_generation
            or not self.is_mounted
        ):
            return None
        try:
            if self.query_one(f"#{FILTER_INPUT_ID}", Input).value != query:
                self._pending_filter_query = None
                self._pending_filter_token = None
                return None
        except (NoMatches, QueryError):
            return None
        self._pending_filter_query = None
        self._pending_filter_token = None
        return token, query

    async def _run_claimed_filter(self, *, token: int, query: str) -> None:
        task = asyncio.current_task()
        try:
            if self.is_mounted:
                await self._apply_filter(query)
                if token != self._filter_generation:
                    self._cancel_preview_debounce()
        finally:
            if self._filter_apply_task is task:
                self._filter_apply_task = None

    async def _drain_filter_updates(self, *, expected_token: int | None = None) -> None:
        while self.is_mounted:
            task = self._filter_apply_task
            if task is asyncio.current_task():
                return
            if task is None:
                if (
                    expected_token is not None
                    and expected_token != self._filter_generation
                ):
                    return
                claimed = self._claim_pending_filter()
                if claimed is None:
                    return
                token, query = claimed
                task = asyncio.create_task(
                    self._run_claimed_filter(token=token, query=query),
                    name="console-reaction-picker-filter",
                )
                self._filter_apply_task = task
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled() or not self.is_mounted:
                    return
                raise

    async def _flush_pending_filter(self) -> None:
        self._cancel_filter_debounce()
        await self._drain_filter_updates()

    def _cancel_filter_apply(self) -> None:
        task = self._filter_apply_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _apply_filter(self, query: str) -> None:
        self._filtered = filter_reaction_options(self._options, query)
        self._highlighted_index = 0
        await self._render_results()

    async def _render_results(self) -> None:
        try:
            results = self.query_one(f"#{RESULTS_CONTAINER_ID}", VerticalScroll)
        except (NoMatches, QueryError):
            return
        await results.remove_children()
        self._row_ids = []
        if not self._filtered:
            copy = (
                "No reactions available."
                if not self._options
                else "No reactions match."
            )
            await results.mount(
                Static(copy, id="console-reaction-picker-empty", markup=False)
            )
        else:
            rows: list[Button] = []
            for index, option in enumerate(self._filtered):
                row_id = f"console-reaction-picker-row-{index}"
                self._row_ids.append(row_id)
                row = Button(
                    self._row_label(index, option),
                    id=row_id,
                    classes=ROW_CLASS,
                    compact=True,
                )
                row.can_focus = False
                rows.append(row)
            await results.mount_all(rows)
        self._sync_highlight()
        self._focus_filter()

    def _row_label(self, index: int, option: ReactionOption) -> str:
        marker = "> " if index == self._highlighted_index else "  "
        animation = " · animated" if option.is_animated else ""
        return f"{marker}{escape_markup(option.display_label)}{animation}"

    def _move_highlight(self, delta: int) -> None:
        if not self._filtered:
            return
        self._highlighted_index = (self._highlighted_index + delta) % len(
            self._filtered
        )
        self._sync_highlight()

    def _sync_highlight(self) -> None:
        self._sync_count_and_actions()
        try:
            results = self.query_one(f"#{RESULTS_CONTAINER_ID}", VerticalScroll)
        except (NoMatches, QueryError):
            return
        highlighted_row_id: str | None = None
        for index, button in enumerate(results.query(f".{ROW_CLASS}")):
            is_highlighted = index == self._highlighted_index
            button.set_class(is_highlighted, ROW_HIGHLIGHTED_CLASS)
            if is_highlighted:
                highlighted_row_id = button.id
            if index < len(self._filtered):
                button.label = self._row_label(index, self._filtered[index])
        if highlighted_row_id is not None:
            self.call_after_refresh(
                self._scroll_highlighted_into_view,
                highlighted_row_id,
            )
        option = self._highlighted_option()
        self._sync_preview_metadata(option)
        if option is None:
            self._cancel_preview_debounce()
        else:
            self._schedule_preview_request(option)

    def _scroll_highlighted_into_view(self, row_id: str) -> None:
        """Reveal the current metadata row after Textual finishes layout."""

        if (
            not 0 <= self._highlighted_index < len(self._row_ids)
            or self._row_ids[self._highlighted_index] != row_id
        ):
            return
        try:
            results = self.query_one(f"#{RESULTS_CONTAINER_ID}", VerticalScroll)
            row = results.query_one(f"#{row_id}", Button)
        except (NoMatches, QueryError):
            return
        results.scroll_to_widget(
            row,
            animate=False,
            force=True,
            immediate=True,
        )

    def _sync_count_and_actions(self) -> None:
        total = len(self._filtered)
        current = self._highlighted_index + 1 if total else 0
        try:
            self.query_one("#console-reaction-picker-count", Static).update(
                f"{current} / {total} reactions"
            )
            self.query_one(
                "#console-reaction-picker-select", Button
            ).disabled = not total
        except (NoMatches, QueryError):
            return

    def _sync_preview_metadata(self, option: ReactionOption | None) -> None:
        try:
            label = self.query_one("#console-reaction-picker-preview-label", Static)
            meta = self.query_one("#console-reaction-picker-preview-meta", Static)
        except (NoMatches, QueryError):
            return
        if option is None:
            label.update("No reaction highlighted")
            meta.update("Adjust the filter or clear the manual override.")
            return
        label.update(option.display_label)
        motion = "Animated" if option.is_animated else "Static"
        meta.update(f"{option.expression_key}\n{motion} · {option.content_type}")
        try:
            self.query_one("#console-reaction-picker-preview-image", Static).update(
                "Loading preview…"
            )
        except (NoMatches, QueryError):
            pass

    def update_preview(self, expression_key: str, renderable: object) -> bool:
        """Show an already-decoded preview only while its reaction is current."""

        option = self._highlighted_option()
        if option is None or option.expression_key != expression_key:
            return False
        try:
            self.query_one("#console-reaction-picker-preview-image", Static).update(
                renderable
            )
        except (NoMatches, QueryError):
            return False
        return True

    def is_preview_current(self, expression_key: str) -> bool:
        """Return whether this mounted picker still highlights one key."""

        option = self._highlighted_option()
        return (
            self.is_mounted
            and option is not None
            and option.expression_key == expression_key
        )

    def _request_preview(self, option: ReactionOption) -> None:
        if option.expression_key == self._last_preview_key:
            return
        self._last_preview_key = option.expression_key
        if self._preview_callback is not None:
            self._preview_callback(option, self)
        else:
            self._emit(ReactionPreviewRequested(option, self))

    def _schedule_preview_request(self, option: ReactionOption) -> None:
        self._cancel_preview_debounce()
        if option.expression_key == self._last_preview_key:
            return
        key = option.expression_key
        token = self._preview_generation
        self._pending_preview_key = key
        self._preview_debounce_timer = self.set_timer(
            PREVIEW_DEBOUNCE_SECONDS,
            lambda: self._emit_settled_preview(key=key, token=token),
        )

    def _cancel_preview_debounce(self) -> None:
        if self._preview_debounce_timer is not None:
            self._preview_debounce_timer.stop()
            self._preview_debounce_timer = None
        self._preview_generation += 1
        self._pending_preview_key = None

    def _emit_settled_preview(self, *, key: str, token: int) -> None:
        if (
            not self.is_mounted
            or token != self._preview_generation
            or key != self._pending_preview_key
        ):
            return
        self._preview_debounce_timer = None
        option = self._highlighted_option()
        if option is None or option.expression_key != key:
            self._pending_preview_key = None
            return
        index = self._highlighted_index
        if not 0 <= index < len(self._row_ids):
            self._pending_preview_key = None
            return
        try:
            row = self.query_one(f"#{self._row_ids[index]}", Button)
        except (NoMatches, QueryError):
            self._pending_preview_key = None
            return
        if not row.has_class(ROW_HIGHLIGHTED_CLASS):
            self._pending_preview_key = None
            return
        self._pending_preview_key = None
        self._request_preview(option)

    def _cancel_pending_updates(self) -> None:
        self._cancel_filter_debounce()
        self._filter_generation += 1
        self._pending_filter_query = None
        self._pending_filter_token = None
        self._cancel_filter_apply()
        self._cancel_preview_debounce()
        if not self._preview_owner_cancelled:
            self._preview_owner_cancelled = True
            if self._preview_cancel_callback is not None:
                self._preview_cancel_callback(self)

    def _emit(self, message: Message) -> None:
        """Post to the Console owner when supplied, otherwise bubble normally."""

        (self._message_target or self).post_message(message)

    def _highlighted_option(self) -> ReactionOption | None:
        if 0 <= self._highlighted_index < len(self._filtered):
            return self._filtered[self._highlighted_index]
        return None

    @on(Button.Pressed, f".{ROW_CLASS}")
    def _row_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        if button_id not in self._row_ids:
            return
        self._highlighted_index = self._row_ids.index(button_id)
        self._sync_highlight()
        self._select_highlighted()

    @on(Button.Pressed, "#console-reaction-picker-select")
    async def _select_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self._flush_pending_filter()
        self._select_highlighted()

    @on(Button.Pressed, "#console-reaction-picker-clear")
    def _clear_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._cancel_pending_updates()
        if self._clear_callback is not None:
            self._clear_callback()
        else:
            self._emit(ReactionCleared())
        self.dismiss(None)

    @on(Button.Pressed, "#console-reaction-picker-cancel")
    async def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    def _select_highlighted(self) -> None:
        option = self._highlighted_option()
        if option is None:
            return
        self._cancel_pending_updates()
        if self._selection_callback is not None:
            self._selection_callback(option)
        else:
            self._emit(ReactionSelected(option))
        self.dismiss(None)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source

        async def cancel_pending_updates() -> None:
            self._cancel_pending_updates()

        await self.run_cancel_effect_once(cancel_pending_updates)
        self.dismiss_safe_once(None)
