"""Console prompt picker modal.

Lets the user search saved prompts and pick one, either to insert its
``user_prompt`` body into the Console composer (``/prompt``, mode
``"insert"``) or to apply its ``system_prompt`` as the session's system
prompt (``/system``, mode ``"apply-system"``). The caller (Tasks 12/14)
supplies an already-adapted ``prompt_search`` closure over the scope
service's ``search_prompts`` seam (Task 6) -- this widget never reaches into
the scope service directly.

Keyboard-first: the filter ``Input`` keeps focus for the whole session; Up/Down
move a synthetic highlighted-row index (``Input`` has no arrow-key bindings in
this Textual version, so the raw key bubbles here); Enter activates the
highlighted row via ``Input.Submitted`` (Input consumes the raw Enter key
itself, so the codebase's established idiom is listening for the bubbled
message rather than intercepting the key). Esc dismisses with ``None`` via the
inherited ``ModalScreen`` binding pattern used by every sibling Console modal.

Note: this screen only dismisses; the CALLER is responsible for returning
focus to the Console composer afterwards (mirrors how ``ConsoleSettingsModal``
and ``ConsoleSaveAsModal`` leave focus restoration to their callers).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Awaitable, Callable, Optional

from rich.markup import escape as escape_markup
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Static

PromptSearch = Callable[[str], Awaitable[list[Mapping[str, object]]]]

MODE_INSERT = "insert"
MODE_APPLY_SYSTEM = "apply-system"
_VALID_MODES = (MODE_INSERT, MODE_APPLY_SYSTEM)

FILTER_INPUT_ID = "console-prompt-picker-filter"
MODAL_ID = "console-prompt-picker-modal"
RESULTS_CONTAINER_ID = "console-prompt-picker-results"
REASON_STATIC_ID = "console-prompt-picker-reason"
EMPTY_STATIC_ID = "console-prompt-picker-empty"
ROW_ID_PREFIX = "console-prompt-picker-row-"
ROW_CLASS = "console-prompt-picker-row"
ROW_HIGHLIGHTED_CLASS = "console-prompt-picker-row-highlighted"
ROW_BLOCKED_CLASS = "console-prompt-picker-row-blocked"

# FTS search debounce; ``prompt_search`` itself is expected to bound each
# page to <= 25 rows (Task 6's ``PromptScopeService.search_prompts``), this
# widget never asks for more nor trims a larger result set silently -- if the
# callable ever returned more than the bound, that's a caller bug to surface,
# not paper over here.
SEARCH_DEBOUNCE_SECONDS = 0.2

EMPTY_STORE_COPY = "No saved prompts yet — create them in Library ▸ Prompts."
NO_SYSTEM_PART_SUFFIX = " (no system part)"
NO_SYSTEM_PART_REASON = "This prompt has no system part to apply as the session system prompt."

_MODE_TITLES = {
    MODE_INSERT: "Insert prompt",
    MODE_APPLY_SYSTEM: "Apply system prompt",
}


class ConsolePromptPickerModal(ModalScreen[Optional[Mapping[str, object]]]):
    """Search and pick a saved prompt for the Console `/prompt`/`/system` commands."""

    BINDINGS = [("escape", "dismiss_picker", "Cancel")]

    def __init__(
        self,
        *,
        mode: str,
        initial_query: str = "",
        prompt_search: PromptSearch,
    ) -> None:
        """Initialize the picker.

        Args:
            mode: Either ``"insert"`` (picking a prompt whose ``user_prompt``
                will be inserted) or ``"apply-system"`` (picking a prompt
                whose ``system_prompt`` will be applied; rows without one are
                shown but refuse selection).
            initial_query: Prefilled filter text (e.g. the ambiguous
                ``/prompt <args>`` the user typed), searched immediately on
                mount without waiting for the debounce.
            prompt_search: Async callable bound to the scope-service search
                seam (Task 6) by the caller; receives the settled filter text
                and returns a bounded (<=25), FTS-ranked list of normalized
                prompt detail mappings.

        Raises:
            ValueError: If ``mode`` isn't a recognized picker mode.
        """
        if mode not in _VALID_MODES:
            raise ValueError(f"Unsupported prompt picker mode: {mode!r}")
        super().__init__()
        self._mode = mode
        self._initial_query = initial_query
        self._prompt_search = prompt_search
        self._results: list[Mapping[str, object]] = []
        self._highlighted_index = 0
        self._search_debounce_timer: Timer | None = None
        self._search_token = 0

    def compose(self) -> ComposeResult:
        with Vertical(id=MODAL_ID):
            yield Static(_MODE_TITLES[self._mode], classes="console-modal-header")
            yield Input(
                value=self._initial_query,
                placeholder="Search prompts…",
                id=FILTER_INPUT_ID,
            )
            reason = Static("", id=REASON_STATIC_ID, markup=False)
            reason.display = False
            yield reason
            with VerticalScroll(id=RESULTS_CONTAINER_ID):
                yield Static(EMPTY_STORE_COPY, id=EMPTY_STATIC_ID, markup=False)

    def on_mount(self) -> None:
        self.query_one(f"#{FILTER_INPUT_ID}", Input).focus()
        # The initial (possibly ambiguous-command-prefilled) query populates
        # the list right away -- only edits made *after* opening debounce.
        self._trigger_search(self._initial_query, debounce=False)

    def action_dismiss_picker(self) -> None:
        self.dismiss(None)

    @on(Input.Changed, f"#{FILTER_INPUT_ID}")
    def _filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._trigger_search(event.value, debounce=True)

    @on(Input.Submitted, f"#{FILTER_INPUT_ID}")
    def _filter_submitted(self, event: Input.Submitted) -> None:
        # Input consumes the raw Enter keypress itself (bound to
        # action_submit) and re-emits it as this message; there is nothing
        # left for a parent on_key handler to intercept, so this is the
        # correct (and this codebase's established) place to react to Enter.
        event.stop()
        self._select_highlighted()

    def on_key(self, event: events.Key) -> None:
        # Input has no up/down bindings in this Textual version, so these
        # bubble here unconsumed while the filter keeps focus.
        if event.key == "down":
            event.stop()
            self._move_highlight(1)
        elif event.key == "up":
            event.stop()
            self._move_highlight(-1)

    @on(Button.Pressed, f".{ROW_CLASS}")
    def _row_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        for index, record in enumerate(self._results):
            if button_id == self._row_id(record, index):
                self._highlighted_index = index
                self._sync_highlight()
                self._select_record(record)
                return

    # -- search -----------------------------------------------------------

    def _trigger_search(self, query: str, *, debounce: bool) -> None:
        self._cancel_search_debounce()
        self._search_token += 1
        token = self._search_token
        if debounce:
            self._search_debounce_timer = self.set_timer(
                SEARCH_DEBOUNCE_SECONDS,
                lambda: self._start_debounced_search(query=query, token=token),
            )
        else:
            self._start_debounced_search(query=query, token=token)

    def _cancel_search_debounce(self) -> None:
        if self._search_debounce_timer is not None:
            self._search_debounce_timer.stop()
            self._search_debounce_timer = None

    def _start_debounced_search(self, *, query: str, token: int) -> None:
        self._search_debounce_timer = None
        self.run_worker(
            self._run_search(query=query, token=token),
            exclusive=True,
            group="console-prompt-picker-search",
        )

    async def _run_search(self, *, query: str, token: int) -> None:
        try:
            results = await self._prompt_search(query)
        except Exception:
            results = []
        if token != self._search_token:
            return  # A newer filter change superseded this in-flight search.
        self._results = list(results or [])
        self._highlighted_index = 0
        await self._render_results()

    # -- rendering ----------------------------------------------------------

    async def _render_results(self) -> None:
        try:
            container = self.query_one(f"#{RESULTS_CONTAINER_ID}", VerticalScroll)
        except (NoMatches, QueryError):
            return  # Modal was dismissed/unmounted while the search was in flight.
        # Awaited (mirrors ConsoleSessionSwitcherModal._refresh_results): the
        # removal must complete before mounting a same-id replacement, or a
        # DuplicateIds error can fire if the message pump hasn't caught up.
        await container.remove_children()
        self._hide_reason()
        if not self._results:
            await container.mount(Static(EMPTY_STORE_COPY, id=EMPTY_STATIC_ID, markup=False))
            return
        await container.mount_all(
            self._build_row_button(index, record) for index, record in enumerate(self._results)
        )
        self._sync_highlight()

    def _build_row_button(self, index: int, record: Mapping[str, object]) -> Button:
        name = escape_markup(str(record.get("name") or "Untitled prompt"))
        blocked = self._is_blocked(record)
        label = f"{name}{NO_SYSTEM_PART_SUFFIX}" if blocked else name
        button = Button(label, id=self._row_id(record, index), classes=ROW_CLASS)
        button.set_class(blocked, ROW_BLOCKED_CLASS)
        return button

    def _row_id(self, record: Mapping[str, object], index: int) -> str:
        for key in ("local_id", "id"):
            value = record.get(key)
            if value not in (None, ""):
                return f"{ROW_ID_PREFIX}{value}"
        return f"{ROW_ID_PREFIX}{index}"

    def _is_blocked(self, record: Mapping[str, object]) -> bool:
        return self._mode == MODE_APPLY_SYSTEM and not self._has_system_part(record)

    @staticmethod
    def _has_system_part(record: Mapping[str, object]) -> bool:
        return bool(str(record.get("system_prompt") or "").strip())

    # -- highlight / selection ------------------------------------------------

    def _move_highlight(self, delta: int) -> None:
        if not self._results:
            return
        self._highlighted_index = (self._highlighted_index + delta) % len(self._results)
        self._sync_highlight()

    def _sync_highlight(self) -> None:
        try:
            container = self.query_one(f"#{RESULTS_CONTAINER_ID}", VerticalScroll)
        except (NoMatches, QueryError):
            return
        for index, button in enumerate(container.query(Button)):
            button.set_class(index == self._highlighted_index, ROW_HIGHLIGHTED_CLASS)

    def _select_highlighted(self) -> None:
        if not (0 <= self._highlighted_index < len(self._results)):
            return
        self._select_record(self._results[self._highlighted_index])

    def _select_record(self, record: Mapping[str, object]) -> None:
        if self._is_blocked(record):
            self._show_reason(NO_SYSTEM_PART_REASON)
            return
        self._cancel_search_debounce()
        self.dismiss(record)

    def _show_reason(self, text: str) -> None:
        try:
            reason = self.query_one(f"#{REASON_STATIC_ID}", Static)
        except (NoMatches, QueryError):
            return
        reason.update(text)
        reason.display = True

    def _hide_reason(self) -> None:
        try:
            reason = self.query_one(f"#{REASON_STATIC_ID}", Static)
        except (NoMatches, QueryError):
            return
        reason.update("")
        reason.display = False
