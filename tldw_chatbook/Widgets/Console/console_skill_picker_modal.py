"""Console skill picker modal.

Lets the user search trusted, user-invocable skills and pick one to run via
the Console `/skill-name` surface (ambiguous-prefix and zero-match-with-args
cases in Task 9's dispatch open this picker prefilled with the typed word).
The caller (Task 9) supplies an already-adapted ``skill_search`` closure over
the scope service's ``get_context`` seam -- ``get_context`` has no
server-side filter, so Task 9's closure is expected to filter the returned
``available_skills`` (already trusted + user-invocable) by query client-side
and bound the result to <= 25 rows; this widget never reaches into the
scope service directly and never re-filters by trust itself.

Unlike ``ConsolePromptPickerModal`` (which has an ``apply-system`` mode whose
rows can be blocked because a prompt might lack a system part), skills have a
single mode here: every row `skill_search` returns is assumed already
eligible to run, so there is no blocked-row/refuse-reason state in this
widget. (Untrusted/absent skills are refused later, at dispatch time, via
``console_skill_resolver.SKILL_UNTRUSTED_REFUSE`` -- this picker is not in
that code path.)

Every other keyboard/focus discipline mirrors ``ConsolePromptPickerModal``
exactly (see that module's docstring for the full rationale): the filter
``Input`` keeps focus for the whole session; Up/Down move a synthetic
highlighted-row index via a raw-key ``on_key`` intercept (``Input`` has no
arrow-key bindings in this Textual version); Enter activates the highlighted
row via the bubbled ``Input.Submitted`` message; Esc dismisses with ``None``;
row ``Button`` widgets and the results ``VerticalScroll`` are both
``can_focus = False`` so real DOM focus can never land on a row (a stray
click still can't desync "what's highlighted" from "what Enter selects").

Note: this screen only dismisses; the CALLER is responsible for returning
focus to the Console composer afterwards (mirrors every sibling Console
modal, including ``ConsolePromptPickerModal``).
"""

from __future__ import annotations

import re
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

from tldw_chatbook.Chat.console_skill_resolver import SKILLS_EMPTY_LIST_ROW

SkillSearch = Callable[[str], Awaitable[list[Mapping[str, object]]]]

FILTER_INPUT_ID = "console-skill-picker-filter"
MODAL_ID = "console-skill-picker-modal"
RESULTS_CONTAINER_ID = "console-skill-picker-results"
EMPTY_STATIC_ID = "console-skill-picker-empty"
ROW_ID_PREFIX = "console-skill-picker-row-"
ROW_CLASS = "console-skill-picker-row"
ROW_HIGHLIGHTED_CLASS = "console-skill-picker-row-highlighted"

# FTS-style search debounce; ``skill_search`` itself is expected to bound
# each page to <= 25 rows (Task 9's client-side filter over `get_context`'s
# `available_skills`), this widget never asks for more nor trims a larger
# result set silently -- if the callable ever returned more than the bound,
# that's a caller bug to surface, not paper over here.
SEARCH_DEBOUNCE_SECONDS = 0.2

EMPTY_STORE_COPY = SKILLS_EMPTY_LIST_ROW

MODAL_TITLE = "Run skill"

# A row's DOM id only needs to be unique + a legal Textual identifier -- it
# is NOT required to be the record's own key. Real skill names are already
# constrained to `^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$` (lowercase
# letters/digits/hyphens, 1-64 chars -- see `tldw_api/skills_schemas.py`),
# which is always a legal + unique id suffix. This widget still guards
# defensively against a malformed/duplicate name (e.g. a stubbed test
# double) rather than trust every caller to have validated it, falling back
# to the row's index -- which is always unique and legal within a single
# render pass -- exactly like `ConsolePromptPickerModal._row_id`'s
# colon-avoidance fallback for composite string ids.
_LEGAL_ID_SUFFIX = re.compile(r"^[A-Za-z0-9_-]+$")


class ConsoleSkillPickerModal(ModalScreen[Optional[Mapping[str, object]]]):
    """Search and pick a trusted, user-invocable skill to run in the Console."""

    BINDINGS = [("escape", "dismiss_picker", "Cancel")]

    def __init__(
        self,
        *,
        initial_query: str = "",
        skill_search: SkillSearch,
    ) -> None:
        """Initialize the picker.

        Args:
            initial_query: Prefilled filter text (e.g. the ambiguous or
                unmatched ``/skill-name`` word Task 9's dispatch opened this
                picker with), searched immediately on mount without waiting
                for the debounce.
            skill_search: Async callable bound to the scope-service seam
                (Task 9) by the caller; receives the settled filter text and
                returns a bounded (<=25) list of trusted, user-invocable
                skill-summary mappings (at least ``{"name", "description"}``
                each) matching it.
        """
        super().__init__()
        self._initial_query = initial_query
        self._skill_search = skill_search
        self._results: list[Mapping[str, object]] = []
        # Parallel to `_results`: the DOM id actually assigned to each row's
        # Button for the current render (see `_row_id`'s fallback rationale).
        # Kept in lockstep with `_results` by `_render_results` alone.
        self._row_ids: list[str] = []
        self._highlighted_index = 0
        self._search_debounce_timer: Timer | None = None
        self._search_token = 0

    def compose(self) -> ComposeResult:
        with Vertical(id=MODAL_ID):
            yield Static(MODAL_TITLE, classes="console-modal-header")
            yield Input(
                value=self._initial_query,
                placeholder="Search skills…",
                id=FILTER_INPUT_ID,
            )
            with VerticalScroll(id=RESULTS_CONTAINER_ID, can_focus=False):
                yield Static(EMPTY_STORE_COPY, id=EMPTY_STATIC_ID, markup=False)

    def on_mount(self) -> None:
        self._focus_filter_input()
        # The initial (possibly ambiguous-word-prefilled) query populates the
        # list right away -- only edits made *after* opening debounce.
        self._trigger_search(self._initial_query, debounce=False)

    def _focus_filter_input(self) -> None:
        # Keyboard-first invariant: the filter Input must keep DOM focus for
        # the *whole* session, or Up/Down and typed characters stop reaching
        # it (row Buttons are can_focus=False precisely so a click can never
        # strand focus on them instead -- see module docstring).
        try:
            self.query_one(f"#{FILTER_INPUT_ID}", Input).focus()
        except (NoMatches, QueryError):
            pass

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
        for index, row_id in enumerate(self._row_ids):
            if button_id == row_id and index < len(self._results):
                self._highlighted_index = index
                self._sync_highlight()
                self._select_record(self._results[index])
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
            group="console-skill-picker-search",
        )

    async def _run_search(self, *, query: str, token: int) -> None:
        try:
            results = await self._skill_search(query)
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
        # Awaited (mirrors ConsolePromptPickerModal): the removal must
        # complete before mounting a same-id replacement, or a DuplicateIds
        # error can fire if the message pump hasn't caught up.
        await container.remove_children()
        self._row_ids = []
        if not self._results:
            await container.mount(Static(EMPTY_STORE_COPY, id=EMPTY_STATIC_ID, markup=False))
            return
        used_ids: set[str] = set()
        buttons = []
        for index, record in enumerate(self._results):
            row_id = self._row_id(record, index, used_ids)
            used_ids.add(row_id)
            self._row_ids.append(row_id)
            buttons.append(self._build_row_button(row_id, record))
        await container.mount_all(buttons)
        self._sync_highlight()
        # Rows may have just been (re)mounted; the filter Input must keep
        # focus regardless (see _focus_filter_input's docstring).
        self._focus_filter_input()

    def _build_row_button(self, row_id: str, record: Mapping[str, object]) -> Button:
        name = escape_markup(str(record.get("name") or "Untitled skill"))
        description = str(record.get("description") or "").strip()
        if description:
            label = f"{name} — {escape_markup(description)}"
        else:
            label = name
        button = Button(label, id=row_id, classes=ROW_CLASS)
        # Non-focusable: a click must never strand real DOM focus on a row
        # (see module docstring for the full rationale).
        button.can_focus = False
        return button

    def _row_id(self, record: Mapping[str, object], index: int, used_ids: set[str]) -> str:
        name = record.get("name")
        if isinstance(name, str) and name:
            candidate = f"{ROW_ID_PREFIX}{name}"
            if _LEGAL_ID_SUFFIX.match(name) and candidate not in used_ids:
                return candidate
        return f"{ROW_ID_PREFIX}idx-{index}"

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
        self._cancel_search_debounce()
        self.dismiss(record)
