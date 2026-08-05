"""Console image-generation style picker modal.

Lets the user search the `/generate-image` style templates -- the 13 builtin
presets plus any user-defined templates loaded from config/the templates dir
(`Media_Creation.generation_templates.get_all_templates`, Task-559 AC4) --
and pick one to insert as an `@<style-id>` token into the Console composer
draft. Reached via the command-palette "Console: Insert image style…" action
(`ChatScreen.action_open_console_style_insert`, mirroring
`action_open_console_prompt_insert`'s guard + launch shape -- see that
method's docstring). This modal never generates an image itself; it only
inserts the token. `/generate-image @<id> ...` is what later resolves
(`console_generate_image.resolve_style_token`) and applies the style at
generation time.

Keyboard/focus discipline mirrors ``ConsoleSkillPickerModal`` exactly (see
that module's docstring for the full rationale): the filter ``Input`` keeps
focus for the whole session; Up/Down move a synthetic highlighted-row index
via a raw-key ``on_key`` intercept (``Input`` has no arrow-key bindings in
this Textual version); Enter activates the highlighted row via the bubbled
``Input.Submitted`` message; Esc dismisses with ``None``; row ``Button``
widgets and the results ``VerticalScroll`` are both ``can_focus = False`` so
real DOM focus can never land on a row.

The searched set is a small, static, in-memory list -- `get_all_templates`
is cached for the process lifetime (see that function's docstring) -- so
there is no injected async search callable, no debounce timer, and no
search-token race to guard against (nothing here ever awaits I/O). Filtering
runs synchronously on every keystroke; only the row mount/unmount itself is
awaited (Textual's ``VerticalScroll.remove_children``/``mount_all`` are
coroutines regardless of where the data came from).

Task-559 AC3 adds a template preview: a detail line below the results list
that shows the highlighted template's base-prompt/negative-prompt snippet
(truncated), updating on every highlight change (arrow keys, click, filter
re-narrow). It uses a plain (``markup=False``) ``Static`` -- template text
is untrusted (user-defined templates), and disabling markup interpretation
entirely is both simpler and strictly safer here than escaping every field,
since it makes bracket-looking content (e.g. a base prompt containing
``[red]``) render literally with no escaping step to forget.

Note: this screen only dismisses; the CALLER is responsible for returning
focus to the Console composer afterwards (mirrors every sibling Console
modal, including ``ConsoleSkillPickerModal``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from rich.markup import escape as escape_markup
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Media_Creation.generation_templates import (
    GenerationTemplate,
    get_all_templates,
)

FILTER_INPUT_ID = "console-style-picker-filter"
MODAL_ID = "console-style-picker-modal"
RESULTS_CONTAINER_ID = "console-style-picker-results"
EMPTY_STATIC_ID = "console-style-picker-empty"
DETAIL_STATIC_ID = "console-style-picker-detail"
ROW_ID_PREFIX = "console-style-picker-row-"
ROW_CLASS = "console-style-picker-row"
ROW_HIGHLIGHTED_CLASS = "console-style-picker-row-highlighted"

EMPTY_STORE_COPY = "No matching styles."
DETAIL_EMPTY_COPY = "Highlight a style to preview its prompt."

MODAL_TITLE = "Insert image style"

_PREVIEW_SNIPPET_MAX_CHARS = 90
"""Max rendered length of each of the base-prompt/negative-prompt snippets
in the detail line (each truncated independently, so a long base prompt
never crowds out the negative-prompt snippet)."""


def _template_matches(template: GenerationTemplate, needle: str) -> bool:
    """Return whether ``template`` matches an already-casefolded ``needle``.

    Args:
        template: Candidate style template (builtin or user-defined).
        needle: Casefolded filter text (empty matches everything).
    """
    if not needle:
        return True
    haystack = (template.id, template.name, template.category)
    return any(needle in field.casefold() for field in haystack)


def search_style_templates(query: str) -> list[GenerationTemplate]:
    """Filter every available template (builtin + user-defined) by id/name/category.

    Args:
        query: Raw filter text as typed into the picker's search box.

    Returns:
        Matching templates in `get_all_templates`' declared order --
        builtins first (their `BUILTIN_TEMPLATES` insertion order, or a
        user template's position if it overrode a builtin id), then any
        additional user-only templates.
    """
    needle = query.strip().casefold()
    return [
        template
        for template in get_all_templates().values()
        if _template_matches(template, needle)
    ]


def _truncate_snippet(text: str, limit: int = _PREVIEW_SNIPPET_MAX_CHARS) -> str:
    """Collapse whitespace and cut ``text`` to ``limit`` chars with a trailing ``…``.

    Mirrors ``console_generate_image.generation_content_marker``'s
    truncation shape. Returns ``""`` unchanged (callers render an
    "(none)"-style placeholder for an empty snippet).
    """
    flattened = " ".join(text.split())
    if len(flattened) > limit:
        return flattened[: limit - 1] + "…"
    return flattened


def format_style_preview(template: GenerationTemplate) -> str:
    """Render the detail-line preview text for ``template``.

    Two truncated lines: the base prompt, then the negative prompt (when
    non-empty). Plain text -- the caller MUST render it through a
    ``markup=False`` ``Static`` (or otherwise escape it) since template text
    is untrusted; this function does not escape anything itself.

    Args:
        template: The highlighted style template.

    Returns:
        Multi-line preview text, never empty (falls back to placeholder
        copy for a template with a blank base prompt, which validation
        should never actually allow through, but this stays defensive).
    """
    base = _truncate_snippet(template.base_prompt or "")
    negative = _truncate_snippet(template.negative_prompt or "")
    lines = [f"Prompt: {base}" if base else "Prompt: (none)"]
    if negative:
        lines.append(f"Negative: {negative}")
    return "\n".join(lines)


class ConsoleStylePickerModal(ModalScreen[Optional[Mapping[str, object]]]):
    """Search and pick a `/generate-image` style template (built-in or user-defined)."""

    BINDINGS = [("escape", "dismiss_picker", "Cancel")]

    def __init__(self, *, initial_query: str = "") -> None:
        """Initialize the picker.

        Args:
            initial_query: Prefilled filter text, searched immediately on
                mount without waiting for user input.
        """
        super().__init__()
        self._initial_query = initial_query
        self._results: list[GenerationTemplate] = []
        # Parallel to `_results`: the DOM id assigned to each row's Button
        # for the current render. Template ids are static, unique,
        # lowercase-with-underscore identifiers (see
        # `generation_templates.BUILTIN_TEMPLATES`), always a legal Textual
        # id suffix -- unlike the skill picker's user-controllable `name`
        # field, no duplicate/malformed-id fallback is needed here.
        self._row_ids: list[str] = []
        self._highlighted_index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id=MODAL_ID):
            yield Static(MODAL_TITLE, classes="console-modal-header")
            yield Input(
                value=self._initial_query,
                placeholder="Search styles…",
                id=FILTER_INPUT_ID,
            )
            with VerticalScroll(id=RESULTS_CONTAINER_ID, can_focus=False):
                yield Static(EMPTY_STORE_COPY, id=EMPTY_STATIC_ID, markup=False)
            # Task-559 AC3: template preview, updates on every highlight
            # change (see `_sync_highlight`). `markup=False` -- template
            # text is untrusted; see module docstring.
            yield Static(DETAIL_EMPTY_COPY, id=DETAIL_STATIC_ID, markup=False)

    async def on_mount(self) -> None:
        self._focus_filter_input()
        await self._apply_filter(self._initial_query)

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
    async def _filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        await self._apply_filter(event.value)

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

    # -- filtering ----------------------------------------------------------

    async def _apply_filter(self, query: str) -> None:
        self._results = search_style_templates(query)
        self._highlighted_index = 0
        await self._render_results()

    # -- rendering ------------------------------------------------------------

    async def _render_results(self) -> None:
        try:
            container = self.query_one(f"#{RESULTS_CONTAINER_ID}", VerticalScroll)
        except (NoMatches, QueryError):
            return  # Modal was dismissed/unmounted mid-render.
        # Awaited (mirrors ConsoleSkillPickerModal): the removal must
        # complete before mounting a same-id replacement, or a DuplicateIds
        # error can fire if the message pump hasn't caught up.
        await container.remove_children()
        self._row_ids = []
        if not self._results:
            await container.mount(Static(EMPTY_STORE_COPY, id=EMPTY_STATIC_ID, markup=False))
            self._sync_detail()  # clears a stale preview from before the filter narrowed to zero
            return
        buttons = []
        for template in self._results:
            row_id = f"{ROW_ID_PREFIX}{template.id}"
            self._row_ids.append(row_id)
            buttons.append(self._build_row_button(row_id, template))
        await container.mount_all(buttons)
        self._sync_highlight()
        # Rows may have just been (re)mounted; the filter Input must keep
        # focus regardless (see _focus_filter_input's docstring).
        self._focus_filter_input()

    def _build_row_button(self, row_id: str, template: GenerationTemplate) -> Button:
        name = escape_markup(template.name)
        category = escape_markup(template.category)
        style_id = escape_markup(template.id)
        label = f"{name} — {category} ({style_id})"
        button = Button(label, id=row_id, classes=ROW_CLASS)
        # Non-focusable: a click must never strand real DOM focus on a row
        # (see module docstring for the full rationale).
        button.can_focus = False
        return button

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
        self._sync_detail()

    def _sync_detail(self) -> None:
        """Refresh the preview detail line for the current highlight (AC3)."""
        try:
            detail = self.query_one(f"#{DETAIL_STATIC_ID}", Static)
        except (NoMatches, QueryError):
            return
        if 0 <= self._highlighted_index < len(self._results):
            template = self._results[self._highlighted_index]
            detail.update(format_style_preview(template))
        else:
            detail.update(DETAIL_EMPTY_COPY)

    def _select_highlighted(self) -> None:
        if not (0 <= self._highlighted_index < len(self._results)):
            return
        self._select_record(self._results[self._highlighted_index])

    def _select_record(self, template: GenerationTemplate) -> None:
        self.dismiss({"id": template.id, "name": template.name})
