"""Persistent content widgets for the Library media viewer."""

import asyncio

from typing import Any

from rich.text import Span, Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Markdown, Static

from tldw_chatbook.Utils.markdown_parsing import front_matter_parser_factory

#: Style carried by every highlighted match…
MATCH_STYLE = "reverse"
#: …and by the one match the Prev/Next navigation currently sits on.
ACTIVE_MATCH_STYLE = "reverse bold"


class RawContentHighlightPlan:
    """The single O(document) highlight pass for one (content, query) pair.

    task-22209: match navigation used to rebuild this entire Rich ``Text``
    per Prev/Next click -- a full ``find_content_matches`` scan plus up to
    three ``Text.append`` calls for every line of the WHOLE document --
    only to move one line's highlight from ``reverse`` to ``reverse bold``.
    The highlight spans are positional and their offsets do NOT move
    between clicks (the characters are identical; that is the same fact
    task-21134's ``layout=False`` rests on), so the plan keeps the built
    ``Text`` and a click rewrites only the one or two ``Span`` entries
    whose style actually changed.

    Spans are appended in ascending line order alongside ``matches``, so
    ``text.spans[i]`` is the highlight for ``matches[i]`` -- the plan owns
    both, derived in the same pass, which is why the renderable no longer
    needs a ``find_content_matches`` scan of its own.

    Attributes:
        matches: Ascending source-line indexes of the highlighted lines.
    """

    __slots__ = ("matches", "_text", "_active_slot")

    def __init__(self, text: Text, matches: tuple[int, ...]) -> None:
        self.matches = matches
        self._text = text
        self._active_slot: int | None = None

    def renderable(self, match_index: int) -> Text:
        """Return the shared highlighted text with ``match_index`` active.

        Args:
            match_index: Zero-based index of the active match; wrapped
                modulo the match count, exactly as the status line wraps it.

        Returns:
            The plan's ``Text``. This is the SAME object across calls -- it
            is handed back to ``Static.update`` so the widget re-visualizes
            it, but nothing is rebuilt.
        """
        slot = match_index % len(self.matches)
        if slot == self._active_slot:
            return self._text
        spans = self._text.spans
        if self._active_slot is not None:
            stale = spans[self._active_slot]
            spans[self._active_slot] = Span(stale.start, stale.end, MATCH_STYLE)
        active = spans[slot]
        spans[slot] = Span(active.start, active.end, ACTIVE_MATCH_STYLE)
        self._active_slot = slot
        return self._text


def build_raw_content_highlight_plan(
    content: str, query: str
) -> RawContentHighlightPlan | str:
    """Build the reusable highlight plan for one document and query.

    This is the one document pass: the loop derives the matching line
    indexes AND their highlight spans together, so the plan's match list
    is aligned with its spans by construction (a separate
    ``find_content_matches`` scan could only re-derive the same answer).
    Matching is case-insensitive on the stripped query, identical to
    ``find_content_matches``: ``needle in line.lower()`` and
    ``line.lower().find(needle) >= 0`` are the same predicate.

    Args:
        content: Source text to display.
        query: Case-insensitive search text to highlight.

    Returns:
        A plan when the query matches at least one line; otherwise the
        plain display string to render as-is (no query, no content, or no
        matches).
    """
    display_content = content or "No stored content."
    normalized_query = query.strip()
    if not normalized_query or not content:
        return display_content
    needle = normalized_query.lower()
    needle_length = len(needle)
    text = Text()
    matches: list[int] = []
    for line_index, line in enumerate(display_content.split("\n")):
        if line_index:
            text.append("\n")
        hit = line.lower().find(needle)
        if hit < 0:
            text.append(line)
            continue
        matches.append(line_index)
        text.append(line[:hit])
        text.append(line[hit : hit + needle_length], style=MATCH_STYLE)
        text.append(line[hit + needle_length :])
    if not matches:
        return display_content
    return RawContentHighlightPlan(text, tuple(matches))


def build_raw_content_renderable(
    content: str, query: str, match_index: int
) -> Text | str:
    """Build raw text content with matching source lines highlighted.

    A one-shot wrapper over ``build_raw_content_highlight_plan``: callers
    that navigate between matches should hold the plan instead (see
    ``LibraryMediaContentBody``), because rebuilding it per click is what
    task-22209 removed.

    Args:
        content: Source text to display.
        query: Case-insensitive search text to highlight.
        match_index: Zero-based index of the active matching line.

    Returns:
        Highlighted Rich text when a match is active; otherwise, the source
        text or its empty-state message.
    """
    plan = build_raw_content_highlight_plan(content, query)
    if isinstance(plan, str):
        return plan
    return plan.renderable(match_index)


class LibraryMediaContentBody(VerticalScroll):
    """Lazily mount and retain the Raw and Rendered media content views."""

    _VALID_MODES = frozenset({"raw", "rendered"})

    def __init__(
        self,
        *,
        content: str,
        is_markdown: bool,
        mode: str,
        query: str,
        match_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.content = content
        self.is_markdown = is_markdown
        self._raw_widget: Static | None = None
        self._markdown_widget: Markdown | None = None
        self._desired_mode = self._normalize_mode(mode)
        self._mount_lock = asyncio.Lock()
        self._query = query
        self._match_index = match_index
        # task-22209: the highlight plan for the CURRENTLY displayed
        # (content, query) pair, plus the pair it was built from. Keyed on
        # the content OBJECT (``is``) and the stripped query: strings are
        # immutable, so an identity hit can never be stale, and a miss only
        # costs the rebuild it would have paid anyway. ``self.content`` is
        # assigned once per widget instance (a content change recomposes
        # the viewer and builds a new body), so the identity check holds for
        # the whole navigable lifetime of one document.
        self._highlight_source: str | None = None
        self._highlight_query: str | None = None
        self._highlight_plan: RawContentHighlightPlan | str | None = None

    def compose(self) -> ComposeResult:
        """Construct only the selected initial content view."""
        if self._desired_mode == "rendered":
            self._markdown_widget = self._build_markdown_widget()
            yield self._markdown_widget
            return
        self._raw_widget = self._build_raw_widget()
        yield self._raw_widget

    async def sync_mode(self, mode: str) -> None:
        """Show the requested view while mounting each view at most once.

        Args:
            mode: Requested content mode, either ``"raw"`` or ``"rendered"``.

        Returns:
            None.

        Raises:
            ValueError: If ``mode`` is not a supported content mode.
        """
        self._desired_mode = self._normalize_mode(mode)
        async with self._mount_lock:
            await self._ensure_mode_mounted(self._desired_mode)
            desired = self._desired_mode
            await self._ensure_mode_mounted(desired)
            if self._raw_widget is not None:
                self._raw_widget.display = desired == "raw"
            if self._markdown_widget is not None:
                self._markdown_widget.display = desired == "rendered"

    def sync_search(self, query: str, match_index: int) -> None:
        """Refresh mounted Raw content while retaining the rendered view.

        Args:
            query: Case-insensitive search text to highlight.
            match_index: Zero-based index of the active matching line.

        Returns:
            None.
        """
        self._query = query
        self._match_index = match_index
        if self._raw_widget is not None:
            # TASK-21134: ``layout=False`` -- a search refresh restyles the
            # SAME characters (the highlight plan only moves the active
            # span's style; it never adds, removes or rewraps a line), so
            # the widget's size cannot change and the layout pass
            # Static.update() arms by default was pure waste on every
            # match-nav click and every keystroke in the search box.
            self._raw_widget.update(
                self._raw_content_renderable(query, match_index),
                layout=False,
            )

    def _raw_content_renderable(self, query: str, match_index: int) -> Text | str:
        """Return the highlighted document, reusing the cached plan.

        task-22209: only a new document or a new query rebuilds the plan
        (one O(document) pass); moving between the matches of the SAME
        (document, query) pair repaints from the plan the widget already
        holds.

        Args:
            query: Case-insensitive search text to highlight.
            match_index: Zero-based index of the active matching line.

        Returns:
            The plan's shared ``Text``, or the plain display string when
            the query is blank or matches nothing.
        """
        normalized_query = query.strip()
        if (
            self._highlight_plan is None
            or self._highlight_source is not self.content
            or self._highlight_query != normalized_query
        ):
            self._highlight_source = self.content
            self._highlight_query = normalized_query
            self._highlight_plan = build_raw_content_highlight_plan(
                self.content, normalized_query
            )
        plan = self._highlight_plan
        if isinstance(plan, str):
            return plan
        return plan.renderable(match_index)

    async def _ensure_mode_mounted(self, mode: str) -> None:
        """Mount the requested view only when it has not been constructed."""
        if mode == "raw":
            if self._raw_widget is None:
                self._raw_widget = self._build_raw_widget()
                await self.mount(self._raw_widget)
            return
        if self._markdown_widget is None:
            self._markdown_widget = self._build_markdown_widget()
            await self.mount(self._markdown_widget)

    def _normalize_mode(self, mode: str) -> str:
        """Validate a mode and force non-Markdown bodies to their Raw view."""
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unsupported Library media content mode: {mode!r}")
        return mode if self.is_markdown else "raw"

    def _build_raw_widget(self) -> Static:
        return Static(
            self._raw_content_renderable(self._query, self._match_index),
            id="library-media-viewer-content-text",
            markup=False,
        )

    def _build_markdown_widget(self) -> Markdown:
        return Markdown(
            self.content or "No stored content.",
            id="library-media-viewer-content-markdown",
            parser_factory=front_matter_parser_factory(),
        )


class LibraryMediaContentSearchControls(Vertical):
    """Maintain search controls while preserving active widget identity."""

    #: Set while a query is active. The app CSS (task-15774) docks the
    #: active controls to the top of the scrolling viewer so the match
    #: count and Prev/Next stay painted at every terminal size -- at 80x24
    #: the in-flow stack above them (Back, title, metadata, section
    #: header) pushed them below the fold exactly while they were in use.
    #: An inactive search stays in flow, so no space is reserved when
    #: nobody is searching.
    ACTIVE_SEARCH_CLASS = "-library-media-search-active"

    DEFAULT_CSS = """
    LibraryMediaContentSearchControls {
        height: auto;
    }
    """

    def __init__(
        self,
        *,
        is_markdown: bool,
        query: str,
        matches: tuple[int, ...],
        match_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.is_markdown = is_markdown
        self.query = query
        self.matches = matches
        self.match_index = match_index
        self.set_class(bool(self.query), self.ACTIVE_SEARCH_CLASS)

    def compose(self) -> ComposeResult:
        yield Input(
            value=self.query,
            placeholder=self._placeholder_text(),
            id="library-media-content-search",
        )
        if not self.query:
            return
        yield Static(
            self._status_text(),
            id="library-media-content-search-status",
            markup=False,
        )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                "◀ Prev",
                id="library-media-content-search-prev",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Next ▶",
                id="library-media-content-search-next",
                classes="library-canvas-action",
                compact=True,
            )

    def sync_query_state(
        self,
        *,
        is_markdown: bool,
        query: str,
        matches: tuple[int, ...],
        match_index: int,
    ) -> None:
        """Synchronize query data, recomposing only when activity changes.

        Args:
            is_markdown: Whether the content supports a rendered Markdown view.
            query: Submitted content-search query.
            matches: Source-line indexes matching ``query``.
            match_index: Zero-based index of the active match.

        Returns:
            None.
        """
        was_active = bool(self.query)
        self.is_markdown = is_markdown
        self.query = query
        self.matches = matches
        self.match_index = match_index
        is_active = bool(query)
        # Dock-on-active (task-15774): the class is on the persistent
        # container itself, so it survives the child recompose below.
        self.set_class(is_active, self.ACTIVE_SEARCH_CLASS)

        if was_active != is_active:
            self.refresh(recompose=True)
            return

        search_input = self.query_one("#library-media-content-search", Input)
        search_input.value = self.query
        search_input.placeholder = self._placeholder_text()
        if is_active:
            self.query_one("#library-media-content-search-status", Static).update(
                self._status_text()
            )

    def sync_match_index(
        self, *, matches: tuple[int, ...], match_index: int
    ) -> None:
        """Synchronize match navigation without rebuilding controls.

        Args:
            matches: Source-line indexes matching the active query.
            match_index: Zero-based index of the active match.

        Returns:
            None.
        """
        self.matches = matches
        self.match_index = match_index
        self.query_one("#library-media-content-search-status", Static).update(
            self._status_text()
        )

    def _placeholder_text(self) -> str:
        return "Search content (raw text)…" if self.is_markdown else "Search content…"

    def _status_text(self) -> str:
        if not self.query:
            return ""
        if not self.matches:
            return "No matches"
        wrapped = self.match_index % len(self.matches)
        return f"Match {wrapped + 1} of {len(self.matches)} matches"
