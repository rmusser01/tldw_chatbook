"""Persistent content widgets for the Library media viewer."""

import asyncio

from typing import Any

from textual.app import ComposeResult
from textual.containers import (
    Container,
    Horizontal,
    ScrollableContainer,
    Vertical,
)
from textual.widgets import Button, Input, Markdown, Static

from tldw_chatbook.Library.library_media_viewer_state import find_content_matches
from tldw_chatbook.Utils.markdown_parsing import front_matter_parser_factory
from tldw_chatbook.Widgets.Library.library_media_raw_view import (
    EMPTY_CONTENT_MESSAGE,
    VirtualizedRawContent,
)


def build_raw_content_match_lines(content: str, query: str) -> tuple[int, ...]:
    """Return the SOURCE line indexes whose text contains ``query``.

    task-22500: the virtualized view styles matches per rendered row, so
    the whole-document ``Text`` this used to build (an O(document) pass on
    every query change) is gone. Only the line list survives -- which is
    all navigation and the "N of M" status ever consumed.

    FINDING 5 fix: this is a thin wrapper over
    ``library_media_viewer_state.find_content_matches`` -- the screen's own
    match-navigation scan -- rather than a second, independently maintained
    copy of the same predicate. The two used to be duplicated verbatim with
    nothing pinning them to agree; delegating makes drift between "which
    lines the status count/Prev/Next see" and "which lines the Raw view
    highlights" structurally impossible instead of merely tested-against.

    Args:
        content: Source text to search.
        query: Case-insensitive search text to match.

    Returns:
        Ascending source-line indexes whose text contains ``query``,
        case-insensitively; empty when the query or content is blank.
    """
    return find_content_matches(content, query)


class LibraryMediaContentBody(Container):
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
        self._raw_widget: VirtualizedRawContent | None = None
        self._markdown_widget: Markdown | None = None
        self._desired_mode = self._normalize_mode(mode)
        self._mount_lock = asyncio.Lock()
        self._query = query
        self._match_index = match_index
        # task-22500 FIX 1: the match-LINE memo for the CURRENTLY displayed
        # (content, query) pair, plus the pair it was built from. Mirrors
        # task-22209's memo shape exactly -- keyed on the content OBJECT
        # (``is``) and the stripped query: strings are immutable, so an
        # identity hit can never be stale, and a miss only costs the scan it
        # would have paid anyway. ``self.content`` is assigned once per
        # widget instance (a content change recomposes the viewer and builds
        # a new body), so the identity check holds for the whole navigable
        # lifetime of one document -- without this, every Prev/Next click
        # (same document, same query) re-scanned the document, reopening the
        # exact per-click cost class task-22209 shipped to close.
        self._match_lines_source: str | None = None
        self._match_lines_query: str | None = None
        self._match_lines: tuple[int, ...] = ()

    @property
    def raw_view(self) -> VirtualizedRawContent | None:
        """The mounted virtualized Raw view.

        This is a LIFETIME accessor, not a mode check: once Raw mode has
        been mounted once, it stays mounted (and this stays non-``None``)
        for the rest of the body's life, even while Rendered is the
        currently displayed mode. Use :attr:`active_mode` to ask which
        mode is actually showing right now.

        Returns:
            The mounted ``VirtualizedRawContent``, or ``None`` if Raw mode
            has never been mounted yet.
        """
        return self._raw_widget

    @property
    def active_mode(self) -> str:
        """The content mode this body is CURRENTLY displaying.

        FINDING 2 fix: ``raw_view is not None`` was being used by callers
        as a stand-in for "Raw mode is active", but it is a lifetime
        accessor (see :attr:`raw_view`) -- once Raw had been shown once, a
        subsequent Rendered<->Raw round-trip left it permanently
        non-``None``, so that check kept treating Rendered mode as if Raw
        were still on screen. This reflects the mode most recently passed
        to :meth:`sync_mode` (or the constructor), which is the actually
        visible one.

        Returns:
            ``"raw"`` or ``"rendered"``.
        """
        return self._desired_mode

    @property
    def scroller(self) -> ScrollableContainer | Container:
        """The scroller for the CURRENT mode.

        Callers used to query this container as a VerticalScroll inside
        try/except; when the type stopped matching they silently no-opped
        and the reader quietly lost scroll restoration.

        Returns:
            The Raw view when Raw is the active, mounted mode; the
            Rendered scroller when it is mounted; otherwise this container
            itself -- a plain ``Container``, not a ``ScrollableContainer``
            -- before either mode has mounted.
        """
        if self._desired_mode == "raw" and self._raw_widget is not None:
            return self._raw_widget
        # Rendered mode scrolls in THIS container (see _apply_mode_overflow):
        # the Markdown is a direct child, exactly as it was when this body
        # still was a VerticalScroll.
        return self

    def compose(self) -> ComposeResult:
        """Construct only the selected initial content view."""
        self._apply_mode_overflow(self._desired_mode)
        if self._desired_mode == "rendered":
            self._markdown_widget = self._build_markdown_widget()
            # Yielded DIRECTLY, with no scroller of its own. A child passed
            # to (or composed inside) a nested container attaches one message
            # cycle after that container, and on a 1 MB document that cycle
            # is the entire markdown parse -- the screen reported the
            # document loaded with no Markdown in the DOM yet, which made
            # TASK-22207's gate fail about half the time. Raw mode brings its
            # own ScrollView; rendered mode scrolls in this container.
            yield self._markdown_widget
            return
        self._raw_widget = self._build_raw_widget()
        yield self._raw_widget

    def _apply_mode_overflow(self, mode: str) -> None:
        """Scroll in this container for Rendered; let Raw scroll itself.

        Args:
            mode: The mode about to be displayed.

        Returns:
            None.
        """
        self.styles.overflow_y = "hidden" if mode == "raw" else "auto"

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
            desired = self._desired_mode
            await self._ensure_mode_mounted(desired)
            self._apply_mode_overflow(desired)
            if self._raw_widget is not None:
                self._raw_widget.display = desired == "raw"
            if self._markdown_widget is not None:
                self._markdown_widget.display = desired == "rendered"

    def sync_search(self, query: str, match_index: int) -> None:
        """Forward a search update to the mounted virtualized Raw view.

        task-22500: the whole-document highlight ``Text`` this used to
        rebuild is gone -- ``VirtualizedRawContent`` restyles each row it
        paints from ``query``/``match_index`` directly, so there is nothing
        document-sized left to build or cache here. The match-LINE list the
        Raw view needs (to know WHICH occurrence is the active one) is
        memoized by ``_matched_lines`` (FIX 1): a Prev/Next click passes the
        SAME query as the previous call, so it reuses the cached list
        instead of re-scanning the document -- ``set_match_lines`` is called
        first so the active/plain distinction is correct on the very next
        repaint ``sync_search`` triggers.

        Args:
            query: Case-insensitive search text to highlight.
            match_index: Zero-based index of the active matching line.

        Returns:
            None.
        """
        self._query = query
        self._match_index = match_index
        if self._raw_widget is not None:
            self._raw_widget.set_match_lines(self._matched_lines(query))
            self._raw_widget.sync_search(query, match_index)

    def _matched_lines(self, query: str) -> tuple[int, ...]:
        """Return the memoized match-line list for ``query`` against ``self.content``.

        task-22500 FIX 1: mirrors task-22209's memo shape -- keyed on the
        content OBJECT (``is``) and the stripped query, since strings are
        immutable so an identity hit can never be stale. Since ``self.content``
        is assigned once per widget instance, the identity half of the key
        only ever misses on the FIRST call for this instance; the query half
        is what actually changes across calls -- never on a Prev/Next click
        (same query), always on a newly submitted query. A blank query
        short-circuits without even calling the scanner, since there is
        nothing to find and nothing worth memoizing.

        Args:
            query: Case-insensitive search text (not yet stripped).

        Returns:
            Ascending source-line indexes matching ``query``; empty when the
            stripped query is blank.
        """
        normalized_query = query.strip()
        if not normalized_query:
            self._match_lines_source = self.content
            self._match_lines_query = ""
            self._match_lines = ()
            return self._match_lines
        if (
            self._match_lines_source is not self.content
            or self._match_lines_query != normalized_query
        ):
            self._match_lines_source = self.content
            self._match_lines_query = normalized_query
            self._match_lines = build_raw_content_match_lines(self.content, query)
        return self._match_lines

    async def _ensure_mode_mounted(self, mode: str) -> None:
        """Mount the requested view only when it has not been constructed."""
        if mode == "raw":
            if self._raw_widget is None:
                self._raw_widget = self._build_raw_widget()
                await self.mount(self._raw_widget)
            return
        if self._markdown_widget is None:
            self._markdown_widget = self._build_markdown_widget()
            # Awaited directly, so the Markdown is guaranteed to be in the
            # DOM the moment this returns.
            await self.mount(self._markdown_widget)

    def _normalize_mode(self, mode: str) -> str:
        """Validate a mode and force non-Markdown bodies to their Raw view."""
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unsupported Library media content mode: {mode!r}")
        return mode if self.is_markdown else "raw"

    def _build_raw_widget(self) -> VirtualizedRawContent:
        """Construct the virtualized Raw content view for the current state.

        task-22500 FIX 2: also seeds ``set_match_lines`` from the memoized
        match list for the body's current query. Without this, a body that
        (re)mounts with an already-active query -- e.g. a recompose during a
        live search, or the Rendered<->Raw toggle -- painted every match
        PLAIN (never bold) until the next ``sync_search`` call, because
        nothing had populated the Raw view's ``_match_lines`` yet.

        Returns:
            A ``VirtualizedRawContent`` seeded with the body's current
            content, search query, active match index, and match-line list.
        """
        widget = VirtualizedRawContent(
            content=self.content,
            query=self._query,
            match_index=self._match_index,
            id="library-media-viewer-content-text",
        )
        widget.set_match_lines(self._matched_lines(self._query))
        return widget

    def _build_markdown_widget(self) -> Markdown:
        return Markdown(
            self.content or EMPTY_CONTENT_MESSAGE,
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
        # task-28002: every child is PERSISTENT and display-gated on query
        # activity, never torn down. The old activity-flip recompose
        # destroyed this Input WHILE IT HELD FOCUS (a first submit is
        # exactly such a flip), leaving screen focus on nothing -- a
        # live-verified total keyboard deadlock, since every Escape gate
        # reads ``self.focused``. Same persistent-child idiom as the
        # task-22207 loading banner.
        is_active = bool(self.query)
        yield Input(
            value=self.query,
            placeholder=self._placeholder_text(),
            id="library-media-content-search",
        )
        status = Static(
            self._status_text(),
            id="library-media-content-search-status",
            markup=False,
        )
        status.display = is_active
        yield status
        toolbar = Horizontal(
            classes="ds-toolbar", id="library-media-content-search-nav"
        )
        toolbar.styles.height = "auto"
        toolbar.display = is_active
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
        """Synchronize query data in place; children are never recomposed.

        Args:
            is_markdown: Whether the content supports a rendered Markdown view.
            query: Submitted content-search query.
            matches: Source-line indexes matching ``query``.
            match_index: Zero-based index of the active match.

        Returns:
            None.
        """
        self.is_markdown = is_markdown
        self.query = query
        self.matches = matches
        self.match_index = match_index
        is_active = bool(query)
        # Dock-on-active (task-15774): the class stays on the persistent
        # container itself.
        self.set_class(is_active, self.ACTIVE_SEARCH_CLASS)

        search_input = self.query_one("#library-media-content-search", Input)
        search_input.value = self.query
        search_input.placeholder = self._placeholder_text()
        status = self.query_one("#library-media-content-search-status", Static)
        status.update(self._status_text())
        status.display = is_active
        self.query_one("#library-media-content-search-nav", Horizontal).display = (
            is_active
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
