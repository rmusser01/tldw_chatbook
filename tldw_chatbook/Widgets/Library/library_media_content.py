"""Persistent content widgets for the Library media viewer."""

import asyncio

from typing import Any

from textual.app import ComposeResult
from textual.containers import (
    Container,
    Horizontal,
    ScrollableContainer,
    Vertical,
    VerticalScroll,
)
from textual.widgets import Button, Input, Markdown, Static

from tldw_chatbook.Utils.markdown_parsing import front_matter_parser_factory
from tldw_chatbook.Widgets.Library.library_media_raw_view import VirtualizedRawContent


def build_raw_content_match_lines(content: str, query: str) -> tuple[int, ...]:
    """Return the SOURCE line indexes whose text contains ``query``.

    task-22500: the virtualized view styles matches per rendered row, so
    the whole-document ``Text`` this used to build (an O(document) pass on
    every query change) is gone. Only the line list survives -- which is
    all navigation and the "N of M" status ever consumed.

    Args:
        content: Source text to search.
        query: Case-insensitive search text to match.

    Returns:
        Ascending source-line indexes whose text contains ``query``,
        case-insensitively; empty when the query or content is blank.
    """
    normalized = query.strip().lower()
    if not normalized or not content:
        return ()
    return tuple(
        index
        for index, line in enumerate(content.split("\n"))
        if normalized in line.lower()
    )


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
        self._markdown_scroll: VerticalScroll | None = None
        self._desired_mode = self._normalize_mode(mode)
        self._mount_lock = asyncio.Lock()
        self._query = query
        self._match_index = match_index

    @property
    def raw_view(self) -> VirtualizedRawContent | None:
        """The mounted virtualized Raw view.

        Returns:
            The mounted ``VirtualizedRawContent``, or ``None`` if Raw mode
            has never been mounted yet.
        """
        return self._raw_widget

    @property
    def scroller(self) -> ScrollableContainer:
        """The scroller for the CURRENT mode.

        Callers used to query this container as a VerticalScroll inside
        try/except; when the type stopped matching they silently no-opped
        and the reader quietly lost scroll restoration.

        Returns:
            The Raw view when Raw is the active, mounted mode; the
            Rendered scroller when it is mounted; otherwise this container
            itself (before either mode has mounted).
        """
        if self._desired_mode == "raw" and self._raw_widget is not None:
            return self._raw_widget
        if self._markdown_scroll is not None:
            return self._markdown_scroll
        return self

    def compose(self) -> ComposeResult:
        """Construct only the selected initial content view."""
        if self._desired_mode == "rendered":
            self._markdown_widget = self._build_markdown_widget()
            self._markdown_scroll = VerticalScroll(
                self._markdown_widget,
                id="library-media-viewer-content-markdown-scroll",
            )
            yield self._markdown_scroll
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
            if self._markdown_scroll is not None:
                self._markdown_scroll.display = desired == "rendered"

    def sync_search(self, query: str, match_index: int) -> None:
        """Forward a search update to the mounted virtualized Raw view.

        task-22500: the whole-document highlight ``Text`` this used to
        rebuild is gone -- ``VirtualizedRawContent`` restyles each row it
        paints from ``query``/``match_index`` directly, so there is nothing
        document-sized left to build or cache here. The match-line list is
        still derived (one O(document) pass, same as before) because the
        Raw view needs it to know WHICH occurrence is the active one --
        ``set_match_lines`` is called first so the active/plain distinction
        is correct on the very next repaint ``sync_search`` triggers.

        Args:
            query: Case-insensitive search text to highlight.
            match_index: Zero-based index of the active matching line.

        Returns:
            None.
        """
        self._query = query
        self._match_index = match_index
        if self._raw_widget is not None:
            self._raw_widget.set_match_lines(
                build_raw_content_match_lines(self.content, query)
            )
            self._raw_widget.sync_search(query, match_index)

    async def _ensure_mode_mounted(self, mode: str) -> None:
        """Mount the requested view only when it has not been constructed."""
        if mode == "raw":
            if self._raw_widget is None:
                self._raw_widget = self._build_raw_widget()
                await self.mount(self._raw_widget)
            return
        if self._markdown_scroll is None:
            self._markdown_widget = self._build_markdown_widget()
            self._markdown_scroll = VerticalScroll(
                self._markdown_widget,
                id="library-media-viewer-content-markdown-scroll",
            )
            await self.mount(self._markdown_scroll)

    def _normalize_mode(self, mode: str) -> str:
        """Validate a mode and force non-Markdown bodies to their Raw view."""
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unsupported Library media content mode: {mode!r}")
        return mode if self.is_markdown else "raw"

    def _build_raw_widget(self) -> VirtualizedRawContent:
        """Construct the virtualized Raw content view for the current state.

        Returns:
            A ``VirtualizedRawContent`` seeded with the body's current
            content, search query, and active match index.
        """
        return VirtualizedRawContent(
            content=self.content,
            query=self._query,
            match_index=self._match_index,
            id="library-media-viewer-content-text",
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
