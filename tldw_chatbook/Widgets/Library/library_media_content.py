"""Persistent content widgets for the Library media viewer."""

import asyncio

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Markdown, Static

from tldw_chatbook.Library.library_media_viewer_state import find_content_matches
from tldw_chatbook.Utils.markdown_parsing import front_matter_parser_factory


def build_raw_content_renderable(
    content: str, query: str, match_index: int
) -> Text | str:
    """Build raw text content with matching source lines highlighted.

    Args:
        content: Source text to display.
        query: Case-insensitive search text to highlight.
        match_index: Zero-based index of the active matching line.

    Returns:
        Highlighted Rich text when a match is active; otherwise, the source
        text or its empty-state message.
    """
    display_content = content or "No stored content."
    normalized_query = query.strip()
    if not normalized_query or not content:
        return display_content
    matches = find_content_matches(content, normalized_query)
    if not matches:
        return display_content
    current_line = matches[match_index % len(matches)]
    needle = normalized_query.lower()
    text = Text()
    for line_index, line in enumerate(display_content.split("\n")):
        if line_index:
            text.append("\n")
        hit = line.lower().find(needle)
        if hit < 0:
            text.append(line)
            continue
        text.append(line[:hit])
        text.append(
            line[hit : hit + len(needle)],
            style="reverse bold" if line_index == current_line else "reverse",
        )
        text.append(line[hit + len(needle) :])
    return text


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
            self._raw_widget.update(
                build_raw_content_renderable(self.content, query, match_index)
            )

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
            build_raw_content_renderable(
                self.content, self._query, self._match_index
            ),
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
