"""Pure state for the unified Console Prompt Library modal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast


PromptModalMode = Literal["browse", "edit", "improve", "recipe"]
PromptSource = Literal["local", "server"]
_MODES = frozenset({"browse", "edit", "improve", "recipe"})
_SOURCES = frozenset({"local", "server"})


@dataclass(frozen=True)
class PromptBrowseResult:
    """One source-scoped page or search result."""

    source: PromptSource
    items: tuple[Mapping[str, Any], ...]
    page: int
    total_pages: int
    total_items: int

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ValueError(f"Unsupported Prompt source: {self.source!r}")
        if self.page < 1 or self.total_pages < 1 or self.total_items < 0:
            raise ValueError("Prompt Browse pagination values are invalid.")


@dataclass(frozen=True)
class ConsolePromptsState:
    """Immutable navigation, Browse, source identity, and dirty state."""

    mode_stack: tuple[PromptModalMode, ...] = ("browse",)
    focus_by_mode: tuple[tuple[PromptModalMode, str], ...] = ()
    source: PromptSource = "local"
    query: str = ""
    page: int = 1
    search_token: int = 0
    selected_source: PromptSource | None = None
    selected_identity: str | None = None
    selected_version: int | None = None
    selected_capabilities: object | None = None
    dirty: bool = False
    working_copy_unsaved: bool = False

    @classmethod
    def initial(cls) -> "ConsolePromptsState":
        """Return the canonical root Browse state."""
        return cls()

    @property
    def mode(self) -> PromptModalMode:
        return self.mode_stack[-1]

    def enter_mode(self, mode: PromptModalMode) -> "ConsolePromptsState":
        if mode not in _MODES:
            raise ValueError(f"Unsupported prompt modal mode: {mode}")
        return replace(self, mode_stack=(*self.mode_stack, mode))

    def replace_mode(self, mode: PromptModalMode) -> "ConsolePromptsState":
        if mode not in _MODES:
            raise ValueError(f"Unsupported prompt modal mode: {mode}")
        return replace(self, mode_stack=(*self.mode_stack[:-1], mode))

    def go_back(self) -> "ConsolePromptsState":
        if len(self.mode_stack) == 1:
            return self
        return replace(
            self,
            mode_stack=self.mode_stack[:-1],
            dirty=False,
            working_copy_unsaved=False,
        )

    def remember_focus(
        self, mode: PromptModalMode, widget_id: str | None
    ) -> "ConsolePromptsState":
        if not widget_id:
            return self
        remembered = dict(self.focus_by_mode)
        remembered[mode] = widget_id
        return replace(self, focus_by_mode=tuple(remembered.items()))

    def focus_for(self, mode: PromptModalMode) -> str | None:
        return dict(self.focus_by_mode).get(mode)

    def with_query(self, query: str) -> "ConsolePromptsState":
        return replace(self, query=str(query), page=1)

    def with_page(self, page: int) -> "ConsolePromptsState":
        return replace(self, page=max(1, int(page)))

    def with_source(self, source: PromptSource) -> "ConsolePromptsState":
        if source not in _SOURCES:
            raise ValueError(f"Unsupported Prompt source: {source!r}")
        return replace(
            self,
            source=cast(PromptSource, source),
            page=1,
            search_token=self.search_token + 1,
        )

    def begin_search(self) -> "ConsolePromptsState":
        return replace(self, search_token=self.search_token + 1)

    def accepts(self, token: int, source: str) -> bool:
        return token == self.search_token and source == self.source

    def select(
        self,
        *,
        identity: str,
        version: int | None,
        source: PromptSource | None = None,
        capabilities: object | None = None,
    ) -> "ConsolePromptsState":
        return replace(
            self,
            selected_source=source or self.source,
            selected_identity=str(identity),
            selected_version=version,
            selected_capabilities=capabilities,
        )

    def with_dirty(self, dirty: bool = True) -> "ConsolePromptsState":
        return replace(self, dirty=bool(dirty))

    def as_unsaved_copy(self, value: bool = True) -> "ConsolePromptsState":
        return replace(self, working_copy_unsaved=bool(value), dirty=False)


__all__ = [
    "ConsolePromptsState",
    "PromptBrowseResult",
    "PromptModalMode",
    "PromptSource",
]
