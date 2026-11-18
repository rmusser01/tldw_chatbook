"""Manual, one-shot Console Library search modal."""

from __future__ import annotations

from typing import Any, Sequence

from tldw_chatbook.Library.library_rag_state import (
    CONSOLE_LIBRARY_SEARCH_QUERY_MAX_CHARS,
    CONSOLE_RAG_DEFAULT_SOURCE_TYPES,
    CONSOLE_RAG_SOURCE_SUMMARY_PREFIX,
    ConsoleLibrarySearchResult,
    normalize_console_rag_source_types,
    sanitize_console_library_rag_query,
)

from .console_rag_settings_modal import (
    ConsoleRagSettingsModal,
    console_rag_source_toggle_label,
)

class ConsoleLibrarySearchModal(ConsoleRagSettingsModal):
    """Search now without reading or changing standing Library policy."""

    BUNDLED_CSS = """
    ConsoleLibrarySearchModal {
        align: center middle;
    }

    ConsoleLibrarySearchModal #console-rag-settings {
        width: 64;
        max-width: 96%;
        height: 20;
        max-height: 96%;
    }
    """

    def __init__(
        self,
        *,
        query: str = "",
        source_types: Sequence[str] = CONSOLE_RAG_DEFAULT_SOURCE_TYPES,
        item_scope_summary: str = "Scope: everything",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            query=query,
            source_types=source_types,
            item_scope_summary=item_scope_summary,
            rag_active=False,
            staged_title="",
            **kwargs,
        )

    def _status_copy(self) -> str:
        return (
            "Manual search for this send only. It stages matching evidence "
            "without changing Automatic retrieval or assistant Library access."
        )

    def _scope_summary(self) -> str:
        """Label source filters as local to this one manual search."""
        return f"This search only · {super()._scope_summary()}"

    def _can_run(self, query: str) -> bool:
        """Gate one-shot search on the same validation used at execution."""
        return bool(sanitize_console_library_rag_query(query)) and bool(
            self._source_types
        )

    def _run_result(self) -> ConsoleLibrarySearchResult:
        return ConsoleLibrarySearchResult(
            query=self._current_query(),
            run=True,
            source_types=self._source_types,
        )


__all__ = [
    "CONSOLE_RAG_DEFAULT_SOURCE_TYPES",
    "CONSOLE_RAG_SOURCE_SUMMARY_PREFIX",
    "CONSOLE_LIBRARY_SEARCH_QUERY_MAX_CHARS",
    "ConsoleLibrarySearchModal",
    "ConsoleLibrarySearchResult",
    "console_rag_source_toggle_label",
    "normalize_console_rag_source_types",
    "sanitize_console_library_rag_query",
]
