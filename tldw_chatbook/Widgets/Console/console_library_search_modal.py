"""Manual, one-shot Console Library search modal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .console_rag_settings_modal import (
    CONSOLE_RAG_DEFAULT_SOURCE_TYPES,
    CONSOLE_RAG_SOURCE_SUMMARY_PREFIX,
    ConsoleRagSettingsModal,
    console_rag_source_toggle_label,
    normalize_console_rag_source_types,
)


@dataclass(frozen=True, slots=True)
class ConsoleLibrarySearchResult:
    """One-shot Library search request returned by the modal."""

    query: str
    run: bool
    source_types: tuple[str, ...] = CONSOLE_RAG_DEFAULT_SOURCE_TYPES


class ConsoleLibrarySearchModal(ConsoleRagSettingsModal):
    """Search now without reading or changing standing Library policy."""

    DEFAULT_CSS = """
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

    def _run_result(self) -> ConsoleLibrarySearchResult:
        return ConsoleLibrarySearchResult(
            query=self._current_query(),
            run=True,
            source_types=self._source_types,
        )


__all__ = [
    "CONSOLE_RAG_DEFAULT_SOURCE_TYPES",
    "CONSOLE_RAG_SOURCE_SUMMARY_PREFIX",
    "ConsoleLibrarySearchModal",
    "ConsoleLibrarySearchResult",
    "console_rag_source_toggle_label",
    "normalize_console_rag_source_types",
]
