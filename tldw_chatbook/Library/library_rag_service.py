"""Service seam for Library-native Search/RAG retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import inspect
from typing import Any, Protocol

from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.UI.destination_recovery import (
    DestinationRecoveryState,
    policy_denied_recovery_state,
)


LIBRARY_RAG_SERVICE_ERROR_SELECTOR = "library-rag-service-error"


class LibraryRagSearchService(Protocol):
    """Protocol implemented by Library Search/RAG retrieval backends."""

    async def search(
        self,
        query: str,
        scope: tuple[str, ...],
        mode: str,
        **kwargs: Any,
    ) -> Any:
        """Run a Library-native retrieval request."""


@dataclass(frozen=True)
class LibraryRagSearchRequest:
    """User query and scope sent to a Library Search/RAG backend."""

    query: str
    source_types: tuple[str, ...]
    mode: str = "rag"
    top_k: int = 5
    include_citations: bool = True


@dataclass(frozen=True)
class LibraryRagSearchOutcome:
    """Normalized result or recovery state for Library Search/RAG."""

    status: str
    results: tuple[LibraryRagResultRow, ...] = ()
    recovery_state: DestinationRecoveryState | None = None
    runtime_backend: str = ""


async def run_library_rag_search(
    app_instance: Any,
    request: LibraryRagSearchRequest,
) -> LibraryRagSearchOutcome:
    """Run Library Search/RAG using the configured app service.

    Args:
        app_instance: Textual app or test fake containing `library_rag_search_service`.
        request: Query, mode, and source scope.

    Returns:
        Normalized retrieval outcome with stable recovery copy on blockers.
    """

    service = getattr(app_instance, "library_rag_search_service", None)
    search = getattr(service, "search", None)
    if not callable(search):
        return LibraryRagSearchOutcome(
            status="blocked",
            recovery_state=_service_unavailable_recovery_state(),
        )

    try:
        raw_result = search(
            request.query,
            request.source_types,
            request.mode,
            top_k=request.top_k,
            include_citations=request.include_citations,
        )
        resolved_result = await _resolve_maybe_awaitable(raw_result)
        return _outcome_from_service_result(resolved_result)
    except PolicyDeniedError as exc:
        return LibraryRagSearchOutcome(
            status="blocked",
            recovery_state=policy_denied_recovery_state(
                exc,
                unavailable_what="Library Search/RAG retrieval",
                stable_selector=LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
                policy_message=exc.user_message,
            ),
        )
    except Exception:
        return LibraryRagSearchOutcome(
            status="failed",
            recovery_state=_retrieval_failed_recovery_state(),
        )


async def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _outcome_from_service_result(raw_result: Any) -> LibraryRagSearchOutcome:
    if isinstance(raw_result, LibraryRagSearchOutcome):
        return raw_result

    runtime_backend = _runtime_backend_from_result(raw_result)
    rows = _normalize_result_rows(raw_result, runtime_backend=runtime_backend)
    if not rows:
        return LibraryRagSearchOutcome(
            status="empty",
            recovery_state=_empty_results_recovery_state(),
            runtime_backend=runtime_backend,
        )
    return LibraryRagSearchOutcome(
        status="ready",
        results=rows,
        runtime_backend=runtime_backend,
    )


def _runtime_backend_from_result(raw_result: Any) -> str:
    if isinstance(raw_result, Mapping):
        return str(raw_result.get("runtime_backend") or "").strip()
    return ""


def _normalize_result_rows(
    raw_result: Any,
    *,
    runtime_backend: str,
) -> tuple[LibraryRagResultRow, ...]:
    raw_items = _raw_items_from_result(raw_result)
    rows: list[LibraryRagResultRow] = []
    for item in raw_items:
        if isinstance(item, LibraryRagResultRow):
            rows.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        values = dict(item)
        if runtime_backend and not values.get("runtime_backend"):
            values["runtime_backend"] = runtime_backend
        rows.append(LibraryRagResultRow.from_result(values))
    return tuple(rows)


def _raw_items_from_result(raw_result: Any) -> tuple[Any, ...]:
    if isinstance(raw_result, Mapping):
        for key in ("results", "items", "documents", "chunks"):
            value = raw_result.get(key)
            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                return tuple(value)
        return ()
    if isinstance(raw_result, Sequence) and not isinstance(raw_result, (str, bytes, bytearray)):
        return tuple(raw_result)
    return ()


def _service_unavailable_recovery_state() -> DestinationRecoveryState:
    return DestinationRecoveryState(
        status_label="Unavailable",
        unavailable_what="Library Search/RAG retrieval",
        why="Library RAG search service is unavailable in this runtime",
        next_action="Configure Library RAG retrieval or use standalone Search/RAG",
        recovery_action="Search/RAG setup",
        authority_owner="Library retrieval service",
        stable_selector=LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
        disabled_tooltip=(
            "Library RAG search service is unavailable in this runtime. "
            "Configure retrieval or use standalone Search/RAG."
        ),
    )


def _empty_results_recovery_state() -> DestinationRecoveryState:
    return DestinationRecoveryState(
        status_label="No results",
        unavailable_what="Library Search/RAG evidence",
        why="No evidence matched the current query",
        next_action="Revise the query or broaden the source scope",
        recovery_action="Query input or source scope",
        authority_owner="Library retrieval",
        stable_selector="library-rag-empty-state",
        disabled_tooltip=(
            "No evidence matched the current query. "
            "Revise the query or broaden the source scope."
        ),
    )


def _retrieval_failed_recovery_state() -> DestinationRecoveryState:
    return DestinationRecoveryState(
        status_label="Retrieval failed",
        unavailable_what="Library Search/RAG retrieval",
        why="Library Search/RAG retrieval failed",
        next_action="Retry the query or check Library indexing",
        recovery_action="Retry",
        authority_owner="Library retrieval",
        stable_selector=LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
        disabled_tooltip=(
            "Library Search/RAG retrieval failed. "
            "Retry the query or check Library indexing."
        ),
    )
