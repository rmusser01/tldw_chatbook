"""Service seam for Library-native Search/RAG retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import inspect
from typing import Any, Protocol

from loguru import logger

from tldw_chatbook.Chat.rag_scope import EffectiveScope
from tldw_chatbook.Library.library_rag_state import (
    LIBRARY_RAG_EMPTY_STATE_SELECTOR,
    LIBRARY_RAG_SERVICE_ERROR_SELECTOR,
    LibraryRagResultRow,
)
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.UI.destination_recovery import (
    DestinationRecoveryState,
    policy_denied_recovery_state,
)


logger = logger.bind(module="LibraryRagService")


class LibraryRagSearchService(Protocol):
    """Protocol implemented by Library Search/RAG retrieval backends."""

    async def search(
        self,
        query: str,
        source_types: tuple[str, ...],
        mode: str,
        *,
        scope: EffectiveScope | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run a Library-native retrieval request.

        Args:
            query: User question or search query to run against Library sources.
            source_types: Selected Library source type identifiers, such as
                notes or media.
            mode: Retrieval mode, currently `rag` or `search`.
            scope: Optional resolved RAG retrieval scope (rag-scope
                narrowing, task-6). Caller-passed only; `None` performs
                unrestricted retrieval.
            **kwargs: Backend-specific options such as `top_k` and
                `include_citations`.

        Returns:
            Raw retrieval backend result that can be normalized into evidence rows.
        """


@dataclass(frozen=True)
class LibraryRagSearchRequest:
    """User query and scope sent to a Library Search/RAG backend."""

    query: str
    source_types: tuple[str, ...]
    mode: str = "rag"
    top_k: int = 5
    include_citations: bool = True
    #: Optional resolved RAG retrieval scope (rag-scope narrowing, task-6).
    #: `None` (the default) performs unrestricted retrieval -- every
    #: Library-screen Search canvas call site leaves this unset (spec
    #: decision D2: the Search canvas stays deliberately unscoped). Only the
    #: Console's "Run Library RAG" seam resolves and sets this.
    scope: EffectiveScope | None = None


@dataclass(frozen=True)
class LibraryRagSearchOutcome:
    """Normalized result or recovery state for Library Search/RAG."""

    status: str
    results: tuple[LibraryRagResultRow, ...] = ()
    recovery_state: DestinationRecoveryState | None = None
    runtime_backend: str = ""
    #: Non-result-shaped notices a backend attaches to this outcome (e.g. the
    #: conversations seam being excluded under an active scope, task-6).
    #: Empty for every outcome that carries no such notice.
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


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
        # `scope=` is only added to the call when the request actually
        # carries one (task-6): backends -- and every existing test
        # fake/Protocol implementer written before `scope` existed -- were
        # never required to accept the keyword, so an always-on
        # `scope=None` would break any of them whose own positional
        # parameter happens to be named `scope` too (the source-types
        # tuple, `request.source_types`, is bound positionally). Leaving it
        # out entirely for the unscoped case (the overwhelming majority of
        # calls -- every Library-screen Search canvas call site, spec D2)
        # keeps the call shape byte-identical to before this parameter
        # existed.
        extra_kwargs: dict[str, Any] = {
            "top_k": request.top_k,
            "include_citations": request.include_citations,
        }
        if request.scope is not None:
            extra_kwargs["scope"] = request.scope
        raw_result = search(
            request.query,
            request.source_types,
            request.mode,
            **extra_kwargs,
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
        logger.opt(exception=True).warning(
            "Library Search/RAG retrieval failed.",
            mode=request.mode,
            top_k=request.top_k,
            source_types=request.source_types,
        )
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
    diagnostics = _diagnostics_from_result(raw_result)
    rows = _normalize_result_rows(raw_result, runtime_backend=runtime_backend)
    if not rows:
        return LibraryRagSearchOutcome(
            status="empty",
            recovery_state=_empty_results_recovery_state(),
            runtime_backend=runtime_backend,
            diagnostics=diagnostics,
        )
    return LibraryRagSearchOutcome(
        status="ready",
        results=rows,
        runtime_backend=runtime_backend,
        diagnostics=diagnostics,
    )


def _diagnostics_from_result(raw_result: Any) -> Mapping[str, Any]:
    if isinstance(raw_result, Mapping):
        diagnostics = raw_result.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            return diagnostics
    return {}


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
    if isinstance(raw_result, Sequence) and not isinstance(
        raw_result, (str, bytes, bytearray)
    ):
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
        stable_selector=LIBRARY_RAG_EMPTY_STATE_SELECTOR,
        disabled_tooltip=(
            "No evidence matched the current query. "
            "Revise the query or broaden the source scope."
        ),
    )


def scope_empty_recovery_state(cause: str | None) -> DestinationRecoveryState:
    """Recovery copy for the caller-side EMPTY-scope short-circuit (task-6).

    Used by the Console's "Run Library RAG" call site, which resolves the
    active conversation's effective RAG retrieval scope *before* calling
    ``run_library_rag_search``/the backend `search()` seam at all: an EMPTY
    resolution (a configured scope with nothing left to search, e.g. its
    items were deleted) means there is nothing to retrieve, so the search
    never runs and this recovery state is rendered directly instead
    (mirrors ``get_rag_context_for_chat``'s task-5 EMPTY short-circuit).

    Args:
        cause: ``EffectiveScope.cause`` explaining why resolution landed on
            EMPTY (``"no-workspace-overlap"`` or ``"deleted-items"``), or
            ``None``/unset.

    Returns:
        Recovery state whose `why` carries the exact
        "Retrieval scope is empty ({cause}); no sources searched." copy.
    """
    reason = cause or "unknown"
    message = f"Retrieval scope is empty ({reason}); no sources searched."
    return DestinationRecoveryState(
        status_label="Scope empty",
        unavailable_what="Library Search/RAG retrieval",
        why=message,
        next_action="Adjust or clear the conversation's retrieval scope",
        recovery_action="Conversation scope",
        authority_owner="Library retrieval service",
        stable_selector=LIBRARY_RAG_EMPTY_STATE_SELECTOR,
        disabled_tooltip=(
            f"{message} Adjust or clear the conversation's retrieval scope."
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
