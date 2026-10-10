"""One bounded automatic Library search for a Console turn (TASK-33621.20).

Automatic Library preparation gives the search a short budget (5 s by
default) so a stuck Library cannot hold a send forever. The Library search
service builds its shared RAG runtime -- embedding model, vector store --
inside its first ``search`` call. That build took 4-5 s with the model
cached and 18 s without it, so the first Automatic send after every launch
spent its whole budget starting the service and paused with a timeout.

The build is therefore run first, through the service's optional
``warm_up`` coroutine, under its own longer bound. Only the search itself
counts against the turn's budget. A service without ``warm_up`` (test
doubles, other backends) keeps the old single-budget behaviour.

A cold build can still take many seconds, so the wait is shown and can be
stopped: the run chip reads "Searching Library…" and Stop is offered while
the search runs. Stop pauses the send exactly as a timeout does, with the
same Retry / Send once without Library / Cancel choices.

Imported lazily by the controller: nothing here runs unless an Automatic
send is preparing, so it adds no work or module to boot.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

#: How long the Library's first-use runtime build may take before the turn
#: pauses with a timeout. Generous on purpose: a first build that has to
#: download the embedding model took 18 s live.
LIBRARY_INITIALIZATION_TIMEOUT_SECONDS = 60.0


class LibrarySearchUnavailable(RuntimeError):
    """The app has no callable Library search service."""


async def run_bounded_library_search(
    service: Any,
    request: Any,
    *,
    search_budget: float,
    initialization_budget: float = LIBRARY_INITIALIZATION_TIMEOUT_SECONDS,
) -> Any:
    """Start the Library service if needed, then run one budgeted search.

    Args:
        service: The app's ``library_rag_search_service``.
        request: A ``LibraryRagSearchRequest`` (query, source types, mode,
            top_k, include_citations, scope).
        search_budget: Seconds the search itself may take.
        initialization_budget: Seconds the service's first-use build may take.

    Returns:
        The service's raw search result, for the caller to normalise.

    Raises:
        LibrarySearchUnavailable: The service has no callable ``search``.
        TimeoutError: The build or the search outlasted its bound.
    """
    search = getattr(service, "search", None)
    if not callable(search):
        raise LibrarySearchUnavailable("library service unavailable")
    warm_up = getattr(service, "warm_up", None)
    if callable(warm_up):
        async with asyncio.timeout(initialization_budget):
            await warm_up()
    kwargs: dict[str, object] = {
        "top_k": request.top_k,
        "include_citations": request.include_citations,
    }
    if request.scope is not None:
        kwargs["scope"] = request.scope
    async with asyncio.timeout(search_budget):
        raw = search(request.query, request.source_types, request.mode, **kwargs)
        if inspect.isawaitable(raw):
            raw = await raw
    return raw


#: The run chip's copy while an automatic Library search (or the Library's
#: first-use build ahead of it) runs.
LIBRARY_SEARCHING_COPY = "Searching Library…"


async def automatic_search_outcome(
    controller: Any, session_id: str, request: Any
) -> tuple[Any, str | None]:
    """Run one automatic Library search for a preparing send, stoppably.

    The search runs as its own task, registered in the controller's
    ``_library_search_tasks`` under the session, so the controller's Stop
    can cancel it (and its Stop button can show) without cancelling the send
    itself. The run chip says what the send is waiting on meanwhile.

    Args:
        controller: The Console chat controller preparing the send.
        session_id: The preparing send's session.
        request: The frozen ``LibraryRagSearchRequest``.

    Returns:
        ``(outcome, None)`` with the normalised search outcome, or
        ``(None, error_code)``: ``library_retrieval_timeout``,
        ``library_retrieval_stopped`` or ``library_retrieval_failed``.

    Raises:
        asyncio.CancelledError: The send itself was cancelled (shutdown,
            session close); only a Stop on the search becomes a pause.
    """
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleRunState,
        ConsoleRunStatus,
    )
    from tldw_chatbook.Library.library_rag_service import (
        _outcome_from_service_result,
    )

    search = asyncio.ensure_future(
        run_bounded_library_search(
            getattr(controller.app, "library_rag_search_service", None),
            request,
            search_budget=controller._library_preparation_timeout,
            initialization_budget=controller._library_initialization_timeout,
        )
    )
    searches: dict[str, asyncio.Task] = controller._library_search_tasks
    searches[session_id] = search
    searching = ConsoleRunState(ConsoleRunStatus.VALIDATING, LIBRARY_SEARCHING_COPY)
    previous = controller.run_state_for(session_id)
    try:
        controller._set_run_state(searching, session_id=session_id)
        return _outcome_from_service_result(await search), None
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if search.cancelled() and (current is None or not current.cancelling()):
            return None, "library_retrieval_stopped"
        raise
    except TimeoutError:
        return None, "library_retrieval_timeout"
    except Exception:  # noqa: BLE001 -- any search fault pauses the send
        return None, "library_retrieval_failed"
    finally:
        if searches.get(session_id) is search:
            del searches[session_id]
        if not search.done():
            search.cancel()
        # Hand the chip back: a Retry's continuation is admitted only from a
        # settled run state, and a first send resumes "Validating provider.".
        if controller.run_state_for(session_id) == searching:
            controller._set_run_state(previous, session_id=session_id)
