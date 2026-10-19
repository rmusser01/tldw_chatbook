"""One bounded automatic Library search for a Console turn (TASK-33621.20).

Automatic Library preparation gives the search a short budget (5 s by
default) so a stuck Library cannot hold a send forever. The Library search
service builds its shared RAG runtime -- embedding model, vector store --
inside its first ``search`` call. That build took 4-5 s with the model
cached and 18 s without it, so the first Automatic send after every launch
spent its whole budget starting the service and paused with a timeout.

The build is therefore run first, through the service's optional
``prewarm`` hook, under its own longer bound. Only the search itself
counts against the turn's budget. A service without ``prewarm`` (test
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

from tldw_chatbook.Chat.console_turn_preparation import (
    LIBRARY_INITIALIZATION_TIMEOUT_SECONDS,
)


class LibrarySearchUnavailable(RuntimeError):
    """The Library cannot search: no service, or its runtime did not build."""


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
        LibrarySearchUnavailable: The service has no callable ``search``, or
            its ``prewarm`` reported that the runtime did not build (missing
            dependencies, a failed build). Searching anyway would only try
            the build again inside the search's short budget.
        TimeoutError: The build or the search outlasted its bound.
    """
    search = getattr(service, "search", None)
    if not callable(search):
        raise LibrarySearchUnavailable("library service unavailable")
    prewarm = getattr(service, "prewarm", None)
    if callable(prewarm):
        async with asyncio.timeout(initialization_budget):
            built = prewarm()
            if inspect.isawaitable(built):
                built = await built
        # Only an explicit False means "did not build"; a hook that returns
        # nothing (other backends, test doubles) just warmed up.
        if built is False:
            raise LibrarySearchUnavailable("library runtime unavailable")
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


def pause_interrupted_retry(controller: Any, preparation: Any) -> None:
    """Return a send whose Retry search was cancelled to a Library pause.

    TASK-33621.20 re-review: a card's Retry runs as a worker on the card, and
    swapping the transcript region out (opening the Console terminal) cancels
    it. Mid-search, that left the send PREPARING with no owner: no card, a
    shelf Discard that released nothing, and every later send refused
    ("Another send is still preparing…"). The send pauses as a stopped Library
    search instead, exactly as a Stop would leave it.

    Args:
        controller: The Console chat controller.
        preparation: The preparation the cancelled Retry had moved to
            PREPARING.
    """
    from tldw_chatbook.Chat.console_chat_controller import ConsolePreparationOutcome
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsolePreparationPauseKind,
        ConsolePreparationTransition,
        ConsoleTurnPreparationState,
        library_pause_copy,
    )

    paused = controller.store.compare_and_set_preparation(
        preparation.session_id,
        ConsolePreparationTransition(
            preparation_id=preparation.preparation_id,
            expected_state=ConsoleTurnPreparationState.PREPARING,
            new_state=ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
            new_attempt_id=None,
        ),
    )
    if paused is None:
        return
    controller._preparation_outcomes[paused.preparation_id] = ConsolePreparationOutcome(
        preparation_id=paused.preparation_id,
        attempt_id=paused.attempt_id,
        state=paused.state,
        evidence_bundle=None,
        contribution=None,
        error_code="library_retrieval_stopped",
    )
    controller._set_run_state(
        ConsoleRunState.blocked(library_pause_copy("library_retrieval_stopped")),
        session_id=paused.session_id,
    )
