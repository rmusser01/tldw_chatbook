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
