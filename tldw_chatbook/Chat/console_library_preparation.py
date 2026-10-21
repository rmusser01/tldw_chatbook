"""Frozen Library preparation delegated from the size-governed controller."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING

from loguru import logger

from tldw_chatbook.Library.library_rag_service import (
    LibraryRagSearchRequest,
    _outcome_from_service_result,
)
from tldw_chatbook.UI.Views.RAGSearch.search_handoff import (
    build_library_rag_evidence_bundle,
)

from .console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparationState,
)

if TYPE_CHECKING:
    from .console_chat_controller import (
        ConsoleChatController,
        ConsolePreparationOutcome,
    )


async def prepare_library_for_turn(
    controller: ConsoleChatController, preparation_id: str
) -> ConsolePreparationOutcome:
    """Run one frozen automatic retrieval attempt and CAS its result."""

    from .console_chat_controller import ConsolePreparationOutcome

    preparation = controller._preparation_by_id(preparation_id)
    if preparation is None:
        raise KeyError(preparation_id)
    if preparation.state is not ConsoleTurnPreparationState.PREPARING:
        existing = controller._preparation_outcomes.get(preparation_id)
        if existing is not None and existing.attempt_id == preparation.attempt_id:
            return existing
        return ConsolePreparationOutcome(
            preparation_id=preparation.preparation_id,
            attempt_id=preparation.attempt_id,
            state=preparation.state,
            evidence_bundle=None,
            contribution=None,
            error_code=None,
        )

    authority = preparation.execution_context.library_authority
    request = LibraryRagSearchRequest(
        query=preparation.executed_draft,
        source_types=controller._frozen_rag_source_types(preparation.execution_context),
        mode="rag",
        top_k=controller._frozen_rag_top_k(preparation.execution_context),
        include_citations=True,
        scope=controller._automatic_scope_for_authority(authority),
    )
    error_code: str | None = None
    try:
        service = getattr(controller.app, "library_rag_search_service", None)
        await controller._prewarm_library_service(service)
        async with asyncio.timeout(controller._library_preparation_timeout):
            search = getattr(service, "search", None)
            if not callable(search):
                raise RuntimeError("library service unavailable")
            kwargs: dict[str, object] = {
                "top_k": request.top_k,
                "include_citations": request.include_citations,
            }
            if request.scope is not None:
                kwargs["scope"] = request.scope
            raw_result = search(
                request.query,
                request.source_types,
                request.mode,
                **kwargs,
            )
            if inspect.isawaitable(raw_result):
                raw_result = await raw_result
            result = _outcome_from_service_result(raw_result)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        result = None
        error_code = "library_retrieval_timeout"
    except Exception as exc:
        result = None
        error_code = "library_retrieval_failed"
        logger.warning(
            "Console Library retrieval failed | exception_type={}",
            type(exc).__name__,
        )

    results = tuple(getattr(result, "results", ()) or ()) if result else ()
    status = str(getattr(result, "status", "") or "") if result else ""
    if error_code is None and status not in {"ready", "empty"}:
        error_code = "library_retrieval_failed"
    if error_code is not None:
        logger.warning("Console Library preparation paused | error_code={}", error_code)
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
        current = paused or controller._preparation_by_id(preparation_id)
        if current is None:
            outcome = ConsolePreparationOutcome(
                preparation_id=preparation.preparation_id,
                attempt_id=preparation.attempt_id,
                state=ConsoleTurnPreparationState.CANCELLED,
                evidence_bundle=None,
                contribution=None,
                error_code=None,
            )
            return outcome
        outcome = ConsolePreparationOutcome(
            preparation_id=current.preparation_id,
            attempt_id=current.attempt_id,
            state=current.state,
            evidence_bundle=None,
            contribution=None,
            error_code=error_code if paused is not None else None,
        )
        controller._preparation_outcomes[preparation_id] = outcome
        return outcome

    bundle = (
        build_library_rag_evidence_bundle(
            results,
            query=preparation.executed_draft,
        )
        if results
        else None
    )
    ready = controller.store.compare_and_set_preparation(
        preparation.session_id,
        ConsolePreparationTransition(
            preparation_id=preparation.preparation_id,
            expected_state=ConsoleTurnPreparationState.PREPARING,
            new_state=ConsoleTurnPreparationState.READY,
            pause_kind=None,
            new_attempt_id=None,
        ),
    )
    current = ready or controller._preparation_by_id(preparation_id)
    if current is None:
        outcome = ConsolePreparationOutcome(
            preparation_id=preparation.preparation_id,
            attempt_id=preparation.attempt_id,
            state=ConsoleTurnPreparationState.CANCELLED,
            evidence_bundle=None,
            contribution=None,
            error_code=None,
        )
        return outcome
    contribution = (
        controller._preparation_contribution("zero_matches", current)
        if ready is not None and bundle is None
        else None
    )
    outcome = ConsolePreparationOutcome(
        preparation_id=current.preparation_id,
        attempt_id=current.attempt_id,
        state=current.state,
        evidence_bundle=bundle if ready is not None else None,
        contribution=contribution,
        error_code=None,
    )
    controller._preparation_outcomes[preparation_id] = outcome
    return outcome
