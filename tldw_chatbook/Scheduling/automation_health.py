"""Read-time health for local automation definitions (schedules-handoff PR-2, Task 6).

`compute_local_health` is never persisted onto the definition row (spec
§7.4): capability/permission state can change between reads (a provider
configured, a service wired up), and a stored value would go stale. Callers
recompute it every time they need it -- today that is
`SchedulingService.run_automation_now`'s pre-dispatch refusal check; a later
PR (PR-6) surfaces it in the UI.
"""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Scheduling.models import Health

_CAPABILITY_UNAVAILABLE_REASON = (
    "Library RAG search is not available in this app instance."
)
_PERMISSION_REQUIRED_REASON = (
    "No LLM provider is configured for automation execution."
)

#: Lazily populated with `automation_execution.resolve_execution_target` on
#: first real use -- `automation_execution` pulls in the Library RAG
#: answer-provider seams, heavier than this module's boot-census budget
#: (ADR-097), so it must not be imported at module scope. Kept as a real
#: module attribute (not a function-local import) so tests can monkeypatch
#: it directly instead of needing the Library seams importable at all.
resolve_execution_target: Any = None


def compute_local_health(app: Any, definition_row: dict) -> tuple[str, str]:
    """Return `(health, reason)` for one local automation definition.

    Checks, in order:

    1. `capability_unavailable` -- `app.library_rag_search_service` is
       absent or `None` (the `recurring_question` family's only retrieval
       seam today).
    2. `permission_required` -- `resolve_execution_target` resolves no
       `provider` at any of its layers (definition `input`, `[scheduling]`
       config, or the Library RAG answer-provider default).
    3. Otherwise `"ready"`, with an empty reason.
    """
    global resolve_execution_target

    if getattr(app, "library_rag_search_service", None) is None:
        return Health.CAPABILITY_UNAVAILABLE.value, _CAPABILITY_UNAVAILABLE_REASON

    if resolve_execution_target is None:
        from tldw_chatbook.Scheduling.automation_execution import (
            resolve_execution_target as _resolve_execution_target,
        )

        resolve_execution_target = _resolve_execution_target

    target = resolve_execution_target(definition_row)
    if not target.get("provider"):
        return Health.PERMISSION_REQUIRED.value, _PERMISSION_REQUIRED_REASON

    return Health.READY.value, ""
