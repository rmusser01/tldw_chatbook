"""Read-time health for local automation definitions (schedules-handoff PR-2, Task 6).

`compute_local_health` is never persisted onto the definition row (spec
§7.4): capability/permission state can change between reads (a provider
configured, a service wired up), and a stored value would go stale. Callers
recompute it every time they need it -- today that is
`SchedulingService.run_automation_now`'s pre-dispatch refusal check; a later
PR (PR-6) surfaces it in the UI.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_chatbook.Scheduling.models import Health
from tldw_chatbook.Scheduling.recurring_question_scope import (
    library_source_types,
    normalize_recurring_question_scope,
)

_CAPABILITY_UNAVAILABLE_REASON = (
    "Library RAG search is not available in this app instance."
)
_PERMISSION_REQUIRED_REASON = (
    "No LLM provider is configured for automation execution."
)

#: `library_source_types`'s Library-plural vocabulary (the same mapping
#: `automation_execution.py` uses for retrieval, via
#: `recurring_question_scope.py` -- imported above, not re-declared) mapped
#: onto the `app` attribute the real keyword-search seam
#: (`Library/library_local_rag_search_service.py`) gates each source on:
#: `_search_media`/`_search_notes` read `media_reading_scope_service`/
#: `notes_scope_service` (never `media_db`/`chachanotes_db` -- those DB
#: handles can be set while the scope service itself is unavailable, or
#: vice versa), `_search_conversations` reads `chachanotes_db` directly.
_SOURCE_DB_ATTR: dict[str, str] = {
    "media": "media_reading_scope_service",
    "notes": "notes_scope_service",
    "conversations": "chachanotes_db",
}

#: Lazily populated with `automation_execution.resolve_execution_target` on
#: first real use -- `automation_execution` pulls in the Library RAG
#: answer-provider seams, heavier than this module's boot-census budget
#: (ADR-097), so it must not be imported at module scope. Kept as a real
#: module attribute (not a function-local import) so tests can monkeypatch
#: it directly instead of needing the Library seams importable at all.
resolve_execution_target: Any = None


def _unreadable_scoped_source(app: Any, definition_row: dict) -> str | None:
    """The first scoped source (Library-plural vocabulary) whose backing
    `app` attribute is missing, or `None` when every scoped source resolves
    to a live DB attribute.

    Normalizes `definition_row["config"]["scope"]` with the same
    `normalize_recurring_question_scope` / `library_source_types` pair
    `automation_execution.py` uses for retrieval, so this check can never
    name a source that a real run would not actually query (and vice
    versa).
    """
    config = definition_row.get("config") if isinstance(definition_row, Mapping) else None
    if not isinstance(config, Mapping):
        config = {}
    normalized_scope, _errors, _warnings = normalize_recurring_question_scope(config.get("scope"))
    for source_type in library_source_types(normalized_scope):
        attr = _SOURCE_DB_ATTR.get(source_type)
        if attr is not None and getattr(app, attr, None) is None:
            return source_type
    return None


def compute_local_health(app: Any, definition_row: dict) -> tuple[str, str]:
    """Return `(health, reason)` for one local automation definition.

    Checks, in order:

    1. `capability_unavailable` -- `app.library_rag_search_service` is
       absent, `None`, or does not expose a callable `search` (the
       `recurring_question` family's only retrieval seam today; a service
       object without a callable `search` is not actually usable, however
       present the attribute is).
    2. `capability_unavailable` -- a source in the definition's scope
       (`config.scope`) resolves to a Library source type whose backing
       `app` attribute (`media_reading_scope_service`/`notes_scope_service`/
       `chachanotes_db` -- see `_SOURCE_DB_ATTR`) is missing. The reason
       names the unreadable source.
    3. `permission_required` -- `resolve_execution_target` resolves no
       `provider` at any of its layers (definition `input`, `[scheduling]`
       config, or the Library RAG answer-provider default).
    4. Otherwise `"ready"`, with an empty reason.

    Args:
        app: The running `TldwCli` app instance, checked for
            `library_rag_search_service`/`media_reading_scope_service`/
            `notes_scope_service`/`chachanotes_db` and passed through to
            `resolve_execution_target`.
        definition_row: The automation-definition row (as a dict) to
            evaluate.

    Returns:
        A `(health, reason)` tuple: `health` is one of `Health`'s string
        values (`capability_unavailable`, `permission_required`, `ready`);
        `reason` is a human-readable explanation, or `""` when `health` is
        `ready`.
    """
    global resolve_execution_target

    service = getattr(app, "library_rag_search_service", None)
    if service is None or not callable(getattr(service, "search", None)):
        return Health.CAPABILITY_UNAVAILABLE.value, _CAPABILITY_UNAVAILABLE_REASON

    unreadable_source = _unreadable_scoped_source(app, definition_row)
    if unreadable_source is not None:
        return (
            Health.CAPABILITY_UNAVAILABLE.value,
            f"The '{unreadable_source}' source is not available in this app instance.",
        )

    if resolve_execution_target is None:
        from tldw_chatbook.Scheduling.automation_execution import (
            resolve_execution_target as _resolve_execution_target,
        )

        resolve_execution_target = _resolve_execution_target

    target = resolve_execution_target(definition_row)
    if not target.get("provider"):
        return Health.PERMISSION_REQUIRED.value, _PERMISSION_REQUIRED_REASON

    return Health.READY.value, ""
