"""Scope normalization for Recurring Question scheduled tasks.

Ported from tldw_server `recurring_question_scope.py` @ 5921014aa9 — byte-parity
except imports; regenerate the parity tests when the server module changes
(spec §7.1 drift rule).

Function docstrings (Args/Returns sections) are a local addition on top of
the ported bodies and are excluded from the parity comparison -- the parity
claim above covers behavior/code, not docstrings. A future re-sync should
diff from each `def` body, not from the docstring text.
"""

from __future__ import annotations

from typing import Any

# Inlined from tldw_server `recurring_question_models.py` @ 5921014aa9
# (only the two constants this module needs).
DEFAULT_SEARCHABLE_SOURCES = ("media_db", "notes", "chats")
SUPPORTED_SCOPE_FIELDS = {
    "mode",
    "sources",
    "collection_ids",
    "tag_ids",
    "saved_search_ids",
    "source_types",
    "date_window",
    "workspace_id",
    "advanced_filters",
}

# Maps server source names onto the retrieval engine vocabulary
# (the `rag_service` docstring's own vocabulary).
_ENGINE_SOURCE_TYPE_MAP = {
    "media_db": "media",
    "notes": "note",
    "chats": "conversation",
}


def normalize_recurring_question_scope(
    scope: Any,
    *,
    available_sources: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize a Recurring Question scope without binding to source-specific UI.

    Args:
        scope: The raw scope dict from a definition's `config.scope` field
            (or any value -- non-dict input normalizes as an empty scope).
        available_sources: The readable searchable source names to resolve
            against. Defaults to `DEFAULT_SEARCHABLE_SOURCES` when omitted.

    Returns:
        A `(normalized_scope, errors, warnings)` tuple. `normalized_scope`
        always carries a `mode`; `errors` and `warnings` are lists of
        field-coded dicts describing any unsupported fields, an unsupported
        mode, an empty resolved scope, or unavailable requested sources.
    """
    readable_sources = list(dict.fromkeys(available_sources or DEFAULT_SEARCHABLE_SOURCES))
    raw_scope = scope if isinstance(scope, dict) else {}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for field in raw_scope:
        if field not in SUPPORTED_SCOPE_FIELDS:
            errors.append(
                {
                    "field": f"config.scope.{field}",
                    "code": "unsupported",
                    "message": f"Unsupported scope field: {field}",
                }
            )

    mode = raw_scope.get("mode")
    if mode is not None and mode not in {"all_searchable_library", "sources"}:
        return {
            "mode": str(mode),
        }, [
            {
                "field": "config.scope.mode",
                "code": "unsupported",
                "message": f"Unsupported scope mode: {mode}",
            }
        ], warnings

    if mode == "all_searchable_library" or (mode is None and "sources" not in raw_scope):
        normalized = {
            "mode": "all_searchable_library",
            "resolved_sources": readable_sources,
        }
        if not readable_sources:
            errors.append(_scope_empty_error())
        return normalized, errors, warnings

    requested_sources = _string_list(raw_scope.get("sources"))
    resolved_sources: list[str] = []
    for source in requested_sources:
        if source in readable_sources:
            resolved_sources.append(source)
        else:
            warnings.append({"code": "source_unavailable", "source": source})

    normalized = {"mode": "sources", "sources": list(dict.fromkeys(resolved_sources))}
    for field in (
        "collection_ids",
        "tag_ids",
        "saved_search_ids",
        "source_types",
        "date_window",
        "workspace_id",
        "advanced_filters",
    ):
        if field in raw_scope:
            normalized[field] = raw_scope[field]

    if not normalized["sources"]:
        errors.append(_scope_empty_error())
    return normalized, errors, warnings


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _scope_empty_error() -> dict[str, str]:
    return {
        "field": "config.scope",
        "code": "scope_empty",
        "message": "Scope must include at least one readable searchable source.",
    }


def engine_source_types(normalized_scope: dict[str, Any]) -> tuple[str, ...]:
    """Map a normalized scope's source names onto retrieval engine source types.

    Reads `resolved_sources` (all-library mode) or `sources` (sources mode).
    Unknown source names are skipped rather than raised.

    Args:
        normalized_scope: A scope dict as returned by
            `normalize_recurring_question_scope`.

    Returns:
        A tuple of retrieval-engine source-type strings (e.g. `"media"`,
        `"note"`, `"conversation"`), in the order the source names appeared.
    """
    names = normalized_scope.get("resolved_sources") or normalized_scope.get("sources") or []
    return tuple(_ENGINE_SOURCE_TYPE_MAP[name] for name in names if name in _ENGINE_SOURCE_TYPE_MAP)
