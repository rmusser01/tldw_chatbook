"""Shared Chat handoff helpers for RAG and web search results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any
from urllib.parse import quote

from ....Chat.citation_evidence_models import (
    EVIDENCE_STATUSES,
    EvidenceBundle,
    EvidenceReference,
)
from ....Chat.chat_handoff_models import ChatHandoffPayload
from ....Utils.input_validation import sanitize_string, validate_text_input


LIBRARY_RAG_PAYLOAD_TEXT_MAX_LENGTH = 1_000
LIBRARY_RAG_PAYLOAD_QUERY_MAX_LENGTH = 2_000
LIBRARY_RAG_PAYLOAD_SNIPPET_MAX_LENGTH = 4_000
LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH = 512


def _normalize_runtime_backend(value: Any) -> str:
    backend = str(value or "local").strip().lower()
    return backend if backend in {"local", "server"} else "local"


def _metadata_for_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(result.get("metadata") or {})
    if "score" in result:
        metadata["score"] = result.get("score")
    if result.get("citations"):
        metadata["citations"] = result.get("citations")
    if result.get("url") and not metadata.get("url"):
        metadata["url"] = result.get("url")
    return metadata


def _result_provenance(result: Any) -> dict[str, Any]:
    provenance = _result_value(result, "provenance", {})
    return dict(provenance) if isinstance(provenance, Mapping) else {}


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _result_or_provenance_value(result: Any, key: str, default: Any = None) -> Any:
    value = _result_value(result, key, None)
    if value not in (None, ""):
        return value
    return _result_provenance(result).get(key, default)


def _validated_payload_text(
    value: Any,
    *,
    fallback: str = "",
    max_length: int = LIBRARY_RAG_PAYLOAD_TEXT_MAX_LENGTH,
) -> str:
    sanitized = sanitize_string(str(value or ""), max_length=max_length)
    text = " ".join(sanitized.strip().split())
    if not text:
        return fallback
    if not validate_text_input(text, max_length=max_length, allow_html=False):
        return fallback
    return text


def _safe_target_part(value: Any) -> str:
    text = _validated_payload_text(
        value,
        fallback="result",
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )
    return quote(text, safe="._:-") or "result"


def _library_rag_result_id(result: Any) -> str:
    explicit_result_id = _validated_payload_text(
        _result_or_provenance_value(result, "result_id"),
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )
    if explicit_result_id:
        return explicit_result_id
    source_id = _validated_payload_text(
        _result_or_provenance_value(result, "source_id"),
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )
    chunk_id = _validated_payload_text(
        _result_or_provenance_value(result, "chunk_id"),
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )
    if source_id and chunk_id:
        return f"{source_id}:{chunk_id}"
    if source_id:
        return source_id
    if chunk_id:
        return chunk_id
    return f"result:{_safe_target_part(_result_value(result, 'title') or 'untitled')}"


def _library_rag_citation_labels(result: Any) -> list[str]:
    citation_labels = _result_value(result, "citation_labels")
    if citation_labels:
        labels: list[str] = []
        for label in citation_labels:
            cleaned = _validated_payload_text(label)
            if cleaned:
                labels.append(cleaned)
        return labels

    labels: list[str] = []
    citations = _result_value(result, "citations", ()) or ()
    if not isinstance(citations, (list, tuple)):
        citations = (citations,)
    for citation in citations:
        if isinstance(citation, dict):
            label = (
                citation.get("label")
                or citation.get("title")
                or citation.get("url")
                or citation.get("source_id")
            )
        else:
            label = getattr(citation, "label", citation)
        cleaned = _validated_payload_text(label)
        if cleaned:
            labels.append(cleaned)
    return labels


def _library_rag_source_authority(runtime_backend: Any) -> str:
    backend = _validated_payload_text(runtime_backend).lower()
    return "server" if backend.startswith("server") or "server" in backend else "local"


def _library_rag_runtime_backend(result: Any) -> str:
    return (
        _validated_payload_text(
            _result_or_provenance_value(result, "runtime_backend")
        )
        or "local"
    )


def _library_rag_target_id(*, source_authority: str, result_id: str) -> str:
    return f"{source_authority}:library-rag:{_safe_target_part(result_id)}"


def _library_rag_title(result: Any) -> str:
    return _validated_payload_text(
        _result_or_provenance_value(result, "title")
        or _result_or_provenance_value(result, "document_title")
        or _result_or_provenance_value(result, "source_title")
        or "",
        fallback="Untitled source",
    )


def _library_rag_score(result: Any) -> float | None:
    value = _result_or_provenance_value(result, "score")
    if value in (None, ""):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score < 0 or score > 1:
        return None
    return score


def _text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_values = value
    elif value not in (None, ""):
        raw_values = (value,)
    else:
        raw_values = ()
    normalized: list[str] = []
    for raw_value in raw_values:
        text = _validated_payload_text(
            raw_value,
            max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
        )
        if text:
            normalized.append(text)
    return tuple(dict.fromkeys(normalized))


def _library_rag_workspace_ids(result: Any) -> tuple[str, ...]:
    workspace_ids = _text_tuple(_result_or_provenance_value(result, "workspace_ids"))
    workspace_id = _validated_payload_text(
        _result_or_provenance_value(result, "workspace_id"),
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )
    if workspace_id:
        workspace_ids = (*workspace_ids, workspace_id)
    return tuple(dict.fromkeys(workspace_ids))


def _library_rag_active_workspace_id(result: Any) -> str:
    return _validated_payload_text(
        _result_or_provenance_value(result, "active_workspace_id"),
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "enabled", "eligible"}:
        return True
    if text in {"0", "false", "no", "n", "disabled", "blocked", "ineligible"}:
        return False
    return None


def _library_rag_active_context_state(result: Any) -> tuple[bool, str]:
    explicit_eligible = _coerce_optional_bool(
        _result_or_provenance_value(result, "active_context_eligible")
    )
    explicit_reason = _validated_payload_text(
        _result_or_provenance_value(result, "eligibility_reason"),
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )
    if explicit_eligible is not None:
        return explicit_eligible, explicit_reason or (
            "explicit_eligible" if explicit_eligible else "explicit_blocked"
        )

    workspace_ids = _library_rag_workspace_ids(result)
    if not workspace_ids:
        return True, "unscoped"

    active_workspace_id = _library_rag_active_workspace_id(result)
    if not active_workspace_id:
        return False, "no_active_workspace"
    if active_workspace_id in workspace_ids:
        return True, "active_workspace_match"
    return False, "cross_workspace"


def _library_rag_evidence_status(result: Any, *, active_context_eligible: bool) -> str:
    if not active_context_eligible:
        return "blocked"
    raw_status = _validated_payload_text(
        _result_or_provenance_value(result, "evidence_status")
        or _result_or_provenance_value(result, "status"),
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    ).lower()
    return raw_status if raw_status in EVIDENCE_STATUSES else "available"


def _library_rag_source_type(result: Any) -> str:
    return _validated_payload_text(
        _result_or_provenance_value(result, "source_type")
        or _result_or_provenance_value(result, "item_type")
        or _result_or_provenance_value(result, "type"),
        fallback="library-rag",
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )


def _library_rag_authority_label(
    *,
    result: Any,
    source_authority: str,
    workspace_ids: tuple[str, ...],
    active_workspace_id: str,
    active_context_eligible: bool,
) -> str:
    explicit_label = _validated_payload_text(
        _result_or_provenance_value(result, "authority_label"),
        max_length=LIBRARY_RAG_PAYLOAD_TEXT_MAX_LENGTH,
    )
    if explicit_label:
        return explicit_label
    if workspace_ids:
        workspace_label = f"Workspace: {', '.join(workspace_ids)}"
        if not active_context_eligible and active_workspace_id:
            return f"{workspace_label} (blocked for active workspace {active_workspace_id})"
        if not active_context_eligible:
            return f"{workspace_label} (blocked until a workspace is active)"
        return workspace_label
    return f"Source authority: {source_authority}"


def _library_rag_result_sequence(results: Any) -> tuple[Any, ...]:
    if isinstance(results, Sequence) and not isinstance(
        results,
        (str, bytes, bytearray, Mapping),
    ):
        return tuple(results)
    if results is None:
        return ()
    return (results,)


def _library_rag_bundle_status(references: Sequence[EvidenceReference]) -> str:
    statuses = {reference.status for reference in references}
    for status in ("available", "blocked", "stale", "unknown", "missing"):
        if status in statuses:
            return status
    return "missing"


def build_library_rag_evidence_bundle(
    results: Any,
    *,
    query: Any,
) -> EvidenceBundle:
    """Build a durable evidence bundle from Library Search/RAG results.

    Args:
        results: One result row or a sequence of result rows from Library Search/RAG.
        query: User query that produced the evidence.

    Returns:
        EvidenceBundle preserving source identity, snippets, source authority,
        workspace eligibility, and active-context availability status.
    """

    query_text = _validated_payload_text(
        query,
        max_length=LIBRARY_RAG_PAYLOAD_QUERY_MAX_LENGTH,
    )
    references: list[EvidenceReference] = []
    for index, result in enumerate(_library_rag_result_sequence(results), start=1):
        result_id = _library_rag_result_id(result)
        runtime_backend = _library_rag_runtime_backend(result)
        source_authority = _library_rag_source_authority(runtime_backend)
        source_id = _validated_payload_text(
            _result_or_provenance_value(result, "source_id"),
            fallback=result_id,
            max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
        )
        chunk_id = _validated_payload_text(
            _result_or_provenance_value(result, "chunk_id"),
            max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
        )
        workspace_ids = _library_rag_workspace_ids(result)
        active_workspace_id = _library_rag_active_workspace_id(result)
        (
            active_context_eligible,
            eligibility_reason,
        ) = _library_rag_active_context_state(result)
        status = _library_rag_evidence_status(
            result,
            active_context_eligible=active_context_eligible,
        )
        target_id = _library_rag_target_id(
            source_authority=source_authority,
            result_id=result_id,
        )
        metadata = {
            **_result_provenance(result),
            "result_id": result_id,
            "chunk_id": chunk_id,
            "citations": _library_rag_citation_labels(result),
            "runtime_backend": runtime_backend,
            "source_authority": source_authority,
            "source_selector_state": source_authority,
            "workspace_ids": workspace_ids,
            "active_workspace_id": active_workspace_id,
            "global_browse_visible": True,
            "active_context_eligible": active_context_eligible,
            "eligibility_reason": eligibility_reason,
        }
        references.append(
            EvidenceReference(
                evidence_id=f"S{index}",
                source_id=source_id,
                source_type=_library_rag_source_type(result),
                title=_library_rag_title(result),
                snippet=_validated_payload_text(
                    _result_value(result, "snippet"),
                    max_length=LIBRARY_RAG_PAYLOAD_SNIPPET_MAX_LENGTH,
                ),
                authority_label=_library_rag_authority_label(
                    result=result,
                    source_authority=source_authority,
                    workspace_ids=workspace_ids,
                    active_workspace_id=active_workspace_id,
                    active_context_eligible=active_context_eligible,
                ),
                workspace_id=workspace_ids[0] if workspace_ids else None,
                source_owner=source_authority,
                content_ref=target_id,
                status=status,
                score=_library_rag_score(result),
                metadata=metadata,
            )
        )

    available_count = sum(reference.status == "available" for reference in references)
    blocked_count = sum(reference.status == "blocked" for reference in references)
    bundle_status = _library_rag_bundle_status(references)
    first_runtime_backend = (
        str(references[0].metadata.get("runtime_backend") or "")
        if references
        else ""
    )
    return EvidenceBundle(
        bundle_id=f"library-rag:{_safe_target_part(query_text or 'evidence')}",
        query=query_text,
        references=tuple(references),
        status=bundle_status,
        metadata={
            "runtime_backend": first_runtime_backend,
            "eligible_reference_count": available_count,
            "blocked_reference_count": blocked_count,
            "global_browse_visible": True,
        },
    )


def build_library_rag_console_live_work_payload(
    result: Any,
    *,
    query: Any,
) -> dict[str, Any]:
    """Build a Console live-work payload for one Library Search/RAG evidence row.

    Args:
        result: Library Search/RAG result object or mapping containing evidence
            metadata such as title, snippet, source ID, chunk ID, runtime backend,
            score, and citations.
        query: User-provided Library Search/RAG query to preserve with the
            staged Console payload after centralized validation.

    Returns:
        A sanitized Console live-work payload containing a stable target ID,
        evidence metadata, citation labels, runtime authority, and source
        selector state. The payload also includes a serialized `EvidenceBundle`
        so downstream Console flows can preserve snippets and authority labels.
    """
    result_id = _library_rag_result_id(result)
    runtime_backend = _library_rag_runtime_backend(result)
    source_authority = _library_rag_source_authority(runtime_backend)
    return {
        "target_id": _library_rag_target_id(
            source_authority=source_authority,
            result_id=result_id,
        ),
        "result_id": result_id,
        "query": _validated_payload_text(
            query,
            max_length=LIBRARY_RAG_PAYLOAD_QUERY_MAX_LENGTH,
        ),
        "title": _library_rag_title(result),
        "source_id": _validated_payload_text(
            _result_or_provenance_value(result, "source_id"),
            max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
        ),
        "chunk_id": _validated_payload_text(
            _result_or_provenance_value(result, "chunk_id"),
            max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
        ),
        "snippet": _validated_payload_text(
            _result_value(result, "snippet"),
            max_length=LIBRARY_RAG_PAYLOAD_SNIPPET_MAX_LENGTH,
        ),
        "citations": _library_rag_citation_labels(result),
        "score": _library_rag_score(result),
        "runtime_backend": runtime_backend,
        "source_authority": source_authority,
        "source_selector_state": source_authority,
        "evidence_bundle": build_library_rag_evidence_bundle(
            result,
            query=query,
        ).to_payload(),
    }


def build_search_chat_handoff_payload(
    result: dict[str, Any],
    *,
    runtime_backend: Any = "local",
) -> ChatHandoffPayload:
    metadata = _metadata_for_result(result)
    source_kind = str(result.get("source") or "unknown").lower()
    is_web = source_kind in {"web", "web_search", "search-web"}
    normalized_backend = _normalize_runtime_backend(runtime_backend)
    source_owner = "server" if is_web else normalized_backend
    source_selector_state = "server" if is_web else normalized_backend
    source_id = str(
        metadata.get("document_id")
        or metadata.get("chunk_id")
        or metadata.get("url")
        or result.get("id")
        or result.get("source_id")
        or ""
    )
    body = str(result.get("content") or result.get("snippet") or "")

    return ChatHandoffPayload.from_source_content(
        source="search-web" if is_web else "search-rag",
        item_type="web-result" if is_web else "rag-result",
        title=str(result.get("title") or "Search Result"),
        body=body,
        content_ref=f"{'search-web' if is_web else 'search-rag'}:{source_id}" if source_id else None,
        source_id=source_id or None,
        display_summary=body[:240],
        suggested_prompt=(
            "Use this web result as source context and preserve attribution in your answer."
            if is_web
            else "Use this retrieved result as context and answer or reason from it carefully."
        ),
        runtime_backend=normalized_backend,
        source_owner=source_owner,
        source_selector_state=source_selector_state,
        discovery_owner="web_search" if is_web else "rag_search",
        discovery_entity_id=source_id or None,
        scope_type="global",
        metadata=metadata,
    )
