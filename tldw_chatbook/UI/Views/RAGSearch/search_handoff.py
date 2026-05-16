"""Shared Chat handoff helpers for RAG and web search results."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

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


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


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
        _result_value(result, "result_id"),
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )
    if explicit_result_id:
        return explicit_result_id
    source_id = _validated_payload_text(
        _result_value(result, "source_id"),
        max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
    )
    chunk_id = _validated_payload_text(
        _result_value(result, "chunk_id"),
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
        selector state.
    """
    result_id = _library_rag_result_id(result)
    runtime_backend = (
        _validated_payload_text(_result_value(result, "runtime_backend")) or "local"
    )
    source_authority = _library_rag_source_authority(runtime_backend)
    return {
        "target_id": f"{source_authority}:library-rag:{_safe_target_part(result_id)}",
        "result_id": result_id,
        "query": _validated_payload_text(
            query,
            max_length=LIBRARY_RAG_PAYLOAD_QUERY_MAX_LENGTH,
        ),
        "title": _validated_payload_text(
            _result_value(result, "title")
            or _result_value(result, "document_title")
            or "",
            fallback="Untitled source",
        ),
        "source_id": _validated_payload_text(
            _result_value(result, "source_id"),
            max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
        ),
        "chunk_id": _validated_payload_text(
            _result_value(result, "chunk_id"),
            max_length=LIBRARY_RAG_PAYLOAD_ID_MAX_LENGTH,
        ),
        "snippet": _validated_payload_text(
            _result_value(result, "snippet"),
            max_length=LIBRARY_RAG_PAYLOAD_SNIPPET_MAX_LENGTH,
        ),
        "citations": _library_rag_citation_labels(result),
        "score": _result_value(result, "score"),
        "runtime_backend": runtime_backend,
        "source_authority": source_authority,
        "source_selector_state": source_authority,
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
