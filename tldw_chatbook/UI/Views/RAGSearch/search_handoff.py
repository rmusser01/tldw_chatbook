"""Shared Chat handoff helpers for RAG and web search results."""

from __future__ import annotations

from typing import Any

from ....Chat.chat_handoff_models import ChatHandoffPayload


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
