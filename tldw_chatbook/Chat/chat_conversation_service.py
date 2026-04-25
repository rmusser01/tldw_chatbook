from __future__ import annotations

from typing import Any, Iterable, Mapping

from tldw_chatbook.tldw_api.chat_conversation_schemas import ALLOWED_CONVERSATION_STATES


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_state(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized not in ALLOWED_CONVERSATION_STATES:
        raise ValueError(f"Invalid state '{value}'. Allowed: {', '.join(ALLOWED_CONVERSATION_STATES)}")
    return normalized


def _normalize_assistant_kind(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized in {"", "generic", "none"}:
        return None
    if normalized not in {"character", "persona"}:
        raise ValueError("assistant_kind must be 'character', 'persona', or null")
    return normalized


def _normalize_scope(scope_type: Any, workspace_id: Any) -> tuple[str, str | None]:
    normalized_workspace_id = _clean_text(workspace_id)
    raw_scope = _clean_text(scope_type)
    normalized_scope = raw_scope.lower() if raw_scope is not None else ("workspace" if normalized_workspace_id else "global")
    if normalized_scope == "global":
        return "global", None
    if normalized_scope != "workspace":
        raise ValueError("scope_type must be 'global' or 'workspace'")
    if not normalized_workspace_id:
        raise ValueError("workspace_id is required when scope_type='workspace'")
    return "workspace", normalized_workspace_id


def _normalize_runtime_backend(value: Any) -> str:
    text = _clean_text(value)
    normalized = (text or "local").lower()
    if normalized not in {"local", "server"}:
        return "local"
    return normalized


def _normalize_discovery_owner(value: Any) -> str:
    text = _clean_text(value)
    normalized = (text or "general_chat").lower()
    if normalized not in {"general_chat", "ccp_character", "ccp_persona"}:
        return "general_chat"
    return normalized


def _normalize_keywords(keyword_rows: Any) -> list[str]:
    if not keyword_rows:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in keyword_rows:
        keyword_text = item
        if isinstance(item, Mapping):
            keyword_text = item.get("keyword")
        text = _clean_text(keyword_text)
        if text is None:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _row_assistant_name(conversation_row: Mapping[str, Any]) -> str | None:
    assistant_kind = _normalize_assistant_kind(conversation_row.get("assistant_kind"))
    assistant_name = _clean_text(conversation_row.get("assistant_name"))
    assistant_id = _clean_text(conversation_row.get("assistant_id"))
    character_id = conversation_row.get("character_id")

    if assistant_name is not None:
        return assistant_name
    if assistant_kind == "character" and character_id is not None:
        return f"Character {character_id}"
    if assistant_kind == "persona" and assistant_id is not None:
        return f"Persona {assistant_id}"
    return None


def derive_conversation_title(
    *,
    assistant_kind: Any = None,
    assistant_name: Any = None,
    fallback_title: Any = None,
    character_id: Any = None,
) -> str:
    title = _clean_text(fallback_title)
    if title is not None:
        return title

    normalized_kind = _normalize_assistant_kind(assistant_kind)
    normalized_name = _clean_text(assistant_name)

    if normalized_kind == "character":
        if normalized_name is not None:
            return f"Chat with {normalized_name}"
        return "Chat with Character"

    if normalized_kind == "persona":
        if normalized_name is not None:
            return f"Chat with {normalized_name}"
        return "Chat with Persona"

    return "New Chat"


def normalize_message_row(message_row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not message_row:
        return None

    message_id = message_row.get("id")
    created_at = message_row.get("created_at") or message_row.get("timestamp")
    topology = {
        "conversation_id": message_row.get("conversation_id"),
        "parent_message_id": message_row.get("parent_message_id"),
        "sender": message_row.get("sender"),
        "role": message_row.get("role") or message_row.get("sender"),
    }
    variant = {
        "variant_of": message_row.get("variant_of"),
        "variant_number": message_row.get("variant_number"),
        "is_selected_variant": bool(message_row.get("is_selected_variant"))
        if message_row.get("is_selected_variant") is not None
        else None,
        "total_variants": message_row.get("total_variants"),
    }

    return {
        "id": message_id,
        "conversation_id": message_row.get("conversation_id"),
        "parent_message_id": message_row.get("parent_message_id"),
        "sender": message_row.get("sender"),
        "content": message_row.get("content") or "",
        "role": message_row.get("role") or message_row.get("sender"),
        "created_at": created_at,
        "timestamp": message_row.get("timestamp") or created_at,
        "last_modified": message_row.get("last_modified"),
        "version": message_row.get("version"),
        "deleted": message_row.get("deleted"),
        "client_id": message_row.get("client_id"),
        "ranking": message_row.get("ranking"),
        "image_data": message_row.get("image_data"),
        "image_mime_type": message_row.get("image_mime_type"),
        "topology": topology,
        "variant": variant,
    }


def normalize_conversation_row(
    conversation_row: Mapping[str, Any] | None,
    *,
    keywords: Iterable[Any] | None = None,
    message_count: int | None = None,
) -> dict[str, Any] | None:
    if not conversation_row:
        return None

    normalized_scope, normalized_workspace_id = _normalize_scope(
        conversation_row.get("scope_type"),
        conversation_row.get("workspace_id"),
    )
    normalized_keywords = _normalize_keywords(keywords if keywords is not None else conversation_row.get("keywords"))
    normalized_state = _normalize_state(conversation_row.get("state")) or "in-progress"
    assistant_kind = _normalize_assistant_kind(conversation_row.get("assistant_kind"))
    assistant_id = _clean_text(conversation_row.get("assistant_id"))
    character_id = conversation_row.get("character_id")
    if assistant_kind == "character" and assistant_id is None and character_id is not None:
        assistant_id = str(character_id)
    normalized_title = derive_conversation_title(
        assistant_kind=conversation_row.get("assistant_kind"),
        assistant_name=_row_assistant_name(conversation_row),
        fallback_title=conversation_row.get("title"),
    )

    return {
        "id": conversation_row.get("id"),
        "scope_type": normalized_scope,
        "workspace_id": normalized_workspace_id,
        "character_id": character_id,
        "assistant_kind": assistant_kind,
        "assistant_id": assistant_id,
        "runtime_backend": _normalize_runtime_backend(conversation_row.get("runtime_backend")),
        "discovery_owner": _normalize_discovery_owner(conversation_row.get("discovery_owner")),
        "discovery_entity_id": _clean_text(conversation_row.get("discovery_entity_id")),
        "persona_memory_mode": _clean_text(conversation_row.get("persona_memory_mode")),
        "title": normalized_title,
        "state": normalized_state,
        "topic_label": _clean_text(conversation_row.get("topic_label")),
        "topic_label_source": _clean_text(conversation_row.get("topic_label_source")),
        "topic_last_tagged_at": conversation_row.get("topic_last_tagged_at"),
        "topic_last_tagged_message_id": _clean_text(conversation_row.get("topic_last_tagged_message_id")),
        "bm25_norm": conversation_row.get("bm25_norm"),
        "last_modified": conversation_row.get("last_modified"),
        "created_at": conversation_row.get("created_at"),
        "message_count": int(message_count if message_count is not None else conversation_row.get("message_count") or 0),
        "keywords": normalized_keywords,
        "cluster_id": _clean_text(conversation_row.get("cluster_id")),
        "source": _clean_text(conversation_row.get("source")),
        "external_ref": _clean_text(conversation_row.get("external_ref")),
        "version": conversation_row.get("version"),
    }


class ChatConversationService:
    def __init__(self, db: Any):
        self.db = db

    def derive_conversation_title(self, conversation_row: Mapping[str, Any] | None) -> str:
        if not conversation_row:
            return derive_conversation_title()
        return derive_conversation_title(
            assistant_kind=conversation_row.get("assistant_kind"),
            assistant_name=_row_assistant_name(conversation_row),
            fallback_title=conversation_row.get("title"),
        )

    def normalize_conversation_row(
        self,
        conversation_row: Mapping[str, Any] | None,
        *,
        keywords: Iterable[Any] | None = None,
        message_count: int | None = None,
    ) -> dict[str, Any] | None:
        return normalize_conversation_row(conversation_row, keywords=keywords, message_count=message_count)

    def normalize_message_row(self, message_row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        return normalize_message_row(message_row)

    def create_conversation(
        self,
        *,
        title: str | None = None,
        conversation_title: str | None = None,
        character_id: int | None = None,
        assistant_kind: str | None = None,
        assistant_id: str | None = None,
        persona_memory_mode: str | None = None,
        runtime_backend: str | None = None,
        discovery_owner: str | None = None,
        discovery_entity_id: str | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        state: str | None = None,
        topic_label: str | None = None,
        source: str | None = None,
        external_ref: str | None = None,
        **extra_fields: Any,
    ) -> str:
        resolved_title = derive_conversation_title(
            assistant_kind=assistant_kind,
            assistant_name=None,
            fallback_title=conversation_title or title,
            character_id=character_id,
        )
        conversation_data = {
            **extra_fields,
            "title": resolved_title,
            "character_id": character_id,
            "assistant_kind": assistant_kind,
            "assistant_id": assistant_id,
            "persona_memory_mode": persona_memory_mode,
            "runtime_backend": runtime_backend,
            "discovery_owner": discovery_owner,
            "discovery_entity_id": discovery_entity_id,
            "scope_type": scope_type,
            "workspace_id": workspace_id,
            "state": state,
            "topic_label": topic_label,
            "source": source,
            "external_ref": external_ref,
        }
        conversation_id = self.db.add_conversation(conversation_data)
        if conversation_id is None:
            raise ValueError("Unable to create chat conversation.")
        return str(conversation_id)

    def delete_conversation(self, conversation_id: str, *, expected_version: int) -> bool:
        return bool(self.db.soft_delete_conversation(conversation_id, expected_version))

    def _fetch_keywords_for_conversations(self, conversation_ids: list[str]) -> dict[str, list[str]]:
        if not conversation_ids:
            return {}
        if hasattr(self.db, "get_keywords_for_conversations"):
            keyword_rows_by_conversation = self.db.get_keywords_for_conversations(conversation_ids)
            return {
                conversation_id: _normalize_keywords(keyword_rows_by_conversation.get(conversation_id, []))
                for conversation_id in conversation_ids
            }
        return {conversation_id: self.get_conversation_keywords(conversation_id) for conversation_id in conversation_ids}

    def get_conversation_keywords(self, conversation_id: str) -> list[str]:
        keyword_rows = self.db.get_keywords_for_conversation(conversation_id)
        return _normalize_keywords(keyword_rows)

    def replace_conversation_keywords(self, conversation_id: str, keywords: Iterable[Any]) -> list[str]:
        normalized_keywords = _normalize_keywords(keywords)
        keyword_ids: list[int] = []
        for keyword_text in normalized_keywords:
            keyword_row = None
            if hasattr(self.db, "get_keyword_by_text"):
                keyword_row = self.db.get_keyword_by_text(keyword_text)
            keyword_id = self._keyword_row_to_id(keyword_row)
            if keyword_id is None:
                keyword_id = self._create_keyword_id(keyword_text)
            if keyword_id is None:
                raise ValueError(f"Unable to resolve keyword '{keyword_text}' to an ID")
            keyword_ids.append(keyword_id)

        self.db.replace_keywords_for_conversation(conversation_id, keyword_ids)
        return normalized_keywords

    def _create_keyword_id(self, keyword_text: str) -> int | None:
        keyword_id = self.db.add_keyword(keyword_text)
        if isinstance(keyword_id, int):
            return keyword_id
        if hasattr(self.db, "get_keyword_by_text"):
            keyword_row = self.db.get_keyword_by_text(keyword_text)
            return self._keyword_row_to_id(keyword_row)
        return None

    @staticmethod
    def _keyword_row_to_id(keyword_row: Any) -> int | None:
        if keyword_row is None:
            return None
        if isinstance(keyword_row, int):
            return keyword_row
        if isinstance(keyword_row, Mapping):
            raw_id = keyword_row.get("id")
            if raw_id is None:
                return None
            return int(raw_id)
        return None

    def get_conversation_metadata(self, conversation_id: str) -> dict[str, Any] | None:
        conversation_row = self.db.get_conversation_by_id(conversation_id)
        if not conversation_row:
            return None
        keywords = self.get_conversation_keywords(conversation_id)
        message_count = conversation_row.get("message_count")
        if message_count is None:
            if hasattr(self.db, "count_messages_for_conversation"):
                message_count = self.db.count_messages_for_conversation(
                    conversation_id,
                    include_deleted=False,
                    include_deleted_conversation=False,
                )
            elif hasattr(self.db, "count_messages_for_conversations"):
                counts = self.db.count_messages_for_conversations(
                    [conversation_id],
                    include_deleted=False,
                    include_deleted_conversation=False,
                )
                message_count = counts.get(conversation_id, 0)
            else:
                message_count = 0
        return normalize_conversation_row(conversation_row, keywords=keywords, message_count=message_count)

    def update_conversation_metadata(
        self,
        conversation_id: str,
        update_data: Mapping[str, Any],
        expected_version: int,
    ) -> bool:
        current_row = None
        if "scope_type" in update_data or "workspace_id" in update_data:
            current_row = self.db.get_conversation_by_id(conversation_id)

        normalized_update: dict[str, Any] = {}
        for key, value in update_data.items():
            if key == "assistant_kind":
                normalized_update[key] = _normalize_assistant_kind(value)
            elif key in {"assistant_id", "persona_memory_mode", "topic_label", "topic_label_source", "topic_last_tagged_message_id", "cluster_id", "source", "external_ref", "title"}:
                normalized_update[key] = _clean_text(value)
            elif key == "character_id":
                normalized_update[key] = value
            elif key == "state":
                normalized_update[key] = _normalize_state(value)
            elif key == "scope_type":
                cleaned_value = _clean_text(value)
                normalized_update[key] = cleaned_value.lower() if cleaned_value is not None else None
            elif key == "workspace_id":
                normalized_update[key] = _clean_text(value)
            else:
                normalized_update[key] = value

        if "scope_type" in normalized_update or "workspace_id" in normalized_update:
            if (
                "workspace_id" in update_data
                and update_data.get("workspace_id") is None
                and "scope_type" not in update_data
                and current_row is not None
                and _clean_text(current_row.get("scope_type")) == "workspace"
            ):
                raise ValueError("workspace_id is required when scope_type='workspace'")

            merged_scope_type = normalized_update.get("scope_type")
            if merged_scope_type is None and current_row is not None:
                merged_scope_type = current_row.get("scope_type")
            merged_workspace_id = normalized_update.get("workspace_id")
            if "workspace_id" not in normalized_update and current_row is not None:
                merged_workspace_id = current_row.get("workspace_id")
            normalized_update["scope_type"], normalized_update["workspace_id"] = _normalize_scope(
                merged_scope_type,
                merged_workspace_id,
            )

        if not normalized_update:
            return False

        return bool(self.db.update_conversation(conversation_id, dict(normalized_update), expected_version))

    def list_conversations(
        self,
        query: str | None = None,
        *,
        limit: int = 50,
        offset: int = 0,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        include_deleted: bool = False,
        deleted_only: bool = False,
        state: str | None = None,
        topic_label: str | None = None,
        character_id: int | None = None,
    ) -> dict[str, Any]:
        effective_scope = scope_type
        if effective_scope is None:
            effective_scope = "workspace" if workspace_id is not None else "global"
        if workspace_id is not None:
            effective_scope = "workspace"

        normalized_scope, normalized_workspace_id = _normalize_scope(effective_scope, workspace_id)
        rows, total, _ = self.db.search_conversations_page(
            query,
            scope_type=normalized_scope,
            workspace_id=normalized_workspace_id,
            include_deleted=include_deleted,
            deleted_only=deleted_only,
            state=_normalize_state(state) if state is not None else None,
            topic_label=_clean_text(topic_label),
            character_id=character_id,
            limit=limit,
            offset=offset,
        )

        conversation_ids = [row.get("id") for row in rows if row.get("id") is not None]
        message_counts = {}
        if conversation_ids:
            message_counts = self.db.count_messages_for_conversations(
                conversation_ids,
                include_deleted=include_deleted or deleted_only,
                include_deleted_conversation=include_deleted or deleted_only,
            )
        keyword_map = self._fetch_keywords_for_conversations(conversation_ids)

        items = []
        for row in rows:
            conversation_id = row.get("id")
            item = normalize_conversation_row(
                row,
                keywords=keyword_map.get(conversation_id, []),
                message_count=message_counts.get(conversation_id, row.get("message_count", 0)),
            )
            if item is not None:
                items.append(item)

        pagination = {
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": offset + len(items) < total,
        }
        return {"items": items, "pagination": pagination}

    def get_conversation_tree(
        self,
        conversation_id: str,
        *,
        root_limit: int = 50,
        root_offset: int = 0,
        order_by_timestamp: str = "ASC",
        depth_cap: int = 50,
    ) -> dict[str, Any]:
        conversation = self.get_conversation_metadata(conversation_id)
        if conversation is None:
            return {
                "conversation": None,
                "root_threads": [],
                "pagination": {
                    "limit": root_limit,
                    "offset": root_offset,
                    "total_root_threads": 0,
                    "has_more": False,
                },
                "depth_cap": depth_cap,
            }

        total_root_threads = self.db.count_root_messages_for_conversation(
            conversation_id,
            include_deleted_conversation=False,
        )
        root_rows = self.db.get_root_messages_for_conversation(
            conversation_id,
            limit=root_limit,
            offset=root_offset,
            order_by_timestamp=order_by_timestamp,
            include_deleted_conversation=False,
        )
        root_threads = self._build_message_tree(
            conversation_id,
            root_rows,
            order_by_timestamp=order_by_timestamp,
            depth_cap=depth_cap,
            depth=1,
            seen_message_ids=set(),
        )

        return {
            "conversation": conversation,
            "root_threads": root_threads,
            "pagination": {
                "limit": root_limit,
                "offset": root_offset,
                "total_root_threads": total_root_threads,
                "has_more": root_offset + len(root_rows) < total_root_threads,
            },
            "depth_cap": depth_cap,
        }

    def _build_message_tree(
        self,
        conversation_id: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        order_by_timestamp: str,
        depth_cap: int,
        depth: int,
        seen_message_ids: set[str],
    ) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        for row in rows:
            message_id = row.get("id")
            normalized_row = normalize_message_row(row)
            if normalized_row is None:
                continue
            if message_id is not None and message_id in seen_message_ids:
                normalized_row["children"] = []
                normalized_row["truncated"] = True
                nodes.append(normalized_row)
                continue

            next_seen = set(seen_message_ids)
            if message_id is not None:
                next_seen.add(message_id)

            if depth >= depth_cap:
                normalized_row["children"] = []
                normalized_row["truncated"] = True
                nodes.append(normalized_row)
                continue

            child_rows = self.db.get_messages_for_conversation_by_parent_ids(
                conversation_id,
                [message_id] if message_id is not None else [],
                order_by_timestamp=order_by_timestamp,
                include_deleted_conversation=False,
            )
            normalized_row["children"] = self._build_message_tree(
                conversation_id,
                child_rows,
                order_by_timestamp=order_by_timestamp,
                depth_cap=depth_cap,
                depth=depth + 1,
                seen_message_ids=next_seen,
            )
            normalized_row["truncated"] = False
            nodes.append(normalized_row)
        return nodes
