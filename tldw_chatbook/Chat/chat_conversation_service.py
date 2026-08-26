from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, cast

from tldw_chatbook.DB.ChaChaNotes_DB import CONVERSATION_SCOPE_ALL

_ASSISTANT_AUTHORITY_UNSET = cast(str | None, object())
_SQLITE_INTEGER_MAX = (1 << 63) - 1

if TYPE_CHECKING:
    from tldw_chatbook.Chat.citation_legacy_migration import (
        CitationLegacyMigrationService,
    )


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_state(value: Any) -> str | None:
    # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
    from tldw_chatbook.tldw_api.chat_conversation_schemas import (
        ALLOWED_CONVERSATION_STATES,
    )

    text = _clean_text(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized not in ALLOWED_CONVERSATION_STATES:
        raise ValueError(
            f"Invalid state '{value}'. Allowed: {', '.join(ALLOWED_CONVERSATION_STATES)}"
        )
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
    normalized_scope = (
        raw_scope.lower()
        if raw_scope is not None
        else ("workspace" if normalized_workspace_id else "global")
    )
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


def normalize_message_row(
    message_row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
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
        "usage_json": message_row.get("usage_json"),
        "metadata_json": message_row.get("metadata_json"),
        "provider_continuation_json": message_row.get("provider_continuation_json"),
        "assistant_generation_state": message_row.get(
            "assistant_generation_state"
        ),
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
    normalized_keywords = _normalize_keywords(
        keywords if keywords is not None else conversation_row.get("keywords")
    )
    normalized_state = _normalize_state(conversation_row.get("state")) or "in-progress"
    assistant_kind = _normalize_assistant_kind(conversation_row.get("assistant_kind"))
    assistant_id = _clean_text(conversation_row.get("assistant_id"))
    character_id = conversation_row.get("character_id")
    if (
        assistant_kind == "character"
        and assistant_id is None
        and character_id is not None
    ):
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
        "assistant_authority_id": _clean_text(
            conversation_row.get("assistant_authority_id")
        ),
        "runtime_backend": _normalize_runtime_backend(
            conversation_row.get("runtime_backend")
        ),
        "discovery_owner": _normalize_discovery_owner(
            conversation_row.get("discovery_owner")
        ),
        "discovery_entity_id": _clean_text(conversation_row.get("discovery_entity_id")),
        "persona_memory_mode": _clean_text(conversation_row.get("persona_memory_mode")),
        "title": normalized_title,
        "state": normalized_state,
        "topic_label": _clean_text(conversation_row.get("topic_label")),
        "topic_label_source": _clean_text(conversation_row.get("topic_label_source")),
        "topic_last_tagged_at": conversation_row.get("topic_last_tagged_at"),
        "topic_last_tagged_message_id": _clean_text(
            conversation_row.get("topic_last_tagged_message_id")
        ),
        "bm25_norm": conversation_row.get("bm25_norm"),
        "last_modified": conversation_row.get("last_modified"),
        "created_at": conversation_row.get("created_at"),
        "deleted": conversation_row.get("deleted"),
        "message_count": int(
            message_count
            if message_count is not None
            else conversation_row.get("message_count") or 0
        ),
        "keywords": normalized_keywords,
        "cluster_id": _clean_text(conversation_row.get("cluster_id")),
        "source": _clean_text(conversation_row.get("source")),
        "external_ref": _clean_text(conversation_row.get("external_ref")),
        "system_prompt": _clean_text(conversation_row.get("system_prompt")),
        "metadata": conversation_row.get("metadata"),
        "version": conversation_row.get("version"),
    }


class ChatConversationService:
    def __init__(
        self,
        db: Any,
        *,
        rag_context_store_path: str | Path | None = None,
        citation_legacy_migration: "CitationLegacyMigrationService | None" = None,
    ):
        self.db = db
        self.rag_context_store_path = (
            Path(rag_context_store_path) if rag_context_store_path else None
        )
        self._rag_context_store: dict[str, Any] | None = None
        self.citation_legacy_migration = citation_legacy_migration

    def set_citation_legacy_migration(
        self,
        migration: "CitationLegacyMigrationService | None",
    ) -> None:
        """Attach the canonical/legacy read boundary after repository wiring."""

        self.citation_legacy_migration = migration

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _load_rag_context_store(self) -> dict[str, Any]:
        if self._rag_context_store is not None:
            return self._rag_context_store
        if (
            self.rag_context_store_path is None
            or not self.rag_context_store_path.exists()
        ):
            self._rag_context_store = {"version": 1, "conversations": {}}
            return self._rag_context_store
        try:
            payload = json.loads(
                self.rag_context_store_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            payload = {}
        conversations = (
            payload.get("conversations") if isinstance(payload, Mapping) else None
        )
        self._rag_context_store = {
            "version": 1,
            "conversations": conversations if isinstance(conversations, dict) else {},
        }
        return self._rag_context_store

    def _save_rag_context_store(self) -> None:
        if self.rag_context_store_path is None or self._rag_context_store is None:
            return
        self.rag_context_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.rag_context_store_path.write_text(
            json.dumps(self._rag_context_store, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def derive_conversation_title(
        self, conversation_row: Mapping[str, Any] | None
    ) -> str:
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
        return normalize_conversation_row(
            conversation_row, keywords=keywords, message_count=message_count
        )

    def normalize_message_row(
        self, message_row: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        return normalize_message_row(message_row)

    def create_conversation(
        self,
        *,
        title: str | None = None,
        conversation_title: str | None = None,
        character_id: int | None = None,
        assistant_kind: str | None = None,
        assistant_id: str | None = None,
        assistant_authority_id: str | None = _ASSISTANT_AUTHORITY_UNSET,
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
        """Create and persist a conversation.

        Args:
            title: Secondary title candidate used only when raw
                ``conversation_title`` is falsey.
            conversation_title: Raw truthy value selected by
                ``conversation_title or title``, including whitespace. Downstream
                cleaning can discard it and derive an assistant title without
                falling back to ``title``.
            character_id: Local character identifier associated with the conversation.
            assistant_kind: Kind of assistant that owns the conversation.
            assistant_id: Stable assistant identifier.
            assistant_authority_id: Provenance authority identifier. Omitting it
                leaves the field absent so eligible DB-owned local inference may
                apply; passing ``None`` explicitly preserves unproven authority.
            persona_memory_mode: Memory behavior for a persona conversation.
            runtime_backend: Backend selected to run the assistant.
            discovery_owner: Owner of the assistant discovery record.
            discovery_entity_id: Discovery record identifier for the assistant.
            scope_type: Scope classification for the conversation.
            workspace_id: Workspace identifier when the scope requires one.
            state: Initial conversation state.
            topic_label: Optional topic label.
            source: Origin that created the conversation.
            external_ref: External source reference.
            **extra_fields: Additional database-supported fields. Explicit named
                fields are assigned after this mapping; recognized keyword names
                bind to the signature rather than this mapping.

        Returns:
            Persisted conversation ID as a string.

        Raises:
            ValueError: If the database cannot create the conversation.
        """
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
        if assistant_authority_id is not _ASSISTANT_AUTHORITY_UNSET:
            conversation_data["assistant_authority_id"] = assistant_authority_id
        conversation_id = self.db.add_conversation(conversation_data)
        if conversation_id is None:
            raise ValueError("Unable to create chat conversation.")
        return str(conversation_id)

    def delete_conversation(
        self, conversation_id: str, *, expected_version: int
    ) -> bool:
        return bool(self.db.soft_delete_conversation(conversation_id, expected_version))

    def restore_conversation(
        self, conversation_id: str, *, expected_version: int
    ) -> bool:
        return bool(self.db.restore_conversation(conversation_id, expected_version))

    def _fetch_keywords_for_conversations(
        self, conversation_ids: list[str]
    ) -> dict[str, list[str]]:
        if not conversation_ids:
            return {}
        if hasattr(self.db, "get_keywords_for_conversations"):
            keyword_rows_by_conversation = self.db.get_keywords_for_conversations(
                conversation_ids
            )
            return {
                conversation_id: _normalize_keywords(
                    keyword_rows_by_conversation.get(conversation_id, [])
                )
                for conversation_id in conversation_ids
            }
        return {
            conversation_id: self.get_conversation_keywords(conversation_id)
            for conversation_id in conversation_ids
        }

    def _normalize_conversation_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        include_deleted: bool,
    ) -> list[dict[str, Any]]:
        rows = list(rows)
        conversation_ids = [
            row.get("id") for row in rows if row.get("id") is not None
        ]
        message_counts = {}
        if conversation_ids:
            message_counts = self.db.count_messages_for_conversations(
                conversation_ids,
                include_deleted=include_deleted,
                include_deleted_conversation=include_deleted,
            )
        keyword_map = self._fetch_keywords_for_conversations(conversation_ids)

        items = []
        for row in rows:
            conversation_id = row.get("id")
            item = normalize_conversation_row(
                row,
                keywords=keyword_map.get(conversation_id, []),
                message_count=message_counts.get(
                    conversation_id, row.get("message_count", 0)
                ),
            )
            if item is not None:
                items.append(item)
        return items

    def get_conversation_keywords(self, conversation_id: str) -> list[str]:
        keyword_rows = self.db.get_keywords_for_conversation(conversation_id)
        return _normalize_keywords(keyword_rows)

    def replace_conversation_keywords(
        self, conversation_id: str, keywords: Iterable[Any]
    ) -> list[str]:
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
        return normalize_conversation_row(
            conversation_row, keywords=keywords, message_count=message_count
        )

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
            elif key == "assistant_authority_id":
                normalized_update[key] = value
            elif key in {
                "assistant_id",
                "persona_memory_mode",
                "topic_label",
                "topic_label_source",
                "topic_last_tagged_message_id",
                "cluster_id",
                "source",
                "external_ref",
                "title",
            }:
                normalized_update[key] = _clean_text(value)
            elif key == "character_id":
                normalized_update[key] = value
            elif key == "state":
                normalized_update[key] = _normalize_state(value)
            elif key == "scope_type":
                cleaned_value = _clean_text(value)
                normalized_update[key] = (
                    cleaned_value.lower() if cleaned_value is not None else None
                )
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
            normalized_update["scope_type"], normalized_update["workspace_id"] = (
                _normalize_scope(
                    merged_scope_type,
                    merged_workspace_id,
                )
            )

        if not normalized_update:
            return False

        return bool(
            self.db.update_conversation(
                conversation_id, dict(normalized_update), expected_version
            )
        )

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
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= _SQLITE_INTEGER_MAX
        ):
            raise ValueError("limit must be a positive integer.")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset <= _SQLITE_INTEGER_MAX
        ):
            raise ValueError("offset must be a non-negative integer.")

        effective_scope = scope_type
        if effective_scope is None:
            effective_scope = "workspace" if workspace_id is not None else "global"
        if workspace_id is not None:
            effective_scope = "workspace"

        if str(effective_scope).strip().lower() == CONVERSATION_SCOPE_ALL:
            # Query-only scope spanning global- and workspace-scoped rows in
            # one page/count. Used by the Library conversations snapshot so
            # Console chats persisted inside a workspace session are listed.
            # An explicit workspace_id always wins (handled above).
            normalized_scope: str = CONVERSATION_SCOPE_ALL
            normalized_workspace_id: str | None = None
        else:
            normalized_scope, normalized_workspace_id = _normalize_scope(
                effective_scope, workspace_id
            )
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

        items = self._normalize_conversation_rows(
            rows, include_deleted=include_deleted or deleted_only
        )

        pagination = {
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": offset + len(items) < total,
        }
        return {"items": items, "pagination": pagination}

    def locate_conversation_page(
        self,
        conversation_id: str,
        query: str | None = None,
        *,
        limit: int = 20,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        include_deleted: bool = False,
        deleted_only: bool = False,
        state: str | None = None,
        topic_label: str | None = None,
        character_id: int | None = None,
    ) -> dict[str, Any] | None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit != 20:
            raise ValueError("limit must be exactly 20.")

        effective_scope = scope_type
        if effective_scope is None:
            effective_scope = "workspace" if workspace_id is not None else "global"
        if workspace_id is not None:
            effective_scope = "workspace"

        if str(effective_scope).strip().lower() == CONVERSATION_SCOPE_ALL:
            normalized_scope: str = CONVERSATION_SCOPE_ALL
            normalized_workspace_id: str | None = None
        else:
            normalized_scope, normalized_workspace_id = _normalize_scope(
                effective_scope, workspace_id
            )
        located = self.db.locate_conversation_page(
            conversation_id,
            query=query,
            scope_type=normalized_scope,
            workspace_id=normalized_workspace_id,
            include_deleted=include_deleted,
            deleted_only=deleted_only,
            state=_normalize_state(state) if state is not None else None,
            topic_label=_clean_text(topic_label),
            character_id=character_id,
            limit=limit,
        )
        if located is None:
            return None

        rows = located.get("rows")
        offset = located.get("offset")
        target_index = located.get("target_index")
        total = located.get("total")
        coordinates = (limit, offset, target_index, total)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in coordinates
        ):
            raise ValueError("Conversation locator coordinates must be integers.")
        expected_offset = (target_index // limit) * limit if limit > 0 else -1
        if (
            limit <= 0
            or target_index < 0
            or total <= target_index
            or offset != expected_offset
        ):
            raise ValueError("Conversation locator offset is not page-aligned.")
        if not isinstance(rows, list) or len(rows) != min(limit, total - offset):
            raise ValueError("Conversation locator returned an invalid bounded page.")
        local_index = target_index - offset
        if (
            local_index < 0
            or local_index >= len(rows)
            or not isinstance(rows[local_index], Mapping)
            or rows[local_index].get("id") != conversation_id
        ):
            raise ValueError("Conversation locator target identity is invalid.")
        row_ids = [row.get("id") for row in rows if isinstance(row, Mapping)]
        if (
            len(row_ids) != len(rows)
            or any(
                not isinstance(row_id, str) or not row_id.strip()
                for row_id in row_ids
            )
            or len(set(row_ids)) != len(row_ids)
        ):
            raise ValueError("Conversation locator page identity is invalid.")

        items = self._normalize_conversation_rows(
            rows, include_deleted=include_deleted or deleted_only
        )
        if len(items) != len(rows) or items[local_index].get("id") != conversation_id:
            raise ValueError(
                "Conversation locator target identity changed during normalization."
            )
        pagination = {
            "limit": limit,
            "offset": offset,
            "page": offset // limit + 1,
            "total": total,
            "target_index": target_index,
            "has_more": offset + len(items) < total,
        }
        return {"items": items, "pagination": pagination}

    # --- Library read seams (task-1337, plan Task 4) ---
    #
    # Thin, agent-facing delegates over the additive DB library read seams.
    # RAG context lives in a JSON sidecar adjunct store owned by this
    # service; library reads never join it, so message responses always
    # carry ``include_rag_context: False``.

    def list_library_conversations(
        self, *, limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        """Page active local conversations for Library agent tools.

        Args:
            limit: Maximum number of conversations to return.
            offset: Number of conversations to skip.

        Returns:
            A bounded page containing items, exact total, offset, and limit.

        Raises:
            CharactersRAGDBError: If the local conversation store cannot be read.
        """
        payload = self.db.list_library_conversations_page(limit=limit, offset=offset)
        return {
            "items": payload["items"],
            "total": payload["total"],
            "offset": offset,
            "limit": limit,
        }

    def search_library_conversations(
        self, *, query: str, limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        """Search active local conversations for Library agent tools.

        Args:
            query: Literal case-insensitive search text.
            limit: Maximum number of conversations to return.
            offset: Number of matching conversations to skip.

        Returns:
            A bounded page with exact total and match evidence.

        Raises:
            CharactersRAGDBError: If the local conversation store cannot be read.
        """
        payload = self.db.search_library_conversations_page(
            query=query, limit=limit, offset=offset
        )
        return {
            "items": payload["items"],
            "total": payload["total"],
            "offset": offset,
            "limit": limit,
        }

    def get_library_conversation_messages(
        self,
        conversation_id: str,
        *,
        message_offset: int = 0,
        message_limit: int = 20,
        max_chars: int = 8000,
        message_id: str | None = None,
        char_start: int = 0,
    ) -> dict[str, Any] | None:
        """Read a text-only, windowed message page for one active conversation.

        Returns None when no active conversation matches ``conversation_id``.

        Args:
            conversation_id: Stable conversation identifier.
            message_offset: Number of messages to skip in page mode.
            message_limit: Maximum messages to return in page mode.
            max_chars: Maximum characters to return per message.
            message_id: Optional single-message continuation target.
            char_start: Zero-based character offset into message text.

        Returns:
            Bounded conversation metadata and messages, or None when absent.

        Raises:
            CharactersRAGDBError: If the local conversation store cannot be read.
        """
        return self.db.get_library_conversation_messages(
            conversation_id,
            message_offset=message_offset,
            message_limit=message_limit,
            max_chars=max_chars,
            message_id=message_id,
            char_start=char_start,
        )

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

        # TASK-22206: ONE conversation-scoped query (no BLOB hydration),
        # then a purely in-memory, iterative tree assembly. The old shape
        # issued one get_messages_for_conversation_by_parent_ids call per
        # node -- each a full-conversation scan under the production query
        # plan (sqlite_stat1 absent) -- and recursed once per message.
        rows = self.db.get_message_tree_rows_for_conversation(
            conversation_id,
            order_by_timestamp=order_by_timestamp,
            include_deleted_conversation=False,
        )
        children_by_parent: dict[Any, list[Mapping[str, Any]]] = {}
        root_rows: list[Mapping[str, Any]] = []
        for row in rows:
            parent_id = row.get("parent_message_id")
            if parent_id is None:
                root_rows.append(row)
            else:
                children_by_parent.setdefault(parent_id, []).append(row)
        # Same predicate the old COUNT query used, computed from the same
        # fetch (conversation-scoped, live rows, live conversation).
        total_root_threads = len(root_rows)
        # Replicate SQL LIMIT/OFFSET semantics for non-positive inputs:
        # a negative OFFSET is 0, a negative LIMIT means "no limit".
        effective_offset = max(0, root_offset)
        if root_limit < 0:
            paged_root_rows = root_rows[effective_offset:]
        else:
            paged_root_rows = root_rows[
                effective_offset : effective_offset + root_limit
            ]

        root_threads, image_pending = self._build_message_tree(
            paged_root_rows,
            children_by_parent,
            depth_cap=depth_cap,
        )
        self._hydrate_tree_images(image_pending)

        return {
            "conversation": conversation,
            "root_threads": root_threads,
            "pagination": {
                "limit": root_limit,
                "offset": root_offset,
                "total_root_threads": total_root_threads,
                "has_more": root_offset + len(paged_root_rows) < total_root_threads,
            },
            "depth_cap": depth_cap,
        }

    def record_message_rag_context(
        self,
        conversation_id: str,
        message_id: str,
        *,
        rag_context: Mapping[str, Any] | None = None,
        citations: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Deprecated compatibility writer available only in recovery mode."""

        migration = self.citation_legacy_migration
        if migration is not None and migration.writes_enabled:
            raise RuntimeError("legacy_rag_context_writes_disabled")
        if hasattr(self.db, "get_message_by_id"):
            message_row = self.db.get_message_by_id(message_id)
            if not message_row:
                raise ValueError("message not found")
            if str(message_row.get("conversation_id")) != str(conversation_id):
                raise ValueError("message does not belong to conversation")

        normalized_citations = []
        for citation in citations or []:
            item = dict(citation)
            item.setdefault("message_id", message_id)
            normalized_citations.append(item)

        record = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "rag_context": dict(rag_context or {}),
            "citations": normalized_citations,
            "last_modified": self._now(),
        }
        store = self._load_rag_context_store()
        conversation_store = store.setdefault("conversations", {}).setdefault(
            str(conversation_id), {}
        )
        conversation_store[str(message_id)] = record
        self._save_rag_context_store()
        return dict(record)

    def record_imported_legacy_citation_context(
        self,
        conversation_id: str,
        message_id: str,
        *,
        rag_context: Mapping[str, Any] | None = None,
        citations: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Persist package-era citations without portable-import semantics."""

        normalized_citations = [dict(item) for item in citations or ()]
        migration = self.citation_legacy_migration
        if migration is None or not migration.writes_enabled:
            return self.record_message_rag_context(
                conversation_id,
                message_id,
                rag_context=rag_context,
                citations=normalized_citations,
            )
        record = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "rag_context": dict(rag_context or {}),
            "citations": normalized_citations,
        }
        result = migration.persist_package_record(
            conversation_id=conversation_id,
            message_id=message_id,
            record=record,
        )
        if result.state.value != "complete":
            raise ValueError(result.reason_code or "legacy_package_citation_failed")
        return record

    def get_messages_with_context(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        order_by_timestamp: str = "ASC",
        include_rag_context: bool = True,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not hasattr(self.db, "get_messages_for_conversation"):
            return []
        rows = self.db.get_messages_for_conversation(
            conversation_id,
            limit=limit,
            offset=offset,
            order_by_timestamp=order_by_timestamp,
        )
        migration = self.citation_legacy_migration
        provenance_state = "legacy_fallback"
        if migration is not None:
            view = migration.read_conversation(
                str(conversation_id),
                verify_canonical=True,
            )
            conversation_store = view.records
            provenance_state = view.state.value
        else:
            conversation_store = (
                self._load_rag_context_store()
                .get("conversations", {})
                .get(str(conversation_id), {})
            )
        messages: list[dict[str, Any]] = []
        for row in rows:
            normalized = normalize_message_row(row)
            if normalized is None:
                continue
            adjunct = conversation_store.get(str(normalized["id"]), {})
            if include_rag_context:
                rag_context = adjunct.get("rag_context")
                if rag_context is None:
                    rag_context = {
                        key: adjunct[key]
                        for key in ("citation_validation", "evidence_bundle")
                        if key in adjunct
                    }
                normalized["rag_context"] = rag_context or None
            normalized["citations"] = list(adjunct.get("citations") or [])
            normalized["citation_provenance_state"] = provenance_state
            messages.append(normalized)
        return messages

    def get_citations(self, conversation_id: str) -> dict[str, Any]:
        migration = self.citation_legacy_migration
        state = "legacy_fallback"
        if migration is not None:
            view = migration.read_conversation(
                str(conversation_id),
                verify_canonical=True,
            )
            conversation_store = view.records
            state = view.state.value
        else:
            conversation_store = (
                self._load_rag_context_store()
                .get("conversations", {})
                .get(str(conversation_id), {})
            )
        citations: list[dict[str, Any]] = []
        for message_id, adjunct in conversation_store.items():
            for citation in adjunct.get("citations") or []:
                item = dict(citation)
                item.setdefault("message_id", message_id)
                citations.append(item)
        result = {
            "conversation_id": conversation_id,
            "citations": citations,
            "total_count": len(citations),
        }
        if migration is not None:
            result["state"] = state
        return result

    def _build_message_tree(
        self,
        paged_root_rows: Sequence[Mapping[str, Any]],
        children_by_parent: Mapping[Any, Sequence[Mapping[str, Any]]],
        *,
        depth_cap: int,
    ) -> tuple[list[dict[str, Any]], list[tuple[Any, dict[str, Any]]]]:
        """Assemble nested tree nodes iteratively from one row fetch.

        TASK-22206: replaces the recursive one-query-per-node walk (O(N^2)
        row scans, RecursionError at ~1000-deep linear conversations).
        Semantics preserved exactly: per-parent child order is the fetch's
        timestamp order (a stable partition of one ``ORDER BY m.timestamp``
        result); a node at ``depth >= depth_cap`` or whose id was already
        visited keeps ``children=[]`` with ``truncated=True``; a row
        ``normalize_message_row`` rejects is dropped along with its subtree;
        a row without an id gets no children. The visited set is global
        rather than the old per-path copy -- the two differ only on inputs a
        real DB cannot produce (a duplicated primary key), and the global
        set is what makes the walk O(N).

        BLOB columns are never touched here: rows carry ``has_image`` and
        image-bearing nodes are returned for one batched hydration pass
        (``_hydrate_tree_images``).

        Args:
            paged_root_rows: The page of root rows, in fetch order.
            children_by_parent: All non-root rows, bucketed by
                ``parent_message_id``, each bucket in fetch order.
            depth_cap: Maximum depth; roots are depth 1.

        Returns:
            The nested root nodes, and ``(message_id, node)`` pairs for
            every node whose row carries an image.
        """
        roots: list[dict[str, Any]] = []
        image_pending: list[tuple[Any, dict[str, Any]]] = []
        seen_message_ids: set[Any] = set()
        # (row, depth, parent's children list); explicit stack keeps
        # arbitrary-depth conversations off the Python recursion limit.
        stack: list[tuple[Mapping[str, Any], int, list[dict[str, Any]]]] = [
            (row, 1, roots) for row in reversed(paged_root_rows)
        ]
        while stack:
            row, depth, siblings = stack.pop()
            message_id = row.get("id")
            normalized_row = normalize_message_row(row)
            if normalized_row is None:
                continue
            if message_id is not None and row.get("has_image"):
                image_pending.append((message_id, normalized_row))
            if message_id is not None and message_id in seen_message_ids:
                normalized_row["children"] = []
                normalized_row["truncated"] = True
                siblings.append(normalized_row)
                continue
            if message_id is not None:
                seen_message_ids.add(message_id)
            if depth >= depth_cap:
                normalized_row["children"] = []
                normalized_row["truncated"] = True
                siblings.append(normalized_row)
                continue
            children: list[dict[str, Any]] = []
            normalized_row["children"] = children
            normalized_row["truncated"] = False
            siblings.append(normalized_row)
            if message_id is not None:
                for child_row in reversed(children_by_parent.get(message_id, ())):
                    stack.append((child_row, depth + 1, children))
        return roots, image_pending

    def _hydrate_tree_images(
        self, image_pending: Sequence[tuple[Any, dict[str, Any]]]
    ) -> None:
        """Fill image BLOBs into built tree nodes, one batched fetch.

        TASK-22206: nodes leave ``_build_message_tree`` with
        ``image_data=None``; the actual BLOBs are read here, once, only for
        the messages that have one. A conversation without images performs
        zero BLOB reads. Skipped silently for DB objects (test fakes) that
        do not expose the batched fetch -- their rows carry ``image_data``
        inline and ``normalize_message_row`` already passed it through.
        """
        if not image_pending:
            return
        fetcher = getattr(self.db, "get_message_images_by_ids", None)
        if not callable(fetcher):
            return
        images = fetcher([message_id for message_id, _node in image_pending])
        for message_id, node in image_pending:
            image_row = images.get(message_id)
            if image_row is None:
                # Deleted between the two reads; the node keeps image_data
                # None, matching a snapshot taken a moment later.
                continue
            node["image_data"] = image_row.get("image_data")
