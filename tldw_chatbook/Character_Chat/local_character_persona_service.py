"""Local character/persona session adapter for the source-aware CCP seam."""

from __future__ import annotations

from typing import Any, Mapping

from ..Chat.chat_conversation_service import ChatConversationService
from ..tldw_api.character_persona_schemas import (
    CharacterChatSessionCreate,
    CharacterChatSessionUpdate,
)


def _model_payload(value: Any, *, exclude_none: bool = True) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=exclude_none, mode="json")
    return dict(value or {})


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class LocalCharacterPersonaService:
    """Wrap local character cards and local CCP chat-session metadata."""

    def __init__(self, db: Any):
        self.db = db
        self.conversations = ChatConversationService(db)

    def _require_db(self) -> Any:
        if self.db is None:
            raise ValueError("Local character/persona backend is unavailable.")
        return self.db

    @staticmethod
    def _session_record(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        normalized = dict(record)
        conversation_id = normalized.get("id")
        normalized.setdefault("backend", "local")
        if conversation_id is not None:
            normalized.setdefault("record_id", f"local:character_chat_session:{conversation_id}")
        return normalized

    @staticmethod
    def _is_ccp_session(record: Mapping[str, Any]) -> bool:
        assistant_kind = _clean_text(record.get("assistant_kind"))
        discovery_owner = _clean_text(record.get("discovery_owner"))
        return assistant_kind in {"character", "persona"} or discovery_owner in {"ccp_character", "ccp_persona"}

    @staticmethod
    def _expected_version(record: Mapping[str, Any], expected_version: int | None) -> int:
        if expected_version is not None:
            return int(expected_version)
        version = record.get("version")
        if version is None:
            raise ValueError("expected_version is required when the local chat session has no version.")
        return int(version)

    def list_characters(self, limit: int = 100, offset: int = 0) -> Any:
        return self._require_db().list_character_cards(limit=limit, offset=offset)

    def create_character_chat_session(self, request_data: Any, **_: Any) -> dict[str, Any]:
        payload = _model_payload(CharacterChatSessionCreate.model_validate(_model_payload(request_data)))
        assistant_kind = payload.get("assistant_kind")
        discovery_owner = "ccp_persona" if assistant_kind == "persona" else "ccp_character"
        conversation_id = self.conversations.create_conversation(
            title=payload.get("title"),
            character_id=payload.get("character_id"),
            assistant_kind=assistant_kind,
            assistant_id=payload.get("assistant_id"),
            persona_memory_mode=payload.get("persona_memory_mode"),
            runtime_backend="local",
            discovery_owner=discovery_owner,
            discovery_entity_id=payload.get("assistant_id"),
            scope_type=payload.get("scope_type"),
            workspace_id=payload.get("workspace_id"),
            state=payload.get("state"),
            topic_label=payload.get("topic_label"),
            cluster_id=payload.get("cluster_id"),
            source=payload.get("source"),
            external_ref=payload.get("external_ref"),
            parent_conversation_id=payload.get("parent_conversation_id"),
            forked_from_message_id=payload.get("forked_from_message_id"),
        )
        record = self.get_character_chat_session(conversation_id, include_deleted=True)
        if record is None:
            raise ValueError("Created local character chat session could not be loaded.")
        return record

    def list_character_chat_sessions(
        self,
        *,
        character_id: int | None = None,
        assistant_kind: str | None = None,
        assistant_id: str | None = None,
        q: str | None = None,
        query: str | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        state: str | None = None,
        include_deleted: bool = False,
        deleted_only: bool = False,
        limit: int = 100,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        fetch_limit = max(limit + offset, limit)
        page = self.conversations.list_conversations(
            query=query or q,
            limit=fetch_limit,
            offset=0,
            scope_type=scope_type,
            workspace_id=workspace_id,
            include_deleted=include_deleted,
            deleted_only=deleted_only,
            state=state,
            character_id=character_id,
        )
        normalized_kind = _clean_text(assistant_kind)
        normalized_assistant_id = _clean_text(assistant_id)
        records = []
        for item in page.get("items", []):
            if not self._is_ccp_session(item):
                continue
            if normalized_kind and item.get("assistant_kind") != normalized_kind:
                continue
            if normalized_assistant_id and str(item.get("assistant_id") or "") != normalized_assistant_id:
                continue
            session_record = self._session_record(item)
            if session_record is not None:
                records.append(session_record)

        return {
            "chats": records[offset : offset + limit],
            "total": len(records),
            "limit": limit,
            "offset": offset,
        }

    def get_character_chat_session(
        self,
        chat_id: str,
        *,
        include_deleted: bool = False,
        **_: Any,
    ) -> dict[str, Any] | None:
        record = self.conversations.get_conversation_metadata(chat_id)
        if record is None and include_deleted:
            raw_record = self._require_db().get_conversation_by_id(chat_id, include_deleted=True)
            record = self.conversations.normalize_conversation_row(raw_record, message_count=0)
        if record is None or not self._is_ccp_session(record):
            return None
        return self._session_record(record)

    def update_character_chat_session(
        self,
        chat_id: str,
        request_data: Any,
        *,
        expected_version: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        current = self.get_character_chat_session(chat_id)
        if current is None:
            raise ValueError(f"Local character chat session '{chat_id}' was not found.")
        payload = _model_payload(CharacterChatSessionUpdate.model_validate(_model_payload(request_data)))
        update_payload = {key: value for key, value in payload.items() if value is not None}
        self.conversations.update_conversation_metadata(
            chat_id,
            update_payload,
            expected_version=self._expected_version(current, expected_version),
        )
        updated = self.get_character_chat_session(chat_id)
        if updated is None:
            raise ValueError(f"Local character chat session '{chat_id}' could not be reloaded after update.")
        return updated

    def delete_character_chat_session(
        self,
        chat_id: str,
        *,
        expected_version: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        current = self.get_character_chat_session(chat_id)
        if current is None:
            raise ValueError(f"Local character chat session '{chat_id}' was not found.")
        self.conversations.delete_conversation(
            chat_id,
            expected_version=self._expected_version(current, expected_version),
        )
        return {"status": "deleted", "chat_id": chat_id}

    def restore_character_chat_session(
        self,
        chat_id: str,
        *,
        expected_version: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        current = self.get_character_chat_session(chat_id, include_deleted=True)
        if current is None:
            raise ValueError(f"Local character chat session '{chat_id}' was not found.")
        if not current.get("deleted"):
            return current
        self.conversations.restore_conversation(
            chat_id,
            expected_version=self._expected_version(current, expected_version),
        )
        restored = self.get_character_chat_session(chat_id)
        if restored is None:
            raise ValueError(f"Local character chat session '{chat_id}' could not be reloaded after restore.")
        return restored

    def export_chat_history(
        self,
        chat_id: str,
        *,
        format: str = "json",
        limit: int = 1000,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any] | str:
        session = self.get_character_chat_session(chat_id)
        if session is None:
            raise ValueError(f"Local character chat session '{chat_id}' was not found.")
        messages = [
            self.conversations.normalize_message_row(message)
            for message in self._require_db().get_messages_for_conversation(chat_id, limit=limit, offset=offset)
        ]
        messages = [message for message in messages if message is not None]
        normalized_format = str(format or "json").strip().lower()
        if normalized_format == "markdown":
            lines = [f"# {session.get('title') or chat_id}", ""]
            for message in messages:
                role = message.get("role") or message.get("sender") or "message"
                lines.extend([f"## {role}", "", str(message.get("content") or ""), ""])
            return "\n".join(lines).rstrip() + "\n"
        if normalized_format != "json":
            raise ValueError("Local character chat export supports json and markdown formats.")
        return {
            "chat_id": chat_id,
            "format": "json",
            "session": session,
            "messages": messages,
        }

    def get_chat_settings(self, chat_id: str, **_: Any) -> Any:
        raise ValueError("Local character chat settings are not available through this scope service yet.")

    def update_chat_settings(self, chat_id: str, request_data: Any, **_: Any) -> Any:
        raise ValueError("Local character chat settings updates are not available through this scope service yet.")

    def export_lorebook_diagnostics(self, chat_id: str, **_: Any) -> Any:
        raise ValueError("Local character lorebook diagnostics are not available through this scope service yet.")

    def list_chat_greetings(self, chat_id: str) -> Any:
        raise ValueError("Local chat greetings are not available yet.")

    def select_chat_greeting(self, chat_id: str, index: int) -> Any:
        raise ValueError("Local chat greetings are not available yet.")

    def list_chat_presets(self) -> Any:
        raise ValueError("Local chat presets are not available yet.")

    def create_chat_preset(self, request_data: Any) -> Any:
        raise ValueError("Local chat presets are not available yet.")

    def update_chat_preset(self, preset_id: str, request_data: Any) -> Any:
        raise ValueError("Local chat presets are not available yet.")

    def delete_chat_preset(self, preset_id: str) -> Any:
        raise ValueError("Local chat presets are not available yet.")


__all__ = ["LocalCharacterPersonaService"]
