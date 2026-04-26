"""Local character/persona session adapter for the source-aware CCP seam."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..Chat.chat_conversation_service import ChatConversationService
from ..tldw_api.character_persona_schemas import (
    CharacterChatSessionCreate,
    CharacterChatSessionUpdate,
    CharacterCreateRequest,
    CharacterExemplarCreate,
    CharacterExemplarSearchRequest,
    CharacterExemplarSelectionDebugRequest,
    CharacterExemplarUpdate,
    CharacterUpdateRequest,
    PersonaExemplarCreate,
    PersonaExemplarImportRequest,
    PersonaExemplarReviewRequest,
    PersonaExemplarUpdate,
    PersonaProfileCreate,
    PersonaProfileUpdate,
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

    def __init__(self, db: Any, *, persona_store_path: str | Path | None = None):
        self.db = db
        self.conversations = ChatConversationService(db)
        self.persona_store_path = Path(persona_store_path).expanduser() if persona_store_path is not None else None
        self._persona_profiles: list[dict[str, Any]] = []
        self._persona_exemplars: list[dict[str, Any]] = []
        self._character_exemplars: list[dict[str, Any]] = []
        self._load_persona_profiles()

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

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_persona_profiles(self) -> None:
        if self.persona_store_path is None or not self.persona_store_path.exists():
            return
        try:
            payload = json.loads(self.persona_store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._persona_profiles = []
            return
        if isinstance(payload, dict):
            profile_records = payload.get("profiles", payload.get("items", []))
            exemplar_records = payload.get("exemplars", [])
            character_exemplar_records = payload.get("character_exemplars", [])
        else:
            profile_records = payload
            exemplar_records = []
            character_exemplar_records = []
        if not isinstance(profile_records, list):
            self._persona_profiles = []
            return
        self._persona_profiles = [dict(item) for item in profile_records if isinstance(item, dict)]
        self._persona_exemplars = [
            dict(item)
            for item in exemplar_records
            if isinstance(item, dict)
        ] if isinstance(exemplar_records, list) else []
        self._character_exemplars = [
            dict(item)
            for item in character_exemplar_records
            if isinstance(item, dict)
        ] if isinstance(character_exemplar_records, list) else []

    def _persist_persona_profiles(self) -> None:
        if self.persona_store_path is None:
            return
        self.persona_store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.persona_store_path.with_suffix(self.persona_store_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(
                {
                    "profiles": self._persona_profiles,
                    "exemplars": self._persona_exemplars,
                    "character_exemplars": self._character_exemplars,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temp_path.replace(self.persona_store_path)

    @staticmethod
    def _persona_profile_view(record: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        normalized.setdefault("backend", "local")
        normalized.setdefault("record_id", f"local:persona_profile:{normalized.get('id')}")
        normalized["deleted"] = bool(normalized.get("deleted", False))
        normalized["version"] = int(normalized.get("version", 1) or 1)
        return normalized

    def _find_persona_profile(self, persona_id: str, *, include_deleted: bool = False) -> dict[str, Any]:
        normalized_id = str(persona_id)
        for record in self._persona_profiles:
            if str(record.get("id")) != normalized_id:
                continue
            if record.get("deleted") and not include_deleted:
                break
            return record
        raise ValueError(f"local_persona_profile_not_found:{persona_id}")

    @staticmethod
    def _check_profile_version(record: Mapping[str, Any], expected_version: int | None, persona_id: str) -> None:
        if expected_version is None:
            return
        if int(record.get("version", 1) or 1) != int(expected_version):
            raise ValueError(f"local_persona_profile_version_conflict:{persona_id}")

    @staticmethod
    def _persona_exemplar_view(record: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        normalized.setdefault("backend", "local")
        normalized.setdefault(
            "record_id",
            f"local:persona_exemplar:{normalized.get('persona_id')}:{normalized.get('id')}",
        )
        normalized["deleted"] = bool(normalized.get("deleted", False))
        normalized["enabled"] = bool(normalized.get("enabled", True))
        normalized["version"] = int(normalized.get("version", 1) or 1)
        return normalized

    def _find_persona_exemplar(
        self,
        persona_id: str,
        exemplar_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        normalized_persona_id = str(persona_id)
        normalized_exemplar_id = str(exemplar_id)
        for record in self._persona_exemplars:
            if str(record.get("persona_id")) != normalized_persona_id:
                continue
            if str(record.get("id")) != normalized_exemplar_id:
                continue
            if record.get("deleted") and not include_deleted:
                break
            return record
        raise ValueError(f"local_persona_exemplar_not_found:{persona_id}:{exemplar_id}")

    @staticmethod
    def _character_exemplar_view(record: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        normalized.setdefault("backend", "local")
        normalized.setdefault(
            "record_id",
            f"local:character_exemplar:{normalized.get('character_id')}:{normalized.get('id')}",
        )
        normalized["deleted"] = bool(normalized.get("deleted", False))
        return normalized

    def _find_character_exemplar(
        self,
        character_id: int,
        exemplar_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        normalized_character_id = int(character_id)
        normalized_exemplar_id = str(exemplar_id)
        for record in self._character_exemplars:
            if int(record.get("character_id")) != normalized_character_id:
                continue
            if str(record.get("id")) != normalized_exemplar_id:
                continue
            if record.get("deleted") and not include_deleted:
                break
            return record
        raise ValueError(f"local_character_exemplar_not_found:{character_id}:{exemplar_id}")

    def _require_character(self, character_id: int) -> None:
        if self.get_character(int(character_id)) is None:
            raise ValueError(f"local_character_not_found:{character_id}")

    def list_characters(self, limit: int = 100, offset: int = 0) -> Any:
        return self._require_db().list_character_cards(limit=limit, offset=offset)

    def search_characters(self, query: str, limit: int = 10) -> Any:
        return self._require_db().search_character_cards(query, limit=limit)

    def get_character(self, character_id: int) -> Any:
        return self._require_db().get_character_card_by_id(int(character_id))

    def create_character(self, request_data: Any) -> dict[str, Any]:
        payload = _model_payload(CharacterCreateRequest.model_validate(_model_payload(request_data)))
        character_id = self._require_db().add_character_card(payload)
        record = self.get_character(int(character_id))
        if record is None:
            raise ValueError("Created local character could not be loaded.")
        return record

    def update_character(
        self,
        character_id: int,
        request_data: Any,
        *,
        expected_version: int,
    ) -> dict[str, Any]:
        payload = _model_payload(
            CharacterUpdateRequest.model_validate(_model_payload(request_data, exclude_none=False))
        )
        payload = {key: value for key, value in payload.items() if value is not None}
        updated = self._require_db().update_character_card(int(character_id), payload, int(expected_version))
        if not updated:
            raise ValueError(f"Local character '{character_id}' could not be updated.")
        record = self.get_character(int(character_id))
        if record is None:
            raise ValueError(f"Local character '{character_id}' could not be loaded after update.")
        return record

    def delete_character(self, character_id: int, *, expected_version: int) -> dict[str, Any]:
        deleted = self._require_db().soft_delete_character_card(int(character_id), int(expected_version))
        if not deleted:
            raise ValueError(f"Local character '{character_id}' could not be deleted.")
        return {"status": "deleted", "character_id": int(character_id)}

    def restore_character(self, character_id: int, *, expected_version: int) -> dict[str, Any]:
        restored = self._require_db().restore_character_card(int(character_id), int(expected_version))
        if not restored:
            raise ValueError(f"Local character '{character_id}' could not be restored.")
        record = self.get_character(int(character_id))
        if record is None:
            raise ValueError(f"Local character '{character_id}' could not be loaded after restore.")
        return record

    def list_persona_profiles(
        self,
        *,
        active_only: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        records = [
            self._persona_profile_view(record)
            for record in self._persona_profiles
            if (include_deleted or not record.get("deleted", False))
            and (not active_only or bool(record.get("is_active", True)))
        ]
        records = sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)
        return records[offset : offset + limit]

    def get_persona_profile(self, persona_id: str) -> dict[str, Any]:
        return self._persona_profile_view(self._find_persona_profile(persona_id))

    def create_persona_profile(self, request_data: Any) -> dict[str, Any]:
        payload = _model_payload(PersonaProfileCreate.model_validate(_model_payload(request_data)))
        persona_id = str(payload.get("id") or f"local-persona-{uuid.uuid4().hex}")
        if any(str(record.get("id")) == persona_id and not record.get("deleted") for record in self._persona_profiles):
            raise ValueError(f"local_persona_profile_exists:{persona_id}")
        now = self._now()
        payload.update(
            {
                "id": persona_id,
                "created_at": now,
                "last_modified": now,
                "version": 1,
                "deleted": False,
            }
        )
        self._persona_profiles.append(payload)
        self._persist_persona_profiles()
        return self._persona_profile_view(payload)

    def update_persona_profile(
        self,
        persona_id: str,
        request_data: Any,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        record = self._find_persona_profile(persona_id)
        self._check_profile_version(record, expected_version, persona_id)
        request = PersonaProfileUpdate.model_validate(_model_payload(request_data))
        payload = request.model_dump(mode="json", exclude_none=True)
        current_version = int(record.get("version", 1) or 1)
        record.update(payload)
        record["last_modified"] = self._now()
        record["version"] = current_version + 1
        self._persist_persona_profiles()
        return self._persona_profile_view(record)

    def delete_persona_profile(
        self,
        persona_id: str,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        record = self._find_persona_profile(persona_id)
        self._check_profile_version(record, expected_version, persona_id)
        record["deleted"] = True
        record["last_modified"] = self._now()
        record["version"] = int(record.get("version", 1) or 1) + 1
        self._persist_persona_profiles()
        return {"status": "deleted", "persona_id": persona_id}

    def restore_persona_profile(self, persona_id: str, expected_version: int) -> dict[str, Any]:
        record = self._find_persona_profile(persona_id, include_deleted=True)
        self._check_profile_version(record, expected_version, persona_id)
        record["deleted"] = False
        record["last_modified"] = self._now()
        record["version"] = int(record.get("version", 1) or 1) + 1
        self._persist_persona_profiles()
        return self._persona_profile_view(record)

    def list_persona_exemplars(
        self,
        persona_id: str,
        *,
        include_disabled: bool = False,
        include_deleted: bool = False,
        include_deleted_personas: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._find_persona_profile(persona_id, include_deleted=include_deleted_personas)
        records = [
            self._persona_exemplar_view(record)
            for record in self._persona_exemplars
            if str(record.get("persona_id")) == str(persona_id)
            and (include_deleted or not record.get("deleted", False))
            and (include_disabled or bool(record.get("enabled", True)))
        ]
        records = sorted(records, key=lambda item: (item.get("priority", 0), item.get("created_at", "")), reverse=True)
        return records[offset : offset + limit]

    def get_persona_exemplar(self, persona_id: str, exemplar_id: str) -> dict[str, Any]:
        self._find_persona_profile(persona_id)
        return self._persona_exemplar_view(self._find_persona_exemplar(persona_id, exemplar_id))

    def create_persona_exemplar(self, persona_id: str, request_data: Any) -> dict[str, Any]:
        self._find_persona_profile(persona_id)
        payload = _model_payload(PersonaExemplarCreate.model_validate(_model_payload(request_data)))
        exemplar_id = str(payload.get("id") or f"local-exemplar-{uuid.uuid4().hex}")
        if any(
            str(record.get("persona_id")) == str(persona_id)
            and str(record.get("id")) == exemplar_id
            and not record.get("deleted")
            for record in self._persona_exemplars
        ):
            raise ValueError(f"local_persona_exemplar_exists:{persona_id}:{exemplar_id}")
        now = self._now()
        payload.update(
            {
                "id": exemplar_id,
                "persona_id": str(persona_id),
                "user_id": "local",
                "created_at": now,
                "last_modified": now,
                "version": 1,
                "deleted": False,
            }
        )
        self._persona_exemplars.append(payload)
        self._persist_persona_profiles()
        return self._persona_exemplar_view(payload)

    def import_persona_exemplars(self, persona_id: str, request_data: Any) -> dict[str, Any]:
        self._find_persona_profile(persona_id)
        request = PersonaExemplarImportRequest.model_validate(_model_payload(request_data))
        lines = [line.strip() for line in request.transcript.splitlines() if line.strip()]
        candidates = lines[: request.max_candidates] or [request.transcript.strip()]
        items = [
            self.create_persona_exemplar(
                persona_id,
                PersonaExemplarCreate(
                    content=content,
                    source_type="transcript_import",
                    source_ref=request.source_ref,
                    notes=request.notes,
                ),
            )
            for content in candidates
        ]
        return {"persona_id": persona_id, "created": len(items), "items": items}

    def update_persona_exemplar(self, persona_id: str, exemplar_id: str, request_data: Any) -> dict[str, Any]:
        self._find_persona_profile(persona_id)
        record = self._find_persona_exemplar(persona_id, exemplar_id)
        request = PersonaExemplarUpdate.model_validate(_model_payload(request_data))
        payload = request.model_dump(mode="json", exclude_none=True)
        record.update(payload)
        record["last_modified"] = self._now()
        record["version"] = int(record.get("version", 1) or 1) + 1
        self._persist_persona_profiles()
        return self._persona_exemplar_view(record)

    def review_persona_exemplar(self, persona_id: str, exemplar_id: str, request_data: Any) -> dict[str, Any]:
        self._find_persona_profile(persona_id)
        record = self._find_persona_exemplar(persona_id, exemplar_id, include_deleted=True)
        request = PersonaExemplarReviewRequest.model_validate(_model_payload(request_data))
        record["enabled"] = request.action == "approve"
        if request.notes is not None:
            record["notes"] = request.notes
        record["last_modified"] = self._now()
        record["version"] = int(record.get("version", 1) or 1) + 1
        self._persist_persona_profiles()
        return self._persona_exemplar_view(record)

    def delete_persona_exemplar(self, persona_id: str, exemplar_id: str) -> dict[str, Any]:
        self._find_persona_profile(persona_id)
        record = self._find_persona_exemplar(persona_id, exemplar_id)
        record["deleted"] = True
        record["last_modified"] = self._now()
        record["version"] = int(record.get("version", 1) or 1) + 1
        self._persist_persona_profiles()
        return {"status": "deleted", "persona_id": persona_id, "exemplar_id": exemplar_id}

    def search_character_exemplars(self, character_id: int, request_data: Any) -> dict[str, Any]:
        self._require_character(character_id)
        request = CharacterExemplarSearchRequest.model_validate(_model_payload(request_data))
        query = str(request.query or "").strip().lower()
        records = [
            self._character_exemplar_view(record)
            for record in self._character_exemplars
            if int(record.get("character_id")) == int(character_id)
            and not record.get("deleted", False)
        ]
        if query:
            records = [record for record in records if query in str(record.get("text") or "").lower()]
        if request.filter.emotion is not None:
            records = [
                record
                for record in records
                if (record.get("labels") or {}).get("emotion") == request.filter.emotion
            ]
        if request.filter.scenario is not None:
            records = [
                record
                for record in records
                if (record.get("labels") or {}).get("scenario") == request.filter.scenario
            ]
        total = len(records)
        page = records[request.offset : request.offset + request.limit]
        return {"items": page, "total": total}

    def get_character_exemplar(
        self,
        character_id: int,
        exemplar_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        self._require_character(character_id)
        return self._character_exemplar_view(
            self._find_character_exemplar(character_id, exemplar_id, include_deleted=include_deleted)
        )

    def create_character_exemplar(self, character_id: int, request_data: Any) -> dict[str, Any]:
        self._require_character(character_id)
        payload = CharacterExemplarCreate.model_validate(_model_payload(request_data)).model_dump(
            mode="json",
            by_alias=True,
        )
        exemplar_id = f"local-character-exemplar-{uuid.uuid4().hex}"
        now = self._now()
        payload.update(
            {
                "id": exemplar_id,
                "character_id": int(character_id),
                "created_at": now,
                "updated_at": now,
                "deleted": False,
            }
        )
        self._character_exemplars.append(payload)
        self._persist_persona_profiles()
        return self._character_exemplar_view(payload)

    def update_character_exemplar(self, character_id: int, exemplar_id: str, request_data: Any) -> dict[str, Any]:
        self._require_character(character_id)
        record = self._find_character_exemplar(character_id, exemplar_id)
        payload = CharacterExemplarUpdate.model_validate(_model_payload(request_data)).model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        record.update(payload)
        record["updated_at"] = self._now()
        self._persist_persona_profiles()
        return self._character_exemplar_view(record)

    def delete_character_exemplar(self, character_id: int, exemplar_id: str) -> dict[str, Any]:
        self._require_character(character_id)
        record = self._find_character_exemplar(character_id, exemplar_id)
        record["deleted"] = True
        record["updated_at"] = self._now()
        self._persist_persona_profiles()
        return {"status": "deleted", "character_id": int(character_id), "exemplar_id": exemplar_id}

    def select_character_exemplars_debug(self, character_id: int, request_data: Any) -> dict[str, Any]:
        self._require_character(character_id)
        request = CharacterExemplarSelectionDebugRequest.model_validate(_model_payload(request_data))
        search_result = self.search_character_exemplars(
            character_id,
            CharacterExemplarSearchRequest(
                query=request.user_turn,
                limit=20,
                offset=0,
            ),
        )
        selected = search_result["items"] or [
            self._character_exemplar_view(record)
            for record in self._character_exemplars
            if int(record.get("character_id")) == int(character_id)
            and not record.get("deleted", False)
        ]
        selected = selected[: max(1, min(len(selected), request.selection_config.budget_tokens))]
        return {
            "selected": selected,
            "coverage": {"selected_count": len(selected), "candidate_count": search_result["total"]},
            "scores": [],
        }

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
