"""Local character/persona session adapter for the source-aware CCP seam."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..Chat.chat_conversation_service import ChatConversationService
from .world_book_manager import WorldBookManager


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
        self.world_books = WorldBookManager(db) if db is not None else None
        self.persona_store_path = Path(persona_store_path).expanduser() if persona_store_path is not None else None
        self._persona_profiles: list[dict[str, Any]] = []
        self._persona_exemplars: list[dict[str, Any]] = []
        self._character_exemplars: list[dict[str, Any]] = []
        self._chat_settings: dict[str, dict[str, Any]] = {}
        self._chat_greeting_selections: dict[str, int] = {}
        self._chat_presets: list[dict[str, Any]] = []
        self._character_memories: list[dict[str, Any]] = []
        self._load_persona_profiles()

    def _require_db(self) -> Any:
        if self.db is None:
            raise ValueError("Local character/persona backend is unavailable.")
        return self.db

    def _require_world_book_manager(self) -> WorldBookManager:
        if self.world_books is None:
            self.world_books = WorldBookManager(self._require_db())
        return self.world_books

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
            chat_settings_records = payload.get("chat_settings", {})
            chat_greeting_selections = payload.get("chat_greeting_selections", {})
            chat_preset_records = payload.get("chat_presets", [])
            character_memory_records = payload.get("character_memories", [])
        else:
            profile_records = payload
            exemplar_records = []
            character_exemplar_records = []
            chat_settings_records = {}
            chat_greeting_selections = {}
            chat_preset_records = []
            character_memory_records = []
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
        self._chat_settings = {
            str(chat_id): dict(settings)
            for chat_id, settings in chat_settings_records.items()
            if isinstance(settings, dict)
        } if isinstance(chat_settings_records, dict) else {}
        self._chat_greeting_selections = {
            str(chat_id): int(index)
            for chat_id, index in chat_greeting_selections.items()
            if isinstance(index, int)
        } if isinstance(chat_greeting_selections, dict) else {}
        self._chat_presets = [
            dict(item)
            for item in chat_preset_records
            if isinstance(item, dict)
        ] if isinstance(chat_preset_records, list) else []
        self._character_memories = [
            dict(item)
            for item in character_memory_records
            if isinstance(item, dict)
        ] if isinstance(character_memory_records, list) else []

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
                    "chat_settings": self._chat_settings,
                    "chat_greeting_selections": self._chat_greeting_selections,
                    "chat_presets": self._chat_presets,
                    "character_memories": self._character_memories,
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

    @staticmethod
    def _builtin_chat_preset() -> dict[str, Any]:
        return {
            "preset_id": "default",
            "name": "Default",
            "builtin": True,
            "section_order": [],
            "section_templates": {},
            "created_at": None,
            "updated_at": None,
        }

    @staticmethod
    def _chat_preset_view(record: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        normalized.setdefault("builtin", False)
        normalized.setdefault("section_order", [])
        normalized.setdefault("section_templates", {})
        normalized.setdefault("created_at", None)
        normalized.setdefault("updated_at", None)
        normalized.setdefault("source", "local")
        return normalized

    def _find_chat_preset(self, preset_id: str, *, include_deleted: bool = False) -> dict[str, Any]:
        normalized_id = str(preset_id)
        for record in self._chat_presets:
            if str(record.get("preset_id")) != normalized_id:
                continue
            if record.get("deleted") and not include_deleted:
                break
            return record
        raise ValueError(f"local_chat_preset_not_found:{preset_id}")

    @staticmethod
    def _character_memory_view(record: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        normalized.setdefault("source", "local")
        normalized.setdefault("archived", False)
        normalized.setdefault("deleted", False)
        normalized.setdefault("version", 1)
        normalized.setdefault(
            "record_id",
            f"local:character_memory:{normalized.get('character_id')}:{normalized.get('id')}",
        )
        return normalized

    def _find_character_memory(
        self,
        character_id: str,
        memory_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        normalized_character_id = str(character_id)
        normalized_memory_id = str(memory_id)
        for record in self._character_memories:
            if str(record.get("character_id")) != normalized_character_id:
                continue
            if str(record.get("id")) != normalized_memory_id:
                continue
            if record.get("deleted") and not include_deleted:
                break
            return record
        raise ValueError(f"local_character_memory_not_found:{character_id}:{memory_id}")

    def _require_chat_session(self, chat_id: str) -> dict[str, Any]:
        session = self.get_character_chat_session(str(chat_id))
        if session is None:
            raise ValueError(f"Local character chat session '{chat_id}' was not found.")
        return session

    def _character_for_session(self, session: Mapping[str, Any]) -> dict[str, Any] | None:
        character_id = session.get("character_id")
        if character_id is None:
            return None
        character = self.get_character(int(character_id))
        return dict(character) if character is not None else None

    @staticmethod
    def _character_greeting_texts(character: Mapping[str, Any] | None) -> list[str]:
        if character is None:
            return []
        first_message = _clean_text(
            character.get("first_message")
            or character.get("first_mes")
            or character.get("greeting")
        )
        greetings: list[str] = []
        if first_message is not None:
            greetings.append(first_message)
        for item in character.get("alternate_greetings") or []:
            text = _clean_text(item)
            if text is not None:
                greetings.append(text)
        return greetings

    def _chat_greeting_items(self, chat_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
        session = self._require_chat_session(chat_id)
        character = self._character_for_session(session)
        greetings = [
            {"index": index, "text": text, "preview": text[:160]}
            for index, text in enumerate(self._character_greeting_texts(character))
        ]
        warning = None
        if character is None:
            warning = "Local chat is not attached to a character card."
        return session, greetings, warning

    def list_characters(self, limit: int = 100, offset: int = 0) -> Any:
        return self._require_db().list_character_cards(limit=limit, offset=offset)

    def search_characters(self, query: str, limit: int = 10) -> Any:
        return self._require_db().search_character_cards(query, limit=limit)

    def get_character(self, character_id: int) -> Any:
        record = self._require_db().get_character_card_by_id(int(character_id))
        if record is None:
            raise ValueError(f"Local character '{character_id}' not found")
        return record

    def create_character(self, request_data: Any) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import CharacterCreateRequest

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import CharacterUpdateRequest

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
        return {"deleted": True, "id": str(character_id)}

    def restore_character(self, character_id: int, *, expected_version: int) -> dict[str, Any]:
        restored = self._require_db().restore_character_card(int(character_id), int(expected_version))
        if not restored:
            raise ValueError(f"Local character '{character_id}' could not be restored.")
        record = self.get_character(int(character_id))
        if record is None:
            raise ValueError(f"Local character '{character_id}' could not be loaded after restore.")
        return record

    @staticmethod
    def _world_book_payload(request_data: Any) -> dict[str, Any]:
        payload = _model_payload(request_data, exclude_none=False)
        return {key: value for key, value in payload.items() if value is not None}

    def list_character_world_books(self, *, include_disabled: bool = False, **_: Any) -> dict[str, Any]:
        items = self._require_world_book_manager().list_world_books(include_disabled=include_disabled)
        return {"world_books": items, "total": len(items)}

    def get_character_world_book(self, world_book_id: int, **_: Any) -> dict[str, Any]:
        record = self._require_world_book_manager().get_world_book(int(world_book_id))
        if record is None:
            raise ValueError(f"Local character world book '{world_book_id}' not found")
        return record

    def create_character_world_book(self, request_data: Any, **_: Any) -> dict[str, Any]:
        payload = self._world_book_payload(request_data)
        world_book_id = self._require_world_book_manager().create_world_book(
            name=payload.get("name"),
            description=payload.get("description"),
            scan_depth=int(payload.get("scan_depth", 3) or 3),
            token_budget=int(payload.get("token_budget", 500) or 500),
            recursive_scanning=bool(payload.get("recursive_scanning", False)),
            enabled=bool(payload.get("enabled", True)),
        )
        return self.get_character_world_book(world_book_id)

    def update_character_world_book(
        self,
        world_book_id: int,
        request_data: Any,
        *,
        expected_version: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        payload = self._world_book_payload(request_data)
        update_fields = {
            key: payload[key]
            for key in (
                "name",
                "description",
                "scan_depth",
                "token_budget",
                "recursive_scanning",
                "enabled",
            )
            if key in payload
        }
        updated = self._require_world_book_manager().update_world_book(
            int(world_book_id),
            expected_version=expected_version,
            **update_fields,
        )
        if not updated:
            raise ValueError(f"Local character world book '{world_book_id}' could not be updated.")
        return self.get_character_world_book(world_book_id)

    def delete_character_world_book(
        self,
        world_book_id: int,
        *,
        expected_version: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        deleted = self._require_world_book_manager().delete_world_book(
            int(world_book_id),
            expected_version=expected_version,
        )
        if not deleted:
            raise ValueError(f"Local character world book '{world_book_id}' could not be deleted.")
        return {"deleted": True, "id": str(world_book_id)}

    def _get_character_world_book_entry(self, entry_id: int) -> dict[str, Any] | None:
        query = """
        SELECT id, world_book_id, keys, content, enabled, position, insertion_order,
               selective, secondary_keys, case_sensitive, extensions, created_at, last_modified
        FROM world_book_entries
        WHERE id = ?
        """
        with self._require_db().transaction() as cursor:
            cursor.execute(query, (int(entry_id),))
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "world_book_id": row[1],
            "keys": json.loads(row[2]) if row[2] else [],
            "content": row[3],
            "enabled": bool(row[4]),
            "position": row[5],
            "insertion_order": row[6],
            "selective": bool(row[7]),
            "secondary_keys": json.loads(row[8]) if row[8] else [],
            "case_sensitive": bool(row[9]),
            "extensions": json.loads(row[10]) if row[10] else {},
            "created_at": row[11],
            "last_modified": row[12],
        }

    def list_character_world_book_entries(
        self,
        world_book_id: int,
        *,
        include_disabled: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        entries = self._require_world_book_manager().get_world_book_entries(
            int(world_book_id),
            enabled_only=not include_disabled,
        )
        return {"entries": entries, "total": len(entries)}

    def get_character_world_book_entry(self, entry_id: int, **_: Any) -> dict[str, Any]:
        entry = self._get_character_world_book_entry(int(entry_id))
        if entry is None:
            raise ValueError(f"Local character world book entry '{entry_id}' not found")
        return entry

    def create_character_world_book_entry(
        self,
        world_book_id: int,
        request_data: Any,
        **_: Any,
    ) -> dict[str, Any]:
        payload = self._world_book_payload(request_data)
        entry_id = self._require_world_book_manager().create_world_book_entry(
            world_book_id=int(world_book_id),
            keys=list(payload.get("keys") or []),
            content=payload.get("content"),
            enabled=bool(payload.get("enabled", True)),
            position=payload.get("position", "before_char"),
            insertion_order=int(payload.get("insertion_order", 0) or 0),
            selective=bool(payload.get("selective", False)),
            secondary_keys=list(payload.get("secondary_keys") or []),
            case_sensitive=bool(payload.get("case_sensitive", False)),
            extensions=dict(payload.get("extensions") or {}),
        )
        return self.get_character_world_book_entry(entry_id)

    def update_character_world_book_entry(
        self,
        entry_id: int,
        request_data: Any,
        **_: Any,
    ) -> dict[str, Any]:
        payload = self._world_book_payload(request_data)
        update_fields = {
            key: payload[key]
            for key in (
                "keys",
                "content",
                "enabled",
                "position",
                "insertion_order",
                "selective",
                "secondary_keys",
                "case_sensitive",
                "extensions",
            )
            if key in payload
        }
        updated = self._require_world_book_manager().update_world_book_entry(int(entry_id), **update_fields)
        if not updated:
            raise ValueError(f"Local character world book entry '{entry_id}' could not be updated.")
        return self.get_character_world_book_entry(entry_id)

    def delete_character_world_book_entry(self, entry_id: int, **_: Any) -> dict[str, Any]:
        deleted = self._require_world_book_manager().delete_world_book_entry(int(entry_id))
        if not deleted:
            raise ValueError(f"Local character world book entry '{entry_id}' could not be deleted.")
        return {"deleted": True, "id": str(entry_id)}

    def attach_character_world_book_to_session(
        self,
        chat_id: str,
        world_book_id: int,
        request_data: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        payload = self._world_book_payload(request_data or {})
        priority = int(payload.get("priority", 0) or 0)
        attached = self._require_world_book_manager().associate_world_book_with_conversation(
            str(chat_id),
            int(world_book_id),
            priority=priority,
        )
        if not attached:
            raise ValueError(f"Local character world book '{world_book_id}' could not be attached.")
        return {"chat_id": str(chat_id), "world_book_id": int(world_book_id), "priority": priority}

    def detach_character_world_book_from_session(self, chat_id: str, world_book_id: int, **_: Any) -> dict[str, Any]:
        self._require_world_book_manager().disassociate_world_book_from_conversation(str(chat_id), int(world_book_id))
        return {"deleted": True, "chat_id": str(chat_id), "world_book_id": int(world_book_id)}

    def list_session_world_books(
        self,
        chat_id: str,
        *,
        include_disabled: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        items = self._require_world_book_manager().get_world_books_for_conversation(
            str(chat_id),
            enabled_only=not include_disabled,
        )
        return {"world_books": items, "total": len(items)}

    def export_character_world_book(self, world_book_id: int, **_: Any) -> dict[str, Any]:
        return self._require_world_book_manager().export_world_book(int(world_book_id))

    def import_character_world_book(
        self,
        request_data: Any,
        *,
        name_override: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        world_book_id = self._require_world_book_manager().import_world_book(
            self._world_book_payload(request_data),
            name_override=name_override,
        )
        return self.get_character_world_book(world_book_id)

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import PersonaProfileCreate

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import PersonaProfileUpdate

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import PersonaExemplarCreate

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import PersonaExemplarCreate, PersonaExemplarImportRequest

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import PersonaExemplarUpdate

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import PersonaExemplarReviewRequest

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import CharacterExemplarSearchRequest

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import CharacterExemplarCreate

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import CharacterExemplarUpdate

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import (
            CharacterExemplarSearchRequest,
            CharacterExemplarSelectionDebugRequest,
        )

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import CharacterChatSessionCreate

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import CharacterChatSessionUpdate

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

    def list_character_chat_messages(
        self,
        chat_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        self._require_chat_session(chat_id)
        messages = [
            self.conversations.normalize_message_row(message)
            for message in self._require_db().get_messages_for_conversation(chat_id, limit=limit, offset=offset)
        ]
        messages = [message for message in messages if message is not None]
        return {"messages": messages, "total": len(messages), "limit": limit, "offset": offset, "source": "local"}

    def list_character_messages(self, chat_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.list_character_chat_messages(chat_id, **kwargs)

    def get_character_chat_message(self, message_id: str, **_: Any) -> dict[str, Any]:
        message = self.conversations.normalize_message_row(self._require_db().get_message_by_id(message_id))
        if message is None:
            raise ValueError(f"local_character_message_not_found:{message_id}")
        self._require_chat_session(str(message["conversation_id"]))
        return message

    def get_character_message(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.get_character_chat_message(message_id, **kwargs)

    def create_character_chat_message(self, chat_id: str, request_data: Any, **_: Any) -> dict[str, Any]:
        self._require_chat_session(chat_id)
        payload = _model_payload(request_data, exclude_none=False)
        role = payload.get("role") or payload.get("sender") or "user"
        message_id = self._require_db().add_message(
            {
                "conversation_id": str(chat_id),
                "sender": payload.get("sender") or role,
                "role": role,
                "content": payload.get("content", ""),
                "parent_message_id": payload.get("parent_message_id"),
            }
        )
        return self.get_character_chat_message(str(message_id))

    def create_character_message(self, chat_id: str, request_data: Any, **kwargs: Any) -> dict[str, Any]:
        return self.create_character_chat_message(chat_id, request_data, **kwargs)

    def update_character_chat_message(
        self,
        message_id: str,
        request_data: Any,
        *,
        expected_version: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        current = self.get_character_chat_message(message_id)
        payload = {
            key: value
            for key, value in _model_payload(request_data, exclude_none=False).items()
            if key in {"content", "ranking", "parent_message_id", "image_data", "image_mime_type"}
        }
        if not payload:
            payload = {"content": current.get("content", "")}
        self._require_db().update_message(
            message_id,
            payload,
            expected_version=int(expected_version if expected_version is not None else current.get("version", 1)),
        )
        return self.get_character_chat_message(message_id)

    def update_character_message(self, message_id: str, request_data: Any, **kwargs: Any) -> dict[str, Any]:
        return self.update_character_chat_message(message_id, request_data, **kwargs)

    def delete_character_chat_message(
        self,
        message_id: str,
        *,
        expected_version: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        current = self.get_character_chat_message(message_id)
        self._require_db().soft_delete_message(
            message_id,
            expected_version=int(expected_version if expected_version is not None else current.get("version", 1)),
        )
        return {"status": "deleted", "message_id": message_id}

    def delete_character_message(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.delete_character_chat_message(message_id, **kwargs)

    def search_character_messages(
        self,
        chat_id: str,
        query: str,
        *,
        limit: int = 100,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        page = self.list_character_chat_messages(chat_id, limit=limit, offset=offset)
        normalized_query = str(query or "").lower()
        messages = [
            message
            for message in page["messages"]
            if normalized_query in str(message.get("content") or "").lower()
        ]
        return {"messages": messages, "total": len(messages), "limit": limit, "offset": offset, "source": "local"}

    def search_character_chat_messages(self, chat_id: str, query: str, **kwargs: Any) -> dict[str, Any]:
        return self.search_character_messages(chat_id, query, **kwargs)

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

    def get_chat_settings(self, chat_id: str, **_: Any) -> dict[str, Any]:
        self._require_chat_session(chat_id)
        settings = dict(self._chat_settings.get(str(chat_id), {}))
        return {"conversation_id": str(chat_id), "settings": settings, "source": "local"}

    def update_chat_settings(self, chat_id: str, request_data: Any, **_: Any) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import ChatSettingsUpdate

        self._require_chat_session(chat_id)
        request = ChatSettingsUpdate.model_validate(_model_payload(request_data, exclude_none=False))
        self._chat_settings[str(chat_id)] = dict(request.settings)
        self._persist_persona_profiles()
        return self.get_chat_settings(chat_id)

    def get_character_chat_settings(self, chat_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.get_chat_settings(chat_id, **kwargs)

    def update_character_chat_settings(self, chat_id: str, request_data: Any, **kwargs: Any) -> dict[str, Any]:
        return self.update_chat_settings(chat_id, request_data, **kwargs)

    def list_character_memories(
        self,
        character_id: str,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        self._require_character(int(character_id))
        filtered = [
            self._character_memory_view(record)
            for record in self._character_memories
            if str(record.get("character_id")) == str(character_id)
            and (include_deleted or not record.get("deleted"))
            and (include_archived or not record.get("archived"))
        ]
        page = filtered[offset : offset + limit]
        return {"memories": page, "total": len(filtered), "limit": limit, "offset": offset, "source": "local"}

    def create_character_memory(self, character_id: str, request_data: Any, **_: Any) -> dict[str, Any]:
        self._require_character(int(character_id))
        payload = _model_payload(request_data, exclude_none=False)
        content = _clean_text(payload.get("content"))
        if content is None:
            raise ValueError("local_character_memory_content_required")
        now = self._now()
        record = {
            "id": str(payload.get("id") or uuid.uuid4()),
            "character_id": str(character_id),
            "content": content,
            "memory_type": payload.get("memory_type") or "manual",
            "archived": bool(payload.get("archived", False)),
            "deleted": False,
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "metadata": payload.get("metadata") or {},
        }
        self._character_memories.append(record)
        self._persist_persona_profiles()
        return self._character_memory_view(record)

    def update_character_memory(
        self,
        character_id: str,
        memory_id: str,
        request_data: Any,
        **_: Any,
    ) -> dict[str, Any]:
        record = self._find_character_memory(character_id, memory_id)
        payload = _model_payload(request_data, exclude_none=False)
        if "content" in payload:
            content = _clean_text(payload.get("content"))
            if content is None:
                raise ValueError("local_character_memory_content_required")
            record["content"] = content
        if "metadata" in payload:
            record["metadata"] = payload.get("metadata") or {}
        record["version"] = int(record.get("version", 1) or 1) + 1
        record["updated_at"] = self._now()
        self._persist_persona_profiles()
        return self._character_memory_view(record)

    def archive_character_memory(
        self,
        character_id: str,
        memory_id: str,
        request_data: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        record = self._find_character_memory(character_id, memory_id)
        payload = _model_payload(request_data or {}, exclude_none=False)
        record["archived"] = bool(payload.get("archived", True))
        record["version"] = int(record.get("version", 1) or 1) + 1
        record["updated_at"] = self._now()
        self._persist_persona_profiles()
        return self._character_memory_view(record)

    def delete_character_memory(self, character_id: str, memory_id: str, **_: Any) -> dict[str, Any]:
        record = self._find_character_memory(character_id, memory_id)
        record["deleted"] = True
        record["version"] = int(record.get("version", 1) or 1) + 1
        record["updated_at"] = self._now()
        self._persist_persona_profiles()
        return {"deleted": True}

    def extract_character_memories(self, character_id: str, request_data: Any, **_: Any) -> dict[str, Any]:
        self._require_character(int(character_id))
        payload = _model_payload(request_data, exclude_none=False)
        chat_id = str(payload.get("chat_id") or "")
        if not chat_id:
            raise ValueError("local_character_memory_chat_id_required")
        limit = int(payload.get("message_limit") or 100)
        messages = self._require_db().get_messages_for_conversation(chat_id, limit=limit, offset=0)
        extracted: list[dict[str, Any]] = []
        skipped_duplicates = 0
        existing = {
            str(record.get("content") or "").strip().lower()
            for record in self._character_memories
            if str(record.get("character_id")) == str(character_id) and not record.get("deleted")
        }
        for message in messages:
            content = str(message.get("content") or "").strip()
            marker = "remember that "
            marker_index = content.lower().find(marker)
            if marker_index < 0:
                continue
            memory_content = content[marker_index + len(marker) :].strip()
            if not memory_content:
                continue
            key = memory_content.lower()
            if key in existing:
                skipped_duplicates += 1
                continue
            memory = self.create_character_memory(
                character_id,
                {"content": memory_content, "memory_type": "extracted", "metadata": {"chat_id": chat_id}},
            )
            existing.add(key)
            extracted.append(memory)
        return {
            "extracted": len(extracted),
            "skipped_duplicates": skipped_duplicates,
            "memories": extracted,
            "source": "local",
        }

    def export_lorebook_diagnostics(self, chat_id: str, **kwargs: Any) -> dict[str, Any]:
        session = self._require_chat_session(chat_id)
        character = self._character_for_session(session)
        world_books = []
        if character is not None:
            for key in ("world_books", "world_info", "character_book"):
                value = character.get(key)
                if value:
                    world_books = value if isinstance(value, list) else [value]
                    break
        return {
            "chat_id": str(chat_id),
            "turns": [],
            "diagnostics": {
                "source": "local",
                "character_id": session.get("character_id"),
                "world_books": world_books,
                "options": dict(kwargs),
            },
        }

    def list_chat_greetings(self, chat_id: str) -> dict[str, Any]:
        session, greetings, warning = self._chat_greeting_items(chat_id)
        current_selection = self._chat_greeting_selections.get(str(chat_id))
        if current_selection is None and greetings:
            current_selection = 0
        if current_selection is not None and not (0 <= current_selection < len(greetings)):
            current_selection = 0 if greetings else None
        character = self._character_for_session(session)
        character_id = session.get("character_id")
        return {
            "chat_id": str(chat_id),
            "character_id": str(character_id) if character_id is not None else None,
            "character_name": character.get("name") if character else None,
            "greetings": greetings,
            "current_selection": current_selection,
            "staleness_warning": warning,
            "source": "local",
        }

    def select_chat_greeting(self, chat_id: str, index: int) -> dict[str, Any]:
        _, greetings, _ = self._chat_greeting_items(chat_id)
        normalized_index = int(index)
        if normalized_index < 0 or normalized_index >= len(greetings):
            raise ValueError(f"local_chat_greeting_not_found:{chat_id}:{index}")
        self._chat_greeting_selections[str(chat_id)] = normalized_index
        self._persist_persona_profiles()
        return {
            "chat_id": str(chat_id),
            "selected_index": normalized_index,
            "greeting_preview": greetings[normalized_index]["preview"],
            "checksum_updated": False,
            "source": "local",
        }

    def list_chat_presets(self) -> dict[str, Any]:
        presets = [self._builtin_chat_preset()]
        presets.extend(
            self._chat_preset_view(record)
            for record in self._chat_presets
            if not record.get("deleted", False)
        )
        return {"presets": presets, "source": "local"}

    def create_chat_preset(self, request_data: Any) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import PresetCreate

        payload = PresetCreate.model_validate(_model_payload(request_data)).model_dump(mode="json")
        preset_id = str(payload["preset_id"])
        if preset_id == "default" or any(
            str(record.get("preset_id")) == preset_id and not record.get("deleted", False)
            for record in self._chat_presets
        ):
            raise ValueError(f"local_chat_preset_exists:{preset_id}")
        now = self._now()
        record = {
            **payload,
            "builtin": False,
            "created_at": now,
            "updated_at": now,
            "deleted": False,
        }
        self._chat_presets.append(record)
        self._persist_persona_profiles()
        return self._chat_preset_view(record)

    def update_chat_preset(self, preset_id: str, request_data: Any) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.character_persona_schemas import PresetUpdate

        if str(preset_id) == "default":
            raise ValueError("local_builtin_chat_preset_read_only:default")
        record = self._find_chat_preset(preset_id)
        payload = PresetUpdate.model_validate(_model_payload(request_data, exclude_none=False)).model_dump(
            mode="json",
            exclude_none=True,
        )
        record.update(payload)
        record["updated_at"] = self._now()
        self._persist_persona_profiles()
        return self._chat_preset_view(record)

    def delete_chat_preset(self, preset_id: str) -> dict[str, Any]:
        if str(preset_id) == "default":
            raise ValueError("local_builtin_chat_preset_read_only:default")
        record = self._find_chat_preset(preset_id)
        record["deleted"] = True
        record["updated_at"] = self._now()
        self._persist_persona_profiles()
        return {"status": "deleted", "preset_id": str(preset_id), "source": "local"}


__all__ = ["LocalCharacterPersonaService"]
