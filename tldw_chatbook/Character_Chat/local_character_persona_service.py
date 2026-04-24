"""Local CCP session/message/memory adapter for the character/persona scope service."""

from __future__ import annotations

import base64
import json
import re
import uuid
from typing import Any, Mapping

from .world_book_manager import WorldBookManager


class LocalCharacterPersonaService:
    """Expose local ChaChaNotes character chat data through server-compatible method names."""

    _REMEMBER_PATTERNS = (
        re.compile(r"\bremember\s+that\s+(?P<content>.+?)[.!?]*$", re.IGNORECASE),
        re.compile(r"\bplease\s+remember\s+(?P<content>.+?)[.!?]*$", re.IGNORECASE),
    )

    def __init__(self, db: Any):
        self.db = db

    @staticmethod
    def _payload_dict(payload: Any, *, exclude_none: bool = True) -> dict[str, Any]:
        if hasattr(payload, "model_dump"):
            return payload.model_dump(exclude_none=exclude_none, mode="json")
        return dict(payload or {})

    def _ensure_adapter_schema(self) -> None:
        conn = self.db.get_connection()
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS LocalCharacterChatSettings (
                chat_id TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL DEFAULT '{}',
                version INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS LocalCharacterMemories (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'manual',
                salience REAL NOT NULL DEFAULT 0.7,
                archived INTEGER NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_local_character_memories_character_archived
            ON LocalCharacterMemories(character_id, archived, updated_at);
            """
        )
        conn.commit()

    def _conversation_record(self, chat_id: str, *, include_deleted: bool = False) -> dict[str, Any]:
        row = self.db.get_conversation_by_id(chat_id, include_deleted=include_deleted)
        if not row:
            raise ValueError(f"Local character chat session '{chat_id}' not found.")
        return row

    def _character_record(self, character_id: int | str) -> dict[str, Any]:
        record = self.db.get_character_card_by_id(int(character_id))
        if record is None:
            raise ValueError(f"Local character '{character_id}' not found.")
        return record

    def _world_book_manager(self) -> WorldBookManager:
        return WorldBookManager(self.db)

    def _world_book_record(self, world_book_id: int | str) -> dict[str, Any]:
        record = self._world_book_manager().get_world_book(int(world_book_id))
        if record is None:
            raise ValueError(f"Local character world book '{world_book_id}' not found.")
        return record

    def _world_book_entry_record(self, entry_id: int | str) -> dict[str, Any]:
        row = self.db.get_connection().execute(
            """
            SELECT id, world_book_id, keys, content, enabled, position, insertion_order,
                   selective, secondary_keys, case_sensitive, extensions, created_at, last_modified
            FROM world_book_entries
            WHERE id = ?
            """,
            (int(entry_id),),
        ).fetchone()
        if row is None:
            raise ValueError(f"Local character world book entry '{entry_id}' not found.")
        return {
            "id": row["id"],
            "world_book_id": row["world_book_id"],
            "keys": json.loads(row["keys"] or "[]"),
            "content": row["content"],
            "enabled": bool(row["enabled"]),
            "position": row["position"],
            "insertion_order": row["insertion_order"],
            "selective": bool(row["selective"]),
            "secondary_keys": json.loads(row["secondary_keys"] or "[]"),
            "case_sensitive": bool(row["case_sensitive"]),
            "extensions": json.loads(row["extensions"] or "{}"),
            "created_at": row["created_at"],
            "last_modified": row["last_modified"],
        }

    def list_characters(self, limit: int = 100, offset: int = 0) -> Any:
        return self.db.list_character_cards(limit=limit, offset=offset)

    def get_character(self, character_id: int | str) -> dict[str, Any]:
        return self._character_record(character_id)

    def create_character(self, request_data: Mapping[str, Any] | Any) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        character_id = self.db.add_character_card(payload)
        if character_id is None:
            raise ValueError("Local character could not be created.")
        return self._character_record(character_id)

    def update_character(
        self,
        character_id: int | str,
        request_data: Mapping[str, Any] | Any,
        expected_version: int,
    ) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        updated = self.db.update_character_card(int(character_id), payload, expected_version=expected_version)
        if not updated:
            raise ValueError(f"Local character '{character_id}' could not be updated.")
        return self._character_record(character_id)

    def delete_character(self, character_id: int | str, expected_version: int | None = None) -> dict[str, Any]:
        if expected_version is None:
            expected_version = int(self._character_record(character_id)["version"])
        deleted = self.db.soft_delete_character_card(int(character_id), expected_version=expected_version)
        if not deleted:
            raise ValueError(f"Local character '{character_id}' could not be deleted.")
        return {"deleted": True, "id": str(character_id)}

    def restore_character(self, character_id: int | str, expected_version: int) -> dict[str, Any]:
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT version, deleted FROM character_cards WHERE id = ?",
                (int(character_id),),
            ).fetchone()
            if row is None:
                raise ValueError(f"Local character '{character_id}' not found.")
            if int(row["version"]) != int(expected_version):
                raise ValueError(
                    f"Local character '{character_id}' version mismatch: "
                    f"expected {expected_version}, found {row['version']}."
                )
            if not bool(row["deleted"]):
                return self._character_record(character_id)
            # The existing FTS update trigger deletes OLD before inserting NEW.
            # Soft-deleted rows are absent from FTS, so seed OLD before undelete.
            conn.execute(
                """
                INSERT INTO character_cards_fts(rowid, name, description, personality, scenario, system_prompt)
                SELECT id, name, description, personality, scenario, system_prompt
                FROM character_cards
                WHERE id = ?
                """,
                (int(character_id),),
            )
            cursor = conn.execute(
                """
                UPDATE character_cards
                SET deleted = 0,
                    version = version + 1,
                    last_modified = CURRENT_TIMESTAMP,
                    client_id = ?
                WHERE id = ? AND version = ? AND deleted = 1
                """,
                (self.db.client_id, int(character_id), int(expected_version)),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Local character '{character_id}' could not be restored.")
        return self._character_record(character_id)

    def create_character_chat_session(
        self,
        request_data: Mapping[str, Any] | Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        character_id = payload.get("character_id")
        assistant_kind = payload.get("assistant_kind")
        assistant_id = payload.get("assistant_id")
        if character_id is not None and not assistant_kind:
            assistant_kind = "character"
            assistant_id = str(character_id)
        discovery_owner = "ccp_persona" if assistant_kind == "persona" else "ccp_character"
        discovery_entity_id = assistant_id or (str(character_id) if character_id is not None else None)
        chat_id = self.db.add_conversation(
            {
                "character_id": character_id,
                "assistant_kind": assistant_kind,
                "assistant_id": assistant_id,
                "persona_memory_mode": payload.get("persona_memory_mode"),
                "title": payload.get("title"),
                "parent_conversation_id": payload.get("parent_conversation_id"),
                "forked_from_message_id": payload.get("forked_from_message_id"),
                "state": payload.get("state"),
                "topic_label": payload.get("topic_label"),
                "cluster_id": payload.get("cluster_id"),
                "source": payload.get("source"),
                "external_ref": payload.get("external_ref"),
                "scope_type": payload.get("scope_type"),
                "workspace_id": payload.get("workspace_id"),
                "runtime_backend": "local",
                "discovery_owner": discovery_owner,
                "discovery_entity_id": discovery_entity_id,
            }
        )
        return self._conversation_record(chat_id)

    def list_character_chat_sessions(self, **kwargs: Any) -> dict[str, Any]:
        limit = int(kwargs.get("limit", 100))
        offset = int(kwargs.get("offset", 0))
        character_id = kwargs.get("character_id")
        if character_id is not None:
            rows = self.db.get_conversations_for_character(int(character_id), limit=limit, offset=offset)
        else:
            rows = self.db.list_all_active_conversations(limit=limit, offset=offset)
            rows = [
                row
                for row in rows
                if row.get("character_id") is not None or row.get("discovery_owner") in {"ccp_character", "ccp_persona"}
            ]
        return {"chats": rows, "total": len(rows), "limit": limit, "offset": offset}

    def get_character_chat_session(self, chat_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._conversation_record(chat_id, include_deleted=bool(kwargs.get("include_deleted", False)))

    def update_character_chat_session(
        self,
        chat_id: str,
        request_data: Mapping[str, Any] | Any,
        *,
        expected_version: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        self.db.update_conversation(chat_id, payload, expected_version=expected_version)
        return self._conversation_record(chat_id)

    def delete_character_chat_session(
        self,
        chat_id: str,
        *,
        expected_version: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if expected_version is None:
            expected_version = int(self._conversation_record(chat_id)["version"])
        self.db.soft_delete_conversation(chat_id, expected_version=expected_version)
        return {"deleted": True, "id": chat_id}

    def get_character_chat_settings(self, chat_id: str, **_kwargs: Any) -> dict[str, Any]:
        self._ensure_adapter_schema()
        self._conversation_record(chat_id)
        row = self.db.get_connection().execute(
            "SELECT settings_json, version FROM LocalCharacterChatSettings WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        if row is None:
            return {"conversation_id": chat_id, "settings": {}, "version": 0}
        return {
            "conversation_id": chat_id,
            "settings": json.loads(row["settings_json"] or "{}"),
            "version": int(row["version"]),
        }

    def update_character_chat_settings(
        self,
        chat_id: str,
        request_data: Mapping[str, Any] | Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self._ensure_adapter_schema()
        self._conversation_record(chat_id)
        payload = self._payload_dict(request_data, exclude_none=False)
        settings = payload.get("settings") or {}
        settings_json = json.dumps(settings, sort_keys=True)
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO LocalCharacterChatSettings (chat_id, settings_json)
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    version = LocalCharacterChatSettings.version + 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, settings_json),
            )
        return self.get_character_chat_settings(chat_id)

    @staticmethod
    def _sender_for_role(role: str | None) -> str:
        role_value = (role or "assistant").lower()
        if role_value == "user":
            return "User"
        if role_value == "system":
            return "System"
        if role_value == "tool":
            return "Tool"
        return "Assistant"

    def create_character_chat_message(
        self,
        chat_id: str,
        request_data: Mapping[str, Any] | Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        image_data = None
        image_mime_type = None
        image_base64 = payload.get("image_base64")
        if image_base64:
            if isinstance(image_base64, str) and "," in image_base64 and image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image_data = base64.b64decode(image_base64)
            image_mime_type = payload.get("image_mime_type") or "image/png"
        message_id = self.db.add_message(
            {
                "conversation_id": chat_id,
                "parent_message_id": payload.get("parent_message_id"),
                "sender": payload.get("sender") or self._sender_for_role(payload.get("role")),
                "role": payload.get("role"),
                "content": payload.get("content") or "",
                "image_data": image_data,
                "image_mime_type": image_mime_type,
            }
        )
        return self.get_character_chat_message(message_id)

    def list_character_chat_messages(self, chat_id: str, **kwargs: Any) -> dict[str, Any]:
        limit = int(kwargs.get("limit", 100))
        offset = int(kwargs.get("offset", 0))
        order = kwargs.get("order_by_timestamp", "ASC")
        rows = self.db.get_messages_for_conversation(
            chat_id,
            limit=limit,
            offset=offset,
            order_by_timestamp=order,
        )
        return {"messages": rows, "total": len(rows), "limit": limit, "offset": offset}

    def get_character_chat_message(self, message_id: str, **_kwargs: Any) -> dict[str, Any]:
        row = self.db.get_message_by_id(message_id)
        if not row:
            raise ValueError(f"Local character chat message '{message_id}' not found.")
        return row

    def update_character_chat_message(
        self,
        message_id: str,
        request_data: Mapping[str, Any] | Any,
        *,
        expected_version: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        update_data = {
            key: value
            for key, value in payload.items()
            if key in {"content", "ranking", "parent_message_id", "feedback"}
        }
        self.db.update_message(message_id, update_data, expected_version=expected_version)
        return self.get_character_chat_message(message_id)

    def delete_character_chat_message(
        self,
        message_id: str,
        *,
        expected_version: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.db.soft_delete_message(message_id, expected_version=expected_version)
        return {"deleted": True, "id": message_id}

    def search_character_chat_messages(self, chat_id: str, query: str, **kwargs: Any) -> dict[str, Any]:
        limit = int(kwargs.get("limit", 10))
        rows = self.db.search_messages_by_content(query, conversation_id=chat_id, limit=limit)
        return {"messages": rows, "total": len(rows), "limit": limit}

    def _memory_record(self, memory_id: str, *, include_archived: bool = False) -> dict[str, Any]:
        self._ensure_adapter_schema()
        query = "SELECT * FROM LocalCharacterMemories WHERE id = ?"
        params: tuple[Any, ...] = (memory_id,)
        if not include_archived:
            query += " AND archived = 0"
        row = self.db.get_connection().execute(query, params).fetchone()
        if not row:
            raise ValueError(f"Local character memory '{memory_id}' not found.")
        return {
            "id": row["id"],
            "character_id": row["character_id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "salience": float(row["salience"]),
            "archived": bool(row["archived"]),
            "version": int(row["version"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_character_memories(self, character_id: str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_adapter_schema()
        include_archived = bool(kwargs.get("include_archived", False))
        query = "SELECT id FROM LocalCharacterMemories WHERE character_id = ?"
        params: list[Any] = [str(character_id)]
        if not include_archived:
            query += " AND archived = 0"
        query += " ORDER BY updated_at DESC, id DESC"
        rows = self.db.get_connection().execute(query, tuple(params)).fetchall()
        memories = [self._memory_record(row["id"], include_archived=include_archived) for row in rows]
        return {"memories": memories, "total": len(memories)}

    def create_character_memory(
        self,
        character_id: str,
        request_data: Mapping[str, Any] | Any,
    ) -> dict[str, Any]:
        self._ensure_adapter_schema()
        payload = self._payload_dict(request_data)
        content = str(payload.get("content") or "").strip()
        if not content:
            raise ValueError("Local character memory content is required.")
        memory_id = str(uuid.uuid4())
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO LocalCharacterMemories (id, character_id, content, memory_type, salience)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    str(character_id),
                    content,
                    payload.get("memory_type") or "manual",
                    float(payload.get("salience", 0.7)),
                ),
            )
        return self._memory_record(memory_id)

    def update_character_memory(
        self,
        character_id: str,
        memory_id: str,
        request_data: Mapping[str, Any] | Any,
    ) -> dict[str, Any]:
        self._ensure_adapter_schema()
        payload = self._payload_dict(request_data)
        allowed = {key: payload[key] for key in ("content", "memory_type", "salience") if key in payload}
        if "content" in allowed:
            allowed["content"] = str(allowed["content"] or "").strip()
            if not allowed["content"]:
                raise ValueError("Local character memory content is required.")
        if not allowed:
            return self._memory_record(memory_id)
        set_clause = ", ".join(f"{key} = ?" for key in allowed)
        params = list(allowed.values()) + [str(character_id), memory_id]
        with self.db.transaction() as conn:
            cursor = conn.execute(
                f"""
                UPDATE LocalCharacterMemories
                SET {set_clause}, version = version + 1, updated_at = CURRENT_TIMESTAMP
                WHERE character_id = ? AND id = ?
                """,
                params,
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Local character memory '{memory_id}' not found.")
        return self._memory_record(memory_id, include_archived=True)

    def archive_character_memory(
        self,
        character_id: str,
        memory_id: str,
        request_data: Mapping[str, Any] | Any,
    ) -> dict[str, Any]:
        self._ensure_adapter_schema()
        payload = self._payload_dict(request_data)
        archived = bool(payload.get("archived", True))
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE LocalCharacterMemories
                SET archived = ?, version = version + 1, updated_at = CURRENT_TIMESTAMP
                WHERE character_id = ? AND id = ?
                """,
                (1 if archived else 0, str(character_id), memory_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Local character memory '{memory_id}' not found.")
        return self._memory_record(memory_id, include_archived=True)

    def delete_character_memory(self, character_id: str, memory_id: str) -> dict[str, Any]:
        self._ensure_adapter_schema()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM LocalCharacterMemories WHERE character_id = ? AND id = ?",
                (str(character_id), memory_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Local character memory '{memory_id}' not found.")
        return {"deleted": True}

    @staticmethod
    def _normalize_memory_for_duplicate_check(content: str) -> str:
        return re.sub(r"\s+", " ", str(content or "").strip().lower()).rstrip(".!?")

    @staticmethod
    def _finish_memory_sentence(content: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(content or "").strip().strip("\"'` "))
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    @classmethod
    def _extract_memory_candidates_from_text(cls, text: str) -> list[str]:
        candidates: list[str] = []
        for raw_sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            for pattern in cls._REMEMBER_PATTERNS:
                match = pattern.search(sentence)
                if match:
                    candidate = cls._finish_memory_sentence(match.group("content"))
                    if candidate:
                        candidates.append(candidate)
                    break
        return candidates

    def extract_character_memories(self, character_id: str, request_data: Mapping[str, Any] | Any) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        chat_id = payload.get("chat_id") or payload.get("conversation_id")
        if not chat_id:
            raise ValueError("Local character memory extraction requires chat_id.")

        conversation = self._conversation_record(str(chat_id))
        conversation_character_id = conversation.get("character_id")
        if conversation_character_id is not None and str(conversation_character_id) != str(character_id):
            raise ValueError("Local character memory extraction chat_id does not belong to character_id.")

        message_limit = max(1, int(payload.get("message_limit") or 25))
        memory_type = str(payload.get("memory_type") or "extracted")
        salience = float(payload.get("salience", 0.6))
        include_assistant = bool(payload.get("include_assistant", False))

        messages = self.list_character_chat_messages(str(chat_id), limit=message_limit).get("messages", [])
        existing = {
            self._normalize_memory_for_duplicate_check(memory.get("content", ""))
            for memory in self.list_character_memories(str(character_id), include_archived=True).get("memories", [])
        }

        created: list[dict[str, Any]] = []
        skipped_duplicates = 0
        for message in messages:
            role = str(message.get("role") or "").lower()
            sender = str(message.get("sender") or "").lower()
            if not include_assistant and role != "user" and sender != "user":
                continue
            for candidate in self._extract_memory_candidates_from_text(str(message.get("content") or "")):
                normalized_candidate = self._normalize_memory_for_duplicate_check(candidate)
                if not normalized_candidate:
                    continue
                if normalized_candidate in existing:
                    skipped_duplicates += 1
                    continue
                existing.add(normalized_candidate)
                created.append(
                    self.create_character_memory(
                        str(character_id),
                        {
                            "content": candidate,
                            "memory_type": memory_type,
                            "salience": salience,
                        },
                    )
                )

        return {
            "extracted": len(created),
            "skipped_duplicates": skipped_duplicates,
            "memories": created,
        }

    def list_character_world_books(self, **kwargs: Any) -> dict[str, Any]:
        include_disabled = bool(kwargs.get("include_disabled", False))
        world_books = self._world_book_manager().list_world_books(include_disabled=include_disabled)
        return {"world_books": world_books, "total": len(world_books)}

    def create_character_world_book(self, request_data: Mapping[str, Any] | Any) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        world_book_id = self._world_book_manager().create_world_book(
            name=str(payload.get("name") or ""),
            description=payload.get("description"),
            scan_depth=int(payload.get("scan_depth", 3)),
            token_budget=int(payload.get("token_budget", 500)),
            recursive_scanning=bool(payload.get("recursive_scanning", False)),
            enabled=bool(payload.get("enabled", True)),
        )
        return self._world_book_record(world_book_id)

    def get_character_world_book(self, world_book_id: int | str) -> dict[str, Any]:
        return self._world_book_record(world_book_id)

    def update_character_world_book(
        self,
        world_book_id: int | str,
        request_data: Mapping[str, Any] | Any,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        updated = self._world_book_manager().update_world_book(
            int(world_book_id),
            name=payload.get("name"),
            description=payload.get("description"),
            scan_depth=payload.get("scan_depth"),
            token_budget=payload.get("token_budget"),
            recursive_scanning=payload.get("recursive_scanning"),
            enabled=payload.get("enabled"),
            expected_version=expected_version,
        )
        if not updated:
            raise ValueError(f"Local character world book '{world_book_id}' not found.")
        return self._world_book_record(world_book_id)

    def delete_character_world_book(
        self,
        world_book_id: int | str,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        deleted = self._world_book_manager().delete_world_book(int(world_book_id), expected_version=expected_version)
        if not deleted:
            raise ValueError(f"Local character world book '{world_book_id}' not found.")
        return {"deleted": True, "id": str(world_book_id)}

    def list_character_world_book_entries(self, world_book_id: int | str, **kwargs: Any) -> dict[str, Any]:
        self._world_book_record(world_book_id)
        enabled_only = bool(kwargs.get("enabled_only", False))
        entries = self._world_book_manager().get_world_book_entries(int(world_book_id), enabled_only=enabled_only)
        return {"entries": entries, "total": len(entries)}

    def create_character_world_book_entry(
        self,
        world_book_id: int | str,
        request_data: Mapping[str, Any] | Any,
    ) -> dict[str, Any]:
        self._world_book_record(world_book_id)
        payload = self._payload_dict(request_data)
        entry_id = self._world_book_manager().create_world_book_entry(
            world_book_id=int(world_book_id),
            keys=list(payload.get("keys") or []),
            content=str(payload.get("content") or ""),
            enabled=bool(payload.get("enabled", True)),
            position=payload.get("position") or "before_char",
            insertion_order=int(payload.get("insertion_order", 0)),
            selective=bool(payload.get("selective", False)),
            secondary_keys=list(payload.get("secondary_keys") or []),
            case_sensitive=bool(payload.get("case_sensitive", False)),
            extensions=dict(payload.get("extensions") or {}),
        )
        return self._world_book_entry_record(entry_id)

    def get_character_world_book_entry(self, entry_id: int | str) -> dict[str, Any]:
        return self._world_book_entry_record(entry_id)

    def update_character_world_book_entry(
        self,
        entry_id: int | str,
        request_data: Mapping[str, Any] | Any,
    ) -> dict[str, Any]:
        self._world_book_entry_record(entry_id)
        payload = self._payload_dict(request_data)
        updated = self._world_book_manager().update_world_book_entry(int(entry_id), **payload)
        if not updated:
            raise ValueError(f"Local character world book entry '{entry_id}' not found.")
        return self._world_book_entry_record(entry_id)

    def delete_character_world_book_entry(self, entry_id: int | str) -> dict[str, Any]:
        deleted = self._world_book_manager().delete_world_book_entry(int(entry_id))
        if not deleted:
            raise ValueError(f"Local character world book entry '{entry_id}' not found.")
        return {"deleted": True, "id": str(entry_id)}

    def attach_character_world_book_to_session(
        self,
        chat_id: str,
        world_book_id: int | str,
        request_data: Mapping[str, Any] | Any | None = None,
    ) -> dict[str, Any]:
        self._conversation_record(chat_id)
        self._world_book_record(world_book_id)
        payload = self._payload_dict(request_data or {})
        priority = int(payload.get("priority", 0))
        self._world_book_manager().associate_world_book_with_conversation(
            chat_id,
            int(world_book_id),
            priority=priority,
        )
        return {"conversation_id": chat_id, "world_book_id": int(world_book_id), "priority": priority}

    def detach_character_world_book_from_session(self, chat_id: str, world_book_id: int | str) -> dict[str, Any]:
        self._conversation_record(chat_id)
        detached = self._world_book_manager().disassociate_world_book_from_conversation(chat_id, int(world_book_id))
        if not detached:
            raise ValueError(
                f"Local character world book '{world_book_id}' is not attached to session '{chat_id}'."
            )
        return {"deleted": True, "conversation_id": chat_id, "world_book_id": int(world_book_id)}

    def list_session_world_books(self, chat_id: str, **kwargs: Any) -> dict[str, Any]:
        self._conversation_record(chat_id)
        enabled_only = not bool(kwargs.get("include_disabled", False))
        world_books = self._world_book_manager().get_world_books_for_conversation(chat_id, enabled_only=enabled_only)
        return {"world_books": world_books, "total": len(world_books)}

    def export_character_world_book(self, world_book_id: int | str) -> dict[str, Any]:
        return self._world_book_manager().export_world_book(int(world_book_id))

    def import_character_world_book(
        self,
        request_data: Mapping[str, Any] | Any,
        *,
        name_override: str | None = None,
    ) -> dict[str, Any]:
        payload = self._payload_dict(request_data)
        world_book_id = self._world_book_manager().import_world_book(payload, name_override=name_override)
        return self._world_book_record(world_book_id)
