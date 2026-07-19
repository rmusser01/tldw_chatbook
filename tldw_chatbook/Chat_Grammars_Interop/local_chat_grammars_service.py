"""Local Chatbook-owned saved chat grammar store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LocalChatGrammarsService:
    """Persist saved grammars for local/offline structured chat output."""

    def __init__(self, *, store_path: str | Path, policy_enforcer: Any | None = None) -> None:
        self.store_path = Path(store_path).expanduser()
        self.policy_enforcer = policy_enforcer
        self._records: list[dict[str, Any]] = []
        self._next_id = 1
        self._load()

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._records = []
            self._next_id = 1
            return
        records = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return
        self._records = [dict(item) for item in records if isinstance(item, dict)]
        max_id = 0
        for record in self._records:
            raw_id = str(record.get("id") or "")
            if raw_id.startswith("local-grammar-"):
                try:
                    max_id = max(max_id, int(raw_id.removeprefix("local-grammar-")))
                except ValueError:
                    continue
        self._next_id = max_id + 1

    def _persist(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps({"items": self._records}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self.store_path)

    @staticmethod
    def _view(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": record["id"],
            "name": record["name"],
            "description": record.get("description"),
            "grammar_text": record["grammar_text"],
            "validation_status": record.get("validation_status", "unchecked"),
            "validation_error": record.get("validation_error"),
            "last_validated_at": record.get("last_validated_at"),
            "is_archived": bool(record.get("is_archived", False)),
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "version": int(record.get("version", 1) or 1),
        }

    def _find(self, grammar_id: str, *, include_archived: bool = False) -> dict[str, Any]:
        for record in self._records:
            if record.get("id") != grammar_id:
                continue
            if record.get("is_archived") and not include_archived:
                break
            return record
        raise ValueError(f"local_chat_grammar_not_found:{grammar_id}")

    async def create_grammar(
        self,
        *,
        name: str,
        grammar_text: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import ChatGrammarCreate

        self._enforce("chat.grammars.create.local")
        request = ChatGrammarCreate(name=name, description=description, grammar_text=grammar_text)
        grammar_id = f"local-grammar-{self._next_id}"
        self._next_id += 1
        now = self._now()
        record = request.model_dump(mode="json")
        record.update(
            {
                "id": grammar_id,
                "validation_status": "unchecked",
                "validation_error": None,
                "last_validated_at": None,
                "is_archived": False,
                "created_at": now,
                "updated_at": now,
                "version": 1,
            }
        )
        self._records.append(record)
        self._persist()
        return self._view(record)

    async def list_grammars(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._enforce("chat.grammars.list.local")
        records = [
            self._view(record)
            for record in self._records
            if include_archived or not record.get("is_archived", False)
        ]
        records = sorted(records, key=lambda item: item["created_at"], reverse=True)
        page = records[offset : offset + limit]
        return {"items": page, "total": len(records), "limit": limit, "offset": offset}

    async def get_grammar(self, grammar_id: str, *, include_archived: bool = False) -> dict[str, Any]:
        self._enforce("chat.grammars.detail.local")
        return self._view(self._find(grammar_id, include_archived=include_archived))

    async def update_grammar(
        self,
        grammar_id: str,
        *,
        version: int | None = None,
        name: str | None = None,
        description: str | None = None,
        grammar_text: str | None = None,
        validation_status: str | None = None,
        validation_error: str | None = None,
        last_validated_at: Any = None,
        is_archived: bool | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import ChatGrammarUpdate

        self._enforce("chat.grammars.update.local")
        record = self._find(grammar_id, include_archived=True)
        current_version = int(record.get("version", 1) or 1)
        if version is not None and int(version) != current_version:
            raise ValueError(f"local_chat_grammar_version_conflict:{grammar_id}")
        update = ChatGrammarUpdate(
            version=version,
            name=name,
            description=description,
            grammar_text=grammar_text,
            validation_status=validation_status,  # type: ignore[arg-type]
            validation_error=validation_error,
            last_validated_at=last_validated_at,
            is_archived=is_archived,
        )
        payload = update.model_dump(mode="json", exclude_none=True)
        payload.pop("version", None)
        record.update(payload)
        record["updated_at"] = self._now()
        record["version"] = current_version + 1
        self._persist()
        return self._view(record)

    async def delete_grammar(self, grammar_id: str, *, hard_delete: bool = False) -> bool:
        self._enforce("chat.grammars.delete.local")
        record = self._find(grammar_id, include_archived=True)
        if hard_delete:
            self._records = [item for item in self._records if item.get("id") != grammar_id]
        else:
            record["is_archived"] = True
            record["updated_at"] = self._now()
            record["version"] = int(record.get("version", 1) or 1) + 1
        self._persist()
        return True
