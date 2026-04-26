"""Local chat dictionary adapter for the source-aware dictionary seam."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import Chat_Dictionary_Lib as cdl


_ENTRY_ID_RE = re.compile(r"^local:chat_dictionary_entry:(?P<dictionary_id>\d+):(?P<index>\d+)$")


def _payload(value: Any, *, exclude_none: bool = True) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=exclude_none, exclude_unset=True, mode="json")
    return dict(value or {})


def _entry_from_payload(value: Any) -> cdl.ChatDictionary:
    data = _payload(value)
    pattern = data.get("pattern", data.get("key", ""))
    if data.get("type") == "regex" and not str(pattern).startswith("/"):
        pattern = f"/{pattern}/"
    replacement = data.get("replacement", data.get("content", ""))
    probability = data.get("probability", 1.0)
    if isinstance(probability, float) and probability <= 1:
        probability = int(probability * 100)
    return cdl.ChatDictionary(
        key=str(pattern),
        content=str(replacement),
        probability=int(probability),
        group=data.get("group"),
        timed_effects=data.get("timed_effects"),
        max_replacements=int(data.get("max_replacements", 1) or 1),
    )


def _entry_to_response(entry: cdl.ChatDictionary, *, dictionary_id: int, index: int) -> dict[str, Any]:
    entry_type = "regex" if entry.is_regex else "literal"
    return {
        "id": f"local:chat_dictionary_entry:{dictionary_id}:{index}",
        "dictionary_id": dictionary_id,
        "index": index,
        "pattern": entry.raw_key,
        "replacement": entry.content,
        "probability": entry.probability / 100,
        "group": entry.group,
        "timed_effects": entry.timed_effects,
        "max_replacements": entry.max_replacements,
        "type": entry_type,
        "enabled": True,
        "source": "local",
    }


def _parse_entry_id(entry_id: str) -> tuple[int, int]:
    match = _ENTRY_ID_RE.match(str(entry_id))
    if match is None:
        raise ValueError(
            "Local chat dictionary entry ids must use local:chat_dictionary_entry:<dictionary_id>:<index>."
        )
    return int(match.group("dictionary_id")), int(match.group("index"))


def _parse_markdown_entries(content: str) -> list[cdl.ChatDictionary]:
    entries: list[cdl.ChatDictionary] = []
    current_key: str | None = None
    current_lines: list[str] = []
    key_pattern = re.compile(r"^\s*([^:\n]+?)\s*:(.*)$")
    terminator = re.compile(r"^\s*---@@@---\s*$")

    def flush_current() -> None:
        nonlocal current_key, current_lines
        if current_key is None:
            return
        entries.append(cdl.ChatDictionary(key=current_key, content="\n".join(current_lines).strip()))
        current_key = None
        current_lines = []

    for raw_line in str(content or "").splitlines():
        if terminator.match(raw_line):
            flush_current()
            continue
        match = key_pattern.match(raw_line)
        if match:
            flush_current()
            key = match.group(1).strip()
            value = match.group(2).strip()
            if value == "|":
                current_key = key
                current_lines = []
            else:
                entries.append(cdl.ChatDictionary(key=key, content=value))
            continue
        if current_key is not None:
            current_lines.append(raw_line)

    flush_current()
    return entries


def _entries_payload(entries: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for entry in entries:
        if hasattr(entry, "to_dict"):
            payloads.append(entry.to_dict())
        elif isinstance(entry, Mapping):
            payloads.append(dict(entry))
    return payloads


class LocalChatDictionaryService:
    """Wrap local chat dictionary storage behind server-shaped methods."""

    def __init__(self, db: Any, *, history_store_path: str | Path | None = None):
        self.db = db
        self.history_store_path = Path(history_store_path).expanduser() if history_store_path is not None else None
        self._history: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._load_history()

    def _require_db(self) -> Any:
        if self.db is None:
            raise ValueError("Local chat dictionary backend is unavailable.")
        return self.db

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_history(self) -> None:
        if self.history_store_path is None or not self.history_store_path.exists():
            return
        try:
            payload = json.loads(self.history_store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._history = {}
            return
        dictionaries = payload.get("dictionaries", payload) if isinstance(payload, dict) else {}
        if not isinstance(dictionaries, dict):
            self._history = {}
            return
        self._history = {}
        for dictionary_id, bucket in dictionaries.items():
            if not isinstance(bucket, dict):
                continue
            activity = bucket.get("activity", [])
            versions = bucket.get("versions", [])
            self._history[str(dictionary_id)] = {
                "activity": [dict(item) for item in activity if isinstance(item, dict)]
                if isinstance(activity, list)
                else [],
                "versions": [dict(item) for item in versions if isinstance(item, dict)]
                if isinstance(versions, list)
                else [],
            }

    def _persist_history(self) -> None:
        if self.history_store_path is None:
            return
        self.history_store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.history_store_path.with_suffix(self.history_store_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps({"dictionaries": self._history}, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temp_path.replace(self.history_store_path)

    def _history_bucket(self, dictionary_id: int) -> dict[str, list[dict[str, Any]]]:
        key = str(int(dictionary_id))
        if key not in self._history:
            self._history[key] = {"activity": [], "versions": []}
        return self._history[key]

    @staticmethod
    def _dictionary_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
        dictionary_id = int(record["id"])
        return {
            "id": dictionary_id,
            "name": record.get("name"),
            "description": record.get("description"),
            "content": record.get("content"),
            "entries": _entries_payload(list(record.get("entries") or [])),
            "strategy": record.get("strategy"),
            "max_tokens": record.get("max_tokens"),
            "enabled": bool(record.get("enabled", True)),
            "created_at": record.get("created_at"),
            "last_modified": record.get("last_modified"),
            "version": int(record.get("version", 1) or 1),
            "source": "local",
        }

    def _record_history(self, dictionary_id: int, action: str, record: Mapping[str, Any]) -> None:
        snapshot = self._dictionary_snapshot(record)
        bucket = self._history_bucket(dictionary_id)
        now = self._now()
        revision = int(snapshot.get("version", 1) or 1)
        bucket["activity"].append(
            {
                "id": f"local:chat_dictionary_activity:{int(dictionary_id)}:{len(bucket['activity']) + 1}",
                "dictionary_id": int(dictionary_id),
                "action": action,
                "revision": revision,
                "created_at": now,
                "source": "local",
            }
        )
        existing_index = next(
            (
                index
                for index, item in enumerate(bucket["versions"])
                if int(item.get("revision", -1)) == revision
            ),
            None,
        )
        version_record = {
            "dictionary_id": int(dictionary_id),
            "revision": revision,
            "action": action,
            "name": snapshot.get("name"),
            "created_at": now,
            "snapshot": snapshot,
            "source": "local",
        }
        if existing_index is None:
            bucket["versions"].append(version_record)
        else:
            bucket["versions"][existing_index] = version_record
        self._persist_history()

    def _ensure_history_baseline(self, dictionary_id: int) -> None:
        bucket = self._history_bucket(dictionary_id)
        if bucket["versions"]:
            return
        record = self.get_dictionary(int(dictionary_id))
        if record is not None:
            self._record_history(int(dictionary_id), "baseline", record)

    def _normalize_dictionary(self, record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        normalized = dict(record)
        dictionary_id = int(normalized["id"])
        normalized["source"] = "local"
        normalized["is_active"] = bool(normalized.get("enabled", True))
        normalized["default_token_budget"] = normalized.get("max_tokens")
        normalized["entries"] = [
            _entry_to_response(entry, dictionary_id=dictionary_id, index=index)
            for index, entry in enumerate(normalized.get("entries") or [])
        ]
        return normalized

    def _load_required_dictionary(self, dictionary_id: int) -> dict[str, Any]:
        record = cdl.load_chat_dictionary(self._require_db(), int(dictionary_id))
        if record is None:
            raise ValueError(f"Local chat dictionary '{dictionary_id}' was not found.")
        return record

    @staticmethod
    def _dictionary_update_payload(request_data: Any) -> dict[str, Any]:
        payload = _payload(request_data, exclude_none=False)
        updates: dict[str, Any] = {}
        for key in ("name", "description", "content", "strategy"):
            if key in payload:
                updates[key] = payload[key]
        if "default_token_budget" in payload:
            updates["max_tokens"] = payload["default_token_budget"]
        if "max_tokens" in payload:
            updates["max_tokens"] = payload["max_tokens"]
        if "is_active" in payload:
            updates["enabled"] = payload["is_active"]
        if "enabled" in payload:
            updates["enabled"] = payload["enabled"]
        if "entries" in payload:
            updates["entries"] = [_entry_from_payload(entry) for entry in payload["entries"] or []]
        return updates

    def list_dictionaries(self, *, include_inactive: bool = False, include_usage: bool = False) -> dict[str, Any]:
        records = cdl.list_chat_dictionaries(
            self._require_db(),
            limit=1000,
            include_disabled=include_inactive,
        )
        dictionaries = []
        for record in records:
            normalized = self._normalize_dictionary(record)
            if normalized is None:
                continue
            if include_usage:
                normalized.setdefault("usage", {"conversation_count": None})
            dictionaries.append(normalized)
        return {"dictionaries": dictionaries, "source": "local"}

    def create_dictionary(self, request_data: Any) -> dict[str, Any]:
        payload = _payload(request_data)
        dictionary_id = cdl.save_chat_dictionary(
            self._require_db(),
            name=payload["name"],
            description=payload.get("description") or "",
            content=payload.get("content"),
            entries=[_entry_from_payload(entry) for entry in payload.get("entries") or []],
            max_tokens=int(payload.get("default_token_budget") or payload.get("max_tokens") or 1000),
            enabled=bool(payload.get("is_active", payload.get("enabled", True))),
        )
        if dictionary_id is None:
            raise ValueError("Local chat dictionary could not be created.")
        record = self.get_dictionary(int(dictionary_id))
        if record is None:
            raise ValueError("Created local chat dictionary could not be loaded.")
        self._record_history(int(dictionary_id), "create", record)
        return record

    def get_dictionary(self, dictionary_id: int) -> dict[str, Any] | None:
        return self._normalize_dictionary(cdl.load_chat_dictionary(self._require_db(), int(dictionary_id)))

    def update_dictionary(
        self,
        dictionary_id: int,
        request_data: Any,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        updates = self._dictionary_update_payload(request_data)
        updated = cdl.update_chat_dictionary(
            self._require_db(),
            int(dictionary_id),
            expected_version=expected_version,
            **updates,
        )
        if not updated:
            raise ValueError(f"Local chat dictionary '{dictionary_id}' could not be updated.")
        record = self.get_dictionary(int(dictionary_id))
        if record is None:
            raise ValueError(f"Local chat dictionary '{dictionary_id}' could not be loaded after update.")
        self._record_history(int(dictionary_id), "update", record)
        return record

    def delete_dictionary(
        self,
        dictionary_id: int,
        *,
        expected_version: int | None = None,
        hard_delete: bool = False,
    ) -> dict[str, Any]:
        if hard_delete:
            raise ValueError("Local chat dictionaries support soft delete only.")
        record = self._load_required_dictionary(int(dictionary_id))
        deleted = cdl.delete_chat_dictionary(self._require_db(), int(dictionary_id), expected_version=expected_version)
        if not deleted:
            raise ValueError(f"Local chat dictionary '{dictionary_id}' could not be deleted.")
        record["deleted"] = True
        record["version"] = int(record.get("version", 1) or 1) + 1
        self._record_history(int(dictionary_id), "delete", record)
        return {"status": "deleted", "dictionary_id": int(dictionary_id), "source": "local"}

    def add_entry(self, dictionary_id: int, request_data: Any) -> dict[str, Any]:
        record = self._load_required_dictionary(int(dictionary_id))
        entries = list(record.get("entries") or [])
        entries.append(_entry_from_payload(request_data))
        self.update_dictionary(
            int(dictionary_id),
            {"entries": _entries_payload(entries)},
            expected_version=record.get("version"),
        )
        return _entry_to_response(entries[-1], dictionary_id=int(dictionary_id), index=len(entries) - 1)

    def list_entries(self, dictionary_id: int, *, group: str | None = None) -> dict[str, Any]:
        record = self._load_required_dictionary(int(dictionary_id))
        entries = []
        for index, entry in enumerate(record.get("entries") or []):
            if group is not None and entry.group != group:
                continue
            entries.append(_entry_to_response(entry, dictionary_id=int(dictionary_id), index=index))
        return {"dictionary_id": int(dictionary_id), "entries": entries, "source": "local"}

    def update_entry(self, entry_id: str, request_data: Any) -> dict[str, Any]:
        dictionary_id, index = _parse_entry_id(entry_id)
        record = self._load_required_dictionary(dictionary_id)
        entries = list(record.get("entries") or [])
        if index >= len(entries):
            raise ValueError(f"Local chat dictionary entry '{entry_id}' was not found.")
        current = entries[index].to_dict()
        current.update(_payload(request_data, exclude_none=False))
        entries[index] = _entry_from_payload(current)
        self.update_dictionary(
            dictionary_id,
            {"entries": _entries_payload(entries)},
            expected_version=record.get("version"),
        )
        return _entry_to_response(entries[index], dictionary_id=dictionary_id, index=index)

    def delete_entry(self, entry_id: str) -> dict[str, Any]:
        dictionary_id, index = _parse_entry_id(entry_id)
        record = self._load_required_dictionary(dictionary_id)
        entries = list(record.get("entries") or [])
        if index >= len(entries):
            raise ValueError(f"Local chat dictionary entry '{entry_id}' was not found.")
        del entries[index]
        self.update_dictionary(
            dictionary_id,
            {"entries": _entries_payload(entries)},
            expected_version=record.get("version"),
        )
        return {"status": "deleted", "entry_id": str(entry_id), "source": "local"}

    def reorder_entries(self, dictionary_id: int, request_data: Any) -> dict[str, Any]:
        payload = _payload(request_data)
        record = self._load_required_dictionary(int(dictionary_id))
        entries = list(record.get("entries") or [])
        selected_indexes = []
        for entry_id in payload.get("entry_ids") or []:
            parsed_dictionary_id, index = _parse_entry_id(entry_id)
            if parsed_dictionary_id != int(dictionary_id):
                raise ValueError("Local chat dictionary entry ids must belong to the dictionary being reordered.")
            selected_indexes.append(index)
        selected = [entries[index] for index in selected_indexes if index < len(entries)]
        remainder = [entry for index, entry in enumerate(entries) if index not in set(selected_indexes)]
        reordered = selected + remainder
        self.update_dictionary(
            int(dictionary_id),
            {"entries": _entries_payload(reordered)},
            expected_version=record.get("version"),
        )
        return {"dictionary_id": int(dictionary_id), "entry_ids": list(payload.get("entry_ids") or []), "source": "local"}

    def process_text(self, request_data: Any) -> dict[str, Any]:
        payload = _payload(request_data)
        text = str(payload.get("text") or "")
        dictionary_id = payload.get("dictionary_id")
        group = payload.get("group")
        token_budget = int(payload.get("token_budget") or 5000)
        dictionaries = []
        if dictionary_id is not None:
            dictionaries.append(self._load_required_dictionary(int(dictionary_id)))
        else:
            dictionaries = [
                self._load_required_dictionary(int(item["id"]))
                for item in cdl.list_chat_dictionaries(self._require_db(), limit=1000)
            ]
        entries: list[cdl.ChatDictionary] = []
        strategy = "sorted_evenly"
        for dictionary in dictionaries:
            strategy = dictionary.get("strategy") or strategy
            for entry in dictionary.get("entries") or []:
                if group is not None and entry.group != group:
                    continue
                entries.append(entry)
        return {
            "text": text,
            "processed_text": cdl.process_user_input(text, entries, max_tokens=token_budget, strategy=strategy),
            "dictionary_id": dictionary_id,
            "source": "local",
        }

    def import_markdown(self, request_data: Any) -> dict[str, Any]:
        payload = _payload(request_data)
        dictionary_id = cdl.save_chat_dictionary(
            self._require_db(),
            name=payload["name"],
            description=payload.get("description") or "",
            content=payload.get("content") or "",
            entries=_parse_markdown_entries(payload.get("content") or ""),
            enabled=bool(payload.get("activate", True)),
        )
        if dictionary_id is None:
            raise ValueError("Local chat dictionary markdown import failed.")
        record = self.get_dictionary(int(dictionary_id))
        if record is not None:
            self._record_history(int(dictionary_id), "import", record)
        return {"dictionary_id": int(dictionary_id), "source": "local"}

    def export_markdown(self, dictionary_id: int) -> dict[str, Any]:
        record = self._load_required_dictionary(int(dictionary_id))
        content = record.get("content")
        if not content:
            lines = []
            for entry in record.get("entries") or []:
                if "\n" in entry.content:
                    lines.extend([f"{entry.raw_key}: |", entry.content, "---@@@---"])
                else:
                    lines.append(f"{entry.raw_key}: {entry.content}")
            content = "\n".join(lines) + ("\n" if lines else "")
        return {"dictionary_id": int(dictionary_id), "name": record.get("name"), "content": content, "source": "local"}

    def import_json(self, request_data: Any) -> dict[str, Any]:
        payload = _payload(request_data)
        data = dict(payload.get("data") or {})
        dictionary_id = cdl.save_chat_dictionary(
            self._require_db(),
            name=data.get("name") or payload.get("name") or "Imported Dictionary",
            description=data.get("description") or "",
            content=data.get("content"),
            entries=[_entry_from_payload(entry) for entry in data.get("entries") or []],
            max_tokens=int(data.get("default_token_budget") or data.get("max_tokens") or 1000),
            enabled=bool(payload.get("activate", data.get("enabled", True))),
        )
        if dictionary_id is None:
            raise ValueError("Local chat dictionary JSON import failed.")
        record = self.get_dictionary(int(dictionary_id))
        if record is not None:
            self._record_history(int(dictionary_id), "import", record)
        return {"dictionary_id": int(dictionary_id), "source": "local"}

    def export_json(self, dictionary_id: int) -> dict[str, Any]:
        record = self._load_required_dictionary(int(dictionary_id))
        return {
            "dictionary_id": int(dictionary_id),
            "data": {
                "name": record.get("name"),
                "description": record.get("description"),
                "content": record.get("content"),
                "entries": _entries_payload(record.get("entries") or []),
                "max_tokens": record.get("max_tokens"),
                "enabled": bool(record.get("enabled")),
                "version": record.get("version"),
            },
            "source": "local",
        }

    def list_activity(self, dictionary_id: int, *, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        self._ensure_history_baseline(int(dictionary_id))
        bucket = self._history_bucket(int(dictionary_id))
        activity = list(reversed(bucket["activity"]))
        page = activity[offset : offset + limit]
        return {
            "dictionary_id": int(dictionary_id),
            "activity": page,
            "total": len(activity),
            "limit": limit,
            "offset": offset,
            "source": "local",
        }

    def list_versions(self, dictionary_id: int, *, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        self._ensure_history_baseline(int(dictionary_id))
        bucket = self._history_bucket(int(dictionary_id))
        versions = sorted(bucket["versions"], key=lambda item: int(item.get("revision", 0)), reverse=True)
        summaries = [
            {key: value for key, value in version.items() if key != "snapshot"}
            for version in versions
        ]
        page = summaries[offset : offset + limit]
        return {
            "dictionary_id": int(dictionary_id),
            "versions": page,
            "total": len(summaries),
            "limit": limit,
            "offset": offset,
            "source": "local",
        }

    def get_version(self, dictionary_id: int, revision: int) -> dict[str, Any]:
        self._ensure_history_baseline(int(dictionary_id))
        bucket = self._history_bucket(int(dictionary_id))
        for version in bucket["versions"]:
            if int(version.get("revision", -1)) == int(revision):
                return dict(version)
        raise ValueError(f"local_chat_dictionary_version_not_found:{dictionary_id}:{revision}")

    def revert_version(self, dictionary_id: int, revision: int) -> dict[str, Any]:
        version = self.get_version(int(dictionary_id), int(revision))
        snapshot = dict(version["snapshot"])
        current = self._load_required_dictionary(int(dictionary_id))
        updated = cdl.update_chat_dictionary(
            self._require_db(),
            int(dictionary_id),
            name=snapshot.get("name"),
            description=snapshot.get("description") or "",
            content=snapshot.get("content"),
            entries=[_entry_from_payload(entry) for entry in snapshot.get("entries") or []],
            strategy=snapshot.get("strategy"),
            max_tokens=snapshot.get("max_tokens"),
            enabled=bool(snapshot.get("enabled", True)),
            expected_version=current.get("version"),
        )
        if not updated:
            raise ValueError(f"Local chat dictionary '{dictionary_id}' could not be reverted.")
        record = self.get_dictionary(int(dictionary_id))
        if record is None:
            raise ValueError(f"Local chat dictionary '{dictionary_id}' could not be loaded after revert.")
        self._record_history(int(dictionary_id), "revert", record)
        record["reverted_to_revision"] = int(revision)
        return record

    def get_statistics(self, dictionary_id: int) -> dict[str, Any]:
        record = self._load_required_dictionary(int(dictionary_id))
        return {
            "dictionary_id": int(dictionary_id),
            "entry_count": len(record.get("entries") or []),
            "enabled": bool(record.get("enabled")),
            "source": "local",
        }


__all__ = ["LocalChatDictionaryService"]
