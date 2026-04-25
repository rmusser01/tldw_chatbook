"""Local chat dictionary adapter for the source-aware dictionary seam."""

from __future__ import annotations

import re
from typing import Any

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


def _entries_payload(entries: list[cdl.ChatDictionary]) -> list[dict[str, Any]]:
    return [entry.to_dict() for entry in entries]


class LocalChatDictionaryService:
    """Wrap local chat dictionary storage behind server-shaped methods."""

    def __init__(self, db: Any):
        self.db = db

    def _require_db(self) -> Any:
        if self.db is None:
            raise ValueError("Local chat dictionary backend is unavailable.")
        return self.db

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
        deleted = cdl.delete_chat_dictionary(self._require_db(), int(dictionary_id), expected_version=expected_version)
        if not deleted:
            raise ValueError(f"Local chat dictionary '{dictionary_id}' could not be deleted.")
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

    def get_statistics(self, dictionary_id: int) -> dict[str, Any]:
        record = self._load_required_dictionary(int(dictionary_id))
        return {
            "dictionary_id": int(dictionary_id),
            "entry_count": len(record.get("entries") or []),
            "enabled": bool(record.get("enabled")),
            "source": "local",
        }


__all__ = ["LocalChatDictionaryService"]
