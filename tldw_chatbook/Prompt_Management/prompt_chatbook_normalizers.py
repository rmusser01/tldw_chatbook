"""Source-aware normalizers for prompt and chatbook parity services."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def _as_dict(record: Any) -> dict[str, Any]:
    if record is None:
        return {}
    if isinstance(record, dict):
        return dict(record)
    model_dump = getattr(record, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json", exclude_none=True))
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if is_dataclass(record):
        return asdict(record)
    return {"value": record}


def _record_identifier(record: dict[str, Any], fallback: str) -> str:
    for key in ("id", "uuid", "job_id", "name"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)

    manifest = record.get("manifest")
    if isinstance(manifest, dict):
        value = manifest.get("name") or manifest.get("id")
        if value is not None and str(value).strip():
            return str(value)

    return fallback


def normalize_prompt_record(source: str, record: Any) -> dict[str, Any]:
    normalized = _as_dict(record)
    identifier = _record_identifier(normalized, "unsaved")
    normalized.setdefault("source", source)
    normalized.setdefault("record_type", "prompt")
    normalized.setdefault("record_id", f"{source}:prompt:{identifier}")
    return normalized


def normalize_prompt_result(source: str, result: Any) -> Any:
    if isinstance(result, list):
        return [normalize_prompt_record(source, item) for item in result]
    if isinstance(result, dict):
        if isinstance(result.get("items"), list):
            normalized = dict(result)
            normalized["items"] = [normalize_prompt_record(source, item) for item in result["items"]]
            return normalized
        return normalize_prompt_record(source, result)
    return result


def normalize_chatbook_record(source: str, record_type: str, record: Any) -> dict[str, Any]:
    normalized = _as_dict(record)
    identifier = _record_identifier(normalized, "unsaved")
    normalized.setdefault("source", source)
    normalized.setdefault("record_type", record_type)
    normalized.setdefault("record_id", f"{source}:{record_type}:{identifier}")
    return normalized


def normalize_chatbook_result(source: str, record_type: str, result: Any) -> Any:
    if isinstance(result, list):
        return [normalize_chatbook_record(source, record_type, item) for item in result]
    if isinstance(result, dict):
        if isinstance(result.get("items"), list):
            normalized = dict(result)
            normalized["items"] = [normalize_chatbook_record(source, record_type, item) for item in result["items"]]
            return normalized
        return normalize_chatbook_record(source, record_type, result)
    return result
