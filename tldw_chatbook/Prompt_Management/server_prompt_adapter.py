"""
Helpers for converting between local prompt rows and server prompt payloads.
"""

import json
from typing import Any, Dict, List, Optional


def _normalize_keywords(keywords: Any) -> List[str]:
    if keywords is None:
        return []
    if isinstance(keywords, list):
        return [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    if isinstance(keywords, str):
        return [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]
    return [str(keyword).strip() for keyword in list(keywords) if str(keyword).strip()]


def _deserialize_prompt_definition(prompt_definition: Any) -> Optional[Any]:
    if prompt_definition is None:
        return None
    if isinstance(prompt_definition, (dict, list)):
        return prompt_definition
    if isinstance(prompt_definition, str):
        stripped = prompt_definition.strip()
        if not stripped:
            return None
        return json.loads(stripped)
    raise ValueError("Prompt definition must be a JSON string, dict, list, or None.")


def local_prompt_to_server_payload(local_prompt: Dict[str, Any]) -> Dict[str, Any]:
    prompt_definition = _deserialize_prompt_definition(local_prompt.get("prompt_definition"))
    payload: Dict[str, Any] = {
        "name": local_prompt.get("name"),
        "author": local_prompt.get("author"),
        "details": local_prompt.get("details"),
        "system_prompt": local_prompt.get("system_prompt"),
        "user_prompt": local_prompt.get("user_prompt"),
        "keywords": sorted(_normalize_keywords(local_prompt.get("keywords"))),
        "prompt_format": local_prompt.get("prompt_format") or "legacy",
        "prompt_schema_version": local_prompt.get("prompt_schema_version"),
        "prompt_definition": prompt_definition,
    }

    for passthrough_key in ("uuid", "version", "deleted"):
        if passthrough_key in local_prompt and local_prompt.get(passthrough_key) is not None:
            payload[passthrough_key] = local_prompt.get(passthrough_key)

    return payload


def local_prompt_to_preview_payload(local_prompt: Dict[str, Any]) -> Dict[str, Any]:
    payload = local_prompt_to_server_payload(local_prompt)
    payload.pop("keywords", None)
    payload.pop("uuid", None)
    payload.pop("version", None)
    payload.pop("deleted", None)
    return payload


def server_prompt_to_local_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    update: Dict[str, Any] = {}

    for field in ("name", "author", "details", "system_prompt", "user_prompt", "prompt_schema_version"):
        if field in payload:
            update[field] = payload.get(field)

    if "keywords" in payload:
        update["keywords"] = sorted(_normalize_keywords(payload.get("keywords")))

    if "prompt_format" in payload:
        update["prompt_format"] = payload.get("prompt_format") or "legacy"

    if "prompt_definition" in payload:
        update["prompt_definition"] = _deserialize_prompt_definition(payload.get("prompt_definition"))

    if "prompt_format" not in update and "prompt_definition" in update and update["prompt_definition"] is not None:
        update["prompt_format"] = "structured"

    return update
