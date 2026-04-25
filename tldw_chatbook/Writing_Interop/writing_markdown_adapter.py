"""Markdown preservation helpers for server writing-scene interop."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


CHATBOOK_MARKDOWN_TYPE = "chatbook-markdown"
CHATBOOK_MARKDOWN_SCHEMA_VERSION = 1


def markdown_to_server_content(markdown: str | None) -> dict[str, Any]:
    """Wrap Chatbook-authored Markdown in a deterministic server content payload."""

    return {
        "type": CHATBOOK_MARKDOWN_TYPE,
        "schema_version": CHATBOOK_MARKDOWN_SCHEMA_VERSION,
        "markdown": str(markdown or ""),
    }


def markdown_to_plain_text(markdown: str | None) -> str:
    """Create a stable plain-text companion for server search/word-count fields."""

    text = str(markdown or "")
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
    text = re.sub(r"[*_`~]+", "", text)
    return text.strip()


def extract_markdown_from_server_scene(record: Mapping[str, Any]) -> tuple[str, str]:
    """Return Markdown plus fidelity label from a server scene payload."""

    explicit_markdown = record.get("content_markdown")
    if explicit_markdown is not None:
        return str(explicit_markdown), "native_markdown"

    content = record.get("content")
    if content is None and record.get("content_json") is not None:
        content = _parse_content_json(record.get("content_json"))

    if isinstance(content, Mapping):
        if (
            content.get("type") == CHATBOOK_MARKDOWN_TYPE
            and content.get("schema_version") == CHATBOOK_MARKDOWN_SCHEMA_VERSION
            and isinstance(content.get("markdown"), str)
        ):
            return content["markdown"], "chatbook_markdown"

    return str(record.get("content_plain") or ""), "plain_text_fallback"


def _parse_content_json(content_json: Any) -> Any:
    if isinstance(content_json, Mapping):
        return content_json
    if not isinstance(content_json, str):
        return None
    try:
        return json.loads(content_json)
    except json.JSONDecodeError:
        return None
