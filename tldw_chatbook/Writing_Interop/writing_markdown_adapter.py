"""Markdown preservation helpers for server writing-scene interop."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


CHATBOOK_MARKDOWN_TYPE = "chatbook-markdown"
CHATBOOK_MARKDOWN_SCHEMA_VERSION = 1


def markdown_to_server_content(markdown: str | None) -> dict[str, Any]:
    """Wrap Chatbook-authored Markdown in a deterministic server content payload."""

    markdown_text = str(markdown or "")
    return {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "tldw_chatbook_markdown": True,
                    "format": "markdown",
                    "version": CHATBOOK_MARKDOWN_SCHEMA_VERSION,
                },
                "content": [{"type": "text", "text": markdown_text}],
            }
        ],
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
        markdown = server_content_to_markdown(content, None)
        if markdown:
            return markdown, "chatbook_markdown"

    return str(record.get("content_plain") or ""), "plain_text_fallback"


def server_content_to_markdown(content: Mapping[str, Any] | None, content_plain: str | None) -> str:
    """Recover Chatbook-authored Markdown from server content or fall back to plain text."""

    if not isinstance(content, Mapping):
        return str(content_plain or "")

    if (
        content.get("type") == CHATBOOK_MARKDOWN_TYPE
        and content.get("schema_version") == CHATBOOK_MARKDOWN_SCHEMA_VERSION
        and isinstance(content.get("markdown"), str)
    ):
        return str(content["markdown"])

    if content.get("type") == "doc":
        for node in content.get("content") or []:
            if not isinstance(node, Mapping):
                continue
            attrs = node.get("attrs")
            if not isinstance(attrs, Mapping) or not attrs.get("tldw_chatbook_markdown"):
                continue
            parts: list[str] = []
            for child in node.get("content") or []:
                if isinstance(child, Mapping) and child.get("type") == "text":
                    parts.append(str(child.get("text") or ""))
            return "".join(parts)

    return str(content_plain or "")


def parse_server_content_json(content_json: Any) -> dict[str, Any] | None:
    """Parse a server content JSON value only when it is a mapping."""

    parsed = _parse_content_json(content_json)
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _parse_content_json(content_json: Any) -> Any:
    if isinstance(content_json, Mapping):
        return content_json
    if not isinstance(content_json, str):
        return None
    try:
        return json.loads(content_json)
    except json.JSONDecodeError:
        return None
