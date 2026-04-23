"""Markdown <-> server TipTap wrapper adapter."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


def markdown_to_server_content(markdown: str) -> dict[str, Any]:
    return {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "tldw_chatbook_markdown": True,
                    "format": "markdown",
                    "version": 1,
                },
                "content": [{"type": "text", "text": markdown}],
            }
        ],
    }


def markdown_to_plain_text(markdown: str) -> str:
    lines = str(markdown or "").splitlines()
    normalized: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        text = re.sub(r"^(#{1,6}\s+)", "", text)
        text = re.sub(r"^([-*+]\s+)", "", text)
        text = re.sub(r"^(\d+\.\s+)", "", text)
        text = re.sub(r"^(>\s+)", "", text)
        text = re.sub(r"[*_~`]+", "", text)
        if text:
            normalized.append(text)

    if normalized:
        return "\n".join(normalized)
    return str(markdown or "").strip()


def server_content_to_markdown(content: Mapping[str, Any] | None, content_plain: str | None) -> str:
    if isinstance(content, Mapping):
        content_list = content.get("content")
        if isinstance(content_list, list) and content_list:
            first = content_list[0]
            if isinstance(first, Mapping):
                attrs = first.get("attrs")
                first_content = first.get("content")
                if (
                    isinstance(attrs, Mapping)
                    and attrs.get("tldw_chatbook_markdown") is True
                    and attrs.get("format") == "markdown"
                    and attrs.get("version") == 1
                    and isinstance(first_content, list)
                    and first_content
                ):
                    text_node = first_content[0]
                    if isinstance(text_node, Mapping):
                        text = text_node.get("text")
                        if isinstance(text, str):
                            return text
    return str(content_plain or "")


def parse_server_content_json(content_json: str | None) -> dict[str, Any] | None:
    if not content_json:
        return None
    try:
        parsed = json.loads(content_json)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None
