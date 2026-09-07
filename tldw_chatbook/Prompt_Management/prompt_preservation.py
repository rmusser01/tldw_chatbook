"""Conservative exact-byte preservation guards for improved prompts."""

from __future__ import annotations

from collections import Counter
import re


_TEMPLATE_PLACEHOLDER = re.compile(
    r"\{\{\s*\$?[A-Za-z_][A-Za-z0-9_. -]*\s*\}\}"
    r"|\{\$[A-Za-z_][A-Za-z0-9_. -]*\}"
    r"|<(?:USER|CHAR)>"
)
_FENCE_OPEN = re.compile(
    r"(?m)^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^\r\n]*)\r?\n"
)
_URL = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
_URL_SENTENCE_PUNCTUATION = ".,;:!?"
_UUID = re.compile(
    r"(?<![0-9A-Fa-f])"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    r"(?![0-9A-Fa-f])"
)
_XML_TOKEN = re.compile(
    r"<\s*(?P<close>/)?\s*(?P<name>[A-Za-z_][A-Za-z0-9_.:-]*)"
    r"(?:\s+[^<>]*?)?\s*(?P<self>/)?\s*>"
)
_OPAQUE_PLACEHOLDER = re.compile(
    r"\[\[TLDW_PROTECTED:[0-9a-f]{20}:\d+:[0-9a-f]{24}\]\]"
)


def _fenced_code(text: str) -> tuple[tuple[str, str, str, str], ...]:
    blocks: list[tuple[str, str, str, str]] = []
    position = 0
    while opening := _FENCE_OPEN.search(text, position):
        fence = opening.group("fence")
        character = fence[0]
        closing = re.compile(
            rf"(?m)^ {{0,3}}(?P<fence>{re.escape(character)}{{{len(fence)},}})"
            r"[ \t]*(?:\r?\n|\Z)"
        ).search(text, opening.end())
        if closing is None:
            position = opening.end()
            continue
        body = text[opening.end() : closing.start()]
        blocks.append((fence, opening.group("info"), body, closing.group("fence")))
        position = closing.end()
    return tuple(blocks)


def _xml_wrappers(text: str) -> tuple[tuple[str, str], ...]:
    tokens = list(_XML_TOKEN.finditer(text))
    stack: list[tuple[str, int]] = []
    matched_indexes: set[int] = set()
    for index, token in enumerate(tokens):
        if token.group("self"):
            continue
        name = token.group("name")
        if not token.group("close"):
            stack.append((name, index))
            continue
        if stack and stack[-1][0] == name:
            _, opening_index = stack.pop()
            matched_indexes.update((opening_index, index))
    return tuple(
        ("close" if token.group("close") else "open", token.group("name"))
        for index, token in enumerate(tokens)
        if index in matched_indexes
    )


def _urls(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _URL.finditer(text):
        value = match.group(0).rstrip(_URL_SENTENCE_PUNCTUATION)
        for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
            while value.endswith(closing) and value.count(closing) > value.count(
                opening
            ):
                value = value[:-1]
        if value:
            values.append(value)
    return tuple(values)


def preservation_violations(source: str, result: str) -> tuple[str, ...]:
    """Return deterministic protected categories whose exact contract changed."""
    if not isinstance(source, str) or not isinstance(result, str):
        raise TypeError("Preservation inputs must be text.")
    violations: list[str] = []
    if Counter(_TEMPLATE_PLACEHOLDER.findall(source)) != Counter(
        _TEMPLATE_PLACEHOLDER.findall(result)
    ):
        violations.append("template_placeholder")
    if _fenced_code(source) != _fenced_code(result):
        violations.append("fenced_code")
    if Counter(_urls(source)) != Counter(_urls(result)):
        violations.append("url")
    if Counter(_UUID.findall(source)) != Counter(_UUID.findall(result)):
        violations.append("uuid")
    if _xml_wrappers(source) != _xml_wrappers(result):
        violations.append("xml_wrapper")
    if _OPAQUE_PLACEHOLDER.findall(source) != _OPAQUE_PLACEHOLDER.findall(result):
        violations.append("opaque_placeholder")
    return tuple(violations)
