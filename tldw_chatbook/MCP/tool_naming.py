"""LLM-facing MCP tool name sanitization.

Tool identifiers handed to LLM providers must be restricted to
`[a-zA-Z0-9_-]` and at most 64 characters. MCP server/tool identities are
otherwise unconstrained (unicode, spaces, arbitrary length), so this module
derives a safe, deterministic, collision-resistant LLM-facing name of the
shape `mcp__<server>__<tool>` from them.

Pure stdlib, no repo dependencies - safe to import from anywhere.
"""

from __future__ import annotations

import hashlib
import re

MAX_NAME_LENGTH = 64

_INVALID_RUN_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_LABEL_PREFIX_RE = re.compile(r"^\w+:")

_HASH_HEX_LENGTH = 8
_HASH_RESERVED = 1 + _HASH_HEX_LENGTH  # "_" + 8 hex chars
_TRUNCATE_BASE_LENGTH = MAX_NAME_LENGTH - _HASH_RESERVED  # 55


def sanitize_component(text: str) -> str:
    """Reduce `text` to the LLM-safe alphabet `[a-zA-Z0-9_-]`.

    Runs of any other character (including unicode) collapse to a single
    `_`; leading/trailing `_` are then stripped. An input that is empty or
    entirely made of invalid characters becomes `"x"` so the result is
    never empty.

    Args:
        text: Arbitrary, possibly unicode, possibly empty text.

    Returns:
        A non-empty string containing only `[a-zA-Z0-9_-]`.
    """
    collapsed = _INVALID_RUN_RE.sub("_", text or "")
    stripped = collapsed.strip("_")
    return stripped or "x"


def _strip_server_label_prefix(server_key: str) -> str:
    """Drop a single leading `<word>:` label prefix, e.g. `local:`/`builtin:`."""
    return _LABEL_PREFIX_RE.sub("", server_key, count=1)


def _hash8(identity: str) -> str:
    """First 8 hex chars of sha256(identity), used as a collision-avoiding tag."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:_HASH_HEX_LENGTH]


def _truncate_with_hash(identity: str, base: str, extra_suffix: str = "") -> str:
    """Deterministically shrink `base` to fit within MAX_NAME_LENGTH.

    Reserves room for `_` + 8 hex chars of sha256(identity) + `extra_suffix`
    at the end, then truncates `base` to whatever room remains. The result
    is always <= MAX_NAME_LENGTH.

    Args:
        identity: The string hashed to produce the disambiguating suffix.
            For `llm_tool_name` this is the *original* pre-sanitization
            `"{server_key}::{tool_name}"` identity, so two long inputs that
            happen to share a truncated visible prefix still (almost
            always) resolve to different final names.
        base: The (already assembled/sanitized) name to truncate.
        extra_suffix: An additional literal suffix (e.g. a dedupe `_2`) to
            preserve after the hash tag.

    Returns:
        `base[:keep] + "_" + hash8 + extra_suffix`, length <= MAX_NAME_LENGTH.
    """
    hash_part = _hash8(identity)
    reserved = _HASH_RESERVED + len(extra_suffix)
    keep = max(MAX_NAME_LENGTH - reserved, 0)
    return f"{base[:keep]}_{hash_part}{extra_suffix}"


def llm_tool_name(server_key: str, tool_name: str) -> str:
    """Build the LLM-facing name for an MCP hub tool.

    Assembles `mcp__<server>__<tool>` from the sanitized components. A
    single leading `<word>:` label prefix on `server_key` (e.g. `local:`,
    `builtin:`, `server:`) is dropped before sanitizing (`tool_name` is
    never prefix-stripped, only sanitized).

    If the assembled name exceeds `MAX_NAME_LENGTH` (64) characters, it is
    truncated to 55 characters and suffixed with `_` + the first 8 hex
    characters of `sha256(f"{server_key}::{tool_name}")` - hashing the
    original, pre-sanitization identity - so the result is always exactly
    64 characters and stays deterministic across calls.

    Args:
        server_key: The MCP server's identity key, optionally prefixed
            with a `<word>:` label (`local:`, `builtin:`, ...).
        tool_name: The tool's raw name as exposed by the MCP server.

    Returns:
        A name matching `[a-zA-Z0-9_-]{1,64}` of the shape
        `mcp__<server>__<tool>`.
    """
    label = _strip_server_label_prefix(server_key)
    name = f"mcp__{sanitize_component(label)}__{sanitize_component(tool_name)}"
    if len(name) <= MAX_NAME_LENGTH:
        return name
    identity = f"{server_key}::{tool_name}"
    return _truncate_with_hash(identity, name)


def dedupe_names(names: list[str]) -> list[str]:
    """Rewrite a list of LLM tool names so every entry is unique.

    The first occurrence of any name is left untouched. Each later
    occurrence of an equal (or newly-colliding) name is suffixed `_2`,
    `_3`, ... in input order. If appending a suffix would push a name past
    `MAX_NAME_LENGTH`, the base name is re-truncated using the same
    hash-truncation mechanism as `llm_tool_name`, hashing the current
    (already-sanitized) name since no earlier server/tool identity is
    available to this function. The result is stable: calling this again
    with the same input (in the same order) yields the same output.

    Args:
        names: LLM-facing tool names, typically produced by
            `llm_tool_name`, possibly containing duplicates.

    Returns:
        A new list of the same length, all entries pairwise-unique, each
        matching `[a-zA-Z0-9_-]{1,64}`.
    """
    used: set[str] = set()
    next_suffix: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name not in used:
            used.add(name)
            result.append(name)
            continue

        n = next_suffix.get(name, 1)
        while True:
            n += 1
            suffix = f"_{n}"
            candidate = name + suffix
            if len(candidate) > MAX_NAME_LENGTH:
                candidate = _truncate_with_hash(name, name, suffix)
            if candidate not in used:
                break
        next_suffix[name] = n
        used.add(candidate)
        result.append(candidate)
    return result
