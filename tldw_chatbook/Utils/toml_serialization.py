"""Keep the existing TOML codecs while preserving literal backslash-x values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import toml


_BASIC_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _contains_literal_backslash_x(value: Any) -> bool:
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if type(current) is str:  # noqa: E721 - inspect builtin values without custom coercion.
            if "\\x" in current:
                return True
        elif type(current) in (dict, list, tuple):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            # Only builtin containers enter this branch; custom mappings keep the legacy route.
            pending.extend(current.values() if type(current) is dict else current)  # noqa: E721
    return False


def _dump_basic_string(value: str) -> str:
    escaped = []
    for character in value:
        replacement = _BASIC_ESCAPES.get(character)
        if replacement is not None:
            escaped.append(replacement)
            continue
        codepoint = ord(character)
        if codepoint < 0x20 or codepoint == 0x7F or 0xD800 <= codepoint <= 0xDFFF:
            # Escaped surrogate values still fail the writer's original
            # tomllib parse-back, rather than reaching an invalid UTF-8 write.
            escaped.append(f"\\u{codepoint:04x}")
        else:
            escaped.append(character)
    return '"' + "".join(escaped) + '"'


def dumps_cli_config(config_data: Mapping[str, Any]) -> str:
    """Serialize config values with the original encoder except literal \\x.

    Args:
        config_data: The existing config writer's mapping of TOML values.

    Returns:
        TOML text using unchanged key and ordinary-value codecs.

    Raises:
        ValueError: The original encoder rejects a circular mapping.
    """
    if not _contains_literal_backslash_x(config_data):
        # Keep the one-argument route, including existing serialization fault
        # controls. No custom encoder is created for ordinary configuration.
        return toml.dumps(config_data)
    encoder = toml.TomlEncoder()
    original_dump_string = encoder.dump_funcs[str]

    def dump_string(value: Any) -> str:
        if type(value) is str and "\\x" in value:  # noqa: E721 - match the original exact-type codec.
            return _dump_basic_string(value)
        return original_dump_string(value)

    encoder.dump_funcs[str] = dump_string
    return toml.dumps(config_data, encoder=encoder)
