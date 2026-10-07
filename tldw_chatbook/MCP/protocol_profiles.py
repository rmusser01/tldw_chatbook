"""Closed MCP wire profiles; no guessed version or deprecated-SSE fallback.

Fixture source pins: modern transport 0cb6c6a, schema 271ecc9;
2025-11-25 transport 977e748; 2025-03-26 transport 2fa3861.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

SUPPORTED_VERSIONS = frozenset({"2026-07-28", "2025-11-25", "2025-03-26"})
META_PREFIX = "io.modelcontextprotocol/"
_HEADER_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")


@dataclass(frozen=True)
class ProtocolProfile:
    """The entire selected wire era, shared by stdio and HTTP."""

    version: str

    @property
    def modern(self) -> bool:
        return self.version == "2026-07-28"

    @property
    def batches(self) -> bool:
        return self.version == "2025-03-26"

    def params(self, params: dict[str, Any], client_name: str) -> dict[str, Any]:
        """Copy parameters, setting host protocol metadata last."""
        if not self.modern:
            return dict(params)
        metadata = params.get("_meta", {})
        if not isinstance(metadata, dict):
            raise ValueError("mcp_protocol_invalid")  # noqa: TRY004
        return {
            **params,
            "_meta": {
                **metadata,
                META_PREFIX + "protocolVersion": self.version,
                META_PREFIX + "clientCapabilities": {},
                META_PREFIX + "clientInfo": {"name": client_name, "version": "1.0.0"},
            },
        }


def protocol_profile(version: str) -> ProtocolProfile:
    """Return only an explicitly qualified profile."""
    if version not in SUPPORTED_VERSIONS:
        raise ValueError("mcp_protocol_unsupported")
    return ProtocolProfile(version)


def header_value(value: str | int | bool) -> str:
    """Encode MCP routing values per modern UTF-8 Base64 sentinel rules."""
    if type(value) is bool:
        value = "true" if value else "false"
    elif type(value) is int:
        if abs(value) > 2**53 - 1:
            raise ValueError("mcp_header_value_invalid")
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("mcp_header_value_invalid")  # noqa: TRY004
    if (
        value != value.strip()
        or any(not (32 <= ord(char) <= 126 or char == "\t") for char in value)
        or (value.startswith("=?base64?") and value.endswith("?="))
    ):
        return "=?base64?" + base64.b64encode(value.encode("utf-8")).decode() + "?="
    return value


def tool_header_paths(
    schema: dict[str, Any],
) -> tuple[tuple[tuple[str, ...], str, str], ...]:
    """Validate annotations everywhere; only properties-only paths are eligible."""
    found: list[tuple[tuple[str, ...], str, str]] = []
    names: set[str] = set()

    def visit(node: Any, path: tuple[str, ...], reachable: bool) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child, path, False)
        elif isinstance(node, dict):
            if "x-mcp-header" in node:
                name, kind = node["x-mcp-header"], node.get("type")
                if (
                    not reachable
                    or not path
                    or not isinstance(name, str)
                    or not _HEADER_TOKEN.fullmatch(name)
                    or name.lower() in names
                    or kind not in ("string", "integer", "boolean")
                ):
                    raise ValueError("mcp_header_schema_invalid")
                names.add(name.lower())
                found.append((path, name, kind))
            for key, child in node.items():
                if key == "properties" and isinstance(child, dict):
                    for name, prop in child.items():
                        visit(prop, (*path, name), reachable)
                elif isinstance(child, (dict, list)):
                    visit(child, path, False)

    visit(schema, (), True)
    return tuple(found)
