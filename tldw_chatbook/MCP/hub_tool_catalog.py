"""Cross-server tool-catalog derivation for the MCP Hub Tools mode.

Pure functions (no Textual, no I/O) that turn the three tool-inventory
shapes the hub already collects — a local external-profile catalog record
(Phase 2 `local_external_catalog()` items), the built-in server's inventory
(`local_service.get_inventory()`), and a remote server's raw tool payload —
into a single normalized `HubTool` shape the Tools mode canvas (T5) and
inspector (T6) can render and filter uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

_MAX_TAGS = 5


@dataclass(frozen=True)
class HubTool:
    """A single tool normalized for cross-server display.

    Attributes:
        server_key: Stable source identifier, e.g. `local:docs`,
            `builtin:tldw_chatbook`, or `server:<target_id>`.
        server_label: Human-readable label for the owning server.
        source: One of `local`, `builtin`, `server`.
        name: Tool name as advertised by the source.
        description: Tool description (may be empty).
        input_schema: JSON schema dict when the source provided a
            non-empty one, else `None`.
        tags: Lowercased, deduplication-free extras (risk class /
            capabilities), capped at 5 entries.
        stale: True when the source's live connection is currently down
            (only meaningful for `local`).
        executable: True when the hub can currently invoke this tool.
    """

    server_key: str
    server_label: str
    source: str  # local|builtin|server
    name: str
    description: str
    input_schema: dict | None
    tags: tuple[str, ...]
    stale: bool
    executable: bool

    @property
    def tool_id(self) -> str:
        return f"{self.server_key}::{self.name}"


def _normalized_schema(raw: Any) -> dict | None:
    if isinstance(raw, dict) and raw:
        return raw
    return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def local_tools_from_record(record: dict) -> list[HubTool]:
    """Derive `HubTool`s from a local external-profile catalog record.

    Args:
        record: One item from `local_external_catalog()` — profile fields
            plus `discovery_snapshot` (with a `tools` list of raw dicts)
            and `is_connected`.

    Returns:
        One `HubTool` per tool in `discovery_snapshot["tools"]`, or an
        empty list when there is no snapshot.
    """
    snapshot = record.get("discovery_snapshot")
    if not isinstance(snapshot, Mapping):
        return []
    raw_tools = snapshot.get("tools")
    if not isinstance(raw_tools, list):
        return []
    profile_id = _text(record.get("profile_id"))
    server_key = f"local:{profile_id}"
    stale = not record.get("is_connected")
    tools: list[HubTool] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        name = _text(raw_tool.get("name"))
        if not name:
            continue
        tools.append(
            HubTool(
                server_key=server_key,
                server_label=profile_id,
                source="local",
                name=name,
                description=_text(raw_tool.get("description")),
                input_schema=_normalized_schema(raw_tool.get("inputSchema")),
                tags=(),
                stale=stale,
                executable=True,
            )
        )
    return tools


def builtin_tools_from_inventory(inventory: dict) -> list[HubTool]:
    """Derive `HubTool`s from the built-in server's inventory.

    Args:
        inventory: `local_service.get_inventory()` payload — a `tools`
            list of `{name, description}` entries.

    Returns:
        One `HubTool` per entry, always executable and never stale, with
        no input schema (the built-in tool registry doesn't expose one).
    """
    raw_tools = inventory.get("tools") if isinstance(inventory, Mapping) else None
    if not isinstance(raw_tools, list):
        return []
    tools: list[HubTool] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        name = _text(raw_tool.get("name"))
        if not name:
            continue
        tools.append(
            HubTool(
                server_key="builtin:tldw_chatbook",
                server_label="tldw_chatbook",
                source="builtin",
                name=name,
                description=_text(raw_tool.get("description")),
                input_schema=None,
                tags=(),
                stale=False,
                executable=True,
            )
        )
    return tools


def _extra_tags(raw_tool: Mapping[str, Any]) -> tuple[str, ...]:
    tags: list[str] = []
    risk_class = raw_tool.get("risk_class")
    if isinstance(risk_class, str) and risk_class.strip():
        tags.append(risk_class.strip().lower())
    capabilities = raw_tool.get("capabilities")
    if isinstance(capabilities, list):
        for entry in capabilities:
            if isinstance(entry, str) and entry.strip():
                tags.append(entry.strip().lower())
    return tuple(tags[:_MAX_TAGS])


def server_tools_from_inventory(payload: dict, *, target_id: str, target_label: str) -> list[HubTool]:
    """Derive `HubTool`s from a remote server's raw tool-inventory payload.

    Reads defensively since the payload comes straight off the wire: skips
    nameless entries and non-dict tool records entirely.

    Args:
        payload: `{"tools": [raw dicts]}` from a server target.
        target_id: Stable identifier for the owning server target.
        target_label: Human-readable label for the owning server target.

    Returns:
        One `HubTool` per valid tool entry. Never executable — server-
        source execution ships in Phase 4 — and never stale (a payload
        that was fetched at all implies a live connection at fetch time).
    """
    raw_tools = payload.get("tools") if isinstance(payload, Mapping) else None
    if not isinstance(raw_tools, list):
        return []
    server_key = f"server:{target_id}"
    tools: list[HubTool] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        name = _text(raw_tool.get("name"))
        if not name:
            continue
        tools.append(
            HubTool(
                server_key=server_key,
                server_label=target_label,
                source="server",
                name=name,
                description=_text(raw_tool.get("description")),
                input_schema=_normalized_schema(raw_tool.get("inputSchema")),
                tags=_extra_tags(raw_tool),
                stale=False,
                executable=False,
            )
        )
    return tools


def filter_tools(
    tools: list[HubTool],
    *,
    server_key: str | None = None,
    text: str | None = None,
) -> list[HubTool]:
    """Filter a `HubTool` list by exact server key and/or free text.

    Args:
        tools: Tools to filter.
        server_key: When given, keep only tools with an exact `server_key`
            match.
        text: When given, keep only tools whose `name` or `description`
            contains this text (case-insensitive).

    Returns:
        The filtered list, preserving input order.
    """
    filtered = tools
    if server_key:
        filtered = [tool for tool in filtered if tool.server_key == server_key]
    if text:
        needle = text.strip().lower()
        if needle:
            filtered = [
                tool
                for tool in filtered
                if needle in tool.name.lower() or needle in tool.description.lower()
            ]
    return filtered
