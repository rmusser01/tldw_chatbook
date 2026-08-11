"""Cross-server tool-catalog derivation for the MCP Hub Tools mode.

Pure functions (no Textual, no I/O) that turn the three tool-inventory
shapes the hub already collects — a local external-profile catalog record
(Phase 2 `local_external_catalog()` items), the built-in server's inventory
(`local_service.get_inventory()`), and a remote server's raw tool payload —
into a single normalized `HubTool` shape the Tools mode canvas (T5) and
inspector (T6) can render and filter uniformly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

_MAX_TAGS = 5
_RESERVED_EXTERNAL_PROFILE_ID = "__local__"


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
        # task-1337 (plan Task 8): defensively COPY a non-empty schema --
        # aliasing the source mapping would let a later mutation of the
        # inventory/record rewrite an already-derived HubTool's schema.
        # Empty/non-dict values still normalize to None; no schema is ever
        # synthesized for entries lacking one.
        return copy.deepcopy(raw)
    return None


def schema_argument_names(input_schema: dict | None) -> set[str]:
    """The schema-approved argument NAMES for a tool's execution-log record.

    Reads only the top-level ``properties`` keys of ``input_schema`` — the
    same shape `mcp_schema_form.parse_schema()` renders the Test Tool form
    from — but stays lenient where that parser is strict about rendering: a
    schema with a nested/unrenderable property (which would make
    `parse_schema()` fall back to the raw-JSON text area) still names its
    top-level arguments correctly here. This function only ever inspects
    and returns argument NAMES, never values.

    Args:
        input_schema: A tool's JSON schema (`HubTool.input_schema`), or
            `None`/malformed when the source didn't advertise one.

    Returns:
        The schema's top-level ``properties`` keys, or an empty set when
        the schema is absent or not the expected shape.
    """
    if not isinstance(input_schema, dict):
        return set()
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return set()
    return {str(name) for name in properties}


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
    profile_id = _text(record.get("profile_id"))
    if profile_id == _RESERVED_EXTERNAL_PROFILE_ID:
        return []
    snapshot = record.get("discovery_snapshot")
    if not isinstance(snapshot, Mapping):
        return []
    raw_tools = snapshot.get("tools")
    if not isinstance(raw_tools, list):
        return []
    server_key = f"local:{profile_id}"
    stale = not record.get("is_connected")
    tools: list[HubTool] = []
    seen_names: set[str] = set()
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        name = _text(raw_tool.get("name"))
        if not name or name in seen_names:
            # Duplicate tool name from the same server: keep only the first
            # occurrence. `HubTool.tool_id` is `f"{server_key}::{name}"`,
            # which is used as a Textual DataTable row key (mcp_tools_mode.py)
            # -- a second row with the same key raises `DuplicateKey` and
            # crashes every mount that renders this catalog (persisted
            # discovery snapshots make this a permanent crash-loop, not a
            # one-off).
            continue
        seen_names.add(name)
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
            list of `{name, description, inputSchema}` entries (RAG-48
            part 1 synthesizes `inputSchema` from each tool function's AST
            signature in `MCP/server.py`).

    Returns:
        One `HubTool` per entry, always executable and never stale, with
        `input_schema` populated when the source entry carries a non-empty
        `inputSchema`, else `None`.
    """
    raw_tools = inventory.get("tools") if isinstance(inventory, Mapping) else None
    if not isinstance(raw_tools, list):
        return []
    tools: list[HubTool] = []
    seen_names: set[str] = set()
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        name = _text(raw_tool.get("name"))
        if not name or name in seen_names:
            # See local_tools_from_record()'s dedup comment -- same
            # DuplicateKey hazard, this server's own tool_id namespace.
            continue
        seen_names.add(name)
        tools.append(
            HubTool(
                server_key="builtin:tldw_chatbook",
                server_label="tldw_chatbook",
                source="builtin",
                name=name,
                description=_text(raw_tool.get("description")),
                input_schema=_normalized_schema(raw_tool.get("inputSchema")),
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


def server_tools_from_inventory(
    payload: dict, *, target_id: str, target_label: str
) -> list[HubTool]:
    """Derive `HubTool`s from a remote server's raw tool-inventory payload.

    Reads defensively since the payload comes straight off the wire: skips
    nameless entries and non-dict tool records entirely.

    Args:
        payload: `{"tools": [raw dicts]}` from a server target.
        target_id: Stable identifier for the owning server target.
        target_label: Human-readable label for the owning server target.

    Returns:
        One `HubTool` per valid tool entry. Never executable — server-
        source tools are display-only — and never stale (a payload
        that was fetched at all implies a live connection at fetch time).
    """
    raw_tools = payload.get("tools") if isinstance(payload, Mapping) else None
    if not isinstance(raw_tools, list):
        return []
    server_key = f"server:{target_id}"
    tools: list[HubTool] = []
    seen_names: set[str] = set()
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        name = _text(raw_tool.get("name"))
        if not name or name in seen_names:
            # See local_tools_from_record()'s dedup comment -- same
            # DuplicateKey hazard, this server's own tool_id namespace.
            continue
        seen_names.add(name)
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
