# tldw_chatbook/Agents/tool_catalog.py
"""ToolProvider capability interface + registry + builtin provider.

This is the plugin seam: MCP (task-201) and Skills (task-200) register as
providers here — the runtime never changes. May import tool_executor
(wrapping it is this module's job); no UI/DB imports.
"""
from __future__ import annotations

import asyncio
import json
from typing import Protocol

from tldw_chatbook.Tools.tool_executor import CalculatorTool, DateTimeTool

from .agent_models import (
    DIRECT_DISCLOSE_THRESHOLD, FIND_TOOLS_NAME, LOAD_TOOLS_NAME,
    RunBudget, SPAWN_TOOL_NAME, ToolCatalogEntry, ToolResult, ToolSchema,
)

SPAWN_TOOL_SCHEMA = ToolSchema(
    id="runtime:spawn_subagent",
    name=SPAWN_TOOL_NAME,
    description=(
        "Delegate a self-contained task to an isolated sub-agent. It sees "
        "only the task text you pass, works on it, and returns a result."),
    parameters={
        "type": "object",
        "properties": {"task": {
            "type": "string",
            "description": "Complete, self-contained task description."}},
        "required": ["task"],
    },
)

FIND_TOOLS_SCHEMA = ToolSchema(
    id="runtime:find_tools",
    name=FIND_TOOLS_NAME,
    description="Search the tool catalog by keyword; returns ids + one-liners.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

LOAD_TOOLS_SCHEMA = ToolSchema(
    id="runtime:load_tools",
    name=LOAD_TOOLS_NAME,
    description="Load full schemas for catalog ids so you can call them.",
    parameters={
        "type": "object",
        "properties": {"ids": {"type": "array",
                               "items": {"type": "string"}}},
        "required": ["ids"],
    },
)


class ToolProvider(Protocol):
    """The capability interface providers implement."""

    def list_catalog(self) -> list[ToolCatalogEntry]: ...

    def load_schema(self, tool_id: str) -> ToolSchema: ...

    def invoke(self, tool_id: str, args: dict) -> ToolResult: ...


class BuiltinToolProvider:
    """Wraps tool_executor's built-in tools behind the provider interface."""

    SOURCE = "builtin"

    def __init__(self) -> None:
        self._tools = {t.name: t for t in (CalculatorTool(), DateTimeTool())}

    def _tool_id(self, name: str) -> str:
        return f"{self.SOURCE}:{name}"

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(id=self._tool_id(t.name), name=t.name,
                             one_line_description=t.description,
                             source=self.SOURCE)
            for t in self._tools.values()
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = tool_id.split(":", 1)[1]
        tool = self._tools[name]
        return ToolSchema(id=tool_id, name=tool.name,
                          description=tool.description,
                          parameters=tool.parameters)

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[1]
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"Unknown builtin tool: {name}")
        try:
            # Providers bridge async tools; the loop's interface is sync.
            # Safe here: the service runs in a worker thread with no
            # running event loop.
            raw = asyncio.run(tool.execute(**args))
        except Exception as exc:  # noqa: BLE001 — captured, never escapes
            return ToolResult(ok=False, error=str(exc))
        if isinstance(raw, dict) and raw.get("error"):
            return ToolResult(ok=False, error=str(raw["error"]))
        content = json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw)
        return ToolResult(ok=True, content=content)


class ToolCatalogRegistry:
    """Ordered provider registry: catalog, search, schema, invocation."""

    def __init__(self) -> None:
        self._providers: list[ToolProvider] = []

    def register_provider(self, provider: ToolProvider) -> None:
        self._providers.append(provider)

    def list_catalog(self) -> list[ToolCatalogEntry]:
        entries: list[ToolCatalogEntry] = []
        for provider in self._providers:
            entries.extend(provider.list_catalog())
        return entries

    def find(self, query: str) -> list[ToolCatalogEntry]:
        needle = query.strip().lower()
        if not needle:
            return []
        return [e for e in self.list_catalog()
                if needle in e.name.lower()
                or needle in e.one_line_description.lower()]

    def _owner_and_id(self, tool_id: str):
        for provider in self._providers:
            if any(e.id == tool_id for e in provider.list_catalog()):
                return provider
        return None

    def load_schema(self, tool_id: str) -> ToolSchema:
        provider = self._owner_and_id(tool_id)
        if provider is None:
            raise KeyError(f"Unknown tool id: {tool_id}")
        return provider.load_schema(tool_id)

    def resolve_name(self, name: str) -> str | None:
        for entry in self.list_catalog():
            if entry.name == name:
                return entry.id
        return None

    def invoke_by_name(self, name: str, args: dict) -> ToolResult:
        tool_id = self.resolve_name(name)
        if tool_id is None:
            return ToolResult(ok=False, error=f"Unknown tool: {name}")
        provider = self._owner_and_id(tool_id)
        return provider.invoke(tool_id, args)


def initial_disclosure(
    registry: ToolCatalogRegistry, budget: RunBudget
) -> tuple[list[ToolSchema], bool]:
    """Small catalog → direct-disclose everything, drop find/load.

    Returns (active schemas, offer_find_load).
    """
    catalog = registry.list_catalog()
    if len(catalog) <= DIRECT_DISCLOSE_THRESHOLD:
        schemas = [registry.load_schema(e.id) for e in catalog]
        return schemas[: budget.max_active_tools], False
    return [], True
