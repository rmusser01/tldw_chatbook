# tldw_chatbook/Agents/tool_catalog.py
"""ToolProvider capability interface + registry + builtin provider.

This is the plugin seam: MCP (task-201) and Skills (task-200) register as
providers here — the runtime never changes. May import tool_executor
(wrapping it is this module's job); no UI/DB imports.
"""
from __future__ import annotations

import asyncio
import json
from typing import Iterable, Mapping, Protocol

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


def intersect_skill_tools(
    skill_allowed_tools: list[str] | None, builtin_names: Iterable[str],
) -> tuple[str, ...]:
    """A skill's `allowed_tools` narrows the runtime builtin set; never grants.

    ``None`` means the skill did not narrow — all builtins pass through.
    Otherwise only names present in both survive, ordered by
    ``builtin_names`` (not the skill's own order) so callers get a stable,
    registry-consistent ordering regardless of how the skill listed them.
    """
    if skill_allowed_tools is None:
        return tuple(builtin_names)
    allowed = set(skill_allowed_tools)
    return tuple(name for name in builtin_names if name in allowed)


class SkillToolProvider:
    """Exposes trusted, model-invocable skills as catalog tools.

    Built from a per-run snapshot of skill summaries (plain mappings with
    "name", "description", "argument_hint") — never imports Skills_Interop
    itself, so this module stays importable without that subsystem and the
    catalog is always as fresh as the snapshot the caller passed in (the
    per-run freshness doctrine: callers re-read skills at run start, not
    once at import time).

    ``invoke()`` deliberately raises: skill tools never execute via plain
    provider.invoke(). They route through the run-scoped spawn executor
    (budget-counted, cancellable, DB-lineage-tracked sub-agent runs — see
    the skills design doc's Architecture section). This method exists only
    to satisfy the ToolProvider protocol; calling it directly is a bug.
    """

    SOURCE = "skill"

    def __init__(self, entries: list[Mapping]) -> None:
        self._entries = list(entries)

    def _tool_id(self, name: str) -> str:
        return f"{self.SOURCE}:{name}"

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(id=self._tool_id(e["name"]), name=e["name"],
                             one_line_description=e["description"],
                             source=self.SOURCE)
            for e in self._entries
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = tool_id.split(":", 1)[1]
        entry = next(e for e in self._entries if e["name"] == name)
        hint = entry.get("argument_hint") or entry["description"]
        return ToolSchema(
            id=tool_id, name=name, description=entry["description"],
            parameters={
                "type": "object",
                "properties": {"args": {"type": "string", "description": hint}},
                "required": [],
            },
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        raise RuntimeError(
            "SkillToolProvider.invoke must not be called; skills route "
            "through the run-scoped spawn executor")


class ToolCatalogRegistry:
    """Ordered provider registry: catalog, search, schema, invocation."""

    def __init__(self) -> None:
        self._providers: list[ToolProvider] = []
        # tool_id -> owning provider, and name -> tool_id, both built
        # together (lazily) by _ensure_catalog_cache() and scoped PER RUN
        # (see reset_catalog_cache()). `None` means "not built yet" and is
        # distinct from an empty-but-built cache. The two dicts are always
        # populated from the SAME `list_catalog()` sweep (see
        # _build_owner_cache()), so a name resolved from `_name_to_id_cache`
        # is always present in `_owner_cache` too — `resolve_name()` and
        # `_owner_and_id()` can never observe different generations of the
        # catalog within one lookup.
        self._owner_cache: dict[str, ToolProvider] | None = None
        self._name_to_id_cache: dict[str, str] | None = None

    def register_provider(self, provider: ToolProvider) -> None:
        self._providers.append(provider)
        # A newly registered provider's tools aren't reflected in any
        # cache already built — invalidate so the next lookup rebuilds it.
        self._owner_cache = None
        self._name_to_id_cache = None

    def reset_catalog_cache(self) -> None:
        """Drop the owner-map/name-map cache; call once at the start of a run.

        Cache scope is PER RUN: the catalog is listed fresh at run start
        (``AgentService.run_turn`` calls this before dispatching), so any
        skill CRUD (or other provider mutation) between runs is always
        picked up. No cross-run invalidation signal is needed beyond this
        single reset — see the skills spec's Catalog scale section.
        """
        self._owner_cache = None
        self._name_to_id_cache = None

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

    def _build_owner_cache(self) -> tuple[dict[str, ToolProvider], dict[str, str]]:
        owner: dict[str, ToolProvider] = {}
        name_to_id: dict[str, str] = {}
        for provider in self._providers:
            for entry in provider.list_catalog():
                owner.setdefault(entry.id, provider)
                # First-registrant-wins, same as the owner map above and in
                # the SAME iteration order — preserves the existing
                # shadowing rule (builtins registered before skills/MCP
                # always win a name collision) without adding a second,
                # independently-ordered pass over the providers.
                name_to_id.setdefault(entry.name, entry.id)
        return owner, name_to_id

    def _ensure_catalog_cache(self) -> None:
        # This is the fix MCP (task-201) also needs: a network-backed
        # provider must not re-list_catalog() per lookup. Both the owner
        # map (id -> provider, used by load_schema()/_owner_and_id()) and
        # the name map (name -> id, used by resolve_name()) are built
        # together from ONE list_catalog() sweep per provider (lazily, on
        # first lookup) and reused for every subsequent lookup — by either
        # map — until reset_catalog_cache() clears both. Previously only
        # the owner map shared this cache; resolve_name() re-listed every
        # provider on every call, so invoke_by_name() (resolve_name() then
        # _owner_and_id()) still paid a full per-provider sweep on every
        # invocation despite the owner-map cache existing.
        if self._owner_cache is None:
            self._owner_cache, self._name_to_id_cache = self._build_owner_cache()

    def _owner_and_id(self, tool_id: str):
        self._ensure_catalog_cache()
        return self._owner_cache.get(tool_id)

    def load_schema(self, tool_id: str) -> ToolSchema:
        provider = self._owner_and_id(tool_id)
        if provider is None:
            raise KeyError(f"Unknown tool id: {tool_id}")
        return provider.load_schema(tool_id)

    def resolve_name(self, name: str) -> str | None:
        self._ensure_catalog_cache()
        return self._name_to_id_cache.get(name)

    def invoke_by_name(self, name: str, args: dict) -> ToolResult:
        tool_id = self.resolve_name(name)
        if tool_id is None:
            return ToolResult(ok=False, error=f"Unknown tool: {name}")
        provider = self._owner_and_id(tool_id)
        if provider is None:
            # Defensive only: resolve_name()/_owner_and_id() now share one
            # cache built atomically from a single list_catalog() sweep, so
            # a name resolved above is always present in the owner map too
            # within the SAME cache generation — this branch is no longer
            # reachable via a same-lookup race. It stays as insurance
            # against a future change to the cache-building code, and never
            # lets a `None` owner surface as an AttributeError.
            return ToolResult(
                ok=False, error=f"Tool provider not found for: {name}")
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
