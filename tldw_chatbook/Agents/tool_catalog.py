# tldw_chatbook/Agents/tool_catalog.py
"""ToolProvider capability interface + registry + builtin provider.

This is the plugin seam: MCP (task-201) and Skills (task-200) register as
providers here — the runtime never changes. May import tool_executor
(wrapping it is this module's job); no UI/DB imports.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable, Mapping, Protocol

from tldw_chatbook.Tools.tool_executor import CalculatorTool, DateTimeTool

from .agent_models import (
    DIRECT_DISCLOSE_THRESHOLD,
    FIND_TOOLS_NAME,
    INSTALL_SKILL_TOOL_NAME,
    LOAD_TOOLS_NAME,
    RUN_SKILL_SCRIPT_TOOL_NAME,
    RunBudget,
    SKILL_FILE_TOOL_NAME,
    SPAWN_TOOL_NAME,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)

SPAWN_TOOL_SCHEMA = ToolSchema(
    id="runtime:spawn_subagent",
    name=SPAWN_TOOL_NAME,
    description=(
        "Delegate a self-contained task to an isolated sub-agent. It sees "
        "only the task text you pass, works on it, and returns a result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Complete, self-contained task description.",
            }
        },
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
        "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["ids"],
    },
)

SKILL_FILE_TOOL_SCHEMA = ToolSchema(
    id="runtime:skill_file",
    name=SKILL_FILE_TOOL_NAME,
    description=(
        "Read a bundled reference file of a skill active in this run. "
        "Args: skill_name (the skill whose bundle to read), path (relative "
        "POSIX path, e.g. references/api.md). Text files only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "skill_name": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["skill_name", "path"],
    },
)

INSTALL_SKILL_TOOL_SCHEMA = ToolSchema(
    id="runtime:install_skill",
    name=INSTALL_SKILL_TOOL_NAME,
    description=(
        "Install a skill from a GitHub repository/tree URL or a direct "
        "https .zip URL. The user is asked to confirm before anything is "
        "downloaded. On success the skill is installed but left pending the "
        "user's review — it cannot run until the user approves it in "
        "Library > Skills. If the repository contains multiple skills, the "
        "tool returns the list of candidates; re-call with a URL that points "
        "at one skill's subdirectory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "A GitHub repo/tree URL or a direct https .zip URL for "
                    "the skill to install."
                ),
            }
        },
        "required": ["url"],
    },
)

RUN_SKILL_SCRIPT_TOOL_SCHEMA = ToolSchema(
    id="runtime:run_skill_script",
    name=RUN_SKILL_SCRIPT_TOOL_NAME,
    description=(
        "Run a script bundled with a trusted skill. The user is asked to "
        "confirm each run unless they have granted this skill standing "
        "permission. The script runs with a scrubbed environment in a "
        "temporary working directory (not the skill's own folder), under CPU "
        "and time limits; only its stdout/stderr and exit code come back, and "
        "any files it writes are discarded. Args: skill_name (the skill that "
        "owns the script), script_path (relative POSIX path, e.g. "
        "scripts/extract.py), args (optional list of string arguments)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "The skill whose bundled script to run.",
            },
            "script_path": {
                "type": "string",
                "description": (
                    "Relative POSIX path of the script inside the skill's "
                    "bundle, e.g. scripts/extract.py."
                ),
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional string arguments passed to the script.",
            },
        },
        "required": ["skill_name", "script_path"],
    },
)


class ToolProvider(Protocol):
    """The capability interface providers implement."""

    def list_catalog(self) -> list[ToolCatalogEntry]: ...

    def load_schema(self, tool_id: str) -> ToolSchema: ...

    def invoke(self, tool_id: str, args: dict) -> ToolResult: ...


def build_builtin_gate(*args: Any, **kwargs: Any) -> Any:
    """Thin, monkeypatchable indirection to the real gate builder.

    Defined here (rather than imported at module scope) so this module
    stays dependency-light: `builtin_tool_gate` (which pulls in the MCP
    permission store) is only imported the first time this is actually
    *called*, not merely when `tool_catalog` itself is imported. Keeping
    it as a real module-level name -- instead of a `from ... import` done
    inline inside `_resolve_gate` -- is what lets tests monkeypatch
    `tool_catalog.build_builtin_gate` directly to prove a bare
    `BuiltinToolProvider()` is gated by default (Constraint 6): a
    function-local import only ever binds a local name, never a module
    attribute, so `monkeypatch.setattr(module, "build_builtin_gate", ...)`
    would have nothing to patch without this indirection.
    """
    from tldw_chatbook.Agents.builtin_tool_gate import (
        build_builtin_gate as _build_builtin_gate,
    )

    return _build_builtin_gate(*args, **kwargs)


class BuiltinToolProvider:
    """Wraps tool_executor's built-in tools behind the provider interface."""

    SOURCE = "builtin"

    def __init__(self, gate: Any | None = None) -> None:
        self._tools = {t.name: t for t in (CalculatorTool(), DateTimeTool())}
        # task-584: surface the app's existing sandbox-rooted file tools to the
        # agent loop. They were registered on the global ToolExecutor but never
        # reachable from here, so retained script output -- deliberately written
        # under the file-tool sandbox root -- had no consumer. Behind the SAME
        # [tools] gates that already govern them, which default to DISABLED:
        # this changes reachability, not the default posture.
        for gate_key, factory_name in (
            ("read_file_enabled", "ReadFileTool"),
            ("list_directory_enabled", "ListDirectoryTool"),
        ):
            try:
                from ..config import get_cli_setting

                if not get_cli_setting("tools", gate_key, False):
                    continue
                from ..Tools import file_operation_tools as _file_tools

                tool = getattr(_file_tools, factory_name)()
            except Exception:  # noqa: BLE001 — an unavailable tool is just absent
                continue
            self._tools[tool.name] = tool
        # `None` means "build the real gate on first use" -- NOT "ungated".
        # Every construction site (console_agent_bridge's default registry
        # and its per-run registry) passes nothing today, so an ungated
        # default would silently leave the shipping path unprotected.
        self._gate = gate

    def _tool_id(self, name: str) -> str:
        return f"{self.SOURCE}:{name}"

    def tool_for(self, name: str) -> Any | None:
        """Return the built-in ``Tool`` registered under ``name``, if any."""
        return self._tools.get(name)

    def _resolve_gate(self) -> Any:
        """Return the provider's gate, building one lazily on first use.

        Note: nothing here calls `begin_turn()` on a lazily-built gate,
        so its permission-store payload is loaded once and never
        invalidated for the life of this provider. Harmless today --
        `build_builtin_gate()`'s default, service-less gate always has
        an empty `{}` payload -- but any gate handed to (or built by)
        this provider must be driven by a caller that calls
        `begin_turn()` once per turn, as `console_chat_controller`'s
        review-hook path already does, or its permission state will
        freeze at first use.
        """
        if self._gate is None:
            # Module-global lookup (not a local import) so a test's
            # monkeypatch of `tool_catalog.build_builtin_gate` is honored.
            self._gate = build_builtin_gate()
        return self._gate

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                id=self._tool_id(t.name),
                name=t.name,
                one_line_description=t.description,
                source=self.SOURCE,
            )
            for t in self._tools.values()
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = tool_id.split(":", 1)[1]
        tool = self._tools[name]
        return ToolSchema(
            id=tool_id,
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[1]
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"Unknown builtin tool: {name}")
        # Defense in depth: the run-level review hook is the primary gate
        # (it batches approvals into one card per turn), but a caller that
        # reaches invoke() without going through it must still not execute
        # ungated. A gate that raises fails CLOSED -- never into the pure
        # loop, which must not see exceptions from tool invocation.
        try:
            refusal = self._resolve_gate().check(tool)
        except Exception as exc:  # noqa: BLE001 — fail closed
            return ToolResult(ok=False, error=f"permission check failed: {exc}")
        if refusal is not None:
            return ToolResult(ok=False, error=refusal)
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
    skill_allowed_tools: list[str] | None,
    builtin_names: Iterable[str],
) -> tuple[str, ...]:
    """A skill's `allowed_tools` narrows the runtime builtin set; never grants.

    ``None`` means the skill did not narrow — all builtins pass through.
    Otherwise only names present in both survive, ordered by
    ``builtin_names`` (not the skill's own order) so callers get a stable,
    registry-consistent ordering regardless of how the skill listed them.

    Args:
        skill_allowed_tools: The skill's own declared ``allowed_tools``
            list (front-matter), or ``None`` when the skill did not narrow
            its child's tool set at all.
        builtin_names: The run's builtin tool names, in registry order —
            the widest set a narrowed list can ever be intersected down
            to; a name absent here can never be granted regardless of what
            the skill declares.

    Returns:
        ``tuple(builtin_names)`` unchanged when ``skill_allowed_tools`` is
        ``None``; otherwise the subset of ``builtin_names`` also present in
        ``skill_allowed_tools``, preserving ``builtin_names``' order.
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
        """Return one cheap-to-list catalog row per skill entry.

        Returns:
            A `ToolCatalogEntry` for each skill this provider was built
            with, in the order the entries were passed to `__init__`; each
            entry's `id` is `"skill:<name>"` and its `source` is
            `SOURCE` (`"skill"`).
        """
        return [
            ToolCatalogEntry(
                id=self._tool_id(e["name"]),
                name=e["name"],
                one_line_description=e["description"],
                source=self.SOURCE,
            )
            for e in self._entries
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        """Return the full tool schema for one previously-listed skill entry.

        Args:
            tool_id: A catalog id previously returned by `list_catalog`
                (``"skill:<name>"``).

        Returns:
            A `ToolSchema` whose single parameter is a free-form ``args``
            string (described by the skill's own ``argument_hint``, or its
            ``description`` when no hint was given) — skills never expose a
            structured parameter schema the way builtin tools do.

        Raises:
            StopIteration: ``tool_id``'s name does not match any entry this
                provider was built with (mirrors `next()`'s own behavior
                with no default; never expected in practice since
                `tool_id` always comes from this provider's own
                `list_catalog`).
        """
        name = tool_id.split(":", 1)[1]
        entry = next(e for e in self._entries if e["name"] == name)
        hint = entry.get("argument_hint") or entry["description"]
        return ToolSchema(
            id=tool_id,
            name=name,
            description=entry["description"],
            parameters={
                "type": "object",
                "properties": {"args": {"type": "string", "description": hint}},
                "required": [],
            },
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        """Never called: satisfies the `ToolProvider` protocol only.

        Args:
            tool_id: Unused — present only to match the protocol shape.
            args: Unused — present only to match the protocol shape.

        Raises:
            RuntimeError: Always. A skill-tool call must route through the
                run-scoped spawn executor (see this class's own docstring
                and `console_agent_bridge._BridgeSkillRunner`), never
                through a plain `ToolProvider.invoke`. Reaching this method
                at all is a caller bug.
        """
        raise RuntimeError(
            "SkillToolProvider.invoke must not be called; skills route "
            "through the run-scoped spawn executor"
        )


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
        # is always present in `_owner_cache` too PROVIDED the two reads
        # aren't interleaved with a concurrent rebuild — true when every
        # `invoke_by_name()` call ran serialized on one thread, which is no
        # longer guaranteed: task-327's per-call timeout runs each call on
        # its own daemon thread and abandons (never joins) one that hangs,
        # so an abandoned call's `resolve_name()`/`_owner_and_id()` pair can
        # now overlap a later call's own pair, or a `register_provider()`
        # invalidation, with no lock guarding `_owner_cache`/
        # `_name_to_id_cache` — two concurrent lookups CAN observe different
        # generations of the catalog.
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
        return [
            e
            for e in self.list_catalog()
            if needle in e.name.lower() or needle in e.one_line_description.lower()
        ]

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
        # Guard BOTH caches, not just _owner_cache: the two stores are
        # assigned together as a tuple below, but task-327's per-call daemon
        # threads mean an abandoned thread can still be mid-flight when
        # reset_catalog_cache() runs on a later call, interleaving with this
        # method elsewhere and leaving one store populated while the other
        # was reset to None. A single-cache guard would then skip the
        # rebuild and leave _name_to_id_cache (or _owner_cache) permanently
        # None for the rest of the run.
        if self._owner_cache is None or self._name_to_id_cache is None:
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
        cache = self._name_to_id_cache
        return cache.get(name) if cache else None

    def invoke_by_name(self, name: str, args: dict) -> ToolResult:
        tool_id = self.resolve_name(name)
        if tool_id is None:
            return ToolResult(ok=False, error=f"Unknown tool: {name}")
        provider = self._owner_and_id(tool_id)
        if provider is None:
            # resolve_name()/_owner_and_id() share one cache built from a
            # single list_catalog() sweep, so within one SERIALIZED lookup
            # a name resolved above is always present in the owner map too.
            # That no longer makes this branch unreachable, though: this
            # registry has no lock, and task-327's per-call timeout can run
            # invoke_by_name() calls concurrently (one call's cache read
            # racing another call's, or a register_provider() rebuild), so
            # `tool_id` above can genuinely belong to a since-superseded
            # generation by the time this line runs. This fallback never
            # lets a `None` owner surface as an AttributeError either way.
            return ToolResult(ok=False, error=f"Tool provider not found for: {name}")
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
