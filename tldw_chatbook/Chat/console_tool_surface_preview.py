"""What the Console context meter counts for the next send's tools.

TASK-34100.5.3 AC#5. The send preflight, its capacity check and the context
meter ask the agent bridge for ``preview_tool_schemas(resolution)``. The
answer has to agree with what the send will disclose, so it is never a list
remembered from an earlier send: it is the send's own first-request schema
planner (``build_first_request_schema_plan``) re-run against the resolution
the caller is asking about. A tool set that fits a 1M-token window directly
can need progressive discovery, or no tools at all, on an 8,192-token one
(review round 1, F2: the remembered 1M rows made the preflight refuse sends
on a small model as unable to fit).

What the planner is re-run over (``ToolSurfacePlanInputs``):

* **Light surface** -- the config-derived light set
  (``plain_chat_tool_surface.light_surface_tool_schemas``, narrowed by the
  same ``light_surface_allowed_tools`` the send applies), plus the tools from
  the user's own sources (MCP servers they added, Library tools, an armed
  shell) that this session's last send carried.
* **Full surface** -- the catalog and planner inputs of this session's last
  full-surface send, else of the most recent one in any session, else the
  bridge's shared registry. The first full-surface send after the switch is
  composed after the preflight runs (``ConsoleChatController._run_agent_
  reply``), so until one has gone out the meter cannot see the composed
  local, MCP and skill tools; the planner still sizes that send itself.

Every remembered catalog is a frozen copy of schemas (``SchemaCatalog``),
never a provider, so remembering one keeps no MCP connection, Library
handle or shell alive.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    RunBudget,
    ToolCatalogEntry,
    ToolSchema,
)

#: Sessions whose last send the meter remembers (oldest dropped first).
MAX_REMEMBERED_SESSIONS = 32

#: The one-row request the planner is re-run with. It needs an active user
#: request to count a request at all: with a local reasoning-replay policy
#: (every llama.cpp resolution carries one) the count projects history around
#: it, and with no messages it raised "A Console provider request requires an
#: active request" -- which the planner swallows into a tool-less plan, so the
#: meter counted no tools for requests that carried the light set (live,
#: review round 1).
_PREVIEW_REQUEST: tuple[dict[str, str], ...] = ({"role": "user", "content": "."},)


class SchemaCatalog:
    """A frozen copy of the tool schemas one send could disclose.

    It answers ``list_catalog``/``load_schema`` like a registry, which is all
    the schema planner reads.
    """

    def __init__(self, pairs: Sequence[tuple[ToolCatalogEntry, ToolSchema]]) -> None:
        self._entries = tuple(entry for entry, _schema in pairs)
        self._schemas = {entry.id: schema for entry, schema in pairs}

    @classmethod
    def from_schemas(cls, schemas: Sequence[ToolSchema], *, source: str) -> SchemaCatalog:
        """Wrap bare schemas, cataloguing each under its own id."""
        return cls(
            [
                (
                    ToolCatalogEntry(
                        id=schema.id,
                        name=schema.name,
                        one_line_description=(schema.description or "").split("\n")[0],
                        source=source,
                    ),
                    schema,
                )
                for schema in schemas
            ]
        )

    @classmethod
    def from_registry(cls, registry: Any, names: Collection[str]) -> SchemaCatalog:
        """Copy the schemas of ``names`` out of a composed registry.

        An entry whose schema cannot be loaded is left out (a preview counts
        what it can).
        """
        wanted = frozenset(names)
        pairs: list[tuple[ToolCatalogEntry, ToolSchema]] = []
        try:
            entries = registry.list_catalog()
        except Exception:  # noqa: BLE001 -- an unreadable registry copies nothing
            return cls(())
        for entry in entries:
            if entry.name not in wanted:
                continue
            try:
                pairs.append((entry, registry.load_schema(entry.id)))
            except Exception:  # noqa: BLE001 -- skip what cannot be loaded
                continue
        return cls(pairs)

    @property
    def schemas(self) -> tuple[ToolSchema, ...]:
        """Every schema, in catalog order."""
        return tuple(self._schemas[entry.id] for entry in self._entries)

    @property
    def names(self) -> tuple[str, ...]:
        """Every tool name, in catalog order."""
        return tuple(entry.name for entry in self._entries)

    def list_catalog(self) -> list[ToolCatalogEntry]:
        """The copied catalog entries, in their original order."""
        return list(self._entries)

    def load_schema(self, tool_id: str) -> ToolSchema:
        """The copied schema for ``tool_id``."""
        return self._schemas[tool_id]


@dataclasses.dataclass(frozen=True, slots=True)
class ToolSurfacePlanInputs:
    """Everything the first-request schema planner needs but the resolution.

    Attributes:
        catalog: The tools the send could disclose.
        allowed_tools: The send's allow-list over ``catalog``.
        config: The send's agent config (model, budget, prompt, reserve).
        schema_flags: The keyword flags the send passed the planner, minus
            ``context_window``.
    """

    catalog: SchemaCatalog
    allowed_tools: tuple[str, ...]
    config: AgentConfig
    schema_flags: Mapping[str, Any]

    def tool_rows(self, resolution: Any, *, native_tools: bool) -> list[dict[str, Any]]:
        """The provider tool rows the planner discloses for ``resolution``.

        The planner is given a one-row stand-in request, not the
        conversation: the meter asks what the tool surface is for this model
        and window; whether the conversation then fits beside it is the
        preflight's own question.

        Args:
            resolution: The provider resolution to size against, or ``None``
                to keep the remembered send's own model and window.
            native_tools: Whether the bridge prefers provider-native tools.

        Returns:
            One OpenAI-shaped ``{"type": "function", ...}`` row per tool.
        """
        from tldw_chatbook.Agents.agent_service import (
            build_first_request_schema_plan,
        )
        from tldw_chatbook.Agents.native_tools import schemas_to_openai_tools
        from tldw_chatbook.Chat.console_history_budget import (
            DEFAULT_RESPONSE_RESERVATION,
        )

        config = dataclasses.replace(self.config, native_tools=native_tools)
        endpoint = config.provider
        context_window = None
        if resolution is not None:
            endpoint = str(
                getattr(resolution, "execution_key", "")
                or getattr(resolution, "provider", "")
                or endpoint
            )
            config = dataclasses.replace(
                config,
                model=str(getattr(resolution, "model", "") or config.model),
                provider=endpoint,
                reasoning_replay=getattr(resolution, "reasoning_replay", None),
                response_reserve_tokens=(
                    getattr(resolution, "max_tokens", None)
                    or DEFAULT_RESPONSE_RESERVATION
                ),
            )
            context_window = getattr(resolution, "context_window", None)
        plan = build_first_request_schema_plan(
            self.catalog,
            self.allowed_tools,
            config,
            endpoint,
            [dict(row) for row in _PREVIEW_REQUEST],
            context_window=context_window,
            **self.schema_flags,
        )
        return schemas_to_openai_tools([*plan.runtime_schemas, *plan.active_schemas])


def light_surface_plan_inputs(
    user_source_schemas: Sequence[ToolSchema] = (),
    *,
    local_tools_available: bool = True,
) -> ToolSurfacePlanInputs:
    """Planner inputs for plain chat's light surface, read fresh from config.

    The schema flags are the light plan's: no runtime tools, no run-log
    tools and no discovery fallback (``build_console_first_request_plan``
    with ``light_surface=True``).

    Args:
        user_source_schemas: Tools from the user's own sources that this
            session's last send carried.
        local_tools_available: ``False`` while the MCP hub's kill switch
            keeps the send from composing local tools.
    """
    from tldw_chatbook.Agents.first_request_window import PLAIN_CHAT_SYSTEM_PROMPT
    from tldw_chatbook.Agents.plain_chat_tool_surface import (
        light_surface_budget,
        light_surface_tool_schemas,
    )

    schemas = light_surface_tool_schemas(
        user_source_schemas, local_tools_available=local_tools_available
    )
    catalog = SchemaCatalog.from_schemas(schemas, source="light_surface")
    names = catalog.names
    return ToolSurfacePlanInputs(
        catalog=catalog,
        allowed_tools=names,
        config=AgentConfig(
            model="",
            system_prompt=PLAIN_CHAT_SYSTEM_PROMPT,
            allowed_tools=names,
            budget=light_surface_budget(RunBudget()),
        ),
        schema_flags={
            "skill_file_enabled": False,
            "install_skill_enabled": False,
            "run_skill_script_enabled": False,
            "run_log_active": False,
            "plain_system_prompt": PLAIN_CHAT_SYSTEM_PROMPT,
            "allow_discovery": False,
        },
    )


def shared_registry_plan_inputs(
    registry: Any, allowed_tools: Sequence[str], budget: RunBudget
) -> ToolSurfacePlanInputs:
    """Planner inputs over the bridge's shared registry (no send yet)."""
    names = tuple(allowed_tools) or tuple(
        entry.name for entry in registry.list_catalog()
    )
    catalog = SchemaCatalog.from_registry(registry, names)
    return ToolSurfacePlanInputs(
        catalog=catalog,
        allowed_tools=names,
        config=AgentConfig(model="", system_prompt="", allowed_tools=names, budget=budget),
        schema_flags={
            "skill_file_enabled": False,
            "install_skill_enabled": False,
            "run_skill_script_enabled": False,
            "run_log_active": False,
        },
    )


class ToolSurfaceMemory:
    """Per-session memory of the last send's tool surface, for the meter.

    Thread-safe: sends record from their worker threads, the meter reads on
    the UI loop.
    """

    def __init__(self, max_sessions: int = MAX_REMEMBERED_SESSIONS) -> None:
        self._lock = threading.Lock()
        self._max = max_sessions
        self._full: OrderedDict[str, ToolSurfacePlanInputs] = OrderedDict()
        self._light_sources: OrderedDict[str, tuple[ToolSchema, ...]] = OrderedDict()

    def _put(self, table: OrderedDict, session_id: str, value: Any) -> None:
        with self._lock:
            table.pop(session_id, None)
            table[session_id] = value
            while len(table) > self._max:
                table.popitem(last=False)

    def remember_full(self, session_id: str, inputs: ToolSurfacePlanInputs) -> None:
        """Record a full-surface send's planner inputs."""
        self._put(self._full, session_id, inputs)

    def remember_light_sources(
        self, session_id: str, schemas: Sequence[ToolSchema]
    ) -> None:
        """Record the user-source tools a light-surface send carried."""
        self._put(self._light_sources, session_id, tuple(schemas))

    def full_inputs(self, session_id: str | None) -> ToolSurfacePlanInputs | None:
        """This session's last full-surface inputs, else the most recent."""
        with self._lock:
            if session_id is not None and session_id in self._full:
                return self._full[session_id]
            return next(reversed(self._full.values()), None)

    def light_sources(self, session_id: str | None) -> tuple[ToolSchema, ...]:
        """The user-source tools this session's last light send carried."""
        if session_id is None:
            return ()
        with self._lock:
            return self._light_sources.get(session_id, ())
