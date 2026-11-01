"""What the Console context meter counts for the next send's tools.

TASK-34100.5.3 AC#5. The send preflight, its capacity check and the context
meter ask the agent bridge for ``preview_tool_surface(resolution)``. The
answer has to agree with what the send will disclose, so it is never a list
remembered from an earlier send: it is the send's own first-request schema
planner (``build_first_request_schema_plan``) re-run against the resolution
and the request the caller is assessing (``preflight_tool_surface``). A tool set
that fits a 1M-token window directly can need progressive discovery, or no
tools at all, on an 8,192-token one (review round 1, F2: the remembered 1M
rows made the preflight refuse sends on a small model as unable to fit); and
one that fits beside a short conversation can be dropped beside a long one
(review round 2, V2-1/R2-1: sized against a one-row stand-in, the meter held
a chat for compaction over 3,351 tokens of tools the send never carried).

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

#: The one-row request the planner is re-run with when the caller has no
#: request to size against. It needs an active user request to count a
#: request at all: with a local reasoning-replay policy (every llama.cpp
#: resolution carries one) the count projects history around it, and with no
#: messages it raised "A Console provider request requires an active request"
#: -- which the planner swallows into a tool-less plan, so the meter counted
#: no tools for requests that carried the light set (live, review round 1).
_PREVIEW_REQUEST: tuple[dict[str, str], ...] = ({"role": "user", "content": "."},)


def split_session_prompt(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split a provider request the way a send does before it plans.

    ``ConsoleChatController._run_agent_reply`` takes a leading system row as
    the session's prompt and hands the rest to the planner as the
    conversation; the meter's re-plan does the same so both size one request.

    Returns:
        ``(session_system_prompt, conversation_rows)``.
    """
    rows = [dict(row) for row in messages]
    if rows and rows[0].get("role") == "system":
        return str(rows[0].get("content", "")), rows[1:]
    return "", rows


@dataclasses.dataclass(frozen=True, slots=True)
class ToolSurfacePreview:
    """Selected provider representation; remembered composition is an estimate."""

    system_prompt: str | None = None
    tools: tuple[Mapping[str, Any], ...] = ()

    def request_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict]:
        """Replace only the session prompt for request accounting."""
        rows = [dict(row) for row in messages]
        if self.system_prompt is None:
            return rows
        system = rows.pop(0) if rows and rows[0].get("role") == "system" else {}
        return [{**system, "role": "system", "content": self.system_prompt}, *rows]

    def restore_session_prompt(
        self, rows: list[dict], original: Sequence[Mapping[str, Any]]
    ) -> list[dict]:
        """Return session inputs to dispatch, which composes its own run prompt."""
        if self.system_prompt is None:
            return rows
        session_system = (
            [dict(original[0])]
            if original and original[0].get("role") == "system"
            else []
        )
        return [*session_system, *rows[1:]]


def preflight_tool_surface(
    bridge: Any | None,
    resolution: Any,
    *,
    session_id: str | None,
    messages: Sequence[Mapping[str, Any]],
    turn_skill_bindings: tuple[str, ...] = (),
) -> ToolSurfacePreview:
    """Project this request with the bridge's planner, without owning sources."""
    preview = getattr(bridge, "preview_tool_surface", None)
    try:
        if callable(preview):
            return preview(
                resolution,
                session_id=session_id,
                messages=messages,
                turn_skill_bindings=turn_skill_bindings,
            )
        legacy = getattr(bridge, "preview_tool_schemas", None)
        if callable(legacy):
            return ToolSurfacePreview(
                tools=tuple(
                    legacy(resolution, session_id=session_id, messages=messages)
                )
            )
    except Exception:  # noqa: BLE001 -- an unavailable estimate never breaks a send
        pass
    return ToolSurfacePreview()


def preflight_tool_rows(
    bridge: Any | None,
    resolution: Any,
    *,
    session_id: str | None,
    messages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Compatibility projection of native rows; fenced schemas live in the prompt."""
    return list(
        preflight_tool_surface(
            bridge, resolution, session_id=session_id, messages=messages
        ).tools
    )


class SchemaCatalog:
    """A frozen copy of the tool schemas one send could disclose.

    It answers ``list_catalog``/``load_schema`` like a registry, which is all
    the schema planner reads.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[ToolCatalogEntry, ToolSchema]],
        *,
        review_owner_present: bool = False,
    ) -> None:
        self._review_owner_present = review_owner_present
        self._entries = tuple(entry for entry, _schema in pairs)
        self._schemas = {entry.id: schema for entry, schema in pairs}

    @classmethod
    def from_schemas(
        cls, schemas: Sequence[ToolSchema], *, source: str
    ) -> SchemaCatalog:
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
        owner_query = getattr(registry, "resolve_owner_for_name", None)
        review_owner_present = bool(
            callable(owner_query) and owner_query("code_review_prepare") is not None
        )
        return cls(pairs, review_owner_present=review_owner_present)

    def resolve_owner_for_name(self, name: str) -> str | None:
        """Project only captured review-owner presence for schema planning.

        This is metadata, never an executable provider or an ownership grant.
        Same-name schemas alone do not establish registration.
        """
        return (
            name
            if name == "code_review_prepare" and self._review_owner_present
            else None
        )

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

    def preview(
        self,
        resolution: Any,
        *,
        native_tools: bool,
        messages: Sequence[Mapping[str, Any]] = (),
        session_system_prompt: str | None = None,
        turn_skill_bindings: tuple[str, ...] = (),
    ) -> ToolSurfacePreview:
        """The selected provider representation for ``resolution``.

        The planner sizes the tools against ``messages`` -- the request the
        caller is assessing -- exactly as the send does: a conversation that
        leaves no room for the direct set gets discovery or no tools, and the
        meter must count that, not the set that fits an empty conversation
        (review round 2, V2-1/R2-1). With no messages a one-row stand-in is
        used.

        Args:
            resolution: The provider resolution to size against, or ``None``
                to keep the remembered send's own model and window.
            native_tools: Whether the bridge prefers provider-native tools.
            messages: The conversation rows the send would carry (no leading
                system row; ``split_session_prompt`` removes it).

        Returns:
            Selected system prompt and native rows; fenced schemas stay in the prompt.
        """
        from tldw_chatbook.Agents.agent_service import (
            build_first_request_schema_plan,
            compose_model_request_surface,
        )
        from tldw_chatbook.Agents.native_tools import provider_supports_native_tools
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
        flags = dict(self.schema_flags)
        if session_system_prompt is not None and flags.get("allow_discovery", True):
            from tldw_chatbook.Chat.console_agent_bridge import (
                compose_agent_system_prompt,
                _append_canvas_discovery_hint,
            )
            from tldw_chatbook.Agents.first_request_window import (
                PLAIN_CHAT_SYSTEM_PROMPT,
            )

            flags.update(
                direct_system_prompt=compose_agent_system_prompt(session_system_prompt),
                discovery_system_prompt=_append_canvas_discovery_hint(
                    compose_agent_system_prompt(
                        session_system_prompt, offer_find_load=True
                    ),
                    self.allowed_tools,
                ),
                plain_system_prompt=session_system_prompt.strip()
                or PLAIN_CHAT_SYSTEM_PROMPT,
                skill_file_enabled=bool(
                    turn_skill_bindings
                    and any(
                        entry.source == "skill" for entry in self.catalog.list_catalog()
                    )
                ),
            )
            config = dataclasses.replace(
                config, system_prompt=flags["direct_system_prompt"]
            )
        plan = build_first_request_schema_plan(
            self.catalog,
            self.allowed_tools,
            config,
            endpoint,
            [dict(row) for row in (messages or _PREVIEW_REQUEST)],
            context_window=context_window,
            **flags,
        )
        runtime = list(plan.runtime_schemas)
        schemas = [*runtime, *plan.active_schemas]
        native = config.native_tools and provider_supports_native_tools(
            config.execution_provider or endpoint,
            reasoning_replay=config.reasoning_replay,
        )
        prompt, tools = compose_model_request_surface(
            dataclasses.replace(config, system_prompt=plan.system_prompt),
            schemas,
            [dict(row) for row in messages],
            native=native,
            log_active=plan.log_active,
        )
        return ToolSurfacePreview(prompt, tuple(tools))

    def tool_rows(
        self,
        resolution: Any,
        *,
        native_tools: bool,
        messages: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Compatibility access to selected native provider rows."""
        return list(
            self.preview(resolution, native_tools=native_tools, messages=messages).tools
        )


def light_surface_plan_inputs(
    user_source_schemas: Sequence[ToolSchema] = (),
    *,
    local_tools_available: bool = True,
    session_system_prompt: str = "",
    previous_config: AgentConfig | None = None,
) -> ToolSurfacePlanInputs:
    """Planner inputs for plain chat's light surface, read fresh from config.

    The schema flags and prompts are the light plan's: no runtime tools, no
    run-log tools, no discovery fallback, and the session's prompt with the
    "answer directly" line (``build_console_first_request_plan`` with
    ``light_surface=True``; both take the prompts from
    ``light_surface_prompts``).

    Args:
        user_source_schemas: Tools from the user's own sources that this
            session's last send carried.
        local_tools_available: ``False`` while the MCP hub's kill switch
            keeps the send from composing local tools.
        session_system_prompt: The session's own system prompt, as the
            request being assessed carries it.
    """
    from tldw_chatbook.Agents.plain_chat_tool_surface import (
        LIGHT_SURFACE_TOOL_NAMES,
        light_surface_budget,
        light_surface_prompts,
        light_surface_tool_schemas,
    )

    schemas = light_surface_tool_schemas(
        user_source_schemas, local_tools_available=local_tools_available
    )
    catalog = SchemaCatalog.from_schemas(schemas, source="light_surface")
    names = catalog.names
    plain_prompt, direct_prompt = light_surface_prompts(session_system_prompt)
    return ToolSurfacePlanInputs(
        catalog=catalog,
        allowed_tools=names,
        config=AgentConfig(
            model="",
            system_prompt=direct_prompt,
            allowed_tools=names,
            budget=light_surface_budget(RunBudget()),
            workspace_context_note=previous_config.workspace_context_note
            if previous_config
            else "",
            personal_context_block=previous_config.personal_context_block
            if previous_config
            else "",
        ),
        schema_flags={
            "skill_file_enabled": False,
            "install_skill_enabled": False,
            "run_skill_script_enabled": False,
            "run_log_active": False,
            "direct_system_prompt": direct_prompt,
            "discovery_system_prompt": plain_prompt,
            "plain_system_prompt": plain_prompt,
            "allow_discovery": False,
            "fallback_tools": LIGHT_SURFACE_TOOL_NAMES,
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
        config=AgentConfig(
            model="", system_prompt="", allowed_tools=names, budget=budget
        ),
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
        self._light_config: OrderedDict[str, AgentConfig] = OrderedDict()
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
        self._put(self._light_config, session_id, inputs.config)

    def remember_light_sources(
        self,
        session_id: str,
        schemas: Sequence[ToolSchema],
        config: AgentConfig | None = None,
    ) -> None:
        """Record the user-source tools a light-surface send carried."""
        self._put(self._light_sources, session_id, tuple(schemas))
        if config is not None:
            self._put(self._light_config, session_id, config)

    def light_config(self, session_id: str | None) -> AgentConfig | None:
        """Last composed workspace/profile context; never provider ownership."""
        with self._lock:
            return self._light_config.get(session_id)

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
