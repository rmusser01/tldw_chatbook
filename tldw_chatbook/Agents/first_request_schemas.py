"""Pure runtime-schema construction for the first-request disclosure planner."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from .agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    MESSAGE_TOOL_NAMES,
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
    AgentConfig,
    AgentDefinition,
    ToolSchema,
)
from .tool_catalog import (
    CHECK_AGENTS_SCHEMA,
    WAIT_AGENTS_SCHEMA,
    SEND_TO_AGENT_SCHEMA,
    FORK_CHAT_TOOL_SCHEMA,
    NEW_CHAT_TOOL_SCHEMA,
    INSTALL_SKILL_TOOL_SCHEMA,
    PREPARE_MANAGED_SKILL_PROMOTION_TOOL_SCHEMA,
    SKILL_FILE_TOOL_SCHEMA,
    RUN_SKILL_SCRIPT_TOOL_SCHEMA,
    SEARCH_RUN_LOG_TOOL_SCHEMA,
    RUN_LOG_STATS_TOOL_SCHEMA,
    RUN_LOG_SLICE_TOOL_SCHEMA,
    LOAD_TOOLS_SCHEMA,
    build_chat_create_schema,
    build_find_tools_schema,
    build_spawn_schema,
)


@dataclasses.dataclass(frozen=True, slots=True)
class FirstRequestSchemaPlan:
    """Pure disclosure/runtime-schema plan for a primary first request."""

    active_schemas: tuple[ToolSchema, ...]
    runtime_schemas: tuple[ToolSchema, ...]
    offer_find_load: bool
    log_active: bool
    system_prompt: str
    agent_definitions: tuple[AgentDefinition, ...] | None = None
    fleet_max_live: int | None = None
    request_fits: bool = True
    # Compact disclosure keeps this ceiling through all request additions.
    schema_input_budget_tokens: int | None = None


def build_runtime_schema_plan(
    active: tuple[ToolSchema, ...],
    offer_find_load: bool,
    system_prompt: str,
    *,
    config: AgentConfig,
    agent_definitions: tuple[AgentDefinition, ...] | None,
    fleet_max_live: int | None,
    agent_kind: str,
    fleet_active: bool,
    progress_available: bool,
    reporting_available: bool,
    peers_available: bool,
    worktree_merge_enabled: bool,
    spawn_override_enabled: bool,
    spawn_override_targets: tuple[tuple[str, tuple[str, ...]], ...],
    skill_file_enabled: bool,
    install_skill_enabled: bool,
    managed_skill_promotion_enabled: bool,
    run_skill_script_enabled: bool,
    fork_chat_enabled: bool,
    new_chat_enabled: bool,
    run_log_active: bool,
    deferred_names: Callable[[], tuple[str, ...]],
) -> FirstRequestSchemaPlan:
    """Compose only runtime schemas enabled for this run and agent kind."""
    runtime: list[ToolSchema] = []
    if config.budget.max_subagents > 0:
        # ADR-147 Task 7: provider/model override args appear ONLY when
        # the operator opted in; the enumerated targets let the master
        # pick a VALID one. With no definitions and the gate closed,
        # build_spawn_schema returns SPAWN_TOOL_SCHEMA itself -- the
        # pre-ADR-147 payload stays byte-identical.
        runtime.append(
            build_spawn_schema(
                agent_definitions or (),
                override_enabled=spawn_override_enabled,
                override_targets=spawn_override_targets,
            )
        )
    if fleet_active and agent_kind == AGENT_KIND_PRIMARY:
        runtime.extend((WAIT_AGENTS_SCHEMA, CHECK_AGENTS_SCHEMA, SEND_TO_AGENT_SCHEMA))
        if worktree_merge_enabled:
            from .tool_catalog import (
                DISCARD_AGENT_WORKTREE_SCHEMA,
                MERGE_AGENT_WORKTREE_SCHEMA,
            )

            runtime.extend((MERGE_AGENT_WORKTREE_SCHEMA, DISCARD_AGENT_WORKTREE_SCHEMA))
    if progress_available and agent_kind == AGENT_KIND_PRIMARY:
        from .fleet_message_tools import (
            READ_AGENT_MESSAGES_SCHEMA,
            READ_INSTRUCTIONS,
        )

        runtime.append(READ_AGENT_MESSAGES_SCHEMA)
        if READ_INSTRUCTIONS not in system_prompt:
            system_prompt += "\n\n" + READ_INSTRUCTIONS
    if reporting_available and agent_kind == AGENT_KIND_SUBAGENT:
        from .fleet_message_tools import (
            REPORT_TO_SUPERVISOR_SCHEMA,
            REPORT_INSTRUCTIONS,
        )

        runtime.append(REPORT_TO_SUPERVISOR_SCHEMA)
        if REPORT_INSTRUCTIONS not in system_prompt:
            system_prompt += "\n\n" + REPORT_INSTRUCTIONS
    if peers_available and agent_kind == AGENT_KIND_SUBAGENT:
        from .fleet_message_tools import (
            LIST_PEER_AGENTS_SCHEMA,
            PEER_INSTRUCTIONS,
            SEND_TO_PEER_SCHEMA,
        )

        runtime.extend((LIST_PEER_AGENTS_SCHEMA, SEND_TO_PEER_SCHEMA))
        if PEER_INSTRUCTIONS not in system_prompt:
            system_prompt += "\n\n" + PEER_INSTRUCTIONS
    if offer_find_load:
        # TASK-26007: the deferred surface NAMES what exists -- the
        # model must never conclude a present capability is absent.
        runtime.extend(
            (
                build_find_tools_schema(deferred_names()),
                LOAD_TOOLS_SCHEMA,
            )
        )
    if skill_file_enabled:
        runtime.append(SKILL_FILE_TOOL_SCHEMA)
    if install_skill_enabled and agent_kind == AGENT_KIND_PRIMARY:
        runtime.append(INSTALL_SKILL_TOOL_SCHEMA)
    if managed_skill_promotion_enabled and agent_kind == AGENT_KIND_PRIMARY:
        runtime.append(PREPARE_MANAGED_SKILL_PROMOTION_TOOL_SCHEMA)
    if run_skill_script_enabled:
        runtime.append(RUN_SKILL_SCRIPT_TOOL_SCHEMA)
    if fork_chat_enabled:
        # TASK-32874: the DISCLOSED schema is built per run (preset
        # roster + gated override args, ADR-147 pattern); the static
        # constant remains only as the nothing-to-add identity base.
        # TASK-32531: disclosed to sub-agents too (children are scoped
        # to the parent conversation; confirm still gates every call).
        runtime.append(
            build_chat_create_schema(
                FORK_CHAT_TOOL_SCHEMA,
                agent_definitions or (),
                override_enabled=spawn_override_enabled,
                override_targets=spawn_override_targets,
            )
        )
    if new_chat_enabled:
        runtime.append(
            build_chat_create_schema(
                NEW_CHAT_TOOL_SCHEMA,
                agent_definitions or (),
                override_enabled=spawn_override_enabled,
                override_targets=spawn_override_targets,
            )
        )
    log_active = bool(
        agent_kind == AGENT_KIND_PRIMARY and run_log_active and (runtime or active)
    )
    if log_active:
        runtime.extend(
            (
                SEARCH_RUN_LOG_TOOL_SCHEMA,
                RUN_LOG_STATS_TOOL_SCHEMA,
                RUN_LOG_SLICE_TOOL_SCHEMA,
            )
        )
    return FirstRequestSchemaPlan(
        active_schemas=active,
        runtime_schemas=tuple(runtime),
        offer_find_load=offer_find_load,
        log_active=log_active,
        system_prompt=system_prompt,
        agent_definitions=agent_definitions,
        fleet_max_live=fleet_max_live,
    )


def build_compact_schema_plan(
    no_tools: FirstRequestSchemaPlan,
    discovery: FirstRequestSchemaPlan,
    *,
    context_limit: int,
    response_reserve_tokens: int,
    progress_available: bool,
    reporting_available: bool,
    peers_available: bool,
    agent_kind: str,
) -> FirstRequestSchemaPlan:
    """Keep discovery and already-gated progress under the same input ceiling."""
    # ADR-253: optional runtime schemas must not crowd out a usable
    # catalog. Keep the caller's session prompt and only the discovery
    # and already-gated progress surface, then cost the whole request.
    from .fleet_message_tools import (
        PEER_INSTRUCTIONS,
        READ_INSTRUCTIONS,
        REPORT_INSTRUCTIONS,
    )

    compact_names = MESSAGE_TOOL_NAMES | {FIND_TOOLS_NAME, LOAD_TOOLS_NAME}
    compact_runtime = tuple(
        schema for schema in discovery.runtime_schemas if schema.name in compact_names
    )
    compact_prompt = (
        no_tools.system_prompt
        + "\n\nUse find_tools to discover available tools, then call load_tools "
        "alone with ids for every tool to retain before calling those tools."
    )
    for enabled, instruction in (
        (progress_available and agent_kind == AGENT_KIND_PRIMARY, READ_INSTRUCTIONS),
        (
            reporting_available and agent_kind == AGENT_KIND_SUBAGENT,
            REPORT_INSTRUCTIONS,
        ),
        (peers_available and agent_kind == AGENT_KIND_SUBAGENT, PEER_INSTRUCTIONS),
    ):
        if enabled and instruction not in compact_prompt:
            compact_prompt += "\n\n" + instruction
    return dataclasses.replace(
        no_tools,
        runtime_schemas=compact_runtime,
        offer_find_load=True,
        system_prompt=compact_prompt,
        schema_input_budget_tokens=context_limit - response_reserve_tokens,
    )
