"""Render admitted Console skills and launch their scoped inline children."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tldw_chatbook.Agents.agent_models import SkillFileBindings, ToolResult
from tldw_chatbook.Agents.tool_catalog import intersect_skill_tools
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError

from .console_skill_resolver import SKILL_UNTRUSTED_REFUSE


class _BridgeSkillRunner:
    """``SkillRunner``: renders a skill, then routes it through THIS run's spawn.

    Built fresh per ``run_reply`` invocation from that run's own eligible
    skill-name set and builtin + local tool names (see
    ``_compose_run_registry_and_allowed``).
    ``run`` re-verifies trust at render time via ``execute_skill`` -- never
    a cached snapshot -- so a skill approved when the catalog was built but
    revoked before the model actually calls it still refuses (mirrors
    ``ConsoleChatController._apply_skill_substitution``'s own re-verification
    discipline for the ``$skill-name`` user-invocation path).
    """

    def __init__(
        self,
        *,
        skills_service: Any,
        skill_names: frozenset[str],
        builtin_names: tuple[str, ...],
        local_names: tuple[str, ...] = (),
        skill_file_bindings: SkillFileBindings | None = None,
        definition_digests: Mapping[str, str] | None = None,
        plugin_entries: Sequence[Mapping[str, Any]] = (),
        owned_mcp_names: tuple[str, ...] = (),
        plugin_registry=None,
        plugin_eligible: frozenset[str] = frozenset(),
        plugin_provider: str = "",
        skill_report_owner: tuple[Any, tuple[str, int, str]] | None = None,
    ) -> None:
        self._skill_report_owner = skill_report_owner
        self._owned_mcp_names = owned_mcp_names
        self._plugin_registry = plugin_registry
        self._plugin_eligible = plugin_eligible
        self._plugin_provider = plugin_provider
        self._plugin_entries = {str(row["tool_name"]): row for row in plugin_entries}
        self._skills_service = skills_service
        self._skill_names = skill_names
        self._builtin_names = builtin_names
        self._local_names = local_names
        self._skill_file_bindings = skill_file_bindings
        self._definition_digests = dict(definition_digests or {})

    def is_skill_tool(self, name: str) -> bool:
        return name in self._skill_names

    def run(self, name: str, args: str, spawn: Callable[..., ToolResult]) -> ToolResult:
        try:
            expected_digest = self._definition_digests.get(name)
            if expected_digest is not None:
                local = getattr(self._skills_service, "local_service", None)
                if local is None and hasattr(self._skills_service, "trust_service"):
                    local = self._skills_service
                trust = getattr(local, "trust_service", None)
                plugin_entry = self._plugin_entries.get(name)
                current_digest = (
                    local.plugin_service.current_definition(
                        name, plugin_entry["plugin_admission"]
                    )
                    if plugin_entry is not None
                    else (
                        trust.current_fingerprint_digest(name)
                        if trust is not None
                        else None
                    )
                )
                if current_digest != expected_digest:
                    raise SkillTrustBlockedError(
                        skill_name=name,
                        reason_code="skill_definition_changed",
                        trust_status="quarantined_modified",
                    )
            result = asyncio.run(
                self._skills_service.execute_skill(
                    name,
                    mode="local",
                    args=args,
                    **(
                        {
                            "plugin_admission": self._plugin_entries[name][
                                "plugin_admission"
                            ]
                        }
                        if name in self._plugin_entries
                        else {}
                    ),
                )
            )
        except PermissionError:
            return ToolResult(
                ok=False, error="Plugin authority changed; review and retry."
            )
        except SkillTrustBlockedError as exc:
            return ToolResult(
                ok=False,
                error=SKILL_UNTRUSTED_REFUSE.format(name=name, reason=exc.reason_code),
            )
        rendered = (
            result.get("rendered_prompt", "") if isinstance(result, Mapping) else ""
        )
        declared_allowed_tools = (
            result.get("allowed_tools") if isinstance(result, Mapping) else None
        )
        model = None
        if name in self._plugin_entries:
            from tldw_chatbook.Chat.provider_readiness import provider_config_key
            from tldw_chatbook.Plugins.host_references import resolve_tool_ceiling

            try:
                declared_allowed_tools = resolve_tool_ceiling(
                    declared_allowed_tools, self._plugin_registry, self._plugin_eligible
                )
                model = result.get("model_override")
                if model and (
                    not self._plugin_provider
                    or provider_config_key(result.get("model_provider", ""))
                    != provider_config_key(self._plugin_provider)
                ):
                    raise PermissionError("plugin_model_parent_mismatch")
            except PermissionError:
                return ToolResult(
                    ok=False,
                    error="Plugin constraints are unavailable; review and retry.",
                )
        # Narrow-only against THIS run's builtin + local tool names: a skill
        # can never grant its child a tool the parent run doesn't have (no
        # runtime tools, no MCP tools, no other skills). Local tools in the
        # child stay approval-gated -- the spawn below shares the parent's
        # review hook and stamp scope. ``None`` (undeclared) passes the full
        # builtin + local set through, matching how native spawn_subagent
        # children already inherit local tools.
        allowed_tools = intersect_skill_tools(
            declared_allowed_tools,
            self._builtin_names
            + self._local_names
            + (self._owned_mcp_names if name in self._plugin_entries else ()),
        )
        # task-4 (skills-fork-reachability): grant the spawned skill's own
        # name skill_file authorization BEFORE spawn -- so the child's very
        # first turn can already read its own bundled reference files (see
        # SkillFileBindings' own docstring: authorization lives here, never
        # in config.allowed_tools) -- then append a "Bundled files" pointer
        # block to the rendered task text whenever execute_skill reported
        # any (absent when the skill has no bundle beyond SKILL.md).
        if self._skill_file_bindings is not None:
            self._skill_file_bindings.authorize(
                name, self._definition_digests.get(name)
            )
        skill_body = rendered
        refs = result.get("reference_files") if isinstance(result, Mapping) else None
        if refs and self._skill_file_bindings is not None:
            rows = ", ".join(
                f"{r['path']} ({r['size']} bytes"
                f"{'' if r.get('is_text', True) else ', binary'})"
                for r in refs
            )
            from tldw_chatbook.Agents.agent_models import carry_plugin_context

            rendered = carry_plugin_context(
                f"{rendered}\n\nBundled files (readable via skill_file): {rows}",
                rendered,
            )
        spawn_kwargs = {"model": model} if model else {}
        if self._skill_report_owner is None or not isinstance(skill_body, str):
            return spawn(rendered, allowed_tools=allowed_tools, **spawn_kwargs)

        from .console_skill_cost_report import (
            SKILL_CHILD_OBSERVATION,
            SkillChildObservation,
            observe_skill_use,
            skill_body_digest,
        )

        owner = self._skill_report_owner
        display = self._plugin_entries.get(name, {}).get("name", name)
        started = time.monotonic()

        def record_use(body_tokens: int | None) -> None:
            observe_skill_use(*owner, (display,), body_tokens=body_tokens, now=started)

        observation = SkillChildObservation(
            len(skill_body), skill_body_digest(skill_body), on_prepared=record_use
        )
        token = SKILL_CHILD_OBSERVATION.set(observation)
        try:
            spawned = spawn(rendered, allowed_tools=allowed_tools, **spawn_kwargs)
        finally:
            SKILL_CHILD_OBSERVATION.reset(token)
        if not observation.prepared and (
            spawned.ok or spawned.dispatch_state == "settled"
        ):
            record_use(None)
        return spawned
