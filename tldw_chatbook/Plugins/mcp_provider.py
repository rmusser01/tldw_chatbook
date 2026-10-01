"""Owned MCP tools compose the existing provider's approval and typed-result path."""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar

from tldw_chatbook.Agents.agent_models import ToolCatalogEntry, ToolResult
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.MCP.connection_ownership import OwnedMCPInvocation, owned_invocation

_permission_profile = ContextVar("plugin_mcp_permission_profile", default=None)


class PluginMCPProvider(MCPToolProvider):
    """Bind one captured plugin scope to exact reviewed tool definitions.

    Permission profile/persona and parent ceilings remain the normal MCP provider
    inputs; owning a package never substitutes for any of those gates.
    """

    def __init__(self, *, plugin_service, ownership, snapshot, **kwargs):
        self.plugins = plugin_service
        self.ownership = ownership
        self.snapshot = snapshot
        mappings = [
            row
            for row in json.loads(snapshot.mappings_json)
            if row["kind"] == "tool" and row["component_id"] in snapshot.selection
        ]
        self._tool_mappings = {row["target_reference"]: row for row in mappings}
        ceiling = kwargs.pop("maximum_tool_ids", None)
        hashes = kwargs.pop("maximum_definition_hashes", None)
        allowed = frozenset(self._tool_mappings)
        if ceiling is not None:
            allowed &= ceiling
        if hashes is not None:
            allowed = frozenset(
                key
                for key in allowed
                if hashes.get(key) == self._tool_mappings[key]["definition_digest"]
            )
        super().__init__(
            maximum_tool_ids=allowed,
            maximum_definition_hashes={
                key: self._tool_mappings[key]["definition_digest"] for key in allowed
            },
            owned_profile_ids=frozenset(
                key.partition("::")[0].removeprefix("local:") for key in allowed
            ),
            **kwargs,
        )

    def _kill_switch_engaged(self) -> bool:
        """Owned authority refuses unavailable switch state without logging it."""
        try:
            return bool(self._service.get_kill_switch())
        except Exception:  # noqa: BLE001
            return True

    def _current_persona_policy(self):
        from tldw_chatbook.Agents.persona_policy import PersonaToolPolicy

        try:
            policy = (
                self._persona_policy_provider()
                if self._persona_policy_provider is not None
                else None
            )
            if policy is not None and not isinstance(policy, PersonaToolPolicy):
                raise TypeError("invalid persona authority")
            return policy
        except Exception:  # noqa: BLE001
            raise PermissionError("plugin_permission_unavailable") from None

    def _persona_floor(self, state, tool):
        from tldw_chatbook.Agents.persona_policy import persona_floor_state

        policy = self._current_persona_policy()
        return (
            state if policy is None else persona_floor_state(state, policy, tool.name)
        )

    async def compose_catalog(self) -> None:
        self._catalog, self._entry_by_llm_name = [], {}
        self._not_connected_count = 0
        with self._decisions_lock:
            self._stamped_decisions.clear()
        try:
            self._current_persona_policy()
            if self._kill_switch_engaged():
                return
            for mapping in self._tool_mappings.values():
                await self.plugins.check_mcp_snapshot(
                    self.snapshot, mapping["component_id"]
                )
            await super().compose_catalog()
            renamed, catalog = {}, []
            for tool, state in self._entry_by_llm_name.values():
                mapping = self._tool_mappings[tool.tool_id]
                name = (
                    "plugin_mcp_"
                    + hashlib.sha256(
                        (
                            self.snapshot.installation_id
                            + ":"
                            + mapping["component_id"]
                            + ":"
                            + tool.name
                        ).encode()
                    ).hexdigest()[:48]
                )
                renamed[name] = (tool, state)
                catalog.append(
                    ToolCatalogEntry(
                        id=name,
                        name=name,
                        one_line_description=tool.description,
                        source="plugin_mcp",
                    )
                )
            self._entry_by_llm_name = renamed
            self._catalog = catalog
            self._current_persona_policy()
            if self._kill_switch_engaged():
                self._catalog, self._entry_by_llm_name = [], {}
        except Exception:  # noqa: BLE001
            self._catalog, self._entry_by_llm_name = [], {}
            self._not_connected_count = 0
            raise PermissionError("plugin_catalog_authority_unavailable") from None

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        try:
            profile = self._profile_kwargs()
        except Exception:  # noqa: BLE001
            return ToolResult(
                ok=False,
                error="plugin_permission_unavailable",
                dispatch_state="not_started",
            )
        token = _permission_profile.set(profile)
        try:
            return super().invoke(tool_id, args)
        finally:
            _permission_profile.reset(token)

    def _check_permission(self, tool, args, decision=None):
        if (
            self._kill_switch_engaged()
            or self._profile_kwargs() != _permission_profile.get()
        ):
            raise PermissionError("plugin_permission_changed")
        state = self._persona_floor(
            self._service.gate_tool_test(tool, **self._profile_kwargs()), tool
        )
        if state.state == "deny":
            raise PermissionError("plugin_permission_changed")
        if (
            decision == "allowed"
            and state.state != "allow"
            and not self._arg_rule_allows_safe(tool, args)
        ):
            raise PermissionError("plugin_permission_changed")
        if decision == "approved-session" and not self._is_session_approved_safe(tool):
            raise PermissionError("plugin_permission_changed")

    def _check_result_permission(self, tool, args, decision):
        # Called under the original invocation's captured permission profile.
        self._check_permission(tool, args, decision)

    def _result_currentness(self, tool, args, decision):
        current = super()._result_currentness(tool, args, decision)
        snapshot = self.snapshot
        mapping = dict(self._tool_mappings[tool.tool_id])

        def check():
            current()
            if (
                self.snapshot != snapshot
                or self._tool_mappings.get(tool.tool_id) != mapping
            ):
                raise PermissionError("plugin_result_authority_changed")
            self.plugins.fences.check_snapshot(snapshot)
            self.plugins._call_from_agent(
                lambda: self.plugins._admission.check(snapshot, mapping["component_id"])
            )
            # That existing owner check can wait; re-read permission/automatic
            # authority after it, preserving the original same-call approval.
            current()

        return check

    def _apply_verdict(self, verdict, tool, args, *, unanswered=False):
        try:
            self._check_permission(tool, args)
        except Exception:  # noqa: BLE001
            return ToolResult(
                ok=False,
                error="plugin_permission_changed",
                dispatch_state="not_started",
            )
        return super()._apply_verdict(verdict, tool, args, unanswered=unanswered)

    def _execute(self, tool, args, *, decision):
        mapping = self._tool_mappings.get(tool.tool_id)
        if mapping is None:
            return ToolResult(
                ok=False, error="plugin_tool_not_reviewed", dispatch_state="not_started"
            )
        try:
            self._check_permission(tool, args, decision)
            self.plugins.fences.check_snapshot(self.snapshot)
            self.plugins._call_from_agent(
                lambda: self.plugins._admission.check(
                    self.snapshot, mapping["component_id"]
                )
            )
        except Exception:  # noqa: BLE001
            return ToolResult(
                ok=False,
                error="plugin_tool_authority_changed",
                dispatch_state="not_started",
            )
        token = owned_invocation.set(
            OwnedMCPInvocation(
                self.ownership,
                self.snapshot,
                mapping["component_id"],
                mapping,
                lambda: self._check_permission(tool, args, decision),
            )
        )
        try:
            result = super()._execute(tool, args, decision=decision)
            try:
                self._check_permission(tool, args, decision)
                self.plugins._call_from_agent(
                    lambda: self.plugins._admission.check(
                        self.snapshot, mapping["component_id"]
                    )
                )
            except Exception:  # noqa: BLE001
                return ToolResult(
                    ok=False,
                    error="plugin_result_revoked",
                    dispatch_state=result.dispatch_state,
                )
            return result
        finally:
            owned_invocation.reset(token)

    async def connect(self, component_id: str) -> dict:
        """Explicit host launch using the reviewed connection and normal launch gate."""
        mappings = [
            row
            for row in json.loads(self.snapshot.mappings_json)
            if row["kind"] == "connection" and row["component_id"] == component_id
        ]
        if len(mappings) != 1:
            raise PermissionError("plugin_connection_not_reviewed")
        token = owned_invocation.set(
            OwnedMCPInvocation(self.ownership, self.snapshot, component_id)
        )
        try:
            return await self.ownership.local.connect_profile(
                mappings[0]["target_reference"].removeprefix("local:")
            )
        finally:
            owned_invocation.reset(token)
