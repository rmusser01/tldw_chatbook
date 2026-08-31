from __future__ import annotations

import asyncio
import inspect
import json
import math
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger

from tldw_chatbook.config import coerce_bool_setting, get_cli_setting
from tldw_chatbook.runtime_policy.types import RuntimeSourceState

from .execution_log import MCPExecutionLog, build_record
from .hub_tool_catalog import (
    HubTool,
    builtin_tools_from_inventory,
    local_tools_from_record,
    schema_argument_names,
)
from .local_control_service import MCPGovernanceDenied
from .local_runtime_delegate import (
    PERMISSION_STATE_UNRESOLVED_CLAUSE,
    RAW_TOOL_CALL_REFUSED_MESSAGE,
    RawToolCallRefusedError,
    capitalize_first,
)
from .permission_prompt_reducer import (
    DEFAULT_MIN_APPROVED_COUNT,
    PermissionPromptRecommendation,
    PermissionPromptReport,
    build_permission_prompt_report,
)
from .permission_store import (
    EffectiveToolState,
    HASH_FREE_SERVER_KEYS,
    MCPPermissionStore,
    definition_hash,
    resolve_effective_state,
    resolve_effective_state_by_key,
)
from .redaction import is_secret_key, redact_mapping
from .readiness import BUILTIN_SERVER_KEY
from .server_target_store import ConfiguredServerTargetStore
from .unified_context_store import UnifiedMCPContextStore
from .unified_control_models import ServerAccessContext, UnifiedMCPContext
from tldw_chatbook.Utils.filesystem_identity import DirectoryChain

if TYPE_CHECKING:
    from .hub_test_execution import (
        LocalHubDecision,
        LocalHubExecutionCoordinator,
        LocalHubExecutionOutcome,
        OneShotLocalHubApproval,
        RegisteredToolTestPreview,
        ToolTestAdmissionBlocked,
        ToolTestAdmissionPreview,
        ToolTestAdmissionStale,
        ToolTestPreviewRegistry,
    )

# Task 6 (PR-T3), Route B: refusal copy for the Advanced runner's
# `tool.execute` action when the Hub's per-tool permission gate resolves the
# named built-in tool to "deny". Mirrors the Test Tool runner's own blocked
# sentence (`_TOOL_TEST_BLOCKED_TEXT`, mcp_workbench.py) but names the tool,
# since the Advanced panel has no tool detail beside it to say which one.
# Deliberately does NOT lead with "Blocked": the inspector renders it under
# `show_tool_result()`'s "Blocked · not run" heading, and every other
# PermissionError this path can surface (the in-process runtime-governance
# profile's own denials, raised by `local_control_service.execute_tool()`)
# reads correctly under that same heading.
#
# ONLY for a genuine resolved "deny" (`state.origin != "gate_error"`) --
# see `_ADVANCED_EXECUTE_GATE_ERROR_MESSAGE` just below for the synthesized
# fail-closed case, where this claim would be false (the tool's actual
# state was never determined).
_ADVANCED_EXECUTE_BLOCKED_MESSAGE = "{tool} is set to Off in Permissions."

# Item 1 (PR-T3 fix round F): honest counterpart to
# `_ADVANCED_EXECUTE_BLOCKED_MESSAGE` above for `execute_advanced_tool()`'s
# OWN fail-closed gate (`gate_tool_test_by_key()` raised, synthesized as
# `EffectiveToolState(state="deny", origin="gate_error")` a few hundred
# lines down) -- the resolver crashed, so the tool's configured state was
# never determined; it is not necessarily "Off" at all. Before this, the
# deny branch below used `_ADVANCED_EXECUTE_BLOCKED_MESSAGE` unconditionally
# for BOTH cases, so a corrupt/erroring permission store told the user a
# confident, false fact about their own configuration -- the THIRD
# occurrence of the pattern task-2536 (fix round B, item 2) fixed on the
# Test Tool panel's blocked-result body and task-2270's rider fixed on its
# quiet decision note.
#
# Fix Round G, Item 7 (review of Fix Round F): the prior version of this
# comment claimed the twin worth matching was `mcp_inspector._UNKNOWN_
# ORIGIN_SENTENCE` -- wrong surface AT THE TIME. `_UNKNOWN_ORIGIN_
# SENTENCE` was then the Permissions-detail-panel/quiet-decision-note
# sentence, still an independently maintained literal; the twin THAT
# round actually converged with was `mcp_workbench._TOOL_TEST_BLOCKED_
# UNKNOWN_TEXT`, the Test Tool panel's own LOUD blocked-run body for this
# identical `gate_error` condition.
#
# Fix Round I, Item 4 (review, recount): a later review found
# `_UNKNOWN_ORIGIN_SENTENCE` was STILL a third, separately maintained
# literal stating this same claim -- reachable whenever a Tools-mode tool
# selection's own `gate_tool_test()` call raises
# (`MCPWorkbench._effective_for_display()`'s single-tool fallback path;
# `_decision_note()`'s OWN former `gate_error` branch, a fourth candidate,
# was proven dead and removed one round earlier -- see that commit for why
# it no longer counts). All three -- this constant, `_TOOL_TEST_BLOCKED_
# UNKNOWN_TEXT`, and `_UNKNOWN_ORIGIN_SENTENCE` -- now derive from one
# shared clause, `local_runtime_delegate.PERMISSION_STATE_UNRESOLVED_
# CLAUSE` (see that module for the sharing rationale -- same
# dependency-safe-common-ground precedent as `RAW_TOOL_CALL_REFUSED_
# MESSAGE` just above), rather than being re-imported into each other:
# this module is imported BY `mcp_workbench.py`/`mcp_inspector.py` (see
# `_ADVANCED_EXECUTE_BLOCKED_MESSAGE`'s own mirrored-not-shared precedent,
# and `_run_advanced_action()`'s docstring in `mcp_inspector.py` for why
# the reverse import would be circular), so deriving all three from a
# FOURTH, dependency-safe module lets a reword of the underlying claim
# change every call site, without any of them importing each other.
#
# `capitalize_first()`, not `.capitalize()`: the latter also lowercases
# everything AFTER the first character, silently mangling any acronym a
# future clause might contain -- see that function's own docstring in
# `local_runtime_delegate.py` for the proof.
_ADVANCED_EXECUTE_GATE_ERROR_MESSAGE = (
    f"{capitalize_first(PERMISSION_STATE_UNRESOLVED_CLAUSE)}."
)

# task-2539 (PR-T3 fix round B, item 3): the exact message
# `execute_hub_tool()` raises below for a server-source `server_key`. Its
# own constant so the raise site and `MCPServerSourceDisplayOnlyError`'s
# default message can never drift apart from each other.
_SERVER_SOURCE_DISPLAY_ONLY_MESSAGE = "Server-source tools are display-only."


class MCPServerSourceDisplayOnlyError(ValueError):
    """`execute_hub_tool()`'s refusal for a server-source ``server_key``.

    Raised ONLY when ``server_key`` is neither ``local:`` nor
    ``builtin:`` -- a server-source tool has no in-process execution path
    through this seam at all (see the Hub's Governance mode for how a
    server-source tool's definition is shown instead). Subclasses
    ``ValueError`` so any existing ``except ValueError`` handler upstream
    keeps working unchanged.

    task-2539: before this, `mcp_workbench._is_permission_refusal()` told
    this refusal apart from an unrelated ``ValueError`` by comparing
    ``str(exc)`` against this exact sentence -- fragile, since nothing
    pinned that string AT THIS RAISE SITE, so an unrelated reword here
    would have silently reverted that classifier's fix with a fully green
    suite. Matching THIS TYPE instead makes the message text purely
    cosmetic for classification; it stays byte-identical to what the bare
    ``ValueError`` this replaces carried, purely so the rendered text is
    unchanged for anyone reading the result.
    """

    def __init__(self, message: str = _SERVER_SOURCE_DISPLAY_ONLY_MESSAGE) -> None:
        super().__init__(message)


class MCPHubGateDeniedError(PermissionError):
    """`execute_advanced_tool()`'s refusal when the Hub's OWN per-tool
    permission gate (Allow/Ask/Off) resolves to "Off".

    Item 2 (PR-T3 fix round D). Deliberately a DIFFERENT type from
    ``local_control_service.MCPGovernanceDenied`` -- that exception names a
    denial from the in-process runtime-governance profile, a separate
    permission system checked one layer further in (inside
    :meth:`UnifiedMCPControlPlaneService.execute_hub_tool`'s ``coro``, via
    ``local_control_service._require_runtime_governance_allowed()``).
    Conflating the two into one type would make it impossible to tell, from
    the exception alone, WHICH gate refused a given call -- exactly the
    distinction Fix Round D, Item 1's `error_category` tokens
    (``"gate_denied"`` here vs. ``"governance_denied"`` there) preserve on
    the execution-log side; this type preserves the same distinction on the
    raise side.

    Subclasses ``PermissionError`` so any existing ``except
    PermissionError`` handler upstream keeps working unchanged.
    """


@dataclass(frozen=True, slots=True)
class _ResolvedHubTest:
    """One fresh, service-owned Hub Test Tool resolution."""

    tool: HubTool
    authority: DirectoryChain | None = None
    safe_authority_label: str | None = None
    unavailable_reason: str | None = None


# Task 6 (PR-T3), Route B, second door: `runtime.request`/`runtime.batch` are
# Advanced descriptors too, and the in-process runtime speaks the real
# protocol -- `{"method": "tools/call"}` reaches the SAME
# `runtime_delegate.execute_tool()` as the `tool.execute` action, with the
# same two holes it had (no Hub permission gate, no execution-log row).
# Gating only `tool.execute` would lock the front door and leave this open,
# so tool execution keeps exactly one door here: the gated, logged one.
# Every other protocol method (tools/list, prompts/list, status/get, ...)
# is untouched -- this runner's diagnostic value is in those.
#
# Fix Round A, Item 2: this refusal (the control-plane pre-dispatch scan,
# below) is one of TWO independent enforcement points -- the other is
# `LocalMCPRuntimeDelegate.request()` itself, the durable backstop for
# callers that reach the delegate without going through this scan at all.
# The message constant is imported from that module, not redefined here, so
# the two can never show the user different copy for the same refusal. This
# scan's own job (see the full rationale on the import site): preserve
# `runtime.batch`'s all-or-nothing property, which a delegate-level refusal
# alone cannot -- the batch runs serially, so a per-item refusal there would
# only stop the offending item, not the ones dispatched before it.
_RAW_TOOL_CALL_REFUSED_MESSAGE = RAW_TOOL_CALL_REFUSED_MESSAGE


class UnifiedMCPControlPlaneService:
    """Destination-local orchestration for local/server Unified MCP browse flows."""

    def __init__(
        self,
        *,
        target_store: ConfiguredServerTargetStore | None,
        context_store: UnifiedMCPContextStore | None,
        local_service: Any,
        server_service: Any,
    ) -> None:
        self.target_store = target_store
        self.context_store = context_store
        self.local_service = local_service
        self.server_service = server_service
        self.context = (
            self.context_store.load()
            if self.context_store is not None
            else UnifiedMCPContext()
        )
        self._execution_log: MCPExecutionLog | None = None
        self._execution_log_init_lock = threading.Lock()
        self._permission_store: MCPPermissionStore | None = None
        self._hub_test_state_lock = threading.Lock()
        self._hub_test_previews: ToolTestPreviewRegistry | None = None
        self._local_hub_execution: LocalHubExecutionCoordinator | None = None
        # Chat bridge (Phase 5): in-memory-only, app-run-lifetime session
        # approvals. Never persisted -- a fresh process/instance starts
        # empty, and `clear_session_approvals()` is the only other way
        # entries leave this set.
        self._session_approvals: set[tuple[str, str]] = set()

    def _ensure_hub_test_state(
        self,
    ) -> tuple[ToolTestPreviewRegistry, LocalHubExecutionCoordinator]:
        """Create first-use Hub test state without charging app startup."""
        previews = self._hub_test_previews
        execution = self._local_hub_execution
        if previews is not None and execution is not None:
            return previews, execution
        with self._hub_test_state_lock:
            previews = self._hub_test_previews
            execution = self._local_hub_execution
            if previews is None:
                from .hub_test_execution import (
                    ToolTestPreviewRegistry,
                )

                previews = ToolTestPreviewRegistry()
                self._hub_test_previews = previews
            if execution is None:
                from .hub_test_execution import LocalHubExecutionCoordinator

                execution = LocalHubExecutionCoordinator()
                self._local_hub_execution = execution
            return previews, execution

    @property
    def selected_source(self) -> str:
        return self.context.selected_source

    async def load_context(self) -> UnifiedMCPContext:
        if self.context_store is not None:
            self.context = self.context_store.load()
        return self.context

    async def select_source(self, source: str) -> UnifiedMCPContext:
        normalized_source = (
            "server" if str(source or "").strip() == "server" else "local"
        )
        self.context = replace(self.context, selected_source=normalized_source)
        if (
            normalized_source == "server"
            and self.context.selected_active_server_id is None
        ):
            default_target = self._resolve_target(None)
            if default_target is not None:
                return await self.select_server_target(default_target.server_id)
        self._persist_context()
        return self.context

    async def select_server_target(self, server_id: str | None) -> UnifiedMCPContext:
        target = self._resolve_target(server_id)
        if target is None:
            raise KeyError(f"Unknown server_id: {server_id}")
        if self.server_service is None:
            raise ValueError("Server Unified MCP service is unavailable.")

        restored_state = self.context.per_server_state.get(
            target.server_id, ServerAccessContext(server_id=target.server_id)
        )
        access_context = await self._maybe_await(
            self.server_service.resolve_access_context(
                target=target,
                selected_scope=restored_state.selected_scope
                or self.context.selected_scope,
                selected_scope_ref=restored_state.selected_scope_ref
                or self.context.selected_scope_ref,
                selected_section=restored_state.selected_section
                or self.context.selected_section,
            )
        )
        self._apply_server_access_context(target.server_id, access_context)
        return self.context

    async def select_scope(
        self, scope: str | None, scope_ref: str | None = None
    ) -> UnifiedMCPContext:
        if self.context.selected_source != "server":
            self.context = replace(
                self.context,
                selected_scope=scope,
                selected_scope_ref=scope_ref,
            )
            self._persist_context()
            return self.context

        target = self._require_active_server_target()
        access_context = await self._maybe_await(
            self.server_service.resolve_access_context(
                target=target,
                selected_scope=scope,
                selected_scope_ref=scope_ref,
                selected_section=self.context.selected_section,
            )
        )
        self._apply_server_access_context(target.server_id, access_context)
        return self.context

    async def select_section(self, section: str | None) -> UnifiedMCPContext:
        if self.context.selected_source != "server":
            self.context = replace(self.context, selected_section=section)
            self._persist_context()
            return self.context

        target = self._require_active_server_target()
        access_context = await self._maybe_await(
            self.server_service.resolve_access_context(
                target=target,
                selected_scope=self.context.selected_scope,
                selected_scope_ref=self.context.selected_scope_ref,
                selected_section=section,
            )
        )
        self._apply_server_access_context(target.server_id, access_context)
        return self.context

    async def load_section(self, section: str | None = None) -> dict[str, Any]:
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "server":
            target = self._require_active_server_target()
            access_context = self.context.per_server_state.get(target.server_id)
            if access_context is None:
                await self.select_server_target(target.server_id)
                access_context = self.context.per_server_state.get(target.server_id)
            if access_context is None:
                raise RuntimeError(
                    "Failed to resolve Unified MCP server access context."
                )
            if access_context.selected_section != effective_section:
                await self.select_section(effective_section)
                access_context = self.context.per_server_state[target.server_id]

            if effective_section == "overview":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_overview(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "inventory":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_inventory(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "catalogs":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_catalogs(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "external_servers":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_external_servers(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "governance":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_governance(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            if effective_section == "advanced":
                return self._with_server_context(
                    await self._maybe_await(
                        self.server_service.get_advanced(
                            target=target,
                            access_context=access_context,
                        )
                    ),
                    section=effective_section,
                )
            raise ValueError(f"Unsupported Unified MCP section: {effective_section}")

        self.context = replace(self.context, selected_section=effective_section)
        self._persist_context()
        if effective_section == "overview":
            return await self._maybe_await(self.local_service.get_overview())
        if effective_section == "inventory":
            return await self._maybe_await(self.local_service.get_inventory())
        if effective_section == "external_servers":
            return await self._maybe_await(self.local_service.get_external_servers())
        if effective_section == "governance":
            return {
                "source": "local",
                "section": "governance",
                "rules": list(
                    await self._maybe_await(self.local_service.get_governance())
                ),
            }
        if effective_section == "advanced":
            return await self._maybe_await(self.local_service.get_advanced())
        raise ValueError(f"Unsupported Unified MCP section: {effective_section}")

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _with_server_context(
        self, payload: dict[str, Any], *, section: str
    ) -> dict[str, Any]:
        return {
            **dict(payload or {}),
            "source": "server",
            "section": section,
        }

    def available_actions(self) -> list[dict[str, Any]]:
        if self.context.selected_source != "server":
            if (self.context.selected_section or "overview") == "inventory":
                return [
                    {
                        "name": "tool.execute",
                        "label": "Execute Local Tool",
                        "action_id": "mcp.runtime.trigger.local",
                        "payload_template": '{"tool_name":"search_notes","arguments":{"query":"example"}}',
                    },
                    {
                        "name": "resource.read",
                        "label": "Read Local Resource",
                        "action_id": "mcp.inventory.observe.local",
                        "payload_template": '{"resource_uri":"note://123"}',
                    },
                    {
                        "name": "prompt.get",
                        "label": "Get Local Prompt",
                        "action_id": "mcp.inventory.observe.local",
                        "payload_template": '{"prompt_name":"summarize_conversation","arguments":{"conversation_id":4}}',
                    },
                ]
            if (self.context.selected_section or "overview") == "external_servers":
                return [
                    {
                        "name": "profile.save",
                        "label": "Save Profile",
                        "action_id": "mcp.external_profiles.configure.local",
                        "payload_template": '{"profile_id":"demo","command":"python","args":["-m","demo.server"]}',
                    },
                    {
                        "name": "profile.delete",
                        "label": "Delete Profile",
                        "action_id": "mcp.external_profiles.configure.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                    {
                        "name": "profile.connect",
                        "label": "Connect Profile",
                        "action_id": "mcp.external_profiles.launch.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                    {
                        "name": "profile.disconnect",
                        "label": "Disconnect Profile",
                        "action_id": "mcp.external_profiles.launch.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                    {
                        "name": "profile.test",
                        "label": "Test Profile",
                        "action_id": "mcp.external_profiles.trigger.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                    {
                        "name": "profile.refresh",
                        "label": "Refresh Profile",
                        "action_id": "mcp.external_profiles.observe.local",
                        "payload_template": '{"profile_id":"demo"}',
                    },
                ]
            if (self.context.selected_section or "overview") == "governance":
                return [
                    {
                        "name": "governance_rule.save",
                        "label": "Save Governance Rule",
                        "action_id": "mcp.governance.configure.local",
                        "payload_template": '{"rule_id":"rule-a","capability_id":"mcp.inventory.list.local","decision":"allow"}',
                    },
                    {
                        "name": "governance_rule.preview",
                        "label": "Preview Governance Decision",
                        "action_id": "mcp.governance.observe.local",
                        "payload_template": '{"capability_id":"mcp.inventory.list.local"}',
                    },
                    {
                        "name": "governance_rule.delete",
                        "label": "Delete Governance Rule",
                        "action_id": "mcp.governance.configure.local",
                        "payload_template": '{"rule_id":"rule-a"}',
                    },
                ]
            if (self.context.selected_section or "overview") == "advanced":
                return [
                    {
                        "name": "runtime.access.preview",
                        "label": "Preview Local Runtime Access",
                        "action_id": "mcp.governance.observe.local",
                        "payload_template": '{"action_name":"tool.execute","payload":{"tool_name":"search_notes","arguments":{"query":"example"}}}',
                    },
                    {
                        "name": "runtime.activity.list",
                        "label": "List Local Runtime Activity",
                        "action_id": "mcp.runtime.observe.local",
                        "payload_template": '{"limit":5}',
                    },
                    {
                        "name": "runtime.protocol.inspect",
                        "label": "Inspect Local Protocol",
                        "action_id": "mcp.runtime.observe.local",
                        "payload_template": "{}",
                    },
                    {
                        "name": "runtime.health.get",
                        "label": "Get Local Runtime Health",
                        "action_id": "mcp.runtime.observe.local",
                        "payload_template": "{}",
                    },
                    {
                        "name": "approval_requests.list",
                        "label": "List Local Approval Requests",
                        "action_id": "mcp.governance.observe.local",
                        "payload_template": "{}",
                    },
                    {
                        "name": "approval_request.approve",
                        "label": "Approve Local Request",
                        "action_id": "mcp.governance.approve.local",
                        "payload_template": '{"request_id":"approval-a"}',
                    },
                    {
                        "name": "approval_request.deny",
                        "label": "Deny Local Request",
                        "action_id": "mcp.governance.approve.local",
                        "payload_template": '{"request_id":"approval-a"}',
                    },
                    {
                        "name": "approval_request.delete",
                        "label": "Delete Local Request",
                        "action_id": "mcp.governance.approve.local",
                        "payload_template": '{"request_id":"approval-a"}',
                    },
                    {
                        "name": "runtime.status.get",
                        "label": "Get Local Runtime Status",
                        "action_id": "mcp.runtime.observe.local",
                        "payload_template": "{}",
                    },
                    {
                        "name": "runtime.request",
                        "label": "Send Local Runtime Request",
                        "action_id": "mcp.runtime.trigger.local",
                        "payload_template": '{"method":"tools/list","params":{}}',
                    },
                    {
                        "name": "runtime.batch",
                        "label": "Run Local Runtime Batch",
                        "action_id": "mcp.runtime.trigger.local",
                        "payload_template": '{"requests":[{"method":"tools/list"},{"method":"prompts/list"}]}',
                    },
                ]
            return []

        if (self.context.selected_section or "overview") == "catalogs":
            if (self.context.selected_scope or "personal") not in {
                "team",
                "org",
                "system_admin",
            }:
                return []
            return [
                {
                    "name": "catalog.create",
                    "label": "Create Catalog",
                    "action_id": "mcp.catalogs.configure.server",
                    "payload_template": '{"name":"Team Catalog","description":"Scoped"}',
                },
                {
                    "name": "catalog.entry.create",
                    "label": "Add Catalog Entry",
                    "action_id": "mcp.catalogs.configure.server",
                    "payload_template": '{"catalog_id":9,"tool_name":"media.search"}',
                },
                {
                    "name": "catalog.delete",
                    "label": "Delete Catalog",
                    "action_id": "mcp.catalogs.configure.server",
                    "payload_template": '{"catalog_id":9}',
                },
                {
                    "name": "catalog.entry.delete",
                    "label": "Delete Catalog Entry",
                    "action_id": "mcp.catalogs.configure.server",
                    "payload_template": '{"catalog_id":9,"tool_name":"media.search"}',
                },
            ]

        if (self.context.selected_section or "overview") == "external_servers":
            if (self.context.selected_scope or "personal") not in {
                "team",
                "org",
                "system_admin",
            }:
                return []
            return [
                {
                    "name": "external_server.create",
                    "label": "Create External Server",
                    "action_id": "mcp.external_servers.configure.server",
                    "payload_template": '{"server_id":"docs","name":"Docs","transport":"http","config":{"url":"https://docs.example/mcp"}}',
                },
                {
                    "name": "external_server.update",
                    "label": "Update External Server",
                    "action_id": "mcp.external_servers.configure.server",
                    "payload_template": '{"server_id":"docs","name":"Docs","enabled":true}',
                },
                {
                    "name": "external_server.delete",
                    "label": "Delete External Server",
                    "action_id": "mcp.external_servers.configure.server",
                    "payload_template": '{"server_id":"docs"}',
                },
                {
                    "name": "external_server.import",
                    "label": "Import External Server",
                    "action_id": "mcp.external_servers.configure.server",
                    "payload_template": '{"server_id":"legacy-docs"}',
                },
                {
                    "name": "external_server.auth_template.update",
                    "label": "Update Auth Template",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","mode":"template","mappings":[]}',
                },
                {
                    "name": "external_server.slots.list",
                    "label": "List Credential Slots",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"server_id":"docs"}',
                },
                {
                    "name": "external_server.slot.create",
                    "label": "Create Credential Slot",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly","display_name":"Read-only token","secret_kind":"bearer_token","privilege_class":"read","is_required":true}',
                },
                {
                    "name": "external_server.slot.update",
                    "label": "Update Credential Slot",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly","display_name":"Read-only token"}',
                },
                {
                    "name": "external_server.slot.delete",
                    "label": "Delete Credential Slot",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "external_server.slot.secret.set",
                    "label": "Set Slot Secret",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly","secret":"replace-me"}',
                },
                {
                    "name": "external_server.slot.secret.clear",
                    "label": "Clear Slot Secret",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "external_server.secret.set",
                    "label": "Set External Server Secret",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"server_id":"docs","secret":"replace-me"}',
                },
            ]

        if (self.context.selected_section or "overview") == "governance":
            return [
                {
                    "name": "permission_profile.create",
                    "label": "Create Permission Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"name":"Default","description":"Scoped profile","mode":"custom","policy_document":{},"is_active":true}',
                },
                {
                    "name": "permission_profile.update",
                    "label": "Update Permission Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"profile_id":1,"name":"Updated Profile"}',
                },
                {
                    "name": "permission_profile.delete",
                    "label": "Delete Permission Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"profile_id":1}',
                },
                {
                    "name": "policy_assignment.create",
                    "label": "Create Policy Assignment",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"target_type":"persona","target_id":"persona-a","profile_id":1,"inline_policy_document":{},"is_active":true}',
                },
                {
                    "name": "policy_assignment.update",
                    "label": "Update Policy Assignment",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2,"approval_policy_id":7}',
                },
                {
                    "name": "policy_assignment.delete",
                    "label": "Delete Policy Assignment",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.override.get",
                    "label": "Get Assignment Override",
                    "action_id": "mcp.governance.observe.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.override.upsert",
                    "label": "Upsert Assignment Override",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2,"override_policy_document":{"allowed_tools":["mcp.tool"]},"is_active":true}',
                },
                {
                    "name": "policy_assignment.override.delete",
                    "label": "Delete Assignment Override",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "approval_policy.create",
                    "label": "Create Approval Policy",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"name":"Default Approval","mode":"ask_every_time","rules":{},"is_active":true}',
                },
                {
                    "name": "approval_policy.update",
                    "label": "Update Approval Policy",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"approval_policy_id":7,"name":"Updated Approval"}',
                },
                {
                    "name": "approval_policy.delete",
                    "label": "Delete Approval Policy",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"approval_policy_id":7}',
                },
                {
                    "name": "approval_decision.create",
                    "label": "Record Approval Decision",
                    "action_id": "mcp.governance.approve.server",
                    "payload_template": '{"approval_policy_id":7,"context_key":"user:7:docs","tool_name":"docs.search","scope_key":"team:21","decision":"approved","duration":"once"}',
                },
                {
                    "name": "policy_assignment.external_access.get",
                    "label": "Preview External Access",
                    "action_id": "mcp.effective_access.observe.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.workspaces.list",
                    "label": "List Assignment Workspaces",
                    "action_id": "mcp.governance.observe.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.workspace.add",
                    "label": "Add Assignment Workspace",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2,"workspace_id":"ws-1"}',
                },
                {
                    "name": "policy_assignment.workspace.delete",
                    "label": "Delete Assignment Workspace",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"assignment_id":2,"workspace_id":"ws-1"}',
                },
                {
                    "name": "permission_profile.bindings.list",
                    "label": "List Profile Bindings",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"profile_id":1}',
                },
                {
                    "name": "permission_profile.binding.upsert",
                    "label": "Upsert Profile Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs","managed_secret_ref_id":"secret-1"}',
                },
                {
                    "name": "permission_profile.slot_binding.upsert",
                    "label": "Upsert Profile Slot Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs","slot_name":"token_readonly","managed_secret_ref_id":"secret-1"}',
                },
                {
                    "name": "permission_profile.binding.delete",
                    "label": "Delete Profile Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs"}',
                },
                {
                    "name": "permission_profile.slot_binding.delete",
                    "label": "Delete Profile Slot Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "permission_profile.slot_status.get",
                    "label": "Get Profile Slot Status",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"profile_id":1,"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "policy_assignment.bindings.list",
                    "label": "List Assignment Bindings",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"assignment_id":2}',
                },
                {
                    "name": "policy_assignment.binding.upsert",
                    "label": "Upsert Assignment Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","binding_mode":"grant","managed_secret_ref_id":"secret-1"}',
                },
                {
                    "name": "policy_assignment.slot_binding.upsert",
                    "label": "Upsert Assignment Slot Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","slot_name":"token_readonly","binding_mode":"grant","managed_secret_ref_id":"secret-1"}',
                },
                {
                    "name": "policy_assignment.binding.delete",
                    "label": "Delete Assignment Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs"}',
                },
                {
                    "name": "policy_assignment.slot_binding.delete",
                    "label": "Delete Assignment Slot Binding",
                    "action_id": "mcp.credentials.configure.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "policy_assignment.slot_status.get",
                    "label": "Get Assignment Slot Status",
                    "action_id": "mcp.credentials.list.server",
                    "payload_template": '{"assignment_id":2,"server_id":"docs","slot_name":"token_readonly"}',
                },
                {
                    "name": "acp_profile.create",
                    "label": "Create ACP Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"name":"Workspace ACP","profile":{},"is_active":true}',
                },
                {
                    "name": "acp_profile.update",
                    "label": "Update ACP Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"profile_id":8,"name":"Updated ACP"}',
                },
                {
                    "name": "acp_profile.delete",
                    "label": "Delete ACP Profile",
                    "action_id": "mcp.governance.configure.server",
                    "payload_template": '{"profile_id":8}',
                },
            ]

        if (self.context.selected_section or "overview") == "advanced":
            actions = [
                {
                    "name": "governance_pack.dry_run",
                    "label": "Dry Run Governance Pack",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"pack":{"manifest":{"pack_id":"baseline","version":"1.0.0"}}}',
                },
                {
                    "name": "governance_pack.source.prepare",
                    "label": "Prepare Governance Pack Source",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"source":{"kind":"git","url":"git@example.com:trusted/repo.git","ref":"main"}}',
                },
                {
                    "name": "governance_pack.source.dry_run",
                    "label": "Dry Run Governance Pack Source",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"candidate_id":"cand-1"}',
                },
                {
                    "name": "governance_pack.check_updates",
                    "label": "Check Governance Pack Updates",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"governance_pack_id":81}',
                },
                {
                    "name": "governance_pack.prepare_upgrade_candidate",
                    "label": "Prepare Governance Pack Upgrade Candidate",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"governance_pack_id":81}',
                },
                {
                    "name": "governance_pack.dry_run_upgrade",
                    "label": "Dry Run Governance Pack Upgrade",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"source_governance_pack_id":81,"pack":{"manifest":{"pack_id":"baseline","version":"1.1.0"}},"planner_inputs_fingerprint":"planner-1","adapter_state_fingerprint":"adapter-1"}',
                },
                {
                    "name": "governance_pack.source.dry_run_upgrade",
                    "label": "Dry Run Governance Pack Source Upgrade",
                    "action_id": "mcp.advanced.trigger.server",
                    "payload_template": '{"candidate_id":"cand-1","source_governance_pack_id":81,"planner_inputs_fingerprint":"planner-1","adapter_state_fingerprint":"adapter-1"}',
                },
                {
                    "name": "governance_pack.import",
                    "label": "Import Governance Pack",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"pack":{"manifest":{"pack_id":"baseline","version":"1.0.0"}}}',
                },
                {
                    "name": "governance_pack.source.import",
                    "label": "Import Governance Pack Source",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"candidate_id":"cand-1"}',
                },
                {
                    "name": "governance_pack.source.execute_upgrade",
                    "label": "Execute Governance Pack Source Upgrade",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"candidate_id":"cand-1","source_governance_pack_id":81,"planner_inputs_fingerprint":"planner-1","adapter_state_fingerprint":"adapter-1"}',
                },
                {
                    "name": "governance_pack.execute_upgrade",
                    "label": "Execute Governance Pack Upgrade",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"source_governance_pack_id":81,"pack":{"manifest":{"pack_id":"baseline","version":"1.1.0"}},"planner_inputs_fingerprint":"planner-1","adapter_state_fingerprint":"adapter-1"}',
                },
                {
                    "name": "governance_pack.detail.get",
                    "label": "Get Governance Pack Detail",
                    "action_id": "mcp.advanced.observe.server",
                    "payload_template": '{"governance_pack_id":81}',
                },
                {
                    "name": "governance_pack.upgrade_history.list",
                    "label": "List Governance Pack Upgrade History",
                    "action_id": "mcp.advanced.observe.server",
                    "payload_template": '{"governance_pack_id":81}',
                },
                {
                    "name": "path_scope_object.create",
                    "label": "Create Path Scope Object",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"name":"Workspace Root","path_scope_document":{"path_scope_mode":"workspace_root"},"is_active":true}',
                },
                {
                    "name": "path_scope_object.update",
                    "label": "Update Path Scope Object",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"path_scope_object_id":5,"name":"Workspace Root Updated"}',
                },
                {
                    "name": "path_scope_object.delete",
                    "label": "Delete Path Scope Object",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"path_scope_object_id":5}',
                },
                {
                    "name": "workspace_set_object.create",
                    "label": "Create Workspace Set",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"name":"Research Set","description":"Trusted workspaces","is_active":true}',
                },
                {
                    "name": "workspace_set_object.update",
                    "label": "Update Workspace Set",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6,"description":"Updated"}',
                },
                {
                    "name": "workspace_set_object.delete",
                    "label": "Delete Workspace Set",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6}',
                },
                {
                    "name": "workspace_set_object.members.list",
                    "label": "List Workspace Set Members",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6}',
                },
                {
                    "name": "workspace_set_object.member.add",
                    "label": "Add Workspace Set Member",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6,"workspace_id":"ws-1"}',
                },
                {
                    "name": "workspace_set_object.member.delete",
                    "label": "Delete Workspace Set Member",
                    "action_id": "mcp.advanced.configure.server",
                    "payload_template": '{"workspace_set_object_id":6,"workspace_id":"ws-1"}',
                },
            ]
            if (self.context.selected_scope or "personal") in {
                "team",
                "org",
                "system_admin",
            }:
                actions[0:0] = [
                    {
                        "name": "capability_mapping.preview",
                        "label": "Preview Capability Mapping",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"mapping_id":"filesystem-write","capability_name":"filesystem.write","resolved_policy_document":{"allowed_tools":["filesystem.write"]},"is_active":true}',
                    },
                    {
                        "name": "capability_mapping.create",
                        "label": "Create Capability Mapping",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"mapping_id":"filesystem-write","title":"Filesystem Write","capability_name":"filesystem.write","resolved_policy_document":{"allowed_tools":["filesystem.write"]},"is_active":true}',
                    },
                    {
                        "name": "capability_mapping.update",
                        "label": "Update Capability Mapping",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"capability_adapter_mapping_id":3,"title":"Filesystem Write Updated"}',
                    },
                    {
                        "name": "capability_mapping.delete",
                        "label": "Delete Capability Mapping",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"capability_adapter_mapping_id":3}',
                    },
                    {
                        "name": "shared_workspace.create",
                        "label": "Create Shared Workspace",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"workspace_id":"shared-ws","display_name":"Shared Workspace","absolute_root":"/srv/shared","is_active":true}',
                    },
                    {
                        "name": "shared_workspace.update",
                        "label": "Update Shared Workspace",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"shared_workspace_id":7,"display_name":"Shared Workspace Updated"}',
                    },
                    {
                        "name": "shared_workspace.delete",
                        "label": "Delete Shared Workspace",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"shared_workspace_id":7}',
                    },
                ]
            if (self.context.selected_scope or "personal") == "system_admin":
                actions.insert(
                    0,
                    {
                        "name": "governance_pack_trust_policy.update",
                        "label": "Update Trust Policy",
                        "action_id": "mcp.advanced.configure.server",
                        "payload_template": '{"mode":"allowlist","allowed_sources":["git@example.com:trusted/repo.git"]}',
                    },
                )
            return actions

        return []

    def runtime_state_override(self) -> RuntimeSourceState:
        if self.context.selected_source != "server":
            return RuntimeSourceState(active_source="local")

        target = self._resolve_target(self.context.selected_active_server_id)
        access_context = None
        if target is not None:
            access_context = self.context.per_server_state.get(target.server_id)
        target_status = (
            access_context.target_status if access_context is not None else None
        )
        return RuntimeSourceState(
            active_source="server",
            active_server_id=(
                target.server_id
                if target is not None
                else self.context.selected_active_server_id
            ),
            server_configured=target is not None,
            server_reachability=(
                target_status.last_known_reachability
                if target_status is not None
                and target_status.last_known_reachability is not None
                else "reachable"
            ),
            server_auth_state=(
                target_status.last_known_auth_state
                if target_status is not None
                and target_status.last_known_auth_state is not None
                else "authenticated"
            ),
            last_known_server_label=(
                target_status.last_known_server_label
                if target_status is not None
                and target_status.last_known_server_label is not None
                else (target.label if target is not None else None)
            ),
        )

    async def run_action(
        self, action_name: str, payload: dict[str, Any] | None = None
    ) -> Any:
        payload = dict(payload or {})
        if self.context.selected_source != "server":
            if action_name == "profile.save":
                return await self._maybe_await(
                    self.local_service.save_external_profile(payload)
                )
            if action_name == "profile.delete":
                return await self._maybe_await(
                    self.local_service.delete_external_profile(
                        self._require_field(payload, "profile_id")
                    )
                )
            if action_name == "profile.connect":
                return await self._maybe_await(
                    self.local_service.connect_profile(
                        self._require_field(payload, "profile_id")
                    )
                )
            if action_name == "profile.disconnect":
                return await self._maybe_await(
                    self.local_service.disconnect_profile(
                        self._require_field(payload, "profile_id")
                    )
                )
            if action_name == "profile.test":
                return await self._maybe_await(
                    self.local_service.test_external_profile(
                        self._require_field(payload, "profile_id")
                    )
                )
            if action_name == "profile.refresh":
                return await self._maybe_await(
                    self.local_service.refresh_external_profile(
                        self._require_field(payload, "profile_id")
                    )
                )
            if action_name == "tool.execute":
                # Task 6 (PR-T3), Route B: NOT a bare `local_service.
                # execute_tool()` call anymore -- see
                # `execute_advanced_tool()` for why that route was the one
                # execution path in the Hub with neither a permission gate
                # nor an execution-log record.
                return await self.execute_advanced_tool(
                    self._require_field(payload, "tool_name"),
                    payload.get("arguments")
                    if isinstance(payload.get("arguments"), dict)
                    else {},
                )
            if action_name == "resource.read":
                return await self._maybe_await(
                    self.local_service.read_resource(
                        self._require_field(payload, "resource_uri")
                    )
                )
            if action_name == "prompt.get":
                return await self._maybe_await(
                    self.local_service.get_prompt(
                        self._require_field(payload, "prompt_name"),
                        payload.get("arguments")
                        if isinstance(payload.get("arguments"), dict)
                        else {},
                    )
                )
            if action_name == "governance_rule.save":
                return await self._maybe_await(
                    self.local_service.save_governance_rule(payload)
                )
            if action_name == "governance_rule.preview":
                return await self._maybe_await(
                    self.local_service.preview_governance_decision(
                        self._require_field(payload, "capability_id")
                    )
                )
            if action_name == "governance_rule.delete":
                return await self._maybe_await(
                    self.local_service.delete_governance_rule(
                        self._require_field(payload, "rule_id")
                    )
                )
            if action_name == "runtime.access.preview":
                nested_payload = (
                    payload.get("payload")
                    if isinstance(payload.get("payload"), dict)
                    else {}
                )
                return await self._maybe_await(
                    self.local_service.preview_runtime_access(
                        self._require_field(payload, "action_name"),
                        nested_payload,
                    )
                )
            if action_name == "runtime.activity.list":
                return await self._maybe_await(
                    self.local_service.get_runtime_activity(
                        int(payload.get("limit", 20))
                    )
                )
            if action_name == "runtime.protocol.inspect":
                return await self._maybe_await(
                    self.local_service.get_runtime_protocol_diagnostics()
                )
            if action_name == "runtime.health.get":
                return await self._maybe_await(self.local_service.get_runtime_health())
            if action_name == "approval_requests.list":
                status = payload.get("status")
                resolved_action_id = payload.get("resolved_action_id")
                return await self._maybe_await(
                    self.local_service.list_approval_requests(
                        str(status).strip() if status is not None else None,
                        str(resolved_action_id).strip()
                        if resolved_action_id is not None
                        else None,
                    )
                )
            if action_name == "approval_request.approve":
                return await self._maybe_await(
                    self.local_service.approve_approval_request(
                        self._require_field(payload, "request_id")
                    )
                )
            if action_name == "approval_request.deny":
                return await self._maybe_await(
                    self.local_service.deny_approval_request(
                        self._require_field(payload, "request_id")
                    )
                )
            if action_name == "approval_request.delete":
                return await self._maybe_await(
                    self.local_service.delete_approval_request(
                        self._require_field(payload, "request_id")
                    )
                )
            if action_name == "runtime.status.get":
                return await self._maybe_await(self.local_service.get_runtime_status())
            if action_name == "runtime.request":
                method = self._require_field(payload, "method")
                self._refuse_raw_tool_call(method)
                return await self._maybe_await(
                    self.local_service.run_runtime_request(
                        method,
                        payload.get("params")
                        if isinstance(payload.get("params"), dict)
                        else {},
                    )
                )
            if action_name == "runtime.batch":
                requests = payload.get("requests")
                if not isinstance(requests, list):
                    raise ValueError("Unified MCP action requires 'requests'.")
                # Checked BEFORE dispatching any of them: the batch runs
                # serially, so a `tools/call` in the middle would already
                # have executed by the time a per-item refusal could
                # report it, and would report as an ordinary batch row.
                #
                # Item 2 (PR-T3 fix round F): normalized ONCE via
                # `_normalize_batch_requests()` -- the same `dict(request)`
                # coercion `local_service.run_runtime_batch()` applies
                # (`local_control_service.py:500`) -- and that SAME
                # normalized list is what both this scan and the dispatch
                # below consume. Before this, the scan checked
                # `isinstance(request, Mapping)` against the RAW items
                # (widened from `dict` to `Mapping` in Fix Round A, Minor
                # #4) while the real dispatcher's `dict(request)` also
                # accepts a list of `(key, value)` pairs -- a `list` is
                # neither, so that shape skipped the scan silently and
                # still ran for real. See `_normalize_batch_requests()`'s
                # own docstring for the full mechanism. The delegate-level
                # refusal (`LocalMCPRuntimeDelegate.request()`) remains the
                # durable backstop for callers that reach it without going
                # through this scan at all -- this fix does not replace
                # that, it closes the gap that let this scan disagree with
                # the dispatcher it is supposed to pre-empt.
                normalized_requests = self._normalize_batch_requests(requests)
                for request in normalized_requests:
                    self._refuse_raw_tool_call(request.get("method"))
                return await self._maybe_await(
                    self.local_service.run_runtime_batch(normalized_requests)
                )
            raise ValueError(f"Unsupported Unified MCP local action: {action_name}")

        target = self._require_active_server_target()
        access_context = self.context.per_server_state.get(target.server_id)
        if access_context is None:
            raise RuntimeError("Failed to resolve Unified MCP server access context.")

        if action_name == "catalog.create":
            return await self._maybe_await(
                self.server_service.create_catalog(
                    target=target, access_context=access_context, payload=payload
                )
            )
        if action_name == "catalog.entry.create":
            catalog_id = self._require_field(payload, "catalog_id")
            entry_payload = {
                key: value for key, value in payload.items() if key != "catalog_id"
            }
            return await self._maybe_await(
                self.server_service.create_catalog_entry(
                    target=target,
                    access_context=access_context,
                    catalog_id=catalog_id,
                    payload=entry_payload,
                )
            )
        if action_name == "catalog.delete":
            return await self._maybe_await(
                self.server_service.delete_catalog(
                    target=target,
                    access_context=access_context,
                    catalog_id=self._require_field(payload, "catalog_id"),
                )
            )
        if action_name == "catalog.entry.delete":
            return await self._maybe_await(
                self.server_service.delete_catalog_entry(
                    target=target,
                    access_context=access_context,
                    catalog_id=self._require_field(payload, "catalog_id"),
                    tool_name=self._require_field(payload, "tool_name"),
                )
            )
        if action_name == "external_server.create":
            return await self._maybe_await(
                self.server_service.create_external_server(
                    target=target, access_context=access_context, payload=payload
                )
            )
        if action_name == "external_server.update":
            server_id = self._require_field(payload, "server_id")
            update_payload = {
                key: value for key, value in payload.items() if key != "server_id"
            }
            return await self._maybe_await(
                self.server_service.update_external_server(
                    target=target,
                    access_context=access_context,
                    server_id=server_id,
                    payload=update_payload,
                )
            )
        if action_name == "external_server.delete":
            return await self._maybe_await(
                self.server_service.delete_external_server(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "external_server.import":
            return await self._maybe_await(
                self.server_service.import_external_server(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "external_server.auth_template.update":
            server_id = self._require_field(payload, "server_id")
            update_payload = {
                key: value for key, value in payload.items() if key != "server_id"
            }
            return await self._maybe_await(
                self.server_service.update_external_server_auth_template(
                    target=target,
                    access_context=access_context,
                    server_id=server_id,
                    payload=update_payload,
                )
            )
        if action_name == "external_server.slots.list":
            return await self._maybe_await(
                self.server_service.list_external_server_credential_slots(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "external_server.slot.create":
            server_id = self._require_field(payload, "server_id")
            slot_payload = {
                key: value for key, value in payload.items() if key != "server_id"
            }
            return await self._maybe_await(
                self.server_service.create_external_server_credential_slot(
                    target=target,
                    access_context=access_context,
                    server_id=server_id,
                    payload=slot_payload,
                )
            )
        if action_name == "external_server.slot.update":
            server_id = self._require_field(payload, "server_id")
            slot_name = self._require_field(payload, "slot_name")
            slot_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"server_id", "slot_name"}
            }
            return await self._maybe_await(
                self.server_service.update_external_server_credential_slot(
                    target=target,
                    access_context=access_context,
                    server_id=server_id,
                    slot_name=slot_name,
                    payload=slot_payload,
                )
            )
        if action_name == "external_server.slot.delete":
            return await self._maybe_await(
                self.server_service.delete_external_server_credential_slot(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "external_server.slot.secret.set":
            return await self._maybe_await(
                self.server_service.set_external_server_slot_secret(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                    secret=self._require_field(payload, "secret"),
                )
            )
        if action_name == "external_server.slot.secret.clear":
            return await self._maybe_await(
                self.server_service.clear_external_server_slot_secret(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "external_server.secret.set":
            return await self._maybe_await(
                self.server_service.set_external_server_secret(
                    target=target,
                    access_context=access_context,
                    server_id=self._require_field(payload, "server_id"),
                    secret=self._require_field(payload, "secret"),
                )
            )
        if action_name == "permission_profile.create":
            return await self._maybe_await(
                self.server_service.create_permission_profile(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "permission_profile.update":
            profile_id = self._require_field(payload, "profile_id")
            update_payload = {
                key: value for key, value in payload.items() if key != "profile_id"
            }
            return await self._maybe_await(
                self.server_service.update_permission_profile(
                    target=target,
                    access_context=access_context,
                    profile_id=profile_id,
                    payload=update_payload,
                )
            )
        if action_name == "permission_profile.delete":
            return await self._maybe_await(
                self.server_service.delete_permission_profile(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                )
            )
        if action_name == "policy_assignment.create":
            return await self._maybe_await(
                self.server_service.create_policy_assignment(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "policy_assignment.update":
            assignment_id = self._require_field(payload, "assignment_id")
            update_payload = {
                key: value for key, value in payload.items() if key != "assignment_id"
            }
            return await self._maybe_await(
                self.server_service.update_policy_assignment(
                    target=target,
                    access_context=access_context,
                    assignment_id=assignment_id,
                    payload=update_payload,
                )
            )
        if action_name == "policy_assignment.delete":
            return await self._maybe_await(
                self.server_service.delete_policy_assignment(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.override.get":
            return await self._maybe_await(
                self.server_service.get_policy_assignment_override(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.override.upsert":
            assignment_id = self._require_field(payload, "assignment_id")
            override_payload = {
                key: value for key, value in payload.items() if key != "assignment_id"
            }
            return await self._maybe_await(
                self.server_service.upsert_policy_assignment_override(
                    target=target,
                    access_context=access_context,
                    assignment_id=assignment_id,
                    payload=override_payload,
                )
            )
        if action_name == "policy_assignment.override.delete":
            return await self._maybe_await(
                self.server_service.delete_policy_assignment_override(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "approval_policy.create":
            return await self._maybe_await(
                self.server_service.create_approval_policy(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "approval_policy.update":
            approval_policy_id = self._require_field(payload, "approval_policy_id")
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "approval_policy_id"
            }
            return await self._maybe_await(
                self.server_service.update_approval_policy(
                    target=target,
                    access_context=access_context,
                    approval_policy_id=approval_policy_id,
                    payload=update_payload,
                )
            )
        if action_name == "approval_policy.delete":
            return await self._maybe_await(
                self.server_service.delete_approval_policy(
                    target=target,
                    access_context=access_context,
                    approval_policy_id=self._require_field(
                        payload, "approval_policy_id"
                    ),
                )
            )
        if action_name == "approval_decision.create":
            return await self._maybe_await(
                self.server_service.create_approval_decision(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "policy_assignment.external_access.get":
            return await self._maybe_await(
                self.server_service.get_assignment_external_access(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.workspaces.list":
            return await self._maybe_await(
                self.server_service.list_policy_assignment_workspaces(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.workspace.add":
            return await self._maybe_await(
                self.server_service.add_policy_assignment_workspace(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    workspace_id=self._require_field(payload, "workspace_id"),
                )
            )
        if action_name == "policy_assignment.workspace.delete":
            return await self._maybe_await(
                self.server_service.delete_policy_assignment_workspace(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    workspace_id=self._require_field(payload, "workspace_id"),
                )
            )
        if action_name == "permission_profile.bindings.list":
            return await self._maybe_await(
                self.server_service.list_profile_credential_bindings(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                )
            )
        if action_name == "permission_profile.binding.upsert":
            profile_id = self._require_field(payload, "profile_id")
            server_id = self._require_field(payload, "server_id")
            binding_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"profile_id", "server_id"}
            }
            return await self._maybe_await(
                self.server_service.upsert_profile_credential_binding(
                    target=target,
                    access_context=access_context,
                    profile_id=profile_id,
                    server_id=server_id,
                    payload=binding_payload,
                )
            )
        if action_name == "permission_profile.slot_binding.upsert":
            profile_id = self._require_field(payload, "profile_id")
            server_id = self._require_field(payload, "server_id")
            slot_name = self._require_field(payload, "slot_name")
            binding_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"profile_id", "server_id", "slot_name"}
            }
            return await self._maybe_await(
                self.server_service.upsert_profile_credential_binding(
                    target=target,
                    access_context=access_context,
                    profile_id=profile_id,
                    server_id=server_id,
                    slot_name=slot_name,
                    payload=binding_payload,
                )
            )
        if action_name == "permission_profile.binding.delete":
            return await self._maybe_await(
                self.server_service.delete_profile_credential_binding(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "permission_profile.slot_binding.delete":
            return await self._maybe_await(
                self.server_service.delete_profile_credential_binding(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "permission_profile.slot_status.get":
            return await self._maybe_await(
                self.server_service.get_profile_slot_credential_status(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "policy_assignment.bindings.list":
            return await self._maybe_await(
                self.server_service.list_assignment_credential_bindings(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                )
            )
        if action_name == "policy_assignment.binding.upsert":
            assignment_id = self._require_field(payload, "assignment_id")
            server_id = self._require_field(payload, "server_id")
            binding_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"assignment_id", "server_id"}
            }
            return await self._maybe_await(
                self.server_service.upsert_assignment_credential_binding(
                    target=target,
                    access_context=access_context,
                    assignment_id=assignment_id,
                    server_id=server_id,
                    payload=binding_payload,
                )
            )
        if action_name == "policy_assignment.slot_binding.upsert":
            assignment_id = self._require_field(payload, "assignment_id")
            server_id = self._require_field(payload, "server_id")
            slot_name = self._require_field(payload, "slot_name")
            binding_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"assignment_id", "server_id", "slot_name"}
            }
            return await self._maybe_await(
                self.server_service.upsert_assignment_credential_binding(
                    target=target,
                    access_context=access_context,
                    assignment_id=assignment_id,
                    server_id=server_id,
                    slot_name=slot_name,
                    payload=binding_payload,
                )
            )
        if action_name == "policy_assignment.binding.delete":
            return await self._maybe_await(
                self.server_service.delete_assignment_credential_binding(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    server_id=self._require_field(payload, "server_id"),
                )
            )
        if action_name == "policy_assignment.slot_binding.delete":
            return await self._maybe_await(
                self.server_service.delete_assignment_credential_binding(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "policy_assignment.slot_status.get":
            return await self._maybe_await(
                self.server_service.get_assignment_slot_credential_status(
                    target=target,
                    access_context=access_context,
                    assignment_id=self._require_field(payload, "assignment_id"),
                    server_id=self._require_field(payload, "server_id"),
                    slot_name=self._require_field(payload, "slot_name"),
                )
            )
        if action_name == "acp_profile.create":
            return await self._maybe_await(
                self.server_service.create_acp_profile(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "acp_profile.update":
            profile_id = self._require_field(payload, "profile_id")
            update_payload = {
                key: value for key, value in payload.items() if key != "profile_id"
            }
            return await self._maybe_await(
                self.server_service.update_acp_profile(
                    target=target,
                    access_context=access_context,
                    profile_id=profile_id,
                    payload=update_payload,
                )
            )
        if action_name == "acp_profile.delete":
            return await self._maybe_await(
                self.server_service.delete_acp_profile(
                    target=target,
                    access_context=access_context,
                    profile_id=self._require_field(payload, "profile_id"),
                )
            )
        if action_name == "governance_pack_trust_policy.update":
            return await self._maybe_await(
                self.server_service.update_governance_pack_trust_policy(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.dry_run":
            return await self._maybe_await(
                self.server_service.dry_run_governance_pack(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.prepare":
            return await self._maybe_await(
                self.server_service.prepare_governance_pack_source(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.dry_run":
            return await self._maybe_await(
                self.server_service.dry_run_governance_pack_source(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.check_updates":
            return await self._maybe_await(
                self.server_service.check_governance_pack_updates(
                    target=target,
                    access_context=access_context,
                    governance_pack_id=self._require_field(
                        payload, "governance_pack_id"
                    ),
                )
            )
        if action_name == "governance_pack.prepare_upgrade_candidate":
            return await self._maybe_await(
                self.server_service.prepare_governance_pack_upgrade_candidate(
                    target=target,
                    access_context=access_context,
                    governance_pack_id=self._require_field(
                        payload, "governance_pack_id"
                    ),
                )
            )
        if action_name == "governance_pack.dry_run_upgrade":
            return await self._maybe_await(
                self.server_service.dry_run_governance_pack_upgrade(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.dry_run_upgrade":
            return await self._maybe_await(
                self.server_service.dry_run_governance_pack_source_upgrade(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.import":
            return await self._maybe_await(
                self.server_service.import_governance_pack(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.import":
            return await self._maybe_await(
                self.server_service.import_governance_pack_source(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.source.execute_upgrade":
            return await self._maybe_await(
                self.server_service.execute_governance_pack_source_upgrade(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.execute_upgrade":
            return await self._maybe_await(
                self.server_service.execute_governance_pack_upgrade(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "governance_pack.detail.get":
            return await self._maybe_await(
                self.server_service.get_governance_pack_detail(
                    target=target,
                    access_context=access_context,
                    governance_pack_id=self._require_field(
                        payload, "governance_pack_id"
                    ),
                )
            )
        if action_name == "governance_pack.upgrade_history.list":
            return await self._maybe_await(
                self.server_service.list_governance_pack_upgrade_history(
                    target=target,
                    access_context=access_context,
                    governance_pack_id=self._require_field(
                        payload, "governance_pack_id"
                    ),
                )
            )
        if action_name == "path_scope_object.create":
            return await self._maybe_await(
                self.server_service.create_path_scope_object(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "path_scope_object.update":
            path_scope_object_id = self._require_field(payload, "path_scope_object_id")
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "path_scope_object_id"
            }
            return await self._maybe_await(
                self.server_service.update_path_scope_object(
                    target=target,
                    access_context=access_context,
                    path_scope_object_id=path_scope_object_id,
                    payload=update_payload,
                )
            )
        if action_name == "path_scope_object.delete":
            return await self._maybe_await(
                self.server_service.delete_path_scope_object(
                    target=target,
                    access_context=access_context,
                    path_scope_object_id=self._require_field(
                        payload, "path_scope_object_id"
                    ),
                )
            )
        if action_name == "capability_mapping.preview":
            return await self._maybe_await(
                self.server_service.preview_capability_mapping(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "capability_mapping.create":
            return await self._maybe_await(
                self.server_service.create_capability_mapping(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "capability_mapping.update":
            capability_adapter_mapping_id = self._require_field(
                payload, "capability_adapter_mapping_id"
            )
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "capability_adapter_mapping_id"
            }
            return await self._maybe_await(
                self.server_service.update_capability_mapping(
                    target=target,
                    access_context=access_context,
                    capability_adapter_mapping_id=capability_adapter_mapping_id,
                    payload=update_payload,
                )
            )
        if action_name == "capability_mapping.delete":
            return await self._maybe_await(
                self.server_service.delete_capability_mapping(
                    target=target,
                    access_context=access_context,
                    capability_adapter_mapping_id=self._require_field(
                        payload, "capability_adapter_mapping_id"
                    ),
                )
            )
        if action_name == "workspace_set_object.create":
            return await self._maybe_await(
                self.server_service.create_workspace_set_object(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "workspace_set_object.update":
            workspace_set_object_id = self._require_field(
                payload, "workspace_set_object_id"
            )
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "workspace_set_object_id"
            }
            return await self._maybe_await(
                self.server_service.update_workspace_set_object(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=workspace_set_object_id,
                    payload=update_payload,
                )
            )
        if action_name == "workspace_set_object.delete":
            return await self._maybe_await(
                self.server_service.delete_workspace_set_object(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=self._require_field(
                        payload, "workspace_set_object_id"
                    ),
                )
            )
        if action_name == "workspace_set_object.members.list":
            return await self._maybe_await(
                self.server_service.list_workspace_set_members(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=self._require_field(
                        payload, "workspace_set_object_id"
                    ),
                )
            )
        if action_name == "workspace_set_object.member.add":
            workspace_set_object_id = self._require_field(
                payload, "workspace_set_object_id"
            )
            member_payload = {
                key: value
                for key, value in payload.items()
                if key != "workspace_set_object_id"
            }
            return await self._maybe_await(
                self.server_service.add_workspace_set_member(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=workspace_set_object_id,
                    payload=member_payload,
                )
            )
        if action_name == "workspace_set_object.member.delete":
            return await self._maybe_await(
                self.server_service.delete_workspace_set_member(
                    target=target,
                    access_context=access_context,
                    workspace_set_object_id=self._require_field(
                        payload, "workspace_set_object_id"
                    ),
                    workspace_id=self._require_field(payload, "workspace_id"),
                )
            )
        if action_name == "shared_workspace.create":
            return await self._maybe_await(
                self.server_service.create_shared_workspace(
                    target=target,
                    access_context=access_context,
                    payload=payload,
                )
            )
        if action_name == "shared_workspace.update":
            shared_workspace_id = self._require_field(payload, "shared_workspace_id")
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "shared_workspace_id"
            }
            return await self._maybe_await(
                self.server_service.update_shared_workspace(
                    target=target,
                    access_context=access_context,
                    shared_workspace_id=shared_workspace_id,
                    payload=update_payload,
                )
            )
        if action_name == "shared_workspace.delete":
            return await self._maybe_await(
                self.server_service.delete_shared_workspace(
                    target=target,
                    access_context=access_context,
                    shared_workspace_id=self._require_field(
                        payload, "shared_workspace_id"
                    ),
                )
            )
        raise ValueError(f"Unsupported Unified MCP server action: {action_name}")

    def _apply_server_access_context(
        self,
        server_id: str,
        access_context: ServerAccessContext,
    ) -> None:
        per_server_state = dict(self.context.per_server_state)
        per_server_state[server_id] = access_context
        self.context = replace(
            self.context,
            selected_source="server",
            selected_active_server_id=server_id,
            selected_scope=access_context.selected_scope,
            selected_scope_ref=access_context.selected_scope_ref,
            selected_section=access_context.selected_section,
            per_server_state=per_server_state,
        )
        self._persist_context()

    def _persist_context(self) -> None:
        if self.context_store is not None:
            self.context_store.save(self.context)

    def _resolve_target(self, server_id: str | None) -> Any:
        if self.target_store is None:
            return None
        return self.target_store.resolve_active_target(server_id)

    def _require_active_server_target(self) -> Any:
        target = self._resolve_target(self.context.selected_active_server_id)
        if target is None:
            raise ValueError("No active Unified MCP server target is selected.")
        return target

    @staticmethod
    def _require_field(payload: dict[str, Any], field_name: str) -> Any:
        value = payload.get(field_name)
        if value in (None, ""):
            raise ValueError(f"Unified MCP action requires '{field_name}'.")
        return value

    # ---- Typed local lifecycle/mutation seam (Phase 2) ----------------------
    # Shared by the Hub UI now and by the Phase 5 chat bridge / agent-runtime
    # MCPToolProvider (task-201) later. Governance enforcement stays inside
    # the local service exactly as run_action's branches rely on it.

    def _lifecycle_timeout(self) -> float:
        try:
            return float(get_cli_setting("mcp", "hub_lifecycle_timeout_seconds", 45))
        except (TypeError, ValueError):
            return 45.0

    def _record_local_attempt(
        self, profile_id: str, action: str, *, ok: bool, error: str | None
    ) -> None:
        store = getattr(self.local_service, "store", None)
        if store is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            previous = store.get_profile_runtime_state(profile_id) or {}
            store.save_profile_runtime_state(
                profile_id,
                {
                    "last_attempt_at": now,
                    "last_action": action,
                    "ok": ok,
                    "last_ok_at": now if ok else previous.get("last_ok_at"),
                    "last_error": None if ok else (error or "")[:300],
                },
            )
        except Exception as exc:
            # Recording is best-effort: it must never mask the lifecycle
            # result or the original exception being propagated.
            logger.warning(
                f"MCP lifecycle attempt record failed for {profile_id}: {exc}"
            )

    async def _run_local_lifecycle(self, action: str, profile_id: str, coro):
        timeout = self._lifecycle_timeout()
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            message = f"Timed out after {timeout:.0f}s"
            self._record_local_attempt(profile_id, action, ok=False, error=message)
            raise RuntimeError(message) from None
        except asyncio.CancelledError:
            self._record_local_attempt(profile_id, action, ok=False, error="Cancelled")
            raise
        except Exception as exc:
            self._record_local_attempt(profile_id, action, ok=False, error=str(exc))
            raise
        self._record_local_attempt(profile_id, action, ok=True, error=None)
        return result

    async def connect_local_profile(self, profile_id: str) -> dict:
        return await self._run_local_lifecycle(
            "connect", profile_id, self.local_service.connect_profile(profile_id)
        )

    async def disconnect_local_profile(self, profile_id: str) -> bool:
        return await self._run_local_lifecycle(
            "disconnect", profile_id, self.local_service.disconnect_profile(profile_id)
        )

    async def test_local_profile(self, profile_id: str) -> dict:
        return await self._run_local_lifecycle(
            "test", profile_id, self.local_service.test_external_profile(profile_id)
        )

    async def refresh_local_profile(self, profile_id: str) -> dict:
        return await self._run_local_lifecycle(
            "refresh",
            profile_id,
            self.local_service.refresh_external_profile(profile_id),
        )

    async def save_local_profile(self, payload: dict) -> dict:
        return self.local_service.save_external_profile(dict(payload or {}))

    async def delete_local_profile(self, profile_id: str) -> bool:
        return bool(self.local_service.delete_external_profile(profile_id))

    async def local_external_catalog(self) -> list[dict]:
        # Records (profile fields + discovery_snapshot + is_connected) still
        # come from the local service so governance enforcement and
        # is_connected (read from the live client sessions) are unchanged.
        # `runtime_state` is merged in from a single store bundle load
        # rather than one `get_profile_runtime_state()` load per record.
        records = list(self.local_service.get_external_servers() or [])
        store = getattr(self.local_service, "store", None)
        runtime_state_by_profile: dict[str, Any] = (
            store.get_catalog_bundle()["profile_runtime_state"] if store else {}
        )
        for record in records:
            profile_id = str(record.get("profile_id") or "")
            record["runtime_state"] = runtime_state_by_profile.get(profile_id)
        return records

    # ---- Typed tool-execution seam (Phase 3) ---------------------------
    # Shared by the Hub Tools mode now and by the Phase 5 chat bridge /
    # agent-runtime MCPToolProvider (task-201) later. Keep this UI-free.

    @property
    def execution_log(self) -> MCPExecutionLog | None:
        if self._execution_log is not None:
            return self._execution_log
        store = getattr(self.local_service, "store", None)
        if store is None:
            return None
        log_path = Path(store.path).with_name("mcp_execution_log.jsonl")
        with self._execution_log_init_lock:
            if self._execution_log is None:
                self._execution_log = MCPExecutionLog(log_path)
            return self._execution_log

    def _record_tool_execution(
        self,
        server_key: str,
        tool_name: str,
        *,
        ok: bool,
        duration_ms: int,
        status: str,
        error_category: str | None,
        exception_type: str | None,
        status_code: int | None,
        arguments: dict[str, Any],
        registered_argument_names: set[str] | None,
        result: Any,
        initiator: str = "test",
        decision: str = "allowed",
    ) -> None:
        # Recording is best-effort: it must never mask the tool result or
        # the tool error being propagated (Phase 2 masking lesson). N1: the
        # `self.execution_log` property access itself must be inside this
        # try too -- it can raise (e.g. `Path(store.path)` oddities), and
        # sitting outside would let that raise straight out of
        # `_record_tool_execution()` into the caller's own try/except
        # around test_hub_tool()'s / execute_hub_tool()'s success/failure
        # paths, masking the tool result exactly like an append() failure
        # would.
        #
        # `initiator`/`decision` default to "test"/"allowed" -- the values
        # `test_hub_tool()` has always recorded -- so callers that don't
        # pass them (there are none left in this module, but external
        # callers via reflection/monkeypatching should not break) keep the
        # original byte-compatible record shape.
        try:
            log = self.execution_log
            if log is None:
                return
            record = build_record(
                server_key=server_key,
                tool_name=tool_name,
                initiator=initiator,
                decision=decision,
                ok=ok,
                status=status,
                duration_ms=duration_ms,
                error_category=error_category,
                exception_type=exception_type,
                status_code=status_code,
                arguments=arguments,
                registered_argument_names=registered_argument_names,
                result=result,
            )
            log.append(record)
        except Exception as exc:
            logger.warning(
                "MCP execution log record failed (exception_type={})",
                type(exc).__name__,
            )

    async def execute_hub_tool(
        self,
        server_key: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        initiator: str = "test",
        decision: str = "allowed",
        timeout_seconds: float | None = None,
        registered_argument_names: set[str] | None = None,
    ) -> dict[str, Any]:
        """Execute one tool call against a local or built-in server.

        The shared execute seam for the Hub's Test Tool runner and the
        Phase 5 chat bridge / agent-runtime ``MCPToolProvider``. Same
        ``local:``/``builtin:`` routing and error semantics as the
        original ``test_hub_tool()`` body this generalizes; callers
        distinguish themselves via ``initiator``/``decision`` so the
        execution-log record reflects who ran the tool and under what
        permission decision. Every attempt — success, failure, or
        timeout — is recorded to the execution log best-effort before the
        result or error propagates. Test Tool cancellation is recorded here;
        agent-bridge cancellation is re-raised for that bridge's outer owner
        to record once.

        Args:
            server_key: Prefixed server key (``local:<profile_id>`` or
                ``builtin:<id>``). Server-source keys are display-only
                and rejected here.
            tool_name: Name of the tool to execute.
            arguments: Tool arguments; defaults to an empty dict.
            initiator: Who initiated the call, recorded on the execution
                log (e.g. ``"test"`` for the Hub UI, ``"agent"`` for the
                chat bridge).
            decision: The permission decision under which the call ran
                (e.g. ``"allowed"``, ``"approved"``, or
                ``"approved-session"``), recorded on the execution log.
            timeout_seconds: Per-call timeout override; defaults to
                :meth:`_tool_call_timeout` (``[mcp]
                tool_call_timeout_seconds``) when omitted.
            registered_argument_names: Optional schema-approved argument
                names. Values are never persisted; supplied unknown names are
                counted.

        Returns:
            The raw result payload from the underlying service call.

        Raises:
            MCPServerSourceDisplayOnlyError: If ``server_key`` is not a
                local/builtin key. A ``ValueError`` subclass -- any
                existing ``except ValueError`` handler still catches it.
            MCPGovernanceDenied: If the in-process runtime-governance
                profile denies the call (raised inside ``coro`` by
                ``local_control_service._require_runtime_governance_
                allowed()``). Recorded honestly (item 1, fix round D) and
                re-raised -- a ``PermissionError`` subclass, so any
                existing ``except PermissionError`` handler upstream still
                catches it.
            RuntimeError: If the tool call fails or exceeds the
                effective timeout.
        """
        normalized_key = str(server_key or "").strip()
        normalized_tool_name = str(tool_name or "").strip()
        normalized_arguments = dict(arguments or {})

        if normalized_key.startswith("local:"):
            profile_id = normalized_key.split(":", 1)[1]
            coro = self.local_service.execute_external_tool(
                profile_id, normalized_tool_name, normalized_arguments
            )
        elif normalized_key.startswith("builtin:"):
            coro = self.local_service.execute_tool(
                normalized_tool_name, normalized_arguments
            )
        else:
            # task-2539: typed, not a bare `ValueError` -- see
            # `MCPServerSourceDisplayOnlyError`'s own docstring for why.
            raise MCPServerSourceDisplayOnlyError()

        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._tool_call_timeout()
        )
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            duration_ms = int((time.monotonic() - started) * 1000)
            message = f"Timed out after {timeout:.0f}s"
            self._record_tool_execution(
                normalized_key,
                normalized_tool_name,
                ok=False,
                duration_ms=duration_ms,
                status="timeout",
                error_category="timeout",
                exception_type="TimeoutError",
                status_code=None,
                arguments=normalized_arguments,
                registered_argument_names=registered_argument_names,
                result=None,
                initiator=initiator,
                decision=decision,
            )
            raise RuntimeError(message) from None
        except asyncio.CancelledError:
            if initiator == "test":
                duration_ms = int((time.monotonic() - started) * 1000)
                self._record_tool_execution(
                    normalized_key,
                    normalized_tool_name,
                    ok=False,
                    duration_ms=duration_ms,
                    status="cancelled",
                    error_category="cancelled",
                    exception_type="CancelledError",
                    status_code=None,
                    arguments=normalized_arguments,
                    registered_argument_names=registered_argument_names,
                    result=None,
                    initiator=initiator,
                    decision=decision,
                )
            raise
        except MCPGovernanceDenied as exc:
            # Item 1 (PR-T3 fix round D). Without this branch, a governance
            # refusal fell into the generic `except Exception` below and was
            # recorded as three false statements about one event: `status=
            # "error"`/`error_category="execution_failed"` (the tool RAN and
            # crashed -- it never ran at all), `duration_ms` measured from
            # `started` (timing a call that never dispatched), and the
            # caller's PRE-COMPUTED `decision` left untouched (describes
            # what the caller expected going in, not what happened -- the
            # governance check that just denied it runs INSIDE `coro`,
            # after this method already committed to that value). Recorded
            # honestly instead, reusing `record_tool_decision()`'s own
            # "never executed" vocabulary (`status="blocked"`,
            # `duration_ms=0`) rather than this method's own "attempted and
            # timed" shape.
            #
            # `error_category="governance_denied"` is a NEW token, not Fix
            # Round B's `"gate_denied"` (`execute_advanced_tool()`'s deny
            # branch, a few hundred lines down): that token names the Hub's
            # OWN Allow/Ask/Off permission gate. This denial comes from the
            # in-process runtime-governance profile
            # (`local_control_service._require_runtime_governance_
            # allowed()`) -- a different system, checked at a different
            # seam, for a different reason -- and reusing "gate_denied"
            # here would conflate the two exactly the way Round B's own
            # docstring warned "policy_denied" would falsely cross-
            # reference the unrelated `runtime_policy` engine's
            # `PolicyDeniedError`. "governance_denied" mirrors this
            # exception's own name (`MCPGovernanceDenied`) instead.
            self._record_tool_execution(
                normalized_key,
                normalized_tool_name,
                ok=False,
                duration_ms=0,
                status="blocked",
                error_category="governance_denied",
                exception_type=type(exc).__name__,
                status_code=None,
                arguments=normalized_arguments,
                registered_argument_names=registered_argument_names,
                result=None,
                initiator=initiator,
                decision="denied",
            )
            raise
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            response = getattr(exc, "response", None)
            raw_status_code = getattr(response, "status_code", None)
            if raw_status_code is None:
                raw_status_code = getattr(exc, "status_code", None)
            try:
                status_code = (
                    int(raw_status_code) if raw_status_code is not None else None
                )
            except (TypeError, ValueError):
                status_code = None
            self._record_tool_execution(
                normalized_key,
                normalized_tool_name,
                ok=False,
                duration_ms=duration_ms,
                status="http_error" if status_code is not None else "error",
                error_category=(
                    "http_error" if status_code is not None else "execution_failed"
                ),
                exception_type=type(exc).__name__,
                status_code=status_code,
                arguments=normalized_arguments,
                registered_argument_names=registered_argument_names,
                result=None,
                initiator=initiator,
                decision=decision,
            )
            raise

        duration_ms = int((time.monotonic() - started) * 1000)
        self._record_tool_execution(
            normalized_key,
            normalized_tool_name,
            ok=True,
            duration_ms=duration_ms,
            status="success",
            error_category=None,
            exception_type=None,
            status_code=None,
            arguments=normalized_arguments,
            registered_argument_names=registered_argument_names,
            result=result,
            initiator=initiator,
            decision=decision,
        )
        return result

    async def test_hub_tool(
        self,
        server_key: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        decision: str = "allowed",
        registered_argument_names: set[str] | None = None,
    ) -> dict[str, Any]:
        """Execute one tool test against a local or built-in server.

        Thin delegate to :meth:`execute_hub_tool` fixed to the Hub Test
        Tool runner's semantics: ``initiator="test"`` and the lifecycle
        timeout (``[mcp] hub_lifecycle_timeout_seconds`` via
        :meth:`_lifecycle_timeout`) rather than the chat-bridge's
        per-call timeout knob -- preserved unchanged so existing callers
        and their pinned tests keep seeing identical behavior.

        Args:
            server_key: Prefixed server key (``local:<profile_id>`` or
                ``builtin:<id>``). Server-source keys are display-only
                and rejected here.
            tool_name: Name of the tool to execute.
            arguments: Tool arguments; defaults to an empty dict.
            decision: The permission decision under which this test run
                dispatched, recorded on the execution log (RAG-51). The
                Hub UI passes ``"approved"`` for an Ask-gated tool the
                user just confirmed; every other caller keeps the default
                ``"allowed"`` this method has always recorded.
            registered_argument_names: Optional schema-approved argument
                names, forwarded unchanged to :meth:`execute_hub_tool`
                (Task 4, PR-T3). Omitted callers keep recording every
                supplied argument as unknown -- byte-identical to
                pre-Task-4 behavior.

        Returns:
            The raw result payload from the underlying service call.

        Raises:
            MCPServerSourceDisplayOnlyError: If ``server_key`` is not a
                local/builtin key. A ``ValueError`` subclass -- any
                existing ``except ValueError`` handler still catches it.
            RuntimeError: If the tool call fails or exceeds the
                configured lifecycle timeout.
        """
        return await self.execute_hub_tool(
            server_key,
            tool_name,
            arguments,
            initiator="test",
            decision=decision,
            timeout_seconds=self._lifecycle_timeout(),
            registered_argument_names=registered_argument_names,
        )

    def prepare_hub_test(self, tool: HubTool) -> ToolTestAdmissionPreview:
        """Resolve and register an immutable preview for one Hub test panel.

        The caller's ``HubTool`` supplies only the exact catalog identity. The
        definition, gate, eligibility, and workspace authority all come from a
        fresh service-owned resolution so a stale UI row cannot mint authority.
        """
        resolved = self._resolve_hub_test(tool.server_key, tool.name)
        if resolved is None:
            raise KeyError(f"Unknown Hub tool: {tool.tool_id}")
        return self._issue_hub_test_preview(resolved)

    def revoke_hub_test_preview(self, nonce: str) -> None:
        """Revoke one prepared Hub test nonce if it is still live."""
        if self._hub_test_previews is not None:
            self._hub_test_previews.revoke(str(nonce or ""))

    async def execute_prepared_hub_test(
        self,
        nonce: str,
        intent: Literal["run", "approve_once"],
        arguments: dict[str, Any],
    ) -> (
        dict[str, Any]
        | LocalHubExecutionOutcome
        | ToolTestAdmissionBlocked
        | ToolTestAdmissionStale
    ):
        """Atomically admit one preview-bound Hub Test Tool invocation.

        Argument validation deliberately precedes nonce consumption. Everything
        after consumption is re-resolved from live service state and compared to
        the immutable preview before either the legacy MCP seam or the private
        local-Hub seam can be reached.
        """
        from .hub_test_execution import (
            ToolTestAdmissionBlocked,
            ToolTestAdmissionStale,
            canonicalize_arguments,
        )

        canonical_bytes, dispatch_arguments = canonicalize_arguments(arguments)
        previews, _execution = self._ensure_hub_test_state()
        registered = previews.consume(str(nonce or ""))
        if registered is None:
            return ToolTestAdmissionStale(reason="preview_unavailable")

        public = registered.public
        if intent not in {"run", "approve_once"}:
            if public.server_key == "local:__local__":
                refresh_task = asyncio.create_task(
                    asyncio.to_thread(self._refresh_hub_test_preview, public)
                )
                cancelled: asyncio.CancelledError | None = None
                while not refresh_task.done():
                    try:
                        await asyncio.shield(refresh_task)
                    except asyncio.CancelledError as exc:
                        if refresh_task.cancelled():
                            raise
                        if cancelled is None:
                            cancelled = exc
                refreshed = refresh_task.result()
                if cancelled is not None and refreshed is not None:
                    self.revoke_hub_test_preview(refreshed.nonce)
                result = ToolTestAdmissionBlocked(
                    reason="intent_invalid", refreshed_preview=refreshed
                )
                try:
                    await self._attempt_local_hub_audit(
                        lambda: self._record_prepared_hub_block(public, result.reason),
                        "Local Hub intent review",
                    )
                except asyncio.CancelledError as exc:
                    if cancelled is None:
                        cancelled = exc
                    if refreshed is not None:
                        self.revoke_hub_test_preview(refreshed.nonce)
                if cancelled is not None:
                    raise cancelled
                return result
            return self._hub_test_blocked(
                public,
                reason="intent_invalid",
                refreshed=self._refresh_hub_test_preview(public),
            )

        if public.server_key == "local:__local__":
            return await self._execute_owned_prepared_local_hub_test(
                registered=registered,
                intent=intent,
                canonical_arguments=canonical_bytes,
                arguments=dispatch_arguments,
            )

        resolved = self._resolve_hub_test(public.server_key, public.tool_name)
        if resolved is None:
            return self._hub_test_stale(public, reason="identity_changed")

        fresh_preview = self._preview_fields(resolved)
        if (
            fresh_preview.server_key != public.server_key
            or fresh_preview.tool_name != public.tool_name
        ):
            return self._hub_test_stale(
                public,
                reason="identity_changed",
                refreshed=self._issue_hub_test_preview(resolved),
            )
        rendered_gate = public.rendered_gate
        fresh_gate = fresh_preview.rendered_gate
        if rendered_gate not in {"allow", "ask"}:
            return self._hub_test_blocked(
                public,
                reason=(
                    "permission_unresolved"
                    if rendered_gate in {"unavailable", "unresolved"}
                    else "permission_denied"
                ),
                refreshed=self._issue_hub_test_preview(resolved),
            )
        if resolved.unavailable_reason is not None:
            return self._hub_test_stale(
                public,
                reason=resolved.unavailable_reason,
                refreshed=self._issue_hub_test_preview(resolved),
            )

        if fresh_preview.definition_hash != public.definition_hash:
            return self._hub_test_stale(
                public,
                reason="definition_changed",
                refreshed=self._issue_hub_test_preview(resolved),
            )
        if (
            fresh_preview.authority_fingerprint != public.authority_fingerprint
            or resolved.authority != registered.authority
            or fresh_preview.safe_authority_label != public.safe_authority_label
        ):
            return self._hub_test_stale(
                public,
                reason="authority_changed",
                refreshed=self._issue_hub_test_preview(resolved),
            )

        if rendered_gate == "allow":
            if intent != "run":
                return self._hub_test_blocked(
                    public,
                    reason="intent_mismatch",
                    refreshed=self._issue_hub_test_preview(resolved),
                )
            if fresh_gate != "allow":
                return self._hub_test_stale(
                    public,
                    reason="gate_changed",
                    refreshed=self._issue_hub_test_preview(resolved),
                )
        elif rendered_gate == "ask":
            if intent != "approve_once":
                return self._hub_test_blocked(
                    public,
                    reason="intent_mismatch",
                    refreshed=self._issue_hub_test_preview(resolved),
                )
            if fresh_gate not in {"ask", "allow"}:
                return self._hub_test_stale(
                    public,
                    reason="gate_changed",
                    refreshed=self._issue_hub_test_preview(resolved),
                )

        # Re-encode the independent object immediately before dispatch. This
        # turns any accidental internal mutation into a closed refusal instead
        # of dispatching arguments different from the admitted canonical bytes.
        dispatch_bytes, dispatch_copy = canonicalize_arguments(dispatch_arguments)
        if dispatch_bytes != canonical_bytes:
            return self._hub_test_blocked(
                public,
                reason="arguments_changed",
                refreshed=self._issue_hub_test_preview(resolved),
            )

        decision = "approved" if fresh_gate == "ask" else "allowed"
        return await self.test_hub_tool(
            resolved.tool.server_key,
            resolved.tool.name,
            dispatch_copy,
            decision=decision,
            registered_argument_names=schema_argument_names(resolved.tool.input_schema),
        )

    @staticmethod
    async def _attempt_local_hub_audit(
        callback: Callable[[], None], label: str
    ) -> None:
        """Await one off-loop audit attempt, preserving caller cancellation."""
        audit_task = asyncio.create_task(asyncio.to_thread(callback))
        cancelled: asyncio.CancelledError | None = None
        failed = False
        while not audit_task.done():
            try:
                await asyncio.shield(audit_task)
            except asyncio.CancelledError as exc:
                if audit_task.cancelled():
                    failed = True
                    break
                if cancelled is None:
                    cancelled = exc
            except BaseException:
                failed = True
                break
        if not failed:
            try:
                audit_task.result()
            except BaseException:
                failed = True
        if failed:
            logger.warning("{} audit failed", label)
        if cancelled is not None:
            raise cancelled

    def _review_prepared_local_hub_test(
        self,
        *,
        registered: RegisteredToolTestPreview,
        intent: Literal["run", "approve_once"],
        canonical_arguments: bytes,
        arguments: dict[str, Any],
    ) -> (
        tuple[_ResolvedHubTest, str, dict[str, Any]]
        | ToolTestAdmissionBlocked
        | ToolTestAdmissionStale
    ):
        """Rebuild and compare one consumed local preview on a worker thread."""
        from .hub_test_execution import (
            ToolTestAdmissionBlocked,
            ToolTestAdmissionStale,
            canonicalize_arguments,
        )

        public = registered.public
        resolved = self._resolve_hub_test(public.server_key, public.tool_name)
        if resolved is None:
            return ToolTestAdmissionStale(reason="identity_changed")

        fresh_preview = self._preview_fields(resolved)
        if (
            fresh_preview.server_key != public.server_key
            or fresh_preview.tool_name != public.tool_name
        ):
            return ToolTestAdmissionStale(
                reason="identity_changed",
                refreshed_preview=self._issue_hub_test_preview(resolved),
            )
        rendered_gate = public.rendered_gate
        fresh_gate = fresh_preview.rendered_gate
        if rendered_gate not in {"allow", "ask"}:
            return ToolTestAdmissionBlocked(
                reason=(
                    "permission_unresolved"
                    if rendered_gate in {"unavailable", "unresolved"}
                    else "permission_denied"
                ),
                refreshed_preview=self._issue_hub_test_preview(resolved),
            )
        if resolved.unavailable_reason is not None:
            return ToolTestAdmissionStale(
                reason=resolved.unavailable_reason,
                refreshed_preview=self._issue_hub_test_preview(resolved),
            )
        if fresh_preview.definition_hash != public.definition_hash:
            return ToolTestAdmissionStale(
                reason="definition_changed",
                refreshed_preview=self._issue_hub_test_preview(resolved),
            )
        if (
            fresh_preview.authority_fingerprint != public.authority_fingerprint
            or resolved.authority != registered.authority
            or fresh_preview.safe_authority_label != public.safe_authority_label
        ):
            return ToolTestAdmissionStale(
                reason="authority_changed",
                refreshed_preview=self._issue_hub_test_preview(resolved),
            )
        if rendered_gate == "allow":
            if intent != "run":
                return ToolTestAdmissionBlocked(
                    reason="intent_mismatch",
                    refreshed_preview=self._issue_hub_test_preview(resolved),
                )
            if fresh_gate != "allow":
                return ToolTestAdmissionStale(
                    reason="gate_changed",
                    refreshed_preview=self._issue_hub_test_preview(resolved),
                )
        else:
            if intent != "approve_once":
                return ToolTestAdmissionBlocked(
                    reason="intent_mismatch",
                    refreshed_preview=self._issue_hub_test_preview(resolved),
                )
            if fresh_gate not in {"ask", "allow"}:
                return ToolTestAdmissionStale(
                    reason="gate_changed",
                    refreshed_preview=self._issue_hub_test_preview(resolved),
                )

        dispatch_bytes, dispatch_copy = canonicalize_arguments(arguments)
        if dispatch_bytes != canonical_arguments:
            return ToolTestAdmissionBlocked(
                reason="arguments_changed",
                refreshed_preview=self._issue_hub_test_preview(resolved),
            )
        return resolved, fresh_gate, dispatch_copy

    def _execute_prepared_local_hub_test(
        self,
        *,
        tool: HubTool,
        authority: DirectoryChain | None,
        intent: Literal["run", "approve_once"],
        fresh_gate: str,
        canonical_arguments: bytes,
        arguments: dict[str, Any],
        started_at: float,
        cancellation_requested: threading.Event,
        caller_task: asyncio.Task[Any] | None,
        handler_started: threading.Event,
        definitive_after_start: threading.Event,
        deadline_ready: threading.Event,
        effective_deadline: dict[str, float],
        approval_state: dict[str, OneShotLocalHubApproval | None],
    ) -> LocalHubExecutionOutcome:
        """Construct, invoke, sanitize, and close one local provider in-worker."""
        from tldw_chatbook.Agents.agent_models import ToolResult
        from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy
        from . import local_server_tools
        from .hub_test_execution import (
            LocalHubExecutionOutcome,
            LocalHubFinalGate,
            LocalHubStatus,
            OneShotLocalHubApproval,
            authority_fingerprint,
        )

        handle = None
        approval_callback: OneShotLocalHubApproval | None = None

        def _duration_ms() -> int:
            return max(0, int((time.monotonic() - started_at) * 1000))

        def _synthetic(
            status: str, category: str, message: str
        ) -> LocalHubExecutionOutcome:
            approval_consumed = (
                approval_callback.consumed if approval_callback is not None else False
            )
            return LocalHubExecutionOutcome(
                decision=(
                    "approved"
                    if approval_consumed
                    else "allowed"
                    if handler_started.is_set()
                    else "denied"
                ),
                status=cast(LocalHubStatus, status),
                error_category=category,
                final_gate=cast(
                    LocalHubFinalGate,
                    "allow" if approval_consumed else fresh_gate,
                ),
                approval_consumed=approval_consumed,
                dispatch_started=handler_started.is_set(),
                provider_terminal="not_started",
                duration_ms=_duration_ms(),
                result=ToolResult(
                    ok=False,
                    error=message,
                    outcome=("cancelled" if status == "cancelled" else "blocked"),
                ),
            )

        def _dispatch_guard() -> bool:
            if cancellation_requested.is_set() or (
                caller_task is not None and caller_task.cancelling()
            ):
                return False
            handler_started.set()
            return True

        def _cancelled() -> bool:
            return cancellation_requested.is_set() or (
                caller_task is not None and bool(caller_task.cancelling())
            )

        try:
            if _cancelled():
                return _synthetic(
                    "cancelled",
                    "cancelled",
                    "Local tool test was cancelled before dispatch.",
                )
            if authority is None:
                return _synthetic(
                    "blocked",
                    "authority_changed",
                    "Selected workspace authority changed.",
                )
            try:
                enabled = coerce_bool_setting(
                    get_cli_setting("console", "local_tools_enabled", True),
                    True,
                )
            except BaseException:
                return _synthetic(
                    "blocked",
                    "local_configuration_unavailable",
                    "Local tool configuration is unavailable.",
                )
            if not enabled:
                return _synthetic(
                    "blocked", "local_tools_disabled", "Local tools are disabled."
                )
            try:
                root = local_server_tools.resolve_server_workspace_root()
                approval_callback = (
                    OneShotLocalHubApproval(
                        invocation_id=secrets.token_urlsafe(),
                        server_key=tool.server_key,
                        tool_name=tool.name,
                        definition_hash=definition_hash(
                            tool.description, tool.input_schema
                        ),
                        authority_fingerprint=authority_fingerprint(authority),
                        canonical_arguments=canonical_arguments,
                    )
                    if intent == "approve_once" and fresh_gate == "ask"
                    else None
                )
                approval_state["callback"] = approval_callback
                handle = local_server_tools.build_hub_local_provider(
                    root,
                    resolve_state=self.gate_tool_test,
                    approval_callback=approval_callback,
                    dispatch_guard=_dispatch_guard,
                )
            except BaseException:
                return _synthetic(
                    "blocked",
                    "local_provider_unavailable",
                    "Local tool provider is unavailable.",
                )
            if handle.authority != authority:
                return _synthetic(
                    "blocked",
                    "authority_changed",
                    "Selected workspace authority changed.",
                )
            provider = handle.provider
            live_tool = next(
                (
                    candidate
                    for candidate in provider.hub_tools()
                    if candidate.server_key == tool.server_key
                    and candidate.name == tool.name
                    and definition_hash(candidate.description, candidate.input_schema)
                    == definition_hash(tool.description, tool.input_schema)
                ),
                None,
            )
            if live_tool is None:
                return _synthetic(
                    "blocked",
                    "local_tool_ineligible",
                    "Local tool is not eligible for Hub execution.",
                )
            if _cancelled():
                return _synthetic(
                    "cancelled",
                    "cancelled",
                    "Local tool test was cancelled before dispatch.",
                )

            policy = provider.execution_policy_for(tool.name)
            if policy is ToolExecutionPolicy.DEFINITIVE_AFTER_START:
                definitive_after_start.set()
            timeout_floor = provider.timeout_for(tool.name)
            if (
                timeout_floor is not None
                and isinstance(timeout_floor, (int, float))
                and not isinstance(timeout_floor, bool)
                and math.isfinite(float(timeout_floor))
                and float(timeout_floor) > 0
            ):
                effective_deadline["value"] = max(
                    effective_deadline["value"],
                    started_at + float(timeout_floor),
                )
            deadline_ready.set()
            if _cancelled():
                return _synthetic(
                    "cancelled",
                    "cancelled",
                    "Local tool test was cancelled before dispatch.",
                )

            if approval_callback is None:
                detail = provider.invoke_detailed(tool.name, arguments)
            else:
                with approval_callback.invocation_scope():
                    detail = provider.invoke_detailed(tool.name, arguments)
            return self._local_hub_outcome_from_detail(
                detail,
                handle.authority.canonical_root,
                _duration_ms(),
            )
        except BaseException:
            return _synthetic(
                "error", "execution_failed", "Local tool execution failed."
            )
        finally:
            if handle is not None:
                try:
                    handle.close()
                except BaseException:
                    pass

    async def _execute_owned_prepared_local_hub_test(
        self,
        *,
        registered: RegisteredToolTestPreview,
        intent: Literal["run", "approve_once"],
        canonical_arguments: bytes,
        arguments: dict[str, Any],
    ) -> LocalHubExecutionOutcome | ToolTestAdmissionBlocked | ToolTestAdmissionStale:
        """Own local review through terminal audit independently of its caller."""
        from tldw_chatbook.Agents.agent_models import ToolResult
        from .hub_test_execution import (
            LocalHubExecutionOutcome,
            LocalHubFinalGate,
            LocalHubStatus,
            ToolTestAdmissionBlocked,
            ToolTestAdmissionStale,
        )

        public = registered.public
        key = (public.server_key, public.tool_name)
        started_at = time.monotonic()
        loop = asyncio.get_running_loop()
        presentation: asyncio.Future[Any] = loop.create_future()
        cancellation_requested = threading.Event()
        caller_task = asyncio.current_task()
        handler_started = threading.Event()
        definitive_after_start = threading.Event()
        deadline_ready = threading.Event()
        effective_deadline = {"value": started_at + 45.0}
        approval_state: dict[str, OneShotLocalHubApproval | None] = {"callback": None}
        gate_state = {"value": public.rendered_gate}
        audit_tool = {
            "value": HubTool(
                server_key=public.server_key,
                server_label="Local workspace",
                source="local",
                name=public.tool_name,
                description="",
                input_schema={"type": "object", "properties": {}},
                tags=(),
                stale=False,
                executable=False,
            )
        }
        sealed = False

        def _duration_ms() -> int:
            return max(0, int((time.monotonic() - started_at) * 1000))

        def _synthetic(
            status: str,
            category: str,
            message: str,
        ) -> LocalHubExecutionOutcome:
            callback = approval_state["callback"]
            approval_consumed = callback.consumed if callback is not None else False
            return LocalHubExecutionOutcome(
                decision=(
                    "approved"
                    if approval_consumed
                    else "allowed"
                    if handler_started.is_set()
                    else "denied"
                ),
                status=cast(LocalHubStatus, status),
                error_category=category,
                final_gate=cast(
                    LocalHubFinalGate,
                    "allow" if approval_consumed else gate_state["value"],
                ),
                approval_consumed=approval_consumed,
                dispatch_started=handler_started.is_set(),
                provider_terminal="not_started",
                duration_ms=_duration_ms(),
                result=ToolResult(
                    ok=False,
                    error=message,
                    outcome=(
                        "timeout"
                        if status == "timeout"
                        else "cancelled"
                        if status == "cancelled"
                        else "blocked"
                    ),
                ),
            )

        async def _seal_local(outcome: LocalHubExecutionOutcome) -> None:
            nonlocal sealed
            if sealed:
                return
            sealed = True
            try:
                await self._attempt_local_hub_audit(
                    lambda: self._record_local_hub_outcome(
                        audit_tool["value"], arguments, outcome
                    ),
                    "Local Hub terminal",
                )
            finally:
                if not presentation.done():
                    presentation.set_result(outcome)

        async def _seal_review(
            result: ToolTestAdmissionBlocked | ToolTestAdmissionStale,
        ) -> None:
            nonlocal sealed
            if sealed:
                return
            sealed = True
            try:
                await self._attempt_local_hub_audit(
                    lambda: self._record_prepared_hub_block(public, result.reason),
                    "Local Hub review",
                )
            finally:
                if not presentation.done():
                    presentation.set_result(result)

        def _transaction():
            try:
                try:
                    lifecycle_timeout = self._lifecycle_timeout()
                except BaseException:
                    lifecycle_timeout = 45.0
                if not math.isfinite(lifecycle_timeout) or lifecycle_timeout <= 0:
                    lifecycle_timeout = 45.0
                effective_deadline["value"] = started_at + float(lifecycle_timeout)
                deadline_ready.set()
                if cancellation_requested.is_set():
                    return _synthetic(
                        "cancelled",
                        "cancelled",
                        "Local tool test was cancelled before dispatch.",
                    )
                reviewed = self._review_prepared_local_hub_test(
                    registered=registered,
                    intent=intent,
                    canonical_arguments=canonical_arguments,
                    arguments=arguments,
                )
                if isinstance(
                    reviewed, (ToolTestAdmissionBlocked, ToolTestAdmissionStale)
                ):
                    return reviewed
                resolved, fresh_gate, dispatch_copy = reviewed
                audit_tool["value"] = resolved.tool
                gate_state["value"] = fresh_gate
                if cancellation_requested.is_set():
                    return _synthetic(
                        "cancelled",
                        "cancelled",
                        "Local tool test was cancelled before dispatch.",
                    )
                return self._execute_prepared_local_hub_test(
                    tool=resolved.tool,
                    authority=resolved.authority,
                    intent=intent,
                    fresh_gate=fresh_gate,
                    canonical_arguments=canonical_arguments,
                    arguments=dispatch_copy,
                    started_at=started_at,
                    cancellation_requested=cancellation_requested,
                    caller_task=caller_task,
                    handler_started=handler_started,
                    definitive_after_start=definitive_after_start,
                    deadline_ready=deadline_ready,
                    effective_deadline=effective_deadline,
                    approval_state=approval_state,
                )
            except BaseException:
                return _synthetic(
                    "error", "execution_failed", "Local tool execution failed."
                )

        async def _owner() -> None:
            worker: asyncio.Task[Any] | None = None
            try:
                worker = asyncio.create_task(asyncio.to_thread(_transaction))
                while not worker.done():
                    definitive_running = (
                        definitive_after_start.is_set() and handler_started.is_set()
                    )
                    if cancellation_requested.is_set() and not definitive_running:
                        await _seal_local(
                            _synthetic(
                                "cancelled",
                                "cancelled",
                                "Local tool test was cancelled before dispatch.",
                            )
                        )
                        return
                    if (
                        deadline_ready.is_set()
                        and time.monotonic() >= effective_deadline["value"]
                        and not definitive_running
                    ):
                        cancellation_requested.set()
                        await _seal_local(
                            _synthetic(
                                "timeout", "timeout", "Local tool test timed out."
                            )
                        )
                        return
                    await asyncio.sleep(0.005)
                result = await worker
                if cancellation_requested.is_set() and not (
                    definitive_after_start.is_set() and handler_started.is_set()
                ):
                    await _seal_local(
                        _synthetic(
                            "cancelled",
                            "cancelled",
                            "Local tool test was cancelled before dispatch.",
                        )
                    )
                elif isinstance(result, LocalHubExecutionOutcome):
                    await _seal_local(result)
                else:
                    await _seal_review(result)
            except asyncio.CancelledError:
                try:
                    await _seal_local(
                        _synthetic(
                            "cancelled",
                            "cancelled",
                            "Local tool test was cancelled before dispatch.",
                        )
                    )
                finally:
                    raise
            except BaseException:
                await _seal_local(
                    _synthetic(
                        "error", "execution_failed", "Local tool execution failed."
                    )
                )
            finally:
                if worker is not None and not worker.done():
                    try:
                        await worker
                    except BaseException:
                        pass

        _previews, execution = self._ensure_hub_test_state()
        owner = execution.start(key, _owner())
        if owner is None:
            outcome = LocalHubExecutionOutcome(
                decision="denied",
                status="blocked",
                error_category="already_active",
                final_gate=public.rendered_gate,
                approval_consumed=False,
                dispatch_started=False,
                provider_terminal="not_started",
                duration_ms=_duration_ms(),
                result=ToolResult.blocked(
                    "A test for this local tool is already active."
                ),
            )
            await self._attempt_local_hub_audit(
                lambda: self._record_local_hub_outcome(
                    audit_tool["value"], arguments, outcome
                ),
                "Local Hub duplicate",
            )
            return outcome
        try:
            return await asyncio.shield(presentation)
        except asyncio.CancelledError:
            cancellation_requested.set()
            if not (definitive_after_start.is_set() and handler_started.is_set()):
                await asyncio.shield(presentation)
            raise

    def hub_test_active(self, server_key: str, tool_name: str) -> bool:
        """Return whether the service owns an active exact local Hub test."""
        execution = self._local_hub_execution
        return (
            execution.active(server_key, tool_name) if execution is not None else False
        )

    @staticmethod
    def _safe_local_hub_result(result: Any, root: Path, *, hide_error: bool) -> Any:
        """Root-redact, secret-redact, and bound one provider ToolResult."""
        from tldw_chatbook.Agents.agent_models import ToolResult
        from tldw_chatbook.Agents.tool_catalog import redact_root_locator

        def _sanitize_json(value: Any) -> Any:
            if isinstance(value, Mapping):
                return redact_mapping(value)
            if isinstance(value, list):
                return [_sanitize_json(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_sanitize_json(item) for item in value)
            return value

        def _scrub_secret_fragments(value: str) -> str:
            """Scrub secret-key assignments even in provider-truncated JSON."""

            def _quoted_end(start: int) -> tuple[int, bool]:
                quote = value[start]
                index = start + 1
                escaped = False
                while index < len(value):
                    character = value[index]
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == quote:
                        return index + 1, True
                    index += 1
                return len(value), False

            def _container_end(start: int) -> tuple[int, bool]:
                pairs = {"{": "}", "[": "]"}
                stack = [pairs[value[start]]]
                index = start + 1
                quote: str | None = None
                escaped = False
                while index < len(value):
                    character = value[index]
                    if quote is not None:
                        if escaped:
                            escaped = False
                        elif character == "\\":
                            escaped = True
                        elif character == quote:
                            quote = None
                    elif character in {'"', "'"}:
                        quote = character
                    elif character in pairs:
                        stack.append(pairs[character])
                    elif character == stack[-1]:
                        stack.pop()
                        if not stack:
                            return index + 1, True
                    index += 1
                return len(value), False

            def _key_at(start: int) -> tuple[str, int] | None:
                if value[start] in {'"', "'"}:
                    end, closed = _quoted_end(start)
                    if not closed:
                        return None
                    raw = value[start:end]
                    try:
                        key = json.loads(raw) if raw.startswith('"') else raw[1:-1]
                    except (TypeError, ValueError, json.JSONDecodeError):
                        key = raw[1:-1]
                    return str(key), end
                if not (value[start].isalnum() or value[start] in "_-"):
                    return None
                end = start + 1
                while end < len(value) and (value[end].isalnum() or value[end] in "_-"):
                    end += 1
                return value[start:end], end

            output: list[str] = []
            retained_from = 0
            index = 0
            while index < len(value):
                candidate = _key_at(index)
                if candidate is None:
                    index += 1
                    continue
                key, key_end = candidate
                separator = key_end
                while separator < len(value) and value[separator].isspace():
                    separator += 1
                if (
                    not is_secret_key(key)
                    or separator >= len(value)
                    or value[separator] not in ":="
                ):
                    index = key_end
                    continue
                secret_start = separator + 1
                while secret_start < len(value) and value[secret_start].isspace():
                    secret_start += 1
                output.append(value[retained_from:secret_start])
                if secret_start < len(value) and value[secret_start] in {'"', "'"}:
                    secret_end, closed = _quoted_end(secret_start)
                    quote = value[secret_start]
                    output.append(f"{quote}***{quote if closed else ''}")
                elif secret_start < len(value) and value[secret_start] in "[{":
                    secret_end, _closed = _container_end(secret_start)
                    output.append('"***"')
                else:
                    secret_end = secret_start
                    while secret_end < len(value) and value[secret_end] not in (
                        ",;\r\n}]"
                    ):
                        secret_end += 1
                    output.append("***")
                retained_from = secret_end
                index = secret_end
            output.append(value[retained_from:])
            return "".join(output)

        def _safe_text(value: str, limit: int) -> str:
            rooted = str(redact_root_locator(str(value), root))
            try:
                decoded = json.loads(rooted)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, (Mapping, list, tuple)):
                rooted = json.dumps(_sanitize_json(decoded), sort_keys=True)
            rooted = _scrub_secret_fragments(rooted)
            encoded = rooted.encode("utf-8")
            if len(encoded) <= limit:
                return rooted
            return encoded[:limit].decode("utf-8", errors="ignore") + "\n… [truncated]"

        return ToolResult(
            ok=bool(result.ok),
            content=_safe_text(result.content, 32 * 1024),
            error=(
                "Local tool execution failed."
                if hide_error
                else _safe_text(result.error, 300)
            ),
            outcome=result.outcome,
        )

    def _local_hub_outcome_from_detail(
        self,
        detail: Any,
        root: Path,
        duration_ms: int,
    ) -> LocalHubExecutionOutcome:
        """Derive a terminal exclusively from structured provider facts."""
        from .hub_test_execution import (
            LocalHubExecutionOutcome,
            LocalHubFinalGate,
            LocalHubProviderTerminal,
            LocalHubStatus,
        )

        raw_reason = str(getattr(detail.reason_code, "value", detail.reason_code))
        raw_terminal = str(
            getattr(detail.provider_terminal, "value", detail.provider_terminal)
        )
        known_reasons = {
            "unknown_tool",
            "invalid_arguments",
            "permission_off",
            "permission_unresolved",
            "approval_refused",
            "approval_timeout",
            "root_changed",
            "authority_unavailable",
            "handler_returned",
            "handler_raised",
        }
        reason = raw_reason if raw_reason in known_reasons else "execution_failed"
        terminal = (
            raw_terminal
            if raw_terminal in {"not_started", "returned", "raised"}
            else "raised"
        )
        malformed = reason != raw_reason or terminal != raw_terminal
        hide_error = malformed or terminal == "raised" or reason == "handler_raised"
        safe_result = self._safe_local_hub_result(
            detail.result, root, hide_error=hide_error
        )
        if malformed or terminal == "raised" or reason == "handler_raised":
            status = "error"
            category = "execution_failed"
        elif terminal == "returned" and reason == "handler_returned":
            status = "success" if detail.result.ok else "error"
            category = None if detail.result.ok else "tool_failed"
        elif terminal == "not_started" and reason != "handler_returned":
            status = "blocked"
            category = reason
        else:
            status = "error"
            category = "execution_failed"
        raw_gate = str(detail.final_gate)
        final_gate = (
            raw_gate
            if raw_gate
            in {
                "allow",
                "deny",
                "gate_error",
                "kill_switch",
                "no_callback",
                "not_checked",
                "timeout",
            }
            else "unresolved"
        )
        approval_consumed = bool(detail.approval_consumed)
        dispatch_started = bool(detail.dispatch_started)
        decision: LocalHubDecision = (
            "approved"
            if approval_consumed
            else "allowed"
            if dispatch_started
            else "denied"
        )
        return LocalHubExecutionOutcome(
            decision=decision,
            status=cast(LocalHubStatus, status),
            error_category=category,
            final_gate=cast(LocalHubFinalGate, final_gate),
            approval_consumed=approval_consumed,
            dispatch_started=dispatch_started,
            provider_terminal=cast(LocalHubProviderTerminal, terminal),
            duration_ms=max(0, int(duration_ms)),
            result=safe_result,
        )

    def _record_local_hub_outcome(
        self,
        tool: HubTool,
        arguments: dict[str, Any],
        outcome: LocalHubExecutionOutcome,
    ) -> None:
        """Attempt one best-effort terminal row from the shared outcome."""
        exception_type = (
            "TimeoutError"
            if outcome.status == "timeout"
            else "CancelledError"
            if outcome.status == "cancelled"
            else "LocalToolError"
            if outcome.status == "error"
            else None
        )
        self._record_tool_execution(
            tool.server_key,
            tool.name,
            ok=outcome.status == "success",
            duration_ms=outcome.duration_ms,
            status=outcome.status,
            error_category=outcome.error_category,
            exception_type=exception_type,
            status_code=None,
            arguments=arguments,
            registered_argument_names=schema_argument_names(tool.input_schema),
            result=outcome.result,
            initiator="test",
            decision=outcome.decision,
        )

    def _resolve_hub_test(
        self, server_key: str, tool_name: str
    ) -> _ResolvedHubTest | None:
        """Recapture one exact live Hub identity and any local authority."""
        normalized_key = str(server_key or "").strip()
        normalized_name = str(tool_name or "").strip()
        if normalized_key == "local:__local__":
            return self._resolve_local_hub_test(normalized_name)

        if normalized_key.startswith("local:"):
            try:
                records = self.local_service.get_external_servers()
            except Exception as exc:  # noqa: BLE001 -- admission fails closed
                logger.warning(
                    "MCP Hub test external catalog unavailable (exception_type={})",
                    type(exc).__name__,
                )
                return None
            if not isinstance(records, list):
                return None
            for record in records or []:
                if not isinstance(record, Mapping):
                    continue
                for candidate in local_tools_from_record(dict(record)):
                    if (
                        candidate.server_key == normalized_key
                        and candidate.name == normalized_name
                    ):
                        return _ResolvedHubTest(tool=candidate)
            return None

        if normalized_key.startswith("builtin:"):
            try:
                inventory = self.local_service.get_inventory()
            except Exception as exc:  # noqa: BLE001 -- admission fails closed
                logger.warning(
                    "MCP Hub test built-in catalog unavailable (exception_type={})",
                    type(exc).__name__,
                )
                return None
            if not isinstance(inventory, Mapping):
                return None
            for candidate in builtin_tools_from_inventory(dict(inventory)):
                if (
                    candidate.server_key == normalized_key
                    and candidate.name == normalized_name
                ):
                    return _ResolvedHubTest(tool=candidate)
        return None

    def _resolve_local_hub_test(self, tool_name: str) -> _ResolvedHubTest | None:
        """Rebuild local inspection and eligible provider state fail-closed."""
        from . import local_server_tools

        inspection_handle = None
        executable_handle = None
        try:
            try:
                inspection_handle = (
                    local_server_tools.build_hub_local_inspection_provider(
                        Path.cwd(),
                        resolve_state=self.gate_tool_test,
                    )
                )
                inspection = next(
                    (
                        tool
                        for tool in inspection_handle.provider.hub_tools()
                        if tool.server_key == "local:__local__"
                        and tool.name == tool_name
                    ),
                    None,
                )
            except Exception as exc:  # noqa: BLE001 -- admission fails closed
                logger.warning(
                    "MCP Hub test local inspection unavailable (exception_type={})",
                    type(exc).__name__,
                )
                return None
            if inspection is None:
                return None

            try:
                enabled = coerce_bool_setting(
                    get_cli_setting("console", "local_tools_enabled", True),
                    True,
                )
            except Exception as exc:  # noqa: BLE001 -- admission fails closed
                logger.warning(
                    "MCP Hub test local configuration unavailable (exception_type={})",
                    type(exc).__name__,
                )
                return _ResolvedHubTest(
                    tool=replace(inspection, executable=False),
                    unavailable_reason="local_configuration_unavailable",
                )
            if not enabled:
                return _ResolvedHubTest(
                    tool=replace(inspection, executable=False),
                    unavailable_reason="local_tools_disabled",
                )

            try:
                root = local_server_tools.resolve_server_workspace_root()
                executable_handle = local_server_tools.build_hub_local_provider(
                    root,
                    resolve_state=self.gate_tool_test,
                    approval_callback=None,
                )
                executable_tools = executable_handle.provider.hub_tools()
            except Exception as exc:  # noqa: BLE001 -- admission fails closed
                logger.warning(
                    "MCP Hub test local provider unavailable (exception_type={})",
                    type(exc).__name__,
                )
                return _ResolvedHubTest(
                    tool=replace(inspection, executable=False),
                    unavailable_reason="local_provider_unavailable",
                )

            current_hash = definition_hash(
                inspection.description, inspection.input_schema
            )
            eligible = next(
                (
                    tool
                    for tool in executable_tools
                    if tool.server_key == inspection.server_key
                    and tool.name == inspection.name
                    and definition_hash(tool.description, tool.input_schema)
                    == current_hash
                ),
                None,
            )
            if eligible is None:
                return _ResolvedHubTest(
                    tool=replace(inspection, executable=False),
                    unavailable_reason="local_tool_ineligible",
                )
            return _ResolvedHubTest(
                tool=replace(inspection, executable=True),
                authority=executable_handle.authority,
                safe_authority_label="Selected workspace",
            )
        finally:
            for handle in (executable_handle, inspection_handle):
                if handle is None:
                    continue
                try:
                    handle.close()
                except Exception as exc:  # noqa: BLE001 -- admission stays bounded
                    logger.warning(
                        "MCP Hub test provider cleanup failed (exception_type={})",
                        type(exc).__name__,
                    )

    def _preview_fields(self, resolved: _ResolvedHubTest) -> ToolTestAdmissionPreview:
        """Build comparison-only preview fields without registering a nonce."""
        from .hub_test_execution import (
            ToolTestAdmissionPreview,
            authority_fingerprint,
        )

        rendered_gate = "unavailable"
        if resolved.tool.executable and resolved.unavailable_reason is None:
            try:
                gate = self.gate_tool_test(resolved.tool)
                rendered_gate = (
                    gate.state
                    if gate.origin != "gate_error"
                    and gate.state in {"allow", "ask", "deny"}
                    else "unresolved"
                )
            except Exception as exc:  # noqa: BLE001 -- gate errors are unavailable
                logger.warning(
                    "MCP Hub test permission unavailable (exception_type={})",
                    type(exc).__name__,
                )
                rendered_gate = "unresolved"
        return ToolTestAdmissionPreview(
            nonce="",
            server_key=resolved.tool.server_key,
            tool_name=resolved.tool.name,
            definition_hash=definition_hash(
                resolved.tool.description, resolved.tool.input_schema
            ),
            rendered_gate=rendered_gate,
            authority_fingerprint=(
                authority_fingerprint(resolved.authority)
                if resolved.authority is not None
                else None
            ),
            safe_authority_label=resolved.safe_authority_label,
        )

    def _issue_hub_test_preview(
        self, resolved: _ResolvedHubTest
    ) -> ToolTestAdmissionPreview:
        fields = self._preview_fields(resolved)
        previews, _execution = self._ensure_hub_test_state()
        return previews.issue(
            server_key=fields.server_key,
            tool_name=fields.tool_name,
            definition_hash=fields.definition_hash,
            rendered_gate=fields.rendered_gate,
            authority=resolved.authority,
            safe_authority_label=fields.safe_authority_label,
        )

    def _refresh_hub_test_preview(
        self, public: ToolTestAdmissionPreview
    ) -> ToolTestAdmissionPreview | None:
        resolved = self._resolve_hub_test(public.server_key, public.tool_name)
        return self._issue_hub_test_preview(resolved) if resolved is not None else None

    def _record_prepared_hub_block(
        self, public: ToolTestAdmissionPreview, reason: str
    ) -> None:
        self.record_tool_decision(
            public.server_key,
            public.tool_name,
            decision="denied",
            initiator="test",
            error_category=reason,
        )

    def _hub_test_blocked(
        self,
        public: ToolTestAdmissionPreview,
        *,
        reason: str,
        refreshed: ToolTestAdmissionPreview | None = None,
    ) -> ToolTestAdmissionBlocked:
        from .hub_test_execution import ToolTestAdmissionBlocked

        self._record_prepared_hub_block(public, reason)
        return ToolTestAdmissionBlocked(reason=reason, refreshed_preview=refreshed)

    def _hub_test_stale(
        self,
        public: ToolTestAdmissionPreview,
        *,
        reason: str,
        refreshed: ToolTestAdmissionPreview | None = None,
    ) -> ToolTestAdmissionStale:
        from .hub_test_execution import ToolTestAdmissionStale

        self._record_prepared_hub_block(public, reason)
        return ToolTestAdmissionStale(
            reason=reason,
            refreshed_preview=(
                refreshed
                if refreshed is not None
                else self._refresh_hub_test_preview(public)
            ),
        )

    def local_hub_tools(self) -> list[HubTool]:
        """Project the full local catalog with exact executable identities.

        Inspection always comes from the ordinary, unfiltered provider. A
        second descriptor-filtered composition contributes only an exact
        ``(server_key, name, definition_hash)`` eligibility set. Configuration
        or filtered-composition failures therefore remove execution authority
        without removing visible local rows.
        """
        from . import local_server_tools

        inspection_handle = None
        executable_handle = None
        try:
            try:
                inspection_handle = (
                    local_server_tools.build_hub_local_inspection_provider(
                        Path.cwd(),
                        resolve_state=self.gate_tool_test,
                    )
                )
                inspection = [
                    replace(tool, executable=False)
                    for tool in inspection_handle.provider.hub_tools()
                ]
            except Exception as exc:  # noqa: BLE001 -- catalog is fail-soft
                logger.warning(
                    "MCP local inspection catalog unavailable (exception_type={})",
                    type(exc).__name__,
                )
                return []

            enabled = coerce_bool_setting(
                get_cli_setting("console", "local_tools_enabled", True),
                True,
            )
            if not enabled:
                return inspection

            try:
                root = local_server_tools.resolve_server_workspace_root()
                executable_handle = local_server_tools.build_hub_local_provider(
                    root,
                    resolve_state=self.gate_tool_test,
                    approval_callback=None,
                )
                executable_identities = {
                    (
                        tool.server_key,
                        tool.name,
                        definition_hash(tool.description, tool.input_schema),
                    )
                    for tool in executable_handle.provider.hub_tools()
                }
            except Exception as exc:  # noqa: BLE001 -- inspection remains useful
                logger.warning(
                    "MCP local executable projection unavailable (exception_type={})",
                    type(exc).__name__,
                )
                return inspection

            return [
                replace(
                    tool,
                    executable=(
                        tool.server_key,
                        tool.name,
                        definition_hash(tool.description, tool.input_schema),
                    )
                    in executable_identities,
                )
                for tool in inspection
            ]
        finally:
            for handle in (executable_handle, inspection_handle):
                if handle is None:
                    continue
                try:
                    handle.close()
                except Exception as exc:  # noqa: BLE001 -- catalog stays fail-soft
                    logger.warning(
                        "MCP local provider cleanup failed (exception_type={})",
                        type(exc).__name__,
                    )

    # ---- Local MCP prompt-reduction recommendations -----------------------
    # This is intentionally local/MCP-only (ADR-081): no shell command
    # analysis, no telemetry, no model-based auto-approval.

    async def _local_prompt_reduction_tools(self) -> list[HubTool]:
        """Collect live local and built-in tools for prompt analysis."""
        tools: list[HubTool] = []
        try:
            records = await self.local_external_catalog()
        except Exception as exc:
            logger.warning(
                "MCP prompt-reduction local catalog read failed (exception_type={})",
                type(exc).__name__,
            )
            records = []
        if isinstance(records, list):
            for record in records:
                if isinstance(record, Mapping):
                    tools.extend(local_tools_from_record(dict(record)))

        local_service = getattr(self, "local_service", None)
        get_inventory = getattr(local_service, "get_inventory", None)
        if callable(get_inventory):
            try:
                inventory = get_inventory()
            except Exception as exc:
                logger.warning(
                    "MCP prompt-reduction built-in inventory read failed "
                    "(exception_type={})",
                    type(exc).__name__,
                )
                inventory = None
            if isinstance(inventory, Mapping):
                tools.extend(builtin_tools_from_inventory(dict(inventory)))
        return tools

    def _recent_execution_records(self, limit: int) -> list[dict[str, Any]]:
        """Read recent MCP execution records defensively."""
        try:
            log = self.execution_log
        except Exception as exc:
            logger.warning(
                "MCP prompt-reduction execution log access failed (exception_type={})",
                type(exc).__name__,
            )
            return []
        if log is None:
            return []
        try:
            return log.read_recent(limit)
        except Exception as exc:
            logger.warning(
                "MCP prompt-reduction execution log read failed (exception_type={})",
                type(exc).__name__,
            )
            return []

    async def _permission_prompt_recommendation_snapshot(
        self,
        *,
        min_approved_count: int,
        limit: int,
    ) -> tuple[PermissionPromptReport, dict[tuple[str, str], HubTool]]:
        """Build a report and tool lookup from one live catalog snapshot."""
        records = self._recent_execution_records(limit)
        tools = await self._local_prompt_reduction_tools()
        states = self.effective_tool_states(tools)
        report = build_permission_prompt_report(
            records,
            tools,
            states,
            min_approved_count=min_approved_count,
        )
        return report, {(tool.server_key, tool.name): tool for tool in tools}

    async def permission_prompt_recommendations(
        self,
        *,
        min_approved_count: int = DEFAULT_MIN_APPROVED_COUNT,
        limit: int = 200,
    ) -> PermissionPromptReport:
        """Return local MCP permission prompt-reduction recommendations.

        Args:
            min_approved_count: Minimum prompted approvals required for a
                recommendation.
            limit: Maximum recent local execution records to analyze.

        Returns:
            A telemetry-free report based on local MCP metadata and current
            permission state.
        """
        report, _tools = await self._permission_prompt_recommendation_snapshot(
            min_approved_count=min_approved_count,
            limit=limit,
        )
        return report

    async def apply_permission_prompt_recommendation(
        self,
        server_key: str,
        tool_name: str,
        *,
        min_approved_count: int = DEFAULT_MIN_APPROVED_COUNT,
    ) -> PermissionPromptRecommendation:
        """Persist a currently recommended tool-level allow.

        Args:
            server_key: Stable key for the recommendation's MCP server.
            tool_name: Name of the recommended tool.
            min_approved_count: Minimum prompted approvals required for a
                recommendation.

        Returns:
            The recommendation whose hash-safe tool-level allow was persisted.

        Raises:
            KeyError: If the requested server and tool pair is not a current
                prompt-reduction recommendation.
            RuntimeError: If the MCP permission store is unavailable.
        """
        normalized_key = str(server_key or "").strip()
        normalized_tool = str(tool_name or "").strip()
        store = self.permission_store
        if store is None:
            raise RuntimeError("MCP permission store unavailable")
        report, tools_by_key = await self._permission_prompt_recommendation_snapshot(
            min_approved_count=min_approved_count,
            limit=200,
        )
        recommendation = next(
            (
                item
                for item in report.recommendations
                if item.server_key == normalized_key
                and item.tool_name == normalized_tool
            ),
            None,
        )
        if recommendation is None:
            raise KeyError(
                "No prompt-reduction recommendation for "
                f"{normalized_key}/{normalized_tool}"
            )

        tool = tools_by_key.get((normalized_key, normalized_tool))
        if tool is None:
            raise KeyError(
                f"No live MCP tool found for {normalized_key}/{normalized_tool}"
            )
        self.set_tool_state(normalized_key, normalized_tool, "allow", tool=tool)
        return recommendation

    @staticmethod
    def _normalize_batch_requests(requests: list[Any]) -> list[dict[str, Any]]:
        """Coerce a ``runtime.batch`` ``requests`` list exactly the way
        :meth:`LocalMCPControlService.run_runtime_batch` does.

        Item 2 (PR-T3 fix round F). The pre-dispatch scan used to check
        ``isinstance(request, Mapping)`` directly against the RAW items --
        but ``local_control_service.run_runtime_batch()``
        (``local_control_service.py:500``) normalizes every item with
        ``dict(request)``, which also accepts a list of ``(key, value)``
        pairs (``dict([["method", "tools/call"]])`` succeeds). A JSON array
        nested inside ``requests`` is exactly that shape -- ``json.loads``
        produces a bare ``list`` for a JSON array, and the Advanced pane's
        raw-JSON textarea is reachable with one -- so it is a ``list``, not
        a ``Mapping``, and the OLD scan skipped it silently while the real
        dispatcher still ran it.

        Called ONCE here and the result is reused for both the scan
        (:meth:`_refuse_raw_tool_call` below) and the actual dispatch, so
        the two can never again disagree about what counts as a request --
        a duplicated coercion rule in two places is how they drifted apart
        the first time. Any item that is not ``dict``-coercible (neither a
        ``Mapping`` nor an iterable of pairs) raises the SAME
        ``TypeError``/``ValueError`` ``run_runtime_batch()`` would have
        raised for it -- this only moves the moment that error surfaces
        earlier, before any item has dispatched, which is a strict
        improvement for the same "check everything before running
        anything" reason the scan exists at all.

        Fix Round H (PR-T3 review), Item 5: called here, in ``run_action()``
        (line ~1271), BEFORE ``local_service.run_runtime_batch()``'s own
        ``_require_allowed("mcp.runtime.trigger.local")`` check (nested one
        layer down, ``local_control_service.py:499``) ever runs. A
        non-``dict``-coercible item (e.g. a bare ``5``) therefore raises
        ``TypeError`` HERE, before the capability check, so a caller
        lacking ``mcp.runtime.trigger.local`` who submits a malformed batch
        gets a shape error instead of a policy denial. DECISION (not a
        defect, left as-is): nothing runs either way -- both paths refuse
        the whole batch before any item dispatches -- and this is
        consistent with the pre-existing ``_refuse_raw_tool_call()`` scan
        immediately below, which ALSO runs at this layer, before the same
        nested permission check, for the identical "validate the whole
        batch before dispatching or gating it" reason (see the comment
        above this call site). Reordering to gate first would mean parsing
        malformed, potentially adversarial batch items before knowing the
        caller may even act -- worse, not better. If this needs revisiting,
        it should move with `_refuse_raw_tool_call()`'s ordering too, not
        change alone.

        Args:
            requests: The raw ``requests`` list from the ``runtime.batch``
                payload.

        Returns:
            One ``dict`` per item, in order.
        """
        return [dict(request) for request in requests]

    def _refuse_raw_tool_call(self, method: Any) -> None:
        """Refuse a raw ``tools/call`` on the runtime request/batch runners.

        Task 6 (PR-T3), Route B. Those runners hand a protocol method
        straight to the in-process runtime, whose ``tools/call`` branch
        (``LocalRuntimeDelegate.request``) calls the same
        ``execute_tool()`` seam :meth:`execute_advanced_tool` gates -- so
        without this they are a second, unlabelled way to run a tool the
        user set to Off, leaving no execution-log row either. Executing a
        tool keeps exactly one door here: ``tool.execute``, which gates
        and records. Every other method is untouched.

        Args:
            method: The protocol method from the request payload; anything
                that is not ``tools/call`` passes through.

        Raises:
            RawToolCallRefusedError: ``method`` is ``tools/call``. A
                ``PermissionError`` subclass -- any existing ``except
                PermissionError`` handler upstream still catches it.
        """
        if str(method or "").strip() == "tools/call":
            # Item 2 (PR-T3 fix round D): typed, not a bare
            # `PermissionError` -- see `RawToolCallRefusedError`'s own
            # docstring for why this and the delegate's own raise site
            # share one type the same way they already share one message.
            raise RawToolCallRefusedError(_RAW_TOOL_CALL_REFUSED_MESSAGE)

    async def execute_advanced_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """Execute one built-in tool for the Advanced runner's ``tool.execute``.

        Task 6 (PR-T3), Route B. The Advanced (legacy control plane) panel
        offers a ``tool.execute`` descriptor whose Run Action button
        reached ``local_service.execute_tool()`` straight through
        :meth:`run_action` -- the ONLY execution path in the Hub that
        touched neither the per-tool permission gate (a tool the user set
        to Off ran anyway) nor :meth:`_record_tool_execution` (which lives
        inside :meth:`execute_hub_tool`), so such a run left no row in the
        audit trail at all. This routes it through the same shared seam as
        the Test Tool runner and the agent bridge, so it is gated and
        logged like every other run.

        The gate is :meth:`gate_tool_test_by_key`, always called with
        ``BUILTIN_SERVER_KEY`` -- the Advanced runner names a tool in raw
        JSON and has no live ``HubTool`` to fingerprint, which is exactly
        the case that seam exists for. It resolves ``deny``/``ask`` at full
        fidelity; an ``allow`` collapses to ``ask`` UNLESS the server key is
        in ``BY_KEY_HASH_FREE_SERVER_KEYS`` (Fix Round A, Item 1; narrowed
        from the wider ``HASH_FREE_SERVER_KEYS`` in Fix Round C, Item 2 --
        see that constant's docstring for why) -- and ``BUILTIN_SERVER_KEY``
        (``"builtin:tldw_chatbook"``) is in both, so a tool explicitly (or
        by inherited default) set to Allow resolves as ``allow`` here, same
        as :meth:`gate_tool_test` would with a live ``HubTool``. A ``deny``
        is refused here (nothing runs, a ``decision="denied"`` row is
        recorded); an ``allow`` executes under ``decision="allowed"``; and
        an ``ask`` executes under ``decision="approved"``, naming the
        Advanced runner's own two-press confirm
        (``MCPInspector._run_advanced_action()``) as the approval -- the
        same contract, and the same audit vocabulary, the agent bridge's
        Ask-then-approved calls already use. That confirm is a UI-level
        mis-click guard independent of the gate state, not a substitute for
        it: a caller that skips it is the one lying to the log, not this
        method, and the gate's hard "Off" verdict is enforced here
        regardless of whether a confirm ever happened.

        A gate resolution that RAISES fails closed (a synthetic deny),
        mirroring ``MCPWorkbench._resolve_test_gate()``: a runtime error
        must never silently expose a tool permissions might forbid.

        No ``registered_argument_names`` is supplied: the payload is
        free-form JSON with no schema behind it, so this path honestly
        records no argument provenance rather than inventing some.

        Args:
            tool_name: Name of the built-in tool to execute.
            arguments: Tool arguments; defaults to an empty dict.

        Returns:
            The raw result payload from the built-in tool call.

        Raises:
            MCPHubGateDeniedError: The tool is set to Off in Permissions
                (the Hub's own gate), OR the gate check itself raised and
                this failed closed -- the two are distinguished by message
                and by the ``error_category`` recorded on the audit row
                (``"gate_denied"`` vs. ``"gate_error"``; see the deny
                branch below), never conflated in the user-facing text. A
                ``PermissionError`` subclass.
            MCPGovernanceDenied: The in-process runtime-governance profile
                denies it, raised further down by ``local_control_
                service.execute_tool()``. A ``PermissionError`` subclass,
                and a DIFFERENT type from ``MCPHubGateDeniedError`` above
                -- see that type's own docstring for why.
            RuntimeError: The tool call fails or exceeds the effective
                timeout.
        """
        normalized_tool_name = str(tool_name or "").strip()
        normalized_arguments = dict(arguments or {})
        try:
            state = self.gate_tool_test_by_key(BUILTIN_SERVER_KEY, normalized_tool_name)
        except Exception as exc:
            logger.warning(
                "MCP advanced tool.execute gate check failed for {}; failing closed: {}",
                normalized_tool_name,
                type(exc).__name__,
            )
            state = EffectiveToolState(state="deny", origin="gate_error")

        if state.state == "deny":
            # Item 1 (PR-T3 fix round F): `state.origin == "gate_error"` is
            # THIS method's own synthesized fail-closed verdict from the
            # `except Exception` above -- the permission RESOLVER raised,
            # not a genuine "Off". Before this branch existed, that case
            # fell straight into the genuine-deny copy/token below, telling
            # the user (and the audit row) a confident, false fact about
            # their configuration -- indistinguishable from a real
            # user-configured deny. Repeats the exact pattern fix round B
            # (task-2536) removed from the Test Tool panel's blocked-result
            # body, and fix round D polished further;
            # `mcp_workbench._resolve_test_gate()` already branches on this
            # same `origin` for the same reason.
            is_gate_error = state.origin == "gate_error"
            blocked_message = (
                _ADVANCED_EXECUTE_GATE_ERROR_MESSAGE
                if is_gate_error
                else _ADVANCED_EXECUTE_BLOCKED_MESSAGE.format(
                    tool=normalized_tool_name or "This tool"
                )
            )
            # Fix Round B, Item 1: Fix Round A had this row reuse the
            # `error=` mechanism to reach `error_category="approval_
            # cancelled"` -- but that category means a user cancelled an
            # approval that was genuinely OFFERED, and that never happened
            # here: the permission GATE denied this outright, before any
            # approval could be offered. Reusing the vocabulary made the
            # row specific and FALSE instead of generic and true. This now
            # passes the honest, explicit `error_category="gate_denied"`
            # token instead -- chosen over "policy_denied" to stay
            # consistent with this exact code path's own vocabulary
            # (`gate_tool_test_by_key()`, `_resolve_test_gate()`,
            # `EffectiveToolState.origin="gate_error"` -- all "gate", never
            # "policy") and to avoid a false cross-reference to the
            # separate runtime-policy engine (`runtime_policy/types.py`'s
            # `PolicyDeniedError`), which is a different subsystem denying
            # for a different reason. The message text itself is still
            # never persisted -- `error_category` is a sanitized token,
            # never free text, by the execution log's metadata-only
            # design.
            #
            # Item 1 (PR-T3 fix round F): `"gate_denied"` itself is ALSO
            # false for the `is_gate_error` case -- it means "the Hub's
            # gate denied", and here the gate never resolved at all.
            #
            # Fix Round H (PR-T3 review), Item 3: corrected the arithmetic
            # below -- `"policy_denied"` is a name this branch REJECTED
            # (see the comment just above), never a token it produces, so
            # it does not count toward "already distinguishes". `"gate_
            # error"` is the THIRD token this branch actually produces,
            # consistent with the TWO real ones that came before it
            # (`"gate_denied"` = the Hub's own Allow/Ask/Off gate genuinely
            # resolving to Off; `"governance_denied"` = the separate
            # in-process runtime-governance profile, a few hundred lines
            # up) -- matching the `EffectiveToolState.origin` value that
            # produced it rather than inventing an unrelated word for the
            # same fact. `"policy_denied"` stays named here ONLY as the
            # rejected alternative -- naming it would have falsely cross-
            # referenced the unrelated `runtime_policy` engine's own
            # `PolicyDeniedError`.
            self.record_tool_decision(
                BUILTIN_SERVER_KEY,
                normalized_tool_name,
                decision="denied",
                initiator="test",
                error_category="gate_error" if is_gate_error else "gate_denied",
            )
            # Item 2 (PR-T3 fix round D): typed, not a bare `PermissionError`
            # -- see `MCPHubGateDeniedError`'s own docstring for why this is
            # a DIFFERENT type from `MCPGovernanceDenied` below.
            raise MCPHubGateDeniedError(blocked_message)

        return await self.execute_hub_tool(
            BUILTIN_SERVER_KEY,
            normalized_tool_name,
            normalized_arguments,
            initiator="test",
            decision="allowed" if state.state == "allow" else "approved",
            timeout_seconds=self._lifecycle_timeout(),
        )

    # ---- Chat bridge seam (Phase 5) -------------------------------------
    # Timeout knobs, in-memory session approvals, and best-effort decision
    # recording for tool calls that stop before execution (denied / timed
    # out waiting for approval). Backs the chat bridge / agent-runtime
    # MCPToolProvider (task-201); UI-free.

    def _tool_call_timeout(self) -> float:
        """Resolve the per-call tool execution timeout.

        Mirrors :meth:`_lifecycle_timeout`'s config-read/fallback guard,
        but reads a distinct config key: the Hub's Test Tool runner and
        the chat bridge intentionally have independently tunable
        timeouts.

        Returns:
            The configured ``[mcp] tool_call_timeout_seconds`` value in
            seconds, falling back to ``60.0`` when unset or unparsable.
        """
        try:
            return float(get_cli_setting("mcp", "tool_call_timeout_seconds", 60.0))
        except (TypeError, ValueError):
            return 60.0

    def approval_timeout_seconds(self) -> float:
        """Resolve how long the chat bridge waits for a human approval.

        Mirrors :meth:`_lifecycle_timeout`'s config-read/fallback guard.

        Returns:
            The configured ``[mcp] approval_timeout_seconds`` value in
            seconds, falling back to ``120.0`` when unset or unparsable.
        """
        try:
            return float(get_cli_setting("mcp", "approval_timeout_seconds", 120.0))
        except (TypeError, ValueError):
            return 120.0

    def approve_for_session(self, server_key: str, tool_name: str) -> None:
        """Grant a session-scoped approval for one server/tool pair.

        Session approvals are held in memory only, for the lifetime of
        this service instance (an app-run singleton) -- they are never
        persisted to disk and do not survive an app restart or a fresh
        instance of this service.

        Args:
            server_key: Prefixed server key the tool belongs to.
            tool_name: Name of the tool being approved.
        """
        self._session_approvals.add((server_key, tool_name))

    def is_session_approved(self, server_key: str, tool_name: str) -> bool:
        """Check whether a server/tool pair has a session-scoped approval.

        Args:
            server_key: Prefixed server key the tool belongs to.
            tool_name: Name of the tool to check.

        Returns:
            ``True`` if :meth:`approve_for_session` was called for this
            exact pair since the last :meth:`clear_session_approvals`
            call (or since this service instance was constructed);
            ``False`` otherwise. Always ``False`` on a fresh instance or
            after an app restart -- the grant is not persisted.
        """
        return (server_key, tool_name) in self._session_approvals

    def clear_session_approvals(self) -> None:
        """Discard every in-memory session approval on this instance."""
        self._session_approvals.clear()

    def record_tool_decision(
        self,
        server_key: str,
        tool_name: str,
        *,
        decision: str,
        initiator: str = "agent",
        error: str | None = None,
        error_category: str | None = None,
    ) -> None:
        """Best-effort log a tool-call decision that never executed.

        For approval outcomes that stop the call before the tool runs
        (denied, timed out waiting for approval) so the execution log
        keeps a complete decision trail even for calls that never
        reached :meth:`execute_hub_tool`. Same never-raise contract as
        :meth:`_record_tool_execution` -- and, per that method's N1
        lesson, the ``self.execution_log`` property access happens
        *inside* the try, since the property itself can raise.

        Fix Round B, Item 1: ``error_category`` was added because the
        pre-existing ``decision``/``error`` derivation below has exactly
        one vocabulary slot for "denied, and we have a reason" --
        ``"approval_cancelled"`` -- and that term has a real, narrower
        meaning: a user actively dismissing/cancelling an approval that
        was genuinely OFFERED (see ``test_record_tool_decision_writes_
        denied_record`` and ``console_chat_controller.py``'s shutdown-
        mid-approval recorder, both of which keep using the derivation
        below unchanged). A permission GATE denying a call outright --
        no approval was ever offered, nobody cancelled anything -- is a
        different fact and needs its own token; forcing it through
        ``error=`` to hit the same branch produced a row that was
        specific and FALSE rather than generic and true (the bug this
        parameter exists to close -- see ``execute_advanced_tool()``'s
        deny branch).

        Args:
            server_key: Prefixed server key the tool belongs to.
            tool_name: Name of the tool the decision applies to.
            decision: Outcome of the decision (e.g. ``"denied"``,
                ``"timeout"``).
            initiator: Who/what produced the decision; defaults to
                ``"agent"``.
            error: Optional human-readable detail for the record. Only
                used, together with ``decision``, to *derive*
                ``error_category`` when ``error_category`` itself is not
                supplied -- the text is never persisted.
            error_category: Optional explicit category token, used
                verbatim (through ``safe_metadata_token()``) instead of
                the ``decision``/``error`` derivation below. Because
                ``safe_metadata_token()`` rejects any value containing
                whitespace, this can only ever carry a single bare token
                (e.g. ``"gate_denied"``), never a sentence -- do not
                retry passing prose here, it will just come back as
                ``"invalid"``.
        """
        try:
            log = self.execution_log
            if log is None:
                return
            record = build_record(
                server_key=server_key,
                tool_name=tool_name,
                initiator=initiator,
                decision=decision,
                ok=False,
                status="blocked",
                duration_ms=0,
                error_category=(
                    error_category
                    if error_category is not None
                    else (
                        "approval_timeout"
                        if "timeout" in decision
                        else "approval_cancelled"
                        if decision == "denied" and error
                        else "denied"
                        if decision == "denied"
                        else "execution_bridge_failed"
                        if error
                        else "blocked"
                    )
                ),
            )
            log.append(record)
        except Exception as exc:
            logger.warning(
                "MCP tool decision record failed (exception_type={})",
                type(exc).__name__,
            )

    # ---- Typed permission methods (Phase 4) ----------------------------
    # Backs the Hub's Permissions mode: effective-state resolution (with
    # the rug-pull downgrade audit), the state setters, and the Test Tool
    # gate. Keep this UI-free -- the Phase 5 chat bridge / agent-runtime
    # MCPToolProvider will call `gate_tool_test`-shaped resolution too.

    @property
    def permission_store(self) -> MCPPermissionStore | None:
        if self._permission_store is not None:
            return self._permission_store
        store = getattr(self.local_service, "store", None)
        if store is None:
            return None
        permissions_path = Path(store.path).with_name("mcp_permissions.json")
        self._permission_store = MCPPermissionStore(permissions_path)
        return self._permission_store

    def effective_tool_states(
        self, tools: list[HubTool], *, profile_id: str = "default"
    ) -> dict[tuple[str, str], EffectiveToolState]:
        """Resolve the effective allow/ask/deny state for every tool in ``tools``.

        Loads the permission-store payload once and resolves every tool
        against it (Task 2's `resolve_effective_state`). Any tool whose
        resolution flags a hash mismatch against an explicit tool-level
        ``allow`` (`EffectiveToolState.config_changed`) has that mismatch
        persisted via `store.mark_config_changed()`; the *first* time that
        transition happens for a given tool, exactly one
        ``decision="downgraded"`` audit record is appended to the
        execution log, best-effort, mirroring `_record_tool_execution`'s
        never-raise contract -- a logging failure must never prevent the
        resolved states from being returned. Later calls see the marker
        already set (`mark_config_changed` returns False) and skip the
        audit.

        `config_changed` is only ever True when the tool carries an
        *explicit* tool-level ``allow`` entry (see
        `resolve_effective_state`): a tool that inherits its state from a
        server or global default has nothing to compare hashes against,
        so it can never trigger a marker or an audit here.

        Workspace assistant defaults (Task 6): ``profile_id`` selects the
        named permission profile the resolution (and any rug-pull marker)
        runs under. The store's chain is PROFILE-MAJOR: the named
        profile's tool/server/global levels settle before the default
        profile's are consulted. Defaults to ``"default"`` --
        byte-identical to the single-profile behavior.

        No store configured -> every tool resolves to
        `EffectiveToolState(state="ask", origin="global_default")` (fail
        closed).
        """
        store = self.permission_store
        if store is None:
            return {
                (tool.server_key, tool.name): EffectiveToolState(
                    state="ask", origin="global_default"
                )
                for tool in tools
            }

        payload = store.load()
        results: dict[tuple[str, str], EffectiveToolState] = {}
        for tool in tools:
            effective = resolve_effective_state(payload, tool, profile_id=profile_id)
            results[(tool.server_key, tool.name)] = effective
            if effective.config_changed:
                self._audit_downgrade_if_fresh(store, tool, profile_id=profile_id)
        return results

    def _audit_downgrade_if_fresh(
        self, store: MCPPermissionStore, tool: HubTool, *, profile_id: str = "default"
    ) -> None:
        # Best-effort, same never-raise contract as `_record_tool_execution`:
        # a persistence/logging failure here must never propagate out of
        # `effective_tool_states()` and mask the resolved states it already
        # computed.
        try:
            newly_marked = store.mark_config_changed(
                tool.server_key, tool.name, profile_id=profile_id
            )
            if not newly_marked:
                return
            log = self.execution_log
            if log is None:
                return
            record = build_record(
                server_key=tool.server_key,
                tool_name=tool.name,
                initiator="system",
                decision="downgraded",
                ok=False,
                status="blocked",
                duration_ms=0,
                error_category="definition_changed",
            )
            log.append(record)
        except Exception as exc:
            logger.warning(
                "MCP permission downgrade audit failed (exception_type={})",
                type(exc).__name__,
            )

    def set_tool_state(
        self,
        server_key: str,
        tool_name: str,
        ui_state: str | None,
        *,
        tool: HubTool | None = None,
        profile_id: str = "default",
    ) -> None:
        """Set (or clear, when ``ui_state`` is None) a tool-level override.

        Args:
            server_key: Owning server's stable key.
            tool_name: Tool name within that server.
            ui_state: One of ``None`` (inherit), ``"allow"``, ``"ask"``,
                ``"deny"``.
            tool: Required when ``ui_state`` is ``"allow"`` -- its
                description/input_schema are fingerprinted into the stored
                ``definition_hash`` the rug-pull guard compares against
                later. Not required (and not hashed) when ``server_key`` is
                in ``HASH_FREE_SERVER_KEYS`` (e.g. ``agent:builtin``).
            profile_id: Permission profile to write (workspace assistant
                defaults, Task 6); defaults to the ``default`` profile --
                byte-identical to the single-profile behavior.

        Raises:
            ValueError: ``ui_state`` is ``"allow"``, ``tool`` is None, and
                ``server_key`` is not in ``HASH_FREE_SERVER_KEYS``.
        """
        store = self.permission_store
        if store is None:
            return
        hash_value: str | None = None
        if ui_state == "allow" and server_key not in HASH_FREE_SERVER_KEYS:
            if tool is None:
                raise ValueError(
                    "tool is required to set state 'allow' (need its description/input_schema)"
                )
            hash_value = definition_hash(tool.description, tool.input_schema)
        store.set_tool_state(
            server_key,
            tool_name,
            ui_state,
            definition_hash=hash_value,
            profile_id=profile_id,
        )

    def set_server_default(
        self, server_key: str, state: str | None, *, profile_id: str = "default"
    ) -> None:
        store = self.permission_store
        if store is None:
            return
        store.set_server_default(server_key, state, profile_id=profile_id)

    def set_global_default(self, state: str, *, profile_id: str = "default") -> None:
        store = self.permission_store
        if store is None:
            return
        store.set_global_default(state, profile_id=profile_id)

    def get_kill_switch(self) -> bool:
        store = self.permission_store
        if store is None:
            return False
        return store.get_kill_switch()

    def set_kill_switch(self, value: bool) -> None:
        store = self.permission_store
        if store is None:
            return
        store.set_kill_switch(value)

    def gate_tool_test(
        self, tool: HubTool, *, profile_id: str = "default"
    ) -> EffectiveToolState:
        """Resolve one tool's effective state for the Hub's Test Tool gate.

        A single fresh `load()` + resolve -- no batching, no audit
        emission (the `effective_tool_states()` sync/render pass owns the
        rug-pull downgrade audit; calling both for the same mismatch would
        double-count it).

        Workspace assistant defaults (Task 6): ``profile_id`` selects the
        named permission profile the resolution runs under (PROFILE-MAJOR
        chain, see `resolve_effective_state`); defaults to ``"default"`` --
        byte-identical to the single-profile behavior.

        Deliberately ignores the kill switch: the switch gates chat
        send-time tool-call assembly for the Phase 5 chat bridge /
        agent-runtime MCPToolProvider, not this operator-initiated Hub
        diagnostic -- an operator explicitly running Test Tool from the
        Hub UI should see the tool's real allow/ask/deny state regardless
        of whether the kill switch happens to be on.

        No store configured -> `EffectiveToolState(state="ask",
        origin="global_default")` (fail closed).
        """
        store = self.permission_store
        if store is None:
            return EffectiveToolState(state="ask", origin="global_default")
        payload = store.load()
        return resolve_effective_state(payload, tool, profile_id=profile_id)

    def gate_tool_test_for_profile(
        self, tool: HubTool, profile_id: str
    ) -> EffectiveToolState:
        """Resolve one tool's gate under an explicit, non-default-carrying
        profile id.

        Workspace assistant defaults (Task 6): the Console's per-workspace
        provider closure (Task 7) holds a workspace's profile id and calls
        this alias so the required argument is positional-by-name at the
        call site -- an explicit ``profile_id`` can never silently fall
        back to ``"default"`` the way an omitted keyword would. Thin
        delegation to :meth:`gate_tool_test`; same no-audit, kill-switch-
        ignoring, fail-closed contract.
        """
        return self.gate_tool_test(tool, profile_id=profile_id)

    def gate_tool_test_by_key(
        self, server_key: str, tool_name: str
    ) -> EffectiveToolState:
        """Resolve one tool's Test Tool gate from the store alone, with no
        live ``HubTool`` to fingerprint.

        I1: the counterpart `gate_tool_test()` needs a `HubTool` to
        hash-compare an explicit ``allow`` against its stored
        ``definition_hash`` (the rug-pull guard) -- this is the seam for
        when the workbench can't produce one anymore (`_tool_for()` came
        back empty: the tool dropped out of `_last_hub_tools` since the
        Test panel opened, e.g. a resync racing a rug-pull refresh).
        Without this, `MCPWorkbench._resolve_test_gate()` had nothing to
        gate a vanished-but-still-denied tool against and fell through to
        an ungated dispatch.

        Deny/ask verdicts resolve at full fidelity (no hash check is
        needed to trust those); an "allow" verdict -- explicit or
        inherited -- can't be trusted without the live definition, so it
        resolves to "ask" instead (see `resolve_effective_state_by_key`).

        No store configured -> `EffectiveToolState(state="ask",
        origin="global_default")` (fail closed), matching
        `gate_tool_test()`.
        """
        store = self.permission_store
        if store is None:
            return EffectiveToolState(state="ask", origin="global_default")
        payload = store.load()
        return resolve_effective_state_by_key(payload, server_key, tool_name)
