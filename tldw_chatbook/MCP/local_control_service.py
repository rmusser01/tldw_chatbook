from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from uuid import uuid4

from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import RuntimeSourceState

from .client import MCPClient
from .local_runtime_delegate import LocalMCPRuntimeDelegate
from .local_store import LocalApprovalRequest, LocalExternalMCPProfile, LocalGovernanceRule, LocalMCPStore

_ENV_PLACEHOLDER_PATTERN = re.compile(r"^\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))$")
_SPAWN_ENV_BASELINE_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
)
_TOOL_ACTION_IDS = {
    "chat_with_llm": "chat.launch.local",
    "chat_with_character": "character.sessions.launch.local",
    "search_rag": "media.reading.list.local",
    "search_conversations": "chat.list.local",
    "create_note": "notes.create.local",
    "search_notes": "notes.list.local",
    "list_characters": "character.persona.list.local",
    "get_conversation_history": "chat.detail.local",
    "export_conversation": "chat.detail.local",
    "ingest_media": "media.ingestion_jobs.launch.local",
}
_RESOURCE_ACTION_IDS = (
    ("conversation://", "chat.detail.local"),
    ("note://", "notes.detail.local"),
    ("character://", "character.persona.detail.local"),
    ("media://", "media.reading.detail.local"),
    ("rag-chunk://", "media.reading.detail.local"),
)
_REQUEST_METHOD_ACTION_IDS = {
    "initialize": "mcp.runtime.observe.local",
    "status/get": "mcp.runtime.observe.local",
    "tools/list": "mcp.inventory.list.local",
    "resources/list": "mcp.inventory.list.local",
    "prompts/list": "mcp.inventory.list.local",
}


def _default_manifest_provider() -> dict[str, Any]:
    from .server import describe_local_mcp_capabilities

    return describe_local_mcp_capabilities()


class LocalMCPControlService:
    def __init__(
        self,
        *,
        store: LocalMCPStore,
        client: MCPClient | None = None,
        manifest_provider: Callable[[], dict[str, Any]] | None = None,
        policy_enforcer: Any | None = None,
        runtime_delegate: LocalMCPRuntimeDelegate | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.manifest_provider = manifest_provider or _default_manifest_provider
        self.policy_enforcer = policy_enforcer
        self.runtime_delegate = runtime_delegate or LocalMCPRuntimeDelegate(
            manifest_provider=self.manifest_provider,
        )
        self._runtime_activity_limit = 50

    def get_overview(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        inventory = self.get_inventory()
        external_servers = self.get_external_servers()
        governance = self.get_governance()
        return {
            "inventory": {
                "tools": len(inventory.get("tools", [])),
                "resources": len(inventory.get("resources", [])),
                "prompts": len(inventory.get("prompts", [])),
            },
            "external_servers": {
                "profiles": len(external_servers),
                "discovery_snapshots": sum(1 for item in external_servers if item.get("discovery_snapshot")),
            },
            "governance": {
                "rules": len(governance),
            },
        }

    def get_inventory(self) -> dict[str, Any]:
        self._require_allowed("mcp.inventory.list.local")
        manifest = self.manifest_provider() or {}
        inventory = dict(manifest)
        inventory["server_id"] = manifest.get("server_id", "local:tldw_chatbook")
        inventory["tools"] = list(manifest.get("tools", []))
        inventory["resources"] = list(manifest.get("resources", []))
        inventory["prompts"] = list(manifest.get("prompts", []))
        return inventory

    def get_external_servers(self) -> list[dict[str, Any]]:
        self._require_allowed("mcp.external_profiles.list.local")
        servers: list[dict[str, Any]] = []
        client = self.client
        active_sessions = getattr(client, "sessions", {}) if client is not None else {}
        for profile in self.store.list_profiles():
            servers.append(
                {
                    **profile.to_dict(),
                    "discovery_snapshot": self.store.get_discovery_snapshot(profile.profile_id),
                    "is_connected": profile.profile_id in active_sessions,
                }
            )
        return servers

    def save_external_profile(
        self,
        profile: Mapping[str, Any] | LocalExternalMCPProfile,
    ) -> dict[str, Any]:
        self._require_allowed("mcp.external_profiles.configure.local")
        strict_input = (
            profile.to_input_dict()
            if isinstance(profile, LocalExternalMCPProfile)
            else profile
        )
        record = LocalExternalMCPProfile.from_input_dict(strict_input)
        return self.store.save_profile(record).to_dict()

    async def connect_profile(self, profile_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.external_profiles.launch.local")
        profile = self.store.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"Unknown profile_id: {profile_id}")

        client = self._get_client()
        resolved_env = self._build_spawn_env(profile)
        connected = await client.connect_to_server(
            profile.profile_id,
            profile.command,
            args=list(profile.args),
            env=resolved_env,
        )
        if connected is False:
            raise RuntimeError(f"Failed to connect profile: {profile.profile_id}")

        snapshot = await client.describe_server(profile.profile_id)
        if not self._has_capabilities(snapshot):
            await self._disconnect_best_effort(client, profile.profile_id)
            raise RuntimeError(f"Connected profile '{profile.profile_id}' returned no discoverable capabilities")
        self.store.save_discovery_snapshot(profile.profile_id, snapshot)
        return snapshot

    async def disconnect_profile(self, profile_id: str) -> bool:
        self._require_allowed("mcp.external_profiles.launch.local")
        client = self._get_client()
        return await client.disconnect_from_server(profile_id)

    async def test_external_profile(self, profile_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.external_profiles.trigger.local")
        snapshot = await self._describe_profile(profile_id, keep_connected=False)
        return {
            "ok": True,
            "profile_id": profile_id,
            "tools": len(snapshot.get("tools", [])),
            "resources": len(snapshot.get("resources", [])),
            "prompts": len(snapshot.get("prompts", [])),
        }

    async def refresh_external_profile(self, profile_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.external_profiles.observe.local")
        return await self._describe_profile(profile_id, keep_connected=True)

    def delete_external_profile(self, profile_id: str) -> bool:
        self._require_allowed("mcp.external_profiles.configure.local")
        return self.store.delete_profile(profile_id)

    def get_governance(self) -> list[dict[str, Any]]:
        self._require_allowed("mcp.governance.list.local")
        return [rule.to_dict() for rule in self.store.list_governance_rules()]

    def list_approval_requests(
        self,
        status: str | None = None,
        resolved_action_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self._require_allowed("mcp.governance.observe.local")
        normalized_status = str(status or "").strip()
        normalized_resolved_action_id = str(resolved_action_id or "").strip()
        requests = [request.to_dict() for request in self.store.list_approval_requests()]
        if normalized_status:
            requests = [
                request
                for request in requests
                if request.get("status") == normalized_status
            ]
        if normalized_resolved_action_id:
            requests = [
                request
                for request in requests
                if request.get("resolved_action_id") == normalized_resolved_action_id
            ]
        return requests

    def get_advanced(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        governance = self.get_governance()
        approval_requests = self.list_approval_requests()
        governance_summary = {
            "rules": len(governance),
            "deny_rules": sum(1 for rule in governance if rule.get("decision") == "deny"),
            "allow_rules": sum(1 for rule in governance if rule.get("decision") == "allow"),
        }
        ask_rules = sum(1 for rule in governance if rule.get("decision") == "ask")
        pending_approvals = sum(1 for request in approval_requests if request.get("status") == "pending")
        if ask_rules:
            governance_summary["ask_rules"] = ask_rules
        if pending_approvals:
            governance_summary["pending_approvals"] = pending_approvals
        payload = {
            "source": "local",
            "section": "advanced",
            "runtime_status": self.runtime_delegate.get_status(),
            "runtime_health": self.runtime_delegate.get_runtime_health(),
            "protocol": self.runtime_delegate.get_protocol_capabilities(),
            "protocol_diagnostics": self.runtime_delegate.get_protocol_diagnostics(),
            "governance": governance_summary,
        }
        recent_activity = self._recent_runtime_activity_entries(limit=5)
        if recent_activity:
            payload["recent_activity_count"] = len(
                self.store.list_runtime_activity(limit=self._runtime_activity_limit)
            )
            payload["recent_activity"] = recent_activity
        return payload

    def save_governance_rule(
        self,
        rule: Mapping[str, Any] | LocalGovernanceRule,
    ) -> dict[str, Any]:
        self._require_allowed("mcp.governance.configure.local")
        record = rule if isinstance(rule, LocalGovernanceRule) else LocalGovernanceRule.from_dict(rule)
        return self.store.save_governance_rule(record).to_dict()

    def delete_governance_rule(self, rule_id: str) -> bool:
        self._require_allowed("mcp.governance.configure.local")
        return self.store.delete_governance_rule(str(rule_id or ""))

    def preview_governance_decision(self, capability_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.governance.observe.local")
        normalized_capability_id = str(capability_id or "").strip()
        matched_rule = self._find_governance_rule(normalized_capability_id)
        return {
            "source": "local",
            "capability_id": normalized_capability_id,
            "decision": matched_rule.decision if matched_rule is not None else "inherit",
            "matched_rule_id": matched_rule.rule_id if matched_rule is not None else None,
            "notes": matched_rule.notes if matched_rule is not None else None,
        }

    def preview_runtime_access(self, action_name: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._require_allowed("mcp.governance.observe.local")
        normalized_action_name = str(action_name or "").strip()
        normalized_payload = dict(payload or {})
        if normalized_action_name == "runtime.batch":
            requests = normalized_payload.get("requests")
            if not isinstance(requests, list):
                raise ValueError("Runtime batch preview requires 'requests'.")
            return {
                "source": "local",
                "action_name": normalized_action_name,
                "items": [
                    self._governance_preview_for_runtime_action(
                        "runtime.request",
                        {
                            "method": request.get("method"),
                            "params": request.get("params") if isinstance(request.get("params"), Mapping) else {},
                        },
                    )
                    for request in requests
                    if isinstance(request, Mapping)
                ],
            }
        return self._governance_preview_for_runtime_action(normalized_action_name, normalized_payload)

    def approve_approval_request(self, request_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.governance.approve.local")
        resolved = self.store.resolve_approval_request(str(request_id or ""), "approved")
        if resolved is None:
            raise KeyError(f"Unknown approval request: {request_id}")
        return resolved.to_dict()

    def deny_approval_request(self, request_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.governance.approve.local")
        resolved = self.store.resolve_approval_request(str(request_id or ""), "denied")
        if resolved is None:
            raise KeyError(f"Unknown approval request: {request_id}")
        return resolved.to_dict()

    def delete_approval_request(self, request_id: str) -> bool:
        self._require_allowed("mcp.governance.approve.local")
        return self.store.delete_approval_request(str(request_id or ""))

    def get_runtime_status(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        return {
            "source": "local",
            "status": self.runtime_delegate.get_status(),
        }

    def get_runtime_health(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        return {
            "source": "local",
            "health": self.runtime_delegate.get_runtime_health(),
        }

    def get_runtime_activity(self, limit: int = 20) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        normalized_limit = max(1, min(int(limit or 20), self._runtime_activity_limit))
        return {
            "source": "local",
            "limit": normalized_limit,
            "entries": self._recent_runtime_activity_entries(limit=normalized_limit),
        }

    def get_runtime_protocol_diagnostics(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        return {
            "source": "local",
            "diagnostics": self.runtime_delegate.get_protocol_diagnostics(),
        }

    async def run_runtime_request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.trigger.local")
        normalized_method = str(method or "").strip()
        normalized_params = dict(params or {})
        try:
            governance = self._require_runtime_governance_allowed(
                "runtime.request",
                {"method": normalized_method, "params": normalized_params},
            )
        except PermissionError as exc:
            governance = self._governance_preview_for_runtime_action(
                "runtime.request",
                {"method": normalized_method, "params": normalized_params},
            )
            self._record_runtime_activity(
                action_name="runtime.request",
                target=normalized_method,
                governance=governance,
                ok=False,
                blocked=True,
                error=str(exc),
            )
            raise
        result = await self.runtime_delegate.request(normalized_method, normalized_params)
        self._record_runtime_activity(
            action_name="runtime.request",
            target=normalized_method,
            governance=governance,
            ok=True,
        )
        return {
            "source": "local",
            "method": normalized_method,
            "params": normalized_params,
            "result": result,
            "governance": self._compact_governance_preview(governance),
        }

    async def run_runtime_batch(self, requests: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.trigger.local")
        normalized_requests = [dict(request) for request in requests]
        results: list[dict[str, Any]] = []
        for index, request in enumerate(normalized_requests):
            method = str(request.get("method") or "").strip()
            params = request.get("params") if isinstance(request.get("params"), Mapping) else {}
            governance = self._governance_preview_for_runtime_action(
                "runtime.request",
                {"method": method, "params": params},
            )
            if governance["decision"] == "deny":
                self._record_runtime_activity(
                    action_name="runtime.request",
                    target=method,
                    governance=governance,
                    ok=False,
                    blocked=True,
                    error=f"Denied by local governance: {governance['resolved_action_id']}",
                )
                results.append(
                    {
                        "index": index,
                        "method": method,
                        "ok": False,
                        "blocked": True,
                        "error": f"Denied by local governance: {governance['resolved_action_id']}",
                        "governance": self._compact_governance_preview(governance),
                    }
                )
                continue
            if governance["decision"] == "ask" and governance.get("approval_status") != "approved":
                if governance.get("approval_request_id") is None:
                    governance = self._create_pending_runtime_approval("runtime.request", {"method": method, "params": params})
                    error_message = f"Approval required: {governance.get('approval_request_id')}"
                else:
                    error_message = self._approval_error_message(governance)
                self._record_runtime_activity(
                    action_name="runtime.request",
                    target=method,
                    governance=governance,
                    ok=False,
                    blocked=True,
                    error=error_message,
                )
                results.append(
                    {
                        "index": index,
                        "method": method,
                        "ok": False,
                        "blocked": True,
                        "error": error_message,
                        "governance": self._compact_governance_preview(governance),
                    }
                )
                continue
            result = await self.runtime_delegate.request(method, params)
            self._record_runtime_activity(
                action_name="runtime.request",
                target=method,
                governance=governance,
                ok=True,
            )
            results.append(
                {
                    "index": index,
                    "method": method,
                    "ok": True,
                    "result": result,
                    "governance": self._compact_governance_preview(governance),
                }
            )
        return {
            "source": "local",
            "results": results,
        }

    async def execute_tool(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.trigger.local")
        normalized_tool_name = str(tool_name or "").strip()
        normalized_arguments = dict(arguments or {})
        try:
            governance = self._require_runtime_governance_allowed(
                "tool.execute",
                {"tool_name": normalized_tool_name, "arguments": normalized_arguments},
            )
        except PermissionError as exc:
            governance = self._governance_preview_for_runtime_action(
                "tool.execute",
                {"tool_name": normalized_tool_name, "arguments": normalized_arguments},
            )
            self._record_runtime_activity(
                action_name="tool.execute",
                target=normalized_tool_name,
                governance=governance,
                ok=False,
                blocked=True,
                error=str(exc),
            )
            raise
        result = await self.runtime_delegate.execute_tool(normalized_tool_name, normalized_arguments)
        self._record_runtime_activity(
            action_name="tool.execute",
            target=normalized_tool_name,
            governance=governance,
            ok=True,
        )
        return {
            "source": "local",
            "tool_name": normalized_tool_name,
            "result": result,
            "governance": self._compact_governance_preview(governance),
        }

    async def read_resource(self, resource_uri: str) -> dict[str, Any]:
        self._require_allowed("mcp.inventory.observe.local")
        normalized_resource_uri = str(resource_uri or "").strip()
        try:
            governance = self._require_runtime_governance_allowed(
                "resource.read",
                {"resource_uri": normalized_resource_uri},
            )
        except PermissionError as exc:
            governance = self._governance_preview_for_runtime_action(
                "resource.read",
                {"resource_uri": normalized_resource_uri},
            )
            self._record_runtime_activity(
                action_name="resource.read",
                target=normalized_resource_uri,
                governance=governance,
                ok=False,
                blocked=True,
                error=str(exc),
            )
            raise
        result = await self.runtime_delegate.read_resource(normalized_resource_uri)
        self._record_runtime_activity(
            action_name="resource.read",
            target=normalized_resource_uri,
            governance=governance,
            ok=True,
        )
        return {
            "source": "local",
            "resource_uri": normalized_resource_uri,
            "result": result,
            "governance": self._compact_governance_preview(governance),
        }

    async def get_prompt(self, prompt_name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._require_allowed("mcp.inventory.observe.local")
        normalized_prompt_name = str(prompt_name or "").strip()
        normalized_arguments = dict(arguments or {})
        try:
            governance = self._require_runtime_governance_allowed(
                "prompt.get",
                {"prompt_name": normalized_prompt_name, "arguments": normalized_arguments},
            )
        except PermissionError as exc:
            governance = self._governance_preview_for_runtime_action(
                "prompt.get",
                {"prompt_name": normalized_prompt_name, "arguments": normalized_arguments},
            )
            self._record_runtime_activity(
                action_name="prompt.get",
                target=normalized_prompt_name,
                governance=governance,
                ok=False,
                blocked=True,
                error=str(exc),
            )
            raise
        messages = await self.runtime_delegate.get_prompt(normalized_prompt_name, normalized_arguments)
        self._record_runtime_activity(
            action_name="prompt.get",
            target=normalized_prompt_name,
            governance=governance,
            ok=True,
        )
        return {
            "source": "local",
            "prompt_name": normalized_prompt_name,
            "arguments": normalized_arguments,
            "messages": messages,
            "governance": self._compact_governance_preview(governance),
        }

    def _get_client(self) -> MCPClient:
        if self.client is None:
            self.client = MCPClient()
        return self.client

    def _build_spawn_env(self, profile: LocalExternalMCPProfile) -> dict[str, str]:
        resolved_env = {
            key: value
            for key in _SPAWN_ENV_BASELINE_KEYS
            if (value := os.environ.get(key)) not in (None, "")
        }
        resolved_env.update(profile.legacy_env_literals)
        resolved_env.update(profile.env_literals)
        for key, placeholder in profile.env_placeholders.items():
            match = _ENV_PLACEHOLDER_PATTERN.fullmatch(placeholder)
            if not match:
                raise RuntimeError(f"Invalid env placeholder for '{key}': {placeholder}")
            env_key = match.group("braced") or match.group("plain")
            env_value = os.environ.get(env_key)
            if env_value in (None, ""):
                raise RuntimeError(f"Missing required environment variable '{env_key}' for profile '{profile.profile_id}'")
            resolved_env[key] = env_value
        return resolved_env

    def _has_capabilities(self, snapshot: Mapping[str, Any]) -> bool:
        return any(snapshot.get(section) for section in ("tools", "resources", "prompts"))

    async def _disconnect_best_effort(self, client: MCPClient, profile_id: str) -> None:
        disconnect = getattr(client, "disconnect_from_server", None)
        if disconnect is None:
            return
        try:
            await disconnect(profile_id)
        except Exception:
            return

    async def _describe_profile(self, profile_id: str, *, keep_connected: bool) -> dict[str, Any]:
        profile = self.store.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"Unknown profile_id: {profile_id}")

        client = self._get_client()
        sessions = getattr(client, "sessions", {})
        was_connected = profile_id in sessions
        if not was_connected:
            await self.connect_profile(profile_id)
        snapshot = await client.describe_server(profile_id)
        self.store.save_discovery_snapshot(profile_id, snapshot)
        if (not was_connected) or (was_connected and not keep_connected):
            await self._disconnect_best_effort(client, profile_id)
        return snapshot

    def _find_governance_rule(self, action_id: str) -> LocalGovernanceRule | None:
        return next(
            (
                rule
                for rule in self.store.list_governance_rules()
                if rule.capability_id == action_id
            ),
            None,
        )

    def _governance_preview_for_runtime_action(
        self,
        action_name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        resolved_action_id, fallback_action_ids = self._resolve_runtime_action_ids(action_name, payload)
        matched_rule = self._find_governance_rule(resolved_action_id)
        if matched_rule is None:
            matched_rule = next(
                (
                    rule
                    for fallback_action_id in fallback_action_ids
                    if (rule := self._find_governance_rule(fallback_action_id)) is not None
                ),
                None,
            )
        capability_entry = CAPABILITY_REGISTRY.get(resolved_action_id)
        governance = {
            "source": "local",
            "action_name": action_name,
            "resolved_action_id": resolved_action_id,
            "registry_capability_id": capability_entry.capability_id if capability_entry is not None else None,
            "decision": matched_rule.decision if matched_rule is not None else "inherit",
            "matched_rule_id": matched_rule.rule_id if matched_rule is not None else None,
            "notes": matched_rule.notes if matched_rule is not None else None,
        }
        if governance["decision"] == "ask":
            approval_request = self._find_latest_approval_request(
                self._approval_fingerprint(str(governance["resolved_action_id"]), payload)
            )
            governance["approval_request_id"] = approval_request.request_id if approval_request is not None else None
            governance["approval_status"] = approval_request.status if approval_request is not None else None
        return governance

    def _find_latest_approval_request(self, payload_fingerprint: str) -> LocalApprovalRequest | None:
        matches = [
            request
            for request in self.store.list_approval_requests()
            if request.payload_fingerprint == payload_fingerprint
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda request: request.updated_at or request.created_at or datetime.min.replace(tzinfo=timezone.utc),
        )

    def _create_pending_runtime_approval(
        self,
        action_name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        governance = self._governance_preview_for_runtime_action(action_name, payload)
        saved_request = self.store.save_approval_request(
            LocalApprovalRequest(
                request_id=f"approval-{uuid4().hex[:12]}",
                action_name=action_name,
                resolved_action_id=str(governance["resolved_action_id"]),
                registry_capability_id=str(governance["registry_capability_id"] or "") or None,
                payload=dict(payload),
                payload_fingerprint=self._approval_fingerprint(str(governance["resolved_action_id"]), payload),
                status="pending",
                matched_rule_id=str(governance["matched_rule_id"] or "") or None,
                notes=str(governance["notes"] or "") or None,
            )
        )
        governance["approval_request_id"] = saved_request.request_id
        governance["approval_status"] = saved_request.status
        return governance

    @staticmethod
    def _approval_error_message(governance: Mapping[str, Any]) -> str:
        if governance.get("approval_status") == "pending":
            return f"Approval pending: {governance.get('approval_request_id')}"
        if governance.get("approval_status") == "denied":
            return f"Approval denied: {governance.get('approval_request_id')}"
        return f"Approval required: {governance.get('approval_request_id')}"

    @staticmethod
    def _approval_fingerprint(resolved_action_id: str, payload: Mapping[str, Any]) -> str:
        canonical_payload = json.dumps(
            {
                "resolved_action_id": resolved_action_id,
                "payload": payload,
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

    def _require_runtime_governance_allowed(
        self,
        action_name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        governance = self._governance_preview_for_runtime_action(action_name, payload)
        if governance["decision"] == "deny":
            raise PermissionError(f"Denied by local governance: {governance['resolved_action_id']}")
        if governance["decision"] == "ask":
            if governance.get("approval_status") == "approved":
                return governance
            if governance.get("approval_request_id") is None:
                governance = self._create_pending_runtime_approval(action_name, payload)
                raise PermissionError(f"Approval required: {governance.get('approval_request_id')}")
            raise PermissionError(self._approval_error_message(governance))
        return governance

    @staticmethod
    def _compact_governance_preview(governance: Mapping[str, Any]) -> dict[str, Any]:
        compact = {
            "resolved_action_id": governance.get("resolved_action_id"),
            "registry_capability_id": governance.get("registry_capability_id"),
            "decision": governance.get("decision"),
            "matched_rule_id": governance.get("matched_rule_id"),
            "notes": governance.get("notes"),
        }
        if governance.get("decision") == "ask" or governance.get("approval_request_id") is not None or governance.get(
            "approval_status"
        ) is not None:
            compact["approval_request_id"] = governance.get("approval_request_id")
            compact["approval_status"] = governance.get("approval_status")
        return compact

    def _recent_runtime_activity_entries(self, *, limit: int) -> list[dict[str, Any]]:
        normalized_limit = max(1, min(int(limit or 20), self._runtime_activity_limit))
        return self.store.list_runtime_activity(limit=normalized_limit)

    def _record_runtime_activity(
        self,
        *,
        action_name: str,
        target: str,
        governance: Mapping[str, Any] | None,
        ok: bool,
        blocked: bool = False,
        error: str | None = None,
    ) -> None:
        entry = {
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "action_name": str(action_name or "").strip(),
            "target": str(target or "").strip(),
            "ok": bool(ok),
            "blocked": bool(blocked),
            "error": str(error) if error is not None else None,
            "resolved_action_id": governance.get("resolved_action_id") if governance is not None else None,
            "decision": governance.get("decision") if governance is not None else None,
            "matched_rule_id": governance.get("matched_rule_id") if governance is not None else None,
            "approval_request_id": governance.get("approval_request_id") if governance is not None else None,
            "approval_status": governance.get("approval_status") if governance is not None else None,
        }
        self.store.record_runtime_activity(entry, limit=self._runtime_activity_limit)

    def _resolve_runtime_action_ids(
        self,
        action_name: str,
        payload: Mapping[str, Any],
    ) -> tuple[str, tuple[str, ...]]:
        if action_name == "tool.execute":
            tool_name = str(payload.get("tool_name") or "").strip()
            resolved_action_id = _TOOL_ACTION_IDS.get(tool_name, "mcp.runtime.trigger.local")
            return resolved_action_id, ("mcp.runtime.trigger.local",)
        if action_name == "resource.read":
            resource_uri = str(payload.get("resource_uri") or "").strip()
            resolved_action_id = next(
                (action_id for prefix, action_id in _RESOURCE_ACTION_IDS if resource_uri.startswith(prefix)),
                "mcp.inventory.observe.local",
            )
            return resolved_action_id, ("mcp.inventory.observe.local",)
        if action_name == "prompt.get":
            return "prompts.preview.local", ("mcp.inventory.observe.local",)
        if action_name == "runtime.status.get":
            return "mcp.runtime.observe.local", ()
        if action_name == "runtime.request":
            method = str(payload.get("method") or "").strip()
            params = payload.get("params") if isinstance(payload.get("params"), Mapping) else {}
            return self._resolve_runtime_request_action_ids(method, params)
        raise ValueError(f"Unsupported local runtime action preview: {action_name}")

    def _resolve_runtime_request_action_ids(
        self,
        method: str,
        params: Mapping[str, Any],
    ) -> tuple[str, tuple[str, ...]]:
        if method in _REQUEST_METHOD_ACTION_IDS:
            return _REQUEST_METHOD_ACTION_IDS[method], ()
        if method == "tools/call":
            tool_name = str(params.get("name") or params.get("tool_name") or "").strip()
            resolved_action_id = _TOOL_ACTION_IDS.get(tool_name, "mcp.runtime.trigger.local")
            return resolved_action_id, ("mcp.runtime.trigger.local",)
        if method == "resources/read":
            resource_uri = str(params.get("uri") or params.get("resource_uri") or "").strip()
            resolved_action_id = next(
                (action_id for prefix, action_id in _RESOURCE_ACTION_IDS if resource_uri.startswith(prefix)),
                "mcp.inventory.observe.local",
            )
            return resolved_action_id, ("mcp.inventory.observe.local",)
        if method == "prompts/get":
            return "prompts.preview.local", ("mcp.inventory.observe.local",)
        return "mcp.runtime.trigger.local", ()

    def _require_allowed(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(
            action_id=action_id,
            runtime_state_override=RuntimeSourceState(active_source="local"),
        )
