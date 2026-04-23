from __future__ import annotations

import os
import re
from typing import Any, Callable, Mapping

from tldw_chatbook.runtime_policy.types import RuntimeSourceState

from .client import MCPClient
from .local_store import LocalExternalMCPProfile, LocalGovernanceRule, LocalMCPStore

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
    ) -> None:
        self.store = store
        self.client = client
        self.manifest_provider = manifest_provider or _default_manifest_provider
        self.policy_enforcer = policy_enforcer

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

    def save_governance_rule(
        self,
        rule: Mapping[str, Any] | LocalGovernanceRule,
    ) -> dict[str, Any]:
        self._require_allowed("mcp.governance.configure.local")
        record = rule if isinstance(rule, LocalGovernanceRule) else LocalGovernanceRule.from_dict(rule)
        return self.store.save_governance_rule(record).to_dict()

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

    def _require_allowed(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(
            action_id=action_id,
            runtime_state_override=RuntimeSourceState(active_source="local"),
        )
