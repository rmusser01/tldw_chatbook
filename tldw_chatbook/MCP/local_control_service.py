from __future__ import annotations

from typing import Any, Callable, Mapping

from .client import MCPClient
from .local_store import LocalExternalMCPProfile, LocalGovernanceRule, LocalMCPStore


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
    ) -> None:
        self.store = store
        self.client = client
        self.manifest_provider = manifest_provider or _default_manifest_provider

    def get_overview(self) -> dict[str, Any]:
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
        manifest = self.manifest_provider() or {}
        return {
            "server_id": manifest.get("server_id", "local:tldw_chatbook"),
            "tools": list(manifest.get("tools", [])),
            "resources": list(manifest.get("resources", [])),
            "prompts": list(manifest.get("prompts", [])),
        }

    def get_external_servers(self) -> list[dict[str, Any]]:
        servers: list[dict[str, Any]] = []
        for profile in self.store.list_profiles():
            servers.append(
                {
                    **profile.to_dict(),
                    "discovery_snapshot": self.store.get_discovery_snapshot(profile.profile_id),
                }
            )
        return servers

    def save_external_profile(
        self,
        profile: Mapping[str, Any] | LocalExternalMCPProfile,
    ) -> dict[str, Any]:
        record = profile if isinstance(profile, LocalExternalMCPProfile) else LocalExternalMCPProfile.from_dict(profile)
        return self.store.save_profile(record).to_dict()

    async def connect_profile(self, profile_id: str) -> dict[str, Any]:
        profile = self.store.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"Unknown profile_id: {profile_id}")

        client = self._get_client()
        connected = await client.connect_to_server(
            profile.profile_id,
            profile.command,
            args=list(profile.args),
            env=dict(profile.env),
        )
        if connected is False:
            raise RuntimeError(f"Failed to connect profile: {profile.profile_id}")

        snapshot = await client.describe_server(profile.profile_id)
        self.store.save_discovery_snapshot(profile.profile_id, snapshot)
        return snapshot

    def get_governance(self) -> list[dict[str, Any]]:
        return [rule.to_dict() for rule in self.store.list_governance_rules()]

    def save_governance_rule(
        self,
        rule: Mapping[str, Any] | LocalGovernanceRule,
    ) -> dict[str, Any]:
        record = rule if isinstance(rule, LocalGovernanceRule) else LocalGovernanceRule.from_dict(rule)
        return self.store.save_governance_rule(record).to_dict()

    def _get_client(self) -> MCPClient:
        if self.client is None:
            self.client = MCPClient()
        return self.client
