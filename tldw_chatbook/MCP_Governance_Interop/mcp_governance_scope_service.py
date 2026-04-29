"""Source-aware routing for Unified MCP governance control surfaces."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class MCPGovernanceBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "mcp_governance.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Remote MCP governance is unavailable in local/offline mode; use Chatbook's local MCP runtime controls instead.",
        "affected_action_ids": [],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES: list[dict[str, Any]] = []


class MCPGovernanceScopeService:
    """Present local and server MCP as one control-plane boundary with explicit source scope."""

    def __init__(self, *, server_service: Any = None, local_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.local_service = local_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: MCPGovernanceBackend | str | None) -> MCPGovernanceBackend:
        if mode is None:
            return MCPGovernanceBackend.SERVER
        if isinstance(mode, MCPGovernanceBackend):
            return mode
        try:
            return MCPGovernanceBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid MCP governance backend: {mode}") from exc

    def _require_server_service(self, mode: MCPGovernanceBackend) -> Any:
        if mode == MCPGovernanceBackend.LOCAL:
            raise ValueError("Remote MCP governance is server-only; local MCP runtime governance remains Chatbook-owned.")
        if self.server_service is None:
            raise ValueError("Server MCP governance backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _record_identifier(item: dict[str, Any]) -> Any:
        return item.get("server_id") or item.get("id") or item.get("mapping_id") or item.get("tool_name") or item.get("name")

    @classmethod
    def _normalize_record(
        cls,
        mode: MCPGovernanceBackend,
        kind: str,
        item: Any,
        *,
        source_id: Any | None = None,
    ) -> dict[str, Any]:
        if not isinstance(item, dict):
            item = {"value": item}
        record = dict(item)
        record.setdefault("backend", mode.value)
        identifier = source_id if source_id is not None else cls._record_identifier(record)
        if identifier is not None:
            record.setdefault("record_id", f"{mode.value}:{kind}:{identifier}")
        return record

    @classmethod
    def _normalize_list(cls, mode: MCPGovernanceBackend, kind: str, result: Any) -> list[dict[str, Any]]:
        if isinstance(result, dict):
            items = result.get("items") or result.get("entries") or result.get("results") or []
        else:
            items = result or []
        return [cls._normalize_record(mode, kind, item) for item in items]

    def list_unsupported_capabilities(
        self,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MCPGovernanceBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def list_external_servers(
        self,
        *,
        mode: MCPGovernanceBackend | str | None = None,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("mcp.governance.external_servers.list.server")
        result = await self._maybe_await(
            service.list_external_servers(owner_scope_type=owner_scope_type, owner_scope_id=owner_scope_id)
        )
        return self._normalize_list(normalized_mode, "mcp_external_server", result)

    async def create_external_server(
        self,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "create_external_server",
            "mcp.governance.external_servers.create.server",
            "mcp_external_server",
            request_data,
            mode=mode,
        )

    async def import_external_server(
        self,
        server_id: str,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "import_external_server",
            "mcp.governance.external_servers.create.server",
            "mcp_external_server",
            server_id,
            mode=mode,
        )

    async def update_external_server(
        self,
        server_id: str,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "update_external_server",
            "mcp.governance.external_servers.update.server",
            "mcp_external_server",
            server_id,
            request_data,
            mode=mode,
        )

    async def delete_external_server(
        self,
        server_id: str,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "delete_external_server",
            "mcp.governance.external_servers.delete.server",
            "mcp_external_server",
            server_id,
            mode=mode,
        )

    async def set_external_server_secret(
        self,
        server_id: str,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "set_external_server_secret",
            "mcp.governance.external_servers.secrets.update.server",
            "mcp_external_server_secret",
            server_id,
            request_data,
            mode=mode,
        )

    async def list_tool_catalogs(
        self,
        *,
        mode: MCPGovernanceBackend | str | None = None,
        scope_type: str,
        scope_id: int,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("mcp.governance.catalogs.list.server")
        if scope_type == "org":
            result = await self._maybe_await(service.list_org_tool_catalogs(org_id=scope_id))
        elif scope_type == "team":
            result = await self._maybe_await(service.list_team_tool_catalogs(team_id=scope_id))
        else:
            raise ValueError("MCP catalog scope_type must be 'org' or 'team'.")
        return [
            self._normalize_record(normalized_mode, "mcp_tool_catalog", item, source_id=f"{scope_type}:{item.get('id')}")
            for item in (result or [])
        ]

    async def create_tool_catalog(
        self,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
        scope_type: str,
        scope_id: int,
    ) -> dict[str, Any]:
        if scope_type == "org":
            result = await self.__getattr_dispatch(
                "create_org_tool_catalog",
                "mcp.governance.catalogs.create.server",
                "mcp_tool_catalog",
                mode=mode,
                org_id=scope_id,
                request_data=request_data,
            )
        elif scope_type == "team":
            result = await self.__getattr_dispatch(
                "create_team_tool_catalog",
                "mcp.governance.catalogs.create.server",
                "mcp_tool_catalog",
                mode=mode,
                team_id=scope_id,
                request_data=request_data,
            )
        else:
            raise ValueError("MCP catalog scope_type must be 'org' or 'team'.")
        result["record_id"] = f"{self._normalize_mode(mode).value}:mcp_tool_catalog:{scope_type}:{result.get('id')}"
        return result

    async def delete_tool_catalog(
        self,
        *,
        mode: MCPGovernanceBackend | str | None = None,
        scope_type: str,
        scope_id: int,
        catalog_id: int,
    ) -> dict[str, Any]:
        if scope_type == "org":
            return await self.__getattr_dispatch(
                "delete_org_tool_catalog",
                "mcp.governance.catalogs.delete.server",
                "mcp_tool_catalog",
                mode=mode,
                org_id=scope_id,
                catalog_id=catalog_id,
            )
        if scope_type == "team":
            return await self.__getattr_dispatch(
                "delete_team_tool_catalog",
                "mcp.governance.catalogs.delete.server",
                "mcp_tool_catalog",
                mode=mode,
                team_id=scope_id,
                catalog_id=catalog_id,
            )
        raise ValueError("MCP catalog scope_type must be 'org' or 'team'.")

    async def add_catalog_entry(
        self,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
        scope_type: str,
        scope_id: int,
        catalog_id: int,
    ) -> dict[str, Any]:
        if scope_type == "org":
            return await self.__getattr_dispatch(
                "add_org_catalog_entry",
                "mcp.governance.catalog_entries.create.server",
                "mcp_tool_catalog_entry",
                mode=mode,
                org_id=scope_id,
                catalog_id=catalog_id,
                request_data=request_data,
            )
        if scope_type == "team":
            return await self.__getattr_dispatch(
                "add_team_catalog_entry",
                "mcp.governance.catalog_entries.create.server",
                "mcp_tool_catalog_entry",
                mode=mode,
                team_id=scope_id,
                catalog_id=catalog_id,
                request_data=request_data,
            )
        raise ValueError("MCP catalog scope_type must be 'org' or 'team'.")

    async def delete_catalog_entry(
        self,
        *,
        mode: MCPGovernanceBackend | str | None = None,
        scope_type: str,
        scope_id: int,
        catalog_id: int,
        tool_name: str,
    ) -> dict[str, Any]:
        if scope_type == "org":
            return await self.__getattr_dispatch(
                "delete_org_catalog_entry",
                "mcp.governance.catalog_entries.delete.server",
                "mcp_tool_catalog_entry",
                mode=mode,
                org_id=scope_id,
                catalog_id=catalog_id,
                tool_name=tool_name,
            )
        if scope_type == "team":
            return await self.__getattr_dispatch(
                "delete_team_catalog_entry",
                "mcp.governance.catalog_entries.delete.server",
                "mcp_tool_catalog_entry",
                mode=mode,
                team_id=scope_id,
                catalog_id=catalog_id,
                tool_name=tool_name,
            )
        raise ValueError("MCP catalog scope_type must be 'org' or 'team'.")

    async def get_effective_policy(
        self,
        *,
        mode: MCPGovernanceBackend | str | None = None,
        persona_id: str | None = None,
        group_id: str | None = None,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("mcp.governance.effective_policy.detail.server")
        result = await self._maybe_await(
            service.get_effective_policy(persona_id=persona_id, group_id=group_id, org_id=org_id, team_id=team_id)
        )
        scope_id = (
            f"team:{team_id}"
            if team_id is not None
            else f"org:{org_id}"
            if org_id is not None
            else persona_id
            or group_id
            or "current_user"
        )
        return self._normalize_record(normalized_mode, "mcp_effective_policy", result, source_id=scope_id)

    async def list_tool_registry(self, *, mode: MCPGovernanceBackend | str | None = None) -> list[dict[str, Any]]:
        return await self.__getattr_dispatch(
            "list_tool_registry",
            "mcp.governance.tool_registry.list.server",
            "mcp_tool_registry_entry",
            mode=mode,
        )

    async def list_tool_registry_modules(self, *, mode: MCPGovernanceBackend | str | None = None) -> list[dict[str, Any]]:
        return await self.__getattr_dispatch(
            "list_tool_registry_modules",
            "mcp.governance.tool_registry.list.server",
            "mcp_tool_registry_module",
            mode=mode,
        )

    async def get_tool_registry_summary(self, *, mode: MCPGovernanceBackend | str | None = None) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "get_tool_registry_summary",
            "mcp.governance.tool_registry.detail.server",
            "mcp_tool_registry_summary",
            mode=mode,
        )

    async def __getattr_dispatch(self, method_name: str, action_id: str, kind: str, *args: Any, **kwargs: Any) -> Any:
        mode = kwargs.pop("mode", None)
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        result = await self._maybe_await(getattr(service, method_name)(*args, **kwargs))
        if isinstance(result, list):
            return self._normalize_list(normalized_mode, kind, result)
        if isinstance(result, dict):
            return self._normalize_record(normalized_mode, kind, result)
        return result

    async def list_permission_profiles(self, *, mode: MCPGovernanceBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.__getattr_dispatch(
            "list_permission_profiles",
            "mcp.governance.permission_profiles.list.server",
            "mcp_permission_profile",
            mode=mode,
            **kwargs,
        )

    async def list_policy_assignments(self, *, mode: MCPGovernanceBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.__getattr_dispatch(
            "list_policy_assignments",
            "mcp.governance.policy_assignments.list.server",
            "mcp_policy_assignment",
            mode=mode,
            **kwargs,
        )

    async def list_approval_policies(self, *, mode: MCPGovernanceBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.__getattr_dispatch(
            "list_approval_policies",
            "mcp.governance.approval_policies.list.server",
            "mcp_approval_policy",
            mode=mode,
            **kwargs,
        )

    async def list_capability_mappings(self, *, mode: MCPGovernanceBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.__getattr_dispatch(
            "list_capability_mappings",
            "mcp.governance.capability_mappings.list.server",
            "mcp_capability_mapping",
            mode=mode,
            **kwargs,
        )

    async def preview_capability_mapping(
        self,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "preview_capability_mapping",
            "mcp.governance.capability_mappings.preview.server",
            "mcp_capability_mapping_preview",
            request_data,
            mode=mode,
        )

    async def create_capability_mapping(
        self,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "create_capability_mapping",
            "mcp.governance.capability_mappings.create.server",
            "mcp_capability_mapping",
            request_data,
            mode=mode,
        )

    async def update_capability_mapping(
        self,
        capability_adapter_mapping_id: int,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "update_capability_mapping",
            "mcp.governance.capability_mappings.update.server",
            "mcp_capability_mapping",
            capability_adapter_mapping_id,
            request_data,
            mode=mode,
        )

    async def delete_capability_mapping(
        self,
        capability_adapter_mapping_id: int,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "delete_capability_mapping",
            "mcp.governance.capability_mappings.delete.server",
            "mcp_capability_mapping",
            capability_adapter_mapping_id,
            mode=mode,
        )

    async def create_permission_profile(
        self,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "create_permission_profile",
            "mcp.governance.permission_profiles.create.server",
            "mcp_permission_profile",
            request_data,
            mode=mode,
        )

    async def update_permission_profile(
        self,
        profile_id: int,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "update_permission_profile",
            "mcp.governance.permission_profiles.update.server",
            "mcp_permission_profile",
            profile_id,
            request_data,
            mode=mode,
        )

    async def delete_permission_profile(
        self,
        profile_id: int,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "delete_permission_profile",
            "mcp.governance.permission_profiles.delete.server",
            "mcp_permission_profile",
            profile_id,
            mode=mode,
        )

    async def create_policy_assignment(
        self,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "create_policy_assignment",
            "mcp.governance.policy_assignments.create.server",
            "mcp_policy_assignment",
            request_data,
            mode=mode,
        )

    async def update_policy_assignment(
        self,
        assignment_id: int,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "update_policy_assignment",
            "mcp.governance.policy_assignments.update.server",
            "mcp_policy_assignment",
            assignment_id,
            request_data,
            mode=mode,
        )

    async def delete_policy_assignment(
        self,
        assignment_id: int,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "delete_policy_assignment",
            "mcp.governance.policy_assignments.delete.server",
            "mcp_policy_assignment",
            assignment_id,
            mode=mode,
        )

    async def create_approval_policy(
        self,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "create_approval_policy",
            "mcp.governance.approval_policies.create.server",
            "mcp_approval_policy",
            request_data,
            mode=mode,
        )

    async def update_approval_policy(
        self,
        approval_policy_id: int,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "update_approval_policy",
            "mcp.governance.approval_policies.update.server",
            "mcp_approval_policy",
            approval_policy_id,
            request_data,
            mode=mode,
        )

    async def delete_approval_policy(
        self,
        approval_policy_id: int,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "delete_approval_policy",
            "mcp.governance.approval_policies.delete.server",
            "mcp_approval_policy",
            approval_policy_id,
            mode=mode,
        )

    async def create_approval_decision(
        self,
        request_data: Any,
        *,
        mode: MCPGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self.__getattr_dispatch(
            "create_approval_decision",
            "mcp.governance.approval_decisions.approve.server",
            "mcp_approval_decision",
            request_data,
            mode=mode,
        )

    async def observe_events(
        self,
        *,
        mode: MCPGovernanceBackend | str | None = None,
        after_event_id: str | None = None,
        event_types: list[str] | None = None,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        replay: bool = True,
    ):
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("mcp.governance.events.observe.server")
        stream_kwargs = {
            "after_event_id": after_event_id,
            "event_types": event_types,
            "replay": replay,
        }
        if owner_scope_type is not None:
            stream_kwargs["owner_scope_type"] = owner_scope_type
        if owner_scope_id is not None:
            stream_kwargs["owner_scope_id"] = owner_scope_id
        async for event in service.stream_events(**stream_kwargs):
            event_dict = dict(event) if isinstance(event, dict) else {"value": event}
            source_id = event_dict.get("event_id") or event_dict.get("id")
            yield self._normalize_record(
                normalized_mode,
                "mcp_governance_event",
                event_dict,
                source_id=source_id,
            )
