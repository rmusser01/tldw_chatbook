"""Server-backed sync send/get transport service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from tldw_chatbook.Sync_Interop.validation import (
    validate_outgoing_envelope_scope,
    validate_pull_pagination_state,
    validate_pulled_response_scope,
    validate_push_response_scope,
)

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    ClientChangesPayload,
    SyncV2ConflictResolveRequest,
    SyncV2DatasetEnrollRequest,
    SyncV2DeviceRegisterRequest,
    SyncV2Envelope,
    SyncV2KeyRecoveryBundleRequest,
    SyncV2PushRequest,
    TLDWAPIClient,
)


class ServerSyncService:
    """Policy-gated access to the server sync transport endpoints."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
        state_repository: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer
        self.state_repository = state_repository

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerSyncService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerSyncService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server sync operations.")

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Sync action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [ServerSyncService._dump(item) for item in response]
        if isinstance(response, dict):
            return {key: ServerSyncService._dump(value) for key, value in response.items()}
        return response

    @staticmethod
    def _coerce_payload(request_data: ClientChangesPayload | Mapping[str, Any]) -> ClientChangesPayload:
        if isinstance(request_data, ClientChangesPayload):
            return request_data
        return ClientChangesPayload.model_validate(request_data)

    async def send_changes(
        self,
        request_data: ClientChangesPayload | Mapping[str, Any],
    ) -> dict[str, Any]:
        self._enforce("sync.changes.launch.server")
        payload = self._coerce_payload(request_data)
        return self._dump(await self._require_client().send_sync_changes(payload))

    async def get_changes(
        self,
        *,
        client_id: str,
        since_change_id: int = 0,
    ) -> dict[str, Any]:
        self._enforce("sync.changes.observe.server")
        return self._dump(
            await self._require_client().get_sync_changes(
                client_id=client_id,
                since_change_id=since_change_id,
            )
        )

    async def run_v2_dry_run(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        display_name: str,
        domains: list[str] | None = None,
        client_version: str | None = None,
        scope_type: str = "personal",
        encryption_policy: str = "client_private_v1",
    ) -> dict[str, Any]:
        """Negotiate Sync v2 state without sending or applying content envelopes."""

        if self.state_repository is None:
            raise ValueError("Sync state repository is required for Sync v2 dry-run.")
        if not server_profile_id:
            raise ValueError("server_profile_id is required")
        if not display_name:
            raise ValueError("display_name is required")

        self._enforce("sync.v2.dry_run.server")
        client = self._require_client()
        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        existing_device_id = profile["device_id"] if profile else None
        existing_dataset_id = profile["dataset_id"] if profile else None

        requested_domains = domains or ["notes", "chat", "workspaces", "source_cache", "media"]
        capabilities = await client.get_sync_v2_capabilities()
        capabilities_record = self._dump(capabilities)
        supported_domains = set(capabilities_record.get("supported_domains", []))
        sync_domains = [domain for domain in requested_domains if domain in supported_domains]
        if not sync_domains:
            raise ValueError("Server does not advertise any requested Sync v2 domains.")

        device = await client.register_sync_v2_device(
            SyncV2DeviceRegisterRequest(
                device_id=existing_device_id,
                display_name=display_name,
                client_type="chatbook",
                client_version=client_version,
                supported_domains=sync_domains,
                capabilities={
                    "dry_run": True,
                    "protocol_version": 2,
                },
            )
        )
        device_record = self._dump(device)
        device_id = str(device_record["device_id"])

        dataset = await client.enroll_sync_v2_dataset(
            SyncV2DatasetEnrollRequest(
                dataset_id=existing_dataset_id,
                device_id=device_id,
                scope_type=scope_type,
                workspace_id=workspace_scope,
                domains=sync_domains,
                encryption_policy=encryption_policy,
                metadata={"dry_run": True},
            )
        )
        dataset_record = self._dump(dataset)
        dataset_id = str(dataset_record["dataset_id"])

        pushed = await client.push_sync_v2_envelopes(
            SyncV2PushRequest(dataset_id=dataset_id, device_id=device_id, envelopes=[])
        )
        push_record = self._dump(pushed)
        validate_push_response_scope(
            dataset_id=dataset_id,
            response_dataset_id=push_record.get("dataset_id"),
            submitted_client_envelope_ids=[],
            accepted=push_record.get("accepted", []),
            rejected=push_record.get("rejected", []),
            conflicts=push_record.get("conflicts", []),
        )
        cursor_record = self.state_repository.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain="sync_v2",
            remote_collection=dataset_id,
        )
        pulled = await client.pull_sync_v2_envelopes(
            dataset_id=dataset_id,
            device_id=device_id,
            cursor=cursor_record.cursor,
            domains=sync_domains,
            page_size=1,
            include_own_changes=False,
        )
        pull_record = self._dump(pulled)
        pulled_envelopes = [
            SyncV2Envelope.model_validate(envelope)
            for envelope in pull_record.get("envelopes", [])
        ]
        validate_pulled_response_scope(
            dataset_id=dataset_id,
            response_dataset_id=pull_record.get("dataset_id"),
            envelopes=pulled_envelopes,
            domains=sync_domains,
            excluded_device_id=device_id,
        )
        validate_pull_pagination_state(
            has_more=pull_record.get("has_more", False),
            next_cursor=pull_record.get("next_cursor"),
            envelope_count=len(pulled_envelopes),
        )

        next_cursor = pull_record.get("next_cursor") or push_record.get("next_cursor") or cursor_record.cursor
        dataset_cursors = dict(dataset_record.get("cursors") or {})
        if next_cursor is not None:
            dataset_cursors["sync_v2"] = next_cursor
            self.state_repository.set_remote_pull_cursor(
                source_authority="server",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                domain="sync_v2",
                remote_collection=dataset_id,
                cursor=next_cursor,
            )

        result = {
            "dry_run": True,
            "server_profile_id": server_profile_id,
            "workspace_scope": workspace_scope,
            "device_id": device_id,
            "dataset_id": dataset_id,
            "domains": sync_domains,
            "pushed_envelopes": len(push_record.get("accepted", [])),
            "pulled_envelopes": len(pull_record.get("envelopes", [])),
            "next_cursor": next_cursor,
            "key_setup_required": bool(dataset_record.get("key_setup_required", False)),
        }
        self.state_repository.set_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            profile_mode="local_first",
            device_id=device_id,
            dataset_id=dataset_id,
            dataset_cursors=dataset_cursors,
            capabilities=capabilities_record,
            dry_run_metadata=result,
            last_error=None,
        )
        return result

    async def store_v2_recovery_bundle(
        self,
        *,
        dataset_id: str,
        device_id: str | None = None,
        wrapped_key_blob: str,
        kdf_metadata: Mapping[str, Any],
        recovery_hint: str | None = None,
        key_purpose: str = "dataset_recovery",
        rotation_of_key_record_id: str | None = None,
    ) -> dict[str, Any]:
        """Store opaque Sync v2 key recovery material on the server."""

        self._enforce("sync.v2.keys.store.server")
        request = SyncV2KeyRecoveryBundleRequest(
            dataset_id=dataset_id,
            device_id=device_id,
            key_purpose=key_purpose,
            wrapped_key_blob=wrapped_key_blob,
            kdf_metadata=dict(kdf_metadata),
            recovery_hint=recovery_hint,
            rotation_of_key_record_id=rotation_of_key_record_id,
        )
        return self._dump(await self._require_client().store_sync_v2_key_recovery_bundle(request))

    async def list_v2_recovery_bundles(
        self,
        *,
        dataset_id: str,
        device_id: str | None = None,
        key_purpose: str | None = "dataset_recovery",
    ) -> dict[str, Any]:
        """Fetch opaque Sync v2 key recovery material from the server."""

        self._enforce("sync.v2.keys.retrieve.server")
        return self._dump(
            await self._require_client().list_sync_v2_key_recovery_bundles(
                dataset_id=dataset_id,
                device_id=device_id,
                key_purpose=key_purpose,
            )
        )

    async def get_v2_restore_manifest(
        self,
        *,
        dataset_ids: list[str] | None = None,
        domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch metadata-only Sync v2 restore manifest records from the server."""

        self._enforce("sync.v2.restore_manifest.observe.server")
        return self._dump(
            await self._require_client().get_sync_v2_restore_manifest(
                dataset_ids=dataset_ids,
                domains=domains,
            )
        )

    async def push_v2_envelopes(
        self,
        *,
        dataset_id: str,
        device_id: str,
        envelopes: list[SyncV2Envelope | Mapping[str, Any]],
        idempotency_key: str | None = None,
        last_known_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Push local-first Sync v2 envelopes through the policy-gated transport."""

        self._enforce("sync.v2.push.server")
        coerced_envelopes = [
            envelope
            if isinstance(envelope, SyncV2Envelope)
            else SyncV2Envelope.model_validate(envelope)
            for envelope in envelopes
        ]
        validate_outgoing_envelope_scope(
            dataset_id=dataset_id,
            device_id=device_id,
            envelopes=coerced_envelopes,
            domains=[],
        )
        request = SyncV2PushRequest(
            dataset_id=dataset_id,
            device_id=device_id,
            envelopes=coerced_envelopes,
            idempotency_key=idempotency_key,
            last_known_cursor=last_known_cursor,
        )
        response = self._dump(await self._require_client().push_sync_v2_envelopes(request))
        validate_push_response_scope(
            dataset_id=dataset_id,
            response_dataset_id=response.get("dataset_id"),
            submitted_client_envelope_ids=[
                envelope.client_envelope_id
                for envelope in coerced_envelopes
            ],
            accepted=response.get("accepted", []),
            rejected=response.get("rejected", []),
            conflicts=response.get("conflicts", []),
        )
        return response

    async def pull_v2_envelopes(
        self,
        *,
        dataset_id: str,
        device_id: str,
        cursor: str | None = None,
        domains: list[str] | None = None,
        page_size: int | None = None,
        include_own_changes: bool = False,
    ) -> dict[str, Any]:
        """Pull selected Sync v2 envelopes for restore or incremental sync."""

        self._enforce("sync.v2.restore.pull.server")
        response = self._dump(
            await self._require_client().pull_sync_v2_envelopes(
                dataset_id=dataset_id,
                device_id=device_id,
                cursor=cursor,
                domains=domains,
                page_size=page_size,
                include_own_changes=include_own_changes,
            )
        )
        envelopes = [
            SyncV2Envelope.model_validate(envelope)
            for envelope in response.get("envelopes", [])
        ]
        validate_pulled_response_scope(
            dataset_id=dataset_id,
            response_dataset_id=response.get("dataset_id"),
            envelopes=envelopes,
            domains=domains,
            excluded_device_id=None if include_own_changes else device_id,
        )
        validate_pull_pagination_state(
            has_more=response.get("has_more", False),
            next_cursor=response.get("next_cursor"),
            envelope_count=len(envelopes),
        )
        return response

    async def list_v2_conflicts(
        self,
        *,
        dataset_id: str,
        status: str = "unresolved",
    ) -> list[dict[str, Any]]:
        """List Sync v2 conflicts that remain visible until explicitly resolved."""

        self._enforce("sync.v2.conflicts.observe.server")
        return self._dump(
            await self._require_client().list_sync_v2_conflicts(
                dataset_id=dataset_id,
                status=status,
            )
        )

    async def resolve_v2_conflict(
        self,
        *,
        conflict_id: str,
        action: str,
        resolution_envelope: Mapping[str, Any] | None = None,
        resolved_by_device_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a Sync v2 conflict via the server conflict API."""

        self._enforce("sync.v2.conflicts.resolve.server")
        request = SyncV2ConflictResolveRequest(
            conflict_id=conflict_id,
            action=action,
            resolution_envelope=resolution_envelope,
            resolved_by_device_id=resolved_by_device_id,
            notes=notes,
        )
        return self._dump(await self._require_client().resolve_sync_v2_conflict(conflict_id, request))
