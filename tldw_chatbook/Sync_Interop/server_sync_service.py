"""Server-backed sync send/get transport service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from tldw_chatbook.Sync_Interop.sync_state import SyncV2ProfileMode
from tldw_chatbook.Sync_Interop.validation import (
    validate_outgoing_envelope_scope,
    validate_pull_pagination_state,
    validate_pulled_response_scope,
    validate_push_response_scope,
)

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError


def _bootstrap_attention_safe_blockers(blockers: tuple[str, ...]) -> bool:
    """Allow only schema/quota facts to reach the typed bootstrap response."""

    return bool(blockers) and all(
        blocker == "personal_context_schema_incompatible"
        or blocker.startswith("personal_context_quota_incompatible:")
        for blocker in blockers
    )

if TYPE_CHECKING:
    from ..tldw_api import ClientChangesPayload, SyncV2Envelope, TLDWAPIClient


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
        state_repository: Any | None = None,
    ) -> "ServerSyncService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
            state_repository=state_repository,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
        state_repository: Any | None = None,
    ) -> "ServerSyncService":
        """Build a lazy Sync service over an application server provider.

        Args:
            provider: Runtime provider used to resolve the current server client.
            policy_enforcer: Optional policy boundary for Sync operations.
            state_repository: Optional repository for Sync v2 profile and cursor state.

        Returns:
            A Sync service that retains the provider and repository without building
            or caching a client locally.
        """
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
            state_repository=state_repository,
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
        require_ui_action_allowed = getattr(
            self.policy_enforcer, "require_ui_action_allowed", None
        )
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None)
                    or "authority_denied",
                    user_message=getattr(decision, "user_message", None)
                    or "Sync action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None)
                    or "server",
                    authority_owner=getattr(decision, "authority_owner", None)
                    or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [ServerSyncService._dump(item) for item in response]
        if isinstance(response, dict):
            return {
                key: ServerSyncService._dump(value) for key, value in response.items()
            }
        return response

    @staticmethod
    def _select_advertised_domains(
        requested_domains: list[str],
        supported_domains: list[str],
    ) -> list[str]:
        selected: list[str] = []
        supported = [str(domain) for domain in supported_domains if str(domain).strip()]
        supported_set = set(supported)
        for requested_domain in requested_domains:
            requested = str(requested_domain).strip()
            if not requested:
                continue
            matches = [requested] if requested in supported_set else []
            prefix = f"{requested}."
            matches.extend(domain for domain in supported if domain.startswith(prefix))
            for domain in matches:
                if domain not in selected:
                    selected.append(domain)
        return selected

    @staticmethod
    def _coerce_payload(
        request_data: ClientChangesPayload | Mapping[str, Any],
    ) -> ClientChangesPayload:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import ClientChangesPayload

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
        profile_mode: str = SyncV2ProfileMode.LOCAL_FIRST_SYNC.value,
    ) -> dict[str, Any]:
        """Negotiate Sync v2 state without sending or applying content envelopes.

        Args:
            server_profile_id: Stable identifier for the configured server profile.
            authenticated_principal_id: Optional user or account identity for scoped state.
            workspace_scope: Optional workspace identifier for workspace-scoped datasets.
            display_name: Human-readable device name to register with the server.
            domains: Candidate domains to request for the dataset.
            client_version: Optional Chatbook client version to advertise.
            scope_type: Dataset scope type requested from the server.
            encryption_policy: Sync v2 encryption policy requested for the dataset.
            profile_mode: Sync v2 profile mode to persist with the negotiated state.

        Returns:
            Dry-run summary with negotiated device, dataset, domain, cursor, and key setup state.

        Raises:
            ValueError: If state storage is unavailable, required identifiers are missing, or
                the server does not support any requested domains.
            PolicyDeniedError: If runtime policy blocks server Sync v2 dry-run access.
        """
        requested_domains = domains or [
            "notes",
            "chat",
            "workspaces",
            "source_cache",
            "media",
        ]
        if any(
            str(domain).strip() == "personal_context"
            or str(domain).strip().startswith("personal_context.")
            for domain in requested_domains
        ):
            raise ValueError("personal_context_requires_reviewed_first_link")

        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import (
            SyncV2CapabilitiesResponse,
            SyncV2DatasetEnrollRequest,
            SyncV2DeviceRegisterRequest,
            SyncV2Envelope,
            SyncV2PushRequest,
        )

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

        capabilities = await client.get_sync_v2_capabilities()
        capabilities_model = (
            capabilities
            if isinstance(capabilities, SyncV2CapabilitiesResponse)
            else SyncV2CapabilitiesResponse.model_validate(capabilities)
        )
        capabilities_record = self._dump(capabilities_model)
        # M1 schema: model_dump() produces "domains"; fall back to "supported_domains"
        # for raw-dict responses (e.g. from test stubs that pre-date M1).
        supported_domains = (
            capabilities_record.get("domains")
            or capabilities_record.get("supported_domains")
            or []
        )
        sync_domains = self._select_advertised_domains(
            requested_domains, supported_domains
        )
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
                    "protocol_version": "sync-v2-m1",
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

        next_cursor = (
            pull_record.get("next_cursor")
            or push_record.get("next_cursor")
            or cursor_record.cursor
        )
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
            profile_mode=profile_mode,
            device_id=device_id,
            dataset_id=dataset_id,
            dataset_cursors=dataset_cursors,
            capabilities=capabilities_record,
            dry_run_metadata=result,
            last_error=None,
        )
        return result

    async def bootstrap_personal_context_link(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        display_name: str,
        wrapping_key_provider: Any,
        client_version: str | None = None,
        required_schema_version: int | None = None,
        required_quotas: Mapping[str, int] | None = None,
        expected_purge_generation: int | None = None,
    ) -> dict[str, Any]:
        """Register secure device custody and fetch a read-only canonical snapshot."""

        from ..tldw_api import (
            SyncPersonalContextBootstrapRequest,
            SyncV2CapabilitiesResponse,
            SyncV2DeviceRegisterRequest,
        )
        from .sync_readiness import (
            PERSONAL_CONTEXT_SYNC_DOMAINS,
            personal_context_sync_readiness,
        )
        from tldw_profile_core import SERIALIZED_SCHEMA_VERSION

        if not server_profile_id or not display_name:
            raise ValueError("personal_context_link_binding_invalid")
        public_key = getattr(wrapping_key_provider, "public_key_pem", None)
        if not isinstance(public_key, str) or not public_key.strip():
            raise ValueError("personal_context_device_key_unavailable")

        self._enforce("sync.v2.personal_context.bootstrap.server")
        client = self._require_client()
        capabilities_response = await client.get_sync_v2_capabilities()
        capabilities = (
            capabilities_response
            if isinstance(capabilities_response, SyncV2CapabilitiesResponse)
            else SyncV2CapabilitiesResponse.model_validate(capabilities_response)
        )
        readiness = personal_context_sync_readiness(
            capabilities, require_writable=False
        )
        if not readiness.write_enabled and not _bootstrap_attention_safe_blockers(
            readiness.blockers
        ):
            raise ValueError(
                ",".join(readiness.blockers) or "personal_context_sync_unavailable"
            )
        profile = (
            self.state_repository.get_sync_v2_profile_state(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=None,
            )
            if self.state_repository is not None
            else None
        )
        device = await client.register_sync_v2_device(
            SyncV2DeviceRegisterRequest(
                device_id=profile["device_id"] if profile else None,
                display_name=display_name,
                client_type="chatbook",
                client_version=client_version,
                supported_domains=list(PERSONAL_CONTEXT_SYNC_DOMAINS),
                capabilities={
                    "protocol_version": "sync-v2-m1",
                    "personal_context": {
                        "schema_version": readiness.negotiated_schema_version
                        or SERIALIZED_SCHEMA_VERSION,
                    },
                    "personal_context_wrapping_public_key": public_key,
                },
            )
        )
        device_id = str(self._dump(device)["device_id"])
        response = await client.bootstrap_sync_v2_personal_context(
            SyncPersonalContextBootstrapRequest(
                device_id=device_id,
                required_schema_version=required_schema_version,
                required_quotas=dict(required_quotas or {}),
                expected_purge_generation=expected_purge_generation,
            )
        )
        record = self._dump(response)
        if not isinstance(record, dict):
            raise ValueError("personal_context_bootstrap_response_invalid")
        return {
            "device_id": device_id,
            **record,
            "_sync_capabilities": {
                "max_batch_size": capabilities.max_batch_size,
            },
        }

    async def complete_personal_context_link(
        self,
        *,
        device_id: str,
        dataset_id: str,
        bootstrap_cursor: str,
    ) -> None:
        """Acknowledge the exact reviewed bootstrap after local convergence."""

        from ..tldw_api import SyncPersonalContextLinkCompleteRequest

        self._enforce("sync.v2.personal_context.complete.server")
        await self._require_client().complete_sync_v2_personal_context_link(
            SyncPersonalContextLinkCompleteRequest(
                device_id=device_id,
                dataset_id=dataset_id,
                bootstrap_cursor=bootstrap_cursor,
            )
        )

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SyncV2KeyRecoveryBundleRequest

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
        return self._dump(
            await self._require_client().store_sync_v2_key_recovery_bundle(request)
        )

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
        domains: list[str] | None = None,
        idempotency_key: str | None = None,
        last_known_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Push non-Personal-Context envelopes through the public transport.

        Args:
            dataset_id: Dataset receiving the envelopes.
            device_id: Local device identity that produced the envelopes.
            envelopes: Sync v2 envelopes or envelope dictionaries to push.
            domains: Optional domain allow-list for scoped outgoing validation.
            idempotency_key: Optional stable key for retry-safe server dispatch.
            last_known_cursor: Optional cursor known before this push.

        Returns:
            Server push response after dataset, envelope ID, and batch integrity validation.

        Raises:
            ValueError: If outgoing envelopes or the server response violate Sync v2 scope.
            PolicyDeniedError: If runtime policy blocks server Sync v2 push access.
        """
        candidate_domains = set(domains or ())
        for envelope in envelopes:
            domain = (
                envelope.get("domain")
                if isinstance(envelope, Mapping)
                else getattr(envelope, "domain", None)
            )
            if isinstance(domain, str):
                candidate_domains.add(domain)
        if any(
            domain == "personal_context"
            or domain.startswith("personal_context.")
            for domain in candidate_domains
        ):
            raise ValueError("personal_context_requires_reviewed_first_link")
        return await self._push_v2_envelopes(
            dataset_id=dataset_id,
            device_id=device_id,
            envelopes=envelopes,
            domains=domains,
            idempotency_key=idempotency_key,
            last_known_cursor=last_known_cursor,
        )

    async def _push_v2_envelopes(
        self,
        *,
        dataset_id: str,
        device_id: str,
        envelopes: list[SyncV2Envelope | Mapping[str, Any]],
        domains: list[str] | None = None,
        idempotency_key: str | None = None,
        last_known_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Validate and dispatch one push after its caller proves domain authority."""

        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SyncV2Envelope, SyncV2PushRequest

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
            domains=(
                domains
                if domains is not None
                else sorted({str(envelope.domain) for envelope in coerced_envelopes})
            ),
        )
        request = SyncV2PushRequest(
            dataset_id=dataset_id,
            device_id=device_id,
            envelopes=coerced_envelopes,
            idempotency_key=idempotency_key,
            last_known_cursor=last_known_cursor,
        )
        response = self._dump(
            await self._require_client().push_sync_v2_envelopes(request)
        )
        validate_push_response_scope(
            dataset_id=dataset_id,
            response_dataset_id=response.get("dataset_id"),
            submitted_client_envelope_ids=[
                envelope.client_envelope_id for envelope in coerced_envelopes
            ],
            accepted=response.get("accepted", []),
            rejected=response.get("rejected", []),
            conflicts=response.get("conflicts", []),
        )
        return response

    async def _push_v2_personal_context_first_link(
        self, **kwargs: Any
    ) -> dict[str, Any]:
        """Private push reached only from exact reviewed first-link convergence."""

        return await self._push_v2_envelopes(**kwargs)

    async def _push_v2_personal_context_complete(
        self, **kwargs: Any
    ) -> dict[str, Any]:
        """Private push reached only after LocalFirst validates the exact receipt."""

        return await self._push_v2_envelopes(**kwargs)

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
        """Pull non-Personal-Context envelopes through the public transport.

        Args:
            dataset_id: Dataset to pull from.
            device_id: Local device identity making the request.
            cursor: Optional cursor for incremental sync.
            domains: Optional domain filter to request and validate.
            page_size: Optional maximum number of envelopes to request.
            include_own_changes: Whether restore flows may include envelopes from this device.

        Returns:
            Server pull response after dataset, domain, device, and pagination validation.

        Raises:
            ValueError: If pulled envelopes or pagination state violate Sync v2 scope.
            PolicyDeniedError: If runtime policy blocks server Sync v2 pull access.
        """
        if domains and any(
            domain == "personal_context"
            or domain.startswith("personal_context.")
            for domain in domains
        ):
            raise ValueError("personal_context_requires_reviewed_first_link")
        return await self._pull_v2_envelopes(
            dataset_id=dataset_id,
            device_id=device_id,
            cursor=cursor,
            domains=domains,
            page_size=page_size,
            include_own_changes=include_own_changes,
        )

    async def _pull_v2_envelopes(
        self,
        *,
        dataset_id: str,
        device_id: str,
        cursor: str | None = None,
        domains: list[str] | None = None,
        page_size: int | None = None,
        include_own_changes: bool = False,
    ) -> dict[str, Any]:
        """Validate and dispatch one pull after its caller proves domain authority."""

        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SyncV2Envelope

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

    async def _pull_v2_personal_context_first_link(
        self,
        *,
        dataset_id: str,
        device_id: str,
        cursor: str | None,
        domains: list[str],
        page_size: int | None,
        include_own_changes: bool,
    ) -> dict[str, Any]:
        """Private transport used only by the exact reviewed reconciliation path."""

        return await self._pull_v2_envelopes(
            dataset_id=dataset_id,
            device_id=device_id,
            cursor=cursor,
            domains=domains,
            page_size=page_size,
            include_own_changes=include_own_changes,
        )

    async def _pull_v2_personal_context_complete(self, **kwargs: Any) -> dict[str, Any]:
        """Private transport reached only after LocalFirst validates the exact receipt."""

        return await self._pull_v2_envelopes(**kwargs)

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
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SyncV2ConflictResolveRequest

        self._enforce("sync.v2.conflicts.resolve.server")
        request = SyncV2ConflictResolveRequest(
            conflict_id=conflict_id,
            action=action,
            resolution_envelope=resolution_envelope,
            resolved_by_device_id=resolved_by_device_id,
            notes=notes,
        )
        return self._dump(
            await self._require_client().resolve_sync_v2_conflict(conflict_id, request)
        )
