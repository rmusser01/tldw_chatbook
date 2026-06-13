"""Local setup flow for Sync v2 dataset key recovery."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Sync_Interop.crypto import wrap_dataset_key_for_recovery
from tldw_chatbook.Sync_Interop.sync_state import SyncV2ProfileMode, is_local_first_sync_profile_mode


class SyncKeyRecoveryService:
    """Wrap local dataset keys and store only recovery material on the server."""

    def __init__(
        self,
        *,
        server_service: Any,
        state_repository: Any,
        dataset_keys: dict[str, bytes] | None = None,
    ) -> None:
        self.server_service = server_service
        self.state_repository = state_repository
        self.dataset_keys = dataset_keys or {}

    async def configure_recovery(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        recovery_secret: str | bytes,
        recovery_hint: str | None = None,
        dataset_key: bytes | None = None,
        key_purpose: str = "dataset_recovery",
        rotation_of_key_record_id: str | None = None,
    ) -> dict[str, Any]:
        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile is None:
            raise ValueError("local_first Sync v2 profile is required")
        if not is_local_first_sync_profile_mode(profile.get("profile_mode")):
            raise ValueError("key recovery setup requires a local_first Sync v2 profile")
        profile_mode = str(profile.get("profile_mode") or SyncV2ProfileMode.LOCAL_FIRST_SYNC.value)

        dataset_id = profile.get("dataset_id")
        device_id = profile.get("device_id")
        if not dataset_id or not device_id:
            raise ValueError("local_first Sync v2 profile requires device_id and dataset_id")

        resolved_dataset_id = str(dataset_id)
        resolved_device_id = str(device_id)
        resolved_key = dataset_key or self.dataset_keys.get(resolved_dataset_id)
        if resolved_key is None:
            raise ValueError("dataset key is required to configure Sync v2 key recovery")

        bundle = wrap_dataset_key_for_recovery(
            resolved_key,
            recovery_secret=recovery_secret,
            recovery_hint=recovery_hint,
        )
        resolved_key_purpose = key_purpose or bundle.key_purpose
        response = self._dump(
            await self.server_service.store_v2_recovery_bundle(
                dataset_id=resolved_dataset_id,
                device_id=resolved_device_id,
                wrapped_key_blob=bundle.wrapped_key_blob,
                kdf_metadata=bundle.kdf_metadata,
                recovery_hint=bundle.recovery_hint,
                key_purpose=resolved_key_purpose,
                rotation_of_key_record_id=rotation_of_key_record_id,
            )
        )

        response_key_purpose = response.get("key_purpose") or resolved_key_purpose
        response_recovery_hint = response.get("recovery_hint") or recovery_hint
        dry_run_metadata = dict(profile.get("dry_run_metadata") or {})
        dry_run_metadata.update(
            {
                "key_recovery_configured": True,
                "key_recovery_available": True,
                "key_recovery_key_record_id": response.get("key_record_id"),
                "key_recovery_hint": response_recovery_hint,
                "key_recovery_key_purpose": response_key_purpose,
            }
        )
        self.state_repository.set_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            profile_mode=profile_mode,
            device_id=resolved_device_id,
            dataset_id=resolved_dataset_id,
            dataset_cursors=dict(profile.get("dataset_cursors") or {}),
            capabilities=dict(profile.get("capabilities") or {}),
            dry_run_metadata=dry_run_metadata,
            last_error=None,
            last_mirror_report_id=profile.get("last_mirror_report_id"),
        )
        return {
            "dataset_id": resolved_dataset_id,
            "device_id": resolved_device_id,
            "key_record_id": response.get("key_record_id"),
            "key_purpose": response_key_purpose,
            "recovery_hint": response_recovery_hint,
            "key_recovery_configured": True,
        }

    @staticmethod
    def _dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [SyncKeyRecoveryService._dump(item) for item in value]
        if isinstance(value, dict):
            return {key: SyncKeyRecoveryService._dump(item) for key, item in value.items()}
        return value
