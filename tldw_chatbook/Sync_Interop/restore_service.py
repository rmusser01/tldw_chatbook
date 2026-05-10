"""Restore manifest and selective pull helpers for Sync v2."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.tldw_api import SyncV2Envelope


class SyncRestoreService:
    """Interpret restore manifests and apply selected Sync v2 envelopes."""

    def __init__(
        self,
        *,
        server_service: Any,
        local_store: Any,
        dataset_keys: dict[str, bytes] | None = None,
    ) -> None:
        self.server_service = server_service
        self.local_store = local_store
        self.dataset_keys = dataset_keys or {}

    async def fetch_manifest(
        self,
        *,
        dataset_ids: list[str] | None = None,
        domains: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._dump(
            await self.server_service.get_v2_restore_manifest(
                dataset_ids=dataset_ids,
                domains=domains,
            )
        )

    async def preview_restore(
        self,
        *,
        dataset_ids: list[str] | None = None,
        domains: list[str] | None = None,
    ) -> dict[str, Any]:
        manifest = await self.fetch_manifest(dataset_ids=dataset_ids, domains=domains)
        datasets = [
            self._preview_dataset(dataset)
            for dataset in manifest.get("datasets", [])
        ]
        return {
            "datasets": datasets,
            "devices": manifest.get("devices", []),
            "generated_at": manifest.get("generated_at"),
            "filters_applied": manifest.get("filters_applied", {}),
        }

    async def restore_selection(
        self,
        *,
        dataset_id: str,
        device_id: str,
        domains: list[str],
        cursor: str | None = None,
        page_size: int | None = None,
        dataset_key: bytes | None = None,
    ) -> dict[str, Any]:
        key = dataset_key or self.dataset_keys.get(dataset_id)
        if key is None:
            raise ValueError("dataset key is required to restore encrypted Sync v2 envelopes")
        pulled = self._dump(
            await self.server_service.pull_v2_envelopes(
                dataset_id=dataset_id,
                device_id=device_id,
                cursor=cursor,
                domains=domains,
                page_size=page_size,
                include_own_changes=False,
            )
        )
        applier = SyncEnvelopeApplier(dataset_key=key, local_store=self.local_store)
        results = [
            applier.apply(SyncV2Envelope.model_validate(envelope))
            for envelope in pulled.get("envelopes", [])
        ]
        return {
            "dataset_id": dataset_id,
            "domains": list(domains),
            "applied": sum(1 for result in results if result.get("status") == "applied"),
            "conflicts": [
                result["conflict"]
                for result in results
                if result.get("status") == "conflict" and "conflict" in result
            ],
            "next_cursor": pulled.get("next_cursor"),
            "has_more": bool(pulled.get("has_more", False)),
            "results": results,
        }

    async def list_conflicts(
        self,
        *,
        dataset_id: str,
        status: str = "unresolved",
    ) -> list[dict[str, Any]]:
        conflicts = await self.server_service.list_v2_conflicts(
            dataset_id=dataset_id,
            status=status,
        )
        return self._dump(conflicts)

    def _preview_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(dataset["dataset_id"])
        encryption_policy = dataset.get("encryption_policy")
        has_local_key = dataset_id in self.dataset_keys
        recovery_available = bool(dataset.get("key_recovery_available", False))
        restore_status = "restorable"
        if encryption_policy == "client_private_v1":
            if has_local_key:
                restore_status = "restorable_with_local_key"
            elif recovery_available:
                restore_status = "recovery_available"
            else:
                restore_status = "locked"
        return {
            **dataset,
            "has_local_key": has_local_key,
            "restore_status": restore_status,
        }

    @staticmethod
    def _dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [SyncRestoreService._dump(item) for item in value]
        if isinstance(value, dict):
            return {key: SyncRestoreService._dump(item) for key, item in value.items()}
        return value
