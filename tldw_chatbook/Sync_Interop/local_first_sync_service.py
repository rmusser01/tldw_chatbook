"""Local-first Sync v2 orchestration for Chatbook."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.tldw_api import SyncV2Envelope


class LocalFirstSyncService:
    """Push local envelopes, pull remote envelopes, and apply them locally."""

    def __init__(
        self,
        *,
        server_service: Any,
        state_repository: Any,
        local_store: Any,
        dataset_keys: dict[str, bytes] | None = None,
    ) -> None:
        self.server_service = server_service
        self.state_repository = state_repository
        self.local_store = local_store
        self.dataset_keys = dataset_keys or {}

    async def sync_once(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        domains: list[str],
        outgoing_envelopes: list[SyncV2Envelope | Mapping[str, Any]] | None = None,
        page_size: int | None = None,
        dataset_key: bytes | None = None,
    ) -> dict[str, Any]:
        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile is None:
            raise ValueError("local_first Sync v2 profile is required")
        if profile.get("profile_mode") != "local_first":
            raise ValueError("sync_once requires a local_first Sync v2 profile")

        dataset_id = profile.get("dataset_id")
        device_id = profile.get("device_id")
        if not dataset_id or not device_id:
            raise ValueError("local_first Sync v2 profile requires device_id and dataset_id")

        key = dataset_key or self.dataset_keys.get(str(dataset_id))
        if key is None:
            raise ValueError("dataset key is required for local_first Sync v2 envelopes")

        cursor_record = self.state_repository.get_remote_pull_cursor(
            source_authority="server",
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain="sync_v2",
            remote_collection=str(dataset_id),
        )

        outbox_entries = self.state_repository.list_pending_sync_v2_outbox_envelopes(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            dataset_id=str(dataset_id),
            domains=list(domains),
        )
        outbox_payloads = [dict(entry["envelope"]) for entry in outbox_entries]
        outgoing_payloads = [
            self._dump_envelope(envelope)
            for envelope in (outgoing_envelopes or [])
        ]
        push_payloads = outbox_payloads + outgoing_payloads
        push_record: dict[str, Any] = {}
        outbox_result = {"dispatched": 0, "retained": 0}
        if push_payloads:
            try:
                push_record = self._dump(
                    await self.server_service.push_v2_envelopes(
                        dataset_id=str(dataset_id),
                        device_id=str(device_id),
                        envelopes=push_payloads,
                        idempotency_key=self._push_idempotency_key(
                            dataset_id=str(dataset_id),
                            device_id=str(device_id),
                            cursor=cursor_record.cursor,
                            envelopes=push_payloads,
                        ),
                        last_known_cursor=cursor_record.cursor,
                    )
                )
            except Exception as exc:
                if outbox_entries:
                    self._record_outbox_push_failure(
                        server_profile_id=server_profile_id,
                        authenticated_principal_id=authenticated_principal_id,
                        workspace_scope=workspace_scope,
                        dataset_id=str(dataset_id),
                        outbox_entries=outbox_entries,
                        exc=exc,
                    )
                self._record_sync_error(profile=profile, stage="push", exc=exc)
                raise
            if outbox_entries:
                outbox_result = self.state_repository.mark_sync_v2_outbox_push_results(
                    server_profile_id=server_profile_id,
                    authenticated_principal_id=authenticated_principal_id,
                    workspace_scope=workspace_scope,
                    dataset_id=str(dataset_id),
                    accepted=push_record.get("accepted", []),
                    rejected=push_record.get("rejected", []),
                    conflicts=push_record.get("conflicts", []),
                )

        try:
            pulled = self._dump(
                await self.server_service.pull_v2_envelopes(
                    dataset_id=str(dataset_id),
                    device_id=str(device_id),
                    cursor=cursor_record.cursor,
                    domains=list(domains),
                    page_size=page_size,
                    include_own_changes=False,
                )
            )
        except Exception as exc:
            self._record_sync_error(profile=profile, stage="pull", exc=exc)
            raise
        applier = SyncEnvelopeApplier(dataset_key=key, local_store=self.local_store)
        try:
            results = [
                applier.apply(SyncV2Envelope.model_validate(envelope))
                for envelope in pulled.get("envelopes", [])
            ]
        except Exception as exc:
            self._record_sync_error(profile=profile, stage="apply", exc=exc)
            raise
        rejected_results = [
            result
            for result in results
            if result.get("status") == "rejected"
        ]
        if rejected_results:
            rejection_message = self._rejection_message(rejected_results)
            self._persist_profile_state(
                profile=profile,
                dataset_cursors=dict(profile.get("dataset_cursors") or {}),
                last_error=f"apply_rejected: {rejection_message}",
            )
            raise ValueError(f"apply rejected: {rejection_message}")
        next_cursor = (
            pulled.get("next_cursor")
            or push_record.get("next_cursor")
            or cursor_record.cursor
        )
        dataset_cursors = dict(profile.get("dataset_cursors") or {})
        if next_cursor is not None:
            dataset_cursors["sync_v2"] = next_cursor
            self.state_repository.set_remote_pull_cursor(
                source_authority="server",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                domain="sync_v2",
                remote_collection=str(dataset_id),
                cursor=next_cursor,
            )
        self._persist_profile_state(
            profile=profile,
            dataset_cursors=dataset_cursors,
            last_error=self._push_partial_failure_message(push_record),
        )

        conflicts = [
            result["conflict"]
            for result in results
            if result.get("status") == "conflict" and "conflict" in result
        ]
        return {
            "dataset_id": str(dataset_id),
            "device_id": str(device_id),
            "domains": list(domains),
            "pushed_envelopes": len(push_record.get("accepted", [])),
            "rejected_envelopes": push_record.get("rejected", []),
            "push_conflicts": push_record.get("conflicts", []),
            "outbox_drained": len(outbox_entries),
            "outbox_dispatched": outbox_result["dispatched"],
            "outbox_retained": outbox_result["retained"],
            "pulled_envelopes": len(pulled.get("envelopes", [])),
            "applied_envelopes": sum(1 for result in results if result.get("status") == "applied"),
            "conflicts": conflicts,
            "next_cursor": next_cursor,
            "has_more": bool(pulled.get("has_more", False)),
            "results": results,
        }

    @staticmethod
    def _dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [LocalFirstSyncService._dump(item) for item in value]
        if isinstance(value, dict):
            return {key: LocalFirstSyncService._dump(item) for key, item in value.items()}
        return value

    @staticmethod
    def _dump_envelope(envelope: SyncV2Envelope | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(envelope, SyncV2Envelope):
            return envelope.model_dump(mode="json")
        return SyncV2Envelope.model_validate(envelope).model_dump(mode="json")

    @staticmethod
    def _push_idempotency_key(
        *,
        dataset_id: str,
        device_id: str,
        cursor: str | None,
        envelopes: list[Mapping[str, Any]],
    ) -> str:
        batch_identity = {
            "cursor": cursor,
            "dataset_id": dataset_id,
            "device_id": device_id,
            "client_envelope_ids": [
                str(envelope.get("client_envelope_id"))
                for envelope in envelopes
            ],
        }
        encoded = json.dumps(
            batch_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sync-v2-push:{hashlib.sha256(encoded).hexdigest()}"

    def _record_sync_error(
        self,
        *,
        profile: Mapping[str, Any],
        stage: str,
        exc: Exception,
    ) -> None:
        self._persist_profile_state(
            profile=profile,
            dataset_cursors=dict(profile.get("dataset_cursors") or {}),
            last_error=f"{stage}_failed: {exc}",
        )

    def _record_outbox_push_failure(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        dataset_id: str,
        outbox_entries: list[Mapping[str, Any]],
        exc: Exception,
    ) -> dict[str, int]:
        return self.state_repository.mark_sync_v2_outbox_push_results(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            dataset_id=dataset_id,
            accepted=[],
            rejected=[
                {
                    "client_envelope_id": entry["client_envelope_id"],
                    "error_code": "push_failed",
                    "message": str(exc),
                    "retryable": True,
                }
                for entry in outbox_entries
            ],
            conflicts=[],
        )

    def _persist_profile_state(
        self,
        *,
        profile: Mapping[str, Any],
        dataset_cursors: Mapping[str, str | int],
        last_error: str | None,
    ) -> dict[str, Any]:
        return self.state_repository.set_sync_v2_profile_state(
            server_profile_id=str(profile["server_profile_id"]),
            authenticated_principal_id=profile.get("authenticated_principal_id"),
            workspace_scope=profile.get("workspace_scope"),
            profile_mode=str(profile.get("profile_mode") or "local_first"),
            device_id=profile.get("device_id"),
            dataset_id=profile.get("dataset_id"),
            dataset_cursors=dict(dataset_cursors),
            capabilities=dict(profile.get("capabilities") or {}),
            dry_run_metadata=dict(profile.get("dry_run_metadata") or {}),
            last_error=last_error,
            last_mirror_report_id=profile.get("last_mirror_report_id"),
        )

    @staticmethod
    def _rejection_message(results: list[Mapping[str, Any]]) -> str:
        error_codes = [
            str(result.get("error_code"))
            for result in results
            if result.get("error_code")
        ]
        if error_codes:
            return ",".join(error_codes)
        return f"{len(results)} envelope apply rejection(s)"

    @staticmethod
    def _push_partial_failure_message(push_record: Mapping[str, Any]) -> str | None:
        rejected = list(push_record.get("rejected", []))
        conflicts = list(push_record.get("conflicts", []))
        if not rejected and not conflicts:
            return None
        codes = [
            str(item.get("error_code"))
            for item in rejected
            if item.get("error_code")
        ]
        codes.extend("conflict" for item in conflicts if item.get("client_envelope_id"))
        if not codes:
            codes.append("unknown")
        return f"push_partial_failure: {','.join(codes)}"
