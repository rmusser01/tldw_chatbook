"""Local-first Sync v2 orchestration for Chatbook."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Sync_Interop.sync_state import is_local_first_sync_profile_mode
from tldw_chatbook.Sync_Interop.validation import (
    validate_pull_pagination_state,
    validate_push_response_scope,
    validate_outgoing_envelope_scope,
    validate_pulled_response_scope,
)
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
        """Run one local-first Sync v2 push, pull, and apply cycle.

        Args:
            server_profile_id: Stable identifier for the configured server profile.
            authenticated_principal_id: Optional user or account identity for scoped state.
            workspace_scope: Optional workspace identifier for workspace-scoped datasets.
            domains: Sync v2 domains to include in this cycle.
            outgoing_envelopes: Additional unsaved envelopes to push with the durable outbox.
            page_size: Optional pull page size to request from the server.
            dataset_key: Optional dataset key override for decrypting pulled envelopes.

        Returns:
            Summary containing pushed, pulled, applied, conflict, cursor, and outbox counts.

        Raises:
            ValueError: If profile state, dataset identity, encryption key, or transport
                validation is invalid.
            Exception: Propagates server transport and local apply failures after recording
                sync state.
        """

        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile is None:
            raise ValueError("local_first Sync v2 profile is required")
        if not is_local_first_sync_profile_mode(profile.get("profile_mode")):
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
        try:
            outbox_parsed = [
                self._coerce_envelope(entry["envelope"])
                for entry in outbox_entries
            ]
            outgoing_parsed = [
                self._coerce_envelope(envelope)
                for envelope in (outgoing_envelopes or [])
            ]
            validate_outgoing_envelope_scope(
                dataset_id=str(dataset_id),
                device_id=str(device_id),
                envelopes=[*outbox_parsed, *outgoing_parsed],
                domains=list(domains),
            )
        except Exception as exc:
            self._record_sync_error(profile=profile, stage="push", exc=exc)
            raise
        outbox_payloads = [
            envelope.model_dump(mode="json")
            for envelope in outbox_parsed
        ]
        outgoing_payloads = [
            envelope.model_dump(mode="json")
            for envelope in outgoing_parsed
        ]
        push_items = [
            {"payload": payload, "outbox_entry": entry}
            for payload, entry in zip(outbox_payloads, outbox_entries)
        ]
        push_items.extend(
            {"payload": payload, "outbox_entry": None}
            for payload in outgoing_payloads
        )
        push_record: dict[str, Any] = {
            "dataset_id": str(dataset_id),
            "accepted": [],
            "rejected": [],
            "conflicts": [],
        }
        outbox_result = {"dispatched": 0, "retained": 0}
        if push_items:
            processed_outbox_ids: set[int] = set()
            push_cursor = cursor_record.cursor
            for batch_items in self._chunk_push_items(
                push_items,
                batch_size=self._max_push_batch_size(
                    profile.get("capabilities"),
                    fallback_size=len(push_items),
                ),
            ):
                batch_payloads = [
                    item["payload"]
                    for item in batch_items
                ]
                try:
                    batch_record = self._dump(
                        await self.server_service.push_v2_envelopes(
                            dataset_id=str(dataset_id),
                            device_id=str(device_id),
                            envelopes=batch_payloads,
                            domains=list(domains),
                            idempotency_key=self._push_idempotency_key(
                                dataset_id=str(dataset_id),
                                device_id=str(device_id),
                                cursor=push_cursor,
                                envelopes=batch_payloads,
                            ),
                            last_known_cursor=push_cursor,
                        )
                    )
                    batch_outbox_entries = [
                        item["outbox_entry"]
                        for item in batch_items
                        if item["outbox_entry"] is not None
                    ]
                except Exception as exc:
                    failed_outbox_entries = [
                        entry
                        for entry in outbox_entries
                        if int(entry["outbox_id"]) not in processed_outbox_ids
                    ]
                    if failed_outbox_entries:
                        self._record_outbox_push_failure(
                            server_profile_id=server_profile_id,
                            authenticated_principal_id=authenticated_principal_id,
                            workspace_scope=workspace_scope,
                            dataset_id=str(dataset_id),
                            outbox_entries=failed_outbox_entries,
                            exc=exc,
                        )
                    self._record_sync_error(profile=profile, stage="push", exc=exc)
                    raise

                try:
                    validate_push_response_scope(
                        dataset_id=str(dataset_id),
                        response_dataset_id=batch_record.get("dataset_id"),
                        submitted_client_envelope_ids=[
                            str(envelope["client_envelope_id"])
                            for envelope in batch_payloads
                        ],
                        accepted=batch_record.get("accepted", []),
                        rejected=batch_record.get("rejected", []),
                        conflicts=batch_record.get("conflicts", []),
                    )
                except Exception as exc:
                    self._record_sync_error(profile=profile, stage="push", exc=exc)
                    raise

                for result_key in ("accepted", "rejected", "conflicts"):
                    push_record[result_key].extend(batch_record.get(result_key, []))
                if batch_record.get("next_cursor") is not None:
                    push_cursor = batch_record.get("next_cursor")
                    push_record["next_cursor"] = push_cursor

                if batch_outbox_entries:
                    batch_result = self.state_repository.mark_sync_v2_outbox_push_results(
                        server_profile_id=server_profile_id,
                        authenticated_principal_id=authenticated_principal_id,
                        workspace_scope=workspace_scope,
                        dataset_id=str(dataset_id),
                        accepted=batch_record.get("accepted", []),
                        rejected=batch_record.get("rejected", []),
                        conflicts=batch_record.get("conflicts", []),
                    )
                    outbox_result["dispatched"] += batch_result["dispatched"]
                    outbox_result["retained"] += batch_result["retained"]
                    processed_outbox_ids.update(
                        int(entry["outbox_id"])
                        for entry in batch_outbox_entries
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
            envelopes = [
                SyncV2Envelope.model_validate(envelope)
                for envelope in pulled.get("envelopes", [])
            ]
            validate_pulled_response_scope(
                dataset_id=str(dataset_id),
                response_dataset_id=pulled.get("dataset_id"),
                envelopes=envelopes,
                domains=list(domains),
                excluded_device_id=str(device_id),
            )
            validate_pull_pagination_state(
                has_more=pulled.get("has_more", False),
                next_cursor=pulled.get("next_cursor"),
                envelope_count=len(envelopes),
            )
            results = [
                applier.apply(envelope)
                for envelope in envelopes
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
        conflicts = [
            result["conflict"]
            for result in results
            if result.get("status") == "conflict" and "conflict" in result
        ]
        self._persist_profile_state(
            profile=profile,
            dataset_cursors=dataset_cursors,
            last_error=self._attention_status_message(push_record, conflicts),
        )

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
        return LocalFirstSyncService._coerce_envelope(envelope).model_dump(mode="json")

    @staticmethod
    def _coerce_envelope(envelope: SyncV2Envelope | Mapping[str, Any]) -> SyncV2Envelope:
        if isinstance(envelope, SyncV2Envelope):
            return envelope
        return SyncV2Envelope.model_validate(envelope)

    @staticmethod
    def _max_push_batch_size(capabilities: Any, *, fallback_size: int) -> int:
        if not isinstance(capabilities, Mapping):
            return max(1, fallback_size)
        raw_batch_size = capabilities.get("max_batch_size")
        if raw_batch_size is None:
            return max(1, fallback_size)
        batch_size = int(raw_batch_size)
        if batch_size <= 0:
            raise ValueError("Sync v2 max_batch_size must be positive")
        return batch_size

    @staticmethod
    def _chunk_push_items(
        push_items: list[Mapping[str, Any]],
        *,
        batch_size: int,
    ) -> list[list[Mapping[str, Any]]]:
        return [
            push_items[index:index + batch_size]
            for index in range(0, len(push_items), batch_size)
        ]

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

    @staticmethod
    def _apply_conflict_message(conflicts: list[Mapping[str, Any]]) -> str | None:
        conflict_types = [
            str(conflict.get("conflict_type"))
            for conflict in conflicts
            if conflict.get("conflict_type")
        ]
        if not conflict_types:
            return None
        return f"apply_conflict: {','.join(conflict_types)}"

    @classmethod
    def _attention_status_message(
        cls,
        push_record: Mapping[str, Any],
        conflicts: list[Mapping[str, Any]],
    ) -> str | None:
        messages = [
            message
            for message in (
                cls._push_partial_failure_message(push_record),
                cls._apply_conflict_message(conflicts),
            )
            if message
        ]
        if not messages:
            return None
        return "; ".join(messages)
