"""Dedicated, fail-closed Personal Context first-link convergence cycle."""

from __future__ import annotations

from typing import Any, Mapping

from .envelope_applier import SyncEnvelopeApplier
from .local_first_sync_service import LocalFirstSyncService
from .validation import (
    validate_outgoing_envelope_scope,
    validate_pull_pagination_state,
    validate_pulled_response_scope,
    validate_push_response_scope,
)


_DOMAINS = [
    "personal_context.manifest",
    "personal_context.scope",
    "personal_context.record",
    "personal_context.proposal",
    "personal_context.purge",
]


class PersonalContextFirstLinkSync:
    """Push and confirm one reviewed Personal Context materialization delta."""

    def __init__(
        self,
        *,
        server_service: Any,
        state_repository: Any,
        dispatcher: Any,
        personal_context_service: Any,
        local_store: Any,
        dataset_keys: Mapping[str, bytes],
    ) -> None:
        self._server = server_service
        self._state = state_repository
        self._dispatcher = dispatcher
        self._profile = personal_context_service
        self._local_store = local_store
        self._dataset_keys = dataset_keys

    def activate_storage_key(self, dataset_id: str, storage_key: bytes) -> None:
        """Install one securely loaded dataset key into the shared runtime cache."""

        if not isinstance(self._dataset_keys, dict) or len(storage_key) != 32:
            raise ValueError("personal_context_staging_key_unavailable")
        self._dataset_keys[dataset_id] = bytes(storage_key)

    async def converge(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        device_id: str,
        dataset_id: str,
        profile_id: str,
        integrity_key_id: str,
        key_record_id: str,
        purge_generation: int,
        bootstrap_cursor: str,
        bootstrap_heads: Mapping[str, Mapping[str, str]],
        expected_heads: Mapping[str, Mapping[str, str]],
    ) -> dict[str, Any]:
        """Confirm the exact reviewed, sync-eligible canonical head set.

        Server-only workspace scopes that remain unbound are retained locally but are
        outside this receipt: they are not agent-accessible and are not eligible for
        Personal Context Sync until the user explicitly maps them.
        """

        binding = {
            "device_id": device_id,
            "dataset_id": dataset_id,
            "profile_id": profile_id,
            "integrity_key_id": integrity_key_id,
            "key_record_id": key_record_id,
            "purge_generation": purge_generation,
            "bootstrap_cursor": bootstrap_cursor,
            "bootstrap_heads": dict(bootstrap_heads),
            "expected_heads": dict(expected_heads),
        }
        state = self._state.get_personal_context_link_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
        )
        if state is None or state.get("state") != "reconciling" or any(
            state.get(key) != value for key, value in binding.items()
        ):
            raise ValueError("personal_context_reconciliation_binding_stale")
        storage_key = self._dataset_keys.get(dataset_id)
        if storage_key is None or len(storage_key) != 32:
            raise ValueError("personal_context_staging_key_unavailable")
        sync_profile = self._state.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=None,
        )
        batch_size = LocalFirstSyncService._max_push_batch_size(
            None if sync_profile is None else sync_profile.get("capabilities"),
            fallback_size=100,
        )
        push_cursor: str | None = bootstrap_cursor
        while True:
            self._dispatcher.dispatch_first_link_reconciliation(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=None,
                dataset_id=dataset_id,
                device_id=device_id,
                storage_key=storage_key,
                limit=batch_size,
                profile_id=profile_id,
                integrity_key_id=integrity_key_id,
                key_record_id=key_record_id,
                purge_generation=purge_generation,
                bootstrap_cursor=bootstrap_cursor,
            )
            outbox = self._state.list_pending_sync_v2_outbox_envelopes(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=None,
                dataset_id=dataset_id,
                domains=_DOMAINS,
            )
            if not outbox:
                break
            envelopes = [
                self._dispatcher.adapter.restore_from_storage(
                    self._coerce(item["envelope"]), storage_key=storage_key
                )
                for item in outbox[:batch_size]
            ]
            validate_outgoing_envelope_scope(
                dataset_id=dataset_id,
                device_id=device_id,
                envelopes=envelopes,
                domains=_DOMAINS,
            )
            payloads = [item.model_dump(mode="json") for item in envelopes]
            push = getattr(
                self._server,
                "_push_v2_personal_context_first_link",
                None,
            )
            if not callable(push):
                raise RuntimeError("personal_context_first_link_sync_unavailable")
            response = await push(
                dataset_id=dataset_id,
                device_id=device_id,
                envelopes=payloads,
                domains=_DOMAINS,
                idempotency_key=LocalFirstSyncService._push_idempotency_key(
                    dataset_id=dataset_id,
                    device_id=device_id,
                    cursor=push_cursor,
                    envelopes=payloads,
                ),
                last_known_cursor=push_cursor,
            )
            validate_push_response_scope(
                dataset_id=dataset_id,
                response_dataset_id=response.get("dataset_id"),
                submitted_client_envelope_ids=[
                    item.client_envelope_id for item in envelopes
                ],
                accepted=response.get("accepted", []),
                rejected=response.get("rejected", []),
                conflicts=response.get("conflicts", []),
            )
            if response.get("rejected") or response.get("conflicts"):
                raise RuntimeError("personal_context_reconciliation_push_rejected")
            self._state.mark_sync_v2_outbox_push_results(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=None,
                dataset_id=dataset_id,
                accepted=response.get("accepted", []),
                rejected=[],
                conflicts=[],
            )
            if response.get("next_cursor") is not None:
                push_cursor = str(response["next_cursor"])
        observed = self._reviewed_bootstrap_heads(
            bootstrap_heads=bootstrap_heads,
            expected_heads=expected_heads,
        )
        cursor: str | None = bootstrap_cursor
        applier = SyncEnvelopeApplier(
            dataset_key=storage_key,
            local_store=self._local_store,
            personal_context_adapter=self._dispatcher.adapter,
            personal_context_service=self._profile,
        )
        while True:
            pull = getattr(
                self._server,
                "_pull_v2_personal_context_first_link",
                None,
            )
            if not callable(pull):
                raise RuntimeError("personal_context_first_link_sync_unavailable")
            pulled = await pull(
                dataset_id=dataset_id,
                device_id=device_id,
                cursor=cursor,
                domains=_DOMAINS,
                page_size=None,
                include_own_changes=True,
            )
            pulled_envelopes = [self._coerce(item) for item in pulled.get("envelopes", [])]
            validate_pulled_response_scope(
                dataset_id=dataset_id,
                response_dataset_id=pulled.get("dataset_id"),
                envelopes=pulled_envelopes,
                domains=_DOMAINS,
                excluded_device_id=None,
            )
            validate_pull_pagination_state(
                has_more=bool(pulled.get("has_more", False)),
                next_cursor=pulled.get("next_cursor"),
                envelope_count=len(pulled_envelopes),
            )
            for envelope in pulled_envelopes:
                writes = getattr(
                    self._profile, "first_link_reconciliation_writes", None
                )
                if callable(writes):
                    with writes(plan_id=str(state["plan_id"])):
                        result = applier.apply(envelope)
                else:
                    result = applier.apply(envelope)
                if result.get("status") not in {"applied", "ignored"}:
                    raise RuntimeError("personal_context_reconciliation_apply_failed")
                if envelope.entity_version is None or envelope.object_id is None:
                    raise RuntimeError("personal_context_reconciliation_version_missing")
                if self._is_reviewed_head(envelope, expected_heads=expected_heads):
                    observed.setdefault(envelope.domain, {})[
                        str(envelope.object_id)
                    ] = str(envelope.entity_version)
            cursor = pulled.get("next_cursor") or cursor
            if not pulled.get("has_more", False):
                break
        if observed != dict(expected_heads):
            raise RuntimeError("personal_context_convergence_unconfirmed")
        return {"confirmed_cursor": cursor, "confirmed_heads": observed}

    @staticmethod
    def _reviewed_bootstrap_heads(
        *,
        bootstrap_heads: Mapping[str, Mapping[str, str]],
        expected_heads: Mapping[str, Mapping[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Project the full server bootstrap onto the reviewed eligible identities."""

        return {
            str(domain): {
                str(object_id): str(bootstrap_version)
                for object_id in expected_domain
                if (
                    bootstrap_version := bootstrap_heads.get(domain, {}).get(object_id)
                )
                is not None
            }
            for domain, expected_domain in expected_heads.items()
        }

    @staticmethod
    def _is_reviewed_head(
        envelope: Any,
        *,
        expected_heads: Mapping[str, Mapping[str, str]],
    ) -> bool:
        """Return whether a pulled head participates in the reviewed receipt."""

        object_id = str(envelope.object_id)
        expected_domain = expected_heads.get(envelope.domain, {})
        if object_id in expected_domain:
            return True
        if envelope.domain == "personal_context.scope":
            return False
        if envelope.domain in {
            "personal_context.record",
            "personal_context.proposal",
        }:
            return str(envelope.parent_id) in expected_heads.get(
                "personal_context.scope", {}
            )
        return True

    @staticmethod
    def _coerce(value: Any):
        from tldw_chatbook.tldw_api import SyncV2Envelope

        return value if isinstance(value, SyncV2Envelope) else SyncV2Envelope.model_validate(value)


__all__ = ["PersonalContextFirstLinkSync"]
