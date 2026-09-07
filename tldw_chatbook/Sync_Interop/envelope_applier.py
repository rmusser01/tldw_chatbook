"""Apply pulled Sync v2 envelopes through local domain adapters."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    read_provider_continuation_json,
)
from tldw_chatbook.Chat.assistant_generation_state import (
    AssistantGenerationState,
    normalize_assistant_generation_state,
)
from tldw_chatbook.Sync_Interop.domain_adapters import (
    ChatSyncAdapter,
    MediaSyncAdapter,
    NotesSyncAdapter,
    SourceCacheSyncAdapter,
    WorkspacesSyncAdapter,
)
from tldw_chatbook.Sync_Interop.domain_adapters.notes_m1 import NotesM1SyncAdapter
from tldw_chatbook.Sync_Interop.domain_adapters._helpers import (
    decrypt_envelope_payload,
)
from tldw_chatbook.Sync_Interop.hashing import (
    canonical_payload_hash,
    canonical_thinking_blocks_json,
)
from tldw_chatbook.Sync_Interop.sync_state import (
    NOTES_ORGANIZATION_DOMAINS,
)

if TYPE_CHECKING:
    from tldw_chatbook.Notes.notes_organization_repository import (
        NotesOrganizationRepository,
    )
    from tldw_chatbook.tldw_api import SyncV2Envelope


class SyncEnvelopeApplier:
    """Route Sync v2 envelopes to small local domain adapters."""

    def __init__(
        self,
        *,
        local_store: Any,
        dataset_key: bytes | None = None,
        notes_mirror: Any = None,
        dataset_id: str | None = None,
        notes_organization_repository: NotesOrganizationRepository | None = None,
        notes_organization_server_profile_id: str | None = None,
        personal_context_adapter: Any = None,
        personal_context_service: Any = None,
    ) -> None:
        self.dataset_key = dataset_key
        self.local_store = local_store
        self.notes_mirror = notes_mirror
        self.dataset_id = dataset_id
        if (
            notes_organization_repository is not None
            and notes_organization_server_profile_id is not None
        ):
            from tldw_chatbook.Notes.notes_organization_repository import (
                NotesOrganizationRepository,
            )

            notes_organization_repository = NotesOrganizationRepository(
                notes_organization_repository.db,
                server_profile_id=notes_organization_server_profile_id,
            )
        self.notes_organization_repository = notes_organization_repository
        self.personal_context_adapter = personal_context_adapter
        self.personal_context_service = personal_context_service
        self.conflicts: list[dict[str, Any]] = []
        self._adapters: dict[str, Any] = {
            "notes": NotesSyncAdapter(),
            "chat": ChatSyncAdapter(),
            "workspaces": WorkspacesSyncAdapter(),
            "source_cache": SourceCacheSyncAdapter(),
            "media": MediaSyncAdapter(),
            "notes.note": NotesM1SyncAdapter(),
        }
        self._notes_organization_adapter: Any | None = None

    def apply(self, envelope: SyncV2Envelope) -> dict[str, Any]:
        if envelope.domain in NOTES_ORGANIZATION_DOMAINS:
            if self._notes_organization_adapter is None:
                from tldw_chatbook.Sync_Interop.domain_adapters.notes_organization import (
                    NotesOrganizationSyncAdapter,
                )

                self._notes_organization_adapter = NotesOrganizationSyncAdapter()
            return self._notes_organization_adapter.apply(
                envelope,
                repository=self.notes_organization_repository,
                restore_intent=(
                    envelope.routing_metadata.get("restore_intent") is True
                ),
                record_conflict=self._record_conflict,
            )
        if envelope.domain.startswith("personal_context."):
            return self._apply_personal_context(envelope)
        adapter = self._adapters.get(envelope.domain)
        if adapter is None:
            return self._record_conflict(
                envelope,
                conflict_type="unsupported_domain",
                message=f"Unsupported Sync v2 domain: {envelope.domain}",
            )
        if isinstance(adapter, NotesM1SyncAdapter):
            return adapter.apply(
                envelope,
                local_store=self.local_store,
                notes_mirror=self.notes_mirror,
                dataset_id=self.dataset_id,
                record_conflict=self._record_conflict,
            )
        if self.dataset_key is None:
            return self._record_conflict(
                envelope,
                conflict_type="missing_dataset_key",
                message="dataset_key is required to apply encrypted Sync v2 envelopes.",
            )
        local_store = self.local_store
        chat_store = None
        if envelope.domain == "chat":
            try:
                chat_store = _ContinuationValidatingChatStore(
                    local_store,
                    claimed_payload_hash=envelope.payload_hash,
                    payload=(
                        decrypt_envelope_payload(envelope, dataset_key=self.dataset_key)
                        if envelope.operation != "delete"
                        else None
                    ),
                )
            except _InvalidChatMessagePayload:
                return self._record_conflict(
                    envelope,
                    conflict_type="invalid_chat_message_payload",
                    message="The chat message payload is invalid.",
                )
            local_store = chat_store
            if chat_store.payload_hash is not None:
                envelope = envelope.model_copy(
                    update={"payload_hash": chat_store.payload_hash}
                )
        try:
            result = adapter.apply(
                envelope,
                dataset_key=self.dataset_key,
                local_store=local_store,
                record_conflict=self._record_conflict,
            )
        except _InvalidChatMessagePayload:
            return self._record_conflict(
                envelope,
                conflict_type="invalid_chat_message_payload",
                message="The chat message payload is invalid.",
            )
        if chat_store is not None and chat_store.warning is not None:
            return {**result, "warning": chat_store.warning}
        return result

    def _apply_personal_context(self, envelope: SyncV2Envelope) -> dict[str, Any]:
        if self.personal_context_adapter is None or self.personal_context_service is None:
            return {
                "status": "rejected",
                "reason_code": "personal_context_runtime_unavailable",
            }
        from tldw_chatbook.Personal_Context.service import ProfileConflictError
        from tldw_chatbook.Sync_Interop.personal_context_adapter import (
            PersonalContextSyncValidationError,
        )

        try:
            self.personal_context_adapter.apply_inbound(
                envelope,
                service=self.personal_context_service,
            )
        except PersonalContextSyncValidationError as exc:
            return {"status": "rejected", "reason_code": exc.reason_code}
        except ProfileConflictError:
            return self._record_conflict(
                envelope,
                conflict_type="personal_context_base_conflict",
                message="Personal Context base state changed.",
            )
        except Exception:
            return {
                "status": "rejected",
                "reason_code": "personal_context_apply_failed",
            }
        return {
            "status": "applied",
            "domain": envelope.domain,
            "entity_id": envelope.entity_id,
        }

    def _record_conflict(
        self,
        envelope: SyncV2Envelope,
        *,
        conflict_type: str,
        message: str | None = None,
    ) -> dict[str, Any]:
        conflict = {
            "domain": envelope.domain,
            "entity_id": envelope.entity_id,
            "stable_key": envelope.stable_key,
            "client_envelope_id": envelope.client_envelope_id,
            "conflict_type": conflict_type,
        }
        if message:
            conflict["message"] = message
        self.conflicts.append(conflict)
        record = getattr(self.local_store, "record_conflict", None)
        if callable(record):
            record(conflict)
        return {"status": "conflict", "conflict": conflict}


class _ContinuationValidatingChatStore:
    """Validate private Chat payload data before forwarding one whole record."""

    def __init__(
        self,
        store: Any,
        *,
        claimed_payload_hash: str,
        payload: dict[str, Any] | None,
    ) -> None:
        self.store = store
        self.warning: str | None = None
        if (
            payload is not None
            and canonical_payload_hash(payload) != claimed_payload_hash
        ):
            raise _InvalidChatMessagePayload
        self.payload = self._normalize_payload(payload) if payload is not None else None
        self.payload_hash = (
            canonical_payload_hash(self.payload) if self.payload is not None else None
        )

    def get_chat_message_hash(self, stable_key: str) -> str | None:
        reader = getattr(self.store, "get_chat_message_hash", None)
        return reader(stable_key) if callable(reader) else None

    def delete_chat_message(self, stable_key: str, payload_hash: str) -> None:
        deleter = getattr(self.store, "delete_chat_message", None)
        if callable(deleter):
            deleter(stable_key, payload_hash)

    def append_chat_message(
        self,
        stable_key: str,
        payload: dict[str, Any],
        payload_hash: str,
    ) -> None:
        if self.payload is None or self.payload_hash is None:
            raise _InvalidChatMessagePayload
        writer = getattr(self.store, "append_chat_message", None)
        if callable(writer):
            writer(stable_key, self.payload, self.payload_hash)

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the one canonical stored/hash payload for old and new envelopes."""
        payload = dict(payload)
        payload.setdefault("assistant_generation_state", None)
        allowed_keys = {"assistant_generation_state", "content", "role"}
        if "provider_continuation_json" in payload:
            allowed_keys.add("provider_continuation_json")
        if "thinking_blocks_json" in payload:
            allowed_keys.add("thinking_blocks_json")
        if (
            set(payload) != allowed_keys
            or type(payload.get("content")) is not str
            or type(payload.get("role")) is not str
        ):
            raise _InvalidChatMessagePayload

        thinking_value = payload.get("thinking_blocks_json")
        if "thinking_blocks_json" in payload:
            if payload["role"] != "assistant" or thinking_value is None:
                raise _InvalidChatMessagePayload
            try:
                payload["thinking_blocks_json"] = canonical_thinking_blocks_json(
                    thinking_value
                )
            except (TypeError, ValueError):
                raise _InvalidChatMessagePayload from None

        private_value = payload.get("provider_continuation_json")
        active_continuation = False
        if "provider_continuation_json" in payload and private_value is not None:
            safe_read = read_provider_continuation_json(private_value)
            if payload.get("role") == "assistant" and safe_read.checkpoint is not None:
                active_continuation = safe_read.checkpoint.state == "active"
                payload = {
                    **payload,
                    "provider_continuation_json": dump_provider_continuation_json(
                        safe_read.checkpoint
                    ),
                }
            else:
                payload = dict(payload)
                payload.pop("provider_continuation_json", None)
                self.warning = safe_read.warning or (
                    "Exact tool continuation was discarded."
                )
        elif "provider_continuation_json" in payload:
            payload.pop("provider_continuation_json", None)

        raw_state = payload["assistant_generation_state"]
        if raw_state is not None and payload["role"] != "assistant":
            raise _InvalidChatMessagePayload
        try:
            state = normalize_assistant_generation_state(
                role=payload["role"],
                raw_state=raw_state,
                has_valid_active_continuation=active_continuation,
            )
        except ValueError:
            raise _InvalidChatMessagePayload from None
        if (
            state is AssistantGenerationState.CONTINUATION_ACTIVE
            and not active_continuation
        ):
            raise _InvalidChatMessagePayload
        payload["assistant_generation_state"] = (
            state.value if state is not None else None
        )

        return payload


class _InvalidChatMessagePayload(Exception):
    """Pulled Chat payload failed its exact compatibility contract."""
