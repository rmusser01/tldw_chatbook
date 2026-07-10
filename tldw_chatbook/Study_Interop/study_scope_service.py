"""Mode-aware routing for the study flashcards compat seam."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping

from .study_normalizers import (
    merge_review_outcome_record,
    normalize_study_deck_record,
    normalize_study_flashcard_record,
    normalize_study_review_candidate,
)


class StudyBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class StudyCompatRecord(dict):
    """Dict record that remains equal to legacy raw payload projections."""

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            return all(self.get(key) == value for key, value in other.items())
        if isinstance(other, (bytes, bytearray)):
            return self.get("content") == bytes(other)
        return super().__eq__(other)


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "study.packs.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Study-pack generation jobs are server-only; use local decks and flashcards offline.",
        "affected_action_ids": [],
    },
    {
        "operation_id": "study.suggestions.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Study suggestions are server-only; use local study review flows offline.",
        "affected_action_ids": [],
    },
    {
        "operation_id": "study.workspace.local",
        "source": "local",
        "supported": False,
        "reason_code": "scope_not_supported",
        "user_message": "Workspace-scoped study decks and reviews are unavailable in local mode; use global local decks or server workspace mode.",
        "affected_action_ids": [
            "study.deck.create.local",
            "study.deck.list.local",
            "study.deck.update.local",
        ],
    },
    {
        "operation_id": "study.flashcard_helpers.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server flashcard helpers such as assets, imports, exports, analytics, tag suggestions, assistant, generation, templates, and review-session discovery are unavailable in local/offline mode.",
        "affected_action_ids": [
            "study.flashcard.analytics.observe.local",
            "study.flashcard.assistant.detail.local",
            "study.flashcard.assistant.launch.local",
            "study.flashcard.assets.create.local",
            "study.flashcard.assets.detail.local",
            "study.flashcard.export.export.local",
            "study.flashcard.generation.launch.local",
            "study.flashcard.import.import.local",
            "study.flashcard.import.preview.local",
            "study.flashcard.review_sessions.list.local",
            "study.flashcard.templates.create.local",
            "study.flashcard.templates.delete.local",
            "study.flashcard.templates.detail.local",
            "study.flashcard.templates.list.local",
            "study.flashcard.templates.update.local",
        ],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES: list[dict[str, Any]] = []


class StudyScopeService:
    """Route study flashcard actions to local or server backends and normalize outputs."""

    _ALLOWED_SCOPE_TYPES = {"global", "workspace"}

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: StudyBackend | str | None) -> StudyBackend:
        if mode is None:
            return StudyBackend.LOCAL
        if isinstance(mode, StudyBackend):
            return mode
        try:
            return StudyBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid study backend: {mode}") from exc

    def _service_for_mode(self, mode: StudyBackend) -> Any:
        if mode == StudyBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local study backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server study backend is unavailable.")
        return self.server_service

    def _server_only_service(self, mode: StudyBackend | str | None, feature_label: str) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == StudyBackend.LOCAL:
            raise ValueError(f"{feature_label} are server-only in Chatbook.")
        return self._service_for_mode(normalized_mode)

    def _enforce_policy(self, mode: StudyBackend, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=f"{action_id}.{mode.value}")

    @staticmethod
    def _with_source(source: str, payload: Any) -> Any:
        if isinstance(payload, Mapping):
            result = dict(payload)
            result.setdefault("source", source)
            return result
        return payload

    @staticmethod
    def _with_server_source(payload: Any) -> Any:
        return StudyScopeService._with_source("server", payload)

    @classmethod
    def _with_template_record(cls, source: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(cls._with_source(source, payload))
        result.setdefault("entity_kind", "flashcard_template")
        template_id = result.get("id")
        if template_id is not None:
            result.setdefault("record_id", f"{source}:flashcard_template:{template_id}")
        return result

    @classmethod
    def _with_template_list(cls, source: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(cls._with_source(source, payload))
        result.setdefault("entity_kind", "flashcard_template_list")
        result["items"] = [
            cls._with_template_record(source, item)
            for item in list(result.get("items") or [])
            if isinstance(item, Mapping)
        ]
        return result

    @classmethod
    def _with_server_template_record(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        return cls._with_template_record("server", payload)

    @classmethod
    def _with_server_template_list(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        return cls._with_template_list("server", payload)

    @classmethod
    def _with_server_flashcard_list(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        return cls._with_flashcard_list("server", payload)

    @classmethod
    def _with_flashcard_list(cls, source: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(cls._with_server_source(payload))
        result["source"] = source
        result.setdefault("entity_kind", "study_flashcard_list")
        result["items"] = [
            normalize_study_flashcard_record(source, item)
            for item in list(result.get("items") or [])
            if isinstance(item, Mapping)
        ]
        return result

    @classmethod
    def _with_server_bulk_update(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        return cls._with_bulk_update("server", payload)

    @classmethod
    def _with_bulk_update(cls, source: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(cls._with_server_source(payload))
        result["source"] = source
        result.setdefault("entity_kind", "study_flashcard_bulk_update")
        normalized_results: list[dict[str, Any]] = []
        for item in list(result.get("results") or []):
            if not isinstance(item, Mapping):
                continue
            normalized = dict(item)
            flashcard = normalized.get("flashcard")
            if isinstance(flashcard, Mapping):
                normalized["flashcard"] = normalize_study_flashcard_record(source, flashcard)
            normalized_results.append(normalized)
        result["results"] = normalized_results
        return result

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _deck_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.deck.{action}.{mode.value}"

    @staticmethod
    def _flashcard_action_id(mode: StudyBackend, *, mutation: bool) -> str:
        # Keep legacy flashcard list/create/move/delete policy behavior stable for
        # the current UI while newer helper routes use explicit flashcard IDs.
        action = "update" if mutation else "detail"
        return f"study.deck.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_tags_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.tags.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_analytics_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.analytics.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_review_sessions_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.review_sessions.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_assistant_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.assistant.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_generation_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.generation.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_template_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.templates.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_asset_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.assets.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_bulk_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.bulk.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_import_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.import.{action}.{mode.value}"

    @staticmethod
    def _study_flashcard_export_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.flashcard.export.{action}.{mode.value}"

    @staticmethod
    def _study_pack_action_id(action: str) -> str:
        return f"study.packs.jobs.{action}.server"

    @staticmethod
    def _study_suggestion_action_id(action: str) -> str:
        return f"study.suggestions.{action}.server"

    @staticmethod
    def _require_server_only(mode: StudyBackend, feature_name: str) -> None:
        if mode != StudyBackend.SERVER:
            raise ValueError(f"{feature_name} are server-only.")

    @staticmethod
    def _coerce_delete_result(result: Any) -> bool:
        if isinstance(result, Mapping):
            if "deleted" in result:
                return bool(result.get("deleted"))
            return str(result.get("status") or "").strip().lower() == "deleted"
        return bool(result)

    @classmethod
    def _to_plain(cls, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return {key: cls._to_plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._to_plain(item) for item in value]
        return value

    @staticmethod
    def _with_backend_record(
        *,
        backend: StudyBackend,
        kind: str,
        payload: Mapping[str, Any],
        source_id: Any,
    ) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", backend.value)
        if source_id is not None:
            record.setdefault("record_id", f"{backend.value}:{kind}:{source_id}")
        return record

    def _normalize_study_pack_payload(self, payload: Any) -> Any:
        if not isinstance(payload, Mapping):
            return payload
        record = dict(payload)
        record.setdefault("backend", StudyBackend.SERVER.value)
        if isinstance(record.get("job"), Mapping):
            job = record["job"]
            record["job"] = self._with_backend_record(
                backend=StudyBackend.SERVER,
                kind="study_pack_job",
                payload=job,
                source_id=job.get("id"),
            )
        if isinstance(record.get("jobs"), list):
            record["jobs"] = [
                self._with_backend_record(
                    backend=StudyBackend.SERVER,
                    kind="study_pack_job",
                    payload=job,
                    source_id=job.get("id"),
                )
                for job in record["jobs"]
                if isinstance(job, Mapping)
            ]
        if isinstance(record.get("study_pack"), Mapping):
            study_pack = record["study_pack"]
            record["study_pack"] = self._with_backend_record(
                backend=StudyBackend.SERVER,
                kind="study_pack",
                payload=study_pack,
                source_id=study_pack.get("id"),
            )
        if "id" in record and "title" in record:
            return self._with_backend_record(
                backend=StudyBackend.SERVER,
                kind="study_pack",
                payload=record,
                source_id=record.get("id"),
            )
        return record

    def _normalize_study_suggestion_payload(self, payload: Any) -> Any:
        if not isinstance(payload, Mapping):
            return payload
        record = dict(payload)
        record.setdefault("backend", StudyBackend.SERVER.value)
        if record.get("anchor_type") is not None and record.get("anchor_id") is not None:
            record.setdefault(
                "record_id",
                f"server:study_suggestion_status:{record.get('anchor_type')}:{record.get('anchor_id')}",
            )
        if isinstance(record.get("snapshot"), Mapping):
            snapshot = record["snapshot"]
            record["snapshot"] = self._with_backend_record(
                backend=StudyBackend.SERVER,
                kind="study_suggestion_snapshot",
                payload=snapshot,
                source_id=snapshot.get("id"),
            )
        if isinstance(record.get("job"), Mapping):
            job = record["job"]
            record["job"] = self._with_backend_record(
                backend=StudyBackend.SERVER,
                kind="study_suggestion_job",
                payload=job,
                source_id=job.get("id"),
            )
        if record.get("snapshot_id") is not None and record.get("selection_fingerprint") is not None:
            record.setdefault(
                "record_id",
                "server:study_suggestion_action:"
                f"{record.get('snapshot_id')}:{record.get('selection_fingerprint')}",
            )
        return record

    def _normalize_tags_payload(
        self,
        *,
        backend: StudyBackend,
        card_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        record = dict(payload or {}) if isinstance(payload, Mapping) else {"tags": payload}
        tags = record.get("tags")
        items = record.get("items")
        if tags is None and items is not None:
            tags = items
        if tags is not None:
            if isinstance(tags, str):
                normalized_tags = [item for item in tags.split() if item]
            else:
                normalized_tags = [str(item).strip() for item in list(tags or []) if str(item).strip()]
            record["tags"] = normalized_tags
            record.setdefault("items", normalized_tags)
            record.setdefault("count", len(normalized_tags))
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_tags")
        record.setdefault("uuid", card_id)
        return StudyCompatRecord(self._with_backend_record(
            backend=backend,
            kind="study_flashcard_tags",
            payload=record,
            source_id=card_id,
        ))

    def _normalize_tag_suggestions_payload(
        self,
        *,
        backend: StudyBackend,
        q: str | None,
        payload: Any,
    ) -> dict[str, Any]:
        record = dict(payload or {}) if isinstance(payload, Mapping) else {"items": list(payload or [])}
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_tag_suggestions")
        source_id = str(q or "all").strip() or "all"
        return StudyCompatRecord(self._with_backend_record(
            backend=backend,
            kind="study_flashcard_tag_suggestions",
            payload=record,
            source_id=source_id,
        ))

    def _normalize_analytics_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
        deck_id: Any = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        record = dict(payload or {}) if isinstance(payload, Mapping) else {"summary": payload}
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_analytics")
        if workspace_id:
            source_id = f"workspace:{workspace_id}"
        elif deck_id is not None:
            source_id = f"deck:{deck_id}"
        else:
            source_id = "all"
        return self._with_backend_record(
            backend=backend,
            kind="study_flashcard_analytics",
            payload=record,
            source_id=source_id,
        )

    def _normalize_review_sessions_payload(
        self,
        *,
        backend: StudyBackend,
        records: Any,
    ) -> list[dict[str, Any]]:
        return [
            self._with_backend_record(
                backend=backend,
                kind="study_review_session",
                payload=record,
                source_id=record.get("id") if isinstance(record, Mapping) else None,
            )
            for record in list(records or [])
        ]

    def _normalize_assistant_payload(
        self,
        *,
        backend: StudyBackend,
        card_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return self._with_backend_record(
                backend=backend,
                kind="study_flashcard_assistant",
                payload={"payload": payload},
                source_id=card_id,
            )
        record = dict(payload)
        record = self._with_backend_record(
            backend=backend,
            kind="study_flashcard_assistant",
            payload=record,
            source_id=card_id,
        )
        if isinstance(record.get("thread"), Mapping):
            thread = dict(record["thread"])
            record["thread"] = self._with_backend_record(
                backend=backend,
                kind="study_flashcard_assistant_thread",
                payload=thread,
                source_id=thread.get("id"),
            )
        for message_key in ("messages",):
            if isinstance(record.get(message_key), list):
                record[message_key] = [
                    self._with_backend_record(
                        backend=backend,
                        kind="study_flashcard_assistant_message",
                        payload=message,
                        source_id=message.get("id") if isinstance(message, Mapping) else None,
                    )
                    for message in record[message_key]
                ]
        for message_key in ("user_message", "assistant_message"):
            if isinstance(record.get(message_key), Mapping):
                message = dict(record[message_key])
                record[message_key] = self._with_backend_record(
                    backend=backend,
                    kind="study_flashcard_assistant_message",
                    payload=message,
                    source_id=message.get("id"),
                )
        return record

    def _normalize_generation_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
    ) -> dict[str, Any]:
        record = dict(payload or {}) if isinstance(payload, Mapping) else {"flashcards": list(payload or [])}
        return self._with_backend_record(
            backend=backend,
            kind="study_flashcard_generation",
            payload=record,
            source_id="transient",
        )

    def _normalize_template_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
    ) -> dict[str, Any]:
        record = dict(payload or {}) if isinstance(payload, Mapping) else {"payload": payload}
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_template")
        return self._with_backend_record(
            backend=backend,
            kind="study_flashcard_template",
            payload=record,
            source_id=record.get("id"),
        )

    def _normalize_template_list_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        if isinstance(payload, Mapping):
            record = dict(payload)
            items = list(record.get("items") or [])
        else:
            items = list(payload or [])
            record = {"items": items, "count": len(items)}
        record["items"] = [
            self._normalize_template_payload(backend=backend, payload=item)
            for item in items
        ]
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_template_list")
        return self._with_backend_record(
            backend=backend,
            kind="study_flashcard_template_list",
            payload=record,
            source_id=f"{limit}:{offset}",
        )

    def _normalize_asset_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
    ) -> dict[str, Any]:
        record = self._to_plain(payload)
        if not isinstance(record, Mapping):
            record = {"payload": record}
        record = dict(record)
        source_id = record.get("asset_uuid") or record.get("uuid")
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_asset")
        return StudyCompatRecord(self._with_backend_record(
            backend=backend,
            kind="study_flashcard_asset",
            payload=record,
            source_id=source_id,
        ))

    def _normalize_asset_content_payload(
        self,
        *,
        backend: StudyBackend,
        asset_uuid: str,
        payload: Any,
    ) -> dict[str, Any]:
        record = self._to_plain(payload)
        if not isinstance(record, Mapping):
            record = {"content": record}
        record = dict(record)
        record.setdefault("asset_uuid", asset_uuid)
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_asset_content")
        return StudyCompatRecord(self._with_backend_record(
            backend=backend,
            kind="study_flashcard_asset_content",
            payload=record,
            source_id=asset_uuid,
        ))

    def _normalize_bulk_create_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
    ) -> dict[str, Any]:
        record = self._to_plain(payload)
        if not isinstance(record, Mapping):
            record = {"items": list(record or [])}
        record = dict(record)
        record["items"] = [
            normalize_study_flashcard_record(backend.value, item)
            for item in list(record.get("items") or [])
        ]
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "study_flashcard_bulk_create")
        return self._with_backend_record(
            backend=backend,
            kind="study_flashcard_bulk_create",
            payload=record,
            source_id="transient",
        )

    def _normalize_bulk_update_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
    ) -> dict[str, Any]:
        record = self._to_plain(payload)
        if not isinstance(record, Mapping):
            record = {"results": list(record or [])}
        record = dict(record)
        normalized_results: list[dict[str, Any]] = []
        for result in list(record.get("results") or []):
            if not isinstance(result, Mapping):
                normalized_results.append({"payload": result})
                continue
            normalized_result = dict(result)
            if isinstance(normalized_result.get("flashcard"), Mapping):
                normalized_result["flashcard"] = normalize_study_flashcard_record(
                    backend.value,
                    normalized_result["flashcard"],
                )
            normalized_results.append(normalized_result)
        record["results"] = normalized_results
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "study_flashcard_bulk_update")
        return self._with_backend_record(
            backend=backend,
            kind="study_flashcard_bulk_update",
            payload=record,
            source_id="transient",
        )

    def _normalize_import_preview_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
    ) -> dict[str, Any]:
        record = self._to_plain(payload)
        if not isinstance(record, Mapping):
            record = {"drafts": list(record or [])}
        record = dict(record)
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_import_preview")
        return StudyCompatRecord(self._with_backend_record(
            backend=backend,
            kind="study_flashcard_import_preview",
            payload=record,
            source_id="transient",
        ))

    def _normalize_import_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
        import_kind: str,
    ) -> dict[str, Any]:
        record = self._to_plain(payload)
        if not isinstance(record, Mapping):
            record = {"items": list(record or [])}
        record = dict(record)
        if isinstance(record.get("items"), list):
            normalized_items: list[Any] = []
            for item in record["items"]:
                if not isinstance(item, Mapping):
                    normalized_items.append(item)
                    continue
                normalized_item = normalize_study_flashcard_record(backend.value, item)
                if item.get("uuid") is not None:
                    normalized_item.setdefault("uuid", item.get("uuid"))
                if item.get("id") is not None:
                    normalized_item.setdefault("id", item.get("id"))
                normalized_items.append(normalized_item)
            record["items"] = normalized_items
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_import")
        return StudyCompatRecord(self._with_backend_record(
            backend=backend,
            kind="study_flashcard_import",
            payload=record,
            source_id=import_kind,
        ))

    def _normalize_export_payload(
        self,
        *,
        backend: StudyBackend,
        payload: Any,
    ) -> dict[str, Any]:
        record = self._to_plain(payload)
        if not isinstance(record, Mapping):
            record = {"content": record}
        record = dict(record)
        record.setdefault("source", backend.value)
        record.setdefault("entity_kind", "flashcard_export")
        return StudyCompatRecord(self._with_backend_record(
            backend=backend,
            kind="study_flashcard_export",
            payload=record,
            source_id="transient",
        ))

    def list_unsupported_capabilities(
        self,
        *,
        mode: StudyBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == StudyBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    @classmethod
    def _normalize_scope(cls, scope_type: str | None, workspace_id: str | None) -> tuple[str, str | None]:
        normalized_scope = str(scope_type or "global").strip().lower()
        if normalized_scope not in cls._ALLOWED_SCOPE_TYPES:
            raise ValueError(f"Invalid study scope_type: {scope_type}")
        normalized_workspace_id = str(workspace_id or "").strip() or None
        if normalized_scope == "workspace" and normalized_workspace_id is None:
            raise ValueError("workspace_id is required when scope_type='workspace'")
        return normalized_scope, normalized_workspace_id

    @staticmethod
    def _filter_deck_scope(record: Mapping[str, Any], *, scope_type: str, workspace_id: str | None) -> bool:
        record_workspace_id = record.get("workspace_id")
        if scope_type == "global":
            return record_workspace_id is None
        return record_workspace_id == workspace_id

    async def _load_scoped_server_decks(
        self,
        *,
        service: Any,
        limit: int,
        scope_type: str,
        workspace_id: str | None,
    ) -> list[dict[str, Any]]:
        page_size = max(1, int(limit or 100))
        fetched_records: list[dict[str, Any]] = []
        page_offset = 0
        while True:
            page = await self._maybe_await(service.list_decks(limit=page_size, offset=page_offset))
            page_items = list(page or [])
            fetched_records.extend(page_items)
            if len(page_items) < page_size:
                break
            page_offset += page_size

        normalized_records = [
            normalize_study_deck_record("server", record)
            for record in fetched_records
            if self._filter_deck_scope(record, scope_type=scope_type, workspace_id=workspace_id)
        ]
        return normalized_records

    async def list_decks(
        self,
        *,
        mode: StudyBackend | str | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._deck_action_id(normalized_mode, "list"))
        normalized_scope_type, normalized_workspace_id = self._normalize_scope(scope_type, workspace_id)
        service = self._service_for_mode(normalized_mode)

        if normalized_mode == StudyBackend.LOCAL:
            if normalized_scope_type == "workspace":
                raise ValueError("Workspace Study is unavailable in local mode")
            records = await self._maybe_await(service.list_decks(limit=limit, offset=offset))
            return [
                normalize_study_deck_record(normalized_mode.value, record)
                for record in list(records or [])
            ]

        if normalized_scope_type == "workspace":
            records = await self._load_scoped_server_decks(
                service=service,
                limit=limit,
                scope_type=normalized_scope_type,
                workspace_id=normalized_workspace_id,
            )
        else:
            records = await self._load_scoped_server_decks(
                service=service,
                limit=limit,
                scope_type=normalized_scope_type,
                workspace_id=None,
            )

        if offset:
            records = records[offset:]
        if limit >= 0:
            records = records[:limit]
        return records

    async def count_decks(
        self,
        *,
        mode: StudyBackend | str | None = None,
    ) -> int:
        """Count non-deleted flashcard decks in the given mode.

        Args:
            mode: The study backend to count in; only ``StudyBackend.LOCAL``
                is supported (see Raises).

        Returns:
            The exact number of non-deleted local decks.

        Raises:
            ValueError: For server mode (no count-only backend seam exists;
                see the inline comment).
        """
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._deck_action_id(normalized_mode, "list"))
        if normalized_mode == StudyBackend.LOCAL:
            return int(await self._maybe_await(self._service_for_mode(normalized_mode).count_decks()))
        # The server study backend does not expose a dedicated count-only
        # seam today (``list_decks``'s server branch pages through the full
        # deck collection to build normalized records). Rather than issuing
        # a full paginated fetch just to read a number, mirror the
        # unsupported-count contract established by
        # ``NotesScopeService.count_notes``.
        raise ValueError("Server study deck counts are not supported; use list_decks for a scoped total.")

    async def count_due_flashcards(
        self,
        *,
        mode: StudyBackend | str | None = None,
    ) -> int:
        """Count flashcards due for review in the given mode.

        Args:
            mode: The study backend to count in; only ``StudyBackend.LOCAL``
                is supported (see Raises).

        Returns:
            The exact number of local flashcards due for review.

        Raises:
            ValueError: For server mode (no count-only backend seam exists;
                see the inline comment).
        """
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=False))
        if normalized_mode == StudyBackend.LOCAL:
            return int(await self._maybe_await(self._service_for_mode(normalized_mode).count_due_flashcards()))
        # Mirror ``count_decks``'s unsupported-count contract: the server
        # study backend has no dedicated due-flashcard count seam.
        raise ValueError("Server due-flashcard counts are not supported; use list_flashcards for a scoped total.")

    async def create_deck(
        self,
        *,
        mode: StudyBackend | str | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        name: str,
        description: str | None = None,
        scheduler_type: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._deck_action_id(normalized_mode, "create"))
        normalized_scope_type, normalized_workspace_id = self._normalize_scope(scope_type, workspace_id)
        if normalized_mode == StudyBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        record = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_deck(
                name=name,
                description=description,
                workspace_id=normalized_workspace_id if normalized_scope_type == "workspace" else None,
                scheduler_type=scheduler_type,
            )
        )
        return normalize_study_deck_record(normalized_mode.value, record)

    async def update_deck(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int,
        name: str | None = None,
        description: str | None = None,
        workspace_id: str | None = None,
        review_prompt_side: str | None = None,
        scheduler_type: str | None = None,
        scheduler_settings: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._deck_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "update_deck"):
            raise ValueError(f"Study deck update is unavailable in {normalized_mode.value} mode.")
        record = await self._maybe_await(
            service.update_deck(
                deck_id,
                name=name,
                description=description,
                workspace_id=workspace_id,
                review_prompt_side=review_prompt_side,
                scheduler_type=scheduler_type,
                scheduler_settings=scheduler_settings,
                expected_version=expected_version,
            )
        )
        return normalize_study_deck_record(normalized_mode.value, record)

    async def list_flashcards(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=False))
        service = self._service_for_mode(normalized_mode)
        raw_records = await self._maybe_await(
            service.list_flashcards(deck_id=deck_id, q=q, limit=limit, offset=offset)
        )
        return [
            normalize_study_flashcard_record(normalized_mode.value, record)
            for record in list(raw_records or [])
        ]

    async def create_flashcard(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int | None,
        front: str,
        back: str,
        tags: list[str] | None = None,
        notes: str | None = None,
        extra: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
        record = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_flashcard(
                deck_id=deck_id,
                front=front,
                back=back,
                tags=tags,
                notes=notes,
                extra=extra,
            )
        )
        return normalize_study_flashcard_record(normalized_mode.value, record)

    async def get_flashcard(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_action_id(normalized_mode, "detail"))
        record = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_flashcard(card_id)
        )
        return normalize_study_flashcard_record(normalized_mode.value, record)

    async def reset_flashcard_scheduling(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        expected_version: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard scheduling reset")
        self._enforce_policy(self._study_flashcard_action_id(normalized_mode, "update"))
        record = await self._maybe_await(
            self._service_for_mode(normalized_mode).reset_flashcard_scheduling(
                card_id,
                expected_version=expected_version,
            )
        )
        return normalize_study_flashcard_record(normalized_mode.value, record)

    async def set_flashcard_tags(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        tags: list[str],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_tags_action_id(normalized_mode, "update"))
        record = await self._maybe_await(
            self._service_for_mode(normalized_mode).set_flashcard_tags(card_id, tags=tags)
        )
        return normalize_study_flashcard_record(normalized_mode.value, record)

    async def get_flashcard_tags(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_tags_action_id(normalized_mode, "list"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_flashcard_tags(card_id)
        )
        return self._normalize_tags_payload(backend=normalized_mode, card_id=card_id, payload=result)

    async def list_flashcard_tag_suggestions(
        self,
        *,
        mode: StudyBackend | str | None = None,
        q: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard tag suggestions")
        self._enforce_policy(self._study_flashcard_tags_action_id(normalized_mode, "list"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_flashcard_tag_suggestions(q=q, limit=limit)
        )
        return self._normalize_tag_suggestions_payload(backend=normalized_mode, q=q, payload=result)

    async def get_flashcard_analytics_summary(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard analytics")
        self._enforce_policy(self._study_flashcard_analytics_action_id(normalized_mode, "observe"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_flashcard_analytics_summary(
                deck_id=deck_id,
                workspace_id=workspace_id,
                include_workspace_items=include_workspace_items,
            )
        )
        return self._normalize_analytics_payload(
            backend=normalized_mode,
            payload=result,
            deck_id=deck_id,
            workspace_id=workspace_id,
        )

    async def list_review_sessions(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int | None = None,
        scope_key: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard review sessions")
        self._enforce_policy(self._study_flashcard_review_sessions_action_id(normalized_mode, "list"))
        records = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_review_sessions(
                deck_id=deck_id,
                scope_key=scope_key,
                status=status,
                limit=limit,
            )
        )
        return self._normalize_review_sessions_payload(backend=normalized_mode, records=records)

    async def get_flashcard_assistant(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard assistant")
        self._enforce_policy(self._study_flashcard_assistant_action_id(normalized_mode, "detail"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_flashcard_assistant(card_id)
        )
        return self._normalize_assistant_payload(backend=normalized_mode, card_id=card_id, payload=result)

    async def respond_flashcard_assistant(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        action: str,
        message: str | None = None,
        input_modality: str = "text",
        provider: str | None = None,
        model: str | None = None,
        expected_thread_version: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard assistant")
        self._enforce_policy(self._study_flashcard_assistant_action_id(normalized_mode, "launch"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).respond_flashcard_assistant(
                card_id,
                action=action,
                message=message,
                input_modality=input_modality,
                provider=provider,
                model=model,
                expected_thread_version=expected_thread_version,
            )
        )
        return self._normalize_assistant_payload(backend=normalized_mode, card_id=card_id, payload=result)

    async def generate_flashcards(
        self,
        *,
        mode: StudyBackend | str | None = None,
        text: str,
        num_cards: int = 10,
        card_type: str = "basic",
        difficulty: str = "mixed",
        focus_topics: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard generation")
        self._enforce_policy(self._study_flashcard_generation_action_id(normalized_mode, "launch"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).generate_flashcards(
                text=text,
                num_cards=num_cards,
                card_type=card_type,
                difficulty=difficulty,
                focus_topics=focus_topics or [],
                provider=provider,
                model=model,
            )
        )
        return self._normalize_generation_payload(backend=normalized_mode, payload=result)

    async def upload_flashcard_asset(
        self,
        *,
        mode: StudyBackend | str | None = None,
        file: tuple[str, bytes, str],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard asset upload")
        self._enforce_policy(self._study_flashcard_asset_action_id(normalized_mode, "create"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).upload_flashcard_asset(file)
        )
        return self._normalize_asset_payload(backend=normalized_mode, payload=result)

    async def get_flashcard_asset_content(
        self,
        *,
        mode: StudyBackend | str | None = None,
        asset_uuid: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard asset content")
        self._enforce_policy(self._study_flashcard_asset_action_id(normalized_mode, "detail"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_flashcard_asset_content(asset_uuid)
        )
        return self._normalize_asset_content_payload(
            backend=normalized_mode,
            asset_uuid=asset_uuid,
            payload=result,
        )

    async def create_flashcards_bulk(
        self,
        *,
        mode: StudyBackend | str | None = None,
        cards: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_bulk_action_id(normalized_mode, "create"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_flashcards_bulk(cards)
        )
        return self._normalize_bulk_create_payload(backend=normalized_mode, payload=result)

    async def update_flashcards_bulk(
        self,
        *,
        mode: StudyBackend | str | None = None,
        updates: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_bulk_action_id(normalized_mode, "update"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).update_flashcards_bulk(updates)
        )
        return self._normalize_bulk_update_payload(backend=normalized_mode, payload=result)

    async def preview_structured_qa_import(
        self,
        *,
        mode: StudyBackend | str | None = None,
        content: str,
        max_lines: int | None = None,
        max_line_length: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard import preview")
        self._enforce_policy(self._study_flashcard_import_action_id(normalized_mode, "preview"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).preview_structured_qa_import(
                content,
                max_lines=max_lines,
                max_line_length=max_line_length,
                max_field_length=max_field_length,
            )
        )
        return self._normalize_import_preview_payload(backend=normalized_mode, payload=result)

    async def import_flashcards(
        self,
        *,
        mode: StudyBackend | str | None = None,
        content: str,
        delimiter: str | None = "\t",
        has_header: bool | None = False,
        max_lines: int | None = None,
        max_line_length: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard import")
        self._enforce_policy(self._study_flashcard_import_action_id(normalized_mode, "import"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).import_flashcards(
                content,
                delimiter=delimiter,
                has_header=has_header,
                max_lines=max_lines,
                max_line_length=max_line_length,
                max_field_length=max_field_length,
            )
        )
        return self._normalize_import_payload(
            backend=normalized_mode,
            payload=result,
            import_kind="structured",
        )

    async def import_flashcards_json(
        self,
        *,
        mode: StudyBackend | str | None = None,
        file: tuple[str, bytes, str],
        max_items: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard JSON import")
        self._enforce_policy(self._study_flashcard_import_action_id(normalized_mode, "import"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).import_flashcards_json(
                file,
                max_items=max_items,
                max_field_length=max_field_length,
            )
        )
        return self._normalize_import_payload(
            backend=normalized_mode,
            payload=result,
            import_kind="json",
        )

    async def import_flashcards_apkg(
        self,
        *,
        mode: StudyBackend | str | None = None,
        file: tuple[str, bytes, str],
        max_items: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard APKG import")
        self._enforce_policy(self._study_flashcard_import_action_id(normalized_mode, "import"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).import_flashcards_apkg(
                file,
                max_items=max_items,
                max_field_length=max_field_length,
            )
        )
        return self._normalize_import_payload(
            backend=normalized_mode,
            payload=result,
            import_kind="apkg",
        )

    async def export_flashcards(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool | None = None,
        tag: str | None = None,
        q: str | None = None,
        format: str = "csv",
        include_reverse: bool | None = None,
        delimiter: str | None = None,
        include_header: bool | None = None,
        extended_header: bool | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard export")
        self._enforce_policy(self._study_flashcard_export_action_id(normalized_mode, "export"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).export_flashcards(
                deck_id=deck_id,
                workspace_id=workspace_id,
                include_workspace_items=include_workspace_items,
                tag=tag,
                q=q,
                format=format,
                include_reverse=include_reverse,
                delimiter=delimiter,
                include_header=include_header,
                extended_header=extended_header,
            )
        )
        return self._normalize_export_payload(backend=normalized_mode, payload=result)

    async def create_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        name: str,
        front_template: str,
        model_type: str = "basic",
        back_template: str | None = None,
        notes_template: str | None = None,
        extra_template: str | None = None,
        placeholder_definitions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard templates")
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "create"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_flashcard_template(
                name=name,
                model_type=model_type,
                front_template=front_template,
                back_template=back_template,
                notes_template=notes_template,
                extra_template=extra_template,
                placeholder_definitions=placeholder_definitions or [],
            )
        )
        return self._normalize_template_payload(backend=normalized_mode, payload=result)

    async def list_flashcard_templates(
        self,
        *,
        mode: StudyBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard templates")
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "list"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_flashcard_templates(limit=limit, offset=offset)
        )
        return self._normalize_template_list_payload(
            backend=normalized_mode,
            payload=result,
            limit=limit,
            offset=offset,
        )

    async def get_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        template_id: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard templates")
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "detail"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_flashcard_template(template_id)
        )
        return self._normalize_template_payload(backend=normalized_mode, payload=result)

    async def update_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        template_id: int,
        name: str | None = None,
        model_type: str | None = None,
        front_template: str | None = None,
        back_template: str | None = None,
        notes_template: str | None = None,
        extra_template: str | None = None,
        placeholder_definitions: list[dict[str, Any]] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard templates")
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "update"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).update_flashcard_template(
                template_id,
                name=name,
                model_type=model_type,
                front_template=front_template,
                back_template=back_template,
                notes_template=notes_template,
                extra_template=extra_template,
                placeholder_definitions=placeholder_definitions,
                expected_version=expected_version,
            )
        )
        return self._normalize_template_payload(backend=normalized_mode, payload=result)

    async def delete_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        template_id: int,
        expected_version: int,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard templates")
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "delete"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).delete_flashcard_template(
                template_id,
                expected_version=expected_version,
            )
        )
        return self._coerce_delete_result(result)

    async def move_flashcard(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        target_deck_id: str | int,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
        record = await self._maybe_await(
            self._service_for_mode(normalized_mode).move_flashcard(
                card_id,
                target_deck_id=target_deck_id,
                expected_version=expected_version,
            )
        )
        if not record:
            return None
        return normalize_study_flashcard_record(normalized_mode.value, record)

    async def delete_flashcard(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        expected_version: int | None = None,
        hard_delete: bool = False,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
        if normalized_mode == StudyBackend.SERVER:
            if expected_version is None:
                raise ValueError("expected_version is required for server flashcard deletion.")
            if expected_version < 1:
                raise ValueError("expected_version must be >= 1 for server flashcard deletion.")
        kwargs: dict[str, Any] = {"expected_version": expected_version}
        if normalized_mode == StudyBackend.LOCAL:
            kwargs["hard_delete"] = hard_delete
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).delete_flashcard(
                card_id,
                **kwargs,
            )
        )
        return self._coerce_delete_result(result)

    async def update_flashcards_bulk(
        self,
        *,
        mode: StudyBackend | str | None = None,
        updates: list[Mapping[str, Any]] | None = None,
        cards: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_bulk_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        update_payload = updates if updates is not None else cards
        if update_payload is None:
            raise ValueError("updates or cards are required for flashcard bulk update.")
        payload = await self._maybe_await(service.update_flashcards_bulk(update_payload))
        return self._normalize_bulk_update_payload(backend=normalized_mode, payload=payload)

    async def reset_flashcard_scheduling(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        expected_version: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(
            service.reset_flashcard_scheduling(card_id, expected_version=expected_version)
        )
        return normalize_study_flashcard_record(normalized_mode.value, record)

    async def set_flashcard_tags(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        tags: list[str],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_tags_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        normalized_tags = [str(tag).strip() for tag in list(tags or []) if str(tag).strip()]
        record = await self._maybe_await(service.set_flashcard_tags(card_id, tags=normalized_tags))
        normalized_record = normalize_study_flashcard_record(normalized_mode.value, record)
        normalized_record["tags"] = normalized_tags
        return normalized_record

    async def get_flashcard_tags(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_tags_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.get_flashcard_tags(card_id))
        return self._normalize_tags_payload(backend=normalized_mode, card_id=card_id, payload=result)

    async def list_flashcard_tag_suggestions(
        self,
        *,
        mode: StudyBackend | str | None = None,
        q: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_tags_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.list_flashcard_tag_suggestions(q=q, limit=limit))
        return self._normalize_tag_suggestions_payload(backend=normalized_mode, q=q, payload=payload)

    async def preview_structured_qa_import(
        self,
        *,
        mode: StudyBackend | str | None = None,
        content: str,
        max_lines: int | None = None,
        max_line_length: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_import_action_id(normalized_mode, "preview"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.preview_structured_qa_import(
                content,
                max_lines=max_lines,
                max_line_length=max_line_length,
                max_field_length=max_field_length,
            )
        )
        return self._normalize_import_preview_payload(backend=normalized_mode, payload=payload)

    async def import_flashcards_tsv(
        self,
        *,
        mode: StudyBackend | str | None = None,
        content: str,
        delimiter: str = "\t",
        has_header: bool = False,
        max_lines: int | None = None,
        max_line_length: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.import_flashcards_tsv(
                content,
                delimiter=delimiter,
                has_header=has_header,
                max_lines=max_lines,
                max_line_length=max_line_length,
                max_field_length=max_field_length,
            )
        )
        result = dict(payload or {})
        result.setdefault("source", normalized_mode.value)
        result.setdefault("entity_kind", "flashcard_import")
        return result

    async def export_flashcards(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: int | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool = False,
        tag: str | None = None,
        q: str | None = None,
        format: str | None = None,
        export_format: str = "csv",
        include_reverse: bool = False,
        delimiter: str = "\t",
        include_header: bool = False,
        extended_header: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_export_action_id(normalized_mode, "export"))
        service = self._service_for_mode(normalized_mode)
        legacy_record_response = format is not None
        requested_format = format or export_format
        payload = await self._maybe_await(
            service.export_flashcards(
                deck_id=deck_id,
                workspace_id=workspace_id,
                include_workspace_items=include_workspace_items,
                tag=tag,
                q=q,
                export_format=requested_format,
                include_reverse=include_reverse,
                delimiter=delimiter,
                include_header=include_header,
                extended_header=extended_header,
            )
        )
        if legacy_record_response:
            if isinstance(payload, Mapping):
                record = dict(payload)
            else:
                record = {
                    "content": payload,
                    "filename": f"flashcards.{requested_format}",
                }
            return self._normalize_export_payload(backend=normalized_mode, payload=record)
        return payload

    async def upload_flashcard_asset(
        self,
        *,
        mode: StudyBackend | str | None = None,
        file: Any = None,
        file_path: Any = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_asset_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        upload_payload = file if file is not None else file_path
        if upload_payload is None:
            raise ValueError("file or file_path is required for flashcard asset upload.")
        payload = await self._maybe_await(service.upload_flashcard_asset(upload_payload))
        return self._normalize_asset_payload(backend=normalized_mode, payload=payload)

    async def get_flashcard_asset_content(
        self,
        *,
        mode: StudyBackend | str | None = None,
        asset_uuid: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_asset_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.get_flashcard_asset_content(asset_uuid))
        return self._normalize_asset_content_payload(
            backend=normalized_mode,
            asset_uuid=asset_uuid,
            payload=payload,
        )

    async def import_flashcards_json_file(
        self,
        *,
        mode: StudyBackend | str | None = None,
        file_path: Any,
        max_items: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.import_flashcards_json_file(
                file_path,
                max_items=max_items,
                max_field_length=max_field_length,
            )
        )
        result = dict(payload or {})
        result.setdefault("source", normalized_mode.value)
        result.setdefault("entity_kind", "flashcard_import")
        return result

    async def import_flashcards_apkg(
        self,
        *,
        mode: StudyBackend | str | None = None,
        file: Any = None,
        file_path: Any = None,
        max_items: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard import")
        self._enforce_policy(self._study_flashcard_import_action_id(normalized_mode, "import"))
        service = self._service_for_mode(normalized_mode)
        import_payload = file if file is not None else file_path
        if import_payload is None:
            raise ValueError("file or file_path is required for APKG import.")
        payload = await self._maybe_await(
            service.import_flashcards_apkg(
                import_payload,
                max_items=max_items,
                max_field_length=max_field_length,
            )
        )
        return self._normalize_import_payload(
            backend=normalized_mode,
            payload=payload,
            import_kind="apkg",
        )

    async def get_flashcard_study_assistant_context(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard study assistant")
        payload = await self._maybe_await(service.get_flashcard_study_assistant_context(card_id))
        result = dict(self._with_server_source(payload or {}))
        result.setdefault("entity_kind", "flashcard_study_assistant_context")
        return result

    async def respond_flashcard_study_assistant(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        action: str,
        message: str | None = None,
        input_modality: str = "text",
        provider: str | None = None,
        model: str | None = None,
        expected_thread_version: int | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard study assistant")
        payload = await self._maybe_await(
            service.respond_flashcard_study_assistant(
                card_id,
                action=action,
                message=message,
                input_modality=input_modality,
                provider=provider,
                model=model,
                expected_thread_version=expected_thread_version,
            )
        )
        result = dict(self._with_server_source(payload or {}))
        result.setdefault("entity_kind", "flashcard_study_assistant_response")
        return result

    async def get_flashcard_analytics_summary(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: int | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Flashcard analytics")
        self._enforce_policy(self._study_flashcard_analytics_action_id(normalized_mode, "observe"))
        payload = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_flashcard_analytics_summary(
                deck_id=deck_id,
                workspace_id=workspace_id,
                include_workspace_items=include_workspace_items,
            )
        )
        return self._normalize_analytics_payload(
            backend=normalized_mode,
            payload=payload,
            deck_id=deck_id,
            workspace_id=workspace_id,
        )

    async def create_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        name: str,
        model_type: str = "basic",
        front_template: str,
        back_template: str | None = None,
        notes_template: str | None = None,
        extra_template: str | None = None,
        placeholder_definitions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.create_flashcard_template(
                name=name,
                model_type=model_type,
                front_template=front_template,
                back_template=back_template,
                notes_template=notes_template,
                extra_template=extra_template,
                placeholder_definitions=placeholder_definitions,
            )
        )
        return self._normalize_template_payload(backend=normalized_mode, payload=payload)

    async def list_flashcard_templates(
        self,
        *,
        mode: StudyBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.list_flashcard_templates(limit=limit, offset=offset))
        return self._normalize_template_list_payload(
            backend=normalized_mode,
            payload=payload,
            limit=limit,
            offset=offset,
        )

    async def get_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        template_id: str | int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.get_flashcard_template(template_id))
        return self._normalize_template_payload(backend=normalized_mode, payload=payload)

    async def update_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        template_id: str | int,
        name: str | None = None,
        model_type: str | None = None,
        front_template: str | None = None,
        back_template: str | None = None,
        notes_template: str | None = None,
        extra_template: str | None = None,
        placeholder_definitions: list[dict[str, Any]] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.update_flashcard_template(
                template_id,
                name=name,
                model_type=model_type,
                front_template=front_template,
                back_template=back_template,
                notes_template=notes_template,
                extra_template=extra_template,
                placeholder_definitions=placeholder_definitions,
                expected_version=expected_version,
            )
        )
        return self._normalize_template_payload(backend=normalized_mode, payload=payload)

    async def delete_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        template_id: str | int,
        expected_version: int,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._study_flashcard_template_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            service.delete_flashcard_template(template_id, expected_version=expected_version)
        )
        return self._coerce_delete_result(result)

    async def generate_study_document(
        self,
        *,
        mode: StudyBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study document generation")
        self._enforce_policy("study.guides.launch.server")
        result = await self._maybe_await(service.generate_study_document(dict(payload)))
        return dict(self._with_server_source(result or {}))

    async def get_study_document_job_status(
        self,
        *,
        mode: StudyBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study document jobs")
        self._enforce_policy("study.guides.observe.server")
        result = await self._maybe_await(service.get_study_document_job_status(job_id))
        return dict(self._with_server_source(result or {}))

    async def cancel_study_document_job(
        self,
        *,
        mode: StudyBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study document jobs")
        self._enforce_policy("study.guides.launch.server")
        result = await self._maybe_await(service.cancel_study_document_job(job_id))
        return dict(self._with_server_source(result or {}))

    async def list_study_documents(
        self,
        *,
        mode: StudyBackend | str | None = None,
        conversation_id: str | None = None,
        document_type: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study documents")
        self._enforce_policy("study.guides.observe.server")
        result = await self._maybe_await(
            service.list_study_documents(
                conversation_id=conversation_id,
                document_type=document_type,
                limit=limit,
            )
        )
        return dict(self._with_server_source(result or {}))

    async def get_study_document(
        self,
        *,
        mode: StudyBackend | str | None = None,
        document_id: int | str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study documents")
        self._enforce_policy("study.guides.observe.server")
        result = await self._maybe_await(service.get_study_document(document_id))
        return dict(self._with_server_source(result or {}))

    async def delete_study_document(
        self,
        *,
        mode: StudyBackend | str | None = None,
        document_id: int | str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study documents")
        self._enforce_policy("study.guides.launch.server")
        result = await self._maybe_await(service.delete_study_document(document_id))
        return dict(self._with_server_source(result or {}))

    async def save_study_document_prompt_config(
        self,
        *,
        mode: StudyBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study document prompts")
        self._enforce_policy("study.guides.launch.server")
        result = await self._maybe_await(service.save_study_document_prompt_config(dict(payload)))
        return dict(self._with_server_source(result or {}))

    async def get_study_document_prompt_config(
        self,
        *,
        mode: StudyBackend | str | None = None,
        document_type: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study document prompts")
        self._enforce_policy("study.guides.observe.server")
        result = await self._maybe_await(service.get_study_document_prompt_config(document_type))
        return dict(self._with_server_source(result or {}))

    async def bulk_generate_study_documents(
        self,
        *,
        mode: StudyBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study document generation")
        self._enforce_policy("study.guides.launch.server")
        result = await self._maybe_await(service.bulk_generate_study_documents(dict(payload)))
        return dict(self._with_server_source(result or {}))

    async def get_study_document_statistics(
        self,
        *,
        mode: StudyBackend | str | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study document statistics")
        self._enforce_policy("study.guides.observe.server")
        result = await self._maybe_await(service.get_study_document_statistics())
        return dict(self._with_server_source(result or {}))

    async def delete_deck(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int,
        expected_version: int | None = None,
        hard_delete: bool = False,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._deck_action_id(normalized_mode, "delete"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).delete_deck(
                deck_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
        )
        return self._coerce_delete_result(result)

    async def get_next_review_candidate(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=False))
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if normalized_mode == StudyBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        payload = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_next_review_candidate(deck_id=deck_id)
        )
        payload = dict(payload or {})
        return normalize_study_review_candidate(
            normalized_mode.value,
            card=payload.get("card"),
            selection_reason=payload.get("selection_reason"),
            review_session=payload.get("review_session"),
        )

    async def submit_flashcard_review(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        rating: int,
        current_card: Mapping[str, Any] | None = None,
        answer_time_ms: int | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if normalized_mode == StudyBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        service = self._service_for_mode(normalized_mode)
        kwargs = {"rating": rating}
        if normalized_mode == StudyBackend.SERVER and answer_time_ms is not None:
            kwargs["answer_time_ms"] = answer_time_ms
        response = await self._maybe_await(service.submit_flashcard_review(card_id, **kwargs))
        return merge_review_outcome_record(
            normalized_mode.value,
            current_card=current_card,
            review_response=response,
            rating=rating,
        )

    async def end_review_session(
        self,
        *,
        mode: StudyBackend | str | None = None,
        review_session_id: int | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if normalized_mode == StudyBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        if normalized_mode == StudyBackend.LOCAL or review_session_id is None:
            return None
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "end_review_session"):
            return None
        return await self._maybe_await(service.end_review_session(review_session_id))

    async def create_study_pack_job(
        self,
        *,
        mode: StudyBackend | str | None = None,
        title: str,
        source_items: list[Mapping[str, Any]],
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Study packs")
        self._enforce_policy(self._study_pack_action_id("launch"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_study_pack_job(
                title=title,
                workspace_id=workspace_id,
                source_items=source_items,
            )
        )
        return self._normalize_study_pack_payload(result)

    async def get_study_pack_job_status(
        self,
        *,
        mode: StudyBackend | str | None = None,
        job_id: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Study packs")
        self._enforce_policy(self._study_pack_action_id("observe"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_study_pack_job_status(job_id)
        )
        return self._normalize_study_pack_payload(result)

    async def list_study_pack_jobs(
        self,
        *,
        mode: StudyBackend | str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Study packs")
        self._enforce_policy(self._study_pack_action_id("list"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_study_pack_jobs(
                status=status,
                limit=limit,
            )
        )
        return self._normalize_study_pack_payload(result)

    async def get_study_pack(
        self,
        *,
        mode: StudyBackend | str | None = None,
        pack_id: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Study packs")
        self._enforce_policy(self._study_pack_action_id("observe"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_study_pack(pack_id)
        )
        return self._normalize_study_pack_payload(result)

    async def regenerate_study_pack(
        self,
        *,
        mode: StudyBackend | str | None = None,
        pack_id: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Study packs")
        self._enforce_policy(self._study_pack_action_id("launch"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).regenerate_study_pack(pack_id)
        )
        return self._normalize_study_pack_payload(result)

    async def get_study_suggestion_status(
        self,
        *,
        mode: StudyBackend | str | None = None,
        anchor_type: str,
        anchor_id: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Study suggestions")
        self._enforce_policy(self._study_suggestion_action_id("list"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_study_suggestion_status(
                anchor_type=anchor_type,
                anchor_id=anchor_id,
            )
        )
        return self._normalize_study_suggestion_payload(result)

    async def get_study_suggestion_snapshot(
        self,
        *,
        mode: StudyBackend | str | None = None,
        snapshot_id: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Study suggestions")
        self._enforce_policy(self._study_suggestion_action_id("observe"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_study_suggestion_snapshot(snapshot_id)
        )
        return self._normalize_study_suggestion_payload(result)

    async def refresh_study_suggestion_snapshot(
        self,
        *,
        mode: StudyBackend | str | None = None,
        snapshot_id: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Study suggestions")
        self._enforce_policy(self._study_suggestion_action_id("launch"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).refresh_study_suggestion_snapshot(
                snapshot_id,
                reason=reason,
            )
        )
        return self._normalize_study_suggestion_payload(result)

    async def trigger_study_suggestion_action(
        self,
        *,
        mode: StudyBackend | str | None = None,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        action_kind: str,
        selected_topic_ids: list[str] | None = None,
        selected_topic_edits: list[dict[str, str]] | None = None,
        manual_topic_labels: list[str] | None = None,
        has_explicit_selection: bool = False,
        generator_version: str = "v1",
        force_regenerate: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_only(normalized_mode, "Study suggestions")
        self._enforce_policy(self._study_suggestion_action_id("configure"))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).trigger_study_suggestion_action(
                snapshot_id,
                target_service=target_service,
                target_type=target_type,
                action_kind=action_kind,
                selected_topic_ids=selected_topic_ids or [],
                selected_topic_edits=selected_topic_edits or [],
                manual_topic_labels=manual_topic_labels or [],
                has_explicit_selection=has_explicit_selection,
                generator_version=generator_version,
                force_regenerate=force_regenerate,
            )
        )
        return self._normalize_study_suggestion_payload(result)
