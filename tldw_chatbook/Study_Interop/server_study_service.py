"""Thin server-backed study service around the shared flashcards API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    BulkGenerateRequest,
    DocumentType,
    FlashcardBulkUpdateItemRequest,
    FlashcardCreateRequest,
    FlashcardDeckCreateRequest,
    FlashcardDeckUpdateRequest,
    FlashcardResetSchedulingRequest,
    FlashcardReviewRequest,
    FlashcardTagsUpdateRequest,
    FlashcardTemplateCreateRequest,
    FlashcardTemplateUpdateRequest,
    FlashcardUpdateRequest,
    FlashcardsImportRequest,
    GenerateDocumentRequest,
    SavePromptConfigRequest,
    StudyAssistantRespondRequest,
    StructuredQaImportPreviewRequest,
    StudyPackCreateJobRequest,
    StudyPackSourceSelection,
    SuggestionActionRequest,
    SuggestionRefreshRequest,
    TLDWAPIClient,
)


class ServerStudyService:
    """Thin wrapper around server-backed flashcard deck/card/review endpoints."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerStudyService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server study operations.")
        return self.client

    @staticmethod
    def _model_to_dict(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    @staticmethod
    def _coerce_deck_id(deck_id: Any) -> Optional[int]:
        if deck_id in {None, ""}:
            return None
        return int(deck_id)

    async def list_decks(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        response = await self._require_client().list_flashcard_decks(limit=limit, offset=offset)
        return [self._model_to_dict(item) for item in list(response or [])]

    async def create_deck(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        scheduler_type: Optional[str] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().create_flashcard_deck(
            FlashcardDeckCreateRequest(
                name=name,
                description=description,
                workspace_id=workspace_id,
                scheduler_type=scheduler_type,
            )
        )
        return self._model_to_dict(response)

    async def update_deck(
        self,
        deck_id: int,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        review_prompt_side: Optional[str] = None,
        scheduler_type: Optional[str] = None,
        scheduler_settings: Optional[dict[str, Any]] = None,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in {
                "name": name,
                "description": description,
                "workspace_id": workspace_id,
                "review_prompt_side": review_prompt_side,
                "scheduler_type": scheduler_type,
                "scheduler_settings": scheduler_settings,
                "expected_version": expected_version,
            }.items()
            if value is not None
        }
        response = await self._require_client().update_flashcard_deck(
            int(deck_id),
            FlashcardDeckUpdateRequest(**payload),
        )
        return self._model_to_dict(response)

    async def list_flashcards(
        self,
        *,
        deck_id: Optional[int] = None,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        response = await self._require_client().list_flashcards(
            deck_id=self._coerce_deck_id(deck_id),
            q=q,
            limit=limit,
            offset=offset,
        )
        payload = self._model_to_dict(response)
        return list(payload.get("items", []))

    async def create_flashcard(
        self,
        *,
        deck_id: int,
        front: str,
        back: str,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        extra: Optional[str] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().create_flashcard(
            FlashcardCreateRequest(
                deck_id=self._coerce_deck_id(deck_id),
                front=front,
                back=back,
                tags=tags,
                notes=notes,
                extra=extra,
                model_type="basic",
            )
        )
        return self._model_to_dict(response)

    async def create_flashcards_bulk(self, cards: list[Mapping[str, Any]]) -> dict[str, Any]:
        response = await self._require_client().create_flashcards_bulk(
            [FlashcardCreateRequest.model_validate(dict(card)) for card in cards]
        )
        return self._model_to_dict(response)

    async def get_flashcard(self, card_id: str) -> dict[str, Any]:
        response = await self._require_client().get_flashcard(card_id)
        return self._model_to_dict(response)

    async def move_flashcard(
        self,
        card_id: str,
        *,
        target_deck_id: int,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        return self._model_to_dict(
            await self._require_client().update_flashcard(
                card_id,
                FlashcardUpdateRequest(
                    deck_id=self._coerce_deck_id(target_deck_id),
                    expected_version=expected_version,
                ),
            )
        )

    async def delete_flashcard(
        self,
        card_id: str,
        *,
        expected_version: int,
    ) -> dict[str, Any]:
        if expected_version is None:
            raise ValueError("expected_version is required for server flashcard deletion.")
        if expected_version < 1:
            raise ValueError("expected_version must be >= 1 for server flashcard deletion.")
        return await self._require_client().delete_flashcard(
            card_id,
            expected_version=expected_version,
        )

    async def update_flashcards_bulk(self, cards: list[Mapping[str, Any]]) -> dict[str, Any]:
        response = await self._require_client().update_flashcards_bulk(
            [FlashcardBulkUpdateItemRequest.model_validate(dict(card)) for card in cards]
        )
        return self._model_to_dict(response)

    async def reset_flashcard_scheduling(
        self,
        card_id: str,
        *,
        expected_version: int,
    ) -> dict[str, Any]:
        response = await self._require_client().reset_flashcard_scheduling(
            card_id,
            FlashcardResetSchedulingRequest(expected_version=expected_version),
        )
        return self._model_to_dict(response)

    async def set_flashcard_tags(
        self,
        card_id: str,
        *,
        tags: list[str],
    ) -> dict[str, Any]:
        response = await self._require_client().set_flashcard_tags(
            card_id,
            FlashcardTagsUpdateRequest(tags=tags),
        )
        return self._model_to_dict(response)

    async def get_flashcard_tags(self, card_id: str) -> dict[str, Any]:
        return self._model_to_dict(await self._require_client().get_flashcard_tags(card_id))

    async def list_flashcard_tag_suggestions(
        self,
        *,
        q: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return self._model_to_dict(
            await self._require_client().list_flashcard_tag_suggestions(q=q, limit=limit)
        )

    async def preview_structured_qa_import(
        self,
        content: str,
        *,
        max_lines: Optional[int] = None,
        max_line_length: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().preview_structured_qa_import(
            StructuredQaImportPreviewRequest(content=content),
            max_lines=max_lines,
            max_line_length=max_line_length,
            max_field_length=max_field_length,
        )
        return self._model_to_dict(response)

    async def import_flashcards_tsv(
        self,
        content: str,
        *,
        delimiter: str = "\t",
        has_header: bool = False,
        max_lines: Optional[int] = None,
        max_line_length: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().import_flashcards_tsv(
            FlashcardsImportRequest(
                content=content,
                delimiter=delimiter,
                has_header=has_header,
            ),
            max_lines=max_lines,
            max_line_length=max_line_length,
            max_field_length=max_field_length,
        )
        return self._model_to_dict(response)

    async def export_flashcards(
        self,
        *,
        deck_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        include_workspace_items: bool = False,
        tag: Optional[str] = None,
        q: Optional[str] = None,
        export_format: str = "csv",
        include_reverse: bool = False,
        delimiter: str = "\t",
        include_header: bool = False,
        extended_header: bool = False,
    ) -> bytes:
        return await self._require_client().export_flashcards(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
            tag=tag,
            q=q,
            export_format=export_format,
            include_reverse=include_reverse,
            delimiter=delimiter,
            include_header=include_header,
            extended_header=extended_header,
        )

    async def upload_flashcard_asset(self, file_path: Any) -> dict[str, Any]:
        response = await self._require_client().upload_flashcard_asset(file_path)
        return self._model_to_dict(response)

    async def get_flashcard_asset_content(self, asset_uuid: str) -> bytes:
        return await self._require_client().get_flashcard_asset_content(asset_uuid)

    async def import_flashcards_json_file(
        self,
        file_path: Any,
        *,
        max_items: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().import_flashcards_json_file(
            file_path,
            max_items=max_items,
            max_field_length=max_field_length,
        )
        return self._model_to_dict(response)

    async def import_flashcards_apkg(
        self,
        file_path: Any,
        *,
        max_items: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().import_flashcards_apkg(
            file_path,
            max_items=max_items,
            max_field_length=max_field_length,
        )
        return self._model_to_dict(response)

    async def get_flashcard_study_assistant_context(self, card_id: str) -> dict[str, Any]:
        response = await self._require_client().get_flashcard_study_assistant_context(card_id)
        return self._model_to_dict(response)

    async def respond_flashcard_study_assistant(
        self,
        card_id: str,
        *,
        action: str,
        message: Optional[str] = None,
        input_modality: str = "text",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        expected_thread_version: Optional[int] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().respond_flashcard_study_assistant(
            card_id,
            StudyAssistantRespondRequest(
                action=action,
                message=message,
                input_modality=input_modality,
                provider=provider,
                model=model,
                expected_thread_version=expected_thread_version,
            ),
        )
        return self._model_to_dict(response)

    async def get_flashcard_analytics_summary(
        self,
        *,
        deck_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        include_workspace_items: bool = False,
    ) -> dict[str, Any]:
        response = await self._require_client().get_flashcard_analytics_summary(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
        )
        payload = self._model_to_dict(response)
        payload["source"] = "server"
        return payload

    async def create_flashcard_template(
        self,
        *,
        name: str,
        model_type: str = "basic",
        front_template: str,
        back_template: Optional[str] = None,
        notes_template: Optional[str] = None,
        extra_template: Optional[str] = None,
        placeholder_definitions: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().create_flashcard_template(
            FlashcardTemplateCreateRequest(
                name=name,
                model_type=model_type,
                front_template=front_template,
                back_template=back_template,
                notes_template=notes_template,
                extra_template=extra_template,
                placeholder_definitions=placeholder_definitions or [],
            )
        )
        return self._model_to_dict(response)

    async def list_flashcard_templates(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        return self._model_to_dict(
            await self._require_client().list_flashcard_templates(limit=limit, offset=offset)
        )

    async def get_flashcard_template(self, template_id: int) -> dict[str, Any]:
        return self._model_to_dict(
            await self._require_client().get_flashcard_template(int(template_id))
        )

    async def update_flashcard_template(
        self,
        template_id: int,
        *,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        front_template: Optional[str] = None,
        back_template: Optional[str] = None,
        notes_template: Optional[str] = None,
        extra_template: Optional[str] = None,
        placeholder_definitions: Optional[list[dict[str, Any]]] = None,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in {
                "name": name,
                "model_type": model_type,
                "front_template": front_template,
                "back_template": back_template,
                "notes_template": notes_template,
                "extra_template": extra_template,
                "placeholder_definitions": placeholder_definitions,
                "expected_version": expected_version,
            }.items()
            if value is not None
        }
        response = await self._require_client().update_flashcard_template(
            int(template_id),
            FlashcardTemplateUpdateRequest(**payload),
        )
        return self._model_to_dict(response)

    async def delete_flashcard_template(self, template_id: int, *, expected_version: int) -> dict[str, Any]:
        return self._model_to_dict(
            await self._require_client().delete_flashcard_template(
                int(template_id),
                expected_version=expected_version,
            )
        )

    async def delete_deck(
        self,
        deck_id: int,
        *,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> Any:
        raise NotImplementedError(
            "Flashcard deck deletion is not supported by the current server API."
        )

    async def get_next_review_candidate(self, *, deck_id: Optional[int] = None) -> dict[str, Any]:
        if deck_id is not None:
            response = await self._require_client().get_next_flashcard_review(deck_id=self._coerce_deck_id(deck_id))
        else:
            response = await self._require_client().get_next_flashcard_review(deck_id=None)
        payload = self._model_to_dict(response)
        return {
            "card": payload.get("card"),
            "selection_reason": payload.get("selection_reason"),
        }

    async def submit_flashcard_review(
        self,
        card_id: str,
        *,
        rating: int,
        answer_time_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().review_flashcard(
            FlashcardReviewRequest(card_uuid=card_id, rating=rating, answer_time_ms=answer_time_ms)
        )
        return self._model_to_dict(response)

    async def end_review_session(self, review_session_id: int) -> dict[str, Any]:
        response = await self._require_client().end_flashcard_review_session(review_session_id)
        return self._model_to_dict(response)

    async def create_study_pack_job(
        self,
        *,
        title: str,
        source_items: list[Mapping[str, Any]],
        workspace_id: Optional[str] = None,
        deck_mode: str = "new",
    ) -> dict[str, Any]:
        response = await self._require_client().create_study_pack_job(
            StudyPackCreateJobRequest(
                title=title,
                workspace_id=workspace_id,
                deck_mode=deck_mode,
                source_items=[
                    StudyPackSourceSelection.model_validate(dict(item))
                    for item in source_items
                ],
            )
        )
        return self._model_to_dict(response)

    async def get_study_pack_job_status(self, job_id: int) -> dict[str, Any]:
        response = await self._require_client().get_study_pack_job_status(int(job_id))
        return self._model_to_dict(response)

    async def get_study_pack(self, pack_id: int) -> dict[str, Any]:
        payload = self._model_to_dict(await self._require_client().get_study_pack(int(pack_id)))
        payload["source"] = "server"
        payload["record_id"] = f"server:study-pack:{payload.get('id')}"
        return payload

    async def regenerate_study_pack(self, pack_id: int) -> dict[str, Any]:
        response = await self._require_client().regenerate_study_pack(int(pack_id))
        return self._model_to_dict(response)

    async def get_study_suggestion_status(self, *, anchor_type: str, anchor_id: int) -> dict[str, Any]:
        response = await self._require_client().get_study_suggestion_status(anchor_type, int(anchor_id))
        payload = self._model_to_dict(response)
        payload["source"] = "server"
        return payload

    async def get_study_suggestion_snapshot(self, snapshot_id: int) -> dict[str, Any]:
        payload = self._model_to_dict(await self._require_client().get_study_suggestion_snapshot(int(snapshot_id)))
        payload["source"] = "server"
        return payload

    async def refresh_study_suggestion_snapshot(
        self,
        snapshot_id: int,
        *,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        response = await self._require_client().refresh_study_suggestion_snapshot(
            int(snapshot_id),
            SuggestionRefreshRequest(reason=reason),
        )
        return self._model_to_dict(response)

    async def trigger_study_suggestion_action(
        self,
        snapshot_id: int,
        *,
        target_service: str,
        target_type: str,
        action_kind: str,
        selected_topic_ids: Optional[list[str]] = None,
        selected_topic_edits: Optional[list[dict[str, str]]] = None,
        manual_topic_labels: Optional[list[str]] = None,
        has_explicit_selection: bool = False,
        generator_version: str = "v1",
        force_regenerate: bool = False,
    ) -> dict[str, Any]:
        response = await self._require_client().trigger_study_suggestion_action(
            int(snapshot_id),
            SuggestionActionRequest(
                target_service=target_service,
                target_type=target_type,
                action_kind=action_kind,
                selected_topic_ids=selected_topic_ids or [],
                selected_topic_edits=selected_topic_edits or [],
                manual_topic_labels=manual_topic_labels or [],
                has_explicit_selection=has_explicit_selection,
                generator_version=generator_version,
                force_regenerate=force_regenerate,
            ),
        )
        return self._model_to_dict(response)

    async def generate_study_document(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().generate_chat_document(
            GenerateDocumentRequest.model_validate(dict(payload))
        )
        return self._model_to_dict(response)

    async def get_study_document_job_status(self, job_id: str) -> dict[str, Any]:
        response = await self._require_client().get_chat_document_job_status(str(job_id))
        return self._model_to_dict(response)

    async def cancel_study_document_job(self, job_id: str) -> dict[str, Any]:
        return await self._require_client().cancel_chat_document_job(str(job_id))

    async def list_study_documents(
        self,
        *,
        conversation_id: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        response = await self._require_client().list_chat_generated_documents(
            conversation_id=conversation_id,
            document_type=DocumentType(document_type) if document_type is not None else None,
            limit=limit,
        )
        return self._model_to_dict(response)

    async def get_study_document(self, document_id: int) -> dict[str, Any]:
        response = await self._require_client().get_chat_generated_document(int(document_id))
        return self._model_to_dict(response)

    async def delete_study_document(self, document_id: int) -> dict[str, Any]:
        return await self._require_client().delete_chat_generated_document(int(document_id))

    async def save_study_document_prompt_config(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().save_chat_document_prompt_config(
            SavePromptConfigRequest.model_validate(dict(payload))
        )
        return self._model_to_dict(response)

    async def get_study_document_prompt_config(self, document_type: str) -> dict[str, Any]:
        response = await self._require_client().get_chat_document_prompt_config(DocumentType(document_type))
        return self._model_to_dict(response)

    async def bulk_generate_study_documents(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().bulk_generate_chat_documents(
            BulkGenerateRequest.model_validate(dict(payload))
        )
        return self._model_to_dict(response)

    async def get_study_document_statistics(self) -> dict[str, Any]:
        response = await self._require_client().get_chat_document_generation_statistics()
        return self._model_to_dict(response)
