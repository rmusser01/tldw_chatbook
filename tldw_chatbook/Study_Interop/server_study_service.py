"""Thin server-backed study service around the shared flashcards API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    FlashcardCreateRequest,
    FlashcardDeckCreateRequest,
    FlashcardReviewRequest,
    FlashcardUpdateRequest,
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
