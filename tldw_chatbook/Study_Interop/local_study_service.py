"""Thin local study wrapper around ChaChaNotes_DB flashcard helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional


class LocalStudyService:
    """Thin sync wrapper around local study helpers."""

    def __init__(self, db: Any):
        self.db = db

    def _require_db(self) -> Any:
        if self.db is None:
            raise ValueError("Local study backend is unavailable.")
        return self.db

    def list_decks(self, *, limit: int = 100, offset: int = 0) -> Any:
        return self._require_db().list_decks(limit=limit, offset=offset)

    def get_deck(self, deck_id: str) -> Any:
        return self._require_db().get_deck(deck_id)

    def create_deck(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        scheduler_type: Optional[str] = None,
    ) -> Any:
        deck_id = self._require_db().create_deck(name, description)
        return self._require_db().get_deck(deck_id)

    def update_deck(
        self,
        deck_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        review_prompt_side: Optional[str] = None,
        scheduler_type: Optional[str] = None,
        scheduler_settings: Optional[dict[str, Any]] = None,
        expected_version: Optional[int] = None,
    ) -> Any:
        metadata = {
            key: value
            for key, value in {
                "review_prompt_side": review_prompt_side,
                "scheduler_type": scheduler_type,
                "scheduler_settings": scheduler_settings,
            }.items()
            if value is not None
        }
        self._require_db().update_deck(
            deck_id,
            name=name,
            description=description,
            metadata=metadata or None,
            expected_version=expected_version,
        )
        return self._require_db().get_deck(deck_id)

    def list_flashcards(
        self,
        *,
        deck_id: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        normalized_q = str(q or "").strip() or None
        return self._require_db().list_flashcards(deck_id=deck_id, q=normalized_q, limit=limit, offset=offset)

    def create_flashcard(
        self,
        *,
        deck_id: str,
        front: str,
        back: str,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        extra: Optional[str] = None,
    ) -> Any:
        metadata = {
            key: value
            for key, value in {"notes": notes, "extra": extra}.items()
            if value not in {None, ""}
        }
        card_id = self._require_db().create_flashcard(
            {
                "deck_id": deck_id,
                "front": front,
                "back": back,
                "tags": " ".join(tags or []),
                "type": "basic",
                "metadata": metadata or None,
            }
        )
        return self._require_db().get_flashcard(card_id)

    def create_flashcards_bulk(self, cards: list[Mapping[str, Any]]) -> dict[str, Any]:
        created_cards = [self.create_flashcard(**dict(card)) for card in cards]
        return {"items": created_cards, "count": len(created_cards)}

    def get_flashcard(self, card_id: str) -> Any:
        return self._require_db().get_flashcard(card_id)

    def update_flashcard(
        self,
        card_id: str,
        *,
        deck_id: Optional[str] = None,
        front: Optional[str] = None,
        back: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        extra: Optional[str] = None,
        expected_version: Optional[int] = None,
        **extra_fields: Any,
    ) -> Any:
        metadata = {
            key: value
            for key, value in {"notes": notes, "extra": extra}.items()
            if value is not None
        }
        self._require_db().update_flashcard(
            card_id,
            deck_id=deck_id,
            front=front,
            back=back,
            tags=tags,
            metadata=metadata or None,
            expected_version=expected_version,
            **extra_fields,
        )
        return self._require_db().get_flashcard(card_id)

    def update_flashcards_bulk(self, cards: list[Mapping[str, Any]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for card in cards:
            payload = dict(card)
            card_id = payload.pop("id", None) or payload.pop("uuid", None)
            if not card_id:
                results.append({"status": "error", "error": "Missing flashcard id", "flashcard": None})
                continue
            flashcard = self.update_flashcard(str(card_id), **payload)
            results.append({"status": "updated", "flashcard": flashcard})
        return {"results": results, "count": len(results)}

    def move_flashcard(
        self,
        card_id: str,
        *,
        target_deck_id: str,
        expected_version: Optional[int] = None,
    ) -> Any:
        moved = self._require_db().move_flashcard(
            card_id,
            target_deck_id,
            expected_version=expected_version,
        )
        if not moved:
            return None
        return self._require_db().get_flashcard(card_id)

    def reset_flashcard_scheduling(
        self,
        card_id: str,
        *,
        expected_version: Optional[int] = None,
    ) -> Any:
        self._require_db().reset_flashcard_scheduling(card_id, expected_version=expected_version)
        return self._require_db().get_flashcard(card_id)

    def set_flashcard_tags(self, card_id: str, *, tags: list[str]) -> Any:
        self._require_db().set_flashcard_tags(card_id, tags=tags)
        return self._require_db().get_flashcard(card_id)

    def get_flashcard_tags(self, card_id: str) -> dict[str, Any]:
        payload = self._require_db().get_flashcard_tags(card_id)
        if isinstance(payload, Mapping):
            items = list(payload.get("items") or [])
        else:
            items = list(payload or [])
        return {"items": items, "count": len(items)}

    def list_flashcard_tag_suggestions(
        self,
        *,
        q: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._require_db().list_flashcard_tag_suggestions(q=q, limit=limit)
        if isinstance(payload, Mapping):
            items = list(payload.get("items") or [])
        else:
            items = list(payload or [])
        return {"items": items, "count": len(items)}

    def delete_flashcard(
        self,
        card_id: str,
        *,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        return bool(
            self._require_db().delete_flashcard(
                card_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
        )

    def delete_deck(
        self,
        deck_id: str,
        *,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        return bool(
            self._require_db().delete_deck(
                deck_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
        )

    def get_next_review_candidate(self, *, deck_id: Optional[str] = None) -> dict[str, Any]:
        cards = self._require_db().get_due_flashcards(deck_id=deck_id, limit=1)
        if not cards:
            return {"card": None, "selection_reason": "none"}
        return {"card": cards[0], "selection_reason": "due"}

    def submit_flashcard_review(self, card_id: str, *, rating: int) -> dict[str, Any]:
        self._require_db().update_flashcard_review(card_id, rating)
        return {"card": self._require_db().get_flashcard(card_id), "rating": rating}

    def end_review_session(self, review_session_id: int) -> None:
        return None
