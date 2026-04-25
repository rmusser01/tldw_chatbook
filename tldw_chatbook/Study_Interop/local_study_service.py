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

    @staticmethod
    def _unsupported_study_packs() -> None:
        raise ValueError("Study packs are server-only.")

    @staticmethod
    def _unsupported_study_suggestions() -> None:
        raise ValueError("Study suggestions are server-only.")

    def create_study_pack_job(self, **kwargs: Any) -> None:
        self._unsupported_study_packs()

    def get_study_pack_job_status(self, job_id: int) -> None:
        self._unsupported_study_packs()

    def get_study_pack(self, pack_id: int) -> None:
        self._unsupported_study_packs()

    def regenerate_study_pack(self, pack_id: int) -> None:
        self._unsupported_study_packs()

    def get_study_suggestion_status(self, **kwargs: Any) -> None:
        self._unsupported_study_suggestions()

    def get_study_suggestion_snapshot(self, snapshot_id: int) -> None:
        self._unsupported_study_suggestions()

    def refresh_study_suggestion_snapshot(self, snapshot_id: int, **kwargs: Any) -> None:
        self._unsupported_study_suggestions()

    def trigger_study_suggestion_action(self, snapshot_id: int, **kwargs: Any) -> None:
        self._unsupported_study_suggestions()
