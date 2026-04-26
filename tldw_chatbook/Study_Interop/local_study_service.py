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

    def update_deck(
        self,
        deck_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        expected_version: Optional[int] = None,
        **_: Any,
    ) -> Any:
        updated = self._require_db().update_deck(
            deck_id,
            name=name,
            description=description,
            expected_version=expected_version,
        )
        if not updated:
            return None
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

    def get_flashcard(self, card_id: str) -> Any:
        return self._require_db().get_flashcard(card_id)

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, str):
            raw_tags = tags.split()
        elif isinstance(tags, (list, tuple, set)):
            raw_tags = list(tags)
        else:
            raw_tags = [tags]
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_tags:
            tag = str(item or "").strip()
            if tag and tag not in seen:
                seen.add(tag)
                normalized.append(tag)
        return normalized

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

    def update_flashcard(
        self,
        card_id: str,
        *,
        deck_id: Optional[str] = None,
        front: Optional[str] = None,
        back: Optional[str] = None,
        tags: Any = None,
        notes: Optional[str] = None,
        extra: Optional[str] = None,
        expected_version: Optional[int] = None,
        card_type: Optional[str] = None,
        **_: Any,
    ) -> Any:
        request = {
            key: value
            for key, value in {
                "deck_id": deck_id,
                "front": front,
                "back": back,
                "tags": tags,
                "notes": notes,
                "extra": extra,
                "expected_version": expected_version,
                "card_type": card_type,
            }.items()
            if value is not None
        }
        updated = self._require_db().update_flashcard(card_id, **request)
        if not updated:
            return None
        return self._require_db().get_flashcard(card_id)

    def get_flashcard_tags(self, card_id: str) -> dict[str, Any]:
        card = self.get_flashcard(card_id)
        return {"uuid": card_id, "tags": self._normalize_tags((card or {}).get("tags"))}

    def set_flashcard_tags(self, card_id: str, *, tags: list[str]) -> Any:
        return self.update_flashcard(card_id, tags=tags)

    def create_flashcards_bulk(self, cards: list[Mapping[str, Any]]) -> dict[str, Any]:
        created: list[Any] = []
        for card in cards:
            created.append(
                self.create_flashcard(
                    deck_id=card["deck_id"],
                    front=str(card.get("front") or ""),
                    back=str(card.get("back") or ""),
                    tags=self._normalize_tags(card.get("tags")),
                    notes=card.get("notes"),
                    extra=card.get("extra"),
                )
            )
        return {"items": created, "count": len(created), "total": len(created)}

    def update_flashcards_bulk(self, updates: list[Mapping[str, Any]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for update in updates:
            card_id = update.get("uuid") or update.get("id") or update.get("card_id")
            if not card_id:
                results.append({"uuid": None, "status": "error", "error": "missing_card_id"})
                continue
            card_id = str(card_id)
            card = self.update_flashcard(
                card_id,
                deck_id=update.get("deck_id"),
                front=update.get("front"),
                back=update.get("back"),
                tags=update.get("tags"),
                notes=update.get("notes"),
                extra=update.get("extra"),
                expected_version=update.get("expected_version"),
                card_type=update.get("card_type") or update.get("type") or update.get("model_type"),
            )
            status = "updated" if card else "not_found"
            results.append({"uuid": card_id, "status": status, "flashcard": card})
        return {"results": results}

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
