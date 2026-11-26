"""Native Study reads through the shared authenticated client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .flashcards_schemas import FlashcardDeckResponse
from .quizzes_schemas import QuizListResponse, QuizResponse


class StudyReadClientMixin:
    if TYPE_CHECKING:

        async def _request(
            self, method: str, endpoint: str, *, params: dict[str, Any] | None = None
        ) -> Any: ...

    async def list_quizzes(
        self,
        *,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
        workspace_id: str | None = None,
    ) -> list[QuizResponse]:
        """List native quizzes, optionally restricted to one workspace."""
        response = await self._request(
            "GET",
            "/api/v1/quizzes",
            params={
                key: value
                for key, value in {
                    "q": q,
                    "limit": limit,
                    "offset": offset,
                    "workspace_id": workspace_id,
                }.items()
                if value is not None
            },
        )
        return QuizListResponse.model_validate(response).items

    async def get_flashcard_deck(self, deck_id: int) -> FlashcardDeckResponse:
        """Read the exact native deck for the authenticated account."""
        response = await self._request("GET", f"/api/v1/flashcards/decks/{deck_id}")
        return FlashcardDeckResponse.model_validate(response)
