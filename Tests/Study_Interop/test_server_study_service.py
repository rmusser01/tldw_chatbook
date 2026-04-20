import pytest

from tldw_chatbook.Study_Interop.server_study_service import ServerStudyService
from tldw_chatbook.tldw_api.flashcards_schemas import FlashcardResponse


CARD_UUID = "00000000-0000-4000-8000-000000000001"


class FakeClient:
    def __init__(self):
        self.calls = []

    async def update_flashcard(self, card_uuid, request_data):
        self.calls.append(("update_flashcard", card_uuid, request_data.model_dump(mode="json", exclude_none=True)))
        return FlashcardResponse.model_validate(
            {
                "uuid": card_uuid,
                "deck_id": 9,
                "front": "Question",
                "back": "Answer",
                "tags": ["science"],
                "ef": 2.5,
                "interval_days": 0,
                "repetitions": 0,
                "lapses": 0,
                "queue_state": "new",
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
                "deleted": False,
                "client_id": "server-client",
                "version": 3,
                "model_type": "basic",
                "reverse": False,
            }
        )

    async def delete_flashcard(self, card_uuid, *, expected_version):
        self.calls.append(("delete_flashcard", card_uuid, expected_version))
        return {"deleted": True}


@pytest.mark.asyncio
async def test_server_study_service_moves_flashcards_via_update_flashcard():
    client = FakeClient()
    service = ServerStudyService(client=client)

    moved = await service.move_flashcard(CARD_UUID, target_deck_id=9, expected_version=2)

    assert moved["uuid"] == CARD_UUID
    assert moved["deck_id"] == 9
    assert client.calls == [
        ("update_flashcard", CARD_UUID, {"deck_id": 9, "expected_version": 2}),
    ]


@pytest.mark.asyncio
async def test_server_study_service_deletes_flashcards_with_expected_version():
    client = FakeClient()
    service = ServerStudyService(client=client)

    deleted = await service.delete_flashcard("card-server-1", expected_version=2)

    assert deleted == {"deleted": True}
    assert client.calls == [("delete_flashcard", "card-server-1", 2)]


@pytest.mark.asyncio
async def test_server_deck_delete_is_explicitly_unsupported():
    server = ServerStudyService(client=FakeClient())

    with pytest.raises(
        NotImplementedError,
        match="Flashcard deck deletion is not supported by the current server API\\.",
    ):
        await server.delete_deck(deck_id=7, expected_version=2)
