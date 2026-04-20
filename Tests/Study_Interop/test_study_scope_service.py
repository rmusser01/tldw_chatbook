import pytest

from tldw_chatbook.Study_Interop.study_scope_service import StudyScopeService


class FakeLocalStudyService:
    def __init__(self):
        self.calls = []

    def list_decks(self, *, limit=100, offset=0):
        self.calls.append(("list_decks", limit, offset))
        return [{"id": "deck-local-1", "name": "Biology", "description": "Cell review"}]

    def create_deck(self, *, name, description=None):
        self.calls.append(("create_deck", name, description))
        return {"id": "deck-local-1", "name": name, "description": description}

    def list_flashcards(self, *, deck_id=None, q=None, limit=100, offset=0):
        self.calls.append(("list_flashcards", deck_id, q, limit, offset))
        return [{"id": "card-local-1", "deck_id": deck_id, "front": "Question", "back": "Answer", "tags": "science"}]

    def create_flashcard(self, *, deck_id, front, back, tags=None, notes=None, extra=None):
        self.calls.append(("create_flashcard", deck_id, front, back, tags, notes, extra))
        return {
            "id": "card-local-1",
            "deck_id": deck_id,
            "front": front,
            "back": back,
            "tags": " ".join(tags or []),
            "notes": notes,
            "extra": extra,
            "type": "basic",
        }

    def get_next_review_candidate(self, *, deck_id=None):
        self.calls.append(("get_next_review_candidate", deck_id))
        return {"card": {"id": "card-local-1", "deck_id": deck_id, "front": "Question", "back": "Answer"}, "selection_reason": "due"}

    def submit_flashcard_review(self, card_id, *, rating):
        self.calls.append(("submit_flashcard_review", card_id, rating))
        return {
            "card": {
                "id": card_id,
                "deck_id": "deck-local-1",
                "front": "Question",
                "back": "Answer",
                "interval": 3,
                "repetitions": 1,
                "ease_factor": 2.6,
            },
            "rating": rating,
        }

    def move_flashcard(self, card_id, *, target_deck_id, expected_version=None):
        self.calls.append(("move_flashcard", card_id, target_deck_id, expected_version))
        return {
            "id": card_id,
            "deck_id": target_deck_id,
            "front": "Question",
            "back": "Answer",
            "tags": "science biology",
            "type": "basic",
            "version": 3,
        }

    def delete_flashcard(self, card_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_flashcard", card_id, expected_version, hard_delete))
        return True

    def delete_deck(self, deck_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_deck", deck_id, expected_version, hard_delete))
        return True

    def end_review_session(self, review_session_id):
        self.calls.append(("end_review_session", review_session_id))
        return None


class FakeServerStudyService:
    def __init__(self):
        self.calls = []

    async def list_decks(self, *, limit=100, offset=0):
        self.calls.append(("list_decks", limit, offset))
        return [{"id": 7, "name": "Biology", "description": "Cell review", "scheduler_type": "fsrs"}]

    async def get_next_review_candidate(self, *, deck_id=None):
        self.calls.append(("get_next_review_candidate", deck_id))
        return {
            "card": {
                "uuid": "card-server-1",
                "deck_id": deck_id,
                "front": "Question",
                "back": "Answer",
                "tags": ["science"],
                "is_cloze": False,
                "ef": 2.5,
                "interval_days": 0,
                "repetitions": 0,
                "lapses": 0,
                "queue_state": "new",
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
                "model_type": "basic",
                "reverse": False,
                "scheduler_type": "fsrs",
                "next_intervals": {"again": "10m", "good": "1d"},
            },
            "selection_reason": "new",
        }

    async def submit_flashcard_review(self, card_id, *, rating, answer_time_ms=None):
        self.calls.append(("submit_flashcard_review", card_id, rating, answer_time_ms))
        return {
            "uuid": card_id,
            "ef": 2.6,
            "interval_days": 3,
            "repetitions": 1,
            "lapses": 0,
            "due_at": "2026-04-23T00:00:00Z",
            "last_reviewed_at": "2026-04-20T00:05:00Z",
            "last_modified": "2026-04-20T00:05:00Z",
            "version": 2,
            "scheduler_type": "fsrs",
            "queue_state": "review",
            "next_intervals": {"again": "10m", "good": "3d"},
            "review_session_id": 41,
        }

    async def move_flashcard(self, card_id, *, target_deck_id, expected_version=None):
        self.calls.append(("move_flashcard", card_id, target_deck_id, expected_version))
        return {
            "uuid": card_id,
            "deck_id": target_deck_id,
            "front": "Question",
            "back": "Answer",
            "tags": ["science"],
            "is_cloze": False,
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "queue_state": "new",
            "created_at": "2026-04-20T00:00:00Z",
            "last_modified": "2026-04-20T00:02:00Z",
            "deleted": False,
            "client_id": "server-client",
            "version": 2,
            "model_type": "basic",
            "reverse": False,
        }

    async def delete_flashcard(self, card_id, *, expected_version=None):
        self.calls.append(("delete_flashcard", card_id, expected_version))
        return {"deleted": True}

    async def delete_deck(self, deck_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_deck", deck_id, expected_version, hard_delete))
        raise NotImplementedError("Flashcard deck deletion is not supported by the current server API.")

    async def end_review_session(self, review_session_id):
        self.calls.append(("end_review_session", review_session_id))
        return {"id": review_session_id, "status": "completed"}


@pytest.mark.asyncio
async def test_scope_service_routes_deck_list_by_backend():
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
    )

    local_decks = await scope.list_decks(mode="local")
    server_decks = await scope.list_decks(mode="server")

    assert local_decks[0]["record_id"] == "local:study_deck:deck-local-1"
    assert server_decks[0]["record_id"] == "server:study_deck:7"
    assert server_decks[0]["scheduler_type"] == "fsrs"


@pytest.mark.asyncio
async def test_scope_service_routes_flashcard_create_and_list_by_backend():
    local = FakeLocalStudyService()
    scope = StudyScopeService(local_service=local, server_service=FakeServerStudyService())

    created = await scope.create_flashcard(
        mode="local",
        deck_id="deck-local-1",
        front="Question",
        back="Answer",
        tags=["science", "biology"],
    )
    listed = await scope.list_flashcards(mode="local", deck_id="deck-local-1", q="bio", limit=10, offset=1)

    assert created["record_id"] == "local:study_flashcard:card-local-1"
    assert listed[0]["deck_record_id"] == "local:study_deck:deck-local-1"
    assert local.calls[0] == ("create_flashcard", "deck-local-1", "Question", "Answer", ["science", "biology"], None, None)
    assert local.calls[1] == ("list_flashcards", "deck-local-1", "bio", 10, 1)


@pytest.mark.asyncio
async def test_scope_service_merges_server_review_outcome_with_current_card():
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=FakeLocalStudyService(), server_service=server)

    candidate = await scope.get_next_review_candidate(mode="server", deck_id=7)
    outcome = await scope.submit_flashcard_review(
        mode="server",
        card_id="card-server-1",
        rating=4,
        current_card=candidate["card"],
    )

    assert outcome["card"]["front"] == "Question"
    assert outcome["card"]["interval_days"] == 3
    assert outcome["review_session"]["review_session_id"] == 41
    assert server.calls == [
        ("get_next_review_candidate", 7),
        ("submit_flashcard_review", "card-server-1", 4, None),
    ]


@pytest.mark.asyncio
async def test_scope_service_ends_server_review_session_and_noops_locally():
    local = FakeLocalStudyService()
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=local, server_service=server)

    local_result = await scope.end_review_session(mode="local", review_session_id=11)
    server_result = await scope.end_review_session(mode="server", review_session_id=41)

    assert local_result is None
    assert server_result["status"] == "completed"
    assert ("end_review_session", 11) not in local.calls
    assert server.calls == [("end_review_session", 41)]


@pytest.mark.asyncio
async def test_scope_service_routes_move_delete_and_local_deck_delete():
    local = FakeLocalStudyService()
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=local, server_service=server)

    moved = await scope.move_flashcard(
        mode="server",
        card_id="card-server-1",
        target_deck_id=9,
        expected_version=2,
    )
    deleted_card = await scope.delete_flashcard(
        mode="server",
        card_id="card-server-1",
        expected_version=3,
    )
    deleted_deck = await scope.delete_deck(
        mode="local",
        deck_id="deck-local-1",
        expected_version=4,
    )

    assert moved["record_id"] == "server:study_flashcard:card-server-1"
    assert moved["deck_record_id"] == "server:study_deck:9"
    assert moved["version"] == 2
    assert deleted_card is True
    assert deleted_deck is True
    assert server.calls == [
        ("move_flashcard", "card-server-1", 9, 2),
        ("delete_flashcard", "card-server-1", 3),
    ]
    assert local.calls == [("delete_deck", "deck-local-1", 4, False)]


@pytest.mark.asyncio
async def test_scope_service_rejects_missing_expected_version_for_server_delete():
    server = FakeServerStudyService()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=server,
    )

    with pytest.raises(
        ValueError,
        match="expected_version is required for server flashcard deletion\\.",
    ):
        await scope.delete_flashcard(mode="server", card_id="card-server-1")

    assert server.calls == []


@pytest.mark.asyncio
async def test_scope_service_accepts_legacy_status_deleted_mapping():
    class LegacyStatusServerStudyService:
        def __init__(self):
            self.calls = []

        async def delete_flashcard(self, card_id, *, expected_version=None):
            self.calls.append(("delete_flashcard", card_id, expected_version))
            return {"status": "deleted"}

    server = LegacyStatusServerStudyService()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=server,
    )

    deleted = await scope.delete_flashcard(
        mode="server",
        card_id="card-server-legacy",
        expected_version=5,
    )

    assert deleted is True
    assert server.calls == [("delete_flashcard", "card-server-legacy", 5)]


@pytest.mark.asyncio
async def test_scope_service_does_not_swallow_server_deck_delete_unsupported_error():
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
    )

    with pytest.raises(
        NotImplementedError,
        match="Flashcard deck deletion is not supported by the current server API\\.",
    ):
        await scope.delete_deck(mode="server", deck_id=7, expected_version=2)
