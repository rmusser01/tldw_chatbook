import pytest

from tldw_chatbook.Study_Interop.study_scope_service import StudyScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


CARD_UUID = "00000000-0000-4000-8000-000000000001"


class FakeLocalStudyService:
    def __init__(self):
        self.calls = []

    def list_decks(self, *, limit=100, offset=0):
        self.calls.append(("list_decks", limit, offset))
        return [{"id": "deck-local-1", "name": "Biology", "description": "Cell review"}]

    def create_deck(self, *, name, description=None):
        self.calls.append(("create_deck", name, description))
        return {"id": "deck-local-1", "name": name, "description": description}

    def update_deck(self, deck_id, **request):
        self.calls.append(("update_deck", deck_id, request))
        return {
            "id": deck_id,
            "name": request.get("name") or "Biology",
            "description": request.get("description"),
            "version": request.get("expected_version", 1) + 1,
        }

    def get_flashcard(self, card_id):
        self.calls.append(("get_flashcard", card_id))
        return {
            "id": card_id,
            "deck_id": "deck-local-1",
            "front": "Question",
            "back": "Answer",
            "tags": "science",
            "type": "basic",
        }

    def set_flashcard_tags(self, card_id, *, tags):
        self.calls.append(("set_flashcard_tags", card_id, tags))
        return {
            "id": card_id,
            "deck_id": "deck-local-1",
            "front": "Question",
            "back": "Answer",
            "tags": " ".join(tags),
            "type": "basic",
            "version": 2,
        }

    def get_flashcard_tags(self, card_id):
        self.calls.append(("get_flashcard_tags", card_id))
        return {"uuid": card_id, "tags": ["science", "biology"]}

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

    def create_flashcards_bulk(self, cards):
        self.calls.append(("create_flashcards_bulk", cards))
        return {
            "items": [
                {
                    "id": "card-local-1",
                    "deck_id": "deck-local-1",
                    "front": "Question",
                    "back": "Answer",
                    "tags": "science",
                    "type": "basic",
                }
            ],
            "count": 1,
            "total": 1,
        }

    def update_flashcards_bulk(self, updates):
        self.calls.append(("update_flashcards_bulk", updates))
        return {
            "results": [
                {
                    "uuid": "card-local-1",
                    "status": "updated",
                    "flashcard": {
                        "id": "card-local-1",
                        "deck_id": "deck-local-1",
                        "front": "Updated",
                        "back": "Answer",
                        "tags": "science",
                        "type": "basic",
                        "version": 2,
                    },
                }
            ]
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

    async def update_deck(self, deck_id, **request):
        self.calls.append(("update_deck", deck_id, request))
        return {
            "id": deck_id,
            "name": request.get("name") or "Biology",
            "description": request.get("description"),
            "workspace_id": request.get("workspace_id"),
            "scheduler_type": request.get("scheduler_type") or "fsrs",
            "version": request.get("expected_version", 1) + 1,
        }

    async def upload_flashcard_asset(self, file):
        self.calls.append(("upload_flashcard_asset", file))
        return {
            "asset_uuid": CARD_UUID,
            "reference": f"flashcard-asset://{CARD_UUID}",
            "markdown_snippet": f"![image](flashcard-asset://{CARD_UUID})",
            "mime_type": "image/png",
            "byte_size": 7,
            "original_filename": "cell.png",
        }

    async def get_flashcard_asset_content(self, asset_uuid):
        self.calls.append(("get_flashcard_asset_content", asset_uuid))
        return {"content": b"pngdata", "content_type": "image/png", "filename": "cell.png"}

    async def create_flashcards_bulk(self, cards):
        self.calls.append(("create_flashcards_bulk", cards))
        return {
            "items": [
                {
                    "uuid": CARD_UUID,
                    "deck_id": 7,
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
                }
            ],
            "count": 1,
            "total": 1,
        }

    async def update_flashcards_bulk(self, updates):
        self.calls.append(("update_flashcards_bulk", updates))
        return {
            "results": [
                {
                    "uuid": CARD_UUID,
                    "status": "updated",
                    "flashcard": {
                        "uuid": CARD_UUID,
                        "deck_id": 7,
                        "front": "Updated",
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
                    },
                }
            ]
        }

    async def get_flashcard(self, card_id):
        self.calls.append(("get_flashcard", card_id))
        return {
            "uuid": card_id,
            "deck_id": 7,
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
        }

    async def reset_flashcard_scheduling(self, card_id, *, expected_version):
        self.calls.append(("reset_flashcard_scheduling", card_id, expected_version))
        return await self.get_flashcard(card_id)

    async def set_flashcard_tags(self, card_id, *, tags):
        self.calls.append(("set_flashcard_tags", card_id, tags))
        return {**(await self.get_flashcard(card_id)), "tags": tags, "version": 2}

    async def get_flashcard_tags(self, card_id):
        self.calls.append(("get_flashcard_tags", card_id))
        return {"uuid": card_id, "tags": ["science", "biology"]}

    async def preview_structured_qa_import(self, content, **limits):
        self.calls.append(("preview_structured_qa_import", content, limits))
        return {
            "drafts": [
                {
                    "front": "What powers the cell?",
                    "back": "ATP",
                    "line_start": 1,
                    "line_end": 2,
                    "tags": ["biology"],
                }
            ],
            "errors": [],
            "detected_format": "qa_labels",
            "skipped_blocks": 0,
        }

    async def import_flashcards(self, content, **request):
        self.calls.append(("import_flashcards", content, request))
        return {"imported": 1, "items": [{"uuid": CARD_UUID, "deck_id": 7}], "errors": []}

    async def import_flashcards_json(self, file, **limits):
        self.calls.append(("import_flashcards_json", file, limits))
        return {"imported": 1, "items": [{"uuid": CARD_UUID, "deck_id": 7}], "errors": []}

    async def import_flashcards_apkg(self, file, **limits):
        self.calls.append(("import_flashcards_apkg", file, limits))
        return {"imported": 1, "items": [{"uuid": CARD_UUID, "deck_id": 7}], "errors": []}

    async def list_flashcard_tag_suggestions(self, *, q=None, limit=50):
        self.calls.append(("list_flashcard_tag_suggestions", q, limit))
        return {"items": [{"tag": "science", "count": 12}], "count": 1}

    async def get_flashcard_analytics_summary(self, *, deck_id=None, workspace_id=None, include_workspace_items=None):
        self.calls.append(("get_flashcard_analytics_summary", deck_id, workspace_id, include_workspace_items))
        return {
            "reviewed_today": 4,
            "study_streak_days": 3,
            "generated_at": "2026-04-20T00:04:00Z",
            "decks": [{"deck_id": deck_id or 7, "deck_name": "Biology", "total": 10}],
        }

    async def list_review_sessions(self, *, deck_id=None, scope_key=None, status=None, limit=20):
        self.calls.append(("list_review_sessions", deck_id, scope_key, status, limit))
        return [{"id": 77, "deck_id": deck_id, "scope_key": scope_key or "deck:7", "status": status or "active"}]

    async def get_flashcard_assistant(self, card_id):
        self.calls.append(("get_flashcard_assistant", card_id))
        return {
            "thread": {"id": 88, "flashcard_uuid": card_id, "version": 1},
            "messages": [{"id": 1, "thread_id": 88, "role": "assistant", "content": "Explanation"}],
            "available_actions": ["explain", "follow_up"],
        }

    async def respond_flashcard_assistant(self, card_id, **request):
        self.calls.append(("respond_flashcard_assistant", card_id, request))
        return {
            "thread": {"id": 88, "flashcard_uuid": card_id, "version": 2},
            "user_message": {"id": 2, "thread_id": 88, "role": "user", "content": request.get("message")},
            "assistant_message": {"id": 3, "thread_id": 88, "role": "assistant", "content": "Because."},
        }

    async def generate_flashcards(self, **request):
        self.calls.append(("generate_flashcards", request))
        return {
            "flashcards": [{"front": "Generated Q", "back": "Generated A", "tags": ["generated"]}],
            "count": 1,
        }

    async def export_flashcards(self, **request):
        self.calls.append(("export_flashcards", request))
        return {
            "content": b"Deck\tFront\tBack\nBio\tQ\tA\n",
            "content_type": "text/tab-separated-values",
            "filename": "flashcards.tsv",
        }

    async def create_flashcard_template(self, **request):
        self.calls.append(("create_flashcard_template", request))
        return {
            "id": 3,
            "name": request["name"],
            "model_type": request.get("model_type", "basic"),
            "front_template": request["front_template"],
            "back_template": request.get("back_template"),
            "deleted": False,
            "client_id": "server-client",
            "version": 1,
        }

    async def list_flashcard_templates(self, *, limit=100, offset=0):
        self.calls.append(("list_flashcard_templates", limit, offset))
        return {
            "items": [
                {
                    "id": 3,
                    "name": "Basic science",
                    "model_type": "basic",
                    "front_template": "{{question}}",
                    "back_template": "{{answer}}",
                    "deleted": False,
                    "client_id": "server-client",
                    "version": 1,
                }
            ],
            "count": 1,
            "total": 1,
        }

    async def get_flashcard_template(self, template_id):
        self.calls.append(("get_flashcard_template", template_id))
        return {
            "id": template_id,
            "name": "Basic science",
            "model_type": "basic",
            "front_template": "{{question}}",
            "back_template": "{{answer}}",
            "deleted": False,
            "client_id": "server-client",
            "version": 1,
        }

    async def update_flashcard_template(self, template_id, **request):
        self.calls.append(("update_flashcard_template", template_id, request))
        return {
            "id": template_id,
            "name": request.get("name", "Basic science"),
            "model_type": request.get("model_type", "basic"),
            "front_template": request.get("front_template", "{{question}}"),
            "back_template": request.get("back_template", "{{answer}}"),
            "deleted": False,
            "client_id": "server-client",
            "version": 2,
        }

    async def delete_flashcard_template(self, template_id, *, expected_version):
        self.calls.append(("delete_flashcard_template", template_id, expected_version))
        return {"deleted": True}

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

    async def create_study_pack_job(self, *, title, workspace_id=None, source_items):
        self.calls.append(("create_study_pack_job", title, workspace_id, source_items))
        return {
            "job": {
                "id": 42,
                "status": "queued",
                "domain": "study_pack",
                "queue": "study",
                "job_type": "generate_study_pack",
            }
        }

    async def get_study_pack_job_status(self, job_id):
        self.calls.append(("get_study_pack_job_status", job_id))
        return {
            "job": {
                "id": job_id,
                "status": "completed",
                "domain": "study_pack",
                "queue": "study",
                "job_type": "generate_study_pack",
            },
            "study_pack": {"id": 9, "title": "Cell biology pack"},
        }

    async def get_study_pack(self, pack_id):
        self.calls.append(("get_study_pack", pack_id))
        return {"id": pack_id, "title": "Cell biology pack"}

    async def regenerate_study_pack(self, pack_id):
        self.calls.append(("regenerate_study_pack", pack_id))
        return {
            "job": {
                "id": 43,
                "status": "queued",
                "domain": "study_pack",
                "queue": "study",
                "job_type": "regenerate_study_pack",
            }
        }

    async def get_study_suggestion_status(self, *, anchor_type, anchor_id):
        self.calls.append(("get_study_suggestion_status", anchor_type, anchor_id))
        return {"anchor_type": anchor_type, "anchor_id": anchor_id, "status": "ready", "snapshot_id": 11}

    async def get_study_suggestion_snapshot(self, snapshot_id):
        self.calls.append(("get_study_suggestion_snapshot", snapshot_id))
        return {"snapshot": {"id": snapshot_id, "payload": {}}, "live_evidence": {}}

    async def refresh_study_suggestion_snapshot(self, snapshot_id, *, reason=None):
        self.calls.append(("refresh_study_suggestion_snapshot", snapshot_id, reason))
        return {"job": {"id": 44, "status": "queued"}}

    async def trigger_study_suggestion_action(self, snapshot_id, **request):
        self.calls.append(("trigger_study_suggestion_action", snapshot_id, request))
        return {
            "disposition": "generated",
            "snapshot_id": snapshot_id,
            "selection_fingerprint": "fp-1",
            "target_service": request["target_service"],
            "target_type": request["target_type"],
            "target_id": "quiz-9",
        }


class PagedFakeServerStudyService:
    def __init__(self, pages, *, fail_on_offset=None):
        self.calls = []
        self.pages = pages
        self.fail_on_offset = fail_on_offset

    async def list_decks(self, *, limit=100, offset=0):
        self.calls.append(("list_decks", limit, offset))
        if self.fail_on_offset is not None and offset == self.fail_on_offset:
            raise RuntimeError("deck page load failed")
        return list(self.pages.get(offset, []))

    async def create_deck(self, *, name, description=None, workspace_id=None, scheduler_type=None):
        self.calls.append(("create_deck", name, description, workspace_id, scheduler_type))
        return {"id": "deck-created", "name": name, "description": description, "workspace_id": workspace_id}


class FakePolicyEnforcer:
    def __init__(self, denied_reason: str | None = None):
        self.denied_reason = denied_reason
        self.calls = []

    @classmethod
    def deny(cls, reason_code: str) -> "FakePolicyEnforcer":
        return cls(denied_reason=reason_code)

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)
        if self.denied_reason is None:
            return
        raise PolicyDeniedError(
            action_id=action_id,
            reason_code=self.denied_reason,
            user_message=f"{action_id} denied",
            effective_source="local",
            authority_owner="server",
        )


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
async def test_scope_service_routes_local_deck_update_and_flashcard_detail():
    local = FakeLocalStudyService()
    scope = StudyScopeService(
        local_service=local,
        server_service=FakeServerStudyService(),
    )

    deck = await scope.update_deck(
        mode="local",
        deck_id="deck-local-1",
        name="Biology Updated",
        description="Cells and genetics",
        expected_version=3,
    )
    card = await scope.get_flashcard(mode="local", card_id="card-local-1")

    assert deck["record_id"] == "local:study_deck:deck-local-1"
    assert card["record_id"] == "local:study_flashcard:card-local-1"
    assert local.calls == [
        (
            "update_deck",
            "deck-local-1",
            {
                "name": "Biology Updated",
                "description": "Cells and genetics",
                "workspace_id": None,
                "review_prompt_side": None,
                "scheduler_type": None,
                "scheduler_settings": None,
                "expected_version": 3,
            },
        ),
        ("get_flashcard", "card-local-1"),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_flashcard_helper_methods_and_normalizes_payloads():
    server = FakeServerStudyService()
    policy_enforcer = FakePolicyEnforcer()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    deck = await scope.update_deck(
        mode="server",
        deck_id=7,
        name="Biology Updated",
        description="Cells and genetics",
        expected_version=3,
    )
    card = await scope.get_flashcard(mode="server", card_id=CARD_UUID)
    reset = await scope.reset_flashcard_scheduling(mode="server", card_id=CARD_UUID, expected_version=3)
    tagged = await scope.set_flashcard_tags(mode="server", card_id=CARD_UUID, tags=["science", "biology"])
    tags = await scope.get_flashcard_tags(mode="server", card_id=CARD_UUID)
    suggestions = await scope.list_flashcard_tag_suggestions(mode="server", q="sci", limit=5)
    analytics = await scope.get_flashcard_analytics_summary(mode="server", deck_id=7)
    sessions = await scope.list_review_sessions(mode="server", deck_id=7, status="active", limit=2)
    assistant = await scope.get_flashcard_assistant(mode="server", card_id=CARD_UUID)
    assistant_response = await scope.respond_flashcard_assistant(
        mode="server",
        card_id=CARD_UUID,
        action="follow_up",
        message="Why?",
        expected_thread_version=1,
    )
    generated = await scope.generate_flashcards(
        mode="server",
        text="Cells divide by mitosis.",
        num_cards=1,
        focus_topics=["mitosis"],
    )

    assert deck["record_id"] == "server:study_deck:7"
    assert card["record_id"] == f"server:study_flashcard:{CARD_UUID}"
    assert reset["record_id"] == f"server:study_flashcard:{CARD_UUID}"
    assert tagged["tags"] == ["science", "biology"]
    assert tags["record_id"] == f"server:study_flashcard_tags:{CARD_UUID}"
    assert suggestions["record_id"] == "server:study_flashcard_tag_suggestions:sci"
    assert analytics["record_id"] == "server:study_flashcard_analytics:deck:7"
    assert sessions[0]["record_id"] == "server:study_review_session:77"
    assert assistant["record_id"] == f"server:study_flashcard_assistant:{CARD_UUID}"
    assert assistant["thread"]["record_id"] == "server:study_flashcard_assistant_thread:88"
    assert assistant_response["assistant_message"]["record_id"] == "server:study_flashcard_assistant_message:3"
    assert generated["record_id"] == "server:study_flashcard_generation:transient"
    assert policy_enforcer.calls == [
        "study.deck.update.server",
        "study.flashcard.detail.server",
        "study.flashcard.update.server",
        "study.flashcard.tags.update.server",
        "study.flashcard.tags.list.server",
        "study.flashcard.tags.list.server",
        "study.flashcard.analytics.observe.server",
        "study.flashcard.review_sessions.list.server",
        "study.flashcard.assistant.detail.server",
        "study.flashcard.assistant.launch.server",
        "study.flashcard.generation.launch.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_flashcard_bulk_import_export_and_assets():
    server = FakeServerStudyService()
    policy_enforcer = FakePolicyEnforcer()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    uploaded = await scope.upload_flashcard_asset(
        mode="server",
        file=("cell.png", b"pngdata", "image/png"),
    )
    content = await scope.get_flashcard_asset_content(mode="server", asset_uuid=CARD_UUID)
    created_bulk = await scope.create_flashcards_bulk(
        mode="server",
        cards=[{"deck_id": 7, "front": "Question", "back": "Answer", "tags": ["science"]}],
    )
    updated_bulk = await scope.update_flashcards_bulk(
        mode="server",
        updates=[{"uuid": CARD_UUID, "front": "Updated", "expected_version": 1}],
    )
    preview = await scope.preview_structured_qa_import(
        mode="server",
        content="Q: What powers the cell?\nA: ATP",
        max_lines=10,
    )
    imported = await scope.import_flashcards(
        mode="server",
        content="Deck\tFront\tBack\nBio\tQ\tA",
        has_header=True,
    )
    imported_json = await scope.import_flashcards_json(
        mode="server",
        file=("cards.json", b"[]", "application/json"),
        max_items=10,
    )
    imported_apkg = await scope.import_flashcards_apkg(
        mode="server",
        file=("cards.apkg", b"apkg", "application/octet-stream"),
    )
    exported = await scope.export_flashcards(
        mode="server",
        deck_id=7,
        format="tsv",
        delimiter="\t",
        include_header=True,
    )

    assert uploaded["record_id"] == f"server:study_flashcard_asset:{CARD_UUID}"
    assert content["record_id"] == f"server:study_flashcard_asset_content:{CARD_UUID}"
    assert created_bulk["record_id"] == "server:study_flashcard_bulk_create:transient"
    assert created_bulk["items"][0]["record_id"] == f"server:study_flashcard:{CARD_UUID}"
    assert updated_bulk["record_id"] == "server:study_flashcard_bulk_update:transient"
    assert updated_bulk["results"][0]["flashcard"]["record_id"] == f"server:study_flashcard:{CARD_UUID}"
    assert preview["record_id"] == "server:study_flashcard_import_preview:transient"
    assert imported["record_id"] == "server:study_flashcard_import:structured"
    assert imported_json["record_id"] == "server:study_flashcard_import:json"
    assert imported_apkg["record_id"] == "server:study_flashcard_import:apkg"
    assert exported["record_id"] == "server:study_flashcard_export:transient"
    assert exported["filename"] == "flashcards.tsv"
    assert policy_enforcer.calls == [
        "study.flashcard.assets.create.server",
        "study.flashcard.assets.detail.server",
        "study.flashcard.bulk.create.server",
        "study.flashcard.bulk.update.server",
        "study.flashcard.import.preview.server",
        "study.flashcard.import.import.server",
        "study.flashcard.import.import.server",
        "study.flashcard.import.import.server",
        "study.flashcard.export.export.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_flashcard_tags_and_bulk_helpers():
    local = FakeLocalStudyService()
    policy_enforcer = FakePolicyEnforcer()
    scope = StudyScopeService(
        local_service=local,
        server_service=FakeServerStudyService(),
        policy_enforcer=policy_enforcer,
    )

    tagged = await scope.set_flashcard_tags(
        mode="local",
        card_id="card-local-1",
        tags=["science", "biology"],
    )
    tags = await scope.get_flashcard_tags(mode="local", card_id="card-local-1")
    created_bulk = await scope.create_flashcards_bulk(
        mode="local",
        cards=[{"deck_id": "deck-local-1", "front": "Question", "back": "Answer", "tags": ["science"]}],
    )
    updated_bulk = await scope.update_flashcards_bulk(
        mode="local",
        updates=[{"uuid": "card-local-1", "front": "Updated", "expected_version": 1}],
    )

    assert tagged["record_id"] == "local:study_flashcard:card-local-1"
    assert tagged["tags"] == ["science", "biology"]
    assert tags["record_id"] == "local:study_flashcard_tags:card-local-1"
    assert tags["tags"] == ["science", "biology"]
    assert created_bulk["record_id"] == "local:study_flashcard_bulk_create:transient"
    assert created_bulk["items"][0]["record_id"] == "local:study_flashcard:card-local-1"
    assert updated_bulk["record_id"] == "local:study_flashcard_bulk_update:transient"
    assert updated_bulk["results"][0]["flashcard"]["record_id"] == "local:study_flashcard:card-local-1"
    assert policy_enforcer.calls == [
        "study.flashcard.tags.update.local",
        "study.flashcard.tags.list.local",
        "study.flashcard.bulk.create.local",
        "study.flashcard.bulk.update.local",
    ]
    assert local.calls == [
        ("set_flashcard_tags", "card-local-1", ["science", "biology"]),
        ("get_flashcard_tags", "card-local-1"),
        ("create_flashcards_bulk", [{"deck_id": "deck-local-1", "front": "Question", "back": "Answer", "tags": ["science"]}]),
        ("update_flashcards_bulk", [{"uuid": "card-local-1", "front": "Updated", "expected_version": 1}]),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_flashcard_template_crud_and_normalizes_payloads():
    server = FakeServerStudyService()
    policy_enforcer = FakePolicyEnforcer()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    created = await scope.create_flashcard_template(
        mode="server",
        name="Basic science",
        front_template="{{question}}",
        back_template="{{answer}}",
    )
    listed = await scope.list_flashcard_templates(mode="server", limit=5, offset=1)
    fetched = await scope.get_flashcard_template(mode="server", template_id=3)
    updated = await scope.update_flashcard_template(
        mode="server",
        template_id=3,
        name="Updated science",
        expected_version=1,
    )
    deleted = await scope.delete_flashcard_template(mode="server", template_id=3, expected_version=2)

    assert created["record_id"] == "server:study_flashcard_template:3"
    assert listed["record_id"] == "server:study_flashcard_template_list:5:1"
    assert listed["items"][0]["record_id"] == "server:study_flashcard_template:3"
    assert fetched["record_id"] == "server:study_flashcard_template:3"
    assert updated["version"] == 2
    assert deleted is True
    assert policy_enforcer.calls == [
        "study.flashcard.templates.create.server",
        "study.flashcard.templates.list.server",
        "study.flashcard.templates.detail.server",
        "study.flashcard.templates.update.server",
        "study.flashcard.templates.delete.server",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("list_flashcard_tag_suggestions", {"q": "sci"}),
        ("get_flashcard_analytics_summary", {"deck_id": "deck-local-1"}),
        ("upload_flashcard_asset", {"file": ("cell.png", b"pngdata", "image/png")}),
        ("get_flashcard_asset_content", {"asset_uuid": "asset-local-1"}),
        ("preview_structured_qa_import", {"content": "Q: What powers the cell?\nA: ATP"}),
        ("import_flashcards", {"content": "Deck\tFront\tBack\nBio\tQ\tA"}),
        ("import_flashcards_json", {"file": ("cards.json", b"[]", "application/json")}),
        ("import_flashcards_apkg", {"file": ("cards.apkg", b"apkg", "application/octet-stream")}),
        ("export_flashcards", {}),
        ("list_review_sessions", {"deck_id": "deck-local-1"}),
        ("get_flashcard_assistant", {"card_id": "card-local-1"}),
        ("respond_flashcard_assistant", {"card_id": "card-local-1", "action": "follow_up", "message": "Why?"}),
        ("generate_flashcards", {"text": "Cells divide by mitosis."}),
        ("list_flashcard_templates", {}),
        ("create_flashcard_template", {"name": "Basic science", "front_template": "{{question}}"}),
        ("get_flashcard_template", {"template_id": 3}),
        ("update_flashcard_template", {"template_id": 3, "name": "Updated science"}),
        ("delete_flashcard_template", {"template_id": 3, "expected_version": 1}),
    ],
)
async def test_scope_service_rejects_remote_only_flashcard_helpers_in_local_mode(method_name, kwargs):
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
    )

    with pytest.raises(ValueError, match="server-only"):
        await getattr(scope, method_name)(mode="local", **kwargs)


@pytest.mark.asyncio
async def test_study_scope_service_denies_server_deck_create_when_runtime_policy_blocks_it():
    policy_enforcer = FakePolicyEnforcer.deny("wrong_source")
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.create_deck(
            mode="server",
            scope_type="global",
            name="Remote deck",
            description="Body",
        )

    assert exc.value.reason_code == "wrong_source"
    assert policy_enforcer.calls == ["study.deck.create.server"]


@pytest.mark.asyncio
async def test_study_scope_service_keeps_normalizing_server_decks_after_policy_passes():
    policy_enforcer = FakePolicyEnforcer()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
        policy_enforcer=policy_enforcer,
    )

    server_decks = await scope.list_decks(mode="server")

    assert server_decks[0]["record_id"] == "server:study_deck:7"
    assert policy_enforcer.calls == ["study.deck.list.server"]


@pytest.mark.asyncio
async def test_study_scope_service_routes_flashcard_mutations_through_deck_policy_proxy():
    policy_enforcer = FakePolicyEnforcer()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
        policy_enforcer=policy_enforcer,
    )

    created = await scope.create_flashcard(
        mode="local",
        deck_id="deck-local-1",
        front="Question",
        back="Answer",
    )

    assert created["record_id"] == "local:study_flashcard:card-local-1"
    assert policy_enforcer.calls == ["study.deck.update.local"]


@pytest.mark.asyncio
async def test_scope_service_filters_global_and_workspace_decks_without_mixing():
    server = PagedFakeServerStudyService(
        {
            0: [
                {"id": 1, "name": "Global One", "workspace_id": None},
                {"id": 2, "name": "Workspace One A", "workspace_id": "ws-1"},
            ],
            2: [
                {"id": 3, "name": "Workspace Two", "workspace_id": "ws-2"},
                {"id": 4, "name": "Global Two", "workspace_id": None},
            ],
            4: [
                {"id": 5, "name": "Workspace One B", "workspace_id": "ws-1"},
            ],
        }
    )
    scope = StudyScopeService(local_service=FakeLocalStudyService(), server_service=server)

    global_decks = await scope.list_decks(mode="server", scope_type="global", limit=2, offset=0)
    workspace_decks = await scope.list_decks(
        mode="server",
        scope_type="workspace",
        workspace_id="ws-1",
        limit=2,
        offset=0,
    )

    assert [deck["record_id"] for deck in global_decks] == ["server:study_deck:1", "server:study_deck:4"]
    assert [deck["record_id"] for deck in workspace_decks] == ["server:study_deck:2", "server:study_deck:5"]
    assert server.calls == [
        ("list_decks", 2, 0),
        ("list_decks", 2, 2),
        ("list_decks", 2, 4),
        ("list_decks", 2, 0),
        ("list_decks", 2, 2),
        ("list_decks", 2, 4),
    ]


@pytest.mark.asyncio
async def test_scope_service_applies_offset_after_client_side_deck_scope_filtering():
    server = PagedFakeServerStudyService(
        {
            0: [
                {"id": 1, "name": "Global One", "workspace_id": None},
                {"id": 2, "name": "Workspace One A", "workspace_id": "ws-1"},
            ],
            2: [
                {"id": 3, "name": "Workspace Two", "workspace_id": "ws-2"},
                {"id": 4, "name": "Global Two", "workspace_id": None},
            ],
            4: [
                {"id": 5, "name": "Workspace One B", "workspace_id": "ws-1"},
            ],
        }
    )
    scope = StudyScopeService(local_service=FakeLocalStudyService(), server_service=server)

    global_decks = await scope.list_decks(mode="server", scope_type="global", limit=2, offset=1)
    workspace_decks = await scope.list_decks(
        mode="server",
        scope_type="workspace",
        workspace_id="ws-1",
        limit=2,
        offset=1,
    )

    assert [deck["record_id"] for deck in global_decks] == ["server:study_deck:4"]
    assert [deck["record_id"] for deck in workspace_decks] == ["server:study_deck:5"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("list_decks", {"mode": "server", "scope_type": "invalid", "limit": 1}),
        (
            "create_deck",
            {"mode": "server", "scope_type": "invalid", "name": "Workspace deck", "description": None},
        ),
    ],
)
async def test_scope_service_rejects_invalid_scope_type_for_scoped_methods(method_name, kwargs):
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
    )

    with pytest.raises(ValueError, match="Invalid study scope_type: invalid"):
        await getattr(scope, method_name)(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("list_decks", {"mode": "server", "scope_type": "workspace", "limit": 1}),
        (
            "create_deck",
            {"mode": "server", "scope_type": "workspace", "name": "Workspace deck", "description": None},
        ),
    ],
)
async def test_scope_service_rejects_missing_workspace_id_for_workspace_scope(method_name, kwargs):
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
    )

    with pytest.raises(ValueError, match="workspace_id is required when scope_type='workspace'"):
        await getattr(scope, method_name)(**kwargs)


@pytest.mark.asyncio
async def test_scope_service_rejects_workspace_scope_in_local_mode():
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
    )

    with pytest.raises(ValueError, match="Workspace Study is unavailable in local mode"):
        await scope.list_decks(mode="local", scope_type="workspace", workspace_id="ws-1")

    with pytest.raises(ValueError, match="Workspace Study is unavailable in local mode"):
        await scope.create_deck(
            mode="local",
            scope_type="workspace",
            workspace_id="ws-1",
            name="Workspace deck",
            description=None,
        )


@pytest.mark.asyncio
async def test_scope_service_rejects_workspace_scope_for_local_review_calls():
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
    )

    with pytest.raises(ValueError, match="Workspace Study is unavailable in local mode"):
        await scope.get_next_review_candidate(
            mode="local",
            scope_type="workspace",
            workspace_id="ws-1",
            deck_id="deck-local-1",
        )

    with pytest.raises(ValueError, match="Workspace Study is unavailable in local mode"):
        await scope.submit_flashcard_review(
            mode="local",
            scope_type="workspace",
            workspace_id="ws-1",
            card_id="card-local-1",
            rating=3,
        )


@pytest.mark.asyncio
async def test_scope_service_forwards_workspace_id_on_server_create_and_nulls_global_create():
    server = PagedFakeServerStudyService({0: []})
    scope = StudyScopeService(local_service=FakeLocalStudyService(), server_service=server)

    await scope.create_deck(
        mode="server",
        scope_type="workspace",
        workspace_id="ws-7",
        name="Workspace deck",
        description="Scoped",
    )
    await scope.create_deck(
        mode="server",
        scope_type="global",
        workspace_id="ws-7",
        name="Global deck",
        description=None,
    )

    assert server.calls == [
        ("create_deck", "Workspace deck", "Scoped", "ws-7", None),
        ("create_deck", "Global deck", None, None, None),
    ]


@pytest.mark.asyncio
async def test_scope_service_keeps_workspace_load_errors_scoped():
    server = PagedFakeServerStudyService(
        {
            0: [
                {"id": 1, "name": "Global One", "workspace_id": None},
                {"id": 2, "name": "Workspace One", "workspace_id": "ws-2"},
            ],
            2: [{"id": 3, "name": "Workspace Two", "workspace_id": "ws-1"}],
        },
        fail_on_offset=2,
    )
    scope = StudyScopeService(local_service=FakeLocalStudyService(), server_service=server)

    with pytest.raises(RuntimeError, match="deck page load failed"):
        await scope.list_decks(mode="server", scope_type="workspace", workspace_id="ws-1", limit=2, offset=0)


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
@pytest.mark.parametrize("expected_version", [0, -1])
async def test_scope_service_rejects_non_positive_expected_version_for_server_delete(expected_version):
    server = FakeServerStudyService()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=server,
    )

    with pytest.raises(
        ValueError,
        match="expected_version must be >= 1 for server flashcard deletion\\.",
    ):
        await scope.delete_flashcard(
            mode="server",
            card_id="card-server-1",
            expected_version=expected_version,
        )

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


@pytest.mark.asyncio
async def test_scope_service_blocks_server_deck_delete_before_dispatch():
    class ServerStudyServiceWithDelete(FakeServerStudyService):
        async def delete_deck(self, deck_id, *, expected_version=None, hard_delete=False):
            self.calls.append(("delete_deck", deck_id, expected_version, hard_delete))
            return {"status": "deleted"}

    server = ServerStudyServiceWithDelete()
    policy_enforcer = FakePolicyEnforcer()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(
        NotImplementedError,
        match="Flashcard deck deletion is not supported by the current server API\\.",
    ):
        await scope.delete_deck(mode="server", deck_id=7, expected_version=2)

    assert server.calls == []
    assert policy_enforcer.calls == ["study.deck.delete.server"]


@pytest.mark.asyncio
async def test_scope_service_routes_study_pack_jobs_to_server_with_policy():
    server = FakeServerStudyService()
    policy_enforcer = FakePolicyEnforcer()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    created = await scope.create_study_pack_job(
        mode="server",
        title="Cell biology pack",
        workspace_id="ws-1",
        source_items=[{"source_type": "note", "source_id": "note-1"}],
    )
    status = await scope.get_study_pack_job_status(mode="server", job_id=42)
    pack = await scope.get_study_pack(mode="server", pack_id=9)
    regenerated = await scope.regenerate_study_pack(mode="server", pack_id=9)

    assert created["job"]["id"] == 42
    assert created["backend"] == "server"
    assert created["job"]["record_id"] == "server:study_pack_job:42"
    assert status["study_pack"]["id"] == 9
    assert status["study_pack"]["record_id"] == "server:study_pack:9"
    assert pack["id"] == 9
    assert pack["record_id"] == "server:study_pack:9"
    assert regenerated["job"]["id"] == 43
    assert regenerated["job"]["record_id"] == "server:study_pack_job:43"
    assert policy_enforcer.calls == [
        "study.packs.jobs.launch.server",
        "study.packs.jobs.observe.server",
        "study.packs.jobs.observe.server",
        "study.packs.jobs.launch.server",
    ]
    assert server.calls[-4:] == [
        ("create_study_pack_job", "Cell biology pack", "ws-1", [{"source_type": "note", "source_id": "note-1"}]),
        ("get_study_pack_job_status", 42),
        ("get_study_pack", 9),
        ("regenerate_study_pack", 9),
    ]


@pytest.mark.asyncio
async def test_scope_service_rejects_study_pack_jobs_in_local_mode_before_dispatch():
    local = FakeLocalStudyService()
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=local, server_service=server)

    with pytest.raises(ValueError, match="Study packs are server-only"):
        await scope.create_study_pack_job(
            mode="local",
            title="Cell biology pack",
            source_items=[{"source_type": "note", "source_id": "note-1"}],
        )

    assert local.calls == []
    assert server.calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_study_suggestions_to_server_with_policy():
    server = FakeServerStudyService()
    policy_enforcer = FakePolicyEnforcer()
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    status = await scope.get_study_suggestion_status(mode="server", anchor_type="deck", anchor_id=7)
    snapshot = await scope.get_study_suggestion_snapshot(mode="server", snapshot_id=11)
    refresh = await scope.refresh_study_suggestion_snapshot(mode="server", snapshot_id=11, reason="manual")
    action = await scope.trigger_study_suggestion_action(
        mode="server",
        snapshot_id=11,
        target_service="quiz",
        target_type="quiz",
        action_kind="generate",
        selected_topic_ids=["mitosis"],
        has_explicit_selection=True,
    )

    assert status["snapshot_id"] == 11
    assert status["record_id"] == "server:study_suggestion_status:deck:7"
    assert snapshot["snapshot"]["id"] == 11
    assert snapshot["snapshot"]["record_id"] == "server:study_suggestion_snapshot:11"
    assert refresh["job"]["id"] == 44
    assert refresh["job"]["record_id"] == "server:study_suggestion_job:44"
    assert action["target_id"] == "quiz-9"
    assert action["record_id"] == "server:study_suggestion_action:11:fp-1"
    assert policy_enforcer.calls == [
        "study.suggestions.list.server",
        "study.suggestions.observe.server",
        "study.suggestions.launch.server",
        "study.suggestions.configure.server",
    ]
    assert server.calls[-4:] == [
        ("get_study_suggestion_status", "deck", 7),
        ("get_study_suggestion_snapshot", 11),
        ("refresh_study_suggestion_snapshot", 11, "manual"),
        (
            "trigger_study_suggestion_action",
            11,
            {
                "target_service": "quiz",
                "target_type": "quiz",
                "action_kind": "generate",
                "selected_topic_ids": ["mitosis"],
                "selected_topic_edits": [],
                "manual_topic_labels": [],
                "has_explicit_selection": True,
                "generator_version": "v1",
                "force_regenerate": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_scope_service_rejects_study_suggestions_in_local_mode_before_dispatch():
    local = FakeLocalStudyService()
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=local, server_service=server)

    with pytest.raises(ValueError, match="Study suggestions are server-only"):
        await scope.get_study_suggestion_status(mode="local", anchor_type="deck", anchor_id=7)

    assert local.calls == []
    assert server.calls == []


def test_scope_service_reports_study_pack_and_suggestion_unsupported_capabilities():
    scope = StudyScopeService(
        local_service=FakeLocalStudyService(),
        server_service=FakeServerStudyService(),
    )

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
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
    assert server_report == [
        {
            "operation_id": "study.deck.delete.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "Flashcard deck deletion is not supported by the current server API.",
            "affected_action_ids": ["study.deck.delete.server"],
        },
        {
            "operation_id": "study.packs.jobs.list.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server study-pack contract exposes launch, job status, pack detail, and regenerate, but not job listing/discovery.",
            "affected_action_ids": ["study.packs.jobs.list.server"],
        }
    ]
