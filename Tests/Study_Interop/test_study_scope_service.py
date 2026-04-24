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

    async def update_deck(self, deck_id, *, name=None, description=None, workspace_id=None, review_prompt_side=None, scheduler_type=None, scheduler_settings=None, expected_version=None):
        self.calls.append(("update_deck", deck_id, name, description, workspace_id, review_prompt_side, scheduler_type, scheduler_settings, expected_version))
        return {
            "id": int(deck_id),
            "name": name or "Biology v2",
            "description": description,
            "workspace_id": workspace_id,
            "review_prompt_side": review_prompt_side or "front",
            "scheduler_type": scheduler_type or "fsrs",
            "deleted": False,
            "client_id": "server-client",
            "version": 2,
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
        return await self.get_flashcard(card_id)

    async def get_flashcard_tags(self, card_id):
        self.calls.append(("get_flashcard_tags", card_id))
        return {"items": ["science", "biology"], "count": 2}

    async def get_flashcard_analytics_summary(self, *, deck_id=None, workspace_id=None, include_workspace_items=False):
        self.calls.append(("get_flashcard_analytics_summary", deck_id, workspace_id, include_workspace_items))
        return {
            "reviewed_today": 3,
            "study_streak_days": 4,
            "generated_at": "2026-04-23T12:00:00Z",
            "decks": [{"deck_id": 7, "deck_name": "Biology"}],
        }

    async def create_flashcards_bulk(self, cards):
        self.calls.append(("create_flashcards_bulk", cards))
        return {
            "items": [
                {
                    "uuid": "card-server-1",
                    "deck_id": cards[0].get("deck_id"),
                    "front": cards[0]["front"],
                    "back": cards[0]["back"],
                    "tags": cards[0].get("tags") or [],
                    "is_cloze": False,
                    "ef": 2.5,
                    "interval_days": 0,
                    "repetitions": 0,
                    "lapses": 0,
                    "queue_state": "new",
                    "deleted": False,
                    "client_id": "server-client",
                    "version": 1,
                    "model_type": "basic",
                    "reverse": False,
                }
            ],
            "count": 1,
        }

    async def update_flashcards_bulk(self, cards):
        self.calls.append(("update_flashcards_bulk", cards))
        return {
            "results": [
                {
                    "uuid": cards[0]["uuid"],
                    "status": "updated",
                    "flashcard": {
                        "uuid": cards[0]["uuid"],
                        "deck_id": 7,
                        "front": "Question",
                        "back": "Answer",
                        "tags": cards[0].get("tags") or [],
                        "is_cloze": False,
                        "ef": 2.5,
                        "interval_days": 0,
                        "repetitions": 0,
                        "lapses": 0,
                        "queue_state": "new",
                        "deleted": False,
                        "client_id": "server-client",
                        "version": 2,
                        "model_type": "basic",
                        "reverse": False,
                    },
                    "error": None,
                }
            ]
        }

    async def list_flashcard_tag_suggestions(self, *, q=None, limit=50):
        self.calls.append(("list_flashcard_tag_suggestions", q, limit))
        return {"items": [{"tag": "biology", "count": 3}], "count": 1}

    async def preview_structured_qa_import(self, content, *, max_lines=None, max_line_length=None, max_field_length=None):
        self.calls.append(("preview_structured_qa_import", content, max_lines, max_line_length, max_field_length))
        return {
            "drafts": [
                {
                    "front": "What powers cells?",
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

    async def import_flashcards_tsv(self, content, *, delimiter="\t", has_header=False, max_lines=None, max_line_length=None, max_field_length=None):
        self.calls.append(("import_flashcards_tsv", content, delimiter, has_header, max_lines, max_line_length, max_field_length))
        return {"imported": 1, "items": [{"uuid": "card-server-1", "deck_id": 7}], "errors": []}

    async def export_flashcards(self, **kwargs):
        self.calls.append(("export_flashcards", kwargs))
        return b"Deck\tFront\tBack\nBiology\tQ\tA\n"

    async def create_flashcard_template(
        self,
        *,
        name,
        model_type="basic",
        front_template,
        back_template=None,
        notes_template=None,
        extra_template=None,
        placeholder_definitions=None,
    ):
        self.calls.append(
            (
                "create_flashcard_template",
                name,
                model_type,
                front_template,
                back_template,
                notes_template,
                extra_template,
                placeholder_definitions,
            )
        )
        return {
            "id": 12,
            "name": name,
            "model_type": model_type,
            "front_template": front_template,
            "back_template": back_template,
            "notes_template": notes_template,
            "extra_template": extra_template,
            "placeholder_definitions": placeholder_definitions or [],
            "deleted": False,
            "client_id": "server-client",
            "version": 1,
        }

    async def list_flashcard_templates(self, *, limit=100, offset=0):
        self.calls.append(("list_flashcard_templates", limit, offset))
        return {
            "items": [
                {
                    "id": 12,
                    "name": "Cloze Drill",
                    "model_type": "cloze",
                    "front_template": "{{statement}}",
                    "placeholder_definitions": [],
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
            "name": "Cloze Drill",
            "model_type": "cloze",
            "front_template": "{{statement}}",
            "placeholder_definitions": [],
            "deleted": False,
            "client_id": "server-client",
            "version": 1,
        }

    async def update_flashcard_template(
        self,
        template_id,
        *,
        name=None,
        model_type=None,
        front_template=None,
        back_template=None,
        notes_template=None,
        extra_template=None,
        placeholder_definitions=None,
        expected_version=None,
    ):
        self.calls.append(
            (
                "update_flashcard_template",
                template_id,
                name,
                model_type,
                front_template,
                back_template,
                notes_template,
                extra_template,
                placeholder_definitions,
                expected_version,
            )
        )
        return {
            "id": template_id,
            "name": name or "Cloze Drill",
            "model_type": model_type or "cloze",
            "front_template": front_template or "{{statement}}",
            "back_template": back_template,
            "notes_template": notes_template,
            "extra_template": extra_template,
            "placeholder_definitions": placeholder_definitions or [],
            "deleted": False,
            "client_id": "server-client",
            "version": 2,
        }

    async def delete_flashcard_template(self, template_id, *, expected_version):
        self.calls.append(("delete_flashcard_template", template_id, expected_version))
        return {"deleted": True}

    async def end_review_session(self, review_session_id):
        self.calls.append(("end_review_session", review_session_id))
        return {"id": review_session_id, "status": "completed"}

    async def create_study_pack_job(self, *, title, source_items, workspace_id=None, deck_mode="new"):
        self.calls.append(("create_study_pack_job", title, source_items, workspace_id, deck_mode))
        return {"job": {"id": 41, "status": "queued"}}

    async def get_study_pack_job_status(self, job_id):
        self.calls.append(("get_study_pack_job_status", job_id))
        return {"job": {"id": int(job_id), "status": "completed"}, "study_pack": {"id": 9, "title": "Pack"}}

    async def get_study_pack(self, pack_id):
        self.calls.append(("get_study_pack", pack_id))
        return {"record_id": f"server:study-pack:{pack_id}", "id": int(pack_id), "title": "Pack"}

    async def regenerate_study_pack(self, pack_id):
        self.calls.append(("regenerate_study_pack", pack_id))
        return {"job": {"id": 42, "status": "queued"}}

    async def get_study_suggestion_status(self, *, anchor_type, anchor_id):
        self.calls.append(("get_study_suggestion_status", anchor_type, anchor_id))
        return {"anchor_type": anchor_type, "anchor_id": int(anchor_id), "status": "ready", "snapshot_id": 23}

    async def get_study_suggestion_snapshot(self, snapshot_id):
        self.calls.append(("get_study_suggestion_snapshot", snapshot_id))
        return {"snapshot": {"id": int(snapshot_id), "payload": {}}, "live_evidence": {}}

    async def refresh_study_suggestion_snapshot(self, snapshot_id, *, reason=None):
        self.calls.append(("refresh_study_suggestion_snapshot", snapshot_id, reason))
        return {"job": {"id": 45, "status": "queued"}}

    async def trigger_study_suggestion_action(self, snapshot_id, **payload):
        self.calls.append(("trigger_study_suggestion_action", snapshot_id, payload))
        return {"snapshot_id": int(snapshot_id), "target_service": payload["target_service"], "target_type": payload["target_type"], "target_id": "7"}


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
async def test_scope_service_routes_study_pack_server_only_actions():
    local = FakeLocalStudyService()
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=local, server_service=server)

    created = await scope.create_study_pack_job(
        mode="server",
        title="Cell Biology Pack",
        workspace_id="ws-1",
        source_items=[{"source_type": "note", "source_id": "note-1"}],
    )
    status = await scope.get_study_pack_job_status(mode="server", job_id=41)
    pack = await scope.get_study_pack(mode="server", pack_id=9)
    regenerated = await scope.regenerate_study_pack(mode="server", pack_id=9)

    assert created["source"] == "server"
    assert created["job"]["id"] == 41
    assert status["study_pack"]["title"] == "Pack"
    assert pack["record_id"] == "server:study-pack:9"
    assert regenerated["job"]["id"] == 42
    assert server.calls[-4:] == [
        ("create_study_pack_job", "Cell Biology Pack", [{"source_type": "note", "source_id": "note-1"}], "ws-1", "new"),
        ("get_study_pack_job_status", 41),
        ("get_study_pack", 9),
        ("regenerate_study_pack", 9),
    ]

    with pytest.raises(ValueError, match="Study packs are server-only"):
        await scope.create_study_pack_job(
            mode="local",
            title="Offline Pack",
            source_items=[{"source_type": "note", "source_id": "note-1"}],
        )


@pytest.mark.asyncio
async def test_scope_service_routes_study_suggestion_server_only_actions():
    local = FakeLocalStudyService()
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=local, server_service=server)

    status = await scope.get_study_suggestion_status(mode="server", anchor_type="quiz_attempt", anchor_id=17)
    snapshot = await scope.get_study_suggestion_snapshot(mode="server", snapshot_id=23)
    refresh = await scope.refresh_study_suggestion_snapshot(mode="server", snapshot_id=23, reason="user-request")
    action = await scope.trigger_study_suggestion_action(
        mode="server",
        snapshot_id=23,
        target_service="flashcards",
        target_type="deck",
        action_kind="generate",
        selected_topic_ids=["topic-1"],
        has_explicit_selection=True,
    )

    assert status["source"] == "server"
    assert status["status"] == "ready"
    assert snapshot["snapshot"]["id"] == 23
    assert refresh["job"]["id"] == 45
    assert action["target_id"] == "7"
    assert server.calls[-4:] == [
        ("get_study_suggestion_status", "quiz_attempt", 17),
        ("get_study_suggestion_snapshot", 23),
        ("refresh_study_suggestion_snapshot", 23, "user-request"),
        (
            "trigger_study_suggestion_action",
            23,
            {
                "target_service": "flashcards",
                "target_type": "deck",
                "action_kind": "generate",
                "selected_topic_ids": ["topic-1"],
                "selected_topic_edits": None,
                "manual_topic_labels": None,
                "has_explicit_selection": True,
                "generator_version": "v1",
                "force_regenerate": False,
            },
        ),
    ]

    with pytest.raises(ValueError, match="Study suggestions are server-only"):
        await scope.get_study_suggestion_status(mode="local", anchor_type="quiz_attempt", anchor_id=17)


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
async def test_scope_service_routes_server_flashcard_management_actions():
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=FakeLocalStudyService(), server_service=server)

    deck = await scope.update_deck(
        mode="server",
        deck_id=7,
        name="Biology v2",
        description="Updated",
        review_prompt_side="back",
        expected_version=1,
    )
    card = await scope.get_flashcard(mode="server", card_id="card-server-1")
    reset = await scope.reset_flashcard_scheduling(
        mode="server",
        card_id="card-server-1",
        expected_version=2,
    )
    tagged = await scope.set_flashcard_tags(
        mode="server",
        card_id="card-server-1",
        tags=["science", "biology"],
    )
    tags = await scope.get_flashcard_tags(mode="server", card_id="card-server-1")
    analytics = await scope.get_flashcard_analytics_summary(
        mode="server",
        deck_id=7,
        workspace_id="ws-1",
        include_workspace_items=True,
    )

    assert deck["record_id"] == "server:study_deck:7"
    assert deck["name"] == "Biology v2"
    assert card["record_id"] == "server:study_flashcard:card-server-1"
    assert reset["record_id"] == "server:study_flashcard:card-server-1"
    assert tagged["record_id"] == "server:study_flashcard:card-server-1"
    assert tags == {"items": ["science", "biology"], "count": 2}
    assert analytics["source"] == "server"
    assert analytics["decks"][0]["deck_name"] == "Biology"
    assert server.calls[:6] == [
        ("update_deck", 7, "Biology v2", "Updated", None, "back", None, None, 1),
        ("get_flashcard", "card-server-1"),
        ("reset_flashcard_scheduling", "card-server-1", 2),
        ("get_flashcard", "card-server-1"),
        ("set_flashcard_tags", "card-server-1", ["science", "biology"]),
        ("get_flashcard", "card-server-1"),
    ]
    assert server.calls[-2:] == [
        ("get_flashcard_tags", "card-server-1"),
        ("get_flashcard_analytics_summary", 7, "ws-1", True),
    ]

    with pytest.raises(ValueError, match="server-only"):
        await scope.reset_flashcard_scheduling(mode="local", card_id="card-local-1", expected_version=1)


@pytest.mark.asyncio
async def test_scope_service_routes_server_flashcard_template_actions():
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=FakeLocalStudyService(), server_service=server)

    created = await scope.create_flashcard_template(
        mode="server",
        name="Cloze Drill",
        model_type="cloze",
        front_template="{{statement}}",
        notes_template="Focus: {{topic}}",
    )
    listed = await scope.list_flashcard_templates(mode="server", limit=25, offset=5)
    fetched = await scope.get_flashcard_template(mode="server", template_id=12)
    updated = await scope.update_flashcard_template(
        mode="server",
        template_id=12,
        notes_template="Updated focus: {{topic}}",
        expected_version=1,
    )
    deleted = await scope.delete_flashcard_template(mode="server", template_id=12, expected_version=2)

    assert created["record_id"] == "server:flashcard_template:12"
    assert listed["source"] == "server"
    assert listed["items"][0]["record_id"] == "server:flashcard_template:12"
    assert fetched["record_id"] == "server:flashcard_template:12"
    assert updated["notes_template"] == "Updated focus: {{topic}}"
    assert deleted is True
    assert server.calls == [
        ("create_flashcard_template", "Cloze Drill", "cloze", "{{statement}}", None, "Focus: {{topic}}", None, None),
        ("list_flashcard_templates", 25, 5),
        ("get_flashcard_template", 12),
        ("update_flashcard_template", 12, None, None, None, None, "Updated focus: {{topic}}", None, None, 1),
        ("delete_flashcard_template", 12, 2),
    ]

    with pytest.raises(ValueError, match="server-only"):
        await scope.list_flashcard_templates(mode="local")


@pytest.mark.asyncio
async def test_scope_service_routes_server_flashcard_bulk_and_tag_suggestion_actions():
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=FakeLocalStudyService(), server_service=server)

    created = await scope.create_flashcards_bulk(
        mode="server",
        cards=[
            {
                "deck_id": 7,
                "front": "Question",
                "back": "Answer",
                "tags": ["biology"],
            }
        ],
    )
    updated = await scope.update_flashcards_bulk(
        mode="server",
        cards=[
            {
                "uuid": "card-server-1",
                "tags": ["biology", "cell"],
                "expected_version": 1,
            }
        ],
    )
    suggestions = await scope.list_flashcard_tag_suggestions(mode="server", q="bio", limit=10)

    assert created["source"] == "server"
    assert created["items"][0]["record_id"] == "server:study_flashcard:card-server-1"
    assert updated["results"][0]["flashcard"]["record_id"] == "server:study_flashcard:card-server-1"
    assert suggestions["items"][0]["tag"] == "biology"
    assert server.calls == [
        (
            "create_flashcards_bulk",
            [{"deck_id": 7, "front": "Question", "back": "Answer", "tags": ["biology"]}],
        ),
        (
            "update_flashcards_bulk",
            [{"uuid": "card-server-1", "tags": ["biology", "cell"], "expected_version": 1}],
        ),
        ("list_flashcard_tag_suggestions", "bio", 10),
    ]

    with pytest.raises(ValueError, match="server-only"):
        await scope.list_flashcard_tag_suggestions(mode="local")


@pytest.mark.asyncio
async def test_scope_service_routes_server_flashcard_import_export_actions():
    server = FakeServerStudyService()
    scope = StudyScopeService(local_service=FakeLocalStudyService(), server_service=server)

    preview = await scope.preview_structured_qa_import(
        mode="server",
        content="Q: What powers cells?\nA: ATP",
        max_lines=25,
    )
    imported = await scope.import_flashcards_tsv(
        mode="server",
        content="Deck\tFront\tBack\tTags\tNotes\nBiology\tQ\tA\tbio\tN",
        has_header=True,
    )
    exported = await scope.export_flashcards(mode="server", deck_id=7, export_format="csv", include_header=True)

    assert preview["source"] == "server"
    assert preview["entity_kind"] == "flashcard_import_preview"
    assert imported["source"] == "server"
    assert imported["items"][0]["uuid"] == "card-server-1"
    assert exported.startswith(b"Deck\tFront")
    assert server.calls == [
        ("preview_structured_qa_import", "Q: What powers cells?\nA: ATP", 25, None, None),
        (
            "import_flashcards_tsv",
            "Deck\tFront\tBack\tTags\tNotes\nBiology\tQ\tA\tbio\tN",
            "\t",
            True,
            None,
            None,
            None,
        ),
        (
            "export_flashcards",
            {
                "deck_id": 7,
                "workspace_id": None,
                "include_workspace_items": False,
                "tag": None,
                "q": None,
                "export_format": "csv",
                "include_reverse": False,
                "delimiter": "\t",
                "include_header": True,
                "extended_header": False,
            },
        ),
    ]

    with pytest.raises(ValueError, match="server-only"):
        await scope.export_flashcards(mode="local")


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
