"""Local-only Prompt Draft Shelf storage and scope-service contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptDraftConflictError,
    PromptDraftShelfFullError,
    PromptScopeService,
)
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.Utils.input_validation import CONSOLE_DRAFT_MAX_LENGTH


@pytest.fixture
def draft_shelf(tmp_path):
    database = PromptsDatabase(tmp_path / "prompt-drafts.db", client_id="draft-shelf")
    local = LocalPromptService(database)
    scope = PromptScopeService(local_service=local, server_service=None)
    try:
        yield database, local, scope
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_scope_round_trip_preserves_exact_multiline_unicode_and_avoids_sync_log(
    draft_shelf,
):
    database, _local, scope = draft_shelf
    content = "  First line  \n\nPaste body: 100% _literal_\nこんにちは\n"

    created = await scope.create_prompt_draft(content=content)
    loaded = await scope.get_prompt_draft(draft_id=created["draft_id"])

    assert loaded["content"] == content
    assert loaded["version"] == 1
    assert loaded["display_name"] == "First line"
    assert loaded["backend"] == "draft_shelf"
    assert (
        database.get_connection().execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
async def test_scope_pages_newest_first_and_searches_content_literally(draft_shelf):
    _database, _local, scope = draft_shelf
    first = await scope.create_prompt_draft(content="first ordinary draft")
    percent = await scope.create_prompt_draft(content="keep 100% literal")
    underscore = await scope.create_prompt_draft(content="keep _literal_ marker")

    page_one = await scope.list_prompt_drafts(page=1, per_page=2)
    page_two = await scope.list_prompt_drafts(page=2, per_page=2)
    percent_search = await scope.list_prompt_drafts(query="%", page=1, per_page=10)
    underscore_search = await scope.list_prompt_drafts(
        query="_LITERAL_", page=1, per_page=10
    )

    assert [item["draft_id"] for item in page_one["items"]] == [
        underscore["draft_id"],
        percent["draft_id"],
    ]
    assert [item["draft_id"] for item in page_two["items"]] == [first["draft_id"]]
    assert page_one["total_items"] == 3
    assert page_one["total_pages"] == 2
    assert [item["draft_id"] for item in percent_search["items"]] == [
        percent["draft_id"]
    ]
    assert [item["draft_id"] for item in underscore_search["items"]] == [
        underscore["draft_id"]
    ]


@pytest.mark.asyncio
async def test_update_and_delete_refuse_stale_reviewed_versions(draft_shelf):
    _database, _local, scope = draft_shelf
    created = await scope.create_prompt_draft(content="before")

    updated = await scope.update_prompt_draft(
        draft_id=created["draft_id"],
        content="after",
        expected_version=1,
    )

    assert updated["content"] == "after"
    assert updated["version"] == 2
    with pytest.raises(PromptDraftConflictError, match="changed"):
        await scope.update_prompt_draft(
            draft_id=created["draft_id"],
            content="stale overwrite",
            expected_version=1,
        )
    with pytest.raises(PromptDraftConflictError, match="changed"):
        await scope.delete_prompt_draft(
            draft_id=created["draft_id"], expected_version=1
        )

    assert await scope.delete_prompt_draft(
        draft_id=created["draft_id"], expected_version=2
    )
    with pytest.raises(KeyError):
        await scope.get_prompt_draft(draft_id=created["draft_id"])


def test_atomic_capacity_refuses_one_of_two_concurrent_100th_creates(draft_shelf):
    database, local, _scope = draft_shelf
    for index in range(99):
        local.create_prompt_draft(f"seed {index}")

    barrier = Barrier(2)

    def create(content: str):
        barrier.wait()
        try:
            return local.create_prompt_draft(content)
        except PromptDraftShelfFullError as exc:
            return exc
        finally:
            database.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(create, ("candidate a", "candidate b")))

    successes = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], PromptDraftShelfFullError)
    assert local.list_prompt_drafts(page=1, per_page=100)["total_items"] == 100

    reviewed = local.list_prompt_drafts(page=1, per_page=1)["items"][0]
    assert local.delete_prompt_draft(
        draft_id=reviewed["draft_id"], expected_version=reviewed["version"]
    )
    replacement = local.create_prompt_draft("replacement after explicit delete")
    assert replacement["content"] == "replacement after explicit delete"
    assert local.list_prompt_drafts(page=1, per_page=100)["total_items"] == 100


def test_draft_page_count_and_rows_share_one_read_snapshot(draft_shelf):
    database, local, _scope = draft_shelf
    first = local.create_prompt_draft("first")
    local.create_prompt_draft("second")
    start_delete = Event()
    delete_finished = Event()
    writer_errors: list[BaseException] = []

    def delete_between_count_and_page() -> None:
        assert start_delete.wait(5)
        try:
            local.delete_prompt_draft(
                draft_id=first["draft_id"], expected_version=first["version"]
            )
        except Exception as exc:  # noqa: BLE001 - assert worker failures in parent
            writer_errors.append(exc)
        finally:
            database.close_connection()
            delete_finished.set()

    connection = database.get_connection()

    def pause_before_page(statement: str) -> None:
        normalized = " ".join(statement.split()).upper()
        if normalized.startswith("SELECT DRAFT_ID, CONTENT"):
            start_delete.set()
            assert delete_finished.wait(5)

    connection.set_trace_callback(pause_before_page)
    worker = Thread(target=delete_between_count_and_page)
    worker.start()
    try:
        result = local.list_prompt_drafts(page=1, per_page=10)
    finally:
        connection.set_trace_callback(None)
        start_delete.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert writer_errors == []
    assert result["total_items"] == 2
    assert len(result["items"]) == 2


@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize(
    "invalid_content",
    ["x" * (CONSOLE_DRAFT_MAX_LENGTH + 1), "prefix\x00suffix"],
)
def test_draft_writes_reject_unsafe_or_oversized_exact_text(
    draft_shelf, operation, invalid_content
):
    _database, local, _scope = draft_shelf
    created = local.create_prompt_draft("valid")

    with pytest.raises(ValueError, match="content"):
        if operation == "create":
            local.create_prompt_draft(invalid_content)
        else:
            local.update_prompt_draft(
                draft_id=created["draft_id"],
                content=invalid_content,
                expected_version=created["version"],
            )

@pytest.mark.asyncio
async def test_scope_rejects_server_mode_before_consulting_server(draft_shelf):
    _database, local, _scope = draft_shelf

    class ExplodingServer:
        def __getattr__(self, name):
            raise AssertionError(f"server draft seam was consulted: {name}")

    scope = PromptScopeService(local_service=local, server_service=ExplodingServer())

    with pytest.raises(ValueError, match="local-only"):
        await scope.list_prompt_drafts(mode="server")


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    [
        ("create_prompt_draft", {"content": "  \n"}, "content"),
        ("list_prompt_drafts", {"page": 0}, "page"),
        ("list_prompt_drafts", {"per_page": 0}, "per_page"),
        ("get_prompt_draft", {"draft_id": 0}, "draft_id"),
        (
            "update_prompt_draft",
            {"draft_id": 1, "content": "valid", "expected_version": 0},
            "expected_version",
        ),
    ],
)
def test_local_service_rejects_invalid_boundaries(draft_shelf, method, kwargs, message):
    _database, local, _scope = draft_shelf

    with pytest.raises((TypeError, ValueError), match=message):
        getattr(local, method)(**kwargs)


def test_draft_shelf_scope_actions_are_registered_as_local_only():
    expected = {
        "prompts.drafts.create.local",
        "prompts.drafts.list.local",
        "prompts.drafts.detail.local",
        "prompts.drafts.update.local",
        "prompts.drafts.delete.local",
    }

    assert expected <= CAPABILITY_REGISTRY.keys()
    assert not any(
        action_id.startswith("prompts.drafts.") and action_id.endswith(".server")
        for action_id in CAPABILITY_REGISTRY
    )


def test_prompts_database_v4_migrates_draft_shelf_schema_to_v5(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "prompt-drafts-v4.db"
    monkeypatch.setattr(PromptsDatabase, "_CURRENT_SCHEMA_VERSION", 4)
    legacy = PromptsDatabase(database_path, client_id="draft-shelf-v4")
    legacy.close_connection()

    monkeypatch.setattr(PromptsDatabase, "_CURRENT_SCHEMA_VERSION", 5)
    migrated = PromptsDatabase(database_path, client_id="draft-shelf-v5")
    try:
        connection = migrated.get_connection()
        version = connection.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()[0]
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(LocalPromptDrafts)")
        }
        assert version == 5
        assert columns == {
            "draft_id",
            "content",
            "created_at",
            "updated_at",
            "version",
        }
    finally:
        migrated.close_connection()
