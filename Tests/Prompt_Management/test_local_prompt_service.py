import pytest

from tldw_chatbook.Prompt_Management.local_prompt_service import LocalPromptService


class FakePromptDB:
    def __init__(self):
        self.updated = None
        self.sync_entries = [
            {
                "change_id": 1,
                "entity": "Prompts",
                "entity_uuid": "prompt-uuid",
                "operation": "create",
                "timestamp": "2026-04-20T00:00:00Z",
                "version": 1,
                "payload": {
                    "id": 1,
                    "uuid": "prompt-uuid",
                    "name": "Original",
                    "author": "Author",
                    "details": "v1 details",
                    "system_prompt": "sys v1",
                    "user_prompt": "user v1",
                    "prompt_format": "legacy",
                    "prompt_schema_version": None,
                    "prompt_definition": None,
                    "artifact_type": "prompt",
                    "version": 1,
                    "last_modified": "2026-04-20T00:00:00Z",
                },
            },
            {
                "change_id": 2,
                "entity": "Prompts",
                "entity_uuid": "other-prompt",
                "operation": "update",
                "timestamp": "2026-04-20T00:01:00Z",
                "version": 2,
                "payload": {"uuid": "other-prompt", "version": 2},
            },
            {
                "change_id": 3,
                "entity": "Prompts",
                "entity_uuid": "prompt-uuid",
                "operation": "update",
                "timestamp": "2026-04-20T00:02:00Z",
                "version": 2,
                "payload": {
                    "id": 1,
                    "uuid": "prompt-uuid",
                    "name": "Updated",
                    "author": "Author",
                    "details": "v2 details",
                    "system_prompt": "sys v2",
                    "user_prompt": "user v2",
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": '{"messages":[{"role":"user","content":"hi"}]}',
                    "artifact_type": "recipe",
                    "version": 2,
                    "last_modified": "2026-04-20T00:02:00Z",
                },
            },
        ]

    def get_sync_log_entries(self, since_change_id=0, limit=None):
        del since_change_id, limit
        return list(self.sync_entries)

    def update_prompt_by_id(self, prompt_id, update_data):
        self.updated = (prompt_id, update_data)
        return "prompt-uuid", "restored"


class FakePromptInterop:
    def __init__(self):
        self.db = FakePromptDB()
        self.prompt = {
            "id": 1,
            "uuid": "prompt-uuid",
            "name": "Updated",
            "prompt_format": "structured",
            "version": 2,
        }

    def fetch_prompt_details(self, prompt_identifier, *, include_deleted=True):
        del prompt_identifier, include_deleted
        return dict(self.prompt)

    def get_db_instance(self):
        return self.db


@pytest.mark.asyncio
async def test_local_prompt_service_lists_prompt_versions_from_sync_log_snapshots():
    service = LocalPromptService(interop_module=FakePromptInterop())

    versions = await service.list_prompt_versions("prompt-uuid")

    assert [version["version"] for version in versions] == [2, 1]
    assert versions[0]["prompt_uuid"] == "prompt-uuid"
    assert versions[0]["operation"] == "update"
    assert (
        versions[0]["prompt_definition"]
        == '{"messages":[{"role":"user","content":"hi"}]}'
    )
    assert versions[0]["artifact_type"] == "recipe"


@pytest.mark.asyncio
async def test_local_prompt_service_restores_prompt_version_from_sync_log_snapshot():
    interop = FakePromptInterop()
    service = LocalPromptService(interop_module=interop)

    restored = await service.restore_prompt_version("prompt-uuid", 1)

    assert restored["uuid"] == "prompt-uuid"
    assert interop.db.updated == (
        1,
        {
            "name": "Original",
            "author": "Author",
            "details": "v1 details",
            "system_prompt": "sys v1",
            "user_prompt": "user v1",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
            "artifact_type": "prompt",
        },
    )


@pytest.mark.asyncio
async def test_local_prompt_service_rejects_missing_prompt_version_snapshot():
    service = LocalPromptService(interop_module=FakePromptInterop())

    with pytest.raises(ValueError, match="Local prompt version 99 was not found"):
        await service.restore_prompt_version("prompt-uuid", 99)


# ---------------------------------------------------------------------------
# Library read seams (task-1337 plan Task 3)
# ---------------------------------------------------------------------------


class FakeLibraryPromptInterop:
    def __init__(self):
        self.calls = []
        self.page_payload = {"items": [{"id": 1, "uuid": "u-1", "name": "One"}], "total": 7}
        self.search_payload = {
            "items": [
                {
                    "id": 2,
                    "uuid": "u-2",
                    "name": "Two",
                    "matched_fields": ["name"],
                    "matched_keywords": [],
                }
            ],
            "total": 3,
        }
        self.overview_payload = {
            "uuid": "u-1",
            "name": "One",
            "version": 4,
            "sections": {"system_prompt": {"total_chars": 42, "preview": "sys"}},
        }
        self.section_payload = {
            "uuid": "u-1",
            "section": "system_prompt",
            "version": 4,
            "total_chars": 42,
            "start": 10,
            "returned_chars": 20,
            "has_more": True,
            "text": "segment",
        }

    def list_library_prompts_page(self, *, limit, offset):
        self.calls.append(("list", limit, offset))
        return dict(self.page_payload)

    def search_library_prompts_page(self, *, query, limit, offset):
        self.calls.append(("search", query, limit, offset))
        return dict(self.search_payload)

    def get_library_prompt_overview(self, prompt_uuid):
        self.calls.append(("overview", prompt_uuid))
        return dict(self.overview_payload)

    def get_library_prompt_section(self, prompt_uuid, *, section, start, max_chars):
        self.calls.append(("section", prompt_uuid, section, start, max_chars))
        return dict(self.section_payload)


@pytest.mark.asyncio
async def test_local_prompt_service_lists_library_page_and_echoes_pagination():
    interop = FakeLibraryPromptInterop()
    service = LocalPromptService(interop_module=interop)

    payload = await service.list_library_prompts(limit=5, offset=10)

    assert interop.calls == [("list", 5, 10)]
    assert payload["total"] == 7
    assert payload["offset"] == 10
    assert payload["limit"] == 5
    assert payload["items"] == interop.page_payload["items"]


@pytest.mark.asyncio
async def test_local_prompt_service_forwards_search_totals_and_match_fields():
    interop = FakeLibraryPromptInterop()
    service = LocalPromptService(interop_module=interop)

    payload = await service.search_library_prompts("quarterly", limit=3, offset=6)

    assert interop.calls == [("search", "quarterly", 3, 6)]
    assert payload["total"] == 3
    assert payload["offset"] == 6
    assert payload["items"][0]["matched_fields"] == ["name"]
    assert "matched_keywords" in payload["items"][0]


@pytest.mark.asyncio
async def test_local_prompt_service_forwards_overview_and_section_window():
    interop = FakeLibraryPromptInterop()
    service = LocalPromptService(interop_module=interop)

    overview = await service.get_library_prompt_overview("u-1")
    section = await service.get_library_prompt_section(
        "u-1", "system_prompt", start=10, max_chars=20
    )

    assert ("overview", "u-1") in interop.calls
    assert ("section", "u-1", "system_prompt", 10, 20) in interop.calls
    assert overview["sections"]["system_prompt"]["total_chars"] == 42
    assert section["text"] == "segment"
    assert section["has_more"] is True
