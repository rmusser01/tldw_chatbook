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
    assert versions[0]["prompt_definition"] == '{"messages":[{"role":"user","content":"hi"}]}'


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
        },
    )


@pytest.mark.asyncio
async def test_local_prompt_service_rejects_missing_prompt_version_snapshot():
    service = LocalPromptService(interop_module=FakePromptInterop())

    with pytest.raises(ValueError, match="Local prompt version 99 was not found"):
        await service.restore_prompt_version("prompt-uuid", 99)
