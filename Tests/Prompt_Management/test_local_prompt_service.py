import pytest

from tldw_chatbook.Prompt_Management.local_prompt_service import LocalPromptService
from tldw_chatbook.Prompt_Management.prompt_batch_models import (
    PromptBatchDeleteResult,
    PromptBatchRestoreResult,
    PromptBatchTarget,
    PromptDeleteReceiptEntry,
    PromptRestoreResultEntry,
)
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService as ScopeLocalPromptService,
)


class FakePromptDB:
    def __init__(self):
        self.updated = None
        self.history_calls = []
        self.history_page = {
            "total_count": 2,
            "has_more": False,
            "next_before_change_id": None,
            "predecessor": None,
            "items": [
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
            ],
        }

    def get_prompt_history_entries(self, entity_uuid, page_size, before_change_id=None):
        self.history_calls.append((entity_uuid, page_size, before_change_id))
        return self.history_page

    def restore_prompt_history_entry(
        self,
        prompt_uuid,
        change_id,
        version,
        expected_version,
        snapshot_validator,
    ):
        del snapshot_validator
        self.updated = (prompt_uuid, change_id, version, expected_version)
        return {
            "outcome": "restored",
            "snapshot_unavailable": False,
            "no_change": False,
            "source_version": version,
            "current_version": expected_version,
            "new_version": expected_version + 1,
            "retained_current_keywords": False,
        }

    def restore_deleted_prompt(self, prompt_id, *, expected_version):
        self.updated = ("deleted", prompt_id, expected_version)
        return {
            "id": 1,
            "uuid": "prompt-uuid",
            "name": "Updated",
            "version": expected_version + 1,
            "deleted": 0,
        }


class FakeAtomicPromptDB:
    def __init__(self):
        self.calls = []
        self.deleted_result = PromptBatchDeleteResult(
            entries=(PromptDeleteReceiptEntry(7, "Seven", "prompt", 4),)
        )
        self.restored_result = PromptBatchRestoreResult(
            entries=(PromptRestoreResultEntry(7, 5),)
        )

    def soft_delete_prompts(self, targets):
        self.calls.append(("delete", targets))
        return self.deleted_result

    def restore_deleted_prompts(self, targets):
        self.calls.append(("restore", targets))
        return self.restored_result


class PromptBatchTargetSubclass(PromptBatchTarget):
    pass


def _forged_prompt_batch_target(local_id, expected_version) -> PromptBatchTarget:
    target = object.__new__(PromptBatchTarget)
    object.__setattr__(target, "local_id", local_id)
    object.__setattr__(target, "expected_version", expected_version)
    return target


INVALID_LOCAL_PROMPT_BATCH_TARGETS = [
    ([], TypeError, "targets"),
    ((), ValueError, "non-empty"),
    ((object(),), TypeError, "targets"),
    ((PromptBatchTargetSubclass(7, 3),), TypeError, "targets"),
    (
        (PromptBatchTarget(7, 3), PromptBatchTarget(7, 4)),
        ValueError,
        "unique local IDs",
    ),
    ((_forged_prompt_batch_target(True, 1),), ValueError, "local_id"),
    ((_forged_prompt_batch_target(0, 1),), ValueError, "local_id"),
    ((_forged_prompt_batch_target(-1, 1),), ValueError, "local_id"),
    ((_forged_prompt_batch_target(2**63, 1),), ValueError, "local_id"),
    ((_forged_prompt_batch_target(1, False),), ValueError, "expected_version"),
    ((_forged_prompt_batch_target(1, 0),), ValueError, "expected_version"),
    ((_forged_prompt_batch_target(1, -1),), ValueError, "expected_version"),
    ((_forged_prompt_batch_target(1, 2**63),), ValueError, "expected_version"),
]


@pytest.mark.parametrize("method_name", ["delete_prompts", "restore_deleted_prompts"])
@pytest.mark.parametrize(
    ("targets", "error_type", "message"), INVALID_LOCAL_PROMPT_BATCH_TARGETS
)
def test_local_prompt_batch_methods_validate_before_database_call(
    method_name, targets, error_type, message
):
    database = FakeAtomicPromptDB()
    service = ScopeLocalPromptService(database)

    with pytest.raises(error_type, match=message):
        getattr(service, method_name)(targets=targets)

    assert database.calls == []


@pytest.mark.parametrize(
    ("method_name", "operation", "result_attribute"),
    [
        ("delete_prompts", "delete", "deleted_result"),
        ("restore_deleted_prompts", "restore", "restored_result"),
    ],
)
def test_local_prompt_batch_methods_pass_exact_canonical_targets_to_database(
    method_name, operation, result_attribute
):
    database = FakeAtomicPromptDB()
    service = ScopeLocalPromptService(database)
    first = PromptBatchTarget(7, 3)
    second = PromptBatchTarget(9, 2)
    canonical = (first, second)

    result = getattr(service, method_name)(targets=canonical)

    assert result is getattr(database, result_attribute)
    assert database.calls[0][0] == operation
    assert database.calls[0][1] is canonical

    database.calls.clear()
    result = getattr(service, method_name)(targets=(second, first))

    assert result is getattr(database, result_attribute)
    assert database.calls == [(operation, canonical)]


def test_local_prompt_batch_methods_are_sync_keyword_only_typed_pass_throughs():
    import inspect
    from typing import get_type_hints

    database = FakeAtomicPromptDB()
    service = ScopeLocalPromptService(database)
    targets = (PromptBatchTarget(7, 3),)

    assert not inspect.iscoroutinefunction(ScopeLocalPromptService.delete_prompts)
    assert not inspect.iscoroutinefunction(
        ScopeLocalPromptService.restore_deleted_prompts
    )
    for method_name, result_type in (
        ("delete_prompts", PromptBatchDeleteResult),
        ("restore_deleted_prompts", PromptBatchRestoreResult),
    ):
        method = getattr(ScopeLocalPromptService, method_name)
        signature = inspect.signature(method)
        assert signature.parameters["targets"].kind is inspect.Parameter.KEYWORD_ONLY
        hints = get_type_hints(method)
        assert hints["targets"] == tuple[PromptBatchTarget, ...]
        assert hints["return"] is result_type

    with pytest.raises(TypeError):
        service.delete_prompts(targets)  # type: ignore[misc]
    with pytest.raises(TypeError):
        service.restore_deleted_prompts(targets)  # type: ignore[misc]

    deleted = service.delete_prompts(targets=targets)
    restored = service.restore_deleted_prompts(targets=targets)

    assert deleted is database.deleted_result
    assert restored is database.restored_result
    assert database.calls[0][0] == "delete"
    assert database.calls[0][1] is targets
    assert database.calls[1][0] == "restore"
    assert database.calls[1][1] is targets


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

    def soft_delete_prompt(self, prompt_identifier, *, expected_version=None):
        self.db.updated = ("delete", prompt_identifier, expected_version)
        return True


@pytest.mark.asyncio
async def test_local_prompt_service_lists_versions_with_one_bounded_database_page_call():
    interop = FakePromptInterop()
    service = LocalPromptService(interop_module=interop)

    page = await service.list_prompt_versions(
        "prompt-uuid", page_size=13, before_change_id=99
    )

    assert interop.db.history_calls == [("prompt-uuid", 13, 99)]
    versions = page["items"]
    assert [version["version"] for version in versions] == [1, 2]
    assert versions[0]["entity_uuid"] == "prompt-uuid"
    assert versions[1]["operation"] == "update"
    assert (
        versions[1]["payload"]["prompt_definition"]
        == '{"messages":[{"role":"user","content":"hi"}]}'
    )
    assert versions[1]["payload"]["artifact_type"] == "recipe"


@pytest.mark.asyncio
async def test_local_prompt_service_restores_exact_retained_version_conditionally():
    interop = FakePromptInterop()
    service = LocalPromptService(interop_module=interop)

    restored = await service.restore_prompt_version(
        "prompt-uuid", change_id=1, version=1, expected_version=2
    )

    assert restored["outcome"] == "restored"
    assert interop.db.updated == ("prompt-uuid", 1, 1, 2)


@pytest.mark.asyncio
async def test_local_prompt_service_restores_exact_deleted_version_conditionally():
    interop = FakePromptInterop()
    service = LocalPromptService(interop_module=interop)

    restored = await service.restore_deleted_prompt(1, expected_version=2)

    assert restored["version"] == 3
    assert restored["deleted"] == 0
    assert interop.db.updated == ("deleted", 1, 2)


@pytest.mark.asyncio
async def test_local_prompt_service_deletes_exact_current_version_conditionally():
    interop = FakePromptInterop()
    service = LocalPromptService(interop_module=interop)

    deleted = await service.delete_prompt(1, expected_version=2)

    assert deleted is True
    assert interop.db.updated == ("delete", 1, 2)


@pytest.mark.asyncio
async def test_local_prompt_service_source_contains_no_whole_sync_log_scan():
    import inspect

    source = inspect.getsource(LocalPromptService)

    assert "get_sync_log_entries" not in source


# ---------------------------------------------------------------------------
# Library read seams (task-1337 plan Task 3)
# ---------------------------------------------------------------------------


class FakeLibraryPromptInterop:
    def __init__(self):
        self.calls = []
        self.page_payload = {
            "items": [{"id": 1, "uuid": "u-1", "name": "One"}],
            "total": 7,
        }
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
