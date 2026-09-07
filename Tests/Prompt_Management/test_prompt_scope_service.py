import inspect
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import pytest

import tldw_chatbook.Prompt_Management.prompt_scope_service as prompt_scope_module
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_batch_models import (
    PromptBatchDeleteResult,
    PromptBatchRestoreResult,
    PromptBatchTarget,
    PromptDeleteReceiptEntry,
    PromptRestoreResultEntry,
)
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptBackend,
    PromptScopeService,
    ServerPromptService,
    build_prompt_scope_service,
)
from tldw_chatbook.Prompt_Management.prompt_normalizers import (
    normalize_prompt_list,
    normalize_prompt_record,
)
from tldw_chatbook.Prompt_Management.prompt_restore_errors import (
    PromptRestoreError,
    PromptRestoreErrorCode,
)
from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
    CANONICAL_JSON_UTF8_V1,
    PromptCapabilityError,
    canonical_json_utf8_size,
    local_prompt_capabilities,
)
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import (
    PaginatedPromptsResponse,
    PromptBriefResponse,
    PromptCollectionCreateResponse,
    PromptCollectionListResponse,
    PromptCollectionResponse,
    PromptResponse,
    PromptVersionResponse,
)


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.actions = []
        self.denied_reason = denied_reason

    @classmethod
    def deny(cls, reason="blocked"):
        return cls(denied_reason=reason)

    def require_allowed(self, *, action_id):
        self.actions.append(action_id)
        if self.denied_reason:
            raise PermissionError(self.denied_reason)


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not build a client")


def modern_prompt_health(
    *,
    structured_kinds=None,
    search=True,
    conditional_update=False,
    measurement=None,
    compiled_lane_limit=12_000,
    definition_limit=200_000,
    request_limit=400_000,
):
    return {
        "status": "healthy",
        "capabilities": {
            "structured_kinds": structured_kinds
            if structured_kinds is not None
            else [
                {"schema_version": 1, "kind": "multi_message"},
                {"schema_version": 2, "kind": "block_prompt"},
                {"schema_version": 2, "kind": "block_recipe"},
            ],
            "artifact_types": ["prompt", "recipe"],
            "search": search,
            "conditional_update": conditional_update,
            "size_limits": {
                "compiled_lane_characters": compiled_lane_limit,
                "definition_utf8_bytes": definition_limit,
                "request_utf8_bytes": request_limit,
                "json_byte_measurement": (
                    measurement
                    if measurement is not None
                    else {
                        "name": "canonical_json_utf8_v1",
                        "encoding": "utf-8",
                        "ensure_ascii": False,
                        "sort_keys": True,
                        "separators": [",", ":"],
                    }
                ),
            },
        },
    }


def block_definition(*, kind="block_prompt", content="hello"):
    return {
        "schema_version": 2,
        "kind": kind,
        "lanes": [
            {"id": "system", "blocks": []},
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "user-1",
                        "title": "User",
                        "syntax": "freeform",
                        "content": content,
                    }
                ],
            },
        ],
    }


def test_prompt_scope_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(prompt_scope_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source
    assert "build_tldw_api_client_from_config" not in source


class FakeLocalPromptService:
    def __init__(self):
        self.calls = []
        self.prompt = {
            "id": 7,
            "uuid": "local-uuid-7",
            "name": "Local Prompt",
            "author": "Local Writer",
            "details": "Local details",
            "system_prompt": "Local system",
            "user_prompt": "Local user",
            "keywords": ["draft"],
            "prompt_format": "legacy",
            "artifact_type": "prompt",
            "version": 3,
            "deleted": False,
        }

    def list_prompts(self, *, page=1, per_page=10, include_deleted=False, **_kwargs):
        self.calls.append(("list_prompts", page, per_page, include_deleted))
        return {
            "items": [self.prompt],
            "total_pages": 1,
            "current_page": page,
            "total_items": 1,
        }

    def get_prompt(self, prompt_identifier, *, include_deleted=False):
        self.calls.append(("get_prompt", prompt_identifier, include_deleted))
        return self.prompt

    def create_prompt(self, payload):
        self.calls.append(("create_prompt", payload))
        return {**self.prompt, **payload, "id": 8, "uuid": "local-uuid-8", "version": 1}

    def update_prompt(self, prompt_identifier, payload):
        self.calls.append(("update_prompt", prompt_identifier, payload))
        return {**self.prompt, **payload}

    def delete_prompt(self, prompt_identifier, *, expected_version=None):
        self.calls.append(("delete_prompt", prompt_identifier, expected_version))
        return True

    def restore_deleted_prompt(self, prompt_identifier, *, expected_version):
        self.calls.append(
            ("restore_deleted_prompt", prompt_identifier, expected_version)
        )
        return {
            **self.prompt,
            "version": expected_version + 1,
            "deleted": 0,
        }

    def count_prompt_versions(self, prompt_identifier):
        self.calls.append(("count_prompt_versions", prompt_identifier))
        return 6

    def list_prompt_versions(
        self, prompt_identifier, *, page_size=25, before_change_id=None
    ):
        self.calls.append(
            ("list_prompt_versions", prompt_identifier, page_size, before_change_id)
        )
        return {
            "items": [
                {
                    "change_id": 42,
                    "entity": "Prompts",
                    "entity_uuid": "local-uuid-7",
                    "operation": "update",
                    "timestamp": "2026-08-08T00:00:00.000Z",
                    "version": 3,
                    "payload": {
                        "name": "Local Prompt",
                        "system_prompt": "Local system",
                        "user_prompt": "Local user",
                    },
                }
            ],
            "predecessor": None,
            "total_count": 1,
            "has_more": False,
            "next_before_change_id": None,
        }

    def restore_prompt_version(
        self, prompt_identifier, *, change_id, version, expected_version
    ):
        self.calls.append(
            (
                "restore_prompt_version",
                prompt_identifier,
                change_id,
                version,
                expected_version,
            )
        )
        return {
            "outcome": "restored",
            "snapshot_unavailable": False,
            "no_change": False,
            "source_version": version,
            "current_version": expected_version,
            "new_version": expected_version + 1,
            "retained_current_keywords": False,
        }

    def create_prompt_collection(self, payload):
        self.calls.append(("create_prompt_collection", payload))
        return {"collection_id": 3}

    def list_prompt_collections(self, *, query="", limit=200, offset=0):
        self.calls.append(("list_prompt_collections", query, limit, offset))
        return {
            "collections": [
                {
                    "collection_id": 3,
                    "name": "Local Collection",
                    "description": "Offline prompts",
                    "prompt_ids": [7],
                }
            ],
            "limit": limit,
            "offset": offset,
            "total": 1,
        }

    def get_prompt_collection(self, collection_id):
        self.calls.append(("get_prompt_collection", collection_id))
        return {
            "collection_id": collection_id,
            "name": "Local Collection",
            "description": "Offline prompts",
            "prompt_ids": [7],
        }

    def update_prompt_collection(self, collection_id, payload):
        self.calls.append(("update_prompt_collection", collection_id, payload))
        return {
            "collection_id": collection_id,
            "name": payload.get("name") or "Local Collection",
            "description": payload.get("description"),
            "prompt_ids": payload.get("prompt_ids") or [],
        }

    def list_prompt_collection_memberships(self, prompt_id):
        self.calls.append(("list_prompt_collection_memberships", prompt_id))
        return {
            "prompt_id": prompt_id,
            "collection_ids": (3, 8),
            "changed": False,
            "saved": True,
        }

    def replace_prompt_collection_memberships(self, prompt_id, collection_ids):
        self.calls.append(
            ("replace_prompt_collection_memberships", prompt_id, collection_ids)
        )
        return {
            "prompt_id": prompt_id,
            "collection_ids": tuple(sorted(collection_ids)),
            "changed": True,
            "saved": True,
        }


class FakeServerPromptService:
    def __init__(self, *, health=None, search_items=None):
        self.calls = []
        self.health = health if health is not None else modern_prompt_health()
        self.search_items = list(search_items or [])
        self.prompt = PromptResponse(
            id=9,
            uuid="server-uuid-9",
            name="Server Prompt",
            author="Server Writer",
            details="Server details",
            system_prompt="Server system",
            user_prompt="Server user",
            keywords=["remote"],
            prompt_format="structured",
            prompt_schema_version=1,
            prompt_definition={"schema_version": 1, "messages": []},
            version=5,
            usage_count=11,
            deleted=False,
        )

    async def list_prompts(
        self,
        *,
        page=1,
        per_page=10,
        include_deleted=False,
        sort_by="last_modified",
        sort_order="desc",
    ):
        self.calls.append(
            ("list_prompts", page, per_page, include_deleted, sort_by, sort_order)
        )
        return PaginatedPromptsResponse(
            items=[
                PromptBriefResponse(
                    id=9,
                    uuid="server-uuid-9",
                    name="Server Prompt",
                    author="Server Writer",
                    usage_count=11,
                )
            ],
            total_pages=2,
            current_page=page,
            total_items=12,
        )

    async def get_prompts_health(self):
        self.calls.append(("get_prompts_health",))
        return self.health

    async def search_prompts(self, **kwargs):
        self.calls.append(("search_prompts", kwargs))
        return {
            "items": self.search_items,
            "total_matches": len(self.search_items),
            "page": kwargs.get("page", 1),
            "per_page": kwargs.get("results_per_page", 20),
        }

    async def get_prompt(self, prompt_identifier, *, include_deleted=False):
        self.calls.append(("get_prompt", prompt_identifier, include_deleted))
        return self.prompt

    async def create_prompt(self, payload):
        self.calls.append(("create_prompt", payload))
        return self.prompt.model_copy(update={"name": payload["name"]})

    async def update_prompt(self, prompt_identifier, payload):
        self.calls.append(("update_prompt", prompt_identifier, payload))
        return self.prompt.model_copy(update=payload)

    async def delete_prompt(self, prompt_identifier):
        self.calls.append(("delete_prompt", prompt_identifier))
        return {}

    async def record_prompt_usage(self, prompt_identifier):
        self.calls.append(("record_prompt_usage", prompt_identifier))
        return self.prompt

    async def list_prompt_versions(self, prompt_identifier):
        self.calls.append(("list_prompt_versions", prompt_identifier))
        return [
            PromptVersionResponse(
                version=5,
                created_at="2026-04-22T00:00:00Z",
                comment="current",
                name="Server Prompt",
            )
        ]

    async def restore_prompt_version(self, prompt_identifier, version):
        self.calls.append(("restore_prompt_version", prompt_identifier, version))
        return self.prompt.model_copy(update={"version": version})

    async def create_prompt_collection(self, payload):
        self.calls.append(("create_prompt_collection", payload))
        return PromptCollectionCreateResponse(collection_id=7)

    async def list_prompt_collections(self, *, limit=200, offset=0):
        self.calls.append(("list_prompt_collections", limit, offset))
        return PromptCollectionListResponse(
            collections=[
                PromptCollectionResponse(
                    collection_id=7,
                    name="Server Collection",
                    description="Remote prompts",
                    prompt_ids=[9],
                )
            ]
        )

    async def get_prompt_collection(self, collection_id):
        self.calls.append(("get_prompt_collection", collection_id))
        return PromptCollectionResponse(
            collection_id=collection_id,
            name="Server Collection",
            description="Remote prompts",
            prompt_ids=[9],
        )

    async def update_prompt_collection(self, collection_id, payload):
        self.calls.append(("update_prompt_collection", collection_id, payload))
        return PromptCollectionResponse(
            collection_id=collection_id,
            name=payload.get("name") or "Server Collection",
            description=payload.get("description"),
            prompt_ids=payload.get("prompt_ids") or [],
        )


class RecordingPromptBrowseDatabase:
    """Signature-real adapter target for the Task 3 database seam."""

    def __init__(self):
        self.calls = []

    def browse_prompts(
        self,
        *,
        query="",
        collection_id=None,
        sort_by="last_modified",
        sort_order="desc",
        page=1,
        page_size=50,
    ):
        self.calls.append(
            {
                "query": query,
                "collection_id": collection_id,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "page": page,
                "page_size": page_size,
            }
        )
        return (
            [
                {
                    "id": 7,
                    "uuid": "local-uuid-7",
                    "name": "Local Prompt",
                    "artifact_type": "prompt",
                }
            ],
            3,
            3,
            201,
        )


def test_browse_prompt_local_adapter_exposes_narrow_keyword_only_signature():
    parameters = inspect.signature(LocalPromptService.browse_prompts).parameters

    assert tuple(parameters) == (
        "self",
        "query",
        "collection_id",
        "sort_by",
        "sort_order",
        "page",
        "page_size",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in parameters.items()
        if name != "self"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local", PromptBackend.LOCAL])
async def test_browse_prompt_routes_normalized_local_scope_through_real_adapter(mode):
    database = RecordingPromptBrowseDatabase()
    policy = FakePolicyEnforcer()
    server = FakeServerPromptService()
    service = PromptScopeService(
        local_service=LocalPromptService(database),
        server_service=server,
        policy_enforcer=policy,
    )

    result = await service.browse_prompts(
        mode=mode,
        query="  alpha beta \n",
        collection_id=8,
        sort_by=" NAME ",
        sort_order=" ASC ",
        page=4,
        page_size=999,
    )

    assert database.calls == [
        {
            "query": "alpha beta",
            "collection_id": 8,
            "sort_by": "name",
            "sort_order": "asc",
            "page": 4,
            "page_size": 100,
        }
    ]
    assert result["items"][0]["id"] == "local:prompt:local-uuid-7"
    assert result["total_items"] == 201
    assert result["total_pages"] == 3
    assert result["current_page"] == 3
    assert result["page"] == 3
    assert result["per_page"] == 100
    assert policy.actions == ["prompts.list.local"]
    assert server.calls == []


@pytest.mark.asyncio
async def test_browse_prompt_omitted_page_size_keeps_generic_fifty_row_default():
    database = RecordingPromptBrowseDatabase()
    service = PromptScopeService(
        local_service=LocalPromptService(database),
        server_service=FakeServerPromptService(),
        policy_enforcer=FakePolicyEnforcer(),
    )

    result = await service.browse_prompts()

    assert database.calls[0]["page_size"] == 50
    assert result["per_page"] == 50


def test_normalize_prompt_list_preserves_divergent_page_aliases():
    normalized = normalize_prompt_list(
        {
            "items": [],
            "total_items": 0,
            "total_pages": 0,
            "current_page": 2,
            "page": 1,
            "per_page": 20,
        },
        backend="local",
        page=9,
        per_page=50,
    )

    assert normalized["current_page"] == 2
    assert normalized["page"] == 1
    assert normalized["per_page"] == 20


def test_normalize_prompt_list_defaults_page_metadata_only_when_keys_are_absent():
    normalized = normalize_prompt_list(
        {"items": [], "total_items": 0, "total_pages": 0},
        backend="local",
        page=4,
        per_page=20,
    )

    assert normalized["current_page"] == 4
    assert normalized["page"] == 4
    assert normalized["per_page"] == 20


@pytest.mark.parametrize(
    "field", ["total_items", "total_pages", "current_page", "page", "per_page"]
)
@pytest.mark.parametrize("value", [True, None, 1.5])
def test_normalize_prompt_list_rejects_present_non_integer_envelope_metadata(
    field, value
):
    payload = {
        "items": [],
        "total_items": 0,
        "total_pages": 0,
        "current_page": 1,
        "page": 1,
        "per_page": 20,
    }
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        normalize_prompt_list(payload, backend="local", page=7, per_page=50)


@pytest.mark.asyncio
async def test_browse_prompt_policy_denial_stops_before_local_adapter_call():
    database = RecordingPromptBrowseDatabase()
    policy = FakePolicyEnforcer.deny()
    service = PromptScopeService(
        LocalPromptService(database), FakeServerPromptService(), policy
    )

    with pytest.raises(PermissionError, match="blocked"):
        await service.browse_prompts()

    assert policy.actions == ["prompts.list.local"]
    assert database.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["server", "all", "mixed"])
async def test_browse_prompt_rejects_non_local_library_modes_without_routing(mode):
    database = RecordingPromptBrowseDatabase()
    server = FakeServerPromptService()
    service = PromptScopeService(LocalPromptService(database), server)

    with pytest.raises(ValueError, match="local|backend"):
        await service.browse_prompts(mode=mode)

    assert database.calls == []
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sort_by": "name; DROP TABLE Prompts"}, "sort_by"),
        ({"sort_order": "sideways"}, "sort_order"),
        ({"collection_id": True}, "collection_id"),
        ({"page": 0}, "page"),
        ({"page_size": 0}, "page_size"),
        ({"query": None}, "query"),
    ],
)
async def test_browse_prompt_rejects_invalid_scope_before_adapter_call(kwargs, message):
    database = RecordingPromptBrowseDatabase()
    policy = FakePolicyEnforcer()
    service = PromptScopeService(
        LocalPromptService(database), FakeServerPromptService(), policy
    )

    with pytest.raises((TypeError, ValueError), match=message):
        await service.browse_prompts(**kwargs)

    assert database.calls == []
    assert policy.actions == []


@pytest.mark.asyncio
async def test_prompt_scope_lists_local_and_server_prompts_with_stable_ids():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(
        local_service=local, server_service=server, policy_enforcer=policy
    )

    local_result = await service.list_prompts(
        mode=PromptBackend.LOCAL, page=1, per_page=10
    )
    server_result = await service.list_prompts(
        mode=PromptBackend.SERVER,
        page=2,
        per_page=25,
        include_deleted=True,
        sort_by="name",
        sort_order="asc",
    )

    assert local_result["items"][0]["id"] == "local:prompt:local-uuid-7"
    assert local_result["items"][0]["backend"] == "local"
    assert local_result["items"][0]["artifact_type"] == "prompt"
    assert server_result["items"][0]["id"] == "server:prompt:server-uuid-9"
    assert server_result["items"][0]["backend"] == "server"
    assert server_result["current_page"] == 2
    assert policy.actions == ["prompts.list.local", "prompts.list.server"]


@pytest.mark.asyncio
async def test_prompt_scope_count_prompt_versions_uses_real_index_only_count(
    monkeypatch,
):
    database = PromptsDatabase(":memory:", client_id="scope-history-count")
    try:
        prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Retained count", author=None, details="v1"
        )
        database.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
        assert database.soft_delete_prompt(prompt_id) is True
        database.add_prompt(name="Unrelated", author=None, details="other")

        count_calls = []
        real_count = database.get_prompt_history_count

        def count_only(entity_uuid):
            count_calls.append(entity_uuid)
            return real_count(entity_uuid)

        monkeypatch.setattr(database, "get_prompt_history_count", count_only)
        monkeypatch.setattr(
            database,
            "get_prompt_history_entries",
            lambda *_args, **_kwargs: pytest.fail("history page must not be read"),
        )
        monkeypatch.setattr(
            database,
            "_decode_prompt_history_row",
            lambda *_args, **_kwargs: pytest.fail(
                "history payload must not be decoded"
            ),
        )
        service = PromptScopeService(
            local_service=LocalPromptService(database),
            server_service=FakeServerPromptService(),
        )

        count = await service.count_prompt_versions(
            mode="local", prompt_identifier=prompt_uuid
        )

        assert count == 2
        assert count_calls == [prompt_uuid]
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_prompt_scope_count_prompt_versions_routes_policy_and_local_adapter():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    service = PromptScopeService(
        local_service=local,
        server_service=FakeServerPromptService(),
        policy_enforcer=policy,
    )

    count = await service.count_prompt_versions(
        mode="local", prompt_identifier="local-uuid-7"
    )

    assert count == 6
    assert local.calls == [("count_prompt_versions", "local-uuid-7")]
    assert policy.actions == ["prompts.versions.list.local"]


@pytest.mark.asyncio
async def test_prompt_scope_count_prompt_versions_fails_truthfully_for_server():
    policy = FakePolicyEnforcer()
    server = FakeServerPromptService()
    service = PromptScopeService(
        local_service=FakeLocalPromptService(),
        server_service=server,
        policy_enforcer=policy,
    )

    with pytest.raises(PromptCapabilityError, match="server.*retained history count"):
        await service.count_prompt_versions(
            mode="server", prompt_identifier="server-uuid-9"
        )

    assert server.calls == []
    assert policy.actions == ["prompts.versions.list.server"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_count", [True, -1, "2", 1.5, None])
async def test_prompt_scope_count_prompt_versions_rejects_invalid_local_counts(
    invalid_count,
):
    local = FakeLocalPromptService()
    local.count_prompt_versions = lambda _identifier: invalid_count
    service = PromptScopeService(
        local_service=local,
        server_service=FakeServerPromptService(),
    )

    with pytest.raises(ValueError, match="non-negative integer"):
        await service.count_prompt_versions(
            mode="local", prompt_identifier="local-uuid-7"
        )


@pytest.mark.parametrize(
    (
        "system_flag",
        "user_flag",
        "system_prompt",
        "user_prompt",
        "expected",
    ),
    [
        (1, 0, None, None, (True, False)),
        (2, -1, None, None, (False, False)),
        ("1", "0", None, None, (False, False)),
        (None, None, "System text", "User text", (True, True)),
    ],
)
def test_prompt_scope_lane_flags_accept_only_boolean_or_sqlite_boolean_values(
    system_flag,
    user_flag,
    system_prompt,
    user_prompt,
    expected,
):
    normalized = PromptScopeService._normalize_prompt_record(
        {
            "id": 7,
            "uuid": "local-uuid-7",
            "name": "Lane flags",
            "has_system_prompt": system_flag,
            "has_user_prompt": user_flag,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        backend="local",
    )

    assert (
        normalized["has_system_prompt"],
        normalized["has_user_prompt"],
    ) == expected


def test_prompt_scope_unknown_remote_artifact_remains_browsable_but_unsupported():
    normalized = PromptScopeService._normalize_prompt_list(
        {
            "items": [
                {
                    "id": 9,
                    "uuid": "known-prompt",
                    "name": "Known Prompt",
                    "artifact_type": "prompt",
                },
                {
                    "id": 10,
                    "uuid": "future-artifact",
                    "name": "Future Artifact",
                    "artifact_type": "workflow",
                    "system_prompt": "Compatibility system",
                    "user_prompt": "Compatibility user",
                },
                {
                    "id": 11,
                    "uuid": "known-recipe",
                    "name": "Known Recipe",
                    "artifact_type": "recipe",
                },
            ],
            "total_pages": 1,
            "current_page": 1,
            "total_items": 3,
        },
        backend="server",
        page=1,
        per_page=10,
    )

    assert [item["name"] for item in normalized["items"]] == [
        "Known Prompt",
        "Future Artifact",
        "Known Recipe",
    ]
    future = normalized["items"][1]
    assert future["artifact_type"] == "unsupported"
    assert future["artifact_type_raw"] == "workflow"
    assert future["definition_state"] == "unsupported"
    assert future["system_prompt"] == "Compatibility system"
    assert future["user_prompt"] == "Compatibility user"


@pytest.mark.asyncio
async def test_prompt_scope_saves_and_deletes_against_selected_backend():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(
        local_service=local, server_service=server, policy_enforcer=policy
    )

    created = await service.save_prompt(
        mode="local",
        name="New Local",
        author="Me",
        details="Details",
        system_prompt="System",
        user_prompt="User",
        keywords=["local"],
        artifact_type="recipe",
    )
    local_updated = await service.save_prompt(
        mode="local",
        prompt_identifier="local-uuid-7",
        details="Locally updated",
        artifact_type="prompt",
        expected_version=3,
    )
    updated = await service.save_prompt(
        mode="server",
        prompt_identifier="server-uuid-9",
        name="Updated Server",
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition={"schema_version": 1, "messages": []},
        artifact_type="recipe",
        expected_version=5,
    )
    deleted = await service.delete_prompt(
        mode="server", prompt_identifier="server-uuid-9"
    )

    assert created["id"] == "local:prompt:local-uuid-8"
    assert created["artifact_type"] == "recipe"
    assert local.calls[0][0] == "create_prompt"
    assert local.calls[0][1]["artifact_type"] == "recipe"
    assert local_updated["details"] == "Locally updated"
    assert local.calls[1] == (
        "update_prompt",
        "local-uuid-7",
        {
            "details": "Locally updated",
            "artifact_type": "prompt",
            "expected_version": 3,
        },
    )
    assert updated["id"] == "server:prompt:server-uuid-9"
    assert updated["name"] == "Updated Server"
    assert server.calls[-2][0] == "update_prompt"
    assert server.calls[-2][2]["artifact_type"] == "recipe"
    assert "expected_version" not in server.calls[-2][2]
    assert deleted is True
    assert policy.actions == [
        "prompts.create.local",
        "prompts.update.local",
        "prompts.update.server",
        "prompts.delete.server",
    ]


@pytest.mark.asyncio
async def test_prompt_scope_restores_deleted_local_prompt_as_conditional_update():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(local, server, policy)

    restored = await service.restore_deleted_prompt(
        mode="local",
        prompt_identifier="local-uuid-7",
        expected_version=4,
    )

    assert restored["id"] == "local:prompt:local-uuid-7"
    assert restored["local_id"] == 7
    assert restored["version"] == 5
    assert local.calls[-1] == (
        "restore_deleted_prompt",
        "local-uuid-7",
        4,
    )
    assert server.calls == []
    assert policy.actions == ["prompts.update.local"]


@pytest.mark.asyncio
async def test_prompt_scope_forwards_expected_version_for_local_delete():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(local, server, policy)

    deleted = await service.delete_prompt(
        mode="local",
        prompt_identifier="local-uuid-7",
        expected_version=4,
    )

    assert deleted is True
    assert local.calls[-1] == ("delete_prompt", "local-uuid-7", 4)
    assert server.calls == []
    assert policy.actions == ["prompts.delete.local"]


@pytest.mark.asyncio
async def test_build_prompt_scope_service_wires_local_and_server_backends_lazily(
    monkeypatch,
):
    client = FakeServerPromptService()
    build_client = Mock(return_value=client)
    monkeypatch.setattr(
        "tldw_chatbook.runtime_policy.bootstrap.build_runtime_api_client_from_config",
        build_client,
    )

    prompt_db = object()
    app_config = {"tldw_api": {"base_url": "http://server.test", "api_key": "token"}}
    service = build_prompt_scope_service(
        prompt_db=prompt_db,
        app_config=app_config,
        policy_enforcer="policy",
    )

    assert isinstance(service, PromptScopeService)
    assert isinstance(service.local_service, LocalPromptService)
    assert service.local_service.prompt_db is prompt_db
    assert isinstance(service.server_service, ServerPromptService)
    assert service.server_service.client is None
    assert service.server_service.client_provider is not None
    assert service.policy_enforcer == "policy"
    build_client.assert_not_called()

    prompts = await service.server_service.list_prompts(page=2, per_page=3)

    assert prompts.items[0].id == 9
    assert service.server_service.client is None
    build_client.assert_called_once_with(app_config)


def test_build_prompt_scope_service_keeps_server_backend_unavailable_without_config():
    service = build_prompt_scope_service(
        prompt_db=None, app_config={}, policy_enforcer=None
    )

    assert service.local_service is None
    assert isinstance(service.server_service, ServerPromptService)
    assert service.server_service.client is None
    assert service.server_service.client_provider is None


@pytest.mark.asyncio
async def test_scope_server_prompt_service_from_config_can_use_provider_backed_client(
    monkeypatch,
):
    build_client = Mock(
        side_effect=AssertionError("legacy config builder should not run")
    )
    monkeypatch.setattr(
        "tldw_chatbook.runtime_policy.bootstrap.build_runtime_api_client",
        build_client,
    )

    provider = FakeClientProvider(FakeServerPromptService())
    service = ServerPromptService.from_config(
        {"tldw_api": {"base_url": "https://example.com"}},
        client_provider=provider,
    )

    result = await service.list_prompts(page=2, per_page=3)

    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 1
    assert result.items[0].id == 9


@pytest.mark.asyncio
async def test_scope_server_prompt_service_from_config_uses_shared_provider_lazily(
    monkeypatch,
):
    client = FakeServerPromptService()
    direct_builder = Mock(
        side_effect=AssertionError("service should not call direct legacy builder")
    )
    provider_builder = Mock(return_value=client)
    monkeypatch.setattr(
        "tldw_chatbook.runtime_policy.bootstrap.build_runtime_api_client",
        direct_builder,
    )
    monkeypatch.setattr(
        "tldw_chatbook.runtime_policy.bootstrap.build_runtime_api_client_from_config",
        provider_builder,
    )

    service = ServerPromptService.from_config(
        {"tldw_api": {"base_url": "https://example.com"}}
    )

    assert isinstance(service, ServerPromptService)
    assert service.client is None
    assert service.client_provider is not None
    direct_builder.assert_not_called()
    provider_builder.assert_not_called()

    result = await service.list_prompts(page=2, per_page=3)

    assert result.items[0].id == 9
    assert service.client is None
    provider_builder.assert_called_once_with(
        {"tldw_api": {"base_url": "https://example.com"}}
    )


@pytest.mark.asyncio
async def test_scope_server_prompt_service_direct_client_takes_precedence_over_provider():
    client = FakeServerPromptService()
    provider = ExplodingClientProvider()
    service = ServerPromptService(client=client, client_provider=provider)

    result = await service.list_prompts(page=2, per_page=3)

    assert result.items[0].id == 9
    assert provider.build_calls == 0


@pytest.mark.asyncio
async def test_prompt_scope_routes_server_usage_versions_and_restore():
    policy = FakePolicyEnforcer()
    server = FakeServerPromptService()
    service = PromptScopeService(
        local_service=FakeLocalPromptService(),
        server_service=server,
        policy_enforcer=policy,
    )

    used = await service.record_prompt_usage(
        mode="server", prompt_identifier="server-uuid-9"
    )
    versions = await service.list_prompt_versions(
        mode="server", prompt_identifier="server-uuid-9"
    )
    restored = await service.restore_prompt_version(
        mode="server",
        prompt_identifier="server-uuid-9",
        version=3,
    )

    assert used["id"] == "server:prompt:server-uuid-9"
    assert versions == [
        {
            "backend": "server",
            "version": 5,
            "created_at": "2026-04-22T00:00:00Z",
            "comment": "current",
            "name": "Server Prompt",
            "author": None,
            "details": None,
            "system_prompt": None,
            "user_prompt": None,
            "prompt_uuid": None,
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
            "artifact_type": "prompt",
            "has_system_prompt": False,
            "has_user_prompt": False,
        }
    ]
    assert restored["version"] == 3
    assert server.calls[-3:] == [
        ("record_prompt_usage", "server-uuid-9"),
        ("list_prompt_versions", "server-uuid-9"),
        ("restore_prompt_version", "server-uuid-9", 3),
    ]
    assert policy.actions[-3:] == [
        "prompts.use.server",
        "prompts.versions.list.server",
        "prompts.versions.restore.server",
    ]


@pytest.mark.asyncio
async def test_prompt_scope_refuses_to_record_recipe_usage_after_authoritative_read():
    server = FakeServerPromptService()
    server.prompt = server.prompt.model_copy(update={"artifact_type": "recipe"})
    service = PromptScopeService(
        local_service=FakeLocalPromptService(),
        server_service=server,
        policy_enforcer=FakePolicyEnforcer(),
    )

    with pytest.raises(ValueError, match="Recipes cannot be used directly"):
        await service.record_prompt_usage(
            mode="server", prompt_identifier="server-uuid-9"
        )

    assert server.calls == [("get_prompt", "server-uuid-9", False)]


@pytest.mark.asyncio
async def test_prompt_scope_refuses_to_record_unknown_artifact_usage():
    server = FakeServerPromptService()
    server.prompt = {
        **server.prompt.model_dump(mode="json"),
        "artifact_type": "workflow",
    }
    service = PromptScopeService(
        local_service=FakeLocalPromptService(),
        server_service=server,
        policy_enforcer=FakePolicyEnforcer(),
    )

    with pytest.raises(ValueError, match="Only Prompt artifacts can be used directly"):
        await service.record_prompt_usage(
            mode="server", prompt_identifier="server-uuid-9"
        )

    assert server.calls == [("get_prompt", "server-uuid-9", False)]


@pytest.mark.asyncio
async def test_prompt_scope_routes_local_retained_history_and_restore_through_live_adapter():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    service = PromptScopeService(
        local_service=local,
        server_service=FakeServerPromptService(),
        policy_enforcer=policy,
    )

    page = await service.list_prompt_versions(
        mode="local", prompt_identifier="local-uuid-7", page_size=7, before_change_id=50
    )
    restored = await service.restore_prompt_version(
        mode="local",
        prompt_identifier="local-uuid-7",
        change_id=42,
        version=3,
        expected_version=3,
    )

    assert page["items"][0]["backend"] == "local"
    assert restored["new_version"] == 4
    assert local.calls[-2:] == [
        ("list_prompt_versions", "local-uuid-7", 7, 50),
        ("restore_prompt_version", "local-uuid-7", 42, 3, 3),
    ]
    assert policy.actions[-2:] == [
        "prompts.versions.list.local",
        "prompts.versions.restore.local",
    ]


@pytest.mark.asyncio
async def test_prompt_scope_app_wired_local_restore_uses_real_retained_history_transaction():
    database = PromptsDatabase(":memory:", client_id="scope-retained-history")
    try:
        prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Original",
            author="Author",
            details="Original details",
            system_prompt="Original system",
            user_prompt="Original user",
            keywords=["original"],
        )
        database.update_prompt_by_id(
            prompt_id,
            {
                "name": "Current",
                "details": "Current details",
                "keywords": ["current"],
            },
            expected_version=1,
        )
        service = PromptScopeService(
            local_service=LocalPromptService(database),
            server_service=FakeServerPromptService(),
        )

        page = await service.list_prompt_versions(
            mode="local", prompt_identifier=prompt_uuid, page_size=10
        )
        source = next(item for item in page["items"] if item["version"] == 1)
        result = await service.restore_prompt_version(
            mode="local",
            prompt_identifier=prompt_uuid,
            change_id=source["change_id"],
            version=source["version"],
            expected_version=2,
        )

        restored = database.fetch_prompt_details(prompt_uuid)
        assert result == {
            "outcome": "restored",
            "snapshot_unavailable": False,
            "no_change": False,
            "source_version": 1,
            "current_version": 2,
            "new_version": 3,
            "retained_current_keywords": False,
        }
        assert restored["name"] == "Original"
        assert restored["keywords"] == ["original"]
        assert restored["version"] == 3
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_local_history_page_and_restore_enforce_current_capabilities_and_legacy_recipe_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = PromptsDatabase(":memory:", client_id="scope-history-capabilities")
    capabilities = replace(local_prompt_capabilities(), compiled_lane_limit=4)
    monkeypatch.setattr(
        prompt_scope_module, "local_prompt_capabilities", lambda: capabilities
    )
    try:
        _legacy_id, legacy_uuid, _message = database.add_prompt(
            name="Legacy Recipe",
            author=None,
            details="literal legacy details",
            system_prompt="[bold]literal legacy system[/bold]",
            user_prompt="literal legacy user",
            prompt_format="legacy",
            artifact_type="recipe",
        )
        definition = block_definition(content="hello")
        _large_id, large_uuid, _message = database.add_prompt(
            name="Large structured Prompt",
            author=None,
            details="literal structured details",
            system_prompt="",
            user_prompt="hello",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=definition,
            artifact_type="prompt",
        )
        service = PromptScopeService(
            local_service=LocalPromptService(database),
            server_service=FakeServerPromptService(),
        )

        cases = (
            (
                legacy_uuid,
                "legacy_recipe",
                "Legacy Recipe snapshots are preview-only.",
            ),
            (
                large_uuid,
                "current_capability_unsupported",
                "This retained version is not supported by current local Prompt "
                "capabilities.",
            ),
        )
        for prompt_uuid, expected_state, expected_reason in cases:
            page = await service.list_prompt_versions(
                mode="local", prompt_identifier=prompt_uuid, page_size=10
            )
            source = page["items"][0]
            assert source["compatibility_state"] == expected_state
            assert source["compatibility_reason"] == expected_reason
            assert source["restore_eligible"] is False

            before_detail = database.fetch_prompt_details(prompt_uuid)
            before_count = database.get_prompt_history_count(prompt_uuid)
            with pytest.raises(PromptRestoreError) as exc_info:
                await service.restore_prompt_version(
                    mode="local",
                    prompt_identifier=prompt_uuid,
                    change_id=source["change_id"],
                    version=source["version"],
                    expected_version=before_detail["version"],
                )

            assert exc_info.value.code is PromptRestoreErrorCode.VALIDATION
            assert database.fetch_prompt_details(prompt_uuid) == before_detail
            assert database.get_prompt_history_count(prompt_uuid) == before_count
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_prompt_scope_structured_compact_snapshot_no_change_preserves_durable_json():
    database = PromptsDatabase(":memory:", client_id="scope-compact-no-change")
    compact_definition = (
        '{"schema_version":2,"kind":"block_prompt","lanes":['
        '{"id":"system","blocks":[]},'
        '{"id":"user","blocks":[{"id":"u","title":"User",'
        '"syntax":"freeform","content":"Hello"}]}]}'
    )
    try:
        _prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Compact structured",
            author=None,
            details="Details",
            system_prompt="",
            user_prompt="Hello",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=compact_definition,
            artifact_type="prompt",
        )
        service = PromptScopeService(
            local_service=LocalPromptService(database),
            server_service=FakeServerPromptService(),
        )
        page = await service.list_prompt_versions(
            mode="local", prompt_identifier=prompt_uuid, page_size=1
        )
        source = page["items"][0]
        before = (
            database.get_connection()
            .execute(
                "SELECT version, prompt_definition FROM Prompts WHERE uuid = ?",
                (prompt_uuid,),
            )
            .fetchone()
        )
        before_history_count = database.get_prompt_history_count(prompt_uuid)

        result = await service.restore_prompt_version(
            mode="local",
            prompt_identifier=prompt_uuid,
            change_id=source["change_id"],
            version=1,
            expected_version=1,
        )

        after = (
            database.get_connection()
            .execute(
                "SELECT version, prompt_definition FROM Prompts WHERE uuid = ?",
                (prompt_uuid,),
            )
            .fetchone()
        )
        assert result["outcome"] == "no_change"
        assert result["new_version"] == 1
        assert tuple(after) == tuple(before)
        assert database.get_prompt_history_count(prompt_uuid) == before_history_count
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_prompt_scope_routes_server_prompt_collections_with_policy():
    policy = FakePolicyEnforcer()
    server = FakeServerPromptService()
    service = PromptScopeService(
        local_service=FakeLocalPromptService(),
        server_service=server,
        policy_enforcer=policy,
    )

    created = await service.create_prompt_collection(
        mode="server",
        name="Server Collection",
        description="Remote prompts",
        prompt_ids=[9],
    )
    listed = await service.list_prompt_collections(mode="server", limit=50, offset=5)
    fetched = await service.get_prompt_collection(mode="server", collection_id=7)
    updated = await service.update_prompt_collection(
        mode="server",
        collection_id=7,
        name="Renamed",
        description="Updated",
        prompt_ids=[9, 10],
    )

    assert created == {
        "id": "server:prompt_collection:7",
        "backend": "server",
        "collection_id": 7,
    }
    assert listed["collections"][0]["id"] == "server:prompt_collection:7"
    assert fetched["name"] == "Server Collection"
    assert updated["name"] == "Renamed"
    assert server.calls[-4:] == [
        (
            "create_prompt_collection",
            {
                "name": "Server Collection",
                "description": "Remote prompts",
                "prompt_ids": [9],
            },
        ),
        ("list_prompt_collections", 50, 5),
        ("get_prompt_collection", 7),
        (
            "update_prompt_collection",
            7,
            {"name": "Renamed", "description": "Updated", "prompt_ids": [9, 10]},
        ),
    ]
    assert policy.actions[-4:] == [
        "prompts.collections.create.server",
        "prompts.collections.list.server",
        "prompts.collections.detail.server",
        "prompts.collections.update.server",
    ]


@pytest.mark.asyncio
async def test_prompt_scope_routes_local_prompt_collections_with_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    service = PromptScopeService(
        local_service=local,
        server_service=FakeServerPromptService(),
        policy_enforcer=policy,
    )

    created = await service.create_prompt_collection(
        mode="local",
        name="Local Collection",
        description="Offline prompts",
        prompt_ids=[7],
    )
    listed = await service.list_prompt_collections(
        mode="local", query="  Collection  ", limit=50, offset=5
    )
    fetched = await service.get_prompt_collection(mode="local", collection_id=3)
    updated = await service.update_prompt_collection(
        mode="local",
        collection_id=3,
        name="Renamed",
        description="Updated",
        prompt_ids=[7, 8],
    )

    assert created == {
        "id": "local:prompt_collection:3",
        "backend": "local",
        "collection_id": 3,
    }
    assert listed["collections"][0]["id"] == "local:prompt_collection:3"
    assert fetched["name"] == "Local Collection"
    assert updated["name"] == "Renamed"
    assert local.calls[-4:] == [
        (
            "create_prompt_collection",
            {
                "name": "Local Collection",
                "description": "Offline prompts",
                "prompt_ids": [7],
            },
        ),
        ("list_prompt_collections", "Collection", 50, 5),
        ("get_prompt_collection", 3),
        (
            "update_prompt_collection",
            3,
            {"name": "Renamed", "description": "Updated", "prompt_ids": [7, 8]},
        ),
    ]
    assert policy.actions[-4:] == [
        "prompts.collections.create.local",
        "prompts.collections.list.local",
        "prompts.collections.detail.local",
        "prompts.collections.update.local",
    ]


@pytest.mark.asyncio
async def test_prompt_scope_routes_local_prompt_memberships_with_bounded_outcomes():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(local, server, policy)

    listed = await service.list_prompt_collection_memberships(mode="local", prompt_id=7)
    replaced = await service.replace_prompt_collection_memberships(
        mode=PromptBackend.LOCAL,
        prompt_id=7,
        collection_ids=[8, 3],
    )

    assert listed == {
        "prompt_id": 7,
        "collection_ids": (3, 8),
        "changed": False,
    }
    assert replaced == {
        "prompt_id": 7,
        "collection_ids": (3, 8),
        "changed": True,
    }
    assert policy.actions == [
        "prompts.collections.detail.local",
        "prompts.collections.update.local",
    ]
    assert local.calls[-2:] == [
        ("list_prompt_collection_memberships", 7),
        ("replace_prompt_collection_memberships", 7, (3, 8)),
    ]
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"collection_ids": (3, 8), "changed": False},
        {"prompt_id": 8, "collection_ids": (3, 8), "changed": False},
    ],
)
async def test_prompt_scope_membership_list_rejects_missing_or_mismatched_response_id(
    response,
):
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    local.list_prompt_collection_memberships = Mock(return_value=response)
    service = PromptScopeService(local, FakeServerPromptService(), policy)

    with pytest.raises(ValueError, match="response prompt_id"):
        await service.list_prompt_collection_memberships(mode="local", prompt_id=7)

    assert policy.actions == ["prompts.collections.detail.local"]
    local.list_prompt_collection_memberships.assert_called_once_with(7)


@pytest.mark.asyncio
async def test_prompt_scope_membership_replace_rejects_different_response_collection_ids():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    local.replace_prompt_collection_memberships = Mock(
        return_value={
            "prompt_id": 7,
            "collection_ids": (3, 9),
            "changed": True,
        }
    )
    service = PromptScopeService(local, FakeServerPromptService(), policy)

    with pytest.raises(ValueError, match="response collection_ids"):
        await service.replace_prompt_collection_memberships(
            mode="local", prompt_id=7, collection_ids=[8, 3]
        )

    assert policy.actions == ["prompts.collections.update.local"]
    local.replace_prompt_collection_memberships.assert_called_once_with(7, (3, 8))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["server", "all", "mixed", None])
async def test_prompt_scope_rejects_non_local_membership_modes_before_policy_or_adapter(
    mode,
):
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(local, server, policy)

    with pytest.raises(ValueError, match="local-only"):
        await service.list_prompt_collection_memberships(mode=mode, prompt_id=7)
    with pytest.raises(ValueError, match="local-only"):
        await service.replace_prompt_collection_memberships(
            mode=mode, prompt_id=7, collection_ids=[3]
        )

    assert policy.actions == []
    assert local.calls == []
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt_id", "collection_ids"),
    [
        (True, [3]),
        (0, [3]),
        (2**63, [3]),
        ("7", [3]),
        (7, None),
        (7, "3"),
        (7, [3, True]),
        (7, [3, 4, 3]),
        (7, [3, 2**63]),
    ],
)
async def test_prompt_scope_rejects_invalid_memberships_before_policy_or_adapter(
    prompt_id, collection_ids
):
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(local, server, policy)

    with pytest.raises(ValueError):
        await service.replace_prompt_collection_memberships(
            mode="local", prompt_id=prompt_id, collection_ids=collection_ids
        )

    assert policy.actions == []
    assert local.calls == []
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_id", [True, 0, -1, 1.5, "7", 2**63])
async def test_prompt_scope_rejects_invalid_membership_list_id_before_policy_or_adapter(
    prompt_id,
):
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(local, server, policy)

    with pytest.raises(ValueError, match="prompt_id"):
        await service.list_prompt_collection_memberships(
            mode="local", prompt_id=prompt_id
        )

    assert policy.actions == []
    assert local.calls == []
    assert server.calls == []


@pytest.mark.asyncio
async def test_prompt_scope_membership_policy_denial_stops_before_local_adapter():
    policy = FakePolicyEnforcer.deny()
    local = FakeLocalPromptService()
    service = PromptScopeService(local, FakeServerPromptService(), policy)

    with pytest.raises(PermissionError):
        await service.list_prompt_collection_memberships(mode="local", prompt_id=7)

    assert policy.actions == ["prompts.collections.detail.local"]
    assert local.calls == []


@pytest.mark.asyncio
async def test_prompt_scope_membership_update_policy_denial_stops_before_local_adapter():
    policy = FakePolicyEnforcer.deny()
    local = FakeLocalPromptService()
    service = PromptScopeService(local, FakeServerPromptService(), policy)

    with pytest.raises(PermissionError):
        await service.replace_prompt_collection_memberships(
            mode="local", prompt_id=7, collection_ids=[3]
        )

    assert policy.actions == ["prompts.collections.update.local"]
    assert local.calls == []


@pytest.mark.asyncio
async def test_prompt_scope_rejects_server_collection_query_before_policy_or_adapter():
    policy = FakePolicyEnforcer()
    server = FakeServerPromptService()
    service = PromptScopeService(
        local_service=FakeLocalPromptService(),
        server_service=server,
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Server prompt collection search"):
        await service.list_prompt_collections(mode="server", query="sales")

    assert policy.actions == []
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": None},
        {"limit": True},
        {"limit": 0},
        {"offset": False},
        {"offset": -1},
        {"offset": 2**63},
    ],
)
async def test_prompt_scope_validates_collection_catalog_before_policy_or_adapter(
    kwargs,
):
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    service = PromptScopeService(
        local_service=local,
        server_service=FakeServerPromptService(),
        policy_enforcer=policy,
    )

    with pytest.raises((TypeError, ValueError)):
        await service.list_prompt_collections(mode="local", **kwargs)

    assert policy.actions == []
    assert local.calls == []


@pytest.mark.asyncio
async def test_prompt_scope_accepts_signed_maximum_collection_catalog_offset():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    service = PromptScopeService(
        local_service=local,
        server_service=FakeServerPromptService(),
        policy_enforcer=policy,
    )

    listed = await service.list_prompt_collections(
        mode="local", limit=1, offset=(2**63) - 1
    )

    assert listed["offset"] == (2**63) - 1
    assert local.calls == [
        ("list_prompt_collections", "", 1, (2**63) - 1),
    ]
    assert policy.actions == ["prompts.collections.list.local"]


def test_local_prompt_service_persists_prompt_collections(tmp_path):
    prompt_db = PromptsDatabase(tmp_path / "prompts.db", client_id="test_client")
    prompt_id, _prompt_uuid, _ = prompt_db.add_prompt(
        name="Local Prompt",
        author="Writer",
        details="Details",
        system_prompt="System",
        user_prompt="User",
        keywords=["draft"],
        overwrite=False,
    )
    second_prompt_id, _second_prompt_uuid, _ = prompt_db.add_prompt(
        name="Second Local Prompt",
        author="Writer",
        details="More details",
        system_prompt="Second system",
        user_prompt="Second user",
        keywords=["draft"],
        overwrite=False,
    )
    service = LocalPromptService(prompt_db)

    created = service.create_prompt_collection(
        {
            "name": "Local Collection",
            "description": "Offline prompts",
            "prompt_ids": [prompt_id],
        }
    )
    listed = service.list_prompt_collections(limit=10, offset=0)
    fetched = service.get_prompt_collection(created["collection_id"])
    updated = service.update_prompt_collection(
        created["collection_id"],
        {
            "name": "Renamed",
            "description": "Updated",
            "prompt_ids": [prompt_id],
        },
    )
    membership_updated = service.update_prompt_collection(
        created["collection_id"],
        {"prompt_ids": [second_prompt_id]},
    )

    assert listed["total"] == 1
    assert fetched["prompt_ids"] == [prompt_id]
    assert updated["name"] == "Renamed"
    assert updated["prompt_ids"] == [prompt_id]
    assert membership_updated["name"] == "Renamed"
    assert membership_updated["prompt_ids"] == [second_prompt_id]


@pytest.mark.asyncio
async def test_local_capabilities_are_frozen_known_in_process_and_use_canonical_limits():
    service = PromptScopeService(
        local_service=FakeLocalPromptService(), server_service=FakeServerPromptService()
    )

    capabilities = await service.get_capabilities(mode="local")

    assert capabilities.backend == "local"
    assert capabilities.structured_kinds == frozenset(
        {(2, "block_prompt"), (2, "block_recipe")}
    )
    assert capabilities.artifact_types == frozenset({"prompt", "recipe"})
    assert capabilities.search is True
    assert capabilities.conditional_update is True
    assert capabilities.compiled_lane_limit == 20_000
    assert capabilities.definition_limit == 256_000
    assert capabilities.request_limit == 512_000
    assert capabilities.json_byte_measurement == CANONICAL_JSON_UTF8_V1
    with pytest.raises(FrozenInstanceError):
        capabilities.search = False


@pytest.mark.asyncio
async def test_modern_server_capabilities_preserve_exact_kinds_and_smaller_limits():
    server = FakeServerPromptService(
        health=modern_prompt_health(conditional_update=True)
    )
    service = PromptScopeService(FakeLocalPromptService(), server)

    capabilities = await service.get_capabilities(mode="server")
    cached = await service.get_capabilities(mode="server")

    assert capabilities.structured_kinds == frozenset(
        {
            (1, "multi_message"),
            (2, "block_prompt"),
            (2, "block_recipe"),
        }
    )
    assert capabilities.artifact_types == frozenset({"prompt", "recipe"})
    assert capabilities.search is True
    assert capabilities.conditional_update is False
    assert capabilities.compiled_lane_limit == 12_000
    assert capabilities.definition_limit == 200_000
    assert capabilities.request_limit == 400_000
    assert capabilities.json_byte_measurement == CANONICAL_JSON_UTF8_V1
    assert cached is capabilities
    assert server.calls == [("get_prompts_health",)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "health",
    [
        {"status": "healthy"},
        {"status": "healthy", "capabilities": "not-an-object"},
        {"status": "healthy", "capabilities": {"structured_kinds": [2]}},
        {
            "status": "healthy",
            "capabilities": {
                "structured_kinds": [{"schema_version": 2, "kind": "block_prompt"}],
                "artifact_types": [{}],
            },
        },
    ],
)
async def test_legacy_or_malformed_server_health_fails_closed_but_remains_browsable(
    health,
):
    server = FakeServerPromptService(health=health)
    service = PromptScopeService(FakeLocalPromptService(), server)

    capabilities = await service.get_capabilities(mode="server")
    listed = await service.list_prompts(mode="server", page=1, per_page=5)

    assert capabilities.structured_kinds == frozenset()
    assert capabilities.artifact_types == frozenset({"prompt"})
    assert capabilities.search is False
    assert capabilities.conditional_update is False
    assert capabilities.compiled_lane_limit == 20_000
    assert capabilities.definition_limit == 256_000
    assert capabilities.request_limit == 512_000
    assert capabilities.json_byte_measurement is None
    assert listed["items"][0]["artifact_type"] == "prompt"


@pytest.mark.asyncio
async def test_single_text_only_server_is_not_inferred_to_support_console_block_v2():
    server = FakeServerPromptService(
        health=modern_prompt_health(
            structured_kinds=[{"schema_version": 2, "kind": "single_text_recipe"}]
        )
    )
    service = PromptScopeService(FakeLocalPromptService(), server)

    capabilities = await service.get_capabilities(mode="server")

    assert capabilities.structured_kinds == frozenset({(2, "single_text_recipe")})
    with pytest.raises(PromptCapabilityError) as exc:
        await service.save_prompt(
            mode="server",
            name="Console Prompt",
            artifact_type="prompt",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=block_definition(),
            user_prompt="hello",
        )
    assert (exc.value.backend, exc.value.capability) == (
        "server",
        "structured kind (2, 'block_prompt')",
    )
    assert not any(call[0] == "create_prompt" for call in server.calls)


@pytest.mark.asyncio
async def test_block_save_does_not_accept_boolean_schema_version_as_an_integer():
    local = FakeLocalPromptService()
    service = PromptScopeService(local, FakeServerPromptService())

    with pytest.raises(PromptCapabilityError):
        await service.save_prompt(
            mode="local",
            name="Boolean Version",
            artifact_type="prompt",
            prompt_format="structured",
            prompt_schema_version=True,
            prompt_definition=block_definition(),
            user_prompt="hello",
        )

    assert not any(call[0] == "create_prompt" for call in local.calls)


@pytest.mark.asyncio
async def test_server_search_routes_non_empty_query_without_hidden_detail_fetches():
    definition = block_definition(content="résumé")
    server = FakeServerPromptService(
        search_items=[
            {
                "id": 41,
                "uuid": "server-search-41",
                "name": "Alpha",
                "artifact_type": "prompt",
                "has_system_prompt": False,
                "has_user_prompt": True,
                "version": 7,
                "prompt_format": "structured",
                "prompt_schema_version": 2,
                "prompt_definition": definition,
                "system_prompt": "",
                "user_prompt": "résumé",
            }
        ]
    )
    service = PromptScopeService(FakeLocalPromptService(), server)

    items = await service.search_prompts(mode="server", query="alpha", limit=25)

    assert server.calls == [
        ("get_prompts_health",),
        (
            "search_prompts",
            {
                "search_query": "alpha",
                "page": 1,
                "results_per_page": 25,
                "include_deleted": False,
            },
        ),
    ]
    assert items[0]["backend"] == "server"
    assert items[0]["source_id"] == "server-search-41"
    assert items[0]["server_id"] == 41
    assert items[0]["version"] == 7
    assert items[0]["artifact_type"] == "prompt"
    assert items[0]["has_system_prompt"] is False
    assert items[0]["has_user_prompt"] is True
    assert items[0]["definition_state"] == "supported_v2"
    assert items[0]["prompt_definition"] == definition


@pytest.mark.asyncio
async def test_empty_server_query_uses_paginated_list_not_search():
    server = FakeServerPromptService()
    service = PromptScopeService(FakeLocalPromptService(), server)

    items = await service.search_prompts(
        mode="server", query="", limit=25, include_deleted=True
    )

    assert items[0]["name"] == "Server Prompt"
    assert server.calls == [("list_prompts", 1, 25, True, "last_modified", "desc")]


@pytest.mark.asyncio
async def test_whitespace_only_server_query_lists_but_nonempty_bytes_are_unchanged():
    server = FakeServerPromptService()
    service = PromptScopeService(FakeLocalPromptService(), server)

    listed = await service.search_prompts(mode="server", query=" \t\n", limit=7)
    await service.search_prompts(mode="server", query="  alpha  ", limit=9)

    assert listed[0]["name"] == "Server Prompt"
    assert server.calls == [
        ("list_prompts", 1, 7, False, "last_modified", "desc"),
        ("get_prompts_health",),
        (
            "search_prompts",
            {
                "search_query": "  alpha  ",
                "page": 1,
                "results_per_page": 9,
                "include_deleted": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_unsupported_or_policy_denied_server_search_is_typed_unavailable():
    unsupported = PromptScopeService(
        FakeLocalPromptService(),
        FakeServerPromptService(health={"status": "healthy"}),
    )
    with pytest.raises(PromptCapabilityError) as unsupported_exc:
        await unsupported.search_prompts(mode="server", query="alpha")
    assert (unsupported_exc.value.backend, unsupported_exc.value.capability) == (
        "server",
        "search",
    )

    denied_server = FakeServerPromptService()
    denied = PromptScopeService(
        FakeLocalPromptService(),
        denied_server,
        policy_enforcer=FakePolicyEnforcer.deny("server policy"),
    )
    with pytest.raises(PromptCapabilityError) as denied_exc:
        await denied.search_prompts(mode="server", query="alpha")
    assert (denied_exc.value.backend, denied_exc.value.capability) == (
        "server",
        "search",
    )
    assert denied_server.calls == []


def test_canonical_json_utf8_size_measures_decoded_unicode_mapping():
    value = {"z": "é", "a": [True, None]}

    assert canonical_json_utf8_size(value) == len(
        '{"a":[true,null],"z":"é"}'.encode("utf-8")
    )


def test_full_malformed_definition_is_classified_without_losing_source_bytes():
    record = {
        "id": 4,
        "name": "Malformed",
        "artifact_type": "prompt",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": "{not-json",
        "system_prompt": "compiled system",
        "user_prompt": "compiled user",
    }

    normalized = normalize_prompt_record(record, backend="server")

    assert normalized["definition_state"] == "malformed"
    assert normalized["prompt_definition"] == "{not-json"


@pytest.mark.asyncio
async def test_malformed_measurement_and_limits_fail_before_structured_persistence():
    malformed_measurement = modern_prompt_health(
        measurement={
            "name": "wire_bytes",
            "encoding": "utf-8",
            "ensure_ascii": False,
            "sort_keys": True,
            "separators": [",", ":"],
        }
    )
    server = FakeServerPromptService(health=malformed_measurement)
    service = PromptScopeService(FakeLocalPromptService(), server)

    capabilities = await service.get_capabilities(mode="server")
    assert capabilities.json_byte_measurement is None
    with pytest.raises(PromptCapabilityError, match="canonical JSON byte measurement"):
        await service.save_prompt(
            mode="server",
            name="Console Prompt",
            artifact_type="prompt",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=block_definition(),
            user_prompt="hello",
        )

    local = FakeLocalPromptService()
    local_service = PromptScopeService(local, server)
    with pytest.raises(ValueError, match="user_prompt.*20000 characters"):
        await local_service.save_prompt(
            mode="local",
            name="Oversized",
            artifact_type="prompt",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=block_definition(content="x" * 20_001),
            user_prompt="x" * 20_001,
        )
    assert not any(call[0] == "create_prompt" for call in local.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("health", "message"),
    [
        (
            modern_prompt_health(definition_limit=100),
            "prompt_definition exceeds 100 UTF-8 bytes",
        ),
        (
            modern_prompt_health(request_limit=250),
            "request exceeds 250 UTF-8 bytes",
        ),
    ],
)
async def test_server_canonical_byte_limits_reject_without_truncation_or_mutation(
    health, message
):
    server = FakeServerPromptService(health=health)
    service = PromptScopeService(FakeLocalPromptService(), server)
    definition = block_definition(content="é" * 40)

    with pytest.raises(ValueError, match=message):
        await service.save_prompt(
            mode="server",
            name="Unicode",
            artifact_type="prompt",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=definition,
            user_prompt="é" * 40,
        )

    assert definition["lanes"][1]["blocks"][0]["content"] == "é" * 40
    assert not any(call[0] == "create_prompt" for call in server.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_format", [None, "legacy"])
async def test_block_v2_evidence_rejects_missing_or_legacy_format_before_mutation(
    prompt_format,
):
    local = FakeLocalPromptService()
    service = PromptScopeService(local, FakeServerPromptService())

    with pytest.raises(PromptCapabilityError, match="valid Console block artifact"):
        await service.save_prompt(
            mode="local",
            name="Inconsistent block",
            artifact_type="prompt",
            prompt_format=prompt_format,
            prompt_schema_version=2,
            prompt_definition=block_definition(),
            user_prompt="caller text",
        )

    assert local.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_user", ["stale caller lane", "x" * 20_001])
async def test_block_save_persists_compiler_lanes_not_caller_compatibility_fields(
    caller_user,
):
    local = FakeLocalPromptService()
    service = PromptScopeService(local, FakeServerPromptService())

    await service.save_prompt(
        mode="local",
        name="Canonical lanes",
        artifact_type="prompt",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=block_definition(content="tiny compiled lane"),
        system_prompt="stale system lane",
        user_prompt=caller_user,
    )

    persisted = local.calls[0][1]
    assert persisted["system_prompt"] == ""
    assert persisted["user_prompt"] == "tiny compiled lane"


@pytest.mark.asyncio
async def test_definition_limit_measures_and_persists_final_normalized_model():
    definition = block_definition(content="tiny")
    block = definition["lanes"][1]["blocks"][0]
    block["xml_tag"] = None
    block["mapping_hint"] = None
    expected_definition = block_definition(content="tiny")
    normalized_size = canonical_json_utf8_size(expected_definition)
    assert canonical_json_utf8_size(definition) > normalized_size

    server = FakeServerPromptService(
        health=modern_prompt_health(definition_limit=normalized_size)
    )
    service = PromptScopeService(FakeLocalPromptService(), server)

    await service.save_prompt(
        mode="server",
        name="Normalized definition",
        artifact_type="prompt",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        user_prompt="caller value",
    )

    sent = next(call[1] for call in server.calls if call[0] == "create_prompt")
    assert sent["prompt_definition"] == expected_definition
    assert sent["user_prompt"] == "tiny"


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_identifier", [None, "server-uuid-9"])
async def test_request_limit_measures_exact_create_or_update_mapping_after_defaults(
    prompt_identifier,
):
    definition = block_definition(content="tiny")
    pre_default_mapping = {
        "name": "Default boundary",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": definition,
    }
    pre_default_size = canonical_json_utf8_size(pre_default_mapping)
    server = FakeServerPromptService(
        health=modern_prompt_health(request_limit=pre_default_size)
    )
    service = PromptScopeService(FakeLocalPromptService(), server)

    with pytest.raises(ValueError, match=f"request exceeds {pre_default_size}"):
        await service.save_prompt(
            mode="server",
            prompt_identifier=prompt_identifier,
            name="Default boundary",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=definition,
        )

    assert not any(
        call[0] in {"create_prompt", "update_prompt"} for call in server.calls
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "measurement",
    [
        {
            "name": "canonical_json_utf8_v1",
            "encoding": "utf-8",
            "ensure_ascii": 0,
            "sort_keys": True,
            "separators": [",", ":"],
        },
        {
            "name": "canonical_json_utf8_v1",
            "encoding": "utf-8",
            "ensure_ascii": False,
            "sort_keys": 1,
            "separators": [",", ":"],
        },
        {
            "name": "canonical_json_utf8_v1",
            "encoding": "utf-8",
            "ensure_ascii": False,
            "sort_keys": True,
            "separators": (",", ":"),
        },
    ],
)
async def test_measurement_descriptor_requires_exact_json_types(measurement):
    server = FakeServerPromptService(
        health=modern_prompt_health(measurement=measurement)
    )
    service = PromptScopeService(FakeLocalPromptService(), server)

    capabilities = await service.get_capabilities(mode="server")

    assert capabilities.json_byte_measurement is None


class FakeBatchLocalPromptService:
    def __init__(self):
        self.calls = []
        self.deleted_result = PromptBatchDeleteResult(
            entries=(
                PromptDeleteReceiptEntry(7, "Seven", "prompt", 4),
                PromptDeleteReceiptEntry(9, "Nine", "recipe", 3),
            )
        )
        self.restored_result = PromptBatchRestoreResult(
            entries=(
                PromptRestoreResultEntry(7, 5),
                PromptRestoreResultEntry(9, 4),
            )
        )

    def delete_prompts(self, *, targets):
        self.calls.append(("delete", targets))
        return self.deleted_result

    def restore_deleted_prompts(self, *, targets):
        self.calls.append(("restore", targets))
        return self.restored_result


class MissingBatchLocalPromptService:
    def __init__(self):
        self.calls = []


class PromptBatchTargetSubclass(PromptBatchTarget):
    pass


def _forged_prompt_batch_target(local_id, expected_version) -> PromptBatchTarget:
    target = object.__new__(PromptBatchTarget)
    object.__setattr__(target, "local_id", local_id)
    object.__setattr__(target, "expected_version", expected_version)
    return target


@pytest.mark.asyncio
async def test_prompt_batch_methods_are_keyword_only_typed_and_return_local_objects():
    from typing import get_type_hints

    local = FakeBatchLocalPromptService()
    policy = FakePolicyEnforcer()
    scope = PromptScopeService(local, FakeServerPromptService(), policy)
    targets = (PromptBatchTarget(7, 3), PromptBatchTarget(9, 2))
    scope._normalize_prompt_record = Mock(
        side_effect=AssertionError("batch results must not be normalized")
    )

    for method_name, result_type in (
        ("delete_prompts", PromptBatchDeleteResult),
        ("restore_deleted_prompts", PromptBatchRestoreResult),
    ):
        method = getattr(PromptScopeService, method_name)
        signature = inspect.signature(method)
        assert signature.parameters["mode"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["targets"].kind is inspect.Parameter.KEYWORD_ONLY
        hints = get_type_hints(method)
        assert hints["targets"] == tuple[PromptBatchTarget, ...]
        assert hints["return"] is result_type

    with pytest.raises(TypeError):
        await scope.delete_prompts("local", targets)  # type: ignore[misc]
    assert policy.actions == []
    assert local.calls == []

    deleted = await scope.delete_prompts(mode="local", targets=targets)
    restore_targets = deleted.targets
    restored = await scope.restore_deleted_prompts(
        mode="local", targets=restore_targets
    )

    assert deleted is local.deleted_result
    assert restored is local.restored_result
    assert policy.actions == ["prompts.delete.local", "prompts.update.local"]
    assert local.calls[0][0] == "delete"
    assert local.calls[0][1] is targets
    assert local.calls[1][0] == "restore"
    assert local.calls[1][1] is restore_targets
    assert scope._normalize_prompt_record.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "local", PromptBackend.LOCAL])
@pytest.mark.parametrize(
    ("method_name", "action", "result_attribute"),
    [
        ("delete_prompts", "prompts.delete.local", "deleted_result"),
        ("restore_deleted_prompts", "prompts.update.local", "restored_result"),
    ],
)
async def test_prompt_batch_methods_accept_established_local_mode_forms_once(
    mode, method_name, action, result_attribute
):
    local = FakeBatchLocalPromptService()
    policy = FakePolicyEnforcer()
    scope = PromptScopeService(local, FakeServerPromptService(), policy)
    targets = (PromptBatchTarget(7, 3), PromptBatchTarget(9, 2))

    result = await getattr(scope, method_name)(mode=mode, targets=targets)

    assert result is getattr(local, result_attribute)
    assert policy.actions == [action]
    assert len(local.calls) == 1
    assert local.calls[0][1] is targets


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "action", "operation"),
    [
        ("delete_prompts", "prompts.delete.local", "delete"),
        ("restore_deleted_prompts", "prompts.update.local", "restore"),
    ],
)
async def test_prompt_batch_methods_canonicalize_targets_before_one_policy_call(
    method_name, action, operation
):
    local = FakeBatchLocalPromptService()
    policy = FakePolicyEnforcer()
    scope = PromptScopeService(local, FakeServerPromptService(), policy)
    targets = (PromptBatchTarget(9, 2), PromptBatchTarget(7, 3))

    await getattr(scope, method_name)(mode="local", targets=targets)

    assert policy.actions == [action]
    assert local.calls == [(operation, tuple(reversed(targets)))]


INVALID_PROMPT_BATCH_TARGETS = [
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


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["delete_prompts", "restore_deleted_prompts"])
@pytest.mark.parametrize(
    ("targets", "error_type", "message"), INVALID_PROMPT_BATCH_TARGETS
)
async def test_prompt_batch_invalid_targets_fail_before_policy_or_backends(
    method_name, targets, error_type, message
):
    local = FakeBatchLocalPromptService()
    server = FakeServerPromptService()
    policy = FakePolicyEnforcer()
    scope = PromptScopeService(local, server, policy)

    with pytest.raises(error_type, match=message):
        await getattr(scope, method_name)(mode="local", targets=targets)

    assert policy.actions == []
    assert local.calls == []
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["delete_prompts", "restore_deleted_prompts"])
@pytest.mark.parametrize("mode", ["LOCAL", "remote", 1])
async def test_prompt_batch_invalid_modes_fail_before_policy_or_backends(
    method_name, mode
):
    local = FakeBatchLocalPromptService()
    server = FakeServerPromptService()
    policy = FakePolicyEnforcer()
    scope = PromptScopeService(local, server, policy)

    with pytest.raises(ValueError, match="Invalid prompt backend"):
        await getattr(scope, method_name)(mode=mode, targets=(PromptBatchTarget(7, 3),))

    assert policy.actions == []
    assert local.calls == []
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["delete_prompts", "restore_deleted_prompts"])
@pytest.mark.parametrize("mode", ["server", PromptBackend.SERVER])
async def test_prompt_batch_server_modes_are_refused_before_policy_or_backends(
    method_name, mode
):
    local = FakeBatchLocalPromptService()
    server = FakeServerPromptService()
    policy = FakePolicyEnforcer()
    scope = PromptScopeService(local, server, policy)

    with pytest.raises(ValueError, match="local-only"):
        await getattr(scope, method_name)(mode=mode, targets=(PromptBatchTarget(7, 3),))

    assert policy.actions == []
    assert local.calls == []
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "action"),
    [
        ("delete_prompts", "prompts.delete.local"),
        ("restore_deleted_prompts", "prompts.update.local"),
    ],
)
async def test_prompt_batch_policy_denial_makes_no_backend_call(method_name, action):
    local = FakeBatchLocalPromptService()
    server = FakeServerPromptService()
    policy = FakePolicyEnforcer.deny()
    scope = PromptScopeService(local, server, policy)

    with pytest.raises(PermissionError, match="blocked"):
        await getattr(scope, method_name)(
            mode="local", targets=(PromptBatchTarget(7, 3),)
        )

    assert policy.actions == [action]
    assert local.calls == []
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "action", "message"),
    [
        ("delete_prompts", "prompts.delete.local", "batch delete"),
        ("restore_deleted_prompts", "prompts.update.local", "batch restore"),
    ],
)
async def test_prompt_batch_missing_local_method_fails_after_one_policy_decision(
    method_name, action, message
):
    local = MissingBatchLocalPromptService()
    server = FakeServerPromptService()
    policy = FakePolicyEnforcer()
    scope = PromptScopeService(local, server, policy)

    with pytest.raises(ValueError, match=message):
        await getattr(scope, method_name)(
            mode="local", targets=(PromptBatchTarget(7, 3),)
        )

    assert policy.actions == [action]
    assert local.calls == []
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "action"),
    [
        ("delete_prompts", "prompts.delete.local"),
        ("restore_deleted_prompts", "prompts.update.local"),
    ],
)
async def test_prompt_batch_missing_local_backend_fails_after_one_policy_decision(
    method_name, action
):
    server = FakeServerPromptService()
    policy = FakePolicyEnforcer()
    scope = PromptScopeService(None, server, policy)

    with pytest.raises(ValueError, match="Local prompt backend is unavailable"):
        await getattr(scope, method_name)(
            mode="local", targets=(PromptBatchTarget(7, 3),)
        )

    assert policy.actions == [action]
    assert server.calls == []


def test_prompt_batch_scope_methods_have_no_post_return_normalizer_or_server_path():
    for method_name in ("delete_prompts", "restore_deleted_prompts"):
        source = inspect.getsource(getattr(PromptScopeService, method_name))
        assert "_normalize_prompt_record" not in source
        assert "server_service" not in source
