from unittest.mock import Mock

import pytest

from tldw_chatbook.RAG_Admin.server_rag_admin_service import ServerRAGAdminService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.tldw_api import (
    ChunkingTemplateCreateRequest,
    ChunkingTemplateDiagnosticsResponse,
    ChunkingTemplateListResponse,
    ChunkingTemplateResponse,
    ChunkingTemplateUpdateRequest,
    EmbeddingCollectionResponse,
    EmbeddingCollectionStatsResponse,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    async def list_chunking_templates(self, **kwargs):
        self.calls.append(("list_chunking_templates", kwargs))
        return ChunkingTemplateListResponse.model_validate(
            {
                "templates": [
                    {
                        "id": 1,
                        "uuid": "cb547cde-3720-4f1e-b3a7-90425ee6b38d",
                        "name": "server-demo",
                        "description": "Server demo",
                        "template_json": '{"chunking": {"method": "words", "config": {"max_size": 256}}}',
                        "is_builtin": False,
                        "tags": ["rag"],
                        "created_at": "2026-04-20T00:00:00Z",
                        "updated_at": "2026-04-20T00:00:00Z",
                        "version": 1,
                        "user_id": "u1",
                    }
                ],
                "total": 1,
            }
        )

    async def create_chunking_template(self, request_data):
        self.calls.append(("create_chunking_template", request_data))
        return ChunkingTemplateResponse.model_validate(
            {
                "id": 2,
                "uuid": "1ce34b1a-5d36-4d4f-b4f4-6cd77b31377e",
                "name": "created",
                "description": "Created template",
                "template_json": '{"chunking": {"method": "words", "config": {"max_size": 512}}}',
                "is_builtin": False,
                "tags": ["notes"],
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T00:00:00Z",
                "version": 1,
                "user_id": "u1",
            }
        )

    async def update_chunking_template(self, template_name, request_data):
        self.calls.append(("update_chunking_template", template_name, request_data))
        return ChunkingTemplateResponse.model_validate(
            {
                "id": 2,
                "uuid": "1ce34b1a-5d36-4d4f-b4f4-6cd77b31377e",
                "name": template_name,
                "description": "Updated template",
                "template_json": '{"chunking": {"method": "sentences", "config": {"max_size": 8}}}',
                "is_builtin": False,
                "tags": ["notes", "updated"],
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T01:00:00Z",
                "version": 2,
                "user_id": "u1",
            }
        )

    async def get_chunking_template_diagnostics(self):
        self.calls.append(("get_chunking_template_diagnostics",))
        return ChunkingTemplateDiagnosticsResponse(
            db_class="sqlite.MediaDatabase",
            capability="native",
            fallback_enabled=True,
            hint="hint",
        )

    async def list_embedding_collections(self):
        self.calls.append(("list_embedding_collections",))
        return [EmbeddingCollectionResponse(name="demo_collection", metadata={"provider": "openai"})]

    async def get_embedding_collection_stats(self, collection_name):
        self.calls.append(("get_embedding_collection_stats", collection_name))
        return EmbeddingCollectionStatsResponse(
            name=collection_name,
            count=3,
            embedding_dimension=1536,
            metadata={"provider": "openai"},
        )

    async def delete_embedding_collection(self, collection_name):
        self.calls.append(("delete_embedding_collection", collection_name))
        return None


@pytest.mark.asyncio
async def test_server_rag_admin_service_builds_requests_and_unwraps_models():
    client = FakeClient()
    service = ServerRAGAdminService(client=client)

    listed = await service.list_templates(include_builtin=False, include_custom=True, tags=["rag"], user_id="u1")
    created = await service.create_template(
        name="created",
        description="Created template",
        tags=["notes"],
        template={"chunking": {"method": "words", "config": {"max_size": 512}}},
        user_id="u1",
    )
    updated = await service.update_template(
        "created",
        description="Updated template",
        tags=["notes", "updated"],
        template={"chunking": {"method": "sentences", "config": {"max_size": 8}}},
    )
    diagnostics = await service.get_template_diagnostics()
    collections = await service.list_collections()
    stats = await service.get_collection_detail("demo_collection")
    await service.delete_collection("demo_collection")

    assert client.calls[0] == (
        "list_chunking_templates",
        {"include_builtin": False, "include_custom": True, "tags": ["rag"], "user_id": "u1"},
    )
    assert isinstance(client.calls[1][1], ChunkingTemplateCreateRequest)
    assert client.calls[1][1].name == "created"
    assert isinstance(client.calls[2][2], ChunkingTemplateUpdateRequest)
    assert client.calls[2][1] == "created"
    assert client.calls[3] == ("get_chunking_template_diagnostics",)
    assert client.calls[4] == ("list_embedding_collections",)
    assert client.calls[5] == ("get_embedding_collection_stats", "demo_collection")
    assert client.calls[6] == ("delete_embedding_collection", "demo_collection")

    assert listed[0]["name"] == "server-demo"
    assert created["name"] == "created"
    assert updated["description"] == "Updated template"
    assert diagnostics["capability"] == "native"
    assert collections[0]["name"] == "demo_collection"
    assert stats["count"] == 3


@pytest.mark.asyncio
async def test_server_rag_admin_service_enforces_policy_actions():
    client = FakeClient()
    policy = Mock()
    service = ServerRAGAdminService(client=client, policy_enforcer=policy)

    await service.list_templates(include_builtin=False, include_custom=True, tags=["rag"], user_id="u1")
    await service.create_template(
        name="created",
        description="Created template",
        tags=["notes"],
        template={"chunking": {"method": "words", "config": {"max_size": 512}}},
        user_id="u1",
    )
    await service.update_template(
        "created",
        description="Updated template",
        tags=["notes", "updated"],
        template={"chunking": {"method": "sentences", "config": {"max_size": 8}}},
    )
    await service.get_template_diagnostics()
    await service.list_collections()
    await service.get_collection_detail("demo_collection")
    await service.delete_collection("demo_collection")

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "rag.template.list.server",
        "rag.template.create.server",
        "rag.template.update.server",
        "rag.admin.observe.server",
        "rag.admin.list.server",
        "rag.admin.observe.server",
        "rag.admin.configure.server",
    ]


@pytest.mark.asyncio
async def test_server_rag_admin_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeClient()
    service = ServerRAGAdminService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_templates()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
