import pytest

from tldw_chatbook.RAG_Admin.rag_admin_scope_service import RAGAdminScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeLocalService:
    def __init__(self, templates=None, collection_detail=None):
        self.templates = templates or []
        self.collection_detail = collection_detail or {}
        self.calls = []

    def list_templates(self, **kwargs):
        self.calls.append(("list_templates", kwargs))
        return list(self.templates)

    def get_collection_detail(self, collection_name):
        self.calls.append(("get_collection_detail", collection_name))
        return dict(self.collection_detail[collection_name])

    def reprocess_media(self, media_id, **options):
        self.calls.append(("reprocess_media", media_id, options))
        return {"backend": "local", "media_id": media_id, "status": "queued", "options": options}


class FakeServerService:
    def __init__(self, templates=None, collection_detail=None):
        self.templates = templates or []
        self.collection_detail = collection_detail or {}
        self.calls = []

    async def list_templates(self, **kwargs):
        self.calls.append(("list_templates", kwargs))
        return list(self.templates)

    async def get_collection_detail(self, collection_name):
        self.calls.append(("get_collection_detail", collection_name))
        return dict(self.collection_detail[collection_name])

    async def reprocess_media(self, media_id, **options):
        self.calls.append(("reprocess_media", media_id, options))
        return {"backend": "server", "media_id": media_id, "status": "queued", "options": options}


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
async def test_scope_service_routes_template_list_by_backend():
    local = FakeLocalService(
        templates=[
            {
                "id": 7,
                "name": "local-demo",
                "description": "Local demo",
                "template_json": '{"chunking": {"method": "words", "config": {"max_size": 100}}}',
                "is_system": 0,
            }
        ]
    )
    server = FakeServerService(
        templates=[
            {
                "id": 8,
                "uuid": "cb547cde-3720-4f1e-b3a7-90425ee6b38d",
                "name": "server-demo",
                "description": "Server demo",
                "template_json": '{"chunking": {"method": "sentences", "config": {"max_size": 8}}}',
                "is_builtin": False,
                "tags": ["rag"],
                "version": 1,
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T00:00:00Z",
            }
        ]
    )
    scope = RAGAdminScopeService(local_service=local, server_service=server)

    local_records = await scope.list_templates(mode="local")
    server_records = await scope.list_templates(mode="server")

    assert local_records[0]["backend"] == "local"
    assert local_records[0]["record_id"] == "local:chunking_template:local-demo"
    assert server_records[0]["backend"] == "server"
    assert server_records[0]["tags"] == ["rag"]


@pytest.mark.asyncio
async def test_rag_admin_scope_service_denies_server_template_listing_in_local_mode():
    policy_enforcer = FakePolicyEnforcer.deny("wrong_source")
    scope = RAGAdminScopeService(
        local_service=FakeLocalService(),
        server_service=FakeServerService(),
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.list_templates(mode="server")

    assert exc.value.reason_code == "wrong_source"
    assert policy_enforcer.calls == ["rag.template.list.server"]


@pytest.mark.asyncio
async def test_scope_service_uses_stats_endpoint_for_server_collection_detail():
    server = FakeServerService(
        collection_detail={
            "demo": {
                "name": "demo",
                "count": 3,
                "embedding_dimension": 1536,
                "metadata": {"provider": "openai"},
            }
        }
    )
    scope = RAGAdminScopeService(local_service=FakeLocalService(), server_service=server)

    detail = await scope.get_collection_detail(mode="server", collection_name="demo")

    assert detail["backend"] == "server"
    assert detail["name"] == "demo"
    assert detail["count"] == 3
    assert server.calls == [("get_collection_detail", "demo")]


@pytest.mark.asyncio
async def test_scope_service_routes_reprocess_media_as_rag_admin_launch():
    local = FakeLocalService()
    server = FakeServerService()
    policy_enforcer = FakePolicyEnforcer()
    scope = RAGAdminScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    local_result = await scope.reprocess_media(
        mode="local",
        media_id=7,
        generate_embeddings=True,
    )
    server_result = await scope.reprocess_media(
        mode="server",
        media_id=8,
        chunking_template_name="server-demo",
    )

    assert local_result["backend"] == "local"
    assert server_result["backend"] == "server"
    assert local.calls == [("reprocess_media", 7, {"generate_embeddings": True})]
    assert server.calls == [("reprocess_media", 8, {"chunking_template_name": "server-demo"})]
    assert policy_enforcer.calls == [
        "rag.admin.launch.local",
        "rag.admin.launch.server",
    ]


def test_scope_service_reports_known_rag_admin_capability_gaps():
    scope = RAGAdminScopeService(
        local_service=FakeLocalService(),
        server_service=FakeServerService(),
    )

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "rag.templates.apply.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local chunking templates can be listed and edited, but the local admin seam does not apply templates to arbitrary text yet.",
            "affected_action_ids": ["rag.admin.launch.local"],
        },
        {
            "operation_id": "rag.templates.tags.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local chunking templates do not persist or filter server-style tags yet.",
            "affected_action_ids": [
                "rag.template.create.local",
                "rag.template.list.local",
                "rag.template.update.local",
            ],
        },
        {
            "operation_id": "rag.collections.export.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local embedding collection export is not exposed by the current RAG admin seam.",
            "affected_action_ids": ["rag.admin.observe.local"],
        },
    ]
    assert server_report == [
        {
            "operation_id": "rag.collections.create.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server embedding admin contract lists, inspects, deletes, and reprocesses collections, but does not expose direct collection creation.",
            "affected_action_ids": ["rag.admin.configure.server", "rag.admin.launch.server"],
        },
        {
            "operation_id": "rag.collections.export.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server embedding admin contract does not expose embedding collection export.",
            "affected_action_ids": ["rag.admin.observe.server"],
        },
    ]
