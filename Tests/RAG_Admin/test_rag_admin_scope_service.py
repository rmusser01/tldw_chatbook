import pytest

from tldw_chatbook.RAG_Admin.rag_admin_scope_service import RAGAdminScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakePolicyEnforcer:
    def __init__(self):
        self.actions = []

    def require_allowed(self, *, action_id):
        self.actions.append(action_id)


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

    def export_collection(self, collection_name, **options):
        self.calls.append(("export_collection", collection_name, options))
        return {
            "backend": "local",
            "name": collection_name,
            "items": [{"id": "a", "document": "alpha"}],
        }


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

    async def create_collection(self, **kwargs):
        self.calls.append(("create_collection", kwargs))
        return {
            "name": kwargs["name"],
            "metadata": {
                "provider": kwargs.get("provider"),
                "embedding_model": kwargs.get("embedding_model"),
                **dict(kwargs.get("metadata") or {}),
            },
        }

    async def reprocess_media(self, media_id, **options):
        self.calls.append(("reprocess_media", media_id, options))
        return {"backend": "server", "media_id": media_id, "status": "queued", "options": options}

    async def get_media_embeddings_status(self, media_id):
        self.calls.append(("get_media_embeddings_status", media_id))
        return {"media_id": media_id, "has_embeddings": True, "embedding_count": 2}

    async def generate_media_embeddings(self, media_id, **options):
        self.calls.append(("generate_media_embeddings", media_id, options))
        return {"media_id": media_id, "status": "accepted", "job_id": "job-7"}

    async def generate_media_embeddings_batch(self, **options):
        self.calls.append(("generate_media_embeddings_batch", options))
        return {"status": "accepted", "job_ids": ["job-7"], "submitted": 1}

    async def search_media_embeddings(self, **options):
        self.calls.append(("search_media_embeddings", options))
        return {"results": [{"id": "chunk-1", "metadata": {"media_id": "7"}}], "count": 1}

    async def delete_media_embeddings(self, media_id):
        self.calls.append(("delete_media_embeddings", media_id))
        return {"status": "success"}

    async def get_media_embedding_job(self, job_id):
        self.calls.append(("get_media_embedding_job", job_id))
        return {"uuid": job_id, "status": "completed"}

    async def list_media_embedding_jobs(self, **options):
        self.calls.append(("list_media_embedding_jobs", options))
        return {"data": [{"uuid": "job-7", "status": "completed"}], "pagination": {"count": 1}}


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
async def test_scope_service_routes_server_collection_create_as_rag_admin_configure():
    server = FakeServerService()
    policy_enforcer = FakePolicyEnforcer()
    scope = RAGAdminScopeService(
        local_service=FakeLocalService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    created = await scope.create_collection(
        mode="server",
        name="new_collection",
        metadata={"purpose": "tests"},
        embedding_model="text-embedding-3-small",
        provider="openai",
    )

    assert created["backend"] == "server"
    assert created["name"] == "new_collection"
    assert created["metadata"]["purpose"] == "tests"
    assert server.calls == [
        (
            "create_collection",
            {
                "name": "new_collection",
                "metadata": {"purpose": "tests"},
                "embedding_model": "text-embedding-3-small",
                "provider": "openai",
            },
        )
    ]
    assert policy_enforcer.calls == ["rag.admin.configure.server"]


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


@pytest.mark.asyncio
async def test_scope_service_routes_server_media_embedding_admin_and_blocks_local_mode():
    server = FakeServerService()
    policy_enforcer = FakePolicyEnforcer()
    scope = RAGAdminScopeService(
        local_service=FakeLocalService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    status = await scope.get_media_embeddings_status(mode="server", media_id=7)
    generated = await scope.generate_media_embeddings(
        mode="server",
        media_id=7,
        embedding_model="text-embedding-3-small",
        chunk_size=512,
    )
    batch = await scope.generate_media_embeddings_batch(mode="server", media_ids=[7])
    search = await scope.search_media_embeddings(mode="server", query="alpha", filters={"media_id": "7"})
    deleted = await scope.delete_media_embeddings(mode="server", media_id=7)
    job = await scope.get_media_embedding_job(mode="server", job_id="job-7")
    jobs = await scope.list_media_embedding_jobs(mode="server", status="completed")

    assert status["backend"] == "server"
    assert generated["backend"] == "server"
    assert batch["backend"] == "server"
    assert search["backend"] == "server"
    assert deleted["backend"] == "server"
    assert job["backend"] == "server"
    assert jobs["backend"] == "server"
    assert server.calls == [
        ("get_media_embeddings_status", 7),
        ("generate_media_embeddings", 7, {"embedding_model": "text-embedding-3-small", "chunk_size": 512}),
        ("generate_media_embeddings_batch", {"media_ids": [7]}),
        ("search_media_embeddings", {"query": "alpha", "filters": {"media_id": "7"}}),
        ("delete_media_embeddings", 7),
        ("get_media_embedding_job", "job-7"),
        ("list_media_embedding_jobs", {"status": "completed"}),
    ]
    assert policy_enforcer.calls == [
        "rag.media_embeddings.status.server",
        "rag.media_embeddings.create.server",
        "rag.media_embeddings.create.server",
        "rag.media_embeddings.search.server",
        "rag.media_embeddings.delete.server",
        "rag.media_embedding_jobs.detail.server",
        "rag.media_embedding_jobs.list.server",
    ]

    with pytest.raises(ValueError, match="server-only"):
        await scope.get_media_embeddings_status(mode="local", media_id=7)


@pytest.mark.asyncio
async def test_scope_service_routes_local_collection_export_as_rag_admin_observe():
    local = FakeLocalService()
    policy_enforcer = FakePolicyEnforcer()
    scope = RAGAdminScopeService(
        local_service=local,
        server_service=FakeServerService(),
        policy_enforcer=policy_enforcer,
    )

    exported = await scope.export_collection(
        mode="local",
        collection_name="demo",
        include_embeddings=False,
    )

    assert exported["backend"] == "local"
    assert exported["name"] == "demo"
    assert local.calls == [("export_collection", "demo", {"include_embeddings": False})]
    assert policy_enforcer.calls == ["rag.admin.observe.local"]


@pytest.mark.asyncio
async def test_scope_service_routes_server_collection_export_when_adapter_provides_it():
    class ServerServiceWithExport(FakeServerService):
        async def export_collection(self, collection_name, **options):
            self.calls.append(("export_collection", collection_name, options))
            return {"backend": "server", "name": collection_name}

    server = ServerServiceWithExport()
    policy_enforcer = FakePolicyEnforcer()
    scope = RAGAdminScopeService(
        local_service=FakeLocalService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    exported = await scope.export_collection(
        mode="server",
        collection_name="demo",
        include_embeddings=False,
        limit=50,
    )

    assert exported == {"backend": "server", "name": "demo"}
    assert server.calls == [("export_collection", "demo", {"include_embeddings": False, "limit": 50})]
    assert policy_enforcer.calls == ["rag.admin.observe.server"]


def test_scope_service_reports_known_rag_admin_capability_gaps():
    scope = RAGAdminScopeService(
        local_service=FakeLocalService(),
        server_service=FakeServerService(),
    )

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "rag.media_embeddings.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Server-style per-media embedding status, generation, search, deletion, and job tracking are not exposed by the local RAG admin seam yet.",
            "affected_action_ids": [],
        },
    ]
    assert server_report == [
        {
            "operation_id": "rag.collections.export.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server embedding admin contract does not expose embedding collection export.",
            "affected_action_ids": ["rag.admin.observe.server"],
        },
    ]
