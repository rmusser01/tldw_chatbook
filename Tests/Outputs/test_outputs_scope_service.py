import pytest

from tldw_chatbook.Outputs_Interop.outputs_scope_service import OutputsScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeOutputsService:
    def __init__(self, source="server"):
        self.source = source
        self.calls = []

    async def list_templates(self, **kwargs):
        self.calls.append(("list_templates", kwargs))
        return {"items": [{"id": 5, "name": "Newsletter"}], "total": 1}

    async def get_template(self, template_id):
        self.calls.append(("get_template", template_id))
        return {"id": template_id, "name": "Newsletter"}

    async def create_template(self, **kwargs):
        self.calls.append(("create_template", kwargs))
        return {"id": 6, "name": kwargs["name"]}

    async def list_artifacts(self, **kwargs):
        self.calls.append(("list_artifacts", kwargs))
        return {"items": [{"id": 9, "title": "Digest"}], "total": 1}

    async def create_artifact(self, **kwargs):
        self.calls.append(("create_artifact", kwargs))
        return {"id": 10, "title": "Digest"}

    async def preview_template(self, template_id, **kwargs):
        self.calls.append(("preview_template", template_id, kwargs))
        return {"rendered": "# Digest", "format": "md"}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_outputs_scope_service_routes_server_templates_artifacts_and_preview():
    server = FakeOutputsService()
    policy = FakePolicyEnforcer()
    scope = OutputsScopeService(local_service=None, server_service=server, policy_enforcer=policy)

    templates = await scope.list_templates(mode="server", q="news")
    template = await scope.get_template(mode="server", template_id=5)
    artifact = await scope.create_artifact(mode="server", template_id=5, data={"title": "Digest"})
    preview = await scope.preview_template(mode="server", template_id=5, data={"title": "Digest"})

    assert templates["items"][0]["record_id"] == "server:output_template:5"
    assert templates["items"][0]["backend"] == "server"
    assert template["record_id"] == "server:output_template:5"
    assert artifact["record_id"] == "server:output_artifact:10"
    assert preview["backend"] == "server"
    assert server.calls == [
        ("list_templates", {"q": "news"}),
        ("get_template", 5),
        ("create_artifact", {"template_id": 5, "data": {"title": "Digest"}}),
        ("preview_template", 5, {"data": {"title": "Digest"}}),
    ]
    assert policy.calls == [
        "outputs.templates.list.server",
        "outputs.templates.detail.server",
        "outputs.artifacts.create.server",
        "outputs.render_jobs.launch.server",
    ]


@pytest.mark.asyncio
async def test_outputs_scope_service_honestly_rejects_local_mode_when_backend_missing():
    policy = FakePolicyEnforcer()
    scope = OutputsScopeService(local_service=None, server_service=FakeOutputsService(), policy_enforcer=policy)

    with pytest.raises(ValueError, match="Local outputs backend is unavailable"):
        await scope.list_templates(mode="local")

    assert policy.calls == ["outputs.templates.list.local"]


@pytest.mark.asyncio
async def test_outputs_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeOutputsService()
    scope = OutputsScopeService(
        local_service=None,
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.list_artifacts(mode="server")

    assert exc.value.reason_code == "wrong_source"
    assert server.calls == []


def test_outputs_scope_service_reports_known_unsupported_capabilities():
    scope = OutputsScopeService(local_service=None, server_service=FakeOutputsService())

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "outputs.local_backend.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_backend_unavailable",
            "user_message": "Local managed output templates, artifacts, and render jobs are not wired in the current Chatbook client.",
            "affected_action_ids": [
                "outputs.templates.list.local",
                "outputs.templates.detail.local",
                "outputs.templates.create.local",
                "outputs.templates.update.local",
                "outputs.templates.delete.local",
                "outputs.artifacts.list.local",
                "outputs.artifacts.detail.local",
                "outputs.artifacts.create.local",
                "outputs.artifacts.update.local",
                "outputs.artifacts.delete.local",
                "outputs.render_jobs.launch.local",
                "outputs.render_jobs.list.local",
                "outputs.render_jobs.detail.local",
                "outputs.render_jobs.observe.local",
            ],
        }
    ]
    assert server_report == [
        {
            "operation_id": "outputs.render_jobs.observe.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server output API supports synchronous template preview and artifact creation, but not first-class render-job listing, detail, or observation.",
            "affected_action_ids": [
                "outputs.render_jobs.list.server",
                "outputs.render_jobs.detail.server",
                "outputs.render_jobs.observe.server",
            ],
        }
    ]
