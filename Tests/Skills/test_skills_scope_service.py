import threading

import pytest

from tldw_chatbook.Library.library_content_evidence import LibraryContentEvidence
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError
from tldw_chatbook.tldw_api.skills_schemas import SkillContextPayload, SkillSummary


class FakeSkillsService:
    def __init__(self):
        self.calls = []

    async def list_skills(self, **kwargs):
        self.calls.append(("list_skills", kwargs))
        return {
            "skills": [{"name": "summarize-notes", "user_invocable": True}],
            "total": 1,
        }

    async def get_context(self):
        self.calls.append(("get_context",))
        return {
            "available_skills": [{"name": "summarize-notes"}],
            "context_text": "- summarize-notes",
        }

    async def get_skill(self, skill_name):
        self.calls.append(("get_skill", skill_name))
        return {"id": "skill-1", "name": skill_name, "version": 1}

    async def import_skill(self, **kwargs):
        self.calls.append(("import_skill", kwargs))
        return {"id": "skill-2", "name": kwargs["name"], "version": 1}

    async def execute_skill(self, skill_name, **kwargs):
        self.calls.append(("execute_skill", skill_name, kwargs))
        return {
            "skill_name": skill_name,
            "rendered_prompt": "Summarize note-1",
            "execution_mode": "inline",
        }


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
async def test_skills_scope_service_routes_server_operations_and_normalizes_records():
    server = FakeSkillsService()
    policy = FakePolicyEnforcer()
    scope = SkillsScopeService(server_service=server, policy_enforcer=policy)

    listed = await scope.list_skills(mode="server", include_hidden=True)
    context = await scope.get_context(mode="server")
    fetched = await scope.get_skill("summarize-notes", mode="server")
    imported = await scope.import_skill(
        mode="server", name="rewrite-draft", content="# Skill", overwrite=True
    )
    executed = await scope.execute_skill(
        "summarize-notes", mode="server", args="note-1"
    )

    assert listed["skills"][0]["record_id"] == "server:skill:summarize-notes"
    assert context["available_skills"][0]["record_id"] == "server:skill:summarize-notes"
    assert fetched["record_id"] == "server:skill:summarize-notes"
    assert imported["record_id"] == "server:skill:rewrite-draft"
    assert executed["record_id"] == "server:skill_execution:summarize-notes"
    assert server.calls == [
        ("list_skills", {"include_hidden": True}),
        ("get_context",),
        ("get_skill", "summarize-notes"),
        (
            "import_skill",
            {"name": "rewrite-draft", "content": "# Skill", "overwrite": True},
        ),
        ("execute_skill", "summarize-notes", {"args": "note-1"}),
    ]
    assert policy.calls == [
        "skills.list.server",
        "skills.context.list.server",
        "skills.detail.server",
        "skills.import.launch.server",
        "skills.execute.launch.server",
    ]


@pytest.mark.asyncio
async def test_skills_scope_service_routes_local_operations_without_server_dispatch():
    local = FakeSkillsService()
    server = FakeSkillsService()
    policy = FakePolicyEnforcer()
    scope = SkillsScopeService(
        local_service=local, server_service=server, policy_enforcer=policy
    )

    listed = await scope.list_skills(mode="local", include_hidden=True)
    context = await scope.get_context(mode="local")
    fetched = await scope.get_skill("summarize-notes", mode="local")
    imported = await scope.import_skill(
        mode="local", name="rewrite-draft", content="# Skill", overwrite=True
    )
    executed = await scope.execute_skill("summarize-notes", mode="local", args="note-1")

    assert listed["backend"] == "local"
    assert listed["skills"][0]["record_id"] == "local:skill:summarize-notes"
    assert context["available_skills"][0]["record_id"] == "local:skill:summarize-notes"
    assert fetched["record_id"] == "local:skill:summarize-notes"
    assert imported["record_id"] == "local:skill:rewrite-draft"
    assert executed["record_id"] == "local:skill_execution:summarize-notes"
    assert local.calls == [
        ("list_skills", {"include_hidden": True}),
        ("get_context",),
        ("get_skill", "summarize-notes"),
        (
            "import_skill",
            {"name": "rewrite-draft", "content": "# Skill", "overwrite": True},
        ),
        ("execute_skill", "summarize-notes", {"args": "note-1"}),
    ]
    assert server.calls == []
    assert policy.calls == [
        "skills.list.local",
        "skills.context.list.local",
        "skills.detail.local",
        "skills.import.launch.local",
        "skills.execute.launch.local",
    ]


@pytest.mark.asyncio
async def test_skills_scope_service_preserves_blocked_local_context_trust_fields():
    class FakeLocalSkillsService(FakeSkillsService):
        async def get_context(self):
            self.calls.append(("get_context",))
            return {
                "available_skills": [],
                "blocked_skills": [
                    {
                        "name": "demo-skill",
                        "trust_status": "quarantined_modified",
                        "trust_reason_code": "skill_modified",
                        "trust_blocked": True,
                        "trust_changed_files": ["SKILL.md"],
                        "trust_manifest_generation": 2,
                        "trust_last_verified_at": "2026-06-25T00:00:00+00:00",
                    }
                ],
                "context_text": "",
            }

    local = FakeLocalSkillsService()
    scope = SkillsScopeService(local_service=local, server_service=FakeSkillsService())

    context = await scope.get_context(mode="local")

    blocked = context["blocked_skills"][0]
    assert blocked["backend"] == "local"
    assert blocked["record_id"] == "local:skill:demo-skill"
    assert blocked["trust_status"] == "quarantined_modified"
    assert blocked["trust_reason_code"] == "skill_modified"
    assert blocked["trust_blocked"] is True
    assert blocked["trust_changed_files"] == ["SKILL.md"]
    assert blocked["trust_manifest_generation"] == 2
    assert blocked["trust_last_verified_at"] == "2026-06-25T00:00:00+00:00"
    assert context["available_skills"] == []


@pytest.mark.asyncio
async def test_skills_user_content_evidence_uses_only_available_managed_skills():
    class ContextService:
        def __init__(self, context):
            self.context = context

        async def get_context(self):
            return self.context

    blocked_only = ContextService(
        {
            "available_skills": [],
            "blocked_skills": [
                {"name": "quarantined", "trust_status": "quarantined_modified"}
            ],
        }
    )
    service = SkillsScopeService(local_service=blocked_only)
    result = await service.get_library_user_content_evidence(mode="local")
    assert type(result) is LibraryContentEvidence
    assert result is LibraryContentEvidence.EMPTY

    user_skill = ContextService({"available_skills": [{"name": "user-skill"}]})
    service = SkillsScopeService(local_service=user_skill)
    assert (
        await service.get_library_user_content_evidence(mode="local")
        is LibraryContentEvidence.HAS_USER_CONTENT
    )


@pytest.mark.asyncio
async def test_skills_user_content_evidence_runs_real_local_scan_off_loop(
    tmp_path, monkeypatch
):
    local = LocalSkillsService(
        store_dir=tmp_path,
        allow_untrusted_without_trust_service=True,
    )
    await local.create_skill(
        name="user-skill",
        content="---\ndescription: User skill\n---\n# User skill\n",
    )
    loop_thread = threading.get_ident()
    scan_threads = []
    load_index = local._load_index

    def tracked_load_index():
        scan_threads.append(threading.get_ident())
        return load_index()

    monkeypatch.setattr(local, "_load_index", tracked_load_index)
    service = SkillsScopeService(local_service=local)

    assert (
        await service.get_library_user_content_evidence(mode="local")
        is LibraryContentEvidence.HAS_USER_CONTENT
    )
    assert scan_threads
    assert all(thread_id != loop_thread for thread_id in scan_threads)


@pytest.mark.asyncio
async def test_skills_local_evidence_owner_has_no_bundled_or_sample_population(
    tmp_path,
):
    local = LocalSkillsService(
        store_dir=tmp_path,
        allow_untrusted_without_trust_service=True,
    )
    assert local.get_library_user_content_evidence_context() == {"available_skills": []}

    await local.import_skill(
        name="user-import",
        content="---\ndescription: Imported by the user\n---\n# User import\n",
    )
    context = local.get_library_user_content_evidence_context()
    assert [item["name"] for item in context["available_skills"]] == ["user-import"]
    assert all(
        key not in context["available_skills"][0]
        for key in ("bundled", "is_sample", "source")
    )
    assert (
        await SkillsScopeService(local_service=local).get_library_user_content_evidence(
            mode="local"
        )
        is LibraryContentEvidence.HAS_USER_CONTENT
    )


@pytest.mark.asyncio
async def test_skills_user_content_evidence_accepts_server_user_skill_and_fails_closed():
    class ContextService:
        def __init__(self, context):
            self.context = context

        async def get_context(self):
            return self.context

    service = SkillsScopeService(
        server_service=ContextService(
            {"available_skills": [{"name": "user-skill"}], "blocked_skills": []}
        )
    )
    assert (
        await service.get_library_user_content_evidence(mode="server")
        is LibraryContentEvidence.UNKNOWN
    )

    production_context = SkillContextPayload(
        available_skills=[
            SkillSummary(
                name="schema-skill",
                user_invocable=True,
                disable_model_invocation=False,
                context="inline",
            )
        ],
        context_text="schema-skill",
    )
    service = SkillsScopeService(server_service=ContextService(production_context))
    assert (
        await service.get_library_user_content_evidence(mode="server")
        is LibraryContentEvidence.UNKNOWN
    )

    service = SkillsScopeService(server_service=ContextService({"skills": []}))
    assert (
        await service.get_library_user_content_evidence(mode="server")
        is LibraryContentEvidence.UNKNOWN
    )


@pytest.mark.asyncio
async def test_skills_scope_service_reports_missing_local_backend_before_dispatch():
    server = FakeSkillsService()
    scope = SkillsScopeService(
        server_service=server, policy_enforcer=FakePolicyEnforcer()
    )

    with pytest.raises(ValueError, match="Local skills backend is unavailable"):
        await scope.list_skills(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_skills_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeSkillsService()
    scope = SkillsScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_auth_required"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.list_skills(mode="server")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_skills_scope_service_reports_known_unsupported_capabilities():
    scope = SkillsScopeService(server_service=FakeSkillsService())
    local_scope = SkillsScopeService(
        local_service=FakeSkillsService(), server_service=FakeSkillsService()
    )

    assert scope.list_unsupported_capabilities(mode="server") == []
    assert local_scope.list_unsupported_capabilities(mode="local") == []
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "skills.local_backend_unavailable",
            "source": "local",
            "supported": False,
            "reason_code": "local_backend_unavailable",
            "user_message": "Local skills backend is unavailable.",
            "affected_action_ids": [],
        }
    ]
