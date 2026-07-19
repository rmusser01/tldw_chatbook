"""Tests for the Skills count seam (Task 1 of the Skills sub-project).

Covers ``LocalSkillsService.count_skills`` (the local backend's total managed
skills count -- trusted ``available_skills`` plus needs-review
``blocked_skills``) and ``SkillsScopeService.count_skills`` (the source-aware
passthrough the Library rail's ``Skills (N)`` row count seam will call).
"""

import pytest

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService


@pytest.mark.asyncio
async def test_count_skills_counts_managed_skills(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path, allow_untrusted_without_trust_service=True)
    assert await svc.count_skills() == 0
    await svc.create_skill(
        name="code-review",
        content="---\nname: code-review\ndescription: Review code\n---\nDo it.")
    await svc.create_skill(
        name="summarize",
        content="---\nname: summarize\ndescription: Summarize text\n---\nGo.")
    assert await svc.count_skills() == 2


@pytest.mark.asyncio
async def test_count_skills_includes_blocked_needs_review_skills(tmp_path):
    """The blocked_skills envelope's needs-review population must count too --
    per the spec, a skill pending trust review is still a managed skill even
    though ``get_context``'s ``available_skills`` (the trusted/invocable
    population) excludes it.

    Mirrors ``test_local_skills_service_without_trust_service_fails_closed_
    by_default`` in ``test_local_skills_service.py``: with no
    ``trust_service`` configured and ``allow_untrusted_without_trust_service``
    left at its default (``False``), a newly created skill is blocked
    (``trust_locked``) until reviewed/approved, landing in
    ``blocked_skills`` rather than ``available_skills``.
    """
    svc = LocalSkillsService(store_dir=tmp_path)
    await svc.create_skill(
        name="code-review",
        content="---\nname: code-review\ndescription: Review code\n---\nDo it.",
    )

    context = await svc.get_context()
    assert context["available_skills"] == []
    assert len(context["blocked_skills"]) == 1

    assert await svc.count_skills() == 1


class _FakeLocalSkillsService:
    def __init__(self, *, count: int):
        self._count = count
        self.calls = []

    async def count_skills(self, **kwargs):
        self.calls.append(kwargs)
        return self._count


class _FakePolicyEnforcer:
    def __init__(self):
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)


@pytest.mark.asyncio
async def test_scope_service_count_skills_routes_to_local_backend_and_returns_int():
    local = _FakeLocalSkillsService(count=3)
    scope = SkillsScopeService(local_service=local)

    result = await scope.count_skills(mode="local")

    assert result == 3
    assert isinstance(result, int)
    assert local.calls == [{}]


@pytest.mark.asyncio
async def test_scope_service_count_skills_enforces_context_list_action_id():
    policy = _FakePolicyEnforcer()
    scope = SkillsScopeService(local_service=_FakeLocalSkillsService(count=5), policy_enforcer=policy)

    result = await scope.count_skills(mode="local")

    assert result == 5
    assert policy.calls == ["skills.context.list.local"]


@pytest.mark.asyncio
async def test_scope_service_count_skills_reports_missing_local_backend_before_dispatch():
    scope = SkillsScopeService(server_service=_FakeLocalSkillsService(count=1))

    with pytest.raises(ValueError, match="Local skills backend is unavailable"):
        await scope.count_skills(mode="local")
