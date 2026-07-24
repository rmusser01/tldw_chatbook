"""Foundation tests for URL-based skill install (task 1 of the SDD plan).

Covers: the fail-closed policy registry entry for remote skill installs, the
public policy-enforcement passthrough on ``SkillsScopeService``, and the
``GitHubAPIClient.get_branches`` pagination fix (GitHub's branches endpoint
defaults to 30 results/page -- installers need the full branch list).
"""

import pytest


def test_policy_action_id_registered():
    # The engine denies unknown ids (fail-closed) -- pin that the new id
    # exists. Mirrors Tests/Skills/test_read_skill_file.py's
    # test_policy_action_id_registered idiom for skills.read_file.
    from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY

    assert "skills.install_remote.launch.local" in CAPABILITY_REGISTRY


def test_scope_service_public_enforce_passthrough():
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    calls = []

    class _Enforcer:
        def require_allowed(self, *, action_id):
            calls.append(action_id)

    scope = SkillsScopeService(
        local_service=None, server_service=None, policy_enforcer=_Enforcer()
    )
    scope.enforce_install_remote()
    assert calls == ["skills.install_remote.launch.local"]
    # No enforcer wired -> no-op, no raise (matches _enforce_policy semantics).
    SkillsScopeService(local_service=None, server_service=None).enforce_install_remote()


@pytest.mark.asyncio
async def test_get_branches_requests_full_page():
    from tldw_chatbook.Utils.github_api_client import GitHubAPIClient

    client = GitHubAPIClient(token="t")
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"name": "main"}]

    class _MockHttpClient:
        async def get(self, url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs.get("params")
            return _Resp()

    # Mirror Tests/Utils/test_github_api_client.py's idiom: set the private
    # cached-client attribute directly rather than patching the `client`
    # property.
    client._client = _MockHttpClient()
    branches = await client.get_branches("o", "r")
    assert branches == ["main"]
    assert captured["params"] == {"per_page": 100}
