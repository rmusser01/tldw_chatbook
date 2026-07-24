"""Foundation tests for URL-based skill install (task 1 of the SDD plan).

Covers: the fail-closed policy registry entry for remote skill installs, the
public policy-enforcement passthrough on ``SkillsScopeService``, and the
``GitHubAPIClient.get_branches`` pagination fix (GitHub's branches endpoint
defaults to 30 results/page -- installers need the full branch list).
"""

import pytest
import httpx


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


from tldw_chatbook.Skills_Interop.skill_remote_fetch import (
    GITHUB_AUTH_HOSTS,
    REMOTE_FETCH_MAX_BYTES,
    DirectZipSource,
    GitHubZipSource,
    RemoteSkillError,
    classify_skill_source_url,
    fetch_zip_bytes,
    resolve_ref_and_subdir,
)


def test_classify_repo_root():
    src = classify_skill_source_url("https://github.com/obra/superpowers")
    assert src == GitHubZipSource(
        owner="obra", repo="superpowers", tree_tail=(), suggested_name="superpowers"
    )


def test_classify_tree_subdir():
    src = classify_skill_source_url(
        "https://github.com/obra/superpowers/tree/main/skills/brainstorming"
    )
    assert isinstance(src, GitHubZipSource)
    assert src.tree_tail == ("main", "skills", "brainstorming")
    assert src.suggested_name == "brainstorming"


def test_classify_release_asset_and_direct_zip():
    rel = classify_skill_source_url(
        "https://github.com/o/r/releases/download/v1/my-skill.zip"
    )
    assert rel == DirectZipSource(
        url="https://github.com/o/r/releases/download/v1/my-skill.zip",
        github_auth=True,
        suggested_name="my-skill",
    )
    direct = classify_skill_source_url("https://example.com/pkg/my-skill.zip")
    assert direct.github_auth is False and direct.suggested_name == "my-skill"


@pytest.mark.parametrize("bad", [
    "http://github.com/o/r",                    # http
    "https://example.com/not-a-zip",            # non-github non-zip
    "git@github.com:o/r.git",                   # git protocol
    "https://github.com/onlyowner",             # no repo
    "ftp://example.com/x.zip",
])
def test_classify_rejects(bad):
    with pytest.raises(RemoteSkillError):
        classify_skill_source_url(bad)


@pytest.mark.asyncio
async def test_ref_split_single_segment_is_ref():
    src = classify_skill_source_url("https://github.com/o/r/tree/v1.2")
    async def _boom(o, r):  # must NOT be called for single-segment tails
        raise AssertionError("branch listing not needed")
    assert await resolve_ref_and_subdir(src, _boom) == ("v1.2", "")


@pytest.mark.asyncio
async def test_ref_split_longest_prefix_branch_wins():
    src = classify_skill_source_url("https://github.com/o/r/tree/feature/foo/skills/x")
    async def _branches(o, r):
        return ["main", "feature/foo", "feature"]
    ref, subdir = await resolve_ref_and_subdir(src, _branches)
    assert (ref, subdir) == ("feature/foo", "skills/x")


@pytest.mark.asyncio
async def test_ref_split_no_match_falls_back_to_first_segment():
    src = classify_skill_source_url("https://github.com/o/r/tree/main/skills/x")
    async def _branches(o, r):
        return ["dev"]  # capped/truncated list without the real branch
    assert await resolve_ref_and_subdir(src, _branches) == ("main", "skills/x")


@pytest.mark.asyncio
async def test_ref_split_api_failure_falls_back():
    src = classify_skill_source_url("https://github.com/o/r/tree/main/skills/x")
    async def _branches(o, r):
        raise RuntimeError("offline")
    assert await resolve_ref_and_subdir(src, _branches) == ("main", "skills/x")


_PUB = lambda host: ["93.184.216.34"]          # public resolver


def _transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_fetch_happy_path_streams_bytes():
    def handler(request):
        return httpx.Response(200, content=b"PK\x03\x04zipbytes")
    out = await fetch_zip_bytes(
        "https://api.github.com/repos/o/r/zipball/HEAD",
        transport=_transport(handler), resolver=_PUB,
    )
    assert out.startswith(b"PK")


@pytest.mark.asyncio
async def test_private_and_mixed_resolution_rejected():
    with pytest.raises(RemoteSkillError):
        await fetch_zip_bytes("https://internal.example/x.zip",
                              transport=_transport(lambda r: httpx.Response(200)),
                              resolver=lambda h: ["10.0.0.5"])
    with pytest.raises(RemoteSkillError):  # mixed public+private -> reject
        await fetch_zip_bytes("https://mixed.example/x.zip",
                              transport=_transport(lambda r: httpx.Response(200)),
                              resolver=lambda h: ["93.184.216.34", "fd00::1"])
    with pytest.raises(RemoteSkillError):  # IP-literal host
        await fetch_zip_bytes("https://169.254.169.254/x.zip",
                              transport=_transport(lambda r: httpx.Response(200)),
                              resolver=_PUB)


@pytest.mark.asyncio
async def test_redirect_hop_revalidated_and_capped():
    def handler(request):
        host = request.url.host
        if host == "a.example":
            return httpx.Response(302, headers={"location": "https://b.example/x.zip"})
        if host == "b.example":
            return httpx.Response(302, headers={"location": "http://c.example/x.zip"})
        raise AssertionError("unexpected host")
    with pytest.raises(RemoteSkillError):   # https->http hop rejected
        await fetch_zip_bytes("https://a.example/x.zip",
                              transport=_transport(handler), resolver=_PUB)

    def loop_handler(request):
        return httpx.Response(302, headers={"location": str(request.url)})
    with pytest.raises(RemoteSkillError):   # >3 hops
        await fetch_zip_bytes("https://a.example/x.zip",
                              transport=_transport(loop_handler), resolver=_PUB)

    def private_hop(request):
        if request.url.host == "a.example":
            return httpx.Response(302, headers={"location": "https://evil.internal/x.zip"})
        raise AssertionError
    with pytest.raises(RemoteSkillError):   # hop host resolves private
        await fetch_zip_bytes(
            "https://a.example/x.zip", transport=_transport(private_hop),
            resolver=lambda h: ["10.1.1.1"] if h == "evil.internal" else ["93.184.216.34"],
        )


@pytest.mark.asyncio
async def test_auth_scoped_to_github_family():
    seen = {}
    def handler(request):
        seen[request.url.host] = request.headers.get("authorization")
        if request.url.host == "api.github.com":
            return httpx.Response(
                302, headers={"location": "https://codeload.github.com/o/r/zip/HEAD"})
        if request.url.host == "codeload.github.com":
            return httpx.Response(
                302, headers={"location": "https://cdn.example.com/x.zip"})
        return httpx.Response(200, content=b"PK\x03\x04")
    await fetch_zip_bytes("https://api.github.com/repos/o/r/zipball/HEAD",
                          token="SECRET", transport=_transport(handler), resolver=_PUB)
    assert seen["api.github.com"] == "token SECRET"
    assert seen["codeload.github.com"] == "token SECRET"
    assert seen["cdn.example.com"] is None    # STRIPPED off-family


@pytest.mark.asyncio
async def test_stream_cap_aborts():
    big = b"x" * (REMOTE_FETCH_MAX_BYTES + 1024)
    def handler(request):
        return httpx.Response(200, content=big)
    with pytest.raises(RemoteSkillError, match="too large"):
        await fetch_zip_bytes("https://example.com/x.zip",
                              transport=_transport(handler), resolver=_PUB)
