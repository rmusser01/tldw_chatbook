"""Foundation tests for URL-based skill install (task 1 of the SDD plan).

Covers: the fail-closed policy registry entry for remote skill installs, the
public policy-enforcement passthrough on ``SkillsScopeService``, and the
``GitHubAPIClient.get_branches`` pagination fix (GitHub's branches endpoint
defaults to 30 results/page -- installers need the full branch list).
"""

import io
import zipfile

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
    re_root_skill_zip,
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


def _zipball(entries, wrapper="repo-abc123/"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries:
            z.writestr(wrapper + name, data)
    return buf.getvalue()


def test_reroot_subdir_install():
    data = _zipball([
        ("skills/brainstorm/SKILL.md", "---\nname: brainstorm\n---\nbody"),
        ("skills/brainstorm/references/api.md", "# api"),
        ("skills/other/SKILL.md", "x"),
        ("README.md", "top"),
    ])
    out, name = re_root_skill_zip(data, subdir="skills/brainstorm", suggested_name="brainstorm")
    with zipfile.ZipFile(io.BytesIO(out)) as z:
        assert set(z.namelist()) == {"SKILL.md", "references/api.md"}
    assert name == "brainstorm"


def test_reroot_root_is_skill_and_unwrapped_asset():
    wrapped = _zipball([("SKILL.md", "body"), ("notes.md", "n")])
    out, _ = re_root_skill_zip(wrapped, subdir="", suggested_name="repo")
    with zipfile.ZipFile(io.BytesIO(out)) as z:
        assert "SKILL.md" in z.namelist()
    unwrapped = _zipball([("SKILL.md", "body")], wrapper="")   # release-asset shape
    out2, _ = re_root_skill_zip(unwrapped, subdir="", suggested_name="asset")
    with zipfile.ZipFile(io.BytesIO(out2)) as z:
        assert "SKILL.md" in z.namelist()


def test_reroot_single_candidate_auto_installs():
    data = _zipball([("skills/only/SKILL.md", "body"), ("README.md", "r")])
    out, name = re_root_skill_zip(data, subdir="", suggested_name="repo")
    assert name == "only"


def test_reroot_many_candidates_error_lists_paths():
    entries = [(f"skills/s{i}/SKILL.md", "b") for i in range(25)]
    data = _zipball(entries)
    with pytest.raises(RemoteSkillError) as exc:
        re_root_skill_zip(data, subdir="", suggested_name="repo")
    msg = str(exc.value)
    assert "skills/s0" in msg and "subdirectory URL" in msg
    assert msg.count("skills/s") <= 20


def test_reroot_zero_candidates_and_corrupt():
    with pytest.raises(RemoteSkillError, match="No SKILL.md"):
        re_root_skill_zip(_zipball([("README.md", "r")]), subdir="", suggested_name="x")
    with pytest.raises(RemoteSkillError):
        re_root_skill_zip(b"not a zip", subdir="", suggested_name="x")


def test_reroot_lying_header_bomb_aborts_bounded():
    # Forge a member whose declared size is tiny but whose stream is huge:
    # simplest honest proxy — a member whose ACTUAL content exceeds the
    # per-file cap; the bounded reader must abort during synthesis.
    from tldw_chatbook.tldw_api.skills_schemas import MAX_SUPPORTING_FILE_BYTES
    big = "x" * (MAX_SUPPORTING_FILE_BYTES + 100)
    data = _zipball([("SKILL.md", "body"), ("references/huge.md", big)])
    with pytest.raises(RemoteSkillError, match="too large"):
        re_root_skill_zip(data, subdir="", suggested_name="repo")


def test_reroot_rejects_zip_slip_relative_member():
    # A member name that -- once the skill_root prefix is stripped -- still
    # carries ".." segments. Raw member selection is a naive
    # ``startswith(skill_root)`` string match, so this still reaches
    # synthesis; the per-member path validator must catch it there.
    data = _zipball([
        ("SKILL.md", "body"),
        ("refs/../../evil.md", "evil"),
    ])
    # Sanity: writestr does not normalize the traversal out of the name.
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert any(n.endswith("refs/../../evil.md") for n in z.namelist())
    with pytest.raises(RemoteSkillError, match="Unsafe file path") as exc_info:
        re_root_skill_zip(data, subdir="", suggested_name="repo")
    assert "evil.md" in str(exc_info.value)


def test_reroot_prunes_junk_before_count_cap():
    # Junk-heavy but otherwise-legitimate bundles must not be falsely
    # rejected by the count cap: the cap applies to the PRUNED member list,
    # matching LocalSkillsService.import_skill_file's own ordering.
    from tldw_chatbook.tldw_api.skills_schemas import MAX_SUPPORTING_FILES_COUNT

    entries = [
        ("SKILL.md", "body"),
        ("references/a.md", "a"),
        ("references/b.md", "b"),
        ("scripts/run.sh", "run"),
    ]
    entries += [
        (f"__pycache__/x{i}.pyc", "junk")
        for i in range(MAX_SUPPORTING_FILES_COUNT + 5)
    ]
    data = _zipball(entries)
    out, name = re_root_skill_zip(data, subdir="", suggested_name="repo")
    with zipfile.ZipFile(io.BytesIO(out)) as z:
        names = z.namelist()
    assert not any("__pycache__" in n for n in names)
    assert set(names) == {
        "SKILL.md", "references/a.md", "references/b.md", "scripts/run.sh",
    }


def _reroot_zip_with_understated_file_size(declared: int = 5) -> bytes:
    """Bridge-shaped analog of ``test_zip_import_bundle.py``'s
    ``_zip_with_understated_file_size``: a member whose central-directory
    ``file_size`` lies smaller than its real (compressed) payload, tripping a
    CRC mismatch on read -- exercising the "corrupt", not "too large", branch
    of ``_read_zip_member_bounded``'s ``ValueError`` contract.
    """
    buf = io.BytesIO()
    z = zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED)
    z.writestr("SKILL.md", b"---\nname: forged\n---\nbody\n")
    info = zipfile.ZipInfo("big.bin")
    z.writestr(info, b"y" * 4096)   # local header + data written with real size
    info.file_size = declared        # mutate BEFORE close -> central dir lies
    z.close()
    return buf.getvalue()


def test_reroot_labels_corrupt_member_not_too_large():
    data = _reroot_zip_with_understated_file_size()
    with pytest.raises(RemoteSkillError, match="Corrupt file in bundle") as exc_info:
        re_root_skill_zip(data, subdir="", suggested_name="x")
    assert "too large" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Task 5: the public install seam (compose classify -> enforce -> fetch ->
# re-root -> the existing import_skill_file seam).
# ---------------------------------------------------------------------------


class _RecordingScopeService:
    """Fake ``SkillsScopeService`` recording call order + captured kwargs.

    ``calls`` is shared with the fake transport handler (appended to
    directly by the test) so a single list proves ordering across BOTH
    the policy gate and the network hop -- not just the two scope-service
    methods on their own.
    """

    def __init__(self, *, enforce_error: Exception | None = None):
        self.calls: list[str] = []
        self._enforce_error = enforce_error
        self.import_content: bytes | None = None
        self.import_kwargs: dict | None = None

    def enforce_install_remote(self) -> None:
        self.calls.append("enforce")
        if self._enforce_error is not None:
            raise self._enforce_error

    async def import_skill_file(self, content, **kwargs):
        self.calls.append("import")
        self.import_content = content
        self.import_kwargs = kwargs
        return {"name": kwargs["filename"].removesuffix(".zip"), "backend": "local"}


@pytest.mark.asyncio
async def test_install_seam_classification_precedes_enforce_for_bad_urls():
    # Classification is pure and may run before enforcement -- an invalid
    # URL is rejected without ever touching the policy gate.
    from tldw_chatbook.Skills_Interop.skill_remote_fetch import install_skill_from_url

    scope = _RecordingScopeService()
    with pytest.raises(RemoteSkillError):
        await install_skill_from_url("http://github.com/o/r", scope_service=scope)
    assert scope.calls == []


@pytest.mark.asyncio
async def test_install_seam_enforces_before_any_network():
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError
    from tldw_chatbook.Skills_Interop.skill_remote_fetch import install_skill_from_url

    denial = PolicyDeniedError(
        action_id="skills.install_remote.launch.local",
        reason_code="policy_denied",
        user_message="Remote skill installs are disabled.",
        effective_source="config",
        authority_owner="local",
    )
    scope = _RecordingScopeService(enforce_error=denial)
    requests_seen: list[httpx.Request] = []

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(200, content=b"PK\x03\x04")

    with pytest.raises(PolicyDeniedError):
        await install_skill_from_url(
            "https://github.com/o/r/tree/main",
            scope_service=scope,
            transport=_transport(handler),
            resolver=_PUB,
        )
    assert scope.calls == ["enforce"]
    assert requests_seen == []


@pytest.mark.asyncio
async def test_install_seam_github_zip_source_happy_path():
    from tldw_chatbook.Skills_Interop.skill_remote_fetch import install_skill_from_url

    zip_bytes = _zipball(
        [("SKILL.md", "---\nname: brainstorm\n---\nbody")],
        wrapper="o-brainstorm-abc123/",
    )
    scope = _RecordingScopeService()

    def handler(request):
        assert request.url.host == "api.github.com"
        assert request.url.path == "/repos/o/brainstorm/zipball/main"
        scope.calls.append("network")
        return httpx.Response(200, content=zip_bytes)

    result = await install_skill_from_url(
        "https://github.com/o/brainstorm/tree/main",
        scope_service=scope,
        transport=_transport(handler),
        resolver=_PUB,
    )

    assert scope.calls == ["enforce", "network", "import"]
    assert scope.import_kwargs == {
        "mode": "local",
        "filename": "brainstorm.zip",
        "content_type": "application/zip",
        "overwrite": False,
        "trust_approved": False,
    }
    with zipfile.ZipFile(io.BytesIO(scope.import_content)) as z:
        assert "SKILL.md" in z.namelist()
    assert result == {"name": "brainstorm", "backend": "local"}


@pytest.mark.asyncio
async def test_install_seam_direct_zip_source_flow():
    from tldw_chatbook.Skills_Interop.skill_remote_fetch import install_skill_from_url

    zip_bytes = _zipball(
        [("SKILL.md", "---\nname: widget\n---\nbody")], wrapper="widget-1.0/"
    )
    scope = _RecordingScopeService()
    seen_auth: list[str | None] = []

    def handler(request):
        seen_auth.append(request.headers.get("authorization"))
        scope.calls.append("network")
        return httpx.Response(200, content=zip_bytes)

    result = await install_skill_from_url(
        "https://example.com/pkg/widget.zip",
        scope_service=scope,
        transport=_transport(handler),
        resolver=_PUB,
    )

    assert scope.calls == ["enforce", "network", "import"]
    assert seen_auth == [None]  # third-party host: never GitHub-token-scoped
    assert scope.import_kwargs["filename"] == "widget.zip"
    assert scope.import_kwargs["overwrite"] is False
    assert scope.import_kwargs["trust_approved"] is False
    assert result == {"name": "widget", "backend": "local"}


@pytest.mark.asyncio
async def test_install_seam_overwrite_flag_passed_through():
    from tldw_chatbook.Skills_Interop.skill_remote_fetch import install_skill_from_url

    zip_bytes = _zipball(
        [("SKILL.md", "---\nname: widget\n---\nbody")], wrapper="widget-1.0/"
    )
    scope = _RecordingScopeService()

    def handler(request):
        return httpx.Response(200, content=zip_bytes)

    await install_skill_from_url(
        "https://example.com/pkg/widget.zip",
        scope_service=scope,
        overwrite=True,
        transport=_transport(handler),
        resolver=_PUB,
    )
    assert scope.import_kwargs["overwrite"] is True


# ---------------------------------------------------------------------------
# Task 6: end-to-end -- REAL LocalSkillsService/SkillsScopeService/
# SkillTrustService behind the public install seam, mocked transport only.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_install_skill_from_github_tree_url_real_services(tmp_path, monkeypatch):
    """Paste a GitHub ``/tree/`` URL -> classify -> real policy gate ->
    real 302 hop (api.github.com -> codeload.github.com) -> bounded re-root
    -> the real hardened ``import_skill_file`` seam, against REAL
    ``LocalSkillsService``/``SkillsScopeService``/``SkillTrustService``
    instances (no fakes for the service layer). Only the two network-facing
    seams are mocked: the httpx transport (``fetch_zip_bytes``'s own
    ``AsyncClient``) and DNS resolution (SSRF host-allow check).

    ``GitHubAPIClient.get_branches`` is monkeypatched directly on the class
    (it owns its own internal ``httpx.AsyncClient``, not the ``transport=``
    override passed to ``install_skill_from_url``, so the fake transport
    below cannot reach it) to return ``["main", "master"]`` -- the exact
    shape ``get_branches``'s real success path returns (a plain
    ``list[str]`` of branch names; see ``GitHubAPIClient.get_branches`` in
    ``tldw_chatbook/Utils/github_api_client.py``). This keeps the test fully
    hermetic -- no live request to ``api.github.com`` for either the branch
    listing or the zip download -- while still exercising the real
    ``resolve_ref_and_subdir`` longest-prefix match, which resolves to
    ``("main", "skills/demo")`` for the pasted ``/tree/main/skills/demo``
    URL.

    Policy enforcement is REAL, not a no-op: a real ``ServicePolicyEnforcer``
    bound to the real ``CAPABILITY_REGISTRY``/``PolicyEngine`` is wired onto
    both ``local_service`` and ``scope_service`` below, mirroring how
    ``app.py`` (~:4536) wires one ``ServicePolicyEnforcer`` instance into
    every scope-service layer, and how
    ``test_skill_editor_opens_under_real_runtime_policy_enforcer`` in
    ``Tests/Skills/test_skills_library_flow.py`` proves the equivalent gate
    for the editor seam. This test therefore proves
    ``skills.install_remote.launch.local`` (and the
    ``skills.import.launch.local`` hop inside ``import_skill_file``)
    genuinely resolve ALLOWED under the real engine's default local-source
    policy -- not merely that ``enforce_install_remote()`` is a harmless
    no-op when no enforcer is wired (that no-op path is already covered by
    ``test_scope_service_public_enforce_passthrough`` above).
    """
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skill_remote_fetch import install_skill_from_url
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
    from tldw_chatbook.Skills_Interop.skill_trust_store import (
        FileSkillTrustGenerationMarkerStore,
        SkillTrustStore,
    )
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    # A real, unlocked trust service, bootstrapped against an EMPTY store --
    # mirrors Tests/Skills/test_skills_library_flow.py's
    # ``_real_trust_service`` + the bootstrap-before-any-skill-exists idiom
    # from ``test_library_shell_create_skill_save_arrives_needs_review_with_panel_primed``,
    # so the skill installed below is unambiguously "added since the
    # baseline" (quarantined_added / trust-pending) rather than merely
    # unreviewed due to a missing trust service.
    trust_service = SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )
    trust_service.unlock_with_passphrase("e2e-passphrase", salt=b"7" * 32)
    trust_service.bootstrap_trust()

    from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    # A REAL policy_enforcer, bound to the real CAPABILITY_REGISTRY/
    # PolicyEngine (not left unset) -- wired onto BOTH local_service and
    # scope_service, mirroring app.py's production wiring (~:4536) of one
    # ServicePolicyEnforcer instance into every scope-service layer. Leaving
    # this unset would make enforce_install_remote() trivially no-op (see
    # test_scope_service_public_enforce_passthrough above) without ever
    # reaching PolicyEngine.evaluate() -- exactly the gate-defect class
    # test_skill_editor_opens_under_real_runtime_policy_enforcer (in
    # Tests/Skills/test_skills_library_flow.py) guards against for the
    # editor seam.
    policy_enforcer = ServicePolicyEnforcer(
        state_provider=lambda: RuntimeSourceState(active_source="local"),
    )
    local_service = LocalSkillsService(
        store_dir=tmp_path,
        trust_service=trust_service,
        policy_enforcer=policy_enforcer,
    )
    scope_service = SkillsScopeService(
        local_service=local_service,
        server_service=None,
        policy_enforcer=policy_enforcer,
    )

    # GitHubAPIClient reads its token from the env var named by
    # [github].api_token_env_var ("GITHUB_API_TOKEN" by default -- confirmed
    # against config.py's default template). install_skill_from_url
    # constructs its own GitHubAPIClient() internally, so the
    # constructor-injection idiom Tests/Utils/test_github_api_client.py uses
    # isn't reachable from here; the env var is the clean seam. The value is
    # deliberately non-real -- it is never sent to the real GitHub API (the
    # branch listing below is monkeypatched, and the zip-fetch hops go
    # through the fake ``transport=`` handler further down), but it still
    # proves auth-header presence on both GitHub-family hops.
    monkeypatch.setenv("GITHUB_API_TOKEN", "e2e-secret-token")

    # Mock GitHubAPIClient.get_branches directly on the class -- it owns its
    # own internal httpx.AsyncClient, not the transport= override passed to
    # install_skill_from_url below, so leaving it unmocked would send a real
    # request (carrying the garbage token above) to
    # api.github.com/repos/obra/superpowers/branches. Patching the method on
    # the class works even though install_skill_from_url does a local
    # `from ..Utils.github_api_client import GitHubAPIClient` and constructs
    # its own instance internally: that import resolves to this same class
    # object, so the patched method is picked up regardless of where/when it
    # is imported. The fake return value matches get_branches's real
    # success-path shape exactly -- a plain list[str] of branch names (see
    # GitHubAPIClient.get_branches in tldw_chatbook/Utils/github_api_client.py,
    # success branch: `[branch["name"] for branch in branches]`). "main" must
    # be present so resolve_ref_and_subdir's longest-prefix match still
    # yields ("main", "skills/demo") for the pasted /tree/main/skills/demo
    # URL.
    from tldw_chatbook.Utils.github_api_client import GitHubAPIClient

    branch_calls: list[tuple[str, str]] = []

    async def _fake_get_branches(self, owner: str, repo: str) -> list[str]:
        branch_calls.append((owner, repo))
        return ["main", "master"]

    monkeypatch.setattr(GitHubAPIClient, "get_branches", _fake_get_branches)

    zip_bytes = _zipball(
        [
            ("skills/demo/SKILL.md", "---\nname: demo\n---\nDemo skill body.\n"),
            ("skills/demo/references/api.md", "# API\n\nReference doc.\n"),
        ],
        wrapper="superpowers-abc/",
    )

    seen_auth: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        seen_auth[host] = request.headers.get("authorization")
        if host == "api.github.com":
            assert request.url.path == "/repos/obra/superpowers/zipball/main"
            return httpx.Response(
                302,
                headers={
                    "location": "https://codeload.github.com/obra/superpowers/legacy.zip/main"
                },
            )
        if host == "codeload.github.com":
            return httpx.Response(200, content=zip_bytes)
        raise AssertionError(f"unexpected host hit: {host}")

    fake_public = lambda host: ["93.184.216.34"]

    result = await install_skill_from_url(
        "https://github.com/obra/superpowers/tree/main/skills/demo",
        scope_service=scope_service,
        transport=_transport(handler),
        resolver=fake_public,
    )

    # The auth header was present on BOTH github-family hops -- the redirect
    # origin AND the codeload landing spot -- proving the 302 re-validates
    # auth-scoping per hop rather than assuming it carries over (the
    # off-family-strip half of this is already covered by
    # test_auth_scoped_to_github_family above; this proves the presence half
    # end to end through the public install seam).
    assert seen_auth == {
        "api.github.com": "token e2e-secret-token",
        "codeload.github.com": "token e2e-secret-token",
    }

    # The mocked branch listing was actually reached (not skipped) and asked
    # for the right repo -- proves the ref/subdir split above is exercising
    # resolve_ref_and_subdir's real longest-prefix logic against the fake
    # data, not merely trusting an unreached mock.
    assert branch_calls == [("obra", "superpowers")]

    assert result["name"] == "demo"
    assert result["backend"] == "local"
    assert result["trust_status"] == "quarantined_added"
    assert result["trust_blocked"] is True

    skill_dir = tmp_path / "skills" / "demo"
    assert (skill_dir / "SKILL.md").is_file()
    assert (skill_dir / "references" / "api.md").is_file()

    # Re-confirm not-trusted through the service's own trust-status accessor
    # -- not by re-deriving it from the install() return value -- so this
    # proves the persisted index + on-disk bundle actually reflect
    # trust-pending, not just the one response object.
    fetched = await scope_service.get_skill("demo", mode="local")
    assert fetched["trust_status"] == "quarantined_added"
    assert fetched["trust_blocked"] is True
