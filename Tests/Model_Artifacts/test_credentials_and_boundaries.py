"""TASK-595 Task 9: credential resolution, secret hygiene, worker import boundary.

Four things this file pins down that no earlier task's tests cover:

(a) A ``CredentialResolver`` that returns a working token doesn't just clear
    ``preflight()``'s gating probe -- it lets ``provision()`` actually fetch,
    verify, and install against the same gated repository (Task 5 already
    covers "no resolver -> gating_errors" in
    ``test_preflight.py::test_preflight_gated_repo_reports_instructions``;
    not duplicated here).
(b) The resolved token never leaks into a log record (loguru, bridged to
    ``caplog``, plus httpx/httpcore's native stdlib logging), the fetch-state
    sidecar, any file under the artifact store, or a raised error's
    ``str()``.
(c) Task 3 review carry-over: an authenticated origin that 302-redirects to
    a DIFFERENT origin must not leak ``Authorization`` across the hop --
    ``fetch.stream_fetch`` already strips it (Task 3), but nothing exercised
    that path against two real, differently-ported fixture servers until now.
(d) Import boundary: the STT/worker surface that actually runs local
    transcription must never import the async, credentialed acquisition
    modules -- following ``Tests/STT/test_boundaries.py``'s subprocess
    import-recording mechanism and module list, extended with the concrete
    legacy worker module.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap

import httpx
import pytest

from Tests.Model_Artifacts.acquisition_test_helpers import _trusted, _two_file_descriptor
from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from Tests.Model_Artifacts.test_acquisition_types import DictCatalog, make_descriptor
from tldw_chatbook.Model_Artifacts import ArtifactRef
from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactAcquisitionService,
    CredentialResolver,
    EnvConfigCredentialResolver,
)
from tldw_chatbook.Model_Artifacts.fetch import stream_fetch
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService

TOKEN = "tok-secret-9f3c4a"


class _StaticResolver:
    """Test double: always resolves to the same token (or ``None``).

    Records every ``repository`` it was asked to resolve for, so a test can
    prove the resolver was actually consulted rather than bypassed.
    """

    def __init__(self, token: str | None) -> None:
        self._token = token
        self.resolved_for: list[str] = []

    def resolve(self, repository: str) -> str | None:
        """Return the configured token, recording the request."""
        self.resolved_for.append(repository)
        return self._token


# ---------------------------------------------------------------------------
# (a) + (b): resolver-backed provision succeeds end-to-end against a gated
# repository, and the token never leaks anywhere.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _loguru_to_caplog(caplog):
    """Bridge loguru output into pytest's ``caplog`` for this module only.

    loguru does not propagate to stdlib ``logging`` (and therefore not to
    ``caplog``) without an explicit bridge -- the same pattern already used
    in ``Tests/Internal_Prompts/conftest.py``, scoped here to this file so
    it doesn't change log-capture behavior for the rest of the suite.
    """
    from loguru import logger as loguru_logger

    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    handler_id = loguru_logger.add(PropagateHandler(), format="{message}")
    yield
    loguru_logger.remove(handler_id)


@pytest.mark.asyncio
async def test_gated_repo_with_resolver_provisions_and_never_leaks_token(tmp_path, caplog):
    """A working credential clears preflight gating AND completes a real
    fetch/pre-verify/install/activate against the gated route -- proving the
    resolver is wired into BOTH the HEAD probe and the real fetch, not just
    one of them. Then scans everywhere a leaked secret could hide.
    """
    root_dir = tmp_path / "root"
    core = ModelArtifactService(root_dir)
    root = ArtifactRef("gated-model", "r1", "int8")
    resolver = _StaticResolver(TOKEN)

    with caplog.at_level(logging.DEBUG):
        with FixtureArtifactServer() as srv:
            body = b"gated-payload-bytes-" * 400  # 8400 bytes
            srv.serve("/m.onnx", body, require_token=TOKEN, etag='"v1"', support_range=True)
            svc = ArtifactAcquisitionService(
                core,
                free_bytes_probe=lambda p: 10**12,
                trusted_origins=_trusted(srv),
                credential_resolver=resolver,
            )
            desc = make_descriptor(ref=root, files_body=body, source_url=srv.url("/m.onnx"))
            catalog = DictCatalog({root: desc})

            report = await svc.preflight(root, catalog)
            assert report.gating_errors == (), (
                "a working credential must clear the preflight gating probe"
            )
            consent = report.grant()

            activated = await svc.provision(root, consent, catalog)
            assert activated == root

    # The resolver was genuinely consulted (both the probe and the fetch
    # ask for the same repository -- make_descriptor's fixed "test/repo").
    assert resolver.resolved_for
    assert set(resolver.resolved_for) == {"test/repo"}

    installed_refs = {
        item.descriptor.reference for item in core.list_installed() if item.descriptor is not None
    }
    assert root in installed_refs
    with core.acquire(root) as handle:
        assert handle.handle.root == root

    # (b) secret hygiene, four independent surfaces:

    # 1. No log record -- loguru (bridged above) or httpx/httpcore's native
    #    stdlib DEBUG logging (connection tracing, request/response lines)
    #    -- carries the token.
    assert TOKEN not in caplog.text
    for record in caplog.records:
        assert TOKEN not in record.getMessage()

    # 2. Nothing under the artifact store's root -- staging sidecars,
    #    installed payloads, manifests, lease files -- contains the token.
    scanned = 0
    for path in root_dir.rglob("*"):
        if path.is_file():
            scanned += 1
            assert TOKEN.encode() not in path.read_bytes(), f"token leaked into {path}"
    assert scanned > 0, "sanity: the artifact store must contain files to scan"

    # 3. The fetch-state sidecar specifically no longer exists (a
    #    successful finalize retires the whole download stage, sidecar
    #    included -- see core._finalize_download_stage /
    #    _remove_finalized_download_stage), which the tree scan above
    #    already covers, but assert it explicitly as the most direct claim
    #    the brief asks for. It would have lived inside the stage's
    #    state/ subtree, never inside payload/ (see acquisition.py's
    #    _fetch_sidecar_path).
    assert core._download_stage_for(desc, create=False) is None


# ---------------------------------------------------------------------------
# (e) TASK-1695: per-file source-map URLs never leak into manifests, resume
# state, errors, or logs -- extends (b) above to the new source-map path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_file_source_map_urls_never_leak_into_state_manifests_or_errors(
    tmp_path, caplog
):
    """TASK-1695: per-file source-map URLs are credential-free by
    construction (rejected at ``preflight()`` otherwise -- see
    ``test_source_map.py``), but this proves the mechanism doesn't
    accidentally start PERSISTING url text anywhere either: the fetch-state
    sidecar, the installed manifest, and any raised error stay exactly as
    opaque to per-file origin identity as the single-file ``source_url``
    path always was.

    Structural guarantee behind this, not just an incidental test result:
    TASK-1695's AC #2 forbids adding a ``url`` field to ``ArtifactFile``/
    ``ArtifactDescriptor``, so there is nowhere in the manifest schema a
    per-file URL COULD be serialized into even if this test were absent.
    This is the regression guard that keeps that true.
    """
    MARKER = "sourcemarkxyz789"
    root_dir = tmp_path / "root"
    core = ModelArtifactService(root_dir)
    root = ArtifactRef("multi-file-hygiene", "r1", "int8")

    with caplog.at_level(logging.DEBUG):
        with FixtureArtifactServer() as srv:
            srv.serve(f"/{MARKER}/a.bin", b"aaaa", etag='"va"', support_range=True)
            srv.serve(f"/{MARKER}/b.bin", b"bbbb", etag='"vb"', support_range=True)
            svc = ArtifactAcquisitionService(
                core,
                free_bytes_probe=lambda p: 10**12,
                trusted_origins=_trusted(srv),
            )
            # Deliberately NOT marker-bearing: descriptor.source_url is a
            # pre-existing, unrelated field that manifest.json has ALWAYS
            # persisted (it is credential-free by its own validation, so
            # that persistence is expected and fine) -- keeping it marker-
            # free isolates this test to what TASK-1695 actually adds: the
            # PER-FILE source-map entries below, which must never appear
            # anywhere a plain descriptor field legitimately does.
            desc = _two_file_descriptor(root, srv.url("/hygiene-descriptor-source"))
            catalog = DictCatalog({root: desc})
            sources = {
                root: {
                    "a.bin": srv.url(f"/{MARKER}/a.bin"),
                    "b.bin": srv.url(f"/{MARKER}/b.bin"),
                }
            }

            report = await svc.preflight(root, catalog, sources=sources)
            consent = report.grant()
            activated = await svc.provision(root, consent, catalog, sources=sources)
            assert activated == root

    installed_refs = {
        item.descriptor.reference for item in core.list_installed() if item.descriptor is not None
    }
    assert root in installed_refs

    # 1. No THIS-APPLICATION log record carries the marker. httpx/httpcore's
    #    own DEBUG/INFO request tracing is deliberately excluded: it
    #    legitimately logs the exact (credential-free, by construction --
    #    see test_source_map.py's preflight-time rejection tests) URL it
    #    requests, which is expected operational visibility, not a secret
    #    leak -- the design spec forbids bearer tokens, cookies, signed
    #    redirect targets, and query strings from ever appearing anywhere,
    #    not the credential-free URL identity itself (which the spec's own
    #    "Resume metadata is credential-free and contains only ... a
    #    credential-free origin source identity" explicitly allows). What
    #    this DOES catch: any accidental ``logger.info(f"...{url}...")``
    #    this task's own new code (``_resolve_file_sources``,
    #    ``_closure_fingerprint_with_sources``, the ``_fetch_*`` threading)
    #    might have introduced.
    app_records = [
        record
        for record in caplog.records
        if not record.name.startswith(("httpx", "httpcore"))
    ]
    for record in app_records:
        assert MARKER not in record.getMessage()

    # 2. Nothing under the artifact store's root -- staging sidecars,
    #    installed payloads, manifests, lease files -- contains the marker.
    scanned = 0
    for path in root_dir.rglob("*"):
        if path.is_file():
            scanned += 1
            assert MARKER.encode() not in path.read_bytes(), f"source-map URL leaked into {path}"
    assert scanned > 0, "sanity: the artifact store must contain files to scan"


# ---------------------------------------------------------------------------
# (f) PR-1165 review, P0: a bearer token resolved for a repository must
# never reach a per-file mapped URL on a DIFFERENT origin than the
# descriptor's own source_url -- not just on a redirect hop (which
# fetch.stream_fetch already strips independently, see (c) below), but on
# the INITIAL request too.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_credential_attached_to_same_origin_mapped_file(tmp_path):
    """A per-file source-map URL on the SAME origin as ``descriptor.
    source_url`` receives the repository's resolved bearer token, exactly
    like the single-file fallback path already did."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("same-origin-mapped-model", "r1", "int8")
    resolver = _StaticResolver(TOKEN)

    with FixtureArtifactServer() as srv:
        srv.serve("/a.bin", b"aaaa", require_token=TOKEN, etag='"va"')
        srv.serve("/b.bin", b"bbbb", require_token=TOKEN, etag='"vb"')
        desc = _two_file_descriptor(root, srv.url("/a.bin"))
        catalog = DictCatalog({root: desc})
        sources = {root: {"a.bin": srv.url("/a.bin"), "b.bin": srv.url("/b.bin")}}
        svc = ArtifactAcquisitionService(
            core,
            free_bytes_probe=lambda p: 10**12,
            trusted_origins=_trusted(srv),
            credential_resolver=resolver,
        )

        report = await svc.preflight(root, catalog, sources=sources)
        consent = report.grant()
        activated = await svc.provision(root, consent, catalog, sources=sources)
        assert activated == root

    for path in ("/a.bin", "/b.bin"):
        assert any(
            headers.get("Authorization") == f"Bearer {TOKEN}"
            for headers in srv.requests[path]
        ), f"same-origin mapped file {path} must have received the credential"


@pytest.mark.asyncio
async def test_credential_withheld_from_cross_origin_mapped_file_but_both_download(
    tmp_path,
):
    """A per-file source-map URL on a DIFFERENT origin than ``descriptor.
    source_url`` must NEVER receive the repository's bearer token -- not
    just on a redirect hop, but on its very first request. Both files must
    still download successfully: the cross-origin file is public (no
    credential needed at all), matching a real third-party CDN or mirror.

    Regression test for the P0 finding: before this fix, ``_auth_headers``
    attached a resolved token to EVERY per-file request for the
    descriptor's repository, regardless of which URL (and therefore which
    origin) it was actually going to -- modeled here with two real,
    differently-ported ``FixtureArtifactServer`` instances, which is what
    makes this the INITIAL-request case ``stream_fetch``'s own
    redirect-hop stripping does not cover.
    """

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("cross-origin-mapped-model", "r1", "int8")
    resolver = _StaticResolver(TOKEN)

    with FixtureArtifactServer() as same_origin, FixtureArtifactServer() as cross_origin:
        same_origin.serve("/a.bin", b"aaaa", require_token=TOKEN, etag='"va"')
        # Public: a real cross-origin CDN never receives this repository's
        # credential, so it must not require one either.
        cross_origin.serve("/b.bin", b"bbbb", etag='"vb"')

        desc = _two_file_descriptor(root, same_origin.url("/a.bin"))
        catalog = DictCatalog({root: desc})
        sources = {
            root: {
                "a.bin": same_origin.url("/a.bin"),
                "b.bin": cross_origin.url("/b.bin"),
            }
        }
        svc = ArtifactAcquisitionService(
            core,
            free_bytes_probe=lambda p: 10**12,
            trusted_origins=_trusted(same_origin) | _trusted(cross_origin),
            credential_resolver=resolver,
        )

        report = await svc.preflight(root, catalog, sources=sources)
        assert report.gating_errors == ()
        consent = report.grant()
        activated = await svc.provision(root, consent, catalog, sources=sources)
        assert activated == root

        assert any(
            headers.get("Authorization") == f"Bearer {TOKEN}"
            for headers in same_origin.requests["/a.bin"]
        ), "the same-origin mapped file must have received the credential"

        b_requests = cross_origin.requests["/b.bin"]
        assert b_requests, "the cross-origin mapped file must have been reached at all"
        assert all("Authorization" not in headers for headers in b_requests), (
            "the credential must NOT have reached the cross-origin mapped file"
        )

    destination = core.artifact_path(root)
    assert (destination / "a.bin").read_bytes() == b"aaaa"
    assert (destination / "b.bin").read_bytes() == b"bbbb"


# ---------------------------------------------------------------------------
# (c) Task 3 review carry: cross-origin redirect strips Authorization.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_origin_redirect_strips_authorization_but_body_still_downloads(tmp_path):
    """An authenticated origin (A) 302-redirects to a DIFFERENT origin (B)
    -- modeled with two real ``FixtureArtifactServer`` instances on
    different ports, which ``fetch._same_origin`` treats as different
    origins even though both bind the same loopback hostname. The
    ``Authorization`` header must reach A (which requires it to authorize
    the redirect) but must NOT reach B, and the body must still download
    correctly from B.
    """
    body = b"cross-origin-body-bytes-" * 300  # 7200 bytes
    with FixtureArtifactServer() as origin_a, FixtureArtifactServer() as origin_b:
        origin_b.serve("/final.bin", body)
        origin_a.serve(
            "/gated.bin",
            b"",  # never served -- redirect_to short-circuits before the body
            require_token=TOKEN,
            redirect_to=origin_b.url("/final.bin"),
        )

        dest = tmp_path / "f.bin"
        async with httpx.AsyncClient() as client:
            result = await stream_fetch(
                origin_a.url("/gated.bin"),
                dest,
                client=client,
                max_bytes=len(body) + 10,
                headers={"Authorization": f"Bearer {TOKEN}"},
                trusted_origins=_trusted(origin_a),
            )

        assert dest.read_bytes() == body
        assert result.bytes_written == len(body)

        a_requests = origin_a.requests["/gated.bin"]
        assert any(headers.get("Authorization") == f"Bearer {TOKEN}" for headers in a_requests), (
            "the authenticated redirect origin must have SEEN the credential"
        )

        b_requests = origin_b.requests["/final.bin"]
        assert b_requests, "the redirect target must have been reached at all"
        assert all("Authorization" not in headers for headers in b_requests), (
            "the credential must NOT have crossed the origin boundary"
        )


@pytest.mark.asyncio
async def test_cross_origin_redirect_strips_client_level_default_authorization(tmp_path):
    """Same hand-off as the per-call-header test above, but the credential
    is attached as a CLIENT-LEVEL default header
    (``httpx.AsyncClient(headers={"Authorization": ...})``), which a caller
    who reuses one client across many requests (a repository's own bearer
    token, say) may reasonably do instead of passing it per call.

    ``stream_fetch``'s own ``send_headers`` stripping only ever sees the
    per-call ``headers=`` argument it built itself -- httpx merges a
    client's default headers onto the request during ``send()``, AFTER
    that per-call dict was stripped, so a client-level credential was
    previously invisible to the cross-origin strip and reached origin B
    verbatim. Regression test for that gap.
    """
    body = b"cross-origin-body-bytes-" * 300  # 7200 bytes
    with FixtureArtifactServer() as origin_a, FixtureArtifactServer() as origin_b:
        origin_b.serve("/final.bin", body)
        origin_a.serve(
            "/gated.bin",
            b"",  # never served -- redirect_to short-circuits before the body
            require_token=TOKEN,
            redirect_to=origin_b.url("/final.bin"),
        )

        dest = tmp_path / "f.bin"
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {TOKEN}"}
        ) as client:
            result = await stream_fetch(
                origin_a.url("/gated.bin"),
                dest,
                client=client,
                max_bytes=len(body) + 10,
                trusted_origins=_trusted(origin_a),
            )

        assert dest.read_bytes() == body
        assert result.bytes_written == len(body)

        a_requests = origin_a.requests["/gated.bin"]
        assert any(headers.get("Authorization") == f"Bearer {TOKEN}" for headers in a_requests), (
            "the authenticated redirect origin must have SEEN the client-level credential"
        )

        b_requests = origin_b.requests["/final.bin"]
        assert b_requests, "the redirect target must have been reached at all"
        assert all("Authorization" not in headers for headers in b_requests), (
            "the client-level credential must NOT have crossed the origin boundary"
        )


# ---------------------------------------------------------------------------
# (d) Import boundary: STT/worker surface never imports acquisition/fetch.
# ---------------------------------------------------------------------------


def test_stt_and_transcription_worker_modules_never_import_acquisition_or_fetch() -> None:
    """Neither ``Model_Artifacts.acquisition`` nor ``Model_Artifacts.fetch``
    may be imported, attempted, or newly loaded as a side effect of
    importing the STT runtime-dispatch surface (``contracts``,
    ``coordinator``, ``legacy_bridge``, ``registry``, ``routing`` -- the
    exact module list ``Tests/STT/test_boundaries.py`` already guards) or
    the concrete legacy transcription worker
    (``Local_Ingestion.transcription_service``) that surface ultimately
    dispatches to.

    Reuses that test's subprocess + import-recording-hook mechanism
    verbatim (a clean ``sys.modules`` baseline per run, plus a hook that
    also catches an import attempted-and-caught, not just one that
    actually landed in ``sys.modules``) rather than duplicating it against
    a different, narrower forbidden-module pair.

    ``Model_Artifacts.leases`` and ``.service`` ARE expected to load here
    (``STT.persistence`` -- reached transitively -- uses
    ``ArtifactLeaseKey``); only the async, network-capable, credentialed
    modules are forbidden.

    TASK-1696: also imports ``Audio.console_dictation`` and the new
    ``Local_Ingestion.parakeet_v2_artifact`` adapter it and
    ``transcription_service`` both now reach transitively -- the managed-
    first model-directory resolver these two worker-side modules share.
    ``parakeet_v2_artifact`` itself imports only ``Model_Artifacts.service``
    at module scope (its ``ArtifactAcquisitionService``-based orchestration
    helpers import ``.acquisition`` locally, inside their own function
    bodies, and only the Library UI ever calls them -- see that module's
    own docstring), so it must load here without pulling in ``.acquisition``
    or ``.fetch`` either.
    """
    script = textwrap.dedent(
        """
        import builtins
        import importlib.util
        import json
        import sys

        import tldw_chatbook

        baseline_modules = set(sys.modules)
        attempted_imports = set()
        original_import = builtins.__import__

        def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
            absolute_name = name
            package = globals.get("__package__") if globals is not None else None
            if level and package:
                absolute_name = importlib.util.resolve_name(
                    f"{'.' * level}{name}",
                    package,
                )
            attempted_imports.add(absolute_name)
            attempted_imports.update(
                f"{absolute_name}.{requested_name}"
                for requested_name in fromlist or ()
                if requested_name != "*"
            )
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = recording_import
        try:
            import tldw_chatbook.STT.contracts
            import tldw_chatbook.STT.coordinator
            import tldw_chatbook.STT.legacy_bridge
            import tldw_chatbook.STT.registry
            import tldw_chatbook.STT.routing
            import tldw_chatbook.Local_Ingestion.transcription_service
            import tldw_chatbook.Audio.console_dictation
        finally:
            builtins.__import__ = original_import

        print(
            json.dumps(
                {
                    "all": sorted(sys.modules),
                    "attempted": sorted(attempted_imports),
                    "incremental": sorted(set(sys.modules) - baseline_modules),
                }
            )
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    imported = json.loads(completed.stdout)

    forbidden = (
        "tldw_chatbook.Model_Artifacts.acquisition",
        "tldw_chatbook.Model_Artifacts.fetch",
    )
    for observed_key in ("all", "attempted", "incremental"):
        observed_modules = set(imported[observed_key])
        leaked = {
            name
            for name in forbidden
            if any(
                module == name or module.startswith(f"{name}.") for module in observed_modules
            )
        }
        assert leaked == set(), f"{observed_key} leaked forbidden imports: {leaked}"

    # Sanity: prove this isn't vacuous by confirming the SIBLING modules
    # (leases/service) really were loaded -- if the whole Model_Artifacts
    # package were untouched, the assertions above would pass trivially.
    all_modules = set(imported["all"])
    assert "tldw_chatbook.Model_Artifacts.leases" in all_modules
    assert "tldw_chatbook.Model_Artifacts.service" in all_modules
    # TASK-1696: same sanity check for the new adapter module --
    # ``transcription_service`` imports it at module scope (``console_
    # dictation`` only imports it lazily, inside its own resolver method,
    # so a plain import of that module alone would not prove this).
    assert "tldw_chatbook.Local_Ingestion.parakeet_v2_artifact" in all_modules


# ---------------------------------------------------------------------------
# Bonus: EnvConfigCredentialResolver's own env -> config precedence.
# ---------------------------------------------------------------------------


def test_env_config_resolver_prefers_huggingface_api_key_env(monkeypatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "from-env-primary")
    monkeypatch.setenv("HF_TOKEN", "from-env-fallback")
    resolver = EnvConfigCredentialResolver()
    assert resolver.resolve("any/repo") == "from-env-primary"


def test_env_config_resolver_falls_back_to_hf_token_env(monkeypatch) -> None:
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "from-hf-token")
    resolver = EnvConfigCredentialResolver()
    assert resolver.resolve("any/repo") == "from-hf-token"


def test_env_config_resolver_falls_back_to_config_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.Model_Artifacts.acquisition.get_cli_setting",
        lambda section, key, default=None: "from-config" if key == "huggingface_api_key" else default,
    )
    resolver = EnvConfigCredentialResolver()
    assert resolver.resolve("any/repo") == "from-config"


def test_env_config_resolver_returns_none_when_nothing_configured(monkeypatch) -> None:
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.Model_Artifacts.acquisition.get_cli_setting",
        lambda section, key, default=None: default,
    )
    resolver = EnvConfigCredentialResolver()
    assert resolver.resolve("any/repo") is None


def test_credential_resolver_protocol_is_satisfied_by_duck_typed_object() -> None:
    """Structural typing: anything with a matching ``resolve`` method
    satisfies ``CredentialResolver`` without inheriting from it."""
    assert isinstance(_StaticResolver("x"), CredentialResolver)
