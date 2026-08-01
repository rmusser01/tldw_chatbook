"""TASK-595 Task 5: consent preflight -- staged credit, space math, gating probe."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse

import httpx
import pytest

from Tests.Model_Artifacts.acquisition_test_helpers import _trusted
from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from Tests.Model_Artifacts.test_acquisition_types import DictCatalog, make_descriptor
from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
)
from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactAcquisitionService,
    CatalogError,
    PreflightNotGrantableError,
)
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService


@pytest.mark.asyncio
async def test_preflight_aggregates_and_grants(tmp_path):
    """download_bytes sums not-installed totals; a clean report grants."""
    core = ModelArtifactService(tmp_path / "root")
    with FixtureArtifactServer() as srv:
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        root = ArtifactRef("root-model", "r1", "int8")
        catalog = DictCatalog(
            {
                root: make_descriptor(
                    ref=root, files_body=body, source_url=srv.url("/m.onnx")
                )
            }
        )
        report = await svc.preflight(root, catalog)
    assert report.download_bytes == 2048
    assert report.sufficient_space is True
    assert report.entries[0].already_installed is False
    assert report.entries[0].provenance == (
        ProvenanceClass.CHATBOOK_CURATED,
    )
    report.grant()  # must not raise


@pytest.mark.asyncio
async def test_preflight_counts_staged_credit(tmp_path):
    """A partial fetch-state sidecar credits already_staged_bytes.

    The staged file itself must actually exist with AT LEAST as many
    bytes as the sidecar claims: credit is capped by the file's real
    on-disk size, not just the sidecar's say-so (see the review finding
    covered by test_preflight_stale_sidecar_credit_capped_by_actual_file_size).

    TASK-1694: staged credit is now read from a service-owned download
    stage's ``state/`` subtree (``core._download_stage_for``), not a bare
    ``staging/managed/<id>/<rev>/<variant>`` directory with a sibling
    sidecar file -- see acquisition.py's ``_fetch_sidecar_path`` and
    ``_staged_bytes_for``.
    """
    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("root-model", "r1", "int8")
    with FixtureArtifactServer() as srv:
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        desc = make_descriptor(ref=root, files_body=body, source_url=srv.url("/m.onnx"))
        catalog = DictCatalog({root: desc})

        stage = core._download_stage_for(desc, create=True)
        (stage.payload / "model.onnx").write_bytes(b"m" * 500)
        # Sidecar lives inside the stage's state/ subtree (see
        # acquisition.py's _fetch_sidecar_path), never inside payload/.
        (stage.state / "fetch-state.json").write_text(
            json.dumps(
                {
                    "files": {
                        "model.onnx": {
                            "etag": '"v1"',
                            "last_modified": None,
                            "bytes_done": 500,
                            "complete": False,
                        }
                    }
                }
            )
        )

        report = await svc.preflight(root, catalog)
    assert report.already_staged_bytes == 500
    assert report.download_bytes == 2048 - 500


@pytest.mark.asyncio
async def test_preflight_insufficient_space_blocks_grant(tmp_path):
    """A tiny free_bytes_probe blocks grant() even with no gating errors."""
    core = ModelArtifactService(tmp_path / "root")
    with FixtureArtifactServer() as srv:
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10, trusted_origins=_trusted(srv)
        )
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        root = ArtifactRef("root-model", "r1", "int8")
        catalog = DictCatalog(
            {
                root: make_descriptor(
                    ref=root, files_body=body, source_url=srv.url("/m.onnx")
                )
            }
        )
        report = await svc.preflight(root, catalog)
    assert report.sufficient_space is False
    with pytest.raises(PreflightNotGrantableError):
        report.grant()


@pytest.mark.asyncio
async def test_preflight_gated_repo_reports_instructions(tmp_path):
    """A 401-gated repository surfaces a gating error without leaking the token."""
    core = ModelArtifactService(tmp_path / "root")
    with FixtureArtifactServer() as srv:
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        body = b"m" * 2048
        srv.serve("/m.onnx", body, require_token="tok-secret")
        root = ArtifactRef("root-model", "r1", "int8")
        catalog = DictCatalog(
            {
                root: make_descriptor(
                    ref=root, files_body=body, source_url=srv.url("/m.onnx")
                )
            }
        )
        report = await svc.preflight(root, catalog)
    assert report.gating_errors, "401 repo must surface a gating error"
    assert all("tok-secret" not in message for message in report.gating_errors)
    with pytest.raises(PreflightNotGrantableError):
        report.grant()


@pytest.mark.asyncio
async def test_preflight_upgrade_retains_prior_active_version(tmp_path):
    """An installed+active v1 is retained while a not-installed v2 downloads fresh.

    v2 must NOT be reported as already_installed; the byte cost of v1 must
    surface as retained_bytes (kept on disk during the upgrade), and v2's
    full size must surface as download_bytes -- no double-counting either
    way.
    """
    core = ModelArtifactService(tmp_path / "root")
    artifact_id = "root-model"
    v1_ref = ArtifactRef(artifact_id, "r1", "int8")
    v2_ref = ArtifactRef(artifact_id, "r2", "int8")
    v1_body = b"v1" * 100  # 200 bytes
    v1_descriptor = make_descriptor(ref=v1_ref, files_body=v1_body)

    # Install and activate v1 directly against the core (copy semantics,
    # matching test_install_consume_source.py's payload construction).
    v1_source = tmp_path / "v1-source"
    v1_source.mkdir()
    (v1_source / v1_descriptor.files[0].path).write_bytes(v1_body)
    core.install(v1_descriptor, v1_source)
    core.activate(v1_ref)

    with FixtureArtifactServer() as srv:
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        v2_body = b"v2" * 2048  # 4096 bytes -- deliberately != v1's size
        srv.serve("/v2.onnx", v2_body)
        v2_descriptor = make_descriptor(
            ref=v2_ref, files_body=v2_body, source_url=srv.url("/v2.onnx")
        )
        catalog = DictCatalog({v2_ref: v2_descriptor})
        report = await svc.preflight(v2_ref, catalog)
    assert report.entries[0].already_installed is False
    assert report.retained_bytes == len(v1_body)
    assert report.download_bytes == len(v2_body)


@pytest.mark.asyncio
async def test_preflight_exact_installed_version_has_no_download_or_retention(tmp_path):
    """Preflighting the exact installed+active version costs and retains nothing.

    No gating probe is expected here either (the sole entry is already
    installed), so this test needs no fixture server at all.
    """
    core = ModelArtifactService(tmp_path / "root")
    root_ref = ArtifactRef("root-model", "r1", "int8")
    body = b"m" * 2048
    desc = make_descriptor(ref=root_ref, files_body=body)
    source = tmp_path / "source"
    source.mkdir()
    (source / desc.files[0].path).write_bytes(body)
    core.install(desc, source)
    core.activate(root_ref)

    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    catalog = DictCatalog({root_ref: desc})
    report = await svc.preflight(root_ref, catalog)

    assert report.entries[0].already_installed is True
    assert report.download_bytes == 0
    assert report.retained_bytes == 0


@pytest.mark.asyncio
async def test_preflight_clamps_oversized_staged_credit_to_entry_total(tmp_path):
    """A stale/corrupt sidecar claiming more bytes than the artifact is clamped.

    Regression test for the review finding: 999_999 claimed bytes_done for a
    2048-byte artifact must not inflate already_staged_bytes past the
    entry's own total, and must not drive download_bytes negative. The
    staged file is written at its full declared size (2048 bytes) so this
    test still exercises the "declared total" clamp specifically, distinct
    from the "actual on-disk size" clamp
    (test_preflight_stale_sidecar_credit_capped_by_actual_file_size).
    """
    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("root-model", "r1", "int8")
    with FixtureArtifactServer() as srv:
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        desc = make_descriptor(ref=root, files_body=body, source_url=srv.url("/m.onnx"))
        catalog = DictCatalog({root: desc})

        # TASK-1694: staged credit now lives under a service-owned
        # download stage's state/ subtree (see _staged_bytes_for).
        stage = core._download_stage_for(desc, create=True)
        (stage.payload / "model.onnx").write_bytes(b"m" * 2048)
        (stage.state / "fetch-state.json").write_text(
            json.dumps(
                {
                    "files": {
                        "model.onnx": {
                            "etag": '"v1"',
                            "last_modified": None,
                            "bytes_done": 999_999,
                            "complete": False,
                        }
                    }
                }
            )
        )

        report = await svc.preflight(root, catalog)
    assert report.already_staged_bytes == 2048
    assert report.download_bytes == 0


@pytest.mark.asyncio
async def test_preflight_stale_sidecar_credit_capped_by_actual_file_size(tmp_path):
    """A sidecar claiming more bytes than are ACTUALLY on disk right now
    must not inflate already_staged_bytes past what could genuinely be
    resumed -- the declared-size clamp alone (the previous behavior)
    doesn't catch this: 5000 is already less than nothing here, so it
    passed the OLD "cap by declared total" check unchanged, yet the
    staged FILE itself holds only 100 bytes. Preflight's space math must
    trust the smaller, real number, or it can approve an acquisition that
    then runs out of space because the "already staged" credit was
    phantom.

    Regression test for the review finding: cap each file's credit by
    ``min(recorded bytes_done, actual file size on disk, declared file
    size)``.
    """
    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("root-model", "r1", "int8")
    with FixtureArtifactServer() as srv:
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        desc = make_descriptor(ref=root, files_body=body, source_url=srv.url("/m.onnx"))
        catalog = DictCatalog({root: desc})

        # TASK-1694: staged credit now lives under a service-owned
        # download stage's state/ subtree (see _staged_bytes_for).
        stage = core._download_stage_for(desc, create=True)
        (stage.payload / "model.onnx").write_bytes(b"m" * 100)  # only 100 bytes ACTUALLY staged
        (stage.state / "fetch-state.json").write_text(
            json.dumps(
                {
                    "files": {
                        "model.onnx": {
                            "etag": '"v1"',
                            "last_modified": None,
                            "bytes_done": 5000,  # claims far more than exists
                            "complete": False,
                        }
                    }
                }
            )
        )

        report = await svc.preflight(root, catalog)
    assert report.already_staged_bytes == 100
    assert report.download_bytes == 2048 - 100


def _two_file_descriptor(ref: ArtifactRef) -> ArtifactDescriptor:
    """A 2-file descriptor with no ``ArtifactSourceMap`` entries anywhere
    in this file's tests -- TASK-1695 defines per-file URLs via an explicit
    caller-supplied source map (see ``test_source_map.py`` for the
    end-to-end multi-file coverage); this fixture instead exercises the
    "resolution still fails when no map is supplied" path, so only the
    file COUNT matters here (mirrors test_provision_fetch.py's identical-
    purpose ``_make_two_file_descriptor``, duplicated locally per this
    suite's convention of not cross-importing test-private helpers)."""

    files = (
        ArtifactFile("a.bin", 4, hashlib.sha256(b"aaaa").hexdigest()),
        ArtifactFile("b.bin", 4, hashlib.sha256(b"bbbb").hexdigest()),
    )
    return ArtifactDescriptor(
        reference=ref,
        model_id="test/model",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="test",
        model_family="test-family",
        upstream_repository="test/repo",
        upstream_revision="main",
        source_url="https://example.test/model",
        precision="int8",
        license_id="test-license",
        license_url="https://example.test/license",
        usage_notice="Test model",
        runtime_name="test-runtime",
        runtime_version_constraint="==1.0.0",
        supported_os=("linux",),
        supported_architectures=("x86-64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=files,
        expected_installed_bytes=8,
        dependencies=(),
    )


@pytest.mark.asyncio
async def test_preflight_multi_file_descriptor_raises_catalog_error(tmp_path):
    """A 2-file descriptor with no ``sources`` map fails ``preflight()``
    itself, before any report or consent exists -- not just later at
    ``provision()``'s fetch phase.

    TASK-1695 note: multi-file descriptors are no longer refused outright
    (see ``test_source_map.py`` for the now-supported end-to-end path via
    an explicit ``ArtifactSourceMap``) -- this test's ``CatalogError`` now
    comes from ``_resolve_file_sources`` finding no resolvable URL for a
    declared file (no map entry, and no single-file fallback since this
    descriptor declares two), not from a blanket multi-file refusal.
    Original regression rationale still holds: previously only
    ``_fetch_artifact`` guarded resolution failures of this shape, so a
    preflight report could be built and consent granted for a closure that
    ``provision()`` would ALWAYS reject -- after already taking the
    exclusive session lease. The spec requires catalog/source problems to
    surface at preflight. No network fixture is needed: ``_aggregate_closure``
    raises before ``preflight()`` ever reaches its gating probe.
    """
    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    catalog = DictCatalog({root: _two_file_descriptor(root)})
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)

    with pytest.raises(CatalogError) as excinfo:
        await svc.preflight(root, catalog)
    assert "multi-file-model" in str(excinfo.value)


@pytest.mark.asyncio
async def test_probe_gating_head_does_not_follow_redirects(tmp_path):
    """The preflight gating HEAD probe must not follow a redirect, even
    when the injected ``client_factory`` seam supplies a client configured
    to do so by default.

    Regression test for the review finding: an injected redirect-following
    client would both bypass this app's own egress SSRF check for the
    redirect target (only ``entry.source_url`` -- origin A here -- is
    checked before the request) AND, when a credential is configured,
    carry the bearer token cross-origin. ``follow_redirects=False`` passed
    explicitly on the ``.head()`` call itself is what guarantees this
    regardless of the client's own configured default -- proven here by
    asserting the redirect target never receives any request at all.
    """
    with FixtureArtifactServer() as origin_a, FixtureArtifactServer() as origin_b:
        origin_b.serve("/final.bin", b"body-bytes")
        origin_a.serve("/gated.bin", b"", redirect_to=origin_b.url("/final.bin"))

        root = ArtifactRef("root-model", "r1", "int8")
        catalog = DictCatalog(
            {root: make_descriptor(ref=root, source_url=origin_a.url("/gated.bin"))}
        )
        core = ModelArtifactService(tmp_path / "root")
        injected_client = httpx.AsyncClient(follow_redirects=True)
        try:
            svc = ArtifactAcquisitionService(
                core,
                free_bytes_probe=lambda p: 10**12,
                trusted_origins=frozenset({urlparse(origin_a.url("/")).hostname}),
                client_factory=lambda: injected_client,
            )
            await svc.preflight(root, catalog)
        finally:
            await injected_client.aclose()

        assert origin_b.requests == {}, (
            "the gating probe must not follow a redirect to a different origin"
        )
