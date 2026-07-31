"""TASK-595 Task 5: consent preflight -- staged credit, space math, gating probe."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from Tests.Model_Artifacts.test_acquisition_types import DictCatalog, make_descriptor
from tldw_chatbook.Model_Artifacts import ArtifactRef
from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactAcquisitionService,
    PreflightNotGrantableError,
)
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService


def _trusted(srv: FixtureArtifactServer) -> frozenset:
    """Trusted-origins set for a fixture server, in egress's real format.

    ``tldw_chatbook.Utils.egress._normalize_trusted`` / ``_post_resolution``
    key membership on the bare, lowercased HOSTNAME (e.g. ``"127.0.0.1"``),
    not a scheme+host+port URL string -- mirrors the ``_trusted`` helper in
    ``Tests/Model_Artifacts/test_stream_fetch.py``. The fixture server binds
    to the loopback IP literal, which classifies as "private" under
    ``_classify_ip`` and would otherwise be egress-blocked; listing it here
    is what lets policy allow the preflight HEAD probe to reach it.
    """
    return frozenset({urlparse(srv.url("/")).hostname})


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
    report.grant()  # must not raise


@pytest.mark.asyncio
async def test_preflight_counts_staged_credit(tmp_path):
    """A partial fetch-state sidecar credits already_staged_bytes."""
    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("root-model", "r1", "int8")
    staged = Path(core.staging_path) / "managed" / "root-model" / "r1" / "int8"
    staged.mkdir(parents=True)
    (staged / "fetch-state.json").write_text(
        json.dumps(
            {
                "files": {
                    "m.onnx": {
                        "etag": '"v1"',
                        "last_modified": None,
                        "bytes_done": 500,
                        "complete": False,
                    }
                }
            }
        )
    )
    with FixtureArtifactServer() as srv:
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        catalog = DictCatalog(
            {
                root: make_descriptor(
                    ref=root, files_body=body, source_url=srv.url("/m.onnx")
                )
            }
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
    entry's own total, and must not drive download_bytes negative.
    """
    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("root-model", "r1", "int8")
    staged = Path(core.staging_path) / "managed" / "root-model" / "r1" / "int8"
    staged.mkdir(parents=True)
    (staged / "fetch-state.json").write_text(
        json.dumps(
            {
                "files": {
                    "m.onnx": {
                        "etag": '"v1"',
                        "last_modified": None,
                        "bytes_done": 999_999,
                        "complete": False,
                    }
                }
            }
        )
    )
    with FixtureArtifactServer() as srv:
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        catalog = DictCatalog(
            {
                root: make_descriptor(
                    ref=root, files_body=body, source_url=srv.url("/m.onnx")
                )
            }
        )
        report = await svc.preflight(root, catalog)
    assert report.already_staged_bytes == 2048
    assert report.download_bytes == 0
