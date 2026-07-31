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
