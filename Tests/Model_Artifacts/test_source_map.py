"""TASK-1695: descriptor + credential-free source-map contract for per-file URLs.

Reconciliation item 2 (see
``Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-
reconciliation.md``, "3. Per-file source URLs"): the parallel TASK-595
branch passes an explicit, credential-free source map alongside the exact
descriptor so per-file URLs never need to enter the frozen
``ArtifactFile``/``ArtifactDescriptor`` schema. This file covers the
contract that replaces the old hard ``CatalogError`` refusal of any
not-yet-installed multi-file descriptor:

- the headline case: a genuine multi-file artifact provisions end-to-end
  with per-file URLs supplied via ``ArtifactSourceMap``;
- back-compat: a single-file descriptor with a bare ``source_url`` and no
  map entry keeps working exactly as before this task existed;
- resolution is TOTAL and validated at ``preflight()`` time, before any
  consent or network: a missing entry, an extra entry, a non-``http(s)``
  scheme, a query string, or userinfo each raise ``CatalogError`` naming
  the artifact and the offending file path, never the URL text itself;
- the consent fingerprint covers caller-supplied source identities, so a
  URL swapped between ``preflight()`` and ``provision()`` raises
  ``ConsentMismatchError`` instead of silently fetching from a different
  origin under stale consent.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace as dc_replace
from urllib.parse import urlparse

import httpx
import pytest

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
    ConsentMismatchError,
)
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService


def _trusted(srv: FixtureArtifactServer) -> frozenset:
    """Trusted-origins set for a fixture server (see test_stream_fetch.py's
    identical helper for why this is the bare hostname, not a URL)."""

    return frozenset({urlparse(srv.url("/")).hostname})


def _two_file_descriptor(
    ref: ArtifactRef,
    source_url: str = "https://example.test/model",
    *,
    role: ArtifactRole = ArtifactRole.ROOT,
) -> ArtifactDescriptor:
    """A genuine 2-file descriptor with real, independently-verifiable
    content -- unlike the pre-1695 fixtures of the same shape (e.g.
    ``test_preflight.py``'s ``_two_file_descriptor``), whose ONLY job was
    to trip the old blanket multi-file refusal, this one is exercised
    end-to-end over a real fixture server, so its declared sizes/digests
    must actually match ``a.bin``/``b.bin``'s served bytes. ``role`` is
    settable because ``core.activate``'s closure resolution rejects an
    installed dependency whose manifest role isn't ``DEPENDENCY``."""

    files = (
        ArtifactFile("a.bin", 4, hashlib.sha256(b"aaaa").hexdigest()),
        ArtifactFile("b.bin", 4, hashlib.sha256(b"bbbb").hexdigest()),
    )
    return ArtifactDescriptor(
        reference=ref,
        model_id="test/model",
        role=role,
        format=ArtifactFormat.ONNX,
        consumer="test",
        model_family="test-family",
        upstream_repository="test/repo",
        upstream_revision="main",
        source_url=source_url,
        precision=ref.variant,
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


def _forbid_client() -> httpx.AsyncClient:
    """``client_factory`` stub that fails any test proving "no network yet".

    Passed as ``ArtifactAcquisitionService(client_factory=_forbid_client)``:
    if ``preflight()`` ever reached its gating probe (or anything else
    touched the network), this raises immediately instead of silently
    succeeding -- proving a validation failure inside ``_aggregate_closure``
    aborts before ``_probe_gating`` (or any other network path) is ever
    reached, not just that the test happens not to hit one.
    """

    raise AssertionError("no network client should be constructed")


# ---------------------------------------------------------------------------
# Headline: a genuine multi-file artifact provisions end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_file_artifact_provisions_end_to_end_with_source_map(tmp_path):
    """A 2-file descriptor -- an unconditional ``CatalogError`` at
    ``preflight()`` before this task -- now provisions successfully
    end-to-end once the caller supplies a per-file ``ArtifactSourceMap``.
    This is the headline behavior TASK-1695 adds: it was categorically
    impossible before (see ``_require_single_file`` in the pre-1695
    history)."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    with FixtureArtifactServer() as srv:
        srv.serve("/a.bin", b"aaaa", etag='"va"', support_range=True)
        srv.serve("/b.bin", b"bbbb", etag='"vb"', support_range=True)
        desc = _two_file_descriptor(root)
        catalog = DictCatalog({root: desc})
        sources = {root: {"a.bin": srv.url("/a.bin"), "b.bin": srv.url("/b.bin")}}
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )

        report = await svc.preflight(root, catalog, sources=sources)
        assert report.gating_errors == ()
        consent = report.grant()

        activated = await svc.provision(root, consent, catalog, sources=sources)
        assert activated == root

        assert srv.requests, "the fixture server must have actually been hit"

    installed_refs = {
        item.descriptor.reference for item in core.list_installed() if item.descriptor is not None
    }
    assert root in installed_refs
    destination = core.artifact_path(root)
    assert (destination / "a.bin").read_bytes() == b"aaaa"
    assert (destination / "b.bin").read_bytes() == b"bbbb"


@pytest.mark.asyncio
async def test_multi_file_artifact_with_dependency_provisions_via_source_map(tmp_path):
    """Per-file source maps compose across a closure: root and dependency
    each resolve their OWN entries independently, keyed by ``ArtifactRef``."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-root", "r1", "int8")
    dep = ArtifactRef("multi-file-dep", "r1", "int8")
    with FixtureArtifactServer() as srv:
        srv.serve("/root-a.bin", b"aaaa", etag='"ra"', support_range=True)
        srv.serve("/root-b.bin", b"bbbb", etag='"rb"', support_range=True)
        srv.serve("/dep-a.bin", b"aaaa", etag='"da"', support_range=True)
        srv.serve("/dep-b.bin", b"bbbb", etag='"db"', support_range=True)

        root_desc = _two_file_descriptor(root)
        # Give the root its own real dependency edge.
        root_desc = dc_replace(root_desc, dependencies=(dep,))
        dep_desc = _two_file_descriptor(dep, role=ArtifactRole.DEPENDENCY)
        catalog = DictCatalog({root: root_desc, dep: dep_desc})
        sources = {
            root: {"a.bin": srv.url("/root-a.bin"), "b.bin": srv.url("/root-b.bin")},
            dep: {"a.bin": srv.url("/dep-a.bin"), "b.bin": srv.url("/dep-b.bin")},
        }
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )

        report = await svc.preflight(root, catalog, sources=sources)
        consent = report.grant()
        activated = await svc.provision(root, consent, catalog, sources=sources)
        assert activated == root

    installed_refs = {
        item.descriptor.reference for item in core.list_installed() if item.descriptor is not None
    }
    assert root in installed_refs
    assert dep in installed_refs
    assert (core.artifact_path(dep) / "a.bin").read_bytes() == b"aaaa"


# ---------------------------------------------------------------------------
# Back-compat: single-file + bare source_url, no map (or an empty one).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_file_source_url_without_source_map_still_works(tmp_path):
    """A single-file descriptor resolves its declared file's URL from
    ``descriptor.source_url`` when ``sources`` is omitted entirely --
    exactly the pre-1695 contract, unchanged."""

    core = ModelArtifactService(tmp_path / "root")
    with FixtureArtifactServer() as srv:
        body = b"single-file-body"
        srv.serve("/m.onnx", body, etag='"v1"', support_range=True)
        root = ArtifactRef("single-file-model", "r1", "int8")
        desc = make_descriptor(ref=root, files_body=body, source_url=srv.url("/m.onnx"))
        catalog = DictCatalog({root: desc})
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )

        report = await svc.preflight(root, catalog)  # sources omitted entirely
        consent = report.grant()
        activated = await svc.provision(root, consent, catalog)  # sources omitted entirely
        assert activated == root

    assert (core.artifact_path(root) / "model.onnx").read_bytes() == body


@pytest.mark.asyncio
async def test_single_file_source_url_with_empty_source_map_still_works(tmp_path):
    """Identical to the above, but ``sources={}`` is passed explicitly --
    an empty (or ref-missing) map must behave exactly like ``None``."""

    core = ModelArtifactService(tmp_path / "root")
    with FixtureArtifactServer() as srv:
        body = b"single-file-body-2"
        srv.serve("/m.onnx", body, etag='"v1"', support_range=True)
        root = ArtifactRef("single-file-model-2", "r1", "int8")
        desc = make_descriptor(ref=root, files_body=body, source_url=srv.url("/m.onnx"))
        catalog = DictCatalog({root: desc})
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )

        report = await svc.preflight(root, catalog, sources={})
        consent = report.grant()
        activated = await svc.provision(root, consent, catalog, sources={})
        assert activated == root

    assert (core.artifact_path(root) / "model.onnx").read_bytes() == body


# ---------------------------------------------------------------------------
# Resolution is total, validated at preflight() -- before any consent or
# network. Each scenario below asserts srv.requests is never touched AND
# (via _forbid_client) that no HTTP client is ever even constructed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_source_map_entry_raises_catalog_error_before_network(tmp_path):
    """A multi-file descriptor with one file's URL missing from the map
    fails preflight() with a CatalogError naming the artifact and the
    unresolved path -- before any consent or network."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    desc = _two_file_descriptor(root)
    catalog = DictCatalog({root: desc})
    svc = ArtifactAcquisitionService(
        core, free_bytes_probe=lambda p: 10**12, client_factory=_forbid_client
    )
    sources = {root: {"a.bin": "https://example.test/a.bin"}}  # b.bin missing

    with pytest.raises(CatalogError) as excinfo:
        await svc.preflight(root, catalog, sources=sources)
    assert "multi-file-model" in str(excinfo.value)
    assert "b.bin" in str(excinfo.value)


@pytest.mark.asyncio
async def test_extra_source_map_entry_raises_catalog_error_before_network(tmp_path):
    """A source-map entry naming a file the descriptor does NOT declare
    fails preflight() with a CatalogError naming the artifact and the
    undeclared path -- before any consent or network."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    desc = _two_file_descriptor(root)
    catalog = DictCatalog({root: desc})
    svc = ArtifactAcquisitionService(
        core, free_bytes_probe=lambda p: 10**12, client_factory=_forbid_client
    )
    sources = {
        root: {
            "a.bin": "https://example.test/a.bin",
            "b.bin": "https://example.test/b.bin",
            "c.bin": "https://example.test/c.bin",  # not declared
        }
    }

    with pytest.raises(CatalogError) as excinfo:
        await svc.preflight(root, catalog, sources=sources)
    assert "multi-file-model" in str(excinfo.value)
    assert "c.bin" in str(excinfo.value)


@pytest.mark.asyncio
async def test_non_http_scheme_source_url_raises_catalog_error_before_network(tmp_path):
    """A non-``http(s)`` scheme (e.g. ``ftp://``) fails preflight() with a
    CatalogError naming the artifact and the offending path -- before any
    consent or network -- and the error never quotes the URL text itself."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    desc = _two_file_descriptor(root)
    catalog = DictCatalog({root: desc})
    svc = ArtifactAcquisitionService(
        core, free_bytes_probe=lambda p: 10**12, client_factory=_forbid_client
    )
    sources = {
        root: {
            "a.bin": "ftp://example.test/a.bin",
            "b.bin": "https://example.test/b.bin",
        }
    }

    with pytest.raises(CatalogError) as excinfo:
        await svc.preflight(root, catalog, sources=sources)
    message = str(excinfo.value)
    assert "multi-file-model" in message
    assert "a.bin" in message
    assert "ftp://example.test/a.bin" not in message


@pytest.mark.asyncio
async def test_query_string_source_url_raises_catalog_error_before_network(tmp_path):
    """A query-string URL is credential-shaped and rejected -- and its
    text (which could carry a signed-URL token) never reaches the error
    message."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    desc = _two_file_descriptor(root)
    catalog = DictCatalog({root: desc})
    svc = ArtifactAcquisitionService(
        core, free_bytes_probe=lambda p: 10**12, client_factory=_forbid_client
    )
    sources = {
        root: {
            "a.bin": "https://example.test/a.bin?token=super-secret-value",
            "b.bin": "https://example.test/b.bin",
        }
    }

    with pytest.raises(CatalogError) as excinfo:
        await svc.preflight(root, catalog, sources=sources)
    message = str(excinfo.value)
    assert "multi-file-model" in message
    assert "a.bin" in message
    assert "super-secret-value" not in message
    assert "token=" not in message


@pytest.mark.asyncio
async def test_userinfo_source_url_raises_catalog_error_before_network(tmp_path):
    """A userinfo (``user:pass@host``) URL is credential-shaped and
    rejected -- and its text never reaches the error message."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    desc = _two_file_descriptor(root)
    catalog = DictCatalog({root: desc})
    svc = ArtifactAcquisitionService(
        core, free_bytes_probe=lambda p: 10**12, client_factory=_forbid_client
    )
    sources = {
        root: {
            "a.bin": "https://svc-user:svc-pass@example.test/a.bin",
            "b.bin": "https://example.test/b.bin",
        }
    }

    with pytest.raises(CatalogError) as excinfo:
        await svc.preflight(root, catalog, sources=sources)
    message = str(excinfo.value)
    assert "multi-file-model" in message
    assert "a.bin" in message
    assert "svc-user" not in message
    assert "svc-pass" not in message


@pytest.mark.asyncio
async def test_resolution_failure_at_provision_also_precedes_any_side_effect(tmp_path):
    """The same total-resolution validation runs again inside provision()'s
    independent catalog/source re-walk (defense-in-depth against a source
    map that changed shape between preflight() and provision()) -- and
    still fails before the session lease does anything durable."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    desc = _two_file_descriptor(root)
    catalog = DictCatalog({root: desc})
    # No client_factory override here: preflight() with a fully-resolved
    # good_sources map legitimately reaches the network gating probe (which
    # tolerates an unreachable "example.test" host by design -- see
    # _probe_gating's own httpx.HTTPError/EgressBlockedError handling).
    # What THIS test proves is narrower: provision()'s re-walk with a
    # BROKEN map fails before touching staging at all.
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    good_sources = {
        root: {
            "a.bin": "https://example.test/a.bin",
            "b.bin": "https://example.test/b.bin",
        }
    }
    report = await svc.preflight(root, catalog, sources=good_sources)
    consent = report.grant()

    broken_sources = {root: {"a.bin": "https://example.test/a.bin"}}  # b.bin missing now
    with pytest.raises(CatalogError):
        await svc.provision(root, consent, catalog, sources=broken_sources)

    # _aggregate_closure raises before the per-artifact loop ever creates a
    # download stage -- so the (always-present, ModelArtifactService creates
    # it eagerly on construction) staging directory stays completely empty.
    assert list((tmp_path / "root" / "staging").iterdir()) == []


# ---------------------------------------------------------------------------
# Consent fingerprint covers caller-supplied source identities.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_url_changed_after_consent_raises_consent_mismatch(tmp_path):
    """Swapping a source-map URL between preflight() and provision() must
    invalidate consent -- the spec's rule that the fingerprint covers
    credential-free source identities, not just the closure's reference
    set."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    desc = _two_file_descriptor(root)
    catalog = DictCatalog({root: desc})
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)

    sources_v1 = {
        root: {
            "a.bin": "https://example.test/a.bin",
            "b.bin": "https://example.test/b.bin",
        }
    }
    report = await svc.preflight(root, catalog, sources=sources_v1)
    consent = report.grant()

    sources_v2 = {
        root: {
            "a.bin": "https://example.test/a-DIFFERENT-origin.bin",
            "b.bin": "https://example.test/b.bin",
        }
    }
    with pytest.raises(ConsentMismatchError):
        await svc.provision(root, consent, catalog, sources=sources_v2)


@pytest.mark.asyncio
async def test_identical_source_map_at_provision_does_not_mismatch(tmp_path):
    """Sanity counterpart to the swap test above: re-supplying the SAME
    source map at provision() must NOT raise -- proves the fingerprint is
    deterministic across two independent resolutions of identical input,
    not merely order-sensitive or otherwise unstable."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    with FixtureArtifactServer() as srv:
        srv.serve("/a.bin", b"aaaa", etag='"va"', support_range=True)
        srv.serve("/b.bin", b"bbbb", etag='"vb"', support_range=True)
        desc = _two_file_descriptor(root)
        catalog = DictCatalog({root: desc})
        sources = {root: {"a.bin": srv.url("/a.bin"), "b.bin": srv.url("/b.bin")}}
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )

        report = await svc.preflight(root, catalog, sources=sources)
        consent = report.grant()
        activated = await svc.provision(root, consent, catalog, sources=sources)
        assert activated == root


@pytest.mark.asyncio
async def test_source_map_fingerprint_matches_plain_closure_fingerprint_when_absent(tmp_path):
    """When no caller ever passes ``sources``, the consent fingerprint is
    BYTE-FOR-BYTE the plain ``closure_fingerprint(root, deps)`` this
    codebase used before TASK-1695 -- the back-compat guarantee that lets
    an ``AcquisitionConsent`` hand-built from that older function (a
    pattern many existing tests and, potentially, callers still use) keep
    matching ``provision()``'s own recomputed report."""

    from tldw_chatbook.Model_Artifacts import closure_fingerprint

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("single-file-model-3", "r1", "int8")
    desc = make_descriptor(ref=root)
    catalog = DictCatalog({root: desc})
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)

    report = await svc.preflight(root, catalog)
    assert report.closure_fingerprint == closure_fingerprint(root, ())
