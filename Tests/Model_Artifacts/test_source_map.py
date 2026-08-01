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

from dataclasses import replace as dc_replace

import httpx
import pytest

from Tests.Model_Artifacts.acquisition_test_helpers import (
    _trusted,
    _two_file_descriptor,
    grant_consent,
)
from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from Tests.Model_Artifacts.test_acquisition_types import DictCatalog, make_descriptor
from tldw_chatbook.Model_Artifacts import ArtifactRef, ArtifactRole
from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactAcquisitionService,
    CatalogError,
    ConsentMismatchError,
)
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService


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
    still fails before the session lease does anything durable.

    PR-1165 review (P2): ``good_sources`` points at a real, loopback
    fixture server (not an ``example.test`` placeholder) so
    ``preflight()``'s network gating probe -- which a fully-resolved
    source map legitimately reaches -- never performs real DNS. What this
    test actually proves is narrower and doesn't depend on that probe's
    outcome either way: provision()'s re-walk with a BROKEN map fails
    before touching staging at all.
    """

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    desc = _two_file_descriptor(root)
    catalog = DictCatalog({root: desc})
    with FixtureArtifactServer() as srv:
        srv.serve("/a.bin", b"aaaa", etag='"va"', support_range=True)
        srv.serve("/b.bin", b"bbbb", etag='"vb"', support_range=True)
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        good_sources = {root: {"a.bin": srv.url("/a.bin"), "b.bin": srv.url("/b.bin")}}
        report = await svc.preflight(root, catalog, sources=good_sources)
        consent = report.grant()

        broken_sources = {root: {"a.bin": srv.url("/a.bin")}}  # b.bin missing now
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
    set.

    PR-1165 review (P2): ``sources_v1`` points at a real, loopback fixture
    server so ``preflight()``'s network gating probe never performs real
    DNS. ``sources_v2`` (never actually fetched -- provision() raises
    before any network access, straight off the recomputed fingerprint)
    stays a syntactically-valid but unserved URL on that same server.
    """

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("multi-file-model", "r1", "int8")
    desc = _two_file_descriptor(root)
    catalog = DictCatalog({root: desc})
    with FixtureArtifactServer() as srv:
        srv.serve("/a.bin", b"aaaa", etag='"va"', support_range=True)
        srv.serve("/b.bin", b"bbbb", etag='"vb"', support_range=True)
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )

        sources_v1 = {root: {"a.bin": srv.url("/a.bin"), "b.bin": srv.url("/b.bin")}}
        report = await svc.preflight(root, catalog, sources=sources_v1)
        consent = report.grant()

        sources_v2 = {
            root: {
                "a.bin": srv.url("/a-DIFFERENT-origin.bin"),
                "b.bin": srv.url("/b.bin"),
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
async def test_single_file_fallback_fingerprint_differs_from_plain_closure_fingerprint(
    tmp_path,
):
    """TASK-1712 (PR-1165 review, P1): closes a residual TASK-1695 left open.

    Superseded assertion: this test used to assert the consent fingerprint
    was BYTE-FOR-BYTE the plain ``closure_fingerprint(root, deps)`` even
    when the closure resolves a not-yet-installed single-file descriptor's
    OWN ``source_url`` (the fallback ``_resolve_file_sources`` uses when no
    explicit ``sources`` map is supplied at all). That was itself the
    consent hole TASK-1712 closes: a caller-supplied ``sources`` entry was
    folded into the fingerprint, but the fallback-resolved ``source_url``
    was not, so a dynamic ``ArtifactCatalog`` (mirror rotation, CDN
    rebalancing -- the protocol guarantees no immutability) could change it
    between ``preflight()`` and ``provision()`` with nothing to catch it.
    Every resolved source now folds in uniformly, so the fingerprint is no
    longer plain-closure-equivalent for ANY not-yet-installed entry,
    caller-supplied source map or not.

    ``grant_consent`` (network-free) is used instead of a real
    ``preflight()`` call -- ``make_descriptor``'s default ``source_url`` is
    an unreachable ``example.test`` placeholder, and this test's own setup
    has no need to touch the network.
    """

    from tldw_chatbook.Model_Artifacts import closure_fingerprint

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("single-file-model-3", "r1", "int8")
    desc = make_descriptor(ref=root)
    catalog = DictCatalog({root: desc})
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)

    consent = grant_consent(svc, root, catalog)
    assert consent.closure_fingerprint != closure_fingerprint(root, ())


@pytest.mark.asyncio
async def test_single_file_source_url_changed_after_consent_raises_consent_mismatch(
    tmp_path,
):
    """TASK-1712's headline acceptance criterion.

    A single-file descriptor with NO explicit source map at all -- its
    file's URL resolved purely via the ``descriptor.source_url`` fallback
    -- must still be consent-gated exactly like a caller-supplied
    ``sources`` entry. Before this fix, ``_closure_fingerprint_with_sources``
    only folded in caller-supplied entries, so this exact scenario --
    ``descriptor.source_url`` changing between ``grant()`` and
    ``provision()`` with no explicit source map involved -- was NOT
    consent-gated: ``provision()`` would silently fetch from the new origin
    under stale consent.
    """

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("single-file-model-4", "r1", "int8")
    desc_v1 = make_descriptor(ref=root, source_url="https://mirror-a.example.test/model")
    catalog_v1 = DictCatalog({root: desc_v1})
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)

    consent = grant_consent(svc, root, catalog_v1)

    desc_v2 = dc_replace(desc_v1, source_url="https://mirror-b.example.test/model")
    catalog_v2 = DictCatalog({root: desc_v2})

    with pytest.raises(ConsentMismatchError):
        await svc.provision(root, consent, catalog_v2)


# ---------------------------------------------------------------------------
# PR-1165 review (P2): the gating probe targets the REAL per-file URLs the
# closure will actually be fetched from, not the descriptor's own
# source_url -- which a caller-supplied source map may never be fetched
# from at all.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gated_mapped_file_detected_at_preflight_even_when_descriptor_url_is_public(
    tmp_path,
):
    """A per-file source-map URL on a gated origin must be probed, even
    though the descriptor's own ``source_url`` (never actually fetched
    from once a source map supplies every declared file) sits on a fully
    public origin.

    Regression test: before this fix, ``_probe_gating`` only ever probed
    ``descriptor.source_url`` -- the public origin here -- so a gated
    mapped file was never even reached during preflight(); its 401 would
    only have surfaced mid-transfer, well after consent was granted.
    """

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("mapped-gated-model", "r1", "int8")
    with FixtureArtifactServer() as public_repo, FixtureArtifactServer() as gated_cdn:
        public_repo.serve("/descriptor-landing", b"unused")
        gated_cdn.serve("/a.bin", b"aaaa", require_token="cdn-secret", etag='"va"')
        gated_cdn.serve("/b.bin", b"bbbb", require_token="cdn-secret", etag='"vb"')

        desc = _two_file_descriptor(root, public_repo.url("/descriptor-landing"))
        catalog = DictCatalog({root: desc})
        sources = {
            root: {"a.bin": gated_cdn.url("/a.bin"), "b.bin": gated_cdn.url("/b.bin")}
        }
        svc = ArtifactAcquisitionService(
            core,
            free_bytes_probe=lambda p: 10**12,
            trusted_origins=_trusted(public_repo) | _trusted(gated_cdn),
        )

        report = await svc.preflight(root, catalog, sources=sources)

    assert report.gating_errors, "the gated mapped-file origin must have been probed"


@pytest.mark.asyncio
async def test_gated_descriptor_url_does_not_block_public_mapped_files(tmp_path):
    """A gated ``descriptor.source_url`` must not block consent for a
    closure whose ACTUAL per-file URLs are all on a public origin --
    nothing will ever be fetched from ``source_url`` once a source map
    supplies every declared file.

    Regression test: before this fix, ``_probe_gating`` probed
    ``descriptor.source_url`` -- gated here -- unconditionally, so
    preflight() reported this closure gated even though the real fetch
    would never touch that origin at all.
    """

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("descriptor-gated-model", "r1", "int8")
    with FixtureArtifactServer() as gated_repo, FixtureArtifactServer() as public_cdn:
        gated_repo.serve("/descriptor-landing", b"unused", require_token="repo-secret")
        public_cdn.serve("/a.bin", b"aaaa", etag='"va"')
        public_cdn.serve("/b.bin", b"bbbb", etag='"vb"')

        desc = _two_file_descriptor(root, gated_repo.url("/descriptor-landing"))
        catalog = DictCatalog({root: desc})
        sources = {
            root: {"a.bin": public_cdn.url("/a.bin"), "b.bin": public_cdn.url("/b.bin")}
        }
        svc = ArtifactAcquisitionService(
            core,
            free_bytes_probe=lambda p: 10**12,
            trusted_origins=_trusted(gated_repo) | _trusted(public_cdn),
        )

        report = await svc.preflight(root, catalog, sources=sources)

    assert report.gating_errors == ()
    report.grant()  # must not raise
