"""Shared test-only helpers for ``Tests/Model_Artifacts``'s acquisition suite.

PR-1165 review (P3): ``_trusted`` existed as byte-identical copies across six
files (``test_stream_fetch.py``, ``test_preflight.py``, ``test_source_map.py``,
``test_credentials_and_boundaries.py``, ``test_provision_install.py``,
``test_provision_fetch.py``), and ``test_credentials_and_boundaries.py``'s
``_two_file_descriptor_for_hygiene`` duplicated ``test_source_map.py``'s
``_two_file_descriptor`` almost line-for-line. Both now live here once.

``grant_consent`` is a third helper this same review's P1 fix (TASK-1720)
needs in many of the same files: folding every resolved per-file source URL
(including the single-file ``source_url`` fallback, not just caller-supplied
``sources`` entries) into the consent fingerprint means a hand-built
``AcquisitionConsent(closure_fingerprint=closure_fingerprint(root, deps))``
no longer matches what ``provision()`` recomputes for any closure with a
not-yet-installed entry. Centralizing the fix avoids reproducing the same
``_aggregate_closure(...).grant()`` one-liner at every one of those call
sites.
"""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse

from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
)
from tldw_chatbook.Model_Artifacts.acquisition import (
    AcquisitionConsent,
    ArtifactAcquisitionService,
    ArtifactCatalog,
    ArtifactSourceMap,
)

from .fixture_http import FixtureArtifactServer


def _trusted(srv: FixtureArtifactServer) -> frozenset:
    """Trusted-origins set for a fixture server, in egress's real format.

    ``tldw_chatbook.Utils.egress._normalize_trusted``/``_post_resolution``
    key membership on the bare, lowercased HOSTNAME (e.g. ``"127.0.0.1"``),
    not a scheme+host+port URL string. The fixture server binds to the
    loopback IP literal, which classifies as "private" under
    ``_classify_ip`` and would otherwise be egress-blocked; listing it here
    is what lets policy allow a probe or fetch to reach it.
    """

    return frozenset({urlparse(srv.url("/")).hostname})


def _two_file_descriptor(
    ref: ArtifactRef,
    source_url: str = "https://example.test/model",
    *,
    role: ArtifactRole = ArtifactRole.ROOT,
) -> ArtifactDescriptor:
    """A genuine 2-file descriptor with real, independently-verifiable content.

    Exercised end-to-end over a real fixture server in most callers, so its
    declared sizes/digests must actually match ``a.bin``/``b.bin``'s served
    bytes. ``role`` is settable because ``core.activate``'s closure
    resolution rejects an installed dependency whose manifest role isn't
    ``DEPENDENCY``.

    Distinct from ``test_preflight.py``'s own ``_two_file_descriptor``,
    which is deliberately NOT this helper: that one only ever needs to trip
    the "no resolvable URL" ``CatalogError`` path (file COUNT is all that
    matters there), never real content, so it stays local to that file.
    """

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


def grant_consent(
    svc: ArtifactAcquisitionService,
    root: ArtifactRef,
    catalog: ArtifactCatalog,
    sources: ArtifactSourceMap | None = None,
) -> AcquisitionConsent:
    """Grant real consent without a network round trip.

    Calls ``ArtifactAcquisitionService._aggregate_closure`` -- the same
    pure, network-free aggregation the public ``preflight()`` wraps with a
    gating probe (see its own docstring: "the only I/O is
    ``core.list_installed()`` and ``core.disk_usage()``") -- and grants the
    resulting report exactly like a caller would. Since TASK-1720 folded
    every resolved per-file source URL, including the single-file
    ``source_url`` fallback, into the real consent fingerprint, a
    hand-built ``AcquisitionConsent(closure_fingerprint=closure_fingerprint(
    root, deps))`` no longer matches what ``provision()`` recomputes for any
    closure containing a not-yet-installed entry. Going through this
    instead of the bare function means a test's consent can never drift
    from the real formula ``provision()`` validates against.

    Deliberately NOT ``await svc.preflight(root, catalog, sources=sources)``
    for every caller: several tests use placeholder (``example.test``) or
    otherwise unreachable ``source_url`` values specifically to keep their
    OWN setup network-free, and a real ``preflight()`` call would gating-
    probe those same URLs. Callers that DO want the real gating probe
    exercised should call ``preflight()`` directly instead of this helper.

    Args:
        svc: The service whose (network-free) aggregation to use.
        root: The root artifact reference to grant consent for.
        catalog: Catalog supplying descriptors for the closure walk.
        sources: Optional per-file ``ArtifactSourceMap``, forwarded
            unchanged.

    Returns:
        A real ``AcquisitionConsent`` matching what ``provision()`` will
        independently recompute for the same closure/catalog/sources.
    """

    _closure, report, _gating_targets, _resolved_sources = svc._aggregate_closure(
        root, catalog, sources
    )
    return report.grant()
