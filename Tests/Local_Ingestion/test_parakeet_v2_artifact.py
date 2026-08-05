"""TASK-1696: the Parakeet v2 managed-download adapter.

Covers the new ``tldw_chatbook.Local_Ingestion.parakeet_v2_artifact``
module: the exact descriptor/source-map/catalog built from the existing
curated installer's pinned constants, the shared managed store root, the
managed-first resolver (using only ``Model_Artifacts.service``, never the
async acquisition/HTTP layer), and an end-to-end managed install driven
through ``run_parakeet_v2_preflight``/``run_parakeet_v2_provision`` against
the localhost fixture server -- no real network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tldw_chatbook.Local_Ingestion import parakeet_v2_artifact as artifact
from tldw_chatbook.Local_Ingestion import parakeet_v2_installer as installer
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService

from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer


def _trusted(srv: FixtureArtifactServer) -> frozenset:
    """Trusted-origins set for a fixture server (see acquisition_test_helpers)."""

    return frozenset({urlparse(srv.url("/")).hostname})


# ---------------------------------------------------------------------------
# Descriptor and source map: single source of truth with the installer.
# ---------------------------------------------------------------------------


def test_descriptor_files_match_installer_pinned_constants_exactly() -> None:
    descriptor = artifact.parakeet_v2_descriptor()

    assert [(f.path, f.size_bytes, f.sha256) for f in descriptor.files] == [
        (bundle.filename, bundle.size_bytes, bundle.sha256)
        for bundle in installer.PARAKEET_V2_FILES
    ]
    assert descriptor.expected_installed_bytes == installer.PARAKEET_V2_TOTAL_BYTES
    assert descriptor.upstream_repository == installer.PARAKEET_V2_REPOSITORY
    assert descriptor.upstream_revision == installer.PARAKEET_V2_REVISION
    assert descriptor.reference.revision == installer.PARAKEET_V2_REVISION
    assert descriptor.license_id == installer.PARAKEET_V2_LICENSE
    assert descriptor.precision == descriptor.reference.variant == "int8"


def test_descriptor_provenance_is_curated_and_locally_recorded_not_verified() -> None:
    """Pins the honest provenance label, not the stronger one.

    Only 2 of the 4 pinned files (``encoder-model.int8.onnx``,
    ``decoder_joint-model.int8.onnx``) are LFS-tracked on HuggingFace for
    the pinned revision -- the only case HuggingFace supplies a
    repository-provided SHA256 for. ``config.json`` and ``vocab.txt`` are
    plain git blobs (only a git SHA1 oid, no SHA256), so their pinned
    digests were necessarily computed locally, not repository-supplied.
    Per ADR-025, a per-artifact provenance claim for a MIXED artifact must
    use the weaker label. If this ever changes to ``INTEGRITY_VERIFIED``,
    that's a claim that ALL four files now carry a repository-supplied
    digest -- re-verify against HuggingFace's tree API for the (possibly
    new) pinned revision before loosening this assertion.
    """
    from tldw_chatbook.Model_Artifacts.service import ProvenanceClass

    descriptor = artifact.parakeet_v2_descriptor()
    assert descriptor.provenance == (
        ProvenanceClass.CHATBOOK_CURATED,
        ProvenanceClass.LOCAL_INTEGRITY_RECORDED,
    )
    assert ProvenanceClass.INTEGRITY_VERIFIED not in descriptor.provenance


def test_source_map_covers_every_declared_file_with_credential_free_https_urls() -> None:
    ref = artifact.parakeet_v2_reference()
    source_map = artifact.parakeet_v2_source_map()

    assert set(source_map) == {ref}
    urls = source_map[ref]
    declared = {bundle.filename for bundle in installer.PARAKEET_V2_FILES}
    assert set(urls) == declared
    for filename, url in urls.items():
        assert url == installer._source_url(filename)
        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.username is None and parsed.password is None
        assert not parsed.query and not parsed.fragment


def test_catalog_returns_descriptor_for_known_ref_and_raises_keyerror_otherwise() -> None:
    catalog = artifact.ParakeetV2Catalog()
    ref = artifact.parakeet_v2_reference()

    resolved = catalog.descriptor(ref)
    assert resolved == artifact.parakeet_v2_descriptor()

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    with pytest.raises(KeyError):
        catalog.descriptor(ArtifactRef("other-model", "r1", "int8"))


# ---------------------------------------------------------------------------
# Managed store root: sibling of the legacy installer's own destination.
# ---------------------------------------------------------------------------


def test_managed_store_root_is_sibling_of_legacy_stt_install_dir() -> None:
    managed_root = artifact.managed_model_artifact_root()
    legacy_dir = installer.parakeet_v2_install_dir()

    assert managed_root.parent == legacy_dir.parent.parent
    assert managed_root.name == "managed"
    assert legacy_dir.parent.name == "stt"


# ---------------------------------------------------------------------------
# Managed-first resolver: only Model_Artifacts.service, never acquisition.
# ---------------------------------------------------------------------------


def test_active_managed_dir_returns_none_when_nothing_installed(tmp_path: Path) -> None:
    core = ModelArtifactService(tmp_path / "root")
    assert artifact.active_managed_parakeet_v2_dir(service=core) is None


def test_active_managed_dir_returns_none_when_store_cannot_be_opened(
    tmp_path: Path, monkeypatch
) -> None:
    # A plain file where the managed store root should be is not a store
    # ``ModelArtifactService`` can open -- treated as "nothing managed yet"
    # rather than propagating the construction error to the caller, which
    # always has a further fallback (the verified legacy bundle).
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setattr(artifact, "managed_model_artifact_root", lambda: blocked)

    assert artifact.active_managed_parakeet_v2_dir() is None


def test_active_managed_dir_finds_ready_active_root_using_only_service_api(
    tmp_path: Path,
) -> None:
    """Prove the resolver works with only ``Model_Artifacts.service`` --
    installs and activates a real artifact through ``core.install()``/
    ``core.activate()`` directly, never through the acquisition layer.

    ``Tests/Model_Artifacts/test_credentials_and_boundaries.py::
    test_stt_and_transcription_worker_modules_never_import_acquisition_or_fetch``
    covers the module-import-graph half of this guarantee (this module is
    now reached transitively through
    ``Local_Ingestion.transcription_service``); this test covers the
    functional half: a managed artifact set up with ONLY the synchronous
    core API is still found.
    """

    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactDescriptor,
        ArtifactFile,
        ArtifactFormat,
        ArtifactRole,
        ProvenanceClass,
    )

    core = ModelArtifactService(tmp_path / "root")
    ref = artifact.parakeet_v2_reference()
    payload = b"tiny-config-bytes"
    descriptor = ArtifactDescriptor(
        reference=ref,
        model_id="test/model",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="test",
        model_family="test-family",
        upstream_repository="test/repo",
        upstream_revision="main",
        source_url="https://example.test/model",
        precision=ref.variant,
        license_id="test-license",
        license_url="https://example.test/license",
        usage_notice="Test model",
        runtime_name="test-runtime",
        runtime_version_constraint="==1.0.0",
        supported_os=("linux",),
        supported_architectures=("x86-64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=(ArtifactFile("config.json", len(payload), hashlib.sha256(payload).hexdigest()),),
        expected_installed_bytes=len(payload),
        dependencies=(),
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_bytes(payload)

    core.install(descriptor, source_dir)
    core.activate(ref)

    resolved = artifact.active_managed_parakeet_v2_dir(service=core)
    assert resolved == core.artifact_path(ref)


def test_active_managed_dir_ignores_non_parakeet_active_artifact(tmp_path: Path) -> None:
    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactDescriptor,
        ArtifactFile,
        ArtifactFormat,
        ArtifactRef,
        ArtifactRole,
        ProvenanceClass,
    )

    core = ModelArtifactService(tmp_path / "root")
    other_ref = ArtifactRef("some-other-model", "main", "fp16")
    payload = b"x"
    descriptor = ArtifactDescriptor(
        reference=other_ref,
        model_id="test/model",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="test",
        model_family="test-family",
        upstream_repository="test/repo",
        upstream_revision="main",
        source_url="https://example.test/model",
        precision=other_ref.variant,
        license_id="test-license",
        license_url="https://example.test/license",
        usage_notice="Test model",
        runtime_name="test-runtime",
        runtime_version_constraint="==1.0.0",
        supported_os=("linux",),
        supported_architectures=("x86-64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=(ArtifactFile("model.bin", len(payload), hashlib.sha256(payload).hexdigest()),),
        expected_installed_bytes=len(payload),
        dependencies=(),
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "model.bin").write_bytes(payload)
    core.install(descriptor, source_dir)
    core.activate(other_ref)

    assert artifact.active_managed_parakeet_v2_dir(service=core) is None


def test_active_managed_dir_rejects_same_id_different_revision(tmp_path: Path) -> None:
    """PR-1167 review (Finding 1): matching on ``artifact_id`` alone would
    wrongly return an active ``parakeet-v2`` artifact pinned to a
    DIFFERENT revision/variant than the one this adapter's descriptor
    declares -- latent today (only one descriptor exists), but a real
    misresolution the instant a second revision/variant is ever installed
    and activated, which is exactly what the managed store supports. The
    resolver must match the FULL ``ArtifactRef`` and fall through (return
    ``None``, letting the caller continue to its next resolution step) for
    anything else.
    """

    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactDescriptor,
        ArtifactFile,
        ArtifactFormat,
        ArtifactRef,
        ArtifactRole,
        ProvenanceClass,
    )

    core = ModelArtifactService(tmp_path / "root")
    # Same artifact_id ("parakeet-v2") as artifact.parakeet_v2_reference(),
    # but a different revision -- a distinct, real artifact once the store
    # holds more than one.
    other_revision_ref = ArtifactRef(
        artifact.PARAKEET_V2_ARTIFACT_ID, "a" * 40, artifact.PARAKEET_V2_VARIANT
    )
    assert other_revision_ref != artifact.parakeet_v2_reference()
    payload = b"different-revision-bytes"
    descriptor = ArtifactDescriptor(
        reference=other_revision_ref,
        model_id="test/model",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="test",
        model_family="test-family",
        upstream_repository="test/repo",
        upstream_revision="main",
        source_url="https://example.test/model",
        precision=other_revision_ref.variant,
        license_id="test-license",
        license_url="https://example.test/license",
        usage_notice="Test model",
        runtime_name="test-runtime",
        runtime_version_constraint="==1.0.0",
        supported_os=("linux",),
        supported_architectures=("x86-64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=(ArtifactFile("config.json", len(payload), hashlib.sha256(payload).hexdigest()),),
        expected_installed_bytes=len(payload),
        dependencies=(),
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_bytes(payload)
    core.install(descriptor, source_dir)
    core.activate(other_revision_ref)

    # The resolver must NOT return the different-revision artifact, even
    # though it shares artifact_id and is genuinely active.
    assert artifact.active_managed_parakeet_v2_dir(service=core) is None


# ---------------------------------------------------------------------------
# End-to-end managed install against the localhost fixture server.
# ---------------------------------------------------------------------------


def _tiny_files(payloads: dict[str, bytes]) -> tuple[installer.BundleFile, ...]:
    return tuple(
        installer.BundleFile(
            filename=filename,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for filename, payload in payloads.items()
    )


@pytest.mark.asyncio
async def test_run_parakeet_v2_preflight_and_provision_against_localhost_fixture(
    tmp_path: Path, monkeypatch
) -> None:
    """preflight -> grant -> provision -> the active managed directory is
    returned, driven entirely through this module's own orchestration
    helpers against a real (localhost) HTTP server -- no real network.

    The pinned bundle is 630+ MiB, so the installer's module-level
    constants are monkeypatched to tiny substitute files first (the same
    pattern ``Tests/Local_Ingestion/test_parakeet_v2_installer.py`` already
    uses for its own network tests) and ``installer._source_url`` is
    monkeypatched to point at the fixture server instead of
    huggingface.co. Both ``parakeet_v2_descriptor()`` and
    ``parakeet_v2_source_map()`` read these as module attributes at call
    time (not a rebound import), so the patch is observed here exactly as
    it would be in production.
    """

    payloads = {
        "config.json": b"tiny-config",
        "vocab.txt": b"tiny-vocab",
        "encoder-model.int8.onnx": b"tiny-encoder-bytes",
        "decoder_joint-model.int8.onnx": b"tiny-decoder-bytes",
    }
    monkeypatch.setattr(installer, "PARAKEET_V2_FILES", _tiny_files(payloads))

    with FixtureArtifactServer() as srv:
        monkeypatch.setattr(
            installer, "_source_url", lambda filename: srv.url(f"/{filename}")
        )
        for filename, payload in payloads.items():
            srv.serve(f"/{filename}", payload, etag=f'"{filename}"', support_range=True)

        core = ModelArtifactService(tmp_path / "root")
        trusted = _trusted(srv)

        report = await artifact.run_parakeet_v2_preflight(
            core=core,
            free_bytes_probe=lambda p: 10**12,
            trusted_origins=trusted,
        )
        assert report.gating_errors == ()
        assert report.sufficient_space
        assert report.download_bytes == sum(len(p) for p in payloads.values())
        assert report.destination == core.artifact_path(artifact.parakeet_v2_reference())

        installed_dir = await artifact.run_parakeet_v2_provision(
            report,
            core=core,
            free_bytes_probe=lambda p: 10**12,
            trusted_origins=trusted,
        )

    assert installed_dir == core.artifact_path(artifact.parakeet_v2_reference())
    for filename, payload in payloads.items():
        assert (installed_dir / filename).read_bytes() == payload

    # The active managed resolver -- the same one the console/batch STT
    # paths use -- must now find exactly this directory.
    assert artifact.active_managed_parakeet_v2_dir(service=core) == installed_dir
