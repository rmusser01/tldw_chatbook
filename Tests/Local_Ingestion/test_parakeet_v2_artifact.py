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

# Network opt-in (task-15111): this module fetches artifacts from an in-
# process HTTP server on an ephemeral loopback port.
# The autouse guard in Tests/conftest.py denies egress by default; every address
# these tests reach is a port this process itself is listening on.
pytestmark = pytest.mark.allow_network


_EXPECTED_ROOTS = {
    ("nemo-parakeet-tdt-0.6b-v2", "int8"): {
        "repo": "istupakov/parakeet-tdt-0.6b-v2-onnx",
        "upstream_revision": "0bbb45a3365852604aef28b538a8f066f4ccaa85",
        "reference_revision": "0bbb45a3365852604aef28b538a8f066f4ccaa85-vad-b3e3ee3cce4c",
        "files": {
            "config.json": (97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466"),
            "vocab.txt": (9384, "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d"),
            "encoder-model.int8.onnx": (652184014, "3e0581fda6ab843888b51e56d7ee78b6d5bc3237ec113af1f732d1d5286aa155"),
            "decoder_joint-model.int8.onnx": (8998286, "a449f49acd68979d418651dd2dcb737cc0f1bf0225e009e29ee326354edbf7d3"),
        },
    },
    ("nemo-parakeet-tdt-0.6b-v2", "f32"): {
        "repo": "istupakov/parakeet-tdt-0.6b-v2-onnx",
        "upstream_revision": "0bbb45a3365852604aef28b538a8f066f4ccaa85",
        "reference_revision": "0bbb45a3365852604aef28b538a8f066f4ccaa85-vad-b3e3ee3cce4c",
        "files": {
            "config.json": (97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466"),
            "vocab.txt": (9384, "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d"),
            "encoder-model.onnx": (41770866, "3987bcd28175d829d12888a996a84e8f62a0e374d9ffd640662c1515adc679d3"),
            "encoder-model.onnx.data": (2435420160, "4dab7362d4874d85965045b1e41b2d61dd2cc0fb25671a7f6b3dc47bf120cc41"),
            "decoder_joint-model.onnx": (35792059, "cbb52a07bd70ab5b67f8439d4b3cd8704b18467b4430bcacb5adabe154b8d191"),
        },
    },
    ("nemo-parakeet-tdt-0.6b-v3", "int8"): {
        "repo": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "upstream_revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
        "reference_revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce-vad-b3e3ee3cce4c",
        "files": {
            "config.json": (97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466"),
            "vocab.txt": (93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
            "encoder-model.int8.onnx": (652183999, "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09"),
            "decoder_joint-model.int8.onnx": (18202004, "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70"),
        },
    },
    ("nemo-parakeet-tdt-0.6b-v3", "f32"): {
        "repo": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "upstream_revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
        "reference_revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce-vad-b3e3ee3cce4c",
        "files": {
            "config.json": (97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466"),
            "vocab.txt": (93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
            "encoder-model.onnx": (41770866, "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1"),
            "encoder-model.onnx.data": (2435420160, "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36"),
            "decoder_joint-model.onnx": (72520893, "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1"),
        },
    },
}


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
    assert descriptor.upstream_revision == installer.PARAKEET_V2_REVISION
    assert descriptor.license_id == installer.PARAKEET_V2_LICENSE
    assert descriptor.precision == descriptor.reference.variant == "int8"
    assert descriptor.dependencies == (artifact.parakeet_vad_reference(),)


@pytest.mark.parametrize("model_precision", _EXPECTED_ROOTS)
def test_managed_root_descriptors_match_task_593_exactly(model_precision) -> None:
    """Every admitted root uses only the measured TASK-593 payload files."""
    model, precision = model_precision
    expected = _EXPECTED_ROOTS[model_precision]

    descriptor = artifact.parakeet_descriptor(model, precision)

    assert descriptor.reference == artifact.parakeet_reference(model, precision)
    assert descriptor.reference.revision == expected["reference_revision"]
    assert descriptor.upstream_repository == expected["repo"]
    assert descriptor.upstream_revision == expected["upstream_revision"]
    assert descriptor.precision == precision
    assert descriptor.license_id == "CC-BY-4.0"
    assert descriptor.supported_os == ("linux", "darwin", "windows")
    assert descriptor.supported_architectures == ("x86-64", "arm64")
    assert descriptor.dependencies == (artifact.parakeet_vad_reference(),)
    assert {
        file.path: (file.size_bytes, file.sha256) for file in descriptor.files
    } == expected["files"]


def test_vad_dependency_is_exact_and_independently_addressable() -> None:
    """Long-form VAD is a verified dependency, not a worker-side download."""
    descriptor = artifact.parakeet_vad_descriptor()

    assert descriptor.reference == artifact.parakeet_vad_reference()
    assert descriptor.reference.revision == "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6"
    assert descriptor.role.value == "dependency"
    assert descriptor.upstream_repository == "istupakov/silero-vad-onnx"
    assert descriptor.upstream_revision == descriptor.reference.revision
    assert descriptor.license_id == "mit"
    assert [(file.path, file.size_bytes, file.sha256) for file in descriptor.files] == [
        (
            "silero_vad.onnx",
            2327524,
            "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        )
    ]
    assert descriptor.dependencies == ()


def test_generic_catalog_and_source_map_cover_each_root_closure() -> None:
    catalog = artifact.ParakeetCatalog()
    source_map = artifact.parakeet_source_map()
    vad_ref = artifact.parakeet_vad_reference()

    assert catalog.descriptor(vad_ref) == artifact.parakeet_vad_descriptor()
    assert set(source_map[vad_ref]) == {"silero_vad.onnx"}
    for model, precision in _EXPECTED_ROOTS:
        ref = artifact.parakeet_reference(model, precision)
        descriptor = catalog.descriptor(ref)
        assert descriptor.dependencies == (vad_ref,)
        assert set(source_map[ref]) == {file.path for file in descriptor.files}


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

    assert set(source_map) == {ref, artifact.parakeet_vad_reference()}
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
    vad_payloads = {"silero_vad.onnx": b"tiny-vad-bytes"}
    monkeypatch.setattr(installer, "PARAKEET_V2_FILES", _tiny_files(payloads))
    monkeypatch.setattr(artifact, "_VAD_FILES", _tiny_files(vad_payloads))

    with FixtureArtifactServer() as srv:
        monkeypatch.setattr(
            artifact,
            "_source_url",
            lambda repository, revision, filename: srv.url(f"/{filename}"),
        )
        for filename, payload in {**payloads, **vad_payloads}.items():
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
        assert {entry.ref for entry in report.entries} == {
            artifact.parakeet_v2_reference(),
            artifact.parakeet_vad_reference(),
        }
        assert report.download_bytes == sum(
            len(payload) for payload in (*payloads.values(), *vad_payloads.values())
        )
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
    vad_dir = core.artifact_path(artifact.parakeet_vad_reference())
    assert (vad_dir / "silero_vad.onnx").read_bytes() == vad_payloads["silero_vad.onnx"]

    # The active managed resolver -- the same one the console/batch STT
    # paths use -- must now find exactly this directory.
    assert artifact.active_managed_parakeet_v2_dir(service=core) == installed_dir


@pytest.mark.asyncio
async def test_vad_only_preflight_and_provision_never_include_a_parakeet_root(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"tiny-vad-only"
    monkeypatch.setattr(
        artifact,
        "_VAD_FILES",
        _tiny_files({"silero_vad.onnx": payload}),
    )

    with FixtureArtifactServer() as srv:
        monkeypatch.setattr(
            artifact,
            "_source_url",
            lambda repository, revision, filename: srv.url(f"/{filename}"),
        )
        srv.serve("/silero_vad.onnx", payload)
        core = ModelArtifactService(tmp_path / "managed")
        report = await artifact.run_parakeet_vad_preflight(
            core=core,
            free_bytes_probe=lambda _path: 10**12,
            trusted_origins=_trusted(srv),
        )

        assert report.root == artifact.parakeet_vad_reference()
        assert [entry.ref for entry in report.entries] == [
            artifact.parakeet_vad_reference()
        ]
        installed = await artifact.run_parakeet_vad_provision(
            report,
            core=core,
            free_bytes_probe=lambda _path: 10**12,
            trusted_origins=_trusted(srv),
        )

    assert installed == core.artifact_path(artifact.parakeet_vad_reference())
    assert (installed / "silero_vad.onnx").read_bytes() == payload
    assert set(srv.requests) == {"/silero_vad.onnx"}


def test_vad_catalog_rejects_every_parakeet_root_reference() -> None:
    catalog = artifact.ParakeetVadCatalog()

    assert catalog.descriptor(
        artifact.parakeet_vad_reference()
    ) == artifact.parakeet_vad_descriptor()
    with pytest.raises(KeyError):
        catalog.descriptor(artifact.parakeet_v2_reference())
