"""TASK-595 Task 4: pure acquisition types and the catalog closure walk."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tldw_chatbook.Model_Artifacts import ArtifactRef, ProvenanceClass
from tldw_chatbook.Model_Artifacts.acquisition import (
    ACQUISITION_SAFETY_MARGIN_BYTES,
    MAX_FILE_REFETCHES,
    AcquisitionConsent,
    ArtifactAcquisitionService,
    CatalogError,
    PreflightNotGrantableError,
    PreflightReport,
    resolve_catalog_closure,
)
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts import ArtifactDescriptor


def make_descriptor(
    ref: ArtifactRef | None = None,
    dependencies: tuple[ArtifactRef, ...] = (),
    files_body: bytes | None = None,
    source_url: str | None = None,
    license_id: str = "test-license",
    license_url: str = "https://example.test/license",
    provenance: tuple[ProvenanceClass, ...] | None = None,
) -> ArtifactDescriptor:
    """Build a minimal descriptor for testing.

    Args:
        ref: The reference to build the descriptor for (defaults to a
            fixed "root" reference).
        dependencies: Dependency references for closure-walk tests.
        files_body: Payload bytes for the single declared file, used by
            preflight/fetch tests that need a real byte count and digest
            (defaults to the historical single-byte ``b"x"`` content).
        source_url: Override for ``source_url``, e.g. a fixture server URL
            (defaults to the historical placeholder URL).
        license_id: Licensing identifier for the descriptor.
        license_url: Credential-free HTTP(S) licensing URL, or an empty
            string for a truthful local-integrity descriptor with an unknown
            license.
        provenance: Descriptor provenance (defaults to Chatbook-curated).
    """
    from tldw_chatbook.Model_Artifacts import (
        ArtifactDescriptor,
        ArtifactFile,
        ArtifactFormat,
        ArtifactRole,
        ProvenanceClass,
    )

    if ref is None:
        ref = ArtifactRef("root", "r" * 40, "int8")

    content = files_body if files_body is not None else b"x"
    files = (
        ArtifactFile("model.onnx", len(content), hashlib.sha256(content).hexdigest()),
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
        source_url=source_url
        if source_url is not None
        else "https://example.test/model",
        precision="int8",
        license_id=license_id,
        license_url=license_url,
        usage_notice="Test model",
        runtime_name="test-runtime",
        runtime_version_constraint="==1.0.0",
        supported_os=("linux",),
        supported_architectures=("x86-64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,)
        if provenance is None
        else provenance,
        files=files,
        expected_installed_bytes=len(content),
        dependencies=dependencies,
    )


class DictCatalog:
    """Simple in-memory catalog for testing."""

    def __init__(self, mapping: dict[ArtifactRef, ArtifactDescriptor]) -> None:
        self._m = mapping

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Retrieve a descriptor by ref."""
        return self._m[ref]


def _ref(a: str = "root", r: str = "r1", v: str = "int8") -> ArtifactRef:
    """Build a test ArtifactRef."""
    return ArtifactRef(a, r, v)


def test_closure_walk_resolves_dependencies_in_stable_order() -> None:
    """Verify resolve_catalog_closure returns sorted by ref."""
    dep = _ref("aaa-dep")
    root = _ref("root")
    catalog = DictCatalog(
        {
            root: make_descriptor(ref=root, dependencies=(dep,)),
            dep: make_descriptor(ref=dep),
        }
    )
    closure = resolve_catalog_closure(root, catalog)
    ids = [d.reference.artifact_id for d in closure]
    assert ids == sorted(ids)
    assert len(closure) == 2


def test_closure_walk_detects_cycles() -> None:
    """Verify resolve_catalog_closure raises CatalogError for cycles."""
    a, b = _ref("a"), _ref("b")
    catalog = DictCatalog(
        {
            a: make_descriptor(ref=a, dependencies=(b,)),
            b: make_descriptor(ref=b, dependencies=(a,)),
        }
    )
    with pytest.raises(CatalogError):
        resolve_catalog_closure(a, catalog)


def test_closure_walk_detects_revision_conflicts() -> None:
    """Verify resolve_catalog_closure raises CatalogError for conflicting revisions."""
    dep1, dep2 = _ref("dep", "r1"), _ref("dep", "r2")
    a, b = _ref("a"), _ref("b")
    root = _ref("root")
    catalog = DictCatalog(
        {
            root: make_descriptor(ref=root, dependencies=(a, b)),
            a: make_descriptor(ref=a, dependencies=(dep1,)),
            b: make_descriptor(ref=b, dependencies=(dep2,)),
            dep1: make_descriptor(ref=dep1),
            dep2: make_descriptor(ref=dep2),
        }
    )
    with pytest.raises(CatalogError):
        resolve_catalog_closure(root, catalog)


def test_unknown_ref_is_a_typed_error() -> None:
    """Verify resolve_catalog_closure raises CatalogError for unknown refs."""
    with pytest.raises(CatalogError):
        resolve_catalog_closure(_ref("missing"), DictCatalog({}))


@pytest.mark.asyncio
async def test_uninstalled_local_integrity_descriptor_cannot_enter_download_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path-private local descriptor cannot reach network acquisition."""
    from tldw_chatbook.Model_Artifacts import ProvenanceClass
    from tldw_chatbook.Model_Artifacts import acquisition as acquisition_module

    local = make_descriptor(
        source_url="",
        license_id="unknown",
        license_url="",
        provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
    )
    probe_called = False
    fetch_called = False

    async def unexpected_probe(*_args: object, **_kwargs: object) -> list[str]:
        nonlocal probe_called
        probe_called = True
        return []

    async def unexpected_fetch(*_args: object, **_kwargs: object) -> object:
        nonlocal fetch_called
        fetch_called = True
        raise AssertionError("local artifact must not be fetched")

    monkeypatch.setattr(ArtifactAcquisitionService, "_probe_gating", unexpected_probe)
    monkeypatch.setattr(acquisition_module, "stream_fetch", unexpected_fetch)
    core = ModelArtifactService(tmp_path / "managed")
    acquisition = ArtifactAcquisitionService(
        core,
        free_bytes_probe=lambda _path: 10**12,
    )

    with pytest.raises(CatalogError, match="local integrity"):
        await acquisition.preflight(
            local.reference, DictCatalog({local.reference: local})
        )

    assert probe_called is False
    assert fetch_called is False


@pytest.mark.asyncio
async def test_uninstalled_local_integrity_descriptor_rejects_source_map_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source map cannot make a local descriptor acquisition-eligible."""
    from tldw_chatbook.Model_Artifacts import acquisition as acquisition_module

    local = make_descriptor(
        source_url="",
        license_id="unknown",
        license_url="",
        provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
    )
    probe_called = False
    fetch_called = False

    async def unexpected_probe(*_args: object, **_kwargs: object) -> list[str]:
        nonlocal probe_called
        probe_called = True
        return []

    async def unexpected_fetch(*_args: object, **_kwargs: object) -> object:
        nonlocal fetch_called
        fetch_called = True
        raise AssertionError("local artifact must not be fetched")

    monkeypatch.setattr(ArtifactAcquisitionService, "_probe_gating", unexpected_probe)
    monkeypatch.setattr(acquisition_module, "stream_fetch", unexpected_fetch)
    core = ModelArtifactService(tmp_path / "managed")
    acquisition = ArtifactAcquisitionService(
        core,
        free_bytes_probe=lambda _path: 10**12,
    )

    with pytest.raises(CatalogError, match="local integrity"):
        await acquisition.preflight(
            local.reference,
            DictCatalog({local.reference: local}),
            sources={local.reference: {"model.onnx": "https://example.test/model"}},
        )

    assert probe_called is False
    assert fetch_called is False


@pytest.mark.asyncio
async def test_installed_local_integrity_descriptor_remains_inventory_resolvable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed local descriptor has no acquisition source requirement."""
    local = make_descriptor(
        source_url="",
        license_id="unknown",
        license_url="",
        provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"x")
    core = ModelArtifactService(tmp_path / "managed")
    core.install(local, source)
    acquisition = ArtifactAcquisitionService(
        core,
        free_bytes_probe=lambda _path: 10**12,
    )

    probe_targets: list[object] = []

    async def record_probe(_service: object, targets: object) -> list[str]:
        probe_targets.extend(targets)  # type: ignore[arg-type]
        return []

    monkeypatch.setattr(ArtifactAcquisitionService, "_probe_gating", record_probe)

    report = await acquisition.preflight(
        local.reference,
        DictCatalog({local.reference: local}),
    )

    assert report.entries[0].already_installed is True
    assert report.download_bytes == 0
    assert probe_targets == []


def _report(**overrides: object) -> PreflightReport:
    """Build a PreflightReport with default values."""
    defaults: dict[str, object] = {
        "root": _ref(),
        "closure_fingerprint": "f" * 64,
        "entries": (),
        "download_bytes": 0,
        "already_staged_bytes": 0,
        "staging_overhead_bytes": 0,
        "retained_bytes": 0,
        "destination": Path("/tmp/x"),
        "free_bytes": 10**12,
        "required_bytes": 10**6,
        "sufficient_space": True,
        "gating_errors": (),
    }
    defaults.update(overrides)
    return PreflightReport(**defaults)  # type: ignore[arg-type]


def test_grant_returns_consent_with_fingerprint() -> None:
    """Verify PreflightReport.grant() returns AcquisitionConsent."""
    consent = _report().grant()
    assert isinstance(consent, AcquisitionConsent)
    assert consent.closure_fingerprint == "f" * 64


def test_grant_refuses_gating_errors_and_insufficient_space() -> None:
    """Verify PreflightReport.grant() raises PreflightNotGrantableError."""
    with pytest.raises(PreflightNotGrantableError):
        _report(gating_errors=("token required",)).grant()
    with pytest.raises(PreflightNotGrantableError):
        _report(sufficient_space=False).grant()


def test_constants_are_defined() -> None:
    """Verify module constants."""
    assert ACQUISITION_SAFETY_MARGIN_BYTES == 256 * 1024 * 1024
    assert MAX_FILE_REFETCHES == 1
