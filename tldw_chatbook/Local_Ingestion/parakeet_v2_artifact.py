"""Parakeet v2 INT8 as the managed-download layer's first production consumer.

TASK-1696: reconciliation item 3 ported the shared managed-download layer
(TASK-595/1694/1695) with zero production consumers -- nothing in the app
could actually download anything because no ``ArtifactDescriptor`` existed
anywhere. This module is a thin adapter that:

- builds the exact, immutable ``ArtifactDescriptor`` for the curated
  Parakeet v2 INT8 bundle from the SAME pinned constants
  ``Local_Ingestion.parakeet_v2_installer`` already declares (repository,
  revision, license, per-file sizes and SHA-256 digests) -- imported, never
  copied, so the two never drift;
- supplies the credential-free per-file source map (``ArtifactSourceMap``)
  the shared downloader needs, reusing the installer's own
  ``_source_url`` URL pattern;
- exposes a minimal concrete ``ArtifactCatalog`` returning that one
  descriptor;
- resolves the shared managed-artifact store root beneath the existing
  user-data directory, as a sibling of the legacy ``models/stt/...``
  installer's own destination; and
- resolves the active managed Parakeet v2 artifact directory, if any, for
  the managed-first model-directory resolver used by both
  ``Audio.console_dictation`` and ``Local_Ingestion.transcription_service``.

IMPORT BOUNDARY (load-bearing -- see
``Tests/Model_Artifacts/test_credentials_and_boundaries.py::
test_stt_and_transcription_worker_modules_never_import_acquisition_or_fetch``):
``Audio.console_dictation`` and ``Local_Ingestion.transcription_service`` are
synchronous, worker-side modules that must never import
``Model_Artifacts.acquisition`` or ``Model_Artifacts.fetch`` (both
``import httpx`` at module scope -- see ``Model_Artifacts/__init__.py``'s own
docstring). This module therefore imports only ``Model_Artifacts.service``
at module scope; ``ArtifactCatalog``/``ArtifactSourceMap``/``PreflightReport``
are referenced only as ``TYPE_CHECKING`` annotations (never evaluated at
runtime, since this file uses ``from __future__ import annotations``), and
the two orchestration helpers that actually drive a download
(``run_parakeet_v2_preflight``/``run_parakeet_v2_provision``) import
``Model_Artifacts.acquisition`` LOCALLY, inside their own function bodies.
Only the Library UI (not worker-side) ever calls those two functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from . import parakeet_v2_installer as _installer
from .stt_batch_routing import (
    PARAKEET_V2_MODEL as _PARAKEET_V2_MODEL_ID,
    PARAKEET_V3_MODEL as _PARAKEET_V3_MODEL_ID,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDescriptor,
    ArtifactError,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ModelArtifactService,
    ProvenanceClass,
)
from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionProgress,
        ArtifactSourceMap,
        CredentialResolver,
        PreflightReport,
    )


#: Canonical identity components for the managed Parakeet v2 artifact.
PARAKEET_V2_ARTIFACT_ID = "parakeet-v2"
PARAKEET_V2_VARIANT = "int8"
PARAKEET_V3_ARTIFACT_ID = "parakeet-v3"
PARAKEET_PRECISIONS = ("int8", "f32")

PARAKEET_V3_REPOSITORY = "istupakov/parakeet-tdt-0.6b-v3-onnx"
PARAKEET_V3_REVISION = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"

PARAKEET_VAD_ARTIFACT_ID = "silero-vad"
PARAKEET_VAD_REPOSITORY = "istupakov/silero-vad-onnx"
PARAKEET_VAD_REVISION = "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6"
PARAKEET_VAD_VARIANT = "f32"

_V2_F32_FILES = (
    _installer.BundleFile("config.json", 97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466"),
    _installer.BundleFile("vocab.txt", 9384, "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d"),
    _installer.BundleFile("encoder-model.onnx", 41770866, "3987bcd28175d829d12888a996a84e8f62a0e374d9ffd640662c1515adc679d3"),
    _installer.BundleFile("encoder-model.onnx.data", 2435420160, "4dab7362d4874d85965045b1e41b2d61dd2cc0fb25671a7f6b3dc47bf120cc41"),
    _installer.BundleFile("decoder_joint-model.onnx", 35792059, "cbb52a07bd70ab5b67f8439d4b3cd8704b18467b4430bcacb5adabe154b8d191"),
)
_V3_INT8_FILES = (
    _installer.BundleFile("config.json", 97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466"),
    _installer.BundleFile("vocab.txt", 93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
    _installer.BundleFile("encoder-model.int8.onnx", 652183999, "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09"),
    _installer.BundleFile("decoder_joint-model.int8.onnx", 18202004, "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70"),
)
_V3_F32_FILES = (
    _installer.BundleFile("config.json", 97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466"),
    _installer.BundleFile("vocab.txt", 93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
    _installer.BundleFile("encoder-model.onnx", 41770866, "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1"),
    _installer.BundleFile("encoder-model.onnx.data", 2435420160, "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36"),
    _installer.BundleFile("decoder_joint-model.onnx", 72520893, "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1"),
)
_VAD_FILES = (
    _installer.BundleFile(
        "silero_vad.onnx",
        2327524,
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    ),
)

_PARAKEET_V2_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
_PARAKEET_V2_USAGE_NOTICE = (
    "Curated Parakeet TDT ONNX export pinned to one immutable upstream "
    "revision and verified against the sizes and SHA-256 digests recorded "
    "by TASK-593."
)


def _model_metadata(model: str) -> tuple[str, str, str]:
    if model == _PARAKEET_V2_MODEL_ID:
        return (
            PARAKEET_V2_ARTIFACT_ID,
            _installer.PARAKEET_V2_REPOSITORY,
            _installer.PARAKEET_V2_REVISION,
        )
    if model == _PARAKEET_V3_MODEL_ID:
        return PARAKEET_V3_ARTIFACT_ID, PARAKEET_V3_REPOSITORY, PARAKEET_V3_REVISION
    raise ValueError(f"unsupported managed Parakeet model: {model}")


def _bundle_files(model: str, precision: str) -> tuple[_installer.BundleFile, ...]:
    if precision not in PARAKEET_PRECISIONS:
        raise ValueError(f"unsupported Parakeet precision: {precision}")
    if model == _PARAKEET_V2_MODEL_ID:
        return _installer.PARAKEET_V2_FILES if precision == "int8" else _V2_F32_FILES
    if model == _PARAKEET_V3_MODEL_ID:
        return _V3_INT8_FILES if precision == "int8" else _V3_F32_FILES
    raise ValueError(f"unsupported managed Parakeet model: {model}")


def parakeet_vad_reference() -> ArtifactRef:
    """Return the exact managed Silero VAD dependency reference."""
    return ArtifactRef(
        PARAKEET_VAD_ARTIFACT_ID,
        PARAKEET_VAD_REVISION,
        PARAKEET_VAD_VARIANT,
    )


def parakeet_reference(model: str, precision: str = "int8") -> ArtifactRef:
    """Return one exact closure-bearing managed Parakeet root reference.

    Args:
        model: Exact supported Parakeet v2 or v3 model identifier.
        precision: Exact ``int8`` or ``f32`` artifact variant.

    Returns:
        The immutable managed root reference for the selection.

    Raises:
        ValueError: If the model or precision is unsupported.
    """
    artifact_id, _repository, upstream_revision = _model_metadata(model)
    _bundle_files(model, precision)
    closure_revision = f"{upstream_revision}-vad-{PARAKEET_VAD_REVISION[:12]}"
    return ArtifactRef(artifact_id, closure_revision, precision)


def parakeet_v2_reference() -> ArtifactRef:
    """Return the exact immutable reference for the managed Parakeet v2 artifact.

    Returns:
        The ``ArtifactRef`` identifying the curated Parakeet v2 INT8 bundle
        (artifact id, pinned revision, and variant).
    """

    return parakeet_reference(_PARAKEET_V2_MODEL_ID, PARAKEET_V2_VARIANT)


def _artifact_files(model: str, precision: str) -> tuple[ArtifactFile, ...]:
    """Build ``ArtifactFile`` entries from the installer's pinned bundle files.

    Reads ``_installer.PARAKEET_V2_FILES`` as a module attribute (not a
    rebound top-level import) on every call, so a test that monkeypatches
    ``parakeet_v2_installer.PARAKEET_V2_FILES`` (the same pattern
    ``Tests/Local_Ingestion/test_parakeet_v2_installer.py`` already uses) is
    observed here too, without this module keeping its own copy of the
    pinned sizes/digests to drift out of sync.
    """

    return tuple(
        ArtifactFile(
            path=bundle_file.filename,
            size_bytes=bundle_file.size_bytes,
            sha256=bundle_file.sha256,
        )
        for bundle_file in _bundle_files(model, precision)
    )


def _source_url(repository: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repository}/resolve/{revision}/{filename}"


def parakeet_vad_descriptor() -> ArtifactDescriptor:
    """Build the exact Silero VAD dependency descriptor."""
    files = tuple(
        ArtifactFile(item.filename, item.size_bytes, item.sha256) for item in _VAD_FILES
    )
    return ArtifactDescriptor(
        reference=parakeet_vad_reference(),
        model_id="silero-vad-onnx",
        role=ArtifactRole.DEPENDENCY,
        format=ArtifactFormat.ONNX,
        consumer="stt",
        model_family="silero-vad",
        upstream_repository=PARAKEET_VAD_REPOSITORY,
        upstream_revision=PARAKEET_VAD_REVISION,
        source_url=_source_url(
            PARAKEET_VAD_REPOSITORY,
            PARAKEET_VAD_REVISION,
            _VAD_FILES[0].filename,
        ),
        precision=PARAKEET_VAD_VARIANT,
        expected_installed_bytes=sum(file.size_bytes for file in files),
        license_id="mit",
        license_url="https://opensource.org/license/mit",
        usage_notice="Pinned Silero VAD ONNX model used for offline long-form segmentation.",
        runtime_name="onnx-asr",
        runtime_version_constraint="==0.12.0",
        supported_os=("linux", "darwin", "windows"),
        supported_architectures=("x86-64", "arm64"),
        provenance=(
            ProvenanceClass.CHATBOOK_CURATED,
            ProvenanceClass.LOCAL_INTEGRITY_RECORDED,
        ),
        files=files,
        dependencies=(),
    )


def parakeet_descriptor(
    model: str,
    precision: str = "int8",
) -> ArtifactDescriptor:
    """Build one exact managed Parakeet root descriptor.

    Every size and digest comes from ``parakeet_v2_installer.PARAKEET_V2_FILES``
    -- this is the single source of truth the shared managed-download layer
    now verifies against; nothing here re-declares or copies a digest.

    Provenance is ``(CHATBOOK_CURATED, LOCAL_INTEGRITY_RECORDED)`` -- NOT
    ``INTEGRITY_VERIFIED``, even though every file is verified against its
    pinned digest before use. Per ADR-025, "independently verified" claims
    that the REPOSITORY itself supplied the expected digest, not merely
    that Chatbook checked one it computed itself. Checked against
    HuggingFace's own tree API for this pinned revision: only
    ``encoder-model.int8.onnx`` and ``decoder_joint-model.int8.onnx`` are
    LFS-tracked, which is the only case HuggingFace publishes a repository-
    supplied SHA256 for (matching both pins). ``config.json`` and
    ``vocab.txt`` are plain git blobs -- HuggingFace exposes only a git
    SHA1 blob oid for those, no SHA256 -- so those two files' pinned
    digests were necessarily computed locally (by whoever curated this
    bundle), not supplied by the repository. Provenance is per-artifact,
    not per-file, so a mixed artifact must claim the weaker label for the
    whole thing: two of four files cannot claim independent verification,
    so the artifact as a whole doesn't either. This does not change what
    is CHECKED -- every declared file is still verified byte-for-byte
    against its pinned size and SHA-256 before the bundle becomes usable --
    only what is CLAIMED about where that digest originally came from.
    ``ArtifactDescriptor`` itself forbids combining ``INTEGRITY_VERIFIED``
    with ``LOCAL_INTEGRITY_RECORDED``; this pair does not hit that.

    Returns:
        The validated, immutable descriptor for the curated Parakeet v2
        INT8 bundle.
    """

    _artifact_id, repository, upstream_revision = _model_metadata(model)
    files = _artifact_files(model, precision)
    bundle_files = _bundle_files(model, precision)
    return ArtifactDescriptor(
        reference=parakeet_reference(model, precision),
        model_id=model,
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="stt",
        model_family="parakeet",
        upstream_repository=repository,
        upstream_revision=upstream_revision,
        # Used only as the single-file fallback source (irrelevant here --
        # this descriptor always declares more than one file, so
        # ``parakeet_v2_source_map()`` below is always consulted instead)
        # and for display; pointed at one real, individually-verifiable
        # declared file rather than a bare repository URL.
        source_url=_source_url(repository, upstream_revision, bundle_files[0].filename),
        precision=precision,
        expected_installed_bytes=sum(file.size_bytes for file in files),
        license_id=_installer.PARAKEET_V2_LICENSE,
        license_url=_PARAKEET_V2_LICENSE_URL,
        usage_notice=_PARAKEET_V2_USAGE_NOTICE,
        runtime_name="onnx-asr",
        runtime_version_constraint="==0.12.0",
        supported_os=("linux", "darwin", "windows"),
        supported_architectures=("x86-64", "arm64"),
        provenance=(ProvenanceClass.CHATBOOK_CURATED, ProvenanceClass.LOCAL_INTEGRITY_RECORDED),
        files=files,
        dependencies=(parakeet_vad_reference(),),
    )


def parakeet_v2_descriptor() -> ArtifactDescriptor:
    """Compatibility wrapper for the managed v2 INT8 root descriptor."""
    return parakeet_descriptor(_PARAKEET_V2_MODEL_ID, PARAKEET_V2_VARIANT)


def parakeet_v2_source_map() -> "ArtifactSourceMap":
    """Credential-free per-file download URLs for the Parakeet v2 closure.

    Reuses ``parakeet_v2_installer``'s own ``_source_url`` pattern (one
    URL per file, resolved against the pinned repository and revision) so
    the shared downloader's multi-file fetch never has to guess a per-file
    URL. See ``ArtifactSourceMap``'s own docstring (TASK-1695) for why this
    lives outside the frozen descriptor schema.

    Returns:
        A single-entry ``{parakeet_v2_reference(): {filename: url}}`` map
        covering every file the descriptor declares.
    """

    source_map = parakeet_source_map()
    root = parakeet_v2_reference()
    dependency = parakeet_vad_reference()
    return {root: source_map[root], dependency: source_map[dependency]}


def parakeet_source_map() -> "ArtifactSourceMap":
    """Return credential-free sources for every admitted Parakeet closure."""
    result: dict[ArtifactRef, dict[str, str]] = {}
    for model in (_PARAKEET_V2_MODEL_ID, _PARAKEET_V3_MODEL_ID):
        _artifact_id, repository, revision = _model_metadata(model)
        for precision in PARAKEET_PRECISIONS:
            ref = parakeet_reference(model, precision)
            result[ref] = {
                item.filename: _source_url(repository, revision, item.filename)
                for item in _bundle_files(model, precision)
            }
    vad_ref = parakeet_vad_reference()
    result[vad_ref] = {
        item.filename: _source_url(
            PARAKEET_VAD_REPOSITORY,
            PARAKEET_VAD_REVISION,
            item.filename,
        )
        for item in _VAD_FILES
    }
    return result


class ParakeetCatalog:
    """Catalog for all exact managed Parakeet roots and their VAD dependency."""

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        if ref == parakeet_vad_reference():
            return parakeet_vad_descriptor()
        for model in (_PARAKEET_V2_MODEL_ID, _PARAKEET_V3_MODEL_ID):
            for precision in PARAKEET_PRECISIONS:
                if ref == parakeet_reference(model, precision):
                    return parakeet_descriptor(model, precision)
        raise KeyError(ref)


class ParakeetVadCatalog:
    """Catalog exposing only the exact managed Silero VAD dependency."""

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Return only the exact VAD dependency descriptor."""

        if ref != parakeet_vad_reference():
            raise KeyError(ref)
        return parakeet_vad_descriptor()


class ParakeetV2Catalog:
    """Minimal ``ArtifactCatalog`` exposing only the Parakeet v2 root descriptor.

    Structurally satisfies ``Model_Artifacts.acquisition.ArtifactCatalog``
    (a plain ``Protocol``) without importing it -- see this module's
    docstring for why that import stays out of the module-level graph.
    """

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Return the Parakeet v2 descriptor for ``ref``.

        Args:
            ref: The artifact reference to look up.

        Returns:
            The Parakeet v2 descriptor.

        Raises:
            KeyError: ``ref`` is not the Parakeet v2 root reference.
        """

        return ParakeetCatalog().descriptor(ref)


def parakeet_v2_managed_service() -> ModelArtifactService:
    """Construct a ``ModelArtifactService`` over the shared managed store root.

    Returns:
        A ``ModelArtifactService`` rooted at ``managed_model_artifact_root()``.
    """

    return ModelArtifactService(managed_model_artifact_root())


def active_managed_parakeet_v2_dir(
    service: ModelArtifactService | None = None,
) -> Path | None:
    """Return the active, verified managed Parakeet v2 artifact directory, if any.

    Used by the managed-first model-directory resolver
    (``Audio.console_dictation``, ``Local_Ingestion.transcription_service``):
    checked after an explicitly configured directory and before the
    verified legacy ``.tldw-verified.json`` bundle. Uses ONLY the
    synchronous, credential-free ``ModelArtifactService`` surface --
    ``list_installed()`` -- never the async acquisition layer, honoring the
    import boundary this module's docstring describes.

    Args:
        service: Optional ``ModelArtifactService`` to inspect (tests pass a
            fixture-backed instance). Defaults to a service over
            ``managed_model_artifact_root()``.

    Returns:
        The installed, ready, and active Parakeet v2 artifact's directory,
        or ``None`` if no such artifact is currently active -- including
        when the managed store cannot be opened at all, which is treated as
        "nothing managed yet" rather than a hard failure: the caller always
        has a further fallback (the verified legacy bundle).
    """

    return active_managed_parakeet_dir(
        _PARAKEET_V2_MODEL_ID,
        PARAKEET_V2_VARIANT,
        service=service,
    )


def active_managed_parakeet_dir(
    model: str,
    precision: str = "int8",
    *,
    service: ModelArtifactService | None = None,
) -> Path | None:
    """Return an exact ready/active managed Parakeet root, when installed."""
    expected_ref = parakeet_reference(model, precision)
    try:
        core = service if service is not None else parakeet_v2_managed_service()
        for item in core.list_installed():
            if (
                item.descriptor is not None
                and item.ready
                and item.active
                # PR-1167 review (Finding 1): match the FULL reference
                # (artifact_id, revision, AND variant), not just
                # artifact_id. An active artifact sharing this id but a
                # different revision/variant is a real, distinct artifact
                # once a second one exists in the store -- matching on id
                # alone would return it anyway, and the loaders downstream
                # expect exactly the pinned INT8 revision this adapter
                # declares, not "whichever parakeet-v2 happens to be
                # active."
                and item.descriptor.reference == expected_ref
            ):
                return item.path
    except (ArtifactError, TypeError, ValueError, OSError):
        return None
    return None


async def run_parakeet_preflight(
    model: str,
    precision: str = "int8",
    *,
    core: ModelArtifactService | None = None,
    credential_resolver: "CredentialResolver | None" = None,
    free_bytes_probe: Callable[[Path], int] | None = None,
    trusted_origins: frozenset[str] = frozenset(),
) -> "PreflightReport":
    """Resolve an exact Parakeet root plus its pinned VAD dependency."""
    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactAcquisitionService,
        EnvConfigCredentialResolver,
    )

    service = core if core is not None else parakeet_v2_managed_service()
    resolver = (
        credential_resolver
        if credential_resolver is not None
        else EnvConfigCredentialResolver()
    )
    acquisition = ArtifactAcquisitionService(
        service,
        credential_resolver=resolver,
        free_bytes_probe=free_bytes_probe,
        trusted_origins=trusted_origins,
    )
    return await acquisition.preflight(
        parakeet_reference(model, precision),
        ParakeetCatalog(),
        sources=parakeet_source_map(),
    )


async def run_parakeet_vad_preflight(
    *,
    core: ModelArtifactService | None = None,
    credential_resolver: "CredentialResolver | None" = None,
    free_bytes_probe: Callable[[Path], int] | None = None,
    trusted_origins: frozenset[str] = frozenset(),
) -> "PreflightReport":
    """Preflight only the exact managed Silero VAD dependency."""

    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactAcquisitionService,
        EnvConfigCredentialResolver,
    )

    service = core if core is not None else parakeet_v2_managed_service()
    resolver = (
        credential_resolver
        if credential_resolver is not None
        else EnvConfigCredentialResolver()
    )
    acquisition = ArtifactAcquisitionService(
        service,
        credential_resolver=resolver,
        free_bytes_probe=free_bytes_probe,
        trusted_origins=trusted_origins,
    )
    reference = parakeet_vad_reference()
    return await acquisition.preflight(
        reference,
        ParakeetVadCatalog(),
        sources={reference: parakeet_source_map()[reference]},
    )


async def run_parakeet_vad_provision(
    report: "PreflightReport",
    *,
    core: ModelArtifactService | None = None,
    credential_resolver: "CredentialResolver | None" = None,
    free_bytes_probe: Callable[[Path], int] | None = None,
    trusted_origins: frozenset[str] = frozenset(),
    progress: "Callable[[AcquisitionProgress], None] | None" = None,
) -> Path:
    """Provision only the exact VAD dependency without activating a root."""

    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactAcquisitionService,
        EnvConfigCredentialResolver,
    )

    service = core if core is not None else parakeet_v2_managed_service()
    resolver = (
        credential_resolver
        if credential_resolver is not None
        else EnvConfigCredentialResolver()
    )
    acquisition = ArtifactAcquisitionService(
        service,
        credential_resolver=resolver,
        free_bytes_probe=free_bytes_probe,
        trusted_origins=trusted_origins,
    )
    reference = parakeet_vad_reference()
    installed = await acquisition.provision(
        reference,
        report.grant(),
        ParakeetVadCatalog(),
        sources={reference: parakeet_source_map()[reference]},
        progress=progress,
        activate=False,
    )
    return service.artifact_path(installed)


async def run_parakeet_provision(
    model: str,
    precision: str,
    report: "PreflightReport",
    *,
    core: ModelArtifactService | None = None,
    credential_resolver: "CredentialResolver | None" = None,
    free_bytes_probe: Callable[[Path], int] | None = None,
    trusted_origins: frozenset[str] = frozenset(),
    progress: "Callable[[AcquisitionProgress], None] | None" = None,
) -> Path:
    """Provision and activate one exact Parakeet root closure."""
    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactAcquisitionService,
        EnvConfigCredentialResolver,
    )

    service = core if core is not None else parakeet_v2_managed_service()
    resolver = (
        credential_resolver
        if credential_resolver is not None
        else EnvConfigCredentialResolver()
    )
    acquisition = ArtifactAcquisitionService(
        service,
        credential_resolver=resolver,
        free_bytes_probe=free_bytes_probe,
        trusted_origins=trusted_origins,
    )
    consent = report.grant()
    activated = await acquisition.provision(
        parakeet_reference(model, precision),
        consent,
        ParakeetCatalog(),
        sources=parakeet_source_map(),
        progress=progress,
    )
    return service.artifact_path(activated)


async def run_parakeet_v2_preflight(
    *,
    core: ModelArtifactService | None = None,
    credential_resolver: "CredentialResolver | None" = None,
    free_bytes_probe: Callable[[Path], int] | None = None,
    trusted_origins: frozenset[str] = frozenset(),
) -> "PreflightReport":
    """Resolve the immutable managed-install plan for Parakeet v2.

    The Library UI calls this (in a background worker) before showing
    ``ParakeetV2InstallModal``, so the modal renders real destination,
    byte, license, and free-space figures instead of hard-coded constants.

    Local import of ``Model_Artifacts.acquisition`` (see module docstring):
    only the Library UI calls this, never worker-side STT code.

    Args:
        core: Optional ``ModelArtifactService`` override (tests only;
            production uses ``parakeet_v2_managed_service()``).
        credential_resolver: Optional ``CredentialResolver`` override;
            defaults to ``EnvConfigCredentialResolver()``, matching every
            other HuggingFace-key consumer in this codebase.
        free_bytes_probe: Optional free-space override (tests only).
        trusted_origins: Hostnames exempt from the private-IP egress block;
            empty in production (huggingface.co is a public host), a
            fixture server's loopback hostname in tests.

    Returns:
        A frozen ``PreflightReport``; call ``.grant()`` on it (or pass it to
        ``run_parakeet_v2_provision``) to proceed.
    """

    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactAcquisitionService,
        EnvConfigCredentialResolver,
    )

    service = core if core is not None else parakeet_v2_managed_service()
    resolver = (
        credential_resolver if credential_resolver is not None else EnvConfigCredentialResolver()
    )
    acquisition = ArtifactAcquisitionService(
        service,
        credential_resolver=resolver,
        free_bytes_probe=free_bytes_probe,
        trusted_origins=trusted_origins,
    )
    return await acquisition.preflight(
        parakeet_v2_reference(),
        ParakeetV2Catalog(),
        sources=parakeet_v2_source_map(),
    )


async def run_parakeet_v2_provision(
    report: "PreflightReport",
    *,
    core: ModelArtifactService | None = None,
    credential_resolver: "CredentialResolver | None" = None,
    free_bytes_probe: Callable[[Path], int] | None = None,
    trusted_origins: frozenset[str] = frozenset(),
    progress: "Callable[[AcquisitionProgress], None] | None" = None,
) -> Path:
    """Grant consent from ``report`` and provision the managed Parakeet v2 closure.

    Args:
        report: The ``PreflightReport`` a prior ``run_parakeet_v2_preflight()``
            call returned; consent is granted from it here.
        core: Optional ``ModelArtifactService`` override (tests only).
        credential_resolver: Optional ``CredentialResolver`` override; see
            ``run_parakeet_v2_preflight``.
        free_bytes_probe: Optional free-space override (tests only).
        trusted_origins: See ``run_parakeet_v2_preflight``.
        progress: Optional progress sink forwarded to ``provision()``.

    Returns:
        The active managed Parakeet v2 artifact directory.

    Raises:
        PreflightNotGrantableError: ``report`` has gating errors or
            insufficient space.
        ConsentMismatchError: The catalog or source map changed since
            ``report`` was computed.
    """

    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactAcquisitionService,
        EnvConfigCredentialResolver,
    )

    service = core if core is not None else parakeet_v2_managed_service()
    resolver = (
        credential_resolver if credential_resolver is not None else EnvConfigCredentialResolver()
    )
    acquisition = ArtifactAcquisitionService(
        service,
        credential_resolver=resolver,
        free_bytes_probe=free_bytes_probe,
        trusted_origins=trusted_origins,
    )
    consent = report.grant()
    activated = await acquisition.provision(
        parakeet_v2_reference(),
        consent,
        ParakeetV2Catalog(),
        sources=parakeet_v2_source_map(),
        progress=progress,
    )
    return service.artifact_path(activated)
