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
from .stt_batch_routing import PARAKEET_V2_MODEL as _PARAKEET_V2_MODEL_ID
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

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionProgress,
        ArtifactCatalog,
        ArtifactSourceMap,
        CredentialResolver,
        PreflightReport,
    )


#: Canonical identity components for the managed Parakeet v2 artifact.
PARAKEET_V2_ARTIFACT_ID = "parakeet-v2"
PARAKEET_V2_VARIANT = "int8"

_PARAKEET_V2_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
_PARAKEET_V2_USAGE_NOTICE = (
    "Curated INT8 ONNX export of istupakov's Parakeet TDT v2 conversion, "
    "pinned to one immutable upstream revision and verified against the "
    "sizes and SHA-256 digests recorded in "
    "Local_Ingestion.parakeet_v2_installer."
)


def parakeet_v2_reference() -> ArtifactRef:
    """Return the exact immutable reference for the managed Parakeet v2 artifact.

    Returns:
        The ``ArtifactRef`` identifying the curated Parakeet v2 INT8 bundle
        (artifact id, pinned revision, and variant).
    """

    return ArtifactRef(
        PARAKEET_V2_ARTIFACT_ID,
        _installer.PARAKEET_V2_REVISION,
        PARAKEET_V2_VARIANT,
    )


def _artifact_files() -> tuple[ArtifactFile, ...]:
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
        for bundle_file in _installer.PARAKEET_V2_FILES
    )


def parakeet_v2_descriptor() -> ArtifactDescriptor:
    """Build the exact, immutable descriptor for the curated Parakeet v2 bundle.

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

    files = _artifact_files()
    return ArtifactDescriptor(
        reference=parakeet_v2_reference(),
        model_id=_PARAKEET_V2_MODEL_ID,
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="stt",
        model_family="parakeet",
        upstream_repository=_installer.PARAKEET_V2_REPOSITORY,
        upstream_revision=_installer.PARAKEET_V2_REVISION,
        # Used only as the single-file fallback source (irrelevant here --
        # this descriptor always declares more than one file, so
        # ``parakeet_v2_source_map()`` below is always consulted instead)
        # and for display; pointed at one real, individually-verifiable
        # declared file rather than a bare repository URL.
        source_url=_installer._source_url(_installer.PARAKEET_V2_FILES[0].filename),
        precision=PARAKEET_V2_VARIANT,
        expected_installed_bytes=sum(file.size_bytes for file in files),
        license_id=_installer.PARAKEET_V2_LICENSE,
        license_url=_PARAKEET_V2_LICENSE_URL,
        usage_notice=_PARAKEET_V2_USAGE_NOTICE,
        runtime_name="onnx-asr",
        runtime_version_constraint=">=0.12.0",
        supported_os=("linux", "darwin", "windows"),
        supported_architectures=("x86-64", "arm64"),
        provenance=(ProvenanceClass.CHATBOOK_CURATED, ProvenanceClass.LOCAL_INTEGRITY_RECORDED),
        files=files,
        dependencies=(),
    )


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

    ref = parakeet_v2_reference()
    return {
        ref: {
            bundle_file.filename: _installer._source_url(bundle_file.filename)
            for bundle_file in _installer.PARAKEET_V2_FILES
        }
    }


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

        if ref != parakeet_v2_reference():
            raise KeyError(ref)
        return parakeet_v2_descriptor()


def managed_model_artifact_root() -> Path:
    """Return the shared managed-artifact store root.

    A sibling of the legacy installer's own ``models/stt/...`` destination
    (``parakeet_v2_installer.parakeet_v2_install_dir()``), both beneath the
    existing user-data directory -- so a fresh install and a legacy one
    never collide on disk. Not Parakeet-specific: every future managed
    artifact this application acquires shares this one
    ``ModelArtifactService`` root, distinguished internally by artifact id,
    revision, and variant.

    Returns:
        The absolute path to the shared managed-artifact store root.
    """

    from tldw_chatbook.Utils.paths import get_user_data_dir

    return get_user_data_dir() / "models" / "managed"


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

    expected_ref = parakeet_v2_reference()
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
