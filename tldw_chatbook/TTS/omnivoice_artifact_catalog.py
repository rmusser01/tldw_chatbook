"""Curated managed-artifact catalog for ct03/omnivoice-onnx-int8hq.

LM weights: Apache-2.0 (k2-fsa/OmniVoice model card; the card adds a
research-use / anti-impersonation disclaimer). Audio tokenizer (encoder +
decoder): Boson Higgs Audio 2 Community License (Llama-3-based). The tokenizer
is required for every synthesis, so its terms bind the whole bundle. Both are
surfaced at consent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
)

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionProgress,
        ArtifactAcquisitionService,
        ArtifactSourceMap,
        CredentialResolver,
        PreflightReport,
    )
    from tldw_chatbook.Model_Artifacts.service import ModelArtifactService


OMNIVOICE_ONNX_REPOSITORY = "ct03/omnivoice-onnx-int8hq"
OMNIVOICE_ONNX_REVISION = "65c840ba966f4b50cd6bd73f234fb1eba72f9a16"

OMNIVOICE_ONNX_ARTIFACT_ID = "omnivoice-onnx-int8hq"
OMNIVOICE_ONNX_VARIANT = "int8hq"

OMNIVOICE_ONNX_REQUIRED_PATHS: tuple[str, ...] = (
    "omnivoice_lm_int8_hq/model.onnx",
    "omnivoice_lm_int8_hq/model.onnx_data",
    "audio_tokenizer_decoder_int8/model.onnx",
    "audio_tokenizer_decoder_int8/model.onnx_data",
    "audio_tokenizer_encoder_int8/model.onnx",
    "audio_tokenizer_encoder_int8/model.onnx_data",
    "tokenizer.json",
    "config.json",
)

# (path, size_bytes, sha256) — verbatim from the HuggingFace tree API
# listing at OMNIVOICE_ONNX_REVISION (fetched 2026-09-23). The six
# .onnx/.onnx_data payloads and tokenizer.json are LFS-tracked, so each
# carries the repository-published LFS sha256. config.json is a plain git
# blob — HuggingFace publishes only a git SHA1 blob oid for it, no SHA256 —
# so its pinned digest was computed locally against the pinned revision,
# exactly like parakeet's plain-blob config.json/vocab.txt handling.
# ``ArtifactFile`` requires a 64-hex sha256 for every declared file, so the
# locally computed digest is recorded rather than a null marker; the weaker
# LOCAL_INTEGRITY_RECORDED provenance below is what carries that fact.
OMNIVOICE_ONNX_FILES: tuple[tuple[str, int, str], ...] = (
    (
        "omnivoice_lm_int8_hq/model.onnx",
        1_344_391,
        "4a79b508bf2ca264e0f5e4207a515dcf089a54281fa97689353eb93af76a944e",
    ),
    (
        "omnivoice_lm_int8_hq/model.onnx_data",
        640_295_438,
        "6d2d2f02895ffe4f0c38e03eed34b0a47f92a3a43165fd4b9c98b0a5fafb50de",
    ),
    (
        "audio_tokenizer_decoder_int8/model.onnx",
        231_012,
        "58833acb4e01b9d7401c45810d4889393eb35fc18a27820a394fcd9e4e947d30",
    ),
    (
        "audio_tokenizer_decoder_int8/model.onnx_data",
        85_458_944,
        "28d6b043b35250e415958f4946dc896f5bd41c69bde043b5113d0634833b027a",
    ),
    (
        "audio_tokenizer_encoder_int8/model.onnx",
        512_598,
        "7876269f66f39548a7ef043772f90e935b43f048e1bc17cbe371cb31d2e1803c",
    ),
    (
        "audio_tokenizer_encoder_int8/model.onnx_data",
        395_225_855,
        "7dabb5617c79ef84352f7c477616884310a33f429036baf32d848f83c1e8b716",
    ),
    (
        "tokenizer.json",
        11_423_986,
        "408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1",
    ),
    (
        "config.json",
        2_238,
        "5e359117e13b420c5e0c925d4aba650d624767131f1d1746928f8b850d5dc372",
    ),
)


def omnivoice_onnx_reference() -> ArtifactRef:
    """Return the exact immutable managed reference for the OmniVoice bundle.

    Returns:
        The ``ArtifactRef`` (artifact id, pinned upstream revision, variant)
        identifying the curated OmniVoice ONNX int8hq root artifact.
    """

    return ArtifactRef(
        OMNIVOICE_ONNX_ARTIFACT_ID,
        OMNIVOICE_ONNX_REVISION,
        OMNIVOICE_ONNX_VARIANT,
    )


def omnivoice_onnx_files() -> tuple[ArtifactFile, ...]:
    """Build the pinned per-file closure from the recorded tree facts.

    Returns:
        One ``ArtifactFile`` per entry in ``OMNIVOICE_ONNX_FILES``, in
        declaration order.
    """

    return tuple(
        ArtifactFile(path=path, size_bytes=size, sha256=digest)
        for path, size, digest in OMNIVOICE_ONNX_FILES
    )


def _source_url(path: str) -> str:
    return (
        f"https://huggingface.co/{OMNIVOICE_ONNX_REPOSITORY}/resolve/"
        f"{OMNIVOICE_ONNX_REVISION}/{path}"
    )


def omnivoice_onnx_descriptor() -> ArtifactDescriptor:
    """Build the exact managed OmniVoice ONNX root descriptor.

    Every size and digest comes verbatim from the HuggingFace tree API at
    the pinned revision (see ``OMNIVOICE_ONNX_FILES``). Provenance is
    ``(CHATBOOK_CURATED, LOCAL_INTEGRITY_RECORDED)`` — not
    ``INTEGRITY_VERIFIED`` — because config.json is a plain git blob whose
    digest HuggingFace does not publish and was therefore computed locally;
    per ADR-025 a mixed artifact must claim the weaker label for the whole
    closure (the same reasoning parakeet's config.json/vocab.txt handling
    records).

    Returns:
        The validated, immutable descriptor for the curated OmniVoice ONNX
        int8hq bundle.
    """

    files = omnivoice_onnx_files()
    return ArtifactDescriptor(
        reference=omnivoice_onnx_reference(),
        model_id=OMNIVOICE_ONNX_ARTIFACT_ID,
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="tts",
        model_family="omnivoice",
        upstream_repository=OMNIVOICE_ONNX_REPOSITORY,
        upstream_revision=OMNIVOICE_ONNX_REVISION,
        # Single-file fallback source only; the multi-file source map below
        # is always consulted for this multi-file artifact. Pointed at one
        # real, individually-verifiable declared file.
        source_url=_source_url(OMNIVOICE_ONNX_FILES[0][0]),
        precision=OMNIVOICE_ONNX_VARIANT,
        expected_installed_bytes=sum(file.size_bytes for file in files),
        license_id="other",
        license_url=f"https://huggingface.co/{OMNIVOICE_ONNX_REPOSITORY}",
        usage_notice=(
            "Two licenses: the LM weights are Apache-2.0 (k2-fsa/OmniVoice; its "
            "model card forbids unauthorized voice cloning, impersonation and "
            "fraud). The audio tokenizer, required for all synthesis, is under "
            "the Boson Higgs Audio 2 Community License (Llama 3-based): "
            "commercial use above 100,000 annual active users needs a license "
            "from Boson AI, use must follow the Llama 3 Acceptable Use Policy, "
            "and products built with it must credit \"Built with Higgs "
            "Materials\". By installing you accept both."
        ),
        runtime_name="onnx-tts",
        # onnxruntime is intentionally unpinned in the omnivoice_tts extra;
        # "none" is the house convention for an unpinned runtime (see
        # remote_huggingface descriptors). An empty string is rejected by
        # the descriptor contract.
        runtime_version_constraint="none",
        # Per-platform evidence is not yet reviewed for this backend; the
        # audio.cpp TTS entries use the same unassigned placeholder until
        # the backend tasks record real platform facts.
        supported_os=("unassigned",),
        supported_architectures=("unassigned",),
        provenance=(
            ProvenanceClass.CHATBOOK_CURATED,
            ProvenanceClass.LOCAL_INTEGRITY_RECORDED,
        ),
        files=files,
    )


def omnivoice_onnx_source_map() -> ArtifactSourceMap:
    """Return credential-free per-file download URLs for the bundle.

    Returns:
        A single-entry ``{omnivoice_onnx_reference(): {path: url}}`` map
        covering every file the descriptor declares.
    """

    return {
        omnivoice_onnx_reference(): {
            file.path: _source_url(file.path) for file in omnivoice_onnx_files()
        }
    }


OmniVoiceSetupState = Literal["engine_missing", "model_missing", "path_invalid", "ready"]

# Per-request and saved OmniVoice seeds are non-negative 31-bit ints; the
# engine, request admission and the setup wizard all share this bound.
OMNIVOICE_SEED_LIMIT = 2**31


def missing_omnivoice_modules() -> list[str]:
    """Return the OmniVoice runtime modules that are not installed.

    Probed with ``find_spec`` (no import).

    Returns:
        The missing module names, in install order.
    """
    from importlib.util import find_spec

    missing: list[str] = []
    for module_name in ("onnxruntime", "tokenizers"):
        try:
            if find_spec(module_name) is None:
                missing.append(module_name)
        except (ImportError, ValueError):
            missing.append(module_name)
    return missing


def omnivoice_setup_state(
    model_root: str | None,
    *,
    missing_modules: Callable[[], list[str]] | None = None,
    managed_root: Callable[[], Path | None] | None = None,
) -> OmniVoiceSetupState:
    """Report what OmniVoice still needs before it can speak here.

    Resolves the model exactly as the engine does (``OMNIVOICE_MODEL_ROOT``
    from the environment, else the configured ``model_root``, else the
    active managed artifact). Does
    filesystem work — call it off the UI thread.

    Args:
        model_root: ``[OmniVoiceSettings] model_root`` (blank/None = managed).
        missing_modules: Dependency probe (defaults to ``missing_omnivoice_modules``).
        managed_root: Managed-artifact lookup (defaults to the engine's).

    Returns:
        ``engine_missing``; ``path_invalid`` when an explicit root (env or
        config) fails its checks — it keeps winning over a managed install,
        so a download cannot help; ``model_missing``; or ``ready``.
    """
    from tldw_chatbook.TTS.backends.omnivoice import (
        OmniVoiceModelError,
        OmniVoiceNotConfiguredError,
        managed_model_root,
        resolve_model_root,
    )

    if (missing_modules or missing_omnivoice_modules)():
        return "engine_missing"
    # Same resolution as TTS_Backends (os.getenv with the config default): a
    # set-but-empty environment variable overrides config and means "managed".
    configured = os.environ.get("OMNIVOICE_MODEL_ROOT", model_root or "")
    try:
        resolve_model_root(
            {"OMNIVOICE_MODEL_ROOT": configured},
            managed_root or managed_model_root,
        )
    except OmniVoiceModelError:
        return "path_invalid" if configured else "model_missing"
    except OmniVoiceNotConfiguredError:
        return "model_missing"
    return "ready"


class OmniVoiceCatalog:
    """Catalog exposing only the curated OmniVoice ONNX descriptor."""

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Return the OmniVoice descriptor for its exact reference.

        Args:
            ref: The artifact reference being looked up.

        Returns:
            The curated OmniVoice ONNX descriptor.

        Raises:
            KeyError: For any other reference.
        """
        if ref != omnivoice_onnx_reference():
            raise KeyError(ref)
        return omnivoice_onnx_descriptor()


def _acquisition(
    core: "ModelArtifactService | None",
    credential_resolver: "CredentialResolver | None",
    free_bytes_probe: "Callable[[Path], int] | None",
) -> "tuple[ModelArtifactService, ArtifactAcquisitionService]":
    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactAcquisitionService,
        EnvConfigCredentialResolver,
    )
    from tldw_chatbook.Model_Artifacts.store import managed_service

    service = core if core is not None else managed_service()
    resolver = (
        credential_resolver
        if credential_resolver is not None
        else EnvConfigCredentialResolver()
    )
    return service, ArtifactAcquisitionService(
        service, credential_resolver=resolver, free_bytes_probe=free_bytes_probe
    )


async def run_omnivoice_preflight(
    *,
    core: "ModelArtifactService | None" = None,
    credential_resolver: "CredentialResolver | None" = None,
    free_bytes_probe: "Callable[[Path], int] | None" = None,
) -> "PreflightReport":
    """Plan the OmniVoice model install (sizes, disk, consent data).

    Args:
        core: Artifact store service (defaults to the managed store).
        credential_resolver: Download credentials (defaults to env/config).
        free_bytes_probe: Free-disk probe override (tests).

    Returns:
        The preflight report the consent dialog renders.
    """
    _service, acquisition = _acquisition(core, credential_resolver, free_bytes_probe)
    return await acquisition.preflight(
        omnivoice_onnx_reference(),
        OmniVoiceCatalog(),
        sources=omnivoice_onnx_source_map(),
    )


async def run_omnivoice_provision(
    report: "PreflightReport",
    *,
    core: "ModelArtifactService | None" = None,
    credential_resolver: "CredentialResolver | None" = None,
    free_bytes_probe: "Callable[[Path], int] | None" = None,
    progress: "Callable[[AcquisitionProgress], None] | None" = None,
) -> Path:
    """Download, verify and activate the OmniVoice model after consent.

    Args:
        report: The consented preflight report from ``run_omnivoice_preflight``.
        core: Artifact store service (defaults to the managed store).
        credential_resolver: Download credentials (defaults to env/config).
        free_bytes_probe: Free-disk probe override (tests).
        progress: Callback receiving download/verify progress.

    Returns:
        The installed artifact directory.
    """
    service, acquisition = _acquisition(core, credential_resolver, free_bytes_probe)
    installed = await acquisition.provision(
        omnivoice_onnx_reference(),
        report.grant(),
        OmniVoiceCatalog(),
        sources=omnivoice_onnx_source_map(),
        progress=progress,
    )
    return service.artifact_path(installed)
