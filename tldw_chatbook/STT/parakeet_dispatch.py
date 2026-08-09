"""Download-free resolution of exact installed Parakeet worker identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    active_managed_parakeet_dir,
    parakeet_reference,
    parakeet_v2_managed_service,
    parakeet_vad_reference,
)
from tldw_chatbook.Local_Ingestion.parakeet_v2_installer import (
    PARAKEET_V2_FILES,
    VERIFICATION_RECEIPT,
    parakeet_v2_install_dir,
    verify_parakeet_v2_bundle,
)
from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root
from tldw_chatbook.Utils.path_validation import validate_path_simple

from .contracts import ExecutionDevice
from .executor import LocalSourceSnapshot, ModelIdentity, snapshot_local_source


@dataclass(frozen=True, slots=True)
class ParakeetDispatch:
    """Exact installed artifact identity and caller-local worker option updates."""

    identity: ModelIdentity
    local_source: LocalSourceSnapshot | None
    managed_store_root: Path | None
    managed_artifact_ref: tuple[str, str, str] | None
    option_updates: Mapping[str, Any]
    managed_dependency_refs: tuple[tuple[str, str, str], ...] = ()


def _configured_paths(model_root: Path, precision: str) -> tuple[Path, ...]:
    filenames = (
        (
            "config.json",
            "vocab.txt",
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
        )
        if precision == "int8"
        else (
            "config.json",
            "vocab.txt",
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
        )
    )
    return tuple(model_root / filename for filename in filenames)


def _dispatch(
    *,
    model_id: str,
    precision: str,
    local_source: LocalSourceSnapshot | None = None,
    root_revision: str | None = None,
    closure_fingerprint: str | None = None,
    managed_store_root: Path | None = None,
    managed_artifact_ref: tuple[str, str, str] | None = None,
    managed_dependency_refs: tuple[tuple[str, str, str], ...] = (),
    option_updates: Mapping[str, Any] | None = None,
) -> ParakeetDispatch:
    return ParakeetDispatch(
        identity=ModelIdentity(
            provider_id="parakeet-onnx",
            model_id=model_id,
            root_revision=root_revision,
            closure_fingerprint=closure_fingerprint,
            precision=precision,
            device=ExecutionDevice.CPU,
            local_snapshot_token=(
                local_source.token if local_source is not None else None
            ),
        ),
        local_source=local_source,
        managed_store_root=managed_store_root,
        managed_artifact_ref=managed_artifact_ref,
        option_updates=MappingProxyType(dict(option_updates or {})),
        managed_dependency_refs=managed_dependency_refs,
    )


def resolve_parakeet_dispatch(
    *,
    model_id: str,
    precision: str,
    model_dir: str | Path | None,
) -> ParakeetDispatch:
    """Resolve an installed configured, managed, or verified legacy artifact.

    Resolution is synchronous and local-only. It never invokes artifact
    acquisition, provisioning, or download code.
    """

    reference = parakeet_reference(model_id, precision)
    if model_dir is not None and str(model_dir).strip():
        model_root = validate_path_simple(model_dir, require_exists=True).absolute()
        local_source = snapshot_local_source(_configured_paths(model_root, precision))
        vad_reference = parakeet_vad_reference()
        return _dispatch(
            model_id=model_id,
            precision=precision,
            local_source=local_source,
            root_revision=reference.revision,
            managed_store_root=managed_model_artifact_root().absolute(),
            managed_dependency_refs=(
                (
                    vad_reference.artifact_id,
                    vad_reference.revision,
                    vad_reference.variant,
                ),
            ),
            option_updates={"transcription_model_dir": str(model_root)},
        )

    service = parakeet_v2_managed_service()
    if (
        active_managed_parakeet_dir(
            model_id,
            precision,
            service=service,
        )
        is not None
    ):
        leased = service.acquire(reference)
        try:
            handle = leased.handle
            if handle.root != reference:
                raise FileNotFoundError(
                    "No installed Parakeet artifact matches the requested identity."
                )
            root_revision = handle.root.revision
            closure_fingerprint = handle.closure_fingerprint
        finally:
            leased.close()
        return _dispatch(
            model_id=model_id,
            precision=precision,
            root_revision=root_revision,
            closure_fingerprint=closure_fingerprint,
            managed_store_root=managed_model_artifact_root().absolute(),
            managed_artifact_ref=(
                reference.artifact_id,
                reference.revision,
                reference.variant,
            ),
        )

    if model_id == PARAKEET_V2_MODEL and precision == "int8":
        legacy_root = parakeet_v2_install_dir().absolute()
        if verify_parakeet_v2_bundle(legacy_root):
            legacy_paths = (
                legacy_root / VERIFICATION_RECEIPT,
                *(legacy_root / item.filename for item in PARAKEET_V2_FILES),
            )
            local_source = snapshot_local_source(legacy_paths)
            return _dispatch(
                model_id=model_id,
                precision=precision,
                local_source=local_source,
                option_updates={
                    "transcription_model_dir": str(legacy_root),
                    "_verify_legacy_parakeet_v2": True,
                },
            )

    raise FileNotFoundError(
        "No installed Parakeet artifact matches the requested model and precision."
    )


__all__ = ["ParakeetDispatch", "resolve_parakeet_dispatch"]
