"""Transient, path-private GGUF source state for local LLM runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactFormat,
    ArtifactIntegrityError,
    ArtifactLeaseTimeoutError,
    ArtifactNotReadyError,
    ArtifactRef,
    ArtifactRole,
    ArtifactStateError,
    InstalledArtifact,
    LeasedArtifactHandle,
    ModelArtifactService,
    ProvenanceClass,
)
from tldw_chatbook.Model_Artifacts.gguf_admission import GGUFError
from tldw_chatbook.UI.Screens.model_browser_state import format_mib


_SOURCE_ERROR_CODES = frozenset(
    {"missing", "not_ready", "integrity", "payload", "busy", "state"}
)


class GGUFSourceError(RuntimeError):
    """Stable path-private managed GGUF preparation failure."""

    def __init__(self, code: str) -> None:
        if code not in _SOURCE_ERROR_CODES:
            raise ValueError("unsupported GGUF source error code")
        self.code = code
        super().__init__("Managed GGUF source is unavailable")


class GGUFSourceMode(StrEnum):
    """One mutually exclusive source of GGUF runtime bytes."""

    EMBEDDED = "embedded"
    MANAGED = "managed"
    EXTERNAL = "external"


@dataclass(frozen=True)
class GGUFSourceSelection:
    """Transient source selection with inactive values retained in memory."""

    mode: GGUFSourceMode
    managed_ref: ArtifactRef | None = None
    external_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Reject coercible values so source authority stays explicit."""
        if type(self.mode) is not GGUFSourceMode:
            raise TypeError("mode must be a GGUFSourceMode")
        if self.managed_ref is not None and type(self.managed_ref) is not ArtifactRef:
            raise TypeError("managed_ref must be an ArtifactRef or None")
        if self.external_path is not None and not isinstance(self.external_path, Path):
            raise TypeError("external_path must be a Path or None")

    @property
    def authority(self) -> str:
        """Return the stable user-facing authority label."""
        return {
            GGUFSourceMode.EMBEDDED: "Embedded",
            GGUFSourceMode.MANAGED: "Managed GGUF",
            GGUFSourceMode.EXTERNAL: "External GGUF",
        }[self.mode]

    def for_mode(self, mode: GGUFSourceMode) -> GGUFSourceSelection:
        """Switch authority without discarding either inactive selection."""
        return replace(self, mode=mode)

    def validate_for(self, provider: str) -> GGUFSourceSelection:
        """Return this selection when its mode is supported by ``provider``."""
        supported = {
            "llamacpp": {GGUFSourceMode.MANAGED, GGUFSourceMode.EXTERNAL},
            "llamafile": set(GGUFSourceMode),
        }
        try:
            modes = supported[provider]
        except KeyError:
            raise ValueError("unsupported GGUF source provider") from None
        if self.mode not in modes:
            raise ValueError(f"{provider} does not support {self.mode.value} mode")
        return self


@dataclass(frozen=True)
class ManagedGGUFChoice:
    """Path-free display state for one exact managed GGUF reference."""

    reference: ArtifactRef
    label: str

    def __post_init__(self) -> None:
        """Keep selector payloads exact and display-ready."""
        if type(self.reference) is not ArtifactRef:
            raise TypeError("reference must be an ArtifactRef")
        if type(self.label) is not str or not self.label:
            raise TypeError("label must be a non-empty string")


def initial_gguf_selection(
    provider: str,
    existing_model_path: str,
) -> GGUFSourceSelection:
    """Map legacy model-path state without filesystem or store access.

    Args:
        provider: Local GGUF runtime provider name.
        existing_model_path: Previously configured external model path, if any.

    Returns:
        A transient source selection preserving the legacy path in place.

    Raises:
        ValueError: If ``provider`` is not a supported GGUF runtime.
        TypeError: If ``existing_model_path`` is not a string.
    """
    if provider not in {"llamacpp", "llamafile"}:
        raise ValueError("unsupported GGUF source provider")
    if type(existing_model_path) is not str:
        raise TypeError("existing_model_path must be a string")

    external_path = Path(existing_model_path) if existing_model_path.strip() else None
    mode = (
        GGUFSourceMode.EMBEDDED
        if provider == "llamafile" and external_path is None
        else GGUFSourceMode.EXTERNAL
    )
    return GGUFSourceSelection(mode=mode, external_path=external_path)


def _managed_provenance_label(provenance: tuple[ProvenanceClass, ...]) -> str:
    if ProvenanceClass.LOCAL_INTEGRITY_RECORDED in provenance:
        return "Managed · local integrity recorded"
    if ProvenanceClass.INTEGRITY_VERIFIED in provenance:
        return "Managed · integrity verified"
    if ProvenanceClass.CHATBOOK_CURATED in provenance:
        return "Managed · Chatbook curated"
    return "Managed"


def managed_gguf_choices(
    installed: Iterable[InstalledArtifact],
) -> tuple[ManagedGGUFChoice, ...]:
    """Return path-free choices for healthy, ready root GGUF artifacts."""
    choices: list[ManagedGGUFChoice] = []
    for item in installed:
        descriptor = item.descriptor
        if not (
            item.ready
            and item.error is None
            and descriptor is not None
            and descriptor.role is ArtifactRole.ROOT
            and descriptor.format is ArtifactFormat.GGUF
        ):
            continue
        choices.append(
            ManagedGGUFChoice(
                reference=descriptor.reference,
                label=" · ".join(
                    (
                        descriptor.model_id,
                        descriptor.precision,
                        format_mib(descriptor.expected_installed_bytes),
                        _managed_provenance_label(descriptor.provenance),
                    )
                ),
            )
        )
    return tuple(choices)


def _managed_inventory_result(
    installed: Iterable[InstalledArtifact],
    reference: ArtifactRef,
) -> tuple[str | None, ArtifactDescriptor | None]:
    matching = tuple(
        item
        for item in installed
        if item.descriptor is not None and item.descriptor.reference == reference
    )
    if not matching:
        return "missing", None
    if any(item.error is not None for item in matching):
        return "state", None
    if not any(item.ready for item in matching):
        return "not_ready", None
    exact = tuple(
        item.descriptor
        for item in matching
        if item.ready
        and item.descriptor is not None
        and item.descriptor.role is ArtifactRole.ROOT
        and item.descriptor.format is ArtifactFormat.GGUF
    )
    if len(exact) != 1:
        return "payload", None
    return None, exact[0]


def acquire_managed_gguf(
    service: ModelArtifactService,
    reference: ArtifactRef,
) -> tuple[Path, LeasedArtifactHandle]:
    """Acquire an exact managed root and resolve its sole declared GGUF payload.

    Args:
        service: Artifact service that owns the managed model.
        reference: Exact root artifact reference to acquire.

    Returns:
        The declared GGUF path and its open artifact lease.

    Raises:
        TypeError: If ``reference`` is not an exact ``ArtifactRef``.
        GGUFSourceError: If the artifact cannot be safely acquired or resolved.
    """
    if type(reference) is not ArtifactRef:
        raise TypeError("reference must be an ArtifactRef")

    leased: LeasedArtifactHandle | None = None
    failure_code: str | None = None
    try:
        leased = service.acquire(reference)
    except ArtifactLeaseTimeoutError:
        failure_code = "busy"
    except ArtifactNotReadyError:
        failure_code = "not_ready"
    except ArtifactIntegrityError:
        failure_code = "integrity"
    except ArtifactStateError:
        failure_code = "state"
    except Exception:
        failure_code = "state"
    if leased is None:
        raise GGUFSourceError(failure_code or "state")

    descriptor: ArtifactDescriptor | None = None
    try:
        failure_code, descriptor = _managed_inventory_result(
            service.list_installed(), reference
        )
        if failure_code is None:
            assert descriptor is not None
            gguf_files = tuple(
                item
                for item in descriptor.files
                if Path(item.path).suffix.casefold() == ".gguf"
            )
            if len(gguf_files) != 1 or leased.handle.root != reference:
                failure_code = "payload"
            else:
                root_path = dict(leased.handle.paths).get(reference)
                if root_path is None:
                    failure_code = "payload"
                else:
                    return root_path / gguf_files[0].path, leased
    except ArtifactNotReadyError:
        failure_code = "not_ready"
    except ArtifactIntegrityError:
        failure_code = "integrity"
    except ArtifactStateError:
        failure_code = "state"
    except Exception:
        failure_code = "state"
    except BaseException:
        try:
            leased.close()
        except BaseException:
            pass
        raise

    try:
        leased.close()
    except Exception:
        failure_code = "state"
    raise GGUFSourceError(failure_code or "state")


def gguf_source_failure_message(error: BaseException) -> str:
    """Map source failures to stable user copy without rendering exception text."""
    if isinstance(error, GGUFSourceError):
        return {
            "missing": (
                "The selected managed GGUF is unavailable. Choose another model or "
                "import it again."
            ),
            "not_ready": (
                "The selected managed GGUF is not ready. Choose another model or "
                "import it again."
            ),
            "integrity": (
                "The selected managed GGUF is corrupt. Delete it and import it again."
            ),
            "payload": (
                "The selected managed GGUF has an invalid payload. Delete it and "
                "import it again."
            ),
            "busy": "The managed model store is busy. Try again.",
            "state": "The managed model store is unavailable. Try again.",
        }[error.code]
    if isinstance(error, ArtifactLeaseTimeoutError):
        return "The managed model store is busy. Try again."
    if isinstance(error, ArtifactNotReadyError):
        return (
            "The selected managed GGUF is not ready. Choose another model or import "
            "it again."
        )
    if isinstance(error, ArtifactIntegrityError):
        return "The selected managed GGUF is corrupt. Delete it and import it again."
    if isinstance(error, ArtifactStateError):
        return "The managed model store is unavailable. Try again."
    if isinstance(error, GGUFError):
        return "The selected file is not a valid GGUF. Choose another file."
    return "The GGUF source could not be prepared. Try again or choose another source."
