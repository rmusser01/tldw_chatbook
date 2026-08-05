"""Shared managed model-artifact contracts.

``.acquisition`` and ``.fetch`` (both ``import httpx`` at module scope) are
deliberately NOT imported eagerly here -- ``Tests/Model_Artifacts/test_service.py``'s
``test_package_import_does_not_load_inference_or_http_runtimes`` and
``test_credentials_and_boundaries.py``'s
``test_stt_and_transcription_worker_modules_never_import_acquisition_or_fetch``
(TASK-595 Task 9) pin the invariant that plain ``import tldw_chatbook.Model_Artifacts``
-- reached transitively by the synchronous STT/transcription worker surface,
which only needs ``ArtifactLeaseKey``/``ModelArtifactService`` -- never pulls
httpx or the async acquisition runtime into that process. Their names are
still in ``__all__`` and resolve correctly on ``from ... import Name`` via
the ``__getattr__`` below (PEP 562): the submodule import happens only the
first time one of those specific names is actually looked up.
"""

from __future__ import annotations

from .leases import (
    ArtifactLeaseCancelledError,
    ArtifactLeaseError,
    ArtifactLeaseKey,
    ArtifactLeaseTimeoutError,
    ArtifactOperationLease,
    ArtifactOperationLeaseSet,
    LeaseMode,
)
from .service import (
    ACQUISITION_SESSION_LEASE_KEY,
    ArtifactConflictError,
    ArtifactDependencyError,
    ArtifactDescriptor,
    ArtifactDescriptorError,
    ArtifactDescriptorParseError,
    ArtifactDescriptorValidationError,
    ArtifactDiskUsage,
    ArtifactError,
    ArtifactFile,
    ArtifactFormat,
    ArtifactHandle,
    ArtifactInUseError,
    ArtifactIntegrityError,
    ArtifactNotReadyError,
    ArtifactPathError,
    ArtifactRef,
    ArtifactRole,
    ArtifactStateError,
    InstalledArtifact,
    LeasedArtifactHandle,
    ModelArtifactService,
    ProvenanceClass,
    ReconcileReport,
    closure_fingerprint,
)

# Names resolved lazily from .acquisition / .fetch -- see module docstring.
_ACQUISITION_NAMES = frozenset(
    {
        "AcquisitionBusyError",
        "AcquisitionConsent",
        "AcquisitionError",
        "AcquisitionProgress",
        "ArtifactAcquisitionService",
        "ArtifactCatalog",
        "ArtifactPreflightEntry",
        "ArtifactSourceMap",
        "CatalogError",
        "ConsentMismatchError",
        "CredentialResolver",
        "EnvConfigCredentialResolver",
        "GatedRepositoryError",
        "InsufficientSpaceError",
        "PreflightNotGrantableError",
        "PreflightReport",
        "TransferError",
    }
)
_FETCH_NAMES = frozenset({"FetchResult", "FetchValidators", "stream_fetch"})


def __getattr__(name: str):
    """Resolve acquisition/fetch names on first access (PEP 562).

    Args:
        name: The attribute being looked up on this package.

    Returns:
        The resolved object from ``.acquisition`` or ``.fetch``.

    Raises:
        AttributeError: ``name`` is not one of this package's public names.
    """

    if name in _ACQUISITION_NAMES:
        from . import acquisition

        return getattr(acquisition, name)
    if name in _FETCH_NAMES:
        from . import fetch

        return getattr(fetch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACQUISITION_SESSION_LEASE_KEY",
    "AcquisitionBusyError",
    "AcquisitionConsent",
    "AcquisitionError",
    "AcquisitionProgress",
    "ArtifactAcquisitionService",
    "ArtifactCatalog",
    "ArtifactConflictError",
    "ArtifactDependencyError",
    "ArtifactDescriptor",
    "ArtifactDescriptorError",
    "ArtifactDescriptorParseError",
    "ArtifactDescriptorValidationError",
    "ArtifactDiskUsage",
    "ArtifactError",
    "ArtifactFile",
    "ArtifactFormat",
    "ArtifactHandle",
    "ArtifactInUseError",
    "ArtifactIntegrityError",
    "ArtifactLeaseCancelledError",
    "ArtifactLeaseError",
    "ArtifactLeaseKey",
    "ArtifactLeaseTimeoutError",
    "ArtifactNotReadyError",
    "ArtifactOperationLease",
    "ArtifactOperationLeaseSet",
    "ArtifactPathError",
    "ArtifactPreflightEntry",
    "ArtifactRef",
    "ArtifactRole",
    "ArtifactSourceMap",
    "ArtifactStateError",
    "CatalogError",
    "ConsentMismatchError",
    "CredentialResolver",
    "EnvConfigCredentialResolver",
    "FetchResult",
    "FetchValidators",
    "GatedRepositoryError",
    "InsufficientSpaceError",
    "InstalledArtifact",
    "LeasedArtifactHandle",
    "LeaseMode",
    "ModelArtifactService",
    "PreflightNotGrantableError",
    "PreflightReport",
    "ProvenanceClass",
    "ReconcileReport",
    "TransferError",
    "closure_fingerprint",
    "stream_fetch",
]
