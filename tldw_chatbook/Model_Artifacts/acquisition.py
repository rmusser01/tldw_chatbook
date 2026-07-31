"""TASK-595: managed model acquisition types and catalog resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Protocol

import httpx

from tldw_chatbook.Utils.egress import EgressBlockedError, check_url_or_raise_async

from .service import ArtifactError, ArtifactRef, closure_fingerprint

if TYPE_CHECKING:
    from .service import ArtifactDescriptor, ModelArtifactService


# Constants per spec (Docs/superpowers/specs/2026-07-30-managed-model-acquisition-design.md)
ACQUISITION_SAFETY_MARGIN_BYTES = 256 * 1024 * 1024
MAX_FILE_REFETCHES = 1

# Bounded timeout for the preflight repository-gating HEAD probe (Task 5).
_PREFLIGHT_PROBE_TIMEOUT_SECONDS = 10.0

# Credential hint named in gating_errors -- never a token value. Matches the
# existing env precedence this codebase already documents (config.py's
# HUGGINGFACE_API_KEY, Constants.py's HF_TOKEN); Task 9 wires an actual
# CredentialResolver against these same names.
_CREDENTIAL_ENV_HINT = "HUGGINGFACE_API_KEY (or HF_TOKEN)"


# Error hierarchy: all subclass ArtifactError
class AcquisitionError(ArtifactError):
    """Base error for acquisition operations."""

    pass


class CatalogError(AcquisitionError):
    """Catalog lookup, cycle, or revision-conflict error."""

    pass


class ConsentMismatchError(AcquisitionError):
    """Closure fingerprint changed between preflight and provision."""

    pass


class PreflightNotGrantableError(AcquisitionError):
    """Preflight report cannot be granted due to gating or space errors."""

    pass


class AcquisitionBusyError(AcquisitionError):
    """Another acquisition session is already active."""

    pass


class InsufficientSpaceError(AcquisitionError):
    """Insufficient free space for acquisition."""

    pass


class GatedRepositoryError(AcquisitionError):
    """Authenticated repository requires credentials or fails access."""

    pass


class TransferError(AcquisitionError):
    """Network or transfer error with optional retry flag."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        """Initialize TransferError with retryable flag.

        Args:
            message: The error message.
            retryable: Whether this error is retryable.
        """
        super().__init__(message)
        self.retryable = retryable


# Protocol for catalog descriptors
class ArtifactCatalog(Protocol):
    """Protocol for artifact descriptor catalogs."""

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Retrieve a descriptor by reference.

        Args:
            ref: The artifact reference.

        Returns:
            The artifact descriptor.

        Raises:
            KeyError: If the reference is not found.
        """
        ...


# Frozen dataclasses per spec
@dataclass(frozen=True)
class ArtifactPreflightEntry:
    """One artifact entry in a preflight report."""

    ref: ArtifactRef
    source_url: str
    repository: str
    revision: str
    license_id: str
    license_url: str
    precision: str
    total_bytes: int
    file_count: int
    already_installed: bool


@dataclass(frozen=True)
class AcquisitionConsent:
    """Consent to acquire a resolved closure."""

    closure_fingerprint: str


@dataclass(frozen=True)
class AcquisitionProgress:
    """Progress during an active acquisition operation."""

    phase: Literal["fetch", "pre-verify", "verify-install", "activate"]
    ref: ArtifactRef
    file: str | None
    bytes_done: int
    bytes_total: int


@dataclass(frozen=True)
class PreflightReport:
    """Preflight report for a closure before acquisition.

    Fields marked as space/bytes are in bytes. The report aggregates
    requirements and fails early on gating or space constraints before
    any actual downloads.
    """

    root: ArtifactRef
    closure_fingerprint: str
    entries: tuple[ArtifactPreflightEntry, ...]
    download_bytes: int
    already_staged_bytes: int
    staging_overhead_bytes: int
    retained_bytes: int
    destination: Path
    free_bytes: int
    required_bytes: int
    sufficient_space: bool
    gating_errors: tuple[str, ...]

    def grant(self) -> AcquisitionConsent:
        """Grant acquisition consent from this preflight report.

        Raises PreflightNotGrantableError if gating_errors is non-empty
        or sufficient_space is False.

        Returns:
            AcquisitionConsent carrying the closure fingerprint.

        Raises:
            PreflightNotGrantableError: If gating errors exist or space is insufficient.
        """
        if self.gating_errors or not self.sufficient_space:
            raise PreflightNotGrantableError(
                f"preflight not grantable: gating_errors={self.gating_errors}, "
                f"sufficient_space={self.sufficient_space}"
            )
        return AcquisitionConsent(closure_fingerprint=self.closure_fingerprint)


def resolve_catalog_closure(
    root: ArtifactRef, catalog: ArtifactCatalog
) -> tuple[ArtifactDescriptor, ...]:
    """Resolve the full dependency closure from catalog descriptors.

    Deliberately not the core's _resolve_closure (which reads installed
    manifests): at preflight, dependencies may not be installed at all.
    Same rules: cycle and revision-conflict detection; stable sorted order.

    Args:
        root: The root artifact reference to resolve from.
        catalog: The catalog providing descriptors for references.

    Returns:
        A tuple of artifact descriptors in stable sorted order (by ref).

    Raises:
        CatalogError: If an unknown ref is encountered, a dependency cycle
            is detected, or two different revisions of the same artifact_id
            appear in the closure.
    """
    resolved: dict[ArtifactRef, ArtifactDescriptor] = {}
    revisions: dict[str, ArtifactRef] = {}
    visiting: set[ArtifactRef] = set()

    def visit(ref: ArtifactRef) -> None:
        if ref in resolved:
            return
        if ref in visiting:
            raise CatalogError(f"dependency cycle at {ref.artifact_id}")
        seen = revisions.get(ref.artifact_id)
        if seen is not None and seen != ref:
            raise CatalogError(
                f"conflicting revisions for {ref.artifact_id}: {seen.revision} vs {ref.revision}"
            )
        visiting.add(ref)
        try:
            descriptor = catalog.descriptor(ref)
        except Exception as exc:
            raise CatalogError(f"unknown artifact {ref.artifact_id}@{ref.revision}") from exc
        revisions[ref.artifact_id] = ref
        for dep in descriptor.dependencies:
            visit(dep)
        visiting.discard(ref)
        resolved[ref] = descriptor

    visit(root)
    return tuple(resolved[ref] for ref in sorted(resolved))


class ArtifactAcquisitionService:
    """Consent-driven managed acquisition composed over the sealed 594 core.

    This service never bypasses ``ModelArtifactService``'s integrity
    guarantees; it only calls its public sync surface (``list_installed()``,
    ``disk_usage()``, ``artifact_path()``, and later ``install()``/lease
    methods via an executor hop) and adds the async, consent-gated flow on
    top: ``preflight()`` here, ``provision()`` in a later task.
    """

    def __init__(
        self,
        core: ModelArtifactService,
        *,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        credential_resolver: Callable[[str], str | None] | None = None,
        free_bytes_probe: Callable[[Path], int] | None = None,
        trusted_origins: frozenset[str] = frozenset(),
    ) -> None:
        """Compose an acquisition service over an existing artifact core.

        Args:
            core: The sealed ``ModelArtifactService`` this service reuses for
                every integrity-bearing operation (verify, promote, leases).
            client_factory: Optional zero-arg factory returning a caller-owned
                ``httpx.AsyncClient`` (reused across probes/fetches). When
                absent, a short-lived client is created and closed per call.
            credential_resolver: Optional repository -> token resolver. Not
                yet consulted by ``preflight()`` (the HEAD probe is always
                anonymous here); accepted now so the constructor shape is
                stable across the whole feature -- a later task wires actual
                credential attachment for gated repositories.
            free_bytes_probe: Optional override returning available free
                bytes at a given path, injected by tests instead of
                ``core.disk_usage().free_bytes`` / ``shutil.disk_usage``.
            trusted_origins: Hostnames exempt from the private-IP egress
                block, threaded into the preflight HEAD probe. Empty in
                production, where repository hosts are public; tests pass
                a fixture server's loopback hostname here.
        """
        self._core = core
        self._client_factory = client_factory
        self._credential_resolver = credential_resolver
        self._free_bytes_probe = free_bytes_probe
        self._trusted_origins = trusted_origins

    async def preflight(self, root: ArtifactRef, catalog: ArtifactCatalog) -> PreflightReport:
        """Aggregate space, staged-credit, and repository-gating checks.

        Walks the full dependency closure from catalog descriptors (not the
        core's installed-manifest walk -- dependencies may not exist on disk
        yet), then aggregates a frozen report covering how many bytes remain
        to download, how much of that is already durably staged, how much
        disk a same-artifact upgrade would retain, and whether any
        repository in the closure answers as gated (401/403) before any
        consent screen is shown.

        Args:
            root: The root artifact reference to resolve and preflight.
            catalog: Catalog supplying descriptors for the closure walk.

        Returns:
            A frozen ``PreflightReport``; call ``.grant()`` on it to obtain
            an ``AcquisitionConsent``.

        Raises:
            CatalogError: Propagated from an unknown ref, a dependency
                cycle, or a conflicting-revision closure.
        """
        closure = resolve_catalog_closure(root, catalog)
        fingerprint = closure_fingerprint(root, (descriptor.reference for descriptor in closure))

        installed = self._core.list_installed()
        installed_refs = {
            item.descriptor.reference for item in installed if item.descriptor is not None
        }
        # The prior active version of THIS artifact_id, only when an upgrade
        # would leave it behind under a different reference than root (the
        # same-ref case is just "already installed", not a retained extra).
        retained_descriptor = next(
            (
                item.descriptor
                for item in installed
                if item.descriptor is not None
                and item.active
                and item.descriptor.reference.artifact_id == root.artifact_id
                and item.descriptor.reference != root
            ),
            None,
        )

        entries: list[ArtifactPreflightEntry] = []
        download_bytes = 0
        already_staged_bytes = 0
        # First not-installed entry per repository, in stable closure order --
        # the representative whose URL gets the one bounded gating probe.
        gating_targets: dict[str, ArtifactPreflightEntry] = {}
        for descriptor in closure:
            ref = descriptor.reference
            already_installed = ref in installed_refs
            entry = ArtifactPreflightEntry(
                ref=ref,
                source_url=descriptor.source_url,
                repository=descriptor.upstream_repository,
                revision=descriptor.upstream_revision,
                license_id=descriptor.license_id,
                license_url=descriptor.license_url,
                precision=descriptor.precision,
                total_bytes=descriptor.expected_installed_bytes,
                file_count=len(descriptor.files),
                already_installed=already_installed,
            )
            entries.append(entry)
            if not already_installed:
                # Clamp per entry: a stale/corrupt sidecar claiming more
                # bytes than this artifact's own declared total must not
                # inflate the credit shown on the consent screen.
                staged = min(self._staged_bytes_for(ref), entry.total_bytes)
                already_staged_bytes += staged
                # Floored PER ENTRY (max(total-staged,0) here, not summed
                # totals minus summed staged then floored once at the end)
                # -- an aggregate subtraction would let one entry's
                # over-claimed credit silently offset another entry's real
                # download cost instead of just clamping its own.
                download_bytes += max(entry.total_bytes - staged, 0)
                gating_targets.setdefault(entry.repository, entry)

        # consume_source install() moves staged bytes into the immutable
        # store (os.replace, or copy+delete of the SAME bytes on EXDEV) --
        # there is no second on-disk copy of the payload to budget for. The
        # field survives at 0 so a future copy-based install path (or a
        # policy change back to consume_source=False) has somewhere honest
        # to report real overhead without an API/signature break.
        staging_overhead_bytes = 0

        retained_bytes = (
            retained_descriptor.expected_installed_bytes if retained_descriptor is not None else 0
        )

        required_bytes = (
            download_bytes
            + staging_overhead_bytes
            + retained_bytes
            + ACQUISITION_SAFETY_MARGIN_BYTES
        )

        destination = self._core.artifact_path(root)
        free_bytes = (
            self._free_bytes_probe(destination)
            if self._free_bytes_probe is not None
            else self._core.disk_usage().free_bytes
        )

        gating_errors = await self._probe_gating(gating_targets.values())

        return PreflightReport(
            root=root,
            closure_fingerprint=fingerprint,
            entries=tuple(entries),
            download_bytes=download_bytes,
            already_staged_bytes=already_staged_bytes,
            staging_overhead_bytes=staging_overhead_bytes,
            retained_bytes=retained_bytes,
            destination=destination,
            free_bytes=free_bytes,
            required_bytes=required_bytes,
            sufficient_space=free_bytes >= required_bytes,
            gating_errors=tuple(gating_errors),
        )

    def _staged_bytes_for(self, ref: ArtifactRef) -> int:
        """Best-effort resumable-byte credit from a fetch-state sidecar.

        A missing or unparseable sidecar reads as zero credit: staged bytes
        are only ever a hint surfaced on the consent screen ("up to N bytes
        already fetched") -- ``provision()`` independently re-validates
        against the server's current validators before trusting any of it.

        Args:
            ref: The artifact reference whose staging directory to inspect.

        Returns:
            The sum of ``bytes_done`` across every file in the sidecar, or 0.
        """
        sidecar = (
            self._core.staging_path
            / "managed"
            / ref.artifact_id
            / ref.revision
            / ref.variant
            / "fetch-state.json"
        )
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        files = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files, dict):
            return 0
        total = 0
        for info in files.values():
            if not isinstance(info, dict):
                continue
            bytes_done = info.get("bytes_done")
            if isinstance(bytes_done, int) and not isinstance(bytes_done, bool) and bytes_done >= 0:
                total += bytes_done
        return total

    async def _probe_gating(self, targets: Iterable[ArtifactPreflightEntry]) -> list[str]:
        """Bounded, anonymous HEAD probe per repository; collect 401/403s.

        Any other outcome -- 2xx/3xx/other 4xx/5xx, timeout, connection
        failure, or an egress-policy block -- is silently non-fatal here:
        those belong to the real transfer's error handling, not consent.
        Only an explicit auth-required status is a gating signal.

        Args:
            targets: One representative entry per unique repository.

        Returns:
            Gating-error messages naming the repository and the credential
            env var to set (never a token value); empty if none are gated.
        """
        targets = list(targets)
        if not targets:
            return []
        client = self._client_factory() if self._client_factory is not None else None
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient()
        assert client is not None
        errors: list[str] = []
        try:
            for entry in targets:
                try:
                    await check_url_or_raise_async(
                        entry.source_url, trusted_origins=self._trusted_origins
                    )
                    response = await client.head(
                        entry.source_url, timeout=_PREFLIGHT_PROBE_TIMEOUT_SECONDS
                    )
                except (httpx.HTTPError, EgressBlockedError):
                    continue
                if response.status_code in (401, 403):
                    errors.append(
                        f"repository '{entry.repository}' requires credentials "
                        f"(HTTP {response.status_code}); set {_CREDENTIAL_ENV_HINT} and retry"
                    )
        finally:
            if owns_client:
                await client.aclose()
        return errors
