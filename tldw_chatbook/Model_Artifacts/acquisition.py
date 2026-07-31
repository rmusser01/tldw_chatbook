"""TASK-595: managed model acquisition types and catalog resolution."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Protocol

import httpx

from tldw_chatbook.Utils.egress import EgressBlockedError, check_url_or_raise_async

from .leases import ArtifactLeaseTimeoutError, ArtifactOperationLease, LeaseMode
from .service import (
    ACQUISITION_SESSION_LEASE_KEY,
    ArtifactError,
    ArtifactRef,
    closure_fingerprint,
)

if TYPE_CHECKING:
    from .service import ArtifactDescriptor, ModelArtifactService


# Constants per spec (Docs/superpowers/specs/2026-07-30-managed-model-acquisition-design.md)
ACQUISITION_SAFETY_MARGIN_BYTES = 256 * 1024 * 1024
MAX_FILE_REFETCHES = 1

# Bounded timeout for the preflight repository-gating HEAD probe (Task 5).
_PREFLIGHT_PROBE_TIMEOUT_SECONDS = 10.0

# Non-blocking acquisition-session-lease timeout (Task 6): an immediate,
# typed AcquisitionBusyError beats a hang -- another process or in-process
# caller already holding the session lease means "busy right now", not
# "worth waiting for".
_SESSION_LEASE_TIMEOUT_SECONDS = 0.1

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


@dataclass
class _ProvisionProgressState:
    """Mutable closure-wide byte accounting threaded through provision's phases.

    Task 6 only constructs one instance per ``provision()`` call and threads
    it, unchanged, into ``_fetch_artifact`` and ``_preverify_artifact`` for
    every artifact in the closure. Tasks 7-8 read and update ``bytes_done``
    as each declared file streams or is pre-verified, and call ``callback``
    with an ``AcquisitionProgress`` event carrying these CLOSURE-WIDE totals
    -- per the design spec, fetch/pre-verify progress is reported summed
    across the whole closure, not reset per artifact or per file.

    Args:
        callback: The caller's optional progress sink, forwarded unchanged
            from ``provision()``'s own ``progress`` keyword argument.
        bytes_total: Total bytes still to download across the whole
            closure (``PreflightReport.download_bytes``), computed once
            before the per-artifact loop starts.
        bytes_done: Running total of bytes fetched so far across every
            artifact already processed in this run; starts at zero.
    """

    callback: Callable[[AcquisitionProgress], None] | None
    bytes_total: int
    bytes_done: int = 0


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
        # In-process serialization for provision() (Task 6): queues same-
        # process concurrent callers so only one at a time ever attempts the
        # OS-backed acquisition-session lease below -- without this, two
        # concurrent in-process calls would race each other for that
        # exclusive lease and one would see AcquisitionBusyError even though
        # no OTHER process is involved, which is the wrong signal in-process
        # (queue and proceed, not "busy"). Safe to construct without a
        # running loop on Python >= 3.10.
        self._lock = asyncio.Lock()

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
        _closure, report, gating_targets = self._aggregate_closure(root, catalog)
        gating_errors = await self._probe_gating(gating_targets.values())
        return replace(report, gating_errors=tuple(gating_errors))

    def _aggregate_closure(
        self,
        root: ArtifactRef,
        catalog: ArtifactCatalog,
    ) -> tuple[tuple[ArtifactDescriptor, ...], PreflightReport, dict[str, ArtifactPreflightEntry]]:
        """Resolve the catalog closure and aggregate space/staged-credit math.

        Pure and network-free: the only I/O is ``core.list_installed()`` and
        ``core.disk_usage()`` (or the injected ``free_bytes_probe``), the
        same fast synchronous calls ``preflight()`` already made directly
        (Task 5) without an executor hop. Extracted (Task 6) so
        ``preflight()`` and ``provision()`` share one aggregation instead of
        two copies that could silently drift apart: ``preflight()`` layers
        its own network gating probe on top of the returned report;
        ``provision()`` re-runs this exact aggregation, still network-free,
        to recheck the closure fingerprint and free space against a
        possibly-drifted catalog without repeating the gating probe a
        second time per run (gating already passed at grant time).

        Args:
            root: The root artifact reference to resolve and aggregate.
            catalog: Catalog supplying descriptors for the closure walk.

        Returns:
            A tuple of: the closure descriptors in stable sorted order; a
            ``PreflightReport`` with ``gating_errors`` deliberately left
            empty (the caller decides whether and how to probe); and the
            per-repository gating-probe targets, for a caller that wants to
            hand them to ``_probe_gating``.

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

        report = PreflightReport(
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
            gating_errors=(),
        )
        return closure, report, gating_targets

    async def provision(
        self,
        root: ArtifactRef,
        consent: AcquisitionConsent,
        catalog: ArtifactCatalog,
        *,
        progress: Callable[[AcquisitionProgress], None] | None = None,
    ) -> ArtifactRef:
        """Acquire and activate one consented closure, resuming idempotently.

        Serializes against every other ``provision()`` call twice over: an
        in-process ``asyncio.Lock`` queues same-process callers first (so
        two concurrent calls in this process never race the OS-backed lease
        against each other and see a spurious busy error), then a single
        exclusive, non-blocking ``ACQUISITION_SESSION_LEASE_KEY`` lease
        serializes against every other OS process. The session lease is
        held for this call's ENTIRE run -- acquired before any phase runs,
        released only in a ``finally`` after activation succeeds or any step
        raises -- because ``reconcile()``'s managed-staging GC
        (``service.py``'s ``_gc_managed_staging``) treats a free session
        lease as permission to delete orphaned download staging; releasing
        early would let a reconcile pass race a live download the same way
        an early Task 2 draft let it race a live ``install()``.

        Re-walks ``catalog`` from ``root`` independently of ``consent``: the
        closure fingerprint is recomputed via the same network-free
        aggregation ``preflight()`` uses and compared against
        ``consent.closure_fingerprint`` -- a changed dependency set since
        ``preflight()`` raises ``ConsentMismatchError``, since consent to
        the old content no longer applies to the new one. Free space is
        rechecked from that same aggregation; this recheck deliberately
        skips the network gating probe (gating already passed at grant
        time, and repeating it here would add a per-provision network round
        trip for no benefit at this phase).

        Artifacts already present in ``core.list_installed()`` are skipped
        entirely, bypassing the fetch/pre-verify/install phases -- this is
        both the idempotent-completion path for a fully provisioned closure
        and the crash-after-install recovery path (Tasks 7-8 exercise the
        latter with a mid-closure crash).

        Args:
            root: The root artifact reference to provision.
            consent: Consent obtained from a prior ``preflight().grant()``
                call.
            catalog: Catalog supplying descriptors for the closure re-walk.
                Not carried by ``consent`` itself -- re-walking is what
                makes the fingerprint drift check meaningful, and the
                freshly resolved descriptors supply the file URLs the
                fetch phase needs.
            progress: Optional sink for ``AcquisitionProgress`` events
                emitted by the fetch and pre-verify phases (Tasks 7-8).

        Returns:
            The activated root artifact reference.

        Raises:
            AcquisitionBusyError: Another acquisition session -- in this
                process or another -- already holds the session lease.
            ConsentMismatchError: The re-walked closure fingerprint no
                longer matches ``consent`` (the catalog changed since
                ``preflight()``).
            InsufficientSpaceError: Free space no longer covers the
                required bytes for this closure.
            CatalogError: Propagated from an unknown ref, a dependency
                cycle, or a conflicting-revision closure during the
                re-walk.
            NotImplementedError: A not-yet-installed artifact reached the
                fetch, pre-verify, or install phase stub (Tasks 7-8 fill
                these in).
        """
        async with self._lock:
            loop = asyncio.get_running_loop()
            lease = ArtifactOperationLease(
                self._core.locks_path,
                ACQUISITION_SESSION_LEASE_KEY,
                LeaseMode.EXCLUSIVE,
                timeout_seconds=_SESSION_LEASE_TIMEOUT_SECONDS,
            )
            try:
                await loop.run_in_executor(None, lease.acquire)
            except ArtifactLeaseTimeoutError as error:
                raise AcquisitionBusyError(
                    "another managed acquisition session is already active"
                ) from error
            try:
                closure, report, _gating_targets = self._aggregate_closure(root, catalog)
                if report.closure_fingerprint != consent.closure_fingerprint:
                    raise ConsentMismatchError(
                        "closure fingerprint changed since consent was granted; "
                        "re-run preflight and obtain new consent"
                    )
                if not report.sufficient_space:
                    raise InsufficientSpaceError(
                        f"required {report.required_bytes} bytes but only "
                        f"{report.free_bytes} free"
                    )

                installed = self._core.list_installed()
                installed_refs = {
                    item.descriptor.reference
                    for item in installed
                    if item.descriptor is not None
                }

                progress_state = _ProvisionProgressState(
                    callback=progress,
                    bytes_total=report.download_bytes,
                )

                for descriptor in closure:
                    if descriptor.reference in installed_refs:
                        continue
                    staging_dir = (
                        self._core.staging_path
                        / "managed"
                        / descriptor.reference.artifact_id
                        / descriptor.reference.revision
                        / descriptor.reference.variant
                    )
                    await self._fetch_artifact(descriptor, staging_dir, progress_state)
                    await self._preverify_artifact(descriptor, staging_dir, progress_state)
                    await self._install_artifact(descriptor, staging_dir)

                return await loop.run_in_executor(None, self._core.activate, root)
            finally:
                await loop.run_in_executor(None, lease.release)

    async def _fetch_artifact(
        self,
        descriptor: ArtifactDescriptor,
        staging_dir: Path,
        progress_state: _ProvisionProgressState,
    ) -> None:
        """Stream every declared file into durable staging with resume support.

        Stub (Task 6); Task 7 implements this phase. The signature is final
        -- ``provision()``'s call site will not change when it does.

        Args:
            descriptor: The artifact whose declared files to fetch.
            staging_dir: The durable ``staging/managed/<id>/<rev>/<variant>``
                directory for this artifact.
            progress_state: Closure-wide progress accounting to read and
                update as bytes stream in.

        Raises:
            NotImplementedError: Always, until Task 7 implements this phase.
        """

        raise NotImplementedError("fetch phase is implemented in a later task")

    async def _preverify_artifact(
        self,
        descriptor: ArtifactDescriptor,
        staging_dir: Path,
        progress_state: _ProvisionProgressState,
    ) -> None:
        """Streaming-verify every staged file's SHA-256 before install.

        Stub (Task 6); Task 8 implements this phase. The signature is final
        -- ``provision()``'s call site will not change when it does.

        Args:
            descriptor: The artifact whose staged files to verify.
            staging_dir: The durable staging directory holding the fetched
                files.
            progress_state: Closure-wide progress accounting to read and
                update as bytes are verified.

        Raises:
            NotImplementedError: Always, until Task 8 implements this phase.
        """

        raise NotImplementedError("pre-verify phase is implemented in a later task")

    async def _install_artifact(
        self,
        descriptor: ArtifactDescriptor,
        staging_dir: Path,
    ) -> None:
        """Promote one pre-verified staged directory into the immutable store.

        Stub (Task 6); Task 8 implements this phase. The signature is final
        -- ``provision()``'s call site will not change when it does.

        Args:
            descriptor: The artifact to install.
            staging_dir: The durable staging directory holding the verified
                files.

        Raises:
            NotImplementedError: Always, until Task 8 implements this phase.
        """

        raise NotImplementedError("install phase is implemented in a later task")

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
