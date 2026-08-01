"""TASK-595: managed model acquisition types and catalog resolution."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import httpx

from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Utils.atomic_file_ops import atomic_write_json
from tldw_chatbook.Utils.egress import EgressBlockedError, check_url_or_raise_async

from .fetch import (
    FetchRestartRequired,
    FetchResult,
    FetchTooLargeError,
    FetchTransportError,
    FetchValidators,
    stream_fetch,
)
from .leases import ArtifactLeaseTimeoutError, ArtifactOperationLease, LeaseMode
from .service import (
    ACQUISITION_SESSION_LEASE_KEY,
    NONBLOCKING_LEASE_TIMEOUT_SECONDS,
    ArtifactConflictError,
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactPathError,
    ArtifactRef,
    ArtifactStateError,
    closure_fingerprint,
)

if TYPE_CHECKING:
    from .service import (
        ArtifactDescriptor,
        ArtifactFile,
        ModelArtifactService,
        _ManagedDownloadStage,
    )


# Constants per spec (Docs/superpowers/specs/2026-07-30-managed-model-acquisition-design.md)
ACQUISITION_SAFETY_MARGIN_BYTES = 256 * 1024 * 1024
MAX_FILE_REFETCHES = 1

# Bounded timeout for the preflight repository-gating HEAD probe (Task 5).
_PREFLIGHT_PROBE_TIMEOUT_SECONDS = 10.0

# Non-blocking acquisition-session-lease timeout: an immediate, typed
# AcquisitionBusyError beats a hang -- another process or in-process caller
# already holding the session lease means "busy right now", not "worth
# waiting for". Imported from service.py (not redefined here) -- this same
# 0.1s value also gates reconcile()'s two staging-GC lease probes, and a
# single shared constant keeps the three call sites from silently drifting
# apart.
_SESSION_LEASE_TIMEOUT_SECONDS = NONBLOCKING_LEASE_TIMEOUT_SECONDS

_CoreCallResult = TypeVar("_CoreCallResult")

# Credential hint named in gating_errors -- never a token value. Matches the
# precedence EnvConfigCredentialResolver (below) actually implements against
# these same names: HUGGINGFACE_API_KEY, then HF_TOKEN, then config.
_CREDENTIAL_ENV_HINT = "HUGGINGFACE_API_KEY (or HF_TOKEN)"

# TASK-1694: the fetch-state sidecar's filename, addressed inside a
# ``state/`` directory that is a SIBLING of the download stage's
# ``payload/`` subtree (see ``_fetch_sidecar_path`` and
# ``ModelArtifactService._download_stage_for``/``_finalize_download_stage``
# in service.py). This supersedes the pre-1694 design, where the sidecar
# lived as a sibling FILE of the (then bare) staging directory
# (``<variant>.fetch-state.json``, named via the now-removed
# ``_FETCH_SIDECAR_SUFFIX``): that convention existed only to keep resume
# metadata out of what ``core.install(..., consume_source=True)``
# validated and promoted. The service-owned download stage makes that
# workaround unnecessary -- ``state/`` is never part of the ``payload/``
# subtree ``_finalize_download_stage`` renames into the immutable
# destination, so resume metadata cannot end up inside a promoted
# artifact by construction, regardless of what acquisition.py names it or
# where inside ``state/`` it puts it. See
# Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-
# reconciliation.md item 1.
_FETCH_STATE_FILENAME = "fetch-state.json"


def _fetch_sidecar_path(staging_dir: Path) -> Path:
    """Fetch-state sidecar path for a download stage's payload directory.

    Args:
        staging_dir: A download stage's ``payload/`` directory (i.e.
            ``_ManagedDownloadStage.payload``, or -- for the standalone
            unit tests that exercise ``_fetch_artifact``/
            ``_preverify_artifact`` directly without a real stage -- any
            directory whose parent also owns a sibling ``state/``
            directory this call may create).

    Returns:
        ``staging_dir.parent / "state" / "fetch-state.json"`` -- inside
        the stage's ``state/`` subtree, never inside ``staging_dir``
        itself (which is exactly what gets promoted on a successful
        finalize).
    """
    return staging_dir.parent / "state" / _FETCH_STATE_FILENAME


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


@runtime_checkable
class CredentialResolver(Protocol):
    """Resolves a per-request bearer token for a repository, without persistence.

    An implementation returns ``None`` when no credential is configured --
    a gated repository with no working credential then fails visibly at
    ``preflight()`` (surfaced via ``PreflightReport.gating_errors``) rather
    than silently proceeding anonymously and failing later, mid-transfer.

    A token this protocol resolves is attached ONLY to the request for the
    entry's OWN origin (see ``ArtifactAcquisitionService._auth_headers``,
    consulted by both the preflight gating probe and the real fetch);
    ``fetch.stream_fetch`` independently strips any ``Authorization`` header
    on a cross-origin redirect hop regardless -- defense in depth, not the
    only guard against a token reaching the wrong host.
    """

    def resolve(self, repository: str) -> str | None:
        """Resolve a bearer token for ``repository``.

        Args:
            repository: The upstream repository identifier, e.g.
                ``ArtifactDescriptor.upstream_repository``.

        Returns:
            A bearer token string, or ``None`` if no credential is
            configured for this repository.
        """
        ...


class EnvConfigCredentialResolver:
    """Default ``CredentialResolver``: env vars, then config -- no keyring yet.

    Precedence mirrors this codebase's existing HuggingFace credential
    lookup exactly: ``config.py``'s ``huggingface_api_key`` resolves
    ``HUGGINGFACE_API_KEY`` from the environment ahead of the ``[API]``
    config section, and ``Constants.py`` separately documents ``HF_TOKEN``
    as the equivalent env name other tooling in this ecosystem (llama.cpp's
    server) already uses. This resolver checks, in order: ``HUGGINGFACE_API_KEY``
    env, then ``HF_TOKEN`` env, then the ``[API] huggingface_api_key``
    config setting via ``config.get_cli_setting``.

    Keyring lookup ("where available", per the design spec's Credentials
    section) is DELIBERATELY not implemented here: this class wires only
    the env/config precedence this repository's other HuggingFace-key
    consumers already use. A keyring backend is a follow-up, not required
    by this task's acceptance criteria.

    Read-only, every call -- a resolved token is never written back to the
    environment, config, or anywhere else.
    """

    def resolve(self, repository: str) -> str | None:
        """Resolve a HuggingFace-style bearer token.

        ``repository`` is accepted to satisfy ``CredentialResolver`` but
        otherwise unused: the token this resolver returns is global to the
        process's configured identity (env or config), not scoped per
        repository -- matching how every other HuggingFace-key consumer in
        this codebase already resolves credentials.

        Args:
            repository: Unused; see class docstring.

        Returns:
            The resolved token, or ``None`` if none is configured anywhere
            in the env/config precedence chain.
        """
        for env_var in ("HUGGINGFACE_API_KEY", "HF_TOKEN"):
            token = os.environ.get(env_var)
            if token:
                return token
        config_token = get_cli_setting("API", "huggingface_api_key", None)
        if isinstance(config_token, str) and config_token:
            return config_token
        return None


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
    every artifact in the closure. Task 7 reads and updates ``bytes_done``
    as each declared file streams; Task 8 reads and updates
    ``preverify_bytes_done`` as each staged file is hashed. The two phases
    are tracked against SEPARATE totals, not just separate running counts:
    ``bytes_total`` is the closure's declared DOWNLOAD bytes
    (``PreflightReport.download_bytes``), which nets out any already-staged
    credit from a resumed run, while ``preverify_bytes_total`` is the
    closure's full declared file size, unaffected by staged credit --
    ``_hash_staged_file`` always re-hashes a staged file's ENTIRE on-disk
    content regardless of how much of it was freshly downloaded versus
    already present when this run started, so reusing the (possibly
    netted-down) fetch total for pre-verify's progress would make ANY
    resumed run's pre-verify events overshoot 100%. ``callback`` is called
    with an ``AcquisitionProgress`` event carrying whichever counter (and
    total) belongs to the active phase.

    Args:
        callback: The caller's optional progress sink, forwarded unchanged
            from ``provision()``'s own ``progress`` keyword argument.
        bytes_total: Total bytes still to download across the whole closure
            (``PreflightReport.download_bytes``), computed once before the
            per-artifact loop starts. Nets out already-staged credit --
            NOT the total bytes pre-verify will hash; see
            ``preverify_bytes_total``.
        preverify_bytes_total: Total bytes pre-verify will hash across the
            whole closure -- the full declared size of every not-yet-
            installed artifact's files (``ArtifactDescriptor.
            expected_installed_bytes``), regardless of how much of that was
            already staged before this run started. Defaults to
            ``bytes_total`` (see ``__post_init__``) so existing direct-
            construction call sites that only exercise the fetch phase are
            unaffected; ``provision()`` itself always passes both totals
            explicitly, independently computed.
        bytes_done: Running total of bytes fetched so far across every
            artifact already processed in this run; starts at zero.
        preverify_bytes_done: Running total of bytes hashed so far during
            the pre-verify phase, across every artifact already processed
            in this run; starts at zero. A pre-verify hash mismatch and
            refetch (Task 8) re-streams and re-hashes the same file, so
            this counter can advance past ``preverify_bytes_total`` in that
            recovery path -- an acceptable cosmetic wrinkle in an uncommon
            retry, not a steady-state property.
    """

    callback: Callable[[AcquisitionProgress], None] | None
    bytes_total: int
    preverify_bytes_total: int | None = None
    bytes_done: int = 0
    preverify_bytes_done: int = 0

    def __post_init__(self) -> None:
        """Default ``preverify_bytes_total`` to ``bytes_total`` when unset."""

        if self.preverify_bytes_total is None:
            self.preverify_bytes_total = self.bytes_total


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


def _require_single_file(descriptor: ArtifactDescriptor) -> None:
    """Raise ``CatalogError`` unless ``descriptor`` declares exactly one file.

    Per-file source URLs are undefined until the catalog work
    (TASK-596/1301) specifies them (see ``_file_url``). Shared by
    ``_aggregate_closure`` -- so a not-yet-installed multi-file descriptor
    fails ``preflight()`` itself, before any report or consent exists --
    and ``_fetch_artifact`` (defense-in-depth: a descriptor's shape could in
    principle change between ``preflight()`` and ``provision()``'s
    independent catalog re-walk).

    Args:
        descriptor: The descriptor to check.

    Raises:
        CatalogError: ``descriptor`` declares more than one file.
    """

    if len(descriptor.files) != 1:
        ref = descriptor.reference
        raise CatalogError(
            f"{ref.artifact_id}@{ref.revision} declares "
            f"{len(descriptor.files)} files, but per-file source URLs "
            "are undefined until the catalog work (TASK-596/1301) "
            "specifies them"
        )


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
        credential_resolver: CredentialResolver | None = None,
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
            credential_resolver: Optional ``CredentialResolver`` (see
                ``EnvConfigCredentialResolver`` for the default env/config
                implementation). ``None`` means every request goes out
                anonymous; a gated repository then fails at ``preflight()``
                with a ``gating_errors`` entry naming the credential env var
                to set, never a token value. When present, a resolved
                token is attached as ``Authorization: Bearer`` to BOTH the
                preflight gating HEAD probe and the real per-file fetch
                (``_auth_headers``), so a working credential clears gating
                and lets the transfer itself succeed against the same
                gated repository.
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

    def _auth_headers(self, repository: str) -> dict[str, str] | None:
        """Resolve an ``Authorization`` header for ``repository``, or ``None``.

        The single seam every credentialed request in this service goes
        through -- the preflight gating HEAD probe (``_probe_gating``) and
        the real per-file fetch (``_fetch_one_file``) both call this rather
        than touching ``self._credential_resolver`` directly, so there is
        exactly one place that ever holds a resolved token in memory here.
        Never logged and never embedded in an error message -- callers that
        need to describe what's being fetched use ``repository`` or the
        file path, never this return value's contents.

        Args:
            repository: The upstream repository identifier to resolve a
                credential for.

        Returns:
            ``{"Authorization": "Bearer <token>"}`` when a resolver is
            configured and returns a truthy token for ``repository``, else
            ``None`` (no header attached -- the request goes out anonymous).
        """
        if self._credential_resolver is None:
            return None
        token = self._credential_resolver.resolve(repository)
        if not token:
            return None
        return {"Authorization": f"Bearer {token}"}

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
                cycle, a conflicting-revision closure, or a not-yet-
                installed entry declaring more than one file (see
                ``_require_single_file``).
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
                cycle, a conflicting-revision closure, or a not-yet-
                installed entry declaring more than one file (see
                ``_require_single_file``).
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
                # Fail loudly here, not just later in _fetch_artifact: a
                # multi-file descriptor that still needs fetching is a
                # catalog-contract problem (per-file URLs are undefined --
                # see _require_single_file), and the spec requires catalog
                # problems to surface at preflight, before any report or
                # consent exists. An ALREADY-installed multi-file
                # descriptor never reaches provision()'s fetch phase (the
                # per-artifact loop skips installed entries outright), so
                # it is deliberately not checked here.
                _require_single_file(descriptor)
                # Clamp per entry: a stale/corrupt sidecar claiming more
                # bytes than this artifact's own declared total must not
                # inflate the credit shown on the consent screen.
                # ``_staged_bytes_for`` already caps each FILE's own credit
                # by its actual on-disk size (not just the declared size),
                # but this outer clamp against the entry's aggregate total
                # stays as defense-in-depth.
                staged = min(self._staged_bytes_for(descriptor), entry.total_bytes)
                already_staged_bytes += staged
                # Floored PER ENTRY (max(total-staged,0) here, not summed
                # totals minus summed staged then floored once at the end)
                # -- an aggregate subtraction would let one entry's
                # over-claimed credit silently offset another entry's real
                # download cost instead of just clamping its own.
                download_bytes += max(entry.total_bytes - staged, 0)
                gating_targets.setdefault(entry.repository, entry)

        # TASK-1694: _install_artifact finalizes a download stage by
        # RENAMING its payload subtree into the immutable destination
        # (core._finalize_download_stage) -- fetched bytes never leave the
        # service's own same-filesystem staging until that single rename,
        # so there is no second on-disk copy of the payload to budget for.
        # The field survives at 0 so a future copy-based finalization path
        # has somewhere honest to report real overhead without an
        # API/signature break.
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
        and the crash-after-install recovery path.

        Every not-yet-installed artifact in the closure runs its three
        phases strictly in order -- fetch (Task 7), pre-verify (Task 8),
        install (Task 8) -- before the loop moves to the next artifact; a
        ``verify-install`` progress event follows each artifact's install.
        Once every artifact is installed, the whole closure is activated
        (``core.activate``) and a final ``activate`` progress event fires.

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
                emitted by every phase: real byte detail for ``fetch`` and
                ``pre-verify``, indeterminate per-artifact events for
                ``verify-install`` and ``activate``.

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
            TransferError: A file failed to fetch, a staged file failed
                pre-verify a second time after one automatic refetch, or
                ``core.install``/``core.activate`` raised a core
                ``ArtifactError`` (integrity, conflict, or state/lease
                contention -- see ``_run_core_call``). Nothing further is
                installed or activated for that artifact once this is
                raised.
        """
        async with self._lock:
            loop = asyncio.get_running_loop()
            lease = ArtifactOperationLease(
                self._core.locks_path,
                ACQUISITION_SESSION_LEASE_KEY,
                LeaseMode.EXCLUSIVE,
                timeout_seconds=_SESSION_LEASE_TIMEOUT_SECONDS,
            )
            # The acquire itself must live INSIDE this try/finally, not
            # before it (a prior review found the release leaked): once
            # lease.acquire() has started running in its worker thread, it
            # cannot be interrupted, so a CancelledError delivered while it
            # is still in flight must not skip past the finally below
            # without first learning whether the acquire actually
            # succeeded. asyncio.shield keeps the executor future running
            # even if THIS await is cancelled; the except-CancelledError
            # branch then waits for that same future to actually settle
            # (ignoring whatever it raises) before re-raising, so
            # `lease.acquired` below always reflects the true outcome
            # instead of a stale "not yet" snapshot.
            acquire_future = loop.run_in_executor(None, lease.acquire)
            try:
                try:
                    await asyncio.shield(acquire_future)
                except ArtifactLeaseTimeoutError as error:
                    raise AcquisitionBusyError(
                        "another managed acquisition session is already active"
                    ) from error
                except asyncio.CancelledError:
                    with contextlib.suppress(BaseException):
                        await acquire_future
                    raise

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
                    preverify_bytes_total=sum(
                        descriptor.expected_installed_bytes
                        for descriptor in closure
                        if descriptor.reference not in installed_refs
                    ),
                )

                for descriptor in closure:
                    if descriptor.reference in installed_refs:
                        continue
                    # TASK-1694: a service-owned, marked download stage
                    # replaces the old bare
                    # ``staging/managed/<id>/<rev>/<variant>`` directory.
                    # ``stage.payload`` is where fetch/pre-verify write and
                    # read declared files -- the exact subtree
                    # ``_install_artifact`` (via
                    # ``core._finalize_download_stage``) verifies and
                    # RENAMES into the immutable destination. Resume
                    # metadata (the fetch-state sidecar) is written under
                    # ``stage.state`` -- a sibling of ``payload``, never
                    # promoted -- see ``_fetch_sidecar_path``.
                    stage = await self._run_core_call(
                        "stage",
                        descriptor.reference,
                        functools.partial(
                            self._core._download_stage_for, descriptor, create=True
                        ),
                    )
                    # create=True always returns a stage (never None); the
                    # Optional return type only covers the create=False
                    # lookup path (see _staged_bytes_for).
                    assert stage is not None
                    await self._fetch_artifact(descriptor, stage.payload, progress_state)
                    await self._preverify_artifact(descriptor, stage.payload, progress_state)
                    await self._install_artifact(descriptor, stage)
                    # _install_artifact's signature is (descriptor, stage)
                    # -- no progress_state -- so the per-artifact
                    # "verify-install" event is emitted here, by the caller
                    # that already has it, immediately after the phase it
                    # describes actually completes.
                    self._emit_indeterminate_progress(
                        progress_state, "verify-install", descriptor.reference
                    )

                activated = await self._run_core_call(
                    "activate", root, functools.partial(self._core.activate, root)
                )
                self._emit_indeterminate_progress(progress_state, "activate", root)
                return activated
            finally:
                if lease.acquired:
                    await loop.run_in_executor(None, lease.release)

    async def _fetch_artifact(
        self,
        descriptor: ArtifactDescriptor,
        staging_dir: Path,
        progress_state: _ProvisionProgressState,
    ) -> None:
        """Stream every declared file into durable staging with resume support.

        For each file declared on ``descriptor``: a sidecar entry already
        marked complete (and whose on-disk size matches) is skipped outright;
        otherwise the durable sidecar's ``bytes_done`` is reconciled against
        the file's ACTUAL on-disk size before any resume is attempted (see
        ``_reconcile_durable_bytes``), then the file is streamed via
        ``fetch.stream_fetch`` -- resumed with a ``Range`` request when
        strong validators survive reconciliation, restarted from zero
        otherwise. ``fetch.FetchRestartRequired`` (validators changed, or
        the server ignored ``Range``) truncates and restarts that one file
        from zero exactly once inline; this is unrelated to Task 8's
        pre-verify refetch-once counter, which guards against a corrupt
        payload that DOWNLOADED cleanly but fails its SHA-256.

        Durability order (spec-mandated): ``stream_fetch`` fsyncs the file's
        data before returning; only after a successful return is the sidecar
        rewritten (atomically, via the same ``atomic_write_json`` the core
        uses) to record the new checkpoint. A file that fails to fetch at
        all -- network drop, ENOSPC, oversized body -- leaves the sidecar
        untouched at its last durable checkpoint; the staging directory and
        any unfsynced on-disk bytes past that checkpoint are left in place
        (never cleaned up here) for a later resume attempt to reconcile.

        Args:
            descriptor: The artifact whose declared files to fetch.
            staging_dir: The durable ``staging/managed/<id>/<rev>/<variant>``
                directory for this artifact.
            progress_state: Closure-wide progress accounting to read and
                update as bytes stream in.

        Raises:
            CatalogError: ``descriptor`` declares more than one file --
                per-file source URLs are undefined until the catalog work
                (TASK-596/1301) specifies them (see ``_file_url``). Raised
                before touching staging, the sidecar, or the network: a
                catalog-contract problem, not a transfer failure.
                ``_aggregate_closure`` already rejects this same shape for
                a not-yet-installed entry at ``preflight()`` time; this is
                defense-in-depth for a descriptor that changed shape
                between ``preflight()`` and ``provision()``'s independent
                catalog re-walk.
            TransferError: A file failed to fetch -- network/transport
                failure, disk I/O failure (e.g. ENOSPC), an egress-policy
                block, or a response body exceeding the file's declared
                size. ``retryable`` is True for transport/I/O failures,
                False for an egress block or an oversized body (the same
                URL would presumably answer the same way again).
            asyncio.CancelledError: Propagated untouched if this call is
                cancelled while awaiting network I/O; honored between
                stream_fetch's internal chunks (asyncio-native, no manual
                polling).
        """

        _require_single_file(descriptor)

        staging_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = _fetch_sidecar_path(staging_dir)
        sidecar = self._load_fetch_sidecar(sidecar_path)

        client = self._client_factory() if self._client_factory is not None else None
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient()
        assert client is not None
        try:
            for file in descriptor.files:
                await self._fetch_one_file(
                    descriptor, file, staging_dir, sidecar, sidecar_path, progress_state, client
                )
        finally:
            if owns_client:
                await client.aclose()

    def _file_url(self, descriptor: ArtifactDescriptor, file: ArtifactFile) -> str:
        """Resolve one declared file's download URL.

        ``ArtifactDescriptor.source_url`` is the only fetchable location
        this schema carries today (Task 596/1301's catalog work owns
        richer per-file source metadata); when a descriptor declares
        exactly one file -- the only shape exercised end-to-end so far --
        ``source_url`` IS that file's URL directly, matching how
        ``_probe_gating`` already treats it (a bare GET/HEAD target, never
        joined with anything).

        ``ArtifactDescriptor`` permits more than one declared file, but
        nothing upstream of this task defines what a multi-file
        descriptor's per-file URLs actually are -- guessing a joined
        ``source_url`` + ``file.path`` URL here would silently fetch the
        WRONG bytes for every file whenever that guess doesn't match
        reality, with no signal to the caller that anything went wrong.
        Failing loudly is safer than guessing quietly: see ``Raises``.

        Args:
            descriptor: The artifact descriptor supplying ``source_url``.
            file: The declared file whose URL to resolve.

        Returns:
            The absolute URL to GET this file's bytes from.

        Raises:
            CatalogError: ``descriptor`` declares more than one file --
                per-file source URLs are undefined until the catalog work
                (TASK-596/1301) specifies them.
        """

        if len(descriptor.files) != 1:
            ref = descriptor.reference
            raise CatalogError(
                f"{ref.artifact_id}@{ref.revision} declares "
                f"{len(descriptor.files)} files, but per-file source URLs "
                "are undefined until the catalog work (TASK-596/1301) "
                "specifies them -- refusing to guess a URL for "
                f"'{file.path}'"
            )
        return descriptor.source_url

    @staticmethod
    def _load_fetch_sidecar(sidecar_path: Path) -> dict:
        """Best-effort load of a fetch-state sidecar; a missing/corrupt one reads empty.

        Args:
            sidecar_path: The ``fetch-state.json`` path for one artifact.

        Returns:
            ``{"files": {...}}`` -- the parsed sidecar, or an empty shell.
        """

        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"files": {}}
        files = payload.get("files") if isinstance(payload, dict) else None
        return {"files": files if isinstance(files, dict) else {}}

    @staticmethod
    def _reconcile_durable_bytes(destination: Path, recorded_done: int) -> int:
        """Trust on-disk bytes only up to the sidecar's durable checkpoint.

        ``fetch.stream_fetch`` resumes by opening its destination in append
        mode at whatever the file's CURRENT length happens to be -- it does
        not seek to ``resume_from`` itself. A file that grew past the last
        durable sidecar checkpoint (a crash or dropped connection after
        partial, un-fsynced writes) must be truncated back down to that
        checkpoint before any resume is attempted, or unverified bytes would
        be silently trusted. Symmetrically, a file SHORTER than the sidecar
        claims means the sidecar itself over-claims -- nothing about it can
        be trusted, so this restarts the file from zero.

        Args:
            destination: The staged file's path (may not exist yet).
            recorded_done: The sidecar's last durably recorded byte count
                for this file (0 if there is no usable entry).

        Returns:
            The byte count now safe to resume from -- always either
            ``recorded_done`` (file truncated down to it, or already
            consistent) or ``0`` (sidecar over-claimed; file removed).
        """

        actual_bytes = destination.stat().st_size if destination.exists() else 0
        if actual_bytes > recorded_done:
            with open(destination, "r+b") as fh:
                fh.truncate(recorded_done)
            return recorded_done
        if actual_bytes < recorded_done:
            if destination.exists():
                destination.unlink()
            return 0
        return recorded_done

    async def _fetch_one_file(
        self,
        descriptor: ArtifactDescriptor,
        file: ArtifactFile,
        staging_dir: Path,
        sidecar: dict,
        sidecar_path: Path,
        progress_state: _ProvisionProgressState,
        client: httpx.AsyncClient,
    ) -> None:
        """Fetch (or skip, or resume, or restart) one declared file.

        Args:
            descriptor: The artifact this file belongs to.
            file: The declared file to fetch.
            staging_dir: The durable staging directory for this artifact.
            sidecar: The mutable in-memory sidecar payload (``{"files": {}}``);
                updated and persisted in place on a successful fetch.
            sidecar_path: Where to atomically persist ``sidecar`` after a
                successful fetch.
            progress_state: Closure-wide progress accounting.
            client: The shared ``httpx.AsyncClient`` for this artifact's
                fetch phase.

        Raises:
            TransferError: See ``_fetch_artifact``.
        """

        destination = staging_dir / file.path
        destination.parent.mkdir(parents=True, exist_ok=True)

        entry = sidecar["files"].get(file.path)
        entry = dict(entry) if isinstance(entry, dict) else {}
        recorded_raw = entry.get("bytes_done")
        recorded_done = (
            recorded_raw
            if isinstance(recorded_raw, int)
            and not isinstance(recorded_raw, bool)
            and recorded_raw >= 0
            else 0
        )

        recorded_done = self._reconcile_durable_bytes(destination, recorded_done)

        # A checkpoint from a PRIOR provision() run can exceed the file's
        # CURRENT declared size if the catalog's declared size for this
        # artifact/revision/variant shrank between runs (a corrected or
        # re-cut upstream entry) -- reconciliation above only cross-checks
        # recorded_done against the file's ACTUAL on-disk bytes, never
        # against file.size_bytes. Left unchecked, this becomes
        # resume_from >= max_bytes inside stream_fetch, which raises
        # FetchTooLargeError -- wrongly surfaced as a non-retryable
        # "upstream body exceeds declared size" failure for what is really
        # just a stale, over-large checkpoint. Normalizing to zero here
        # restarts the file cleanly instead: stream_fetch's mode="wb" path
        # (resume_from == 0) truncates whatever stale bytes are on disk.
        if recorded_done > file.size_bytes:
            recorded_done = 0

        if recorded_done == file.size_bytes:
            # Reconciliation already confirms the on-disk file is exactly
            # the declared size -- nothing left to fetch. SHA-256 content
            # correctness is Task 8's pre-verify job, not this phase's.
            # A zero-byte declared file reconciles to recorded_done == 0
            # == size_bytes WITHOUT ever creating the destination (nothing
            # to stream, nothing to reconcile against) -- create it empty
            # so Task 8's pre-verify hashes a real empty file instead of
            # raising FileNotFoundError on a "complete" file that was
            # never actually written.
            if not destination.exists():
                destination.touch()
            if not entry.get("complete") or entry.get("bytes_done") != recorded_done:
                sidecar["files"][file.path] = {
                    "etag": entry.get("etag"),
                    "last_modified": entry.get("last_modified"),
                    "bytes_done": recorded_done,
                    "complete": True,
                }
                atomic_write_json(sidecar_path, sidecar)
            return

        validators: FetchValidators | None = None
        if entry.get("etag") is not None or entry.get("last_modified") is not None:
            validators = FetchValidators(
                etag=entry.get("etag"), last_modified=entry.get("last_modified")
            )
        resume_from = (
            recorded_done if recorded_done and validators is not None and validators.strong else 0
        )

        url = self._file_url(descriptor, file)
        headers = self._auth_headers(descriptor.upstream_repository)

        def on_chunk(count: int) -> None:
            progress_state.bytes_done += count
            if progress_state.callback is not None:
                progress_state.callback(
                    AcquisitionProgress(
                        phase="fetch",
                        ref=descriptor.reference,
                        file=file.path,
                        bytes_done=progress_state.bytes_done,
                        bytes_total=progress_state.bytes_total,
                    )
                )

        try:
            result, used_resume_from = await self._stream_with_restart(
                url,
                destination,
                client=client,
                max_bytes=file.size_bytes,
                resume_from=resume_from,
                validators=validators,
                headers=headers,
                on_chunk=on_chunk,
            )
        except OSError as exc:
            raise TransferError(
                f"I/O error fetching '{file.path}': {exc}", retryable=True
            ) from exc
        except FetchTooLargeError as exc:
            raise TransferError(
                f"upstream body exceeds declared size for '{file.path}': {exc}",
                retryable=False,
            ) from exc
        except FetchTransportError as exc:
            raise TransferError(
                f"transport error fetching '{file.path}': {exc}", retryable=True
            ) from exc
        except EgressBlockedError as exc:
            # Never a raw exception mid-provision (spec's never-trap rule):
            # unlike _probe_gating's own best-effort HEAD probe (which
            # silently skips a blocked URL -- consent doesn't hinge on it),
            # an egress block during the real fetch means these bytes
            # genuinely cannot be retrieved. Not retryable: it's a policy
            # decision on this URL, not a transient network condition.
            raise TransferError(
                f"egress policy blocked fetching '{file.path}': {exc}", retryable=False
            ) from exc

        total_done = used_resume_from + result.bytes_written
        sidecar["files"][file.path] = {
            "etag": result.validators.etag,
            "last_modified": result.validators.last_modified,
            "bytes_done": total_done,
            "complete": total_done == file.size_bytes,
        }
        atomic_write_json(sidecar_path, sidecar)

    async def _stream_with_restart(
        self,
        url: str,
        destination: Path,
        *,
        client: httpx.AsyncClient,
        max_bytes: int,
        resume_from: int,
        validators: FetchValidators | None,
        headers: Mapping[str, str] | None = None,
        on_chunk: Callable[[int], None],
    ) -> tuple[FetchResult, int]:
        """Call ``stream_fetch``, restarting once from zero on FetchRestartRequired.

        Args:
            url: The file's download URL.
            destination: Where to stream the file's bytes.
            client: The shared ``httpx.AsyncClient``.
            max_bytes: The hard bound on the final file size (the
                descriptor's declared ``size_bytes`` for this file).
            resume_from: Durable bytes already on disk (post-reconciliation).
            validators: Validators the existing bytes were fetched under.
            headers: Extra headers (``_auth_headers``'s ``Authorization``,
                if any) for the request to ``url``'s own origin -- carried
                through to BOTH the initial attempt and a from-zero restart;
                ``stream_fetch`` itself strips them on any cross-origin
                redirect hop regardless of what's passed here.
            on_chunk: Progress callback forwarded to ``stream_fetch``.

        Returns:
            The successful ``FetchResult`` together with the ``resume_from``
            value actually used to obtain it (``0`` if a restart occurred).

        Raises:
            FetchRestartRequired: Raised again if even a from-zero attempt
                is rejected (not expected in practice -- ``stream_fetch``
                only raises this for a nonzero ``resume_from``).
            FetchTooLargeError: Propagated from ``stream_fetch``.
            FetchTransportError: Propagated from ``stream_fetch``.
            OSError: Propagated from a local disk failure (e.g. ENOSPC).
        """

        try:
            result = await stream_fetch(
                url,
                destination,
                client=client,
                max_bytes=max_bytes,
                resume_from=resume_from,
                validators=validators,
                headers=headers,
                trusted_origins=self._trusted_origins,
                on_chunk=on_chunk,
            )
            return result, resume_from
        except FetchRestartRequired:
            if resume_from == 0:
                raise
            if destination.exists():
                destination.unlink()
            result = await stream_fetch(
                url,
                destination,
                client=client,
                max_bytes=max_bytes,
                resume_from=0,
                validators=None,
                headers=headers,
                trusted_origins=self._trusted_origins,
                on_chunk=on_chunk,
            )
            return result, 0

    async def _preverify_artifact(
        self,
        descriptor: ArtifactDescriptor,
        staging_dir: Path,
        progress_state: _ProvisionProgressState,
    ) -> None:
        """Streaming-verify every staged file's SHA-256 before install.

        A mismatch does not fail outright: the offending file is deleted,
        its sidecar entry is reset (so ``_fetch_artifact`` treats it as
        never fetched), and the whole artifact is refetched once via the
        existing ``_fetch_artifact`` path before hashing is retried. Only a
        SECOND mismatch for the same file raises -- this bounds automatic
        recovery to ``MAX_FILE_REFETCHES`` (1) per spec, distinct from
        ``_fetch_artifact``'s own restart-on-``FetchRestartRequired``
        handling (that guards a download that never completed cleanly;
        this guards a download that completed but is wrong).

        Args:
            descriptor: The artifact whose staged files to verify.
            staging_dir: The durable staging directory holding the fetched
                files.
            progress_state: Closure-wide progress accounting to read and
                update as bytes are verified.

        Raises:
            TransferError: A staged file still fails its declared SHA-256
                after ``MAX_FILE_REFETCHES`` refetch attempts. ``retryable``
                is True -- a subsequent ``provision()`` call may still
                succeed (e.g. once the corrupt content upstream is fixed).
        """

        for file in descriptor.files:
            await self._preverify_one_file(descriptor, file, staging_dir, progress_state)

    async def _preverify_one_file(
        self,
        descriptor: ArtifactDescriptor,
        file: ArtifactFile,
        staging_dir: Path,
        progress_state: _ProvisionProgressState,
    ) -> None:
        """Hash one staged file, refetching once via ``_fetch_artifact`` on mismatch.

        Args:
            descriptor: The artifact this file belongs to.
            file: The declared file to verify.
            staging_dir: The durable staging directory holding the fetched
                files.
            progress_state: Closure-wide progress accounting.

        Raises:
            TransferError: See ``_preverify_artifact``.
        """

        destination = staging_dir / file.path
        sidecar_path = _fetch_sidecar_path(staging_dir)
        attempts_used = 0
        loop = asyncio.get_running_loop()
        while True:
            digest = await loop.run_in_executor(
                None,
                functools.partial(
                    self._hash_staged_file, destination, descriptor, file, progress_state, loop
                ),
            )
            if digest == file.sha256:
                return
            if attempts_used >= MAX_FILE_REFETCHES:
                raise TransferError(
                    f"staged file '{file.path}' still fails SHA-256 verification "
                    f"after {MAX_FILE_REFETCHES} refetch(es)",
                    retryable=True,
                )
            attempts_used += 1
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            sidecar = self._load_fetch_sidecar(sidecar_path)
            if sidecar["files"].pop(file.path, None) is not None:
                atomic_write_json(sidecar_path, sidecar)
            await self._fetch_artifact(descriptor, staging_dir, progress_state)

    @staticmethod
    def _hash_staged_file(
        destination: Path,
        descriptor: ArtifactDescriptor,
        file: ArtifactFile,
        progress_state: _ProvisionProgressState,
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        """Stream one staged file's SHA-256, reporting real byte progress.

        Runs entirely off the event loop: ``_preverify_one_file`` invokes
        this via ``loop.run_in_executor``, never directly from an async
        context. Hashing a staged file synchronously inside a coroutine
        would block every other coroutine (including any other artifact's
        fetch/pre-verify progress, and any UI event loop this service
        shares) for as long as a multi-gigabyte file takes to read and
        digest -- there is no ``await`` anywhere in a tight
        ``hashlib.sha256().update()`` loop to yield control back.

        Progress-callback invocation is marshalled back onto ``loop`` via
        ``call_soon_threadsafe`` rather than called directly from this
        (executor) thread: ``progress_state.callback`` is caller-supplied
        and may update UI state that, per this codebase's threading
        convention (``CLAUDE.md``'s "Background Work" pattern -- workers
        call back via ``call_from_thread``, never directly), must only
        ever be touched from the event-loop thread. ``call_soon_threadsafe``
        is the non-Textual-specific equivalent: it schedules the callback
        to run on ``loop`` without blocking this thread's chunk-read loop
        waiting for it. Byte counters on ``progress_state`` are still
        updated directly from this thread -- safe because
        ``_preverify_artifact`` processes one file at a time, so only one
        executor call is ever hashing at once; no concurrent writer exists
        to race against.

        A zero-byte declared file (Task 7 guarantees the destination exists
        even then) reads zero chunks and hashes to the empty digest --
        never a special case, matching how ``_fetch_one_file`` already
        treats a zero-byte file as "real, just empty" rather than absent.

        Args:
            destination: The staged file's path (must exist).
            descriptor: The artifact this file belongs to, for the
                progress event's ``ref``.
            file: The declared file being hashed, for the progress event's
                ``file`` and to size chunk reads.
            progress_state: Closure-wide progress accounting to read and
                update as bytes are hashed.
            loop: The event loop ``progress_state.callback`` must be
                invoked on.

        Returns:
            The lowercase hex SHA-256 digest of the file's current content.
        """

        digest = hashlib.sha256()
        with destination.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                progress_state.preverify_bytes_done += len(chunk)
                if progress_state.callback is not None:
                    event = AcquisitionProgress(
                        phase="pre-verify",
                        ref=descriptor.reference,
                        file=file.path,
                        bytes_done=progress_state.preverify_bytes_done,
                        bytes_total=progress_state.preverify_bytes_total,
                    )
                    loop.call_soon_threadsafe(progress_state.callback, event)
        return digest.hexdigest()

    async def _install_artifact(
        self,
        descriptor: ArtifactDescriptor,
        stage: _ManagedDownloadStage,
    ) -> None:
        """Finalize one pre-verified download stage into the immutable store.

        TASK-1694: retargets this phase at
        ``core._finalize_download_stage`` -- the service-owned payload-
        subtree finalization seam ported from the parallel TASK-595
        implementation (see
        Docs/superpowers/reviews/2026-08-01-task-595-duplicate-
        implementation-reconciliation.md, item 1) -- instead of
        ``core.install(..., consume_source=True)``. ``stage.payload`` (what
        ``_fetch_artifact``/``_preverify_artifact`` wrote and verified) is
        the ONLY subtree the core renames into the immutable destination;
        ``stage.marker``/``stage.state`` (the fetch-state sidecar's home,
        see ``_fetch_sidecar_path``) never enter it, so resume metadata
        cannot end up inside a promoted artifact by construction -- this
        is what makes the OLD sibling-sidecar workaround unnecessary and
        structurally eliminates the "retryable install failure destroys
        the resumable download" bug class: finalization never moves any
        bytes out of ``stage.payload`` until every verification and lease
        has already succeeded, and a failure at any step before that final
        rename leaves ``stage`` -- payload, state, and marker -- completely
        untouched for a later ``provision()`` attempt to resume from.

        On success, ``core._finalize_download_stage`` also retires the
        whole stage operation directory (payload having been promoted,
        plus the marker and state subtree) so a later ``reconcile()``
        staging GC sees nothing left to classify for this artifact; this
        phase does not need its own cleanup step.

        Args:
            descriptor: The artifact to install.
            stage: The service-owned download stage holding the verified
                payload (obtained via ``core._download_stage_for`` earlier
                in ``provision()``'s per-artifact loop).

        Raises:
            TransferError: ``core._finalize_download_stage`` raised
                ``ArtifactIntegrityError``, ``ArtifactConflictError``,
                ``ArtifactPathError``, or ``ArtifactStateError`` --
                wrapped by ``_run_core_call`` with ``retryable`` set
                accordingly (see there). On failure, ``stage`` is left
                completely in place for a resumed ``provision()`` attempt.
        """

        await self._run_core_call(
            "finalize",
            descriptor.reference,
            functools.partial(self._core._finalize_download_stage, descriptor, stage),
        )

    async def _run_core_call(
        self,
        operation: Literal["stage", "finalize", "activate"],
        ref: ArtifactRef,
        func: Callable[[], _CoreCallResult],
    ) -> _CoreCallResult:
        """Run one synchronous core call in the executor, never trapping raw.

        ``core._download_stage_for``, ``core._finalize_download_stage``
        (TASK-1694: replaces the old ``core.install(...,
        consume_source=True)`` call this wrapping originally covered), and
        ``core.activate`` are the core entry points this service reaches
        via a bare executor hop; all three can raise the core's own
        ``ArtifactError`` subclasses (integrity, conflict, path safety, or
        lease/state contention), which would otherwise escape
        ``provision()`` untouched -- breaking the spec's never-trap rule
        that every acquisition-surfaced failure is a typed, retryable-
        flagged ``AcquisitionError``. ``ArtifactIntegrityError``,
        ``ArtifactConflictError``, and ``ArtifactPathError`` are not
        retryable: the same staged content, the same conflicting
        destination, or the same unsafe/invalid path would fail again.
        ``ArtifactStateError`` (and its subclasses, e.g.
        ``ArtifactInUseError`` -- lease-timeout/contention style failures)
        is retryable: a later attempt may find the contended resource free.
        The original error is preserved as ``__cause__`` (``raise ... from
        exc``); the wrapped message names only the operation, artifact id,
        and revision -- never a path.

        Args:
            operation: Which core call this is, for the error message only.
            ref: The artifact reference the call concerns.
            func: A zero-argument callable wrapping the real core call (see
                ``functools.partial`` at both call sites).

        Returns:
            The core call's own return value, unchanged.

        Raises:
            TransferError: ``func`` raised ``ArtifactIntegrityError``,
                ``ArtifactConflictError``, ``ArtifactPathError``, or
                ``ArtifactStateError`` (or a subclass of any of these).
        """

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, func)
        except (ArtifactIntegrityError, ArtifactConflictError, ArtifactPathError) as exc:
            raise TransferError(
                f"{operation} failed for {ref.artifact_id}@{ref.revision}: {exc}",
                retryable=False,
            ) from exc
        except ArtifactStateError as exc:
            raise TransferError(
                f"{operation} failed for {ref.artifact_id}@{ref.revision}: {exc}",
                retryable=True,
            ) from exc

    @staticmethod
    def _emit_indeterminate_progress(
        progress_state: _ProvisionProgressState,
        phase: Literal["verify-install", "activate"],
        ref: ArtifactRef,
    ) -> None:
        """Emit one per-artifact progress event with no byte detail.

        Per spec, ``verify-install`` and ``activate`` events carry no real
        byte accounting (unlike ``fetch``/``pre-verify``) -- they mark a
        phase transition, not a position within a stream.

        Args:
            progress_state: Closure-wide progress accounting, for its
                caller-supplied ``callback`` (may be None).
            phase: Which indeterminate phase this event marks.
            ref: The artifact reference this event is about.
        """

        if progress_state.callback is not None:
            progress_state.callback(
                AcquisitionProgress(
                    phase=phase, ref=ref, file=None, bytes_done=0, bytes_total=0
                )
            )

    def _staged_bytes_for(self, descriptor: ArtifactDescriptor) -> int:
        """Best-effort resumable-byte credit from a fetch-state sidecar.

        A missing or unparseable sidecar reads as zero credit: staged bytes
        are only ever a hint surfaced on the consent screen ("up to N bytes
        already fetched") -- ``provision()`` independently re-validates
        against the server's current validators before trusting any of it.
        TASK-1694: the sidecar now lives under a service-owned download
        stage's ``state/`` subtree (see ``_fetch_sidecar_path``), so this
        looks the stage up via ``core._download_stage_for(descriptor,
        create=False)`` rather than computing a bare
        ``staging/managed/<id>/<rev>/<variant>`` path directly. A stage
        that does not exist, or one whose marker/layout fails the core's
        own validation, reads as zero credit -- same best-effort contract
        as a missing or corrupt sidecar file always had.

        Each declared file's credit is capped by THREE independent limits:
        the sidecar's own recorded ``bytes_done``, the file's declared
        size, and -- the gap this closes -- the file's ACTUAL size on disk
        right now. Capping by the declared size alone (the previous
        behavior) still let a stale or hand-corrupted sidecar claim bytes
        for a file that was truncated, never written, or removed out from
        under it, inflating the credit preflight uses to decide whether
        there's enough free space for the REMAINING download -- a report
        computed against phantom credit can approve an acquisition that
        then runs out of space partway through. A sidecar entry naming a
        file this descriptor doesn't actually declare is ignored outright,
        not just uncapped: crediting it at all would credit bytes for
        something ``provision()`` will never look at.

        Args:
            descriptor: The artifact descriptor whose download stage and
                declared files to inspect.

        Returns:
            The sum of capped per-file credit, or 0 for a missing,
            invalid, or unparseable stage/sidecar.
        """
        try:
            stage = self._core._download_stage_for(descriptor, create=False)
        except ArtifactError:
            return 0
        if stage is None:
            return 0
        sidecar = _fetch_sidecar_path(stage.payload)
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        files = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files, dict):
            return 0
        declared_sizes = {file.path: file.size_bytes for file in descriptor.files}
        total = 0
        for file_path, info in files.items():
            if file_path not in declared_sizes:
                continue
            if not isinstance(info, dict):
                continue
            bytes_done = info.get("bytes_done")
            if not (
                isinstance(bytes_done, int)
                and not isinstance(bytes_done, bool)
                and bytes_done >= 0
            ):
                continue
            try:
                actual_size = (stage.payload / file_path).stat().st_size
            except OSError:
                actual_size = 0
            total += min(bytes_done, actual_size, declared_sizes[file_path])
        return total

    async def _probe_gating(self, targets: Iterable[ArtifactPreflightEntry]) -> list[str]:
        """Bounded HEAD probe per repository; collect 401/403s.

        Anonymous unless a ``credential_resolver`` is configured AND
        resolves a token for the entry's repository, in which case the
        probe carries the same ``Authorization`` header the real fetch
        would use (``_auth_headers``) -- a working credential clears
        gating here exactly as it would clear the real transfer, instead
        of preflight reporting a repository gated that provision() would
        actually be able to reach.

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
                        entry.source_url,
                        timeout=_PREFLIGHT_PROBE_TIMEOUT_SECONDS,
                        headers=self._auth_headers(entry.repository),
                        # Explicit regardless of the client's own default:
                        # client_factory is a public seam, and an injected
                        # client configured to follow redirects would
                        # otherwise both bypass this app's own egress check
                        # for the redirect target (only entry.source_url is
                        # checked above) and carry the bearer token
                        # cross-origin.
                        follow_redirects=False,
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
