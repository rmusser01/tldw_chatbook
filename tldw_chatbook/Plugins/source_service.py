"""Catalog discovery facade on the original plugin storage worker and authority."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import Field, field_validator

from .acquisition import (
    acquire_snapshot,
    cleanup_acquisitions,
    release_snapshot,
    verify_reference,
    verify_snapshot,
)
from .catalogs import (
    CATALOG_PATHS,
    MAX_CATALOG_BYTES,
    CatalogEntry,
    CatalogSnapshot,
    catalog_local_path,
    parse_catalog,
    select_catalog,
)
from .git_source import SourceLocator, parse_source_locator, select_local_source
from .imports import read_selected_file
from .inspection import catalog_overlay_from, inspect_package
from .models import (
    AcquiredSnapshot,
    AcquisitionReference,
    CatalogBinding,
    FrozenModel,
    PackageInspection,
)
from .package_files import canonical_json
from .registry import validate_page

if TYPE_CHECKING:
    from .coordinator import PluginCoordinator
    from .imports import ImportPreview, ImportResult
    from .review import PluginReview
    from .service import PluginService


class SourceBinding(FrozenModel):
    version: Literal[1] = 1
    namespace_id: str = Field(min_length=1, max_length=200)
    source_id: str = Field(min_length=1, max_length=100)
    locator: SourceLocator
    catalog_path: str | None = None
    current: AcquisitionReference | None = None


class SourceSummary(FrozenModel):
    """Untrusted display metadata; actions authenticate the stored source afresh."""

    source_id: str = Field(min_length=1, max_length=100)
    locator: SourceLocator | None
    catalog_path: (
        Literal[
            ".agents/plugins/marketplace.json",
            ".cursor-plugin/marketplace.json",
            ".claude-plugin/marketplace.json",
        ]
        | None
    )
    cached: bool
    fetched_at: str | None = Field(default=None, max_length=64)
    refresh_error: (
        Literal[
            "not_refreshed",
            "cache_unavailable",
            "source_metadata_invalid",
            "catalog_refresh_failed",
        ]
        | None
    ) = None

    @field_validator("locator")
    @classmethod
    def bounded_locator(cls, value):
        if value is not None:
            for field, maximum in (
                ("origin", 4096),
                ("repository", 4096),
                ("ref", 1024),
                ("subdirectory", 4096),
            ):
                text = getattr(value, field)
                if text is not None and len(text.encode()) > maximum:
                    raise ValueError("source_display_limit")
        return value


def _locator(reference):
    return SourceLocator(
        **{name: getattr(reference, name) for name in SourceLocator.model_fields}
    )


def _validate_locator(locator):
    from urllib.parse import urlencode

    locator = SourceLocator.model_validate_json(locator.model_dump_json())
    if locator.transport == "local":
        expected = select_local_source(
            Path(locator.repository), ref=locator.ref, subdirectory=locator.subdirectory
        )
    else:
        selectors = {"ref": locator.ref}
        if locator.subdirectory:
            selectors["subdir"] = locator.subdirectory
        expected = parse_source_locator(locator.url + "#" + urlencode(selectors))
    if locator != expected:
        raise ValueError("source_locator_invalid")
    return locator


def _update_locator(catalog, entry):
    if entry.local_path is None:
        return SourceLocator.model_validate_json(entry.package_locator)
    source = _locator(catalog)
    subdirectory = "/".join(
        part for part in (source.subdirectory, entry.local_path) if part
    )
    return source.model_copy(update={"subdirectory": subdirectory})


def _read_snapshot(reference, source_id, path, *, fetched_at=""):
    root = Path(reference.materialized_identity)
    selected = select_catalog(root, path)
    data = read_selected_file(root, selected)
    return selected, parse_catalog(
        data,
        source_id=source_id,
        catalog_path=selected,
        catalog_commit=reference.commit,
        fetched_at=fetched_at,
    )


def _inspect_entry(root, entry, *, expected: PackageInspection | None = None):
    overlay = catalog_overlay_from(entry.overlays)
    executable = {
        "skills",
        "rules",
        "agents",
        "commands",
        "hooks",
        "mcpServers",
        "variables",
        "apps",
    }
    if expected is not None:
        from .package_files import capture_package
        from .retained_inspection import _reinspect_expected

        # A retained entry supplies its exact overlay to the same closed recipes.
        # This is comparison only; callers first verify signed producer custody.
        selected_overlay = (
            None
            if expected.dialect == "portable"
            and not executable.intersection(overlay or {})
            else overlay
        )
        return _reinspect_expected(
            capture_package(root), expected, catalog_overlay=selected_overlay
        )
    initial = inspect_package(root)
    if initial.dialect == "portable" and not executable.intersection(overlay or {}):
        # Display-only metadata is retained by the artifact/entry binding.
        return initial
    dialect = (
        "openai"
        if initial.dialect == "portable"
        and any(candidate.dialect == "openai" for candidate in initial.candidates)
        else initial.dialect
    )
    return inspect_package(root, dialect=dialect, catalog_overlay=overlay)


def bind_source_inspection(
    raw: PackageInspection,
    *,
    catalog_binding: CatalogBinding | None = None,
    acquisition_reference: AcquisitionReference | None = None,
) -> PackageInspection:
    """Derive stable revision identity from raw interpretation and exact provenance."""
    if catalog_binding is not None and acquisition_reference is not None:
        raise ValueError("ambiguous_acquisition_binding")
    if catalog_binding is None and acquisition_reference is None:
        return raw
    if raw.catalog_binding is not None or raw.acquisition_reference is not None:
        raise ValueError("raw_inspection_required")

    def stable(reference, locator=None):
        return {
            **(locator or _locator(reference)).model_dump(mode="json"),
            **{
                name: getattr(reference, name)
                for name in ("commit", "tree", "content_digest", "source_digest")
            },
        }

    if catalog_binding is not None:
        binding = catalog_binding
        update = SourceLocator.model_validate_json(binding.update_locator_json)
        provenance = {
            "kind": "catalog",
            "source_id": binding.source_id,
            "catalog_path": binding.catalog_path,
            "catalog_digest": binding.catalog_digest,
            "entry_identity": binding.entry_identity,
            "entry_digest": binding.entry_digest,
            "update_locator": update.model_dump(mode="json"),
            "catalog": stable(binding.catalog),
            "package": stable(binding.package, update),
        }
    else:
        provenance = {"kind": "direct", "package": stable(acquisition_reference)}
    digest = hashlib.sha256(
        canonical_json(
            {
                "domain": "plugin-source-revision-v1",
                "effective": raw.effective_digest,
                "provenance": provenance,
            }
        ).encode()
    ).hexdigest()
    return raw.model_copy(
        update={
            "effective_digest": digest if raw.effective_digest is not None else None,
            "catalog_binding": catalog_binding,
            "acquisition_reference": acquisition_reference,
        }
    )


def validate_inspection_source(
    coordinator: PluginCoordinator, inspection: PackageInspection
) -> None:
    """Validate retained provenance with the actual current producer owner."""
    if (
        inspection.catalog_binding is not None
        and inspection.acquisition_reference is not None
    ):
        raise ValueError("ambiguous_acquisition_binding")
    if inspection.catalog_binding is not None:
        validate_catalog_binding(coordinator, inspection)
    elif inspection.acquisition_reference is not None:
        reference = inspection.acquisition_reference
        verify_reference(coordinator.owner, coordinator.authority, reference)
        if (
            reference.content_digest != inspection.content_digest
            or reference.source_digest != inspection.source_digest
        ):
            raise ValueError("acquisition_interpretation_changed")


def validate_catalog_binding(
    coordinator: PluginCoordinator, inspection: PackageInspection
) -> None:
    """Reconstruct the exact reviewed entry using owned retained bytes only."""
    binding = CatalogBinding.model_validate_json(
        inspection.catalog_binding.model_dump_json()
    )
    for reference in (binding.catalog, binding.package):
        verify_reference(coordinator.owner, coordinator.authority, reference)
    _, snapshot = _read_snapshot(
        binding.catalog, binding.source_id, binding.catalog_path
    )
    entry = next(
        (row for row in snapshot.entries if row.identity == binding.entry_identity),
        None,
    )
    if (
        snapshot.digest != binding.catalog_digest
        or entry is None
        or entry.support != "supported"
        or entry.digest != binding.entry_digest
    ):
        raise ValueError("catalog_retained_entry_changed")
    if (
        _update_locator(binding.catalog, entry).model_dump_json()
        != binding.update_locator_json
    ):
        raise ValueError("catalog_update_origin_changed")
    package_locator = _locator(binding.package)
    if entry.local_path is not None:
        expected = select_local_source(
            catalog_local_path(
                Path(binding.catalog.materialized_identity), entry.local_path
            )
        )
    else:
        expected = SourceLocator.model_validate_json(entry.package_locator)
    if package_locator != expected:
        raise ValueError("catalog_package_origin_changed")
    observed = _inspect_entry(
        Path(binding.package.materialized_identity), entry, expected=inspection
    )
    if (
        observed.content_digest != inspection.content_digest
        or observed.effective_digest != inspection.effective_digest
        or observed.overlay_identities != inspection.overlay_identities
    ):
        raise ValueError("catalog_package_interpretation_changed")


@dataclass(frozen=True)
class CandidateDiscardReview:
    """Session-issued exact reference release; never a producer snapshot."""

    reference: AcquisitionReference
    token: str
    expires_at: float
    authority_marker: object
    protected_identity: str


class SourceService:
    """Async IO uses PluginService; synchronous search reads an immutable view."""

    def __init__(self, plugin_service: PluginService):
        self.plugins = plugin_service
        self._snapshots = ()
        self._sources = ()
        self._candidates = {}
        self._candidate_inputs = {}

    @property
    def _coordinator(self) -> PluginCoordinator:
        return self.plugins._coordinator

    def _rows(self):
        return self._coordinator.registry.list_marketplace_sources()

    def _binding(self, row):
        value = json.loads(row["binding_json"])
        self._coordinator.authority.verify_marketplace_source(value, row["binding_mac"])
        binding = SourceBinding.model_validate(value)
        if binding.source_id != row["source_id"] or binding.catalog_path not in (
            None,
            *CATALOG_PATHS,
        ):
            raise ValueError("source_binding_invalid")
        return binding

    def _row(self, source_id):
        row = next((row for row in self._rows() if row["source_id"] == source_id), None)
        if row is None:
            raise ValueError("source_removed")
        return row

    def _write(self, binding, snapshot=None, error=None, *, expected_binding=None):
        encoded = snapshot.model_dump_json() if snapshot is not None else None
        if encoded is not None and len(encoded.encode()) > MAX_CATALOG_BYTES:
            raise ValueError("catalog_cache_limit")
        payload = binding.model_dump(mode="json")
        mac = self._coordinator.authority.sign_marketplace_source(payload)
        self._coordinator.registry.write_marketplace_source(
            binding.source_id,
            canonical_json(payload),
            mac,
            encoded,
            error,
            expected_binding=expected_binding,
        )
        self._load_cached()

    def _load_cached(self):
        snapshots = []
        summaries = []
        for row in self._rows():
            snapshot = None
            error = "cache_unavailable" if row["snapshot_present"] else "not_refreshed"
            try:
                snapshot = CatalogSnapshot.model_validate_json(row["snapshot_json"])
                if snapshot.source_id != row["source_id"]:
                    raise ValueError("source_cache_mismatch")
                error = "catalog_refresh_failed" if row["refresh_error"] else None
                snapshot = snapshot.model_copy(update={"refresh_error": error})
                snapshots.append(snapshot)
            except (TypeError, ValueError):
                snapshot = None
            try:
                display = SourceBinding.model_validate_json(row["binding_json"])
                summary = SourceSummary(
                    source_id=row["source_id"],
                    locator=display.locator,
                    catalog_path=display.catalog_path,
                    cached=bool(snapshot and snapshot.digest),
                    fetched_at=(
                        snapshot.fetched_at if snapshot and snapshot.digest else None
                    ),
                    refresh_error=error,
                )
            except (TypeError, ValueError):
                summary = SourceSummary(
                    source_id=row["source_id"],
                    locator=None,
                    catalog_path=None,
                    cached=bool(snapshot and snapshot.digest),
                    refresh_error="source_metadata_invalid",
                )
            summaries.append(summary)
        self._snapshots = tuple(snapshots)
        self._sources = tuple(summaries)
        return self._snapshots

    async def load_cached(self) -> tuple[CatalogSnapshot, ...]:
        """Load bounded display rows on the storage worker, including while locked."""
        return await self.plugins._call(self._load_cached)

    def list_sources(self) -> tuple[SourceSummary, ...]:
        """List all configured source display rows without disk or authentication."""
        return self._sources

    def search(
        self, query: str, *, offset: int, limit: int = 50
    ) -> tuple[CatalogEntry, ...]:
        """Search cached display metadata without disk or network work."""
        validate_page(limit, offset)
        if not isinstance(query, str) or len(query) > 1024:
            raise ValueError("catalog_query_invalid")
        query = query.casefold()
        rows = (
            entry
            for snapshot in self._snapshots
            for entry in snapshot.entries
            if query in entry.name.casefold()
        )
        from itertools import islice

        return tuple(islice(rows, offset, offset + limit))

    async def add(
        self, locator: SourceLocator, *, catalog_path: str | None = None
    ) -> str:
        """Record a newly selected immutable source without fetching or installing."""

        def operation():
            if catalog_path not in (None, *CATALOG_PATHS):
                raise ValueError("catalog_selection_invalid")
            selected = _validate_locator(locator)
            binding = SourceBinding(
                namespace_id=self._coordinator.authority.namespace_id(),
                source_id=uuid4().hex,
                locator=selected,
                catalog_path=catalog_path,
            )
            self._write(binding)
            return binding.source_id

        return await self.plugins._call(operation)

    async def _acquire(self, locator, purpose):
        owner, authority = self._coordinator.owner, self._coordinator.authority
        identity = uuid4().hex
        result = await acquire_snapshot(
            locator,
            identity,
            owner.acquisition_destination(identity),
            owner=owner,
            authority=authority,
            _purpose=purpose,
        )
        return result, verify_snapshot(owner, authority, result)

    async def refresh(self, source_id: str) -> CatalogSnapshot:
        """Fetch the authenticated source; retain last successful discovery on failure."""

        async def operation():
            row = self._row(source_id)
            binding = self._binding(row)
            result = None
            try:
                result, reference = await self._acquire(
                    binding.locator, "catalog-cache"
                )
                path, snapshot = _read_snapshot(
                    reference,
                    source_id,
                    binding.catalog_path,
                    fetched_at=datetime.now(UTC).isoformat(),
                )
                updated = binding.model_copy(
                    update={"current": reference, "catalog_path": path}
                )
                error = None
            except (ValueError, OSError):
                if result is not None:
                    release_snapshot(
                        self._coordinator.owner, self._coordinator.authority, result
                    )
                    result = None
                try:
                    previous = CatalogSnapshot.model_validate_json(row["snapshot_json"])
                    if previous.source_id != source_id:
                        raise ValueError("source_cache_mismatch")
                except (TypeError, ValueError):
                    previous = CatalogSnapshot(
                        source_id=source_id,
                        catalog_commit=None,
                        digest="",
                        fetched_at="",
                        entries=(),
                    )
                error = "catalog_refresh_failed"
                snapshot = previous.model_copy(update={"refresh_error": error})
                updated = binding
            try:
                self._write(
                    updated,
                    snapshot,
                    error,
                    expected_binding=(row["binding_json"], row["binding_mac"]),
                )
            except ValueError as exc:
                if result is not None and exc.args == ("source_changed",):
                    release_snapshot(
                        self._coordinator.owner, self._coordinator.authority, result
                    )
                raise
            return snapshot

        return await self.plugins._call(operation)

    async def _candidate(
        self,
        binding,
        entry_identity,
        *,
        selected_locator=None,
        catalog=None,
        installation_id=None,
    ):
        reference = catalog or binding.current
        if reference is None:
            raise ValueError("catalog_refresh_required")
        verify_reference(
            self._coordinator.owner, self._coordinator.authority, reference
        )
        path, snapshot = _read_snapshot(
            reference, binding.source_id, binding.catalog_path
        )
        entry = next(
            (row for row in snapshot.entries if row.identity == entry_identity), None
        )
        if entry is None or entry.support != "supported":
            raise ValueError("catalog_entry_unavailable")
        if entry.local_path is not None:
            locator = select_local_source(
                catalog_local_path(
                    Path(reference.materialized_identity), entry.local_path
                )
            )
        else:
            locator = SourceLocator.model_validate_json(entry.package_locator)
            if selected_locator is None or selected_locator != locator:
                raise ValueError("catalog_origin_review_required")
        result, package = await self._acquire(locator, "package-candidate")
        try:
            inspection = _inspect_entry(Path(package.materialized_identity), entry)
            retained = CatalogBinding(
                source_id=binding.source_id,
                catalog=reference,
                package=package,
                catalog_path=path,
                catalog_digest=snapshot.digest,
                entry_identity=entry.identity,
                entry_digest=entry.digest,
                update_locator_json=_update_locator(reference, entry).model_dump_json(),
            )
            inspection = bind_source_inspection(
                inspection, catalog_binding=retained
            ).model_copy(
                update={"materialized_identity": package.materialized_identity}
            )
            validate_catalog_binding(self._coordinator, inspection)
            self._remember(inspection, installation_id=installation_id)
            return inspection
        except BaseException:
            release_snapshot(
                self._coordinator.owner, self._coordinator.authority, result
            )
            raise

    async def review_entry(
        self,
        source_id: str,
        entry_identity: str,
        *,
        selection: tuple[str, ...],
        workspace_id: str | None,
        selected_locator: SourceLocator | None = None,
        enable_in_workspace: bool = False,
    ) -> PluginReview:
        """Review the exact selected entry and explicit remote locator for installation."""

        async def operation():
            binding = self._binding(self._row(source_id))
            inspection = await self._candidate(
                binding, entry_identity, selected_locator=selected_locator
            )
            try:
                self.plugins._admission._workspace(workspace_id)
                return self._coordinator.review(
                    inspection,
                    selection=selection,
                    workspace_id=workspace_id,
                    enable_in_workspace=enable_in_workspace,
                )
            except BaseException:
                self._discard_candidate(inspection)
                raise

        return await self.plugins._call(operation)

    def _remember(self, inspection, *, installation_id=None):
        from .review import inspection_identity

        references = (
            (inspection.catalog_binding.catalog, inspection.catalog_binding.package)
            if inspection.catalog_binding
            else (inspection.acquisition_reference,)
        )
        self._candidates[references[-1].operation_id] = references
        self._candidate_inputs[references[-1].operation_id] = (
            inspection_identity(inspection),
            installation_id,
        )

    async def review_acquired(
        self,
        snapshot: AcquiredSnapshot,
        *,
        selection: tuple[str, ...],
        workspace_id: str | None,
        dialect: str | None = None,
        enable_in_workspace: bool = False,
    ) -> PluginReview:
        """Authenticate the complete direct producer result before ordinary review."""

        def operation():
            reference = verify_snapshot(
                self._coordinator.owner, self._coordinator.authority, snapshot
            )
            inspection = bind_source_inspection(
                (
                    self._selected_acquired_inspection(snapshot, reference, dialect)
                    if dialect is not None
                    else snapshot.inspection
                ),
                acquisition_reference=reference,
            )
            validate_inspection_source(self._coordinator, inspection)
            self.plugins._admission._workspace(workspace_id)
            self._remember(inspection)
            return self._coordinator.review(
                inspection,
                selection=selection,
                workspace_id=workspace_id,
                enable_in_workspace=enable_in_workspace,
            )

        return await self.plugins._call(operation)

    @staticmethod
    def _selected_acquired_inspection(snapshot, reference, dialect):
        """Choose format over verified bytes, retaining authenticated physical provenance."""
        observed = inspect_package(
            Path(reference.materialized_identity), dialect=dialect
        )
        original = snapshot.inspection
        return observed.model_copy(
            update={
                "source_identity": original.source_identity,
                "materialized_identity": reference.materialized_identity,
                "source_digest": original.source_digest,
                "link_targets": dict(original.link_targets),
            }
        )

    async def inspect_acquired(self, snapshot: AcquiredSnapshot, *, dialect=None):
        """Authenticate the unchanged producer before choosing an interpretation."""

        def operation():
            reference = verify_snapshot(
                self._coordinator.owner, self._coordinator.authority, snapshot
            )
            inspection = self._selected_acquired_inspection(
                snapshot, reference, dialect
            )
            return bind_source_inspection(inspection, acquisition_reference=reference)

        return await self.plugins._call(operation)

    async def review_candidate(
        self,
        candidate: PackageInspection,
        *,
        selection,
        workspace_id,
        enable_in_workspace=False,
    ):
        """Reissue exact retained candidate inputs, without reacquiring a source."""

        def operation():
            from .review import inspection_identity

            reference = (
                candidate.catalog_binding.package
                if candidate.catalog_binding
                else candidate.acquisition_reference
            )
            if reference is None or self._candidate_inputs.get(
                reference.operation_id
            ) != (inspection_identity(candidate), None):
                raise ValueError("candidate_review_required")
            validate_inspection_source(self._coordinator, candidate)
            self.plugins._admission._workspace(workspace_id)
            return self._coordinator.review(
                candidate,
                selection=selection,
                workspace_id=workspace_id,
                enable_in_workspace=enable_in_workspace,
            )

        return await self.plugins._call(operation)

    async def list_persisted_candidates(self, *, limit=50, offset=0):
        """List authenticated custody; unresolved inventory protects every entry."""
        validate_page(limit, offset)

        def operation():
            from .acquisition import _inventory, _qualify

            owner, authority = self._coordinator.owner, self._coordinator.authority
            records = _inventory(owner, authority)
            try:
                protected, roots = self._protected()
                unresolved = None
            except (ValueError, OSError):
                protected, roots, unresolved = (
                    set(),
                    set(),
                    "acquisition_dependencies_unknown",
                )
            rows = []
            for record in sorted(records, key=lambda value: value["operation_id"])[
                offset : offset + limit
            ]:
                reason = unresolved
                reference = (
                    AcquisitionReference.model_validate(record["result_reference"])
                    if record.get("result_reference")
                    else None
                )
                if record["phase"] != "terminal" or reference is None:
                    reason = reason or "acquisition_custody_unsettled"
                elif not record["recovery_referenced"]:
                    reason = reason or "reference_already_released"
                elif reference.operation_id in protected or self._contains_locator(
                    reference, roots
                ):
                    reason = reason or "acquisition_reference_in_use"
                else:
                    try:
                        _qualify(owner, record)
                        verify_reference(owner, authority, reference)
                    except (ValueError, OSError):
                        reason = "acquisition_custody_recovery_required"
                rows.append(
                    {
                        "operation_id": record["operation_id"],
                        "reference": reference,
                        "phase": record["phase"],
                        "reason": reason,
                    }
                )
            return tuple(rows)

        return await self.plugins._call(operation)

    def _reference_discard_identity(self, reference):
        from .acquisition import _inventory, _qualify

        owner, authority = self._coordinator.owner, self._coordinator.authority
        protected, roots = self._protected()
        if reference.operation_id in protected or self._contains_locator(
            reference, roots
        ):
            raise ValueError("acquisition_reference_in_use")
        verify_reference(owner, authority, reference)
        record = next(
            row
            for row in _inventory(owner, authority)
            if row["operation_id"] == reference.operation_id
        )
        _qualify(owner, record)
        return canonical_json(
            {"record": record, "protected": sorted(protected), "roots": sorted(roots)}
        )

    async def review_candidate_discard(self, reference: AcquisitionReference):
        def operation():
            cache = getattr(self, "_discard_reviews", {})
            cache = {
                key: value
                for key, value in cache.items()
                if value.expires_at > time.monotonic()
            }
            if len(cache) >= 1000:
                raise ValueError("candidate_review_limit")
            review = CandidateDiscardReview(
                reference,
                uuid4().hex,
                time.monotonic() + 900,
                self._coordinator.authority.load_marker(),
                self._reference_discard_identity(reference),
            )
            cache[review.token] = review
            self._discard_reviews = cache
            return review

        return await self.plugins._call(operation)

    async def discard_candidate_reference(self, review: CandidateDiscardReview):
        def operation():
            if (
                not isinstance(review, CandidateDiscardReview)
                or getattr(self, "_discard_reviews", {}).get(review.token) is not review
                or time.monotonic() >= review.expires_at
                or self._coordinator.authority.load_marker() != review.authority_marker
                or self._reference_discard_identity(review.reference)
                != review.protected_identity
            ):
                raise ValueError("candidate_discard_review_stale")
            release_snapshot(
                self._coordinator.owner, self._coordinator.authority, review.reference
            )
            self._discard_reviews.pop(review.token)

        await self.plugins._call(operation)

    async def check_updates(
        self, installation_id: str
    ) -> tuple[PackageInspection, ...]:
        """Acquire candidates from retained original provenance without applying them."""

        async def operation():
            state = self._coordinator.published_snapshot()
            installed = next(
                row
                for row in state["installations"]
                if row["installation_id"] == installation_id
            )
            revision = next(
                row
                for row in state["revisions"]
                if row["installation_id"] == installation_id
                and row["revision_digest"] == installed["revision_digest"]
            )
            if revision.get("acquisition_reference"):
                retained_direct = AcquisitionReference.model_validate(
                    revision["acquisition_reference"]
                )
                verify_reference(
                    self._coordinator.owner,
                    self._coordinator.authority,
                    retained_direct,
                )
                result, reference = await self._acquire(
                    _locator(retained_direct), "package-candidate"
                )
                inspection = bind_source_inspection(
                    result.inspection, acquisition_reference=reference
                )
                validate_inspection_source(self._coordinator, inspection)
                self._remember(inspection, installation_id=installation_id)
                return (inspection,)
            if not revision.get("catalog_binding"):
                return ()
            retained = CatalogBinding.model_validate(revision["catalog_binding"])
            self._binding(self._row(retained.source_id))
            source = _locator(retained.catalog)
            result, reference = await self._acquire(source, "catalog-candidate")
            try:
                binding = SourceBinding(
                    namespace_id=self._coordinator.authority.namespace_id(),
                    source_id=retained.source_id,
                    locator=source,
                    catalog_path=retained.catalog_path,
                    current=reference,
                )
                _, snapshot = _read_snapshot(
                    reference, retained.source_id, retained.catalog_path
                )
                entry = next(
                    (
                        row
                        for row in snapshot.entries
                        if row.identity == retained.entry_identity
                    ),
                    None,
                )
                if (
                    entry is None
                    or entry.support != "supported"
                    or _update_locator(reference, entry).model_dump_json()
                    != retained.update_locator_json
                ):
                    raise ValueError("catalog_update_origin_changed")
                selected = (
                    SourceLocator.model_validate_json(retained.update_locator_json)
                    if entry.local_path is None
                    else None
                )
                return (
                    await self._candidate(
                        binding,
                        entry.identity,
                        selected_locator=selected,
                        installation_id=installation_id,
                    ),
                )
            except BaseException:
                release_snapshot(
                    self._coordinator.owner, self._coordinator.authority, result
                )
                raise

        return await self.plugins._call(operation)

    async def review_update(
        self, installation_id: str, inspection: PackageInspection
    ) -> PluginReview:
        """Review an unchanged candidate issued by this facade against current state."""

        def operation():
            from .review import inspection_identity

            reference = (
                inspection.catalog_binding.package
                if inspection.catalog_binding
                else inspection.acquisition_reference
            )
            if reference is None or self._candidate_inputs.get(
                reference.operation_id
            ) != (inspection_identity(inspection), installation_id):
                raise ValueError("candidate_review_required")
            validate_inspection_source(self._coordinator, inspection)
            return self._coordinator.review_revision(installation_id, inspection)

        return await self.plugins._call(operation)

    async def review_rebind(
        self,
        installation_id: str,
        source_id: str,
        entry_identity: str,
        *,
        selected_locator: SourceLocator | None = None,
    ) -> PluginReview:
        """Review an explicitly selected replacement source as a new revision."""

        async def operation():
            binding = self._binding(self._row(source_id))
            inspection = await self._candidate(
                binding, entry_identity, selected_locator=selected_locator
            )
            try:
                return self._coordinator.review_revision(installation_id, inspection)
            except BaseException:
                self._discard_candidate(inspection)
                raise

        return await self.plugins._call(operation)

    async def remove(self, source_id: str) -> None:
        """Remove discovery state while preserving installed provenance and bytes."""

        def operation():
            self._binding(self._row(source_id))
            self._coordinator.registry.remove_marketplace_source(source_id)
            self._load_cached()

        await self.plugins._call(operation)

    def _protected(self):
        import os

        from .acquisition import MAX_CUSTODY_RECORDS, _inventory, _qualify
        from .package_files import _directory_fd
        from .retention import transition_inventory

        owner, authority = self._coordinator.owner, self._coordinator.authority
        records = _inventory(owner, authority)
        known = {record["name"] for record in records}
        anchor = owner.root / "staging"
        if anchor.exists():
            with _directory_fd(anchor) as fd, os.scandir(fd) as entries:
                for index, entry in enumerate(entries):
                    if index >= MAX_CUSTODY_RECORDS or entry.name not in known:
                        raise ValueError("acquisition_dependencies_unknown")
        references = {}
        roots = set()
        for record in records:
            if record["phase"] == "pending":
                raise ValueError("acquisition_dependencies_unsettled")
            _qualify(
                owner, record, leaf=record["phase"] != "cleaned", artifact_only=True
            )
            if record["recovery_referenced"]:
                reference = AcquisitionReference.model_validate(
                    record.get("result_reference")
                )
                if (
                    reference.operation_id != record["operation_id"]
                    or reference.acquisition_nonce != record["nonce"]
                    or reference.materialized_identity
                    != str(owner.acquisition_destination(record["operation_id"]))
                ):
                    raise ValueError("acquisition_dependencies_changed")
                references[reference.operation_id] = reference
                if reference.transport == "local":
                    roots.add(reference.repository)
        protected = set()

        def protect(reference):
            if references.get(reference.operation_id) != reference:
                raise ValueError("acquisition_dependencies_missing")
            protected.add(reference.operation_id)

        for row in self._rows():
            binding = self._binding(row)
            if binding.current:
                protect(binding.current)
            if binding.locator.transport == "local":
                roots.add(binding.locator.repository)
        states = [authority.verify_current()] + [
            item.snapshot for item in transition_inventory(authority)
        ]
        from .data_cleanup import RootReview
        from .review import PluginReview

        reviews = tuple(self._coordinator._reviews.values())
        for review in reviews:
            if isinstance(review, RootReview):
                # Root reviews retain their original authority and physical custody.
                states.append(json.loads(review.authority_json))
                roots.update(row["path"] for row in review.targets)
            elif not isinstance(review, PluginReview):
                # Domain refusal is shared with other unresolved custody states.
                raise ValueError("acquisition_dependencies_unknown")  # noqa: TRY004
        for state in states:
            for row in state["revisions"]:
                if row.get("acquisition_reference"):
                    protect(
                        AcquisitionReference.model_validate(
                            row["acquisition_reference"]
                        )
                    )
                if row.get("catalog_binding"):
                    binding = CatalogBinding.model_validate(row["catalog_binding"])
                    protect(binding.catalog)
                    protect(binding.package)
        for review in reviews:
            if isinstance(review, RootReview):
                continue
            inspection = review.inspection
            if inspection.acquisition_reference:
                protect(inspection.acquisition_reference)
            if inspection.catalog_binding:
                protect(inspection.catalog_binding.catalog)
                protect(inspection.catalog_binding.package)
        for candidate in self._candidates.values():
            for reference in candidate:
                protect(reference)
        return protected, roots

    @staticmethod
    def _contains_locator(reference, roots):
        return any(
            Path(root).is_relative_to(reference.materialized_identity) for root in roots
        )

    async def cleanup(self) -> tuple[str, ...]:
        """Release only unshared I4 bases, then use the original age/custody cleanup."""

        def operation():
            from .acquisition import _inventory, _qualify, _save

            protected, source_roots = self._protected()
            owner, authority = self._coordinator.owner, self._coordinator.authority
            for record in _inventory(owner, authority):
                if (
                    record.get("purpose") not in {"catalog-cache", "import-copy"}
                    or record["phase"] != "terminal"
                    or not record["recovery_referenced"]
                    or record["operation_id"] in protected
                ):
                    continue
                reference = record.get("result_reference")
                if reference and self._contains_locator(
                    AcquisitionReference.model_validate(reference), source_roots
                ):
                    continue
                _qualify(owner, record)
                record["recovery_referenced"] = False
                _save(owner, authority, record)
            return cleanup_acquisitions(owner, authority)

        return await self.plugins._call(operation)

    async def commit_import(self, preview: ImportPreview) -> ImportResult:
        """Copy exactly the previewed allowlist; imports never install or connect."""

        async def operation():
            import tempfile

            from .imports import (
                ImportFailure,
                ImportResult,
                _capture_digest,
                _catalog_digest,
                inspect_app_import,
            )
            from .package_files import MAX_PACKAGE_BYTES, capture_package
            from .retention import require_capacity

            root = Path(preview.selected_root)
            if inspect_app_import(preview.app, root) != preview:
                raise ValueError("import_preview_changed")
            sources, packages, exclusions = [], [], list(preview.exclusions)
            failures = list(preview.failures)
            for candidate in preview.candidates:
                if candidate.kind == "package":
                    result, reference = await self._acquire(
                        select_local_source(root), "import-copy"
                    )
                    if result.inspection.content_digest != candidate.digest:
                        raise ValueError("import_preview_changed")
                    result, reference = await self._acquire(
                        select_local_source(Path(reference.materialized_identity)),
                        "package-candidate",
                    )
                    packages.append(result)
                    self._candidates[reference.operation_id] = (reference,)
                    continue
                if candidate.kind != "catalog":
                    exclusions.append("import_source_unsupported")
                    continue
                try:
                    data = read_selected_file(root, candidate.catalog_path)
                    snapshot = parse_catalog(
                        data, source_id="import", catalog_path=candidate.catalog_path
                    )
                    files = {candidate.catalog_path: (data, 0)}
                    total = len(data)
                    captures = {}
                    for entry in snapshot.entries:
                        if (
                            entry.support != "supported"
                            or entry.local_path is None
                            or entry.local_path in captures
                        ):
                            continue
                        capture = capture_package(
                            catalog_local_path(root, entry.local_path)
                        )
                        captures[entry.local_path] = capture
                        for name, member in capture.files.items():
                            target = entry.local_path + "/" + name
                            if target not in files:
                                total += len(member.data)
                                if total > MAX_PACKAGE_BYTES:
                                    raise ValueError("import_bytes_limit")
                                files[target] = (member.data, member.executable_mode)
                        await asyncio.sleep(0)
                except (OSError, ValueError):
                    if inspect_app_import(preview.app, root) != preview:
                        raise ValueError("import_preview_changed") from None
                    failures.append(
                        ImportFailure(
                            path=candidate.catalog_path, code="import_item_unavailable"
                        )
                    )
                    continue
                try:
                    package_digests = {
                        path: _capture_digest(capture)
                        for path, capture in captures.items()
                    }
                except ValueError:
                    raise ValueError("import_preview_changed") from None
                if (
                    tuple(sorted(package_digests)) != candidate.package_paths
                    or _catalog_digest(snapshot, package_digests) != candidate.digest
                    or inspect_app_import(preview.app, root) != preview
                ):
                    raise ValueError("import_preview_changed")
                require_capacity(self._coordinator.owner.root, total)
                with tempfile.TemporaryDirectory(
                    prefix="chatbook-plugin-import-"
                ) as temporary:
                    copied_root = Path(temporary).resolve()
                    try:
                        for name, (body, executable) in files.items():
                            target = copied_root / name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with target.open("xb") as stream:
                                for offset in range(0, len(body), 65536):
                                    stream.write(body[offset : offset + 65536])
                                    await asyncio.sleep(0)
                            target.chmod(0o400 | executable)
                    except OSError:
                        failures.append(
                            ImportFailure(
                                path=candidate.catalog_path, code="import_copy_failed"
                            )
                        )
                        continue
                    try:
                        _, reference = await self._acquire(
                            select_local_source(copied_root), "import-copy"
                        )
                    finally:
                        # I3 may retain a draining producer after awaiter cancellation.
                        drained = asyncio.gather(
                            *tuple(self._coordinator.owner.acquisition_tasks),
                            return_exceptions=True,
                        )
                        while not drained.done():
                            try:
                                await asyncio.shield(drained)
                            except asyncio.CancelledError:
                                continue
                source_id = uuid4().hex
                locator = select_local_source(Path(reference.materialized_identity))
                _, current = await self._acquire(locator, "catalog-cache")
                binding = SourceBinding(
                    namespace_id=self._coordinator.authority.namespace_id(),
                    source_id=source_id,
                    locator=locator,
                    catalog_path=candidate.catalog_path,
                    current=current,
                )
                _, copied = _read_snapshot(
                    current,
                    source_id,
                    candidate.catalog_path,
                    fetched_at=datetime.now(UTC).isoformat(),
                )
                self._write(binding, copied)
                sources.append(source_id)
            return ImportResult(
                source_ids=tuple(sources),
                packages=tuple(packages),
                exclusions=tuple(sorted(set(exclusions))),
                failures=tuple(failures),
            )

        return await self.plugins._call(operation)

    def _discard_candidate(self, candidate) -> None:
        from .data_cleanup import RootReview
        from .models import AcquiredSnapshot
        from .review import inspection_identity

        if isinstance(candidate, AcquiredSnapshot):
            reference = verify_snapshot(
                self._coordinator.owner, self._coordinator.authority, candidate
            )
            references = (reference,)
        else:
            reference = (
                candidate.catalog_binding.package
                if candidate.catalog_binding
                else candidate.acquisition_reference
            )
            if reference is None or self._candidate_inputs.get(
                reference.operation_id, (None, None)
            )[0] != inspection_identity(candidate):
                raise ValueError("candidate_review_required")
            references = self._candidates[reference.operation_id]
        # Validate the complete mixed registry before invalidating any draft.
        self._protected()
        self._candidates.pop(reference.operation_id, None)
        self._candidate_inputs.pop(reference.operation_id, None)
        for token, review in tuple(self._coordinator._reviews.items()):
            if isinstance(review, RootReview):
                continue
            owned = (
                review.inspection.catalog_binding.package
                if review.inspection.catalog_binding
                else review.inspection.acquisition_reference
            )
            if owned and owned.operation_id == reference.operation_id:
                self._coordinator._reviews.pop(token)
        for owned in reversed(references):
            protected, roots = self._protected()
            if owned.operation_id not in protected and not self._contains_locator(
                owned, roots
            ):
                verify_reference(
                    self._coordinator.owner, self._coordinator.authority, owned
                )
                release_snapshot(
                    self._coordinator.owner, self._coordinator.authority, owned
                )

    async def discard_candidate(
        self, candidate: AcquiredSnapshot | PackageInspection
    ) -> None:
        """Invalidate candidate reviews; retain installed or shared dependencies."""
        await self.plugins._call(lambda: self._discard_candidate(candidate))
