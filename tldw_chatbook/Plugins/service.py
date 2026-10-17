"""Lazy app-owned plugin storage worker and typed Console facade."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .admission import LivePluginFences, PluginAdmission, PluginUnavailable
from .review import OperationReceipt, PluginReview
from .revocation import RevocationRequest, RevocationTarget

if TYPE_CHECKING:
    from .data_cleanup import DataRootRef, RootReview
    from .git_source import SourceLocator
    from .models import AcquiredSnapshot, InspectionEvidence


@dataclass(frozen=True)
class PluginRunOwnership:
    """Exact host-owned run association consumed by revocation and drain."""

    installation_id: str
    workspace_id: str | None
    revision_digest: str
    pending_id: str
    run_id: str
    handle_id: str | None
    lease_token: str
    cancel: Callable[[], object]
    completed: threading.Event
    turn_id: str | None = None
    parent_run_id: str | None = None
    generations: tuple[tuple[str, str, int], ...] = ()
    component_ceiling: tuple[str, ...] | None = None



@dataclass(frozen=True)
class ManagementComponent:
    """Display metadata without executable definitions or host mappings."""

    component_id: str
    kind: str
    support: str
    evidence: tuple[InspectionEvidence, ...]
    dependencies: tuple[str, ...]
    blockers: tuple[str, ...]

@dataclass(frozen=True)
class ManagementInstallation:
    """Captured display state; never an admission or mutation capability."""

    installation_id: str
    revision_digest: str
    alias: str
    workspace_id: str | None
    activation_default: bool
    override: str
    effective_active: bool
    trusted: bool
    selected_components: tuple[str, ...]
    ready_components: tuple[str, ...]
    components: tuple[ManagementComponent, ...]
    readiness_error: str | None
    source_id: str | None
    origin: str
    ref: str | None
    commit: str | None
    dialect: str | None


class PluginService:
    """Own one persistent worker; construction performs no IO or keyring work."""

    def __init__(
        self,
        profile_root: Path,
        *,
        workspace_lookup: Callable[[str], object | None],
        marker_store_factory: Callable[[Path], Any] | None = None,
        accept_reduced_protection: bool = False,
        mcp_mapping_owner=None,
        model_mapping_owner=None,
    ) -> None:
        self.profile_root = Path(profile_root)
        self.workspace_lookup = workspace_lookup
        self._marker_factory = marker_store_factory
        self._reduced = accept_reduced_protection
        self._mcp_mapping_owner = mcp_mapping_owner
        self._model_mapping_owner = model_mapping_owner
        self.fences = LivePluginFences()
        from .revisions import RevisionDrain

        self.revision_drain = RevisionDrain(
            self.fences,
            lambda identity: self._call(
                lambda: self._coordinator._drain_inventory(identity)
            ),
        )
        self._lock = self.fences.live_lock
        self._thread = None
        self._closed = False
        self._removal_reviews = {}
        self._removal_requests = {}
        self._closing = False
        self._ordinary_calls = set()
        self._ready = Future()
        self._catalog = {}
        self._details = ()
        self._published_revisions = {}
        self._published_generations = {}
        self._snapshots = {}
        self._admitted = {}
        self._admission_keys = {}
        self._retired = set()
        self._pending_custody = {}
        self._root_runs = {}
        self._bound_runs = {}
        self._terminal_runs = set()
        self._actor_components = {}
        self._live_runs = self.fences.runs
        self._revision_refresh_tasks: set[asyncio.Task] = set()

    def _require_open(self, *, terminal=False):
        with self._lock:
            if self._closed:
                raise PluginUnavailable("plugin_service_closed")
            if self._closing and not terminal:
                raise PluginUnavailable("plugin_service_closing")

    def _start(self, *, terminal=False):
        with self._lock:
            self._require_open(terminal=terminal)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._worker, name="chatbook-plugin-storage", daemon=True
                )
                self._thread.start()
        return self._ready

    def _worker(self):
        loop = owner = registry = None
        try:
            from tldw_chatbook.DB.base_db import operation_owned_connection
            from tldw_chatbook.Skills_Interop.local_skills_service import (
                default_local_skills_store_dir,
            )

            from .authority_store import (
                KeyringPluginMarkerStore,
                PluginAuthorityStore,
                default_plugin_authority_dir,
            )
            from .coordinator import PluginCoordinator
            from .registry import PluginRegistry
            from .runtime_owner import PluginRuntimeOwner

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            owner = PluginRuntimeOwner(self.profile_root / "plugins")
            if not owner.try_acquire():
                raise PluginUnavailable("plugin_owner_busy")
            registry = PluginRegistry(owner.root / "registry.sqlite3", owner=owner)
            trust_root = default_plugin_authority_dir(
                default_local_skills_store_dir(self.profile_root)
            )
            marker = (self._marker_factory or KeyringPluginMarkerStore)(trust_root)
            authority = PluginAuthorityStore(
                trust_root, marker, accept_reduced_protection=self._reduced
            )
            self._coordinator = PluginCoordinator(
                registry,
                authority,
                owner,
                fences=self.fences,
                mcp_mapping_owner=self._mcp_mapping_owner,
                model_mapping_owner=self._model_mapping_owner,
            )
            self._coordinator.on_data_change = self._refresh
            self._admission = PluginAdmission(
                self._coordinator, self.workspace_lookup, fences=self.fences
            )
            self._loop = loop
            self._ready.set_result(loop)
            workspace_db = getattr(
                getattr(self.workspace_lookup, "__self__", None), "db", None
            )
            with operation_owned_connection(workspace_db):
                loop.run_forever()
        # Publish all startup failures to waiters before releasing the owner.
        except BaseException as error:  # noqa: BLE001
            if not self._ready.done():
                self._ready.set_exception(error)
        finally:
            if registry is not None:
                registry.close()
            if owner is not None:
                owner.close()
            if loop is not None:
                loop.close()

    async def _invoke(self, callback, *, terminal=False):
        # Admission is checked again on the worker: a callback queued before
        # shutdown may not run after the final clean checkpoint.
        self._require_open(terminal=terminal)
        task = asyncio.current_task()
        if not terminal:
            self._ordinary_calls.add(task)
        try:
            result = callback()
            return await result if inspect.isawaitable(result) else result
        finally:
            self._ordinary_calls.discard(task)

    async def _call(self, callback, *, terminal=False):
        loop = await asyncio.wrap_future(self._start(terminal=terminal))
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(
                self._invoke(callback, terminal=terminal), loop
            )
        )

    def _call_from_agent(self, callback, *, terminal=False):
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("plugin synchronous checks require an agent worker")
        loop = self._start(terminal=terminal).result()
        return asyncio.run_coroutine_threadsafe(
            self._invoke(callback, terminal=terminal), loop
        ).result()

    async def bootstrap(self, passphrase: str) -> None:
        def operation():
            self._coordinator.bootstrap(passphrase)
            self._refresh()

        await self._call(operation)

    async def unlock(self, passphrase: str) -> None:
        async def operation():
            self._coordinator.authority.unlock(passphrase)
            await self._coordinator.recover()
            self._refresh()

        await self._call(operation)

    async def authority_status(self) -> str:
        """Expose the existing authority posture without retrieving secrets."""
        try:
            return await self._call(lambda: self._coordinator.authority.posture())
        except PluginUnavailable as error:
            if str(error) == "plugin_owner_busy":
                return "plugin_owner_busy"
            raise

    async def recover(self) -> tuple[OperationReceipt, ...]:
        async def operation():
            receipts = await self._coordinator.recover()
            self._refresh()
            return receipts

        return await self._call(operation)

    async def management_details(
        self, installation_id: str, *, workspace_id: str | None
    ):
        """Explicit current-byte verification for management, never a run grant."""

        def operation():
            from .data_cleanup import root_ref
            from .recovery import retained_inspections

            self._admission._workspace(workspace_id)
            state = self._coordinator.published_snapshot()
            installed = next(
                (
                    row
                    for row in state["installations"]
                    if row["installation_id"] == installation_id
                ),
                None,
            )
            if installed is None:
                raise ValueError("installation_unavailable")
            inspection = retained_inspections(state, coordinator=self._coordinator)[
                (installation_id, installed["revision_digest"])
            ]
            return {
                "installation": deepcopy(installed),
                "inspection": inspection,
                "workspace_id": workspace_id,
                "activation": tuple(
                    deepcopy(row)
                    for row in state["activation"]
                    if row["installation_id"] == installation_id
                ),
                "mappings": tuple(
                    deepcopy(row)
                    for row in state["mappings"]
                    if row["installation_id"] == installation_id
                ),
                "revisions": tuple(
                    row["revision_digest"]
                    for row in state["revisions"]
                    if row["installation_id"] == installation_id
                ),
                "roots": tuple(
                    root_ref(row)
                    for row in state["data_roots"]
                    if row["installation_id"] == installation_id
                    or (row.get("custody") or {}).get("attached_installation_id")
                    == installation_id
                ),
                "selection": tuple(
                    row["component_id"]
                    for row in state["selections"]
                    if row["installation_id"] == installation_id
                    and row["revision_digest"] == installed["revision_digest"]
                    and row["selected"]
                ),
            }

        return await self._call(operation)

    async def list_operations(self, *, installation_id=None, limit=50, offset=0):
        """Discover authenticated durable history independently of SQLite hints."""
        from .registry import validate_page

        validate_page(limit, offset)

        async def operation():
            receipts = await self._coordinator.recover()
            authority = self._coordinator.authority
            marker = authority.load_marker()
            rows = []
            for receipt in receipts:
                if receipt.phase == "recovery_required":
                    return (
                        {
                            "installation_id": None,
                            "kind": "recovery",
                            "receipt": receipt,
                        },
                    )
                result = (
                    authority.verify_current()
                    if receipt.operation_id == marker.operation_id
                    else authority.verify_transition(receipt.operation_id).snapshot
                )["operation_result"]
                if (
                    installation_id is None
                    or result["installation_id"] == installation_id
                ):
                    rows.append(
                        {
                            "installation_id": result["installation_id"],
                            "kind": result["kind"],
                            "receipt": receipt,
                        }
                    )
            return tuple(rows[offset : offset + limit])

        return await self._call(operation)

    async def operation_status(self, identity: str):
        """Read current custody separately from durable commitment."""
        with self._lock:
            for operation in self.fences.operations.values():
                if identity in {operation.operation_id, operation.durable_id}:
                    return {
                        "receipt": operation.status(),
                        "blockers": tuple(
                            {"owner_token": token, "reason": "unresolved_owner"}
                            for token in operation.unresolved_tokens
                        )
                        + tuple(
                            {"run_id": record.run_id, "reason": "stopping"}
                            for record in operation.records
                            if not record.completed.is_set()
                        ),
                        "cancel_operation": False,
                    }
            for ticket in self.fences.drains.values():
                if ticket.operation_id == identity:
                    return {
                        "receipt": OperationReceipt(
                            identity, ticket.phase, ticket.phase == "complete"
                        ),
                        "blockers": self.revision_drain.blockers(ticket.token),
                        "cancel_operation": ticket.phase in {"reserved", "waiting"},
                    }

        def root_operation():
            from .data_cleanup import root_ref

            usage = self._coordinator.root_usage
            operation = usage.operations.get(identity)
            if operation is None:
                return None
            return {
                "receipt": operation.receipt,
                "blockers": usage.blockers(
                    tuple(root_ref(row) for row in operation.review.targets)
                ),
                "cancel_operation": operation.receipt.phase == "waiting",
                "cancel_work": operation.receipt.phase == "waiting",
            }

        root_state = await self._call(root_operation)
        if root_state is not None:
            return root_state
        receipt = await self.lookup_operation(identity)
        return {"receipt": receipt, "blockers": (), "cancel_operation": False}

    async def data_status(
        self, roots=None, *, installation_id: str | None = None, limit=50, offset=0
    ):
        def operation():
            from .data_cleanup import open_root, root_ref
            from .registry import validate_page

            validate_page(limit, offset)
            state = self._coordinator.published_snapshot()
            rows = state["data_roots"]
            if installation_id is not None:
                rows = [
                    row for row in rows if row["installation_id"] == installation_id
                ]
            if roots is not None:
                wanted = {ref.root_id: ref for ref in roots}
                rows = [row for row in rows if row["root_id"] in wanted]
                if {row["root_id"]: root_ref(row) for row in rows} != wanted:
                    raise ValueError("root_review_stale")
            result = []
            for row in rows[offset : offset + limit]:
                ref = root_ref(row)
                reason = self._coordinator.root_usage.recovery.get(ref.root_id)
                try:
                    with open_root(self._coordinator.owner, row):
                        pass
                except (ValueError, OSError) as error:
                    reason = (
                        str(error)
                        if isinstance(error, PermissionError)
                        else "root_identity_unqualified"
                    )
                result.append(
                    {
                        "root": ref,
                        "recovery_reason": reason,
                        "cleaned_absent": reason is None
                        and (row.get("custody") or {}).get("state") == "cleaned_absent",
                        "cleanup": deepcopy(row.get("cleanup")),
                        **self._coordinator.root_usage.blocker_status((ref,)),
                    }
                )
            return tuple(result)

        return await self._call(operation)

    async def review_authority_reset(self):
        def operation():
            import time

            from .package_files import canonical_json
            from .review import AuthorityResetReview

            authority = self._coordinator.authority
            posture = authority.posture()
            if posture not in {"ready", "ready_reduced", "locked", "recovery_required"}:
                raise ValueError("plugin_reset_unavailable")
            roots = self._coordinator.registry.authority_projection(
                operation_result=None
            )["data_roots"]
            review = AuthorityResetReview(
                uuid4().hex,
                uuid4().hex,
                authority.namespace_id(),
                posture,
                authority.load_marker(),
                canonical_json(roots),
                time.monotonic() + 900,
            )
            self._reset_review = review
            return review

        return await self._call(operation)

    async def reset_authority(self, review):
        def operation():
            import time

            from .package_files import canonical_json

            authority = self._coordinator.authority
            if (
                getattr(self, "_reset_review", None) is not review
                or review.expires_at <= time.monotonic()
                or review.namespace_id != authority.namespace_id()
                or review.marker != authority.load_marker()
                or review.posture != authority.posture()
                or review.roots_json
                != canonical_json(
                    self._coordinator.registry.authority_projection(
                        operation_result=None
                    )["data_roots"]
                )
            ):
                raise ValueError("plugin_reset_review_stale")
            result = self._coordinator.reset(operation_id=review.operation_id)
            self._reset_review = None
            self._refresh()
            return result

        return await self._call(operation)

    async def review_data_boot_reconciliation(self, roots):
        return await self._call(
            lambda: self._coordinator.review_data_boot_reconciliation(roots)
        )

    async def review_install(
        self,
        root: Path,
        *,
        selection: tuple[str, ...],
        workspace_id: str | None,
        dialect: str | None = None,
        enable_in_workspace: bool = False,
    ) -> PluginReview:
        def operation():
            from .inspection import inspect_package

            self._admission._workspace(workspace_id)
            return self._coordinator.review(
                inspect_package(root, dialect=dialect),
                selection=selection,
                workspace_id=workspace_id,
                enable_in_workspace=enable_in_workspace,
            )

        return await self._call(operation)

    async def acquire_snapshot(
        self, source: SourceLocator, operation_id: str, destination: Path
    ) -> AcquiredSnapshot:
        """Acquire a disabled candidate on the existing owner and worker."""
        from .acquisition import acquire_snapshot

        return await self._call(
            lambda: acquire_snapshot(
                source,
                operation_id,
                destination,
                owner=self._coordinator.owner,
                authority=self._coordinator.authority,
            )
        )

    async def acquisition_destination(self, operation_id: str) -> Path:
        return await self._call(
            lambda: self._coordinator.owner.acquisition_destination(operation_id)
        )

    async def release_acquired_snapshot(self, snapshot: AcquiredSnapshot) -> None:
        """Release the result reference after consumption or explicit discard."""
        from .acquisition import release_snapshot

        await self._call(
            lambda: release_snapshot(
                self._coordinator.owner,
                self._coordinator.authority,
                snapshot,
            )
        )

    async def cleanup_acquisitions(self) -> tuple[str, ...]:
        """Reclaim only released, exact terminal acquisitions older than 24 hours."""
        from .acquisition import cleanup_acquisitions

        return await self._call(
            lambda: cleanup_acquisitions(
                self._coordinator.owner,
                self._coordinator.authority,
            )
        )

    async def review_data_creation(
        self, installation_id: str, *, workspace_id: str | None = None
    ):
        def operation():
            self._admission._workspace(workspace_id)
            return self._coordinator.review_data_creation(
                installation_id, workspace_id=workspace_id
            )

        return await self._call(operation)

    async def create_data(self, review: RootReview, operation_id: str) -> DataRootRef:
        return await self._call(
            lambda: self._coordinator.create_data(review, operation_id)
        )

    async def review_data_deletion(self, roots: tuple[DataRootRef, ...]) -> RootReview:
        return await self._call(lambda: self._coordinator.review_data_deletion(roots))

    async def review_data_attachment(
        self, roots: tuple[DataRootRef, ...], installation_id: str | None
    ) -> RootReview:
        return await self._call(
            lambda: self._coordinator.review_data_attachment(roots, installation_id)
        )

    async def delete_data(
        self, roots: tuple[DataRootRef, ...], operation_id: str
    ) -> OperationReceipt:
        from .data_cleanup import begin_data_operation

        self._require_open()
        begin_data_operation(self._coordinator, roots, operation_id)
        return await self._call(
            lambda: self._coordinator.delete_data(roots, operation_id)
        )

    async def cancel_data_work(self, operation_id: str) -> None:
        await self._call(lambda: self._coordinator.cancel_data_work(operation_id))

    async def cancel_data_deletion(self, operation_id: str) -> None:
        await self._call(lambda: self._coordinator.cancel_data_deletion(operation_id))

    async def review_data_reconciliation(self, roots, *, confirm_quiescence):
        return await self._call(
            lambda: self._coordinator.review_data_reconciliation(
                roots, confirm_quiescence=confirm_quiescence
            )
        )

    async def reconcile_data(
        self, review: RootReview, operation_id: str
    ) -> tuple[DataRootRef, ...]:
        return await self._call(
            lambda: self._coordinator.reconcile_data(review, operation_id)
        )

    async def review_data_cleanup_resume(
        self, roots: tuple[DataRootRef, ...]
    ) -> RootReview:
        return await self._call(
            lambda: self._coordinator.review_data_cleanup_resume(roots)
        )

    async def reserve_data_user(
        self,
        operation_id: str,
        installation_id: str,
        workspace_id: str | None,
        revision_digest: str,
        *,
        roots,
        cancel=None,
    ):
        """Host producer reserves all actual grants before spawn or handle access."""

        def reserve():
            token = self._coordinator.owner.reserve_launch(
                operation_id,
                installation_id,
                workspace_id,
                revision_digest,
                roots=roots,
            )
            if cancel is not None:
                try:
                    self._coordinator.root_usage.retain_cancel(token, cancel)
                except BaseException:
                    self._coordinator.owner.settle_process(token, True)
                    raise
            return token

        return await self._call(reserve)

    async def publish_data_user(self, token: str, provenance: dict) -> None:
        await self._call(
            lambda: self._coordinator.owner.publish_process(token, provenance)
        )

    async def settle_data_user(self, token: str, *, confirmed: bool) -> None:
        """Only actual host terminal evidence permits confirmed settlement."""
        await self._call(
            lambda: self._coordinator.owner.settle_process(token, confirmed),
            terminal=True,
        )

    async def retain_revisions(self, installation_id: str) -> OperationReceipt:
        """Run owned bounded retention through the existing commit worker."""
        try:
            return await self._call(
                lambda: self._coordinator.retain_revisions(installation_id)
            )
        finally:
            await self._call(self._refresh)

    async def review_revision(self, installation_id: str, root: Path) -> PluginReview:
        def operation():
            from .inspection import inspect_package

            return self._coordinator.review_revision(
                installation_id, inspect_package(root)
            )

        return await self._call(operation)

    async def review_rollback(
        self, installation_id: str, revision_digest: str
    ) -> PluginReview:
        return await self._call(
            lambda: self._coordinator.review_rollback(installation_id, revision_digest)
        )

    async def apply_revision(
        self, review: PluginReview, operation_id: str
    ) -> OperationReceipt:
        self._require_open()
        self.revision_drain.activate(
            review.drain_token, review=review, operation_id=operation_id
        )

        def refresh_completed(task):
            self._revision_refresh_tasks.discard(task)
            self._refresh()

        async def operation():
            try:
                return await self._coordinator.apply_revision(review, operation_id)
            finally:
                task = self.revision_drain.ticket(review.drain_token).task
                # The coordinator retains Apply when this caller leaves. Its
                # terminal publication must refresh the facade independently.
                if (
                    task is not None
                    and not task.done()
                    and task not in self._revision_refresh_tasks
                ):
                    self._revision_refresh_tasks.add(task)
                    task.add_done_callback(refresh_completed)
                self._refresh()

        return await self._call(operation)

    async def review_trust(self, installation_id: str) -> PluginReview:
        return await self._call(lambda: self._coordinator.review_trust(installation_id))

    async def review_selection(
        self, installation_id: str, *, selection: tuple[str, ...]
    ) -> PluginReview:
        return await self._call(
            lambda: self._coordinator.review_selection(
                installation_id, selection=selection
            )
        )

    def cancel_revision(self, review: PluginReview) -> None:
        ticket = self.revision_drain.ticket(review.drain_token)
        if (
            ticket.review_token != review.token
            or ticket.installation_id != review.installation_id
        ):
            raise ValueError("plugin_drain_review_changed")
        self.revision_drain.cancel(review.drain_token)

    def cancel_revision_work(self, review: PluginReview) -> None:
        ticket = self.revision_drain.ticket(review.drain_token)
        if (
            ticket.review_token != review.token
            or ticket.installation_id != review.installation_id
        ):
            raise ValueError("plugin_drain_review_changed")
        self.revision_drain.cancel_work(review.drain_token)

    async def review_configuration(
        self,
        installation_id: str,
        *,
        connections=None,
        tools=None,
        tool_references=None,
        models=None,
        component_tools=None,
    ) -> PluginReview:
        """Capture current host references on the existing plugin storage worker."""
        return await self._call(
            lambda: self._coordinator.review_configuration(
                installation_id,
                connections=connections,
                tools=tools,
                tool_references=tool_references,
                models=models,
                component_tools=component_tools,
            )
        )

    async def management_snapshot(
        self,
        workspace_id: str | None,
        *,
        query: str = "",
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[ManagementInstallation, ...]:
        """Read one bounded page of current metadata on the existing worker."""
        from .registry import validate_page

        validate_page(limit, offset)
        if not isinstance(query, str) or len(query) > 1024:
            raise ValueError("plugin_query_invalid")

        def operation():
            from .admission import effective_activation

            self._admission._workspace(workspace_id)
            authority = self._coordinator.published_snapshot()
            installed = sorted(
                (
                    row
                    for row in authority["installations"]
                    if query.casefold()
                    in (row.get("alias") or row["installation_id"]).casefold()
                ),
                key=lambda row: row["installation_id"],
            )[offset : offset + limit]
            authority, inspections = self._coordinator.recorded_inspections(
                tuple(
                    (row["installation_id"], row["revision_digest"])
                    for row in installed
                )
            )
            result = []
            for row in installed:
                identity, revision = row["installation_id"], row["revision_digest"]
                inspection = inspections[(identity, revision)]
                override = next(
                    (
                        item["intent"]
                        for item in authority["activation"]
                        if item["installation_id"] == identity
                        and item["workspace_id"] == workspace_id
                    ),
                    "inherit",
                )
                selected = tuple(
                    sorted(
                        item["component_id"]
                        for item in authority["selections"]
                        if item["installation_id"] == identity
                        and item["revision_digest"] == revision
                        and item["selected"]
                    )
                )
                trusted = any(
                    item["installation_id"] == identity
                    and item["revision_digest"] == revision
                    and item["reviewed"]
                    for item in authority["revision_trust"]
                )
                ready, error = (), None
                try:
                    ready = self._admission.recorded_eligibility(
                        identity,
                        workspace_id,
                        authority=authority,
                        inspection=inspection,
                    )
                except PluginUnavailable:
                    error = "plugin_components_unavailable"
                binding = inspection.catalog_binding
                reference = (
                    binding.package if binding else inspection.acquisition_reference
                )
                result.append(
                    ManagementInstallation(
                        identity,
                        revision,
                        row.get("alias") or identity,
                        workspace_id,
                        bool(row["activation_default"]),
                        override,
                        effective_activation(bool(row["activation_default"]), override),
                        trusted,
                        selected,
                        ready,
                        tuple(
                            ManagementComponent(
                                item.component_id,
                                item.kind,
                                item.support,
                                item.evidence,
                                item.dependencies,
                                item.activation_blockers,
                            )
                            for item in sorted(
                                inspection.inventory.values(),
                                key=lambda item: item.component_id,
                            )
                        ),
                        error,
                        binding.source_id if binding else None,
                        (
                            reference.origin + reference.repository
                            if reference
                            else inspection.source_identity
                        ),
                        reference.ref if reference else None,
                        reference.commit if reference else None,
                        inspection.dialect,
                    )
                )
            return tuple(result)

        try:
            return await self._call(operation)
        except (OSError, ValueError, PermissionError) as error:
            raise PluginUnavailable("plugin_management_unavailable") from error

    async def management_readme(
        self,
        installation_id: str,
        revision_digest: str,
        *,
        workspace_id: str | None,
    ) -> str:
        """Read only the explicitly selected current revision's bounded README."""

        def operation():
            import os
            import stat

            from .package_files import (
                _directory_fd,
                _member_fd,
                validate_relative_member,
            )

            self._admission._workspace(workspace_id)
            _, inspections = self._coordinator.recorded_inspections(
                ((installation_id, revision_digest),)
            )
            inspection = inspections[(installation_id, revision_digest)]
            try:
                with (
                    _directory_fd(Path(inspection.materialized_identity)) as root_fd,
                    _member_fd(root_fd, validate_relative_member("README.md")) as fd,
                ):
                    before = os.fstat(fd)
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError("plugin_readme_invalid")
                    preview = bytearray()
                    while len(preview) <= 65536:
                        chunk = os.read(fd, 65537 - len(preview))
                        if not chunk:
                            break
                        preview.extend(chunk)
                    after = os.fstat(fd)
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ):
                        raise ValueError("plugin_readme_changed")
                # Bound rendered UTF-8 too: replacement characters can expand
                # malformed source bytes. Never split a displayed code point.
                encoded = bytes(preview).decode("utf-8", errors="replace").encode()
                if len(encoded) > 65536:
                    marker = b"\n[truncated]"
                    return (
                        encoded[: 65536 - len(marker)].decode("utf-8", errors="ignore")
                        + marker.decode()
                    )
                return encoded.decode()
            except FileNotFoundError:
                return ""

        try:
            return await self._call(operation)
        except (OSError, ValueError, PermissionError) as error:
            raise PluginUnavailable("plugin_details_unavailable") from error

    async def capture_mcp_snapshot(
        self,
        installation_id: str,
        workspace_id: str | None,
        run_id: str,
        *,
        component_ceiling: tuple[str, ...] | None = None,
    ):
        """Capture immutable scope authority, including all requested prerequisites."""
        return await self._call(
            lambda: self._admission.capture(
                installation_id,
                workspace_id,
                run_id,
                component_ceiling=component_ceiling,
            )
        )

    def check_effect_actor(
        self, snapshot, component_id: str, *, run_id: str | None = None
    ) -> None:
        """Empty host child ceilings cannot inherit a parent's owned provider."""
        from tldw_chatbook.Agents.run_context import current_run_actor

        actor = current_run_actor()
        run_id = run_id or (actor.run_id if actor else "")
        if not run_id:
            return
        identity = (
            snapshot.installation_id,
            snapshot.revision_digest,
            snapshot.workspace_id,
            component_id,
        )
        with self._lock:
            if run_id in self._terminal_runs:
                raise PluginUnavailable("plugin_actor_terminal")
            ceiling = self._actor_components.get(run_id)
            if ceiling is not None and identity not in ceiling:
                raise PluginUnavailable("plugin_actor_ceiling_unavailable")
            if (
                ceiling is None
                and actor is not None
                and actor.run_id == run_id
                and actor.kind == "subagent"
            ):
                raise PluginUnavailable("plugin_actor_unbound")

    async def check_mcp_snapshot(self, snapshot, component_id: str) -> None:
        self.check_effect_actor(snapshot, component_id)
        self.fences.check_snapshot(snapshot)
        await self._call(lambda: self._admission.check(snapshot, component_id))

    async def review_activation(
        self, installation_id: str, *, workspace_id: str | None, intent: str
    ) -> PluginReview:
        def operation():
            self._admission._workspace(workspace_id)
            return self._coordinator.review_activation(
                installation_id, workspace_id=workspace_id, intent=intent
            )

        return await self._call(operation)

    async def commit(self, review: PluginReview, operation_id: str) -> OperationReceipt:
        from .revocation import revocation_for_review

        if review.kind in {"revoke", "uninstall"}:
            if operation_id != review.operation_id:
                raise ValueError("original issued review required")
            return await self.finish_revocation(self.begin_reviewed_removal(review))
        if review.kind in {"update", "select"}:
            return await self.apply_revision(review, operation_id)
        target = revocation_for_review(review)
        if target is not None:
            return await self._revoke(target, operation_id, "activate", review=review)

        async def operation():
            self._admission._workspace(review.workspace_id)
            try:
                return await self._coordinator.commit(review, operation_id)
            finally:
                self._refresh()

        return await self._call(operation)

    def _refresh(self):
        """Publish metadata eligibility, never package bodies or a runtime grant."""
        from .components import component_summary
        from .recovery import retained_inspections

        catalog, details = {}, []
        revisions, generations = {}, {}
        try:
            authority = self._coordinator.published_snapshot()
            revisions = {
                row["installation_id"]: row["revision_digest"]
                for row in authority["installations"]
            }
            generations = {
                (row["installation_id"], row["scope_kind"], row["workspace_id"]): row[
                    "generation"
                ]
                for row in authority["authority_generations"]
            }
            inspections = retained_inspections(authority, coordinator=self._coordinator)
            for installation in authority["installations"]:
                identity = installation["installation_id"]
                inspection = inspections[(identity, installation["revision_digest"])]
                selected = {
                    row["component_id"]
                    for row in authority["selections"]
                    if row["installation_id"] == identity
                    and row["revision_digest"] == installation["revision_digest"]
                    and row["selected"]
                }
                for component in inspection.inventory.values():
                    row = component_summary(
                        identity,
                        inspection,
                        component.component_id,
                        installation.get("alias") or identity,
                    )
                    row.update(
                        plugin_selected=component.component_id in selected,
                        plugin_support=component.support,
                        plugin_dependencies=component.dependencies,
                    )
                    details.append(row)
                scopes = {None} | {
                    row["workspace_id"]
                    for row in authority["activation"]
                    if row["installation_id"] == identity
                }
                for workspace in scopes:
                    rows = []
                    try:
                        snapshot = self._admission.capture(
                            identity, workspace, "catalog", fresh=False
                        )
                        token = uuid4().hex
                        self._snapshots[token] = snapshot
                        for component_id in snapshot.selection:
                            row = component_summary(
                                identity,
                                inspection,
                                component_id,
                                snapshot.alias,
                                json.loads(snapshot.mappings_json),
                            )
                            row.update(
                                plugin_ceiling=token,
                                plugin_workspace_id=workspace,
                                trust_blocked=False,
                            )
                            rows.append(row)
                    except PluginUnavailable:
                        pass
                    catalog[(identity, workspace)] = tuple(rows)
        except (ValueError, OSError, PermissionError):
            catalog, details = {}, []
            revisions, generations = {}, {}
        with self._lock:
            self._catalog = catalog
            self._details = tuple(details)
            self._published_revisions = revisions
            self._published_generations = generations

    def capture_maximum(self, workspace_id: str | None) -> dict:
        """Read the worker-published immutable metadata ceiling without IO."""
        with self._lock:
            catalog = self._catalog
            identities = sorted({identity for identity, _ in catalog})
            rows = []
            for identity in identities:
                try:
                    self.fences.check(identity, workspace_id)
                except PluginUnavailable:
                    continue
                selected = catalog.get(
                    (identity, workspace_id), catalog.get((identity, None), ())
                )
                for row in selected:
                    prior = self._snapshots[row["plugin_ceiling"]]
                    generations = prior.generations
                    if workspace_id is not None and prior.workspace_id is None:
                        generations = (*generations, ("workspace", workspace_id, 0))
                    try:
                        self.fences.current(identity, generations)
                        self.fences.check_snapshot(prior)
                    except PluginUnavailable:
                        continue
                    rows.append(dict(deepcopy(row), plugin_workspace_id=workspace_id))
        return {
            "backend": "local",
            "available_skills": rows,
            "blocked_skills": [],
            "context_text": "",
        }

    def published_current(self, snapshot) -> bool:
        """Check worker-published authority generations without effect-lane IO."""
        with self._lock:
            self.fences.check_snapshot(snapshot)
            return self._published_revisions.get(
                snapshot.installation_id
            ) == snapshot.revision_digest and all(
                self._published_generations.get(
                    (snapshot.installation_id, kind, workspace), 0
                )
                == generation
                for kind, workspace, generation in snapshot.generations
            )

    def list_components(self, workspace_id: str | None = None) -> list[dict]:
        """Expose disabled/dependent metadata without reading instruction bodies."""
        maximum = self.capture_maximum(workspace_id)
        eligible = {
            (row["plugin_installation_id"], row["plugin_component_id"])
            for row in maximum["available_skills"]
        }
        with self._lock:
            rows = [deepcopy(row) for row in self._details]
        for row in rows:
            installation = row["plugin_installation_id"]
            row["plugin_available"] = (
                installation,
                row["plugin_component_id"],
            ) in eligible
            if row["plugin_available"]:
                row[
                    "plugin_blockers"
                ] = []  # Current admission resolved these requirements.
            elif row["plugin_selected"]:
                row["plugin_blockers"] += [
                    "dependency_unavailable:" + key
                    for key in row["plugin_dependencies"]
                    if (installation, key) not in eligible
                ]
                if not row["plugin_blockers"]:
                    row["plugin_blockers"] = ["plugin_requirements_unavailable"]
        return rows

    def list_skills(self) -> list[dict]:
        with self._lock:
            return [
                deepcopy(row) for row in self._details if row["plugin_kind"] == "skill"
            ]

    def owns(self, name: str) -> bool:
        from .skill_provider import owned_identifier

        return owned_identifier(name) or any(
            name in {row["name"], row["tool_name"], row["record_id"]}
            for row in self.list_skills()
        )

    def fleet_resume_ceiling(self, pin, run_id, handle_id, entries):
        """Verify host-retained transcript before resolving/launching a new child."""
        from .continuation import fleet_ceiling

        return self._call_from_agent(
            lambda: fleet_ceiling(self, pin, run_id, handle_id, entries)
        )

    def constrain_entries(self, entries, ceiling):
        """Narrow inherited producer entries without minting any new admission."""
        if ceiling is None:
            return tuple(entries)
        with self._lock:
            return tuple(
                row
                for row in entries
                if (snapshot := self._admitted.get(row.get("plugin_admission")))
                is not None
                and row.get("plugin_component_id")
                in ceiling.get(snapshot.installation_id, ())
            )

    def capture_resume_pin(
        self, entries, run_id: str, conversation_id: str, message_id: str
    ) -> str:
        """Capture the original host-bound producer on the existing plugin worker."""
        from .continuation import capture_pin

        return self._call_from_agent(
            lambda: capture_pin(self, entries, run_id, conversation_id, message_id)
        )

    def seal_resume_checkpoint(
        self, pin: str, checkpoint, conversation_id: str, message_id: str
    ):
        """Finalize private checkpoint bytes without consulting current selections."""
        from .continuation import seal_checkpoint

        return self._call_from_agent(
            lambda: seal_checkpoint(self, pin, checkpoint, conversation_id, message_id)
        )

    async def resume_maximum(
        self, maximum, checkpoint, conversation_id: str, message_id: str
    ) -> dict:
        """Verify an archived owner before assembling a new run's context."""
        from .continuation import ResumeConstraint, constrain_maximum

        constraint = (
            ResumeConstraint(checkpoint, conversation_id, message_id)
            if checkpoint.schema_version == 2
            else "zero"
        )
        return await self._call(lambda: constrain_maximum(self, maximum, constraint))

    async def admit(self, maximum: Mapping[str, Any], run_id: str) -> dict:
        def operation():
            from .continuation import constrain_maximum

            # Files, SQLite and package bytes are verified on the worker without
            # holding the lock needed by immediate live revocation.
            narrowed = constrain_maximum(
                self, maximum, maximum.get("plugin_resume_constraint")
            )
            return admit_narrowed(narrowed)

        def admit_narrowed(maximum):
            if run_id in self._retired:
                raise PluginUnavailable("plugin_turn_retired")
            turn_id = maximum.get("plugin_turn_id")
            ceiling = frozenset(
                (
                    row.get("plugin_ceiling"),
                    row.get("plugin_component_id"),
                    row.get("plugin_workspace_id"),
                )
                for row in maximum.get("available_skills", ())
                if row.get("plugin_owned")
            )
            prior_custody = self._pending_custody.get(run_id)
            if prior_custody is not None:
                if prior_custody[0] != turn_id:
                    raise PluginUnavailable("plugin_turn_identity_changed")
                if not ceiling <= prior_custody[1]:
                    raise PluginUnavailable("plugin_ceiling_changed")
            result = dict(maximum)
            rows = []
            for entry in maximum.get("available_skills", ()):
                row = dict(entry)
                if row.get("plugin_owned"):
                    prior = self._snapshots.get(row.get("plugin_ceiling"))
                    if prior is None:
                        raise PluginUnavailable("plugin_ceiling_unavailable")
                    workspace = row.get("plugin_workspace_id")
                    generations = prior.generations
                    if workspace is not None and prior.workspace_id is None:
                        generations = tuple(
                            sorted((*generations, ("workspace", workspace, 0)))
                        )
                    snapshot = replace(
                        prior,
                        workspace_id=workspace,
                        run_id=run_id,
                        generations=generations,
                        live_generations=(
                            tuple(
                                sorted(
                                    (
                                        *prior.live_generations,
                                        ("workspace", workspace, 0),
                                    )
                                )
                            )
                            if workspace is not None and prior.workspace_id is None
                            else prior.live_generations
                        ),
                    )
                    if maximum.get("plugin_resume_constraint") is not None:
                        selected = {
                            item.get("plugin_component_id")
                            for item in maximum.get("available_skills", ())
                            if item.get("plugin_ceiling") == row.get("plugin_ceiling")
                            and item.get("plugin_workspace_id") == workspace
                        }
                        snapshot = replace(
                            snapshot,
                            selection=tuple(
                                key for key in snapshot.selection if key in selected
                            ),
                        )
                    self.fences.require_admission(
                        snapshot.installation_id, snapshot.revision_digest
                    )
                    self._admission.check(snapshot, row["plugin_component_id"])
                    key = (run_id, row["plugin_ceiling"], workspace)
                    token = self._admission_keys.get(key)
                    with self._lock:
                        self.fences.check_snapshot(snapshot)
                        self.fences.require_admission(
                            snapshot.installation_id, snapshot.revision_digest
                        )
                        if run_id in self._retired:
                            raise PluginUnavailable("plugin_turn_retired")
                        if token is None:
                            token = uuid4().hex
                            self._admission_keys[key] = token
                            self._admitted[token] = snapshot
                    from .components import component_summary

                    row = dict(
                        component_summary(
                            snapshot.installation_id,
                            snapshot.inspection,
                            row["plugin_component_id"],
                            snapshot.alias,
                            json.loads(snapshot.mappings_json),
                        ),
                        plugin_ceiling=entry["plugin_ceiling"],
                        plugin_workspace_id=workspace,
                        plugin_admission=token,
                        plugin_run_id=run_id,
                        trust_blocked=False,
                    )
                rows.append(row)
            with self._lock:
                if run_id in self._retired:
                    raise PluginUnavailable("plugin_turn_retired")
                # Verification of a later installation may have stalled after an
                # earlier one passed. Publish the whole admitted set atomically
                # against current live generations and drain state, without I/O.
                for row in rows:
                    if row.get("plugin_owned"):
                        snapshot = self._admitted[row["plugin_admission"]]
                        self.fences.check_snapshot(snapshot)
                        self.fences.require_admission(
                            snapshot.installation_id, snapshot.revision_digest
                        )
                self._pending_custody.setdefault(run_id, (turn_id, ceiling))
                result["available_skills"] = rows
                return result

        return await self._call(operation)

    def _checked(
        self, name: str, token: str, *, binding: bool = False,
        component_kind: str | None = None,
    ):
        from .components import component_summary

        snapshot = self._admitted.get(token)
        if snapshot is None:
            raise PluginUnavailable("plugin_run_admission_required")
        from tldw_chatbook.Agents.run_context import current_run_id

        actor_id = current_run_id()
        with self._lock:
            if snapshot.run_id in self._retired and not any(
                record.pending_id == snapshot.run_id and record.run_id == actor_id
                for record in self._live_runs.values()
            ):
                raise PluginUnavailable("plugin_turn_retired")
        for component_id in snapshot.selection:
            if (
                component_kind is not None
                and snapshot.inspection.inventory[component_id].kind != component_kind
            ):
                continue
            row = component_summary(
                snapshot.installation_id,
                snapshot.inspection,
                component_id,
                snapshot.alias,
                json.loads(snapshot.mappings_json),
            )
            if name in {row["name"], row["tool_name"], row["record_id"]}:
                if actor_id and not binding:
                    record = self._live_runs.get((snapshot.installation_id, actor_id))
                    if (
                        record is None
                        or record.component_ceiling is not None
                        and component_id not in record.component_ceiling
                    ):
                        raise PluginUnavailable("plugin_actor_ceiling_unavailable")
                self._admission.check(snapshot, component_id)
                return snapshot, component_id, row
        raise PluginUnavailable("plugin_component_not_admitted")

    async def get_skill(self, name: str) -> dict:
        """Explicit detail reads remain owned and never imply standalone trust."""

        def operation():
            from .components import component_summary
            from .package_files import capture_package
            from .recovery import retained_inspections

            authority = self._coordinator.published_snapshot()
            inspections = retained_inspections(authority, coordinator=self._coordinator)
            for installed in authority["installations"]:
                identity = installed["installation_id"]
                inspection = inspections[(identity, installed["revision_digest"])]
                for component in inspection.inventory.values():
                    if component.kind != "skill":
                        continue
                    row = component_summary(
                        identity,
                        inspection,
                        component.component_id,
                        installed.get("alias") or identity,
                    )
                    if name in {row["name"], row["tool_name"], row["record_id"]}:
                        capture = capture_package(
                            Path(inspection.materialized_identity)
                        )
                        if (
                            capture.errors
                            or capture.digest != inspection.content_digest
                        ):
                            raise PluginUnavailable("plugin_material_changed")
                        return dict(
                            row, content=capture.read(component.path).decode("utf-8")
                        )
            raise PluginUnavailable("plugin_skill_unavailable")

        return await self._call(operation)

    async def execute_skill(
        self, name: str, *, admission_token: str, args: str | None = None
    ) -> dict:
        def operation():
            from .skill_provider import render_skill

            snapshot, component_id, row = self._checked(name, admission_token)
            kind = snapshot.inspection.inventory[component_id].kind
            if kind == "skill":
                result = render_skill(snapshot, component_id, row, args or "")
            elif kind in {"command", "rule"}:
                from tldw_chatbook.Agents.agent_models import carry_plugin_context

                from .commands import command_arguments, render_command
                from .components import component_body
                from .context import instruction_block

                if kind == "command":
                    blocks = render_command(
                        snapshot, component_id, command_arguments(args or "")
                    )
                    bodies = [block["content"] for block in blocks]
                    rendered = carry_plugin_context("\n\n".join(bodies), *bodies)
                else:
                    if row["plugin_rule_mode"] != "manual":
                        raise PluginUnavailable("plugin_rule_manual_required")
                    rendered = instruction_block(
                        snapshot.installation_id,
                        component_id,
                        snapshot.revision_digest,
                        component_body(snapshot, component_id),
                        args or "",
                    )
                result = {
                    "rendered_prompt": rendered,
                    "execution_mode": "inline",
                    "plugin_owned": True,
                    "allowed_tools": [],
                    "skill_name": name,
                }
            else:
                raise PluginUnavailable("plugin_manual_component_unavailable")
            self._admission.check(snapshot, component_id)
            return result

        return await self._call(operation)

    async def render_rules(self, entries) -> tuple[dict, ...]:
        """Read complete active rules under the exact admitted component ceiling."""

        def operation():
            from .components import component_body
            from .context import instruction_block

            result = []
            for row in sorted(
                entries,
                key=lambda item: (
                    item.get("plugin_installation_id", ""),
                    item.get("plugin_component_id", ""),
                ),
            ):
                if (
                    row.get("plugin_kind") != "rule"
                    or row.get("plugin_rule_mode") != "always"
                ):
                    continue
                snapshot, component_id, _summary = self._checked(
                    row["name"], row["plugin_admission"]
                )
                body = instruction_block(
                    snapshot.installation_id,
                    component_id,
                    snapshot.revision_digest,
                    component_body(snapshot, component_id),
                )
                self._admission.check(snapshot, component_id)
                result.append({"role": "user", "content": body})
            return tuple(result)

        return await self._call(operation)

    async def component_snapshots(self, maximum):
        """Resolve already admitted selections through the actual immutable ceiling."""
        admitted = await self.admit(maximum, maximum["plugin_run_id"])
        entries = tuple(
            row for row in admitted["available_skills"] if row.get("plugin_owned")
        )

        def operation():
            snapshots = {}
            for entry in entries:
                snapshot, _component, _row = self._checked(
                    entry["name"], entry["plugin_admission"], binding=True
                )
                snapshots[snapshot.installation_id] = snapshot
            return tuple(snapshots[key] for key in sorted(snapshots))

        return await self._call(operation)

    async def hook_configuration(self, maximum):
        """Pin owned handlers from the same captured Console component ceiling."""
        from .hooks import NativeHooks

        snapshots = await self.component_snapshots(maximum)
        return await self._call(lambda: NativeHooks(self, snapshots))

    async def agent_presets(
        self, entries, eligible: frozenset[str], *, parent_provider: str = "",
        registry: Any = None,
    ) -> tuple:
        """Project admitted ephemeral presets before ordinary child planning."""

        def operation():
            from .agent_presets import agent_definition

            result = []
            for row in entries:
                if row.get("plugin_kind") == "agent":
                    snapshot, component_id, _summary = self._checked(
                        row["name"], row["plugin_admission"], binding=True
                    )
                    try:
                        result.append(
                            agent_definition(
                                snapshot,
                                component_id,
                                eligible,
                                parent_provider=parent_provider,
                                registry=registry,
                            )
                        )
                    except PluginUnavailable as error:
                        if str(error) != "plugin_model_parent_mismatch":
                            raise
                    self._admission.check(snapshot, component_id)
            return tuple(result)

        return await self._call(operation)

    async def read_skill_file(
        self, name: str, path: str, *, admission_token: str
    ) -> dict:
        def operation():
            from .context import instruction_block
            from .package_files import capture_package, validate_relative_member

            snapshot, component_id, row = self._checked(name, admission_token, component_kind="skill")
            relative = validate_relative_member(path)
            component = snapshot.inspection.inventory[component_id]
            member = str(Path(component.path).parent / relative)
            capture = capture_package(Path(snapshot.inspection.materialized_identity))
            if capture.errors or capture.digest != snapshot.inspection.content_digest:
                raise PluginUnavailable("plugin_material_changed")
            data = capture.read(member)
            try:
                body = data.decode("utf-8")
            except UnicodeError:
                raise PluginUnavailable("plugin_file_not_text") from None
            result = {
                "content": instruction_block(
                    snapshot.installation_id,
                    component_id,
                    snapshot.revision_digest,
                    body,
                ),
                "size": len(data),
                "truncated": False,
                "record_id": row["record_id"],
                "plugin_owned": True,
            }
            self._admission.check(snapshot, component_id)
            return result

        return await self._call(operation)

    def current_definition(self, name: str, admission_token: str) -> str:
        return self._call_from_agent(
            lambda: self._checked(name, admission_token)[2]["definition_digest"]
        )

    async def check_entries(self, entries: Sequence[Mapping[str, Any]]) -> None:
        self.check_entries_live(entries)
        await self._call(
            lambda: [
                self._checked(row["name"], row["plugin_admission"])
                for row in entries
                if row.get("plugin_owned")
            ]
        )

    def check_entries_from_agent(self, entries: Sequence[Mapping[str, Any]]) -> None:
        self.check_entries_live(entries)
        self._call_from_agent(
            lambda: [
                self._checked(row["name"], row["plugin_admission"])
                for row in entries
                if row.get("plugin_owned")
            ]
        )

    def check_entries_live(self, entries: Sequence[Mapping[str, Any]]) -> None:
        """Immediate refusal for callbacks; passing is not durable authorization."""
        with self._lock:
            for row in entries:
                if row.get("plugin_owned"):
                    snapshot = self._admitted.get(row.get("plugin_admission"))
                    if snapshot is None:
                        raise PluginUnavailable("plugin_run_admission_required")
                    self.fences.check_snapshot(snapshot)

    async def review_removal(
        self, installation_id: str, *, uninstall: bool
    ) -> PluginReview:
        """Retain an actual worker-issued review before management confirmation."""
        import time

        review = await self._call(
            lambda: self._coordinator.review_removal(
                installation_id, uninstall=uninstall
            )
        )
        with self._lock:
            self._removal_reviews = {
                token: value
                for token, value in self._removal_reviews.items()
                if value.expires_at > time.monotonic()
                or token in self._removal_requests
            }
            if len(self._removal_reviews) >= 1000:
                raise RuntimeError("plugin_removal_review_limit")
            self._removal_reviews[review.token] = review
        return review

    def begin_reviewed_removal(self, review: PluginReview) -> RevocationRequest:
        """Seal now using original custody; authoritative revalidation stays on worker."""
        import time

        with self._lock:
            if (
                type(review) is not PluginReview
                or self._removal_reviews.get(review.token) is not review
                or review.kind not in {"revoke", "uninstall"}
                or not review.installation_id
            ):
                raise ValueError("plugin_removal_review_unavailable")
            prior = self._removal_requests.get(review.token)
            if prior is not None:
                return prior
            if review.expires_at <= time.monotonic():
                raise ValueError("plugin_removal_review_expired")
            request = self._begin_revocation(
                RevocationTarget(review.installation_id, None, True),
                review.kind,
                review=review,
            )
            self._removal_requests[review.token] = request
            return request

    def begin_disable(self, target: RevocationTarget) -> RevocationRequest:
        """Retain caller custody and seal/cancel before any storage-worker access."""
        return self._begin_revocation(target, "revoke")

    def begin_uninstall(self, installation_id: str) -> RevocationRequest:
        return self._begin_revocation(
            RevocationTarget(installation_id, None, True), "uninstall"
        )

    def _begin_revocation(self, target, kind, *, review=None):
        # Reject a missing event loop before sealing rather than losing task custody.
        asyncio.get_running_loop()
        self._require_open()
        request = self.fences.request(target, kind, review)
        operation = self.fences.operations[request.request_id]
        self._start_revocation(operation)
        return request

    def _start_revocation(self, operation):
        self.fences.require_current(operation)
        with self._lock:
            if operation.task is None or operation.task.done():

                async def persist():
                    async def on_worker():
                        try:
                            return await self._coordinator._revoke(
                                operation.target,
                                operation.operation_id,
                                operation.kind,
                                review=operation.review,
                            )
                        finally:
                            self._refresh()

                    try:
                        return await self._call(on_worker)
                    except Exception as error:
                        from .revocation import RevocationConflict, RevocationFailure

                        if isinstance(
                            error, (RevocationFailure, RevocationConflict, ValueError)
                        ):
                            raise
                        operation.receipt = replace(
                            operation.receipt, persistence_error=type(error).__name__
                        )
                        raise RevocationFailure(operation.status(), error) from error

                operation.task = asyncio.create_task(persist())
                operation.task.add_done_callback(
                    lambda task: None if task.cancelled() else task.exception()
                )
        return operation.task

    async def finish_revocation(self, request: RevocationRequest) -> OperationReceipt:
        if not isinstance(request, RevocationRequest):
            raise TypeError("retained revocation request required")
        with self._lock:
            operation = self.fences.operations.get(request.request_id)
            if operation is None:
                raise ValueError("plugin_revocation_request_unavailable")
        await asyncio.shield(self._start_revocation(operation))
        return operation.status()

    async def disable(self, target: RevocationTarget) -> OperationReceipt:
        return await self.finish_revocation(self.begin_disable(target))

    async def uninstall(self, installation_id: str) -> OperationReceipt:
        return await self.finish_revocation(self.begin_uninstall(installation_id))

    async def _revoke(self, target, operation_id, kind, *, review=None):
        if review is None or operation_id != review.operation_id:
            raise ValueError("original issued review required")
        return await self.finish_revocation(
            self._begin_revocation(target, kind, review=review)
        )

    def revocation_status(self, request: RevocationRequest) -> OperationReceipt:
        """Read current session custody without storage or a replacement mutation."""
        with self._lock:
            return self.fences.operations[request.request_id].status()

    async def lookup_operation(self, identity: str) -> OperationReceipt:
        """Reconcile retained evidence only; unknown identities never start work."""
        return await self._call(lambda: self._coordinator.lookup_operation(identity))

    def live_runs(self) -> tuple[PluginRunOwnership, ...]:
        """Read exact cancellation/completion handles without waiting for storage."""
        with self._lock:
            return tuple(self._live_runs.values())

    def bind_run(
        self,
        entries: Sequence[Mapping[str, Any]],
        run_id: str,
        cancel: Callable[[], object],
        handle_id: str | None = None,
        *,
        parent_run_id: str | None = None,
        component_ceiling: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Bind a real host run before effects, preserving its originating scope."""
        if not run_id:
            raise PluginUnavailable("plugin_run_identity_required")

        def operation():
            ceiling = frozenset(
                (
                    row["plugin_installation_id"],
                    row["plugin_revision"],
                    row["plugin_workspace_id"],
                    row["plugin_component_id"],
                )
                for row in entries
                if row.get("plugin_owned")
                and (
                    component_ceiling is None
                    or row["plugin_component_id"]
                    in component_ceiling.get(row["plugin_installation_id"], ())
                )
            )

            def check_ceiling():
                if run_id in self._terminal_runs:
                    raise PluginUnavailable("plugin_run_terminal")
                if parent_run_id is not None:
                    parent = self._actor_components.get(parent_run_id)
                    if parent is None or not ceiling <= parent:
                        raise PluginUnavailable("plugin_parent_ceiling_unavailable")
                prior = self._actor_components.get(run_id)
                if prior is not None and prior != ceiling:
                    raise PluginUnavailable("plugin_actor_identity_reused")

            with self._lock:
                check_ceiling()
            for row in entries:
                if not row.get("plugin_owned"):
                    continue
                snapshot, _, _ = self._checked(
                    row["name"], row["plugin_admission"], binding=True
                )
                key = (snapshot.installation_id, run_id)
                with self._lock:
                    if self._closing:
                        raise PluginUnavailable("plugin_service_closing")
                    if run_id in self._terminal_runs:
                        raise PluginUnavailable("plugin_run_terminal")
                    existing = self._live_runs.get(key)
                    if existing is not None:
                        if existing.pending_id != snapshot.run_id:
                            raise PluginUnavailable("plugin_run_identity_reused")
                        continue
                    if snapshot.run_id in self._retired:
                        raise PluginUnavailable("plugin_turn_retired")
                    if parent_run_id is None:
                        root = self._root_runs.get(snapshot.run_id)
                        if root is not None and root != run_id:
                            raise PluginUnavailable("plugin_root_identity_reused")
                    elif parent_run_id not in self._bound_runs.get(snapshot.run_id, ()):
                        raise PluginUnavailable("plugin_parent_identity_unavailable")
                    self.fences.check_snapshot(snapshot)
                    self.fences.require_admission(
                        snapshot.installation_id, snapshot.revision_digest
                    )
                # Durable reservation can block; never hold the live seal lock
                # while performing SQLite or filesystem work.
                owner = self._coordinator.owner
                token = owner.reserve_launch(
                    snapshot.run_id,
                    snapshot.installation_id,
                    snapshot.workspace_id,
                    snapshot.revision_digest,
                    root_coverage="qualified_none",
                )
                try:
                    owner.publish_process(
                        token,
                        {
                            "runtime": "console",
                            "run_id": run_id,
                            "handle_id": handle_id,
                            "pending_id": snapshot.run_id,
                        },
                    )
                    with self._lock:
                        if run_id in self._terminal_runs:
                            raise PluginUnavailable("plugin_run_terminal")
                        if snapshot.run_id in self._retired:
                            raise PluginUnavailable("plugin_turn_retired")
                        self.fences.check_snapshot(snapshot)
                        self.fences.require_admission(
                            snapshot.installation_id, snapshot.revision_digest
                        )
                        self._live_runs[key] = PluginRunOwnership(
                            snapshot.installation_id,
                            snapshot.workspace_id,
                            snapshot.revision_digest,
                            snapshot.run_id,
                            run_id,
                            handle_id,
                            token,
                            cancel,
                            threading.Event(),
                            self._pending_custody[snapshot.run_id][0],
                            parent_run_id,
                            snapshot.generations,
                            (
                                None
                                if component_ceiling is None
                                else tuple(
                                    component_ceiling.get(snapshot.installation_id, ())
                                )
                            ),
                        )
                        if parent_run_id is None:
                            self._root_runs[snapshot.run_id] = run_id
                        self._bound_runs.setdefault(snapshot.run_id, set()).add(run_id)
                except BaseException:
                    # bind_run has not returned: the host cannot have launched
                    # this run, so this unused reservation has no live writers.
                    owner.settle_process(token, True)
                    raise

            with self._lock:
                check_ceiling()
                self._actor_components[run_id] = ceiling

        self._call_from_agent(operation)

    def complete_run(self, run_id: str) -> None:
        """Record actual host terminal evidence; coroutine cancellation is insufficient."""
        with self._lock:
            self._terminal_runs.add(run_id)
            records = [
                record for record in self._live_runs.values() if record.run_id == run_id
            ]
            for record in records:
                record.completed.set()
                self._live_runs.pop((record.installation_id, run_id), None)
        if records:
            self._call_from_agent(
                lambda: [
                    self._coordinator.owner.settle_process(record.lease_token, True)
                    for record in records
                ],
                terminal=True,
            )

    async def retire_pending(self, pending_id: str) -> None:
        """Fence reuse of a completed/rejected turn while retaining live children."""
        with self._lock:
            self._retired.add(pending_id)
        # Existing bound runs retain their immutable admissions until the host
        # reports terminal evidence. Merely returning from submit cannot drain.

    async def aclose(self) -> None:
        """Close after Console's owning lifecycle has drained its work."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            if hasattr(self, "_coordinator"):
                self._coordinator.root_usage.closed = True
            if any(
                (operation.task is not None and not operation.task.done())
                or operation.cleanup_tasks
                for operation in self.fences.operations.values()
            ):
                raise PluginUnavailable("plugin_revocation_cleanup_not_drained")
            if any(
                ticket.task is not None and not ticket.task.done()
                for ticket in self.fences.drains.values()
            ):
                raise PluginUnavailable("plugin_revision_work_not_drained")
            if self._live_runs:
                raise PluginUnavailable("plugin_owned_work_not_drained")
            thread = self._thread
        if thread is not None:

            async def finalize():
                if self._ordinary_calls:
                    raise PluginUnavailable("plugin_calls_not_drained")
                if any(
                    not task.done()
                    for task in self._coordinator.owner.acquisition_tasks
                ):
                    raise PluginUnavailable("plugin_acquisition_cleanup_not_drained")
                usage = self._coordinator.root_usage
                if any(
                    operation.task is not None and not operation.task.done()
                    for operation in usage.operations.values()
                ):
                    raise PluginUnavailable("plugin_data_cleanup_not_drained")
                usage.shutdown()

            # Root settlement runs on the owning worker; callers can still submit
            # exact terminal evidence after a refused close, but never new grants.
            if self._ready.done() and self._ready.exception() is None:
                await self._call(finalize, terminal=True)
        with self._lock:
            self._closed = True
            self._catalog = {}
        if thread is not None:
            try:
                loop = await asyncio.wrap_future(self._ready)
            except Exception:  # noqa: BLE001 -- a failed worker still needs joining.
                loop = None
            if loop is not None:
                loop.call_soon_threadsafe(loop.stop)
            await asyncio.to_thread(thread.join)
