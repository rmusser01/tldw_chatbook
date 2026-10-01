"""Owned reviewed installs: durable SQLite success precedes certification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .authority import PluginMarker, snapshot_digest
from .authority_store import PluginAuthorityStore
from .inspection import inspect_package
from .models import PackageInspection
from .package_files import canonical_json, capture_package, materialize_package
from .registry import PluginRegistry
from .review import OperationReceipt, PluginReview, inspection_identity, reinspect
from .revocation import RevocationRequest, RevocationTarget
from .runtime_owner import PluginRuntimeOwner

if TYPE_CHECKING:
    from .data_cleanup import DataRootRef, RootReview


REVIEW_SECONDS = 15 * 60
FREE_RESERVE_BYTES = 100 * 1024 * 1024


class PluginCoordinator:
    """Serialize storage on the caller's persistent, non-UI worker/event loop.

    The app constructs, calls and closes this stack on one worker. This class
    never moves a live SQLite connection between arbitrary executor threads.
    ``progress`` reports durable milestones, without package bodies or secrets.
    It may raise; callers then retain the operation ID and reconcile recovery.
    """

    def __init__(
        self,
        registry: PluginRegistry,
        authority: PluginAuthorityStore,
        owner: PluginRuntimeOwner,
        *,
        fences=None,
        mcp_mapping_owner=None,
    ) -> None:
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("plugin coordinator requires a dedicated worker")
        owner.require_owner(registry.path.parent)
        # Probe SQLite's real affinity now, instead of weakening check_same_thread.
        _ = registry.schema_version
        self.registry, self.authority, self.owner = registry, authority, owner
        self.mcp_mapping_owner = mcp_mapping_owner
        self._thread = threading.get_ident()
        self._loop = asyncio.get_event_loop()
        from .admission import LivePluginFences

        self.fences = fences or LivePluginFences()
        from .revisions import RevisionDrain

        self.revision_drain = RevisionDrain(self.fences, self._drain_inventory)
        from .data_cleanup import RootUsage

        self.root_usage = RootUsage(self)
        owner.root_usage = self.root_usage
        self._revocation_reviews = {}
        self._reviews: dict[str, PluginReview] = {}
        self._published: dict | None = None
        self.progress: Callable[[str], None] | None = None
        self.on_data_change = None

    async def _drain_inventory(self, installation_id: str) -> tuple[str, ...]:
        self._require_worker()
        ownership = getattr(self.mcp_mapping_owner, "connection_ownership", None)
        if ownership is not None:
            with self.fences.live_lock:
                revisions = {
                    ticket.revision_digest
                    for ticket in self.fences.drains.values()
                    if ticket.installation_id == installation_id
                    and ticket.phase == "waiting"
                }
            for revision in revisions:
                await asyncio.wrap_future(
                    ownership._schedule(
                        ownership.close_idle_revision(installation_id, revision)
                    )
                )
        return self.owner.unsettled_tokens(installation_id)

    def _require_worker(self) -> None:
        if (
            threading.get_ident() != self._thread
            or asyncio.get_running_loop() is not self._loop
        ):
            raise RuntimeError("plugin coordinator requires its persistent worker loop")
        self.owner.require_owner(self.registry.path.parent)

    def bootstrap(self, passphrase: str) -> None:
        """Explicit owner-gated setup; never bootstrap over registry authority."""
        self._require_worker()
        prior = self.registry.authority_projection(operation_result=None)
        if prior["installations"] or prior["data_roots"]:
            raise ValueError(
                "existing installation or root custody requires reviewed recovery"
            )
        data_anchor = self.owner.root / "data"
        if os.path.lexists(data_anchor):
            raise ValueError("retained data anchor requires reviewed recovery")
        self.authority.bootstrap(passphrase)
        self._published = self.authority.verify_current()
        self.root_usage.initialize(self._published, fresh=True)

    def reset(self, *, operation_id: str) -> Path | None:
        """Perform an explicitly reviewed plugin-only reset with a retained ID."""
        self._require_worker()
        if self.registry._connection.execute(
            "SELECT 1 FROM data_roots LIMIT 1"
        ).fetchone():
            self.root_usage.require_dirty()
            self.root_usage.recovery.update(
                {
                    row[0]: "root_reset_retained_custody"
                    for row in self.registry._connection.execute(
                        "SELECT root_id FROM data_roots"
                    )
                }
            )
        self._published = None
        self._reviews.clear()
        return self.authority.reset(operation_id=operation_id)

    def _issue_review(self, review, *, live_operation=None):
        from dataclasses import replace

        nonce = secrets.token_hex(32)
        live = None
        if live_operation is not None:
            self.fences.require_current(live_operation)
            if time.monotonic() >= live_operation.expires_at:
                raise ValueError("plugin_revocation_request_expired")
            nonce = "".join(live_operation.operation_id.split(".")[1:])
            live = {
                "request_id": live_operation.operation_id,
                "target": {
                    "installation_id": live_operation.target.installation_id,
                    "workspace_id": live_operation.target.workspace_id,
                    "everywhere": live_operation.target.everywhere,
                    "global_default": live_operation.target.global_default,
                },
                "scope_versions": [
                    [list(scope), version]
                    for scope, version in live_operation.scope_versions
                ],
            }
            review = replace(
                review, expires_at=min(review.expires_at, live_operation.expires_at)
            )
        binding = {
            "marker": review.authority_marker.model_dump(),
            "authority_digest": hashlib.sha256(
                review.authority_json.encode()
            ).hexdigest(),
            "review_token": review.token,
            "installation_id": review.installation_id,
            "alias": review.alias,
            "inspection_digest": hashlib.sha256(
                inspection_identity(review.inspection).encode()
            ).hexdigest(),
            "selection": list(review.selection),
            "workspace_id": review.workspace_id,
            "kind": review.kind,
            "intent": review.intent,
            "previous_revision": review.previous_revision,
            "drain_token": review.drain_token,
            "rollback": review.rollback,
            "retired_revisions": list(review.retired_revisions),
            "retirement_json": review.retirement_json,
            "mappings_json": getattr(review, "mappings_json", "[]"),
            "live": live,
        }
        result = {
            "installation_id": review.installation_id,
            "kind": review.kind,
            "revision_digest": review.inspection.effective_digest,
            "result": "committed",
        }
        if review.kind == "retain":
            result["retired_revisions"] = json.loads(review.retirement_json)
        operation_id = self.authority.issue_operation_id(
            review.authority_marker.generation + 1,
            hashlib.sha256(canonical_json(binding).encode()).hexdigest(),
            result,
            nonce,
        )
        return replace(review, operation_id=operation_id)

    def review(
        self,
        inspection: PackageInspection,
        *,
        selection: tuple[str, ...],
        workspace_id: str | None,
    ) -> PluginReview:
        """Capture a new local install identity and all current authority inputs."""
        self._require_worker()
        baseline = self.published_snapshot()
        if inspection.rejected or inspection.effective_digest is None:
            raise ValueError("package cannot be installed")
        if (
            type(selection) is not tuple
            or len(set(selection)) != len(selection)
            or not set(selection) <= inspection.inventory.keys()
        ):
            raise ValueError("invalid reviewed selection")
        if workspace_id is not None:
            from .authority import Activation

            Activation(
                installation_id="review", workspace_id=workspace_id, intent="disabled"
            )
        self._verify_reviewed_package(inspection)
        installation_id = uuid4().hex
        source = Path(inspection.materialized_identity or inspection.source_identity)
        package_name = capture_package(source).document(inspection.root_manifest)[
            "name"
        ]
        aliases = {row.get("alias") for row in baseline["installations"]}
        alias = package_name
        if alias in aliases:
            for length in range(8, len(installation_id) + 1, 4):
                alias = f"{package_name}-{installation_id[:length]}"
                if alias not in aliases:
                    break
            else:
                raise ValueError("installation alias collision")
        review = PluginReview(
            installation_id=installation_id,
            alias=alias,
            inspection=inspection,
            selection=tuple(sorted(selection)),
            workspace_id=workspace_id,
            authority_marker=self.authority.load_marker(),
            authority_json=canonical_json(baseline),
            token=uuid4().hex,
            expires_at=time.monotonic() + REVIEW_SECONDS,
        )
        review = self._issue_review(review)
        self._reviews[review.token] = review
        return review

    def review_revision(
        self,
        installation_id: str,
        inspection: PackageInspection,
        *,
        rollback: bool = False,
    ) -> PluginReview:
        """Review a replacement against current policy without closing admission."""
        from dataclasses import replace

        from .revisions import preserve_selection

        prior = self._review_existing(installation_id, kind="update")
        if inspection.rejected or inspection.effective_digest is None:
            raise ValueError("package cannot be installed")
        if inspection.effective_digest == prior.inspection.effective_digest:
            raise ValueError("revision is already current")
        self._verify_reviewed_package(inspection)
        available = frozenset(
            key
            for key, component in inspection.inventory.items()
            if component.support in {"supported", "adapted"}
            and key in prior.inspection.inventory
            and prior.inspection.inventory[key].support in {"supported", "adapted"}
        )
        token = self.revision_drain.reserve(
            installation_id,
            prior.inspection.effective_digest,
            review_token=prior.token,
            expires_at=prior.expires_at,
        )
        review = replace(
            prior,
            inspection=inspection,
            selection=tuple(
                sorted(preserve_selection(frozenset(prior.selection), available))
            ),
            previous_revision=prior.inspection.effective_digest,
            drain_token=token,
            rollback=rollback,
        )
        review = self._issue_review(review)
        self._reviews[review.token] = review
        return review

    def review_rollback(
        self, installation_id: str, revision_digest: str
    ) -> PluginReview:
        """Rollback reviews retained bytes against today's authority, never old grants."""
        from .recovery import retained_inspections

        inspection = retained_inspections(self.published_snapshot()).get(
            (installation_id, revision_digest)
        )
        if inspection is None:
            raise ValueError("rollback revision unavailable")
        return self.review_revision(installation_id, inspection, rollback=True)

    async def apply_revision(
        self, review: PluginReview, operation_id: str
    ) -> OperationReceipt:
        """Activate the reviewed fence and retain application after a waiter leaves."""
        self._require_worker()
        if review.kind != "update" or self._reviews.get(review.token) != review:
            raise ValueError("invalid revision review")
        ticket = self.revision_drain.activate(
            review.drain_token, review=review, operation_id=operation_id
        )
        if (
            ticket.task is None
            or ticket.task.done()
            and ticket.task.exception() is not None
        ):

            async def apply():
                while True:
                    # Storage may block; never hold the admission/cancellation lock.
                    unsettled = await self._drain_inventory(review.installation_id)
                    self.revision_drain.record_inventory(ticket.token, unsettled)
                    with self.fences.live_lock:
                        if ticket.phase == "cancelled":
                            raise ValueError("plugin_drain_cancelled_or_expired")
                        ready = not self.revision_drain.blockers(ticket.token)
                        if ready:
                            ticket.phase = "committing"
                    if not ready:
                        await asyncio.sleep(0.01)
                        continue
                    try:
                        receipt = await self.commit(review, operation_id)
                    except BaseException:
                        with self.fences.live_lock:
                            ticket.phase = "waiting"
                        raise
                    with self.fences.live_lock:
                        ticket.phase = "complete"
                    from dataclasses import replace

                    try:
                        cleanup = await self.retain_revisions(review.installation_id)
                        if cleanup.cleanup_pending:
                            receipt = replace(
                                receipt,
                                cleanup_pending=True,
                                cleanup_errors=cleanup.cleanup_errors,
                            )
                    except (OSError, ValueError, PermissionError):
                        receipt = replace(
                            receipt,
                            cleanup_pending=True,
                            cleanup_errors=("retention_pending",),
                        )
                    return receipt

            ticket.task = asyncio.create_task(apply())
            ticket.task.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        return await asyncio.shield(ticket.task)

    def review_trust(self, installation_id: str) -> PluginReview:
        """Review the exact currently installed bytes for explicit trust."""
        return self._review_existing(installation_id, kind="trust")

    def review_activation(
        self, installation_id: str, *, workspace_id: str | None, intent: str
    ) -> PluginReview:
        """Review one explicit activation scope without changing selection."""
        from .authority import Activation

        if workspace_id is not None and (
            not isinstance(workspace_id, str)
            or not workspace_id.strip()
            or workspace_id in {"global", "workspace-default"}
        ):
            raise ValueError("activation override requires a named workspace")
        Activation(
            installation_id=installation_id,
            workspace_id=workspace_id or "global",
            intent=intent,
        )
        if workspace_id is None and intent == "inherit":
            raise ValueError("global activation cannot inherit")
        return self._review_existing(
            installation_id, kind="activate", workspace_id=workspace_id, intent=intent
        )

    def _review_existing(
        self,
        installation_id: str,
        *,
        kind: str,
        workspace_id: str | None = None,
        intent: str | None = None,
        live_operation=None,
        retired_revisions: tuple[str, ...] = (),
        retirement_json: str | None = None,
        mappings_json: str = "[]",
    ) -> PluginReview:
        from .recovery import retained_inspections

        self._require_worker()
        baseline = self.published_snapshot()
        installed = next(
            (
                row
                for row in baseline["installations"]
                if row["installation_id"] == installation_id
            ),
            None,
        )
        if installed is None:
            raise ValueError("installation unavailable")
        inspection = retained_inspections(baseline)[
            (installation_id, installed["revision_digest"])
        ]
        review = PluginReview(
            installation_id=installation_id,
            inspection=inspection,
            selection=tuple(
                row["component_id"]
                for row in baseline["selections"]
                if row["installation_id"] == installation_id
                and row["revision_digest"] == installed["revision_digest"]
                and row["selected"]
            ),
            workspace_id=workspace_id,
            authority_marker=self.authority.load_marker(),
            authority_json=canonical_json(baseline),
            token=uuid4().hex,
            expires_at=time.monotonic() + REVIEW_SECONDS,
            kind=kind,
            intent=intent,
            alias=installed.get("alias"),
            retired_revisions=retired_revisions,
            retirement_json=retirement_json,
            mappings_json=mappings_json,
        )
        review = self._issue_review(review, live_operation=live_operation)
        self._reviews[review.token] = review
        return review

    def review_configuration(
        self,
        installation_id: str,
        *,
        connections: dict[str, str],
        tools: dict[str, tuple[str, ...]] | None = None,
    ) -> PluginReview:
        """Review complete current owner references; never accept caller-made grants."""
        from tldw_chatbook.MCP.local_control_service import LocalMCPControlService

        from .recovery import retained_inspections

        self._require_worker()
        owner = self.mcp_mapping_owner
        if not isinstance(owner, LocalMCPControlService):
            raise PermissionError("plugin_mapping_owner_unavailable")
        snapshot = self.published_snapshot()
        installed = next(
            row
            for row in snapshot["installations"]
            if row["installation_id"] == installation_id
        )
        inspection = retained_inspections(snapshot)[
            (installation_id, installed["revision_digest"])
        ]
        selection = {
            row["component_id"]
            for row in snapshot["selections"]
            if row["installation_id"] == installation_id
            and row["revision_digest"] == installed["revision_digest"]
            and row["selected"]
        }
        if (
            not connections
            or len(connections) > 512
            or set(tools or {}) - connections.keys()
        ):
            raise ValueError("plugin_mapping_invalid")
        mappings = []
        for component_id, profile_id in sorted(connections.items()):
            if component_id not in selection:
                raise PermissionError("plugin_component_not_selected")
            mapping = owner.capture_connection_mapping(
                installation_id=installation_id,
                mapping_id=component_id,
                inspection=inspection,
                component_id=component_id,
                profile_id=profile_id,
            )
            mappings.append(mapping)
            for name in sorted(set((tools or {}).get(component_id, ()))):
                mappings.append(owner.capture_tool_mapping(mapping, inspection, name))
        return self._review_existing(
            installation_id, kind="configure", mappings_json=canonical_json(mappings)
        )

    async def retain_revisions(self, installation_id: str) -> OperationReceipt:
        """Compact references and reconcile retained authenticated cleanup custody."""
        from dataclasses import replace

        from .recovery import retained_inspections
        from .retention import (
            cleanup_retained_operation,
            eligible_revisions,
            transition_inventory,
        )

        if any(item.phase == "recovery_required" for item in await self.recover()):
            raise PermissionError("plugin recovery required")
        prior_result = None
        for evidence in transition_inventory(self.authority):
            result = evidence.snapshot["operation_result"]
            if (
                evidence.committed
                and result["kind"] == "retain"
                and result["installation_id"] == installation_id
            ):
                errors = cleanup_retained_operation(self, evidence)
                prior_result = OperationReceipt(
                    result["operation_id"],
                    "complete",
                    True,
                    cleanup_pending=bool(errors),
                    cleanup_errors=errors,
                )
                if errors:
                    return prior_result
        baseline = self.published_snapshot()
        candidates = eligible_revisions(self, installation_id)[:1000]
        if not candidates:
            return prior_result or OperationReceipt(None, "nothing_eligible", False)
        inspections = retained_inspections(baseline)
        rows = []
        root_info = self.owner.root.lstat()
        for digest in candidates:
            path = Path(inspections[(installation_id, digest)].materialized_identity)
            info, anchor = path.lstat(), path.parent.lstat()
            rows.append(
                {
                    "revision_digest": digest,
                    "materialized_identity": str(path),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "root_device": root_info.st_dev,
                    "root_inode": root_info.st_ino,
                    "anchor_device": anchor.st_dev,
                    "anchor_inode": anchor.st_ino,
                }
            )
        review = self._review_existing(
            installation_id,
            kind="retain",
            retired_revisions=candidates,
            retirement_json=canonical_json(rows),
        )
        result = await self.commit(review, review.operation_id)
        errors = cleanup_retained_operation(
            self, self.authority.verify_transition(review.operation_id)
        )
        return replace(result, cleanup_pending=bool(errors), cleanup_errors=errors)

    def _apply_review(
        self, cursor, review: PluginReview, retained: PackageInspection
    ) -> None:
        if review.kind == "root_data":
            from .data_cleanup import apply_root_review

            apply_root_review(self, cursor, review)
            return
        if review.kind == "retain":
            for digest in review.retired_revisions:
                for table in (
                    "revision_trust",
                    "selections",
                    "components",
                    "revisions",
                ):
                    cursor.execute(
                        f"DELETE FROM {table} WHERE installation_id=? AND revision_digest=?",
                        (review.installation_id, digest),
                    )
                cursor.execute(
                    "DELETE FROM receipts WHERE receipt_id=?",
                    ("revision:" + review.installation_id + ":" + digest,),
                )
            return
        if review.kind == "configure":
            mappings = json.loads(review.mappings_json)
            for mapping in mappings:
                self.mcp_mapping_owner.validate_connection_mapping(mapping, retained)
            cursor.execute(
                "DELETE FROM mappings WHERE installation_id=?",
                (review.installation_id,),
            )
            for mapping in mappings:
                cursor.execute(
                    "INSERT INTO mappings VALUES (?, ?, ?)",
                    (
                        review.installation_id,
                        mapping["mapping_id"],
                        canonical_json(mapping),
                    ),
                )
            cursor.execute(
                "INSERT INTO authority_generations VALUES (?, 'installation', '', 1, 0) ON CONFLICT(installation_id, scope_kind, workspace_id) DO UPDATE SET generation=generation+1",
                (review.installation_id,),
            )
            return
        if review.kind == "install":
            self.registry.insert_installation(
                cursor,
                review.installation_id,
                retained,
                review.selection,
                alias=review.alias,
            )
            return
        if review.kind == "update":
            revision = retained.effective_digest
            cursor.execute(
                "INSERT INTO revisions VALUES (?, ?, ?) ON CONFLICT(installation_id, revision_digest) DO NOTHING",
                (review.installation_id, revision, retained.model_dump_json()),
            )
            for component in retained.inventory.values():
                cursor.execute(
                    "INSERT INTO components VALUES (?, ?, ?, ?) ON CONFLICT(installation_id, revision_digest, component_id) DO NOTHING",
                    (
                        review.installation_id,
                        revision,
                        component.component_id,
                        component.model_dump_json(),
                    ),
                )
                cursor.execute(
                    "INSERT INTO selections VALUES (?, ?, ?, ?) ON CONFLICT(installation_id, revision_digest, component_id) DO UPDATE SET selected=excluded.selected",
                    (
                        review.installation_id,
                        revision,
                        component.component_id,
                        component.component_id in review.selection,
                    ),
                )
            cursor.execute(
                "INSERT INTO revision_trust VALUES (?, ?, 1) ON CONFLICT(installation_id, revision_digest) DO UPDATE SET reviewed=1",
                (review.installation_id, revision),
            )
            cursor.execute(
                "UPDATE installations SET revision_digest=? WHERE installation_id=?",
                (revision, review.installation_id),
            )
            scope, workspace = "installation", ""
        elif review.kind in {"revoke", "uninstall"}:
            operation = self._revocation_reviews[review.token]
            target = operation.target
            if review.kind == "uninstall":
                cursor.execute(
                    "DELETE FROM receipts WHERE operation_id=?",
                    ("revision:" + review.installation_id,),
                )
                cursor.execute(
                    "INSERT INTO tombstones VALUES (?, ?, ?)",
                    (
                        review.installation_id,
                        review.authority_marker.generation + 1,
                        review.operation_id,
                    ),
                )
                for table in (
                    "selections",
                    "components",
                    "revision_trust",
                    "revisions",
                    "activation",
                    "sources",
                    "mappings",
                    "authority_generations",
                    "installations",
                ):
                    cursor.execute(
                        f"DELETE FROM {table} WHERE installation_id=?",
                        (review.installation_id,),
                    )
                return
            scope, workspace = target.scope[1:]
            if target.everywhere:
                cursor.execute(
                    "UPDATE installations SET activation_default=0 WHERE installation_id=?",
                    (review.installation_id,),
                )
                cursor.execute(
                    "UPDATE activation SET intent='disabled' WHERE installation_id=?",
                    (review.installation_id,),
                )
            elif target.global_default:
                cursor.execute(
                    "UPDATE installations SET activation_default=0 WHERE installation_id=?",
                    (review.installation_id,),
                )
            else:
                cursor.execute(
                    "INSERT INTO activation VALUES (?, ?, 'disabled') ON CONFLICT(installation_id, workspace_id) DO UPDATE SET intent='disabled'",
                    (review.installation_id, workspace),
                )
        elif review.kind == "trust":
            cursor.execute(
                "UPDATE revision_trust SET reviewed=1 WHERE installation_id=? AND revision_digest=?",
                (review.installation_id, retained.effective_digest),
            )
            scope, workspace = "installation", ""
        elif review.kind == "activate":
            workspace = review.workspace_id or ""
            scope = "workspace" if review.workspace_id is not None else "global_default"
            if review.workspace_id is None:
                cursor.execute(
                    "UPDATE installations SET activation_default=? WHERE installation_id=?",
                    (review.intent == "enabled", review.installation_id),
                )
            else:
                cursor.execute(
                    "INSERT INTO activation VALUES (?, ?, ?) ON CONFLICT(installation_id, workspace_id) DO UPDATE SET intent=excluded.intent",
                    (review.installation_id, workspace, review.intent),
                )
        else:
            raise ValueError("invalid review operation")
        cursor.execute(
            "INSERT INTO authority_generations VALUES (?, ?, ?, 1, 0) ON CONFLICT(installation_id, scope_kind, workspace_id) DO UPDATE SET generation=generation+1",
            (review.installation_id, scope, workspace),
        )

    @staticmethod
    def _verify_reviewed_package(inspection: PackageInspection) -> None:
        if inspection.materialized_identity:
            reinspect(inspection, Path(inspection.materialized_identity))
        else:
            current = inspect_package(
                Path(inspection.source_identity), dialect=inspection.dialect
            )
            if inspection_identity(current) != inspection_identity(inspection):
                raise ValueError("stale review package inputs")

    def published_snapshot(self) -> dict:
        """Return authenticated projection only while all publication gates agree."""
        self._require_worker()
        if self._published is None:
            raise PermissionError("plugin publication fenced; recovery required")
        current = self.authority.verify_current()
        projection = self.registry.authority_projection(
            operation_result=current["operation_result"]
        )
        if current != self._published or projection != current:
            self._published = None
            raise PermissionError(
                "plugin publication differs from authenticated authority"
            )
        return current

    def _milestone(self, phase: str) -> None:
        self._require_worker()
        if self.progress is not None:
            self.progress(phase)

    def _materialize(self, review: PluginReview) -> PackageInspection:
        root = self.owner.root / "packages"
        if review.kind == "update":
            baseline = json.loads(review.authority_json)
            prior = next(
                (
                    row
                    for row in baseline["revisions"]
                    if row["installation_id"] == review.installation_id
                    and row["revision_digest"] == review.inspection.effective_digest
                ),
                None,
            )
            if prior is not None:
                reinspect(review.inspection, Path(prior["materialized_identity"]))
                return review.inspection
            root = self.owner.root / "packages-revisions" / review.installation_id
        # Create owned containers through descriptors, before any path-based
        # materialization. A linked replacement parent must not create directories
        # in an unrelated tree even when the later quota check would refuse it.
        self.owner.require_owner(self.owner.root)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        opened = [os.open(self.owner.root, flags)]
        try:
            self.owner.require_owner(self.owner.root)
            named, opened_root = self.owner.root.lstat(), os.fstat(opened[0])
            if (named.st_dev, named.st_ino) != (opened_root.st_dev, opened_root.st_ino):
                raise PermissionError("plugin root changed")
            for part in root.relative_to(self.owner.root).parts:
                parent_fd = opened[-1]
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                except FileExistsError:
                    pass
                opened.append(os.open(part, flags, dir_fd=parent_fd))
                os.fsync(parent_fd)
        finally:
            for fd in reversed(opened):
                os.close(fd)
        destination = root / (
            review.inspection.effective_digest
            if review.kind == "update"
            else review.installation_id
        )
        source = Path(
            review.inspection.materialized_identity or review.inspection.source_identity
        )
        # F1 bounds captures; reserve capacity before writing and let actual
        # filesystem failures propagate without reporting commitment.
        capture = capture_package(source)
        if capture.errors:
            raise ValueError("stale or unsafe package capture")
        size = sum(len(member.data) for member in capture.files.values())
        from .retention import require_capacity

        require_capacity(self.owner.root, 0 if os.path.lexists(destination) else size)
        retained = review.inspection.model_copy(
            update={"materialized_identity": str(destination)}
        )
        if not os.path.lexists(destination):
            copied = materialize_package(source, destination)
            if copied.content_digest != retained.content_digest:
                raise ValueError("stale package during materialization")
        reinspect(retained, destination)
        # F1 copies/validates; the coordinator owns durability before preparation.
        for directory, _, filenames in os.walk(destination, topdown=False):
            for name in filenames:
                fd = os.open(Path(directory) / name, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        for directory in (root, self.owner.root):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return retained

    async def commit(self, review: PluginReview, operation_id: str) -> OperationReceipt:
        """Revalidate, prepare, durably commit, certify, mark, then publish."""
        from .revocation import revocation_for_review

        if operation_id != review.operation_id or not operation_id:
            raise ValueError("operation ID belongs to a different review")
        if review.kind == "update":
            ticket = self.revision_drain.ticket(review.drain_token)
            if (
                ticket.phase != "committing"
                or ticket.review_token != review.token
                or ticket.operation_id != operation_id
            ):
                return await self.apply_revision(review, operation_id)
        target = revocation_for_review(review)
        if target is not None and review.token not in self._revocation_reviews:
            return await self.finish_revocation(
                self._begin_revocation(target, "activate", review=review)
            )
        self._require_worker()
        if review.kind == "root_data":
            if review.token not in self.root_usage.committing:
                raise PermissionError("root_lifecycle_entry_required")
            if not self.root_usage.dirty:
                raise PermissionError("root_dirty_checkpoint_required")
            if review.phase in {"waiting", "deleting"} and any(
                row["root_id"] not in self.root_usage.root_fences
                for row in review.targets
            ):
                raise PermissionError("root_live_fence_required")
        if review.kind == "activate" and target is None:
            self.fences.require_enable_reconciled(
                review.installation_id, review.workspace_id
            )
        # Validate the ID with the same closed schema used by protected authority.
        PluginMarker(
            generation=1, operation_id=operation_id, recovery_snapshot_digest="0" * 64
        )
        if self._reviews.get(review.token) != review:
            raise ValueError("invalid review token or changed review")
        receipts = await self.recover()
        prior = next(
            (item for item in receipts if item.operation_id == operation_id), None
        )
        if prior is not None:
            marker = self.authority.load_marker()
            if (
                prior.committed
                and marker is not None
                and marker.operation_id == operation_id
            ):
                result = self.authority.verify_current()["operation_result"]
            else:
                result = self.authority.verify_transition(operation_id).snapshot[
                    "operation_result"
                ]
            if (
                result["installation_id"] != review.installation_id
                or result["revision_digest"]
                != (
                    None
                    if review.kind == "root_data"
                    else review.inspection.effective_digest
                )
                or result["kind"] != review.kind
            ):
                raise ValueError("operation ID belongs to a different review")
            if review.kind in {"trust", "activate"}:
                evidence = self.authority.verify_transition(operation_id)
                if evidence.old != review.authority_marker:
                    raise ValueError("operation ID belongs to a different review")
                baseline = json.loads(review.authority_json)
                scope = (
                    "installation"
                    if review.kind == "trust"
                    else (
                        "global_default" if review.workspace_id is None else "workspace"
                    )
                )
                workspace = review.workspace_id or ""

                def scoped_generation(snapshot):
                    return next(
                        (
                            row["generation"]
                            for row in snapshot["authority_generations"]
                            if row["installation_id"] == review.installation_id
                            and row["scope_kind"] == scope
                            and row["workspace_id"] == workspace
                        ),
                        0,
                    )

                if (
                    scoped_generation(evidence.snapshot)
                    != scoped_generation(baseline) + 1
                ):
                    raise ValueError("operation ID belongs to a different review")
                if review.kind == "activate":
                    if review.workspace_id is None:
                        intent = next(
                            row["activation_default"]
                            for row in evidence.snapshot["installations"]
                            if row["installation_id"] == review.installation_id
                        )
                        matches = intent == (review.intent == "enabled")
                    else:
                        intent = next(
                            (
                                row["intent"]
                                for row in evidence.snapshot["activation"]
                                if row["installation_id"] == review.installation_id
                                and row["workspace_id"] == review.workspace_id
                            ),
                            "inherit",
                        )
                        matches = intent == review.intent
                    if not matches:
                        raise ValueError("operation ID belongs to a different review")
                    if prior.committed and prior.phase == "complete" and target is None:
                        current = self.published_snapshot()

                        def resume_authority(snapshot):
                            installed = next(
                                (
                                    row
                                    for row in snapshot["installations"]
                                    if row["installation_id"] == review.installation_id
                                ),
                                None,
                            )
                            if installed is None:
                                return None
                            scopes = {("installation", ""), (scope, workspace)}
                            if review.intent == "inherit":
                                scopes.add(("global_default", ""))
                            return (
                                installed["revision_digest"],
                                installed.get("alias"),
                                (
                                    installed["activation_default"]
                                    if review.workspace_id is None
                                    or review.intent == "inherit"
                                    else None
                                ),
                                [
                                    row
                                    for row in snapshot["activation"]
                                    if row["installation_id"] == review.installation_id
                                    and row["workspace_id"] == review.workspace_id
                                ],
                                [
                                    row
                                    for row in snapshot["authority_generations"]
                                    if row["installation_id"] == review.installation_id
                                    and (row["scope_kind"], row["workspace_id"])
                                    in scopes
                                ],
                            )

                        # A historical receipt is not a fresh enable. Only the
                        # still-current scoped authority may release its fence;
                        # unrelated namespace publications do not prevent resume.
                        if resume_authority(current) == resume_authority(
                            evidence.snapshot
                        ):
                            with self.fences.live_lock:
                                self.fences.require_enable_reconciled(
                                    review.installation_id, review.workspace_id
                                )
                                self.fences.reconcile_enable(
                                    review.installation_id, review.workspace_id
                                )
            return prior
        if any(item.phase == "recovery_required" for item in receipts):
            raise PermissionError("plugin recovery required")
        if time.monotonic() >= review.expires_at:
            raise ValueError("stale expired review")
        if (
            self.authority.load_marker() != review.authority_marker
            or canonical_json(self.published_snapshot()) != review.authority_json
        ):
            raise ValueError("stale review authority")
        from .retention import prune_transition_history

        prune_transition_history(self)
        if review.kind != "root_data":
            self._verify_reviewed_package(review.inspection)
        if review.kind == "retain":
            from .retention import eligible_revisions

            if not set(review.retired_revisions) <= set(
                eligible_revisions(self, review.installation_id)
            ):
                raise ValueError("retention ownership changed")
        self._published = None
        retained = (
            self._materialize(review)
            if review.kind in {"install", "update"}
            else None
            if review.kind == "root_data"
            else review.inspection
        )
        self._milestone("materialized")
        if review.token in self._revocation_reviews:
            self.fences.require_current(self._revocation_reviews[review.token])
        result = {
            "operation_id": operation_id,
            "installation_id": review.installation_id,
            "kind": review.kind,
            "revision_digest": (
                retained.effective_digest if retained is not None else None
            ),
            "result": "committed",
        }
        if review.kind == "root_data":
            result["root_result"] = review.result
        if review.kind == "retain":
            result["retired_revisions"] = json.loads(review.retirement_json)
        with self.registry.transaction() as cursor:
            self._apply_review(cursor, review, retained)
            snapshot = self.registry.authority_projection(operation_result=result)
            new = PluginMarker(
                generation=review.authority_marker.generation + 1,
                operation_id=operation_id,
                recovery_snapshot_digest=snapshot_digest(snapshot),
            )
            self.authority.prepare(snapshot, review.authority_marker, new)
            self._milestone("prepared")
            self.registry.write_operation(cursor, result, phase="committed")
        if review.token in self._revocation_reviews:
            self._revocation_reviews[review.token].receipt = OperationReceipt(
                operation_id, "recovery_required", True
            )
        self._milestone("registry_committed")
        # This is the only production certificate issuance call. The guarded
        # registry context above has returned after its actual SQLite COMMIT.
        self.authority.certify_commit(review.authority_marker, new)
        self._milestone("certified")
        if self.registry.authority_projection(operation_result=result) != snapshot:
            raise ValueError("registry changed before marker publication")
        self.authority.advance_marker(review.authority_marker, new)
        self._milestone("marker_advanced")
        with self.registry.transaction() as cursor:
            self.registry.write_operation(cursor, result, phase="complete")
        self._published = snapshot
        if review.kind == "activate" and target is None:
            self.fences.reconcile_enable(review.installation_id, review.workspace_id)
        if review.kind == "root_data" and self.on_data_change is not None:
            self.on_data_change()
        self._milestone("published")
        return OperationReceipt(operation_id, "complete", True)

    def review_data_creation(
        self, installation_id: str, *, workspace_id: str | None = None
    ):
        from .data_cleanup import review_creation

        self._require_worker()
        return review_creation(self, installation_id, workspace_id)

    async def create_data(self, review: RootReview, operation_id: str) -> DataRootRef:
        from .data_cleanup import create_data

        self._require_worker()
        return await create_data(self, review, operation_id)

    def review_data_deletion(self, roots: tuple[DataRootRef, ...]) -> RootReview:
        from .data_cleanup import review_deletion

        self._require_worker()
        return review_deletion(self, roots)

    def review_data_attachment(
        self, roots: tuple[DataRootRef, ...], installation_id: str | None
    ) -> RootReview:
        from .data_cleanup import review_deletion

        self._require_worker()
        return review_deletion(self, roots, action="attach", attachment=installation_id)

    async def delete_data(
        self, roots: tuple[DataRootRef, ...], operation_id: str
    ) -> OperationReceipt:
        from .data_cleanup import delete_data

        self._require_worker()
        return await delete_data(self, roots, operation_id)

    def review_data_reconciliation(self, roots, *, confirm_quiescence):
        from .data_cleanup import review_reconciliation

        self._require_worker()
        return review_reconciliation(self, roots, confirm_quiescence)

    async def reconcile_data(
        self, review: RootReview, operation_id: str
    ) -> tuple[DataRootRef, ...]:
        from .data_cleanup import reconcile_data

        self._require_worker()
        return await reconcile_data(self, review, operation_id)

    def review_data_cleanup_resume(self, roots: tuple[DataRootRef, ...]) -> RootReview:
        from .data_cleanup import review_resume

        self._require_worker()
        return review_resume(self, roots)

    def cancel_data_work(self, operation_id: str) -> None:
        self.root_usage.cancel_work(operation_id)

    async def cancel_data_deletion(self, operation_id: str) -> None:
        from .data_cleanup import cancel_data_deletion

        self._require_worker()
        await cancel_data_deletion(self, operation_id)

    def begin_disable(self, target: RevocationTarget) -> RevocationRequest:
        return self._begin_revocation(target, "revoke")

    def begin_uninstall(self, installation_id: str) -> RevocationRequest:
        return self._begin_revocation(
            RevocationTarget(installation_id, None, True), "uninstall"
        )

    def _begin_revocation(self, target, kind, *, review=None):
        self._require_worker()
        request = self.fences.request(target, kind, review)
        operation = self.fences.operations[request.request_id]
        self._start_revocation(operation)
        return request

    def _start_revocation(self, operation):
        self.fences.require_current(operation)
        if operation.task is None or operation.task.done():
            operation.task = asyncio.create_task(
                self._revoke(
                    operation.target,
                    operation.operation_id,
                    operation.kind,
                    review=operation.review,
                )
            )
            operation.task.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        return operation.task

    async def finish_revocation(self, request: RevocationRequest) -> OperationReceipt:
        self._require_worker()
        if (
            not isinstance(request, RevocationRequest)
            or request.request_id not in self.fences.operations
        ):
            raise ValueError("plugin_revocation_request_unavailable")
        operation = self.fences.operations[request.request_id]
        await asyncio.shield(self._start_revocation(operation))
        return operation.status()

    async def disable(self, target: RevocationTarget) -> OperationReceipt:
        return await self.finish_revocation(self.begin_disable(target))

    async def uninstall(self, installation_id: str) -> OperationReceipt:
        return await self.finish_revocation(self.begin_uninstall(installation_id))

    async def lookup_operation(self, identity: str) -> OperationReceipt:
        self._require_worker()
        import re

        nonce = None
        if re.fullmatch(r"lr1\.[0-9a-f]{32}\.[0-9a-f]{32}", identity):
            nonce = "".join(identity.split(".")[1:])
        receipts = await self.recover()
        if any(item.phase == "recovery_required" for item in receipts):
            return OperationReceipt(
                None, "recovery_required", False, request_id=identity if nonce else None
            )
        matches = []
        marker = self.authority.load_marker()
        for receipt in receipts:
            result = (
                self.authority.verify_current()["operation_result"]
                if receipt.operation_id == marker.operation_id
                else self.authority.verify_transition(receipt.operation_id).snapshot[
                    "operation_result"
                ]
            )
            if result["kind"] == "root_data":
                from dataclasses import replace

                group = result["root_result"]["group_id"]
                pending_roots = [
                    row
                    for row in self.published_snapshot()["data_roots"]
                    if (row.get("cleanup") or {}).get("group_id") == group
                ]
                receipt = replace(
                    receipt,
                    phase=(
                        pending_roots[0]["cleanup"]["phase"]
                        if pending_roots
                        else result["root_result"]["phase"]
                    ),
                    cleanup_pending=bool(pending_roots),
                )
            if result["kind"] == "retain":
                from dataclasses import replace

                from .retention import cleanup_retained_operation

                receipt = replace(
                    receipt,
                    cleanup_pending=bool(
                        cleanup_retained_operation(
                            self,
                            self.authority.verify_transition(receipt.operation_id),
                            remove=False,
                        )
                    ),
                )
            if receipt.operation_id == identity or (
                nonce
                and result["operation_id"].startswith("pi1.")
                and self.authority.verify_operation_id(
                    result["operation_id"], result
                ).nonce
                == nonce
            ):
                matches.append(receipt)
        if len(matches) > 1:
            raise ValueError("ambiguous retained request identity")
        if matches:
            from dataclasses import replace

            return replace(matches[0], request_id=identity if nonce else None)
        return OperationReceipt(
            None,
            "unavailable_or_expired",
            False,
            request_id=identity if nonce else None,
        )

    async def _revoke(self, target, operation_id, kind, *, review=None):
        from dataclasses import replace

        from .revocation import RevocationConflict, RevocationFailure

        operation = self.fences.begin(target, operation_id, kind, review=review)
        self._require_worker()
        self.fences.require_current(operation)
        operation_id = operation.durable_id
        if operation.review is not None:
            self._revocation_reviews[operation.review.token] = operation
        try:
            with self.fences.live_lock:
                known_tokens = {
                    record.lease_token
                    for record in (*operation.records, *self.fences.runs.values())
                }
            operation.unresolved_tokens = tuple(
                token
                for token in self.owner.unsettled_tokens(
                    target.installation_id,
                    (
                        None
                        if target.everywhere or target.global_default
                        else target.workspace_id
                    ),
                )
                if token not in known_tokens
            )
            operation.runtime_observed = True
            if operation.receipt.phase != "complete":
                receipts = await self.recover()
                self.fences.require_current(operation)
                prior = next(
                    (item for item in receipts if item.operation_id == operation_id),
                    None,
                )
                if prior is not None:
                    # Custody pins request identity within this live owner. Cross-session
                    # issuance/legacy retry migration belongs to F7 (R32).
                    evidence = self.authority.verify_transition(operation_id)
                    result = evidence.snapshot["operation_result"]
                    if (
                        result["installation_id"] != target.installation_id
                        or result["kind"] != kind
                    ):
                        raise RevocationConflict(
                            "operation ID belongs to a different review"
                        )
                    if operation.review is None:
                        raise ValueError(
                            "plugin retry requires original session custody"
                        )
                    checked = await self.commit(operation.review, operation_id)
                    operation.receipt = replace(
                        checked,
                        committed=checked.committed or operation.receipt.committed,
                    )
                else:
                    if any(item.phase == "recovery_required" for item in receipts):
                        raise PermissionError("plugin recovery required")
                    if operation.review is None:
                        operation.review = self._review_existing(
                            target.installation_id,
                            kind=kind,
                            workspace_id=target.workspace_id,
                            live_operation=operation,
                        )
                        self.fences.require_current(operation)
                        operation.durable_id = operation.review.operation_id
                        operation_id = operation.durable_id
                        self._revocation_reviews[operation.review.token] = operation
                    self.fences.require_current(operation)
                    operation.receipt = await self.commit(
                        operation.review, operation_id
                    )
        except RevocationConflict:
            raise
        except Exception as error:
            if (
                isinstance(error, ValueError)
                and str(error) == "operation ID belongs to a different review"
            ):
                raise
            operation.receipt = replace(
                operation.receipt, persistence_error=type(error).__name__
            )
            raise RevocationFailure(operation.status(), error) from error
        if kind == "uninstall" and operation.receipt.phase == "complete":
            operation.files_pending = True
            if not operation.unresolved_tokens and all(
                record.completed.is_set() for record in operation.records
            ):
                try:
                    self._remove_uninstalled_package(target.installation_id)
                    operation.files_pending = False
                except (OSError, ValueError) as error:
                    operation.cleanup_errors.append(error)
        return operation.status()

    def _remove_uninstalled_package(self, installation_id: str) -> None:
        """Unlink only beneath the qualified owner, without following parent links."""
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open(self.owner.root, flags)
        try:
            self.owner.require_owner(self.owner.root)
            opened, named = os.fstat(root_fd), self.owner.root.lstat()
            if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                raise ValueError("plugin removal root changed")
            for subtree in ("packages", "packages-revisions"):
                try:
                    packages_fd = os.open(subtree, flags, dir_fd=root_fd)
                except FileNotFoundError:
                    continue
                try:
                    try:
                        shutil.rmtree(installation_id, dir_fd=packages_fd)
                    except FileNotFoundError:
                        pass
                    os.fsync(packages_fd)
                finally:
                    os.close(packages_fd)
        finally:
            os.close(root_fd)

    async def recover(self) -> tuple[OperationReceipt, ...]:
        """Reconcile authenticated transitions under the same storage owner."""
        self._require_worker()
        self._published = None
        from .recovery import recover_coordinator

        return recover_coordinator(self)
