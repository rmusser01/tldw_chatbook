"""Owned reviewed installs: durable SQLite success precedes certification."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from .authority import PluginMarker, snapshot_digest
from .authority_store import PluginAuthorityStore
from .inspection import inspect_package
from .models import PackageInspection
from .package_files import canonical_json, capture_package, materialize_package
from .registry import PluginRegistry
from .review import OperationReceipt, PluginReview, inspection_identity, reinspect
from .revocation import RevocationTarget
from .runtime_owner import PluginRuntimeOwner

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
    ) -> None:
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("plugin coordinator requires a dedicated worker")
        owner.require_owner(registry.path.parent)
        # Probe SQLite's real affinity now, instead of weakening check_same_thread.
        _ = registry.schema_version
        self.registry, self.authority, self.owner = registry, authority, owner
        self._thread = threading.get_ident()
        self._loop = asyncio.get_event_loop()
        from .admission import LivePluginFences

        self.fences = fences or LivePluginFences()
        self._revocation_reviews = {}
        self._reviews: dict[str, PluginReview] = {}
        self._published: dict | None = None
        self.progress: Callable[[str], None] | None = None

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
        if self.registry.authority_projection(operation_result=None)["installations"]:
            raise ValueError("existing installations require reviewed recovery")
        self.authority.bootstrap(passphrase)
        self._published = self.authority.verify_current()

    def reset(self, *, operation_id: str) -> Path | None:
        """Perform an explicitly reviewed plugin-only reset with a retained ID."""
        self._require_worker()
        self._published = None
        self._reviews.clear()
        return self.authority.reset(operation_id=operation_id)

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
        self._reviews[review.token] = review
        return review

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
        )
        self._reviews[review.token] = review
        return review

    def _apply_review(
        self, cursor, review: PluginReview, retained: PackageInspection
    ) -> None:
        if review.kind == "install":
            self.registry.insert_installation(
                cursor,
                review.installation_id,
                retained,
                review.selection,
                alias=review.alias,
            )
            return
        if review.kind in {"revoke", "uninstall"}:
            operation = self._revocation_reviews[review.token]
            target = operation.target
            if review.kind == "uninstall":
                cursor.execute(
                    "INSERT INTO tombstones VALUES (?, ?, ?)",
                    (
                        review.installation_id,
                        review.authority_marker.generation + 1,
                        operation.operation_id,
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
        root.mkdir(mode=0o700, exist_ok=True)
        destination = root / review.installation_id
        source = Path(
            review.inspection.materialized_identity or review.inspection.source_identity
        )
        # F1 bounds captures; reserve capacity before writing and let actual
        # filesystem failures propagate without reporting commitment.
        capture = capture_package(source)
        if capture.errors:
            raise ValueError("stale or unsafe package capture")
        size = sum(len(member.data) for member in capture.files.values())
        if shutil.disk_usage(root).free < size + FREE_RESERVE_BYTES:
            raise OSError("insufficient plugin storage reserve")
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

        target = revocation_for_review(review)
        if target is not None and review.token not in self._revocation_reviews:
            return await self._revoke(target, operation_id, "activate", review=review)
        self._require_worker()
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
                or result["revision_digest"] != review.inspection.effective_digest
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
        self._verify_reviewed_package(review.inspection)
        self._published = None
        retained = (
            self._materialize(review) if review.kind == "install" else review.inspection
        )
        self._milestone("materialized")
        result = {
            "operation_id": operation_id,
            "installation_id": review.installation_id,
            "kind": review.kind,
            "revision_digest": retained.effective_digest,
            "result": "committed",
        }
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
        self._milestone("published")
        return OperationReceipt(operation_id, "complete", True)

    async def disable(
        self, target: RevocationTarget, operation_id: str
    ) -> OperationReceipt:
        """Share the facade's immediate live owner before persistence begins."""
        return await self._revoke(target, operation_id, "revoke")

    async def uninstall(
        self, installation_id: str, operation_id: str
    ) -> OperationReceipt:
        return await self._revoke(
            RevocationTarget(installation_id, None, True), operation_id, "uninstall"
        )

    async def _revoke(self, target, operation_id, kind, *, review=None):
        from dataclasses import replace

        from .revocation import RevocationConflict, RevocationFailure

        operation = self.fences.begin(target, operation_id, kind, review=review)
        self._require_worker()
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
                        )
                        self._revocation_reviews[operation.review.token] = operation
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
            try:
                packages_fd = os.open("packages", flags, dir_fd=root_fd)
            except FileNotFoundError:
                return
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
