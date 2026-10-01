"""Run-scoped native plugin admission under authenticated current authority."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

from .coordinator import PluginCoordinator
from .models import PackageInspection
from .package_files import canonical_json
from .recovery import retained_inspections
from .revocation import RevocationTarget


def effective_activation(default: bool, override: str) -> bool:
    """Resolve the explicit workspace intent against the global default."""
    if override == "inherit":
        return default
    if override in {"enabled", "disabled"}:
        return override == "enabled"
    raise ValueError("plugin_activation_invalid")


class PluginUnavailable(PermissionError):
    """Current plugin authority cannot authorize the requested effect."""


@dataclass(frozen=True)
class RunPluginSnapshot:
    """Immutable run ceiling; never a substitute for current authorization."""

    installation_id: str
    revision_digest: str
    workspace_id: str | None
    run_id: str
    selection: tuple[str, ...]
    mappings_json: str
    generations: tuple[tuple[str, str, int], ...]
    dependencies: tuple[tuple[str, tuple[str, ...]], ...]
    inspection: PackageInspection
    alias: str
    live_generations: tuple[tuple[str, str, int], ...] = ()
    data_roots_json: str = "[]"
    root_epochs: tuple[tuple[str, int], ...] = ()


class LivePluginFences:
    """Immediate scope seal, independent of the storage worker's queue.

    The revocation lifecycle owner may seal before awaiting durable work. A seal
    is session-only and never asserts durable disable or confirmed cleanup.
    """

    def __init__(self) -> None:
        self.live_lock = RLock()
        self._sealed: set[tuple[str, str | None]] = set()
        self.epochs = {}
        self.root_epochs = {}
        self.blocked = set()
        self.runs = {}
        self.operations = {}
        self.drains = {}
        self.session_nonce = uuid4().hex

    def require_admission(self, installation_id, revision_digest):
        with self.live_lock:
            if any(
                ticket.installation_id == installation_id
                and ticket.revision_digest == revision_digest
                and ticket.phase in {"waiting", "committing"}
                for ticket in self.drains.values()
            ):
                raise PluginUnavailable("plugin_revision_draining")

    def current(self, installation_id, generations):
        with self.live_lock:
            keys = [
                (installation_id, kind, workspace) for kind, workspace, _ in generations
            ]
            if any(key in self.blocked for key in keys):
                raise PluginUnavailable("plugin_scope_sealed")
            return tuple(
                (
                    kind,
                    workspace,
                    self.epochs.get((installation_id, kind, workspace), 0),
                )
                for _, kind, workspace in keys
            )

    def check_snapshot(self, snapshot):
        self.check(snapshot.installation_id, snapshot.workspace_id)
        with self.live_lock:
            if any(
                self.root_epochs.get(root_id, 0) != epoch
                for root_id, epoch in snapshot.root_epochs
            ):
                raise PluginUnavailable("plugin_root_generation_changed")
        if (
            self.current(snapshot.installation_id, snapshot.generations)
            != snapshot.live_generations
        ):
            raise PluginUnavailable("plugin_live_generation_changed")

    def seal_target(self, target: RevocationTarget) -> tuple[str, ...]:
        with self.live_lock:
            self.epochs[target.scope] = self.epochs.get(target.scope, 0) + 1
            self.blocked.add(target.scope)
            return tuple(
                record.lease_token
                for record in self.runs.values()
                if target.matches(record)
            )

    def request(self, target, kind="revoke", review=None):
        from .revocation import RevocationRequest

        if review is None:
            request_id = f"lr1.{self.session_nonce}.{uuid4().hex}"
        else:
            parts = review.operation_id.split(".")
            if len(parts) != 5 or parts[0] != "pi1":
                raise ValueError("review has no issued identity")
            nonce = parts[3]
            request_id = f"lr1.{nonce[:32]}.{nonce[32:]}"
        self.begin(target, request_id, kind, review=review)
        return RevocationRequest(request_id)

    def require_current(self, operation):
        from dataclasses import replace

        from .revocation import RevocationConflict

        with self.live_lock:
            if operation.receipt.phase != "complete" and any(
                self.epochs.get(scope, 0) != generation
                for scope, generation in operation.scope_versions
            ):
                operation.receipt = replace(operation.receipt, phase="superseded")
                raise RevocationConflict("plugin revocation request superseded")

    def begin(self, target, operation_id, kind="revoke", review=None):
        from .authority import PluginMarker
        from .review import OperationReceipt
        from .revocation import RevocationOperation

        PluginMarker(
            generation=1, operation_id=operation_id, recovery_snapshot_digest="0" * 64
        )
        with self.live_lock:
            prior = self.operations.get(operation_id)
            if prior is not None:
                if (prior.target, prior.kind) != (target, kind) or (
                    review is not None and prior.review != review
                ):
                    raise ValueError("plugin operation identity conflict")
                self.require_current(prior)
                return prior
            tokens = self.seal_target(target)
            records = tuple(
                record for record in self.runs.values() if record.lease_token in tokens
            )
            operation = RevocationOperation(
                target,
                kind,
                operation_id,
                records,
                OperationReceipt(None, "session_only", False, request_id=operation_id),
                review=review,
                durable_id=review.operation_id if review is not None else None,
                scope_versions=tuple(
                    (scope, self.epochs.get(scope, 0))
                    for scope in sorted(
                        {target.scope, (target.installation_id, "installation", "")}
                    )
                ),
            )
            self.operations[operation_id] = operation
            operation.cancel_owned()
            return operation

    def reconcile_enable(self, installation_id, workspace_id):
        with self.live_lock:
            self.blocked.discard((installation_id, "installation", ""))
            self.blocked.discard(
                (
                    installation_id,
                    "global_default" if workspace_id is None else "workspace",
                    workspace_id or "",
                )
            )

    def require_enable_reconciled(self, installation_id, workspace_id):
        with self.live_lock:
            reconciled = set()
            for op in reversed(tuple(self.operations.values())):
                target = op.target
                if target.installation_id != installation_id:
                    continue
                if op.receipt.phase == "complete":
                    reconciled.add(target.scope)
                    continue
                if (
                    target.scope in reconciled
                    or (installation_id, "installation", "") in reconciled
                ):
                    continue
                if (
                    target.everywhere
                    or (target.global_default and workspace_id is None)
                    or (
                        not target.global_default
                        and target.workspace_id == workspace_id
                    )
                ):
                    raise PluginUnavailable("plugin_revocation_reconciliation_required")

    def seal(self, installation_id: str, workspace_id: str | None = None) -> None:
        with self.live_lock:
            self._sealed.add((installation_id, workspace_id))

    def check(self, installation_id: str, workspace_id: str | None) -> None:
        with self.live_lock:
            if (installation_id, None) in self._sealed or (
                installation_id,
                workspace_id,
            ) in self._sealed:
                raise PluginUnavailable("plugin_scope_sealed")


class PluginAdmission:
    """Current checks on the same persistent worker as committed authority."""

    def __init__(
        self,
        coordinator: PluginCoordinator,
        workspace_lookup: Callable,
        *,
        fences: LivePluginFences | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.workspace_lookup = workspace_lookup
        self.fences = fences or coordinator.fences

    def seal(self, target: RevocationTarget) -> tuple[str, ...]:
        """Fence the exact scope synchronously; return retained cancellation tokens."""
        return self.fences.seal_target(target)

    def _workspace(self, workspace_id: str | None) -> None:
        if workspace_id is None or workspace_id in {"global", "workspace-default"}:
            return
        if not isinstance(workspace_id, str) or not workspace_id:
            raise PluginUnavailable("plugin_workspace_invalid")
        record = self.workspace_lookup(workspace_id)
        if record is None or getattr(record, "archived", True):
            raise PluginUnavailable("plugin_workspace_unavailable")

    def capture(
        self,
        installation_id: str,
        workspace_id: str | None,
        run_id: str,
        *,
        fresh: bool = True,
    ) -> RunPluginSnapshot:
        """Capture a trusted, enabled revision and dependency-complete skills."""
        self.fences.check(installation_id, workspace_id)
        self._workspace(workspace_id)
        if not isinstance(run_id, str) or not run_id.strip():
            raise PluginUnavailable("plugin_run_identity_required")
        try:
            authority = self.coordinator.published_snapshot()
            installed = next(
                row
                for row in authority["installations"]
                if row["installation_id"] == installation_id
            )
            revision = installed["revision_digest"]
            if fresh:
                self.fences.require_admission(installation_id, revision)
            alias = installed.get("alias")
            if not alias:
                raise PluginUnavailable("plugin_alias_review_required")
            trusted = any(
                row["installation_id"] == installation_id
                and row["revision_digest"] == revision
                and row["reviewed"]
                for row in authority["revision_trust"]
            )
            override = next(
                (
                    row["intent"]
                    for row in authority["activation"]
                    if row["installation_id"] == installation_id
                    and row["workspace_id"] == workspace_id
                ),
                "inherit",
            )
            if not trusted or not effective_activation(
                installed["activation_default"], override
            ):
                raise PluginUnavailable("plugin_not_enabled_or_trusted")
            # Reinspection verifies retained bytes AND their complete interpretation.
            inspection = retained_inspections(
                {
                    **authority,
                    "revisions": [
                        row
                        for row in authority["revisions"]
                        if row["installation_id"] == installation_id
                        and row["revision_digest"] == revision
                    ],
                }
            )[(installation_id, revision)]
            if inspection.rejected or inspection.activation_blockers:
                raise PluginUnavailable("plugin_constraints_blocked")
            selection = frozenset(
                row["component_id"]
                for row in authority["selections"]
                if row["installation_id"] == installation_id
                and row["revision_digest"] == revision
                and row["selected"]
            )

            def ready(component_id: str, visiting: frozenset[str]) -> bool:
                component = inspection.inventory.get(component_id)
                if (
                    component_id in visiting
                    or component_id not in selection
                    or component is None
                ):
                    return False
                # F5 has only native skill execution. Other providers must prove
                # readiness through their owning services in their increments.
                if (
                    component.kind != "skill"
                    or component.support not in {"supported", "adapted"}
                    or component.activation_blockers
                ):
                    return False
                from .skill_provider import skill_summary

                if skill_summary(installation_id, inspection, component_id, alias)[
                    "plugin_blockers"
                ]:
                    return False
                return all(
                    ready(dep, visiting | {component_id})
                    for dep in component.dependencies
                )

            eligible = tuple(
                sorted(key for key in selection if ready(key, frozenset()))
            )
            if not eligible:
                raise PluginUnavailable("plugin_components_unavailable")
            scopes = {("installation", "")}
            if override == "inherit":
                scopes.add(("global_default", ""))
            if workspace_id is not None:
                scopes.add(("workspace", workspace_id))
            rows = [
                row
                for row in authority["authority_generations"]
                if row["installation_id"] == installation_id
                and (row["scope_kind"], row["workspace_id"]) in scopes
            ]
            if any(row["revoked"] for row in rows):
                raise PluginUnavailable("plugin_revoked")
            generations = tuple(
                sorted(
                    (
                        kind,
                        workspace,
                        next(
                            (
                                row["generation"]
                                for row in rows
                                if row["scope_kind"] == kind
                                and row["workspace_id"] == workspace
                            ),
                            0,
                        ),
                    )
                    for kind, workspace in scopes
                )
            )
            mappings = canonical_json(
                [
                    row
                    for row in authority["mappings"]
                    if row["installation_id"] == installation_id
                ]
            )
            if mappings != "[]":
                raise PluginUnavailable("plugin_mapping_owner_unavailable")
            from .data_cleanup import applicable_roots

            roots = applicable_roots(authority, installation_id, workspace_id)
            with self.fences.live_lock:
                root_epochs = tuple(
                    (row["root_id"], self.fences.root_epochs.get(row["root_id"], 0))
                    for row in roots
                )
            self.fences.check(installation_id, workspace_id)
            return RunPluginSnapshot(
                installation_id,
                revision,
                workspace_id,
                run_id,
                eligible,
                mappings,
                generations,
                tuple(
                    (key, inspection.inventory[key].dependencies) for key in eligible
                ),
                inspection,
                alias,
                self.fences.current(installation_id, generations),
                canonical_json(roots),
                root_epochs,
            )
        except PluginUnavailable:
            raise
        except Exception as error:
            raise PluginUnavailable("plugin_current_authority_unavailable") from error

    def check(self, snapshot: RunPluginSnapshot, component_id: str) -> None:
        """Refuse changed installation/scope/material; unrelated markers may move."""
        self.fences.check_snapshot(snapshot)
        if component_id not in snapshot.selection:
            raise PluginUnavailable("plugin_component_not_admitted")
        current = self.capture(
            snapshot.installation_id,
            snapshot.workspace_id,
            snapshot.run_id,
            fresh=False,
        )
        if (
            current.revision_digest,
            current.generations,
            current.mappings_json,
            current.dependencies,
            current.alias,
            current.data_roots_json,
        ) != (
            snapshot.revision_digest,
            snapshot.generations,
            snapshot.mappings_json,
            snapshot.dependencies,
            snapshot.alias,
            snapshot.data_roots_json,
        ) or not set(snapshot.selection) <= set(current.selection):
            # An archived ceiling may deliberately narrow the current selection.
            # Exact generations still detect every intervening authority edit.
            raise PluginUnavailable("plugin_admission_changed")
