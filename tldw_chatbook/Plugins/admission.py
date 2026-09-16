"""Run-scoped native plugin admission under authenticated current authority."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from .coordinator import PluginCoordinator
from .models import PackageInspection
from .package_files import canonical_json
from .recovery import retained_inspections


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


class LivePluginFences:
    """Immediate scope seal, independent of the storage worker's queue.

    The revocation lifecycle owner may seal before awaiting durable work. A seal
    is session-only and never asserts durable disable or confirmed cleanup.
    """

    def __init__(self) -> None:
        self.live_lock = RLock()
        self._sealed: set[tuple[str, str | None]] = set()

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
        self.fences = fences or LivePluginFences()

    def _workspace(self, workspace_id: str | None) -> None:
        if workspace_id is None or workspace_id in {"global", "workspace-default"}:
            return
        if not isinstance(workspace_id, str) or not workspace_id:
            raise PluginUnavailable("plugin_workspace_invalid")
        record = self.workspace_lookup(workspace_id)
        if record is None or getattr(record, "archived", True):
            raise PluginUnavailable("plugin_workspace_unavailable")

    def capture(
        self, installation_id: str, workspace_id: str | None, run_id: str
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
            )
        except PluginUnavailable:
            raise
        except Exception as error:
            raise PluginUnavailable("plugin_current_authority_unavailable") from error

    def check(self, snapshot: RunPluginSnapshot, component_id: str) -> None:
        """Refuse changed installation/scope/material; unrelated markers may move."""
        if component_id not in snapshot.selection:
            raise PluginUnavailable("plugin_component_not_admitted")
        current = self.capture(
            snapshot.installation_id, snapshot.workspace_id, snapshot.run_id
        )
        if (
            current.revision_digest,
            current.generations,
            current.mappings_json,
            current.dependencies,
            current.selection,
            current.alias,
        ) != (
            snapshot.revision_digest,
            snapshot.generations,
            snapshot.mappings_json,
            snapshot.dependencies,
            snapshot.selection,
            snapshot.alias,
        ):
            raise PluginUnavailable("plugin_admission_changed")
