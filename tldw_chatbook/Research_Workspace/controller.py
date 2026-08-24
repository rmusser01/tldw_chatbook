"""Textual-free state fencing for Research Workspace requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .contracts import (
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    ResearchCapability,
    ResearchWorkspacePort,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
)


@dataclass(frozen=True, slots=True)
class ResearchRequestContext:
    ref: QualifiedWorkspaceRef
    capability_revision: str
    context_revision: int


@dataclass(frozen=True, slots=True)
class ResearchWorkspaceCatalogState:
    """One authority's qualified catalog or its typed recovery state."""

    data_source: WorkspaceDataSource
    context_revision: int
    catalog_generation: int
    workspaces: tuple[ResearchWorkspaceSummary, ...] = ()
    recovery: ResearchCapability | None = None


class ResearchWorkspaceController:
    """Own normalized state and prevent late async results from repainting."""

    def __init__(
        self, ports: Mapping[WorkspaceDataSource, ResearchWorkspacePort]
    ) -> None:
        self._ports = dict(ports)
        self._context_revision = 0
        self._catalog_generation = 0
        self._selected_data_source = WorkspaceDataSource.LOCAL
        self._catalog_states: dict[
            WorkspaceDataSource, ResearchWorkspaceCatalogState
        ] = {}
        self._selected_ref: QualifiedWorkspaceRef | None = None
        self._capability_revision = ""
        self._canonical_workspaces: dict[
            QualifiedWorkspaceRef, ResearchWorkspaceSummary
        ] = {}
        self.visible_workspace: ResearchWorkspaceSummary | None = None

    @property
    def context_revision(self) -> int:
        return self._context_revision

    @property
    def catalog_generation(self) -> int:
        return self._catalog_generation

    @property
    def selected_ref(self) -> QualifiedWorkspaceRef | None:
        return self._selected_ref

    @property
    def selected_data_source(self) -> WorkspaceDataSource:
        return self._selected_data_source

    @property
    def catalog_state(self) -> ResearchWorkspaceCatalogState | None:
        return self._catalog_states.get(self._selected_data_source)

    def port_for_data_source(
        self, data_source: WorkspaceDataSource
    ) -> ResearchWorkspacePort | None:
        """Return the configured owner port without substituting another authority."""

        return self._ports.get(WorkspaceDataSource(data_source))

    def select_data_source(self, data_source: WorkspaceDataSource) -> int:
        """Select one explicit catalog and clear the prior visible workspace."""

        selected = WorkspaceDataSource(data_source)
        if selected is self._selected_data_source:
            return self._context_revision
        self._selected_data_source = selected
        self._selected_ref = None
        self._capability_revision = ""
        self.visible_workspace = None
        self._context_revision += 1
        return self._context_revision

    async def refresh_workspace_catalog(self) -> ResearchWorkspaceCatalogState:
        """Load only the selected authority or return its explicit recovery."""

        data_source = self._selected_data_source
        context_revision = self._context_revision
        self._catalog_generation += 1
        catalog_generation = self._catalog_generation
        port = self._ports.get(data_source)
        if port is None:
            owner = data_source.value
            state = ResearchWorkspaceCatalogState(
                data_source=data_source,
                recovery=ResearchCapability(
                    available=False,
                    reason_code=f"{owner}_service_unavailable",
                    user_message=(
                        f"The {data_source.value.title()} workspace service is unavailable."
                    ),
                    owner=owner,
                    recovery_action=(
                        "Restart after local storage is available."
                        if data_source is WorkspaceDataSource.LOCAL
                        else "Configure or choose a server, then retry."
                    ),
                ),
                context_revision=context_revision,
                catalog_generation=catalog_generation,
            )
        else:
            try:
                workspaces = await port.list_workspaces(include_archived=False)
            except CapabilityUnavailableError as exc:
                state = ResearchWorkspaceCatalogState(
                    data_source=data_source,
                    recovery=exc.capability,
                    context_revision=context_revision,
                    catalog_generation=catalog_generation,
                )
            else:
                state = ResearchWorkspaceCatalogState(
                    data_source=data_source,
                    workspaces=tuple(workspaces),
                    context_revision=context_revision,
                    catalog_generation=catalog_generation,
                )
        if self.is_current_catalog_state(state):
            self._catalog_states[data_source] = state
        return state

    def is_current_catalog_state(self, state: ResearchWorkspaceCatalogState) -> bool:
        """Return whether a catalog result still owns the selected generation."""

        if state.data_source is not self._selected_data_source:
            return False
        return (
            state.context_revision == self._context_revision
            and state.catalog_generation == self._catalog_generation
        )

    def select_workspace(
        self, ref: QualifiedWorkspaceRef, *, capability_revision: str = ""
    ) -> int:
        self._selected_data_source = ref.data_source
        self._selected_ref = ref
        self._capability_revision = capability_revision.strip()
        self._context_revision += 1
        self.visible_workspace = self._canonical_workspaces.get(ref)
        return self._context_revision

    def set_capability_revision(self, capability_revision: str) -> int:
        normalized = capability_revision.strip()
        if normalized != self._capability_revision:
            self._capability_revision = normalized
            self._context_revision += 1
        return self._context_revision

    def capture_request(self) -> ResearchRequestContext:
        if self._selected_ref is None:
            raise RuntimeError("No Research workspace is selected")
        return ResearchRequestContext(
            ref=self._selected_ref,
            capability_revision=self._capability_revision,
            context_revision=self._context_revision,
        )

    def is_current_request(self, capture: ResearchRequestContext) -> bool:
        """Return whether a qualified request may still update presentation state."""

        return (
            capture.context_revision == self._context_revision
            and capture.ref == self._selected_ref
            and capture.capability_revision == self._capability_revision
        )

    async def refresh_selected_workspace(self) -> bool:
        capture = self.capture_request()
        port = self._ports.get(capture.ref.data_source)
        if port is None:
            raise RuntimeError(
                f"No adapter is configured for {capture.ref.data_source.value}"
            )
        result = await port.get_workspace(capture.ref)
        if result is None:
            return False
        return self.accept_workspace_result(capture, result)

    def accept_workspace_result(
        self,
        capture: ResearchRequestContext,
        result: ResearchWorkspaceSummary,
    ) -> bool:
        if result.ref != capture.ref:
            raise ValueError("Request returned a mismatched workspace ref")
        self._canonical_workspaces[capture.ref] = result
        if not self.is_current_request(capture):
            return False
        self.visible_workspace = result
        return True

    def canonical_workspace(
        self, ref: QualifiedWorkspaceRef
    ) -> ResearchWorkspaceSummary | None:
        return self._canonical_workspaces.get(ref)
