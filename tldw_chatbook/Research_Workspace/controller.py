"""Textual-free state fencing for Research Workspace requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .contracts import (
    BoundedPageResult,
    CapabilityUnavailableError,
    MAX_RESEARCH_SELECTION_IDS,
    MAX_RESEARCH_SELECTION_ROWS,
    QualifiedWorkspaceRef,
    ResearchCatalogItem,
    ResearchCapability,
    ResearchSourcePreview,
    ResearchSourcePage,
    ResearchSourceSummary,
    SourceSelectionResult,
    ResearchWorkspacePort,
    ResearchWorkspaceSummary,
    SourceReadiness,
    WorkspaceDataSource,
    require_capability,
)
from .source_operations import ResearchSourceOperation


@dataclass(frozen=True, slots=True)
class ResearchRequestContext:
    ref: QualifiedWorkspaceRef
    capability_revision: str
    context_revision: int


@dataclass(frozen=True, slots=True)
class ResearchSurfaceRequest:
    """One qualified request plus its independently monotonic surface token."""

    context: ResearchRequestContext
    surface: str
    generation: int


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
        self._surface_generations = {
            "association": 0,
            "capabilities": 0,
            "sources": 0,
            "catalog": 0,
            "readiness": 0,
            "preview": 0,
            "selection": 0,
        }
        self._canonical_sources: dict[
            tuple[QualifiedWorkspaceRef, str], ResearchSourceSummary
        ] = {}
        self._canonical_catalog_items: dict[
            tuple[QualifiedWorkspaceRef, str], ResearchCatalogItem
        ] = {}
        self._canonical_readiness: dict[
            tuple[QualifiedWorkspaceRef, str], SourceReadiness
        ] = {}
        self._canonical_previews: dict[
            tuple[QualifiedWorkspaceRef, str], ResearchSourcePreview
        ] = {}
        self.visible_workspace: ResearchWorkspaceSummary | None = None
        self.visible_source_page: ResearchSourcePage | None = None
        self.visible_catalog_page: BoundedPageResult[ResearchCatalogItem] | None = None
        self.visible_readiness: tuple[SourceReadiness, ...] = ()
        self.visible_preview: ResearchSourcePreview | None = None
        self.visible_capabilities: Mapping[str, ResearchCapability] = {}
        self.desired_source_ids: tuple[str, ...] = ()

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
        self._clear_visible_source_state()
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
        self._clear_visible_source_state()
        return self._context_revision

    def set_capability_revision(self, capability_revision: str) -> int:
        normalized = capability_revision.strip()
        if normalized != self._capability_revision:
            self._capability_revision = normalized
            self._context_revision += 1
            self._clear_visible_source_state()
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

    async def refresh_selected_sources(
        self, *, limit: int = 100, offset: int = 0
    ) -> bool:
        """Refresh the attached-source page without accepting a late result."""

        capture = self._capture_surface("sources")
        result = await self._port_for(capture).list_sources(
            capture.context.ref, limit=limit, offset=offset
        )
        if not isinstance(result, ResearchSourcePage):
            raise TypeError("Source listing returned an invalid owner page")
        self._validate_result_refs(result.items, capture.context.ref)
        if not self._is_current_surface(capture):
            return False
        self.visible_source_page = result
        self.desired_source_ids = result.desired_source_ids
        for source in result.items:
            self._canonical_sources[(source.ref, source.source_id)] = source
        return True

    async def refresh_selected_capabilities(self) -> bool:
        """Refresh only the selected qualified authority's action projection."""

        capture = self._capture_surface("capabilities")
        result = await self._port_for(capture).capabilities(capture.context.ref)
        if not isinstance(result, Mapping) or any(
            not isinstance(name, str) or not isinstance(capability, ResearchCapability)
            for name, capability in result.items()
        ):
            raise TypeError("Capability projection returned invalid entries")
        if not self._is_current_surface(capture):
            return False
        self.visible_capabilities = dict(result)
        return True

    async def require_workspace_capability(
        self, ref: QualifiedWorkspaceRef, capability_name: str
    ) -> ResearchCapability:
        """Preflight one mutation against its explicit captured owner."""

        if not isinstance(ref, QualifiedWorkspaceRef):
            raise TypeError("ref must be QualifiedWorkspaceRef")
        projection = await self._port_for_ref(ref).capabilities(ref)
        if not isinstance(projection, Mapping) or any(
            not isinstance(name, str) or not isinstance(capability, ResearchCapability)
            for name, capability in projection.items()
        ):
            raise TypeError("Capability projection returned invalid entries")
        return require_capability(projection, capability_name)

    async def search_selected_catalog(
        self,
        *,
        query: str = "",
        source_types: tuple[str, ...] = (),
        sort_by: str = "updated_desc",
        limit: int = 25,
        offset: int = 0,
    ) -> bool:
        """Search only the selected authority's canonical Media catalog."""

        capture = self._capture_surface("catalog")
        result = await self._port_for(capture).search_catalog(
            capture.context.ref,
            query=query,
            source_types=source_types,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
        )
        self._validate_result_refs(result.items, capture.context.ref)
        if not self._is_current_surface(capture):
            return False
        self.visible_catalog_page = result
        for item in result.items:
            self._canonical_catalog_items[(item.ref, item.catalog_item_id)] = item
        return True

    async def refresh_selected_readiness(
        self, *, source_ids: tuple[str, ...] = ()
    ) -> bool:
        """Refresh readiness without changing the user's desired selection."""

        capture = self._capture_surface("readiness")
        result = tuple(
            await self._port_for(capture).get_readiness(
                capture.context.ref, source_ids=source_ids
            )
        )
        self._validate_result_refs(result, capture.context.ref)
        if not self._is_current_surface(capture):
            return False
        self.visible_readiness = result
        for readiness in result:
            self._canonical_readiness[(readiness.ref, readiness.source_id)] = readiness
        return True

    async def preview_selected_source(
        self,
        source_id: str,
        *,
        max_chars: int = 3000,
        snippet_limit: int = 3,
    ) -> bool:
        """Preview one source while fencing workspace and capability changes."""

        capture = self._capture_surface("preview")
        result = await self._port_for(capture).preview_source(
            capture.context.ref,
            source_id,
            max_chars=max_chars,
            snippet_limit=snippet_limit,
        )
        self._validate_result_refs((result,), capture.context.ref)
        if not self._is_current_surface(capture):
            return False
        self.visible_preview = result
        self._canonical_previews[(result.ref, result.source_id)] = result
        return True

    async def set_selected_scope(self, source_ids: tuple[str, ...]) -> bool:
        """Persist desired selection and reconcile from the authority owner."""

        capture = self._capture_surface("selection")
        self._surface_generations["sources"] += 1
        result = await self._port_for(capture).set_selected_scope(
            capture.context.ref, source_ids
        )
        if not isinstance(result, SourceSelectionResult):
            raise TypeError("Selection reconciliation returned an invalid result")
        self._validate_result_refs((result, *result.sources), capture.context.ref)
        if len(result.desired_source_ids) != len(source_ids) or frozenset(
            result.desired_source_ids
        ) != frozenset(source_ids):
            raise ValueError("Selection reconciliation did not match requested scope")
        return self._accept_selection_result(capture, source_ids, result)

    async def select_all_sources(self) -> bool:
        """Persist every exact owner ID, never just the currently visible page."""

        capture = self._capture_surface("selection")
        self._surface_generations["sources"] += 1
        port = self._port_for(capture)
        owner_ids: list[str] = []
        offset = 0
        total: int | None = None
        while total is None or offset < total:
            page = await port.list_sources(
                capture.context.ref, limit=100, offset=offset
            )
            if not isinstance(page, ResearchSourcePage):
                raise TypeError("Source listing returned an invalid owner page")
            self._validate_result_refs(page.items, capture.context.ref)
            if total is None:
                total = page.total
                if total > MAX_RESEARCH_SELECTION_IDS:
                    raise ValueError("Source owner exceeds the bounded selection limit")
            elif page.total != total:
                raise ValueError("Source owner changed during select all")
            owner_ids.extend(
                source.catalog_item_id
                if capture.context.ref.data_source is WorkspaceDataSource.LOCAL
                else source.source_id
                for source in page.items
            )
            offset += len(page.items)
            if not page.items and offset < total:
                raise ValueError("Source owner returned an incomplete page")
        requested = tuple(owner_ids)
        result = await port.set_selected_scope(capture.context.ref, requested)
        if not isinstance(result, SourceSelectionResult):
            raise TypeError("Selection reconciliation returned an invalid result")
        self._validate_result_refs((result, *result.sources), capture.context.ref)
        return self._accept_selection_result(capture, requested, result)

    async def attach_selected_existing(
        self,
        *,
        catalog_item_id: str,
        idempotency_key: str,
        desired_selected: bool = True,
    ) -> ResearchSourceOperation | None:
        """Attach one catalog item through the captured authority's durable seam."""

        capture = self._capture_surface("association")
        self._surface_generations["sources"] += 1
        result = await self.attach_existing(
            capture.context.ref,
            catalog_item_id=catalog_item_id,
            desired_selected=desired_selected,
            idempotency_key=idempotency_key,
        )
        return result if self._is_current_surface(capture) else None

    async def attach_existing(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        catalog_item_id: str,
        idempotency_key: str,
        desired_selected: bool = True,
    ) -> ResearchSourceOperation:
        """Attach to an explicit captured owner even after visible navigation."""

        result = await self._port_for_ref(ref).attach_existing(
            ref,
            catalog_item_id=catalog_item_id,
            desired_selected=desired_selected,
            idempotency_key=idempotency_key,
        )
        if not isinstance(result, ResearchSourceOperation):
            raise TypeError("Source attachment returned an invalid operation")
        self._validate_operation_ref(result, ref)
        return result

    async def remove_selected_source(
        self, source_id: str, *, expected_version: int | None = None
    ) -> bool:
        """Remove only the captured workspace association, never catalog media."""

        capture = self._capture_surface("association")
        self._surface_generations["sources"] += 1
        removed = await self._port_for(capture).remove_source(
            capture.context.ref,
            source_id,
            expected_version=expected_version,
        )
        if type(removed) is not bool:
            raise TypeError("Source removal returned an invalid result")
        if not self._is_current_surface(capture):
            return False
        if removed:
            self._canonical_sources.pop((capture.context.ref, source_id), None)
        return removed

    async def reorder_selected_sources(
        self, ordered_source_ids: tuple[str, ...]
    ) -> bool:
        """Persist an exact manual order through the selected owner capability."""

        capture = self._capture_surface("association")
        self._surface_generations["sources"] += 1
        rows = tuple(
            await self._port_for(capture).reorder_sources(
                capture.context.ref, ordered_source_ids
            )
        )
        self._validate_result_refs(rows, capture.context.ref)
        if tuple(source.source_id for source in rows) != ordered_source_ids:
            raise ValueError("Source reorder did not return the requested exact order")
        if not self._is_current_surface(capture):
            return False
        for source in rows:
            self._canonical_sources[(source.ref, source.source_id)] = source
        return True

    async def move_selected_source(self, source_id: str, *, delta: int) -> bool:
        """Move one source in the exact bounded owner order."""

        if delta not in {-1, 1}:
            raise ValueError("delta must be -1 or 1")
        capture = self._capture_surface("association")
        self._surface_generations["sources"] += 1
        port = self._port_for(capture)
        page = await port.list_sources(
            capture.context.ref,
            limit=MAX_RESEARCH_SELECTION_ROWS,
            offset=0,
        )
        if not isinstance(page, ResearchSourcePage):
            raise TypeError("Source listing returned an invalid owner page")
        self._validate_result_refs(page.items, capture.context.ref)
        if page.total is None or page.total > MAX_RESEARCH_SELECTION_ROWS:
            raise ValueError("Source owner exceeds the bounded reorder limit")
        if page.offset != 0 or len(page.items) != page.total or page.has_more:
            raise ValueError("Source owner returned an incomplete reorder snapshot")
        ordered = [source.source_id for source in page.items]
        try:
            index = ordered.index(source_id)
        except ValueError:
            raise ValueError("Source is outside the exact owner order") from None
        target = index + delta
        if target < 0 or target >= len(ordered):
            return False
        ordered[index], ordered[target] = ordered[target], ordered[index]
        requested = tuple(ordered)
        rows = tuple(
            await port.reorder_sources(capture.context.ref, requested)
        )
        self._validate_result_refs(rows, capture.context.ref)
        if tuple(source.source_id for source in rows) != requested:
            raise ValueError("Source reorder did not return the requested exact order")
        if not self._is_current_surface(capture):
            return False
        for source in rows:
            self._canonical_sources[(source.ref, source.source_id)] = source
        return True

    def canonical_source(
        self, ref: QualifiedWorkspaceRef, source_id: str
    ) -> ResearchSourceSummary | None:
        return self._canonical_sources.get((ref, source_id))

    def canonical_catalog_item(
        self, ref: QualifiedWorkspaceRef, catalog_item_id: str
    ) -> ResearchCatalogItem | None:
        return self._canonical_catalog_items.get((ref, catalog_item_id))

    def canonical_source_readiness(
        self, ref: QualifiedWorkspaceRef, source_id: str
    ) -> SourceReadiness | None:
        return self._canonical_readiness.get((ref, source_id))

    def canonical_source_preview(
        self, ref: QualifiedWorkspaceRef, source_id: str
    ) -> ResearchSourcePreview | None:
        return self._canonical_previews.get((ref, source_id))

    def _capture_surface(self, surface: str) -> ResearchSurfaceRequest:
        self._surface_generations[surface] += 1
        return ResearchSurfaceRequest(
            context=self.capture_request(),
            surface=surface,
            generation=self._surface_generations[surface],
        )

    def _is_current_surface(self, capture: ResearchSurfaceRequest) -> bool:
        return self.is_current_request(capture.context) and (
            capture.generation == self._surface_generations[capture.surface]
        )

    def _port_for(self, capture: ResearchSurfaceRequest) -> ResearchWorkspacePort:
        return self._port_for_ref(capture.context.ref)

    def _port_for_ref(self, ref: QualifiedWorkspaceRef) -> ResearchWorkspacePort:
        port = self._ports.get(ref.data_source)
        if port is None:
            raise RuntimeError(f"No adapter is configured for {ref.data_source.value}")
        return port

    @staticmethod
    def _validate_result_refs(results: object, ref: QualifiedWorkspaceRef) -> None:
        for result in results:
            if result.ref != ref:
                raise ValueError("Request returned a mismatched workspace ref")

    def _clear_visible_source_state(self) -> None:
        self.visible_source_page = None
        self.visible_catalog_page = None
        self.visible_readiness = ()
        self.visible_preview = None
        self.visible_capabilities = {}
        self.desired_source_ids = ()

    def _accept_selection_result(
        self,
        capture: ResearchSurfaceRequest,
        requested: tuple[str, ...],
        result: SourceSelectionResult,
    ) -> bool:
        if len(result.desired_source_ids) != len(requested) or frozenset(
            result.desired_source_ids
        ) != frozenset(requested):
            raise ValueError("Selection reconciliation did not match requested scope")
        if not self._is_current_surface(capture):
            return False
        self._surface_generations["sources"] += 1
        self.desired_source_ids = result.desired_source_ids
        for source in result.sources:
            self._canonical_sources[(source.ref, source.source_id)] = source
        return True

    @staticmethod
    def _validate_operation_ref(
        operation: ResearchSourceOperation, ref: QualifiedWorkspaceRef
    ) -> None:
        if (
            operation.data_source is not ref.data_source
            or operation.workspace_id != ref.workspace_id
            or operation.server_profile_id != ref.server_profile_id
            or operation.principal_id != ref.principal_id
        ):
            raise ValueError("Request returned a mismatched workspace operation")
