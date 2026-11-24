"""Search one qualified workspace catalog using its existing owner pages."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from .contracts import QualifiedWorkspaceRef, ResearchWorkspacePort
from .controller import ResearchWorkspaceCatalogState
from .quick_notes import ResearchNotePageRequest
from .studio_models import WorkspaceOutputRef


WorkspaceSearchKind = Literal["workspace", "source", "note", "output"]
SEARCH_OWNER_PAGE_SIZE = 25
MAX_SEARCH_QUERY = 240


@dataclass(frozen=True, slots=True)
class WorkspaceSearchHit:
    workspace: QualifiedWorkspaceRef
    workspace_name: str
    kind: WorkspaceSearchKind
    title: str
    item_id: str
    owner_offset: int = 0
    output: WorkspaceOutputRef | None = None


async def _owner_pages(fetch):
    offset = 0
    while True:
        page = await fetch(offset)
        yield offset, page
        if not page.has_more:
            return
        if not page.items:
            raise ValueError("Workspace search received an incomplete owner page")
        offset += len(page.items)


async def search_workspace(
    port: ResearchWorkspacePort, catalog: ResearchWorkspaceCatalogState, query: str
) -> AsyncIterator[WorkspaceSearchHit]:
    """Yield matches with canonical IDs, reading just one owner page at a time."""
    query = query.strip()
    if not query or len(query) > MAX_SEARCH_QUERY:
        raise ValueError("Enter text to search the current workspace data")
    prefix, colon, rest = query.partition(":")
    kinds = ("workspace", "source", "note", "output")
    scope = prefix.casefold()
    if colon and scope in kinds:
        kinds = (scope,)
        query = rest.strip()
    needle = query.casefold()
    if catalog.recovery is not None:
        raise ValueError(catalog.recovery.user_message)
    for workspace in catalog.workspaces:
        ref, name = workspace.ref, workspace.name
        if ref.data_source is not catalog.data_source or workspace.archived:
            raise ValueError(
                "Search found a workspace outside the selected active catalog"
            )
        if (
            "workspace" in kinds
            and needle in f"{name} {workspace.description}".casefold()
        ):
            yield WorkspaceSearchHit(ref, name, "workspace", name, ref.workspace_id)
        if "source" in kinds:
            async for offset, page in _owner_pages(
                lambda offset, ref=ref: port.list_sources(
                    ref, limit=SEARCH_OWNER_PAGE_SIZE, offset=offset
                )
            ):
                for source in page.items:
                    if source.ref != ref:
                        raise ValueError(
                            "Search source belongs to a different workspace"
                        )
                    if needle in source.title.casefold():
                        yield WorkspaceSearchHit(
                            ref, name, "source", source.title, source.source_id, offset
                        )
        if "note" in kinds and callable(
            list_notes := getattr(port, "list_notes", None)
        ):
            async for _, page in _owner_pages(
                lambda offset, ref=ref: list_notes(
                    ref,
                    ResearchNotePageRequest(
                        query=query, limit=SEARCH_OWNER_PAGE_SIZE, offset=offset
                    ),
                )
            ):
                for note in page.items:
                    if note.ref != ref:
                        raise ValueError("Search note belongs to a different workspace")
                    yield WorkspaceSearchHit(
                        ref, name, "note", note.title, note.note_id
                    )
        if "output" in kinds:
            async for output, title in _output_titles(port, ref):
                if needle in title.casefold():
                    yield WorkspaceSearchHit(
                        ref, name, "output", title, output.owner_id, output=output
                    )


async def _output_titles(port, ref):
    list_records = getattr(port, "list_output_records", None)
    list_outputs = getattr(port, "list_outputs", None)
    if not callable(list_records) and not callable(list_outputs):
        return
    offset = 0
    while True:
        kwargs = {"limit": SEARCH_OWNER_PAGE_SIZE, "offset": offset}
        if callable(list_records):
            page, records = await list_records(ref, **kwargs)
        else:
            page = await list_outputs(ref, **kwargs)
            records = None
        for output in page.items:
            if output.workspace != ref:
                raise ValueError("Search output belongs to a different workspace")
            record = (
                records.get(output)
                if records is not None
                else await port.resolve_output(output)
            )
            title = record.get("title") or record.get("name") if record else None
            if isinstance(title, str) and title.strip():
                yield output, title
        if not page.has_more:
            return
        if not page.items:
            raise ValueError("Workspace search received an incomplete output page")
        offset += len(page.items)
