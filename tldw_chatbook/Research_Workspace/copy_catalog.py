"""Read the scoped catalog and transfer policy for workspace publication."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker
from tldw_chatbook.Workspaces.models import WorkspaceTransferPolicy

from .contracts import (
    MAX_RESEARCH_SELECTION_IDS,
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
)
from .copy_manifest import ResearchCopyCandidate
from .copy_models import CopyItemKind, CopyOwnerRef
from .quick_notes import ResearchNotePageRequest
from .studio_models import StudioOwnerKind


async def _page_items(fetch: Callable) -> tuple:
    items = []
    offset = 0
    while True:
        page = await fetch(offset)
        items.extend(page.items)
        if len(items) > MAX_RESEARCH_SELECTION_IDS or (
            page.has_more and not page.items
        ):
            raise ValueError("Copy inspection exceeded the workspace page limit")
        if not page.has_more:
            return tuple(items)
        if len(items) == MAX_RESEARCH_SELECTION_IDS:
            raise ValueError("Copy inspection exceeded the workspace page limit")
        offset += len(page.items)


async def capture_copy_candidates(
    port, ref: QualifiedWorkspaceRef
) -> tuple[ResearchCopyCandidate, ...]:
    """Enumerate the exact native items associated with one workspace."""
    candidates = {}
    for source in await _page_items(
        lambda offset: port.list_sources(ref, limit=100, offset=offset)
    ):
        owner = CopyOwnerRef(
            ref,
            CopyItemKind.MEDIA,
            source.catalog_item_id,
            source.catalog_item_version,
        )
        candidates[owner.kind, owner.owner_id] = ResearchCopyCandidate(
            owner, membership_id=source.source_id
        )
    try:
        notes = await _page_items(
            lambda offset: port.list_notes(
                ref, ResearchNotePageRequest(limit=100, offset=offset)
            )
        )
    except CapabilityUnavailableError:
        notes = ()
    for note in notes:
        owner = CopyOwnerRef(ref, CopyItemKind.NOTE, note.note_id, note.version)
        candidates[owner.kind, owner.owner_id] = ResearchCopyCandidate(
            owner, estimated_bytes=len(note.content.encode("utf-8"))
        )
    try:
        outputs = await _page_items(
            lambda offset: port.list_outputs(ref, limit=100, offset=offset)
        )
    except CapabilityUnavailableError:
        outputs = ()
    for output in outputs:
        if output.workspace != ref:
            raise ValueError("The output list returned a different workspace")
        if output.owner_kind in {
            StudioOwnerKind.LOCAL_STUDY_DECK,
            StudioOwnerKind.SERVER_STUDY_DECK,
        }:
            kind = CopyItemKind.STUDY_DECK
        elif output.owner_kind in {
            StudioOwnerKind.LOCAL_QUIZ,
            StudioOwnerKind.SERVER_QUIZ,
        }:
            kind = CopyItemKind.QUIZ
        else:
            kind = CopyItemKind.ARTIFACT
        owner = CopyOwnerRef(ref, kind, output.owner_id, output.owner_version)
        candidates[kind, owner.owner_id] = ResearchCopyCandidate(owner)
    return tuple(candidates.values())


async def get_local_copy_policy(
    registry, candidate: ResearchCopyCandidate, *, require_membership: bool = False
) -> WorkspaceTransferPolicy:
    """Apply the outbound policy on the exact workspace membership."""
    ref = candidate.source
    item_type, role = {
        CopyItemKind.MEDIA: ("media", "source"),
        CopyItemKind.NOTE: ("note", "note"),
        CopyItemKind.ARTIFACT: ("chatbook", "output"),
        CopyItemKind.STUDY_DECK: ("study_deck", "output"),
        CopyItemKind.QUIZ: ("study_quiz", "output"),
    }.get(ref.kind, ("", ""))
    if not item_type:
        return candidate.transfer_policy
    links = await asyncio.to_thread(
        run_finite_local_worker, registry.get_item_memberships, item_type, ref.owner_id
    )
    match = next(
        (
            link
            for link in links
            if link.workspace_id == ref.workspace.workspace_id and link.role == role
        ),
        None,
    )
    if match is None and require_membership:
        raise ValueError(
            "A workspace membership changed. Review the selection and try again."
        )
    requested = WorkspaceTransferPolicy(candidate.transfer_policy)
    if requested is WorkspaceTransferPolicy.LOCAL_ONLY or match is None:
        return requested
    stored = WorkspaceTransferPolicy(match.transfer_policy)
    return requested if stored is WorkspaceTransferPolicy.COPY else stored
