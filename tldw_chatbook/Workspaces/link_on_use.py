"""Link-on-use: the one seam Console hand-offs share (TASK-34000.24).

Using a Library item in Console links it to the active workspace on the way
-- the decision recorded under TASK-32107 for conversations, ported to notes
in task-32536 and to media here. This module is that algorithm written once:

* read the item's memberships first (``link_membership`` is INSERT OR IGNORE
  and returns the EXISTING row, so a press reaching it through stale cached
  eligibility would otherwise claim a link it never made and an Undo would
  remove someone else's membership -- the Qodo finding on PR #2603);
* write only when the membership is absent;
* name the workspace by its display name (``Local Default`` for the built-in
  default, the record's ``name`` otherwise -- the ``display_state`` rule,
  imported, not copied), never its raw id;
* never raise: every failure is a ``blocked_reason`` sentence the caller can
  show as-is.

Notes and conversations keep their own link code for now (they are Done
tasks); the media hand-off and Search ▸ evidence ▸ ``u`` consume this seam.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .display_state import LibraryWorkspaceSourceRow, library_workspace_display_name

#: The existing no-active-workspace sentence (``eligibility.py``): the one
#: block a link cannot resolve.
NO_ACTIVE_WORKSPACE_COPY = (
    "Select an active workspace before using this item in Console."
)

#: Retrieval may hand a media record back wrapped in its legacy document
#: prefix or local scope envelope (the same two wrappers
#: ``LibraryRagResultRow.resolve_local_open_id`` accepts). Only the record
#: number is compared; a dash-separated fixture id such as ``media-1`` is a
#: literal row id in the Library test fixtures, so it matches by equality
#: first and is never unwrapped away from a row that spells it.
_MEDIA_ID_WRAPPER = re.compile(r"(?:local:media:|media[_-])?([0-9]+)")


@dataclass(frozen=True)
class LinkOnUseResult:
    """What one link-on-use call did.

    Attributes:
        workspace_id: The active workspace the item belongs to once this
            returns, or ``""`` when blocked.
        workspace_name: That workspace's display name, or ``""`` when blocked.
        written: Whether THIS call inserted the membership (the only case an
            Undo may remove it).
        blocked_reason: A whole sentence when nothing could be linked;
            ``""`` otherwise.
    """

    workspace_id: str = ""
    workspace_name: str = ""
    written: bool = False
    blocked_reason: str = ""

    @property
    def linked(self) -> bool:
        """Whether the item is a member of the named workspace now."""
        return bool(self.workspace_id) and not self.blocked_reason


def link_item_to_active_workspace(
    registry: Any,
    *,
    item_type: str,
    item_id: str,
    title: str = "",
) -> LinkOnUseResult:
    """Link one visible item into the active workspace, writing only if absent.

    Args:
        registry: The workspace registry service, or ``None`` when
            workspaces are unavailable.
        item_type: Canonical item type as memberships carry it (``"media"``,
            ``"note"``, ``"conversation"``).
        item_id: The item's membership id -- the SAME id the Library row
            model reads for the record (see ``resolve_workspace_item_id``).
        title: Title stored on a freshly written membership.

    Returns:
        A ``LinkOnUseResult``; never raises.
    """
    clean_type = str(item_type or "").strip()
    clean_id = str(item_id or "").strip()
    if registry is None or not clean_type or not clean_id:
        return LinkOnUseResult(blocked_reason=NO_ACTIVE_WORKSPACE_COPY)
    try:
        active = registry.get_active_workspace()
    except Exception as error:
        logger.warning(
            "Link on use: the active workspace could not be read",
            item_type=clean_type,
            error_type=type(error).__name__,
        )
        return LinkOnUseResult(
            blocked_reason=_write_failed_copy(clean_type, "the active workspace")
        )
    if active is None:
        return LinkOnUseResult(blocked_reason=NO_ACTIVE_WORKSPACE_COPY)
    workspace_id = str(active.workspace_id)
    workspace_name = library_workspace_display_name(active)
    try:
        already_linked = any(
            membership.workspace_id == workspace_id
            for membership in registry.get_item_memberships(
                item_type=clean_type, item_id=clean_id
            )
        )
        if not already_linked:
            registry.link_membership(
                workspace_id,
                item_type=clean_type,
                item_id=clean_id,
                title=str(title or "").strip() or clean_id,
            )
    except Exception as error:
        logger.warning(
            "Link on use: the item could not be linked to the active workspace",
            item_type=clean_type,
            error_type=type(error).__name__,
        )
        return LinkOnUseResult(
            blocked_reason=_write_failed_copy(clean_type, workspace_name)
        )
    return LinkOnUseResult(
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        written=not already_linked,
    )


def unlink_item_from_workspace(
    registry: Any,
    workspace_id: str,
    *,
    item_type: str,
    item_id: str,
) -> bool:
    """Remove one membership -- the inverse of a fresh link-on-use write.

    Takes the workspace id the RECEIPT names, never "whatever is active now":
    the active workspace can change under a standing receipt, and an Undo
    must only remove the row the press added.

    Args:
        registry: The workspace registry service, or ``None``.
        workspace_id: Workspace the link was written into.
        item_type: Membership item type.
        item_id: Membership item id.

    Returns:
        Whether a membership was removed; ``False`` on any failure.
    """
    clean_workspace = str(workspace_id or "").strip()
    clean_type = str(item_type or "").strip()
    clean_id = str(item_id or "").strip()
    if registry is None or not clean_workspace or not clean_type or not clean_id:
        return False
    try:
        return bool(
            registry.unlink_membership(
                clean_workspace, item_type=clean_type, item_id=clean_id
            )
        )
    except Exception as error:
        logger.warning(
            "Link on use: the membership could not be removed",
            item_type=clean_type,
            error_type=type(error).__name__,
        )
        return False


def resolve_workspace_item_id(
    rows: Sequence[LibraryWorkspaceSourceRow],
    item_type: str,
    source_id: str,
) -> str:
    """Map a hand-off's source id onto the id the Library row model reads.

    A membership written under any other spelling would leave the gate
    closed after linking: ``_library_workspace_source_row`` reads the record's
    own ``id``, while retrieval may carry the same media record wrapped
    (``local:media:3``, ``media_3``).

    Args:
        rows: The depth state's source rows.
        item_type: Canonical item type (``"media"``, ``"note"``,
            ``"conversation"``).
        source_id: The hand-off's source id.

    Returns:
        The matching row's ``item_id``; the raw ``source_id`` when no row
        matches (never a guess).
    """
    clean_type = str(item_type or "").strip()
    clean_source = str(source_id or "").strip()
    if not clean_source:
        return ""
    candidates = [row for row in rows if row.item_type == clean_type]
    for row in candidates:
        if row.item_id == clean_source:
            return row.item_id
    if clean_type != "media":
        return clean_source
    source_number = _media_record_number(clean_source)
    if source_number is None:
        return clean_source
    for row in candidates:
        if _media_record_number(row.item_id) == source_number:
            return row.item_id
    return clean_source


def _media_record_number(value: str) -> int | None:
    match = _MEDIA_ID_WRAPPER.fullmatch(value)
    if match is None:
        return None
    number = int(match[1])
    return number if number > 0 else None


def _write_failed_copy(item_type: str, workspace_name: str) -> str:
    return (
        f"Can't use this {item_type} in Console — it could not be linked to "
        f"{workspace_name}. Next: try again."
    )
