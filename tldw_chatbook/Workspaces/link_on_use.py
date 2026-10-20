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

``decide_link_on_use`` (fix round 1, I-2) is the ONE gate-to-link decision
both entry points -- the Media reader's ``c`` and Search ▸ evidence ▸ ``u`` --
call, so they cannot disagree about what a row's block means or what an item
the depth state never sampled should do. Notes and conversations keep their
own link code for now (they are Done tasks).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from loguru import logger

from .display_state import LibraryWorkspaceSourceRow, library_workspace_display_name
from .eligibility import linkable_ineligibility_label

#: The existing no-active-workspace sentence (``eligibility.py``). Under
#: task-15120 no active workspace IS the global scope, so the callers treat
#: this block as "stage as-is", never as a refusal.
NO_ACTIVE_WORKSPACE_COPY = (
    "Select an active workspace before using this item in Console."
)


def registry_unavailable_copy(item_type: str) -> str:
    """The notes hand-off's sentence for a missing registry (fix round 1, M-6).

    Args:
        item_type: Canonical item type, for the sentence.

    Returns:
        A whole sentence with the next step.
    """
    return (
        f"Can't use this {item_type or 'item'} in Console — the workspace "
        "registry is unavailable. Next: restart Chatbook, then try again."
    )


class UnlinkOutcome(Enum):
    """What ``unlink_item_from_workspace`` found (fix round 1, M-4).

    ``ABSENT`` -- the membership was already gone (removed elsewhere) -- is
    the outcome an Undo wanted and must not read as a failure to retry.
    """

    REMOVED = "removed"
    ABSENT = "absent"
    FAILED = "failed"


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


@dataclass(frozen=True)
class LinkOnUseDecision:
    """The one answer both entry points act on.

    Attributes:
        kind: ``"refuse"`` (show ``copy``, stage nothing), ``"linked"`` (the
            item is a member now -- ``result`` says whether this press wrote
            it -- stage it and show the receipt) or ``"stage"`` (stage as-is,
            no link, no receipt: a non-local item, or the global scope).
        copy: The refusal sentence for ``"refuse"``; ``""`` otherwise.
        result: The seam's result for ``"linked"``; ``None`` otherwise.
    """

    kind: Literal["refuse", "linked", "stage"]
    copy: str = ""
    result: LinkOnUseResult | None = None


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
            model reads for the record (the record's own id).
        title: Title stored on a freshly written membership.

    Returns:
        A ``LinkOnUseResult``; never raises.
    """
    clean_type = str(item_type or "").strip()
    clean_id = str(item_id or "").strip()
    if registry is None or not clean_type or not clean_id:
        return LinkOnUseResult(blocked_reason=registry_unavailable_copy(clean_type))
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
) -> UnlinkOutcome:
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
        ``REMOVED`` when a membership was deleted, ``ABSENT`` when there was
        none to delete, ``FAILED`` on any failure (including no registry).
    """
    clean_workspace = str(workspace_id or "").strip()
    clean_type = str(item_type or "").strip()
    clean_id = str(item_id or "").strip()
    if registry is None or not clean_workspace or not clean_type or not clean_id:
        return UnlinkOutcome.FAILED
    try:
        removed = bool(
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
        return UnlinkOutcome.FAILED
    return UnlinkOutcome.REMOVED if removed else UnlinkOutcome.ABSENT


def decide_link_on_use(
    rows: Sequence[LibraryWorkspaceSourceRow] | None,
    registry: Any,
    *,
    item_type: str,
    item_id: str,
    title: str = "",
    local: bool = True,
) -> LinkOnUseDecision:
    """Decide what using one item in Console does to its workspace membership.

    The rules, in order (fix round 1, I-1/I-2):

    * a non-local item (a server detail, a server result) stages as-is --
      there is no local membership to write;
    * a depth-state row that EXISTS and carries a block a link cannot resolve
      refuses with that row's own sentence;
    * otherwise the seam links: a row the depth state never sampled (the
      first-page sample holds 50 local media; the power user's library is
      larger) is linked exactly like a sampled, linkable row -- the seam
      reads memberships itself and needs no row;
    * a seam block means the global scope (no active workspace: stage
      as-is, task-15120) or, for a row the gate already called eligible,
      stage as-is too; any other block refuses with the seam's sentence.

    Args:
        rows: The depth state's source rows, or ``None``.
        registry: The workspace registry service, or ``None``.
        item_type: Canonical item type (``"media"``, ``"note"``,
            ``"conversation"``).
        item_id: The item's record id (the one the row model reads).
        title: Title for a freshly written membership.
        local: Whether the item is a local record.

    Returns:
        A ``LinkOnUseDecision``; never raises.
    """
    clean_type = str(item_type or "").strip()
    clean_id = str(item_id or "").strip()
    if not local or not clean_type or not clean_id:
        return LinkOnUseDecision(kind="stage")
    row = next(
        (
            candidate
            for candidate in (rows or ())
            if candidate.item_type == clean_type and candidate.item_id == clean_id
        ),
        None,
    )
    if (
        row is not None
        and not row.active_context_eligible
        and not linkable_ineligibility_label(row.reason_code)
    ):
        return LinkOnUseDecision(
            kind="refuse",
            copy=row.recovery_copy
            or f"This {clean_type} can't be used in Console from this workspace.",
        )
    result = link_item_to_active_workspace(
        registry, item_type=clean_type, item_id=clean_id, title=title
    )
    if result.blocked_reason:
        if result.blocked_reason == NO_ACTIVE_WORKSPACE_COPY or (
            row is not None and row.active_context_eligible
        ):
            return LinkOnUseDecision(kind="stage")
        return LinkOnUseDecision(kind="refuse", copy=result.blocked_reason)
    return LinkOnUseDecision(kind="linked", result=result)


def _write_failed_copy(item_type: str, workspace_name: str) -> str:
    return (
        f"Can't use this {item_type} in Console — it could not be linked to "
        f"{workspace_name}. Next: try again."
    )
