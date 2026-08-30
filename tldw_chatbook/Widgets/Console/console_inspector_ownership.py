"""Semantic ownership policy for Console Inspector content."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    CONSOLE_INSPECTOR_REVIEW_CHANGES_ID,
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
)


class InspectorOwnershipPolicy(Enum):
    """Behavior when an Inspector snapshot contains unowned content."""

    STRICT = "strict"
    RESILIENT = "resilient"


class UnownedInspectorContentError(ValueError):
    """Raised when STRICT ownership encounters an invalid stable identifier."""

    def __init__(self, identifiers: tuple[str, ...]) -> None:
        self.identifiers = identifiers
        super().__init__("Unowned Inspector content: " + ", ".join(identifiers))


ROW_IDS = {
    "Run recipe": "console-inspector-run-recipe",
    "Live work": "console-inspector-live-work",
    "Setup": "console-inspector-setup",
    "Send blocked": "console-inspector-send-blocked",
    "Recovery action": "console-inspector-recovery-action",
    "Blocked impact": "console-inspector-blocked-impact",
    "Next action": "console-inspector-next-action",
    "Provider": "console-inspector-provider",
    # TASK-24610: the run inspector's retrieval-status row. It was called
    # "Sources", which is what the staged-context tray heading, the pinned
    # authority row and the status chip all call STAGED CONTEXT -- four
    # visible-at-once uses of one noun for two concepts. The widget id keeps
    # its historical name so DOM and CSS references stay stable.
    "Retrieval": "console-inspector-retrieval",
    # The pre-rename label is retained as a classification alias and is NOT
    # emitted by any producer. It cannot simply be deleted: this classifier
    # is STRICT and RAISES `UnownedInspectorContentError` on a label it does
    # not own, so a persisted or replayed snapshot carrying the old label
    # would crash the Inspector rather than mislabel a row. It keeps a
    # DISTINCT widget id: sharing one with "Retrieval" makes a state holding
    # both mount two widgets under the same DOM id.
    "Sources": "console-inspector-sources",
    "Tools": "console-inspector-tools",
    "MCP": "console-inspector-mcp",
    "RAG/source": "console-inspector-rag-source",
    "Evidence": "console-inspector-evidence",
    "Authority": "console-inspector-authority",
    "Artifacts": "console-inspector-artifacts",
    "Approvals": "console-inspector-approvals",
    "Selected message": "console-inspector-selected-message",
    "Selected conversation": "console-inspector-selected-conversation",
    "Conversation source": "console-inspector-conversation-source",
    "Workspace": "console-inspector-workspace",
    "Resume state": "console-inspector-resume-state",
    "Prefill (next send only)": "console-inspector-prefill-one-shot",
    "Prefill (pinned)": "console-inspector-prefill-pinned",
    "Session provider": "console-inspector-session-provider",
    "Session model": "console-inspector-session-model",
    "Session endpoint": "console-inspector-session-endpoint",
    "Session sampling": "console-inspector-session-sampling",
    "Session persona": "console-inspector-session-persona",
    "Message actions": "console-inspector-message-actions",
    "Keyboard": "console-inspector-message-keyboard",
    "Variants": "console-inspector-message-variants",
    "Excerpt": "console-inspector-message-excerpt",
    "Delete confirmation": "console-inspector-delete-confirmation",
}

ROW_GROUPS = (
    (
        "Run",
        "console-inspector-run-heading",
        (
            "Run recipe",
            "Live work",
            "Setup",
            "Send blocked",
            "Recovery action",
            "Blocked impact",
            "Next action",
            "Provider",
        ),
    ),
    (
        "Source Readiness",
        "console-inspector-source-readiness-heading",
        # "Sources" is the pre-TASK-24610 alias for "Retrieval"; see ROW_IDS.
        ("Retrieval", "Sources", "RAG/source", "Evidence", "Authority"),
    ),
    ("Tools", "console-inspector-tools-heading", ("Tools", "MCP")),
    ("Approvals", "console-inspector-approvals-heading", ("Approvals",)),
    ("Artifacts", "console-inspector-artifacts-heading", ("Artifacts",)),
    (
        "Selected Conversation",
        "console-inspector-selected-conversation-heading",
        (
            "Selected conversation",
            "Conversation source",
            "Workspace",
            "Resume state",
            "Prefill (next send only)",
            "Prefill (pinned)",
        ),
    ),
    (
        "Session Defaults",
        "console-inspector-session-defaults-heading",
        (
            "Session provider",
            "Session model",
            "Session endpoint",
            "Session sampling",
            "Session persona",
        ),
    ),
    (
        "Selected Message",
        "console-inspector-selected-message-heading",
        (
            "Selected message",
            "Message actions",
            "Keyboard",
            "Variants",
            "Excerpt",
            "Delete confirmation",
        ),
    ),
    ("Changes", "console-inspector-changes-heading", ()),
)

ACTION_GROUPS = {
    "Artifacts": (CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,),
    "Approvals": (CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,),
    "Changes": (CONSOLE_INSPECTOR_REVIEW_CHANGES_ID,),
}

ROW_OWNERS = {
    label: owner for owner, _heading_id, labels in ROW_GROUPS for label in labels
}
ACTION_OWNERS = {
    action_id: owner
    for owner, action_ids in ACTION_GROUPS.items()
    for action_id in action_ids
}
DYNAMIC_COLLECTION_OWNERS = {
    "dictionary_rows": "Chat Dictionaries",
    "dictionary_actions": "Chat Dictionaries",
    "world_book_rows": "World Books",
    "world_book_actions": "World Books",
}

_RESERVED_WIDGET_IDS = {
    "console-inspector-run-status-summary",
}


@dataclass(frozen=True)
class OwnedInspectorRow:
    """One row accepted into the canonical mounted projection."""

    widget_id: str
    owner: str
    row: ConsoleDisplayRow


@dataclass(frozen=True)
class OwnedInspectorAction:
    """One action accepted into the canonical mounted projection."""

    owner: str
    action: ConsoleInspectorAction


@dataclass(frozen=True)
class InspectorOwnedContent:
    """Canonical safe render projection plus its structural fingerprint."""

    rows: tuple[OwnedInspectorRow, ...]
    actions: tuple[OwnedInspectorAction, ...]
    dictionary_rows: tuple[OwnedInspectorRow, ...]
    dictionary_actions: tuple[OwnedInspectorAction, ...]
    world_book_rows: tuple[OwnedInspectorRow, ...]
    world_book_actions: tuple[OwnedInspectorAction, ...]
    unknown_identifiers: tuple[str, ...]

    @property
    def incomplete(self) -> bool:
        """Whether the source snapshot contained unknown or colliding content."""
        return bool(self.unknown_identifiers)

    def rows_for(self, owner: str) -> tuple[OwnedInspectorRow, ...]:
        """Return accepted ordinary rows owned by one direct boundary."""
        return tuple(entry for entry in self.rows if entry.owner == owner)

    def actions_for(self, owner: str) -> tuple[ConsoleInspectorAction, ...]:
        """Return accepted ordinary actions owned by one direct boundary."""
        return tuple(entry.action for entry in self.actions if entry.owner == owner)

    @property
    def known_actions(self) -> tuple[ConsoleInspectorAction, ...]:
        """Return ordinary actions accepted by the ownership inventory."""
        return tuple(entry.action for entry in self.actions)


def classify_inspector_content(
    state: ConsoleInspectorState,
    policy: InspectorOwnershipPolicy,
) -> InspectorOwnedContent:
    """Build the sole safe render projection for one Inspector snapshot.

    The classifier owns no process configuration. It validates both inventory
    membership and the Textual IDs the composed tree would mount. RESILIENT
    mode retains one deterministic inventory-ordered item and omits later
    collisions; STRICT raises before a caller can replace the mounted tree.

    Args:
        state: Inspector display snapshot to classify.
        policy: Injected handling policy for invalid rows and actions.

    Returns:
        The filtered ownership projection and safe structural fingerprint.

    Raises:
        UnownedInspectorContentError: If STRICT policy finds invalid content.
    """
    invalid: set[str] = set()
    claimed_ids = set(_RESERVED_WIDGET_IDS)

    def claim_row(
        *, widget_id: str, owner: str, row: ConsoleDisplayRow, identifier: str
    ) -> OwnedInspectorRow | None:
        if widget_id in claimed_ids:
            invalid.add(identifier)
            return None
        claimed_ids.add(widget_id)
        return OwnedInspectorRow(widget_id=widget_id, owner=owner, row=row)

    def claim_action(
        *, owner: str, action: ConsoleInspectorAction
    ) -> OwnedInspectorAction | None:
        action_ids = {action.widget_id}
        if not action.enabled and action.disabled_reason:
            action_ids.add(f"{action.widget_id}-reason")
        if action_ids & claimed_ids:
            invalid.add(f"action:{action.widget_id}")
            return None
        claimed_ids.update(action_ids)
        return OwnedInspectorAction(owner=owner, action=action)

    rows_by_label: dict[str, list[ConsoleDisplayRow]] = {}
    for row in state.rows:
        if row.label not in ROW_OWNERS:
            invalid.add(f"row:{row.label}")
            continue
        rows_by_label.setdefault(row.label, []).append(row)

    for owner, heading_id, labels in ROW_GROUPS:
        has_rows = any(rows_by_label.get(label) for label in labels)
        has_enabled_action = any(
            ACTION_OWNERS.get(action.widget_id) == owner and action.enabled
            for action in state.actions
        )
        if has_rows or has_enabled_action:
            claimed_ids.add(heading_id)
    if state.dictionary_rows or state.dictionary_actions:
        claimed_ids.add("console-inspector-dictionaries-heading")
    if state.world_book_rows or state.world_book_actions:
        claimed_ids.add("console-inspector-worldbooks-heading")

    rows: list[OwnedInspectorRow] = []
    for owner, _heading_id, labels in ROW_GROUPS:
        for label in labels:
            for row in rows_by_label.get(label, ()):
                entry = claim_row(
                    widget_id=ROW_IDS[label],
                    owner=owner,
                    row=row,
                    identifier=f"row:{label}",
                )
                if entry is not None:
                    rows.append(entry)

    ordinary_actions: list[OwnedInspectorAction] = []
    for owner, _heading_id, _labels in ROW_GROUPS:
        for action in state.actions:
            if ACTION_OWNERS.get(action.widget_id) != owner:
                continue
            entry = claim_action(owner=owner, action=action)
            if entry is not None:
                ordinary_actions.append(entry)
    invalid.update(
        f"action:{action.widget_id}"
        for action in state.actions
        if action.widget_id not in ACTION_OWNERS
    )

    def project_dynamic_rows(
        collection_name: str, prefix: str
    ) -> tuple[OwnedInspectorRow, ...]:
        owner = DYNAMIC_COLLECTION_OWNERS[collection_name]
        projected: list[OwnedInspectorRow] = []
        for index, row in enumerate(getattr(state, collection_name)):
            widget_id = f"{prefix}-{index}"
            entry = claim_row(
                widget_id=widget_id,
                owner=owner,
                row=row,
                identifier=f"row:{widget_id}",
            )
            if entry is not None:
                projected.append(entry)
        return tuple(projected)

    def project_dynamic_actions(
        collection_name: str,
    ) -> tuple[OwnedInspectorAction, ...]:
        owner = DYNAMIC_COLLECTION_OWNERS[collection_name]
        projected: list[OwnedInspectorAction] = []
        for action in getattr(state, collection_name):
            entry = claim_action(owner=owner, action=action)
            if entry is not None:
                projected.append(entry)
        return tuple(projected)

    dictionary_rows = project_dynamic_rows(
        "dictionary_rows", "console-inspector-dictionaries-row"
    )
    dictionary_actions = project_dynamic_actions("dictionary_actions")
    world_book_rows = project_dynamic_rows(
        "world_book_rows", "console-inspector-worldbooks-row"
    )
    world_book_actions = project_dynamic_actions("world_book_actions")

    unknown_identifiers = tuple(sorted(invalid))
    if unknown_identifiers and policy is InspectorOwnershipPolicy.STRICT:
        raise UnownedInspectorContentError(unknown_identifiers)
    return InspectorOwnedContent(
        rows=tuple(rows),
        actions=tuple(ordinary_actions),
        dictionary_rows=dictionary_rows,
        dictionary_actions=dictionary_actions,
        world_book_rows=world_book_rows,
        world_book_actions=world_book_actions,
        unknown_identifiers=unknown_identifiers,
    )
