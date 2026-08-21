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
    """Raised when STRICT ownership encounters an unknown stable identifier."""

    def __init__(self, identifiers: tuple[str, ...]) -> None:
        self.identifiers = identifiers
        super().__init__("Unowned Inspector content: " + ", ".join(identifiers))


INSPECTOR_BOUNDARY_ORDER = (
    "Sources",
    "Scope",
    "Changed Files",
    "Run status",
    "Run",
    "Source Readiness",
    "Tools",
    "Approvals",
    "Artifacts",
    "Selected Conversation",
    "Session Defaults",
    "Selected Message",
    "Changes",
    "Chat Dictionaries",
    "World Books",
    "Session Settings",
    "Live Work",
)

SPECIALIZED_CONTENT_OWNERS = {
    "console-project-instruction-status": "Sources",
    "console-staged-context-tray": "Sources",
    "console-retrieval-scope-row": "Scope",
    "console-changed-files-section": "Changed Files",
    "console-inspector-run-status-summary": "Run status",
    "console-settings-summary": "Session Settings",
    "console-pending-launch-card": "Live Work",
    "console-live-work-source-readiness": "Live Work",
}

ROW_IDS = {
    "Run recipe": "console-inspector-run-recipe",
    "Live work": "console-inspector-live-work",
    "Setup": "console-inspector-setup",
    "Send blocked": "console-inspector-send-blocked",
    "Recovery action": "console-inspector-recovery-action",
    "Blocked impact": "console-inspector-blocked-impact",
    "Next action": "console-inspector-next-action",
    "Provider": "console-inspector-provider",
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
        ("Sources", "RAG/source", "Evidence", "Authority"),
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


@dataclass(frozen=True)
class InspectorOwnedContent:
    """Known content plus safe identifiers for anything unowned."""

    state: ConsoleInspectorState
    row_owners: tuple[str | None, ...]
    action_owners: tuple[str | None, ...]
    unknown_identifiers: tuple[str, ...]

    @property
    def incomplete(self) -> bool:
        """Whether the source snapshot contained unowned content."""
        return bool(self.unknown_identifiers)

    def rows_for(self, owner: str) -> tuple[tuple[int, ConsoleDisplayRow], ...]:
        """Return indexed, known rows owned by one direct boundary."""
        return tuple(
            (index, row)
            for index, (row, row_owner) in enumerate(
                zip(self.state.rows, self.row_owners)
            )
            if row_owner == owner
        )

    def actions_for(self, owner: str) -> tuple[ConsoleInspectorAction, ...]:
        """Return known actions owned by one direct boundary."""
        return tuple(
            action
            for action, action_owner in zip(self.state.actions, self.action_owners)
            if action_owner == owner
        )

    @property
    def known_actions(self) -> tuple[ConsoleInspectorAction, ...]:
        """Return ordinary actions accepted by the ownership inventory."""
        return tuple(
            action
            for action, owner in zip(self.state.actions, self.action_owners)
            if owner is not None
        )


def classify_inspector_content(
    state: ConsoleInspectorState,
    policy: InspectorOwnershipPolicy,
) -> InspectorOwnedContent:
    """Classify one Inspector snapshot without consulting process state.

    Args:
        state: Inspector display snapshot to classify.
        policy: Injected handling policy for unowned rows and actions.

    Returns:
        The known ownership projection and safe structural fingerprint.

    Raises:
        UnownedInspectorContentError: If STRICT policy finds unknown content.
    """
    row_owners = tuple(ROW_OWNERS.get(row.label) for row in state.rows)
    action_owners = tuple(
        ACTION_OWNERS.get(action.widget_id) for action in state.actions
    )
    unknown_identifiers = tuple(
        sorted(
            {
                *(
                    f"row:{row.label}"
                    for row, owner in zip(state.rows, row_owners)
                    if owner is None
                ),
                *(
                    f"action:{action.widget_id}"
                    for action, owner in zip(state.actions, action_owners)
                    if owner is None
                ),
            }
        )
    )
    if unknown_identifiers and policy is InspectorOwnershipPolicy.STRICT:
        raise UnownedInspectorContentError(unknown_identifiers)
    return InspectorOwnedContent(
        state=state,
        row_owners=row_owners,
        action_owners=action_owners,
        unknown_identifiers=unknown_identifiers,
    )


def dynamic_item_owner(
    state: ConsoleInspectorState,
    collection_name: str,
    index: int,
) -> str:
    """Return the explicit owner for one dictionary or World Book item.

    Accessing the item as part of the lookup makes the inventory assertion
    cover real emitted collection members rather than field names alone.
    """
    collection = getattr(state, collection_name)
    collection[index]
    return DYNAMIC_COLLECTION_OWNERS[collection_name]
