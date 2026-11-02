# conflict_resolver.py
# Description: Handle conflicts when importing chatbook content
#
"""
Conflict Resolver
-----------------

Handles conflict resolution when importing content that may already exist.

Supported strategies: ASK (interactive), SKIP, RENAME and REPLACE. MERGE is
deliberately NOT a supported strategy: it was never implemented anywhere
(the resolver used to rewrite it to RENAME without a word, and the dead
``merge_notes`` helper was reachable by no caller), so the resolver now
rejects it loudly and ``ChatbookImporter.import_chatbook`` refuses the
strategy up front, before any item is written (TASK-38594).
"""

from enum import Enum
from typing import Dict, Any, Optional, Callable


_PREVIEW_CHARS = 100


def _preview(text: Any) -> str:
    """Shorten `text` for a conflict prompt, marking it only if it was cut.

    An unconditional ``+ "..."`` tells the user a 3-character description was
    truncated when it was not (tier-2 review S17 P3).
    """
    value = str(text or "")
    if len(value) <= _PREVIEW_CHARS:
        return value
    return value[:_PREVIEW_CHARS] + "..."


class ConflictResolution(Enum):
    """Conflict resolution strategies."""

    ASK = "ask"  # Ask user for each conflict (interactive)
    SKIP = "skip"  # Skip conflicting items
    RENAME = "rename"  # Rename imported items
    REPLACE = "replace"  # Replace existing items
    MERGE = "merge"  # NOT IMPLEMENTED: rejected loudly by the importer


def _reject_merge(strategy: ConflictResolution) -> None:
    """Refuse the never-implemented merge strategy instead of substituting.

    The resolver used to quietly rewrite MERGE to RENAME "for now"; a
    strategy the user picked that the code does not honour must be an
    error, never a silent substitute (TASK-38594).
    """
    if strategy == ConflictResolution.MERGE:
        raise ValueError(
            "Conflict resolution strategy 'merge' is not implemented; "
            "use skip, rename or replace."
        )


class ConflictResolver:
    """Handle conflicts during chatbook import."""

    def __init__(self, ask_callback: Optional[Callable] = None):
        """
        Initialize conflict resolver.

        Args:
            ask_callback: Optional callback for interactive conflict resolution
        """
        self.ask_callback = ask_callback

    def resolve_conversation_conflict(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        default_resolution: ConflictResolution,
    ) -> ConflictResolution:
        """
        Resolve conflict for conversations.

        Args:
            existing: Existing conversation data
            incoming: Incoming conversation data
            default_resolution: Default resolution strategy

        Returns:
            Resolution to apply

        Raises:
            ValueError: If ``default_resolution`` is MERGE (unimplemented).
        """
        if default_resolution == ConflictResolution.ASK:
            if not self.ask_callback:
                return ConflictResolution.SKIP
            # Show conflict details and ask user
            conflict_info = {
                "type": "conversation",
                "existing_name": existing.get("conversation_name", "Unknown"),
                "existing_created": existing.get("created_at", "Unknown"),
                "existing_messages": existing.get("message_count", 0),
                "incoming_name": incoming.get("name", "Unknown"),
                "incoming_created": incoming.get("created_at", "Unknown"),
                "incoming_messages": len(incoming.get("messages", [])),
            }

            return self.ask_callback(conflict_info)

        _reject_merge(default_resolution)
        return default_resolution

    def resolve_note_conflict(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        default_resolution: ConflictResolution,
    ) -> ConflictResolution:
        """
        Resolve conflict for notes.

        Args:
            existing: Existing note data
            incoming: Incoming note data
            default_resolution: Default resolution strategy

        Returns:
            Resolution to apply

        Raises:
            ValueError: If ``default_resolution`` is MERGE (unimplemented).
        """
        if default_resolution == ConflictResolution.ASK:
            if not self.ask_callback:
                return ConflictResolution.SKIP
            # Show conflict details and ask user
            conflict_info = {
                "type": "note",
                "existing_title": existing.get("title", "Unknown"),
                "existing_created": existing.get("created_at", "Unknown"),
                "existing_updated": existing.get("updated_at", "Unknown"),
                "existing_length": len(existing.get("content", "")),
                "incoming_title": incoming.get("title", "Unknown"),
                "incoming_length": len(incoming.get("content", "")),
            }

            return self.ask_callback(conflict_info)

        _reject_merge(default_resolution)
        return default_resolution

    def resolve_character_conflict(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        default_resolution: ConflictResolution,
    ) -> ConflictResolution:
        """
        Resolve conflict for characters.

        Args:
            existing: Existing character data
            incoming: Incoming character data
            default_resolution: Default resolution strategy

        Returns:
            Resolution to apply

        Raises:
            ValueError: If ``default_resolution`` is MERGE (unimplemented).
        """
        if default_resolution == ConflictResolution.ASK:
            if not self.ask_callback:
                return ConflictResolution.SKIP
            # Show conflict details and ask user
            conflict_info = {
                "type": "character",
                "existing_name": existing.get("name", "Unknown"),
                "existing_created": existing.get("created_at", "Unknown"),
                "existing_description": _preview(existing.get("description")),
                "incoming_name": incoming.get("name", "Unknown"),
                "incoming_description": _preview(incoming.get("description")),
            }

            return self.ask_callback(conflict_info)

        _reject_merge(default_resolution)
        return default_resolution

    def resolve_prompt_conflict(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        default_resolution: ConflictResolution,
    ) -> ConflictResolution:
        """
        Resolve conflict for prompts.

        Args:
            existing: Existing prompt data
            incoming: Incoming prompt data
            default_resolution: Default resolution strategy

        Returns:
            Resolution to apply

        Raises:
            ValueError: If ``default_resolution`` is MERGE (unimplemented).
        """
        if default_resolution == ConflictResolution.ASK:
            if not self.ask_callback:
                return ConflictResolution.SKIP
            # Show conflict details and ask user
            conflict_info = {
                "type": "prompt",
                "existing_name": existing.get("name", "Unknown"),
                "existing_created": existing.get("created_at", "Unknown"),
                "incoming_name": incoming.get("name", "Unknown"),
            }

            return self.ask_callback(conflict_info)

        _reject_merge(default_resolution)
        return default_resolution
