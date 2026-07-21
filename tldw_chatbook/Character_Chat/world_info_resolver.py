"""Shared world-info send-path resolver (Roleplay P2g).

Builds the world-info-injected message text for a send, composing the same
sources the legacy chat_events path does (conversation-attached books ∪
character-attached snapshots ∪ a native character_book) — so the native Console
(P2g-1) and, later, the legacy path (P2g-3) share one faithful implementation.
Never raises: any problem returns the message text unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


def _collect_active_world_books(
    db: Any, conversation_id: Optional[str], char_data: Optional[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect the world books that apply to this send.

    Args:
        db: A ``CharactersRAGDB`` (or None).
        conversation_id: The active conversation (string UUID) or None.
        char_data: The active character record, or None (native Console).

    Returns:
        ``(world_books, has_character_book)`` — the conversation-attached books
        unioned with character-attached snapshots (conversation wins on a name
        collision), and whether ``char_data`` carries a native ``character_book``.
        Never raises.
    """
    world_books: List[Dict[str, Any]] = []
    if conversation_id and db is not None:
        try:
            from .world_book_manager import WorldBookManager

            world_books = WorldBookManager(db).get_world_books_for_conversation(
                str(conversation_id), enabled_only=True
            )
        except Exception:
            logger.opt(exception=True).debug(
                "world-info: could not load conversation world books"
            )
            world_books = []

    has_character_book = False
    extensions = char_data.get("extensions", {}) if isinstance(char_data, dict) else {}
    if isinstance(extensions, dict) and extensions.get("character_book"):
        has_character_book = True

    try:
        from .world_book_manager import resolve_character_world_books

        character_books = resolve_character_world_books(
            char_data, {str(b.get("name")) for b in world_books}
        )
    except Exception:
        character_books = []
    if character_books:
        world_books = world_books + character_books

    return world_books, has_character_book


def apply_world_info_to_message(
    db: Any,
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
    message_text: str,
    history: List[Dict[str, Any]],
) -> str:
    """Return ``message_text`` with matched world-info injected, or unchanged.

    Args:
        db: A ``CharactersRAGDB`` (or None).
        conversation_id: The active conversation (string UUID) or None.
        char_data: The active character record, or None (conversation-only).
        message_text: The current user message text (already plain string).
        history: Prior messages as ``{"role","content": str}`` (string content;
            the caller normalizes multimodal content to text before calling).

    Returns:
        The message text wrapped with world-info injections in the order
        ``at_start → before_char → message → after_char → at_end`` (``"\\n\\n"``
        separated), or the original ``message_text`` when nothing matches / no
        books / no conversation / any error. Never raises.
    """
    if not isinstance(message_text, str):
        return message_text
    try:
        world_books, has_character_book = _collect_active_world_books(
            db, conversation_id, char_data
        )
        if not (has_character_book or world_books):
            return message_text
        from .world_info_processor import WorldInfoProcessor

        processor = WorldInfoProcessor(
            character_data=char_data if has_character_book else None,
            world_books=world_books or None,
        )
        result = processor.process_messages(message_text, history or [])
        if not result.get("matched_entries"):
            return message_text
        formatted = processor.format_injections(result.get("injections", {}))
        parts: List[str] = []
        if formatted.get("at_start"):
            parts.append(formatted["at_start"])
        if formatted.get("before_char"):
            parts.append(formatted["before_char"])
        parts.append(message_text)
        if formatted.get("after_char"):
            parts.append(formatted["after_char"])
        if formatted.get("at_end"):
            parts.append(formatted["at_end"])
        return "\n\n".join(parts)
    except Exception:
        logger.opt(exception=True).debug(
            "world-info: apply failed; returning message text unchanged"
        )
        return message_text


__all__ = ["apply_world_info_to_message"]
