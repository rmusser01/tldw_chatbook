"""Shared world-info send-path resolver (Roleplay P2g).

Builds the world-info-injected message text for a send, composing the same
sources the legacy chat_events path does (conversation-attached books ∪
character-attached snapshots ∪ a native character_book) — so the native Console
(P2g-1) and, later, the legacy path (P2g-3) share one faithful implementation.
Never raises: any problem returns the message text unchanged.

Cold-start caching (ADR-221): the built ``WorldInfoProcessor`` is cached per
``(conversation_id, character_id, card_version)`` and validated against the store's
monotonic ``WorldBookManager.generation`` — an unchanged book set means the
second and later sends perform no book queries, no entry JSON parsing, no
entry processing and no pattern compilation.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

# Bounded LRU: at most this many conversations keep a cached processor
# (ADR-221). Memory is bounded by 8 × the active book set.
_PROCESSOR_CACHE_MAX_CONVERSATIONS = 8


class _CacheEntry:
    """One cached cold start (ADR-221). Never holds the db object itself —
    only a weakref, so a re-opened database never inherits another
    connection's processor."""

    __slots__ = ("generation", "db_ref", "processor", "books")

    def __init__(
        self,
        generation: int,
        db_ref: weakref.ref,
        processor: Any,
        books: List[Dict[str, Any]],
    ) -> None:
        self.generation = generation
        self.db_ref = db_ref
        self.processor = processor
        self.books = books


# Module-level because the resolver is a module of free functions called from
# ``asyncio.to_thread`` workers (Console) and Textual workers — lock-guarded,
# with the lock held only for dict access, never during the fetch or build.
_processor_cache: "OrderedDict[Tuple[Any, ...], _CacheEntry]" = OrderedDict()
_processor_cache_lock = threading.Lock()


def _clear_world_info_cache() -> None:
    """Drop every cached cold start (test seam; the cache is otherwise
    self-invalidating via the store generation)."""
    with _processor_cache_lock:
        _processor_cache.clear()


def _cache_key(conversation_id: Any, char_data: Any) -> Optional[Tuple[Any, ...]]:
    character_id = char_data.get("id") if isinstance(char_data, dict) else None
    version = char_data.get("version") if isinstance(char_data, dict) else None
    if not isinstance(version, int):
        version = None
    if isinstance(char_data, dict) and char_data.get("extensions") and version is None:
        # Imported/ad-hoc cards without a revision have no safe reuse token.
        return None
    return (str(conversation_id), character_id, version)


def _cache_get(db: Any, key: Tuple[Any, ...], generation: int) -> Optional[Any]:
    try:
        with _processor_cache_lock:
            entry = _processor_cache.get(key)
            if entry is None:
                return None
            if entry.generation != generation or entry.db_ref() is not db:
                # Store mutated (or the db object changed): stale by contract.
                del _processor_cache[key]
                return None
            _processor_cache.move_to_end(key)  # LRU touch
            return entry.processor
    except Exception:
        return None


def _cache_put(
    db: Any,
    key: Tuple[Any, ...],
    generation: int,
    processor: Any,
    books: List[Dict[str, Any]],
) -> None:
    try:
        entry = _CacheEntry(generation, weakref.ref(db), processor, books)
    except TypeError:  # db does not support weak references
        return
    with _processor_cache_lock:
        _processor_cache[key] = entry
        _processor_cache.move_to_end(key)
        while len(_processor_cache) > _PROCESSOR_CACHE_MAX_CONVERSATIONS:
            _processor_cache.popitem(last=False)


def _store_generation(db: Any) -> Optional[int]:
    """Current world-book store generation, or None when unreadable (no cache)."""
    try:
        from .world_book_manager import WorldBookManager

        return WorldBookManager(db).cache_generation
    except Exception:
        return None


def _collect_active_world_books(
    db: Any,
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
    books: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect the world books that apply to this send.

    Args:
        db: A ``CharactersRAGDB`` (or None).
        conversation_id: The active conversation (string UUID) or None.
        char_data: The active character record, or None (native Console).
        books: Pre-collected conversation books (ADR-221 console seam). When
            given, the manager fetch is skipped entirely; ``None`` means
            collect from the store as before.

    Returns:
        ``(world_books, has_character_book)`` — the conversation-attached books
        unioned with character-attached snapshots (conversation wins on a name
        collision), and whether ``char_data`` carries a native ``character_book``.
        Never raises.
    """
    world_books: List[Dict[str, Any]] = []
    if books is not None:
        world_books = [b for b in books if isinstance(b, dict)]
    elif conversation_id and db is not None:
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


def resolve_world_info_injection(
    db: Any,
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
    message_text: str,
    history: List[Dict[str, Any]],
    books: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[str, int]:
    """Return ``(injected_message_text, matched_entry_count)``.

    Same collect→build→process→format→join as ``apply_world_info_to_message``,
    but also reports how many world-info entries matched (for the legacy
    ``[World Info: N entries]`` indicator).

    Cold starts are cached per ``(conversation_id, character_id, card_version)`` and
    invalidated by the store generation (ADR-221): with an unchanged book set,
    repeated sends in one conversation reuse one built ``WorldInfoProcessor``.
    The cache is only used on the self-fetch path; passing ``books`` (the
    console seam for pre-collected books) bypasses it.

    Args:
        db: A ``CharactersRAGDB`` (or None).
        conversation_id: The active conversation (string UUID) or None.
        char_data: The active character record, or None (conversation-only).
        message_text: The current user message text (already plain string).
        history: Prior messages as ``{"role","content": str}`` (string content;
            the caller normalizes multimodal content to text before calling).
        books: Pre-collected conversation world books (ADR-221 console seam).
            When given, the manager fetch is skipped; ``None`` collects (and
            caches) from the store.

    Returns:
        ``(text, count)`` — the message text wrapped with world-info injections
        in the order ``at_start → before_char → message → after_char → at_end``
        (``"\\n\\n"`` separated) plus the number of matched entries, or
        ``(message_text, 0)`` when nothing matches / no books / no conversation /
        any error. Never raises.
    """
    if not isinstance(message_text, str):
        return message_text, 0
    try:
        from .world_info_processor import WorldInfoProcessor

        processor = None
        if books is None and conversation_id and db is not None:
            key = _cache_key(conversation_id, char_data)
            generation = _store_generation(db)
            if generation is not None and key is not None:
                processor = _cache_get(db, key, generation)
                if processor is None:
                    world_books, has_character_book = _collect_active_world_books(
                        db, conversation_id, char_data
                    )
                    if not (has_character_book or world_books):
                        # No books and no character book: nothing to cache —
                        # the next send repeats only the (cheap) empty fetch.
                        return message_text, 0
                    processor = WorldInfoProcessor(
                        character_data=char_data if has_character_book else None,
                        world_books=world_books or None,
                    )
                    _cache_put(db, key, generation, processor, world_books)

        if processor is None:
            world_books, has_character_book = _collect_active_world_books(
                db, conversation_id, char_data, books=books
            )
            if not (has_character_book or world_books):
                return message_text, 0
            processor = WorldInfoProcessor(
                character_data=char_data if has_character_book else None,
                world_books=world_books or None,
            )

        result = processor.process_messages(message_text, history or [])
        matched = result.get("matched_entries") or []
        if not matched:
            return message_text, 0
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
        return "\n\n".join(parts), len(matched)
    except Exception:
        logger.opt(exception=True).debug(
            "world-info: apply failed; returning message text unchanged"
        )
        return message_text, 0


def apply_world_info_to_message(
    db: Any,
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
    message_text: str,
    history: List[Dict[str, Any]],
    books: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Return ``message_text`` with matched world-info injected, or unchanged.

    Args:
        db: A ``CharactersRAGDB`` (or None).
        conversation_id: The active conversation (string UUID) or None.
        char_data: The active character record, or None (conversation-only).
        message_text: The current user message text (already plain string).
        history: Prior messages as ``{"role","content": str}`` (string content;
            the caller normalizes multimodal content to text before calling).
        books: Pre-collected conversation world books (ADR-221 console seam);
            skips the manager fetch when given.

    Returns:
        The message text wrapped with world-info injections in the order
        ``at_start → before_char → message → after_char → at_end``
        (``"\\n\\n"`` separated), or the original ``message_text`` when
        nothing matches / no books / no conversation / any error. Never raises.
    """
    return resolve_world_info_injection(
        db, conversation_id, char_data, message_text, history, books=books
    )[0]


def summarize_active_world_books(
    db: Any,
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """World-book "what's in play" summary for the Console inspector (never raises).

    Shows ALL attached books (enabled + disabled) — the *attachment picture* —
    via ``get_world_books_for_conversation(enabled_only=False)``, NOT the
    send-path ``_collect_active_world_books`` (which is enabled-only). ``char_data``
    is accepted for signature parity but P2g-2 is conversation-only (native
    ``char_data=None``).

    Args:
        db: A ``CharactersRAGDB`` (or None).
        conversation_id: The active conversation (string UUID) or None.
        char_data: Unused on native Console (accepted for parity).

    Returns:
        ``{"world_books": [{"name": str, "enabled": bool, "entry_count": int}],
        "source": "local"}``. ``{"world_books": [], "source": "local"}`` on no
        conversation / no books / any error.
    """
    if not conversation_id or db is None:
        return {"world_books": [], "source": "local"}
    try:
        from .world_book_manager import WorldBookManager

        books = WorldBookManager(db).get_world_books_for_conversation(
            str(conversation_id), enabled_only=False
        )
    except Exception:
        logger.opt(exception=True).debug(
            "world-info: could not summarize conversation world books"
        )
        return {"world_books": [], "source": "local"}
    world_books = []
    for book in books:
        if not isinstance(book, dict):
            continue
        entries = book.get("entries")
        world_books.append(
            {
                "name": str(book.get("name") or "Unnamed"),
                "enabled": bool(book.get("enabled", True)),
                "entry_count": len(entries) if isinstance(entries, list) else 0,
            }
        )
    return {"world_books": world_books, "source": "local"}


__all__ = [
    "apply_world_info_to_message",
    "resolve_world_info_injection",
    "summarize_active_world_books",
]
