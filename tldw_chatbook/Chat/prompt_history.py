"""
Persistent prompt history for the chat input (JSONL-backed).

Stores one JSON object per line — ``{"input": ..., "timestamp": ...}`` — in a
per-user data file. All file IO runs in a worker thread (``asyncio.to_thread``)
so the Textual event loop is never blocked.

Recall uses shell-style indexing: index 0 is the *live draft* pseudo-entry (the
in-progress text stashed while navigating), and negative indexes walk backwards
through stored entries (-1 is the most recent prompt). ``clamp_index`` provides
the validate_*-style clamping used by the input widget to keep navigation in
bounds. ``complete`` powers fish-shell-style ghost text: the most recent entry
matching the current text as a prefix wins.

Growth is bounded by ``max_entries``: load keeps only the most recent entries
in memory, and append rewrites the file with the tail once the cap is exceeded.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TypedDict

from loguru import logger

DEFAULT_MAX_ENTRIES = 1000


def default_prompt_history_path() -> Path:
    """Return the default per-user prompt history file path."""
    from ..config import get_user_data_dir

    return get_user_data_dir() / "prompt_history.jsonl"


class HistoryEntry(TypedDict):
    """A single entry in the history file."""

    input: str
    timestamp: float


class PromptHistory:
    """Manages a JSONL prompt-history file with async IO and draft stashing."""

    def __init__(self, path: Path | str, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        """Initialize the history store.

        Args:
            path: Path of the JSONL history file (created on first write).
            max_entries: Cap on stored entries; the most recent entries win.
        """
        self.path = Path(path)
        self.max_entries = max_entries
        self._entries: list[HistoryEntry] = []
        self._current: str | None = None
        self._loaded: bool = False
        # Serializes append() so a whole-file cap rewrite can never
        # interleave with another append from a concurrent send.
        self._append_lock = asyncio.Lock()
        # TASK-22218: monotonic counter bumped on every ``_entries`` mutation
        # (load, optimistic append, cap trim, write-failure rollback). Lets a
        # consumer key a cache on "has the history changed?" without hashing
        # up to ``max_entries`` entries -- the composer's blink-tick render
        # memo is the consumer that motivated it.
        self._revision: int = 0

    @property
    def size(self) -> int:
        """Number of stored entries (excludes the live draft pseudo-entry)."""
        return len(self._entries)

    @property
    def revision(self) -> int:
        """Counter that advances whenever the stored entries change.

        Cheap invalidation key for caches over ``complete()``/``get_entry``
        results: equal revisions guarantee the stored entries are unchanged.
        The live-draft stash (``stash_draft``/``clear_draft``) does not
        advance it -- the stash never affects ``complete()``.
        """
        return self._revision

    @property
    def current(self) -> str:
        """The stashed live-draft text, or an empty string when not stashed."""
        return self._current or ""

    def stash_draft(self, text: str) -> None:
        """Stash in-progress text so history recall never loses it.

        Args:
            text: The current in-progress draft to preserve as the live
                (index 0) pseudo-entry.
        """
        self._current = text

    def clear_draft(self) -> None:
        """Drop the stashed draft (e.g. after a successful send)."""
        self._current = None

    def clamp_index(self, index: int) -> int:
        """Clamp a history index into the valid range ``[-size, 0]``.

        Args:
            index: Requested index; 0 is the live draft, negatives walk back.

        Returns:
            The clamped index.
        """
        return max(-self.size, min(0, index))

    async def load(self) -> None:
        """Load entries from the history file, off the event loop.

        Only the most recent ``max_entries`` entries are kept in memory.
        Corrupt lines are tolerated; read failures log and leave an empty
        history.
        """
        if self._loaded:
            return

        def read_history() -> list[HistoryEntry]:
            """Read and parse the JSONL file in a worker thread."""
            if not self.path.exists():
                return []
            entries: list[HistoryEntry] = []
            with self.path.open("r", encoding="utf-8") as history_file:
                for line in history_file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Tolerate corrupt lines; keep the rest.
                    text = entry.get("input")
                    if isinstance(text, str) and text:
                        timestamp = entry.get("timestamp")
                        entries.append(
                            {
                                "input": text,
                                "timestamp": timestamp
                                if isinstance(timestamp, (int, float))
                                else 0.0,
                            }
                        )
            return entries

        try:
            entries = await asyncio.to_thread(read_history)
        except Exception as error:
            logger.warning(f"Could not read prompt history {self.path}: {error}")
            entries = []
        self._entries = entries[-self.max_entries :]
        self._revision += 1
        self._loaded = True

    async def append(self, text: str) -> bool:
        """Append a prompt to the history (serialized, fire-and-forget friendly).

        Concurrent appends are serialized with an internal lock so a
        whole-file cap rewrite can never interleave with another append.

        Args:
            text: The prompt text to record.

        Returns:
            True when the entry was recorded, False on write failure or when
            the entry was skipped (empty text or consecutive duplicate).
        """
        async with self._append_lock:
            return await self._append_impl(text)

    async def _append_impl(self, text: str) -> bool:
        """Append a prompt to the history (fire-and-forget friendly).

        Consecutive duplicates are skipped. The in-memory entry is added
        optimistically before the awaited write so two rapid identical sends
        cannot both pass the dedupe check; it is rolled back on write failure.
        When the cap is exceeded the file is rewritten with the tail. The file
        IO runs in a worker thread; failures are logged and reported, never
        raised.

        Args:
            text: The prompt text to record.

        Returns:
            True when the entry was recorded, False on write failure or when
            the entry was skipped (empty text or consecutive duplicate).
        """
        if not text:
            return False
        if not self._loaded:
            await self.load()
        if self._entries and self._entries[-1]["input"] == text:
            return False  # Consecutive duplicate — already recorded.

        entry: HistoryEntry = {"input": text, "timestamp": time.time()}
        # Optimistic in-memory update (rolled back on write failure) so a
        # second identical send racing this one hits the dedupe check above.
        self._entries.append(entry)
        self._revision += 1
        excess = len(self._entries) - self.max_entries
        dropped: list[HistoryEntry] = []
        if excess > 0:
            dropped = self._entries[:excess]
            del self._entries[:excess]
            self._revision += 1

        def write_history() -> None:
            """Write to the JSONL file in a worker thread."""
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if dropped:
                # Cap exceeded — rewrite the file with the retained tail.
                with self.path.open("w", encoding="utf-8") as history_file:
                    for retained in self._entries:
                        history_file.write(
                            f"{json.dumps(retained, ensure_ascii=False)}\n"
                        )
            else:
                with self.path.open("a", encoding="utf-8") as history_file:
                    history_file.write(f"{json.dumps(entry, ensure_ascii=False)}\n")

        try:
            await asyncio.to_thread(write_history)
        except Exception as error:
            logger.warning(f"Could not write prompt history {self.path}: {error}")
            if dropped:
                self._entries = dropped + self._entries
            try:
                self._entries.remove(entry)
            except ValueError:
                pass
            self._revision += 1
            return False
        self._current = None
        return True

    async def get_entry(self, index: int) -> HistoryEntry:
        """Get a history entry by shell-style index.

        Args:
            index: 0 for the live draft pseudo-entry, negative indexes for
                stored entries (-1 is the most recent).

        Returns:
            The history entry. Stored entries carry their persisted timestamp;
            the live draft pseudo-entry reports the current time.

        Raises:
            IndexError: When the index is out of range.
        """
        if index > 0:
            raise IndexError("History indices must be 0 or negative.")
        if not self._loaded:
            await self.load()
        if index == 0:
            return {"input": self.current, "timestamp": time.time()}
        try:
            return self._entries[index]
        except IndexError:
            raise IndexError(f"No history entry at index {index}") from None

    def complete(self, prefix: str) -> str | None:
        """Return the most recent entry starting with ``prefix``.

        Used for ghost-text suggestions; exact matches are excluded so a fully
        typed prompt never suggests itself.

        Args:
            prefix: The current input text.

        Returns:
            The matching entry, or None when no entry matches.
        """
        if not prefix:
            return None
        for entry in reversed(self._entries):
            text = entry["input"]
            if text.startswith(prefix) and text != prefix:
                return text
        return None
