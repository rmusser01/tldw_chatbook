"""
Persistent prompt history for the chat input (JSONL-backed).

Stores one JSON object per line — ``{"input": ..., "timestamp": ...}`` — in a
per-user data file. All file IO runs in a source-owned executor callback
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

from ..Backup_Recovery import raw_participants as raw
from ..Backup_Recovery.async_file_participants import _FileJob
from ..Backup_Recovery.profile_paths import lexical_path

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

    def __init__(
        self, path: Path | str, max_entries: int = DEFAULT_MAX_ENTRIES
    ) -> None:
        """Initialize the history store.

        Args:
            path: Path of the JSONL history file (created on first write).
            max_entries: Cap on stored entries; the most recent entries win.
        """
        self.path = lexical_path(path)
        self.max_entries = max_entries
        self._entries: list[HistoryEntry] = []
        self._current: str | None = None
        self._draft_revision = 0
        self._loaded: bool = False
        # Serializes append() so a whole-file cap rewrite can never
        # interleave with another append from a concurrent send.
        self._append_lock = asyncio.Lock()
        self.persistence_error: str | None = None
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
        self._draft_revision += 1

    def clear_draft(self) -> None:
        """Drop the stashed draft (e.g. after a successful send)."""
        self._current = None
        self._draft_revision += 1

    def clamp_index(self, index: int) -> int:
        """Clamp a history index into the valid range ``[-size, 0]``.

        Args:
            index: Requested index; 0 is the live draft, negatives walk back.

        Returns:
            The clamped index.
        """
        return max(-self.size, min(0, index))

    def _history_io(self, selected, payload):
        """Own the complete real thread scope; payload is a fixed write snapshot."""
        with raw._scope(
            self, "prompt_history", writing=payload is not None, selected_read=selected
        ) as operation:
            if payload is None:
                entries = []
                try:
                    with raw._file(operation, selected, "r") as history_file:
                        for line in history_file:
                            try:
                                entry = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(entry, dict):
                                continue
                            text, timestamp = entry.get("input"), entry.get("timestamp")
                            if isinstance(text, str) and text:
                                entries.append(
                                    {
                                        "input": text,
                                        "timestamp": timestamp
                                        if isinstance(timestamp, (int, float))
                                        else 0.0,
                                    }
                                )
                except FileNotFoundError:
                    pass
                return entries
            entries, rewrite = payload
            raw._mkdirs(operation)
            destination = (
                selected.with_suffix(selected.suffix + ".tmp") if rewrite else selected
            )
            append_started = False
            try:
                with raw._file(
                    operation, destination, "w" if rewrite else "a"
                ) as history_file:
                    for text, timestamp in entries:
                        line = f"{json.dumps({'input': text, 'timestamp': timestamp}, ensure_ascii=False)}\n"
                        append_started = not rewrite
                        history_file.write(line)
                if rewrite:
                    raw._replace(operation, destination, selected)
            except BaseException:
                # An interrupted append may already have changed durable bytes.
                # Keep the source/native evidence for recovery rather than claim drain.
                if append_started:
                    raw._states[operation].uncertain = True
                raise
            finally:
                if rewrite:
                    raw._remove_temporary(operation, destination)

    async def _load_locked(self) -> None:
        if self._loaded:
            return
        with _FileJob(self, "prompt_history") as job:
            outcome = await job.run(None)
            if outcome.error is None:
                self._entries = outcome.value[-self.max_entries :]
                self._revision += 1
                self._loaded = True
                self.persistence_error = None
            else:
                self.persistence_error = type(outcome.error).__name__
                logger.warning(
                    "Could not read prompt history: {}", self.persistence_error
                )
            if outcome.cancelled:
                raise asyncio.CancelledError
            if outcome.error is not None:
                raise outcome.error

    async def load(self) -> None:
        """Load once under the same lock as append; failed/refused loads can retry."""
        async with self._append_lock:
            try:
                await self._load_locked()
            except Exception as error:
                self.persistence_error = type(error).__name__
                logger.warning(
                    "Could not read prompt history: {}", self.persistence_error
                )

    async def append(self, text: str) -> bool:
        """Record a prompt, preserving ordered IO and cache delivery on cancellation.

        Args:
            text: Prompt text to record.

        Returns:
            True on persistence, False for skipped entries or persistence refusal/failure.
        """
        async with self._append_lock:
            return await self._append_impl(text)

    async def _append_impl(self, text: str) -> bool:
        if not text:
            return False
        try:
            await self._load_locked()
            with _FileJob(self, "prompt_history") as job:
                if self._entries and self._entries[-1]["input"] == text:
                    return False
                draft_revision = self._draft_revision
                entry = {"input": text, "timestamp": time.time()}
                candidate = self._entries + [entry]
                rewrite = len(candidate) > self.max_entries
                candidate = candidate[-self.max_entries :]
                snapshot = tuple(
                    (e["input"], e["timestamp"])
                    for e in (candidate if rewrite else [entry])
                )
                outcome = await job.run((snapshot, rewrite))
                if outcome.error is None:
                    self._entries = candidate
                    self._revision += 1
                    if self._draft_revision == draft_revision:
                        self._current = None
                    self.persistence_error = None
                else:
                    self.persistence_error = type(outcome.error).__name__
                    logger.warning(
                        "Could not write prompt history: {}", self.persistence_error
                    )
                if outcome.cancelled:
                    raise asyncio.CancelledError
                return outcome.error is None
        except Exception as error:
            self.persistence_error = type(error).__name__
            logger.warning("Could not write prompt history: {}", self.persistence_error)
            return False

    async def persistence_safe_point(self) -> bool:
        """Wait for actual source bookkeeping; never save/discard the live draft."""
        async with self._append_lock:
            return self.persistence_error is None

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
