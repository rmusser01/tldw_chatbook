"""Detached prompt recall values; all persistence remains in the backend."""

from __future__ import annotations

import json

from .console_execution_protocol import ConsoleProtocolError
from .prompt_history import PromptHistory


def decode_prompt_history(payload: dict) -> tuple:
    """Validate the original stored texts/timestamps without a file owner."""
    if (
        type(payload) is not dict or set(payload) != {"revision", "entries", "loaded", "error"}
        or type(payload["revision"]) is not int or payload["revision"] < 0
        or type(payload["entries"]) is not list
        or type(payload["loaded"]) is not bool
        or payload["error"] is not None and type(payload["error"]) is not str
    ):
        raise ConsoleProtocolError("invalid_prompt_history_projection")
    entries = []
    for entry in payload["entries"]:
        if (
            type(entry) is not dict or set(entry) != {"input", "timestamp"}
            or type(entry["input"]) is not str
            or type(entry["timestamp"]) is not str
        ):
            raise ConsoleProtocolError("invalid_prompt_history_entry")
        try:
            stamp = json.loads(entry["timestamp"])
        except (ValueError, TypeError) as error:
            raise ConsoleProtocolError("invalid_prompt_history_timestamp") from error
        # The original JSONL loader accepts bool and nonfinite float timestamps.
        # A designated scalar JSON string preserves that domain without relaxing
        # the outer transport's finite JSON grammar.
        if type(stamp) not in (bool, int, float):
            raise ConsoleProtocolError("invalid_prompt_history_timestamp")
        entries.append((entry["input"], stamp))
    return payload["revision"], tuple(entries), payload["loaded"], payload["error"]


class ConsolePromptHistoryView:
    """Original recall arithmetic over published entries and a UI draft stash.

    This is not a PromptHistory persistence owner. It has no path, file job,
    append method or storage callbacks. The composer retains its original
    completion and recall behavior through the original pure implementations.
    """

    size = PromptHistory.size
    revision = PromptHistory.revision
    current = PromptHistory.current
    entry_texts = PromptHistory.entry_texts
    stash_draft = PromptHistory.stash_draft
    clear_draft = PromptHistory.clear_draft
    clamp_index = PromptHistory.clamp_index
    complete = PromptHistory.complete
    get_entry = PromptHistory.get_entry

    def __init__(self, client) -> None:
        self._client = client
        self._entries = []
        self._revision = 0
        self._current = None
        self._draft_revision = 0
        self._loaded = False
        self.persistence_error = None

    def publish(self, captured: tuple) -> None:
        """Adopt one canonically decoded backend publication."""
        revision, entries, loaded, error = captured
        self._entries = [{"input": text, "timestamp": stamp} for text, stamp in entries]
        self._revision = revision
        self._loaded, self.persistence_error = loaded, error

    async def load(self) -> None:
        """The named projection already joined the backend's original load."""
        if not self._loaded:
            await self._client.refresh_prompt_history(self)
