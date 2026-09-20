"""Legacy raw-line view over a hosted engine stream (TASK-32852).

The pre-ADR-062 provider handlers streamed newline-terminated ``data: ...``
lines with a trailing ``data: [DONE]\\n\\n`` sentinel, and the Console
gateway's relay still consumes exactly that shape. Providers migrated onto
:mod:`tldw_chatbook.LLM_Calls.hosted_chat` keep the consumer contract through
this shim: each engine event is re-serialized as one data line, the stream
ends with exactly one sentinel, and closing is a method forward to the
engine stream -- never a yield-after-exit -- so a consumer Stop closes the
owned response/session exactly once and without ``RuntimeError``.

Migrated defects this shape exists INSTEAD of: the old generators yielded
the sentinel inside ``finally`` (Stop raised ``generator ignored
GeneratorExit`` and postponed the response close), relayed the provider's own
``[DONE]`` and then a synthetic one (every normal completion ended with a
duplicate), and swallowed stream-read failures into synthetic error chunks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatStream, HostedChatTurn

_DONE_SENTINEL = "data: [DONE]\n\n"


class LegacyLineStream(Iterator[str]):
    """Re-serialize engine stream events as legacy raw SSE lines."""

    def __init__(self, stream: HostedChatStream) -> None:
        self._stream = stream
        self._sentinel_sent = False

    def __iter__(self) -> LegacyLineStream:
        return self

    def __next__(self) -> str:
        if self._sentinel_sent:
            raise StopIteration
        try:
            event = next(self._stream)
        except StopIteration:
            self._sentinel_sent = True
            return _DONE_SENTINEL
        return f"data: {json.dumps(event)}\n"

    @property
    def terminal_turn(self) -> HostedChatTurn:
        """Terminal metadata after clean stream exhaustion."""
        return self._stream.terminal_turn

    def close(self) -> None:
        """Close the owned response/session pair exactly once."""
        self._stream.close()
