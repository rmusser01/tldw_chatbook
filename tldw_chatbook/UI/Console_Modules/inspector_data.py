"""Pure Conversation Inspector turn previews and exchange loading."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import (  # noqa: UP035 - preserve moved annotation bindings.
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture
    from tldw_chatbook.Chat.console_trace_projection import ProjectedTraceCall


def _console_inspector_turn_preview(content: Any) -> str:
    """Best-effort short text preview for one Conversation Inspector
    Costs-tab turn row (task-8 review finding 5).

    ``ConsoleChatMessage.content`` is declared ``str``, but a multimodal
    (structured, OpenAI-style content-block list) message is not
    guaranteed to have been coerced to text by the time it reaches here --
    several other modules in this codebase (``Chat/Chat_Functions.py``,
    ``console_provider_gateway.py``) carry their own ``isinstance(content,
    str)`` guards for exactly this reason. Slicing a list with ``[:60]``
    would silently yield up to 60 LIST ELEMENTS, not characters -- not a
    preview, and not obviously wrong-looking in a diff either. Falls back
    to the first text block's text (bounded to 60 chars, matching the str
    path), or ``""`` when nothing text-shaped is found -- never a
    fabricated summary.
    """
    if isinstance(content, str):
        return content[:60]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    return text[:60]
    return ""


def _build_console_inspector_exchanges_loader(
    messages_by_native_id: Mapping[str, Any],
    projected_calls_reader: Callable[[str], Sequence[ProjectedTraceCall]],
    abandoned_run_tags_for: Callable[[str], AbstractSet[str]] | None = None,
) -> Callable[[str], Awaitable[list[tuple[ExchangeCapture, bool]]]]:
    """Build the Costs-tab ``exchanges_loader`` for
    ``ConsoleConversationInspector`` (task-8, extended task-9).

    A standalone function rather than a method-local closure specifically
    so it is unit-testable without mounting a ``ChatScreen`` (review
    finding 6) -- pure extraction, no behavior change from the closure
    this replaced in ``ChatScreen._build_console_inspector_cost_data``.

    Args:
        messages_by_native_id: ``ConsoleChatMessage.id`` -> the matching
            in-memory message, for the native-first check.
        projected_calls_reader: Store-owned persisted-message reader returning
            discriminated normalized/legacy calls. Called lazily only on the
            durable fallback path, so this Textual helper never receives a DB
            handle and an ephemeral session never performs durable I/O.
        abandoned_run_tags_for: Optional ``native_message_id ->
            {run_tag, ...}`` lookup (task-9; ``ConsoleChatStore.
            abandoned_exchange_run_tags`` in production) used ONLY on the
            native-capture path to resolve each capture's real
            ``abandoned`` flag. Defaults to ``None``, which preserves the
            task-8 behavior of reporting ``abandoned=False`` for every
            native capture -- kept optional (rather than required) so the
            existing unit tests in
            ``Tests/UI/test_chat_screen_console_inspector_loader.py``,
            which construct this loader with just the first two
            positional args, are unaffected.

    Returns:
        An async ``native_message_id -> [(capture, abandoned), ...]``
        callable (see ``console_conversation_inspector``'s module
        docstring for the pair contract and the ordering caveat -- callers
        must NOT trust the returned order, only ``(created_at, seq)``).
        Prefers ``message.exchanges`` (native, in-memory captures resolve
        ``abandoned`` via ``abandoned_run_tags_for`` when supplied, else
        always ``False``) and only falls back to a threaded
        ``get_message_exchanges`` + ``capture_from_blob`` read when there
        is no native capture AND the message has a
        ``persisted_message_id`` (an ephemeral session has neither, so it
        returns ``[]`` without any durable read). Corrupt legacy isolation and
        normalized-first selection belong to the injected projection.
    """

    async def _exchanges_loader(
        native_message_id: str,
    ) -> list[tuple[ExchangeCapture, bool]]:
        message = messages_by_native_id.get(native_message_id)
        if message is not None and message.exchanges:
            # Native captures win when present -- they are fresher than
            # whatever was last flushed to the DB.
            abandoned_tags: AbstractSet[str] = (
                abandoned_run_tags_for(native_message_id)
                if abandoned_run_tags_for is not None
                else frozenset()
            )
            return [
                (capture, capture.run_tag in abandoned_tags)
                for capture in message.exchanges
            ]
        persisted_id = message.persisted_message_id if message is not None else None
        if not persisted_id:
            return []

        def _read() -> list[tuple[ExchangeCapture, bool]]:
            return [
                (
                    replace(
                        projected.capture,
                        trace_provenance=projected.provenance,
                        trace_chronology=projected.chronology,
                        trace_uncertainty=projected.uncertainty_codes,
                    ),
                    projected.abandoned,
                )
                for projected in projected_calls_reader(persisted_id)
            ]

        return await asyncio.to_thread(_read)

    return _exchanges_loader
