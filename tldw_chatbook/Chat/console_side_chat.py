"""Ephemeral console side chat: answer questions about selected transcript text.

Persistence-free by construction: only ConsoleProviderGateway.stream_chat is
used (its contract bypasses Console history and persistence). The message
list never leaves the calling modal.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.config import DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE

__all__ = [
    "DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE",
    "SIDE_CHAT_BUFFER_CAP",
    "SIDE_CHAT_SYSTEM_PROMPT",
    "ConsoleSideChatService",
    "SideChatOutcome",
    "cap_reply_buffer",
    "render_prompt",
]

SIDE_CHAT_BUFFER_CAP = 100_000
SIDE_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about text the user "
    "selected in their console session. Be concise and specific."
)


@dataclass(frozen=True)
class SideChatOutcome:
    text: str
    provider: str
    model: str
    status: str  # "complete" | "cancelled" | "provider_error"
    error: str = ""


def render_prompt(template: str, selection: str) -> str:
    """Render the side-chat prompt.

    ``{selection}`` is replaced via ``str.replace`` (other braces stay
    literal); a blank template falls back to the default; a template missing
    the placeholder gets the selection appended on a new line.
    """
    tmpl = template.strip() or DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE
    if "{selection}" in tmpl:
        return tmpl.replace("{selection}", selection)
    return f"{tmpl}\n{selection}"


def cap_reply_buffer(text: str) -> str:
    """Keep the tail of an oversized reply."""
    if len(text) <= SIDE_CHAT_BUFFER_CAP:
        return text
    return "…\n" + text[-SIDE_CHAT_BUFFER_CAP:]


def _build_selection(
    provider_selection: ConsoleProviderSelection | None,
    sidechat_model: str,
) -> ConsoleProviderSelection | None:
    """Apply the side-chat model-resolution precedence rules.

    - "provider/model" (contains "/"): derive both from the string.
    - bare model: keep the session provider, override ``explicit_model``.
    - empty: reuse the session selection as-is.
    - no session selection and no qualified model: nothing to send with.
    """
    model = (sidechat_model or "").strip()
    if model and "/" in model:
        provider_part, _, model_part = model.partition("/")
        return ConsoleProviderSelection(
            provider=provider_part, explicit_model=model_part, streaming=True
        )
    if provider_selection is None:
        return None
    if model:
        return replace(provider_selection, explicit_model=model, streaming=True)
    return replace(provider_selection, streaming=True)


class ConsoleSideChatService:
    """Streams one side-chat completion over the provider gateway.

    The service is headless (no Textual widgets, no store writes): callers
    drive ``run`` from a modal and keep ownership of the message list.
    """

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway

    async def run(
        self,
        *,
        selection_quote: str,
        prompt: str,
        provider_selection: ConsoleProviderSelection | None = None,
        sidechat_model: str = "",
    ) -> AsyncIterator[str | SideChatOutcome]:
        """Stream one side-chat completion.

        Yields ``str`` deltas as produced; the final yield is a
        :class:`SideChatOutcome` (callers detect it via ``isinstance``).
        """
        selection = _build_selection(provider_selection, sidechat_model)
        if selection is None:
            yield SideChatOutcome(
                text="",
                provider="",
                model="",
                status="provider_error",
                error="No provider available for the side chat.",
            )
            return

        messages = [
            {"role": "system", "content": SIDE_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt + "\n\nSelected text:\n" + selection_quote},
        ]

        provider = selection.provider
        model = selection.explicit_model or ""
        parts: list[str] = []
        try:
            resolution = await self.gateway.resolve_for_send(selection)
            provider = str(getattr(resolution, "provider", "") or "")
            model = str(getattr(resolution, "model", "") or "").strip()
            if not getattr(resolution, "ready", True) or not model:
                # Blocked resolutions (missing key, model-less provider) do not
                # raise; stream_chat would silently yield zero chunks. Surface
                # the blocker instead (cf. UI/Console_Modules/prompts.py).
                yield SideChatOutcome(
                    text="",
                    provider=provider,
                    model=model,
                    status="provider_error",
                    error=str(getattr(resolution, "visible_copy", "") or "")
                    or "Choose a ready provider and model, then reopen the side chat.",
                )
                return
            async for item in self.gateway.stream_chat(resolution, messages):
                if not isinstance(item, str):
                    continue  # non-str items (tool calls) are not side-chat text
                parts.append(item)
                yield item
        except ChatProviderError as exc:
            # Keep any already-streamed deltas in the outcome: the modal
            # renders outcome.text as the final reply, so text="" would wipe
            # a partial answer the user was reading mid-stream.
            yield SideChatOutcome(
                text=cap_reply_buffer("".join(parts)),
                provider=provider,
                model=model,
                status="provider_error",
                error=str(exc),
            )
            return
        except asyncio.CancelledError:
            yield SideChatOutcome(
                text=cap_reply_buffer("".join(parts)),
                provider=provider,
                model=model,
                status="cancelled",
            )
            raise

        yield SideChatOutcome(
            text=cap_reply_buffer("".join(parts)),
            provider=provider,
            model=model,
            status="complete",
        )
