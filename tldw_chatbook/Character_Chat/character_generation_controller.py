"""Runs character-generation requests against the Console's active provider.

Sits between the pure request contract (``character_generation``) and the
editor UI: builds messages, runs them, and normalizes the reply into something
safe to drop into a character field.

The provider call is injectable (``runner``). The default runner uses the same
``ConsoleProviderGateway`` path the rest of the app sends through, so
generation always uses whatever provider and model the Console is currently
configured with — the author never has to configure a second one.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Mapping

from loguru import logger

from .character_generation import (
    GENERATABLE_FIELDS,
    CharacterFieldContextMode,
    CharacterGenerationError,
    build_field_generation_messages,
    build_whole_character_messages,
    parse_whole_character_response,
)

#: Called with each incremental chunk of accumulated text as it arrives, so the
#: UI can show progress during a long generation instead of a blank preview.
ChunkCallback = Callable[[str], None]
MessageRunner = Callable[..., Awaitable[str]]

_FENCE = re.compile(r"^```(?:[a-zA-Z]*)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_wrapping(text: str, field_label: str) -> str:
    """Remove the chrome models add despite being told not to.

    Strips a surrounding markdown fence, surrounding quotes, and a restated
    ``Field:`` prefix. These are formatting artifacts, not card text, and the
    author would otherwise have to delete them by hand on every generation.

    Args:
        text: Raw model reply.
        field_label: Human field label (e.g. ``"first message"``) whose
            restated prefix should be dropped.

    Returns:
        Cleaned field text.
    """
    cleaned = text.strip()
    fenced = _FENCE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    prefix = re.compile(rf"^{re.escape(field_label)}\s*:\s*", re.IGNORECASE)
    cleaned = prefix.sub("", cleaned, count=1).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def build_gateway_runner(
    *,
    gateway_factory: Callable[[], Any],
    selection_factory: Callable[[], Any],
) -> MessageRunner:
    """Build a runner that sends through the Console provider gateway.

    Generation deliberately reuses Console's own provider path so the author
    configures exactly one provider. Chunks are accumulated into the final
    reply and, when ``on_chunk`` is supplied, relayed as they arrive so the UI
    can fill the preview live instead of sitting blank for the whole request.

    Args:
        gateway_factory: Returns the ``ConsoleProviderGateway`` to send with.
        selection_factory: Returns the ``ConsoleProviderSelection`` describing
            Console's current provider/model/endpoint.

    Returns:
        An async runner suitable for ``CharacterGenerationController``.
    """

    async def _run(
        messages: list[dict[str, str]], on_chunk: ChunkCallback | None = None
    ) -> str:
        gateway = gateway_factory()
        resolution = await gateway.resolve_for_send(selection_factory())
        if not getattr(resolution, "ready", False):
            # Reuse Console's own recovery copy rather than inventing a second
            # vocabulary for "your provider is not set up".
            raise CharacterGenerationError(
                str(getattr(resolution, "visible_copy", "") or "")
                or "the configured provider is not ready to send"
            )
        chunks: list[str] = []
        async for chunk in gateway.stream_chat(resolution, messages):
            if isinstance(chunk, str):
                chunks.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
        return "".join(chunks)

    return _run


class CharacterGenerationController:
    """Generates character-card text using the Console's active provider."""

    def __init__(self, *, runner: MessageRunner | None = None) -> None:
        """Initialize the controller.

        Args:
            runner: Async callable taking provider messages and returning the
                reply text. Injected in tests; production callers pass the
                gateway-backed runner built by ``build_gateway_runner``.
        """
        self._runner = runner

    async def _run(
        self,
        messages: list[dict[str, str]],
        on_chunk: ChunkCallback | None = None,
    ) -> str:
        if self._runner is None:
            raise CharacterGenerationError(
                "character generation has no provider runner configured"
            )
        try:
            if on_chunk is None:
                return await self._runner(messages)
            return await self._runner(messages, on_chunk=on_chunk)
        except CharacterGenerationError:
            raise
        except Exception as exc:  # provider/transport failures
            logger.opt(exception=True).debug("Character generation call failed.")
            # One error type reaches the UI, with the provider's own words kept:
            # "connection refused" is what tells the author what to fix.
            raise CharacterGenerationError(str(exc) or type(exc).__name__) from exc

    async def generate_field(
        self,
        field: str,
        record: Mapping[str, Any],
        *,
        context_mode: CharacterFieldContextMode = "whole_character",
        instruction: str | None = None,
        on_chunk: ChunkCallback | None = None,
    ) -> str:
        """Generate finished text for one character field.

        Args:
            field: Record key to generate.
            record: The character record being edited.
            context_mode: How much of the character to show the model.
            instruction: Optional extra steer from the author.
            on_chunk: Optional callback receiving the accumulated text so far
                on every provider chunk, so the preview can fill in live.

        Returns:
            Cleaned field text, ready to preview.

        Raises:
            CharacterGenerationError: On an unknown field, a provider failure,
                or an empty reply.
        """
        messages = build_field_generation_messages(
            field, record, context_mode=context_mode, instruction=instruction
        )
        if on_chunk is None:
            reply = await self._run(messages)
        else:
            # Report the accumulated text, not the raw fragment: the preview
            # renders a whole field, and partial text is only useful as the
            # growing field it is becoming.
            accumulated: list[str] = []

            def _relay(chunk: str) -> None:
                accumulated.append(chunk)
                on_chunk("".join(accumulated))

            reply = await self._run(messages, _relay)
        text = _strip_wrapping(reply or "", GENERATABLE_FIELDS[field])
        if not text:
            # Never hand back "" -- accepting it would silently blank a field
            # the author may have spent real time writing.
            raise CharacterGenerationError(
                "the model returned an empty result for this field"
            )
        return text

    async def generate_whole_character(self, concept: str) -> dict[str, str]:
        """Draft a whole character from a one-line concept.

        Args:
            concept: The author's idea for the character.

        Returns:
            Mapping of card field -> text for the fields the model supplied.

        Raises:
            CharacterGenerationError: On a blank concept, a provider failure,
                or a reply that is not a usable character object.
        """
        messages = build_whole_character_messages(concept)
        reply = await self._run(messages)
        return parse_whole_character_response(reply or "")
