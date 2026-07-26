"""Controller tests for LLM-assisted character generation.

The controller owns "turn a generation request into finished text": it builds
messages via the pure contract, runs them through an injected runner, and
normalizes the reply. The runner is injectable so these tests exercise the
real control flow without a provider.
"""

import pytest

from tldw_chatbook.Character_Chat.character_generation import (
    CharacterGenerationError,
)
from tldw_chatbook.Character_Chat.character_generation_controller import (
    CharacterGenerationController,
)

pytestmark = pytest.mark.asyncio


def _controller(reply, *, capture=None):
    async def _runner(messages):
        if capture is not None:
            capture.append(messages)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return CharacterGenerationController(runner=_runner)


async def test_generate_field_returns_finished_text():
    controller = _controller("A guarded archivist of drowned books.")

    text = await controller.generate_field(
        "description", {"name": "Seraphina"}, context_mode="whole_character"
    )

    assert text == "A guarded archivist of drowned books."


async def test_generate_field_strips_surrounding_quotes_and_fences():
    """Models wrap field text despite being told not to; the field must be clean."""
    controller = _controller('```\n"A guarded archivist."\n```')

    text = await controller.generate_field(
        "description", {"name": "Seraphina"}, context_mode="whole_character"
    )

    assert text == "A guarded archivist."


async def test_generate_field_strips_a_restated_field_label():
    """A leading 'Description:' echo is chrome, not card text."""
    controller = _controller("Description: A guarded archivist.")

    text = await controller.generate_field(
        "description", {"name": "Seraphina"}, context_mode="whole_character"
    )

    assert text == "A guarded archivist."


async def test_generate_field_rejects_an_empty_reply():
    """An empty generation must raise, not silently blank the author's field."""
    controller = _controller("   ")

    with pytest.raises(CharacterGenerationError):
        await controller.generate_field(
            "description", {"name": "Seraphina"}, context_mode="whole_character"
        )


async def test_generate_field_passes_the_selected_context_mode_through():
    captured: list = []
    controller = _controller("text", capture=captured)

    await controller.generate_field(
        "scenario",
        {"name": "S", "description": "desc-x", "personality": "pers-y"},
        context_mode="field_and_description",
    )

    body = "\n".join(str(m["content"]) for m in captured[0])
    assert "desc-x" in body
    assert "pers-y" not in body


async def test_generate_whole_character_returns_parsed_fields():
    controller = _controller(
        '{"name": "Seraphina", "description": "An archivist.", '
        '"first_message": "You came."}'
    )

    fields = await controller.generate_whole_character("a drowned-library archivist")

    assert fields["name"] == "Seraphina"
    assert fields["first_message"] == "You came."


async def test_generate_whole_character_surfaces_unparseable_replies():
    controller = _controller("Sure! Here is a great character for you.")

    with pytest.raises(CharacterGenerationError):
        await controller.generate_whole_character("a drowned-library archivist")


async def test_runner_failure_is_reported_as_a_generation_error():
    """Provider failures must arrive as one error type the UI can render."""
    controller = _controller(RuntimeError("connection refused"))

    with pytest.raises(CharacterGenerationError) as excinfo:
        await controller.generate_field(
            "description", {"name": "S"}, context_mode="whole_character"
        )

    assert "connection refused" in str(excinfo.value)


# --- The gateway-backed runner: generation uses the Console's active provider ---
# The author configures one provider, in Console. Character generation must not
# ask them to configure a second one, and must not silently fall back to some
# other provider when Console's is not ready.


async def test_gateway_runner_accumulates_streamed_chunks():
    """Generation is non-streaming to the user, so chunks must be joined."""
    from tldw_chatbook.Character_Chat.character_generation_controller import (
        build_gateway_runner,
    )

    class _Resolution:
        ready = True
        visible_copy = ""

    class _Gateway:
        async def resolve_for_send(self, selection):
            return _Resolution()

        async def stream_chat(self, resolution, messages):
            for chunk in ("A guarded ", "archivist."):
                yield chunk

    runner = build_gateway_runner(
        gateway_factory=lambda: _Gateway(),
        selection_factory=lambda: object(),
    )

    assert await runner([{"role": "user", "content": "x"}]) == "A guarded archivist."


async def test_gateway_runner_reports_an_unready_provider():
    """An unready provider must surface its own recovery copy, not a crash."""
    from tldw_chatbook.Character_Chat.character_generation_controller import (
        build_gateway_runner,
    )

    class _Resolution:
        ready = False
        visible_copy = "Add an API key for OpenAI in [api_settings.openai]."

    class _Gateway:
        async def resolve_for_send(self, selection):
            return _Resolution()

        async def stream_chat(self, resolution, messages):  # pragma: no cover
            raise AssertionError("must not send when the provider is not ready")
            yield ""

    runner = build_gateway_runner(
        gateway_factory=lambda: _Gateway(),
        selection_factory=lambda: object(),
    )

    with pytest.raises(CharacterGenerationError) as excinfo:
        await runner([{"role": "user", "content": "x"}])

    assert "Add an API key" in str(excinfo.value)


async def test_generate_field_streams_partial_text_as_it_arrives():
    """Long generations must show progress, not just a disabled button.

    The preview is the surface the author watches; buffering the whole reply
    left it blank for the entire request.
    """
    from tldw_chatbook.Character_Chat.character_generation_controller import (
        CharacterGenerationController,
    )

    async def _runner(messages, on_chunk=None):
        for chunk in ("A guarded ", "archivist ", "of drowned books."):
            if on_chunk is not None:
                on_chunk(chunk)
        return "A guarded archivist of drowned books."

    seen: list[str] = []
    controller = CharacterGenerationController(runner=_runner)

    text = await controller.generate_field(
        "description",
        {"name": "Seraphina"},
        context_mode="whole_character",
        on_chunk=seen.append,
    )

    assert seen == ["A guarded ", "A guarded archivist ", "A guarded archivist of drowned books."]
    assert text == "A guarded archivist of drowned books."


async def test_gateway_runner_reports_chunks_as_they_stream():
    from tldw_chatbook.Character_Chat.character_generation_controller import (
        build_gateway_runner,
    )

    class _Resolution:
        ready = True
        visible_copy = ""

    class _Gateway:
        async def resolve_for_send(self, selection):
            return _Resolution()

        async def stream_chat(self, resolution, messages):
            for chunk in ("one ", "two"):
                yield chunk

    seen: list[str] = []
    runner = build_gateway_runner(
        gateway_factory=lambda: _Gateway(), selection_factory=lambda: object()
    )

    assert await runner([{"role": "user", "content": "x"}], on_chunk=seen.append) == "one two"
    assert seen == ["one ", "two"]
