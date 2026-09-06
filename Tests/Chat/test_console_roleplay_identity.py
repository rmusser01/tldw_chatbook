from dataclasses import dataclass

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariantSet,
)
from tldw_chatbook.Chat.console_roleplay_identity import (
    ChatDisplayNameError,
    ConsolePresentationContext,
    ConsoleTranscriptStyle,
    effective_user_display_name,
    expand_character_template,
    normalize_chat_display_name,
    resolve_console_message_presentation,
)


def test_effective_name_prefers_override_then_global_then_user():
    assert effective_user_display_name("Captain Rowan", "Default") == "Captain Rowan"
    assert effective_user_display_name(None, "Default") == "Default"
    assert effective_user_display_name(None, "   ") == "User"


def test_name_validation_uses_terminal_cells_and_rejects_controls():
    assert normalize_chat_display_name("  海の人  ", blank_means_none=False) == "海の人"
    assert (
        normalize_chat_display_name("👩‍🚀 Rowan", blank_means_none=False)
        == "👩‍🚀 Rowan"
    )
    with pytest.raises(ChatDisplayNameError, match="48 terminal cells"):
        normalize_chat_display_name("界" * 25, blank_means_none=False)
    with pytest.raises(ChatDisplayNameError, match="control"):
        normalize_chat_display_name("Rowan\nAdmin", blank_means_none=False)
    with pytest.raises(ChatDisplayNameError, match="control"):
        normalize_chat_display_name("Rowan\u202eAdmin", blank_means_none=False)


def test_blank_override_clears_while_blank_global_falls_back():
    assert normalize_chat_display_name("  ", blank_means_none=True) is None
    assert normalize_chat_display_name("  ", blank_means_none=False) == "User"


@pytest.mark.parametrize("token", ["{{user}}", "{{random_user}}", "<USER>"])
def test_user_aliases_expand_only_once(token):
    result = expand_character_template(
        f"Hello {token}",
        user_name="Archivist {{character}}",
        character_name="Alraune",
    )
    assert result == "Hello Archivist {{character}}"


def test_character_aliases_share_the_loaded_name():
    source = "{{char}}/{{character}}/{{persona}}/<CHAR> greets {{user}}"
    assert (
        expand_character_template(source, user_name="Rowan", character_name="Alraune")
        == "Alraune/Alraune/Alraune/Alraune greets Rowan"
    )


def test_case_and_unknown_tokens_stay_literal():
    assert (
        expand_character_template(
            "{{User}} {{unknown}}", user_name="Rowan", character_name="Alraune"
        )
        == "{{User}} {{unknown}}"
    )


@dataclass(frozen=True)
class GreetingMetadata:
    """Future closed greeting provenance supplied by Task 2."""

    template_kind: str = ""
    template_source: str = ""


def user_message(content: str) -> ConsoleChatMessage:
    return ConsoleChatMessage(role=ConsoleMessageRole.USER, content=content)


def assistant_message(
    content: str, *, metadata: GreetingMetadata | None = None
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
        metadata=metadata,  # type: ignore[arg-type]
    )


def test_character_rows_use_named_speakers_and_roleplay_classes():
    context = ConsolePresentationContext(
        user_name="Captain [Rowan]",
        assistant_kind="character",
        character_name="Alraune",
        revision=7,
    )
    user = resolve_console_message_presentation(user_message("Hi"), context)
    assistant = resolve_console_message_presentation(
        assistant_message("Hello"), context
    )
    assert (user.speaker_label, user.row_class) == (
        "Captain [Rowan]",
        "console-transcript-message-roleplay-user",
    )
    assert (assistant.speaker_label, assistant.row_class) == (
        "Alraune",
        "console-transcript-message-roleplay-character",
    )
    assert user.revision_token[-1] == 7


@pytest.mark.parametrize("assistant_kind", ["generic", "persona", None])
def test_non_character_assistant_label_stays_assistant(assistant_kind):
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind=assistant_kind, character_name="Ada"
    )
    presentation = resolve_console_message_presentation(
        assistant_message("Hello"), context
    )
    assert presentation.speaker_label == "Assistant"
    assert presentation.row_class == "console-transcript-message-role-assistant"


def test_generic_user_rows_use_custom_name_with_role_accent():
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind="generic", character_name=None
    )
    presentation = resolve_console_message_presentation(user_message("Hi"), context)
    assert presentation.speaker_label == "Rowan"
    assert presentation.row_class == "console-transcript-message-role-user"


def test_character_session_without_name_falls_back_to_neutral_assistant():
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind="character", character_name="  "
    )
    presentation = resolve_console_message_presentation(
        assistant_message("Hello"), context
    )
    assert presentation.speaker_label == "Assistant"
    assert presentation.row_class == "console-transcript-message-role-assistant"


def test_neutral_style_removes_role_classes_without_changing_speaker_identity():
    context = ConsolePresentationContext(
        user_name="Rowan",
        assistant_kind="character",
        character_name="Alraune",
        transcript_style=ConsoleTranscriptStyle.NEUTRAL,
    )

    user = resolve_console_message_presentation(user_message("Hi"), context)
    assistant = resolve_console_message_presentation(
        assistant_message("Hello"), context
    )

    assert (user.speaker_label, user.row_class) == ("Rowan", None)
    assert (assistant.speaker_label, assistant.row_class) == ("Alraune", None)


def test_resolver_expands_only_trusted_seeded_character_greetings():
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind="character", character_name="Alraune"
    )
    greeting = assistant_message(
        "Safe projection",
        metadata=GreetingMetadata(
            template_kind="character_greeting",
            template_source="Hello {{user}} from {{character}}.",
        ),
    )
    ordinary = assistant_message("Hello {{user}}")

    assert resolve_console_message_presentation(greeting, context).content == (
        "Hello Rowan from Alraune."
    )
    assert resolve_console_message_presentation(ordinary, context).content == (
        "Hello {{user}}"
    )


def test_character_speaker_is_one_line_but_greeting_expands_exact_raw_name():
    raw_name = "Lady\n\t[bold]Nyx[/bold]\x00"
    context = ConsolePresentationContext(
        user_name="Rowan",
        assistant_kind="character",
        character_name=raw_name,
    )
    greeting = assistant_message(
        "stored projection",
        metadata=GreetingMetadata(
            template_kind="character_greeting",
            template_source="Hello from {{character}}.",
        ),
    )

    presentation = resolve_console_message_presentation(greeting, context)

    assert presentation.speaker_label == "Lady [bold]Nyx[/bold]?"
    assert "\n" not in presentation.speaker_label
    assert "\t" not in presentation.speaker_label
    assert presentation.content == f"Hello from {raw_name}."


@pytest.mark.parametrize(
    "metadata",
    [
        GreetingMetadata(template_kind="ordinary", template_source="Hello {{user}}"),
        GreetingMetadata(template_kind="character_greeting", template_source="  "),
    ],
)
def test_greeting_expansion_requires_recognized_kind_and_nonblank_source(metadata):
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind="character", character_name="Alraune"
    )
    message = assistant_message("Safe {{user}} projection", metadata=metadata)

    assert resolve_console_message_presentation(message, context).content == (
        "Safe {{user}} projection"
    )


def test_greeting_expansion_requires_named_character_session():
    message = assistant_message(
        "Safe projection",
        metadata=GreetingMetadata(
            template_kind="character_greeting", template_source="Hello {{user}}."
        ),
    )
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind="character", character_name=" "
    )

    assert resolve_console_message_presentation(message, context).content == (
        "Safe projection"
    )


def test_resolver_uses_the_current_variant_and_leaves_system_and_tool_content_literal():
    variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1", contents=["first", "second"], selected_index=1
    )
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind="character", character_name="Alraune"
    )
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="stale", variants=variants
    )
    system = ConsoleChatMessage(role=ConsoleMessageRole.SYSTEM, content="{{user}}")
    tool = ConsoleChatMessage(role=ConsoleMessageRole.TOOL, content="{{character}}")

    assert resolve_console_message_presentation(assistant, context).content == "second"
    assert resolve_console_message_presentation(system, context).content == "{{user}}"
    assert resolve_console_message_presentation(tool, context).content == (
        "{{character}}"
    )
