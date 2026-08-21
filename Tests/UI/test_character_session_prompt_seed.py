"""Character Start Chat prompt seeding tests (task-1530, task-1744).

The Personas Start Chat handoff builds the Console session's system prompt
and seeded greeting from the character card. Card text is written against
SillyTavern-style macros, so ``{{char}}``/``{{user}}`` (and aliases) must be
resolved before any of it reaches session settings or the provider.

task-1744: ``_character_session_prompt_seed`` now assembles its system
prompt through the same shared joiner
(``Character_Chat_Lib.compose_character_card_text``) as the character-probe
eval engine's ``compose_system_prompt`` AND the Personas preview pane's
``build_preview_system_prompt``, so a card composes to the same system
prompt through any of the three callers -- see
``test_console_engine_and_preview_compose_byte_identical_system_prompts``
below, which is the whole point of the shared function.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.content import Content

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Console_Modules.session import _character_session_prompt_seed
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterChoice,
)

from Tests.UI.test_destination_shells import _build_test_app


def test_macros_resolved_in_system_prompt_and_greeting():
    """Card fields with {{char}}/{{user}} macros resolve in prompt and greeting."""
    card = {
        "name": "Elara",
        "system_prompt": "Play {{char}} faithfully.",
        "personality": "",
        "description": "{{char}} guides {{user}} through the city.",
        "scenario": "{{user}} meets {{char}} at dusk.",
        "first_message": "Hello {{user}}, I am {{char}}.",
    }

    seed = _character_session_prompt_seed(card)

    assert seed.name == "Elara"
    assert "{{user}}" in seed.system_template
    assert "{{char}}" in seed.system_template
    assert "{{" not in seed.system_prompt
    assert "{{" not in seed.greeting
    assert "Elara guides User through the city." in seed.system_prompt
    assert "User meets Elara at dusk." in seed.system_prompt
    assert seed.greeting_template == "Hello {{user}}, I am {{char}}."
    assert seed.greeting == "Hello User, I am Elara."


def test_character_alias_macros_resolve_to_character_name():
    """{{character}}/{{persona}} aliases resolve to the character's name.

    task-1744: ``description`` is labelled ("Description: ...") by the
    shared joiner, matching the character-probe engine's own labelling --
    this pins the label, not just the macro resolution the test name
    describes.
    """
    card = {
        "name": "Elara",
        "description": "{{character}} and {{persona}} both mean the AI.",
        "first_message": "",
    }

    seed = _character_session_prompt_seed(card)

    assert seed.system_template == (
        "Description: {{character}} and {{persona}} both mean the AI."
    )
    assert seed.system_prompt == "Description: Elara and Elara both mean the AI."
    assert seed.greeting_template == ""
    assert seed.greeting == ""


def test_empty_card_falls_back_to_defaults():
    """An empty card yields the name hint, default prompt, and no greeting."""
    seed = _character_session_prompt_seed(
        {}, name_hint="Hinted"
    )

    assert seed.name == "Hinted"
    assert seed.system_template == "Stay in character."
    assert seed.system_prompt == "Stay in character."
    assert seed.greeting_template == ""
    assert seed.greeting == ""


def test_whitespace_only_card_falls_back_to_stay_in_character():
    """PR review (task-1744): a card whose fields are all whitespace (not
    missing, not "") must compose to an empty string internally so
    Console's "Stay in character." fallback actually fires. Before the fix,
    a whitespace-only labelled field left a dangling "Personality:" (or
    similar) that was non-empty, so the composed text was never "" and the
    fallback never triggered -- the card silently got a bare label as its
    entire system prompt instead."""
    card = {
        "name": "Vex",
        "system_prompt": "   ",
        "personality": "\t",
        "description": "\n",
        "scenario": " ",
        "message_example": "  ",
        "post_history_instructions": " \t ",
        "first_message": "",
    }

    seed = _character_session_prompt_seed(card)

    assert seed.system_template == "Stay in character."
    assert seed.system_prompt == "Stay in character."


def test_whitespace_only_card_agrees_with_the_preview_builder():
    """The same whitespace-only card must compose the same way through the
    Personas preview pane's builder too -- all three surfaces (Console, the
    engine, the preview) share one composer, so a fix to the composer's
    whitespace handling must show up identically everywhere."""
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        build_preview_system_prompt,
    )

    card = {
        "name": "Vex",
        "system_prompt": "   ",
        "personality": "\t",
        "description": "\n",
        "scenario": " ",
        "message_example": "  ",
        "post_history_instructions": " \t ",
    }

    seed = _character_session_prompt_seed(card)
    preview_system_prompt = build_preview_system_prompt(card, greeting="")

    assert seed.system_prompt == "Stay in character."
    assert preview_system_prompt == "Stay in character."
    assert seed.system_prompt == preview_system_prompt


def test_message_example_and_post_history_instructions_reach_the_prompt():
    """task-1744: Console used to omit both fields entirely -- a character's
    example dialogue and post-history instructions shape its voice as much
    as personality or scenario, and dropping them meant Console did not
    behave as the card's author wrote it."""
    card = {
        "name": "Vex",
        "system_prompt": "You are Vex.",
        "message_example": "<START>\nVex: Try me.",
        "post_history_instructions": "Never break character.",
        "first_message": "",
    }

    seed = _character_session_prompt_seed(card)

    assert "Example dialogue:\n<START>\nVex: Try me." in seed.system_prompt
    assert "Never break character." in seed.system_prompt


def test_seed_uses_effective_name_once_without_recursing_inserted_macros():
    """Sequential replacement would rewrite the macro inside the user name."""
    card = {
        "name": "Alraune",
        "system_prompt": "Guard {{user}} beside {{character}}.",
        "first_message": "Hello, {{user}}. I am {{char}}.",
    }

    seed = _character_session_prompt_seed(
        card,
        user_name="Captain {{character}}",
    )

    assert seed.system_template == "Guard {{user}} beside {{character}}."
    assert seed.system_prompt == "Guard Captain {{character}} beside Alraune."
    assert seed.greeting_template == "Hello, {{user}}. I am {{char}}."
    assert seed.greeting == "Hello, Captain {{character}}. I am Alraune."


def _roleplay_card(*, card_id: int = 7, name: str = "Alraune") -> dict:
    return {
        "id": card_id,
        "name": name,
        "system_prompt": "Protect {{user}} as {{character}}.",
        "first_message": "Hello, {{user}}.",
    }


def _character_screen(monkeypatch, card: dict) -> ChatScreen:
    app = _build_test_app()
    app.app_config.setdefault("chat_defaults", {})["user_display_name"] = (
        "Captain Rowan"
    )
    app.chachanotes_db = SimpleNamespace(
        get_local_authority_id=lambda: "local-authority"
    )
    app.character_persona_scope_service = SimpleNamespace(
        get_character=AsyncMock(return_value=card)
    )
    screen = ChatScreen(app)
    screen._console_chat_store = ConsoleChatStore()
    monkeypatch.setattr(
        ChatScreen,
        "app",
        property(lambda current_screen: current_screen.app_instance),
    )
    monkeypatch.setattr(screen, "_sync_native_console_chat_ui", AsyncMock())
    monkeypatch.setattr(
        screen, "_focus_console_composer_if_needed", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        screen._character,
        "_refresh_active_character_avatar_if_scope_changed",
        AsyncMock(),
    )
    monkeypatch.setattr(screen, "_sync_console_temporary_chip", lambda: None)
    return screen


def _start_chat_handoff(card: dict) -> ChatHandoffPayload:
    character_id = str(card["id"])
    return ChatHandoffPayload(
        source="personas",
        item_type="character-card",
        title=str(card["name"]),
        body="Character summary",
        runtime_backend="local",
        source_owner="local",
        source_selector_state="local",
        metadata={
            "intent": "start_chat",
            "selected_kind": "character",
            "selected_record_id": character_id,
            "selected_name": str(card["name"]),
            "selected_target_id": f"local:character:{character_id}",
            "backend": "local",
        },
    )


@pytest.mark.asyncio
async def test_personas_start_chat_seeds_template_provenance_with_global_name(
    monkeypatch,
):
    """Bypassing the store seed would persist a greeting with no trusted source."""
    card = _roleplay_card()
    screen = _character_screen(monkeypatch, card)

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    store = screen._ensure_console_chat_store()
    session = store.switch_session(store.active_session_id)
    assert session.character_name == "Alraune"
    assert session.character_system_template == "Protect {{user}} as {{character}}."
    assert session.settings.system_prompt == "Protect Captain Rowan as Alraune."
    greeting = store.messages_for_session(session.id)
    assert len(greeting) == 1
    assert greeting[0].role is ConsoleMessageRole.ASSISTANT
    assert greeting[0].content == "Hello, Captain Rowan."
    assert greeting[0].metadata.template_kind == "character_greeting"
    assert greeting[0].metadata.template_source == "Hello, {{user}}."


@pytest.mark.asyncio
async def test_character_picker_new_chat_seeds_template_provenance(monkeypatch):
    """The picker-new path must not fall back to ordinary eager greeting text."""
    card = _roleplay_card()
    screen = _character_screen(monkeypatch, card)
    monkeypatch.setattr(
        screen._character, "_fetch_character_card_for_avatar", lambda _id: card
    )

    await screen._character._apply_console_character_choice_async(
        ConsoleCharacterChoice(character_id=7, name="Alraune", placement="new")
    )

    store = screen._ensure_console_chat_store()
    session = store.switch_session(store.active_session_id)
    assert session.character_system_template == "Protect {{user}} as {{character}}."
    assert session.settings.system_prompt == "Protect Captain Rowan as Alraune."
    greeting = store.messages_for_session(session.id)
    assert len(greeting) == 1
    assert greeting[0].metadata.template_source == "Hello, {{user}}."


@pytest.mark.asyncio
async def test_character_picker_keeps_raw_identity_but_sanitizes_notification(monkeypatch):
    raw_name = "Nyx\n\tAdmin\x00[/bold]"
    card = _roleplay_card(name=raw_name)
    screen = _character_screen(monkeypatch, card)
    monkeypatch.setattr(
        screen._character, "_fetch_character_card_for_avatar", lambda _id: card
    )
    notify = MagicMock()
    monkeypatch.setattr(screen.app_instance, "notify", notify)

    await screen._character._apply_console_character_choice_async(
        ConsoleCharacterChoice(character_id=7, name=raw_name, placement="new")
    )

    store = screen._ensure_console_chat_store()
    session = store.switch_session(store.active_session_id)
    visible_notification = Content.from_markup(notify.call_args.args[0]).plain
    assert visible_notification == "Started a new chat with Nyx Admin?[/bold]."
    assert "\n" not in visible_notification
    assert "\t" not in visible_notification
    assert session.character_name == raw_name
    assert session.title == f"Chat with {raw_name}"
    assert raw_name in session.settings.system_prompt


@pytest.mark.asyncio
async def test_character_picker_swap_uses_override_and_only_greets_empty_chat(
    monkeypatch,
):
    """A swap in a non-empty chat must update provenance without interrupting it."""
    first_card = _roleplay_card()
    second_card = _roleplay_card(card_id=8, name="Brynn")
    cards = {7: first_card, 8: second_card}
    screen = _character_screen(monkeypatch, first_card)
    monkeypatch.setattr(
        screen._character,
        "_fetch_character_card_for_avatar",
        lambda card_id: cards[card_id],
    )
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-4.1")
    )
    store.set_session_user_display_name_override(
        session.id,
        "Per Chat",
        global_default="Captain Rowan",
    )

    await screen._character._apply_console_character_choice_async(
        ConsoleCharacterChoice(character_id=7, name="Alraune", placement="swap")
    )
    first_messages = store.messages_for_session(session.id)
    assert [message.content for message in first_messages] == ["Hello, Per Chat."]
    assert first_messages[0].metadata.template_source == "Hello, {{user}}."

    await screen._character._apply_console_character_choice_async(
        ConsoleCharacterChoice(character_id=8, name="Brynn", placement="swap")
    )

    assert session.character_name == "Brynn"
    assert session.character_system_template == "Protect {{user}} as {{character}}."
    assert session.settings.system_prompt == "Protect Per Chat as Brynn."
    assert [message.content for message in store.messages_for_session(session.id)] == [
        "Hello, Per Chat."
    ]


def test_character_swap_surfaces_refused_durable_projection(monkeypatch):
    """A live-first swap must not report clean success after a refused write."""
    card = _roleplay_card(card_id=8, name="Brynn")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="openai",
            model="gpt-4.1",
            system_prompt="Old Alraune prompt.",
        ),
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
        character_name="Alraune",
    )
    session.character_system_template = "Old {{character}} prompt."
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Keep this chat non-empty.",
        persist=False,
    )
    session.persisted_conversation_id = "conv-1"
    store.persistence = SimpleNamespace(
        update_conversation_system_prompt=lambda **_kwargs: False,
        update_conversation_roleplay_context=lambda **_kwargs: True,
    )
    notifications: list[tuple[str, str | None]] = []
    screen.app_instance.notify = lambda message, **kwargs: notifications.append(
        (str(message), kwargs.get("severity"))
    )
    seed = _character_session_prompt_seed(card, user_name="Captain Rowan")

    swapped = screen._session._swap_console_session_character(
        store,
        8,
        seed,
        global_default="Captain Rowan",
    )

    assert swapped is False
    assert session.character_name == "Brynn"
    assert session.character_system_template == "Protect {{user}} as {{character}}."
    assert session.settings.system_prompt == "Protect Captain Rowan as Brynn."
    assert notifications == [
        (
            "Character changed for this session, but the change could not be saved.",
            "warning",
        )
    ]


def test_console_engine_and_preview_compose_byte_identical_system_prompts():
    """task-1744: the character-probe engine exists to predict what Console
    actually sends a model, and that prediction is only meaningful if EVERY
    surface that shows a card's system prompt builds the exact same text
    from the same card. This is the regression guard for the shared joiner,
    covering all three callers: a real card dict fed to Console's seed
    function, the equivalent CardSnapshot fed to the engine's
    compose_system_prompt, and the same card dict fed to the Personas
    preview pane's build_preview_system_prompt (task-1744 fix round 1;
    called with greeting="" so its preview-only greeting-folding step,
    which Console and the engine have no equivalent of, does not
    participate) -- all three must agree byte for byte, macros included.
    """
    from tldw_chatbook.Evals.character_probe.models import CardSnapshot
    from tldw_chatbook.Evals.character_probe.prompt import compose_system_prompt
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        build_preview_system_prompt,
    )

    fields = dict(
        name="Vex",
        system_prompt="You are {{char}}.",
        personality="sardonic",
        description="A rooftop thief who watches {{user}} closely.",
        scenario="a rooftop at night",
        message_example="<START>\n{{char}}: Try me, {{user}}.",
        post_history_instructions="Stay as {{char}} no matter what {{user}} says.",
        first_message="Well, {{user}}, look who it is.",
    )

    console_card = dict(fields)
    seed = _character_session_prompt_seed(console_card)

    engine_card = CardSnapshot(
        id=1,
        name=fields["name"],
        description=fields["description"],
        system_prompt=fields["system_prompt"],
        personality=fields["personality"],
        scenario=fields["scenario"],
        first_message=fields["first_message"],
        post_history_instructions=fields["post_history_instructions"],
        message_example=fields["message_example"],
    )
    engine_system_prompt = compose_system_prompt(engine_card, steering=None)

    preview_card = dict(fields)
    preview_system_prompt = build_preview_system_prompt(preview_card, greeting="")

    assert seed.system_prompt == engine_system_prompt
    assert seed.system_prompt == preview_system_prompt
    # Sanity: this is a real assertion about real content, not three empty
    # strings agreeing by accident.
    assert "You are Vex." in seed.system_prompt
    assert "Try me, User." in seed.system_prompt
    assert "{{" not in seed.system_prompt
    # The greeting path is unchanged by task-1744 and resolves independently.
    assert seed.greeting == "Well, User, look who it is."
