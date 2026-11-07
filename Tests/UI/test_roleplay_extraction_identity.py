"""Moved Roleplay values retain their public identities and live callback seams."""

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_item_widget_and_authority_models_keep_original_public_identity():
    from tldw_chatbook.UI.Persona_Modules import roleplay_frame_state as frame
    from tldw_chatbook.UI.Persona_Modules import roleplay_list_window_state as items
    from tldw_chatbook.UI.Screens import personas_screen as screen
    from tldw_chatbook.Widgets.Persona_Widgets import personas_library_pane as pane
    from tldw_chatbook.Widgets.Persona_Widgets import personas_state as state
    from tldw_chatbook.Widgets.Persona_Widgets import roleplay_item_list as widget

    assert pane.RoleplayItemList is widget.RoleplayItemList
    assert pane.RoleplayItemList.WindowWanted is widget.RoleplayItemList.WindowWanted
    assert pane.LibraryRow is widget.LibraryRow
    assert frame.ItemPrompt is items.ItemPrompt
    assert frame.NewChoice is items.NewChoice
    assert frame.ITEM_KINDS is items.ITEM_KINDS
    assert screen._ActorPackPortraitChoice is state._ActorPackPortraitChoice
    assert screen._VisualIdentityPreviewSnapshot is state._VisualIdentityPreviewSnapshot
    assert screen._CharacterTTSControlSnapshot is state._CharacterTTSControlSnapshot


def test_seed_facades_read_current_expander_and_return_original_value_types(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import session

    calls = []

    def expand(template, **kwargs):
        calls.append((template, kwargs))
        return "current:" + template

    monkeypatch.setattr(session, "expand_character_template", expand)
    character = session._character_session_prompt_seed(
        {"name": "Actor", "system_prompt": "instruction", "first_message": "hello"}
    )
    persona = session._persona_session_prompt_seed(
        {"name": "Writer", "system_prompt": "persona instruction"}
    )
    assert type(character) is session.CharacterSessionPromptSeed
    assert type(persona) is session.PersonaSessionPromptSeed
    assert character.system_prompt == "current:" + character.system_template
    assert character.greeting == "current:hello"
    assert persona.system_prompt == "current:persona instruction"
    assert len(calls) == 3


def test_transcript_wrap_cache_keeps_live_content_projection(monkeypatch):
    from tldw_chatbook.Widgets.Console import console_transcript as transcript

    calls = []
    original_content = transcript.Content

    class CountingContent(original_content):
        def wrap(self, width, *args, **kwargs):
            calls.append(width)
            return super().wrap(width, *args, **kwargs)

    monkeypatch.setattr(transcript, "Content", CountingContent)
    transcript._body_wrap_table.cache_clear()
    try:
        assert transcript._body_cell_to_offset("abcdefghij", 4, 2, 1) == 6
        assert transcript._body_cell_to_offset("abcdefghij", 4, 3, 1) == 7
        assert transcript._body_cell_to_offset("abcdefghij", 5, 1, 1) == 6
        assert calls == [4, 5]
        assert transcript._body_wrap_table.cache_info().hits == 1
    finally:
        transcript._body_wrap_table.cache_clear()


def test_portrait_projection_reads_the_screen_validator_seam(monkeypatch):
    from tldw_chatbook.UI.Screens import personas_screen as screen

    calls = []
    portrait = b"\x89PNG\r\n\x1a\nretained test bytes"

    def validate(name, data):
        calls.append((name, data))

    monkeypatch.setattr(screen, "validate_actor_portrait", validate)
    choices = screen.PersonasScreen._eligible_actor_pack_portraits(
        [{"id": 17, "version": 3, "name": " Actor ", "image": portrait}]
    )
    assert calls == [("portrait.png", portrait)]
    assert len(choices) == 1
    assert type(choices[0]) is screen._ActorPackPortraitChoice
    assert (choices[0].character_id, choices[0].revision, choices[0].name) == (
        17, 3, "Actor"
    )
