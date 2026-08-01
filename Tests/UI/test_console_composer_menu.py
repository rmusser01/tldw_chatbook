"""Composer overflow menu: entries, image command, impersonate insertion.

tasks 1680-1683.
"""

import pytest

from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    ACTION_GENERATE_CAPTION,
    ACTION_GENERATE_IMAGE,
    ACTION_IMPERSONATE,
    ACTION_NARRATE_CONVERSATION,
    build_composer_menu_entries,
)
from tldw_chatbook.Widgets.Console.console_generate_image_modal import (
    DEFAULT_CHOICE,
    build_generate_image_command,
)


@pytest.mark.unit
def test_menu_lists_the_four_requested_actions_in_order():
    """task-1680: the menu carries exactly the requested entries."""
    ids = [e.action_id for e in build_composer_menu_entries()]
    assert ids == [
        ACTION_GENERATE_IMAGE,
        ACTION_GENERATE_CAPTION,
        ACTION_NARRATE_CONVERSATION,
        ACTION_IMPERSONATE,
    ]


@pytest.mark.unit
def test_caption_entry_requires_an_attachment():
    """task-1682: captioning needs an image, and says so when there is none."""
    without = {e.action_id: e for e in build_composer_menu_entries()}[
        ACTION_GENERATE_CAPTION
    ]
    assert without.enabled is False
    assert "Attach" in without.description

    with_attachment = {
        e.action_id: e
        for e in build_composer_menu_entries(has_attachment=True)
    }[ACTION_GENERATE_CAPTION]
    assert with_attachment.enabled is True


@pytest.mark.unit
def test_generate_image_command_matches_the_documented_grammar():
    """task-1681: ``:backend`` and ``@style`` lead, then the prompt."""
    assert build_generate_image_command(prompt="a fox") == "/generate-image a fox"
    assert (
        build_generate_image_command(prompt="a fox", backend="swarmui", style="anime")
        == "/generate-image :swarmui @anime a fox"
    )
    # Defaults are omitted rather than emitted as literal tokens.
    assert (
        build_generate_image_command(
            prompt="a fox", backend=DEFAULT_CHOICE, style=DEFAULT_CHOICE
        )
        == "/generate-image a fox"
    )
    assert build_generate_image_command(prompt="  spaced  ") == "/generate-image spaced"


@pytest.mark.unit
def test_impersonate_appends_then_replaces_its_own_text():
    """task-1683: never clobber the user's text; replace only our own.

    Drives the real screen methods against fakes: the first suggestion is
    appended on a new line after existing text, and a second suggestion
    replaces the first rather than stacking.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    class _Composer:
        def __init__(self, text: str) -> None:
            self.text = text

        def draft_text(self) -> str:
            return self.text

        def load_draft(self, text: str) -> None:
            self.text = text

    class _Store:
        def __init__(self) -> None:
            self.active_session_id = "s1"
            self.drafts = {"s1": ""}

        def set_session_draft(self, session_id, text):
            self.drafts[session_id] = text

        def session_draft(self, session_id):
            return self.drafts[session_id]

    screen = ChatScreen.__new__(ChatScreen)
    composer = _Composer("my own words")
    store = _Store()
    screen._console_composer_or_none = lambda: composer
    screen._ensure_console_chat_store = lambda: store

    screen._replace_console_impersonate_text("s1", "FIRST draft")
    assert composer.text == "my own words\nFIRST draft"

    screen._replace_console_impersonate_text("s1", "SECOND draft")
    assert composer.text == "my own words\nSECOND draft", "must replace, not stack"
    assert "FIRST draft" not in composer.text


@pytest.mark.unit
def test_impersonate_appends_when_the_user_edited_our_text():
    """If the user changed our suggestion, appending beats rewriting it."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    class _Composer:
        def __init__(self, text: str) -> None:
            self.text = text

        def draft_text(self) -> str:
            return self.text

        def load_draft(self, text: str) -> None:
            self.text = text

    class _Store:
        def __init__(self) -> None:
            self.active_session_id = "s1"
            self.drafts = {"s1": ""}

        def set_session_draft(self, session_id, text):
            self.drafts[session_id] = text

        def session_draft(self, session_id):
            return self.drafts[session_id]

    screen = ChatScreen.__new__(ChatScreen)
    composer = _Composer("")
    store = _Store()
    screen._console_composer_or_none = lambda: composer
    screen._ensure_console_chat_store = lambda: store

    screen._replace_console_impersonate_text("s1", "generated")
    composer.load_draft("generated, then I edited it")

    screen._replace_console_impersonate_text("s1", "fresh")
    assert composer.text == "generated, then I edited it\nfresh"
