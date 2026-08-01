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


@pytest.mark.unit
def test_draft_addition_never_doubles_a_newline():
    """Qodo PR #1160: a draft already ending in a newline gained a second.

    That put inserted text after a blank line instead of on the next one.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    assert ChatScreen._draft_addition("", "x") == "x"
    assert ChatScreen._draft_addition("   ", "x") == "x"
    assert ChatScreen._draft_addition("hello", "x") == "\nx"
    assert ChatScreen._draft_addition("hello\n", "x") == "x"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_menu_opens_from_the_composer_button_and_returns_an_action():
    """Integration: drive the real modal through its UI boundary.

    Qodo PR #1160 asked for coverage past the unit level: this mounts the
    menu in a running app, presses the Generate Image row, and asserts the
    screen receives that action id.
    """
    from textual.app import App
    from textual.widgets import Button

    from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
        ConsoleComposerMenuModal,
    )

    class _Host(App):
        pass

    received: list[str | None] = []
    app = _Host()
    async with app.run_test() as pilot:
        await app.push_screen(
            ConsoleComposerMenuModal(has_attachment=False), callback=received.append
        )
        await pilot.pause()

        caption = app.screen.query_one(
            f"#console-composer-menu-{ACTION_GENERATE_CAPTION}", Button
        )
        assert caption.disabled is True, "caption must be gated without an attachment"

        await pilot.click(f"#console-composer-menu-{ACTION_GENERATE_IMAGE}")
        await pilot.pause()
        await pilot.pause()

    assert received == [ACTION_GENERATE_IMAGE]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_image_modal_returns_the_composed_command():
    """Integration: type a prompt, accept, and get the command back."""
    from textual.app import App
    from textual.widgets import Input

    from tldw_chatbook.Widgets.Console.console_generate_image_modal import (
        ConsoleGenerateImageModal,
    )

    class _Host(App):
        pass

    received: list[str | None] = []
    app = _Host()
    async with app.run_test() as pilot:
        await app.push_screen(
            ConsoleGenerateImageModal(backends=("swarmui",), styles={"anime": "Anime"}),
            callback=received.append,
        )
        await pilot.pause()
        app.screen.query_one("#console-generate-image-prompt", Input).value = "a fox"
        await pilot.pause()
        await pilot.click("#console-generate-image-accept")
        await pilot.pause()
        await pilot.pause()

    assert received == ["/generate-image a fox"]


def _fake_controller_with(messages):
    """Build a controller stub whose transcript is ``messages``."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    class _Msg:
        def __init__(self, role, content, status="ok"):
            self.role = role
            self.content = content
            self.status = status

    class _Store:
        def messages_for_session(self, session_id):
            return [
                _Msg(ConsoleMessageRole.USER if r == "user" else
                     ConsoleMessageRole.ASSISTANT, c, s)
                for r, c, s in messages
            ]

    controller = ConsoleChatController.__new__(ConsoleChatController)
    controller.store = _Store()
    return controller


@pytest.mark.unit
@pytest.mark.asyncio
async def test_impersonate_payload_obeys_the_provider_contract():
    """cubic PR #1160: the request must be user-first and user-final.

    A seeded character greeting is an assistant-first row, which strict
    providers reject (task-427); a completed turn leaves the transcript
    ending on an assistant row, which user-final providers reject.
    """
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    captured: dict[str, list] = {}

    controller = _fake_controller_with(
        [
            ("assistant", "Greetings, traveller.", "ok"),   # seeded greeting
            ("user", "hello", "ok"),
            ("assistant", "broken reply", "failed"),        # failed row
            ("assistant", "a real reply", "ok"),            # ends assistant
        ]
    )

    class _Resolution:
        ready = True

    async def _resolve(_selection):
        return _Resolution()

    async def _collect(_resolution, messages):
        captured["messages"] = messages
        return "drafted reply"

    controller.provider_gateway = type(
        "_G", (), {"resolve_for_send": staticmethod(_resolve)}
    )()
    controller._provider_selection = lambda: None
    controller._collect_summary_completion = _collect
    controller._seeded_greeting_text = ConsoleChatController._seeded_greeting_text

    result = await controller.impersonate_user_reply("s1")
    assert result.text == "drafted reply"

    messages = captured["messages"]
    roles = [m["role"] for m in messages]
    assert roles[0] == ConsoleMessageRole.SYSTEM.value
    # First conversation row is the USER turn -- the greeting was dropped.
    assert roles[1] == ConsoleMessageRole.USER.value
    assert "Greetings, traveller." not in messages[1]["content"]
    # The greeting still reaches the model, folded into the system row.
    assert "Greetings, traveller." in messages[0]["content"]
    # The failed row never ships.
    blob = " ".join(str(m["content"]) for m in messages)
    assert "broken reply" not in blob
    # And the array ends on a USER turn.
    assert roles[-1] == ConsoleMessageRole.USER.value


@pytest.mark.unit
def test_transcript_trimming_keeps_newest_and_stays_user_first():
    """Long threads drop OLD turns, never leading with an assistant row."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    controller = ConsoleChatController.__new__(ConsoleChatController)
    rows = []
    for i in range(200):
        rows.append({"role": "user", "content": "u" * 400})
        rows.append({"role": "assistant", "content": "a" * 400})

    kept = ConsoleChatController._trim_transcript_to_budget(controller, rows)

    assert len(kept) < len(rows), "a huge thread must be trimmed"
    assert kept[0]["role"] == "user", "must never lead with an assistant row"
    assert kept[-1] == rows[-1], "the newest turn is always kept"
