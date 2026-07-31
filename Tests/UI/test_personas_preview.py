"""Mounted tests for the Personas preview-conversation pane."""

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    PreviewGreetingSelected,
    PreviewOpenInConsoleRequested,
    PreviewReplyRequested,
    PreviewResetRequested,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
    PersonasPreviewPane,
)

pytestmark = pytest.mark.asyncio


class PreviewApp(App):
    def __init__(self):
        super().__init__()
        self.replies: list[str] = []
        self.resets = 0
        self.opens = 0

    def compose(self):
        yield PersonasPreviewPane(id="personas-preview-pane")

    def on_preview_reply_requested(self, message: PreviewReplyRequested) -> None:
        self.replies.append(message.user_message)

    def on_preview_reset_requested(self, message: PreviewResetRequested) -> None:
        self.resets += 1

    def on_preview_open_in_console_requested(
        self, message: PreviewOpenInConsoleRequested
    ) -> None:
        self.opens += 1


def _line_texts(app) -> list[str]:
    return [str(line.renderable) for line in app.query(".personas-preview-line")]


async def test_collapsed_by_default_and_toggle_expands():
    app = PreviewApp()
    async with app.run_test() as pilot:
        body = pilot.app.query_one("#personas-preview-body")
        assert body.display is False
        pilot.app.query_one("#personas-preview-toggle", Button).press()
        await pilot.pause()
        assert body.display is True
        pilot.app.query_one("#personas-preview-toggle", Button).press()
        await pilot.pause()
        assert body.display is False


async def test_buttons_carry_shared_flat_button_classes():
    app = PreviewApp()
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#personas-preview-test-reply", Button).has_class(
            "console-action-secondary"
        )
        assert pilot.app.query_one("#personas-preview-reset", Button).has_class(
            "console-action-subdued"
        )
        assert pilot.app.query_one("#personas-preview-open-console", Button).has_class(
            "console-action-subdued"
        )
        assert pilot.app.query_one("#personas-preview-toggle", Button).has_class(
            "console-action-subdued"
        )


async def test_expand_api_shows_body():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        assert pilot.app.query_one("#personas-preview-body").display is True


async def test_seed_append_reset_roundtrip():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Greetings, detective.")
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Greetings, detective."]
        pane.append_user("Who are you?")
        pane.append_reply("Your humble narrator.")
        await pilot.pause()
        assert _line_texts(pilot.app) == [
            "character: Greetings, detective.",
            "User: Who are you?",
            "character: Your humble narrator.",
        ]
        assert pane.transcript_text() == (
            "character: Greetings, detective.\n"
            "User: Who are you?\n"
            "character: Your humble narrator."
        )
        pane.set_status("Ready")
        await pane.reset()
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Greetings, detective."]
        assert (
            str(pilot.app.query_one("#personas-preview-status", Static).renderable)
            == ""
        )


async def test_seed_empty_greeting_clears_transcript():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        pane.append_user("Hi")
        await pilot.pause()
        await pane.seed_greeting("")
        await pilot.pause()
        assert _line_texts(pilot.app) == []
        assert pane.transcript_text() == ""


async def test_status_region_renders_below_transcript():
    """The status line is its own region beneath the transcript, never above.

    Regression guard: a provider/error status must not interleave above the
    chronological greeting -> you -> character transcript. We assert DOM
    ordering (transcript precedes status precedes input) and that, with both
    a transcript and a status set, the greeting/user lines still read in
    order while the status sits in its separate region.
    """
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Greetings, detective.")
        pane.append_user("Who are you?")
        pane.set_status("anthropic is not ready - configure in Settings")
        await pilot.pause()

        body = pilot.app.query_one("#personas-preview-body")
        child_ids = [c.id for c in body.children]
        transcript_pos = child_ids.index("personas-preview-transcript")
        status_pos = child_ids.index("personas-preview-status")
        input_pos = child_ids.index("personas-preview-input")
        # Status must sit between the transcript and the input, not above it.
        assert transcript_pos < status_pos < input_pos

        # The transcript itself stays in chronological order.
        assert _line_texts(pilot.app) == [
            "character: Greetings, detective.",
            "User: Who are you?",
        ]
        # The status lives in its own region, separate from the transcript.
        assert "anthropic is not ready" in str(
            pilot.app.query_one("#personas-preview-status", Static).renderable
        )


async def test_double_seed_same_tick_does_not_crash():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("First greeting.")
        await pane.seed_greeting("Second greeting.")
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Second greeting."]


async def test_transcript_lines_carry_role_classes():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        pane.append_user("Hi")
        await pilot.pause()
        user_lines = pilot.app.query(".personas-preview-line-you")
        character_lines = pilot.app.query(".personas-preview-line-character")
        assert [str(line.renderable) for line in user_lines] == ["User: Hi"]
        assert [str(line.renderable) for line in character_lines] == ["character: Hello."]


async def test_test_reply_posts_message_and_clears_input():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        pilot.app.query_one("#personas-preview-input", Input).value = "Hi there"
        pilot.app.query_one("#personas-preview-test-reply", Button).press()
        await pilot.pause()
        assert pilot.app.replies == ["Hi there"]
        assert pilot.app.query_one("#personas-preview-input", Input).value == ""
        assert _line_texts(pilot.app) == ["User: Hi there"]


async def test_enter_in_input_submits_like_test_reply():
    """Enter in the preview input takes the same path as the Test Reply button."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        field = pilot.app.query_one("#personas-preview-input", Input)
        field.focus()
        await pilot.pause()
        field.value = "Hi there"
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.replies == ["Hi there"]
        assert field.value == ""
        assert _line_texts(pilot.app) == ["User: Hi there"]


async def test_oversized_message_is_rejected_with_readable_status():
    """A message over the cap is not posted; the input keeps the draft."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        field = pilot.app.query_one("#personas-preview-input", Input)
        oversized = "x" * 4001
        field.value = oversized
        pilot.app.query_one("#personas-preview-test-reply", Button).press()
        await pilot.pause()
        assert pilot.app.replies == []
        assert _line_texts(pilot.app) == []
        assert field.value == oversized  # the draft stays editable
        status = str(pilot.app.query_one("#personas-preview-status", Static).renderable)
        assert "Message too long (max 4000 characters)." == status


async def test_scripty_message_is_rejected_with_readable_status():
    """Injection-shaped input is rejected at the boundary, never posted."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        field = pilot.app.query_one("#personas-preview-input", Input)
        field.value = "<script>alert(1)</script>"
        pilot.app.query_one("#personas-preview-test-reply", Button).press()
        await pilot.pause()
        assert pilot.app.replies == []
        assert _line_texts(pilot.app) == []
        status = str(pilot.app.query_one("#personas-preview-status", Static).renderable)
        assert status.strip()
        assert "Traceback" not in status


async def test_enter_with_empty_input_is_a_noop():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        field = pilot.app.query_one("#personas-preview-input", Input)
        field.focus()
        await pilot.pause()
        field.value = "   "
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.replies == []
        assert _line_texts(pilot.app) == []


async def test_streaming_reply_updates_one_line_progressively():
    """begin_reply/append_reply_chunk grow a single character line in place."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        await pilot.pause()
        pane.begin_reply()
        pane.append_reply_chunk("Your humble ")
        await pilot.pause()
        assert _line_texts(pilot.app) == [
            "character: Hello.",
            "character: Your humble ",
        ]
        pane.append_reply_chunk("narrator.")
        await pilot.pause()
        assert _line_texts(pilot.app) == [
            "character: Hello.",
            "character: Your humble narrator.",
        ]
        assert pane.transcript_text() == (
            "character: Hello.\ncharacter: Your humble narrator."
        )
        pane.finalize_reply()
        # A finalized line is committed: a later discard must not remove it.
        await pane.discard_partial_reply()
        await pilot.pause()
        assert "character: Your humble narrator." in pane.transcript_text()


async def test_discard_partial_reply_removes_in_progress_line():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        await pilot.pause()
        pane.begin_reply()
        pane.append_reply_chunk("Half a tho")
        await pilot.pause()
        await pane.discard_partial_reply()
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Hello."]
        assert pane.transcript_text() == "character: Hello."
        # Discard with no partial in progress is a no-op.
        await pane.discard_partial_reply()
        pane.append_user("Still works")
        await pilot.pause()
        assert pane.transcript_text() == "character: Hello.\nUser: Still works"


async def test_seed_greeting_clears_partial_reply_state():
    """A reseed mid-stream wipes the partial line; discard after is a no-op."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello.")
        pane.begin_reply()
        pane.append_reply_chunk("Half a tho")
        await pilot.pause()
        await pane.seed_greeting("Fresh greeting.")
        await pilot.pause()
        await pane.discard_partial_reply()
        await pilot.pause()
        assert _line_texts(pilot.app) == ["character: Fresh greeting."]
        assert pane.transcript_text() == "character: Fresh greeting."


async def test_test_reply_with_empty_input_is_a_noop():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        pilot.app.query_one("#personas-preview-input", Input).value = "   "
        pilot.app.query_one("#personas-preview-test-reply", Button).press()
        await pilot.pause()
        assert pilot.app.replies == []
        assert _line_texts(pilot.app) == []


async def test_reset_button_restores_greeting_and_posts_reset():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pane.seed_greeting("Hello.")
        pane.append_user("Hi")
        pane.append_reply("Hey.")
        pane.set_status("Ready")
        await pilot.pause()
        pilot.app.query_one("#personas-preview-reset", Button).press()
        await pilot.pause()
        assert pilot.app.resets == 1
        assert _line_texts(pilot.app) == ["character: Hello."]
        assert (
            str(pilot.app.query_one("#personas-preview-status", Static).renderable)
            == ""
        )


async def test_open_in_console_posts_message():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pilot.pause()
        pilot.app.query_one("#personas-preview-open-console", Button).press()
        await pilot.pause()
        assert pilot.app.opens == 1


async def test_markup_like_transcript_content_renders_without_raising():
    """Greeting/user/reply text with Rich-markup-looking content renders literally."""
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.expand()
        await pane.seed_greeting("[/oops]")
        pane.append_user("[/bad user]")
        pane.append_reply("[bold]unclosed")
        await pilot.pause()  # would raise MarkupError at render with markup on
        assert _line_texts(pilot.app) == [
            "character: [/oops]",
            "User: [/bad user]",
            "character: [bold]unclosed",
        ]


async def test_status_is_readable():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        pane.set_status("Running")
        await pilot.pause()
        status = str(pilot.app.query_one("#personas-preview-status", Static).renderable)
        assert status == "Running"
        assert "Traceback" not in status


async def test_greeting_text_property_returns_seeded_greeting():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello, traveller.")
        assert pane.greeting_text == "Hello, traveller."
        pane.refresh_greeting_seed("Updated greeting.")
        assert pane.greeting_text == "Updated greeting."


async def test_speaker_labels_use_character_name():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_speakers(character="Sherlock Holmes")
        await pane.seed_greeting("Greetings.")
        await pilot.pause()
        pane.append_user("Hi")
        pane.append_reply("Elementary.")
        await pilot.pause()
        assert _line_texts(app) == [
            "Sherlock Holmes: Greetings.",
            "User: Hi",
            "Sherlock Holmes: Elementary.",
        ]
        assert pane.transcript_text() == (
            "Sherlock Holmes: Greetings.\nUser: Hi\nSherlock Holmes: Elementary."
        )


async def test_empty_preview_uses_neutral_user_label():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("")
        assert pane.transcript_text() == ""
        assert pane._user_label == "User"
        pane.append_user("Hi")
        pane.append_reply("Hello.")
        await pilot.pause()
        assert _line_texts(app) == ["User: Hi", "character: Hello."]


async def test_set_speakers_ignores_empty_name():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_speakers(character="")
        pane.append_reply("Hi.")
        await pilot.pause()
        assert _line_texts(app) == ["character: Hi."]


async def test_styled_line_italicizes_action_and_escapes_markup():
    app = PreviewApp()
    async with app.run_test():
        pane = app.query_one(PersonasPreviewPane)
        waves = pane._styled_line("*waves*")
        assert str(waves) == "waves"
        assert any("italic" in str(span.style) for span in waves.spans)
        assert str(pane._styled_line("[/oops]")) == "[/oops]"
        assert str(pane._styled_line("you: 5 * 3")) == "you: 5 * 3"


async def test_action_span_renders_italic_not_literal_asterisks():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.append_reply("*smiles warmly*")
        await pilot.pause()
        line = app.query(".personas-preview-line").last()
        assert "*" not in str(line.renderable)
        assert "smiles warmly" in str(line.renderable)
        assert any("italic" in str(s.style) for s in line.renderable.spans)


async def test_set_speakers_relabels_existing_character_lines():
    # task-437 review: a rename mid-conversation relabels already-rendered
    # character lines (no stale/mixed prefixes); user lines are untouched.
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_speakers(character="Alice")
        await pane.seed_greeting("Hi.")
        pane.append_user("hello")
        pane.append_reply("hey")
        await pilot.pause()
        pane.set_speakers(character="Bob")
        await pilot.pause()
        assert pane.transcript_text() == "Bob: Hi.\nUser: hello\nBob: hey"
        assert "Alice" not in pane.transcript_text()
        assert _line_texts(app) == ["Bob: Hi.", "User: hello", "Bob: hey"]


async def test_reset_speakers_restores_defaults():
    # task-437: leaving a character context must drop the stale name so a later
    # reply renders under the neutral default, not the previous character's name.
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_speakers(character="Alice")
        pane.reset_speakers()
        pane.append_user("Hi")
        pane.append_reply("Hello.")
        await pilot.pause()
        assert pane._user_label == "User"
        assert _line_texts(app) == ["User: Hi", "character: Hello."]


# ===== TASK-438: alternate-greeting selector =====


async def test_greeting_selector_hidden_without_alternates():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_greetings(["Only greeting."])
        await pilot.pause()
        assert app.query_one("#personas-preview-greeting-row").display is False


async def test_greeting_selector_shown_with_alternates():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_greetings(["Primary.", "Alt one.", "Alt two."])
        await pilot.pause()
        assert app.query_one("#personas-preview-greeting-row").display is True
        select = app.query_one("#personas-preview-greeting-select", Select)
        assert len(list(select._options)) == 3  # 3 greetings


async def test_choosing_greeting_posts_message():
    posted: list[int] = []
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)

        original_post_message = pane.post_message

        def _capture(message):
            if isinstance(message, PreviewGreetingSelected):
                posted.append(message.index)
            return original_post_message(message)

        pane.post_message = _capture

        pane.set_greetings(["Primary.", "Alt one."])
        await pilot.pause()
        select = app.query_one("#personas-preview-greeting-select", Select)
        select.value = 1
        await pilot.pause()
        # set_greetings populates the Select under prevent(Select.Changed), so the
        # only PreviewGreetingSelected is this genuine user pick (index 1) - no
        # spurious programmatic index-0 post (task-438 review).
        assert posted == [1]


# ===== TASK-617.1: Personas never supply the human identity =====


@pytest.fixture
def _seed_legacy_human_identity(monkeypatch):
    """Expose a retired config value through the existing config read seam."""
    from tldw_chatbook import config as config_module

    legacy_key = "_".join(("active", "user", "profile"))
    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda: {"character_defaults": {legacy_key: "Sam"}},
    )

    def _forbid_write(*args, **kwargs):
        raise AssertionError("preview must not mutate the legacy human pointer")

    monkeypatch.setattr(
        config_module,
        "save_settings_to_cli_config",
        _forbid_write,
    )


class _ProfileService:
    """Sync profile service spy that must stay outside preview rendering."""

    def __init__(self, names):
        self._names = list(names)
        self.calls = 0

    def list_persona_profiles(self, **kwargs):
        self.calls += 1
        return [{"name": name} for name in self._names]


class _ControllerScreen:
    """Minimal screen double driving the controller against the mounted pane."""

    def __init__(self, app, profile_service=None):
        self._app = app
        self.app_instance = SimpleNamespace(
            local_character_persona_service=profile_service
        )

    def query_one(self, selector, *args):
        return self._app.query_one(PersonasPreviewPane)


async def test_greeting_uses_literal_user_and_preview_label(
    _seed_legacy_human_identity,
):
    """A legacy profile value cannot replace the neutral human identity."""
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        PersonasPreviewController,
    )

    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        service = _ProfileService(["Sam"])
        controller = PersonasPreviewController(_ControllerScreen(app, service))
        seed = controller._load_greetings(
            {"first_message": "Hello {{user}}, I am {{char}}."}, "Elara"
        )
        assert seed == "Hello User, I am Elara."
        assert pane._user_label == "User"
        assert service.calls == 0
        await pane.seed_greeting(seed)
        pane.append_user("hi")
        await pilot.pause()
        assert "User: hi" in pane.transcript_text()


class _ServerScopeService:
    """Async profile service spy that must stay outside preview rendering."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    async def list_persona_profiles(self, **kwargs):
        self.calls += 1
        return self._payload


@pytest.mark.parametrize("mode", ("local", "server"))
async def test_runtime_modes_share_neutral_user_substitution(
    mode,
    _seed_legacy_human_identity,
):
    """Local and server modes both render the same literal human identity."""
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        PersonasPreviewController,
    )

    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        local_service = _ProfileService(["Sam"])
        server_service = _ServerScopeService(
            [{"name": "Sam"}]
        )
        screen = _ControllerScreen(app, local_service)
        screen.app_instance.character_persona_scope_service = server_service
        screen.persona_handler = SimpleNamespace(current_mode=lambda: mode)
        screen.workers = SimpleNamespace(cancel_group=lambda *a, **k: None)
        controller = PersonasPreviewController(screen)

        await controller.reset_for_character(
            character_id="7",
            character_name="Elara",
            record={"first_message": "Hello {{user}}, I am {{char}}."},
        )
        await pilot.pause()

        assert pane.transcript_text() == "character: Hello User, I am Elara."
        assert pane._user_label == "User"
        assert local_service.calls == 0
        assert server_service.calls == 0


async def test_alias_tokens_substitute_character_name_with_neutral_user():
    """Character-side aliases stay independent from the neutral human label."""
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        PersonasPreviewController,
    )

    app = PreviewApp()
    async with app.run_test():
        controller = PersonasPreviewController(
            _ControllerScreen(app, _ProfileService(["Sam"]))
        )
        seed = controller._load_greetings(
            {"first_message": "Hello {{user}}, I am {{persona}}/{{character}}."},
            "Elara",
        )
        assert seed == "Hello User, I am Elara/Elara."


# ---- build_preview_system_prompt (tasks 1530/1531) ----


def test_build_preview_system_prompt_resolves_macros():
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        build_preview_system_prompt,
    )

    record = {"name": "Elara", "description": "{{char}} guides {{user}}."}

    assert build_preview_system_prompt(record) == "Elara guides User."


def test_build_preview_system_prompt_folds_greeting_after_prompt():
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        build_preview_system_prompt,
    )

    out = build_preview_system_prompt(
        {"name": "Elara", "description": "Guide the user."},
        greeting="Hello, traveler.",
    )

    assert out.startswith("Guide the user.")
    assert "Hello, traveler." in out


def test_build_preview_system_prompt_empty_record_falls_back():
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        build_preview_system_prompt,
    )

    assert build_preview_system_prompt({}) == "Stay in character."
