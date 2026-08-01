"""Composer overflow menu: entries, image command, impersonate insertion.

tasks 1680-1683.
"""

import pytest

from tldw_chatbook.Chat.console_ephemeral import ACTION_SAVE_CHAT
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
def test_caption_entry_requires_an_IMAGE_attachment():
    """task-1682: captioning needs an image, and names the blocking case.

    The entry is disabled rather than hidden in both blocked cases, so an
    unavailable action still explains itself.
    """
    def caption(kind):
        return {
            e.action_id: e for e in build_composer_menu_entries(attachment_kind=kind)
        }[ACTION_GENERATE_CAPTION]

    nothing = caption("none")
    assert nothing.enabled is False
    assert "Attach an image" in nothing.description

    # A PDF is staged: the entry stays visible and says why it can't act.
    other = caption("other")
    assert other.enabled is False
    assert "not an image" in other.description

    image = caption("image")
    assert image.enabled is True
    assert "Caption" in image.description


@pytest.mark.unit
def test_attachment_kind_reads_the_staged_records():
    """The screen classifies real staged attachments, not the chip label."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    class _A:
        def __init__(self, mime, file_type=""):
            self.mime_type = mime
            self.file_type = file_type

    class _Store:
        def __init__(self, pendings):
            self.active_session_id = "s1"
            self._pendings = pendings

        def pending_attachments(self, session_id):
            return self._pendings

    screen = ChatScreen.__new__(ChatScreen)

    screen._ensure_console_chat_store = lambda: _Store([])
    assert screen._console_pending_attachment_kind() == "none"

    screen._ensure_console_chat_store = lambda: _Store([_A("application/pdf")])
    assert screen._console_pending_attachment_kind() == "other"

    screen._ensure_console_chat_store = lambda: _Store([_A("image/png")])
    assert screen._console_pending_attachment_kind() == "image"

    # Mixed staging counts as captionable.
    screen._ensure_console_chat_store = lambda: _Store(
        [_A("application/pdf"), _A("", file_type="image")]
    )
    assert screen._console_pending_attachment_kind() == "image"


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
            ConsoleComposerMenuModal(attachment_kind="none"), callback=received.append
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


@pytest.mark.unit
def test_temporary_tab_has_a_free_chord_and_a_palette_entry():
    """Alt+T must not collide, and the palette path must exist regardless.

    A chord that a terminal swallows is not a guaranteed path; the palette
    entry is. Both are asserted so neither can quietly disappear.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    keys = [b.key for b in ChatScreen.BINDINGS]
    assert keys.count("alt+t") == 1
    assert [b.action for b in ChatScreen.BINDINGS if b.key == "alt+t"] == [
        "new_temporary_console_tab"
    ]
    assert callable(ChatScreen.action_new_temporary_console_tab)


@pytest.mark.unit
def test_controller_new_session_can_be_born_temporary():
    """`ephemeral` reaches the store, not just the controller signature."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    assert store.create_session(title="A").ephemeral is False
    assert store.create_session(title="B", ephemeral=True).ephemeral is True


@pytest.mark.unit
def test_temporary_chip_is_hidden_outside_a_temporary_chat():
    """The chip says one thing; when it does not apply it vanishes."""
    from tldw_chatbook.Chat.console_ephemeral import TEMPORARY_LABEL
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleStatusChips,
    )

    label, tooltip, hidden = ConsoleStatusChips._temporary_chip_render(True)
    assert label == TEMPORARY_LABEL
    assert hidden is False
    assert "not saved" in tooltip.lower()

    _label, _tooltip, hidden_normal = ConsoleStatusChips._temporary_chip_render(False)
    assert hidden_normal is True


@pytest.mark.unit
def test_console_active_session_is_ephemeral_reads_the_active_flag():
    """The shared accessor Task 8/9 build on: store-only, no widget needed.

    ``ConsoleChatStore`` has no public single-session getter, so this reads
    ``sessions()`` + ``active_session_id`` -- the same public surface any
    other caller has.
    """
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    screen._console_chat_store = None
    assert screen._console_active_session_is_ephemeral() is False

    store = ConsoleChatStore()
    screen._console_chat_store = store
    normal = store.create_session(title="Normal")
    temp = store.create_session(title="Temp", ephemeral=True)

    store.switch_session(normal.id)
    assert screen._console_active_session_is_ephemeral() is False

    store.switch_session(temp.id)
    assert screen._console_active_session_is_ephemeral() is True


@pytest.mark.asyncio
async def test_activate_console_session_for_workspace_keeps_temporary_chip_honest():
    """task-7 review: three callers reach ``_activate_console_session_for_
    workspace`` and none of them refresh the temporary chip afterward --
    they only await ``_sync_native_console_chat_ui()``, which never touches
    it (same reason it never touches the scope chip). Tests the
    CONSEQUENCE (chip visibility after the real switch/create branches
    run), in both directions, so a push that hardcoded either answer would
    fail this.
    """
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleTemporaryChip,
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()

        normal = store.create_session(title="Normal", workspace_id="workspace-b")
        temp = store.create_session(
            title="Temp", workspace_id="workspace-a", ephemeral=True
        )
        # `temp` is active now (`create_session` activates inline). Establish
        # a verified-correct baseline via the real sync path rather than
        # trusting the chip's compose-time default.
        assert store.active_session_id == temp.id
        console._sync_console_temporary_chip()
        chip = console.query_one("#console-temporary-chip", ConsoleTemporaryChip)
        assert chip.display is True

        # "switch to an existing session in the workspace" branch.
        console._activate_console_session_for_workspace("workspace-b")
        await pilot.pause()
        assert store.active_session_id == normal.id
        assert chip.display is False, (
            "switching to a saved session's workspace must clear a stale "
            "Temporary chip"
        )

        # Same branch, opposite direction.
        console._activate_console_session_for_workspace("workspace-a")
        await pilot.pause()
        assert store.active_session_id == temp.id
        assert chip.display is True
        assert "Temporary" in str(chip.render())

        # "create a new session for the workspace" branch: a brand-new
        # workspace with no existing session. `create_session` here never
        # passes `ephemeral=True`, so switching there from the ephemeral
        # session must also hide the chip.
        console._activate_console_session_for_workspace("workspace-c")
        await pilot.pause()
        assert store.active_session_id not in (normal.id, temp.id)
        assert chip.display is False


@pytest.mark.asyncio
async def test_character_picker_new_chat_clears_a_stale_temporary_chip():
    """task-7 review: the character picker's "new chat" placement creates
    and activates a session that is never ephemeral (``create_session`` is
    called with no ``ephemeral=`` there), but the method only awaits
    ``_sync_native_console_chat_ui()`` afterward, which never touches the
    temporary chip.

    Only one direction is tested: this specific call site cannot create an
    ephemeral session (its ``create_session`` call has no ``ephemeral=``
    argument at all), so there is no "chip becomes visible" case reachable
    from here to assert against.
    """
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
        ConsoleCharacterChoice,
    )
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleTemporaryChip,
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()

        temp = store.create_session(title="Temp", ephemeral=True)
        store.switch_session(temp.id)
        console._sync_console_temporary_chip()
        chip = console.query_one("#console-temporary-chip", ConsoleTemporaryChip)
        assert chip.display is True

        # Bypass the real character-card DB lookup (irrelevant to this
        # regression) while exercising the real create/switch/notify body.
        console._fetch_character_card_for_avatar = lambda character_id: {
            "name": "Nova",
            "first_message": "",
        }
        await console._apply_console_character_choice_async(
            ConsoleCharacterChoice(character_id=1, name="Nova", placement="new")
        )
        await pilot.pause()

        assert store.active_session_id != temp.id
        assert chip.display is False


@pytest.mark.asyncio
async def test_workspace_conversation_row_click_on_open_tab_keeps_temporary_chip_honest():
    """task-7 review: clicking an already-open tab's workspace-conversation
    row (the branch that skips ``_resume_console_workspace_conversation``
    because the session is already open) only awaited
    ``_sync_native_console_chat_ui()`` before this fix. Ephemeral (native,
    unpersisted) sessions appear in this same row list --
    ``_native_console_browser_rows`` has no ephemeral filter -- so both
    directions are genuinely reachable here: clicking an open ephemeral
    tab's row must show the chip, and clicking a saved/normal tab's row
    must hide it.
    """
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_console_native_chat_flow import (
        _click_console_workspace_conversation_for_session,
        _configure_native_ready_console,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleTemporaryChip,
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()

        normal = store.create_session(title="Normal chat")
        temp = store.create_session(title="Temp chat", ephemeral=True)
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        chip = console.query_one("#console-temporary-chip", ConsoleTemporaryChip)
        # `temp` is active (created last). Establish a verified-correct
        # baseline via the real sync path.
        assert store.active_session_id == temp.id
        console._sync_console_temporary_chip()
        assert chip.display is True

        await _click_console_workspace_conversation_for_session(
            console, pilot, store, normal.id
        )
        await pilot.pause(0.2)
        assert chip.display is False, (
            "switching onto the already-open saved tab must clear a stale "
            "Temporary chip"
        )

        await _click_console_workspace_conversation_for_session(
            console, pilot, store, temp.id
        )
        await pilot.pause(0.2)
        assert chip.display is True
        assert "Temporary" in str(chip.render())


@pytest.mark.unit
def test_temporary_chip_posts_save_requested_on_activation():
    """The chip is the save affordance: activating it posts ``SaveRequested``.

    Mirrors ``ConsoleApprovalsChip``'s own activation contract test -- Task 8
    wires the handler for this message.
    """
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleTemporaryChip,
    )

    chip = ConsoleTemporaryChip.__new__(ConsoleTemporaryChip)
    posted: list[object] = []
    chip.post_message = lambda message: posted.append(message)  # type: ignore[assignment]

    chip.action_save_chat()

    assert any(isinstance(m, ConsoleTemporaryChip.SaveRequested) for m in posted)


@pytest.mark.unit
def test_save_this_chat_appears_only_in_a_temporary_chat():
    """The entry is meaningless in a normal chat, so it is absent, not disabled.

    This is the one case where hiding beats disabling: a disabled "Save this
    chat" on an already-saved conversation would read as a failure.
    """
    from tldw_chatbook.Chat.console_ephemeral import ACTION_SAVE_CHAT

    normal = [e.action_id for e in build_composer_menu_entries()]
    assert ACTION_SAVE_CHAT not in normal

    temporary = build_composer_menu_entries(ephemeral=True)
    ids = [e.action_id for e in temporary]
    assert ids[0] == ACTION_SAVE_CHAT, "the escape hatch goes first"
    entry = temporary[0]
    assert entry.enabled is True
    assert "not saved" in entry.description.lower()


def _bare_promote_screen(store):
    """Build a ``ChatScreen`` stand-in wired to a fake Console store.

    Shared by the ``_promote_console_temporary_session`` tests below: each
    one only differs in what the fake store does, and in which of the
    captured lists it inspects afterwards.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    screen._console_chat_store = store
    screen._ensure_console_chat_store = lambda: store

    chip_calls: list[bool] = []
    screen._sync_console_temporary_chip = lambda: chip_calls.append(
        screen._console_active_session_is_ephemeral()
    )

    invalidated: list[bool] = []
    screen._invalidate_console_persisted_rows_cache = lambda: invalidated.append(True)

    dispatched: list[object] = []
    screen.run_worker = lambda coroutine, **_kwargs: dispatched.append(coroutine)

    notifications: list[tuple[str, str]] = []

    class _App:
        def notify(self, message, severity="information"):
            notifications.append((message, severity))

    screen.app_instance = _App()
    return screen, chip_calls, invalidated, dispatched, notifications


class _PromoteSession:
    def __init__(self, id_: str, ephemeral: bool) -> None:
        self.id = id_
        self.ephemeral = ephemeral


@pytest.mark.unit
def test_promote_console_temporary_session_saves_and_refreshes_the_chip():
    """The effect that matters: promotion ran and the chip actually clears.

    Guards the task-8 hazard directly: a handler that shows a toast and
    never reaches ``promote_ephemeral_session`` looks identical from the
    outside. This asserts the store call happened and the session came back
    non-temporary, not merely that a notification fired.
    """

    class _Store:
        def __init__(self) -> None:
            self.active_session_id = "s1"
            self._session = _PromoteSession("s1", ephemeral=True)
            self.promote_calls: list[str] = []

        def sessions(self):
            return [self._session]

        def promote_ephemeral_session(self, session_id):
            self.promote_calls.append(session_id)
            self._session.ephemeral = False
            return "conv-123"

    store = _Store()
    screen, chip_calls, invalidated, dispatched, notifications = (
        _bare_promote_screen(store)
    )

    screen._promote_console_temporary_session()

    assert store.promote_calls == ["s1"], "the store's promotion must actually run"
    assert store._session.ephemeral is False, "the session must come back non-temporary"
    assert chip_calls == [False], "the chip refresh must see the now-saved session"
    assert invalidated == [True]
    assert len(dispatched) == 1, "_sync_native_console_chat_ui must run as a worker"
    dispatched[0].close()
    assert notifications == [("Chat saved.", "information")]


@pytest.mark.unit
def test_promote_console_temporary_session_restores_temporary_state_on_failure():
    """A failing save must leave the chat temporary, not silently persisted."""

    class _Store:
        def __init__(self) -> None:
            self.active_session_id = "s1"
            self._session = _PromoteSession("s1", ephemeral=True)

        def sessions(self):
            return [self._session]

        def promote_ephemeral_session(self, session_id):
            # Mirrors promote_ephemeral_session's own contract: restore to
            # temporary before re-raising.
            self._session.ephemeral = True
            raise RuntimeError("db exploded")

    store = _Store()
    screen, chip_calls, invalidated, dispatched, notifications = (
        _bare_promote_screen(store)
    )

    screen._promote_console_temporary_session()

    assert store._session.ephemeral is True, "must stay temporary after a failed save"
    assert chip_calls == [], "a failed save must not tell the chip to disappear"
    assert invalidated == []
    assert dispatched == []
    assert notifications == [
        ("Could not save this chat. It is still temporary.", "error")
    ]


@pytest.mark.unit
def test_promote_console_temporary_session_is_silent_when_already_saved():
    """``promote_ephemeral_session`` returning ``None`` means nothing to do.

    Per its contract this covers an already-saved session or no configured
    adapter -- either way idempotent, so no toast and no chip churn.
    """

    class _Store:
        def __init__(self) -> None:
            self.active_session_id = "s1"

        def promote_ephemeral_session(self, session_id):
            return None

    store = _Store()
    screen, chip_calls, invalidated, dispatched, notifications = (
        _bare_promote_screen(store)
    )

    screen._promote_console_temporary_session()

    assert chip_calls == []
    assert invalidated == []
    assert dispatched == []
    assert notifications == []


@pytest.mark.unit
def test_save_chat_menu_choice_dispatches_to_the_promote_handler():
    """The composer-menu row for ``ACTION_SAVE_CHAT`` reaches the real save path."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    calls: list[bool] = []
    screen._promote_console_temporary_session = lambda: calls.append(True)

    screen._handle_console_composer_menu_choice(ACTION_SAVE_CHAT)

    assert calls == [True]


@pytest.mark.unit
def test_temporary_chip_save_requested_reaches_the_promote_handler():
    """The chip's activation message (task-7) drives the same save path."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleTemporaryChip,
    )

    screen = ChatScreen.__new__(ChatScreen)
    calls: list[bool] = []
    screen._promote_console_temporary_session = lambda: calls.append(True)

    event = ConsoleTemporaryChip.SaveRequested()
    stopped: list[bool] = []
    event.stop = lambda: stopped.append(True)

    screen.on_console_temporary_chip_save(event)

    assert stopped == [True], "the chip's own click/activation handling must not also fire"
    assert calls == [True]


@pytest.mark.unit
def test_artifact_actions_are_disabled_with_a_reason_in_a_temporary_chat():
    """Disabled and explained, never hidden -- and still enabled normally.

    The second half is the control: an assertion that an action is disabled
    proves nothing unless the same call proves it is enabled otherwise.
    """
    from tldw_chatbook.Chat.console_ephemeral import blocked_reason
    from tldw_chatbook.Widgets.Console.console_workbench_state import (
        build_console_workbench_state,
    )
    from tldw_chatbook.Chat.console_display_state import ConsoleControlState

    menu = {
        e.action_id: e
        for e in build_composer_menu_entries(ephemeral=True)
    }
    image = menu[ACTION_GENERATE_IMAGE]
    assert image.enabled is False
    assert image.description == blocked_reason("generate-image", ephemeral=True)

    normal = {e.action_id: e for e in build_composer_menu_entries()}
    assert normal[ACTION_GENERATE_IMAGE].enabled is True

    # ConsoleControlState has seven required label fields and no defaults.
    control_state = ConsoleControlState(
        provider_label="Provider: stub",
        model_label="Model: stub",
        assistant_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0",
        tools_label="Tools: 0",
        approvals_label="Approvals: 0",
    )

    def chatbook_action(**kwargs):
        state = build_console_workbench_state(
            control_state=control_state, can_save_chatbook=True, **kwargs
        )
        return {a.id: a for a in state.actions}["save-chatbook"]

    blocked = chatbook_action(ephemeral=True)
    assert blocked.disabled is True
    assert blocked.tooltip == blocked_reason("save-chatbook", ephemeral=True)
    assert chatbook_action().disabled is False
