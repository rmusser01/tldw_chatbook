"""TASK-33620.2: a staged image the model can't take is its own blocker.

On dev f9baac8706 the screen passed the vision refusal ("Console send
blocked: <model> can't accept images. Remove the attachment, ...") through
the composer's ``setup_blocked_reason`` slot. ``build_console_disabled_reason``
then matched the word "model" in it and painted "Send blocked — choose a
model to continue ›", and the setup-blocked class turned that strip into an
``app.run_setup_wizard`` link: one click sent a set-up user into the
first-run wizard.

Every test here drives the real ChatScreen, store and composer. The model is
the harness's llama.cpp ``local-model``, which the real capability tables do
not list, so the real classifier refuses the image: nothing is stubbed.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

# The harness mounts the real ChatScreen, controller and store.
pytestmark = pytest.mark.bootstrap_profile

MODEL = "local-model"
NOTICE = f"{MODEL} can't accept images"
STRIP_REASON = f"Send blocked — {NOTICE}"


def _image() -> PendingAttachment:
    return PendingAttachment(
        file_path="/tmp/leaf.png",
        display_name="leaf.png",
        file_type="image",
        insert_mode="attachment",
        data=b"\x89PNG-bytes",
        mime_type="image/png",
        original_size=116,
        processed_size=116,
    )


async def _stage_image(console, pilot, draft: str = ""):
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    store = console._ensure_console_chat_store()
    session_id = store.ensure_session().id
    composer.load_draft(draft)
    assert store.add_pending_attachment(session_id, _image())
    console._sync_console_control_bar()
    await pilot.pause()
    return composer, store, session_id


def _row_text(app, widget) -> str:
    """The text the compositor paints across ``widget``'s own region."""
    region = widget.region
    assert region.width > 0 and region.height > 0, widget
    screen = app.screen
    assert screen.region.contains_region(region), (widget, region, screen.region)
    hit, _ = app.get_widget_at(region.x, region.y)
    assert hit is widget or widget in hit.ancestors_with_self, (hit, widget)
    strips = screen._compositor.render_strips()
    return "".join(
        strip.crop(region.x, region.right).text
        for strip in strips[region.y : region.bottom]
    )


@pytest.mark.asyncio
async def test_a_vision_block_names_the_model_and_never_links_to_setup():
    _app, host = _ready_host()
    async with host.run_test(size=(235, 52)) as pilot:
        console = await _mounted_console(host, pilot)
        composer, _store, _session = await _stage_image(console, pilot)

        # The real classifier's verbatim refusal is what the screen holds.
        verbatim = console._console_attachment_blocked_reason()
        assert verbatim.startswith(
            f"Console send blocked: {MODEL} can't accept images."
        )

        strip = composer.query_one("#console-send-disabled-reason", Static)
        assert strip.content.plain == STRIP_REASON
        assert "choose a model" not in strip.content.plain
        assert "run_setup_wizard" not in strip.content.markup
        assert not composer.has_class("console-composer-setup-blocked")
        send = composer.query_one("#console-send-message", Button)
        assert send.disabled
        assert send.tooltip == verbatim

        block = console.query_one("#console-attachment-block")
        assert block.display
        copy = block.query_one("#console-attachment-block-copy", Static)
        assert copy.content.plain == NOTICE
        labels = [str(b.label) for b in block.query(Button)]
        assert labels == ["Remove image", "Switch model"]
        # No control in the row is the wizard either. (A click is no check
        # here: the harness app has no `run_setup_wizard` action, so the
        # pre-fix link was inert under test; its markup above is the check --
        # on dev it read "[@click=app.run_setup_wizard]Send blocked — choose
        # a model to continue ›[/...]".)
        switch = block.query_one("#console-attachment-block-switch-model", Button)
        assert switch.action == "screen.open_console_model_popover"
        assert (
            block.query_one("#console-attachment-block-remove", Button).action is None
        )


@pytest.mark.asyncio
async def test_switch_model_opens_the_console_model_picker_and_keeps_draft_and_image():
    from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover

    _app, host = _ready_host()
    async with host.run_test(size=(235, 52)) as pilot:
        console = await _mounted_console(host, pilot)
        composer, store, session_id = await _stage_image(
            console, pilot, draft="what colour is this leaf?"
        )

        await pilot.click("#console-attachment-block-switch-model")
        await pilot.pause()

        assert isinstance(host.screen, ConsoleModelPopover), host.screen_stack
        assert composer.draft_text() == "what colour is this leaf?"
        assert [p.display_name for p in store.pending_attachments(session_id)] == [
            "leaf.png"
        ]


@pytest.mark.asyncio
async def test_remove_image_clears_the_image_and_the_block_and_keeps_the_draft():
    _app, host = _ready_host()
    async with host.run_test(size=(235, 52)) as pilot:
        console = await _mounted_console(host, pilot)
        composer, store, session_id = await _stage_image(
            console, pilot, draft="never mind the picture"
        )

        await pilot.click("#console-attachment-block-remove")
        await pilot.pause()

        assert store.pending_attachments(session_id) == []
        assert not console.query_one("#console-attachment-block").display
        assert composer.draft_text() == "never mind the picture"
        assert not composer.query_one("#console-send-message", Button).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (235, 52)], ids=["80x24", "235x52"])
async def test_the_chip_the_block_copy_and_both_actions_are_painted_whole(size):
    _app, host = _ready_host()
    async with host.run_test(size=size) as pilot:
        console = await _mounted_console(host, pilot)
        composer, _store, _session = await _stage_image(console, pilot)
        await pilot.pause()

        chip = composer.query_one("#console-attachment-indicator", Static)
        assert "leaf.png · 116 B" in _row_text(host, chip)
        block = console.query_one("#console-attachment-block")
        copy = block.query_one("#console-attachment-block-copy", Static)
        assert _row_text(host, copy).strip() == NOTICE
        for button_id, label in (
            ("#console-attachment-block-remove", "Remove image"),
            ("#console-attachment-block-switch-model", "Switch model"),
        ):
            assert label in _row_text(host, block.query_one(button_id, Button))


@pytest.mark.asyncio
async def test_both_actions_are_a_shift_tab_away_and_enter_opens_the_picker():
    from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover

    _app, host = _ready_host()
    async with host.run_test(size=(80, 24)) as pilot:
        console = await _mounted_console(host, pilot)
        composer, store, session_id = await _stage_image(console, pilot)
        composer.focus()
        await pilot.pause()

        reached = []
        for _ in range(4):
            await pilot.press("shift+tab")
            await pilot.pause()
            reached.append(host.focused.id)
            if host.focused.id == "console-attachment-block-remove":
                break
        assert reached[-2:] == [
            "console-attachment-block-switch-model",
            "console-attachment-block-remove",
        ], reached

        await pilot.press("tab")
        await pilot.pause()
        assert host.focused.id == "console-attachment-block-switch-model"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(host.screen, ConsoleModelPopover), host.screen_stack
        assert len(store.pending_attachments(session_id)) == 1


@pytest.mark.asyncio
async def test_a_switch_that_clears_the_block_leaves_the_keyboard_in_the_draft(
    monkeypatch,
):
    """Live 2026-10-29: after Switch model fixed the block, typed keys vanished.

    The picker's dismissal restores focus to its opener, the Switch model
    button, which is still shown at that moment, so it calls
    ``opener.focus()`` -- and Textual runs that later (``call_later``). The
    model change's sync then hides the row and hands focus to the draft,
    and the deferred focus lands on the hidden button after it: every key
    went nowhere. Instrumented, two of three harness runs took that order.
    This replays it in one tick, through the real dismissal restore.
    """
    import weakref

    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Widgets.modal_dismissal import _restore_focus_after_dismissal

    capable = {"on": False}
    monkeypatch.setattr(
        attachment_core, "is_vision_capable", lambda _p, _m: capable["on"]
    )
    _app, host = _ready_host()
    async with host.run_test(size=(235, 52)) as pilot:
        console = await _mounted_console(host, pilot)
        composer, store, session_id = await _stage_image(
            console, pilot, draft="describe it"
        )
        switch = console.query_one("#console-attachment-block-switch-model", Button)
        switch.focus()
        await pilot.pause()
        assert host.focused is switch

        capable["on"] = True  # the picked model reads images
        _restore_focus_after_dismissal(  # the picker's dismissal: row shown
            host, console, weakref.ref(switch), switch.id
        )
        assert console._sync_console_control_bar() is True  # then the sync
        await pilot.pause()

        assert not console.query_one("#console-attachment-block").display
        assert host.focused is composer, host.focused
        await pilot.press("Z")
        await pilot.pause()
        assert composer.draft_text() == "describe itZ"
        assert len(store.pending_attachments(session_id)) == 1


@pytest.mark.asyncio
async def test_enter_on_remove_image_hands_the_keyboard_back_to_the_composer():
    _app, host = _ready_host()
    async with host.run_test(size=(80, 24)) as pilot:
        console = await _mounted_console(host, pilot)
        composer, store, session_id = await _stage_image(
            console, pilot, draft="keep typing after this"
        )
        console.query_one("#console-attachment-block-remove", Button).focus()
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

        assert store.pending_attachments(session_id) == []
        assert host.focused is composer, host.focused
        assert composer.draft_text() == "keep typing after this"


@pytest.mark.asyncio
async def test_with_no_model_chosen_setup_owns_the_strip_and_the_row_stays_hidden(
    monkeypatch,
):
    """No model chosen is the setup blocker's to say, not the vision row's.

    The row used to read "The selected model can't accept images" beneath a
    strip that says "choose a model": a claim about a model nobody picked.
    The image still holds Send; once a model is chosen the row names it.
    """
    from dataclasses import replace

    _app, host = _ready_host()
    async with host.run_test(size=(235, 52)) as pilot:
        console = await _mounted_console(host, pilot)
        selection = console._build_console_provider_selection
        model_chosen = {"on": False}

        def no_model_until_chosen():
            # A provider entry with no model configured (a fresh llama.cpp).
            chosen = selection()
            if model_chosen["on"]:
                return chosen
            return replace(chosen, explicit_model=None, configured_model=None)

        monkeypatch.setattr(
            console, "_build_console_provider_selection", no_model_until_chosen
        )
        composer, store, session_id = await _stage_image(console, pilot)

        assert console._console_setup_blocked_reason()
        strip = composer.query_one("#console-send-disabled-reason", Static)
        assert strip.content.plain == "Send blocked — choose a model to continue ›"
        assert composer.query_one("#console-send-message", Button).disabled
        block = console.query_one("#console-attachment-block")
        assert not block.display
        assert block.disabled  # hidden, so no focus can land on it
        assert len(store.pending_attachments(session_id)) == 1

        model_chosen["on"] = True
        console._sync_console_control_bar()
        await pilot.pause()

        assert block.display
        copy = block.query_one("#console-attachment-block-copy", Static)
        assert copy.content.plain == NOTICE
        assert strip.content.plain == STRIP_REASON


@pytest.mark.asyncio
async def test_a_long_model_id_is_ellipsized_and_both_actions_stay_whole_at_80x24(
    monkeypatch,
):
    """AC#5 for an id too long for the row: the copy gives, the actions don't.

    At 80 columns the copy has about 48 cells beside the two buttons. A long
    local id is cut with an ellipsis; its tooltip keeps the full refusal.
    """
    from dataclasses import replace

    long_model = "qwen2.5-coder-32b-instruct-q4_k_m.gguf"
    _app, host = _ready_host()
    async with host.run_test(size=(80, 24)) as pilot:
        console = await _mounted_console(host, pilot)
        selection = console._build_console_provider_selection
        monkeypatch.setattr(
            console,
            "_build_console_provider_selection",
            lambda: replace(selection(), explicit_model=long_model),
        )
        _composer, _store, _session = await _stage_image(console, pilot)
        await pilot.pause()

        block = console.query_one("#console-attachment-block")
        copy = block.query_one("#console-attachment-block-copy", Static)
        assert copy.content.plain == f"{long_model} can't accept images"
        painted = _row_text(host, copy).strip()
        assert painted.startswith(long_model[:20]) and painted.endswith("…"), painted
        assert copy.tooltip.startswith(
            f"Console send blocked: {long_model} can't accept images."
        )
        for button_id, label in (
            ("#console-attachment-block-remove", "Remove image"),
            ("#console-attachment-block-switch-model", "Switch model"),
        ):
            assert label in _row_text(host, block.query_one(button_id, Button))
