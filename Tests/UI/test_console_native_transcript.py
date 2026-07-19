import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariantSet,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_message_actions import (
    ConsoleMessageActionService,
    ConsoleSaveDestination,
)
from tldw_chatbook.Chat.console_onboarding_state import ConsoleSetupCardState
from tldw_chatbook.Widgets.Console.console_save_as_modal import ConsoleSaveAsModal
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)


class TranscriptHarness(App):
    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello", id="m1"),
                ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m2"),
            ]
        )
        yield transcript


class EmptyTranscriptHarness(App):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


class MutableTranscriptHarness(App):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


class SaveAsModalHarness(App):
    def __init__(self, destinations: list[ConsoleSaveDestination] | None = None) -> None:
        super().__init__()
        self.destinations = save_as_modal_destinations() if destinations is None else destinations
        self.selected_destination: str | None = None

    def on_mount(self) -> None:
        self.push_screen(
            ConsoleSaveAsModal(destinations=self.destinations),
            self._capture_destination,
        )

    def _capture_destination(self, destination: str | None) -> None:
        self.selected_destination = destination


def save_as_modal_destinations() -> list[ConsoleSaveDestination]:
    return [
        ConsoleSaveDestination(label="Chatbook", available=True, reason=""),
        ConsoleSaveDestination(
            label="Note",
            available=False,
            reason="Notes service is not ready in this session.",
        ),
    ]


def test_console_transcript_renderable_uses_full_width_rules():
    transcript = ConsoleTranscript()
    transcript.set_messages(
        [
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello"),
            ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="world"),
        ]
    )

    plain = transcript.to_plain_text(width=40)

    assert "─" * 40 in plain
    assert "User" in plain
    assert "Assistant" in plain
    assert "hello" in plain
    assert "world" in plain


def test_console_transcript_widget_rules_are_long_enough_to_clip_full_width():
    transcript = ConsoleTranscript()
    transcript.set_messages(
        [
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello"),
        ]
    )

    first_rule = transcript._message_widgets()[0]
    renderable = getattr(first_rule, "renderable", "")

    assert len(str(renderable)) >= 160


@pytest.mark.asyncio
async def test_console_transcript_append_preserves_existing_message_rows():
    app = MutableTranscriptHarness()
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content=f"message {index}", id=f"m{index}")
        for index in range(12)
    ]

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        before_counts = transcript.row_build_counts()

        transcript.set_messages(
            messages
            + [ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="new answer", id="m-new")]
        )
        await transcript.refresh_messages()
        after_counts = transcript.row_build_counts()

    for message in messages:
        assert after_counts[f"message:{message.id}"] == before_counts[f"message:{message.id}"]
    assert after_counts["message:m-new"] == 1


@pytest.mark.asyncio
async def test_console_transcript_streaming_update_preserves_unrelated_message_rows():
    app = MutableTranscriptHarness()
    user = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="prompt", id="m-user")
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial",
        id="m-assistant",
        status="streaming",
    )
    trailing = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="next", id="m-next")

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([user, assistant, trailing])
        await transcript.refresh_messages()
        before_counts = transcript.row_build_counts()

        assistant.content = "partial response"
        transcript.set_messages([user, assistant, trailing])
        await transcript.refresh_messages()
        after_counts = transcript.row_build_counts()
        rendered = transcript.query_one("#console-message-m-assistant", Static)

    assert after_counts["message:m-user"] == before_counts["message:m-user"]
    assert after_counts["message:m-next"] == before_counts["message:m-next"]
    assert after_counts["message:m-assistant"] == before_counts["message:m-assistant"]
    assert "partial response" in str(rendered.renderable)


@pytest.mark.asyncio
async def test_console_transcript_selection_update_preserves_message_rows():
    app = MutableTranscriptHarness()
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="prompt", id="m-user"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m-assistant"),
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="followup", id="m-followup"),
    ]

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()

        transcript.selected_message_id = "m-user"
        await transcript.refresh_messages()
        before_counts = transcript.row_build_counts()

        transcript.selected_message_id = "m-assistant"
        await transcript.refresh_messages()
        after_counts = transcript.row_build_counts()

    for message in messages:
        assert after_counts[f"message:{message.id}"] == before_counts[f"message:{message.id}"]
    assert "actions:m-user" not in transcript.row_render_signatures()
    assert "actions:m-assistant" in transcript.row_render_signatures()


@pytest.mark.asyncio
async def test_console_transcript_removes_build_counts_for_stale_rows():
    app = MutableTranscriptHarness()
    removed = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="remove me", id="m-removed")
    kept = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="keep me", id="m-kept")

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([removed, kept])
        await transcript.refresh_messages()
        assert "message:m-removed" in transcript.row_build_counts()

        transcript.set_messages([kept])
        await transcript.refresh_messages()
        build_counts = transcript.row_build_counts()

    assert "rule:m-removed" not in build_counts
    assert "message:m-removed" not in build_counts
    assert "message:m-kept" in build_counts


def test_console_transcript_compose_resets_build_count_bookkeeping():
    transcript = ConsoleTranscript()
    transcript._row_build_counts["message:stale"] = 3
    transcript.set_messages(
        [
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER,
                content="current",
                id="m-current",
            )
        ]
    )

    list(transcript.compose())

    build_counts = transcript.row_build_counts()
    assert "message:stale" not in build_counts
    assert build_counts["message:m-current"] == 1


@pytest.mark.asyncio
async def test_console_transcript_empty_state_accepts_setup_copy():
    app = EmptyTranscriptHarness()

    async with app.run_test() as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)

        transcript.sync_empty_state(
            ConsoleSetupCardState(
                mode="ready_line",
                body_copy="Choose a model in Console Settings to start chatting.",
            )
        )
        await pilot.pause()

        empty_state = transcript.query_one(".console-transcript-empty-state", Static)
        empty_text = getattr(empty_state.renderable, "plain", str(empty_state.renderable))
        assert empty_text == "Choose a model in Console Settings to start chatting."


def test_console_transcript_selected_message_shows_action_row():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1")
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    plain = transcript.to_plain_text(width=80)

    assert "Copy Edit Save as... ♻ ---> 👍 👎 🗑" in plain
    assert "|" not in plain


def test_console_user_message_regenerate_action_is_disabled_and_blocks_dispatch():
    message = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="prompt", id="m1")
    service = ConsoleMessageActionService()

    regenerate = next(
        action
        for action in service.available_actions(message)
        if action.action_id == "regenerate"
    )
    result = service.dispatch("regenerate", message)

    assert not regenerate.enabled
    assert regenerate.disabled_reason == "Only assistant messages can be regenerated."
    assert result.status == "blocked"
    assert result.visible_copy == "Only assistant messages can be regenerated."


def test_console_transcript_selected_message_does_not_apply_inline_border_geometry():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1")
    widget = ConsoleTranscriptMessage(message, selected=True)

    assert widget.has_class("console-transcript-message-selected")
    assert "solid" not in repr(widget.styles.border)

    widget.sync_message(message, selected=False)

    assert not widget.has_class("console-transcript-message-selected")
    assert "solid" not in repr(widget.styles.border)

    widget.sync_message(message, selected=True)

    assert widget.has_class("console-transcript-message-selected")
    assert "solid" not in repr(widget.styles.border)


def test_console_transcript_action_row_stays_within_terminal_width_budget():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1")
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    action_row = next(
        line
        for line in transcript.to_plain_text(width=48).splitlines()
        if line.startswith("Copy")
    )

    assert action_row == "Copy Edit Save as... ♻ ---> 👍 👎 🗑"
    assert len(action_row) <= 48


def test_console_transcript_selected_message_explains_icon_actions():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1")
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    rendered = transcript.to_plain_text(width=80)

    assert "Copy Edit Save as... ♻ ---> 👍 👎 🗑" in rendered
    assert "Guide: ♻ Regenerate  ---> Continue  👍/👎 Rate  🗑 Delete" in rendered


def test_console_transcript_variant_navigation_changes_displayed_content():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="first", id="m1")
    message.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    assert "first" in transcript.to_plain_text(width=80)

    transcript.select_next_variant("m1")

    rendered = transcript.to_plain_text(width=80)
    assert "second" in rendered
    assert "first" not in rendered
    assert " < " in f" {rendered} "
    assert " > " in f" {rendered} "


def test_console_transcript_variant_action_row_stays_within_terminal_width_budget():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="first", id="m1")
    message.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    action_row = next(
        line
        for line in transcript.to_plain_text(width=48).splitlines()
        if line.startswith("Copy")
    )

    assert action_row == "Copy Edit Save as... < > ♻ ---> 👍 👎 🗑"
    assert len(action_row) <= 52


def test_console_transcript_failed_action_row_includes_retry_without_exceeding_budget():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="failed",
        id="m1",
        status="failed",
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    action_row = next(
        line
        for line in transcript.to_plain_text(width=48).splitlines()
        if line.startswith("Copy")
    )

    assert action_row == "Copy Edit Save as... Try ♻ ---> 👍 👎 🗑"
    assert len(action_row) <= 52


@pytest.mark.asyncio
async def test_console_transcript_keyboard_selects_messages_and_enter_shows_actions():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.press("tab")
        await pilot.press("down")
        await pilot.press("enter")
        text = _visible_text(app)

    assert "Copy" in text
    assert "Save as..." in text
    assert "♻" in text
    assert "👍" in text
    assert "👎" in text
    assert "🗑" in text
    assert "|" not in text


@pytest.mark.asyncio
async def test_console_transcript_click_selects_message_and_shows_actions():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        text = _visible_text(app)

    assert "Copy" in text
    assert "Save as..." in text
    assert "♻" in text
    assert "👍" in text
    assert "👎" in text
    assert "🗑" in text
    assert "Guide: ♻ Regenerate" in text
    assert "|" not in text


@pytest.mark.asyncio
async def test_console_selected_message_uses_class_without_inline_frame():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await _wait_for_selector(app, pilot, "#console-message-m2")
        selected = app.query_one("#console-message-m2")

    assert "console-transcript-message-selected" in selected.classes
    assert "solid" not in repr(selected.styles.border)


@pytest.mark.asyncio
async def test_console_transcript_action_buttons_have_stable_ids():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-copy-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-save-as-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-regenerate-m2")
        text = _visible_text(app)

    assert "Copy" in text
    assert "Save as..." in text
    assert "♻" in text
    assert "👍" in text
    assert "👎" in text
    assert "🗑" in text
    assert "|" not in text


@pytest.mark.asyncio
async def test_console_transcript_action_tooltips_explain_compact_labels():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-continue-m2")
        continue_action = app.query_one("#console-message-action-continue-m2")
        save_action = app.query_one("#console-message-action-save-as-m2")

    assert "extend" in str(continue_action.tooltip).lower()
    assert "destination" in str(save_action.tooltip).lower()


@pytest.mark.asyncio
async def test_console_transcript_escape_collapses_selected_action_row():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        assert "Save as..." in _visible_text(app)

        await pilot.press("escape")

        assert "Save as..." not in _visible_text(app)


@pytest.mark.asyncio
async def test_save_as_modal_lists_available_and_unavailable_destinations():
    app = SaveAsModalHarness()

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.1)
        await _wait_for_selector(app.screen, pilot, "#console-save-as-modal")
        text = _visible_text(app.screen)

    assert "Chatbook" in text
    assert "Note (unavailable)" in text
    assert "Notes service is not ready in this session." in text
    assert "WIP" not in text


@pytest.mark.asyncio
async def test_save_as_modal_available_destination_is_clickable_control():
    app = SaveAsModalHarness(
        destinations=[
            ConsoleSaveDestination(label="Chatbook", available=True, reason=""),
            ConsoleSaveDestination(
                label="Note",
                available=False,
                reason="Notes service is not ready in this session.",
            ),
        ]
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-save-as-destination-chatbook")
        destination_button = app.screen.query_one("#console-save-as-destination-chatbook", Button)
        text = _visible_text(app.screen)

        assert destination_button.disabled is False
        assert "Note (unavailable)" in text
        assert not app.screen.query("#console-save-as-destination-note")

        await pilot.click("#console-save-as-destination-chatbook")
        await pilot.pause(0.1)

    assert app.selected_destination == "Chatbook"


@pytest.mark.asyncio
async def test_save_as_modal_harness_preserves_empty_destination_list():
    app = SaveAsModalHarness(destinations=[])

    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-save-as-modal")
        text = _visible_text(app.screen)

    assert "No Save as destinations are wired for selected messages yet." in text
    assert "Chatbook" not in text
    assert "Note" not in text


@pytest.mark.asyncio
async def test_console_mounts_native_transcript_region():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        assert console.query_one("#console-native-transcript", ConsoleTranscript)


@pytest.mark.asyncio
async def test_console_tab_reaches_major_console_screen_regions():
    """Tab traversal reaches the major Console regions in the post-onboarding state.

    First-run focus is intentionally owned by the blocking setup modal (see
    ``ConsoleSetupModal.is_blocking``), which traps Tab until setup completes.
    That is by design and covered separately. This test marks onboarding
    complete (the same ``first_send_completed`` flag the app persists after a
    real first send) so the modal renders in its non-blocking "quiet" mode
    and Tab is free to reach the workbench regions during normal use.
    """
    app = _build_test_app()
    app.app_config.setdefault("console", {})["onboarding"] = {"first_send_completed": True}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-staged-context-tray")
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_selector(console, pilot, "#console-inspector-rail-open")

        seen_focus_ids = set()
        for _ in range(64):
            focused = getattr(console.app, "focused", None)
            if focused is not None and getattr(focused, "id", None):
                seen_focus_ids.add(focused.id)
            await pilot.press("tab")

    assert "console-native-transcript" in seen_focus_ids
    assert "console-native-composer" in seen_focus_ids
    assert "console-inspector-rail-open" in seen_focus_ids


def test_console_streaming_assistant_row_shows_generating_placeholder_until_first_token():
    """Between send-accepted and first token the assistant row must not be empty."""
    from tldw_chatbook.Widgets.Console.console_transcript import (
        CONSOLE_GENERATING_PLACEHOLDER,
        _message_body,
        _message_render_text,
    )

    pending = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        id="m-generating",
        status="streaming",
    )
    assert _message_body(pending) == CONSOLE_GENERATING_PLACEHOLDER
    rendered = _message_render_text(pending, selected=False)
    assert CONSOLE_GENERATING_PLACEHOLDER in rendered.plain

    # First streamed token replaces the placeholder immediately.
    started = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Once",
        id="m-generating",
        status="streaming",
    )
    assert _message_body(started) == "Once [streaming]"

    # Other terminal statuses keep their existing suffix rendering.
    failed = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        id="m-failed",
        status="failed",
    )
    assert _message_body(failed) == "[failed]"


def test_image_message_row_renders_chip_line():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="what is this?",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        attachment_label="photo.png · 11 B",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼 photo.png · 11 B" in rendered.plain


def test_image_only_message_row_renders_chip_without_body():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼" in rendered.plain


def test_save_image_action_only_offered_for_image_messages():
    service = ConsoleMessageActionService()
    plain = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="text")
    with_image = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    plain_ids = [action.action_id for action in service.available_actions(plain)]
    image_ids = [action.action_id for action in service.available_actions(with_image)]
    assert "save-image" not in plain_ids
    assert "save-image" in image_ids

    result = service.dispatch("save-image", with_image)
    assert result.status == "completed"
    assert result.visible_copy == "Saving image to disk."


def test_image_chip_falls_back_to_mime_and_size_without_label():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="",
        image_data=b"x" * 2048,
        image_mime_type="image/png",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼 image/png · 2 KB" in rendered.plain


def test_image_chip_metadata_only_keeps_bare_mime():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_mime_type="image/png",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼 image/png" in rendered.plain


def test_multi_attachment_message_renders_chip_per_attachment():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="three pics",
    )
    message.attachments = (
        MessageAttachment(data=b"1", mime_type="image/png", display_name="a.png", position=0),
        MessageAttachment(data=b"22", mime_type="image/jpeg", display_name="b.jpg", position=1),
        MessageAttachment(data=None, mime_type="image/png", display_name="", position=2),
    )
    message.image_data = b"1"
    message.image_mime_type = "image/png"
    message.attachment_label = "a.png"

    rendered = _message_render_text(message, selected=False)
    plain = rendered.plain
    assert "🖼 a.png" in plain
    assert "🖼 b.jpg" in plain
    assert plain.count("🖼") == 3  # dataless third falls back to mime label


def _image_row_spec(message_id: str, mode: str = "pixels"):
    from PIL import Image as PILImage
    from rich_pixels import Pixels

    from tldw_chatbook.Chat.console_image_view import ConsoleImageRowSpec

    pil = PILImage.new("RGB", (16, 16), (10, 120, 40))
    return ConsoleImageRowSpec(
        message_id=message_id,
        mode=mode,
        pixels=Pixels.from_image(pil) if mode == "pixels" else None,
        pil=pil if mode == "graphics" else None,
    )


def test_transcript_emits_image_row_when_spec_present():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])
    transcript.set_image_specs({message.id: _image_row_spec(message.id)})

    rows = transcript._transcript_rows()
    kinds = [row.kind for row in rows]
    assert "image" in kinds
    image_row = next(row for row in rows if row.kind == "image")
    assert image_row.key == f"image:{message.id}"
    assert image_row.signature == ("image", message.id, "pixels")
    # Order: message row immediately precedes its image row.
    message_index = kinds.index("message")
    assert kinds[message_index + 1] == "image"


def test_transcript_omits_image_row_without_spec_or_when_hidden():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])
    # No specs set at all -> no image rows (unmounted-test posture).
    assert all(row.kind != "image" for row in transcript._transcript_rows())
    # Hidden mode is expressed by the screen simply omitting the spec.
    transcript.set_image_specs({})
    assert all(row.kind != "image" for row in transcript._transcript_rows())


def test_image_row_signature_stable_across_streaming_ticks():
    transcript = ConsoleTranscript()
    user = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="", status="streaming"
    )
    transcript.set_messages([user, assistant])
    transcript.set_image_specs({user.id: _image_row_spec(user.id)})

    first = next(r for r in transcript._transcript_rows() if r.kind == "image")
    assistant.content = "more streamed text"
    transcript.set_messages([user, assistant])
    second = next(r for r in transcript._transcript_rows() if r.kind == "image")
    assert first.signature == second.signature


def test_image_row_widget_builds_for_both_modes():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])

    transcript.set_image_specs({message.id: _image_row_spec(message.id, "pixels")})
    pixels_row = next(r for r in transcript._transcript_rows() if r.kind == "image")
    pixels_widget = transcript._build_row_widget(pixels_row, track=False)
    assert pixels_widget.id == f"console-image-{message.id}"

    transcript.set_image_specs({message.id: _image_row_spec(message.id, "graphics")})
    graphics_row = next(r for r in transcript._transcript_rows() if r.kind == "image")
    graphics_widget = transcript._build_row_widget(graphics_row, track=False)
    assert graphics_widget.id == f"console-image-{message.id}"
    assert graphics_widget.styles.max_width.value == 80
    assert graphics_widget.styles.max_height.value == 40


def test_image_row_rebuild_tracked_on_mode_change():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])
    transcript.set_image_specs({message.id: _image_row_spec(message.id, "pixels")})
    rows = transcript._transcript_rows()
    image_row = next(r for r in rows if r.kind == "image")
    widget = transcript._build_row_widget(image_row, track=True)
    assert transcript.row_build_counts()[f"image:{message.id}"] == 1

    transcript.set_image_specs({message.id: _image_row_spec(message.id, "graphics")})
    new_row = next(r for r in transcript._transcript_rows() if r.kind == "image")
    assert new_row.signature != image_row.signature
    updated = transcript._update_row_widget(widget, new_row)
    assert updated is not widget
    assert transcript.row_build_counts()[f"image:{message.id}"] == 2


def test_toggle_image_view_action_offered_and_dispatched_for_image_messages():
    service = ConsoleMessageActionService()
    plain = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="text")
    with_image = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    plain_ids = [action.action_id for action in service.available_actions(plain)]
    image_ids = [action.action_id for action in service.available_actions(with_image)]
    assert "toggle-image-view" not in plain_ids
    assert "toggle-image-view" in image_ids
    assert image_ids.index("toggle-image-view") < image_ids.index("save-image")

    result = service.dispatch("toggle-image-view", with_image)
    assert result.status == "completed"
    assert result.visible_copy == "Toggled image view."
    assert result.target_message_id == with_image.id


# ---------------------------------------------------------------------------
# TASK-259: per-message row-signature cache. Derivation of the expensive row
# render payload must be O(changed messages) per changed tick, with
# correctness preserved for delete, reorder, and variant-switch.
# ---------------------------------------------------------------------------


def _cache_test_messages(count: int) -> list[ConsoleChatMessage]:
    return [
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER if index % 2 == 0 else ConsoleMessageRole.ASSISTANT,
            content=f"message body {index}",
            id=f"m{index}",
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_transcript_signature_derivation_is_o_changed_messages():
    """A changed tick re-derives only the changed message's render signature."""
    app = MutableTranscriptHarness()
    messages = _cache_test_messages(10)

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        baseline = transcript.message_signature_compute_counts()
        assert all(count == 1 for count in baseline.values())

        # Unchanged tick (same objects re-set, as the 0.2s sync does).
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        assert transcript.message_signature_compute_counts() == baseline

        # Changed tick: exactly one message's content changed.
        messages[4].content = "message body 4 (edited)"
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        after = transcript.message_signature_compute_counts()
        rendered = transcript.query_one("#console-message-m4", Static)
        assert "message body 4 (edited)" in str(rendered.renderable)

    assert after["m4"] == baseline["m4"] + 1
    for message in messages:
        if message.id == "m4":
            continue
        assert after[message.id] == baseline[message.id]


@pytest.mark.asyncio
async def test_transcript_signature_cache_miss_on_equal_length_edit():
    """The cache must key on content, never on content length alone."""
    app = MutableTranscriptHarness()
    message = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="aaaa", id="m-edit")

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([message])
        await transcript.refresh_messages()

        message.content = "bbbb"  # same length, same status
        transcript.set_messages([message])
        await transcript.refresh_messages()
        rendered = transcript.query_one("#console-message-m-edit", Static)

        assert "bbbb" in str(rendered.renderable)
        assert transcript.message_signature_compute_counts()["m-edit"] == 2


@pytest.mark.asyncio
async def test_transcript_signature_cache_survives_delete():
    """Deleting a message prunes its cache entry and its mounted row."""
    app = MutableTranscriptHarness()
    messages = _cache_test_messages(5)

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        assert "m2" in transcript.message_signature_cache_ids()

        survivors = [message for message in messages if message.id != "m2"]
        transcript.set_messages(survivors)
        await transcript.refresh_messages()

        assert "m2" not in transcript.message_signature_cache_ids()
        assert len(transcript.query("#console-message-m2")) == 0
        # Survivors were not re-derived by the delete.
        counts = transcript.message_signature_compute_counts()
        assert all(counts[message.id] == 1 for message in survivors)

        # Re-adding the same id with different content renders the new text.
        replacement = ConsoleChatMessage(
            role=ConsoleMessageRole.USER, content="replacement body", id="m2"
        )
        transcript.set_messages(survivors + [replacement])
        await transcript.refresh_messages()
        rendered = transcript.query_one("#console-message-m2", Static)
        assert "replacement body" in str(rendered.renderable)


@pytest.mark.asyncio
async def test_transcript_signature_cache_survives_reorder():
    """Reordering messages re-derives nothing and renders the new order."""
    app = MutableTranscriptHarness()
    messages = _cache_test_messages(4)

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        baseline = transcript.message_signature_compute_counts()

        reordered = [messages[2], messages[0], messages[3], messages[1]]
        transcript.set_messages(reordered)
        await transcript.refresh_messages()

        assert transcript.message_signature_compute_counts() == baseline
        rendered_ids = [
            widget.message_id
            for widget in transcript.query(ConsoleTranscriptMessage)
        ]

    assert rendered_ids == ["m2", "m0", "m3", "m1"]


@pytest.mark.asyncio
async def test_transcript_signature_cache_survives_variant_switch():
    """Switching variants re-derives only that message and shows the variant."""
    app = MutableTranscriptHarness()
    plain = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="prompt", id="m-plain")
    varied = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="first answer", id="m-varied"
    )
    varied.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first answer", "second answer"],
    )

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([plain, varied])
        await transcript.refresh_messages()
        baseline = transcript.message_signature_compute_counts()
        rendered = transcript.query_one("#console-message-m-varied", Static)
        assert "first answer" in str(rendered.renderable)

        varied.variants.selected_index = 1
        transcript.set_messages([plain, varied])
        await transcript.refresh_messages()
        after = transcript.message_signature_compute_counts()
        rendered = transcript.query_one("#console-message-m-varied", Static)

        assert "second answer" in str(rendered.renderable)
        assert after["m-varied"] == baseline["m-varied"] + 1
        assert after["m-plain"] == baseline["m-plain"]


@pytest.mark.asyncio
async def test_transcript_signature_cache_miss_on_status_change():
    """A status-only change (streaming -> complete) re-derives the row."""
    app = MutableTranscriptHarness()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial",
        id="m-stream",
        status="streaming",
    )

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([message])
        await transcript.refresh_messages()
        rendered = transcript.query_one("#console-message-m-stream", Static)
        assert "[streaming]" in str(rendered.renderable)

        message.status = "complete"
        transcript.set_messages([message])
        await transcript.refresh_messages()
        rendered = transcript.query_one("#console-message-m-stream", Static)

        assert "[streaming]" not in str(rendered.renderable)
        assert transcript.message_signature_compute_counts()["m-stream"] == 2
