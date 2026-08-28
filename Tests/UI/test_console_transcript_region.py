"""Characterisation of the Console transcript region, written BEFORE its extraction.

Wave-3 console decomposition, task 2. Every assertion here was run green
against ``ChatScreen`` as it stood at ``4348d5b1c`` -- i.e. with the
transcript block still composed inline inside ``compose_content`` and the
four transcript-DOM methods still defined on the screen -- and committed
separately, before a single production line moved. That ordering is the
point: a characterisation written after the move can only prove the tests
still pass, never that the behaviour they describe is the behaviour that
existed beforehand.

Two things are pinned, matching the two ways this extraction could go
wrong:

1. **The composed block** -- the ``#console-main-column`` >
   ``#console-transcript-region`` > ``#console-session-surface`` >
   ``#console-native-transcript`` chain, the inline sizing the workspace
   grid depends on, and the then-deliberately top-less frame on
   ``#console-transcript-region`` (``_frame_console_region(..., top=False)``),
   which made the transcript read as continuous with the control bar above it.
   This is the historical pre-extraction shape, not the current borderless
   ``edges=()`` contract. Ids and nesting are the extraction's stated contract,
   so they are asserted as a parent/child chain rather than by bare presence:
   a region widget that mounted the right ids in the wrong place would pass
   a presence check.

2. **The four transcript-DOM methods** that move into the region --
   ``_capture_console_transcript_reading_state``,
   ``_restore_console_transcript_reading_state``,
   ``_clear_native_console_message_selection`` and
   ``_note_console_follow_intent`` -- each driven against a REAL mounted
   transcript rendering REAL store-persisted rows (``store.append_message``
   -> ``_sync_native_console_chat_ui`` -> the widget), never a stub. They
   are called here through their SCREEN-level names on purpose: post-move
   those names are delegations, so this file is the direct evidence that
   the delegation table works.
"""

import inspect
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console import ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleMarkdownMessage
from tldw_chatbook.UI.Console_Modules.frame import frame_console_region
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    ProviderContinuationTranscriptRegion,
)
from tldw_chatbook.UI.Console_Modules.transcript import ConsoleTranscriptRegion


class _SpeechHeaderHarness(App):
    """Mount completed assistant rows without selecting either one."""

    CSS_PATH = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self) -> None:
        super().__init__()
        self.messages = [
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="First answer.",
                status="complete",
                id="speech-a",
            ),
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="Second answer.",
                status="complete",
                id="speech-b",
            ),
        ]

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")

    def on_mount(self) -> None:
        transcript = self.query_one(ConsoleTranscript)
        transcript.set_messages(self.messages)
        self.call_later(transcript.refresh_messages)


class _TranscriptRegionHarness(App):
    def __init__(self, region_type: type[ConsoleTranscriptRegion]) -> None:
        super().__init__()
        self.region_type = region_type

    def compose(self) -> ComposeResult:
        kwargs = {
            "session_surface_builder": lambda: Static("Visible transcript"),
        }
        if self.region_type is ProviderContinuationTranscriptRegion:
            kwargs.update(
                recovery_message_builder=lambda: None,
                on_recovery_action=lambda *_args: True,
            )
        yield self.region_type(**kwargs)


def _speech_status(transcript: ConsoleTranscript, message_id: str) -> str:
    try:
        label = transcript.query_one(
            f"#console-message-speech-status-{message_id}", Static
        )
    except NoMatches:
        return ""  # no speech presentation mounted = no status
    return str(label.renderable)


class _ProductionStyledConsoleHarness(ConsoleHarness):
    """Mount the real Console with the application stylesheet tier."""

    CSS_PATH = str(BUNDLED_STYLESHEET)


def _ready_console_host() -> _ProductionStyledConsoleHarness:
    """Build a Console harness whose provider readiness is already satisfied."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    return _ProductionStyledConsoleHarness(app)


def _tail_is_anchored(transcript: ConsoleTranscript) -> bool:
    """Return Textual's semantic tail-follow state, including manual release.

    ``ScrollView.is_anchored`` stays True after ``release_anchor()`` -- the
    release is recorded separately on ``_anchor_released``. This is the same
    pair ``_capture_console_transcript_reading_state`` itself reads, so the
    assertions below check exactly the state the captured snapshot means
    (and ``test_console_composer_collapse`` carries an identical helper for
    the same reason).
    """
    return bool(
        transcript.is_anchored and not getattr(transcript, "_anchor_released", False)
    )


async def _mounted_console(host: ConsoleHarness, pilot):
    """Return the mounted Console screen once its composer exists."""
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    return console


async def _seed_transcript(console, pilot):
    """Persist enough rows to make the mounted transcript scrollable.

    Mirrors ``test_console_composer_collapse._seed_overflowing_transcript``:
    rows go into the real ``ConsoleChatStore`` and reach the widget through
    the production sync path, so what the assertions below read back is the
    transcript's response to persisted state, not a hand-set attribute.

    The condition wait is load-bearing: ``max_scroll_y`` is derived from the
    laid-out virtual size, so the rows must be measured before the anchor
    assertions can distinguish the transcript tail from position zero.

    Args:
        console: The mounted ``ChatScreen``.
        pilot: The Textual pilot driving that screen.

    Returns:
        A ``(transcript, selected_message_id)`` pair: the mounted
        ``ConsoleTranscript`` and the id of the last row appended, already
        selected on it.
    """
    store = console._ensure_console_chat_store()
    selected_message_id = ""
    for index in range(24):
        message = store.append_message(
            store.active_session_id,
            role=(
                ConsoleMessageRole.USER
                if index % 2 == 0
                else ConsoleMessageRole.ASSISTANT
            ),
            content="\n".join(f"message {index} line {line}" for line in range(3)),
        )
        selected_message_id = message.id
    await console._sync_native_console_chat_ui()
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    for _ in range(50):
        if transcript.max_scroll_y > 0:
            break
        await pilot.pause(0.02)
    assert transcript.max_scroll_y > 0
    transcript.select_message(selected_message_id)
    return transcript, selected_message_id


def test_frame_helper_keeps_legacy_side_edges_when_top_and_bottom_are_false():
    region = frame_console_region(Vertical(), top=False, bottom=False)

    assert region.styles.border_top[0] in {"", "none"}
    assert region.styles.border_right[0] == "solid"
    assert region.styles.border_bottom[0] in {"", "none"}
    assert region.styles.border_left[0] == "solid"


@pytest.mark.parametrize(
    "region_type",
    [ConsoleTranscriptRegion, ProviderContinuationTranscriptRegion],
)
def test_both_transcript_compose_paths_request_explicitly_borderless_edges(
    region_type: type[ConsoleTranscriptRegion],
):
    source = inspect.getsource(region_type.compose)

    assert "edges=()," in source
    assert "top=False" not in source
    assert "bottom=False" not in source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "region_type",
    [ConsoleTranscriptRegion, ProviderContinuationTranscriptRegion],
)
async def test_both_transcript_compose_paths_are_borderless_at_runtime(
    region_type: type[ConsoleTranscriptRegion],
):
    app = _TranscriptRegionHarness(region_type)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        border = app.screen.query_one("#console-transcript-region").styles.border

        assert all(edge[0] in {"", "none"} for edge in border)


@pytest.mark.asyncio
async def test_transcript_region_composes_its_ids_in_the_documented_nesting():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)

        main_column = console.query_one("#console-main-column")
        transcript_region = console.query_one("#console-transcript-region")
        session_surface = console.query_one("#console-session-surface")
        transcript = console.query_one("#console-native-transcript")

        # The chain, link by link -- presence alone would not catch a
        # region mounted as a sibling instead of a parent.
        assert main_column.parent is console.query_one("#console-workspace-grid")
        assert transcript_region.parent is main_column
        assert session_surface.parent is transcript_region
        assert transcript_region in main_column.children
        assert session_surface in transcript_region.children
        assert transcript in session_surface.query(ConsoleTranscript).nodes

        # `console-region` is what the shell's own styling keys off.
        assert transcript_region.has_class("console-region")


@pytest.mark.asyncio
async def test_transcript_region_keeps_inline_sizing_without_owning_frame_edges():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)

        main_column = console.query_one("#console-main-column")
        transcript_region = console.query_one("#console-transcript-region")
        left_rail = console.query_one("#console-left-rail")

        # Inline sizing set at compose time (the stylesheet carries
        # different values, so these prove the inline assignments survived).
        assert main_column.styles.width is not None
        assert main_column.styles.width.value == 13
        assert main_column.styles.min_width is not None
        assert main_column.styles.min_width.value == 56
        assert main_column.styles.min_height is not None
        assert main_column.styles.min_height.value == 0

        # The main column really is the dominant pane at this width.
        assert main_column.region.width > left_rail.region.width

        # TASK-20937.3: the transcript keeps its sizing role but owns no
        # frame edge; the grid and rails own the surrounding separators.
        border = transcript_region.styles.border
        assert border.top[0] in {"", "none"}
        assert border.right[0] in {"", "none"}
        assert border.bottom[0] in {"", "none"}
        assert border.left[0] in {"", "none"}
        assert transcript_region.has_class("console-frame-solid")


@pytest.mark.asyncio
async def test_manual_reading_state_captures_and_restores_scroll_and_selection():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript, selected = await _seed_transcript(console, pilot)
        transcript.release_anchor()
        transcript.scroll_to(y=2, animate=False)
        await pilot.pause()

        captured = console._capture_console_transcript_reading_state()
        assert captured is not None
        assert captured.anchored is False
        assert captured.scroll_y == pytest.approx(2.0)
        assert captured.selected_message_id == selected

        # Move away from the captured position in both dimensions.
        transcript.scroll_to(y=0, animate=False)
        transcript.selected_message_id = None
        await pilot.pause()

        console._restore_console_transcript_reading_state(captured)
        await pilot.pause()

        assert transcript.selected_message_id == selected
        assert transcript.scroll_y == pytest.approx(2.0)
        assert _tail_is_anchored(transcript) is False


@pytest.mark.asyncio
async def test_anchored_reading_state_restores_tail_follow():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript, selected = await _seed_transcript(console, pilot)
        transcript.anchor()
        await pilot.pause()

        captured = console._capture_console_transcript_reading_state()
        assert captured is not None
        assert captured.anchored is True

        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert _tail_is_anchored(transcript) is False

        console._restore_console_transcript_reading_state(captured)
        await pilot.pause()

        assert _tail_is_anchored(transcript) is True
        assert transcript.selected_message_id == selected


@pytest.mark.asyncio
async def test_restoring_a_missing_reading_state_is_inert():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript, selected = await _seed_transcript(console, pilot)
        await pilot.pause()

        console._restore_console_transcript_reading_state(None)
        await pilot.pause()

        assert transcript.selected_message_id == selected


@pytest.mark.asyncio
async def test_clear_message_selection_clears_the_mounted_transcript():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript, selected = await _seed_transcript(console, pilot)
        await pilot.pause()
        assert transcript.selected_message_id == selected

        console._clear_native_console_message_selection()
        await pilot.pause()

        assert transcript.selected_message_id is None


@pytest.mark.asyncio
async def test_note_follow_intent_stamps_the_mounted_transcript():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript, _selected = await _seed_transcript(console, pilot)
        transcript.release_anchor()
        await pilot.pause()
        before = transcript._follow_intent_time

        console._note_console_follow_intent()

        # TASK-336's whole mechanism is "the most recent of (follow intent,
        # user scroll) wins", so the stamp must land AFTER the release above.
        assert transcript._follow_intent_time > before
        assert transcript._follow_intent_time >= transcript._user_scroll_time


@pytest.mark.asyncio
async def test_idle_speech_lives_in_selected_action_row_not_header():
    app = _SpeechHeaderHarness()
    async with app.run_test(size=(140, 32)) as pilot:
        await _wait_for_selector(app, pilot, "#console-message-speech-a")
        transcript = app.query_one(ConsoleTranscript)

        # Unselected: no header speech action anywhere (idle Speak is not
        # part of the normal per-message chrome).
        assert transcript.selected_message_id is None
        assert len(app.query("#console-message-speech-action-speech-a")) == 0
        assert len(app.query("#console-message-speech-action-speech-b")) == 0

        transcript.select_message("speech-a")
        await pilot.pause()

        # Selected: Speak appears in the ACTION ROW with the other
        # per-message options, on that message only.
        action_row = app.query_one("#console-message-actions-speech-a")
        action_ids = {child.id for child in action_row.children}
        assert "console-message-action-copy-speech-a" in action_ids
        assert "console-message-action-edit-speech-a" in action_ids
        assert "console-message-action-speak-speech-a" in action_ids
        assert len(app.query("#console-message-action-speak-speech-b")) == 0
        # The header still hosts no speech control while idle.
        assert len(app.query("#console-message-speech-action-speech-a")) == 0


@pytest.mark.asyncio
async def test_message_header_tracks_speech_lifecycle_without_recreating_row():
    app = _SpeechHeaderHarness()
    async with app.run_test(size=(140, 32)) as pilot:
        await _wait_for_selector(app, pilot, "#console-message-speech-a")
        transcript = app.query_one(ConsoleTranscript)
        row = app.query_one("#console-message-speech-a", ConsoleMarkdownMessage)

        assert transcript.set_speech_state("speech-a", "generating") is True
        await pilot.pause()
        action = app.query_one("#console-message-speech-action-speech-a", Button)
        assert app.query_one("#console-message-speech-a") is row
        assert action.disabled is True
        assert _speech_status(transcript, "speech-a") == "Generating"

        assert transcript.set_speech_state("speech-a", "playing") is True
        await pilot.pause()
        playing = app.query_one("#console-message-speech-action-speech-a", Button)
        assert app.query_one("#console-message-speech-a") is row
        assert playing is action
        assert playing.disabled is False
        assert _speech_status(transcript, "speech-a") == "Playing"

        assert transcript.set_speech_state("speech-a", "stopped") is True
        await pilot.pause()
        stopped = app.query_one("#console-message-speech-action-speech-a", Button)
        assert stopped is action
        assert _speech_status(transcript, "speech-a") == "Stopped"

        assert transcript.set_speech_state("speech-a", "playing") is False
        await pilot.pause()
        assert app.query_one("#console-message-speech-action-speech-a", Button) is action
        assert _speech_status(transcript, "speech-a") == "Stopped"


@pytest.mark.asyncio
async def test_new_active_speech_stops_prior_header_and_failure_clears_on_selection():
    app = _SpeechHeaderHarness()
    async with app.run_test(size=(140, 32)) as pilot:
        await _wait_for_selector(app, pilot, "#console-message-speech-a")
        transcript = app.query_one(ConsoleTranscript)

        assert transcript.set_speech_state("speech-a", "generating") is True
        assert transcript.set_speech_state("speech-a", "playing") is True
        assert transcript.set_speech_state("speech-b", "generating") is True
        await pilot.pause()

        assert _speech_status(transcript, "speech-a") == "Stopped"
        assert len(app.query("#console-message-speech-action-speech-a")) == 1
        assert _speech_status(transcript, "speech-b") == "Generating"

        assert transcript.set_speech_state("speech-b", "failed") is True
        await pilot.pause()
        assert _speech_status(transcript, "speech-b") == "Failed"

        transcript.select_message("speech-a")
        await pilot.pause()
        assert _speech_status(transcript, "speech-b") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [90, 140])
async def test_speech_header_reserves_nonoverlapping_space(width: int):
    app = _SpeechHeaderHarness()
    async with app.run_test(size=(width, 32)) as pilot:
        await _wait_for_selector(app, pilot, "#console-message-speech-a")
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_speech_state("speech-a", "generating")
        await pilot.pause()

        row = app.query_one("#console-message-speech-a")
        header = app.query_one("#console-message-header-speech-a")
        speaker = header.query_one(".console-transcript-speaker-label")
        speech = header.query_one(".console-message-speech-presentation")
        status = header.query_one(".console-message-speech-status")
        action = header.query_one(".console-message-speech-action")

        assert header.region.width <= row.region.width
        assert speaker.region.x + speaker.region.width <= speech.region.x
        assert status.region.x + status.region.width <= action.region.x
        assert action.region.x + action.region.width <= row.region.x + row.region.width
        assert header.region.height == 1
