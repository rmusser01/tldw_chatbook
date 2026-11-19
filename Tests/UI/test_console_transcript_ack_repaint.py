"""TASK-37145: keep the committed acknowledgement through native echo replacement."""

import asyncio

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_console_transcript_poll_stop_removal import _slow_unmount
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.UI.Console_Modules.send_acknowledgement import (
    ConsoleSendAcknowledgement,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

DRAFT = "acknowledgement repaint witness"


class _TranscriptHost(ConsolidatedCSSApp):
    CSS_PATH = list(APP_STYLESHEETS)

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")
        yield Static("repaint control", id="repaint-control", classes="h-1")


class _Commits:
    """Record active-screen compositor commits, including failure-only context."""

    def __init__(self, app, transcript, ack) -> None:
        self.frames: list[dict] = []
        self.phase = "acknowledgement"
        self.original = app._display

        def display(screen, update) -> None:
            self.original(screen, update)
            if screen is app.screen and update is not None:
                self.frames.append(
                    {
                        "time": asyncio.get_running_loop().time(),
                        "screen": screen.id,
                        "update": type(update).__name__,
                        "phase": self.phase,
                        "ack_active": ack.active_for("s1"),
                        "supplied_ids": [
                            message.id for message in transcript._messages
                        ],
                        "rows": [
                            (row.id, row.is_attached, tuple(row.region))
                            for row in transcript.query(".console-transcript-message")
                        ],
                        "text": "\n".join(
                            strip.text for strip in screen._compositor.render_strips()
                        ),
                    }
                )

        app._display = display


async def _until(predicate) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            # Pilot waits on every mounted pump, including the held Unmount.
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("cancel_replacement", [False, True], ids=["echo", "cancel"])
async def test_acknowledgement_replacement_keeps_committed_frame(
    request: pytest.FixtureRequest, monkeypatch, cancel_replacement: bool
) -> None:
    # Removing the repaint transaction must fail the echo arm: the held native
    # exit permits several compositor commits with neither acknowledged body.
    app = _TranscriptHost()
    entered, release, finished = (asyncio.Event() for _ in range(3))
    ack = ConsoleSendAcknowledgement(lambda: None, lambda session_id: [])
    token = ack.begin("s1", DRAFT)
    assert token is not None
    echo = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content=DRAFT, id="store-echo", status="completed"
    )
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        commits = _Commits(app, transcript, ack)
        timer = None
        doomed = None
        try:
            transcript.set_messages(ack.project("s1", []), session_id="s1")
            await transcript.refresh_messages()
            await _until(lambda: commits.frames and DRAFT in commits.frames[-1]["text"])
            ack_id = ack.pending_row_id("s1")
            doomed = transcript.query_one(f"#console-message-{ack_id}")
            _slow_unmount(monkeypatch, doomed, entered, release)
            with ack.dispatching(token):
                ack.custody_callback("s1")
                ack.mark_admitted("s1")
            replacement = ack.project("s1", [echo])
            assert replacement == [echo] and replacement[0] is echo
            assert not ack.active_for("s1")
            start = len(commits.frames)
            commits.phase = "replacement"
            started = False

            async def paint() -> None:
                nonlocal started
                if started:
                    return
                started = True
                try:
                    transcript.set_messages(replacement, session_id="s1")
                    await transcript.refresh_messages()
                finally:
                    finished.set()

            timer = app.set_interval(0.01, paint)
            await asyncio.wait_for(entered.wait(), 5)
            assert transcript._messages == [echo]
            assert not finished.is_set()
            if cancel_replacement:
                timer.stop()
                await asyncio.wait_for(finished.wait(), 5)
                commits.phase = "cancelled"
                app.query_one("#repaint-control", Static).update("repaint resumed")
                app.screen.refresh(layout=True)
                await _until(lambda: "repaint resumed" in commits.frames[-1]["text"])
            else:
                # These are observation opportunities, not a repair delay. On
                # the fixed arm there need not be another commit while held.
                for _ in range(8):
                    app.screen.refresh(layout=True)
                    await asyncio.sleep(0.03)
                observed = commits.frames[start:]
                missing = [frame for frame in observed if DRAFT not in frame["text"]]
                assert (
                    not missing
                ), f"{len(missing)} missing compositor commits: {missing!r}"
                assert commits.frames[-1]["text"].count(DRAFT) == 1, commits.frames
        finally:
            release.set()
            try:
                if timer is not None:
                    try:
                        await asyncio.wait_for(finished.wait(), 5)
                    finally:
                        timer.stop()
                if doomed is not None:
                    await _until(lambda: doomed.parent is None)
                commits.phase = "settled"
                await transcript.refresh_messages()
                await pilot.pause()
            finally:
                app._display = commits.original
        assert [
            row.message_id for row in transcript.query(".console-transcript-message")
        ] == ["store-echo"]
        assert doomed._task is None or not doomed._task.cancelled()
        assert commits.frames[-1]["text"].count(DRAFT) == 1, commits.frames[-1]
        assert "Sending" not in commits.frames[-1]["text"], commits.frames[-1]
