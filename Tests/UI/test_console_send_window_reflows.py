"""A later send moves the Console transcript twice between Enter and the provider.

TASK-33620.15.26 and TASK-33620.15.20 (owner decision 2026-10-29: keep the
transcript unbounded, cut its whole-transcript reflows per send instead).

Measured headless on origin/dev 38c9d2edcf (160x45, a chat longer than the
screen, a 65-character prompt, Enter in the composer), every reflow from Enter
to the provider call recorded with the regions it changed:

* On both send paths the composer grew from one row to two shortly after
  Enter and shrank back once the sent draft cleared. The Send reason strip
  ("Queue opens once this turn is accepted") took about 40 cells from the
  draft, and the sent text, which fitted one row at the idle width, wrapped to
  two. Each height change resized the transcript (one reflow) and then moved
  all of its rows (a second one).
* A stock send (the received-intent path a real composer send takes) then
  added the store's echo and the reply placeholder.
* A send on the acknowledged path (the mounted send harness's, and custom,
  command and queue dispatch) first added the "Sending…" row, then the echo
  replaced it -- a row shorter, so every row above it moved -- and the reply
  placeholder moved them all again 120-150 ms later: five whole-transcript
  reflows in all.

Live, each such reflow and its paint is the window's longest UI-loop callback
(TASK-33620.15.20). Now the composer keeps its rows while it shows the text a
send took (the strip still shows beside it), and on the acknowledged path the
"Sending…" row stands in for the echo until the echo's turn writes its next
row, so the echo and its placeholder land in one layout pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass, field

import pytest
from textual._compositor import Compositor
from textual.widget import Widget
from textual.widgets import Static

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from Tests.private_profile import private_profile_test
from Tests.UI.test_console_hook_review_send_freeze import _key
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: 65 characters: one row at the idle composer width at 160x45 (95 cells), two
#: rows beside the Send reason strip (54 cells).
PROMPT = "Growth check: in one short sentence, share a fact about lanterns."
REPLY = (
    "A reply paragraph with enough words to wrap across the transcript width "
    "at least once or twice.\n\nA second paragraph."
)
#: Each warm-up reply is taller than half the transcript, so the chat already
#: overflows (and shows its scrollbar) before the measured send.
WARM_UP_REPLY = "\n\n".join(f"Warm-up paragraph {n}." for n in range(1, 10))


@dataclass
class _Reflow:
    """What one Console reflow changed."""

    rows_moved: int
    row_keys: frozenset[str]
    composer_height: int
    draft_empty: bool
    strip_copy: str
    strip_shown: bool


@dataclass
class _Window:
    """Each Console reflow from Enter until the reply placeholder is laid out.

    The provider's stream waits for that layout (a real provider takes far
    longer to its first token), so no reply text moves the transcript inside
    the record. ``post`` is the number of reflows made before the provider was
    asked to stream: ``reflows[:post]`` is the Enter-to-POST window.
    """

    console: object = None
    pilot: object = None
    recording: bool = False
    rows_at_enter: dict[str, object] = field(default_factory=dict)
    reflows: list[_Reflow] = field(default_factory=list)
    stream_calls: int = 0
    post: int = 0
    draft_width_at_enter: int = 0
    draft_width_beside_strip: int = 0

    def placeholder_laid_out(self) -> bool:
        return any(
            key.startswith("assistant-turn:") and key not in self.rows_at_enter
            for reflow in list(self.reflows)
            for key in reflow.row_keys
        )

    def provider_called(self) -> bool:
        """Note the provider call; True when the measured send made it."""
        self.stream_calls += 1
        if not self.recording:
            return False
        self.post = len(self.reflows)
        return True

    def entry(self, compositor: Compositor, before: dict[Widget, object]) -> _Reflow:
        console = self.console
        transcript = console.query_one("#console-native-transcript")
        composer = console._console_composer_or_none()
        strip = console.query_one("#console-send-disabled-reason", Static)
        current = compositor._full_map
        moved = sum(
            1
            for widget in self.rows_at_enter.values()
            if widget in before
            and widget in current
            and before[widget] != current[widget].region
        )
        keys = frozenset(
            key for key, widget in transcript._row_widgets.items() if widget in current
        )
        return _Reflow(
            rows_moved=moved,
            row_keys=keys,
            composer_height=current[composer].region.height,
            draft_empty=not composer.draft_text(),
            strip_copy=str(strip.content),
            strip_shown=bool(
                strip.display and strip in current and current[strip].region.width > 0
            ),
        )


def _record_reflows(monkeypatch, window: _Window) -> None:
    original = Compositor.reflow

    def reflow(compositor, parent, size):
        if not window.recording or parent is not window.console:
            return original(compositor, parent, size)
        before = {
            widget: geometry.region for widget, geometry in compositor._full_map.items()
        }
        result = original(compositor, parent, size)
        window.reflows.append(window.entry(compositor, before))
        return result

    monkeypatch.setattr(Compositor, "reflow", reflow)


async def _strip_beside_the_sent_text(window: _Window) -> None:
    """Wait until the Send reason strip has narrowed the sent text, re-wrapped.

    Bounded: 5 s for the strip to be laid out beside the text, then until the
    composer changes height (origin/dev's re-wrap) or 0.5 s pass.
    """
    console = window.console
    strip = console.query_one("#console-send-disabled-reason", Static)
    draft = console.query_one("#console-command-visible-text")
    composer = console._console_composer_or_none()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not (
        strip.display
        and strip.region.width > 0
        and 0 < draft.region.width < window.draft_width_at_enter
    ):
        await asyncio.sleep(0.02)
    height = composer.region.height
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and composer.region.height == height:
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)  # The new height's transcript move, laid out.


async def _until(pilot, predicate, what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate(), what


async def _drive(host, pilot, console, composer, store, window, finished) -> None:
    """Two finished turns, then one recorded Enter send of ``PROMPT``."""
    transcript = console.query_one("#console-native-transcript")
    await pilot.pause(0.5)
    for index, text in enumerate(("Warm-up one.", "Warm-up two.", PROMPT)):
        session = store.ensure_session()
        before = len(store.messages_for_session(session.id))
        composer.load_draft(text)
        composer.focus()
        await _until(
            pilot,
            lambda: not console.query_one("#console-send-message").disabled,
            "Send never took the draft",
        )
        await pilot.pause(0.3)
        if index == 2:
            # A later send in a chat longer than the screen, the everyday
            # case: the first overflow adds the scrollbar and re-wraps
            # every row, once per chat.
            assert transcript.max_scroll_y > 0, "the chat fits the screen"
            window.rows_at_enter = dict(transcript._row_widgets)
            window.draft_width_at_enter = console.query_one(
                "#console-command-visible-text"
            ).region.width
            window.reflows.clear()
            window.recording = True
        calls = window.stream_calls
        _key(host, "enter", "\r")
        await _until(pilot, lambda: window.stream_calls > calls, "no provider call")
        await _until(
            pilot,
            lambda: (
                len(store.messages_for_session(session.id)) >= before + 2
                and (REPLY if index == 2 else WARM_UP_REPLY)
                in store.messages_for_session(session.id)[-1].content
                and finished(session.id)
            ),
            "the reply never finished",
        )
        await pilot.pause(0.3)


@contextlib.asynccontextmanager
async def _acknowledged_later_send(monkeypatch, tmp_path):
    """A later send on the acknowledged path (the mounted send harness's).

    Yields:
        ``(window, console, composer, store)`` once the measured reply has
        finished; ``window.reflows`` holds the reflows from Enter until the
        reply placeholder was laid out.
    """
    from Tests.UI.test_console_composer_run_controls import _mounted
    from Tests.UI.test_console_send_transition_layout_requests import _console_host

    window = _Window()
    _record_reflows(monkeypatch, window)
    gateway, host = _console_host(tmp_path)

    class _Replying(type(gateway)):
        async def stream_chat(self, resolution, messages, **kwargs):
            if not window.provider_called():
                yield WARM_UP_REPLY
                return
            deadline = time.monotonic() + 10
            while not window.placeholder_laid_out() and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            window.recording = False
            yield REPLY

    gateway.__class__ = _Replying
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        window.console, window.pilot = console, pilot
        store = console._ensure_console_chat_store()
        await _drive(
            host,
            pilot,
            console,
            composer,
            store,
            window,
            lambda _session_id: console._console_transcript_sync_timer is None,
        )
        yield window, console, composer, store


@contextlib.asynccontextmanager
async def _stock_later_send(monkeypatch):
    """A later stock send: the received-intent path a real composer send takes.

    The real ChatScreen, runtime, controller, store and file-backed database;
    only the provider call is a double, durable trace capture is off (the
    harness has no normalized trace writer, so a later Capture-On send would
    stop at "Trace capture could not start"), and the measured send's archive
    check waits for the Send reason strip to narrow the sent text (see
    ``slow_archive_check``).

    Yields:
        As :func:`_acknowledged_later_send`.
    """
    from tldw_chatbook import config
    from tldw_chatbook.Chat import (
        console_provider_gateway,
        conversation_archive_actions,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_console_configuration_worker_feedback import (
        _VisibleNavigatedConsoleHarness,
    )
    from Tests.UI.test_console_hook_review_send_freeze import _task_factory
    from Tests.UI.test_console_native_chat_flow import (
        _persist_console_provider_config,
    )

    window = _Window()
    _record_reflows(monkeypatch, window)
    app = _build_test_app(user_data_dir=config.get_user_data_dir())
    database = CharactersRAGDB(config.get_chachanotes_db_path(), client_id="p26")
    app.chachanotes_db = database
    app.local_chat_conversation_service = ChatConversationService(database)
    _persist_console_provider_config(
        app,
        provider="openai",
        model="gpt-4o",
        provider_settings={"api_key": "synthetic-test-key"},
    )
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-4o"
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="send-window", name="Send window")
    registry.set_active_workspace("send-window")
    monkeypatch.setattr(
        console_provider_gateway.ConsoleProviderGateway,
        "supports_durable_capture",
        property(lambda _gateway: False),
    )

    def reply(**_kwargs):
        if not window.provider_called():
            return WARM_UP_REPLY
        if threading.current_thread() is not threading.main_thread():
            deadline = time.monotonic() + 10
            while not window.placeholder_laid_out() and time.monotonic() < deadline:
                time.sleep(0.02)
        window.recording = False
        return REPLY

    monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", reply)
    archive_check = conversation_archive_actions.conversation_send_refusal

    async def slow_archive_check(app_, conversation_id):
        # Live, the composer shows the sent text for about 0.9 s while the
        # send prepares; headless the preparation can end before the
        # composer's next refresh. This awaited check runs right after the
        # composer turns to "Preparing...", so on the measured send it waits
        # until the strip is laid out beside the sent text and the composer
        # has had time to re-wrap it (on origin/dev, to two rows): a fixed
        # 0.5 s hold let the send take the text first in some runs.
        if window.recording:
            await _strip_beside_the_sent_text(window)
        return await archive_check(app_, conversation_id)

    monkeypatch.setattr(
        conversation_archive_actions, "conversation_send_refusal", slow_archive_check
    )
    host = _VisibleNavigatedConsoleHarness(app)
    runtime = None
    try:
        with _task_factory("eager"):
            async with host.run_test(size=(160, 45)) as pilot:
                await _until(
                    pilot,
                    lambda: (
                        isinstance(host.screen, ChatScreen)
                        and host.screen.is_mounted
                        and host.screen._console_attach_reconciled
                        and not host.screen._console_attach_reconcile_running
                    ),
                    "the Console never attached",
                )
                console = host.screen
                window.console, window.pilot = console, pilot
                runtime = console._console_runtime()
                store = console._ensure_console_chat_controller().store
                composer = console._console_composer_or_none()
                await _drive(
                    host,
                    pilot,
                    console,
                    composer,
                    store,
                    window,
                    lambda session_id: not runtime.has_custodied_turns(session_id),
                )
                yield window, console, composer, store
    finally:
        if runtime is not None:
            await runtime.dispose()
        with database.quiesce_connections(timeout_seconds=5):
            pass


def _composer_findings(window: _Window, console, composer) -> list[str]:
    """TASK-33620.15.26 AC#2/#3/#4: the composer keeps its height from Enter."""
    reflows = window.reflows
    findings: list[str] = []
    heights = [reflow.composer_height for reflow in reflows]
    if heights != [1] * len(heights):
        findings.append(
            f"the composer changed height after Enter (the provider was called "
            f"after reflow {window.post}): {heights}"
        )
    # Positive controls: the strip was laid out beside the text being sent
    # (the condition that grew the composer on origin/dev), and the prompt
    # fits one row at the idle width but wraps at the width the strip leaves.
    assert [r for r in reflows if r.strip_shown and not r.draft_empty], (
        "the reason strip never sat beside the sent text"
    )
    idle_width = window.draft_width_at_enter - 2
    assert composer._visible_draft_row_count(PROMPT, idle_width) == 1
    beside_width = window.draft_width_beside_strip - 2
    assert 0 < beside_width < idle_width
    assert composer._visible_draft_row_count(PROMPT, beside_width) == 2
    if not [r for r in reflows if r.draft_empty and r.strip_shown]:
        findings.append("the Send reason strip never showed in the window")
    # AC#4: whenever the draft is empty, the Send reason copy is shown.
    hidden = [
        (reflow.strip_copy, reflow.strip_shown)
        for reflow in reflows
        if reflow.draft_empty and not (reflow.strip_shown and reflow.strip_copy)
    ]
    if hidden:
        findings.append(f"an empty draft without its Send reason: {hidden}")
    return findings


def _transcript_findings(window: _Window, store, *, same_reflow: bool) -> list[str]:
    """TASK-33620.15.20: at most two whole-transcript reflows.

    Args:
        window: The recorded send.
        store: The Console chat store.
        same_reflow: Also require the echo and its placeholder to land in one
            reflow (the acknowledged path, where a "Sending…" row stands in).
    """
    messages = store.messages_for_session(store.active_session_id)
    echo = next(m for m in reversed(messages) if m.role is ConsoleMessageRole.USER)
    reply = messages[-1]
    assert reply.role is ConsoleMessageRole.ASSISTANT
    reflows = window.reflows
    rows = len(window.rows_at_enter)
    assert rows >= 4, f"the chat had {rows} rows before the measured send"
    whole = [i for i, reflow in enumerate(reflows) if reflow.rows_moved * 2 >= rows]
    first_echo = next(
        (i for i, r in enumerate(reflows) if f"message:{echo.id}" in r.row_keys),
        None,
    )
    first_reply = next(
        (
            i
            for i, r in enumerate(reflows)
            if f"assistant-turn:{reply.id}" in r.row_keys
        ),
        None,
    )
    summary = (
        f"whole-transcript reflows {[(i, reflows[i].rows_moved) for i in whole]} "
        f"of {len(reflows)} (rows {rows}); echo first laid out in reflow "
        f"{first_echo}, its placeholder in {first_reply}"
    )
    assert first_echo is not None and first_reply is not None, summary
    # A positive control: the ledger sees the transcript move when the send's
    # first row lands (the "Sending…" row, or a stock send's echo).
    assert whole and whole[0] <= first_echo, summary
    findings = []
    if len(whole) > 2:
        findings.append(f"more than two reflows moved the transcript: {summary}")
    if same_reflow and first_echo != first_reply:
        findings.append(f"the echo and its placeholder landed apart: {summary}")
    return findings


async def _measure_draft_beside_strip(pilot, console, composer, window) -> None:
    """Lay the queue copy out on the idle composer and read the draft's width.

    The positive control: the width the strip leaves the draft when it shows
    "Queue opens once this turn is accepted" at 160x45.
    """
    from tldw_chatbook.Chat.console_display_state import QUEUE_REASON_PREPARING

    draft = console.query_one("#console-command-visible-text")
    composer._sync_send_disabled_reason(QUEUE_REASON_PREPARING, muted=False)
    await pilot.pause(0.2)
    window.draft_width_beside_strip = draft.region.width
    composer._sync_send_disabled_reason("", muted=True)
    composer._sync_current_action_state()
    await pilot.pause(0.2)


@private_profile_test
async def test_a_stock_later_send_keeps_the_composer_one_row_tall(
    monkeypatch, request, tmp_path
) -> None:
    """The live path: no composer height change, two transcript moves at most.

    Red on origin/dev 38c9d2edcf in 3 of 3 runs: the composer read 1, 2, 2, 2,
    2, 1, ... rows as the reason strip narrowed the sent draft to two rows and
    the draft then cleared.
    """
    async with _stock_later_send(monkeypatch) as (window, console, composer, store):
        assert window.post, "no reflow was recorded between Enter and the provider call"
        await _measure_draft_beside_strip(window.pilot, console, composer, window)
        findings = _composer_findings(window, console, composer)
        findings += _transcript_findings(window, store, same_reflow=False)
        assert findings == [], "\n".join(findings)


@private_profile_test
async def test_an_acknowledged_later_send_lands_its_echo_with_the_placeholder(
    monkeypatch, request, tmp_path
) -> None:
    """The acknowledged path: the "Sending…" row, then echo and placeholder.

    One app run checks the composer and the transcript and reports every
    finding, so neither hides the other. Red on origin/dev 38c9d2edcf for both:
    composer heights 1, 2, 2, 2, 2, 1, ...; five whole-transcript reflows, the
    echo first laid out in reflow 7 and its placeholder in reflow 9.
    """
    async with _acknowledged_later_send(monkeypatch, tmp_path) as (
        window,
        console,
        composer,
        store,
    ):
        assert window.post, "no reflow was recorded between Enter and the provider call"
        await _measure_draft_beside_strip(window.pilot, console, composer, window)
        findings = _composer_findings(window, console, composer)
        findings += _transcript_findings(window, store, same_reflow=True)
        assert findings == [], "\n".join(findings)


async def test_the_composer_keeps_its_rows_while_a_send_takes_its_text(
    monkeypatch,
) -> None:
    """The rule on the composer alone (TASK-33620.15.26).

    The draft's width is set directly (the standalone composer has no Console
    row to narrow it): at the narrow width the text wraps to two rows. While
    the composer shows the text a send took, unedited, it keeps the rows it
    had; an edit or a refused send makes the draft the user's again, and the
    empty draft after the send is one row. New API on this branch
    (``note_draft_sending``); the mounted sends above are the base-build reds.
    """
    from Tests.UI.test_console_run_tick_layout_requests import _Host
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    composer = ConsoleComposerBar()
    width = {"cells": 80}
    async with _Host(composer).run_test(size=(160, 10)) as pilot:
        monkeypatch.setattr(composer, "_draft_render_width", lambda: width["cells"])

        def rows(cells: int) -> int:
            width["cells"] = cells
            composer._refresh_visible_draft()
            return composer._draft_geometry_signature[0]

        text = "Growth check: in one short sentence, share a fact about lanterns."
        composer.load_draft(text)
        await pilot.pause()
        assert rows(80) == 1
        # Positive control: a draft no send has taken wraps when narrowed.
        assert rows(50) == 2
        assert rows(80) == 1
        composer.note_draft_sending(composer.capture_draft_for_send())
        _show_preparing(composer)
        assert rows(50) == 1, "the text being sent grew the composer"
        composer.insert_text(" Now.")
        assert rows(50) == 2, "an edited draft is the user's again"
        assert rows(80) == 1
        composer.note_draft_sending(composer.capture_draft_for_send())
        assert rows(50) == 1
        composer.note_draft_sending(None)  # refused: the draft is the user's
        assert rows(50) == 2
        assert rows(80) == 1
        stash = composer.capture_draft_for_send()
        composer.note_draft_sending(stash)
        assert rows(50) == 1
        assert composer.commit_captured_draft(stash)
        assert rows(50) == 1 and composer.draft_text() == ""


def _show_preparing(composer) -> None:
    """Sync the composer as a send in flight leaves it: Send refuses, Preparing."""
    from tldw_chatbook.Chat.console_display_state import (
        QUEUE_REASON_PREPARING,
        SEND_LABEL_PREPARING,
    )

    composer.sync_action_state(
        has_draft=True,
        run_active=False,
        can_save_chatbook=False,
        send_blocked=True,
        queue_blocked_reason=QUEUE_REASON_PREPARING,
        send_label=SEND_LABEL_PREPARING,
    )


async def test_a_refused_send_gives_its_draft_its_rows_back(monkeypatch) -> None:
    """The kept rows last only while a send of the shown text is in flight.

    Review of TASK-33620.15.26 (2026-10-29): the mark ``_schedule`` sets was
    cleared only by the received-intent path's synchronous refusal. A send
    refused any other way (an unknown ``/command``'s hint, a needs-review
    skill, a dispatch gate, a stock turn refused after its admission) kept
    the draft and the mark, and the composer then kept cutting the draft to
    the rows it last showed, outside any send. The rows are kept only while
    the composer refuses Send for a turn being prepared: once it no longer
    does, the draft gets its rows back, mark or no mark.
    """
    from Tests.UI.test_console_run_tick_layout_requests import _Host
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    composer = ConsoleComposerBar()
    width = {"cells": 80}
    async with _Host(composer).run_test(size=(160, 10)) as pilot:
        monkeypatch.setattr(composer, "_draft_render_width", lambda: width["cells"])

        def rows(cells: int) -> int:
            width["cells"] = cells
            composer._refresh_visible_draft()
            return composer._draft_geometry_signature[0]

        composer.load_draft(
            "/nosuchcommand in one short sentence, share a fact about lanterns ok"
        )
        await pilot.pause()
        assert rows(80) == 1
        # A command dispatch marks the draft but never prepares a turn; its
        # hint leaves the draft with the user.
        composer.note_draft_sending(composer.capture_draft_for_send())
        assert rows(50) == 2, "a draft no turn is preparing was cut to its row"
        assert rows(80) == 1
        # A send that reached "Preparing..." and was then refused: the screen
        # syncs the composer back to an idle Send and never clears the mark.
        composer.note_draft_sending(composer.capture_draft_for_send())
        _show_preparing(composer)
        assert rows(50) == 1, "positive control: the send in flight keeps its rows"
        composer.sync_action_state(
            has_draft=True, run_active=False, can_save_chatbook=False
        )
        assert rows(50) == 2, "a refused send's draft stayed cut to one row"
        # A refusal that clears the mark while "Preparing..." still shows
        # re-windows the draft itself: nothing else may refresh it soon.
        assert rows(80) == 1
        composer.note_draft_sending(composer.capture_draft_for_send())
        _show_preparing(composer)
        await pilot.pause()  # The strip's own re-window runs first.
        assert rows(50) == 1
        composer.note_draft_sending(None)
        await pilot.pause()
        assert composer._draft_geometry_signature[0] == 2, (
            "clearing the mark left the draft cut to one row"
        )


@private_profile_test
async def test_an_unknown_command_keeps_its_draft_and_its_rows(
    monkeypatch, request, tmp_path
) -> None:
    """A refused send leaves a draft that re-wraps like any other.

    Review of TASK-33620.15.26 (2026-10-29), reproduced on the mounted
    Console: Enter on an unknown ``/command`` shows its hint and keeps the
    draft, but the mark the send set stayed, so after a resize to 90 columns
    the draft needed 3 rows and the composer still showed 1 until the user
    typed.
    """
    from Tests.UI.test_console_composer_run_controls import _mounted
    from Tests.UI.test_console_send_transition_layout_requests import _console_host
    from tldw_chatbook.UI.Console_Modules.send_acknowledgement import (
        FLIGHT_ATTRIBUTE,
    )

    draft = "/nosuchcommand in one short sentence, share a fact about lanterns ok"
    _gateway, host = _console_host(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        await pilot.pause(0.5)
        composer.load_draft(draft)
        composer.focus()
        await _until(
            pilot,
            lambda: not console.query_one("#console-send-message").disabled,
            "Send never took the draft",
        )
        _key(host, "enter", "\r")

        def settled() -> bool:
            flight = getattr(console, FLIGHT_ATTRIBUTE, None)
            return (
                console._console_unknown_send_armed == draft
                and console._console_pending_send is None
                and (flight is None or not flight.tasks)
            )

        await _until(pilot, settled, "the unknown command's hint never showed")
        await pilot.pause(0.3)
        assert composer.draft_text() == draft, "the hint did not keep the draft"
        assert composer._draft_geometry_signature[0] == 1
        await pilot.resize_terminal(90, 45)
        await pilot.pause(0.5)
        text = console.query_one("#console-command-visible-text")
        needed = composer._visible_draft_row_count(
            composer._display_draft_text(),
            composer._draft_render_width(),
            reserve_trailing_cell=composer.has_focus_within,
        )
        assert needed > 1, "positive control: the draft wraps at 90 columns"
        shown = (composer._draft_geometry_signature[0], text.region.height)
        assert shown == (needed, needed), (
            f"the refused draft needs {needed} rows at 90 columns; the composer "
            f"shows {shown[0]} (laid out {shown[1]})"
        )
