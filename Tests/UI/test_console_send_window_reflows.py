"""A later send moves the Console transcript twice between Enter and the provider.

TASK-33620.15.26 and TASK-33620.15.20 (owner decision 2026-10-29: keep the
transcript unbounded, cut its whole-transcript reflows per send instead).

Measured headless on origin/dev 38c9d2edcf (160x45, a chat's later sends, a
65-character prompt, Enter in the composer): every reflow from Enter to the
provider's stream call was recorded with the regions it changed. Five of them
moved every transcript row:

* the "Sending…" acknowledgement row joining the bottom-anchored transcript;
* the composer growing from one row to two, about 40-240 ms after Enter, and
  shrinking back once the sent draft cleared. Each height change resized the
  transcript (one reflow) and then moved all of its rows (a second one): the
  Send reason strip ("Queue opens once this turn is accepted") took about 40
  cells from the draft, and the sent text, which fitted one row at the idle
  width, wrapped to two;
* the store's echo replacing the acknowledgement row: the echo has no
  "Sending…" line, so it is a row shorter and everything above it moved;
* the reply placeholder joining the transcript, 120-150 ms later.

Live, each such reflow and its paint is the window's longest UI-loop callback
(TASK-33620.15.20). Here the composer keeps its height while the composer
still shows the text being sent, and the acknowledgement row stands in for the
echo until the echo's turn writes its next row, so the echo and the reply
placeholder land in one layout pass.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    recording: bool = False
    rows_at_enter: dict[str, object] = field(default_factory=dict)
    reflows: list[_Reflow] = field(default_factory=list)
    stream_calls: int = 0
    post: int = 0
    draft_width_at_enter: int = 0

    def placeholder_laid_out(self) -> bool:
        return any(
            key.startswith("assistant-turn:") and key not in self.rows_at_enter
            for reflow in self.reflows
            for key in reflow.row_keys
        )

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


async def _until(pilot, predicate, what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate(), what


@contextlib.asynccontextmanager
async def _later_send(monkeypatch, tmp_path):
    """Two finished turns, then one Enter send of ``PROMPT``, recorded.

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
            window.stream_calls += 1
            if not window.recording:
                yield WARM_UP_REPLY
                return
            window.post = len(window.reflows)
            deadline = time.monotonic() + 10
            while not window.placeholder_laid_out() and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            window.recording = False
            yield REPLY

    gateway.__class__ = _Replying
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        window.console = console
        store = console._ensure_console_chat_store()
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
                    and console._console_transcript_sync_timer is None
                ),
                "the reply never finished",
            )
            await pilot.pause(0.3)
        yield window, console, composer, store


@private_profile_test
async def test_a_one_row_send_keeps_the_composer_one_row_until_the_provider_call(
    monkeypatch, request, tmp_path
) -> None:
    """TASK-33620.15.26 AC#2/#3: no composer height change from Enter to the POST.

    Red on origin/dev 38c9d2edcf: the heights read 1, 2, 2, 2, 1, ... as the
    reason strip narrowed the sent draft to two rows and the draft then cleared.
    """
    async with _later_send(monkeypatch, tmp_path) as (window, console, composer, _):
        reflows = window.reflows
        assert window.post, "no reflow was recorded between Enter and the provider call"
        heights = [reflow.composer_height for reflow in reflows]
        assert heights == [1] * len(heights), (
            f"the composer changed height after Enter (the provider was called "
            f"after reflow {window.post}): {heights}"
        )
        # Positive controls: the prompt fits one row at the idle width and
        # wraps beside the reason strip, so a strip shown next to the sent
        # text would have grown the composer; the strip did show once the
        # draft was empty.
        idle_width = window.draft_width_at_enter - 2
        assert composer._visible_draft_row_count(PROMPT, idle_width) == 1
        shown = [
            reflow for reflow in reflows if reflow.draft_empty and reflow.strip_shown
        ]
        assert shown, "the Send reason strip never showed in the window"
        draft = console.query_one("#console-command-visible-text")
        assert draft.region.width - 2 < idle_width
        assert composer._visible_draft_row_count(PROMPT, draft.region.width - 2) == 2
        # AC#4: whenever the draft is empty, the Send reason copy is shown.
        hidden = [
            (reflow.strip_copy, reflow.strip_shown)
            for reflow in reflows
            if reflow.draft_empty and not (reflow.strip_shown and reflow.strip_copy)
        ]
        assert hidden == [], f"an empty draft without its Send reason: {hidden}"


@private_profile_test
async def test_the_user_row_and_its_reply_placeholder_land_in_one_reflow(
    monkeypatch, request, tmp_path
) -> None:
    """TASK-33620.15.20: at most two whole-transcript reflows per later send.

    Red on origin/dev 38c9d2edcf: five reflows moved every transcript row, and
    the echo and the placeholder first appeared in different reflows.
    """
    async with _later_send(monkeypatch, tmp_path) as (window, _, _, store):
        messages = store.messages_for_session(store.active_session_id)
        echo = next(m for m in reversed(messages) if m.role is ConsoleMessageRole.USER)
        reply = messages[-1]
        assert reply.role is ConsoleMessageRole.ASSISTANT
        reflows = window.reflows
        rows = len(window.rows_at_enter)
        assert rows >= 4, f"the chat had {rows} rows before the measured send"
        whole = [
            index
            for index, reflow in enumerate(reflows)
            if reflow.rows_moved * 2 >= rows
        ]
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
        # The acknowledgement row moves the transcript at Enter (a positive
        # control: the ledger sees a whole-transcript move) and the echo with
        # its placeholder moves it once more.
        assert whole and whole[0] < first_echo, summary
        assert len(whole) <= 2, summary
        assert first_echo == first_reply, summary
