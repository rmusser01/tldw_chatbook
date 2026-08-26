"""TASK-15456: Console streaming defers syntax highlighting for open fences.

Textual's ``Markdown.append()`` re-parses only from the last completed
top-level block (textual/widgets/_markdown.py:1445-1509); while a code fence
is open, every append re-parses the whole fence-so-far and ``MarkdownFence``
re-runs Pygments over it synchronously on the event loop
(textual/widgets/_markdown.py:895-901) -- on every 0.2s streaming sync tick.

These tests cover:
- The conservative fence-parity detector in isolation (pure function).
- That a long, still-open fence defers ``MarkdownFence.highlight()`` work to
  a bounded number of calls, not one per tick.
- That the final rendered message (post fence-close / stream-end) is
  byte-identical to an unthrottled render of the same content.
- That plain multi-block prose streaming is untouched: still exactly one
  ``Markdown.append()`` per delta, no batching introduced.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Markdown
from textual.widgets._markdown import MarkdownFence

import tldw_chatbook.Widgets.Console.console_transcript as console_transcript_module
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleTranscript,
    _console_markdown_body_ends_in_open_fence,
)


class MarkdownHarness(App):
    CSS = "ConsoleTranscript { height: 40; }"

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _assistant(content: str, status: str = "streaming", **kwargs) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
        status=status,
        id=kwargs.pop("id", "a1"),
        **kwargs,
    )


class _FakeClock:
    """A controllable stand-in for ``time.monotonic`` so tests don't depend
    on real wall-clock timing (flaky under CI load)."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# The detector, in isolation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [
        ("", False),
        ("just prose, no fences at all", False),
        ("some `inline code` and more prose", False),
        ("```python\nprint(1)\n", True),  # opened, never closed
        ("```python\nprint(1)\n```\n", False),  # closed
        ("```python\nprint(1)\n```\nmore prose after", False),
        ("text\n\n```\ncode\n", True),
        ("~~~\ncode\n", True),  # tilde fence, open
        ("~~~\ncode\n~~~\n", False),  # tilde fence, closed
        ("````\ncode with ``` inside\n````\n", False),  # longer run closes
        ("````\ncode with ``` inside\n", True),  # longer run, still open
        ("  ```python\ncode\n  ```\n", False),  # <=3 leading spaces is valid
        ("    ```python\ncode\n", False),  # 4-space indent: not a fence line
        ("```one```\n", False),  # backticks with content after -- no close
    ],
)
def test_fence_detector_matches_expected_state(body: str, expected: bool) -> None:
    assert _console_markdown_body_ends_in_open_fence(body) is expected


def test_fence_detector_stays_closed_when_info_string_has_a_backtick() -> None:
    # CommonMark: a backtick-fence info string can't itself contain a
    # backtick, so this line can't validly open one. Conservative: treat as
    # ordinary text rather than guessing.
    body = "```so `me` info\nstill just text\n"
    assert _console_markdown_body_ends_in_open_fence(body) is False


def test_fence_detector_requires_matching_or_longer_run_to_close() -> None:
    # Opened with 4 backticks; a 3-backtick line inside can't close it.
    body = "````python\n```\nmore code\n"
    assert _console_markdown_body_ends_in_open_fence(body) is True
    # A 4-backtick (or longer) line does close it.
    assert (
        _console_markdown_body_ends_in_open_fence("````python\n```\nmore code\n````\n")
        is False
    )


# ---------------------------------------------------------------------------
# Evidence: highlight work during a long open-fence stream is bounded.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_open_fence_stream_defers_highlight_work(monkeypatch):
    """AC#1 evidence: streaming a long single code fence no longer
    re-highlights the full fence every tick.

    Without the throttle, each of the 30 simulated ticks below calls
    ``Markdown.append()`` while the body still ends inside the open fence,
    and every such call constructs a fresh ``MarkdownFence`` (re-running
    Pygments over the whole fence-so-far) -- counted here via
    ``MarkdownFence.__init__`` invocations, which fire exactly once per
    fence (re)parse regardless of anything else touching the DOM. (An
    earlier version of this test counted ``MarkdownFence.highlight()``
    calls directly and was contaminated by an unrelated source: Textual's
    ``Stylesheet.apply`` calls ``notify_style_update()`` -- which also
    calls ``highlight()`` -- on every node in a CSS-reapplied subtree,
    which this test's per-tick header/class updates trigger regardless of
    whether the fence content changed. Counting reconstructions instead of
    raw highlight calls isolates the mechanism this task actually changes.)
    This test fails (would be "born red") against that shape: 30 ticks
    would produce 30 reconstructions, violating the bound asserted below.
    """
    clock = _FakeClock()
    monkeypatch.setattr(console_transcript_module, "monotonic", clock)

    highlight_calls: list[int] = []
    original_init = MarkdownFence.__init__

    def counting_init(self, markdown, token, code):
        highlight_calls.append(len(code))
        return original_init(self, markdown, token, code)

    monkeypatch.setattr(MarkdownFence, "__init__", counting_init)

    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        body = "```python\n"
        transcript.set_messages([_assistant(body)])
        await transcript.refresh_messages()
        await pilot.pause()
        highlight_calls.clear()  # ignore the initial full-parse mount

        TICKS = 30
        for i in range(TICKS):
            body += f"x_{i} = {i}\n"
            transcript.set_messages([_assistant(body)])
            await transcript.refresh_messages()
            await pilot.pause()
            clock.advance(0.2)  # matches the real 0.2s transcript sync tick

        # Bounded: nowhere near one highlight call per tick. The throttle's
        # deadline is 1.0s at a simulated 0.2s cadence, so ~5 ticks land
        # between flushes -- generous headroom below TICKS either way.
        assert 0 < len(highlight_calls) < TICKS // 2, highlight_calls

        # Close the fence; the closing delta must flush immediately
        # (fence-close is a "not deferrable" transition), independent of
        # the deadline.
        calls_before_close = len(highlight_calls)
        body += "```\n"
        transcript.set_messages([_assistant(body, status="complete")])
        await transcript.refresh_messages()
        await pilot.pause()
        assert len(highlight_calls) == calls_before_close + 1

        row = transcript.query_one("#console-message-a1", ConsoleMarkdownMessage)
        markdown = row.query_one(Markdown)
        assert markdown.source == body


@pytest.mark.asyncio
async def test_no_open_fence_never_defers(monkeypatch):
    """A message that never opens a fence must never accumulate a pending
    buffer -- the deferral path is scoped exactly to open-fence streaming."""
    clock = _FakeClock()
    monkeypatch.setattr(console_transcript_module, "monotonic", clock)

    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        body = "intro "
        transcript.set_messages([_assistant(body)])
        await transcript.refresh_messages()
        await pilot.pause()

        for i in range(10):
            body += f"word{i} "
            transcript.set_messages([_assistant(body)])
            await transcript.refresh_messages()
            await pilot.pause()
            # No wall-clock advance: if this content were mistakenly treated
            # as fence-interior, the whole 1.0s deadline would still be
            # unexpired and everything would wrongly stay buffered.

        row = transcript.query_one("#console-message-a1", ConsoleMarkdownMessage)
        assert row._pending_fence_delta == ""
        assert row._fence_defer_deadline is None
        markdown = row.query_one(Markdown)
        assert markdown.source == body


# ---------------------------------------------------------------------------
# AC#2: final rendered output identical to an unthrottled render.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_render_after_fence_close_matches_unthrottled_render(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(console_transcript_module, "monotonic", clock)

    final_body = (
        "Here is an example:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n\n"
        "for i in range(5):\n"
        "    print(add(i, i))\n"
        "```\n\n"
        "That's the whole snippet."
    )

    # Path A: stream it in incrementally, character-by-character-ish chunks,
    # advancing the fake clock so several deadline flushes fire mid-fence.
    app_a = MarkdownHarness()
    async with app_a.run_test() as pilot_a:
        transcript_a = app_a.query_one(ConsoleTranscript)
        streamed = ""
        transcript_a.set_messages([_assistant(streamed)])
        await transcript_a.refresh_messages()
        await pilot_a.pause()
        chunk_size = 7
        for start in range(0, len(final_body), chunk_size):
            streamed = final_body[: start + chunk_size]
            transcript_a.set_messages([_assistant(streamed)])
            await transcript_a.refresh_messages()
            await pilot_a.pause()
            clock.advance(0.2)
        # Final chunk plus stream completion.
        transcript_a.set_messages([_assistant(final_body, status="complete")])
        await transcript_a.refresh_messages()
        await pilot_a.pause()

        row_a = transcript_a.query_one("#console-message-a1", ConsoleMarkdownMessage)
        markdown_a = row_a.query_one(Markdown)
        assert markdown_a.source == final_body
        fences_a = list(markdown_a.query(MarkdownFence))
        assert len(fences_a) == 1

    # Path B: render the identical final content directly, never streamed.
    app_b = MarkdownHarness()
    async with app_b.run_test() as pilot_b:
        transcript_b = app_b.query_one(ConsoleTranscript)
        transcript_b.set_messages([_assistant(final_body, status="complete", id="a1")])
        await transcript_b.refresh_messages()
        await pilot_b.pause()

        row_b = transcript_b.query_one("#console-message-a1", ConsoleMarkdownMessage)
        markdown_b = row_b.query_one(Markdown)
        assert markdown_b.source == final_body
        fences_b = list(markdown_b.query(MarkdownFence))
        assert len(fences_b) == 1

    # Same source, same code captured per fence, and identical highlighted
    # output (same code + language + theme -> deterministic Pygments output).
    assert fences_a[0].code == fences_b[0].code
    assert fences_a[0].lexer == fences_b[0].lexer
    assert fences_a[0]._highlighted_code.plain == fences_b[0]._highlighted_code.plain
    assert list(fences_a[0]._highlighted_code.spans) == list(
        fences_b[0]._highlighted_code.spans
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["complete", "stopped", "failed"])
async def test_stream_ending_mid_open_fence_flushes_buffered_tail(
    monkeypatch, terminal_status: str
):
    """AC#2 regression (fix round 1): a stream that ends while its fence is
    STILL OPEN must not strand buffered content.

    Reproduces the exact production path a code reviewer caught the first
    version of this task missing: the transcript reconciler calls
    ``sync_message`` on a status-only change (status is part of the row
    signature), which is exactly what "user presses Stop mid-code-block" or
    a length-limit cutoff looks like -- the message body stops growing and
    the NEXT (and only remaining) call carries the SAME body text with a
    new terminal status. `sync_message`'s "body unchanged" fast path used
    to return before ever looking at the pending fence buffer, permanently
    stranding whatever was deferred. This drives the body deep enough into
    an open fence that the throttle is guaranteed to be holding a
    non-empty buffer (the fake clock never advances far enough to cross
    the deadline on its own), then flips status with NO further body
    change -- the one shape that previously lost content.
    """
    clock = _FakeClock()
    monkeypatch.setattr(console_transcript_module, "monotonic", clock)

    final_body = (
        "Cutting off mid-block:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n\n"
        "for i in range(3):\n"
        "    print(add(i, i))\n"
    )  # deliberately never closed -- fence is still open when streaming ends

    app_a = MarkdownHarness()
    async with app_a.run_test() as pilot_a:
        transcript_a = app_a.query_one(ConsoleTranscript)
        streamed = ""
        transcript_a.set_messages([_assistant(streamed)])
        await transcript_a.refresh_messages()
        await pilot_a.pause()

        chunk_size = 5
        for start in range(0, len(final_body), chunk_size):
            streamed = final_body[: start + chunk_size]
            transcript_a.set_messages([_assistant(streamed)])
            await transcript_a.refresh_messages()
            await pilot_a.pause()
            # Advance well under the 1.0s deadline so nothing flushes on
            # its own -- only the terminal-status transition below can be
            # responsible for delivering the tail.
            clock.advance(0.05)

        row_a = transcript_a.query_one("#console-message-a1", ConsoleMarkdownMessage)
        # Sanity: the throttle really is holding something back at this
        # point, and the widget's own source really is behind the logical
        # body -- otherwise this test would pass for the wrong reason.
        assert row_a._pending_fence_delta != ""
        markdown_a = row_a.query_one(Markdown)
        assert markdown_a.source != streamed

        # Same body text, terminal status, no further growth -- the exact
        # shape that used to hit the early return and strand the buffer.
        transcript_a.set_messages([_assistant(streamed, status=terminal_status)])
        await transcript_a.refresh_messages()
        await pilot_a.pause()

        assert row_a._pending_fence_delta == ""
        assert markdown_a.source == final_body
        fences_a = list(markdown_a.query(MarkdownFence))
        assert len(fences_a) == 1

    # Reference: render the identical final (still-open-fence) content
    # directly, never streamed/throttled at all.
    app_b = MarkdownHarness()
    async with app_b.run_test() as pilot_b:
        transcript_b = app_b.query_one(ConsoleTranscript)
        transcript_b.set_messages(
            [_assistant(final_body, status=terminal_status, id="a1")]
        )
        await transcript_b.refresh_messages()
        await pilot_b.pause()

        row_b = transcript_b.query_one("#console-message-a1", ConsoleMarkdownMessage)
        markdown_b = row_b.query_one(Markdown)
        assert markdown_b.source == final_body
        fences_b = list(markdown_b.query(MarkdownFence))
        assert len(fences_b) == 1

    assert fences_a[0].code == fences_b[0].code
    assert fences_a[0].lexer == fences_b[0].lexer
    assert fences_a[0]._highlighted_code.plain == fences_b[0]._highlighted_code.plain
    assert list(fences_a[0]._highlighted_code.spans) == list(
        fences_b[0]._highlighted_code.spans
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["complete", "stopped", "failed"])
async def test_status_only_change_with_no_pending_buffer_is_a_no_op(
    monkeypatch, terminal_status: str
):
    """Companion to the regression above: when there is nothing buffered,
    a status-only change must stay a true no-op (no spurious append)."""
    clock = _FakeClock()
    monkeypatch.setattr(console_transcript_module, "monotonic", clock)

    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        body = "plain prose, no fence"
        transcript.set_messages([_assistant(body)])
        await transcript.refresh_messages()
        await pilot.pause()

        row = transcript.query_one("#console-message-a1", ConsoleMarkdownMessage)
        markdown = row.query_one(Markdown)
        append_calls = []
        original_append = markdown.append
        monkeypatch.setattr(
            markdown,
            "append",
            lambda text: append_calls.append(text) or original_append(text),
        )

        assert row._pending_fence_delta == ""
        transcript.set_messages([_assistant(body, status=terminal_status)])
        await transcript.refresh_messages()
        await pilot.pause()

        assert append_calls == []


# ---------------------------------------------------------------------------
# AC#3: multi-block prose streaming is untouched.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_block_prose_streaming_appends_every_delta(monkeypatch):
    """No fence anywhere in the stream: every tick's delta is still applied
    immediately via exactly one ``Markdown.append()`` call, same as before
    this task -- batching is scoped to open-fence bodies only."""
    clock = _FakeClock()
    monkeypatch.setattr(console_transcript_module, "monotonic", clock)

    app = MarkdownHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        body = "# Heading\n\n"
        transcript.set_messages([_assistant(body)])
        await transcript.refresh_messages()
        await pilot.pause()

        row = transcript.query_one("#console-message-a1", ConsoleMarkdownMessage)
        markdown = row.query_one(Markdown)
        append_calls = []
        original_append = markdown.append
        monkeypatch.setattr(
            markdown,
            "append",
            lambda text: append_calls.append(text) or original_append(text),
        )

        paragraphs = [
            "First paragraph grows one word at a time. ",
            "\n\nSecond paragraph, a fresh block. ",
            "\n\n- item one\n- item two\n",
        ]
        for paragraph in paragraphs:
            for word in paragraph.split(" "):
                body += word + " "
                transcript.set_messages([_assistant(body)])
                await transcript.refresh_messages()
                await pilot.pause()
                # No clock advance -- if prose were mistakenly throttled by
                # the fence deadline, a 0-advance run would defer everything
                # into one call instead of one-per-delta.

        expected_ticks = sum(len(p.split(" ")) for p in paragraphs)
        assert len(append_calls) == expected_ticks
