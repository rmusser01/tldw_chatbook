"""AudioBook auto-chapter detection must not crash the paste box (task-15478).

`on_text_area_changed` (the handler behind the "Paste Text" content-preview
box) used to gate chapter detection on `query_one("#auto-chapters-switch",
Switch)`. That id was composed nowhere: the "Chapter Settings" collapsible
that used to own it was replaced by `ChapterEditorWidget` in commit
`256911ea6` ("audiobook work", 2025-07-22), and the four query sites were
never updated. The result was a `NoMatches` raised on every keystroke in the
paste box.

The fix keeps auto-detection (its switch defaulted to on, and there is no UI
left to turn it off) but:
  - drops the dead query on all four sites;
  - moves the keystroke-path call behind a debounce timer, since detection
    walks the *entire* pasted text with `ChapterDetector.detect_chapters`
    and pops a notify toast -- unacceptable once per keystroke;
  - moves the detection itself off the event loop entirely (review round
    2): `ChapterDetector.detect_chapters` was benchmarked at ~19ms/90k
    words, ~60ms/300k, ~200ms on a 6MB paste -- past the repo's 100ms
    worker budget for exactly the large pastes an audiobook feature invites
    -- so it now runs in a `@work(thread=True)` worker with results
    marshaled back via `call_from_thread`, for all four call sites, not
    just the debounced one;
  - only pops the "Detected N chapters" toast when N actually changes since
    the last toast, so a debounced re-paste that keeps re-running detection
    does not spam a toast per settle;
  - (review round 3) guards against a slower, superseded detection
    overwriting a faster, newer one with a monotonically-increasing
    dispatch generation (`exclusive=True` alone cannot interrupt an
    already-running OS thread);
  - (review round 3) resets the toast-dedup memory on every new "Import
    Content" action, so a genuinely new import that happens to detect the
    same chapter count as a previous session still gets its own toast.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import TextArea

from tldw_chatbook.TTS.audiobook_generator import Chapter, ChapterDetector
from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield AudioBookGenerationWidget()


def _make_large_book(n_words: int, *, words_per_chapter: int = 2000) -> str:
    """Build synthetic book text with a "Chapter N" header every so often.

    Mirrors the shape the reviewer benchmarked ChapterDetector against
    (~19ms/90k words, ~60ms/300k, ~200ms on a 6MB paste): plain prose lines
    interspersed with chapter markers the detector's regexes actually match,
    not degenerate all-blank or all-matching input.

    The default density (one chapter every 2000 words: 999 chapters for a
    3,000,000-word book) is the ORIGINAL one. Task-15478 temporarily reduced
    it to 60,000 words/chapter because populating that many chapter rows in
    one reactive update intermittently tripped a then-unowned race in
    `ChapterEditorWidget`/`Select`'s mount sequence (observed once in a
    full-file run: `NoMatches: No nodes match 'SelectOverlay'` out of the
    remount's Select). Task-15773 fixed that race at the source -- `chapters`
    no longer recomposes the widget, so a population mounts nothing -- and
    restored this density; the dedicated regression tests live in
    `Tests/Widgets/test_chapter_editor_widget_population_race.py`.
    """
    words_per_line = 12
    lines: list[str] = []
    total = 0
    chapter = 1
    while total < n_words:
        if total % words_per_chapter == 0:
            lines.append(f"Chapter {chapter}")
            chapter += 1
        lines.append(" ".join(["word"] * words_per_line))
        total += words_per_line
    return "\n".join(lines)


def _stub_chapter(number: int) -> Chapter:
    return Chapter(
        number=number,
        title=f"Chapter {number}",
        content="Some words go here for this chapter.",
        start_position=0,
        end_position=0,
    )


@pytest.mark.asyncio
async def test_typing_in_the_paste_box_raises_no_exception():
    """Regression repro: this used to raise NoMatches on every keystroke.

    Simulates real typing (one `TextArea.Changed` message per keypress) in
    the enabled content-preview TextArea and asserts the app is still alive
    and the widget observed the content afterward.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        content_preview = app.query_one("#content-preview", TextArea)
        content_preview.disabled = False
        content_preview.focus()

        for ch in "Hi there":
            await pilot.press(ch if ch != " " else "space")

        await pilot.pause()

        assert widget.content_text == "Hi there"


@pytest.mark.asyncio
async def test_a_burst_of_keystrokes_runs_detection_once_after_it_settles():
    """Detection must run off the keystroke path (task-15478 AC #2).

    A burst of changes inside the debounce window must not call
    `_detect_chapters` synchronously; it should fire exactly once, only
    after the input goes quiet for the debounce interval.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        content_preview = app.query_one("#content-preview", TextArea)
        content_preview.disabled = False

        detect = Mock()
        widget._detect_chapters = detect

        debounce = widget._CHAPTER_DETECT_DEBOUNCE_SECONDS

        # A rapid burst: several changes, each well inside the debounce
        # window of the previous one.
        content_preview.text = "Chapter 1"
        await pilot.pause(debounce / 4)
        content_preview.text = "Chapter 1\nChapter 2"
        await pilot.pause(debounce / 4)
        content_preview.text = "Chapter 1\nChapter 2\nChapter 3"

        # Still inside the window from the last change: no call yet.
        await pilot.pause(debounce / 4)
        assert detect.call_count == 0

        # Let the debounce interval elapse from the *last* change.
        await pilot.pause(debounce + 0.2)

        assert detect.call_count == 1


@pytest.mark.asyncio
async def test_file_import_still_auto_detects_without_the_missing_switch(tmp_path):
    """The three one-shot import paths (file/notes/conversation) are not on
    a keystroke path, so they keep running detection unconditionally --
    matching the removed switch's `value=True` default, with no dead query.

    Detection itself now runs in a background thread worker (review round
    2), so this waits for it to actually finish rather than asserting
    immediately after dispatch.
    """
    book_path = tmp_path / "book.txt"
    book_path.write_text("Chapter 1\nOnce upon a time.\n\nChapter 2\nThe end.\n")

    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)

        assert widget.detected_chapters == []

        widget._handle_file_selection(str(book_path))
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert widget.content_text.startswith("Chapter 1")
        assert len(widget.detected_chapters) >= 1


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_large_paste_chapter_detection():
    """The actual perf bug (task-15478 review round 2): chapter detection
    used to run synchronously on the event loop per debounce settle.
    `ChapterDetector.detect_chapters` was benchmarked at ~19ms/90k words,
    ~60ms/300k, ~200ms on a 6MB paste -- well past the repo's 100ms worker
    budget, for exactly the large pastes an audiobook feature invites.

    Load-robust design (task-15478 review round 3): an earlier version of
    this test asserted an absolute heartbeat-count threshold (`>= 10`),
    which failed 4/6 runs under real machine load (0-9 heartbeats) despite
    the fix being structurally sound -- the threshold conflated "is the
    mechanism correct" with "is this machine fast enough right now".
    Instead, this compares two arms measured back to back IN THE SAME RUN,
    under identical load: (A) a synchronous control call, made directly
    against `ChapterDetector` in the test itself with no `await` inside its
    wrapper, and (B) the real threaded call through the widget. Arm A is
    *guaranteed* zero heartbeats by construction, not by luck: a coroutine
    with no internal `await` point never yields control back to the event
    loop, so a concurrently-scheduled heartbeat task cannot run even once
    while it executes -- this is a hard property of Python's cooperative
    scheduling, true on any machine at any speed. Arm B only has to beat
    that guaranteed zero, which survives load far better than clearing an
    absolute count: as long as the loop gets scheduled even once during
    detection, the assertion holds. The 0-vs-N mutation contrast is thus
    built into the test itself rather than depended on as a threshold.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        content_preview = app.query_one("#content-preview", TextArea)
        content_preview.disabled = False

        # ~3,000,000 words / ~15MB -- comfortably north of the reviewer's 6MB
        # reference point (measured ~650-700ms of detection on this
        # machine), for a workload big enough that even a single heartbeat
        # tick landing during it is a meaningful signal.
        big_content = _make_large_book(3_000_000)
        widget.content_text = big_content

        async def _count_heartbeats(run_blocking_work) -> int:
            heartbeats = 0
            stop = asyncio.Event()

            async def _heartbeat() -> None:
                nonlocal heartbeats
                while not stop.is_set():
                    heartbeats += 1
                    await asyncio.sleep(0.005)

            hb_task = asyncio.create_task(_heartbeat())
            try:
                await run_blocking_work()
            finally:
                stop.set()
                await hb_task
            return heartbeats

        # Arm A: synchronous control. Same content, same event loop, same
        # machine load as arm B below -- this is what "the loop is frozen"
        # looks like, right now, on this exact machine.
        async def _synchronous_control() -> None:
            ChapterDetector.detect_chapters(big_content)

        sync_heartbeats = await _count_heartbeats(_synchronous_control)

        # Arm B: the real threaded call through the widget.
        async def _threaded_call() -> None:
            widget._detect_chapters()
            await app.workers.wait_for_complete()

        threaded_heartbeats = await _count_heartbeats(_threaded_call)

        # Diagnostic evidence (task-15478 review round 3, item 3's "prove
        # it" ask): visible with `pytest -s`, harmless otherwise.
        print(
            f"[heartbeat-seam] sync={sync_heartbeats} "
            f"threaded={threaded_heartbeats}"
        )

        assert widget.detected_chapters
        assert sync_heartbeats == 0, (
            f"synchronous control arm ticked {sync_heartbeats} times -- "
            "this should be structurally impossible (a coroutine with no "
            "internal await point cannot yield); investigate before "
            "trusting the comparison below"
        )
        assert threaded_heartbeats > sync_heartbeats, (
            f"threaded call ({threaded_heartbeats} heartbeats) did not "
            f"beat the synchronous control ({sync_heartbeats}) -- the "
            "event loop looks starved during chapter detection"
        )


@pytest.mark.asyncio
async def test_notify_only_fires_when_the_chapter_count_changes():
    """Minor fix (task-15478 review round 2): a debounced re-paste can
    re-run detection several times as the user keeps typing; before this,
    every settle popped its own "Detected N chapters" toast even when N
    hadn't moved. This is the "same session" direction: same-count spam
    within one session (no new `_import_content` action in between) must
    still be suppressed after round 3's dedup-reset fix.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        generation = widget._chapter_detect_generation

        notify = Mock()
        app.notify = notify

        # Same count twice in a row, no import action in between: only the
        # first call should toast.
        widget._apply_detected_chapters(
            [_stub_chapter(1), _stub_chapter(2)], generation
        )
        await pilot.pause()
        widget._apply_detected_chapters(
            [_stub_chapter(1), _stub_chapter(2)], generation
        )
        await pilot.pause()
        assert notify.call_count == 1

        # Count changes: a new toast fires.
        widget._apply_detected_chapters([_stub_chapter(1)], generation)
        await pilot.pause()
        assert notify.call_count == 2


@pytest.mark.asyncio
async def test_notify_dedup_resets_on_a_new_import_action():
    """The other direction (task-15478 review round 3): the dedup memory
    used to never reset, so a genuinely NEW import that happens to detect
    the same chapter count as whatever the previous session last toasted
    was silently un-toasted -- reproduced by the reviewer. `_import_content`
    (the single dispatcher all four source types funnel through, including
    "paste": `_import_from_paste` is the app's own signal that the user is
    about to start a fresh paste session) is the chosen reset seam -- see
    its docstring/comment for the justification.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        generation = widget._chapter_detect_generation

        # First "session": detect 2 chapters, toast once.
        widget._apply_detected_chapters(
            [_stub_chapter(1), _stub_chapter(2)], generation
        )
        await pilot.pause()

        notify = Mock()
        app.notify = notify

        # A brand-new import action begins. Call `_import_content` directly
        # rather than driving it through the "Import From" Select: that
        # widget's `options=[(id, label), ...]` are backwards from what
        # `_import_content`'s `if import_source == "file":` branches expect
        # (a separate, pre-existing bug, out of scope here -- confirmed via
        # `sel._options`, whose values are the display labels, not the
        # lowercase source ids, so no branch can ever match). The reset
        # line runs unconditionally as `_import_content`'s first action,
        # before any source branch is even reached, so calling it directly
        # still exercises exactly the reset behavior under test regardless
        # of that separate bug. `reset_mock()` guards against a toast from
        # a source branch that DOES fire, in case that bug gets fixed later
        # and this test isn't revisited.
        widget._import_content()
        await pilot.pause()
        notify.reset_mock()

        # The new session detects the SAME count (2) as the prior session's
        # last toast -- it must still fire, not be silently suppressed by
        # memory belonging to the previous, now-superseded session.
        widget._apply_detected_chapters(
            [_stub_chapter(1), _stub_chapter(2)], widget._chapter_detect_generation
        )
        await pilot.pause()
        assert notify.call_count == 1


@pytest.mark.asyncio
async def test_apply_detected_chapters_rejects_a_stale_generation():
    """The core guard, isolated from worker/thread timing (task-15478
    review round 3): `_apply_detected_chapters` must drop a result whose
    generation no longer matches the latest dispatched one, rather than
    overwrite whatever a newer, already-applied result set.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)

        # Simulate: two detections were dispatched (generations 1 then 2);
        # generation 2's (fresher) result already landed...
        widget._chapter_detect_generation = 2
        widget._apply_detected_chapters([_stub_chapter(2)], 2)
        await pilot.pause()
        assert [c.number for c in widget.detected_chapters] == [2]

        # ...then generation 1's (staler) result arrives late, exactly as
        # it would if its OS thread simply kept running past its own
        # supersession. It must be dropped, not applied.
        widget._apply_detected_chapters([_stub_chapter(999)], 1)
        await pilot.pause()
        assert [c.number for c in widget.detected_chapters] == [2]


@pytest.mark.asyncio
async def test_a_slower_superseded_detection_never_overwrites_a_faster_one(
    monkeypatch,
):
    """Reviewer's exact repro shape (task-15478 review round 3): dispatch
    slow-A then fast-B in one exclusive group; only B's result must stick.

    `exclusive=True` on `_detect_chapters_worker` cancels a *queued* worker
    in the same group; it cannot interrupt one already executing on its OS
    thread (`Worker.cancel()` cancels the wrapping asyncio Task, not the
    underlying thread). The reviewer reproduced 3/3 a slower, superseded
    worker's `call_from_thread` overwriting a newer result -- realistic
    because the three one-shot import paths have no debounce between them,
    and detection can run up to ~700ms.

    Uses real threads with `threading.Event` rendezvous rather than content
    size to force A slower than B, so the ordering is deterministic
    (load-independent) rather than a timing race: A blocks until B's result
    has actually landed, then proceeds and attempts its own (stale) apply.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)

        a_started = threading.Event()
        b_applied = threading.Event()

        def fake_detect_chapters(content: str):
            if content == "SLOW":
                a_started.set()
                # Block until B's (fresher) result has actually been
                # applied to the widget -- this is what "A's OS thread
                # outlives B, despite being logically superseded" looks
                # like, deterministically, independent of real detector
                # speed or machine load.
                if not b_applied.wait(timeout=5):
                    raise AssertionError("B's result never landed within 5s")
                return [_stub_chapter(999)]  # A's STALE result
            return [_stub_chapter(1)]  # B's FRESH result

        monkeypatch.setattr(
            ChapterDetector,
            "detect_chapters",
            staticmethod(fake_detect_chapters),
        )

        # Dispatch A ("slow"): starts running on a worker thread and blocks.
        widget.content_text = "SLOW"
        widget._detect_chapters()

        for _ in range(500):
            if a_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert a_started.is_set(), "worker A never started"

        # Dispatch B ("fast"), in the same exclusive group as A. Its own
        # detect_chapters call returns immediately (no blocking branch).
        widget.content_text = "FAST"
        widget._detect_chapters()

        for _ in range(500):
            if widget.detected_chapters and widget.detected_chapters[0].number == 1:
                break
            await asyncio.sleep(0.01)
        assert (
            widget.detected_chapters and widget.detected_chapters[0].number == 1
        ), "B's (fast, newer) result never applied"

        # Release A. Its thread resumes, computes its STALE result, and
        # attempts to marshal it back -- this must be dropped.
        b_applied.set()
        await asyncio.sleep(0.3)

        assert widget.detected_chapters and widget.detected_chapters[0].number == 1, (
            "A's stale result overwrote B's newer one: "
            f"detected_chapters={[c.number for c in widget.detected_chapters]}"
        )
