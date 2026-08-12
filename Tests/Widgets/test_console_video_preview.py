"""ConsoleVideoPreview lifecycle + ConsoleVideoCard preview mount tests."""

from __future__ import annotations

import asyncio
from threading import Event, Thread, get_ident
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from loguru import logger
from PIL import Image
from textual.app import App, ComposeResult
from textual.geometry import Region
from textual.widgets import Button
from textual.worker import Worker, WorkerState

from tldw_chatbook.Widgets.Console.console_video_card import (
    ConsoleVideoCard,
    ConsoleVideoCardSpec,
)
from tldw_chatbook.Widgets.Console.console_video_preview import (
    ConsoleVideoPreview,
    progress_line,
)
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata


PRIVATE_PATH = "/private/PRIVATE-PATH-SENTINEL.mp4"
PRIVATE_ERROR = "PRIVATE-ERROR-SENTINEL"


class _ObservedPreview(ConsoleVideoPreview):
    """Preview that records its own non-bubbling worker state messages."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.worker_states: list[WorkerState] = []
        self.seen_workers: list[Worker[Any]] = []
        super().__init__(*args, **kwargs)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        self.worker_states.append(event.state)
        if event.worker not in self.seen_workers:
            self.seen_workers.append(event.worker)


class _PreviewApp(App[None]):
    """Small mounted host that exposes a second independently clickable action."""

    def __init__(
        self,
        *,
        preview_class: type[_ObservedPreview] = _ObservedPreview,
        two_previews: bool = False,
    ) -> None:
        super().__init__()
        self.preview = preview_class(PRIVATE_PATH, duration_seconds=6.0)
        self.other_preview = (
            preview_class("/private/other.mp4", duration_seconds=4.0)
            if two_previews
            else None
        )
        self.action_count = 0
        self.action_seen = Event()

    def compose(self) -> ComposeResult:
        yield self.preview
        if self.other_preview is not None:
            yield self.other_preview
        yield Button("Other action", id="other-action")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "other-action":
            self.action_count += 1
            self.action_seen.set()


class _Source:
    """Minimal decoder seam; individual tests override only their needed phase."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.closed = Event()
        self.close_thread: int | None = None

    def check_eligible(self) -> tuple[bool, str]:
        return True, ""

    def iter_frames(self) -> Iterator[tuple[float, Any]]:
        return iter(())

    def close(self) -> None:
        self.close_thread = get_ident()
        self.closed.set()


@pytest.fixture(autouse=True)
def _isolate_registry():
    ConsoleVideoPreview._active = None
    yield
    ConsoleVideoPreview._active = None


def _patch_source(monkeypatch: pytest.MonkeyPatch, factory: Any) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_video_preview.av_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_video_preview.AvFrameSource",
        factory,
    )


async def _wait(event: Event, *, timeout: float = 2.0) -> None:
    assert await asyncio.to_thread(event.wait, timeout), (
        "synchronization event timed out"
    )


async def _finish_workers(app: _PreviewApp, pilot: Any) -> None:
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert app.preview.seen_workers
    assert WorkerState.ERROR not in app.preview.worker_states
    assert all(
        worker.state is not WorkerState.ERROR for worker in app.preview.seen_workers
    )


def _responsive_release_watchdog(
    entered: Event,
    responsive_action: Event,
    release: Event,
    result: list[bool],
) -> Thread:
    """Release a deliberately blocked seam even when the UI incorrectly blocks."""

    def watch() -> None:
        if not entered.wait(2.0):
            result.append(False)
        else:
            result.append(responsive_action.wait(0.5))
        release.set()

    thread = Thread(target=watch, daemon=True)
    thread.start()
    return thread


def _assert_sanitized(
    records: list[str],
    phase: str,
    error_type: str = "RuntimeError",
) -> None:
    matching = [record for record in records if "component=inline_preview" in record]
    assert matching
    assert any(f"phase={phase}" in record for record in matching)
    assert any(f"error_type={error_type}" in record for record in matching)
    joined = "\n".join(records)
    assert PRIVATE_ERROR not in joined
    assert PRIVATE_PATH not in joined
    assert "Traceback" not in joined


# -- progress line (pure) ----------------------------------------------------


def test_progress_line_bar_and_clock():
    line = progress_line(3.0, 6.0, width=10)
    assert line.startswith("▓▓▓▓▓░░░░░")
    assert "0:03 / 0:06" in line


def test_progress_line_unknown_duration():
    assert progress_line(3.0, None) == "0:03 / --:--"


def test_default_state_is_poster(monkeypatch):
    _patch_source(monkeypatch, _Source)
    preview = ConsoleVideoPreview(PRIVATE_PATH, duration_seconds=6.0)
    assert preview.state == "poster"
    assert preview._source is None
    assert "click to play" in preview._poster_text()


# -- real Textual worker / bridge regressions -------------------------------


@pytest.mark.asyncio
async def test_real_click_renders_frame_reaches_eof_and_keeps_app_responsive(
    monkeypatch,
):
    sources: list[_Source] = []

    class FrameSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            sources.append(self)
            self.decode_thread: int | None = None

        def iter_frames(self):
            self.decode_thread = get_ident()
            yield 1.25, Image.new("RGB", (4, 3), "#13579b")

    _patch_source(monkeypatch, FrameSource)
    app = _PreviewApp()
    ui_thread = get_ident()
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _finish_workers(app, pilot)
        await pilot.click("#other-action")

        assert app.action_count == 1
        assert app.preview._pixels is not None
        assert app.preview._position == 1.25
        assert app.preview.state == "paused"
        assert app.preview._source is None
        assert app.preview._offscreen_timer is None
        assert ConsoleVideoPreview._active is None
        assert sources[0].decode_thread not in {None, ui_thread}
        assert sources[0].close_thread == sources[0].decode_thread


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["construct", "probe"])
async def test_blocked_open_or_probe_does_not_block_click_or_other_ui_action(
    monkeypatch,
    blocked_phase,
):
    entered = Event()
    release = Event()
    responsive: list[bool] = []

    class BlockedSource(_Source):
        def __init__(self, path: str) -> None:
            if blocked_phase == "construct":
                entered.set()
                assert release.wait(2.0)
            super().__init__(path)

        def check_eligible(self):
            if blocked_phase == "probe":
                entered.set()
                assert release.wait(2.0)
            return True, ""

    _patch_source(monkeypatch, BlockedSource)
    app = _PreviewApp()
    watchdog = _responsive_release_watchdog(
        entered,
        app.action_seen,
        release,
        responsive,
    )
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await pilot.click("#other-action")
        await _finish_workers(app, pilot)

    watchdog.join(1.0)
    assert not watchdog.is_alive()
    assert responsive == [True]
    assert app.action_count == 1


@pytest.mark.asyncio
async def test_pause_during_blocked_decode_is_nonblocking_and_worker_closes_source(
    monkeypatch,
):
    decode_entered = Event()
    release = Event()
    close_entered = Event()
    responsive: list[bool] = []
    sources: list[_Source] = []

    class BlockedDecodeSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            sources.append(self)

        def iter_frames(self):
            decode_entered.set()
            assert release.wait(2.0)
            if False:
                yield 0.0, None

        def close(self) -> None:
            self.close_thread = get_ident()
            close_entered.set()
            assert release.wait(2.0)
            self.closed.set()

    _patch_source(monkeypatch, BlockedDecodeSource)
    app = _PreviewApp()
    ui_thread = get_ident()
    watchdog = _responsive_release_watchdog(
        decode_entered,
        app.action_seen,
        release,
        responsive,
    )
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _wait(decode_entered)
        await pilot.click(f"#{app.preview.id}")
        await pilot.click("#other-action")
        await _wait(close_entered)
        await _finish_workers(app, pilot)

        assert app.preview.state == "paused"
        assert app.preview._source is None
        assert app.preview._offscreen_timer is None
        assert ConsoleVideoPreview._active is None

    watchdog.join(1.0)
    assert responsive == [True]
    assert sources[0].close_thread not in {None, ui_thread}


@pytest.mark.asyncio
async def test_resume_generation_rejects_late_old_frame_eof_and_cleanup(monkeypatch):
    old_entered = Event()
    release_old = Event()
    old_closed = Event()
    new_entered = Event()
    release_new = Event()
    new_closed = Event()
    sources: list[_Source] = []

    class OldSource(_Source):
        def iter_frames(self):
            old_entered.set()
            assert release_old.wait(2.0)
            yield 1.0, Image.new("RGB", (2, 2), "red")

        def close(self) -> None:
            super().close()
            old_closed.set()

    class NewSource(_Source):
        def iter_frames(self):
            yield 2.0, Image.new("RGB", (2, 2), "blue")
            new_entered.set()  # the preceding bridge has returned: frame rendered
            assert release_new.wait(2.0)

        def close(self) -> None:
            super().close()
            new_closed.set()

    def source_factory(path: str) -> _Source:
        source = OldSource(path) if not sources else NewSource(path)
        sources.append(source)
        return source

    _patch_source(monkeypatch, source_factory)
    app = _PreviewApp()
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _wait(old_entered)
        old_run = app.preview._run
        assert old_run is not None and old_run.generation == 1
        await pilot.click(f"#{app.preview.id}")  # pause generation 1
        await pilot.click(f"#{app.preview.id}")  # start generation 2
        await _wait(new_entered)

        replacement_run = app.preview._run
        assert replacement_run is not None and replacement_run.generation == 2
        replacement_source = app.preview._source
        replacement_pixels = app.preview._pixels
        assert app.preview._position == 2.0

        release_old.set()
        await _wait(old_closed)
        await pilot.pause()

        assert app.preview.state == "playing"
        assert app.preview._run is replacement_run
        assert app.preview._source is replacement_source
        assert app.preview._pixels is replacement_pixels
        assert app.preview._position == 2.0
        assert ConsoleVideoPreview._active is app.preview

        release_new.set()
        await _wait(new_closed)
        await _finish_workers(app, pilot)
        assert app.preview.state == "paused"


class _ControllableRegionPreview(_ObservedPreview):
    force_offscreen = False

    @property
    def region(self):
        if self.force_offscreen:
            return Region(0, -2, 1, 1)
        return super().region


@pytest.mark.asyncio
async def test_old_timer_callback_cannot_pause_replacement_generation(monkeypatch):
    entered = [Event(), Event()]
    releases = [Event(), Event()]
    sources: list[_Source] = []

    class HeldSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self.index = len(sources)
            sources.append(self)

        def iter_frames(self):
            entered[self.index].set()
            assert releases[self.index].wait(2.0)
            if False:
                yield 0.0, None

    _patch_source(monkeypatch, HeldSource)
    app = _PreviewApp(preview_class=_ControllableRegionPreview)
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _wait(entered[0])
        old_timer = app.preview._offscreen_timer
        assert old_timer is not None
        old_callback = old_timer._callback
        assert old_callback is not None

        await pilot.click(f"#{app.preview.id}")
        await pilot.click(f"#{app.preview.id}")
        await _wait(entered[1])
        replacement_run = app.preview._run
        replacement_source = app.preview._source
        replacement_timer = app.preview._offscreen_timer
        app.preview.force_offscreen = True

        old_callback()

        assert app.preview.state == "playing"
        assert app.preview._run is replacement_run
        assert app.preview._source is replacement_source
        assert app.preview._offscreen_timer is replacement_timer
        assert ConsoleVideoPreview._active is app.preview

        releases[0].set()
        releases[1].set()
        await _wait(sources[0].closed)
        await _wait(sources[1].closed)
        await _finish_workers(app, pilot)


@pytest.mark.asyncio
async def test_stale_eof_cannot_pause_replacement_generation(monkeypatch):
    old_entered = Event()
    release_old = Event()
    new_entered = Event()
    release_new = Event()
    sources: list[_Source] = []

    class OldEofSource(_Source):
        def iter_frames(self):
            old_entered.set()
            assert release_old.wait(2.0)
            if False:
                yield 0.0, None

    class NewSource(_Source):
        def iter_frames(self):
            yield 2.0, Image.new("RGB", (2, 2), "blue")
            new_entered.set()
            assert release_new.wait(2.0)

    def source_factory(path: str) -> _Source:
        source = OldEofSource(path) if not sources else NewSource(path)
        sources.append(source)
        return source

    _patch_source(monkeypatch, source_factory)
    app = _PreviewApp()
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _wait(old_entered)
        await pilot.click(f"#{app.preview.id}")
        await pilot.click(f"#{app.preview.id}")
        await _wait(new_entered)
        replacement_run = app.preview._run
        replacement_source = app.preview._source

        release_old.set()
        await _wait(sources[0].closed)
        await pilot.pause()

        assert app.preview.state == "playing"
        assert app.preview._run is replacement_run
        assert app.preview._source is replacement_source
        assert app.preview._eligible
        assert ConsoleVideoPreview._active is app.preview

        release_new.set()
        await _wait(sources[1].closed)
        await _finish_workers(app, pilot)


@pytest.mark.asyncio
async def test_stale_decode_failure_cannot_degrade_replacement_generation(
    monkeypatch,
):
    old_entered = Event()
    release_old = Event()
    new_entered = Event()
    release_new = Event()
    sources: list[_Source] = []

    class OldFailureSource(_Source):
        def iter_frames(self):
            old_entered.set()
            assert release_old.wait(2.0)
            raise RuntimeError(PRIVATE_ERROR)
            yield

    class NewSource(_Source):
        def iter_frames(self):
            yield 2.0, Image.new("RGB", (2, 2), "blue")
            new_entered.set()
            assert release_new.wait(2.0)

    def source_factory(path: str) -> _Source:
        source = OldFailureSource(path) if not sources else NewSource(path)
        sources.append(source)
        return source

    _patch_source(monkeypatch, source_factory)
    app = _PreviewApp()
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _wait(old_entered)
        await pilot.click(f"#{app.preview.id}")
        await pilot.click(f"#{app.preview.id}")
        await _wait(new_entered)
        replacement_run = app.preview._run
        replacement_source = app.preview._source

        release_old.set()
        await _wait(sources[0].closed)
        await pilot.pause()

        assert app.preview.state == "playing"
        assert app.preview._run is replacement_run
        assert app.preview._source is replacement_source
        assert app.preview._eligible
        assert "preview stopped" not in app.preview._poster_text().lower()
        assert ConsoleVideoPreview._active is app.preview

        release_new.set()
        await _wait(sources[1].closed)
        await _finish_workers(app, pilot)


@pytest.mark.asyncio
async def test_current_run_rejects_wrong_source_for_frame_eof_and_failure(monkeypatch):
    entered = Event()
    release = Event()
    sources: list[_Source] = []

    class HeldSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            sources.append(self)

        def iter_frames(self):
            entered.set()
            assert release.wait(2.0)
            if False:
                yield 0.0, None

    _patch_source(monkeypatch, HeldSource)
    app = _PreviewApp()
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _wait(entered)
        run = app.preview._run
        source = app.preview._source
        timer = app.preview._offscreen_timer
        wrong_source = _Source("/private/wrong-source.mp4")
        assert run is not None and source is not None

        assert not app.preview._show_frame(
            run,
            wrong_source,
            9.0,
            Image.new("RGB", (2, 2), "red"),
        )
        assert not app.preview._finish_run(run, wrong_source)
        assert not app.preview._degrade_run(run, wrong_source)

        assert app.preview.state == "playing"
        assert app.preview._run is run
        assert app.preview._source is source
        assert app.preview._offscreen_timer is timer
        assert app.preview._position is None
        assert app.preview._pixels is None
        assert app.preview._eligible
        assert ConsoleVideoPreview._active is app.preview

        release.set()
        await _wait(sources[0].closed)
        await _finish_workers(app, pilot)


@pytest.mark.asyncio
async def test_immediate_eof_observes_timer_before_decode_and_cleans_every_owner(
    monkeypatch,
):
    timer_seen: list[bool] = []
    sources: list[_Source] = []
    app_holder: dict[str, _PreviewApp] = {}

    class ImmediateEofSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            sources.append(self)

        def iter_frames(self):
            timer_seen.append(app_holder["app"].preview._offscreen_timer is not None)
            if False:
                yield 0.0, None

    _patch_source(monkeypatch, ImmediateEofSource)
    app = app_holder["app"] = _PreviewApp()
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _finish_workers(app, pilot)

        assert timer_seen == [True]
        assert sources[0].closed.is_set()
        assert app.preview.state == "paused"
        assert app.preview._run is None
        assert app.preview._source is None
        assert app.preview._offscreen_timer is None
        assert ConsoleVideoPreview._active is None


@pytest.mark.asyncio
async def test_unmount_rejects_late_frame_and_eof_but_worker_closes_source(monkeypatch):
    decode_entered = Event()
    release = Event()
    sources: list[_Source] = []

    class LateSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            sources.append(self)

        def iter_frames(self):
            decode_entered.set()
            assert release.wait(2.0)
            yield 3.0, Image.new("RGB", (2, 2), "green")

    _patch_source(monkeypatch, LateSource)
    app = _PreviewApp()
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _wait(decode_entered)
        await app.preview.remove()
        assert not app.preview.is_attached
        release.set()
        await _wait(sources[0].closed)
        await _finish_workers(app, pilot)

        assert app.preview._pixels is None
        assert app.preview._position is None
        assert app.preview._run is None
        assert app.preview._source is None
        assert app.preview._offscreen_timer is None
        assert ConsoleVideoPreview._active is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_phase", "expected_phase"),
    [
        ("construct", "open"),
        ("probe", "open"),
        ("decode", "decode"),
        ("render", "render"),
    ],
)
async def test_open_decode_and_render_failures_degrade_without_leaking_private_data(
    monkeypatch,
    failure_phase,
    expected_phase,
):
    sources: list[_Source] = []

    class BadImage:
        def copy(self):
            raise RuntimeError(PRIVATE_ERROR)

    class FailingSource(_Source):
        def __init__(self, path: str) -> None:
            if failure_phase == "construct":
                raise RuntimeError(PRIVATE_ERROR)
            super().__init__(path)
            sources.append(self)

        def check_eligible(self):
            if failure_phase == "probe":
                raise RuntimeError(PRIVATE_ERROR)
            return True, ""

        def iter_frames(self):
            if failure_phase == "decode":
                raise RuntimeError(PRIVATE_ERROR)
            yield 0.5, BadImage()

    _patch_source(monkeypatch, FailingSource)
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        app = _PreviewApp()
        async with app.run_test() as pilot:
            await pilot.click(f"#{app.preview.id}")
            await _finish_workers(app, pilot)
            await pilot.click("#other-action")

            assert app.action_count == 1
            assert app.preview.state == "poster"
            assert not app.preview._eligible
            user_copy = app.preview._poster_text()
            assert "preview stopped" in user_copy.lower()
            assert "Play" in user_copy or "system player" in user_copy
            assert PRIVATE_ERROR not in user_copy
            assert PRIVATE_PATH not in user_copy
            if sources:
                assert sources[0].closed.is_set()
    finally:
        logger.remove(sink)

    _assert_sanitized(records, expected_phase)


@pytest.mark.asyncio
async def test_timer_stop_failure_is_contained_and_source_still_closes_on_worker(
    monkeypatch,
):
    decode_entered = Event()
    release = Event()
    sources: list[_Source] = []
    records: list[str] = []

    class BlockedSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            sources.append(self)

        def iter_frames(self):
            decode_entered.set()
            assert release.wait(2.0)
            if False:
                yield 0.0, None

    _patch_source(monkeypatch, BlockedSource)
    app = _PreviewApp()
    ui_thread = get_ident()
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        async with app.run_test() as pilot:
            await pilot.click(f"#{app.preview.id}")
            await _wait(decode_entered)
            timer = app.preview._offscreen_timer
            assert timer is not None
            timer_type = type(timer)
            original_stop = timer_type.stop

            def fail_selected_timer(self):
                if self is timer:
                    raise RuntimeError(PRIVATE_ERROR)
                return original_stop(self)

            monkeypatch.setattr(timer_type, "stop", fail_selected_timer)
            await pilot.click(f"#{app.preview.id}")
            release.set()
            await _wait(sources[0].closed)
            await _finish_workers(app, pilot)

            assert app.preview.state == "paused"
            assert app.preview._offscreen_timer is None
            assert app.preview._source is None
            assert sources[0].close_thread not in {None, ui_thread}
    finally:
        logger.remove(sink)

    _assert_sanitized(records, "cleanup")


@pytest.mark.asyncio
async def test_source_close_failure_is_contained_and_sanitized(monkeypatch):
    close_attempted = Event()
    sources: list[_Source] = []
    records: list[str] = []

    class CloseFailureSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            sources.append(self)

        def close(self) -> None:
            self.close_thread = get_ident()
            close_attempted.set()
            raise RuntimeError(PRIVATE_ERROR)

    _patch_source(monkeypatch, CloseFailureSource)
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        app = _PreviewApp()
        ui_thread = get_ident()
        async with app.run_test() as pilot:
            await pilot.click(f"#{app.preview.id}")
            await _wait(close_attempted)
            await _finish_workers(app, pilot)

            assert app.preview.state == "paused"
            assert app.preview._source is None
            assert app.preview._offscreen_timer is None
            assert app.preview._run is None
            assert close_attempted.is_set()
        assert close_attempted.is_set()
        assert app.preview._source is None
        assert sources[0].close_thread not in {None, ui_thread}
    finally:
        logger.remove(sink)

    _assert_sanitized(records, "cleanup")


@pytest.mark.asyncio
async def test_bridge_refusal_is_attempted_once_without_direct_ui_fallback(monkeypatch):
    accepted_directly = Event()
    sources: list[_Source] = []
    bridge_calls: list[tuple[str, int]] = []
    records: list[str] = []

    class BridgeObservedPreview(_ObservedPreview):
        def _accept_source(self, *args):
            accepted_directly.set()
            return super()._accept_source(*args)

    class RefusedSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            sources.append(self)

    _patch_source(monkeypatch, RefusedSource)
    app = _PreviewApp(preview_class=BridgeObservedPreview)
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        async with app.run_test() as pilot:

            def refuse(callback, *args, **kwargs):
                bridge_calls.append((callback.__name__, get_ident()))
                raise RuntimeError(PRIVATE_ERROR)

            monkeypatch.setattr(app, "call_from_thread", refuse)
            await pilot.click(f"#{app.preview.id}")
            await _wait(sources[0].closed)
            await _finish_workers(app, pilot)

            assert bridge_calls == [("_accept_source", sources[0].close_thread)]
            assert not accepted_directly.is_set()
            assert app.preview._source is None
            assert app.preview._pixels is None
    finally:
        logger.remove(sink)

    _assert_sanitized(records, "frame_dispatch")


# -- one-active and off-screen policy ---------------------------------------


@pytest.mark.asyncio
async def test_one_active_preview_rule_uses_real_workers(monkeypatch):
    entered = [Event(), Event()]
    releases = [Event(), Event()]
    sources: list[_Source] = []

    class HeldSource(_Source):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self.index = len(sources)
            sources.append(self)

        def iter_frames(self):
            entered[self.index].set()
            assert releases[self.index].wait(2.0)
            if False:
                yield 0.0, None

    _patch_source(monkeypatch, HeldSource)
    app = _PreviewApp(two_previews=True)
    assert app.other_preview is not None
    async with app.run_test() as pilot:
        await pilot.click(f"#{app.preview.id}")
        await _wait(entered[0])
        await pilot.click(f"#{app.other_preview.id}")
        await _wait(entered[1])

        assert app.preview.state == "paused"
        assert app.other_preview.state == "playing"
        assert ConsoleVideoPreview._active is app.other_preview

        releases[0].set()
        releases[1].set()
        await _finish_workers(app, pilot)


class _OffscreenPreview(ConsoleVideoPreview):
    def __init__(self, region, screen, **kwargs):
        super().__init__(**kwargs)
        self._fake_region = region
        self._fake_screen = screen

    @property
    def region(self):
        return self._fake_region

    @property
    def screen(self):
        return self._fake_screen

    @property
    def is_attached(self):
        return True


def _offscreen_widget(monkeypatch, bottom, top, height=24):
    _patch_source(monkeypatch, _Source)
    region = SimpleNamespace(bottom=bottom, top=top)
    screen = SimpleNamespace(size=SimpleNamespace(height=height))
    widget = _OffscreenPreview(
        region,
        screen,
        file_path=PRIVATE_PATH,
        duration_seconds=5.0,
    )
    widget.state = "playing"
    run = SimpleNamespace(cancelled=Event())
    source = _Source(PRIVATE_PATH)
    widget._run = run
    widget._source = source
    ConsoleVideoPreview._active = widget
    return widget, run, source


def test_offscreen_scroll_pauses(monkeypatch):
    widget, run, source = _offscreen_widget(monkeypatch, bottom=-10, top=-60)
    widget._pause_if_offscreen(run, source)
    assert widget.state == "paused"
    assert ConsoleVideoPreview._active is None


def test_onscreen_does_not_pause(monkeypatch):
    widget, run, source = _offscreen_widget(monkeypatch, bottom=10, top=0)
    widget._pause_if_offscreen(run, source)
    assert widget.state == "playing"


def test_ineligible_preview_never_starts(monkeypatch):
    _patch_source(monkeypatch, _Source)
    widget = ConsoleVideoPreview(
        PRIVATE_PATH,
        duration_seconds=90.0,
        eligible=False,
        ineligible_reason="clip is 90s -- previews cap at 30s",
    )
    widget.play()
    assert widget.state == "poster"
    assert "previews cap" in widget._poster_text()


def test_missing_av_shows_install_guidance(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_video_preview.av_available",
        lambda: False,
    )
    widget = ConsoleVideoPreview(PRIVATE_PATH, duration_seconds=5.0)
    widget.play()
    assert widget.state == "poster"
    assert "video_playback" in widget._poster_text()
    assert ConsoleVideoPreview._active is None


# -- card mounts preview only when ready ------------------------------------


def _card_spec(status="ready", **meta_overrides):
    meta_kwargs = {
        "name": "clip",
        "prompt": "p",
        "backend": "minimax",
        "duration_seconds": 6.0,
    }
    meta_kwargs.update(meta_overrides)
    return ConsoleVideoCardSpec(
        message_id="m1",
        meta=VideoGenerationMetadata(**meta_kwargs),
        status=status,
        file_path="/tmp/clip.mp4" if status == "ready" else None,
    )


def test_ready_card_composes_preview(monkeypatch):
    _patch_source(monkeypatch, _Source)
    card = ConsoleVideoCard(_card_spec())
    children = list(card.compose())
    assert isinstance(children[0], ConsoleVideoPreview)
    assert children[0]._eligible


def test_expired_card_has_no_preview():
    card = ConsoleVideoCard(_card_spec(status="expired"))
    children = list(card.compose())
    assert not any(isinstance(child, ConsoleVideoPreview) for child in children)


def test_oversized_clip_preview_composes_ineligible(monkeypatch):
    _patch_source(monkeypatch, _Source)
    card = ConsoleVideoCard(_card_spec(duration_seconds=90.0))
    children = list(card.compose())
    preview = children[0]
    assert isinstance(preview, ConsoleVideoPreview)
    assert not preview._eligible
    assert "previews cap" in preview._ineligible_reason
