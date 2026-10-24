"""Disposable, content-only playback for one mounted Console expression."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from typing import Any

from textual import errors
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from tldw_chatbook.Audio.speech_presentation import SpeechPresentationSnapshot
from tldw_chatbook.Character_Chat.portrait_performance import (
    FRAME_ROLES,
    FrameRole,
    PortraitPerformanceRecord,
)
from tldw_chatbook.Chat.character_expression_playback import (
    PreparedExpression,
    PreparedPerformance,
    prepare_expression,
    prepare_performance,
)
from tldw_chatbook.Chat.portrait_performance_state import PortraitPerformanceClock


class CharacterExpressionAvatar(Widget):
    """Own one bounded sequence and timer, never a character or database binding."""

    def __init__(
        self,
        data: bytes,
        *,
        box: tuple[int, int],
        animate: bool,
        is_current: Callable[[], bool],
        monochrome: bool,
        mode: str,
        id: str,
        fallback_data: bytes = b"",
        decode_identity: tuple[str, ...] | None = None,
        fallback_decode_identity: tuple[str, ...] | None = None,
        performance: PortraitPerformanceRecord | None = None,
        read_speech: Callable[[], SpeechPresentationSnapshot | None] | None = None,
    ) -> None:
        super().__init__(id=id)
        self._data = data
        self._fallback_data = fallback_data
        self._box = box
        self._animate = animate
        self._is_current = is_current
        self._monochrome = monochrome
        self._mode = mode
        self._decode_identity = decode_identity
        self._fallback_decode_identity = fallback_decode_identity
        self._performance = performance
        self._read_speech = read_speech
        self._preview_mode: str | None = None
        self._performance_clock: PortraitPerformanceClock | None = None
        self._last_pose: tuple[int, float] | None = None
        self._performance_disabled = False
        self._pose_image: PreparedExpression | None = None
        self._pose_job: asyncio.Task | None = None
        self._prepared: PreparedExpression | PreparedPerformance | None = None
        self._job: asyncio.Task | None = None
        self._timer = None
        self._render_job: asyncio.Task | None = None
        self._elapsed_ms = 0.0
        self._last_tick: float | None = None
        self._frame_index = -1
        self._painting = False
        self._disposed = False
        self._graphics: Any = None
        self._surface: Static | None = None
        self.remove_class(*(name for name in self.classes if name.startswith("h-")))
        self.remove_class(*(name for name in self.classes if name.startswith("w-")))
        # ds-runtime: The owning rail fits this image box to its measured columns and image aspect ratio.
        self.set_styles(width=box[0], height=box[1])

    @property
    def current_image(self) -> Any | None:
        """The displayed prepared frame; viewer callers must copy before retaining."""
        if self._prepared is None or self._frame_index < 0:
            return None
        if self._pose_image is not None:
            return self._pose_image.frames[0]
        return self._prepared.frames[self._frame_index]

    def compose(self) -> ComposeResult:
        surface = Static("Preparing expression…")
        self._surface = surface
        surface.remove_class(*(name for name in surface.classes if name.startswith("h-")))
        surface.remove_class(*(name for name in surface.classes if name.startswith("w-")))
        # ds-runtime: The owning rail fits this image box to its measured columns and image aspect ratio.
        surface.set_styles(width=self._box[0], height=self._box[1])
        yield surface

    def on_mount(self) -> None:
        self.call_after_refresh(self._start)

    def _start(self) -> None:
        if not self._disposed:
            self._job = asyncio.create_task(self._prepare())

    def _current(self) -> bool:
        return (
            not self._disposed
            and self.is_mounted
            and self._surface is not None
            and self._surface.parent is self
            and self._surface.is_mounted
            and self._is_current()
        )

    def _visible(self) -> bool:
        visible = (
            self._current()
            and self.screen.is_current
            and self.is_on_screen
            and self.visible
            and all(getattr(node, "display", True) for node in self.ancestors_with_self)
        )
        if not visible:
            return False
        try:
            top, _region = self.screen.get_widget_at(*self.region.center)
        except errors.NoWidget:
            return False
        return self in top.ancestors_with_self

    async def _prepare(self) -> None:
        # Pixel mosaics need 2x2 subpixels. Graphics gets a modest larger
        # raster, still bounded by the displayed cell box and shared budget.
        scale_x, scale_y = (
            (8, 16) if self._mode == "graphics" and not self._monochrome else (2, 2)
        )
        prepared = None
        try:
            try:
                if self._performance is not None and self._animate:
                    try:
                        prepared = await asyncio.to_thread(
                            prepare_performance,
                            self._performance,
                            (self._box[0] * scale_x, self._box[1] * scale_y),
                        )
                    except (
                        ValueError,
                        OSError,
                        ImportError,
                        RuntimeError,
                        MemoryError,
                    ):
                        prepared = await asyncio.to_thread(
                            prepare_expression,
                            self._data,
                            (self._box[0] * scale_x, self._box[1] * scale_y),
                            animate=False,
                            identity=self._decode_identity,
                        )
                        prepared.fallback_reason = (
                            "Performance unavailable; showing the selected still."
                        )
                else:
                    prepared = await asyncio.to_thread(
                        prepare_expression,
                        self._data,
                        (self._box[0] * scale_x, self._box[1] * scale_y),
                        animate=self._animate,
                        identity=self._decode_identity,
                    )
            except (ValueError, OSError):
                if not self._fallback_data or not self._current():
                    raise
                prepared = await asyncio.to_thread(
                    prepare_expression,
                    self._fallback_data,
                    (self._box[0] * scale_x, self._box[1] * scale_y),
                    animate=False,
                    identity=self._fallback_decode_identity,
                )
                prepared.fallback_reason = (
                    "Expression unavailable; showing the neutral portrait."
                )
            if not self._current():
                prepared.close()
                return
            self._prepared = prepared
            if isinstance(prepared, PreparedPerformance):
                motion = prepared.motion
                self._performance_clock = PortraitPerformanceClock(
                    motion=motion,
                    interval=lambda: random.uniform(
                        motion.blink_min_s, motion.blink_max_s
                    ),
                )
            self.tooltip = prepared.fallback_reason or None
            await self._paint(0)
            if self._current() and len(prepared.frames) > 1:
                self._last_tick = None
                self._timer = self.set_interval(1 / 30, self._tick)
        except (ValueError, OSError, ImportError, RuntimeError, MemoryError):
            if self._prepared is not None:
                self._prepared.close()
                self._prepared = None
            if self._current():
                self._surface.update("no avatar", layout=False)
                self.tooltip = (
                    "Expression unavailable within image limits; using text fallback."
                )
        finally:
            self._data = b""
            self._performance = None
            self._fallback_data = b""
            self._decode_identity = None
            self._fallback_decode_identity = None

    def _adopt_pose(self, pose: PreparedExpression | None) -> None:
        old, self._pose_image = self._pose_image, pose
        if old is not None:
            old.close()

    async def _paint(self, index: int, *, offset_px: float | None = 0.0) -> bool:
        if self._prepared is None or not self._current() or self._painting:
            return False
        if self._render_job is not None and not self._render_job.done():
            return False
        if self._pose_job is not None:
            if not self._pose_job.done():
                return False
            if not self._pose_job.cancelled() and self._pose_job.exception() is None:
                previous = self._pose_job.result()
                if previous is not self._pose_image:
                    previous.close()
        self._painting = True
        pose = None
        try:
            if (
                isinstance(self._prepared, PreparedPerformance)
                and offset_px is not None
            ):
                self._pose_job = asyncio.create_task(
                    asyncio.to_thread(
                        self._prepared.render_pose,
                        index,
                        offset_px,
                    )
                )
                pose = await asyncio.shield(self._pose_job)
                if not self._current() or not self._pose_is_current(index):
                    return False
                image = pose.frames[0]
            else:
                image = self._prepared.frames[index]
            if self._mode == "graphics" and not self._monochrome:
                try:
                    if self._graphics is None:
                        from textual_image.widget import Image as GraphicsImage

                        class FixedImage(
                            GraphicsImage, Renderable=GraphicsImage._Renderable
                        ):
                            def refresh(
                                self,
                                *regions,
                                repaint=True,
                                layout=False,
                                recompose=False,
                            ):
                                # Both frame dimensions and the cell box are fixed.
                                return super().refresh(
                                    *regions,
                                    repaint=repaint,
                                    layout=False,
                                    recompose=recompose,
                                )

                        self._graphics = FixedImage(image)
                        self._graphics.remove_class(*(name for name in self._graphics.classes if name.startswith("h-")))
                        self._graphics.remove_class(*(name for name in self._graphics.classes if name.startswith("w-")))
                        # ds-runtime: The owning rail fits this image box to its measured columns and image aspect ratio.
                        self._graphics.set_styles(
                            width=self._box[0], height=self._box[1]
                        )
                        await self.mount(self._graphics)
                        if not self._current() or not self._pose_is_current(index):
                            return False
                        self._surface.display = False
                    else:
                        self._graphics.image = image
                    self._frame_index = index
                    self._adopt_pose(pose)
                    pose = None
                    return True
                except (
                    ImportError,
                    OSError,
                    ValueError,
                    RuntimeError,
                    TypeError,
                    AttributeError,
                ):
                    self._mode = "pixels"
                    self.tooltip = (
                        "Graphics unavailable; showing terminal pixel expressions."
                    )
            from tldw_chatbook.UI.Console_Modules.character_avatar_layout import (
                render_character_avatar_mosaic,
            )

            self._render_job = asyncio.create_task(
                asyncio.to_thread(
                    render_character_avatar_mosaic,
                    image,
                    *self._box,
                    monochrome=self._monochrome,
                )
            )
            renderable = await asyncio.shield(self._render_job)
            if not self._current() or not self._pose_is_current(index):
                return False
            self._surface.update(renderable, layout=False)
            self._frame_index = index
            self._adopt_pose(pose)
            pose = None
            return True
        finally:
            if pose is not None and (
                self._render_job is None or self._render_job.done()
            ):
                pose.close()
            self._painting = False

    def _speech(self) -> SpeechPresentationSnapshot | None:
        if self._read_speech is None:
            return None
        try:
            return self._read_speech()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def _mouth(self, now: float) -> str:
        speech = self._speech()
        age = now - speech.observed_at if speech is not None else float("inf")
        return (
            speech.mouth
            if speech is not None
            and speech.state == "playing"
            and speech.timing_valid
            and 0 <= age <= 0.1
            else "rest"
        )

    def set_preview_mode(self, mode: FrameRole | str) -> None:
        """Select an offline pose/cycle/idle without creating a speech owner."""
        if mode not in (*FRAME_ROLES, "cycle", "idle"):
            raise ValueError("portrait_performance_preview_mode_invalid")
        self._preview_mode = mode
        if isinstance(self._prepared, PreparedPerformance):
            motion = self._prepared.motion
            self._performance_clock = PortraitPerformanceClock(
                motion=motion,
                interval=lambda: random.uniform(motion.blink_min_s, motion.blink_max_s),
            )
        self._elapsed_ms = 0.0
        self._last_tick = None
        self._last_pose = None

    def _preview_index(self) -> int | None:
        if self._preview_mode == "cycle":
            return int(
                self._elapsed_ms / (self._prepared.motion.blink_hold_s * 1000)
            ) % len(FRAME_ROLES)
        if self._preview_mode in FRAME_ROLES:
            return FRAME_ROLES.index(self._preview_mode)
        return None

    def _pose_is_current(self, index: int) -> bool:
        if (
            isinstance(self._prepared, PreparedPerformance)
            and self._preview_mode is not None
        ):
            preview_index = self._preview_index()
            return (
                index == preview_index if preview_index is not None else index % 3 == 0
            )
        if not isinstance(self._prepared, PreparedPerformance) or index % 3 == 0:
            return True
        return index % 3 == ("rest", "small", "open").index(
            self._mouth(time.monotonic())
        )

    async def _tick(self) -> None:
        if not self._current():
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            self._last_tick = None
            return
        if not self._visible():
            self._last_tick = None
            return
        now = time.monotonic()
        if self._last_tick is not None:
            self._elapsed_ms += (now - self._last_tick) * 1000
        self._last_tick = now
        if self._prepared is None:
            return
        speech = self._speech()
        if isinstance(self._prepared, PreparedPerformance):
            if self._performance_disabled:
                return
            mouth = self._mouth(now)
            pose = self._performance_clock.sample(mouth, now=self._elapsed_ms / 1000)
            preview_index = self._preview_index()
            key = (
                (preview_index, 0.0)
                if preview_index is not None
                else (FRAME_ROLES.index(pose.role), round(pose.offset_px, 2))
            )
            if key != self._last_pose:
                try:
                    if await self._paint(key[0], offset_px=key[1]):
                        self._last_pose = key
                except (ValueError, OSError, ImportError, RuntimeError, MemoryError):
                    self._performance_disabled = True
                    if self._timer is not None:
                        self._timer.stop()
                        self._timer = None
                    self.tooltip = (
                        "Performance unavailable; showing the selected still."
                    )
                    await self._paint(0, offset_px=None)
            return
        if speech is not None and speech.state in {"preparing", "playing", "paused"}:
            self._last_tick = None
            if self._frame_index != 0:
                await self._paint(0)
            return
        index, finished = self._prepared.frame_at(self._elapsed_ms)
        if index != self._frame_index:
            try:
                await self._paint(index)
            except (ValueError, OSError, ImportError, RuntimeError, MemoryError):
                self.tooltip = "Animation stopped; keeping the last available frame."
                finished = True
        if finished and self._timer is not None:
            self._timer.stop()
            self._timer = None

    async def on_unmount(self) -> None:
        self._disposed = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._job is not None:
            # Do not cancel to_thread: that would orphan the codec's result
            # and its reservation. A stale completion closes its own buffers.
            await asyncio.shield(self._job)
        if self._render_job is not None:
            # The frame path has already surfaced any render failure; cleanup
            # still waits for the thread before closing its input image.
            await asyncio.gather(self._render_job, return_exceptions=True)
            self._render_job = None
        if self._pose_job is not None:
            results = await asyncio.gather(self._pose_job, return_exceptions=True)
            if (
                isinstance(results[0], PreparedExpression)
                and results[0] is not self._pose_image
            ):
                results[0].close()
            self._pose_job = None
        if self._graphics is not None:
            self._graphics.image = None
        self._adopt_pose(None)
        if self._prepared is not None:
            self._prepared.close()
            self._prepared = None
