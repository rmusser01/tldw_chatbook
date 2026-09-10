"""Disposable, content-only playback for one mounted Console expression."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from textual import errors
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from tldw_chatbook.Chat.character_expression_playback import (
    PreparedExpression,
    prepare_expression,
)


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
    ) -> None:
        super().__init__(id=id)
        self._data = data
        self._fallback_data = fallback_data
        self._box = box
        self._animate = animate
        self._is_current = is_current
        self._monochrome = monochrome
        self._mode = mode
        self._prepared: PreparedExpression | None = None
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
        self.styles.width, self.styles.height = box

    @property
    def current_image(self) -> Any | None:
        """The displayed prepared frame; viewer callers must copy before retaining."""
        if self._prepared is None or self._frame_index < 0:
            return None
        return self._prepared.frames[self._frame_index]

    def compose(self) -> ComposeResult:
        surface = Static("Preparing expression…")
        self._surface = surface
        surface.styles.width, surface.styles.height = self._box
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
        scale_x, scale_y = (8, 16) if self._mode == "graphics" else (2, 2)
        prepared = None
        try:
            try:
                prepared = await asyncio.to_thread(
                    prepare_expression,
                    self._data,
                    (self._box[0] * scale_x, self._box[1] * scale_y),
                    animate=self._animate,
                )
            except (ValueError, OSError):
                if not self._fallback_data or not self._current():
                    raise
                prepared = await asyncio.to_thread(
                    prepare_expression,
                    self._fallback_data,
                    (self._box[0] * scale_x, self._box[1] * scale_y),
                    animate=False,
                )
                prepared.fallback_reason = (
                    "Expression unavailable; showing the neutral portrait."
                )
            if not self._current():
                prepared.close()
                return
            self._prepared = prepared
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
            self._fallback_data = b""

    async def _paint(self, index: int) -> None:
        if self._prepared is None or not self._current() or self._painting:
            return
        self._painting = True
        try:
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
                        self._graphics.styles.width, self._graphics.styles.height = (
                            self._box
                        )
                        await self.mount(self._graphics)
                        if not self._current():
                            return
                        self._surface.display = False
                    else:
                        self._graphics.image = image
                    self._frame_index = index
                    return
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
            if not self._current():
                return
            self._surface.update(renderable, layout=False)
            self._frame_index = index
        finally:
            self._painting = False

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
        if self._graphics is not None:
            self._graphics.image = None
        if self._prepared is not None:
            self._prepared.close()
            self._prepared = None
