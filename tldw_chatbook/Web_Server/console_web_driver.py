"""Textual's web driver with an explicit terminal reattachment paint boundary."""

from textual import events
from textual.drivers.web_driver import WebDriver
from textual.geometry import Size


class ConsoleWebDriver(WebDriver):
    """Restore one terminal view without restarting the retained App or input."""

    def on_meta(self, packet_type: str, payload: dict[str, object]) -> None:
        """Marshal reattachment from the input thread onto the App thread."""
        if packet_type != "console_reattach":
            return super().on_meta(packet_type, payload)
        width, height, generation = (
            payload.get("width"),
            payload.get("height"),
            payload.get("generation"),
        )
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or not isinstance(generation, str)
            or not generation
        ):
            return
        self._app.call_from_thread(self.reattach, width, height, generation)

    def reattach(self, width: int, height: int, generation: str) -> None:
        """Restore modes and repaint after resize, then publish matching readiness."""
        self._attachment_generation = generation

        def ready() -> None:
            if self._attachment_generation == generation:
                screen = self._app.screen
                self._app._display(
                    screen,
                    screen._compositor.render_update(
                        full=True, screen_stack=self._app._background_screens
                    ),
                )
                self.flush()
                self.write_meta(
                    {"type": "console_attachment_ready", "generation": generation}
                )

        def repaint() -> None:
            if self._attachment_generation == generation:
                self._app.screen.refresh(repaint=True, layout=True)
                self._app.call_after_refresh(ready)

        async def begin() -> None:
            if self._attachment_generation != generation:
                return
            self.write_meta(
                {"type": "console_attachment_begin", "generation": generation}
            )
            self.write("\x1b[?1049h")
            self._enable_mouse_support()
            self.write("\x1b[?25l")
            self._enable_bracketed_paste()
            self._request_terminal_sync_mode_support()
            self.flush()
            self._size = width, height
            size = Size(width, height)
            await self._app._on_resize(events.Resize(size, size))
            # Flush Textual's resize debounce before scheduling the paint boundary.
            self._app._check_resize()
            self._app.call_after_refresh(repaint)

        self._app.call_after_refresh(begin)
