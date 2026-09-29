"""Scratch probe plugin (TASK-33003.1 review round 1, finding 7).

Runs each selected test's app at 211x44 (overriding the test's own size) and,
as the app context exits, writes the screen as plain text to
``$CAP_OUT/<test name>.txt``. The same export path as Textual's
``export_screenshot``, rendered to text instead of SVG. A forced size can make
a test's own assertions fail; the capture is still written. Tests do not
change; fixtures and isolation run as usual.
"""

import io
import os
import re
from contextlib import asynccontextmanager

import textual.app
from rich.console import Console

_ORIG = textual.app.App.run_test
_OUT = os.environ["CAP_OUT"]
_NODE = {"id": ""}


def pytest_runtest_setup(item):
    _NODE["id"] = item.name


def _text(app) -> str:
    width, height = app.size
    console = Console(
        width=width, height=height, file=io.StringIO(), record=True,
        legacy_windows=False, safe_box=False, color_system=None,
    )
    console.print(
        app.screen._compositor.render_update(
            full=True, screen_stack=app._background_screens
        )
    )
    return console.export_text()


@asynccontextmanager
async def _run_test(self, *args, **kwargs):
    kwargs["size"] = (211, 44)
    async with _ORIG(self, *args, **kwargs) as pilot:
        try:
            yield pilot
        finally:
            await pilot.pause()
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", _NODE["id"])[:120]
            with open(os.path.join(_OUT, f"{name}.txt"), "w", encoding="utf-8") as fh:
                fh.write(_text(self))


textual.app.App.run_test = _run_test
