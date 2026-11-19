"""Native Logs resize qualification; run inside Windows Terminal, not Pilot.

Use --before to select stock Textual. Resize/maximize, then Ctrl+Q to exit.
Only synthetic log content, dimensions and driver metadata are recorded.
"""

# Profile selection must precede imports that read Chatbook configuration.
# ruff: noqa: E402

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--before", action="store_true")
parser.add_argument("--label")
parser.add_argument("--profile", type=Path, required=True)
args = parser.parse_args()
# The automation executor exports TERM=dumb; a native Terminal launch normally
# has no such override. Do not let Rich's dumb-terminal 80x25 fallback taint QA.
removed_dumb_term = os.environ.get("TERM") == "dumb"
if removed_dumb_term:
    os.environ.pop("TERM")

from tldw_chatbook.Utils import platform_files

if not args.profile.exists():
    platform_files.os.mkdir(args.profile, 0o700)
os.environ.update(
    HOME=str(args.profile),
    USERPROFILE=str(args.profile),
    XDG_CONFIG_HOME=str(args.profile),
    XDG_DATA_HOME=str(args.profile),
    TLDW_CONFIG_PATH=str(args.profile / "config.toml"),
    PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
)

from loguru import logger

logger.remove()
import textual
from textual.drivers import win32
from textual.drivers.windows_driver import WindowsDriver
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.UI.Screens.logs_screen import LogsScreen
from tldw_chatbook.Utils.text_selection_crash_guard import TextualAppGuards

label = args.label or ("before-logs" if args.before else "after-logs")
output = Path(__file__).parent
stream = (output / f"{label}.jsonl").open("x", encoding="utf-8")
lock = threading.Lock()
stop = threading.Event()
started = time.monotonic()


def record(kind, **values):
    with lock:
        stream.write(
            json.dumps(
                {
                    "seconds": round(time.monotonic() - started, 3),
                    "kind": kind,
                    **values,
                }
            )
            + "\n"
        )
        stream.flush()


def observe_windows():
    previous = None
    while not stop.is_set():
        size = list(os.get_terminal_size(sys.__stdout__.fileno()))
        if size != previous:
            record("windows_size", size=size)
            previous = size
        stop.wait(0.25)


class Probe(TextualAppGuards, ConsolidatedCSSApp):
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]
    TITLE = "TLDW resize " + label
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    async def on_mount(self):
        record(
            "driver",
            name=type(self._driver).__name__,
            input_mode=win32.get_console_mode(sys.__stdin__),
            output_isatty=sys.__stdout__.isatty(),
            render_isatty=self.console.file.isatty(),
            windows_terminal=bool(os.environ.get("WT_SESSION")),
            removed_automation_term_dumb=removed_dumb_term,
            textual=textual.__version__,
        )
        data = SimpleNamespace(
            app_config={},
            _log_records=deque(
                ("INFO", "resize.probe", f"12:00:00 resize fixture {i}")
                for i in range(60)
            ),
        )
        await self.push_screen(LogsScreen(data))
        self.previous = None
        self.set_interval(0.5, self.snapshot)
        self.set_timer(900, self.exit)

    def on_resize(self, event):
        record("app_resize_event", size=list(event.size))

    def snapshot(self):
        selectors = (
            "#screen-content",
            "#logs-destination-header",
            "#logs-window",
            "#app-log-display",
            "#screen-footer-status",
        )
        values = dict(
            app=list(self.size),
            screen=list(self.screen.size),
            regions={s: list(self.screen.query_one(s).region) for s in selectors},
        )
        if values != self.previous:
            record("layout", **values)
            self.save_screenshot(
                f"{label}-{self.size.width}x{self.size.height}.svg", str(output)
            )
            self.previous = values


observer = threading.Thread(target=observe_windows, daemon=True)
observer.start()
try:
    Probe(driver_class=WindowsDriver if args.before else None).run()
finally:
    stop.set()
    observer.join(1)
    record("finished")
    stream.close()
