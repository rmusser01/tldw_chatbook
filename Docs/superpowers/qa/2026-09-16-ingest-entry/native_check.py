"""Inspect Library import recovery and keyboard entry in an isolated native TldwCli profile."""

import asyncio
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

root = Path(sys.argv[1]).resolve()
socket, session = sys.argv[2:4]
os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
os.environ["XDG_DATA_HOME"] = str(root / "data")
os.environ["XDG_CONFIG_HOME"] = str(root / "config")

os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
os.environ.pop("NO_COLOR", None)

from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.app import TldwCli

app = TldwCli()
result = {"pid": os.getpid(), "steps": []}


def record():
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def painted(widget):
    region = widget.region
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


async def tmux(*args):
    return await asyncio.to_thread(
        subprocess.run,
        ["/opt/homebrew/bin/tmux", "-L", socket, *args],
        check=True,
        text=True,
        capture_output=True,
    )


async def journey(pilot):
    async def wait_for(predicate, label):
        deadline = asyncio.get_running_loop().time() + 20
        while asyncio.get_running_loop().time() < deadline:
            try:
                if predicate():
                    await pilot.pause()
                    return
            except (NoMatches, QueryError):
                pass
            await pilot.pause(0.03)
        raise AssertionError(f"{label}: focus={app.screen.focused!r}")

    async def focus_is(selector, label):
        await wait_for(
            lambda: (
                app.screen.focused is app.screen.query_one(selector)
                and label in painted(app.screen.focused)
            ),
            f"visible focus {selector}",
        )

    async def screenshot(name):
        await wait_for(lambda: not app.query("Toast"), "notices expired")
        app.save_screenshot(name, path=str(root))

    async def stage(path, predicate):
        field = app.screen.query_one("#library-ingest-path", Input)
        field.focus()
        field.value = str(path)
        await wait_for(
            lambda: (
                app.screen._ingest_state.form.path == str(path)
                and app.screen._ingest_state.form.preflight is not None
                and not app.screen._ingest_state.form.preflight_checking
                and predicate(app.screen._ingest_state.form.preflight)
            ),
            f"preflight {path.name}",
        )

    def gate_text():
        gate = app.screen.query_one("#library-ingest-start-quiet-line", Static)
        return " ".join(painted(gate).split())

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        assert type(app._driver).__name__ == "LinuxDriver"
        result["driver"] = type(app._driver).__name__
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "resize",
            )
            app.theme = theme
            await pilot.press("i")
            await wait_for(lambda: bool(screen.query("#library-ingest-path")), "Import")
            entry = screen.query_one("#library-ingest-path", Input)
            await focus_is(
                "#library-ingest-path",
                Path(entry.value).name if entry.value else "Path to a local",
            )
            title = screen.query_one("#library-ingest-title", Input)
            if not title.value:
                title.focus()
                await pilot.press("d", "r", "a", "f", "t")
            assert title.value == "draft"
            await stage(root / "sources/missing.txt", lambda p: p.path_invalid)
            await wait_for(lambda: "file or folder." in gate_text(), "missing recovery")
            assert screen.query_one("#library-ingest-start", Button).disabled
            await screenshot(f"missing-{width}.svg")

            await stage(
                root / "sources/empty", lambda p: p.total_files == 0 and not p.errors
            )
            await wait_for(lambda: "or a single file." in gate_text(), "empty recovery")
            assert screen.query_one("#library-ingest-start", Button).disabled
            assert screen.query_one("#library-ingest-title") is title
            await screenshot(f"empty-{width}.svg")

            await pilot.press("tab")
            await focus_is("#library-ingest-browse", "Browse")
            await pilot.press("tab")
            await focus_is("#library-ingest-clear-path", "Clear")
            await pilot.press("enter")
            await focus_is("#library-ingest-path", "Path to a local")
            await pilot.press("x")
            assert screen.query_one("#library-ingest-path", Input).value == "x"
            assert screen._ingest_state.form.title == "draft"
            await stage(
                root / "sources/ready.txt", lambda p: bool(p.type_groups.get("generic"))
            )
            await wait_for(
                lambda: not screen.query_one("#library-ingest-start", Button).disabled,
                "Start enabled",
            )
            assert gate_text() == ""
            assert (
                screen.query_one("#library-ingest-start-quiet-line").region.height >= 1
            )
            assert screen.query_one("#library-ingest-title") is title
            assert title.value == "draft"
            for _ in range(12):
                if screen.focused is screen.query_one("#library-ingest-start"):
                    break
                await pilot.press("tab")
            await focus_is("#library-ingest-start", "Start import")
            assert not app.library_ingest_jobs.jobs()
            await screenshot(f"ready-{width}.svg")
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "missing_recovery_readable": True,
                    "empty_recovery_readable": True,
                    "clear_enter_returns_focus": True,
                    "metadata_retained": True,
                    "ready_start_reachable_by_tab": True,
                    "jobs_queued": 0,
                }
            )
            record()
            await pilot.press("escape")
            await wait_for(
                lambda: bool(screen.query("#library-hub-action-import")), "hub"
            )
        result["passed"] = True
        record()
        await tmux("send-keys", "-t", session, "C-q")
    except Exception:  # noqa: BLE001 - preserve evidence before normal shutdown
        result["passed"] = False
        result["error"] = traceback.format_exc()
        app.save_screenshot("failed-state.svg", path=str(root))
        record()
        if isinstance(app.screen, ModalScreen):
            await pilot.press("escape")
        await tmux("send-keys", "-t", session, "C-q")


app.run(auto_pilot=journey, size=(170, 48))
result["app_run_returned"] = True
record()
raise SystemExit(0 if result.get("passed") else 1)
