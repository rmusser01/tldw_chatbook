"""Native tmux resize journey; accepts a disposable profile and owned session."""

import asyncio
import hashlib
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

from textual import events
from textual.widgets import (
    Button,
    Input,
    TextArea,
)

from tldw_chatbook.app import (
    TldwCli,
)

app = TldwCli()
result = {"pid": os.getpid(), "steps": [], "native_resize": True}


def save_result():
    (root / "result.json").write_text(json.dumps(result, indent=2))


def painted(field):
    region = field.region
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


async def journey(pilot):
    async def wait_for(predicate, label):
        deadline = asyncio.get_running_loop().time() + 30
        while asyncio.get_running_loop().time() < deadline:
            if predicate():
                await pilot.pause()
                return
            await pilot.pause(0.05)
        raise AssertionError(label)

    async def resize(width, height):
        subprocess.run(  # noqa: ASYNC221 - bounded native resize signal
            [
                "/opt/homebrew/bin/tmux",
                "-L",
                socket,
                "resize-window",
                "-t",
                session,
                "-x",
                str(width),
                "-y",
                str(height),
            ],
            check=True,
            capture_output=True,
        )
        await wait_for(
            lambda: tuple(app.size) == (width, height), "native terminal resize"
        )
        await pilot.pause()

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        result["driver"] = type(app._driver).__name__
        assert result["driver"] == "LinuxDriver"
        assert tuple(app.size) == (80, 24)
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        await screen._select_library_rail_row("create-prompt")
        await wait_for(lambda: bool(screen.query("#library-prompt-name")), "new editor")
        await wait_for(
            lambda: screen._library_prompt_browse_controller.result.status == "ready",
            "Items",
        )
        screen.query_one("#library-prompt-mode-basic", Button).focus()
        await pilot.press("enter")
        name = screen.query_one("#library-prompt-name", Input)
        name.value = "Native resize audit " + str(os.getpid())
        content = screen.query_one("#library-prompt-user", TextArea)
        content.text = "Keep this message visible."
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).focus()
        await pilot.press("enter")
        await wait_for(
            lambda: (
                screen._prompts_state.selected_prompt_id is not None
                and not screen._prompts_state.dirty
                and not screen._library_prompt_write_worker_is_active()
            ),
            "save",
        )
        await wait_for(
            lambda: (
                screen._library_snapshot_rendered_generation
                == screen._library_snapshot_state_generation
                and screen._library_prompt_browse_controller.result.status == "ready"
            ),
            "saved Items",
        )
        result["prompt_id"] = screen._prompts_state.selected_prompt_id
        for theme in ("textual-dark", "textual-light"):
            app.theme = theme
            for mode in ("basic", "advanced"):
                screen.query_one(f"#library-prompt-mode-{mode}", Button).focus()
                await pilot.press("enter")
                await wait_for(
                    lambda mode=mode: screen._prompts_state.editor_mode == mode,
                    "editor mode",
                )
                field = (
                    content
                    if mode == "basic"
                    else screen.query(".prompt-block-content").first(TextArea)
                )
                field.focus()
                await pilot.pause()
                before = hashlib.sha256((root / "config.toml").read_bytes()).hexdigest()
                generation = (
                    screen._library_prompt_browse_controller.result.request_token
                )
                for width, height in ((170, 48), (80, 24), (170, 48), (170, 24)):
                    await resize(width, height)
                    assert screen.focused is field, repr(screen.focused)
                    assert field.text == "Keep this message visible."
                    assert "Keep this" in painted(field), repr(field.region)
                    assert (
                        screen._library_prompt_browse_controller.result.request_token
                        == generation
                    )
                    assert (
                        hashlib.sha256((root / "config.toml").read_bytes()).hexdigest()
                        == before
                    )
                    result["steps"].append(
                        {
                            "theme": theme,
                            "mode": mode,
                            "size": [width, height],
                            "focus": field.id,
                            "region": list(field.region),
                            "painted": True,
                            "preferences_unchanged": True,
                            "browse_generation_unchanged": True,
                        }
                    )
                    if (width, height) == (80, 24) or (
                        theme == "textual-dark" and height == 48
                    ):
                        app.save_screenshot(
                            f"{mode}-{theme}-{width}x{height}.svg", path=str(root)
                        )
        await resize(80, 24)
        await wait_for(
            lambda: (
                not any(
                    worker.is_running
                    for worker in app.workers
                    if worker.group.startswith("library")
                )
            ),
            "Library workers settled",
        )
        result["passed"] = True
        result["quit_size"] = list(app.size)
        result["normal_quit_requested"] = True
        save_result()
        # Queue the real global key and return. Waiting on Pilot.pause after
        # shutting down can leave the autopilot waiting on closed message pumps.
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - record any probe failure and exit
        result["passed"] = False
        result["error"] = traceback.format_exc()
        save_result()
        app.exit()


app.run(auto_pilot=journey, size=(80, 24))
result["app_run_returned"] = True
save_result()
raise SystemExit(0 if result.get("passed") else 1)
