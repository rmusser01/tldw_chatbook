"""Exercise Prompt History in an isolated native TldwCli profile."""

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

from textual import events
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Input, TextArea

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

app = TldwCli()
result = {"pid": os.getpid(), "steps": []}
TITLE = "#library-prompt-history-collapsible > CollapsibleTitle"


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
        raise AssertionError(label)

    async def focus_on(selector, label):
        await wait_for(
            lambda: (
                screen.focused is screen.query_one(selector)
                and label in painted(screen.focused)
            ),
            f"readable focus {selector}",
        )

    async def more_history():
        screen.query_one("#library-prompt-more-actions", Button).focus()
        if not screen.query_one("#library-prompt-more-actions-region").display:
            await pilot.press("enter")
        for _ in range(6):
            await pilot.press("tab")
            if screen.focused.id == "library-prompt-more-history":
                assert "History" in painted(screen.focused)
                await pilot.press("enter")
                return
        raise AssertionError("History menu entry")

    async def tab_to(selector, label):
        for _ in range(16):
            if screen.focused is screen.query_one(selector):
                break
            await pilot.press("tab")
        await focus_on(selector, label)

    async def confirm_restore(confirm):
        await wait_for(
            lambda: not screen.focused.has_class("-active"), "Restore press complete"
        )
        await pilot.press("enter")
        await wait_for(
            lambda: isinstance(app.screen, ConfirmationDialog), "restore confirmation"
        )
        assert "creates a new current version" in app.screen.message
        assert app.screen.focused.id == "cancel-button"
        if confirm:
            await pilot.press("tab")
            assert app.screen.focused.id == "confirm-button"
            assert "Restore" in painted(app.screen.focused)
            await pilot.press("enter")
        else:
            await pilot.press("escape")
        await wait_for(lambda: app.screen is screen, "confirmation closed")

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        result["driver"] = type(app._driver).__name__
        assert result["driver"] == "LinuxDriver"
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
                "native size",
            )
            app.theme = theme
            await screen._select_library_rail_row("create-prompt")
            await wait_for(
                lambda: bool(screen.query("#library-prompt-name")), "new editor"
            )
            screen.query_one("#library-prompt-mode-basic", Button).focus()
            await pilot.press("enter")
            name = f"Native history {os.getpid()} {width} café"
            screen.query_one("#library-prompt-name", Input).value = name
            for version in range(1, 13):
                screen.query_one(
                    "#library-prompt-user", TextArea
                ).text = f"Version {version}\n\nKeep [bold] literal text."
                await pilot.pause()
                screen.query_one("#library-prompt-save", Button).focus()
                await pilot.press("enter")
                await wait_for(
                    lambda version=version: (
                        screen._prompts_state.version == version
                        and not screen._prompts_state.dirty
                        and not screen._library_prompt_write_worker_is_active()
                    ),
                    "saved version",
                )
            prompt_id = screen._prompts_state.selected_prompt_id
            original_field = screen.query_one("#library-prompt-user", TextArea)
            await more_history()
            await wait_for(
                lambda: len(screen.query(".library-prompt-history-row")) == 10,
                "first history page",
            )
            await focus_on(TITLE, "Retained history (12)")
            assert screen._prompts_state.editor_mode == "info"
            await pilot.press("tab", "enter")
            await wait_for(
                lambda: screen._library_prompt_history_state.selected is not None,
                "preview",
            )
            assert screen.query_one(
                "#library-prompt-history-user", TextArea
            ).text.startswith("Version 12")
            assert screen.query_one("#library-prompt-history-user", TextArea).read_only
            assert screen.query_one("#library-prompt-user", TextArea) is original_field
            await tab_to("#library-prompt-history-load-older", "Load older versions")
            await pilot.press("enter")
            await wait_for(
                lambda: len(screen.query(".library-prompt-history-row")) == 12,
                "older page",
            )
            assert screen.focused.source_version == 2
            await pilot.press("tab", "enter")
            await wait_for(
                lambda: (
                    screen._library_prompt_history_state.selected.source_version == 1
                ),
                "oldest version",
            )
            assert screen.focused.source_version == 1
            app.save_screenshot(f"selection-{width}.svg", path=str(root))
            await tab_to("#library-prompt-history-restore", "Restore selected version")
            await confirm_restore(False)
            await focus_on(
                "#library-prompt-history-restore", "Restore selected version"
            )
            assert screen._prompts_state.version == 12
            app.save_screenshot(f"cancel-{width}.svg", path=str(root))
            await confirm_restore(True)
            await wait_for(
                lambda: screen._prompts_state.version == 13, "restored version"
            )
            await focus_on(TITLE, "Retained history (13)")
            assert (
                screen.query_one("#library-prompt-user", TextArea).text
                == "Version 1\n\nKeep [bold] literal text."
            )
            app.save_screenshot(f"restored-{width}.svg", path=str(root))
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "prompt_id": prompt_id,
                    "name": name,
                    "versions_created_through_ui": 12,
                    "history_from_basic_reveals_info": True,
                    "first_page_count": 10,
                    "older_page_count": 12,
                    "literal_read_only_preview": True,
                    "older_page_focus_version": 2,
                    "cancel_returns_restore_focus": True,
                    "restore_current_version": 13,
                    "restored_user_body": "Version 1\n\nKeep [bold] literal text.",
                }
            )
            record()
        result["passed"] = True
        result["normal_quit_requested"] = True
        result["quit_size"] = list(app.size)
        record()
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - Record a failed native assertion before cleanup.
        result["passed"] = False
        result["error"] = traceback.format_exc()
        app.save_screenshot("failed-state.svg", path=str(root))
        record()
        if isinstance(app.screen, ConfirmationDialog):
            await pilot.press("escape")
        app.post_message(events.Key("ctrl+q", "\x11"))


app.run(auto_pilot=journey, size=(170, 48))
result["app_run_returned"] = True
record()
raise SystemExit(0 if result.get("passed") else 1)
