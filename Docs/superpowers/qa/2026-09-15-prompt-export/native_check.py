"""Run the actual Prompt Copy and Export controls in a private native profile."""

import asyncio
import base64
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
from tldw_chatbook.Prompt_Management.Prompts_Interop import (
    parse_markdown_prompts_from_content,
)
from tldw_chatbook.Third_Party.textual_fspicker import FileSave
from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileNameInput
from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation

app = TldwCli()
result = {"pid": os.getpid(), "steps": []}
clipboard_writes = []


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
        deadline = asyncio.get_running_loop().time() + 25
        while asyncio.get_running_loop().time() < deadline:
            try:
                ready = predicate()
            except (NoMatches, QueryError):
                ready = False
            if ready:
                await pilot.pause()
                return
            await pilot.pause(0.03)
        raise AssertionError(label)

    async def inline_result(message):
        await wait_for(
            lambda: screen._prompts_state.status == message, "inline feedback"
        )
        status = screen.query_one("#library-prompt-save-status")
        assert message in painted(status)
        assert not [
            note
            for note in app._notifications
            if "prompt" in note.message.lower() or "clipboard" in note.message.lower()
        ]

    async def more(action):
        opener = screen.query_one("#library-prompt-more-actions", Button)
        opener.focus()
        if not screen.query_one("#library-prompt-more-actions-region").display:
            await pilot.press("enter")
        for _ in range(6):
            await pilot.press("tab")
            if screen.focused.id == f"library-prompt-{action}":
                label = str(screen.focused.label).replace("…", "")
                start = asyncio.get_running_loop().time()
                initially_painted = label in painted(screen.focused)
                await wait_for(
                    lambda label=label: label in painted(screen.focused),
                    "focused action paint",
                )
                if not initially_painted:
                    result.setdefault("deferred_action_paint", []).append(
                        {
                            "action": action,
                            "size": list(app.size),
                            "seconds": asyncio.get_running_loop().time() - start,
                        }
                    )
                await pilot.press("enter")
                return
        raise AssertionError(action)

    async def return_to_export():
        await wait_for(
            lambda: (
                app.screen is screen
                and screen.focused is screen.query_one("#library-prompt-export", Button)
                and "Export" in painted(screen.focused)
            ),
            "export focus return",
        )

    async def open_export():
        await more("export")
        await wait_for(
            lambda: (
                isinstance(app.screen, FileSave)
                and isinstance(app.screen.focused, FileNameInput)
            ),
            "filename focus",
        )
        dialog = app.screen
        filename = dialog.query_one(FileNameInput)
        assert filename.value == original_name + ".md"
        assert "café.md" in painted(filename)
        return dialog, filename

    async def choose(dialog, filename, relative_path, capture=None):
        # Navigate to the private folder before capturing the picker.
        await pilot.press("ctrl+a")
        filename.post_message(events.Paste(str(root / "exports")))
        await pilot.pause()
        await pilot.press("enter")
        await wait_for(
            lambda: (
                dialog.query_one(DirectoryNavigation).location == root / "exports"
                and filename.value == ""
            ),
            "private export folder",
        )
        filename.focus()
        await pilot.press("ctrl+a")
        filename.post_message(events.Paste(relative_path))
        await pilot.pause()
        assert filename.value == relative_path
        await pilot.press("tab")
        assert dialog.focused is dialog.query_one("#select", Button)
        assert "Save" in painted(dialog.focused)
        if capture:
            app.save_screenshot(capture, path=str(root))
        await pilot.press("enter")

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        result["driver"] = type(app._driver).__name__
        assert result["driver"] == "LinuxDriver"
        write = app._driver.write

        def capture_clipboard_sequence(data):
            if data.startswith("\x1b]52;c;"):
                clipboard_writes.append(base64.b64decode(data[7:-1]).decode("utf-8"))
            return write(data)

        app._driver.write = capture_clipboard_sequence
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        await screen._select_library_rail_row("create-prompt")
        await wait_for(lambda: bool(screen.query("#library-prompt-name")), "new editor")
        screen.query_one("#library-prompt-mode-basic", Button).focus()
        await pilot.press("enter")
        original_name = f"Native export {os.getpid()} café"
        original_body = "Keep this reusable message.\n\nWith a second paragraph."
        screen.query_one("#library-prompt-name", Input).value = original_name
        screen.query_one("#library-prompt-user", TextArea).text = original_body
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).focus()
        await pilot.press("enter")
        await wait_for(
            lambda: (
                screen._prompts_state.selected_prompt_id is not None
                and not screen._prompts_state.dirty
                and not screen._library_prompt_write_worker_is_active()
            ),
            "saved source",
        )
        result["prompt_id"] = screen._prompts_state.selected_prompt_id
        field = screen.query_one("#library-prompt-user", TextArea)
        normal_copy = app.copy_to_clipboard

        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "native resize",
            )
            app.theme = theme
            await pilot.pause()
            await more("copy")
            copied = app.clipboard
            assert clipboard_writes[-1] == copied
            parsed = parse_markdown_prompts_from_content(copied)
            assert len(parsed) == 1 and parsed[0]["user_prompt"] == original_body
            assert parsed[0]["name"] == original_name
            assert screen.focused.id == "library-prompt-copy"
            await inline_result("Prompt copied to clipboard as markdown!")

            app.copy_to_clipboard = None
            await more("copy")
            await inline_result("Clipboard copy is unavailable in this runtime.")
            app.copy_to_clipboard = normal_copy

            await open_export()
            await pilot.press("escape")
            await return_to_export()
            await inline_result("Prompt export cancelled.")
            assert screen.query_one("#library-prompt-user", TextArea) is field

            dialog, filename = await open_export()
            await choose(dialog, filename, f"missing/failure-{width}.md")
            await inline_result("Error exporting prompt: FileNotFoundError")
            await return_to_export()
            await pilot.pause(0.1)
            capture = await tmux("capture-pane", "-p", "-t", session)
            (root / f"failure-{width}.txt").write_text(capture.stdout)
            assert "Error exporting prompt" in capture.stdout
            app.save_screenshot(f"failure-{width}.svg", path=str(root))
            assert screen.query_one("#library-prompt-user", TextArea) is field

            dialog, filename = await open_export()
            output_name = f"roundtrip-{width}.md"
            await choose(dialog, filename, output_name, f"picker-{width}.svg")
            destination = root / "exports" / output_name
            await wait_for(destination.exists, "export file")
            await return_to_export()
            await inline_result(f"Prompt exported successfully to {output_name}")
            app.save_screenshot(f"success-{width}.svg", path=str(root))
            assert destination.read_text() == copied
            assert screen.query_one("#library-prompt-user", TextArea) is field
            assert field.text == original_body
            assert (
                screen._prompts_state.version == 1 and not screen._prompts_state.dirty
            )
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "clipboard_osc52_matches_markdown": True,
                    "clipboard_unavailable_reported": True,
                    "inline_results_painted_without_prompt_toasts": True,
                    "filename_initial_focus_and_label": True,
                    "cancel_returns_export_focus": True,
                    "write_failure_visible_in_terminal_capture": True,
                    "retry_written_file_matches_clipboard": True,
                    "live_field_identity_preserved": True,
                    "output": f"exports/{output_name}",
                }
            )
            record()
        result["passed"] = True
        result["normal_quit_requested"] = True
        result["quit_size"] = list(app.size)
        record()
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - Record assertion/runtime failures before normal cleanup.
        result["passed"] = False
        result["error"] = traceback.format_exc()
        app.save_screenshot("failed-state.svg", path=str(root))
        record()
        if isinstance(app.screen, FileSave):
            await pilot.press("escape")
        app.post_message(events.Key("ctrl+q", "\x11"))


app.run(auto_pilot=journey, size=(170, 48))
result["app_run_returned"] = True
record()
raise SystemExit(0 if result.get("passed") else 1)
