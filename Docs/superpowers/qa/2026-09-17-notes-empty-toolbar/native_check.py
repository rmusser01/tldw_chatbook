"""TASK-32752: empty Notes action paint in the real terminal app.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
No provider request is sent. This qualifies integrated rendering and lifecycle,
not generation or the complete Notes workflow.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}) + "\n")
    socket, session = sys.argv[2:4]
    qa = Path(__file__).resolve().parents[1]
    runpy.run_path(str(qa / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    for key in ("NO_COLOR", "OPENAI_API_KEY", "LLAMA_CPP_API_KEY"):
        os.environ.pop(key, None)

    from textual.css.query import NoMatches, QueryError
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli

    source = Path(__file__).resolve().parents[4]
    assert Path(app_module.__file__).resolve() == source / "tldw_chatbook/app.py"
    probe_terminal()
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            path: hashlib.sha256((source / path).read_bytes()).hexdigest()
            for path in (
                "tldw_chatbook/Widgets/Library/library_notes_canvas.py",
                "tldw_chatbook/Widgets/Console/console_bounded_section.py",
                "tldw_chatbook/Widgets/Library/library_file_notes_workspace.py",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
                "tldw_chatbook/css/screen_agentic_library.tcss",
            )
        },
        "cells": [],
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

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
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 40
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(label)

        async def capture(stem):
            await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
            await pilot.wait_for_scheduled_animations()
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            assert app.console.file.isatty()
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Startup")
            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    app.theme = theme
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(size[0]),
                        "-y",
                        str(size[1]),
                    )
                    await wait_for(lambda size=size: tuple(app.size) == size, "Resize")
                    await pilot.press("ctrl+2")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "ChatScreen", "Console"
                    )
                    await wait_for(
                        lambda: bool(app.screen.query("#console-native-composer")),
                        "Composer mounted",
                    )
                    await pilot.press("tab")
                    await pilot.pause()
                    console_focus = app.screen.focused
                    assert console_focus is not None and console_focus.is_attached
                    assert console_focus in app.screen._compositor.visible_widgets
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    await pilot.press("ctrl+3")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "LibraryScreen", "Library"
                    )
                    await app.screen._select_library_rail_row("browse-notes")
                    await wait_for(
                        lambda: bool(app.screen.query("#library-notes-canvas")),
                        "Notes canvas",
                    )
                    await pilot.press("f6")
                    await pilot.pause()
                    focused = app.screen.focused
                    assert focused is not None and focused.is_attached
                    assert focused in app.screen._compositor.visible_widgets
                    canvas = app.screen.query_one("#library-notes-canvas")
                    actions = list(
                        canvas.query(".library-notes-tree-action-row Button")
                    )
                    assert len(actions) == (1 if canvas.compact else 4)
                    strips = app.screen._compositor.render_strips()
                    for button in actions:
                        assert button.region.width > 0
                        assert button.region.x >= canvas.content_region.x
                        assert button.region.right <= canvas.content_region.right
                        painted = "\n".join(
                            strip.text
                            for strip in strips[button.region.y : button.region.bottom]
                        )
                        assert str(button.label) in painted
                    result.setdefault("action_rows", []).append(
                        {
                            "theme": theme,
                            "size": size,
                            "contract_width": canvas.pane_width,
                            "content_width": canvas.content_region.width,
                            "actions": [
                                {
                                    "label": str(button.label),
                                    "disabled": button.disabled,
                                    "region": list(button.region),
                                }
                                for button in actions
                            ],
                        }
                    )
                    await capture(stem + "-notes")
                    result["cells"].append({"theme": theme, "size": size})
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain failed-run evidence
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
