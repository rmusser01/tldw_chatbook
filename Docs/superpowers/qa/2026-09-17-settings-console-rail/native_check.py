"""TASK-32759 native Console rail Settings review with private configuration.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Only a private rail-label draft is toggled and restored.
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
    socket, session = sys.argv[2:4]
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
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
    for key in (
        "NO_COLOR",
        "OPENAI_API_KEY",
        "LLAMA_CPP_API_KEY",
        "SEARX_URL",
        "SERPER_API_KEY",
    ):
        os.environ.pop(key, None)
    config_path = root / "config.toml"

    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Checkbox
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli

    probe_terminal()
    app = TldwCli()
    source = Path(__file__).resolve().parents[4]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            p: hashlib.sha256((source / p).read_bytes()).hexdigest()
            for p in (
                "tldw_chatbook/UI/Screens/settings_screen.py",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
                "tldw_chatbook/css/features/_settings.tcss",
                "tldw_chatbook/css/screen_agentic_settings.tcss",
            )
        },
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
            deadline = asyncio.get_running_loop().time() + 35
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(label)

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            assert app.console.file.isatty()
            result.update(driver="LinuxDriver", tty_streams=True, lock_acquired=True)
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Startup")
            await pilot.press("f4")
            await wait_for(
                lambda: type(app.screen).__name__ == "SettingsScreen", "Settings"
            )
            selector = "#settings-console-stack-collapsed-rail-labels"
            for theme in ("textual-dark", "textual-light"):
                for size in ((190, 55), (80, 24)):
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
                    await wait_for(
                        lambda size=size: tuple(app.size) == size, "Terminal size"
                    )
                    await pilot.press("escape", "/", *"vertical", "enter")
                    await wait_for(
                        lambda: app.screen.focused is app.screen.query_one(selector),
                        "Search focuses rail-label toggle",
                    )
                    await pilot.wait_for_scheduled_animations()
                    screen = app.screen
                    toggle = screen.query_one(selector, Checkbox)
                    viewport = screen.query_one("#settings-detail-pane-body")
                    assert viewport.content_region.contains_region(toggle.region)
                    region, clip = screen._compositor.visible_widgets[toggle]
                    assert region.intersection(clip) == region
                    original = toggle.value
                    active = app.app_config.get("console", {}).get(
                        "stack_collapsed_rail_labels", False
                    )
                    before = config_path.read_bytes()
                    await pilot.press("space")
                    await pilot.pause()
                    assert toggle.value is not original
                    assert (
                        app.app_config.get("console", {}).get(
                            "stack_collapsed_rail_labels", False
                        )
                        == active
                    )
                    assert config_path.read_bytes() == before
                    await wait_for(
                        lambda screen=screen: not screen.query("Toast"), "Notices clear"
                    )
                    strips = screen._compositor.render_strips()
                    painted = "\n".join(
                        strips[y].crop(region.x, region.right).text
                        for y in range(region.y, region.bottom)
                    )
                    assert str(toggle.label) in painted, painted
                    assert "X" in painted, painted
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    app.save_screenshot(stem + ".svg", path=str(evidence))
                    pane = await tmux("capture-pane", "-p", "-t", session)
                    (evidence / (stem + ".txt")).write_text(pane.stdout)
                    await pilot.press("space")
                    await pilot.pause()
                    assert toggle.value is original
                    assert config_path.read_bytes() == before
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "region": list(region),
                            "clip": list(clip),
                            "painted": painted,
                            "draft_restored_without_write": True,
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain failed-run diagnostics
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(190, 55))
    result.update(app_run_returned=True)
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
