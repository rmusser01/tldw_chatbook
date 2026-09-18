"""TASK-32761 native Console Behavior and Storage review with private configuration.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
All settings writes and paths remain inside the private profile.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tomllib
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
    from textual.widgets import Button, Input
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
            handles = {
                key: value
                for key, value in vars(app).items()
                if key.endswith("_db") and value is not None
            }
            assert handles
            result["live_database_attributes"] = sorted(handles)

            async def category(name, value):
                await pilot.press("escape", "/", *name, "enter")
                await wait_for(lambda: app.screen.active_category == value, name)

            async def focus(selector):
                control = app.screen.query_one(selector)
                control.focus()
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
                region, clip = app.screen._compositor.visible_widgets[control]
                assert region.intersection(clip) == region
                return control

            async def edit(selector, value):
                field = await focus(selector)
                await pilot.press("home", "shift+end", "backspace", *value)
                await wait_for(lambda: field.value == value, "Field edit")
                return field

            async def press(selector):
                await focus(selector)
                await pilot.press("enter")
                await pilot.pause()

            async def capture(stem):
                await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
                app.save_screenshot(stem + ".svg", path=str(evidence))
                pane = await tmux("capture-pane", "-p", "-t", session)
                (evidence / (stem + ".txt")).write_text(pane.stdout)

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
                    await category("Console Behavior", "console-behavior")
                    screen = app.screen
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    threshold_id = "#settings-console-paste-collapse-threshold"
                    original = screen.query_one(threshold_id, Input).value
                    before = config_path.read_bytes()
                    await edit(threshold_id, "")
                    await pilot.press("escape", "s")
                    await pilot.pause()
                    assert "must be a whole number" in screen._console_behavior_result
                    await edit(threshold_id, str(int(original) + 1))
                    assert config_path.read_bytes() == before
                    await pilot.press("escape", "r")
                    await pilot.pause()
                    await press("#confirm-button")
                    await wait_for(
                        lambda screen=screen: app.screen is screen, "Revert returned"
                    )
                    assert screen.query_one(threshold_id, Input).value == original
                    assert config_path.read_bytes() == before
                    for key in ("remote-images", "status-row-position"):
                        toggle = await focus(f"#settings-console-{key}-toggle")
                        original_label = str(toggle.label)
                        for attempt in range(2):
                            await wait_for(
                                lambda toggle=toggle: not toggle.has_class("-active"),
                                "Button ready",
                            )
                            result["toggle_attempt"] = {
                                "key": key,
                                "attempt": attempt,
                                "before": str(toggle.label),
                            }
                            record()
                            await pilot.press("enter")
                            await wait_for(
                                lambda key=key, screen=screen: (
                                    not screen._console_toggle_writes[key].running
                                ),
                                "Immediate save",
                            )
                            assert (str(toggle.label) == original_label) is (
                                attempt == 1
                            )
                        saved = tomllib.loads(config_path.read_text())
                        if key == "remote-images":
                            assert (
                                saved["chat"]["images"]["render_remote_images"]
                                == screen._remote_images_enabled()
                            )
                        else:
                            assert (
                                saved["console"]["status_chips_position"]
                                == screen._status_row_position_value()
                            )
                    await wait_for(
                        lambda: not app.screen.query("Toast"),
                        "Notices clear before label check",
                    )
                    checkbox = await focus("#settings-console-exchange-capture-enabled")
                    region = checkbox.region
                    painted = "\n".join(
                        strips.crop(region.x, region.right).text
                        for strips in screen._compositor.render_strips()[
                            region.y : region.bottom
                        ]
                    )
                    assert str(checkbox.label) in painted, (
                        str(checkbox.label),
                        painted,
                        region,
                    )
                    await capture(stem + "-console")

                    await category("Storage", "storage")
                    path_id = "#settings-storage-workspaces-db-path"
                    old_path = screen.query_one(path_id, Input).value
                    before = config_path.read_bytes()
                    await edit(path_id, "../outside.db")
                    assert screen.query_one("#settings-save-category", Button).disabled
                    target = root / ("next-launch-" + stem) / "workspaces.db"
                    await edit(path_id, str(target))
                    await press("#settings-check-storage")
                    await wait_for(
                        lambda screen=screen: (
                            "running" not in screen._storage_check_text().lower()
                        ),
                        "Storage check",
                    )
                    assert not target.parent.exists()
                    assert config_path.read_bytes() == before
                    await category("Overview", "overview")
                    await category("Storage", "storage")
                    assert screen.query_one(path_id, Input).value == str(target)
                    await pilot.press("escape", "r")
                    await pilot.pause()
                    await press("#confirm-button")
                    await wait_for(
                        lambda screen=screen: app.screen is screen,
                        "Storage revert returned",
                    )
                    assert screen.query_one(path_id, Input).value == old_path
                    await edit(path_id, str(target))
                    await pilot.press("escape", "s")
                    await wait_for(
                        lambda screen=screen: (
                            not screen._category_has_unsaved_changes(
                                screen.active_category
                            )
                        ),
                        "Storage save",
                    )
                    assert tomllib.loads(config_path.read_text())["database"][
                        "workspaces_db_path"
                    ] == str(target)
                    assert "Restart Chatbook" in screen._storage_result
                    assert not target.parent.exists()
                    assert all(
                        getattr(app, key) is handle for key, handle in handles.items()
                    )
                    await focus(path_id)
                    await pilot.press("end")
                    await capture(stem + "-storage")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "console_validation_and_revert": True,
                            "instant_toggles_saved_and_restored": True,
                            "storage_validation_check_navigation_revert_save": True,
                            "target_not_created": str(target),
                            "database_handles_unchanged": True,
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
