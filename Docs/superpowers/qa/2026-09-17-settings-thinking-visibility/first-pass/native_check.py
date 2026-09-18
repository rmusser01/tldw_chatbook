"""TASK-32763 native thinking-visibility Settings review with private configuration.

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
    from textual.widgets import Static
    from textual_image._terminal import probe_terminal

    from tldw_chatbook import config
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
                "tldw_chatbook/config.py",
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
                    selector = "#settings-console-show-model-thinking"
                    checkbox = await focus(selector)
                    assert checkbox.value is False
                    before = config_path.read_bytes()
                    os.chmod(root, 0o500)
                    try:
                        await pilot.press("space")
                        await wait_for(
                            lambda screen=screen: (
                                not screen._console_toggle_writes[
                                    "model-thinking"
                                ].running
                            ),
                            "Failed write settled",
                        )
                        assert checkbox.value is False
                        assert config_path.read_bytes() == before
                        assert (
                            "prior setting was restored"
                            in screen._console_behavior_result
                        )
                        await capture(stem + "-failed")
                    finally:
                        os.chmod(root, 0o700)
                    await pilot.press("space")
                    await wait_for(
                        lambda screen=screen: (
                            not screen._console_toggle_writes["model-thinking"].running
                        ),
                        "Retry saved",
                    )
                    assert checkbox.value is True
                    assert (
                        tomllib.loads(config_path.read_text())["console"][
                            "show_model_thinking"
                        ]
                        is True
                    )
                    assert app.app_config["console"]["show_model_thinking"] is True
                    await pilot.press("ctrl+2")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "ChatScreen",
                        "Console departure",
                    )
                    screen_was_removed = not screen.is_attached
                    await pilot.press("f4")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "SettingsScreen",
                        "Settings return",
                    )
                    await category("Console Behavior", "console-behavior")
                    screen = app.screen
                    checkbox = await focus(selector)
                    assert checkbox.value is True
                    assert "visibility saved" in str(
                        screen.query_one(
                            "#settings-console-model-thinking-result", Static
                        ).render()
                    )
                    await capture(stem + "-saved")
                    # Real external mutation followed by the visible Reload Config action.
                    mutation = config.apply_settings_mutation_to_cli_config(
                        {"console": {"show_model_thinking": False}}
                    )
                    assert mutation.failure_phase is None
                    await category("Diagnostics", "diagnostics")
                    await focus("#settings-reload-config")
                    await pilot.press("enter")
                    await wait_for(
                        lambda screen=screen: (
                            screen._diagnostics_reload_result == "Config reload: loaded"
                        ),
                        "Reload completed",
                    )
                    await category("Console Behavior", "console-behavior")
                    checkbox = await focus(selector)
                    assert checkbox.value is False
                    await pilot.press("space")
                    await wait_for(
                        lambda screen=screen: (
                            not screen._console_toggle_writes["model-thinking"].running
                        ),
                        "Post-reload choice saved",
                    )
                    assert (
                        tomllib.loads(config_path.read_text())["console"][
                            "show_model_thinking"
                        ]
                        is True
                    )
                    await pilot.press("space")
                    await wait_for(
                        lambda screen=screen: (
                            not screen._console_toggle_writes["model-thinking"].running
                        ),
                        "Original visibility restored",
                    )
                    assert (
                        tomllib.loads(config_path.read_text())["console"][
                            "show_model_thinking"
                        ]
                        is False
                    )
                    assert all(
                        getattr(app, key) is handle for key, handle in handles.items()
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "readonly_directory_failure": True,
                            "failure_restored_prior_value": True,
                            "keyboard_retry_saved": True,
                            "screen_removed_during_native_departure": screen_was_removed,
                            "departure_return_retained_value_and_receipt": True,
                            "diagnostics_reload_then_save": True,
                            "original_visibility_restored": True,
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
