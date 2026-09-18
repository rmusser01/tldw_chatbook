"""TASK-32764 native Full trace-view consent review with private configuration.

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
    from textual.widgets import Checkbox
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

            async def choose_viewer(value):
                viewer = await focus("#settings-console-trace-viewer-profile")
                await pilot.press(
                    "enter", "end" if value == "full" else "home", "enter"
                )
                assert viewer.value == value

            async def apply_capture(*, confirm=None):
                await focus("#settings-console-exchange-capture-apply")
                await pilot.press("enter")
                if confirm is not None:
                    await wait_for(
                        lambda: type(app.screen).__name__ == "ConfirmationDialog",
                        "Full viewer consent",
                    )
                    if confirm:
                        await focus("#confirm-button")
                        await pilot.press("enter")
                    else:
                        await pilot.press("escape")
                await wait_for(
                    lambda: (
                        type(app.screen).__name__ == "SettingsScreen"
                        and not app.screen._console_capture_applying
                    ),
                    "Capture settings settled",
                )

            def saved():
                return tomllib.loads(config_path.read_text())["console"]

            def assert_values(viewer, pii=False):
                value = saved()
                assert value["trace_viewer_profile"] == viewer
                assert value["exchange_capture"] is False
                assert value["exchange_capture_pii_redaction"] is pii
                policy = config.runtime_capture_policy()
                assert policy.viewer_profile == viewer
                assert policy.enabled is False
                assert policy.pii_redaction_enabled is pii

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
                    assert_values("safe")
                    await choose_viewer("full")
                    before = config_path.read_bytes()
                    await focus("#settings-console-exchange-capture-apply")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "ConfirmationDialog",
                        "Consent",
                    )
                    for selector in (
                        ".dialog-message",
                        "#cancel-button",
                        "#confirm-button",
                    ):
                        widget = app.screen.query_one(selector)
                        region, clip = app.screen._compositor.visible_widgets[widget]
                        assert region.intersection(clip) == region
                    assert "PII detectors missed" in app.screen.message
                    await focus("#cancel-button")
                    await capture(stem + "-consent")
                    await pilot.press("escape")
                    await wait_for(
                        lambda screen=screen: (
                            app.screen is screen
                            and not screen._console_capture_applying
                        ),
                        "Cancelled",
                    )
                    assert config_path.read_bytes() == before
                    assert (
                        screen.focused.id == "settings-console-exchange-capture-apply"
                    )
                    assert_values("safe")
                    os.chmod(root, 0o500)
                    try:
                        await apply_capture(confirm=True)
                        assert config_path.read_bytes() == before
                        assert_values("safe")
                        assert "Failed" in screen._console_capture_status
                        assert (
                            screen.focused.id
                            == "settings-console-exchange-capture-apply"
                        )
                        await capture(stem + "-failed")
                    finally:
                        os.chmod(root, 0o700)
                    await apply_capture(confirm=True)
                    assert_values("full")
                    await pilot.press("ctrl+2")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "ChatScreen",
                        "Console departure",
                    )
                    removed = not screen.is_attached
                    await pilot.press("f4")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "SettingsScreen",
                        "Settings return",
                    )
                    await category("Console Behavior", "console-behavior")
                    screen = app.screen
                    assert (
                        await focus("#settings-console-trace-viewer-profile")
                    ).value == "full"
                    await capture(stem + "-saved")
                    await choose_viewer("safe")
                    await apply_capture()
                    assert_values("safe")
                    # A newer independent PII choice must fence this pending consent.
                    await choose_viewer("full")
                    await focus("#settings-console-exchange-capture-apply")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "ConfirmationDialog",
                        "Stale consent",
                    )
                    mutation = config.apply_settings_mutation_to_cli_config(
                        {"console": {"exchange_capture_pii_redaction": True}}
                    )
                    assert mutation.failure_phase is None
                    await focus("#confirm-button")
                    await pilot.press("enter")
                    await wait_for(
                        lambda screen=screen: (
                            app.screen is screen
                            and not screen._console_capture_applying
                        ),
                        "Stale apply rejected",
                    )
                    assert "settings changed" in screen._console_capture_status
                    assert_values("safe", True)
                    await category("Diagnostics", "diagnostics")
                    await pilot.press("t")
                    await wait_for(
                        lambda screen=screen: (
                            screen._diagnostics_reload_result.startswith(
                                "Config reload: loaded"
                            )
                        ),
                        "Reload completed",
                    )
                    await category("Console Behavior", "console-behavior")
                    assert (
                        screen.query_one(
                            "#settings-console-trace-pii-redaction", Checkbox
                        ).value
                        is True
                    )
                    await choose_viewer("full")
                    await apply_capture(confirm=True)
                    assert_values("full", True)
                    await choose_viewer("safe")
                    await focus("#settings-console-trace-pii-redaction")
                    await pilot.press("space")
                    await apply_capture()
                    assert_values("safe")
                    assert all(
                        getattr(app, key) is handle for key, handle in handles.items()
                    )
                    controller = getattr(
                        getattr(app, "console_runtime", None), "chat_controller", None
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "full_disclosure_visible": True,
                            "escape_preserves_file_and_safe_view": True,
                            "readonly_failure_preserves_safe_view": True,
                            "keyboard_retry_saves_full_view": True,
                            "settings_removed_on_departure": removed,
                            "recreated_settings_load_full_view": True,
                            "stale_consent_rejected": True,
                            "reload_then_confirmation_saves": True,
                            "pii_choice_preserved_independently": True,
                            "original_capture_and_viewer_values_restored": True,
                            "active_controller_session": bool(
                                controller and controller.store.active_session_id
                            ),
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
