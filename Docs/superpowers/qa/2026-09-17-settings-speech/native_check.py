"""TASK-32756 native Speech Settings review with private configuration.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
No speech runtime, synthesis, model download or external provider is invoked.
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

import toml


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
    from textual.widgets import Button, Select, Static
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
                "tldw_chatbook/UI/Screens/settings_speech_tts.py",
                "tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py",
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

        async def settle():
            await wait_for(
                lambda: not getattr(app.screen, "_category_pane_swap_pending", False),
                "Category settled",
            )
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def category(name):
            await pilot.press("escape", "/", *name, "enter")
            await settle()

        async def tab_to(selector):
            for _ in range(100):
                target = app.screen.query_one(selector)
                if app.screen.focused is target:
                    await wait_for(
                        lambda: not app.screen.query("Toast"),
                        "Notices clear before inspecting focus",
                    )
                    await pilot.wait_for_scheduled_animations()
                    await pilot.pause()
                    geometry = app.screen._compositor.visible_widgets.get(target)
                    assert (
                        geometry
                        and geometry[0].intersection(geometry[1]) == geometry[0]
                    ), selector
                    if isinstance(target, Button):
                        region = target.region
                        strips = app.screen._compositor.render_strips()
                        painted = "\n".join(
                            strips[y].crop(region.x, region.right).text
                            for y in range(
                                max(0, region.y), min(len(strips), region.bottom)
                            )
                        )
                        assert str(target.label) in painted, (selector, painted)
                    return target
                await pilot.press("tab")
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            raise AssertionError("Tab did not reach " + selector)

        async def capture(stem):
            await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert (
                sys.stdout.isatty()
                and sys.stderr.isatty()
                and app.console.file.isatty()
            )
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Startup")
            await pilot.press("f4")
            await wait_for(
                lambda: type(app.screen).__name__ == "SettingsScreen", "Settings"
            )
            baseline = toml.loads(config_path.read_text())
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
                    await category("Speech & TTS")
                    await wait_for(
                        lambda: bool(app.screen.query("#settings-speech-tts-panel")),
                        "Speech panel",
                    )
                    screen = app.screen
                    panel = screen.query_one("#settings-speech-tts-panel")
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    before = config_path.read_bytes()
                    for field in (
                        "default-profile",
                        "default-provider",
                        "model-policy",
                        "model-value",
                        "voice-policy",
                        "voice-value",
                        "output-format",
                        "speed",
                        "configure-provider",
                        "openai-base-url",
                    ):
                        await tab_to(f"#settings-speech-{field}")
                    for action in (
                        "save",
                        "revert",
                        "restore-defaults",
                        "open-lab-bottom",
                    ):
                        await tab_to(f"#settings-speech-{action}")
                    speed = await tab_to("#settings-speech-speed")
                    await pilot.press("home", "shift+end", "backspace", *"invalid")
                    await tab_to("#settings-speech-save")
                    await pilot.press("enter")
                    await wait_for(
                        lambda screen=screen, speed=speed: screen.focused is speed,
                        "Validation focus",
                    )
                    assert str(
                        screen.query_one(
                            "#settings-speech-speed-error", Static
                        ).renderable
                    )
                    assert config_path.read_bytes() == before
                    await tab_to("#settings-speech-revert")
                    await pilot.press("enter")
                    await wait_for(
                        lambda panel=panel: not panel.has_unsaved_changes(), "Revert"
                    )
                    await tab_to("#settings-speech-voice-value")
                    await capture(stem + "-voice")
                    await tab_to("#settings-speech-browse-voices")
                    await tab_to("#settings-speech-voice-value")
                    await pilot.press("enter", "end", "enter")
                    await wait_for(
                        lambda: bool(
                            app.screen.query("#settings-speech-custom-id-value")
                        ),
                        "Custom ID dialog",
                    )
                    await tab_to("#settings-speech-custom-id-value")
                    custom_voice = "review-" + stem
                    await pilot.press("home", "shift+end", "backspace", *custom_voice)
                    await tab_to("#settings-speech-custom-id-confirm")
                    await capture(stem + "-custom")
                    await pilot.press("enter")
                    await wait_for(
                        lambda screen=screen, custom_voice=custom_voice: (
                            app.screen is screen
                            and screen.query_one(
                                "#settings-speech-voice-value", Select
                            ).value
                            == custom_voice
                        ),
                        "Custom ID draft",
                    )
                    assert (
                        panel.has_unsaved_changes()
                        and config_path.read_bytes() == before
                    )
                    await tab_to("#settings-speech-save")
                    await pilot.press("enter")
                    await wait_for(
                        lambda panel=panel: (
                            not panel.has_unsaved_changes()
                            and panel._latest_request_id is None
                        ),
                        "Real local Save",
                    )
                    assert "saved" in panel.result_text.lower(), panel.result_text
                    saved = toml.loads(config_path.read_text())
                    expected = toml.loads(toml.dumps(baseline))
                    expected["app_tts"]["default_voice"] = custom_voice
                    expected["tts_settings"]["default_tts_voice"] = custom_voice
                    assert saved == expected, "Only the default voice may change"
                    baseline = saved
                    await tab_to("#settings-speech-save")
                    await capture(stem + "-saved")
                    await tab_to("#settings-speech-speed")
                    await pilot.press("home", "shift+end", "backspace", *"2.0")
                    for choice in ("cancel", "discard"):
                        await pilot.press("escape", "/", *"Overview", "enter")
                        await wait_for(
                            lambda: bool(
                                app.screen.query("#global-speech-tts-leave-cancel")
                            ),
                            "Leave dialog",
                        )
                        for action in ("cancel", "discard", "save"):
                            await tab_to("#global-speech-tts-leave-" + action)
                        await tab_to("#global-speech-tts-leave-" + choice)
                        if choice == "discard":
                            await capture(stem + "-leave")
                        await pilot.press("enter")
                        await wait_for(
                            lambda screen=screen: app.screen is screen,
                            "Leave dialog dismissed",
                        )
                        await settle()
                        if choice == "cancel":
                            assert (
                                screen.active_category == "speech-tts"
                                and panel.has_unsaved_changes()
                            )
                        else:
                            assert screen.active_category == "overview"
                    assert toml.loads(config_path.read_text()) == saved
                    await category("Speech & TTS")
                    panel = screen.query_one("#settings-speech-tts-panel")
                    await tab_to("#settings-speech-speed")
                    speed_value = 1.5 if size[0] == 190 else 1.25
                    await pilot.press(
                        "home", "shift+end", "backspace", *str(speed_value)
                    )
                    await pilot.press("escape", "/", *"Overview", "enter")
                    await wait_for(
                        lambda: bool(app.screen.query("#global-speech-tts-leave-save")),
                        "Save and continue dialog",
                    )
                    await tab_to("#global-speech-tts-leave-save")
                    await pilot.press("enter")
                    await wait_for(
                        lambda screen=screen: (
                            app.screen is screen
                            and screen.active_category == "overview"
                        ),
                        "Saved before navigation",
                    )
                    expected["app_tts"]["default_speed"] = speed_value
                    expected["tts_settings"]["default_openai_tts_speed"] = speed_value
                    assert toml.loads(config_path.read_text()) == expected
                    baseline = expected
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "visible_keyboard_actions": True,
                            "local_validation_and_revert": True,
                            "custom_voice_exact_saved_delta": True,
                            "guarded_leave_cancel_discard_and_save": True,
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
