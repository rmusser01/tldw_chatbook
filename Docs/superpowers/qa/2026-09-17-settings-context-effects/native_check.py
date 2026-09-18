"""TASK-32767 native Console context and background effects with private configuration.

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
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli

    probe_terminal()
    app = TldwCli()
    assert app.app_config["console"]["background_effects"]["fps"] == 6
    source = Path(__file__).resolve().parents[4]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            p: hashlib.sha256((source / p).read_bytes()).hexdigest()
            for p in (
                "tldw_chatbook/UI/Screens/settings_screen.py",
                "tldw_chatbook/UI/Screens/chat_screen.py",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
                "tldw_chatbook/css/features/_settings.tcss",
                "tldw_chatbook/Utils/console_background_effects.py",
                "tldw_chatbook/Widgets/Console/console_background_effect.py",
                "tldw_chatbook/Widgets/Console/console_transcript.py",
                "tldw_chatbook/css/widget_defaults_self.tcss",
                "tldw_chatbook/Chat/console_context_policy.py",
                "tldw_chatbook/config.py",
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

            from tldw_chatbook.app import TabNavigationProvider
            from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
            from tldw_chatbook.Chat.console_context_policy import merge_context_policy
            from tldw_chatbook.Constants import TAB_CHAT

            async def console_screen():
                TabNavigationProvider(screen=app.screen).switch_tab(TAB_CHAT)
                await wait_for(
                    lambda: type(app.screen).__name__ == "ChatScreen", "Console"
                )
                await wait_for(
                    lambda: bool(app.screen.query("#console-native-transcript")),
                    "Transcript",
                )
                return app.screen

            async def choose(selector, index, expected):
                control = await focus(selector)
                await pilot.press("enter", "home", *(["down"] * index), "enter")
                await wait_for(
                    lambda: control.value == expected, "Selected " + selector
                )
                return control

            def painted(widget):
                region = widget.region
                return "\n".join(
                    strip.crop(region.x, region.right).text
                    for strip in app.screen._compositor.render_strips()[
                        max(0, region.y) : region.bottom
                    ]
                )

            prefix = "#settings-console-context-"
            background = "#settings-console-background-effect-"
            console = await console_screen()
            transcript = console.query_one("#console-native-transcript")
            store = console._ensure_console_chat_store()
            chat_session = store.ensure_session()
            for role, text in (
                (ConsoleMessageRole.USER, "Keep this sample question visible."),
                (ConsoleMessageRole.ASSISTANT, "Keep this sample answer unchanged."),
            ):
                message = store.append_message(chat_session.id, role=role, content=text)
                if role is ConsoleMessageRole.USER:
                    user_message_id = message.id
            await console._sync_native_console_chat_ui()
            await transcript.refresh_messages()
            await wait_for(
                lambda: "Keep this sample answer unchanged." in painted(transcript),
                "Seeded transcript is painted",
            )
            before_transcript = transcript.to_plain_text()
            assert "Keep this sample question visible." in before_transcript
            assert "Keep this sample answer unchanged." in before_transcript
            result["transcript_fixture"] = (
                "Two local synthetic messages; no provider call"
            )
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
                    await pilot.press("f4")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "SettingsScreen",
                        "Settings",
                    )
                    await category("Console Behavior", "console-behavior")
                    screen = app.screen
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    before = config_path.read_bytes()
                    await edit(prefix + "target-percent", "75")
                    await pilot.press("escape", "s")
                    await wait_for(
                        lambda screen=screen: (
                            "15 percentage points" in screen._console_behavior_result
                        ),
                        "Invalid ratio refused",
                    )
                    assert config_path.read_bytes() == before
                    await choose(prefix + "budget-mode", 1, "custom")
                    await edit(prefix + "budget-tokens", "64000")
                    await choose(prefix + "compaction-mode", 1, "automatic")
                    await choose(prefix + "compaction-representation", 2, "hybrid")
                    await edit(prefix + "trigger-percent", "85")
                    await edit(prefix + "target-percent", "55")
                    await edit(prefix + "summary-max-tokens", "1536")
                    await choose(prefix + "failure-behavior", 1, "omit_older_context")
                    await choose(
                        prefix + "carry-forward-mode", 1, "memory_with_latest_exchange"
                    )
                    toggle = await focus(background + "enabled")
                    if not toggle.value:
                        await pilot.press("space")
                        await wait_for(
                            lambda toggle=toggle: toggle.value, "Effects enabled"
                        )
                    await choose(background + "type", 3, "matrix")
                    await choose(background + "scope", 1, "transcript")
                    assert (
                        "Workbench scope is not available"
                        in screen._console_behavior_result
                    )
                    await choose(background + "intensity", 2, "high")
                    await edit(background + "fps", "13")
                    await pilot.press("escape", "s")
                    await wait_for(
                        lambda screen=screen: (
                            "between 1 and 12" in screen._console_behavior_result
                        ),
                        "Invalid frame rate refused",
                    )
                    assert config_path.read_bytes() == before
                    await edit(background + "fps", "9")
                    await pilot.press("escape", "s")
                    await wait_for(
                        lambda screen=screen: (
                            not screen._category_has_unsaved_changes(
                                screen.active_category
                            )
                        ),
                        "Saved context and effects",
                    )
                    saved = tomllib.loads(config_path.read_text())["console"]
                    assert saved["compaction_target_ratio"] == 0.55
                    assert saved["conversation_budget_tokens"] == 64000
                    assert saved["background_effects"] == {
                        "enabled": True,
                        "effect": "matrix",
                        "scope": "transcript",
                        "intensity": "high",
                        "fps": 9,
                    }
                    await wait_for(
                        lambda: not app.screen.query("Toast"), "Save notice clear"
                    )
                    target = await focus(prefix + "target-percent")
                    label = target.parent.query_one(".settings-input-label")
                    assert "Reduce context to (%)" in painted(label)
                    await capture(stem + "-context")
                    await focus(background + "fps")
                    await capture(stem + "-effects-settings")
                    console = await console_screen()
                    effect = console.query_one("#console-transcript-background-effect")
                    await wait_for(
                        lambda effect=effect: (
                            effect.is_effect_active and effect._timer is not None
                        ),
                        "Effect active",
                    )
                    assert effect.frame_rate == 9
                    assert effect.settings.effect == "matrix"
                    assert not effect.can_focus
                    assert console.query_one("#console-native-transcript") is transcript
                    assert transcript.to_plain_text() == before_transcript
                    controller = console._ensure_console_chat_controller()
                    policy = merge_context_policy(
                        global_overrides=controller._global_context_policy_overrides()
                    )
                    assert policy.custom_budget_tokens == 64000
                    assert policy.budget_mode.value == "custom"
                    assert policy.compaction_mode.value == "automatic"
                    assert policy.summary_max_tokens == 1536
                    assert policy.failure_behavior.value == "omit_older_context"
                    assert (
                        policy.carry_forward_mode.value == "memory_with_latest_exchange"
                    )
                    assert policy.compaction_representation.value == "hybrid"
                    assert policy.target_ratio == 0.55 and policy.trigger_ratio == 0.85
                    first_row = transcript.query_one(
                        f"#console-message-{user_message_id}"
                    )
                    region = transcript.content_region
                    exposed_rows = max(0, first_row.region.y - region.y - 1)
                    if exposed_rows >= 2:
                        background_paint = "".join(
                            strip.crop(region.x, region.right).text
                            for strip in app.screen._compositor.render_strips()[
                                region.y : first_row.region.y - 1
                            ]
                        )
                        assert any(char.isalnum() for char in background_paint)
                    await capture(stem + "-console-active")
                    await pilot.press("f4")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "SettingsScreen",
                        "Settings return",
                    )
                    await category("Console Behavior", "console-behavior")
                    toggle = await focus(background + "enabled")
                    await pilot.press("space", "escape", "s")
                    await wait_for(
                        lambda: (
                            not app.screen._category_has_unsaved_changes(
                                app.screen.active_category
                            )
                        ),
                        "Effects disabled save",
                    )
                    console = await console_screen()
                    await wait_for(
                        lambda effect=effect: (
                            not effect.is_effect_active and effect._timer is None
                        ),
                        "Effect stopped",
                    )
                    assert console.query_one("#console-native-transcript") is transcript
                    assert transcript.to_plain_text() == before_transcript
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "invalid_ratio_and_fps_refused": True,
                            "context_default_controller_parity": True,
                            "workbench_fallback": True,
                            "live_effect_started_and_stopped": True,
                            "exposed_background_rows": exposed_rows,
                            "effect_painted_when_exposed": exposed_rows >= 2,
                            "transcript_identity_and_text_unchanged": True,
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

    app.run(auto_pilot=journey, size=(170, 48))
    result.update(app_run_returned=True)
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
