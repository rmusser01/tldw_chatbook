"""TASK-32757 native Interface Settings review with private configuration.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Only local Appearance, Theme and Splash settings are edited.
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
    from textual.widgets import Button, Input, Select
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
                "tldw_chatbook/Widgets/settings_theme_editor.py",
                "tldw_chatbook/Widgets/settings_splash_screen_viewer.py",
                "tldw_chatbook/css/components/_settings_splash_theme.tcss",
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
            assert tuple(app.size) == size, (stem, app.size, size)
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

            async def edit(selector, value):
                field = await tab_to(selector)
                await pilot.press("home", "shift+end", "backspace", *str(value))
                await settle()
                assert field.value == str(value)
                return field

            async def choose(selector, value):
                select = await tab_to(selector)
                options = [item[1] for item in select._options]
                index = options.index(value)
                await pilot.press("enter", "home", *(["down"] * index), "enter")
                await settle()
                assert select.value == value

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
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    await category("Appearance")
                    screen = app.screen
                    before = config_path.read_bytes()
                    original_font = screen.query_one(
                        "#settings-appearance-font-size", Input
                    ).value
                    for suffix in (
                        "theme",
                        "palette-theme-limit",
                        "font-size",
                        "density",
                        "transcript-style",
                        "character-expression-mode",
                        "animations-enabled",
                        "reduce-motion",
                        "ascii-glyphs",
                        "smooth-scrolling",
                    ):
                        await tab_to(f"#settings-appearance-{suffix}")
                    await edit("#settings-appearance-font-size", "99")
                    await pilot.press("escape", "s")
                    await settle()
                    assert config_path.read_bytes() == before
                    await capture(stem + "-appearance-invalid")
                    await pilot.press("escape", "r")
                    await wait_for(
                        lambda: bool(app.screen.query("#confirm-button")),
                        "Revert dialog",
                    )
                    await tab_to("#confirm-button")
                    await pilot.press("enter")
                    await settle()
                    assert (
                        screen.query_one("#settings-appearance-font-size", Input).value
                        == original_font
                    )
                    font = 18 if size[0] == 190 else 17
                    await edit("#settings-appearance-font-size", font)
                    await tab_to("#settings-preview-appearance")
                    await pilot.press("enter")
                    await settle()
                    assert config_path.read_bytes() == before
                    await pilot.press("escape", "s")
                    await wait_for(
                        lambda font=font: (
                            toml.loads(config_path.read_text())["web_server"][
                                "font_size"
                            ]
                            == font
                        ),
                        "Appearance saved",
                    )
                    baseline["web_server"]["font_size"] = font
                    assert toml.loads(config_path.read_text()) == baseline
                    await tab_to("#settings-appearance-font-size")
                    await capture(stem + "-appearance-saved")

                    app.theme = theme
                    await category("Theme")
                    editor = screen.query_one("#settings-theme-editor")
                    tree = await tab_to("#settings-theme-tree")
                    await pilot.press("home")
                    for _ in range(150):
                        if str(tree.cursor_node.label) == "textual-light":
                            break
                        if (
                            tree.cursor_node.allow_expand
                            and not tree.cursor_node.is_expanded
                        ):
                            await pilot.press("right")
                        else:
                            await pilot.press("down")
                    assert str(tree.cursor_node.label) == "textual-light"
                    await pilot.press("enter")
                    await settle()
                    assert app.theme == theme
                    await tab_to("#settings-theme-new")
                    await pilot.press("enter")
                    await settle()
                    name = f"review_{theme.replace('-', '_')}_{size[0]}"
                    await edit("#settings-theme-name", name)
                    await edit("#settings-theme-color-primary", "#2277AA")
                    assert app.theme == theme
                    await capture(stem + "-theme-palette")
                    await tab_to("#settings-theme-apply")
                    await pilot.press("enter")
                    await settle()
                    assert app.theme == "custom_" + name
                    await tab_to("#settings-theme-save")
                    await pilot.press("enter")
                    theme_path = editor.custom_themes_path / (name + ".toml")
                    await wait_for(
                        lambda theme_path=theme_path, editor=editor: (
                            theme_path.exists() and not editor.is_modified
                        ),
                        "Theme file saved",
                    )
                    assert (
                        toml.loads(theme_path.read_text())["colors"]["primary"]
                        == "#2277AA"
                    )
                    assert toml.loads(config_path.read_text()) == baseline
                    await tab_to("#settings-theme-set-default")
                    await pilot.press("enter")
                    await settle()
                    baseline["general"]["default_theme"] = name
                    assert toml.loads(config_path.read_text()) == baseline
                    await capture(stem + "-theme-actions")
                    await category("Appearance")
                    assert (
                        screen.query_one("#settings-appearance-theme", Select).value
                        == name
                    )

                    app.theme = theme
                    await category("Splash Screen")
                    viewer = screen.query_one("#settings-splash-screen-viewer")
                    for suffix in ("enabled", "show-progress", "skip-on-keypress"):
                        control = await tab_to(f"#settings-splash-{suffix}")
                        value = not control.value
                        await pilot.press("space")
                        key = suffix.replace("-", "_")
                        await wait_for(
                            lambda key=key, value=value, viewer=viewer: (
                                viewer._config[key] == value
                                and key not in viewer._pending_values
                            ),
                            "Splash toggle saved",
                        )
                        baseline["splash_screen"][key] = value
                    for suffix, value in (
                        ("duration", 1.75 if size[0] == 190 else 1.25),
                        ("animation-speed", 1.5 if size[0] == 190 else 1.25),
                    ):
                        await edit(f"#settings-splash-{suffix}", value)
                        await pilot.press("enter")
                        key = suffix.replace("-", "_")
                        await wait_for(
                            lambda key=key, value=value, viewer=viewer: (
                                viewer._config[key] == value
                                and key not in viewer._pending_values
                            ),
                            "Splash number saved",
                        )
                        section = (
                            baseline["splash_screen"]["effects"]
                            if key == "animation_speed"
                            else baseline["splash_screen"]
                        )
                        section[key] = value
                    card = "default" if size[0] == 190 else "minimal_fade"
                    await choose("#settings-splash-default-select", card)
                    await wait_for(
                        lambda viewer=viewer: not viewer._pending_values,
                        "Default card saved",
                    )
                    baseline["splash_screen"]["card_selection"] = card
                    assert toml.loads(config_path.read_text()) == baseline
                    default_card = await tab_to("#settings-splash-default-select")
                    region = default_card.region
                    strips = screen._compositor.render_strips()
                    painted = "\n".join(
                        strips[y].crop(region.x, region.right).text
                        for y in range(region.y, region.bottom)
                    )
                    assert card in painted, (card, painted)
                    await capture(stem + "-splash-default")
                    await tab_to("#settings-splash-animation-speed")
                    await capture(stem + "-splash-settings")
                    card_list = await tab_to("#settings-splash-card-list")
                    index = [option.id for option in card_list.options].index(card)
                    await pilot.press("home", *(["down"] * index))
                    await settle()
                    assert viewer.selected_card == card
                    await tab_to("#settings-splash-play")
                    await pilot.press("enter")
                    await settle()
                    assert viewer.query_one("#settings-splash-preview-scroll").children
                    await capture(stem + "-splash-preview")
                    assert toml.loads(config_path.read_text()) == baseline
                    await category("Overview")
                    await category("Splash Screen")
                    reloaded = screen.query_one("#settings-splash-screen-viewer")
                    assert (
                        reloaded._config["animation_speed"]
                        == baseline["splash_screen"]["effects"]["animation_speed"]
                    )
                    assert reloaded._config["card_selection"] == card
                    await category("Overview")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "appearance_validation_preview_revert_save": True,
                            "theme_browse_edit_apply_save_launch": True,
                            "launch_default_visible_in_appearance": True,
                            "splash_exact_saved_values_and_preview": True,
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
