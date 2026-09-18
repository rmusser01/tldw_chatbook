"""TASK-32779 native Tool Profile export recovery.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
All export and fixture writes stay in the private profile.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
import zipfile
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

    from textual.css.query import NoMatches, QueryError
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
                "tldw_chatbook/app.py",
                "tldw_chatbook/Tool_Packs/service.py",
                "tldw_chatbook/Tool_Packs/export.py",
                "tldw_chatbook/Tool_Packs/publication.py",
                "tldw_chatbook/Widgets/enhanced_file_picker.py",
                "tldw_chatbook/Widgets/Settings_Widgets/tool_pack_import_review.py",
                "tldw_chatbook/Widgets/Settings_Widgets/tool_profiles_panel.py",
                "tldw_chatbook/Workspaces/agent_provisioning.py",
                "tldw_chatbook/UI/Screens/settings_screen.py",
                "tldw_chatbook/Widgets/workspace_persona_default.py",
                "tldw_chatbook/Widgets/workspace_create_modal.py",
                "tldw_chatbook/Character_Chat/local_character_persona_service.py",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
                "tldw_chatbook/css/features/_settings.tcss",
                "tldw_chatbook/css/core/_variables.tcss",
                "tldw_chatbook/css/components/_dialogs.tcss",
                "tldw_chatbook/css/widget_defaults_scoped.tcss",
                "tldw_chatbook/css/widget_defaults_self.tcss",
                "tldw_chatbook/css/screen_agentic_settings.tcss",
                "tldw_chatbook/Workspaces/registry_service.py",
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
                await pilot.pause()
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
                if isinstance(control, Button):
                    await wait_for(
                        lambda: not control.has_class("-active"), "Button ready"
                    )
                region, clip = app.screen._compositor.visible_widgets[control]
                assert region.intersection(clip) == region
                return control

            async def edit(selector, value):
                field = await focus(selector)
                await pilot.press("home", "shift+end", "backspace", *value)
                await wait_for(lambda: field.value == value, "Field edit")
                return field

            async def capture(stem):
                await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
                app.save_screenshot(stem + ".svg", path=str(evidence))
                pane = await tmux("capture-pane", "-p", "-t", session)
                (evidence / (stem + ".txt")).write_text(pane.stdout)

            from textual.widgets import Button, Input, Static

            from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave
            from tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review import (
                ToolPackExportReviewModal,
            )

            await category("Tool Profiles", "tool-profiles")
            await wait_for(
                lambda: app.tool_pack_service is not None, "Tool Profile service"
            )
            store = app.unified_mcp_service.permission_store
            await asyncio.to_thread(store.ensure_profile, "native-export")
            policy_before = store.path.read_bytes()
            exports = root / "exports"
            exports.mkdir()
            result["fixture_setup"] = (
                "One private local profile; real app inventory, capture, review and publication. No provider requests."
            )

            async def press(selector):
                await focus(selector)
                await pilot.press("enter")
                await pilot.pause()

            async def begin_export():
                settings = app.screen
                settings._request_tool_profiles_listing()
                await wait_for(
                    lambda settings=settings: (
                        settings._tool_profiles_listing.by_id("native-export")
                        is not None
                    ),
                    "Profile listing",
                )
                await wait_for(
                    lambda settings=settings: any(
                        str(w.renderable) == "native-export"
                        for w in settings.query(".tool-profile-title")
                    ),
                    "Rendered profile",
                )
                index = next(
                    i
                    for i, row in enumerate(settings._tool_profiles_listing.profiles)
                    if row.profile_id == "native-export"
                )
                await press(f"#tool-profile-export-{index}")
                await wait_for(
                    lambda: isinstance(app.screen, ToolPackExportReviewModal),
                    "Export review",
                )
                return settings

            for theme in ("textual-dark", "textual-light"):
                for size in ((80, 24), (170, 48)):
                    stem = f"{theme}-{size[0]}x{size[1]}"
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
                        lambda size=size: (app.size.width, app.size.height) == size,
                        "Terminal resized",
                    )
                    await category("Tool Profiles", "tool-profiles")
                    settings = await begin_export()
                    await focus("#tool-pack-export-continue")
                    await capture(stem + "-review")
                    await press("#tool-pack-export-continue")
                    await wait_for(
                        lambda: isinstance(app.screen, EnhancedFileSave), "Save picker"
                    )
                    first = app.screen
                    existing = exports / (stem + "-existing.tldw-tool-pack")
                    existing.write_bytes(b"existing archive must remain unchanged")
                    await edit("#filename-input", str(existing))
                    await pilot.press("enter")
                    await wait_for(
                        lambda first=first: (
                            isinstance(app.screen, EnhancedFileSave)
                            and app.screen is not first
                        ),
                        "Destination recovery",
                    )
                    await wait_for(
                        lambda: (
                            "new filename"
                            in str(
                                app.screen.query_one("#error-line", Static).renderable
                            )
                        ),
                        "Recovery copy",
                    )
                    assert (
                        app.screen.query_one("#filename-input", Input).value
                        == existing.name
                    )
                    await focus("#filename-input")
                    await capture(stem + "-recovery")
                    target = exports / (stem + "-saved.tldw-tool-pack")
                    await edit("#filename-input", target.name)
                    await pilot.press("enter")
                    await wait_for(
                        lambda settings=settings: (
                            app.screen is settings
                            and settings._tool_profiles_result.startswith(
                                "Exported Tool Pack"
                            )
                        ),
                        "Archive saved",
                    )
                    assert (
                        existing.read_bytes()
                        == b"existing archive must remain unchanged"
                    )
                    with zipfile.ZipFile(target) as archive:
                        manifest = json.loads(archive.read("tool-pack.json"))
                    assert manifest["profile"]["suggested_id"] == "native-export"
                    settings.query_one("#tool-profiles-result").scroll_visible(
                        animate=False
                    )
                    await pilot.pause()
                    await capture(stem + "-saved")
                    await begin_export()
                    await press("#tool-pack-export-continue")
                    await wait_for(
                        lambda: isinstance(app.screen, EnhancedFileSave),
                        "Cancel picker",
                    )
                    await edit("#filename-input", str(existing))
                    first = app.screen
                    await pilot.press("enter")
                    await wait_for(
                        lambda first=first: (
                            isinstance(app.screen, EnhancedFileSave)
                            and app.screen is not first
                        ),
                        "Cancel recovery",
                    )
                    await pilot.press("escape")
                    await wait_for(
                        lambda settings=settings: (
                            app.screen is settings
                            and settings._tool_profiles_result == "Export cancelled"
                        ),
                        "Export cancelled",
                    )
                    assert (
                        len(list(exports.iterdir())) == (len(result["cells"]) + 1) * 2
                    )
                    assert (
                        existing.read_bytes()
                        == b"existing archive must remain unchanged"
                    )
                    assert store.path.read_bytes() == policy_before
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "archive_sha256": hashlib.sha256(
                                target.read_bytes()
                            ).hexdigest(),
                            "manifest": manifest,
                            "existing_preserved": True,
                            "cancel_no_write": True,
                            "policy_unchanged": True,
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve failed-run diagnostics
            result.update(
                passed=False,
                error=traceback.format_exc(),
                focused_id=getattr(app.focused, "id", None),
                focused_region=str(getattr(app.focused, "region", None)),
            )
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            if type(app.screen).__name__ in {
                "ToolPackExportReviewModal",
                "EnhancedFileSave",
            }:
                await pilot.press("escape")
                await pilot.pause()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result.update(app_run_returned=True)
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
