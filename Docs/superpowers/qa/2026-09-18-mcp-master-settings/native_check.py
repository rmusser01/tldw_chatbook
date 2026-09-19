"""TASK-32793 native ordered master-switch persistence across both MCP controls.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
All workflow and fixture writes stay in the private profile.
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

    from textual.css.query import NoMatches, QueryError
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit

    claim_process_exit()
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
                "tldw_chatbook/UI/Screens/mcp_screen.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_rail.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py",
                "tldw_chatbook/MCP/local_config_saves.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_local_master_button.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_inspector.py",
                "tldw_chatbook/css/components/_agentic_terminal.tcss",
                "tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_workbench.py",
                "tldw_chatbook/Tool_Packs/service.py",
                "tldw_chatbook/Tool_Packs/export.py",
                "tldw_chatbook/Tool_Packs/publication.py",
                "tldw_chatbook/Tool_Packs/activation.py",
                "tldw_chatbook/Tool_Packs/importer.py",
                "tldw_chatbook/Tool_Packs/removal.py",
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
        import threading

        import toml
        from textual.widgets import Button

        from tldw_chatbook.config import get_cli_setting
        from tldw_chatbook.UI.MCP_Modules import mcp_workbench as module
        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 35
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await settle()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(label)

        async def settle():
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        def visible(control):
            region, clip = app.screen._compositor.visible_widgets[control]
            assert region.intersection(clip) == region

        async def focus(selector):
            control = app.screen.query_one(selector)
            assert not control.disabled
            control.focus()
            await settle()
            assert app.focused is control
            visible(control)
            if isinstance(control, Button):
                await wait_for(lambda: not control.has_class("-active"), "Button ready")
            return control

        def paint(region):
            return "\n".join(
                strip.crop(region.x, region.right).text
                for strip in app.screen._compositor.render_strips()[
                    region.y : region.bottom
                ]
            )

        def compact(text):
            return "".join(text.split())

        async def capture(stem):
            await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            assert app.console.file.isatty()
            result.update(driver="LinuxDriver", tty_streams=True, lock_acquired=True)
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Startup")
            result["live_database_attributes"] = sorted(
                key
                for key, value in vars(app).items()
                if key.endswith("_db") and value is not None
            )
            await app.handle_screen_navigation(NavigateToScreen("mcp"))
            await wait_for(
                lambda: getattr(app.screen, "screen_name", None) == "mcp",
                "MCP destination",
            )
            workbench = app.screen.workbench
            await wait_for(
                lambda: (
                    not workbench.is_loading
                    and not workbench._reloading
                    and bool(workbench.query("#mcp-perm-table"))
                ),
                "MCP loaded",
            )
            store = app.unified_mcp_service.permission_store
            policy_before = store.read_snapshot_strict().payload["profiles"]
            assert get_cli_setting("console", "local_tools_enabled", True) is True
            result["fixture_setup"] = (
                "Fresh private profile and real configuration service. "
                "Production navigation, keyboard controls; a controlled thread barrier "
                "holds the first write before delegating to the unmodified atomic writer. "
                "No provider/tool requests or permission writes."
            )
            real_write = module._persist_mcp_local_setting

            def receipt(selector):
                return str(app.screen.query_one(selector).renderable)

            async def mode(name):
                await focus("#mcp-mode-" + name)
                await pilot.press("enter")
                await settle()

            async def show_receipt(selector, text):
                control = app.screen.query_one(selector)
                control.scroll_visible(animate=False, immediate=True)
                await settle()
                visible(control)
                assert compact(text) in compact(paint(control.region))

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
                    await mode("tools")
                    entered, release = threading.Event(), threading.Event()
                    calls = []

                    def held(
                        request, value, calls=calls, entered=entered, release=release
                    ):
                        calls.append(value)
                        if value is False:
                            entered.set()
                            assert release.wait(35)
                        return real_write(request, value)

                    module._persist_mcp_local_setting = held
                    try:
                        await focus("#mcp-tools-local-enabled")
                        await pilot.press("enter")
                        await wait_for(entered.is_set, "Admitted off write held")
                        await workbench.reload()
                        await settle()
                        assert "off" in str(
                            app.screen.query_one("#mcp-tools-local-enabled").label
                        )
                        assert get_cli_setting("console", "local_tools_enabled") is True
                        await show_receipt(
                            "#mcp-tools-local-config-status", "Saving tools off"
                        )
                        await capture(stem + "-tools-pending")
                        await mode("servers")
                        key = next(
                            s.server_key
                            for s in workbench._snapshots
                            if s.source == "builtin"
                        )
                        await workbench._select_server_key(key)
                        await settle()
                        toggle = await focus("#mcp-gate-local_tools_enabled")
                        assert "off" in str(toggle.label)
                        assert not app.screen.query_one(
                            "#mcp-gate-local-master-off-note"
                        ).display
                        await pilot.press("enter")
                        await wait_for(
                            lambda: (
                                app._mcp_local_config_saves.state_for(
                                    "local_tools_enabled"
                                ).request.value
                                is True
                            ),
                            "Reversal admitted from Servers",
                        )
                        assert calls == [False]
                        assert not app.screen.query_one(
                            "#mcp-gate-local-master-off-note"
                        ).display
                        assert not app.screen.query_one(
                            "#mcp-gate-web_deep_search_enabled"
                        ).disabled
                        await show_receipt(
                            "#mcp-gate-local-master-status", "Saving tools on"
                        )
                        await capture(stem + "-servers-pending")
                        release.set()
                        await wait_for(
                            lambda: (
                                app._mcp_local_config_saves.state_for(
                                    "local_tools_enabled"
                                ).result
                                is not None
                                and "Tools on saved"
                                in receipt("#mcp-gate-local-master-status")
                            ),
                            "Both writes finished in order",
                        )
                        assert calls == [False, True]
                        assert (
                            toml.loads((root / "config.toml").read_text())["console"][
                                "local_tools_enabled"
                            ]
                            is True
                        )
                        assert get_cli_setting("console", "local_tools_enabled") is True
                        await show_receipt(
                            "#mcp-gate-local-master-status", "Tools on saved."
                        )
                        await capture(stem + "-servers-saved")
                        await mode("tools")
                        await focus("#mcp-tools-local-enabled")
                        await show_receipt(
                            "#mcp-tools-local-config-status", "Tools on saved."
                        )
                        await capture(stem + "-tools-saved")
                    finally:
                        release.set()
                        module._persist_mcp_local_setting = real_write
                    assert (
                        store.read_snapshot_strict().payload["profiles"]
                        == policy_before
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "pending_off_survives_refresh": True,
                            "pending_does_not_change_runtime_authority": True,
                            "servers_reverses_latest_pending_choice": True,
                            "writes_finish_in_order": calls,
                            "real_file_and_cache_agree": True,
                            "both_receipts_fully_painted": True,
                            "permission_profiles_unchanged": True,
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain failed-run diagnostics
            result.update(
                passed=False,
                error=traceback.format_exc(),
                focused_id=getattr(app.focused, "id", None),
                focused_region=str(getattr(app.focused, "region", None)),
            )
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
