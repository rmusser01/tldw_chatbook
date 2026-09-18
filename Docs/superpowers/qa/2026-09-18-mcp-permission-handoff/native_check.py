"""TASK-32785 native Settings Edit handoff and permission matrix visibility.

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

            from textual.widgets import Button, DataTable, Static

            from tldw_chatbook.Tool_Packs.publication import CapturedToolPackDestination
            from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import (
                MCPPermissionsMode,
            )
            from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
            from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen
            from tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review import (
                ToolPackImportOptionsModal,
                ToolPackImportReviewModal,
            )
            from tldw_chatbook.Widgets.Settings_Widgets.tool_profiles_panel import (
                ToolProfilesPanel,
            )

            await category("Tool Profiles", "tool-profiles")
            await wait_for(
                lambda: app.tool_pack_service is not None, "Tool Profile service"
            )
            service = app.tool_pack_service
            store = app.unified_mcp_service.permission_store
            await asyncio.to_thread(store.ensure_profile, "native-source")
            fixture = root / "native-source.tldw-tool-pack"
            review = await asyncio.to_thread(
                service.capture_export,
                "native-source",
                display_name="Native fixture",
                suggested_id="native-source",
            )
            destination = await asyncio.to_thread(
                CapturedToolPackDestination.capture, fixture
            )
            await asyncio.to_thread(service.publish_export, review, destination)
            fixture_digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            result["fixture_setup"] = (
                "One private source profile and real service-exported archive. Import, Edit routing and permission mutations use the real UI/services. No provider requests."
            )

            async def press(selector):
                await focus(selector)
                await pilot.press("enter")
                await pilot.pause()

            async def expect_focus(selector):
                control = app.screen.query_one(selector)
                await wait_for(lambda: app.focused is control, "Focus " + selector)
                await pilot.wait_for_scheduled_animations()
                region, clip = app.screen._compositor.visible_widgets[control]
                assert region.intersection(clip) == region
                return control

            async def begin_import(profile_id):
                await press("#tool-profiles-import")
                await wait_for(
                    lambda: isinstance(app.screen, EnhancedFileOpen),
                    "Import file picker",
                )
                await pilot.press("ctrl+l")
                await edit("#path-input", str(fixture))
                await pilot.press("enter")
                await wait_for(
                    lambda: isinstance(app.screen, ToolPackImportOptionsModal),
                    "Import options",
                )
                await edit("#tool-pack-import-profile-id", profile_id)

            async def inspect():
                await press("#tool-pack-import-options-review")
                await wait_for(
                    lambda: isinstance(app.screen, ToolPackImportReviewModal),
                    "Import review",
                )

            async def listing_ready(settings, profile_id):
                await wait_for(
                    lambda: (
                        settings._tool_profiles_listing.by_id(profile_id) is not None
                    ),
                    "Listed " + profile_id,
                )
                await wait_for(
                    lambda: (
                        profile_id in settings.query_one(ToolProfilesPanel).profile_ids
                    ),
                    "Rendered " + profile_id,
                )

            def action_selector(settings, profile_id, action):
                index = settings.query_one(ToolProfilesPanel).profile_ids.index(
                    profile_id
                )
                return f"#tool-profile-{action}-{index}"

            for theme in ("textual-dark", "textual-light"):
                for size in ((80, 24), (170, 48)):
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    profile_id = f"native-{theme.removeprefix('textual-')}-{size[0]}"
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
                    settings = app.screen
                    settings._request_tool_profiles_listing()
                    await listing_ready(settings, "native-source")
                    await begin_import(profile_id)
                    await inspect()
                    await press("#tool-pack-import-unbound")
                    await wait_for(
                        lambda settings=settings, profile_id=profile_id: (
                            app.screen is settings
                            and settings._tool_profiles_result.startswith(
                                "Imported " + profile_id
                            )
                        ),
                        "Imported unbound",
                    )
                    await listing_ready(settings, profile_id)
                    await expect_focus("#tool-profiles-import")
                    default_policy = store.read_snapshot_strict().payload["profiles"][
                        "default"
                    ]
                    profile = store.read_snapshot_strict().payload["profiles"][
                        profile_id
                    ]
                    revision = profile["tool_pack_lifecycle"]["revision"]
                    await press(action_selector(settings, profile_id, "edit"))
                    await wait_for(
                        lambda: isinstance(app.screen, MCPScreen), "MCP destination"
                    )
                    workbench = app.screen.workbench
                    await wait_for(
                        lambda workbench=workbench: (
                            workbench is not None
                            and not workbench.is_loading
                            and not workbench._reloading
                            and workbench._pending_view_state is None
                        ),
                        "Permissions loaded",
                    )
                    table = await expect_focus("#mcp-perm-table")
                    assert isinstance(table, DataTable)
                    canvas = workbench.query_one(MCPPermissionsMode)
                    assert workbench._tool_policy_profile_id == profile_id
                    assert workbench._tool_policy_profile_context.revision == revision
                    await capture(stem + "-matrix")
                    await pilot.press("ctrl+end")
                    await pilot.wait_for_scheduled_animations()
                    await pilot.pause()
                    assert table.cursor_row == table.row_count - 1
                    row = table._get_row_region(table.cursor_row)
                    y = table.content_region.y + row.y - int(table.scroll_y)
                    clip = app.screen._compositor.visible_widgets[table][1]
                    assert clip.y <= y and y + row.height <= clip.bottom
                    await capture(stem + "-last-row")
                    await pilot.press("end")
                    await pilot.wait_for_scheduled_animations()
                    await pilot.pause()
                    state_column = table._get_column_region(1)
                    state_x = (
                        table.content_region.x + state_column.x - int(table.scroll_x)
                    )
                    assert (
                        clip.x <= state_x and state_x + state_column.width <= clip.right
                    )
                    await capture(stem + "-state-column")
                    await pilot.press("home")
                    await pilot.wait_for_scheduled_animations()
                    await pilot.press("ctrl+home", "space")
                    await wait_for(
                        lambda profile_id=profile_id, revision=revision: (
                            store.read_snapshot_strict().payload["profiles"][
                                profile_id
                            ]["tool_pack_lifecycle"]["revision"]
                            > revision
                        ),
                        "Permission saved",
                    )
                    updated = store.read_snapshot_strict().payload["profiles"][
                        profile_id
                    ]
                    assert updated["global_default"] != profile.get(
                        "global_default", "ask"
                    )
                    assert (
                        store.read_snapshot_strict().payload["profiles"]["default"]
                        == default_policy
                    )
                    saved_bytes = store.path.read_bytes()
                    canvas.focus()
                    await pilot.pause()
                    await pilot.press("end", "space")
                    await pilot.wait_for_scheduled_animations()
                    await pilot.pause()
                    assert (
                        app.focused is canvas and store.path.read_bytes() == saved_bytes
                    )
                    await pilot.press("home")
                    await pilot.wait_for_scheduled_animations()
                    preview = canvas.query_one("#mcp-perm-preview", Static)
                    for _ in range(120):
                        pair = app.screen._compositor.visible_widgets.get(preview)
                        if pair and pair[0].intersection(pair[1]) == pair[0]:
                            break
                        await pilot.press("down")
                        await pilot.wait_for_scheduled_animations()
                    region, clip = app.screen._compositor.visible_widgets[preview]
                    assert region.intersection(clip) == region
                    await capture(stem + "-feedback")
                    workbench.set_mode("audit")
                    await pilot.pause()
                    await pilot.press("f4")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "SettingsScreen",
                        "Return Settings",
                    )
                    await category("Tool Profiles", "tool-profiles")
                    settings = app.screen
                    await listing_ready(settings, profile_id)
                    await press(action_selector(settings, profile_id, "edit"))
                    await wait_for(
                        lambda: isinstance(app.screen, MCPScreen), "Restored MCP"
                    )
                    workbench = app.screen.workbench
                    await wait_for(
                        lambda workbench=workbench: (
                            workbench is not None
                            and not workbench.is_loading
                            and not workbench._reloading
                            and workbench._pending_view_state is None
                        ),
                        "Restored permissions loaded",
                    )
                    await expect_focus("#mcp-perm-table")
                    assert workbench._tool_policy_profile_id == profile_id
                    assert (
                        workbench._tool_policy_profile_context.revision
                        == updated["tool_pack_lifecycle"]["revision"]
                    )
                    await capture(stem + "-restored")
                    assert (
                        hashlib.sha256(fixture.read_bytes()).hexdigest()
                        == fixture_digest
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "profile_id": profile_id,
                            "imported_unbound": True,
                            "exact_profile_visible": True,
                            "last_row_visible": True,
                            "permission_edit_saved": True,
                            "default_policy_unchanged": True,
                            "canvas_space_no_write": True,
                            "feedback_visible": True,
                            "restored_revision_visible": True,
                            "revision_before": revision,
                            "revision_after": updated["tool_pack_lifecycle"][
                                "revision"
                            ],
                            "fixture_sha256": fixture_digest,
                        }
                    )
                    record()
                    await pilot.press("f4")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "SettingsScreen",
                        "Next Settings",
                    )
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
                "ToolPackImportOptionsModal",
                "ToolPackImportReviewModal",
                "EnhancedFileOpen",
                "ConfirmationDialog",
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
