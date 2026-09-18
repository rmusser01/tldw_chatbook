"""TASK-32777 native cold workspace Persona provisioning.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
All workspace, Persona and context writes stay in the private profile.
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

            from textual.widgets import Button, Select

            from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
            from tldw_chatbook.Widgets.workspace_persona_default import (
                WorkspacePersonaChoice,
                WorkspacePersonaDefaultModal,
            )
            from tldw_chatbook.Workspaces.registry_service import (
                LocalWorkspaceRegistryService,
            )

            registry = app.workspace_registry_service
            personas = app.local_character_persona_service
            assert app.tool_pack_service is None
            assert app._tool_pack_guard_bootstrap.active_guard is None
            before_personas = len(personas.list_persona_profiles(limit=10000))
            result["fixture_setup"] = (
                "Cold Settings creation without opening Tool Profiles. First cell is cold; "
                "remaining cells verify the initialized path across size/theme changes. "
                "Real app-owned composition, real Persona/permission stores and reopened SQLite. "
                "No provider requests."
            )

            async def settled():
                await wait_for(
                    lambda: (
                        not getattr(app.screen, "_category_pane_swap_pending", False)
                    ),
                    "Pane settled",
                )
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()

            async def press(selector):
                await focus(selector)
                await pilot.press("enter")
                await pilot.pause()

            def persisted(workspace_id):
                db = WorkspaceDB(
                    root / "data/db/workspaces.db", client_id="verification"
                )
                try:
                    return LocalWorkspaceRegistryService(db).get_workspace(workspace_id)
                finally:
                    db.close()

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
                    await category("Workspaces", "workspaces")
                    await settled()
                    settings = app.screen
                    await press("#settings-workspace-create")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "WorkspaceCreateModal",
                        "Create dialog",
                    )
                    await edit("#workspace-create-name", stem)
                    assert (
                        app.screen.query_one("#workspace-default-persona", Select).value
                        is WorkspacePersonaChoice.AUTO
                    )
                    await focus("#workspace-default-persona")
                    await capture(stem + "-automatic")
                    cold = app.tool_pack_service is None
                    await press("#workspace-create-confirm")
                    await wait_for(
                        lambda settings=settings: app.screen is settings, "Created"
                    )
                    workspace = next(
                        w for w in registry.list_workspaces() if w.name == stem
                    )
                    saved = persisted(workspace.workspace_id)
                    assert saved.assistant_defaults is not None
                    assert not saved.assistant_defaults_explicit_none
                    profile_id = "ws-" + saved.workspace_id
                    assert saved.assistant_defaults.tool_policy_profile_id == profile_id
                    persona = personas.get_persona_profile(
                        saved.assistant_defaults.assistant_id
                    )
                    assert persona["name"] == stem + " Agent"
                    assert (
                        profile_id
                        in app.unified_mcp_service.permission_store.list_profiles()
                    )
                    assert (
                        app._tool_pack_guard_bootstrap.active_guard
                        is app.tool_pack_service.binding_guard
                    )
                    assert (
                        len(personas.list_persona_profiles(limit=10000))
                        == before_personas + len(result["cells"]) + 1
                    )
                    await settled()
                    await press("#settings-workspace-row-" + workspace.workspace_id)
                    await settled()
                    await focus("#settings-workspace-persona-picker")
                    await capture(stem + "-saved")
                    await app.push_screen(
                        WorkspacePersonaDefaultModal(
                            registry, personas, workspace.workspace_id
                        )
                    )
                    await focus("#workspace-default-persona")
                    assert (
                        app.screen.query_one("#workspace-default-persona", Select).value
                        == saved.assistant_defaults.assistant_id
                    )
                    await capture(stem + "-reopened")
                    await pilot.press("escape")
                    await wait_for(
                        lambda settings=settings: app.screen is settings,
                        "Details closed",
                    )
                    assert (
                        persisted(workspace.workspace_id).assistant_defaults
                        == saved.assistant_defaults
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "cold": cold,
                            "workspace_id": workspace.workspace_id,
                            "persona_id": persona["id"],
                            "profile_id": profile_id,
                            "real_guard_active": True,
                            "reopened_database_verified": True,
                            "cancel_preserves_defaults": True,
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
                "WorkspacePersonaDefaultModal",
                "WorkspaceCreateModal",
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
