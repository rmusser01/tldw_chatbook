"""PR2707 Qodo follow-up: bounded workspace Persona pages and saved identities.

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

            from textual.widgets import Button, Checkbox, OptionList, Select

            from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
            from tldw_chatbook.Widgets.workspace_persona_default import (
                WorkspacePersonaChoice,
                WorkspacePersonaDefaultModal,
            )
            from tldw_chatbook.Workspaces.registry_service import (
                LocalWorkspaceRegistryService,
            )

            await category("Tool Profiles", "tool-profiles")
            await wait_for(lambda: app.tool_pack_service is not None, "Profiles ready")
            registry = app.workspace_registry_service
            personas = app.local_character_persona_service
            ids = [f"writer-{i:03}" for i in range(205)] + ["none", "auto"]
            for persona_id in ids:
                personas.create_persona_profile(
                    {
                        "id": persona_id,
                        "name": f"Writer [{persona_id}]",
                        "system_prompt": "Help.",
                    }
                )
            result["fixture_setup"] = (
                "Real Persona service and registry, 207 saved identities. Tool Profiles initialized first. "
                "Creation and assistant changes enter through Settings; production default modal mounted "
                "directly to isolate its choices (Console entry covered by targeted test). No provider requests."
            )
            folder = root.with_name(root.name + "-project")
            folder.mkdir()

            async def settled():
                await wait_for(
                    lambda: (
                        not getattr(app.screen, "_category_pane_swap_pending", False)
                    ),
                    "Pane settled",
                )
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()

            async def press(selector, key="enter"):
                await focus(selector)
                await pilot.press(key)
                await pilot.pause()

            def paint():
                return "\n".join(
                    strip.text for strip in app.screen._compositor.render_strips()
                )

            async def choose(selector, value):
                control = await focus(selector)
                values = [v for _, v in control._options]
                index = values.index(value)
                await pilot.press("enter", "home", *(["down"] * index), "enter")
                await wait_for(lambda: control.value == value, "Choice selected")
                return control

            def persisted(workspace_id):
                db = WorkspaceDB(
                    root / "data/db/workspaces.db", client_id="verification"
                )
                try:
                    return LocalWorkspaceRegistryService(db).get_workspace(workspace_id)
                finally:
                    db.close()

            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    persona_id = "none" if theme == "textual-dark" else "auto"
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
                        lambda size=size: (
                            app.size.width == size[0] and app.size.height == size[1]
                        ),
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
                    control = await choose("#workspace-default-persona", persona_id)
                    assert len(control._options) <= 103
                    assert f"Writer [{persona_id}]" in paint()
                    assert not app.screen.query_one(
                        "#workspace-default-memory", Select
                    ).disabled
                    await capture(stem + "-create")
                    await edit("#workspace-create-folder-path", str(folder))
                    await press("#workspace-create-folder-add")
                    await wait_for(lambda: bool(app.screen._folders), "Folder added")
                    await settled()
                    assert (
                        app.screen.query_one("#workspace-default-persona", Select).value
                        == persona_id
                    )
                    await press("#workspace-create-confirm")
                    await wait_for(
                        lambda settings=settings: app.screen is settings, "Created"
                    )
                    workspace = next(
                        w for w in registry.list_workspaces() if w.name == stem
                    )
                    saved = persisted(workspace.workspace_id)
                    assert saved.assistant_defaults.assistant_id == persona_id
                    assert not saved.assistant_defaults_explicit_none
                    assert len(personas.list_persona_profiles(limit=1000)) == 207
                    original = saved.assistant_defaults

                    await app.push_screen(
                        WorkspacePersonaDefaultModal(
                            registry, personas, workspace.workspace_id
                        )
                    )
                    await focus("#workspace-default-persona")
                    assert f"Writer [{persona_id}]" in paint()
                    await press("#workspace-persona-next")
                    await press("#workspace-persona-next")
                    control = await focus("#workspace-default-persona")
                    assert len(control._options) <= 102
                    assert control.value == persona_id
                    assert "writer-000" in {v for _, v in control._options}
                    assert app.screen.query_one("#workspace-persona-next", Button).disabled
                    await capture(stem + "-saved")
                    await press("#workspace-default-apply")
                    await wait_for(
                        lambda settings=settings: app.screen is settings,
                        "No-edit apply",
                    )
                    assert (
                        persisted(workspace.workspace_id).assistant_defaults == original
                    )

                    await app.push_screen(
                        WorkspacePersonaDefaultModal(
                            registry, personas, workspace.workspace_id
                        )
                    )
                    await choose("#workspace-default-memory", "read_write")
                    await press("#workspace-default-apply")
                    assert isinstance(app.screen, WorkspacePersonaDefaultModal)
                    assert (
                        persisted(workspace.workspace_id).assistant_defaults == original
                    )
                    assert "Confirm read and write" in paint()
                    await capture(stem + "-confirmation")
                    await press("#workspace-default-memory-confirm", "space")
                    assert app.screen.query_one(
                        "#workspace-default-memory-confirm", Checkbox
                    ).value
                    await press("#workspace-default-apply")
                    await wait_for(
                        lambda settings=settings: app.screen is settings,
                        "Confirmed memory",
                    )
                    confirmed = persisted(workspace.workspace_id).assistant_defaults
                    assert confirmed.assistant_id == persona_id
                    assert confirmed.persona_memory_mode == "read_write"
                    assert (
                        confirmed.tool_policy_profile_id
                        == original.tool_policy_profile_id
                    )

                    await app.push_screen(
                        WorkspacePersonaDefaultModal(
                            registry, personas, workspace.workspace_id
                        )
                    )
                    await choose(
                        "#workspace-default-persona", WorkspacePersonaChoice.NONE
                    )
                    await press("#workspace-default-cancel")
                    await wait_for(
                        lambda settings=settings: app.screen is settings,
                        "Cancel preserved",
                    )
                    assert (
                        persisted(workspace.workspace_id).assistant_defaults
                        == confirmed
                    )

                    await category("Workspaces", "workspaces")
                    await settled()
                    await press("#settings-workspace-row-" + workspace.workspace_id)
                    await settled()
                    picker = await focus("#settings-workspace-persona-picker")
                    assert isinstance(picker, OptionList) and picker.option_count <= 101
                    for _ in range(2):
                        await press("#settings-workspace-persona-next")
                        await settled()
                        picker = await focus("#settings-workspace-persona-picker")
                        assert picker.option_count <= 101
                    assert app.screen.query_one("#settings-workspace-persona-next", Button).disabled
                    target_index = next(
                        i for i in range(picker.option_count)
                        if picker.get_option_at_index(i).persona_id == "writer-000"
                    )
                    await pilot.press("home", *(["down"] * target_index), "enter")
                    await settled()
                    assert (
                        settings._settings_workspace_assistant_pending["persona_id"]
                        == "writer-000"
                    )
                    await focus("#settings-workspace-persona-picker")
                    assert "Writer [writer-000]" in paint()
                    await capture(stem + "-catalog")
                    await press("#settings-workspace-persona-previous")
                    await settled()
                    picker = await focus("#settings-workspace-persona-picker")
                    assert picker.get_option_at_index(picker.highlighted).persona_id == "writer-000"
                    await press("#settings-workspace-memory-toggle")
                    await wait_for(
                        lambda workspace=workspace: (
                            persisted(
                                workspace.workspace_id
                            ).assistant_defaults.assistant_id
                            == "writer-000"
                        ),
                        "Oldest Persona applied",
                    )
                    await settled()
                    await focus("#settings-workspace-persona-picker")
                    await capture(stem + "-applied")
                    final = persisted(workspace.workspace_id).assistant_defaults
                    assert (
                        final.assistant_id == "writer-000"
                        and final.persona_memory_mode == "read_only"
                    )
                    assert (
                        final.tool_policy_profile_id == original.tool_policy_profile_id
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "workspace_id": workspace.workspace_id,
                            "created_persona_id": persona_id,
                            "catalog_count": 207,
                            "recompose_preserves_id": True,
                            "no_edit_apply_preserves_id": True,
                            "read_write_requires_confirmation": True,
                            "cancel_preserves_defaults": True,
                            "final_persona_id": final.assistant_id,
                            "tool_profile_preserved": True,
                            "reopened_database_checked_after_each_apply": True,
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
    result.update(
        app_run_returned=True,
        app_return_code=app.return_code,
        app_exception=type(app._exception).__name__ if app._exception is not None else None,
    )
    if app.return_code != 0 or app._exception is not None:
        result["passed"] = False
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
