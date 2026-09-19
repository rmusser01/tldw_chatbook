"""TASK-32773 native workspace lifecycle with real private registry and services.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
All workspace fixtures and settings writes stay in the private profile.
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
                "tldw_chatbook/UI/Screens/settings_screen.py",
                "tldw_chatbook/Widgets/workspace_create_modal.py",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
                "tldw_chatbook/css/features/_settings.tcss",
                "tldw_chatbook/css/core/_variables.tcss",
                "tldw_chatbook/css/components/_dialogs.tcss",
                "tldw_chatbook/css/widget_defaults_scoped.tcss",
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

            async def press(selector):
                await focus(selector)
                await pilot.press("enter")
                await pilot.pause()

            async def capture(stem):
                await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
                app.save_screenshot(stem + ".svg", path=str(evidence))
                pane = await tmux("capture-pane", "-p", "-t", session)
                (evidence / (stem + ".txt")).write_text(pane.stdout)

            from textual.widgets import Button, Input, Select, Static

            registry = app.workspace_registry_service
            result["fixture_setup"] = (
                "Real registry; sibling folders removed and restored to exercise binding recovery"
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

            def painted(widget):
                region, clip = app.screen._compositor.visible_widgets[widget]
                assert region.intersection(clip) == region, (widget.id, region, clip)
                return "\n".join(
                    strip.crop(region.x, region.right).text
                    for strip in app.screen._compositor.render_strips()[
                        max(0, region.y) : region.bottom
                    ]
                )

            async def receipt(selector, needle):
                await wait_for(
                    lambda: (
                        needle in str(app.screen.query_one(selector, Static).renderable)
                    ),
                    needle,
                )
                await settled()
                status = app.screen.query_one(selector, Static)
                assert " ".join(str(status.renderable).split()) in " ".join(
                    painted(status).split()
                )
                assert app.focused.is_attached
                painted(app.focused)

            async def none_persona():
                await focus("#workspace-default-persona")
                await pilot.press("enter", "home", "down", "enter")
                await wait_for(
                    lambda: (
                        app.screen.query_one("#workspace-default-persona", Select).value
                        == "none"
                    ),
                    "None Persona",
                )

            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    name = f"Review {len(result['cells']) + 1}"
                    literal_name = f"[red]{name}[/red]"
                    folder = root.with_name(root.name + "-fixtures") / stem
                    folder.mkdir(parents=True)
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
                    await category("Overview", "overview")
                    await category("Workspaces", "workspaces")
                    await settled()
                    settings = app.screen
                    await press("#settings-workspace-create")
                    await wait_for(
                        lambda settings=settings: app.screen is not settings,
                        "Create dialog",
                    )
                    await edit("#workspace-create-name", "Default")
                    await pilot.press("enter")
                    await receipt("#workspace-create-error", "already exists")
                    await capture(stem + "-create-error")
                    await pilot.press("escape")
                    await wait_for(
                        lambda settings=settings: app.screen is settings,
                        "Create cancelled",
                    )
                    before_ids = {
                        r.workspace_id
                        for r in registry.list_workspaces(include_archived=True)
                    }
                    await press("#settings-workspace-create")
                    await edit("#workspace-create-name", name)
                    await none_persona()
                    active = await focus("#workspace-create-make-active")
                    if active.value:
                        await pilot.press("space")
                    assert not active.value
                    await edit("#workspace-create-folder-path", str(folder))
                    await press("#workspace-create-folder-add")
                    await wait_for(
                        lambda: app.screen.query("#workspace-create-folder-remove-0"),
                        "Folder added",
                    )
                    modal = app.screen
                    folder.rmdir()
                    await press("#workspace-create-confirm")
                    await wait_for(
                        lambda: (
                            str(
                                app.screen.query_one(
                                    "#workspace-create-confirm", Button
                                ).label
                            )
                            == "Retry folders"
                        ),
                        "Partial create",
                    )
                    await receipt("#workspace-create-error", "Created " + name)
                    assert modal.query_one("#workspace-create-name", Input).disabled
                    assert modal.query_one("#workspace-create-make-active").disabled
                    workspace = next(
                        r for r in registry.list_workspaces() if r.name == name
                    )
                    workspace_id = workspace.workspace_id
                    assert {
                        r.workspace_id
                        for r in registry.list_workspaces(include_archived=True)
                    } - before_ids == {workspace_id}
                    await capture(stem + "-partial-create")
                    folder.mkdir()
                    await press("#workspace-create-confirm")
                    await wait_for(
                        lambda settings=settings: app.screen is settings,
                        "Retry completed",
                    )
                    await settled()
                    assert len(registry.list_folder_bindings(workspace_id)) == 1
                    assert (
                        len([r for r in registry.list_workspaces() if r.name == name])
                        == 1
                    )
                    await press("#settings-workspace-row-" + workspace_id)
                    await settled()
                    await edit("#settings-workspace-rename-input", "Default")
                    await press("#settings-workspace-rename-apply")
                    await receipt(
                        "#settings-workspace-lifecycle-result", "already exists"
                    )
                    assert (
                        settings.query_one(
                            "#settings-workspace-rename-input", Input
                        ).value
                        == "Default"
                    )
                    await edit("#settings-workspace-rename-input", literal_name)
                    await press("#settings-workspace-rename-apply")
                    await receipt(
                        "#settings-workspace-lifecycle-result", "Workspace renamed"
                    )
                    assert registry.get_workspace(workspace_id).name == literal_name
                    await press("#settings-workspace-set-active")
                    await receipt("#settings-workspace-lifecycle-result", "now active")
                    assert app.focused.id == "settings-workspace-archive"
                    assert registry.get_active_workspace().workspace_id == workspace_id
                    await press("#settings-workspace-archive")
                    await wait_for(
                        lambda settings=settings: app.screen is not settings,
                        "Archive confirmation",
                    )
                    assert literal_name in painted(
                        app.screen.query_one(".dialog-message")
                    )
                    await capture(stem + "-archive-confirmation")
                    await pilot.press("escape")
                    await wait_for(
                        lambda settings=settings: app.screen is settings,
                        "Archive cancelled",
                    )
                    assert not registry.get_workspace(workspace_id).archived
                    painted(app.focused)
                    await press("#settings-workspace-archive")
                    await wait_for(
                        lambda settings=settings: app.screen is not settings,
                        "Archive confirmation again",
                    )
                    await press("#confirm-button")
                    await wait_for(
                        lambda settings=settings: (
                            app.screen is settings
                            and bool(settings.query("#settings-workspace-archive-undo"))
                        ),
                        "Archive completed",
                    )
                    await settled()
                    assert registry.get_workspace(workspace_id).archived
                    assert registry.get_active_workspace().workspace_id != workspace_id
                    assert literal_name in str(
                        settings.query_one(
                            "#settings-workspaces-result", Static
                        ).renderable
                    )
                    await press("#settings-workspace-archive-undo")
                    await wait_for(
                        lambda workspace_id=workspace_id: (
                            not registry.get_workspace(workspace_id).archived
                        ),
                        "Undo completed",
                    )
                    await receipt(
                        "#settings-workspace-lifecycle-result",
                        "Active workspace unchanged",
                    )
                    assert app.focused.id == "settings-workspace-set-active"
                    active_id = registry.get_active_workspace().workspace_id
                    assert active_id != workspace_id
                    await press("#settings-workspace-archive")
                    await wait_for(
                        lambda settings=settings: app.screen is not settings,
                        "Archive before conflict",
                    )
                    await press("#confirm-button")
                    await wait_for(
                        lambda settings=settings, workspace_id=workspace_id: (
                            app.screen is settings
                            and registry.get_workspace(workspace_id).archived
                        ),
                        "Second archive complete",
                    )
                    registry.create_workspace(
                        workspace_id="collision-" + stem, name=literal_name
                    )
                    await settled()
                    await press("#settings-workspace-archive-view")
                    await settled()
                    await press("#settings-workspace-unarchive")
                    await receipt(
                        "#settings-workspace-lifecycle-result", "already exists"
                    )
                    await capture(stem + "-restore-conflict")
                    await edit("#settings-workspace-restore-name", "Restored " + name)
                    await press("#settings-workspace-unarchive")
                    await wait_for(
                        lambda workspace_id=workspace_id: (
                            not registry.get_workspace(workspace_id).archived
                        ),
                        "Restore as completed",
                    )
                    await receipt(
                        "#settings-workspace-lifecycle-result", "choose Set active"
                    )
                    assert app.focused.id == "settings-workspace-set-active"
                    assert (
                        registry.get_workspace(workspace_id).name == "Restored " + name
                    )
                    assert registry.get_active_workspace().workspace_id == active_id
                    await capture(stem + "-restored")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "workspace_id": workspace_id,
                            "create_cancel_unchanged": True,
                            "partial_retry_one_workspace_one_binding": True,
                            "rename_draft_and_feedback_visible": True,
                            "activation_focus_visible": True,
                            "archive_literal_confirmation_and_cancel": True,
                            "archive_and_undo": True,
                            "restore_as_conflict_recovery": True,
                            "restore_preserves_active_workspace": True,
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
