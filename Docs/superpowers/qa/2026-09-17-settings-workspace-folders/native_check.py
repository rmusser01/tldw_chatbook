"""TASK-32768 native workspace folder feedback with a private registry.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Settings writes stay in the private profile; bound files use an owned sibling.
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
                "tldw_chatbook/css/tldw_cli_modular.tcss",
                "tldw_chatbook/css/features/_settings.tcss",
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

            from textual.widgets import Static

            registry = app.workspace_registry_service
            registry.create_workspace(
                workspace_id="ws-folders", name="Research project"
            )
            registry.create_workspace(workspace_id="ws-other", name="Other project")
            # The real registry correctly refuses folders inside application data.
            folder = (
                root.with_name(root.name + "-project-files") / "research-project[notes]"
            )
            folder.mkdir(parents=True)
            marker = folder / "retained.txt"
            marker.write_text("Removing a binding leaves these files alone.\n")
            result["workspace_setup"] = (
                "Two named workspaces seeded via the real registry"
            )

            async def settled():
                await wait_for(
                    lambda: not app.screen._category_pane_swap_pending,
                    "Workspace pane settled",
                )
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()

            def painted(widget):
                region, clip = app.screen._compositor.visible_widgets[widget]
                assert region.intersection(clip) == region
                return "\n".join(
                    strip.crop(region.x, region.right).text
                    for strip in app.screen._compositor.render_strips()[
                        max(0, region.y) : region.bottom
                    ]
                )

            async def feedback(needle):
                await settled()
                await wait_for(
                    lambda: any(
                        needle in str(w.renderable) for w in app.screen.query(Static)
                    ),
                    needle,
                )
                await settled()
                matches = [
                    w for w in app.screen.query(Static) if needle in str(w.renderable)
                ]
                assert len(matches) == 1
                receipt = matches[0]
                assert " ".join(str(receipt.renderable).split()) in " ".join(
                    painted(receipt).split()
                )
                painted(app.focused)

            async def add_path(value):
                await edit("#settings-workspace-folder-path", value)
                await pilot.press("tab")
                assert app.focused.id == "settings-workspace-folder-add"
                await pilot.press("enter")
                await settled()

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
                    await category("Workspaces", "workspaces")
                    await settled()
                    await press("#settings-workspace-row-ws-folders")
                    await settled()
                    assert not app.screen.query("#settings-save-category")
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    await add_path("/missing-workspace-folder")
                    await feedback("not a directory")
                    assert (
                        app.screen.query_one("#settings-workspace-folder-path").value
                        == "/missing-workspace-folder"
                    )
                    assert registry.list_folder_bindings("ws-folders") == ()
                    await capture(stem + "-invalid")
                    await add_path(str(folder))
                    await feedback("Folder added (read-only).")
                    (binding,) = registry.list_folder_bindings("ws-folders")
                    assert binding.locator == str(folder.resolve())
                    assert binding.metadata["access"] == "ro"
                    assert app.focused.id == "settings-workspace-folder-add"
                    toggle = f"#settings-workspace-folder-toggle-{binding.binding_id}"

                    async def binding_state(access, *, toggle=toggle, binding=binding):
                        await focus(toggle)
                        row = app.screen.query_one(
                            f"#settings-workspace-folder-{binding.binding_id}"
                        )
                        text = "".join(painted(row).split())
                        assert f"[{access}]" in text
                        assert "research-project[notes]" in text

                    await binding_state("ro")
                    await press(toggle)
                    await feedback("Folder access: read-write.")
                    assert (
                        registry.list_folder_bindings("ws-folders")[0].metadata[
                            "access"
                        ]
                        == "rw"
                    )
                    await binding_state("rw")
                    await capture(stem + "-write")
                    await press(toggle)
                    await feedback("Folder access: read-only.")
                    assert (
                        registry.list_folder_bindings("ws-folders")[0].metadata[
                            "access"
                        ]
                        == "ro"
                    )
                    await press("#settings-workspace-row-ws-other")
                    await settled()
                    assert not any(
                        "Folder access:" in str(w.renderable)
                        for w in app.screen.query(Static)
                    )
                    await press("#settings-workspace-row-ws-folders")
                    await settled()
                    await binding_state("ro")
                    await press(
                        f"#settings-workspace-folder-remove-{binding.binding_id}"
                    )
                    await feedback("Folder removed.")
                    assert registry.list_folder_bindings("ws-folders") == ()
                    assert app.focused is app.screen.query_one(
                        "#settings-workspace-folder-add"
                    )
                    assert app.focused.is_attached
                    await pilot.press("tab")
                    await settled()
                    assert app.focused.is_attached
                    assert app.focused.id != "settings-workspace-folder-add"
                    painted(app.focused)
                    await pilot.press("shift+tab")
                    await settled()
                    assert app.focused is app.screen.query_one(
                        "#settings-workspace-folder-add"
                    )
                    assert (
                        marker.read_text()
                        == "Removing a binding leaves these files alone.\n"
                    )
                    assert (
                        registry.get_active_workspace().workspace_id
                        == "workspace-default"
                    )
                    await capture(stem + "-removed")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "invalid_path_visible_and_retained": True,
                            "add_defaults_read_only": True,
                            "write_then_read_only_reaches_registry": True,
                            "feedback_scoped_to_workspace": True,
                            "remove_retains_files_and_focus": True,
                            "active_workspace_unchanged": True,
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
