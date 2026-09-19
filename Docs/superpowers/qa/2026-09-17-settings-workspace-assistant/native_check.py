"""TASK-32769 native assistant defaults with real private registry and services.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
All assistant/profile fixtures and settings writes stay in the private profile.
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

            from textual.widgets import Button, Static

            from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults

            await category("Tool Profiles", "tool-profiles")
            await wait_for(lambda: app.tool_pack_service is not None, "Profile service")
            registry = app.workspace_registry_service
            personas = app.local_character_persona_service
            for pid, name in (
                ("review-helper", "Research helper"),
                ("review-other", "Other helper"),
            ):
                personas.create_persona_profile({"id": pid, "name": name})
            store = app.unified_mcp_service.permission_store
            store.ensure_profile("review-research")
            store.ensure_profile("review-other")
            for i in range(20):
                personas.create_persona_profile(
                    {"id": f"extra-{i}", "name": f"Extra helper {i:02d}"}
                )
                store.ensure_profile(f"extra-{i:02d}")
            registry.create_workspace(
                workspace_id="ws-assistant", name="Research project"
            )
            registry.create_workspace(workspace_id="ws-other", name="Other project")
            result["fixture_setup"] = (
                "Real local Persona service, permission store, registry and Tool Profile guard"
            )

            def saved():
                return registry.get_workspace("ws-assistant").assistant_defaults

            def assert_default(persona, profile, memory="read_only"):
                assert saved() == WorkspaceAssistantDefaults(
                    assistant_id=persona,
                    tool_policy_profile_id=profile,
                    persona_memory_mode=memory,
                )

            async def settled():
                await wait_for(
                    lambda: not app.screen._category_pane_swap_pending, "Pane settled"
                )
                await wait_for(
                    lambda: (
                        not any(
                            w.group == "settings-workspace-assistant-apply"
                            and w.is_running
                            for w in app.workers
                        )
                    ),
                    "Apply settled",
                )
                await wait_for(
                    lambda: not app.screen._category_pane_swap_pending,
                    "Applied pane settled",
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

            async def choose(kind, identity):
                before = saved()
                picker = await focus(f"#settings-workspace-{kind}-picker")
                index = next(
                    i
                    for i in range(picker.option_count)
                    if getattr(picker.get_option_at_index(i), kind + "_id") == identity
                )
                await pilot.press("home", *(["down"] * index), "enter")
                await settled()
                assert saved() == before
                picker = app.screen.query_one(f"#settings-workspace-{kind}-picker")
                option = picker.get_option_at_index(picker.highlighted)
                assert str(option.prompt) in painted(picker)

            async def receipt(needle):
                await settled()
                status = app.screen.query_one(
                    "#settings-workspace-assistant-result", Static
                )
                assert needle in str(status.renderable)
                assert " ".join(str(status.renderable).split()) in " ".join(
                    painted(status).split()
                )
                assert app.focused.is_attached
                painted(app.focused)

            async def apply(memory="read_only"):
                button = await focus("#settings-workspace-memory-toggle")
                assert str(button.label) == f"Apply (memory: {memory})"
                assert str(button.label) in painted(button)
                before = saved()
                await pilot.press("enter")
                await settled()
                if memory == "read_write":
                    assert saved() == before
                    assert str(button.label) == "Confirm read_write?"
                    await receipt("press again to confirm")
                    await press("#settings-workspace-memory-toggle")
                    await settled()
                await receipt("Default assistant applied")

            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    registry.set_assistant_defaults(
                        "ws-assistant",
                        WorkspaceAssistantDefaults(
                            assistant_id="review-helper",
                            tool_policy_profile_id="review-research",
                        ),
                    )
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
                    await press("#settings-workspace-row-ws-assistant")
                    await settled()
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    await choose("persona", "review-other")
                    await focus("#settings-workspace-memory-toggle")
                    await receipt("Persona staged")
                    await capture(stem + "-staged")
                    await apply()
                    assert_default("review-other", "review-research")
                    await choose("profile", "review-other")
                    await apply()
                    assert_default("review-other", "review-other")
                    await press("#settings-workspace-memory-toggle")
                    await receipt("press again to confirm")
                    assert_default("review-other", "review-other")
                    await capture(stem + "-confirm")
                    await press("#settings-workspace-memory-toggle")
                    await receipt("Default assistant applied")
                    assert_default("review-other", "review-other", "read_write")
                    await choose("profile", "review-research")
                    await apply("read_write")
                    assert_default("review-other", "review-research", "read_write")
                    await capture(stem + "-applied")
                    await choose("persona", "review-helper")
                    await apply()
                    assert_default("review-helper", "review-research")
                    await press("#settings-workspace-assistant-clear")
                    await receipt("Default assistant cleared")
                    assert saved() is None
                    await pilot.press("tab")
                    await settled()
                    assert app.focused.is_attached
                    if size[0] == 170:
                        assert app.focused.id == "settings-impact-pane-body"
                        painted(app.focused)
                    else:
                        assert app.focused.id == "nav-home"
                        # The compact header clips button chrome; verify the
                        # visible focused label rather than its outer border.
                        region, clip = app.screen._compositor.visible_widgets[
                            app.focused
                        ]
                        visible = region.intersection(clip)
                        nav_text = " ".join(
                            strip.crop(visible.x, visible.right).text
                            for strip in app.screen._compositor.render_strips()[
                                visible.y : visible.bottom
                            ]
                        )
                        assert "Home" in nav_text
                    assert (
                        registry.get_active_workspace().workspace_id
                        == "workspace-default"
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "persona_keeps_profile": True,
                            "profile_keeps_persona_and_memory": True,
                            "read_write_requires_separate_press": True,
                            "new_persona_read_only": True,
                            "clear_persists": True,
                            "painted_feedback_and_attached_focus": True,
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
