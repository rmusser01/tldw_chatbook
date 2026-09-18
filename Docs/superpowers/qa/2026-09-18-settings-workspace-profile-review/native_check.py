"""TASK-32774 native imported profile review with real private registry and services.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
All imported profile fixtures and settings writes stay in the private profile.
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
                "tldw_chatbook/Widgets/Settings_Widgets/tool_pack_import_review.py",
                "tldw_chatbook/Tool_Packs/binding.py",
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

            async def press(selector):
                await focus(selector)
                await pilot.press("enter")
                await pilot.pause()

            async def capture(stem):
                await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
                app.save_screenshot(stem + ".svg", path=str(evidence))
                pane = await tmux("capture-pane", "-p", "-t", session)
                (evidence / (stem + ".txt")).write_text(pane.stdout)

            from textual.widgets import Button, Collapsible, Static

            from tldw_chatbook.Tool_Packs.export import write_tool_pack_archive
            from tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review import (
                ToolProfileFirstBindReviewModal,
            )
            from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults

            await category("Tool Profiles", "tool-profiles")
            await wait_for(lambda: app.tool_pack_service is not None, "Profile service")
            service = app.tool_pack_service
            store = app.unified_mcp_service.permission_store
            registry = app.workspace_registry_service
            app.local_character_persona_service.create_persona_profile(
                {"id": "review-helper", "name": "Review [helper]"}
            )
            store.ensure_profile("review-source")
            store.set_global_default("deny", profile_id="review-source")
            store.set_server_default("agent:builtin", "ask", profile_id="review-source")
            exported = await asyncio.to_thread(
                service.capture_export,
                "review-source",
                display_name="Review profile",
                suggested_id="review-import",
            )
            archive = root / "review.tldw-tool-pack"
            with archive.open("wb") as output:
                write_tool_pack_archive(exported.snapshot, output)
            result["fixture_setup"] = (
                "Real production catalog export, serialized archive, inspected unbound import, "
                "receipt store, permission lifecycle and workspace registry; no tool execution"
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

            async def receipt(needle):
                await wait_for(
                    lambda: (
                        needle
                        in str(
                            app.screen.query_one(
                                "#settings-workspace-assistant-result", Static
                            ).renderable
                        )
                    ),
                    needle,
                )
                await settled()
                status = app.screen.query_one(
                    "#settings-workspace-assistant-result", Static
                )
                assert " ".join(str(status.renderable).split()) in " ".join(
                    painted(status).split()
                )
                assert app.focused.id == "settings-workspace-memory-toggle"
                painted(app.focused)

            async def choose_profile(profile_id):
                picker = await focus("#settings-workspace-profile-picker")
                index = next(
                    i
                    for i in range(picker.option_count)
                    if picker.get_option_at_index(i).profile_id == profile_id
                )
                await pilot.press("home", *(["down"] * index), "enter")
                await settled()

            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    profile_id = "import-" + stem
                    workspace_id = "ws-" + stem
                    inspection = await asyncio.to_thread(
                        service.inspect_import,
                        archive,
                        destination_id=profile_id,
                    )
                    installed = await asyncio.to_thread(
                        service.import_unbound, inspection
                    )
                    assert installed.installed.profile_id == profile_id
                    registry.create_workspace(
                        workspace_id=workspace_id, name="Review " + stem
                    )
                    before = WorkspaceAssistantDefaults(assistant_id="review-helper")
                    registry.set_assistant_defaults(workspace_id, before)
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
                    await press("#settings-workspace-row-" + workspace_id)
                    await settled()
                    await choose_profile(profile_id)
                    assert (
                        registry.get_workspace(workspace_id).assistant_defaults
                        == before
                    )

                    async def review(workspace_id=workspace_id, before=before):
                        await press("#settings-workspace-memory-toggle")
                        await wait_for(
                            lambda: isinstance(
                                app.screen, ToolProfileFirstBindReviewModal
                            ),
                            "First-bind review",
                        )
                        await pilot.pause()
                        assert app.focused.id == "tool-profile-bind-confirm"
                        for selector in (
                            "#tool-profile-bind-title",
                            "#tool-profile-bind-confirm",
                            "#tool-profile-bind-cancel",
                        ):
                            control = app.screen.query_one(selector)
                            text = (
                                str(control.label)
                                if isinstance(control, Button)
                                else str(control.renderable)
                            )
                            assert text in painted(control)
                        assert (
                            registry.get_workspace(workspace_id).assistant_defaults
                            == before
                        )
                        return app.screen

                    modal = await review()
                    assert (
                        modal.review.profile_id == profile_id
                        and modal.review.revision == 1
                    )
                    await capture(stem + "-target")
                    # Real Tab routing reaches the scroll body, then the expandable details.
                    for _ in range(12):
                        await pilot.press("tab")
                        await pilot.pause()
                        if app.focused.id == "tool-profile-bind-scroll":
                            break
                    assert app.focused.id == "tool-profile-bind-scroll"
                    await pilot.press("end")
                    await pilot.wait_for_scheduled_animations()
                    for section in modal.query(Collapsible):
                        title = section.query_one("CollapsibleTitle")
                        title.focus()
                        await pilot.pause()
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        painted(title)
                        await pilot.press("enter")
                        await pilot.pause()
                        assert not section.collapsed
                    await focus("#tool-profile-bind-scroll")
                    await pilot.press("end")
                    await pilot.wait_for_scheduled_animations()
                    await pilot.pause()
                    boundary = modal.query(".tool-profile-bind-copy").last()
                    assert "separate from the read_write" in str(boundary.renderable)
                    assert " ".join(str(boundary.renderable).split()) in " ".join(
                        painted(boundary).split()
                    )
                    await capture(stem + "-policy")
                    await pilot.press("escape")
                    await wait_for(
                        lambda settings=settings: app.screen is settings, "Cancelled"
                    )
                    await receipt("bind cancelled")
                    assert (
                        registry.get_workspace(workspace_id).assistant_defaults
                        == before
                    )
                    assert (
                        settings._settings_workspace_assistant_pending["profile_id"]
                        == profile_id
                    )
                    assert store.read_snapshot_strict().payload["profiles"][profile_id][
                        "tool_pack_lifecycle"
                    ]["first_bind_confirmation_required"]
                    await capture(stem + "-cancelled")

                    modal = await review()
                    store.set_global_default(
                        "ask",
                        profile_id=profile_id,
                        expected_revision=modal.review.revision,
                        expected_profile_digest=modal.review.policy_digest,
                    )
                    await press("#tool-profile-bind-confirm")
                    await wait_for(
                        lambda settings=settings: app.screen is settings,
                        "Changed-policy return",
                    )
                    await receipt("confirmation_stale")
                    assert (
                        registry.get_workspace(workspace_id).assistant_defaults
                        == before
                    )
                    await capture(stem + "-stale")
                    modal = await review()
                    assert modal.review.revision == 2
                    await press("#tool-profile-bind-confirm")
                    await wait_for(
                        lambda settings=settings: app.screen is settings,
                        "Confirmed return",
                    )
                    await receipt("Default assistant applied")
                    intended = WorkspaceAssistantDefaults(
                        assistant_id="review-helper", tool_policy_profile_id=profile_id
                    )
                    assert (
                        registry.get_workspace(workspace_id).assistant_defaults
                        == intended
                    )
                    assert not store.read_snapshot_strict().payload["profiles"][
                        profile_id
                    ]["tool_pack_lifecycle"]["first_bind_confirmation_required"]
                    assert settings._settings_workspace_assistant_pending is None
                    status = settings.query_one("#settings-workspace-assistant-status")
                    assert "Review [helper]" in " ".join(painted(status).split())
                    picker = settings.query_one("#settings-workspace-persona-picker")
                    assert "Review [helper]" in " ".join(painted(picker).split())
                    await capture(stem + "-applied")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "profile_id": profile_id,
                            "workspace_id": workspace_id,
                            "unbound_import": True,
                            "keyboard_policy_details": True,
                            "cancel_preserves_defaults_and_draft": True,
                            "changed_revision_refused": True,
                            "retry_binds_exact_defaults": True,
                            "first_bind_marker_cleared": True,
                            "return_focus_visible": True,
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
            if type(app.screen).__name__ == "ToolProfileFirstBindReviewModal":
                await pilot.press("escape")
                await pilot.pause()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result.update(app_run_returned=True)
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
