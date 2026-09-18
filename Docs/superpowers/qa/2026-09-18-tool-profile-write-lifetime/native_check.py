"""TASK-32787 native Tool Profile recreation and owned shutdown.

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
import threading
import time
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
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit

    claim_process_exit()
    shutdown_watcher = [None]
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
                "tldw_chatbook/Tool_Packs/operations.py",
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
        block_release = None
        block_task = None
        shutdown_requested = False

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
                assert not control.disabled
                control.focus()
                await pilot.pause()
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
                if isinstance(control, Button):
                    await wait_for(
                        lambda: not control.has_class("-active"), "Button ready"
                    )
                assert app.focused is control
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

            from textual.widgets import Button, Static

            from tldw_chatbook.Tool_Packs.publication import CapturedToolPackDestination
            from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
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
                "One private source profile and real service-exported archive. Import and removal use the real UI/service. A fixture thread holds the existing lifecycle mutation lock to delay real removal while Settings is destroyed and recreated; a fifth confirmed write is released only after normal shutdown closes admission. No service replacements or provider requests."
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

            cells = [
                (theme, size, False)
                for theme in ("textual-dark", "textual-light")
                for size in ((80, 24), (170, 48))
            ]
            cells.append(("textual-dark", (80, 24), True))
            for theme, size, shutdown_case in cells:
                stem = f"{theme}-{size[0]}x{size[1]}"
                profile_id = (
                    "native-shutdown"
                    if shutdown_case
                    else f"native-{theme.removeprefix('textual-')}-{size[0]}"
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
                imported_bytes = store.path.read_bytes()
                prior = store.read_snapshot_strict().payload["profiles"][profile_id]
                remove = action_selector(settings, profile_id, "remove")
                await press(remove)
                await wait_for(
                    lambda: isinstance(app.screen, ConfirmationDialog),
                    "Removal review",
                )
                await focus("#confirm-button")
                if not shutdown_case:
                    await capture(stem + "-review")
                locked, block_release = threading.Event(), threading.Event()

                def hold_mutation(release=block_release, entered=locked):
                    with service.lifecycle.mutation():
                        entered.set()
                        assert release.wait(45), "Owned lifecycle hold timed out"

                block_task = asyncio.create_task(asyncio.to_thread(hold_mutation))
                await wait_for(locked.is_set, "Lifecycle lock held")
                await press("#confirm-button")
                await wait_for(
                    lambda settings=settings: (
                        app._tool_profile_operations.pending("remove") is not None
                    ),
                    "Removal admitted",
                )
                owner = app._tool_profile_operations
                pending = owner.pending("remove")
                if shutdown_case:
                    await capture("shutdown-pending")
                    result["shutdown"] = {
                        "profile_id": profile_id,
                        "prior_revision": prior["tool_pack_lifecycle"]["revision"],
                        "store_relative_path": str(store.path.relative_to(root)),
                    }

                    def release_after_close(release=block_release, owner=owner):
                        try:
                            deadline = time.monotonic() + 15
                            while not getattr(
                                app, "_tool_profile_operations_closed", False
                            ):
                                if time.monotonic() > deadline:
                                    raise AssertionError(
                                        "Shutdown never closed admission"
                                    )
                                time.sleep(0.02)
                            result["shutdown"].update(
                                observed_closed_admission=True,
                                owned_write_pending=owner.pending("remove") is not None,
                                watchdog_armed=any(
                                    t.name == "tldw-exit-watchdog"
                                    for t in threading.enumerate()
                                ),
                            )
                        except Exception:  # noqa: BLE001 - retain private native failure evidence
                            result["shutdown"]["error"] = traceback.format_exc()
                        finally:
                            release.set()

                    shutdown_watcher[0] = threading.Thread(
                        target=release_after_close,
                        name="native-shutdown-release",
                        daemon=True,
                    )
                    shutdown_watcher[0].start()
                    shutdown_requested = True
                    return
                first = next(
                    w for w in app.workers if w.group == "settings-tool-pack-remove"
                )
                await press(remove)
                assert app.screen is settings and owner.pending("remove") is pending
                old_settings = settings
                await pilot.press("f3")
                await wait_for(
                    lambda old_settings=old_settings: not old_settings.is_attached,
                    "Settings destroyed",
                )
                assert first.is_cancelled and owner.pending("remove") is pending
                await pilot.press("f4")
                await wait_for(
                    lambda old_settings=old_settings: (
                        type(app.screen).__name__ == "SettingsScreen"
                        and app.screen is not old_settings
                    ),
                    "New Settings",
                )
                settings = app.screen
                await category("Tool Profiles", "tool-profiles")
                await wait_for(
                    lambda settings=settings: (
                        "in progress" in settings._tool_profiles_result
                    ),
                    "Replayed pending write",
                )
                assert settings._tool_profiles_listing.unavailable_category == "loading"
                newer_focus = app.focused
                assert (
                    settings._tool_profiles_result
                    == "Removal in progress. Wait for it to finish."
                )
                assert store.path.read_bytes() == imported_bytes
                receipt = next(
                    w
                    for w in settings.query(Static)
                    if w.display and str(w.renderable) == settings._tool_profiles_result
                )
                region, clip = app.screen._compositor.visible_widgets[receipt]
                assert region.intersection(clip) == region
                await capture(stem + "-pending")
                block_release.set()
                await block_task
                block_task = None
                await wait_for(
                    lambda settings=settings, profile_id=profile_id: (
                        settings._tool_profiles_result.startswith(
                            "Removed " + profile_id
                        )
                        and profile_id
                        not in settings.query_one(ToolProfilesPanel).profile_ids
                    ),
                    "Removed profile",
                )
                assert app.focused is newer_focus
                await focus("#tool-profiles-import")
                await expect_focus("#tool-profiles-import")
                assert owner.pending("remove") is None
                receipt = settings.query_one("#tool-profiles-result", Static)
                region, clip = app.screen._compositor.visible_widgets[receipt]
                assert region.intersection(clip) == region
                tombstone = store.read_snapshot_strict().payload["profiles"][profile_id]
                assert tombstone["profile_kind"] == "tool_pack_tombstone"
                assert (
                    tombstone["tool_pack_lifecycle"]["revision"]
                    == prior["tool_pack_lifecycle"]["revision"] + 1
                )
                await capture(stem + "-removed")
                completed_settings = settings
                await pilot.press("f3")
                await wait_for(
                    lambda completed_settings=completed_settings: (
                        not completed_settings.is_attached
                    ),
                    "Completed Settings destroyed",
                )
                await pilot.press("f4")
                await wait_for(
                    lambda completed_settings=completed_settings: (
                        type(app.screen).__name__ == "SettingsScreen"
                        and app.screen is not completed_settings
                    ),
                    "Completed Settings recreated",
                )
                settings = app.screen
                await category("Tool Profiles", "tool-profiles")
                await wait_for(
                    lambda settings=settings, profile_id=profile_id: (
                        settings._tool_profiles_result.startswith(
                            "Removed " + profile_id
                        )
                    ),
                    "Replayed completion",
                )
                await focus("#tool-profiles-import")
                await capture(stem + "-restored")
                assert (
                    hashlib.sha256(fixture.read_bytes()).hexdigest() == fixture_digest
                )
                result["cells"].append(
                    {
                        "theme": theme,
                        "size": size,
                        "profile_id": profile_id,
                        "imported_unbound": True,
                        "pending_policy_bytes_unchanged": True,
                        "screen_destroyed": True,
                        "observer_cancelled_promptly": True,
                        "pre_departure_repeat_kept_owned_write": True,
                        "recreated_listing_waited_for_real_lock": True,
                        "newer_category_focus_retained": True,
                        "completion_replayed_after_recreation": True,
                        "visible_pending_receipt": True,
                        "removed_tombstone": True,
                        "single_revision_increment": True,
                        "continuation": "tool-profiles-import",
                        "fixture_sha256": fixture_digest,
                    }
                )
                record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain private native failure evidence  # noqa: BLE001 - preserve failed-run diagnostics
            result.update(
                passed=False,
                error=traceback.format_exc(),
                focused_id=getattr(app.focused, "id", None),
                focused_region=str(getattr(app.focused, "region", None)),
            )
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            if not shutdown_requested:
                if block_release is not None:
                    block_release.set()
                if block_task is not None:
                    await block_task
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
    if shutdown_watcher[0] is not None:
        shutdown_watcher[0].join(timeout=1)
        try:
            shutdown = result["shutdown"]
            assert not shutdown_watcher[0].is_alive()
            assert not shutdown.get("error")
            assert (
                shutdown["observed_closed_admission"]
                and shutdown["owned_write_pending"]
                and shutdown["watchdog_armed"]
            )
            payload = json.loads((root / shutdown["store_relative_path"]).read_text())
            tombstone = payload["profiles"][shutdown["profile_id"]]
            assert tombstone["profile_kind"] == "tool_pack_tombstone"
            assert (
                tombstone["tool_pack_lifecycle"]["revision"]
                == shutdown["prior_revision"] + 1
            )
            assert app._tool_profile_operations.pending("remove") is None
            shutdown.update(
                removed_tombstone=True,
                single_revision_increment=True,
                owned_write_drained=True,
            )
            result["passed"] = len(result["cells"]) == 4
        except Exception:  # noqa: BLE001 - retain private native failure evidence
            result.update(passed=False, error=traceback.format_exc())
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
