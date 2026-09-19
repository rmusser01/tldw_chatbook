"""TASK-32771 native Change Review with real private registry and services.

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
import threading
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
                "tldw_chatbook/Widgets/Settings_Widgets/workspace_change_review.py",
                "tldw_chatbook/Workspaces/change_review_consent.py",
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

            from tldw_chatbook.Widgets.Settings_Widgets.workspace_change_review import (
                WorkspaceChangeReviewPanel,
            )
            from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService

            registry = app.workspace_registry_service
            service = app.change_review_consent_service
            real_initialize = service._initialize_root
            real_retry = service.retry_failed_roots
            entered = threading.Event()
            release = threading.Event()
            calls = {}

            def initialize(path):
                calls[path] = calls.get(path, 0) + 1
                entered.set()
                if not release.wait(35):
                    raise RuntimeError("audit did not release initialization")
                if calls[path] == 1:
                    raise RuntimeError("synthetic first-attempt initialization failure")
                real_initialize(path)

            service._initialize_root = initialize
            result["fixture_setup"] = (
                "Real registry/consent/initializer/shadow Git, controlled first-attempt failure and release"
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

            async def receipt(needle):
                await settled()
                status = app.screen.query_one(
                    "#settings-workspace-change-review-result", Static
                )
                assert needle in str(status.renderable)
                assert " ".join(str(status.renderable).split()) in " ".join(
                    painted(status).split()
                )
                assert app.focused.is_attached
                painted(app.focused)

            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    workspace_id = "ws-" + stem
                    folder = root.with_name(root.name + "-fixtures") / stem
                    folder.mkdir(parents=True)
                    sample = folder / "sample.txt"
                    sample.write_text("Private Change Review fixture.\n")
                    before = hashlib.sha256(sample.read_bytes()).hexdigest()
                    registry.create_workspace(
                        workspace_id=workspace_id,
                        name=f"Review {len(result['cells']) + 1}",
                    )
                    registry.add_folder_binding(workspace_id, folder)
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
                    # Registry setup is external to the UI; remount the list
                    # before selecting the next private fixture workspace.
                    await category("Overview", "overview")
                    await category("Workspaces", "workspaces")
                    await settled()
                    await press("#settings-workspace-row-" + workspace_id)
                    await settled()
                    panel = app.screen.query_one(WorkspaceChangeReviewPanel)
                    assert not registry.change_review_enabled(workspace_id)
                    registry.set_change_review_enabled(workspace_id, True)
                    await press("#settings-workspace-change-review-toggle")
                    await receipt("changed elsewhere")
                    assert registry.change_review_enabled(workspace_id)
                    await capture(stem + "-conflict")
                    await press("#settings-workspace-change-review-toggle")
                    await receipt("existing history is retained")
                    assert not registry.change_review_enabled(workspace_id)
                    release.clear()
                    entered.clear()
                    await press("#settings-workspace-change-review-toggle")
                    await receipt("enabled")
                    await wait_for(entered.is_set, "Initialization started")
                    await wait_for(
                        lambda: (
                            app.screen.query_one(
                                "#settings-workspace-change-review-preparing"
                            ).display
                        ),
                        "Preparing",
                    )
                    rename = await edit(
                        "#settings-workspace-rename-input", "Draft name"
                    )
                    release.set()
                    await wait_for(
                        lambda: (
                            app.screen.query_one(
                                "#settings-workspace-change-review-failed"
                            ).display
                        ),
                        "Failure published",
                    )
                    assert (
                        app.screen.query_one("#settings-workspace-rename-input")
                        is rename
                    )
                    assert rename.value == "Draft name" and app.focused is rename
                    assert "Draft name" in painted(rename)

                    def fail_retry(_workspace_id):
                        raise RuntimeError("synthetic sensitive retry failure")

                    service.retry_failed_roots = fail_retry
                    await press("#settings-workspace-change-review-retry")
                    await receipt("could not be retried")
                    await capture(stem + "-retry")
                    service.retry_failed_roots = real_retry
                    release.clear()
                    entered.clear()
                    await press("#settings-workspace-change-review-retry")
                    await receipt("Retry scheduled for 1")
                    await wait_for(entered.is_set, "Retry started")
                    assert calls[str(folder)] == 2
                    assert app.focused.id == "settings-workspace-change-review-toggle"
                    assert real_retry(workspace_id) == 0
                    release.set()
                    await wait_for(
                        lambda: (
                            app.screen.query_one(
                                "#settings-workspace-change-review-ready"
                            ).display
                        ),
                        "Real Git preparation complete",
                    )
                    await receipt("Retry scheduled for 1")
                    ready = app.screen.query_one(
                        "#settings-workspace-change-review-ready"
                    )
                    assert "ready for 1" in painted(ready)
                    await capture(stem + "-ready")
                    assert app.screen.query_one(WorkspaceChangeReviewPanel) is panel
                    assert (
                        app.screen.query_one("#settings-workspace-rename-input")
                        is rename
                    )
                    assert rename.value == "Draft name"
                    shadow = ShadowRepoService().repo_for_root(folder)
                    assert shadow.git_dir.is_relative_to(root)

                    def git(*args, shadow=shadow):
                        return subprocess.run(
                            ["git", "--git-dir", str(shadow.git_dir), *args],
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout.strip()

                    head = git("rev-parse", "HEAD")
                    assert (
                        git("show", "HEAD:sample.txt")
                        == "Private Change Review fixture."
                    )
                    await press("#settings-workspace-change-review-toggle")
                    await receipt("existing history is retained")
                    assert not registry.change_review_enabled(workspace_id)
                    assert git("rev-parse", "HEAD") == head
                    assert not (folder / ".git").exists()
                    assert hashlib.sha256(sample.read_bytes()).hexdigest() == before
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "revision_conflict_keeps_consent": True,
                            "failed_retry_is_explicit": True,
                            "preparing_failed_ready_updated_in_place": True,
                            "draft_and_focus_retained": True,
                            "one_retry_scheduled": True,
                            "real_snapshot_head": head,
                            "shadow_git_dir": str(shadow.git_dir),
                            "fixture_unchanged": True,
                            "disable_retains_history": True,
                        }
                    )
                    record()
            service._initialize_root = real_initialize
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
            if "release" in locals():
                release.set()
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result.update(app_run_returned=True)
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
