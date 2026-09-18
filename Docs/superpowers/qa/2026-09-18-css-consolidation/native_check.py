"""TASK-32813: native TTY paint and baseline cascade for representative dialogs."""

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
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    runpy.run_path(str(here.parent / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
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
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(here))
    import migration_parity
    from textual_image._terminal import probe_terminal

    from Tests.UI.test_css_consolidation_recovery_dialogs import make_dialog
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Screens.artifact_share_dialog import ArtifactShareDialog
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Widgets.Console.console_appearance_picker_modal import (
        ConsoleAppearancePickerModal,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementModal,
    )

    claim_process_exit()
    probe_terminal()
    app = TldwCli()
    files = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=repo, text=True
    ).splitlines()
    files += [
        str(Path(__file__).relative_to(repo)),
        str((here / "migration_parity.py").relative_to(repo)),
        "Tests/UI/test_css_consolidation_recovery_dialogs.py",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "source_hashes": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in sorted(set(files))
            if name.endswith((".py", ".tcss"))
        },
        "fixture_setup": "Real TldwCli on native terminal; explicit read-only dialog inputs. No publish, save, import or recovery approval is activated.",
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2))

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    async def journey(pilot):
        async def settle():
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            async with asyncio.timeout(35):
                while not predicate():
                    await pilot.pause(0.03)
            await settle()

        async def capture(stem):
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

        cases = [
            ("artifact-share", lambda: ArtifactShareDialog([]), "#share-cancel"),
            (
                "buddy-management",
                lambda: BuddyManagementModal(
                    buddies=(("Review Buddy", "review-buddy"),)
                ),
                "#buddy-cancel",
            ),
            (
                "appearance",
                lambda: ConsoleAppearancePickerModal(
                    conversation_id="css-audit", conversation_title="CSS review"
                ),
                "#console-appearance-picker-cancel",
            ),
            (
                "markdown",
                lambda: make_dialog("markdown", root),
                "#console-save-markdown-cancel",
            ),
            (
                "notes-recovery",
                lambda: make_dialog("notes", root),
                "#notes-recovery-close",
            ),
            (
                "skills-recovery",
                lambda: make_dialog("skills", root),
                "#skills-recovery-cancel",
            ),
            (
                "roleplay-recovery",
                lambda: make_dialog("roleplay", root),
                "#roleplay-draft-recovery-stay",
            ),
        ]
        modal = None
        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert (
                sys.stdout.isatty()
                and sys.stderr.isatty()
                and app.console.file.isatty()
            )
            result.update(driver="LinuxDriver", tty_streams=True, lock_acquired=True)
            await wait_for(lambda: getattr(app, "_ui_ready", False), "startup")
            for theme in ("textual-dark", "textual-light"):
                for width, height in ((80, 24), (170, 48)):
                    app.theme = theme
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(width),
                        "-y",
                        str(height),
                    )
                    await wait_for(
                        lambda width=width, height=height: (
                            (app.size.width, app.size.height) == (width, height)
                        ),
                        "terminal resized",
                    )
                    for name, factory, selector in cases:
                        outcomes = []
                        modal = factory()
                        app.push_screen(modal, outcomes.append)
                        await settle()
                        cancel_control = modal.query_one(selector)
                        control = (
                            modal.query_one("#console-appearance-picker-filter")
                            if name == "appearance"
                            else cancel_control
                        )
                        control.scroll_visible(animate=False)
                        control.focus()
                        await settle()
                        assert app.focused is control
                        region, clip = modal._compositor.visible_widgets[control]
                        assert region.intersection(clip) == region
                        migration_parity.compare(app)
                        assert not migration_parity._mismatches, (
                            migration_parity._mismatches[:3]
                        )
                        cancel_geometry = modal._compositor.visible_widgets.get(
                            cancel_control
                        )
                        cancel_visible = bool(
                            cancel_geometry
                            and cancel_geometry[0].intersection(cancel_geometry[1])
                            == cancel_geometry[0]
                        )
                        if not cancel_visible:
                            assert name == "appearance" and (width, height) == (80, 24)
                        await wait_for(
                            lambda: not app.screen.query("Toast"),
                            "notifications cleared",
                        )
                        stem = f"{theme}-{width}x{height}-{name}"
                        await capture(stem)
                        control.disabled = True
                        await settle()
                        migration_parity.compare(app)
                        assert not migration_parity._mismatches, (
                            migration_parity._mismatches[:3]
                        )
                        control.disabled = False
                        await settle()
                        cancel_control.focus()
                        await settle()
                        assert app.focused is cancel_control
                        await pilot.press("enter")
                        await settle()
                        assert modal not in app.screen_stack and outcomes == [None], (
                            name,
                            outcomes,
                        )
                        result["cells"].append(
                            {
                                "theme": theme,
                                "size": [width, height],
                                "dialog": name,
                                "focused_control": "#" + control.id,
                                "cancel_control_fully_visible": cancel_visible,
                                "existing_visibility_limit": (
                                    "Appearance Cancel row is outside 80x24 viewport with original computed styles"
                                    if not cancel_visible
                                    else None
                                ),
                                "focus_control_fully_visible": True,
                                "focused_and_disabled_computed_rules_match_baseline": True,
                                "cancelled_without_commit": True,
                            }
                        )
                        record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain native failure and shutdown evidence
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            result["parity_classes"] = {
                name: len(states) for name, states in migration_parity._records.items()
            }
            result["parity_mismatches"] = migration_parity._mismatches
            record()
            if modal is not None and modal in app.screen_stack:
                await modal.request_safe_cancel(source="qa-failure-cleanup")
                await settle()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    result["app_return_code"] = app.return_code
    result["app_exception"] = (
        type(app._exception).__name__ if app._exception is not None else None
    )
    if app.return_code != 0 or app._exception is not None:
        result["passed"] = False
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
