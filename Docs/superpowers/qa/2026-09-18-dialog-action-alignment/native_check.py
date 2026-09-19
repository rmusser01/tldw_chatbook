"""Native palette/gallery action layout and direct Roleplay recovery continuity."""

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
    from textual.command import CommandList
    from textual.widgets import Button
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
        RoleplayDraftRecoveryDialog,
    )
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit

    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    probe_terminal()
    app = TldwCli()
    sources = [
        "tldw_chatbook/UI/Navigation/character_conversation_navigation.py",
        "tldw_chatbook/css/components/_dialogs.tcss",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/core/_variables.tcss",
        "tldw_chatbook/css/widget_defaults_scoped.tcss",
        "tldw_chatbook/css/widget_defaults_self.tcss",
        str(Path(__file__).relative_to(repo)),
        "Tests/UI/test_roleplay_recovery_layout.py",
        "Tests/UI/test_dialog_action_alignment.py",
        "tldw_chatbook/Widgets/pattern_gallery.py",
        "tldw_chatbook/css/components/_buttons.tcss",
    ]
    assert all((repo / p).is_file() for p in sources), sources
    result = {
        "pid": os.getpid(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sources
        },
        "cells": [],
        "fixture_scope": "Real TldwCli: Ctrl+P, typed Pattern Gallery search, Enter to the real gallery, direct focus of its Cancel then Tab to Delete, Escape to host. Direct recovery modal with all four failed domains checks continuity; callbacks do not exercise save workers or normal partial-save entry.",
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

    def visible(modal, widget):
        region, clip = modal._compositor.visible_widgets[widget]
        assert region.intersection(clip) == region, (widget.id, region, clip)

    def painted(modal, widget):
        r = widget.content_region
        return "\n".join(
            s.crop(r.x, r.right).text
            for s in modal._compositor.render_strips()[r.y : r.bottom]
        )

    async def journey(pilot):
        modal = None

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
            domains = (
                "character form",
                "character visuals",
                "Persona visuals",
                "attachments",
            )
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
                        lambda width=width, height=height: app.size == (width, height),
                        "resize",
                    )
                    host = app.screen
                    await pilot.press("ctrl+p")
                    await wait_for(
                        lambda: app.screen.id == "--command-palette", "palette open"
                    )
                    palette = app.screen
                    await pilot.press(*"Pattern Gallery")
                    commands = palette.query_one(CommandList)
                    await wait_for(
                        lambda commands=commands: (
                            commands.option_count == 1
                            and "Design System: Pattern Gallery"
                            in str(commands.get_option_at_index(0).prompt)
                        ),
                        "gallery search result",
                    )
                    await pilot.press("enter")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "PatternGalleryScreen",
                        "gallery open",
                    )
                    gallery = app.screen
                    await wait_for(
                        lambda gallery=gallery: not gallery.query("Toast"),
                        "gallery notifications cleared",
                    )
                    row = gallery.query_one(".pg-dialog-frame .dialog-buttons")
                    cancel, delete = list(row.query(Button))
                    cancel.focus()
                    await settle()
                    await pilot.press("tab")
                    await settle()
                    assert app.focused is delete
                    assert (
                        delete.region.right + delete.styles.margin.right
                        == row.content_region.right
                    )
                    for button in (cancel, delete):
                        visible(gallery, button)
                        assert str(button.label) in painted(gallery, button)
                    gallery_regions = {
                        "row": list(row.region),
                        "cancel": list(cancel.region),
                        "delete": list(delete.region),
                    }
                    await capture(f"{theme}-{width}x{height}-gallery")
                    await pilot.press("escape")
                    await settle()
                    assert app.screen is host
                    outcomes = []
                    modal = RoleplayDraftRecoveryDialog(domains)
                    app.push_screen(modal, outcomes.append)
                    await settle()
                    await wait_for(
                        lambda modal=modal: not modal.query("Toast"),
                        "notifications cleared",
                    )
                    frame = modal.query_one("#roleplay-draft-recovery-dialog")
                    assert 0 < frame.region.width < width
                    assert 0 < frame.region.height < height
                    assert abs(frame.region.x * 2 + frame.region.width - width) <= 1
                    assert abs(frame.region.y * 2 + frame.region.height - height) <= 1
                    frame_region = list(frame.region)
                    failed = modal.query_one("#roleplay-draft-recovery-domains")
                    visible(modal, failed)
                    assert "Failed: " + ", ".join(domains) in " ".join(
                        painted(modal, failed).split()
                    )
                    title = modal.query_one("Static")
                    visible(modal, title)
                    assert "Some Roleplay drafts could not be saved" in " ".join(
                        painted(modal, title).split()
                    )
                    retry = modal.query_one("#roleplay-draft-retry", Button)
                    stay = modal.query_one("#roleplay-draft-recovery-stay", Button)
                    assert app.focused is retry
                    assert (
                        stay.region.right + stay.styles.margin.right
                        == frame.content_region.right
                    )
                    for button in (retry, stay):
                        visible(modal, button)
                        assert str(button.label) in painted(modal, button)
                    await pilot.press("tab")
                    await settle()
                    assert app.focused is stay
                    visible(modal, stay)
                    stem = f"{theme}-{width}x{height}-roleplay"
                    await capture(stem)
                    await pilot.press("enter")
                    await settle()
                    assert modal not in app.screen_stack and outcomes == [None]
                    modal = RoleplayDraftRecoveryDialog(domains)
                    app.push_screen(modal, outcomes.append)
                    await settle()
                    assert await pilot.click("#roleplay-draft-retry")
                    await settle()
                    assert modal not in app.screen_stack and outcomes == [None, "retry"]
                    modal = RoleplayDraftRecoveryDialog(domains)
                    app.push_screen(modal, outcomes.append)
                    await settle()
                    await pilot.press("escape")
                    await settle()
                    assert modal not in app.screen_stack and outcomes == [
                        None,
                        "retry",
                        None,
                    ]
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "frame": frame_region,
                            "gallery_regions": gallery_regions,
                            "gallery_palette_entry_focus_and_escape": True,
                            "complete_failure_copy_painted": True,
                            "actions_fully_visible": True,
                            "keyboard_stay_pointer_retry_escape_results": outcomes,
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - record unexpected native journey failures
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            if modal is not None and modal in app.screen_stack:
                modal.action_stay()
                await settle()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result.update(
        app_run_returned=True,
        app_return_code=app.return_code,
        app_exception=type(app._exception).__name__
        if app._exception is not None
        else None,
    )
    if app.return_code != 0 or app._exception is not None:
        result["passed"] = False
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
