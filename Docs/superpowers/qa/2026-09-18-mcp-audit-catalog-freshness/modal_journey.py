"""Native modal geometry and paint check after private-profile validation."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import traceback
from argparse import Namespace
from pathlib import Path

from loguru import logger
from textual.containers import Vertical
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Utils.app_shutdown import claim_process_exit
from tldw_chatbook.Utils.file_extraction import ExtractedFile
from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol
from tldw_chatbook.Widgets.file_extraction_dialog import FileExtractionDialog
from tldw_chatbook.Widgets.Persona_Widgets.conversation_attach_picker import (
    ConversationAttachPicker,
)
from tldw_chatbook.Widgets.Persona_Widgets.dictionary_attach_picker import (
    DictionaryAttachPicker,
)


def run(args: Namespace, attempts: list[str]) -> int:
    """Capture the three wide-modal selector consumers without saving or attaching.

    Args:
        args: Validated private root, tmux socket/session and executable path.
        attempts: Network attempts collected by the launcher's socket guard.

    Returns:
        Zero for success, or one for a journey or application failure.
    """
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    root, tmux_socket, session = args.root, args.tmux_socket, args.session
    args_tmux_path = args.tmux_path
    logger.remove()
    logger.add(root / "native.log", level="INFO")
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    warm_up_image_protocol()
    app = TldwCli()
    paths = [
        "tldw_chatbook/UI/MCP_Modules/" + p + ".py"
        for p in ("mcp_audit_mode", "mcp_workbench", "mcp_inspector")
    ]
    paths += [
        "tldw_chatbook/css/" + p
        for p in (
            "tldw_cli_modular.tcss",
            "widget_defaults_scoped.tcss",
            "widget_defaults_self.tcss",
            "core/_variables.tcss",
            "components/_agentic_terminal.tcss",
        )
    ]
    paths += [
        "tldw_chatbook/MCP/execution_log.py",
        "tldw_chatbook/MCP/local_control_service.py",
        "tldw_chatbook/MCP/unified_control_plane_service.py",
        "tldw_chatbook/UI/MCP_Modules/mcp_inspector.py",
    ]
    paths += [
        "tldw_chatbook/Widgets/file_extraction_dialog.py",
        "tldw_chatbook/Widgets/Persona_Widgets/dictionary_attach_picker.py",
        "tldw_chatbook/Widgets/Persona_Widgets/conversation_attach_picker.py",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths
        },
        "journey_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(
            (here / "modal_check.py").read_bytes()
        ).hexdigest(),
        "fixture_scope": "Three real modals opened directly with synthetic in-memory contents. Cancel only; no file saves, attachments, tool execution or external servers.",
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            [args_tmux_path, "-L", tmux_socket, *args],
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
                while True:
                    try:
                        if predicate():
                            break
                    except (NoMatches, QueryError):
                        pass
                    await pilot.pause(0.03)
            await settle()

        async def focus(selector):
            control = app.screen.query_one(selector)
            control.focus()
            await settle()
            assert control.has_focus
            region, clip = app.screen._compositor.visible_widgets[control]
            assert region.width > 0 and region.height > 0, (selector, region)
            assert region.intersection(clip) == region, (selector, region, clip)
            if isinstance(control, Button):
                hit, _ = app.screen.get_widget_at(
                    region.x + region.width // 2, region.y + region.height // 2
                )
                assert hit is control, (selector, hit)
                painted = "\n".join(
                    strip.crop(region.x, region.right).text
                    for strip in app.screen._compositor.render_strips()[
                        region.y : region.bottom
                    ]
                )
                assert control.label.plain in painted, (selector, painted)
                await wait_for(lambda: not control.has_class("-active"), "button ready")
            return control

        async def capture(stem):
            await wait_for(lambda: not app._notifications, "notices clear")
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
            await app.handle_screen_navigation(NavigateToScreen("mcp"))
            await wait_for(
                lambda: getattr(app.screen, "screen_name", None) == "mcp", "MCP"
            )
            await wait_for(lambda: not app.screen.workbench.is_loading, "catalog")
            profiles = (
                app.unified_mcp_service.permission_store.read_snapshot_strict().payload[
                    "profiles"
                ]
            )
            for theme in ("textual-dark", "textual-light"):
                for width, height in ((120, 40), (170, 48)):
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
                    rows = [
                        {
                            "conversation_id": "review-only",
                            "title": "Review conversation",
                        }
                    ]
                    dialogs = [
                        (
                            "extraction",
                            FileExtractionDialog(
                                [
                                    ExtractedFile(
                                        filename="review.txt",
                                        content="Review only",
                                        language="text",
                                        start_pos=0,
                                        end_pos=11,
                                    )
                                ]
                            ),
                            "#cancel",
                            150,
                        ),
                        (
                            "dictionary",
                            DictionaryAttachPicker(rows),
                            "#dict-attach-cancel",
                            120,
                        ),
                        (
                            "conversation",
                            ConversationAttachPicker(rows),
                            "#conversation-attach-cancel",
                            120,
                        ),
                    ]
                    for name, modal, cancel, cap in dialogs:
                        previous = app.screen
                        await app.push_screen(modal)
                        await settle()
                        await focus(cancel)
                        body = modal.query_one(
                            f"{type(modal).__name__} > Vertical", Vertical
                        )
                        expected = (
                            min(cap, width * 85 // 100)
                            if width >= 150
                            else min(
                                100 if name == "extraction" else 80,
                                width * (80 if name == "extraction" else 60) // 100,
                            )
                        )
                        assert body.region.width == expected, (
                            name,
                            body.region,
                            expected,
                        )
                        stem = f"{theme}-{width}x{height}-{name}"
                        await capture(stem)
                        result["cells"].append(
                            {
                                "theme": theme,
                                "size": [width, height],
                                "modal": name,
                                "body_region": list(body.region),
                            }
                        )
                        await pilot.press("enter")
                        await wait_for(
                            lambda previous=previous: app.screen is previous, "cancel"
                        )
                        record()
            assert (
                app.unified_mcp_service.permission_store.read_snapshot_strict().payload[
                    "profiles"
                ]
                == profiles
            )
            result["permission_profiles_unchanged"] = True
            result["record_count"] = len(
                app.unified_mcp_service.execution_log.read_recent(200)
            )
            assert result["record_count"] == 0
            result["network_attempts"] = list(attempts)
            assert not attempts
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve failure and shut down the native app
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            if isinstance(
                app.screen,
                (
                    FileExtractionDialog,
                    DictionaryAttachPicker,
                    ConversationAttachPicker,
                ),
            ):
                await app.pop_screen()
                await settle()
            record()
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
    return 0 if result.get("passed") else 1
