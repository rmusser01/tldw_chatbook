"""Native saved-Prompt action journey in an already-isolated profile."""

import asyncio
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

root = Path(sys.argv[1]).resolve()
socket, session = sys.argv[2:4]
os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
os.environ["XDG_DATA_HOME"] = str(root / "data")
os.environ["XDG_CONFIG_HOME"] = str(root / "config")

from textual import events
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Library.prompt_delete_confirmation_modal import (
    PromptDeleteConfirmationModal,
)

app = TldwCli()
result = {"pid": os.getpid(), "steps": []}


def record():
    (root / "result.json").write_text(json.dumps(result, indent=2))


def painted(widget):
    region = widget.region
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


async def journey(pilot):
    async def wait_for(predicate, label):
        deadline = asyncio.get_running_loop().time() + 30
        while asyncio.get_running_loop().time() < deadline:
            try:
                ready = predicate()
            except Exception:  # noqa: BLE001 - transient probe predicate during UI replacement
                ready = False
            if ready:
                await pilot.pause()
                return
            await pilot.pause(0.05)
        raise AssertionError(label)

    async def more(action):
        opener = screen.query_one("#library-prompt-more-actions", Button)
        opener.focus()
        if not screen.query_one("#library-prompt-more-actions-region").display:
            await pilot.press("enter")
        for _ in range(6):
            await pilot.press("tab")
            if screen.focused.id == f"library-prompt-{action}":
                assert str(screen.focused.label).replace("…", "") in painted(
                    screen.focused
                )
                await pilot.press("enter")
                return
        raise AssertionError(action)

    async def answer_delete(confirm):
        await wait_for(
            lambda: isinstance(app.screen, PromptDeleteConfirmationModal),
            "confirmation",
        )
        button = app.screen.query_one(
            "#prompt-delete-confirm" if confirm else "#prompt-delete-cancel", Button
        )
        button.focus()
        await pilot.pause()
        assert str(button.label) in painted(button)
        await pilot.press("enter")

    async def settled_list(total):
        await wait_for(
            lambda: (
                screen._prompts_state.view == "list"
                and not screen._prompts_state.mutation_in_flight
                and screen._library_prompt_browse_controller.result.status == "ready"
                and screen._library_prompt_browse_controller.result.total_items == total
            ),
            "list settlement",
        )

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        result["driver"] = type(app._driver).__name__
        assert result["driver"] == "LinuxDriver"
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        await screen._select_library_rail_row("create-prompt")
        await wait_for(lambda: bool(screen.query("#library-prompt-name")), "new editor")
        screen.query_one("#library-prompt-mode-basic", Button).focus()
        await pilot.press("enter")
        original_name = f"Native actions audit {os.getpid()}"
        screen.query_one("#library-prompt-name", Input).value = original_name
        screen.query_one(
            "#library-prompt-user", TextArea
        ).text = "Keep this reusable message."
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).focus()
        await pilot.press("enter")
        await wait_for(
            lambda: (
                screen._prompts_state.selected_prompt_id is not None
                and not screen._prompts_state.dirty
                and not screen._library_prompt_write_worker_is_active()
                and screen._library_prompt_browse_controller.result.status == "ready"
            ),
            "original saved",
        )
        original_id = screen._prompts_state.selected_prompt_id
        result["original_id"] = original_id
        count = screen._library_prompt_browse_controller.result.total_items
        actual_delete = app.prompt_scope_service.delete_prompts
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await asyncio.to_thread(
                subprocess.run,
                [
                    "/opt/homebrew/bin/tmux",
                    "-L",
                    socket,
                    "resize-window",
                    "-t",
                    session,
                    "-x",
                    str(width),
                    "-y",
                    str(height),
                ],
                check=True,
                capture_output=True,
            )
            await wait_for(lambda: tuple(app.size) == (width, height), "native resize")  # noqa: B023 - awaited before advancing the loop
            app.theme = theme
            await pilot.pause()
            if screen._prompts_state.view == "list":
                screen.query_one(f"#library-prompt-row-{original_id}", Button).focus()
                await pilot.press("enter")
                await wait_for(
                    lambda: (
                        screen._prompts_state.selected_prompt_id == original_id
                        and bool(screen.query("#library-prompt-more-actions"))
                    ),
                    "reopen original",
                )
            opener = screen.query_one("#library-prompt-more-actions", Button)
            opener.focus()
            await pilot.press("enter")
            for suffix, label in (
                ("export", "Export"),
                ("copy", "Copy Markdown"),
                ("duplicate", "Duplicate"),
                ("more-collections", "Collections"),
                ("more-history", "History"),
                ("delete", "Delete"),
            ):
                await pilot.press("tab")
                assert screen.focused.id == f"library-prompt-{suffix}"
                assert label in painted(screen.focused)
            app.save_screenshot(f"menu-{width}.svg", path=str(root))
            await pilot.press("escape")
            assert screen.focused is opener
            await more("duplicate")
            await wait_for(
                lambda: (
                    screen.query_one("#library-prompt-name", Input).value
                    == original_name + " (copy)"
                ),
                "duplicate",
            )
            assert (
                screen._prompts_state.selected_prompt_id is None
                and screen._prompts_state.dirty
            )
            assert (
                screen.query_one("#library-prompt-user", TextArea).text
                == "Keep this reusable message."
            )
            screen.query_one(
                "#library-prompt-name", Input
            ).value = f"{original_name} (copy) {width}"
            await pilot.pause()
            screen.query_one("#library-prompt-save", Button).focus()
            await pilot.press("enter")
            await wait_for(
                lambda: (
                    screen._prompts_state.selected_prompt_id not in (None, original_id)
                    and not screen._prompts_state.dirty
                    and screen._library_prompt_browse_controller.result.total_items
                    == count + 1
                ),
                "copy save",
            )
            copy_id = screen._prompts_state.selected_prompt_id
            field = screen.query_one("#library-prompt-user", TextArea)
            await more("delete")
            await answer_delete(False)
            await pilot.pause()
            assert screen.query_one("#library-prompt-user", TextArea) is field
            attempts = 0

            async def fail_once(**kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("simulated storage unavailable")
                return await actual_delete(**kwargs)

            app.prompt_scope_service.delete_prompts = fail_once
            await more("delete")
            await answer_delete(True)
            await wait_for(
                lambda: (
                    screen._prompts_state.status
                    == "Could not delete this prompt. Nothing was deleted."
                    and not screen._prompts_state.mutation_in_flight
                ),
                "delete failure",
            )
            assert screen.query_one("#library-prompt-user", TextArea) is field
            assert "Nothing was deleted" in painted(
                screen.query_one("#library-prompt-save-status", Static)
            )
            assert "Delete" in painted(
                screen.query_one("#library-prompt-delete", Button)
            )
            app.save_screenshot(f"failure-{width}.svg", path=str(root))
            for recovery in ("undo", "receipt-dismiss"):
                await more("delete")
                await answer_delete(True)
                await settled_list(count)
                action = screen.query_one(f"#library-prompts-delete-{recovery}", Button)
                action.focus()
                await pilot.pause()
                assert screen.focused is action
                assert str(action.label) in painted(action)
                if recovery == "undo":
                    app.save_screenshot(f"receipt-{width}.svg", path=str(root))
                await pilot.press("enter")
                await wait_for(
                    lambda: (
                        not screen.query("#library-prompts-delete-receipt-copy")
                        and not screen._prompts_state.mutation_in_flight
                    ),
                    "receipt removed",
                )
                if recovery == "undo":
                    await settled_list(count + 1)
                    screen.query_one(f"#library-prompt-row-{copy_id}", Button).focus()
                    await pilot.press("enter")
                    await wait_for(
                        lambda: (
                            screen._prompts_state.selected_prompt_id == copy_id  # noqa: B023 - awaited before advancing the loop
                            and bool(screen.query("#library-prompt-more-actions"))
                        ),
                        "restored copy",
                    )
                    assert screen._prompts_state.version == 3
            assert attempts == 3
            app.prompt_scope_service.delete_prompts = actual_delete
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "copy_id": copy_id,
                    "six_actions_painted": True,
                    "escape_returns_opener": True,
                    "duplicate_detached": True,
                    "cancel_retains_field": True,
                    "failure_retains_field": True,
                    "error_and_delete_painted": True,
                    "undo_restores_version": 3,
                    "dismiss_keeps_deletion": True,
                    "delete_attempts": attempts,
                }
            )
        await wait_for(
            lambda: (
                not any(
                    w.is_running for w in app.workers if w.group.startswith("library")
                )
            ),
            "Library workers",
        )
        result["passed"] = True
        result["quit_size"] = list(app.size)
        result["normal_quit_requested"] = True
        record()
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - save the complete native probe failure
        result["passed"] = False
        result["error"] = traceback.format_exc()
        record()
        app.exit()


app.run(auto_pilot=journey, size=(80, 24))
result["app_run_returned"] = True
record()
raise SystemExit(0 if result.get("passed") else 1)
