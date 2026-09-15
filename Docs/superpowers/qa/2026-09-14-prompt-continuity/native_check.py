"""Bounded native-driver Prompt audit using disposable local storage."""
import asyncio
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

root = Path(__file__).resolve().parent
width, height = map(int, sys.argv[1:3])
live = root / f"live-{width}"
previous = root.parent / "2026-09-14-library-remaining" / "live"
if not live.exists():
    live.mkdir()
    shutil.copytree(previous / "data", live / "data")
    for name in ("config.toml", "runtime_policy.json", "ui_state.toml"):
        (live / name).write_text(
            (previous / name).read_text().replace(str(previous), str(live))
        )
os.environ["TLDW_CONFIG_PATH"] = str(live / "config.toml")
os.environ["XDG_DATA_HOME"] = str(live / "data")
os.environ["XDG_CONFIG_HOME"] = str(live / "config")

from textual.widgets import Button, Input, Static, TextArea
from tldw_chatbook.app import TldwCli

app = TldwCli()
result = {"steps": [], "expected_size": [width, height]}

async def journey(pilot):
    async def wait_for(predicate, label):
        deadline = asyncio.get_running_loop().time() + 30
        while asyncio.get_running_loop().time() < deadline:
            if predicate():
                await pilot.pause()
                return
            await pilot.pause(0.05)
        raise AssertionError(label)

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        result["size"] = list(app.size)
        result["driver"] = type(app._driver).__name__
        assert list(app.size) == [width, height]
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        await screen._select_library_rail_row("create-prompt")
        await wait_for(lambda: bool(screen.query("#library-prompt-name")), "new editor")
        await wait_for(
            lambda: screen._library_prompt_browse_controller.result.status == "ready",
            "New prompt Items settlement",
        )
        before_count = screen._library_prompt_browse_controller.result.total_items
        screen.query_one("#library-prompt-mode-basic", Button).focus()
        await pilot.press("enter")
        await wait_for(lambda: screen._prompts_state.editor_mode == "basic", "Basic")
        name = screen.query_one("#library-prompt-name", Input)
        name.value = f"Prompt continuity audit {width} {os.getpid()}"
        content = screen.query_one("#library-prompt-user", TextArea)
        expected_text = "Retain this message template after saving."
        content.text = expected_text
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).focus()
        await pilot.press("enter")
        await wait_for(
            lambda: screen._prompts_state.selected_prompt_id is not None
            and not screen._prompts_state.dirty,
            "saved identity",
        )
        prompt_id = screen._prompts_state.selected_prompt_id
        await wait_for(
            lambda: screen._library_prompt_browse_controller.result.status == "ready"
            and screen._library_prompt_browse_controller.result.total_items == before_count + 1,
            "post-save Items refresh",
        )
        await wait_for(
            lambda: bool(screen.query(f"#library-prompt-row-{prompt_id}")), "saved row"
        )
        assert screen.query_one("#library-prompt-user") is content
        assert screen.query_one("#library-prompt-name") is name
        assert not screen.query_one("#library-prompt-save", Button).display
        assert screen.query_one("#library-prompt-insert-console", Button).display
        await wait_for(lambda: not screen._library_prompt_write_worker_is_active(), "save worker settlement")
        assert all("Unsaved changes" not in str(status.renderable) for status in screen.query(".prompt-block-status"))
        assert "structured format" in str(screen.query_one("#library-prompt-artifact-status", Static).renderable)
        result["steps"].append({"saved_prompt_id": prompt_id, "items_before": before_count,
                                "items_after": before_count + 1, "fields_retained": True})
        for theme in ("textual-dark", "textual-light"):
            app.theme = theme
            await pilot.pause()
            for mode in ("basic", "advanced"):
                screen.query_one(f"#library-prompt-mode-{mode}", Button).focus()
                await pilot.press("enter")
                await wait_for(lambda: screen._prompts_state.editor_mode == mode, mode)
                target = content if mode == "basic" else screen.query(".prompt-block-content").first(TextArea)
                assert target.text == expected_text
                target.focus()
                await pilot.pause()
                assert screen.focused is target
                app.save_screenshot(f"{mode}-{theme}-{width}.svg", path=str(root))
                result["steps"].append({"mode": mode, "theme": theme, "focus": target.id,
                                        "region": list(target.region)})
        screen.query_one("#library-prompt-back", Button).focus()
        await pilot.press("enter")
        await wait_for(lambda: bool(screen.query("#library-prompt-work-empty")), "Back")
        await wait_for(
            lambda: screen._library_prompt_browse_controller.result.status == "ready",
            "return Items settlement",
        )
        screen.query_one(f"#library-prompt-row-{prompt_id}", Button).focus()
        await pilot.press("enter")
        await wait_for(lambda: bool(screen.query("#library-prompt-user")), "reopened")
        assert screen.query_one("#library-prompt-user", TextArea).text == expected_text
        result["steps"].append({"reopened_prompt_id": prompt_id, "text_matches": True,
                                "version": screen._prompts_state.version})
        await wait_for(
            lambda: screen._library_snapshot_rendered_generation == screen._library_snapshot_state_generation
            and not any(worker.is_running for worker in app.workers if worker.group.startswith("library")),
            "Library workers settled before quit",
        )
        result["passed"] = True
    except Exception:
        result["passed"] = False
        result["error"] = traceback.format_exc()
    finally:
        (root / f"native-result-{width}.json").write_text(json.dumps(result, indent=2))
        await pilot.press("ctrl+q")
        app.exit()

app.run(auto_pilot=journey, size=(width, height))
raise SystemExit(0 if result.get("passed") else 1)
