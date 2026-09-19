"""Exercise Skills editing in an isolated native TldwCli profile."""

import asyncio
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

root = Path(sys.argv[1]).resolve()
socket, session = sys.argv[2:4]
os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
os.environ["XDG_DATA_HOME"] = str(root / "data")
os.environ["XDG_CONFIG_HOME"] = str(root / "config")

from textual import events
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Input, TextArea

from tldw_chatbook.app import TldwCli

app = TldwCli()
result = {"pid": os.getpid(), "steps": []}


def record():
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def painted(widget):
    region = widget.region
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


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
        deadline = asyncio.get_running_loop().time() + 20
        while asyncio.get_running_loop().time() < deadline:
            try:
                if predicate():
                    await pilot.pause()
                    return
            except (NoMatches, QueryError):
                pass
            await pilot.pause(0.03)
        raise AssertionError(f"{label}: focus={app.screen.focused!r}")

    async def activate(selector, label):
        await wait_for(lambda: bool(app.screen.query(selector)), selector)

        def focus_current():
            target = app.screen.query_one(selector)
            target.focus()
            return target.has_focus and target.region.width > 0

        await wait_for(focus_current, f"focus {selector}")
        target = app.screen.query_one(selector)
        if label:
            await wait_for(lambda: label in painted(target), f"readable {selector}")
        await wait_for(lambda: not target.has_class("-active"), "action feedback")
        await pilot.press("enter")
        await pilot.pause()

    async def capture(name):
        await wait_for(lambda: not app.query("Toast"), "notices expired")
        app.save_screenshot(name, path=str(root))

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        result["driver"] = type(app._driver).__name__
        assert result["driver"] == "LinuxDriver"
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        local = app.local_skills_service
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "size",
            )
            app.theme = theme
            name = f"native-edit-{os.getpid()}-{width}"
            await local.create_skill(
                name=name,
                content=f"---\nname: {name}\ndescription: Saved description\nallowed-tools: fs_read fs_read unknown_tool\n---\nOriginal body.",
            )
            await screen._select_library_rail_row("browse-skills")
            await wait_for(
                lambda name=name: bool(screen.query(f"#library-skill-row-{name}")),
                "skill row",
            )
            shell = screen.query_one("#library-skills-reader-shell")
            await wait_for(
                lambda shell=shell: shell.effective_layout.reader_width > 0,
                "Skills layout",
            )
            if not shell.effective_layout.items_open:
                await activate("#library-skills-items-grip", None)
            await activate(f"#library-skill-row-{name}", name)
            await activate("#library-skill-mode-edit", "Edit")
            await wait_for(lambda: bool(screen.query("#library-skill-body")), "editor")
            body = screen.query_one("#library-skill-body", TextArea)
            description = screen.query_one("#library-skill-description", Input)
            description.value = "Edited café [bold]"
            body.text = "Saved body {literal}."
            await pilot.pause()
            await activate("#library-skill-editor-mode", "Show advanced")
            assert screen.query_one("#library-skill-body") is body
            await capture(f"advanced-{width}.svg")
            await activate("#library-skill-editor-mode", "Show basic")
            await activate("#library-skill-save", "Save changes")
            await wait_for(
                lambda: (
                    not screen._skills_state.dirty
                    and not screen._skills_state.mutation_in_flight
                ),
                "saved",
            )
            await wait_for(
                lambda: screen.focused is screen.query_one("#library-skill-back"),
                "save focus",
            )
            await capture(f"saved-{width}.svg")
            saved = await local.get_skill(name)
            assert "Saved body {literal}." in saved["content"]
            metadata = yaml.safe_load(saved["content"].split("---", 2)[1])
            assert metadata["description"] == "Edited café [bold]"
            assert metadata["allowed_tools"] == ["fs_read", "fs_read", "unknown_tool"]
            if shell.effective_layout.items_open:
                await activate("#library-skills-items-grip", "<---")
            await activate("#library-skill-back", "Back to list")
            await wait_for(
                lambda: (
                    screen.focused is not None
                    and (screen.focused.id or "").startswith("library-skill-row-")
                ),
                "Back focus",
            )
            await activate(f"#library-skill-row-{name}", name)
            await activate("#library-skill-mode-edit", "Edit")
            await wait_for(
                lambda: bool(screen.query("#library-skill-description")), "reopened"
            )
            screen.query_one("#library-skill-description", Input).value = "Discard me"
            await pilot.pause()
            await pilot.press("escape")
            assert screen._skills_state.dirty
            await activate("#library-skill-discard", "Discard changes")
            await wait_for(lambda: screen._skills_state.view == "list", "discarded")
            await wait_for(
                lambda: (
                    screen.focused is not None
                    and (screen.focused.id or "").startswith("library-skill-row-")
                ),
                "discard focus",
            )
            await capture(f"discarded-{width}.svg")
            assert (await local.get_skill(name))["content"] == saved["content"]
            await screen._select_library_rail_row("create-skill")
            await wait_for(
                lambda: bool(screen.query("#library-skill-name")), "New skill"
            )
            screen.query_one("#library-skill-name", Input).value = "cancelled-native"
            screen.query_one("#library-skill-body", TextArea).text = "Never saved."
            await pilot.pause()
            await activate("#library-skill-cancel", "Cancel")
            await wait_for(
                lambda: (
                    screen._library_selected_row_id == "browse-skills"
                    and screen.focused is not None
                    and (
                        (screen.focused.id or "").startswith("library-skill-row-")
                        or screen.focused.id == "library-skills-filter"
                    )
                    and screen.focused is screen.query_one(f"#{screen.focused.id}")
                    and screen.focused.region.width > 0
                ),
                "Cancel focus",
            )
            await capture(f"cancelled-{width}.svg")
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "name": name,
                    "saved_version": saved["version"],
                    "disclosure_preserved_body": True,
                    "save_focus": True,
                    "cancel_focus": screen.focused.id,
                    "exact_allowlist": True,
                    "dirty_escape_preserved": True,
                    "discard_preserved_saved_content": True,
                }
            )
            record()
        result["passed"] = True
        result["normal_quit_requested"] = True
        record()
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - record failures before normal native shutdown
        result["passed"] = False
        result["error"] = traceback.format_exc()
        app.save_screenshot("failed-state.svg", path=str(root))
        record()
        if isinstance(app.screen, ModalScreen):
            await pilot.press("escape")
        app.post_message(events.Key("ctrl+q", "\x11"))


app.run(auto_pilot=journey, size=(170, 48))
result["app_run_returned"] = True
record()
raise SystemExit(0 if result.get("passed") else 1)
