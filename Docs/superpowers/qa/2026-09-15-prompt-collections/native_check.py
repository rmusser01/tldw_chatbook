"""Exercise Prompt Collections in an isolated native TldwCli profile."""

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
        raise AssertionError(label)

    async def focus_on(surface, selector, label):
        await wait_for(
            lambda: (
                surface.focused is surface.query_one(selector)
                and label in painted(surface.focused)
            ),
            f"readable {selector}",
        )

    async def activate(surface, selector, label):
        surface.query_one(selector).focus()
        await focus_on(surface, selector, label)
        await wait_for(
            lambda: not surface.focused.has_class("-active"), "prior press finished"
        )
        await pilot.press("enter")

    async def tab_to(surface, selector, label):
        for _ in range(15):
            if surface.focused is surface.query_one(selector):
                break
            await pilot.press("tab")
        await focus_on(surface, selector, label)

    async def manager_ready():
        await wait_for(
            lambda: (
                isinstance(app.screen, ModalScreen)
                and app.screen._catalog.status in {"ready", "empty"}
                and app.screen.focused
                is app.screen.query_one("#prompt-collection-manager-search")
            ),
            "manager ready",
        )
        return app.screen

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        result["driver"] = type(app._driver).__name__
        assert result["driver"] == "LinuxDriver"
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "native size",
            )
            app.theme = theme
            await screen._select_library_rail_row("create-prompt")
            await wait_for(
                lambda: bool(screen.query("#library-prompt-name")), "new editor"
            )
            await activate(screen, "#library-prompt-mode-basic", "Basic")
            name = f"Native collections {os.getpid()} {width}"
            screen.query_one("#library-prompt-name", Input).value = name
            screen.query_one(
                "#library-prompt-user", TextArea
            ).text = "Reusable [bold] message."
            await pilot.pause()
            await activate(screen, "#library-prompt-save", "Save")
            await wait_for(
                lambda: (
                    screen._prompts_state.version == 1
                    and not screen._prompts_state.dirty
                    and screen._library_prompt_collections_controller.membership_state.can_manage
                ),
                "saved Prompt",
            )
            prompt_id = screen._prompts_state.selected_prompt_id
            original = screen.query_one("#library-prompt-user", TextArea)
            await activate(screen, "#library-prompt-more-actions", "More actions")
            await tab_to(screen, "#library-prompt-more-collections", "Collections")
            await pilot.press("enter")
            modal = await manager_ready()
            collection_name = f"[bold] café {width}"
            modal.query_one(
                "#prompt-collection-manager-new-name", Input
            ).value = collection_name
            await activate(modal, "#prompt-collection-manager-create", "New collection")
            await wait_for(
                lambda modal=modal: modal._outcome == "Collection created.",
                "collection created",
            )
            collection_id = next(
                i.collection_id
                for i in modal._catalog.items
                if i.name == collection_name
            )
            await focus_on(
                modal, "#prompt-collection-manager-new-name", collection_name
            )
            await activate(modal, "#prompt-collection-manager-create", "New collection")
            await wait_for(
                lambda modal=modal: (
                    modal._outcome == "Name already exists — choose another."
                ),
                "collision",
            )
            await focus_on(
                modal, "#prompt-collection-manager-new-name", collection_name
            )
            app.save_screenshot(f"collision-{width}.svg", path=str(root))
            member = f"#prompt-collection-manager-member-{collection_id}"
            modal.query_one(member).focus()
            await focus_on(modal, member, collection_name)
            renamed = f"[bold] café renamed {width}"
            modal.query_one(
                "#prompt-collection-manager-new-name", Input
            ).value = renamed
            await activate(
                modal, "#prompt-collection-manager-rename", "Rename selected"
            )
            await wait_for(
                lambda modal=modal: modal._outcome == "Collection renamed.", "renamed"
            )
            await focus_on(modal, member, renamed)
            await pilot.press("space")
            await tab_to(modal, "#prompt-collection-manager-done", "Done")
            await pilot.press("enter")
            await wait_for(lambda: app.screen is screen, "manager Done")
            await focus_on(
                screen, "#library-prompt-memberships-manage", "Manage collections"
            )
            assert screen._prompts_state.editor_mode == "info"
            assert not screen.query_one("#library-prompt-more-actions-region").display
            screen.query_one("#library-prompt-name", Input).value = name + " unsaved"
            await wait_for(lambda: screen._prompts_state.dirty, "dirty title")
            await pilot.press("tab")
            await focus_on(
                screen, "#library-prompt-memberships-apply", "Apply memberships"
            )
            app.save_screenshot(f"staged-{width}.svg", path=str(root))
            await pilot.press("enter")
            await wait_for(
                lambda: (
                    screen._library_prompt_collections_controller.membership_state.status
                    == "success"
                ),
                "applied",
            )
            assert screen._prompts_state.dirty
            assert screen.query_one("#library-prompt-user", TextArea) is original
            await activate(
                screen, "#library-prompt-memberships-manage", "Manage collections"
            )
            modal = await manager_ready()
            modal.query_one(member).focus()
            await focus_on(modal, member, renamed)
            await pilot.press("space", "escape")
            await wait_for(lambda: app.screen is screen, "manager Cancel")
            await focus_on(
                screen, "#library-prompt-memberships-manage", "Manage collections"
            )
            state = screen._library_prompt_collections_controller.membership_state
            assert state.staged_ids == state.applied_ids == (collection_id,)
            app.save_screenshot(f"applied-{width}.svg", path=str(root))
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "prompt_id": prompt_id,
                    "prompt_name": name,
                    "prompt_version": 1,
                    "collection_id": collection_id,
                    "collection_name": renamed,
                    "basic_menu_reveals_info": True,
                    "create_collision_retains_name": True,
                    "rename_literal_name": True,
                    "done_stages_cancel_retains_applied": True,
                    "apply_preserves_dirty_draft_and_fields": True,
                }
            )
            record()
            await activate(screen, "#library-prompt-discard", "Cancel")
            await wait_for(
                lambda: screen._prompts_state.view == "list", "discard unsaved title"
            )
        result["passed"] = True
        result["normal_quit_requested"] = True
        record()
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - Record native assertion failures before cleanup.
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
