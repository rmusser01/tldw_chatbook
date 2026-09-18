"""Exercise compact Collections actions and search in an isolated native TldwCli profile."""

import asyncio
import json
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

root = Path(sys.argv[1]).resolve()
socket, session = sys.argv[2:4]
os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
os.environ["XDG_DATA_HOME"] = str(root / "data")
os.environ["XDG_CONFIG_HOME"] = str(root / "config")

os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
os.environ.pop("NO_COLOR", None)

from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Input, TextArea

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Library.collections_capture_models import (
    CaptureSaveRequest,
)

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

    async def expect_focus(selector, label):
        await wait_for(
            lambda: (
                app.screen.focused is app.screen.query_one(selector)
                and label in painted(app.screen.focused)
            ),
            f"natural focus {selector}",
        )

    async def capture(name):
        await wait_for(lambda: not app.query("Toast"), "notices expired")
        app.save_screenshot(name, path=str(root))

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        assert type(app._driver).__name__ == "LinuxDriver"
        result["driver"] = type(app._driver).__name__
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        app.ensure_collections_capture_services()
        scope = app.collections_capture_scope_service
        key = scope.active_authority.key
        identities = []
        for title in ("Alpha", "Beta"):
            saved = await scope.save_capture(
                CaptureSaveRequest(
                    key,
                    f"https://example.test/{title.lower()}",
                    title=title,
                    text_content=f"Readable {title} body. " * 60,
                    freeform_note=f"Saved {title} note",
                )
            )
            identities.append(saved.capture.identity)
        result["capture_ids"] = [i.capture_id for i in identities]
        await screen._select_library_rail_row("browse-collections")
        engine = screen._library_collections_capture_controller
        prior_draft = None
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "resize",
            )
            app.theme = theme
            await screen._select_library_collection_capture(identities[0])
            await wait_for(lambda: engine.state.identity_actions_enabled, "Alpha ready")
            shell = screen.query_one("#library-collections-reader-shell")
            if not shell.effective_layout.items_open:
                await activate("#library-collections-items-grip", "--->")
            await activate("#library-collections-mode-notes", "Notes")
            note = screen.query_one("#library-collections-freeform-note", TextArea)
            if prior_draft:
                assert note.text == prior_draft
            draft = f"Unsaved native draft at {width} columns"
            note.text = draft
            prior_draft = draft
            await pilot.pause()
            screen.query_one("#library-collections-mark-read").focus()
            await expect_focus("#library-collections-mark-read", "Mark Read")
            controls = (
                ("favorite", "Favorite"),
                ("archive", "Move to Archive"),
                ("open-original", "Open Original"),
                ("more", "More"),
                ("mode-read", "Read"),
                ("mode-highlights", "Highlights"),
                ("mode-notes", "Notes"),
                ("mode-info", "Info"),
            )
            for name, label in controls:
                await pilot.press("tab")
                await expect_focus(f"#library-collections-{name}", label)
            assert screen.query_one("#library-collections-freeform-note") is note
            assert note.text == draft
            await capture(f"controls-{width}.svg")
            for name, label in reversed(controls[:-1]):
                await pilot.press("shift+tab")
                await expect_focus(f"#library-collections-{name}", label)
            before = (await scope.get_detail(identities[0])).capture.favorite
            await pilot.press("enter")
            await wait_for(
                lambda before=before: (
                    engine.state.loaded_detail.capture.favorite != before
                ),
                "Favorite changed",
            )
            assert (await scope.get_detail(identities[0])).capture.favorite != before
            await expect_focus("#library-collections-favorite", "Favorite")
            request = replace(
                engine.state.requested_scope,
                search="Alpha",
                sort="relevance",
                statuses=("saved",),
            )
            await screen._collections_controller._apply_library_collection_capture_request(
                request
            )
            await wait_for(
                lambda: engine.state.page is not None and engine.state.page.total == 1,
                "search ready",
            )
            field = screen.query_one("#library-collections-filter", Input)
            field.focus()
            await pilot.pause()
            await pilot.press(
                "home", "shift+end", "backspace", "space", "space", "space", "enter"
            )
            await wait_for(
                lambda: engine.state.page is not None and engine.state.page.total == 2,
                "search cleared",
            )
            assert engine.state.applied_scope.sort == "saved_desc"
            assert engine.state.applied_scope.search == ""
            assert engine.state.applied_scope.statuses == ("saved",)
            await expect_focus("#library-collections-filter", "Filter captures")
            await capture(f"search-{width}.svg")
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "items_open": True,
                    "tab_and_reverse_tab_labels_readable": True,
                    "favorite_enter_persisted": True,
                    "focus_retained_after_action": True,
                    "draft_retained": True,
                    "empty_search_recovered": True,
                    "valid_sort_and_saved_scope": True,
                }
            )
            record()
        result["persisted"] = []
        for identity in identities:
            detail = (await scope.get_detail(identity)).capture
            assert detail.freeform_note == f"Saved {detail.title} note"
            assert detail.status == "saved" and not detail.favorite
            result["persisted"].append(
                {
                    "capture_id": identity.capture_id,
                    "title": detail.title,
                    "note": detail.freeform_note,
                    "status": detail.status,
                    "favorite": detail.favorite,
                }
            )
        result["passed"] = True
        record()
        await tmux("send-keys", "-t", session, "C-q")
    except Exception:  # noqa: BLE001 - preserve evidence before normal shutdown
        result["passed"] = False
        result["error"] = traceback.format_exc()
        app.save_screenshot("failed-state.svg", path=str(root))
        record()
        if isinstance(app.screen, ModalScreen):
            await pilot.press("escape")
        await tmux("send-keys", "-t", session, "C-q")


app.run(auto_pilot=journey, size=(170, 48))
result["app_run_returned"] = True
record()
raise SystemExit(0 if result.get("passed") else 1)
