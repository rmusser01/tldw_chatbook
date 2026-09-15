"""Exercise Collections reader continuity in an isolated native TldwCli profile."""

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

os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
os.environ.pop("NO_COLOR", None)

from textual import events
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Input, TextArea

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Library.collections_capture_models import CaptureSaveRequest

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
            await scope.save_highlight(
                saved.capture.identity, quote=f"{title} annotation"
            )
        result["capture_ids"] = [identity.capture_id for identity in identities]
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
            await screen._select_library_rail_row("browse-collections")
            engine = screen._library_collections_capture_controller
            await wait_for(
                lambda engine=engine: engine.state.identity_actions_enabled,
                "capture ready",
            )
            shell = screen.query_one("#library-collections-reader-shell")
            if not shell.effective_layout.library_open:
                await activate("#library-collections-library-grip", "--->")
            await activate("#library-collections-scope-all", "All Captures")
            await wait_for(
                lambda engine=engine: engine.state.identity_actions_enabled, "all scope"
            )
            await screen._select_library_collection_capture(identities[0])
            await activate("#library-collections-mode-notes", "Notes")
            await wait_for(
                lambda: bool(screen.query("#library-collections-freeform-note")),
                "Notes",
            )
            draft = f"Native note at {width} columns"
            screen.query_one(
                "#library-collections-freeform-note", TextArea
            ).text = draft
            await pilot.pause()
            await activate("#library-collections-more", "More")
            await expect_focus("#library-collections-more", "More")
            assert (
                screen.query_one("#library-collections-freeform-note", TextArea).text
                == draft
            )
            await activate("#library-collections-more", "More")
            await activate("#library-collections-mode-info", "Info")
            await expect_focus("#library-collections-mode-info", "Info")
            await activate("#library-collections-mode-notes", "Notes")
            assert (
                screen.query_one("#library-collections-freeform-note", TextArea).text
                == draft
            )
            await activate(
                "#library-collections-freeform-note-save", "Save capture note"
            )
            await wait_for(
                lambda: (
                    screen._collections_state.action_status == "Capture note saved."
                ),
                "saved note",
            )
            assert (
                await scope.get_detail(identities[0])
            ).capture.freeform_note == draft
            await expect_focus(
                "#library-collections-freeform-note-save", "Save capture note"
            )
            await capture(f"note-{width}.svg")
            await activate("#library-collections-mode-highlights", "Highlights")
            await wait_for(
                lambda: any(
                    h.quote == "Alpha annotation"
                    for h in screen._collections_state.highlights
                ),
                "Alpha highlights",
            )
            screen.query_one(
                "#library-collections-highlight-quote", TextArea
            ).text = f"Quote at {width}"
            screen.query_one(
                "#library-collections-highlight-note", Input
            ).value = "Native draft"
            await pilot.pause()
            await activate("#library-collections-mode-info", "Info")
            await activate("#library-collections-mode-highlights", "Highlights")
            assert (
                screen.query_one("#library-collections-highlight-quote", TextArea).text
                == f"Quote at {width}"
            )
            await activate("#library-collections-highlight-save", "Add highlight")
            await wait_for(
                lambda: screen._collections_state.action_status == "Highlight saved.",
                "saved highlight",
            )
            assert (
                screen.query_one("#library-collections-highlight-quote", TextArea).text
                == ""
            )
            await screen._select_library_collection_capture(identities[1])
            await wait_for(
                lambda: (
                    [h.quote for h in screen._collections_state.highlights]
                    == ["Beta annotation"]
                ),
                "Beta highlights",
            )
            screen.query_one("#library-collections-mode-highlights").focus()
            await expect_focus("#library-collections-mode-highlights", "Highlights")
            await pilot.press("end")
            await wait_for(
                lambda: (
                    "Beta annotation"
                    in painted(screen.query_one("#library-collections-work"))
                ),
                "painted Beta annotation",
            )
            await capture(f"highlights-{width}.svg")
            shell = screen.query_one("#library-collections-reader-shell")
            if not shell.effective_layout.library_open:
                await activate("#library-collections-library-grip", "--->")
            await activate("#library-collections-scope-saved", "Saved")
            await wait_for(
                lambda engine=engine: engine.state.identity_actions_enabled,
                "saved scope",
            )
            first = engine.state.selected_identity
            await activate("#library-collections-archive", "Move to Archive")
            await wait_for(
                lambda engine=engine, first=first: (
                    engine.state.identity_actions_enabled
                    and engine.state.selected_identity != first
                ),
                "successor reader",
            )
            assert (
                engine.state.loaded_detail.capture.identity
                == engine.state.selected_identity
            )
            screen.query_one("#library-collections-archive-undo").focus()
            await expect_focus("#library-collections-archive-undo", "Undo")
            await capture(f"undo-{width}.svg")
            await activate("#library-collections-archive-undo", "Undo")
            await wait_for(
                lambda engine=engine: not engine.state.visible_archive_receipts,
                "Undo settled",
            )
            assert (await scope.get_detail(first)).capture.status == "saved"
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "drafts_preserved": True,
                    "saved_note_exact": True,
                    "saved_note_action_focused_and_visible": True,
                    "selected_highlight_painted": True,
                    "highlight_saved_once_and_cleared": True,
                    "highlights_follow_capture": True,
                    "archive_loads_successor": True,
                    "undo_reachable_and_restores_saved": True,
                }
            )
            record()
        result["persisted"] = []
        for identity in identities:
            detail = (await scope.get_detail(identity)).capture
            highlights = (await scope.list_highlights(identity)).items
            result["persisted"].append(
                {
                    "capture_id": identity.capture_id,
                    "title": detail.title,
                    "note": detail.freeform_note,
                    "status": detail.status,
                    "highlights": [h.quote for h in highlights],
                }
            )
        assert result["persisted"][0]["note"] == "Native note at 80 columns"
        assert set(result["persisted"][0]["highlights"]) == {
            "Alpha annotation",
            "Quote at 170",
            "Quote at 80",
        }
        assert result["persisted"][1]["highlights"] == ["Beta annotation"]
        result["passed"] = True
        record()
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - preserve evidence before normal shutdown
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
