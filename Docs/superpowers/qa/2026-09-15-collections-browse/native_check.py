"""Exercise Collections browse controls in an isolated native TldwCli profile."""

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
from textual.widgets import Button, Input

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Library.collections_capture_models import (
    CapturePageRequest,
    CaptureSaveRequest,
    CollectionsCaptureError,
    SavedCaptureSearch,
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
        for n in range(21):
            await scope.save_saved_search(
                SavedCaptureSearch(
                    key,
                    "new",
                    f"Research {n:02}",
                    CapturePageRequest(key, search="Alpha", statuses=("saved",)),
                    "",
                    "",
                    1,
                )
            )
        result["capture_ids"] = [i.capture_id for i in identities]
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            for identity in identities:
                detail = (await scope.get_detail(identity)).capture
                if detail.status != "saved":
                    await scope.update_capture(
                        identity, detail.revision, {"status": "saved"}
                    )
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

            async def show_pane(name):
                shell = screen.query_one("#library-collections-reader-shell")
                if not getattr(shell.effective_layout, f"{name}_open"):
                    await activate(f"#library-collections-{name}-grip", "--->")

            await show_pane("library")
            await activate("#library-collections-scope-saved", "Saved")
            request = replace(
                engine.state.requested_scope,
                search="missing",
                sort="relevance",
                domain="absent.test",
                tags=("absent",),
                date_from="2001-01-01",
                date_to="2001-12-31",
            )
            await screen._collections_controller._apply_library_collection_capture_request(
                request
            )
            await wait_for(
                lambda engine=engine: engine.state.page.total == 0,
                "empty filtered scope",
            )
            await show_pane("items")
            await activate("#library-collections-filters", "Filters")
            await activate("#library-collections-filters-clear", "Clear")
            await wait_for(
                lambda engine=engine: engine.state.page.total == 2,
                "Clear restores results",
            )
            current = engine.state.applied_scope
            assert current.search == "" and current.sort == "saved_desc"
            assert (
                current.statuses == ("saved",)
                and current.tags == ()
                and current.domain is None
            )
            assert current.date_from is None and current.date_to is None
            assert screen.query_one("#library-collections-filter", Input).value == ""
            await expect_focus("#library-collections-filters-clear", "Clear")
            await capture(f"clear-{width}.svg")
            await activate("#library-collections-filters", "Filters")
            await show_pane("library")
            before = engine.state.applied_scope
            await activate("#library-collections-more-saved-searches", "More searches")
            await wait_for(
                lambda: len(screen._collections_state.saved_searches) == 1,
                "second saved-search page",
            )
            assert engine.state.applied_scope == before
            search = screen._collections_state.saved_searches[0]
            selector = f"#library-collections-saved-search-{search.search_id}"
            await expect_focus(selector, search.name)
            await capture(f"searches-{width}.svg")
            await pilot.press("enter")
            await wait_for(
                lambda engine=engine: engine.state.page.total == 1,
                "saved search applied",
            )
            assert engine.state.applied_scope.search == "Alpha"
            assert engine.state.applied_scope.statuses == ("saved",)
            assert engine.state.page.items[0].title == "Alpha"
            await activate("#library-collections-previous-saved-searches", "Previous")
            await wait_for(
                lambda: len(screen._collections_state.saved_searches) == 20,
                "first saved-search page",
            )
            assert (
                engine.state.page.total == 1
                and engine.state.applied_scope.search == "Alpha"
            )
            await expect_focus(
                "#library-collections-more-saved-searches", "More searches"
            )
            await activate("#library-collections-scope-all", "All Captures")
            await screen._select_library_collection_capture(identities[0])
            await wait_for(
                lambda engine=engine: engine.state.identity_actions_enabled,
                "reader ready",
            )
            await activate("#library-collections-mark-read", "Mark Read")
            await wait_for(
                lambda engine=engine: (
                    engine.state.loaded_detail.capture.status == "read"
                ),
                "marked read",
            )
            await activate("#library-collections-archive", "Move to Archive")
            await wait_for(
                lambda: bool(screen.query("#library-collections-archive-undo")),
                "archive receipt",
            )
            archived = (await scope.get_detail(identities[0])).capture
            assert archived.status == "archived"
            button = screen.query_one("#library-collections-archive", Button)
            assert button.disabled and "already" in str(button.tooltip).lower()
            await wait_for(
                lambda button=button: "Archived" in painted(button),
                "readable archived state",
            )
            try:
                await scope.archive(identities[0], archived.revision)
            except CollectionsCaptureError as error:
                assert "already_archived" in str(error)
            else:
                raise AssertionError("Repeated Archive was accepted")
            assert (
                await scope.get_detail(identities[0])
            ).capture.revision == archived.revision
            screen.query_one("#library-collections-archive-undo").focus()
            await expect_focus("#library-collections-archive-undo", "Undo")
            await capture(f"archive-{width}.svg")
            await pilot.press("enter")
            await wait_for(
                lambda engine=engine: not engine.state.visible_archive_receipts,
                "Undo settled",
            )
            assert (await scope.get_detail(identities[0])).capture.status == "read"
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "clear_restores_two_saved_results": True,
                    "clear_remains_visible_and_focused": True,
                    "clear_resets_all_filters_and_relevance": True,
                    "more_reaches_second_page_and_preserves_scope": True,
                    "saved_search_enter_filters_to_alpha": True,
                    "previous_preserves_active_search": True,
                    "repeated_archive_disabled_and_rejected": True,
                    "original_undo_restores_read": True,
                }
            )
            record()
        result["persisted"] = []
        for identity in identities:
            detail = (await scope.get_detail(identity)).capture
            result["persisted"].append(
                {
                    "capture_id": identity.capture_id,
                    "title": detail.title,
                    "note": detail.freeform_note,
                    "status": detail.status,
                }
            )
        assert [c["status"] for c in result["persisted"]] == ["read", "saved"]
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
