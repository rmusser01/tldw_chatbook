"""Verify Conversations editing and reading with real private local data."""

import asyncio
import json
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


def main():
    root = Path(sys.argv[1]).resolve()
    socket, session = sys.argv[2:4]
    guard = runpy.run_path(
        str(Path(__file__).parents[1] / "2026-09-16-ingest-lifecycle/native_check.py")
    )["validate_profile"]
    guard(root)
    os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
    os.environ["XDG_DATA_HOME"] = str(root / "data")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    os.environ.pop("NO_COLOR", None)

    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, Input
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli

    probe_terminal()
    app = TldwCli()
    result = {"pid": os.getpid(), "steps": []}
    fixtures = []

    def record():
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def seed():
        for name, count in (("Alpha planning", 24), ("Beta review", 2)):
            cid = str(uuid5(NAMESPACE_URL, "task-32701/" + name))
            assert (
                app.chachanotes_db.add_conversation({"id": cid, "title": name}) == cid
            )
            messages = []
            for index in range(1, count + 1):
                body = f"Saved message {index}. {name}: local continuity fixture."
                mid = app.chachanotes_db.add_message(
                    {"conversation_id": cid, "sender": "User", "content": body}
                )
                assert mid
                messages.append({"id": mid, "content": body})
            fixtures.append({"id": cid, "title": name, "messages": messages})
        (root / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")

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
            deadline = asyncio.get_running_loop().time() + 30
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(f"{label}: {app.screen.focused!r}")

        async def press(selector):
            button = app.screen.query_one(selector, Button)
            assert not button.disabled
            button.focus()
            await pilot.pause()
            await pilot.press("enter")

        async def resize(width, height):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(lambda: tuple(app.size) == (width, height), "native resize")
            await pilot.pause()

        async def capture(name):
            await wait_for(lambda: not app.screen.query("Toast"), "notices expired")
            app.save_screenshot(name + ".svg", path=str(root))

        try:
            assert app._instance_lock_status.acquired
            result["exclusive_profile"] = True
            await wait_for(lambda: getattr(app, "_ui_ready", False), "ready")
            assert type(app._driver).__name__ == "LinuxDriver"
            result["driver"] = type(app._driver).__name__
            await resize(170, 48)
            await asyncio.to_thread(seed)
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: type(app.screen).__name__ == "LibraryScreen", "Library"
            )
            screen = app.screen
            await wait_for(
                lambda: bool(screen.query("#library-row-browse-conversations")),
                "conversation navigation",
            )
            await press("#library-row-browse-conversations")
            await wait_for(
                lambda: len(screen.query(".library-conversation-row")) == 2,
                "saved conversations",
            )
            alpha_id = fixtures[0]["id"]
            alpha = next(
                row
                for row in screen.query(".library-conversation-row")
                if row.conversation_id == alpha_id
            )
            alpha.focus()
            await wait_for(lambda: screen.focused is alpha, "Alpha list focus")
            await pilot.press("enter")
            await wait_for(
                lambda: (
                    screen._conversations_state.reader_state.loaded_id == alpha_id
                    and screen._conversations_state.reader_state.complete
                ),
                "Alpha transcript",
            )
            reader = screen.query_one("#library-conversation-reader")
            for theme in ("textual-dark", "textual-light"):
                app.theme = theme
                await resize(170, 48)
                for selector, label in (
                    ("#library-conversations-filter", "filter"),
                    ("#library-conversation-reader-find", "find"),
                ):
                    field = screen.query_one(selector, Input)
                    field.focus()
                    field.value = "unsubmitted query"
                    await pilot.press("end", "shift+left", "shift+left")
                    selection = field.selection
                    for width, height in ((80, 24), (170, 48), (170, 24)):
                        await resize(width, height)
                        assert screen.query_one(selector) is field
                        assert (
                            field.value == "unsubmitted query"
                            and field.selection == selection
                        )
                        if (
                            label == "filter"
                            and not screen._conversations_state.reader_layout.items_open
                        ):
                            assert screen.focused is screen.query_one(
                                "#library-conversations-items-grip"
                            )
                            await pilot.press("enter")
                            await pilot.pause()
                        assert screen.focused is field
                        assert "query" in painted(field)
                        if width == 80 or (label == "find" and height == 48):
                            await capture(f"{label}-{theme}-{width}")
                    await pilot.press("x")
                    assert field.value == "unsubmitted quex"
                    field.value = ""
                    result["steps"].append(
                        {
                            "theme": theme,
                            "editor": label,
                            "draft_selection_focus_and_typing_retained": True,
                        }
                    )
                    record()
                find = screen.query_one("#library-conversation-reader-find", Input)
                find.focus()
                find.value = "Saved message 20."
                await pilot.press("enter")
                mid = fixtures[0]["messages"][19]["id"]
                await wait_for(
                    lambda mid=mid: any(
                        m.message_id == mid
                        for m in screen._conversations_state.reader_state.find_matches
                    ),
                    "find result",
                )
                match = next(
                    row
                    for row in screen.query(".library-conversation-reader-message")
                    if row.message_id == mid
                )
                await wait_for(
                    lambda match=match: screen.focused is match, "found message focus"
                )
                for width, height in ((80, 24), (170, 48), (170, 24)):
                    await resize(width, height)
                    assert screen.query_one("#library-conversation-reader") is reader
                    assert screen.focused is match
                    await wait_for(
                        lambda match=match: "Saved message 20." in painted(match),
                        "visible found message",
                    )
                    if width == 80:
                        await capture(f"match-{theme}-80")
                await resize(80, 24)
                info = screen.query_one("#library-conversation-reader-info", Button)
                info.focus()
                await pilot.pause()
                assert screen.focused is info
                await pilot.press("enter")
                assert screen._conversations_state.reader_state.mode == "info"
                await press("#library-conversation-reader-read")
                for width, height in ((170, 48), (80, 24)):
                    await resize(width, height)
                    if not screen._conversations_state.reader_layout.items_open:
                        await press("#library-conversations-items-grip")
                    field = screen.query_one("#library-conversations-filter", Input)
                    field.focus()
                    field.value = "nothing-matches"
                    await pilot.press("enter")
                    await wait_for(
                        lambda: bool(
                            screen.query("#library-conversations-empty-clear-filter")
                        ),
                        "empty results",
                    )
                    clear = screen.query_one(
                        "#library-conversations-empty-clear-filter", Button
                    )
                    clear.focus()
                    await pilot.pause()
                    assert "Clear filter" in painted(clear)
                    if width == 80:
                        await capture(f"empty-filter-{theme}-80")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: len(screen.query(".library-conversation-row")) == 2,
                        "clear results",
                    )
                    await wait_for(
                        lambda: (
                            screen.focused
                            is screen.query_one("#library-conversations-filter")
                        ),
                        "filter return",
                    )
                    await pilot.press("b")
                    field = screen.query_one("#library-conversations-filter", Input)
                    assert field.value == "b" and "b" in painted(field)
                    field.value = ""
                await resize(170, 48)
                result["steps"].append(
                    {
                        "theme": theme,
                        "find_visible_through_resize": True,
                        "new_focus_respected": True,
                        "empty_filter_recovered_at": [[170, 48], [80, 24]],
                    }
                )
                record()
                # Clearing the list filter can select Beta; choose Alpha for the next pass.
                alpha = next(
                    row
                    for row in screen.query(".library-conversation-row")
                    if row.conversation_id == alpha_id
                )
                alpha.focus()
                await pilot.press("enter")
                await wait_for(
                    lambda: (
                        screen._conversations_state.reader_state.loaded_id == alpha_id
                        and screen._conversations_state.reader_state.complete
                    ),
                    "Alpha restored",
                )
            result["passed"] = True
            record()
            await tmux("send-keys", "-t", session, "C-q")
        except Exception:  # noqa: BLE001 - preserve evidence before normal shutdown
            result["passed"] = False
            result["error"] = traceback.format_exc()
            app.save_screenshot("failed-state.svg", path=str(root))
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
