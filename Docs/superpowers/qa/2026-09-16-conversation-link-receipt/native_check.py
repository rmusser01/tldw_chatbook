"""Qualify retained link receipts in the real app using private local data.

Workspace changes use the real registry API while Library remains mounted;
this does not qualify navigation through the Console workspace switcher.
"""

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
    from textual.widgets import Button, Static
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli

    probe_terminal()
    app = TldwCli()
    cid = str(uuid5(NAMESPACE_URL, "task-32388/conversation"))
    a, b = [str(uuid5(NAMESPACE_URL, "task-32388/" + label)) for label in ("a", "b")]
    body = "A private saved message for workspace receipt verification."
    result = {"pid": os.getpid(), "steps": []}

    def record():
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def seed():
        registry = app.workspace_registry_service
        registry.create_workspace(workspace_id=a, name="Receipt Alpha")
        registry.create_workspace(workspace_id=b, name="Receipt Beta")
        registry.set_active_workspace(a)
        assert (
            app.chachanotes_db.add_conversation({"id": cid, "title": "Receipt fixture"})
            == cid
        )
        mid = app.chachanotes_db.add_message(
            {"conversation_id": cid, "sender": "User", "content": body}
        )
        registry.link_membership(
            b, item_type="conversation", item_id=cid, title="Receipt fixture"
        )
        (root / "fixtures.json").write_text(
            json.dumps(
                {
                    "conversation_id": cid,
                    "message_id": mid,
                    "body": body,
                    "a": a,
                    "b": b,
                },
                indent=2,
            )
            + "\n"
        )

    def memberships():
        return {
            m.workspace_id
            for m in app.workspace_registry_service.get_item_memberships(
                item_type="conversation", item_id=cid
            )
        }

    def painted(widget):
        region = widget.region
        strips = list(app.screen._compositor.render_strips())
        return "\n".join(
            strips[y].crop(region.x, region.right).text
            for y in range(max(0, region.y), min(region.bottom, len(strips)))
        )

    async def tmux(*args):
        await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", socket, *args],
            check=True,
            capture_output=True,
            text=True,
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
            assert button.display and not button.disabled
            button.focus()
            await wait_for(lambda: app.screen.focused is button, selector)
            await pilot.press("enter")

        async def resize(width, height):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(lambda: tuple(app.size) == (width, height), "native resize")

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
            await press("#library-row-browse-conversations")
            await wait_for(
                lambda: (
                    screen._conversations_state.reader_state.loaded_id == cid
                    and screen._conversations_state.reader_state.complete
                ),
                "saved transcript",
            )

            async def activate(workspace_id, recompose):
                app.workspace_registry_service.set_active_workspace(workspace_id)
                screen._invalidate_library_workspace_depth_state()
                if recompose:
                    previous = screen.query_one("#library-conversation-reader")
                    screen.refresh(recompose=True)
                    await wait_for(
                        lambda: (
                            screen.query_one("#library-conversation-reader")
                            is not previous
                        ),
                        "reader recompose",
                    )
                else:
                    screen._sync_library_conversation_reader()
                await pilot.pause()

            for theme in ("textual-dark", "textual-light"):
                app.theme = theme
                for width, height in ((170, 48), (80, 24)):
                    await resize(width, height)
                    assert memberships() == {b}
                    await press("#library-conversation-link-workspace")
                    await wait_for(lambda: memberships() == {a, b}, "new Alpha link")
                    receipt = screen.query_one(
                        "#library-conversation-link-receipt", Static
                    )
                    undo = screen.query_one("#library-conversation-link-undo", Button)
                    assert receipt.display and undo.display
                    undo.focus()
                    await pilot.pause()
                    assert "Receipt Alpha" in painted(receipt)
                    assert "Undo link" in painted(undo)
                    await capture(f"linked-{theme}-{width}")

                    await activate(b, recompose=width == 80)
                    assert not screen.query_one(
                        "#library-conversation-link-receipt"
                    ).display
                    assert not screen.query_one(
                        "#library-conversation-link-undo"
                    ).display
                    assert not screen.query_one(
                        "#library-conversation-use-source", Button
                    ).disabled
                    assert memberships() == {a, b}
                    screen.query_one(
                        "#library-conversation-reader-read", Button
                    ).focus()
                    await pilot.pause()
                    await capture(f"other-workspace-{theme}-{width}")

                    await activate(a, recompose=width == 80)
                    receipt = screen.query_one(
                        "#library-conversation-link-receipt", Static
                    )
                    assert receipt.display and "Receipt Alpha" in str(
                        receipt.renderable
                    )
                    await press("#library-conversation-link-undo")
                    await wait_for(lambda: memberships() == {b}, "exact Alpha Undo")
                    assert not screen.query_one(
                        "#library-conversation-link-receipt"
                    ).display
                    result["steps"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "refresh": "recompose" if width == 80 else "retained",
                            "receipt_hidden_in_beta": True,
                            "receipt_restored_in_alpha": True,
                            "undo_removed_only_alpha": True,
                        }
                    )
                    record()
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
