"""Exercise archive/recovery controls with real SQLite in two native processes.

Usage: native_check.py PROFILE TMUX_SOCKET SESSION exercise|restart
"""

import asyncio
import hashlib
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
    socket, session, phase = sys.argv[2:5]
    if phase not in {"exercise", "restart"}:
        raise ValueError("Choose exercise or restart")
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "2026-09-16-ingest-lifecycle/native_check.py"
        )
    )["validate_profile"](root)
    evidence = root / phase
    evidence.mkdir(exist_ok=False)
    os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
    os.environ["XDG_DATA_HOME"] = str(root / "data")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    os.environ.pop("NO_COLOR", None)

    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, Input
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli

    assert Path(app_module.__file__).resolve() == (
        Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    )
    probe_terminal()
    app = TldwCli()
    result = {
        "phase": phase,
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "steps": [],
    }
    fixtures_path = root / "fixtures.json"
    fixtures = json.loads(fixtures_path.read_text()) if fixtures_path.exists() else []

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def seed():
        assert not fixtures
        for name in ("Alpha", "Beta"):
            cid = str(uuid5(NAMESPACE_URL, "task-32703/" + name))
            title = "Archive audit " + name
            assert (
                app.chachanotes_db.add_conversation({"id": cid, "title": title}) == cid
            )
            messages = []
            for sender in ("User", "Assistant"):
                body = (
                    f"{name} saved {sender} message: archive must preserve this text."
                )
                mid = app.chachanotes_db.add_message(
                    {"conversation_id": cid, "sender": sender, "content": body}
                )
                messages.append({"id": mid, "content": body})
            fixtures.append({"id": cid, "title": title, "messages": messages})
        fixtures_path.write_text(json.dumps(fixtures, indent=2) + "\n")

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
            capture_output=True,
            text=True,
        )

    async def journey(pilot):
        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 45
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(f"{label}; focus={app.screen.focused!r}")

        async def press(selector):
            await wait_for(
                lambda: (
                    bool(app.screen.query(selector))
                    and not app.screen.query_one(selector, Button).disabled
                ),
                "enabled " + selector,
            )
            button = app.screen.query_one(selector, Button)
            button.focus()
            await wait_for(
                lambda: (
                    app.screen.focused is button
                    and str(button.label) in painted(button)
                ),
                "painted focus " + selector,
            )
            await pilot.press("enter")

        async def pane(items):
            if screen._conversations_state.reader_layout.items_open != items:
                await press("#library-conversations-items-grip")
                await wait_for(
                    lambda: (
                        screen._conversations_state.reader_layout.items_open == items
                    ),
                    "Items visibility",
                )

        async def settled(total):
            await wait_for(
                lambda: (
                    app.screen is screen
                    and not screen._conversations_state.loading
                    and not screen._conversation_recovery().busy
                    and screen._conversations_state.total == total
                ),
                f"settled count {total}",
            )
            assert screen._conversations_state.requested_query == "Archive audit"

        def archived(identity):
            return bool(app.chachanotes_db.get_conversation_by_id(identity)["archived"])

        async def scope(name, total):
            await pane(True)
            await press("#library-conversations-scope-" + name)
            await settled(total)
            assert screen._conversation_recovery().scope == name

        async def open_alpha():
            await pane(True)
            await wait_for(
                lambda: any(
                    row.conversation_id == alpha
                    for row in screen.query(".library-conversation-row")
                ),
                "Alpha row",
            )
            row = next(
                row
                for row in screen.query(".library-conversation-row")
                if row.conversation_id == alpha
            )
            row.focus()
            await wait_for(
                lambda: screen.focused is row and "Alpha" in painted(row),
                "Alpha painted",
            )
            await pilot.press("enter")
            await wait_for(
                lambda: (
                    screen._conversations_state.reader_state.loaded_id == alpha
                    and screen._conversations_state.reader_state.loaded_actions_eligible
                ),
                "Alpha reader",
            )
            if app.size.width == 80:
                await pane(False)

        async def mutate(
            action, expected_archived, total, *, cancel=False, capture_name=None
        ):
            before_version = app.chachanotes_db.get_conversation_by_id(alpha)["version"]
            await press("#library-conversation-" + action)
            await wait_for(
                lambda: (
                    app.screen is not screen
                    and bool(app.screen.query("#confirm-button"))
                ),
                "confirmation",
            )
            if capture_name:
                await capture(capture_name)
            if cancel:
                await pilot.press("escape")
            else:
                await press("#confirm-button")
            await settled(total)
            assert archived(alpha) is expected_archived
            assert not archived(beta)
            assert screen._conversations_state.reader_state.loaded_id == alpha
            after_version = app.chachanotes_db.get_conversation_by_id(alpha)["version"]
            if cancel:
                assert after_version == before_version
            else:
                assert after_version == before_version + 1
                assert screen._conversation_recovery().receipt_versions == {
                    alpha: after_version
                }

        async def capture(name):
            await wait_for(lambda: not app.screen.query("Toast"), "notices expired")
            app.save_screenshot(name + ".svg", path=str(evidence))

        try:
            assert app._instance_lock_status.acquired
            result["exclusive_profile"] = True
            await wait_for(
                lambda: (
                    getattr(app, "_ui_ready", False)
                    and type(app.screen).__name__ == "ChatScreen"
                ),
                "Console ready",
            )
            assert type(app._driver).__name__ == "LinuxDriver"
            store = app.screen._console_chat_store
            active_session = store.active_session_id
            session_ids = [s.id for s in store.sessions()]
            draft = store.session_draft(active_session)
            await tmux("resize-window", "-t", session, "-x", "170", "-y", "48")
            await wait_for(lambda: tuple(app.size) == (170, 48), "initial size")
            if phase == "exercise":
                await asyncio.to_thread(seed)
            alpha, beta = (f["id"] for f in fixtures)
            assert archived(alpha) is (phase == "restart")
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: type(app.screen).__name__ == "LibraryScreen", "Library"
            )
            screen = app.screen
            await press("#library-row-browse-conversations")
            await wait_for(
                lambda: (
                    not screen._conversations_state.loading
                    and bool(screen.query("#library-conversations-filter"))
                ),
                "conversation list",
            )
            await pane(True)
            field = screen.query_one("#library-conversations-filter", Input)
            field.value = "Archive audit"
            field.focus()
            await pilot.press("enter")
            await settled(2 if phase == "exercise" else 1)
            if phase == "exercise":
                for theme in ("textual-dark", "textual-light"):
                    app.theme = theme
                    for width, height in ((170, 48), (80, 24)):
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
                            lambda width=width, height=height: (
                                tuple(app.size) == (width, height)
                            ),
                            "matrix resize",
                        )
                        await scope("active", 2)
                        await open_alpha()
                        await mutate(
                            "archive",
                            False,
                            2,
                            cancel=True,
                            capture_name=f"confirm-{theme}-{width}",
                        )
                        await mutate("archive", True, 1)
                        await pane(True)
                        undo = screen.query_one("#library-conversations-undo", Button)
                        undo.focus()
                        await wait_for(
                            lambda undo=undo: "Undo" in painted(undo), "Undo visible"
                        )
                        await capture(f"receipt-{theme}-{width}")
                        await press("#library-conversations-undo")
                        await settled(2)
                        assert not archived(alpha)
                        await open_alpha()
                        await mutate("archive", True, 1)
                        await pane(True)
                        await press("#library-conversations-view-archived")
                        await settled(1)
                        await open_alpha()
                        await mutate("restore", False, 0)
                        await pane(True)
                        await press("#library-conversations-undo")
                        await settled(1)
                        assert archived(alpha)
                        await open_alpha()
                        await mutate("restore", False, 0)
                        result["steps"].append(
                            {
                                "theme": theme,
                                "size": [width, height],
                                "cancel_archive_undo_restore_undo_passed": True,
                                "query_retained": True,
                            }
                        )
                await tmux("resize-window", "-t", session, "-x", "170", "-y", "48")
                await wait_for(lambda: tuple(app.size) == (170, 48), "final size")
                await scope("active", 2)
                await open_alpha()
                await mutate("archive", True, 1)
                result["left_archived_for_restart"] = alpha
            else:
                await scope("archived", 1)
                await open_alpha()
                await capture("archived-after-restart")
                await mutate("restore", False, 0)
                await scope("active", 2)
                await open_alpha()
                await capture("restored-after-restart")
                result["restored_after_restart"] = alpha
            assert store.active_session_id == active_session
            assert [s.id for s in store.sessions()] == session_ids
            assert store.session_draft(active_session) == draft
            result["console_context_unchanged"] = True
            for fixture in fixtures:
                actual = app.chachanotes_db.get_messages_for_conversation(fixture["id"])
                assert {m["id"]: m["content"] for m in actual} == {
                    m["id"]: m["content"] for m in fixture["messages"]
                }
            result["messages_unchanged"] = True
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve native failure before normal shutdown
            result["passed"] = False
            result["error"] = traceback.format_exc()
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
