"""Qualify exact Library Resume in a guarded native private profile.

Usage: native_check.py PROFILE TMUX_SOCKET SESSION
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
    socket, terminal = sys.argv[2:4]
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "2026-09-16-ingest-lifecycle/native_check.py"
        )
    )["validate_profile"](root)
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
    os.environ["XDG_DATA_HOME"] = str(root / "data")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    os.environ.pop("NO_COLOR", None)

    from textual import events
    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, Input
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    assert Path(app_module.__file__).resolve() == (
        Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    )
    probe_terminal()
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "steps": [],
    }
    fixtures = []
    draft = "Unrelated unfinished draft: keep [this] and café."

    def write(name, value):
        (evidence / name).write_text(json.dumps(value, indent=2, default=str) + "\n")

    def record():
        write("result.json", result)

    def seed():
        service = ChatConversationService(app.chachanotes_db)
        for label in ("DarkWide", "DarkCompact", "LightWide", "LightCompact"):
            cid = str(uuid5(NAMESPACE_URL, "task-32705/" + label))
            title = "Resume audit " + label
            assert (
                service.create_conversation(
                    id=cid, title=title, scope_type="global", state="in-progress"
                )
                == cid
            )
            messages = []
            for i, (role, text) in enumerate(
                (
                    ("user", label + " original question"),
                    ("assistant", label + " older selected answer"),
                    ("assistant", label + " newer off-path answer"),
                )
            ):
                mid = str(uuid5(NAMESPACE_URL, cid + "/" + str(i)))
                row = {
                    "id": mid,
                    "conversation_id": cid,
                    "sender": role,
                    "role": role,
                    "content": text,
                    "timestamp": f"2026-01-01T00:00:0{i}.000000+00:00",
                    "parent_message_id": messages[0]["id"] if i else None,
                }
                assert app.chachanotes_db.add_message(row) == mid
                messages.append(row)
            app.chachanotes_db.set_conversation_active_leaf(cid, messages[1]["id"])
            fixtures.append({"id": cid, "title": title, "messages": messages})
        write("fixtures.json", fixtures)

    def source_snapshot():
        return [
            {
                "conversation": app.chachanotes_db.get_conversation_by_id(f["id"]),
                "messages": app.chachanotes_db.get_messages_for_conversation(f["id"]),
            }
            for f in fixtures
        ]

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

        async def press(selector, text=None):
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
                    and (text or str(button.label)) in painted(button)
                ),
                "painted focus " + selector,
            )
            await pilot.press("enter")

        async def capture(name):
            await wait_for(lambda: not app.screen.query("Toast"), "notices expired")
            app.save_screenshot(name + ".svg", path=str(evidence))

        async def open_reader(fixture):
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: type(app.screen).__name__ == "LibraryScreen", "Library"
            )
            screen = app.screen
            if not screen.query("#library-conversations-filter"):
                await press("#library-row-browse-conversations")
            await wait_for(
                lambda: (
                    bool(screen.query("#library-conversations-filter"))
                    and not screen._conversations_state.loading
                ),
                "Conversations",
            )
            if not screen._conversations_state.reader_layout.items_open:
                await press("#library-conversations-items-grip")
            field = screen.query_one("#library-conversations-filter", Input)
            field.value = fixture["title"]
            field.focus()
            await pilot.press("enter")
            await wait_for(
                lambda: (
                    not screen._conversations_state.loading
                    and screen._conversations_state.total == 1
                    and any(
                        row.conversation_id == fixture["id"]
                        for row in screen.query(".library-conversation-row")
                    )
                ),
                "exact filtered row",
            )
            row = next(
                row
                for row in screen.query(".library-conversation-row")
                if row.conversation_id == fixture["id"]
            )
            row.focus()
            await wait_for(
                lambda: screen.focused is row and fixture["title"] in painted(row),
                "row painted",
            )
            await pilot.press("enter")
            await wait_for(
                lambda: (
                    screen._conversations_state.reader_state.loaded_id == fixture["id"]
                    and screen._conversations_state.reader_state.loaded_actions_eligible
                ),
                "exact reader ready",
            )
            if (
                app.size.width == 80
                and screen._conversations_state.reader_layout.items_open
            ):
                await press("#library-conversations-items-grip")

        def check_session(fixture, expected_id=None):
            matches = [
                s
                for s in store.sessions()
                if s.persisted_conversation_id == fixture["id"]
            ]
            assert len(matches) == 1
            resumed = matches[0]
            assert store.active_session_id == resumed.id
            if expected_id:
                assert resumed.id == expected_id
            visible = store.messages_for_session(resumed.id)
            assert [
                (m.persisted_message_id, m.content, m.role.value) for m in visible
            ] == [(m["id"], m["content"], m["role"]) for m in fixture["messages"][:2]]
            nodes = store.all_messages_for_session(resumed.id)
            assert len(nodes) == 3
            assert {m.persisted_message_id: m.content for m in nodes} == {
                m["id"]: m["content"] for m in fixture["messages"]
            }
            siblings, index, count = store.siblings_at(visible[-1].id)
            assert index == 0 and count == 2
            assert {m.persisted_message_id for m in siblings} == {
                m["id"] for m in fixture["messages"][1:]
            }
            assert store.session_draft(keeper) == draft
            assert console._console_composer_or_none().draft_text() == ""
            return resumed.id

        async def resume(fixture, expected_id=None):
            channel = HandoffChannel.CONSOLE_CONVERSATION_RESUME
            handoffs = app.pending_handoffs
            revision = handoffs._slot_for(channel).revision + 1
            await press("#library-conversation-open-console")
            await wait_for(
                lambda: (
                    app.screen is console
                    and any(
                        s.id == store.active_session_id
                        and s.persisted_conversation_id == fixture["id"]
                        for s in store.sessions()
                    )
                ),
                "exact Console activation",
            )
            await wait_for(
                lambda: (
                    handoffs._slot_for(channel).revision == revision
                    and handoffs.exact_revision_status(channel, revision) == "settled"
                ),
                "resume acknowledged",
            )
            await wait_for(
                lambda: (
                    fixture["messages"][1]["content"]
                    in " ".join(
                        " ".join(
                            strip.text for strip in console._compositor.render_strips()
                        ).split()
                    )
                ),
                "older selected answer painted",
            )
            return check_session(fixture, expected_id)

        async def keeper_tab():
            await press("#console-session-tab-" + keeper)
            await wait_for(
                lambda: (
                    store.active_session_id == keeper
                    and console._console_composer_or_none().draft_text() == draft
                ),
                "unrelated draft visible",
            )
            composer = console._console_composer_or_none()
            composer.focus()
            await wait_for(
                lambda: all(
                    part in painted(composer)
                    for part in (
                        "Unrelated unfinished",
                        "draft: keep",
                        "[this]",
                        "café.",
                    )
                ),
                "draft painted",
            )

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            result["exclusive_profile"] = True
            await wait_for(
                lambda: (
                    getattr(app, "_ui_ready", False)
                    and type(app.screen).__name__ == "ChatScreen"
                ),
                "Console ready",
            )
            console = app.screen
            store = console._console_chat_store
            keeper = store.active_session_id
            composer = console._console_composer_or_none()
            composer.focus()
            app.post_message(events.Paste(draft))
            await wait_for(
                lambda: composer.draft_text() == draft,
                "populated keeper draft",
            )
            result["keeper_session_id"] = keeper
            await asyncio.to_thread(seed)
            before = source_snapshot()
            write("source-before.json", before)
            matrix = [
                (theme, size)
                for theme in ("textual-dark", "textual-light")
                for size in ((170, 48), (80, 24))
            ]
            for fixture, (theme, (width, height)) in zip(fixtures, matrix, strict=True):
                app.theme = theme
                await tmux(
                    "resize-window", "-t", terminal, "-x", str(width), "-y", str(height)
                )
                await wait_for(
                    lambda width=width, height=height: (
                        tuple(app.size) == (width, height)
                    ),
                    "matrix size",
                )
                assert not any(
                    s.persisted_conversation_id == fixture["id"]
                    for s in store.sessions()
                )
                previous_ids = [s.id for s in store.sessions()]
                await open_reader(fixture)
                button = app.screen.query_one(
                    "#library-conversation-open-console", Button
                )
                button.focus()
                await wait_for(
                    lambda button=button: "Resume conversation" in painted(button),
                    "Resume painted",
                )
                name = theme + "-" + str(width)
                await capture(name + "-reader")
                resumed_id = await resume(fixture)
                assert [s.id for s in store.sessions()] == previous_ids + [resumed_id]
                await capture(name + "-resumed")
                await keeper_tab()
                await capture(name + "-draft")
                await open_reader(fixture)
                assert await resume(fixture, resumed_id) == resumed_id
                assert [s.id for s in store.sessions()] == previous_ids + [resumed_id]
                await keeper_tab()
                result["steps"].append(
                    {
                        "theme": theme,
                        "size": [width, height],
                        "conversation_id": fixture["id"],
                        "session_id": resumed_id,
                        "cold_original_branch": True,
                        "off_path_sibling_retained": True,
                        "warm_session_reused": True,
                        "unrelated_populated_draft_retained_and_painted": True,
                    }
                )
                record()
            after = source_snapshot()
            write("source-after.json", after)
            assert after == before
            result["source_records_unchanged"] = True
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain evidence before normal exit
            result["passed"] = False
            result["error"] = traceback.format_exc()
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", terminal, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
