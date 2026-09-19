"""Qualify Library source excerpts through send-time capture in a guarded native private profile.

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
from dataclasses import asdict
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
    from textual.widgets import Button, Input, Static
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        capture_console_staged_evidence_for_chat,
    )
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
    media_fixtures = []
    draft = "Unrelated unfinished draft: keep [this] and café."

    def write(name, value):
        (evidence / name).write_text(json.dumps(value, indent=2, default=str) + "\n")

    def record():
        write("result.json", result)

    def seed():
        service = ChatConversationService(app.chachanotes_db)
        for label in (
            "DarkWide",
            "DarkCompact",
            "LightWide",
            "LightCompact",
            "Foreign",
        ):
            cid = str(uuid5(NAMESPACE_URL, "task-2376/" + label))
            title = "Source audit " + label
            assert (
                service.create_conversation(
                    id=cid, title=title, scope_type="global", state="in-progress"
                )
                == cid
            )
            messages = []
            for i, role in enumerate(("user", "assistant")):
                mid = str(uuid5(NAMESPACE_URL, cid + "/" + str(i)))
                row = {
                    "id": mid,
                    "conversation_id": cid,
                    "sender": role,
                    "role": role,
                    "content": label + " " + role + " source body: exact [text], café.",
                    "timestamp": f"2026-01-01T00:00:0{i}.000000+00:00",
                    "parent_message_id": messages[0]["id"] if i else None,
                }
                assert app.chachanotes_db.add_message(row) == mid
                messages.append(row)
            fixtures.append({"id": cid, "title": title, "messages": messages})
        registry = app.workspace_registry_service
        registry.create_workspace(
            workspace_id="source-other-workspace", name="Other source workspace"
        )
        registry.link_membership(
            "source-other-workspace",
            item_type="conversation",
            item_id=fixtures[-1]["id"],
            title=fixtures[-1]["title"],
        )
        for label in ("DarkWide", "DarkCompact", "LightWide", "LightCompact"):
            content = (label + " stored media: exact [text], café.\n") * 30
            media_id, _, _ = app.media_db.add_media_with_keywords(
                title="Excerpt media " + label,
                media_type="document",
                content=content,
                keywords=["excerpt-audit"],
            )
            assert media_id is not None
            media_fixtures.append(
                {"id": media_id, "title": "Excerpt media " + label, "content": content}
            )
            registry.link_membership(
                registry.get_active_workspace().workspace_id,
                item_type="media",
                item_id=str(media_id),
                title="Excerpt media " + label,
            )
        write("fixtures.json", fixtures)
        write("media-fixtures.json", media_fixtures)
        write(
            "media-before.json",
            [app.media_db.get_media_by_id(f["id"]) for f in media_fixtures],
        )

    def memberships(fixture):
        return [
            asdict(item)
            for item in app.workspace_registry_service.get_item_memberships(
                item_type="conversation", item_id=fixture["id"]
            )
        ]

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
                if (
                    screen.query("#library-media-back")
                    and not screen._media_state.reader_layout.library_open
                ):
                    await press("#library-browse-library-grip")
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
            await wait_for(lambda: screen.focused is field, "filter focused")
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

        async def return_reader(fixture):
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: type(app.screen).__name__ == "LibraryScreen", "Library returned"
            )
            screen = app.screen
            await wait_for(
                lambda: (
                    not screen._conversations_state.loading
                    and screen._conversations_state.reader_state.loaded_id
                    == fixture["id"]
                    and screen._conversations_state.reader_state.loaded_actions_eligible
                ),
                "retained source reader",
            )
            return screen

        async def check_context(fixture):
            launch = console._pending_console_launch_context
            assert launch is not None
            assert launch.source == "library" and launch.title == fixture["title"]
            payload = launch.payload
            assert payload["source_id"] == fixture["id"]
            assert payload["item_type"] == "conversation"
            assert payload["metadata"]["conversation_id"] == fixture["id"]
            assert payload["runtime_backend"] == "local"
            refs = payload["evidence_bundle"]["references"]
            assert len(refs) == 1 and refs[0]["source_id"] == fixture["id"]
            assert refs[0]["snippet"] == payload["snippet"]
            captured = await capture_console_staged_evidence_for_chat(
                app, launch, user_message="What did this source say?"
            )
            assert captured.context and "CHAT HISTORY" in captured.context
            for message in fixture["messages"]:
                assert message["content"] in refs[0]["snippet"]
                assert message["content"] in captured.context
            assert "Conversation staged:" not in captured.context
            write(
                f"capture-{fixture['title'].split()[-1]}-conversation.json",
                {
                    "payload": payload,
                    "context": captured.context,
                },
            )
            assert store.active_session_id == keeper
            assert [s.id for s in store.sessions()] == session_ids
            assert store.session_draft(keeper) == draft
            assert console._console_composer_or_none().draft_text() == draft
            assert not store.messages_for_session(keeper)
            return dict(payload)

        async def use_source(fixture):
            channel = HandoffChannel.CHAT
            handoffs = app.pending_handoffs
            revision = handoffs._slot_for(channel).revision + 1
            await press("#library-conversation-use-source")
            await wait_for(
                lambda: (
                    app.screen is console
                    and console._pending_console_launch_context is not None
                    and console._pending_console_launch_context.payload.get("source_id")
                    == fixture["id"]
                ),
                "exact source staged",
            )
            await wait_for(
                lambda: (
                    handoffs._slot_for(channel).revision == revision
                    and handoffs.exact_revision_status(channel, revision) == "settled"
                ),
                "source handoff settled",
            )
            await wait_for(
                lambda: bool(console.query("#console-unstage-evidence")),
                "Unstage available",
            )
            button = console.query_one("#console-unstage-evidence", Button)
            button.focus()
            await wait_for(
                lambda: (
                    str(button.label) in painted(button)
                    and fixture["title"]
                    in painted(console.query_one("#console-staged-evidence-row-0"))
                ),
                "source and Unstage painted",
            )
            return await check_context(fixture)

        async def media_journey(fixture, name):
            # Start on the retained conversation reader after Undo. Open its
            # real Nav disclosure before selecting a different destination.
            if not app.screen._conversations_state.reader_layout.library_open:
                await press("#library-conversations-library-grip")
            await press("#library-row-browse-media")
            screen = app.screen
            await wait_for(
                lambda: bool(screen.query("#library-media-filter")), "Media filter"
            )
            if not screen._media_state.reader_layout.items_open:
                await press("#library-browse-items-grip")
            field = screen.query_one("#library-media-filter", Input)
            field.value = fixture["title"]
            field.focus()
            await wait_for(lambda: screen.focused is field, "filter focused")
            await pilot.press("enter")
            await wait_for(
                lambda: any(
                    str(row.media_id) == f"local:media:{fixture['id']}"
                    for row in screen.query(".library-media-row")
                ),
                "exact media row",
            )
            row = next(
                row
                for row in screen.query(".library-media-row")
                if str(row.media_id) == f"local:media:{fixture['id']}"
            )
            row.focus()
            await wait_for(lambda: screen.focused is row, "media row focus")
            await pilot.press("enter")
            await wait_for(
                lambda: (
                    screen._media_state.reader_session.loaded_backing_id
                    == fixture["id"]
                    and screen._media_state.reader_session.pending_request is None
                ),
                "media detail ready",
            )
            source = screen._selected_media_handoff_payload()
            assert source is not None and source.body_truncated
            assert source.source_id == str(fixture["id"])
            await press("#library-media-use-in-chat")
            await wait_for(
                lambda: (
                    app.screen is console
                    and console._pending_console_launch_context is not None
                    and console._pending_console_launch_context.payload.get("source_id")
                    == source.source_id
                ),
                "media staged",
            )
            launch = console._pending_console_launch_context
            captured = await capture_console_staged_evidence_for_chat(
                app, launch, user_message="What does this media say?"
            )
            snippet = launch.payload["evidence_bundle"]["references"][0]["snippet"]
            assert fixture["content"][:500] in snippet
            assert captured.context and fixture["content"][:500] in captured.context
            assert fixture["content"] not in captured.context
            assert "Media staged:" not in captured.context
            assert store.active_session_id == keeper
            assert [s.id for s in store.sessions()] == session_ids
            assert store.session_draft(keeper) == draft
            assert console._console_composer_or_none().draft_text() == draft
            assert not store.messages_for_session(keeper)
            write(
                "capture-" + name + "-media.json",
                {"payload": launch.payload, "context": captured.context},
            )
            await capture(name + "-media-staged")
            await press("#console-unstage-evidence")
            await wait_for(
                lambda: console._pending_console_launch_context is None,
                "media unstaged",
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
            session_ids = [s.id for s in store.sessions()]
            composer = console._console_composer_or_none()
            composer.focus()
            app.post_message(events.Paste(draft))
            await wait_for(lambda: composer.draft_text() == draft, "populated draft")
            registry = app.workspace_registry_service
            workspace = registry.get_active_workspace()
            assert workspace is not None
            result["keeper_session_id"] = keeper
            result["workspace_id"] = workspace.workspace_id
            await asyncio.to_thread(seed)
            before = source_snapshot()
            foreign_before = memberships(fixtures[-1])
            write("source-before.json", before)
            write("foreign-membership.json", foreign_before)
            matrix = [
                (theme, size)
                for theme in ("textual-dark", "textual-light")
                for size in ((170, 48), (80, 24))
            ]
            for fixture, (theme, (width, height)) in zip(
                fixtures[:4], matrix, strict=True
            ):
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
                assert memberships(fixture) == []
                await open_reader(fixture)
                screen = app.screen
                notice = screen.query_one(
                    "#library-conversation-open-console-blocked", Static
                )
                assert "adds it to the active workspace first" in str(notice.render())
                button = screen.query_one("#library-conversation-use-source", Button)
                button.focus()
                await wait_for(
                    lambda button=button: "Use as source" in painted(button),
                    "source action painted",
                )
                name = theme + "-" + str(width)
                await capture(name + "-reader")
                first_payload = await use_source(fixture)
                linked = memberships(fixture)
                assert (
                    len(linked) == 1
                    and linked[0]["workspace_id"] == workspace.workspace_id
                )
                await capture(name + "-staged")
                screen = await return_reader(fixture)
                await wait_for(
                    lambda screen=screen: (
                        bool(screen.query("#library-conversation-link-undo"))
                        and screen.query_one("#library-conversation-link-undo").display
                    ),
                    "link receipt retained",
                )
                assert (
                    screen._conversations_state.reader_loaded_metadata[
                        "_workspace_link_receipt_id"
                    ]
                    == workspace.workspace_id
                )
                depth = screen._library_workspace_depth_state()
                assert any(
                    row.item_id == fixtures[-1]["id"]
                    and not row.active_context_eligible
                    for row in depth.source_rows
                )
                assert not depth.context_handoff_enabled
                second_payload = await use_source(fixture)
                assert second_payload == first_payload
                assert memberships(fixture) == linked
                await press("#console-unstage-evidence")
                await wait_for(
                    lambda: (
                        console._pending_console_launch_context is None
                        and not console.query("#console-staged-evidence-row-0")
                    ),
                    "source unstaged",
                )
                assert store.active_session_id == keeper
                assert console._console_composer_or_none().draft_text() == draft
                screen = await return_reader(fixture)
                undo = screen.query_one("#library-conversation-link-undo", Button)
                undo.focus()
                await wait_for(
                    lambda undo=undo: str(undo.label) in painted(undo),
                    "Undo link painted",
                )
                await capture(name + "-linked")
                await press("#library-conversation-link-undo")
                await wait_for(
                    lambda fixture=fixture: memberships(fixture) == [], "link undone"
                )
                assert memberships(fixtures[-1]) == foreign_before
                assert "adds it to the active workspace first" in str(
                    screen.query_one(
                        "#library-conversation-open-console-blocked", Static
                    ).render()
                )
                assert [s.id for s in store.sessions()] == session_ids
                assert store.session_draft(keeper) == draft
                media = media_fixtures[len(result["steps"])]
                await media_journey(media, name)
                result["steps"].append(
                    {
                        "theme": theme,
                        "size": [width, height],
                        "conversation_id": fixture["id"],
                        "linked_membership": linked[0],
                        "unlinked_then_existing_source_staged": True,
                        "no_duplicate_membership_or_session": True,
                        "populated_draft_preserved": True,
                        "unstage_and_undo_link_passed": True,
                        "foreign_membership_unchanged": True,
                        "media_id": media["id"],
                        "actual_media_excerpt_captured": True,
                        "evidence_snippet": first_payload["snippet"],
                        "actual_conversation_excerpt_captured": True,
                    }
                )
                record()
            media_after = [
                app.media_db.get_media_by_id(f["id"]) for f in media_fixtures
            ]
            write("media-after.json", media_after)
            assert json.loads(
                (evidence / "media-before.json").read_text()
            ) == json.loads((evidence / "media-after.json").read_text())
            after = source_snapshot()
            write("source-after.json", after)
            assert before == after
            assert not store.messages_for_session(keeper)
            result["source_records_unchanged"] = True
            result["no_send_or_new_messages"] = True
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
