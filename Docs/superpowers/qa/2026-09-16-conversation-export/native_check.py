"""Verify real Conversations ZIP exports in a disposable native profile.

Usage: native_check.py PROFILE TMUX_SOCKET SESSION
"""

import asyncio
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


def main():
    root = Path(sys.argv[1]).resolve()
    socket, session = sys.argv[2:4]
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "2026-09-16-ingest-lifecycle/native_check.py"
        )
    )["validate_profile"](root)
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    exports = root / "exports"
    exports.mkdir(exist_ok=False)
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
    from tldw_chatbook.Third_Party.textual_fspicker import FileSave
    from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileNameInput
    from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation

    assert Path(app_module.__file__).resolve() == (
        Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    )
    probe_terminal()
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "steps": [],
        "artifacts": [],
    }
    fixtures = []

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def seed():
        for name in ("Alpha", "Beta", "Gamma"):
            cid = str(uuid5(NAMESPACE_URL, "task-32704/" + name))
            title = "Export audit " + name
            assert (
                app.chachanotes_db.add_conversation({"id": cid, "title": title}) == cid
            )
            messages = []
            for sender in ("User", "Assistant"):
                body = f"{name} {sender}: exact [brackets], café and newline.\nSecond line."
                mid = app.chachanotes_db.add_message(
                    {"conversation_id": cid, "sender": sender, "content": body}
                )
                messages.append({"id": mid, "content": body})
            if name == "Gamma":
                version = app.chachanotes_db.get_conversation_by_id(cid)["version"]
                changed = app.chachanotes_db.set_conversations_archived(
                    [cid], archived=True, expected_versions={cid: version}
                )
                assert changed["changed"] == {cid: version + 1}
            fixtures.append({"id": cid, "title": title, "messages": messages})
        note = app.chachanotes_db.add_note(
            title="Unrelated export audit note",
            content="Must not enter a conversation bundle.",
        )
        (evidence / "fixtures.json").write_text(
            json.dumps({"conversations": fixtures, "unrelated_note_id": note}, indent=2)
            + "\n"
        )

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

    def inspect_zip(path, expected_ids, label):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            manifest = json.loads(archive.read("manifest.json"))
            items = manifest["content_items"]
            assert {item["id"] for item in items} == set(expected_ids), items
            assert all(item["type"] == "conversation" for item in items), items
            assert not any(
                name.startswith("content/notes/") for name in archive.namelist()
            )
            payloads = {}
            for item in items:
                payload = json.loads(archive.read(item["file_path"]))
                fixture = next(f for f in fixtures if f["id"] == item["id"])
                assert payload["id"] == fixture["id"]
                assert payload["name"] == fixture["title"]
                assert {m["id"]: m["content"] for m in payload["messages"]} == {
                    m["id"]: m["content"] for m in fixture["messages"]
                }
                payloads[item["id"]] = payload
        copy = evidence / (label + ".zip")
        shutil.copyfile(path, copy)
        artifact = {
            "label": label,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "item_count": len(items),
            "conversation_ids": sorted(expected_ids),
            "exact_messages": True,
            "unrelated_notes_absent": True,
        }
        result["artifacts"].append(artifact)
        (evidence / (label + "-contents.json")).write_text(
            json.dumps({"manifest": manifest, "conversations": payloads}, indent=2)
            + "\n"
        )
        return artifact

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

        async def press(selector, visible_label=None):
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
                    and (visible_label or str(button.label)) in painted(button)
                ),
                "painted focus " + selector,
            )
            await pilot.press("enter")

        async def capture(name):
            await wait_for(lambda: not app.screen.query("Toast"), "notices expired")
            app.save_screenshot(name + ".svg", path=str(evidence))

        async def filtered_conversations():
            await wait_for(
                lambda: (
                    app.screen is screen
                    and bool(screen.query("#library-conversations-filter"))
                    and not screen._conversations_state.loading
                ),
                "Conversations returned",
            )
            assert screen._conversations_state.requested_query == "Alpha"
            assert screen._conversations_state.total == 1
            if not screen._conversations_state.reader_layout.items_open:
                await press("#library-conversations-items-grip")

        async def export_form(expected_count, selected=False):
            await wait_for(
                lambda: (
                    bool(screen.query("#library-export-name"))
                    and screen._export_state.counts is not None
                ),
                "Export counts",
            )
            assert screen._export_state.counts["conversations"] == expected_count
            assert sum(screen._export_state.counts.values()) == expected_count
            line = str(screen.query_one("#library-export-scope-line", Static).render())
            assert ("Selected conversations" if selected else "Conversations") in line
            assert not screen.query("#library-export-quality")

        async def choose_destination(filename, *, cancel=False):
            await press("#library-export-destination")
            await wait_for(lambda: isinstance(app.screen, FileSave), "FileSave")
            dialog = app.screen
            field = dialog.query_one(FileNameInput)
            await wait_for(lambda: dialog.focused is field, "filename focus")
            if cancel:
                await pilot.press("escape")
                await wait_for(lambda: app.screen is screen, "picker cancelled")
                return
            await pilot.press("ctrl+a")
            field.post_message(events.Paste(str(exports)))
            await pilot.pause()
            await pilot.press("enter")
            await wait_for(
                lambda: (
                    dialog.query_one(DirectoryNavigation).location == exports
                    and field.value == ""
                ),
                "private export folder",
            )
            field.focus()
            field.post_message(events.Paste(filename))
            await pilot.pause()
            assert field.value == filename
            await pilot.press("tab")
            assert dialog.focused is dialog.query_one("#select", Button)
            assert "Save" in painted(dialog.focused)
            await pilot.press("enter")
            await wait_for(lambda: app.screen is screen, "destination selected")

        async def run_export(path, expected_ids, label):
            await press("#library-export-submit")
            await wait_for(
                lambda: (
                    not screen._export_state.running
                    and screen._export_state.last_path == str(path)
                    and path.exists()
                ),
                "export finished",
            )
            assert not screen._export_state.error, screen._export_state.error
            artifact = await asyncio.to_thread(inspect_zip, path, expected_ids, label)
            assert screen._export_state.last_items == artifact["item_count"]
            assert screen._export_state.last_bytes == artifact["size_bytes"]
            receipt = screen.query_one("#library-export-last-line", Static)
            assert "exported" in str(receipt.render())
            assert str(path) in str(receipt.render())
            screen.query_one("#library-export-submit", Button).focus()
            await wait_for(lambda: "exported" in painted(receipt), "painted receipt")
            await capture(label + "-receipt")

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
            store = app.screen._console_chat_store
            context = (
                store.active_session_id,
                [s.id for s in store.sessions()],
                store.session_draft(store.active_session_id),
            )
            await asyncio.to_thread(seed)
            before = source_snapshot()
            alpha, beta, _gamma = (f["id"] for f in fixtures)
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: type(app.screen).__name__ == "LibraryScreen", "Library"
            )
            screen = app.screen
            await press("#library-row-browse-conversations")
            await wait_for(
                lambda: (
                    bool(screen.query("#library-conversations-filter"))
                    and not screen._conversations_state.loading
                ),
                "Conversations ready",
            )
            field = screen.query_one("#library-conversations-filter", Input)
            field.value = "Alpha"
            field.focus()
            await pilot.press("enter")
            await wait_for(
                lambda: screen._conversations_state.total == 1, "filtered Alpha"
            )
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
                        "matrix size",
                    )
                    await filtered_conversations()
                    if screen._conversations_state.select_mode:
                        await press("#library-conversations-select-toggle")
                    await press("#library-conversations-export")
                    await export_form(2)
                    assert screen.query_one("#library-export-submit", Button).disabled
                    assert "Choose a destination" in str(
                        screen.query_one(
                            "#library-export-submit-reason", Static
                        ).render()
                    )
                    await choose_destination("unused.txt", cancel=True)
                    assert not screen._export_state.form.get("destination")
                    name = f"bundle-{theme}-{width}"
                    await choose_destination(name + ".txt")
                    path = exports / (name + ".zip")
                    assert screen._export_state.form["destination"] == str(path)
                    await run_export(path, [alpha, beta], name + "-whole")
                    await pilot.press("escape")
                    await filtered_conversations()
                    await press("#library-conversations-select-toggle")
                    await wait_for(
                        lambda: screen._conversations_state.select_mode, "select mode"
                    )
                    await press("#library-conversation-row-0", "Export audit Alpha")
                    await wait_for(
                        lambda: screen._conversations_state.row_selection.count == 1,
                        "Alpha selected",
                    )
                    assert screen._conversations_state.row_selection.is_selected(alpha)
                    await press("#library-conversations-export-selected")
                    await export_form(1, selected=True)
                    before_overwrite = hashlib.sha256(path.read_bytes()).hexdigest()
                    await choose_destination(name + ".txt")
                    assert screen._export_state.form["destination_exists"]
                    assert "Overwrites" in str(
                        screen.query_one(
                            "#library-export-overwrite-line", Static
                        ).render()
                    )
                    assert (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        == before_overwrite
                    )
                    screen.query_one("#library-export-submit", Button).focus()
                    await wait_for(
                        lambda: (
                            "Overwrites"
                            in painted(
                                screen.query_one("#library-export-overwrite-line")
                            )
                        ),
                        "painted overwrite",
                    )
                    await capture(name + "-overwrite")
                    await run_export(path, [alpha], name + "-selected")
                    assert (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        != before_overwrite
                    )
                    await pilot.press("escape")
                    await filtered_conversations()
                    result["steps"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "whole_active_scope": 2,
                            "selected_scope": 1,
                            "cancel_normalize_overwrite_return_passed": True,
                        }
                    )
                    record()
            assert source_snapshot() == before
            assert (
                store.active_session_id,
                [s.id for s in store.sessions()],
                store.session_draft(store.active_session_id),
            ) == context
            result["source_records_unchanged"] = True
            result["console_context_unchanged"] = True
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
