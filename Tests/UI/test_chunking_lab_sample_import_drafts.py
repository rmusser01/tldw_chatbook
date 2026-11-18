"""Real file reads must not overwrite newer accepted Chunking Lab sample intent."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager

import pytest
from textual.widgets import Input, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_chunking_lab_screen import settle_lab


@asynccontextmanager
async def mounted_lab():
    # Construct after private-profile activation; do not run unrelated app services.
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.chunking_lab_screen import ChunkingLabScreen

    owner_app = _build_test_app()
    host = ConsolidatedCSSApp()
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = ChunkingLabScreen(owner_app)
            await host.push_screen(screen)
            await asyncio.wait_for(screen.wait_until_ready(), 10)
            await settle_lab(host, screen, pilot)
            assert screen.coordinator is not None
            try:
                yield host, screen, pilot
            finally:
                await screen.drain_edits()
                await settle_lab(host, screen, pilot)
                await host.pop_screen()
    finally:
        owner = getattr(owner_app, "_chunking_lab_coordinator", None)
        if owner is not None:
            await owner.close()
            owner_app._chunking_lab_coordinator = None


def sample(screen):
    session = screen.coordinator.session
    return session.samples[session.view["sample_hash"]]


@asynccontextmanager
async def held_read(monkeypatch, action, selected):
    from tldw_chatbook.UI.Screens import chunking_lab_screen as module

    entered, release = threading.Event(), threading.Event()
    name = "read_sample_excerpt" if action == "excerpt" else "read_sample_file"
    original = getattr(module, name)

    def read(*args):
        result = original(*args)
        if args[0] == str(selected):
            entered.set()
            assert release.wait(20), "sample read gate was not released"
        return result

    monkeypatch.setattr(module, name, read)
    try:
        yield entered, release
    finally:
        release.set()


async def entered_read(entered):
    async with asyncio.timeout(10):
        while not entered.is_set():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["sample-file", "excerpt"])
@pytest.mark.parametrize("later", ["untouched", "text", "clear", "back-to-baseline"])
@private_profile_test
async def test_pending_sample_read_preserves_accepted_input(
    request, tmp_path, monkeypatch, action, later
):
    selected = tmp_path / "selected.txt"
    selected.write_bytes(b"selected\r\nfile content")
    async with mounted_lab() as (host, screen, pilot):
        editor = screen.query_one("#lab-sample-text", TextArea)
        editor.load_text("seed")
        await pilot.pause()
        await screen.drain_edits()
        async with held_read(monkeypatch, action, selected) as (entered, release):
            pending = asyncio.create_task(
                screen.file_operation(
                    action,
                    {
                        "path": str(selected),
                        "start": "2",
                        "end": "12",
                    },
                )
            )
            try:
                await entered_read(entered)
                expected = (
                    "selected\r\nfile content"
                    if action == "sample-file"
                    else "lected\r\nfi"
                )
                if later != "untouched":
                    editor.focus()
                    await pilot.press("x")
                    await pilot.pause()
                    await screen.drain_edits()
                    assert editor.text == "xseed"
                    assert sample(screen)["text"] == "xseed"
                    if later == "clear":
                        await pilot.press("end", *(["backspace"] * 5))
                    elif later == "back-to-baseline":
                        await pilot.press("backspace")
                    await pilot.pause()
                    await screen.drain_edits()
                    expected = {
                        "text": "xseed",
                        "clear": "",
                        "back-to-baseline": "seed",
                    }[later]
                    assert editor.text == expected
                    assert sample(screen)["text"] == expected
                release.set()
                await asyncio.wait_for(pending, 10)
                await settle_lab(host, screen, pilot)
                assert editor.text == expected
                assert sample(screen)["text"] == expected
                if later == "untouched":
                    source = (
                        {"kind": "file", "name": selected.name}
                        if action == "sample-file"
                        else {
                            "kind": "file_excerpt",
                            "name": selected.name,
                            "start": 2,
                            "end": 12,
                        }
                    )
                    assert sample(screen)["source"] == source
                else:
                    assert sample(screen)["source"] == {"kind": "paste"}
            finally:
                release.set()
                await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["sample-file", "excerpt"])
@private_profile_test
async def test_newer_sample_read_supersedes_older_completion(
    request, tmp_path, monkeypatch, action
):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("older contents", encoding="utf-8")
    second.write_text("newer contents", encoding="utf-8")
    async with mounted_lab() as (host, screen, pilot):
        async with held_read(monkeypatch, action, first) as (entered, release):
            pending = asyncio.create_task(
                screen.file_operation(
                    action, {"path": str(first), "start": "0", "end": "5"}
                )
            )
            try:
                await entered_read(entered)
                await screen.file_operation(
                    action, {"path": str(second), "start": "0", "end": "5"}
                )
                expected = "newer contents" if action == "sample-file" else "newer"
                assert sample(screen)["text"] == expected
                release.set()
                await pending
                await settle_lab(host, screen, pilot)
                assert screen.query_one("#lab-sample-text", TextArea).text == expected
                assert sample(screen)["text"] == expected
                assert sample(screen)["source"]["name"] == second.name
            finally:
                release.set()
                await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["sample-file", "excerpt"])
@pytest.mark.parametrize("failure", ["missing", "size", "utf8"])
@private_profile_test
async def test_sample_read_failure_preserves_authored_recovery(
    request, tmp_path, action, failure
):
    selected = tmp_path / "invalid.txt"
    if failure == "size":
        selected.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    elif failure == "utf8":
        selected.write_bytes(b"\xff" * 4)
    async with mounted_lab() as (host, screen, pilot):
        editor = screen.query_one("#lab-sample-text", TextArea)
        editor.focus()
        await pilot.press("x")
        await pilot.pause()
        await screen.drain_edits()
        assert editor.text == sample(screen)["text"] == "x"
        before = screen.coordinator.session
        error = UnicodeDecodeError if failure == "utf8" else ValueError
        message = {"missing": "does not exist", "size": "2 MiB", "utf8": "utf-8"}[
            failure
        ]
        with pytest.raises(error, match=message):
            await screen.file_operation(
                action,
                {
                    "path": str(selected),
                    "start": "0",
                    "end": str(2 * 1024 * 1024 + 1) if failure == "size" else "3",
                },
            )
        await settle_lab(host, screen, pilot)
        assert screen.coordinator.session is before
        assert editor.text == sample(screen)["text"] == "x"
        await screen.coordinator.cancel()
        from tldw_chatbook.Chunking.lab_recovery import export_recovery, parse_recovery

        restored = parse_recovery(export_recovery(screen.coordinator.session))
        assert restored.samples[restored.view["sample_hash"]]["text"] == "x"
    # Reopen the actual SQLite owner after its writer has closed.
    reopened = await screen.app_instance.get_chunking_lab_coordinator()
    try:
        restored = reopened.session
        assert restored.samples[restored.view["sample_hash"]]["text"] == "x"
        assert restored.samples[restored.view["sample_hash"]]["source"] == {
            "kind": "paste"
        }
    finally:
        await reopened.close()
        screen.app_instance._chunking_lab_coordinator = None


@pytest.mark.asyncio
@private_profile_test
async def test_sample_read_sees_keyboard_edit_before_parent_changed_delivery(
    request, tmp_path, monkeypatch
):
    selected = tmp_path / "selected.txt"
    selected.write_text("older contents", encoding="utf-8")
    async with mounted_lab() as (host, screen, pilot):
        editor = screen.query_one("#lab-sample-text", TextArea)
        editor.load_text("seed")
        editor.focus()
        await pilot.pause()
        await screen.drain_edits()
        parent_entered, parent_release = asyncio.Event(), asyncio.Event()

        async def hold_parent():
            parent_entered.set()
            await asyncio.wait_for(parent_release.wait(), 10)

        async with held_read(monkeypatch, "sample-file", selected) as (
            entered,
            release,
        ):
            pending = asyncio.create_task(
                screen.file_operation("sample-file", {"path": str(selected)})
            )
            key = None
            try:
                await entered_read(entered)
                screen.call_next(hold_parent)
                await asyncio.wait_for(parent_entered.wait(), 3)
                key = asyncio.create_task(pilot.press("x"))
                async with asyncio.timeout(3):
                    while editor.text != "xseed":
                        await asyncio.sleep(0.01)
                assert sample(screen)["text"] == "seed"  # Parent message remains held.
                release.set()
                await asyncio.wait_for(pending, 3)
                assert editor.text == "xseed"
                parent_release.set()
                await key
                await pilot.pause()
                await screen.drain_edits()
                assert sample(screen)["text"] == editor.text == "xseed"
            finally:
                parent_release.set()
                release.set()
                await pending
                if key is not None:
                    await key


@pytest.mark.asyncio
@private_profile_test
async def test_sample_read_rechecks_intent_after_queued_transition_computes(
    request, tmp_path, monkeypatch
):
    from tldw_chatbook.Chunking import lab_state

    selected = tmp_path / "selected.txt"
    selected.write_text("older contents", encoding="utf-8")
    entered, release = threading.Event(), threading.Event()
    original = lab_state.replace_sample

    def held_transition(session, text, source):
        changed = original(session, text, source)
        if text == "older contents":
            entered.set()
            assert release.wait(15), "queued transition gate was not released"
        return changed

    monkeypatch.setattr(lab_state, "replace_sample", held_transition)
    async with mounted_lab() as (host, screen, pilot):
        editor = screen.query_one("#lab-sample-text", TextArea)
        editor.focus()
        await pilot.pause()
        pending = asyncio.create_task(
            screen.file_operation("sample-file", {"path": str(selected)})
        )
        try:
            await entered_read(entered)
            await pilot.press("x")
            assert editor.text == "x"
            release.set()
            await asyncio.wait_for(pending, 10)
            await pilot.pause()
            await screen.drain_edits()
            await settle_lab(host, screen, pilot)
            assert editor.text == sample(screen)["text"] == "x"
            # The stale file must never become the undo baseline of the real edit.
            screen.queue_edit(lab_state.undo_edit)
            await screen.drain_edits()
            assert editor.text == sample(screen)["text"] == ""
        finally:
            release.set()
            await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["sample-file", "excerpt"])
@private_profile_test
async def test_newer_failed_read_revokes_older_sample_intent(
    request, tmp_path, monkeypatch, action
):
    selected = tmp_path / "older.txt"
    selected.write_text("older contents", encoding="utf-8")
    async with mounted_lab() as (host, screen, pilot):
        editor = screen.query_one("#lab-sample-text", TextArea)
        editor.focus()
        await pilot.press("x")
        await pilot.pause()
        await screen.drain_edits()
        assert sample(screen)["text"] == editor.text == "x"
        async with held_read(monkeypatch, action, selected) as (entered, release):
            pending = asyncio.create_task(
                screen.file_operation(
                    action, {"path": str(selected), "start": "0", "end": "5"}
                )
            )
            try:
                await entered_read(entered)
                with pytest.raises((ValueError, OSError)):
                    await screen.file_operation(
                        action,
                        {
                            "path": str(tmp_path / "missing.txt"),
                            "start": "0",
                            "end": "5",
                        },
                    )
                release.set()
                await pending
                await settle_lab(host, screen, pilot)
                assert editor.text == sample(screen)["text"] == "x"
                assert sample(screen)["source"] == {"kind": "paste"}
            finally:
                release.set()
                await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["sample-file", "excerpt"])
@private_profile_test
async def test_pending_sample_read_retains_unrelated_recipe_edit(
    request, tmp_path, monkeypatch, action
):
    selected = tmp_path / "selected.txt"
    selected.write_bytes(b"selected file content")
    async with mounted_lab() as (host, screen, pilot):
        editor = screen.query_one("#lab-sample-text", TextArea)
        initial_sample = sample(screen)
        candidate_id = next(iter(screen.coordinator.session.candidates))
        assert await pilot.click("#lab-show-configure")
        maximum = screen.query_one("#lab-max-size", Input)
        async with held_read(monkeypatch, action, selected) as (entered, release):
            pending = asyncio.create_task(
                screen.file_operation(
                    action, {"path": str(selected), "start": "0", "end": "8"}
                )
            )
            try:
                await entered_read(entered)
                maximum.focus()
                await pilot.press("home", "shift+end", *"512")
                await pilot.pause()
                await screen.drain_edits()
                await settle_lab(host, screen, pilot)
                draft = screen.coordinator.session.candidates[candidate_id]["draft"]
                assert (
                    json.loads(draft["parsed_json"])["chunking"]["config"]["max_size"]
                    == 512
                )
                assert not draft["pending_controls"]
                assert maximum.value == "512"
                assert sample(screen) == initial_sample
                release.set()
                await asyncio.wait_for(pending, 10)
                await settle_lab(host, screen, pilot)
                expected = (
                    "selected file content" if action == "sample-file" else "selected"
                )
                assert editor.text == sample(screen)["text"] == expected
                source = {"kind": "file", "name": selected.name}
                if action == "excerpt":
                    source.update(kind="file_excerpt", start=0, end=8)
                assert sample(screen)["source"] == source
                assert (
                    screen.coordinator.session.candidates[candidate_id]["draft"]
                    == draft
                )
                assert maximum.value == "512"
            finally:
                release.set()
                await pending
