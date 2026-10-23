"""Mounted Prompt edits survive pending structural and persistence publication."""

import asyncio
import threading

import pytest
from textual.widgets import Button, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _open_prompt_editor,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor


async def _until(predicate):
    async with asyncio.timeout(12):
        while not predicate():
            await asyncio.sleep(0.01)


def _host(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.app_config.setdefault("library", {})["prompt_editor_mode"] = "basic"
    return db, service, LibraryProductionCSSHarness(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["system", "user"])
@pytest.mark.parametrize("later", ["none", "key", "clear", "other-lane"])
@private_profile_test
async def test_first_block_mount_keeps_live_input(
    request, tmp_path, monkeypatch, lane, later
):
    db, _, host = _host(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    reconcile = PromptBlockEditor._reconcile_lane
    held = False

    async def held_reconcile(editor, lane_id, *, previous_count):
        nonlocal held
        await reconcile(editor, lane_id, previous_count=previous_count)
        if not held and lane_id == lane and previous_count == 0:
            held = True
            entered.set()
            await asyncio.wait_for(release.wait(), 30)

    monkeypatch.setattr(PromptBlockEditor, "_reconcile_lane", held_reconcile)
    delivery = None
    try:
        async with host.run_test(size=(170, 48)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-create-prompt", Button).press()
            await _wait_for_selector(screen, pilot, "#library-prompt-name")
            body = screen.query_one(f"#library-prompt-{lane}", TextArea)
            other_lane = "user" if lane == "system" else "system"
            other = screen.query_one(f"#library-prompt-{other_lane}", TextArea)
            body.focus()
            body.insert("a")
            await _until(entered.is_set)
            editor = screen.query_one(PromptBlockEditor)
            first = editor._state.definition.lanes[0 if lane == "system" else 1].blocks[
                0
            ]
            assert body.text == "a" and not body.read_only and not body.disabled
            if later == "key":
                delivery = asyncio.create_task(pilot.press("b"))
                await _until(lambda: body.text == "ab")
            elif later == "clear":
                body.clear()
                assert body.text == ""
            elif later == "other-lane":
                other.focus()
                delivery = asyncio.create_task(pilot.press("z"))
                await _until(lambda: other.text == "z")
            expected = "ab" if later == "key" else "" if later == "clear" else "a"
            release.set()
            if delivery is not None:
                await asyncio.wait_for(delivery, 15)
            for _ in range(3):
                await pilot.pause()
            assert body.text == expected, "older reconciliation replaced accepted input"
            assert screen.query_one(f"#library-prompt-{lane}", TextArea) is body
            state = screen._prompts_state.block_state
            assert getattr(state, f"compiled_{lane}") == expected
            assert other.text == ("z" if later == "other-lane" else "")
            assert getattr(state, f"compiled_{other_lane}") == other.text
            blocks = state.definition.lanes[0 if lane == "system" else 1].blocks
            assert len(blocks) == 1
            assert blocks[0].id == first.id
            assert blocks[0].title == first.title
            assert blocks[0].syntax == first.syntax
            assert len(
                {b.id for ln in state.definition.lanes for b in ln.blocks}
            ) == sum(len(ln.blocks) for ln in state.definition.lanes)
            assert editor._state.definition == state.definition
            # A subsequent edit continues on the same child and block identity.
            body.focus()
            await pilot.press("end", "c")
            await pilot.pause()
            assert body.text == expected + "c"
            assert (
                getattr(screen._prompts_state.block_state, f"compiled_{lane}")
                == body.text
            )
    finally:
        release.set()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing, mode, later, failure",
    [
        (False, "basic", "none", False),
        (True, "basic", "none", False),
        (False, "basic", "body", False),
        (True, "basic", "body", False),
        (True, "basic", "queued-body", False),
        (False, "advanced", "queued-body", False),
        (True, "advanced", "body", False),
        (False, "basic", "queued-metadata", False),
        (True, "advanced", "metadata", False),
        (True, "basic", "body", True),
    ],
)
@private_profile_test
async def test_save_acknowledges_only_submitted_prompt(
    request, tmp_path, monkeypatch, existing, mode, later, failure
):
    db, service, host = _host(tmp_path)
    seed_id = None
    if existing:
        seed_id, _, _ = db.add_prompt(
            name="pending-save", author="", details="", user_prompt="Baseline"
        )
    entered, release = threading.Event(), threading.Event()
    save = service.save_prompt
    attempts = 0

    async def held_save(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            entered.set()
            assert release.wait(30), "test write gate timed out"
            if failure:
                raise OSError("Definite pre-write failure")
        return await save(*args, **kwargs)

    monkeypatch.setattr(service, "save_prompt", held_save)
    try:
        async with host.run_test(size=(170, 48)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            if existing:
                await _open_prompt_editor(screen, pilot, seed_id)
            else:
                screen.query_one("#library-row-create-prompt", Button).press()
            name = await _wait_for_selector(screen, pilot, "#library-prompt-name")
            name.value = "pending-save"
            basic = screen.query_one("#library-prompt-user", TextArea)
            basic.text = "Submitted body"
            await pilot.pause()
            if mode == "advanced":
                screen.query_one("#library-prompt-mode-advanced", Button).press()
                await pilot.pause()
                body = screen.query_one(".prompt-block-content", TextArea)
            else:
                body = basic
            editor = screen.query_one(PromptBlockEditor)
            block_id = editor.state.definition.lanes[1].blocks[0].id
            run_call = screen._run_library_service_call
            immediate = []

            def edit_queued():
                if later == "queued-body":
                    body.move_cursor((0, len(body.text)))
                    body.insert("x")
                    assert body.text == "Submitted bodyx"
                else:
                    name.replace("newer name ", 0, len(name.value))
                    assert name.value == "newer name "

            async def capture_result(fn, *args, **kwargs):
                result = await run_call(fn, *args, **kwargs)
                if fn is held_save and attempts == 1 and later.startswith("queued"):
                    # Same child edit primitive as native input; Changed is queued.
                    edit_queued()
                    immediate.append((body.text, name.value))
                return result

            monkeypatch.setattr(screen, "_run_library_service_call", capture_result)
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot, entered.is_set, message="Save did not enter writer"
            )
            assert not body.disabled and not body.read_only
            if later == "body":
                body.focus()
                await pilot.press("end", "x")
                assert body.text == "Submitted bodyx"
            elif later == "metadata":
                name.focus()
                await pilot.press("end", "x")
                assert name.value == "pending-savex"
            release.set()
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_prompt_write_worker_is_active(),
                message="Prompt write did not settle",
            )
            await pilot.pause()
            prompt_id = screen._prompts_state.selected_prompt_id
            assert prompt_id == seed_id if existing else isinstance(prompt_id, int)
            stored = db.fetch_prompt_details(prompt_id)
            expected_body = "Submitted bodyx" if "body" in later else "Submitted body"
            expected_name = (
                "newer name "
                if later == "queued-metadata"
                else "pending-savex"
                if later == "metadata"
                else "pending-save"
            )
            assert body.text == expected_body
            assert name.value == expected_name
            assert screen.query_one("#library-prompt-user", TextArea) is basic
            assert screen._prompts_state.dirty is (later != "none")
            assert screen._prompts_state.block_state.compiled_user == expected_body
            assert editor.state.compiled_user == expected_body
            assert basic.text == expected_body
            if later.startswith("queued"):
                assert immediate
            if failure:
                assert stored["user_prompt"] == "Baseline"
                assert stored["version"] == 1
            else:
                assert stored["user_prompt"] == "Submitted body"
                assert stored["name"] == "pending-save"
                assert (
                    screen._prompts_state.version
                    == stored["version"]
                    == (2 if existing else 1)
                )
            if later != "none":
                screen.query_one("#library-prompt-back", Button).press()
                await pilot.pause()
                assert not await screen.flush_pending_work(quitting=True)
                assert screen._prompts_state.view == "editor"
                assert body.text == expected_body
                # Retrying saves current draft into this same row, using adopted version.
                screen.query_one("#library-prompt-save", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: attempts == 2
                    and not screen._library_prompt_write_worker_is_active(),
                    message="Second Save did not settle",
                )
                await pilot.pause()
                assert screen._prompts_state.selected_prompt_id == prompt_id
                final = db.fetch_prompt_details(prompt_id)
                assert final["user_prompt"] == expected_body
                assert final["name"] == expected_name.strip()
                assert final["version"] == stored["version"] + 1
                assert not screen._prompts_state.dirty
            assert editor.state.definition.lanes[1].blocks[0].id == block_id
    finally:
        release.set()
        db.close()


@pytest.mark.asyncio
@private_profile_test
async def test_save_keeps_newer_input_before_older_semantic_message(
    tmp_path, request, monkeypatch
):
    from tldw_chatbook.Widgets.Library.library_prompt_work_pane import (
        LibraryPromptWorkPane,
    )

    db, service, host = _host(tmp_path)
    entered, release = threading.Event(), threading.Event()
    semantic_entered, semantic_release = asyncio.Event(), asyncio.Event()
    save = service.save_prompt
    original = LibraryPromptWorkPane.on_prompt_block_editor_block_field_changed
    held = []

    async def held_save(*args, **kwargs):
        entered.set()
        assert release.wait(40)
        return await save(*args, **kwargs)

    async def held_semantic(canvas, event):
        if entered.is_set() and not held:
            held.append(event)
            semantic_entered.set()
            await asyncio.wait_for(semantic_release.wait(), 40)
        original(canvas, event)

    monkeypatch.setattr(service, "save_prompt", held_save)
    monkeypatch.setattr(
        LibraryPromptWorkPane,
        "on_prompt_block_editor_block_field_changed",
        held_semantic,
    )
    delivery = None
    try:
        async with host.run_test(size=(170, 48)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-create-prompt", Button).press()
            name = await _wait_for_selector(screen, pilot, "#library-prompt-name")
            name.value = "semantic-save"
            screen.query_one("#library-prompt-user", TextArea).text = "Submitted body"
            await pilot.pause()
            screen.query_one("#library-prompt-mode-advanced", Button).press()
            await pilot.pause()
            body = screen.query_one(".prompt-block-content", TextArea)
            body.focus()
            await pilot.pause()
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot, entered.is_set, message="Save did not enter writer"
            )
            body.move_cursor((0, len(body.text)))
            body.insert("x")
            await _until(semantic_entered.is_set)
            assert held[0].state.compiled_user == "Submitted bodyx"
            assert not body.disabled and not body.read_only
            delivery = asyncio.create_task(pilot.press("y"))
            await _until(lambda: body.text == "Submitted bodyxy")
            release.set()
            await _until(lambda: not screen._library_prompt_write_worker_is_active())
            acknowledged_text = body.text
            semantic_release.set()
            await asyncio.wait_for(delivery, 15)
            await pilot.pause()
            assert (
                acknowledged_text == "Submitted bodyxy"
            ), "acknowledgement lost queued live input"
            assert body.text == "Submitted bodyxy"
            assert (
                screen.query_one("#library-prompt-user", TextArea).text
                == "Submitted bodyxy"
            )
            assert screen._prompts_state.block_state.compiled_user == "Submitted bodyxy"
            assert screen._prompts_state.dirty
            stored = db.fetch_prompt_details(screen._prompts_state.selected_prompt_id)
            assert stored["user_prompt"] == "Submitted body"
    finally:
        release.set()
        semantic_release.set()
        db.close()
