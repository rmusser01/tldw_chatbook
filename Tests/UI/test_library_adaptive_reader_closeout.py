"""Durable cross-destination adaptive-reader closeout regressions."""

from __future__ import annotations

import inspect
import threading
from pathlib import Path

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.UI.test_library_conversation_reader import (
    _GatedVersionConversationService,
    _OutOfOrderConversationService,
    _conversation_records,
)
from Tests.UI.test_library_media_side_by_side import _many_media_items
from Tests.UI.test_library_prompts_canvas import _real_prompt_scope_service
from Tests.UI.test_library_prompts_canvas import _open_prompts_list
from Tests.UI.test_library_shell import (
    LibraryGlobalKeyProductionCSSHarness,
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_library_skills_reader import _wire_skills
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.Widgets.workbench_focus import _available_targets
from tldw_chatbook.config import load_settings


DESTINATIONS = ("media", "conversations", "notes", "prompts", "skills")
SIZES = ((160, 50), (120, 35), (100, 30), (80, 24), (160, 50))

DESTINATION_CONTRACT = {
    "media": (
        "#library-row-browse-media",
        "#library-media-reader-shell",
        "#library-media-row-1",
        "_library_media_reader_preferences",
        "_library_media_reader_layout",
    ),
    "conversations": (
        "#library-row-browse-conversations",
        "#library-conversations-reader-shell",
        "#library-conversation-row-1",
        "_library_conversation_reader_preferences",
        "_library_conversation_reader_layout",
    ),
    "notes": (
        "#library-row-browse-notes",
        "#library-notes-reader-shell",
        "#library-notes-row-1",
        "_library_notes_reader_preferences",
        "_library_notes_reader_layout",
    ),
    "prompts": (
        "#library-row-browse-prompts",
        "#library-prompts-reader-shell",
        ".library-prompt-row",
        "_library_prompts_reader_preferences",
        "_library_prompts_reader_layout",
    ),
    "skills": (
        "#library-row-browse-skills",
        "#library-skills-reader-shell",
        "#library-skill-row-review-skill",
        "_library_skills_reader_preferences",
        "_library_skills_reader_layout",
    ),
}


def _instrument_resize_service_seams(monkeypatch, app) -> dict[str, int]:
    """Count every destination list/detail seam after initial settlement."""
    counts: dict[str, int] = {}
    services = {
        "media": (
            app.media_reading_scope_service,
            ("search_media", "get_media_item", "get_reading_progress"),
        ),
        "conversations": (
            app.chat_conversation_scope_service,
            ("list_conversations",),
        ),
        "conversation_detail": (
            app.local_chat_conversation_service,
            ("get_library_conversation_messages",),
        ),
        "notes": (app.notes_scope_service, ("list_notes", "get_note_detail")),
        "prompts": (app.prompt_scope_service, ("list_prompts", "get_prompt")),
        "skills": (app.skills_scope_service, ("get_context", "get_skill")),
    }
    for owner, (service, names) in services.items():
        for name in names:
            original = getattr(service, name, None)
            if not callable(original):
                continue
            key = f"{owner}.{name}"
            counts[key] = 0

            async def counted(*args, _original=original, _key=key, **kwargs):
                counts[_key] += 1
                result = _original(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result

            monkeypatch.setattr(service, name, counted)
    return counts


async def _seed_closeout_app(root: Path):
    """Build one production-shaped app from established destination fixtures."""
    root.mkdir(parents=True, exist_ok=True)
    app = _build_test_app()
    local_skills = _wire_skills(app, root / "skills")
    await local_skills.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
        supporting_files={"references/guide.md": "Read this guide."},
    )
    await local_skills.create_skill(
        name="review-skill",
        content="---\nname: review-skill\ndescription: Review skill\n---\nReview exactly.",
    )
    records = [dict(record, version=7) for record in _conversation_records()]
    _seed_conversations(
        app,
        records,
        notes=_two_notes(),
        media=_many_media_items(4),
    )
    conversation_service = _GatedVersionConversationService(7)
    conversation_service.release.set()
    app.local_chat_conversation_service = conversation_service
    prompt_db, prompt_service = _real_prompt_scope_service(root)
    for index in range(2):
        prompt_db.add_prompt(
            name=f"Closeout prompt {index + 1}",
            author="Ada",
            details="Closeout fixture",
            system_prompt="Be exact.",
            user_prompt=f"Summarize item {index + 1}.",
            keywords=["closeout"],
        )
    app.prompt_scope_service = prompt_service
    return app, prompt_db


async def _open_destination(screen, pilot, destination: str):
    rail, shell_selector, second_selector, _preferences, _layout = DESTINATION_CONTRACT[
        destination
    ]
    mounted_shells = [
        candidate
        for contract in DESTINATION_CONTRACT.values()
        for candidate in screen.query(contract[1])
    ]
    restore_closed_library = bool(
        mounted_shells
        and {
            getattr(screen, contract[3]).library_open
            for contract in DESTINATION_CONTRACT.values()
        }
        == {False}
    )
    if restore_closed_library:
        mounted_shells[0].library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                {
                    getattr(screen, contract[3]).library_open
                    for contract in DESTINATION_CONTRACT.values()
                }
                == {True}
                and screen.query_one(rail, Button).region.area > 0
            ),
            message=f"Library restore grip did not expose {destination}",
        )
    screen.query_one(rail, Button).press()
    shell = await _wait_for_selector(screen, pilot, shell_selector)
    if destination == "prompts":
        await _open_prompts_list(screen, pilot)
        shell = await _wait_for_selector(screen, pilot, shell_selector)
    second = await _wait_for_selector(screen, pilot, second_selector)
    if destination == "prompts":
        rows = list(screen.query(".library-prompt-row"))
        assert len(rows) >= 2
        second = rows[1]
    expected = str(
        getattr(
            second,
            {
                "media": "media_id",
                "conversations": "conversation_id",
                "notes": "note_id",
                "prompts": "prompt_id",
                "skills": "skill_name",
            }[destination],
        )
    )
    already_selected = {
        "media": lambda: (
            str(screen._library_media_reader_session.selected_id or "")
            == str(second.media_id)
            and bool(screen.query("#library-media-viewer-title"))
        ),
        "conversations": lambda: (
            str(screen._library_conversation_reader_state.selected_id or "")
            == str(second.conversation_id)
        ),
        "notes": lambda: str(screen._selected_note_id or "") == str(second.note_id),
        "prompts": lambda: str(screen._selected_prompt_id) == expected,
        "skills": lambda: (
            screen._library_skill_editor_state is not None
            and screen._library_skill_editor_state.name == str(second.skill_name)
        ),
    }[destination]
    if not already_selected():
        second.press()
    if destination == "media":
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_reader_session.selected_id == expected
                and screen._library_media_reader_session.loaded_id == expected
                and screen._library_media_reader_session.pending_request is None
            ),
            message="Media second selection did not settle",
        )
        await _wait_for_selector(
            screen,
            pilot,
            (f"#library-media-reader-mode-{screen._library_media_reader_session.mode}"),
        )
    elif destination == "conversations":
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_conversation_reader_state.selected_id == expected
                and screen._library_conversation_reader_state.loaded_id == expected
                and not screen._library_conversation_reader_state.loading
            ),
            message="Conversation second selection did not settle",
        )
        await _wait_for_selector(screen, pilot, "#library-conversation-reader-info")
    elif destination == "notes":
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._selected_note_id == expected
                and screen._library_note_load_state == "loaded"
                and screen._library_note_session.snapshot is not None
                and screen._library_note_session.snapshot.note_id == expected
            ),
            message="Note second selection did not settle",
        )
        await _wait_for_selector(screen, pilot, "#library-note-title")
        await _wait_for_selector(screen, pilot, "#library-note-preview")
    elif destination == "prompts":
        await _wait_for_condition(
            pilot,
            lambda: (
                str(screen._selected_prompt_id) == expected
                and str(screen._library_prompt_loaded_id) == expected
                and screen._library_prompt_detail is not None
                and not screen._library_prompt_detail_loading
            ),
            message="Prompt second selection did not settle",
        )
        await _wait_for_selector(screen, pilot, "#library-prompt-name")
        await _wait_for_selector(screen, pilot, "#library-prompt-mode-info")
    else:
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._selected_skill_name == expected
                and screen._library_skill_editor_state is not None
                and screen._library_skill_editor_state.name == expected
                and not screen._library_skill_detail_loading
            ),
            message="Skill second selection did not settle",
        )
        await _wait_for_selector(screen, pilot, "#library-skill-mode-overview")
    if restore_closed_library:
        generation = screen._library_reader_persistence_generations["library"]
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                {
                    getattr(screen, contract[3]).library_open
                    for contract in DESTINATION_CONTRACT.values()
                }
                == {False}
                and screen._library_reader_persistence_generations["library"]
                > generation
                and screen._library_reader_durable_generations["library"] > generation
                and not screen._library_reader_durable_preferences["library"]
            ),
            message=f"Shared Library choice did not restore after {destination}",
        )
    return shell


def _destination_state(screen, destination: str) -> tuple[object, ...]:
    shell_selector = DESTINATION_CONTRACT[destination][1]
    preferences_name = DESTINATION_CONTRACT[destination][3]
    layout_name = DESTINATION_CONTRACT[destination][4]
    shell = screen.query_one(shell_selector)
    if destination == "media":
        semantic = (
            screen._selected_media_id,
            screen._library_media_reader_session.loaded_id,
            screen._library_media_reader_session.mode,
        )
    elif destination == "conversations":
        semantic = (
            screen._library_conversation_reader_state.selected_id,
            screen._library_conversation_reader_state.loaded_id,
            screen._library_conversation_reader_state.mode,
        )
    elif destination == "notes":
        mode = (
            "context"
            if screen._library_note_context
            else "preview"
            if screen._library_note_preview
            else "edit"
        )
        semantic = (screen._selected_note_id, mode)
    elif destination == "prompts":
        semantic = (screen._selected_prompt_id, screen._library_prompt_editor_mode)
    else:
        semantic = (
            screen._library_skill_editor_state.name,
            screen._library_skill_reader_mode,
        )
    return (
        id(shell),
        id(shell.items),
        id(shell.work),
        getattr(screen, preferences_name),
        getattr(screen, layout_name),
        semantic,
    )


async def _focus_closeout_work_via_f6(
    screen, pilot, shell, destination: str
) -> tuple[str, str]:
    """Reach the active Work region through the app-owned visible F6 route."""
    available = _available_targets(screen, screen._library_workbench_focus_targets())
    expected = next(
        (
            target
            for pane, target in available
            if pane is shell.work or shell.work in pane.ancestors
        ),
        None,
    )
    assert expected is not None, f"{destination} has no reachable Work focus target"
    for _target in range(len(available) + 1):
        await pilot.press("f6")
        if screen.focused is expected:
            break
    assert screen.focused is expected
    assert screen.focused is shell or shell in screen.focused.ancestors
    return "work", str(screen.focused.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", DESTINATIONS)
async def test_closeout_resize_is_presentation_only(
    destination: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, prompt_db = await _seed_closeout_app(tmp_path / destination)
    host = LibraryProductionCSSHarness(app)
    try:
        async with host.run_test(size=SIZES[0]) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_destination(screen, pilot, destination)
            before = _destination_state(screen, destination)
            service_counts = _instrument_resize_service_seams(monkeypatch, app)
            seam_counts = {"config": 0, "write": 0, "worker": 0, "poll": 0}
            original_get = library_screen_module.get_cli_setting
            original_save = library_screen_module.save_setting_to_cli_config
            original_worker = screen.run_worker
            original_interval = screen.set_interval

            def counted_get(*args, **kwargs):
                seam_counts["config"] += 1
                return original_get(*args, **kwargs)

            def counted_save(*args, **kwargs):
                seam_counts["write"] += 1
                return original_save(*args, **kwargs)

            def counted_worker(*args, **kwargs):
                seam_counts["worker"] += 1
                return original_worker(*args, **kwargs)

            def counted_interval(*args, **kwargs):
                seam_counts["poll"] += 1
                return original_interval(*args, **kwargs)

            monkeypatch.setattr(library_screen_module, "get_cli_setting", counted_get)
            monkeypatch.setattr(
                library_screen_module, "save_setting_to_cli_config", counted_save
            )
            monkeypatch.setattr(screen, "run_worker", counted_worker)
            monkeypatch.setattr(screen, "set_interval", counted_interval)
            for width, height in SIZES[1:]:
                await pilot.resize_terminal(width, height)
                await _wait_for_condition(
                    pilot,
                    lambda width=width: screen.size.width == width,
                    message=f"Resize to {width} did not settle",
                )
            after = _destination_state(screen, destination)
            assert after[:3] == before[:3]
            assert after[5] == before[5]
            assert set(service_counts.values()) == {0}
            assert seam_counts == {"config": 0, "write": 0, "worker": 0, "poll": 0}
    finally:
        prompt_db.close()


@pytest.mark.asyncio
async def test_closeout_preferences_restore_in_fresh_screen(tmp_path: Path) -> None:
    expected = {
        "media": True,
        "conversations": False,
        "notes": True,
        "prompts": False,
        "skills": True,
    }
    first_config = load_settings(force_reload=True)
    first_app, first_prompt_db = await _seed_closeout_app(tmp_path / "first")
    first_app.app_config = first_config
    first_host = LibraryProductionCSSHarness(first_app)
    try:
        async with first_host.run_test(size=(160, 50)) as pilot:
            screen = _active_library_screen(first_host)
            await _wait_for_library_shell(screen, pilot)
            shell = None
            for destination, items_open in expected.items():
                shell = await _open_destination(screen, pilot, destination)
                preferences = getattr(screen, DESTINATION_CONTRACT[destination][3])
                if preferences.items_open is items_open:
                    continue
                authority = f"{destination}_items"
                generation = screen._library_reader_persistence_generations[authority]
                shell.items_grip.press()
                await _wait_for_condition(
                    pilot,
                    lambda authority=authority, generation=generation, destination=destination, items_open=items_open: (
                        getattr(screen, DESTINATION_CONTRACT[destination][3]).items_open
                        is items_open
                        and screen._library_reader_persistence_generations[authority]
                        > generation
                        and screen._library_reader_durable_generations[authority]
                        > generation
                    ),
                    message=f"{destination} Items choice did not persist",
                )
            assert shell is not None
            library_generation = screen._library_reader_persistence_generations[
                "library"
            ]
            shell.library_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    all(
                        not getattr(screen, contract[3]).library_open
                        for contract in DESTINATION_CONTRACT.values()
                    )
                    and screen._library_reader_persistence_generations["library"]
                    > library_generation
                    and screen._library_reader_durable_generations["library"]
                    > library_generation
                ),
                message="Mounted screen did not persist shared Library choice",
            )
    finally:
        first_prompt_db.close()

    fresh_config = load_settings(force_reload=True)
    fresh_app, fresh_prompt_db = await _seed_closeout_app(tmp_path / "fresh")
    fresh_app.app_config = fresh_config
    host = LibraryProductionCSSHarness(fresh_app)
    try:
        async with host.run_test(size=(160, 50)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            for destination, items_open in expected.items():
                await _open_destination(screen, pilot, destination)
                preferences = getattr(screen, DESTINATION_CONTRACT[destination][3])
                assert preferences.library_open is False
                assert preferences.items_open is items_open
    finally:
        fresh_prompt_db.close()


@pytest.mark.asyncio
async def test_closeout_single_app_route_cycle(tmp_path: Path) -> None:
    app, prompt_db = await _seed_closeout_app(tmp_path / "cycle")
    host = LibraryGlobalKeyProductionCSSHarness(app)
    remembered: dict[str, tuple[object, ...]] = {}
    remembered_focus: dict[str, tuple[str, str]] = {}
    expected_items = {destination: True for destination in DESTINATIONS}
    notes_before = tuple(dict(note) for note in app.notes_scope_service.notes)
    prompts_before = tuple(prompt_db.get_prompt_by_id(index) for index in (1, 2))
    stale_service: _OutOfOrderConversationService | None = None
    stale_target_id: str | None = None
    try:
        async with host.run_test(size=(160, 50)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            for destination in DESTINATIONS:
                shell = await _open_destination(screen, pilot, destination)
                if destination == "media":
                    screen.query_one(
                        "#library-media-reader-select-info", Button
                    ).press()
                    await _wait_for_condition(
                        pilot,
                        lambda: screen._library_media_reader_session.mode == "info",
                        message="Media route mode did not settle",
                    )
                    library_generation = screen._library_reader_persistence_generations[
                        "library"
                    ]
                    shell.library_grip.press()
                    await _wait_for_condition(
                        pilot,
                        lambda: (
                            all(
                                not getattr(screen, contract[3]).library_open
                                for contract in DESTINATION_CONTRACT.values()
                            )
                            and screen._library_reader_durable_generations["library"]
                            > library_generation
                            and not screen._library_reader_durable_preferences[
                                "library"
                            ]
                        ),
                        message="Shared durable Library preference did not close",
                    )
                elif destination == "notes":
                    body = screen.query_one("#library-note-body", TextArea)
                    body.text = "Closeout route draft"
                    screen.query_one("#library-note-preview", Button).press()
                elif destination == "prompts":
                    name = screen.query_one("#library-prompt-name", Input)
                    name.value = f"{name.value} route draft"
                    await _wait_for_selector(
                        screen,
                        pilot,
                        "#library-prompt-mode-info",
                    )
                    screen.query_one("#library-prompt-mode-info", Button).press()
                    await _wait_for_condition(
                        pilot,
                        lambda: screen._library_prompt_editor_mode == "info",
                        message="Prompt Info mode did not settle",
                    )
                elif destination == "skills":
                    screen.query_one("#library-skill-mode-edit", Button).press()
                    await _wait_for_condition(
                        pilot,
                        lambda: screen._library_skill_reader_mode == "edit",
                        message="Skills Edit mode did not settle",
                    )
                if destination == "notes":
                    shell.items_grip.press()
                    await _wait_for_condition(
                        pilot,
                        lambda: not screen._library_notes_reader_preferences.items_open,
                        message="Notes Items preference did not close",
                    )
                    expected_items["notes"] = False
                if destination == "conversations":
                    stale_service = _OutOfOrderConversationService()
                    stale_first_receipt = threading.Event()
                    original_conversation_read = (
                        stale_service.get_library_conversation_messages
                    )

                    def read_with_first_receipt(conversation_id: str, **kwargs):
                        is_first = not stale_service.calls
                        try:
                            return original_conversation_read(conversation_id, **kwargs)
                        finally:
                            if is_first:
                                stale_first_receipt.set()

                    stale_service.get_library_conversation_messages = (
                        read_with_first_receipt
                    )
                    app.local_chat_conversation_service = stale_service
                    rows = list(screen.query(".library-conversation-row"))
                    assert len(rows) >= 2
                    rows[0].press()
                    await _wait_for_condition(
                        pilot,
                        stale_service.first_started.is_set,
                        message="Stale Conversation A worker did not start",
                    )
                    current_rows = list(screen.query(".library-conversation-row"))
                    assert len(current_rows) >= 2
                    current_rows[1].press()
                    second_id = str(current_rows[1].conversation_id)
                    stale_target_id = second_id
                    await _wait_for_condition(
                        pilot,
                        lambda: (
                            screen._library_conversation_reader_state.selected_id
                            == second_id
                            and screen._library_conversation_reader_state.loaded_id
                            == second_id
                            and not screen._library_conversation_reader_state.loading
                        ),
                        message=lambda: (
                            "Conversation B did not win rapid A-to-B selection: "
                            f"state={screen._library_conversation_reader_state!r}; "
                            f"calls={stale_service.calls!r}"
                        ),
                    )
                remembered_focus[destination] = await _focus_closeout_work_via_f6(
                    screen, pilot, shell, destination
                )
                remembered[destination] = _destination_state(screen, destination)

            for destination in DESTINATIONS:
                shell = await _open_destination(screen, pilot, destination)
                if destination == "notes" and stale_service is not None:
                    stale_service.release_first.set()
                    await _wait_for_condition(
                        pilot,
                        stale_first_receipt.is_set,
                        message="Stale Conversation A service receipt did not settle",
                    )
                    await screen.workers.wait_for_complete()
                    await _wait_for_condition(
                        pilot,
                        lambda: (
                            stale_target_id is not None
                            and screen._library_selected_row_id == "browse-notes"
                            and screen._library_conversation_reader_state.selected_id
                            == stale_target_id
                            and screen._library_conversation_reader_state.loaded_id
                            == stale_target_id
                            and not screen._library_conversation_reader_state.loading
                        ),
                        message="Late Conversation worker escaped its route fence",
                    )
                current = _destination_state(screen, destination)
                assert current[3] == remembered[destination][3]
                assert current[5] == remembered[destination][5]
                assert len(screen.query(DESTINATION_CONTRACT[destination][1])) == 1
                assert (
                    sum(
                        len(screen.query(contract[1]))
                        for contract in DESTINATION_CONTRACT.values()
                    )
                    == 1
                )
                assert shell.work.is_mounted and shell.work.display
                assert {
                    getattr(screen, contract[3]).library_open
                    for contract in DESTINATION_CONTRACT.values()
                } == {False}
                assert not screen._library_reader_durable_preferences["library"]
                assert (
                    getattr(screen, DESTINATION_CONTRACT[destination][3]).items_open
                    is expected_items[destination]
                )
                restored_focus = await _focus_closeout_work_via_f6(
                    screen, pilot, shell, destination
                )
                assert restored_focus[0] == remembered_focus[destination][0] == "work"
                if destination == "notes":
                    assert screen.query_one("#library-note-body", TextArea).text == (
                        "Closeout route draft"
                    )
                elif destination == "prompts":
                    assert screen.query_one(
                        "#library-prompt-name", Input
                    ).value.endswith(" route draft")
            assert screen._library_notes_reader_preferences.items_open is False
            assert screen._library_media_reader_preferences.items_open is True
            assert (
                tuple(dict(note) for note in app.notes_scope_service.notes)
                == notes_before
            )
            assert app.notes_scope_service.save_calls == []
            assert tuple(prompt_db.get_prompt_by_id(index) for index in (1, 2)) == (
                prompts_before
            )
    finally:
        if stale_service is not None:
            stale_service.release_first.set()
        prompt_db.close()
