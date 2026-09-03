"""Durable cross-destination adaptive-reader closeout regressions."""

from __future__ import annotations

import asyncio
import inspect
import operator
import threading
from pathlib import Path

import pytest
from textual.containers import Vertical
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
from Tests.UI.test_destination_shells import StaticLibraryNotesScopeService
from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.Library.collections_capture_models import (
    CapturePageRequest,
    CaptureSaveRequest,
)
from tldw_chatbook.Notes.note_folder_models import (
    NoteFolder,
    NoteFolderChildPage,
    NoteFolderMembership,
    NotePlacementPage,
    NotePlacementRecord,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.Widgets.workbench_focus import _available_targets
from tldw_chatbook.config import load_settings


DESTINATIONS = ("media", "collections", "conversations", "notes", "prompts", "skills")
SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))

DESTINATION_CONTRACT = {
    "media": (
        "#library-row-browse-media",
        "#library-media-reader-shell",
        "#library-media-row-1",
        "_library_media_reader_preferences",
        "_library_media_reader_layout",
    ),
    "collections": (
        "#library-row-browse-collections",
        "#library-collections-reader-shell",
        "#library-collections-row-1",
        # Task 7: collections' own reader_preferences/reader_layout fields
        # moved to ``screen._collections_state.<field>`` -- same extra hop
        # as conversations' own entry below. Every call site reads these
        # two contract entries through ``operator.attrgetter`` instead of
        # plain ``getattr``, so it stays a passthrough for the other four,
        # not-yet-extracted, destinations' flat attribute names.
        "_collections_state.reader_preferences",
        "_collections_state.reader_layout",
    ),
    "conversations": (
        "#library-row-browse-conversations",
        "#library-conversations-reader-shell",
        "#library-conversation-row-1",
        # Task 9: conversations' own reader_preferences/reader_layout
        # fields moved to ``screen._conversations_state.<field>`` -- one
        # extra hop a bare ``getattr(screen, name)`` can't express. Every
        # call site below reads these two contract entries through
        # ``operator.attrgetter`` instead of plain ``getattr`` so it stays
        # a passthrough for the other four, not-yet-extracted, destinations'
        # flat attribute names.
        "_conversations_state.reader_preferences",
        "_conversations_state.reader_layout",
    ),
    "notes": (
        "#library-row-browse-notes",
        "#library-notes-reader-shell",
        "#library-notes-tree-note-2",
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


class _CloseoutSimpleTreeNotesService(StaticLibraryNotesScopeService):
    """Give the established two-note closeout fixture the current tree seam."""

    async def page_note_folder_children(self, **kwargs):
        return NoteFolderChildPage(
            folders=(),
            total_folders=0,
            start_offset=kwargs["offset"],
            previous_offset=None,
            next_offset=None,
        )

    async def page_note_placements(self, **kwargs):
        offset = kwargs["offset"]
        records = tuple(
            NotePlacementRecord(note=dict(note), folder_id=None, membership=None)
            for note in self.notes[offset : offset + kwargs["limit"]]
        )
        return NotePlacementPage(
            placements=records,
            total_placements=len(self.notes),
            start_offset=offset,
            previous_offset=None,
            next_offset=None,
        )


class _CloseoutPagedNotesService(StaticLibraryNotesScopeService):
    """Bounded branch fixture for production-CSS geometry and focus evidence."""

    def __init__(self) -> None:
        notes = tuple(
            {
                "id": f"note-{index:02d}",
                "title": f"Long identifying Notes title {index:02d} " + "details " * 8,
                "content": "Closeout fixture",
                "version": 1,
                "created_at": "2026-08-29T00:00:00Z",
                "updated_at": "2026-08-29T00:00:00Z",
            }
            for index in range(70)
        )
        super().__init__(notes)
        self.fail_next_personal_page = False
        self.failure_entered = asyncio.Event()
        self.failure_release = asyncio.Event()
        self.gate_success = False
        self.success_entered = asyncio.Event()
        self.success_release = asyncio.Event()

    @staticmethod
    def _folder(parent_id: str | None, index: int) -> NoteFolder:
        folder_id = "personal" if parent_id is None and index == 0 else (
            f"root-{index:02d}" if parent_id is None else f"child-{index:02d}"
        )
        name = (
            "00 Personal research with a deliberately identifying long title"
            if folder_id == "personal"
            else f"{index:02d} Long identifying folder title " + "details " * 6
        )
        parent_path = "" if parent_id is None else "/Personal"
        path = f"{parent_path}/{name}"
        return NoteFolder(
            folder_id=folder_id,
            parent_id=parent_id,
            name=name,
            path=path,
            normalized_path=path.casefold(),
            version=1,
            deleted=False,
        )

    @staticmethod
    def _folder_page(parent_id: str | None, offset: int) -> NoteFolderChildPage:
        total = 25
        stop = min(offset + 20, total)
        folders = tuple(
            _CloseoutPagedNotesService._folder(parent_id, index)
            for index in range(offset, stop)
        )
        return NoteFolderChildPage(
            folders=folders,
            total_folders=total,
            start_offset=offset,
            previous_offset=None if offset == 0 else max(0, offset - 20),
            next_offset=stop if stop < total else None,
        )

    @staticmethod
    def _placement_page(
        parent_id: str | None, offset: int, *, total: int
    ) -> NotePlacementPage:
        stop = min(offset + 20, total)
        records = []
        for index in range(offset, stop):
            note_id = f"note-{index:02d}"
            membership = (
                NoteFolderMembership(
                    membership_id=f"membership-{index:02d}",
                    folder_id=parent_id,
                    note_id=note_id,
                    ownership="manual",
                    owner_id="",
                    owner_active=True,
                    version=1,
                )
                if parent_id is not None
                else None
            )
            records.append(
                NotePlacementRecord(
                    note={
                        "id": note_id,
                        "title": (
                            f"Long identifying Notes title {index:02d} "
                            + "details " * 8
                        ),
                        "content": "Closeout fixture",
                        "version": 1,
                    },
                    folder_id=parent_id,
                    membership=membership,
                )
            )
        return NotePlacementPage(
            placements=tuple(records),
            total_placements=total,
            start_offset=offset,
            previous_offset=None if offset == 0 else max(0, offset - 20),
            next_offset=stop if stop < total else None,
        )

    async def page_note_folder_children(self, **kwargs):
        return self._folder_page(kwargs["parent_id"], kwargs["offset"])

    async def page_note_placements(self, **kwargs):
        parent_id = kwargs["parent_id"]
        offset = kwargs["offset"]
        if parent_id == "personal" and offset == 20 and self.fail_next_personal_page:
            self.fail_next_personal_page = False
            self.failure_entered.set()
            await self.failure_release.wait()
            raise RuntimeError("one-shot closeout branch failure")
        if parent_id == "personal" and offset == 20 and self.gate_success:
            self.gate_success = False
            self.success_entered.set()
            await self.success_release.wait()
        return self._placement_page(
            parent_id,
            offset,
            total=25 if parent_id is None else 45,
        )


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
        "collections": (
            app.collections_capture_scope_service,
            ("list_page", "get_detail"),
        ),
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
    app.notes_scope_service = _CloseoutSimpleTreeNotesService(_two_notes())
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
    if destination == "collections":
        scope = screen.app_instance.collections_capture_scope_service
        authority = scope.active_authority
        assert authority is not None
        page = await scope.list_page(CapturePageRequest(authority.key))
        for index in range(page.total, 2):
            await scope.save_capture(
                CaptureSaveRequest(
                    authority.key,
                    f"https://example.test/closeout-{index + 1}",
                    title=f"Closeout capture {index + 1}",
                    text_content=f"Capture body {index + 1}",
                )
            )
    mounted_shells = [
        candidate
        for contract in DESTINATION_CONTRACT.values()
        for candidate in screen.query(contract[1])
    ]
    restore_closed_library = bool(
        mounted_shells
        and {
            operator.attrgetter(contract[3])(screen).library_open
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
                    operator.attrgetter(contract[3])(screen).library_open
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
    if destination == "collections":
        expected = str(second.capture_identity.capture_id)
    else:
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
        "collections": lambda: (
            screen._library_collections_capture_controller.state.selected_identity
            == second.capture_identity
            and screen._library_collections_capture_controller.state.loaded_detail
            is not None
            and screen._library_collections_capture_controller.state.loaded_detail.capture.identity
            == second.capture_identity
        ),
        "conversations": lambda: (
            str(screen._conversations_state.reader_state.selected_id or "")
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
    elif destination == "collections":
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_collections_capture_controller.state.selected_identity
                == second.capture_identity
                and screen._library_collections_capture_controller.state.loaded_detail
                is not None
                and screen._library_collections_capture_controller.state.loaded_detail.capture.identity
                == second.capture_identity
                and not screen._library_collections_capture_controller.state.detail_loading
            ),
            message="Collections second selection did not settle",
        )
        await _wait_for_selector(screen, pilot, "#library-collections-reader-title")
    elif destination == "conversations":
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._conversations_state.reader_state.selected_id == expected
                and screen._conversations_state.reader_state.loaded_id == expected
                and not screen._conversations_state.reader_state.loading
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
        # Collections repaints its reader while page/detail settlement lands;
        # always operate on the current shell after that transition.
        shell = screen.query_one(shell_selector)
        if not shell.effective_layout.library_open:
            shell.library_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: shell.effective_layout.library_open,
                message=f"Library pane did not reopen after {destination}",
            )
        generation = screen._library_reader_persistence_generations["library"]
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                {
                    operator.attrgetter(contract[3])(screen).library_open
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
    elif destination == "collections":
        state = screen._library_collections_capture_controller.state
        semantic = (
            state.selected_identity,
            state.loaded_detail.capture.identity if state.loaded_detail else None,
            screen._collections_state.reader_mode,
        )
    elif destination == "conversations":
        semantic = (
            screen._conversations_state.reader_state.selected_id,
            screen._conversations_state.reader_state.loaded_id,
            screen._conversations_state.reader_state.mode,
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
        operator.attrgetter(preferences_name)(screen),
        operator.attrgetter(layout_name)(screen),
        semantic,
    )


def _durable_live_oracle(
    screen,
    shell,
    destination: str,
    size: tuple[int, int],
    *,
    observations: dict[str, object],
) -> dict[str, object]:
    """Capture the bounded structured truth shared by durable live journeys."""
    preferences = operator.attrgetter(DESTINATION_CONTRACT[destination][3])(screen)
    layout = operator.attrgetter(DESTINATION_CONTRACT[destination][4])(screen)
    if destination == "media":
        state = screen._library_media_reader_session
        record = {
            "selected": state.selected_id,
            "pending": state.pending_request,
            "loaded": state.loaded_id,
            "mode": state.mode,
        }
    elif destination == "collections":
        state = screen._library_collections_capture_controller.state
        record = {
            "selected": (
                state.selected_identity.capture_id
                if state.selected_identity is not None
                else None
            ),
            "pending": (
                state.selected_identity.capture_id
                if state.detail_loading and state.selected_identity is not None
                else None
            ),
            "loaded": (
                state.loaded_detail.capture.identity.capture_id
                if state.loaded_detail is not None
                else None
            ),
            "mode": screen._collections_state.reader_mode,
        }
    elif destination == "conversations":
        state = screen._conversations_state.reader_state
        record = {
            "selected": state.selected_id,
            "pending": state.selected_id if state.loading else None,
            "loaded": state.loaded_id,
            "mode": state.mode,
        }
    elif destination == "notes":
        snapshot = screen._library_note_session.snapshot
        record = {
            "selected": screen._selected_note_id,
            "pending": (
                screen._selected_note_id
                if screen._library_note_load_state == "loading"
                else None
            ),
            "loaded": snapshot.note_id if snapshot is not None else None,
            "mode": (
                "context"
                if screen._library_note_context
                else "preview"
                if screen._library_note_preview
                else "edit"
            ),
        }
    elif destination == "prompts":
        record = {
            "selected": screen._selected_prompt_id,
            "pending": (
                screen._selected_prompt_id
                if screen._library_prompt_detail_loading
                else None
            ),
            "loaded": screen._library_prompt_loaded_id,
            "mode": screen._library_prompt_editor_mode,
        }
    else:
        state = screen._library_skill_editor_state
        record = {
            "selected": screen._selected_skill_name,
            "pending": (
                screen._selected_skill_name
                if screen._library_skill_detail_loading
                else None
            ),
            "loaded": state.name if state is not None else None,
            "mode": screen._library_skill_reader_mode,
        }
    regions = {
        name: {
            "x": widget.region.x,
            "y": widget.region.y,
            "width": widget.region.width,
            "height": widget.region.height,
        }
        for name, widget in {
            "library": shell.library,
            "items": shell.items,
            "work": shell.work,
        }.items()
    }
    return {
        "status": "PASS",
        "destination": destination,
        "final_destination": destination,
        "terminal_size": list(size),
        "contained": all(
            region["x"] >= 0
            and region["y"] >= 0
            and region["x"] + region["width"] <= size[0]
            and region["y"] + region["height"] <= size[1]
            for region in regions.values()
            if region["width"] and region["height"]
        ),
        "regions": regions,
        "identities": {
            "shell": shell.id,
            "items": shell.items.id,
            "work": shell.work.id,
        },
        "focus_owner": getattr(screen.focused, "id", None) or "work",
        "record": record,
        "preferences": {
            "requested_library_open": preferences.library_open,
            "requested_items_open": preferences.items_open,
            "effective_library_open": layout.library_open,
            "effective_items_open": layout.items_open,
        },
        "host_worker_groups": sorted(
            str(worker.group) for worker in screen.workers if not worker.is_finished
        ),
        "visible_controls": [
            button.id
            for button in shell.query(Button)
            if button.id and button.display and button.region.area
        ],
        "compositor_text": "\n".join(
            strip.text for strip in screen._compositor.render_strips()
        ),
        "cleanup_owner_counts": {},
        "observations": observations,
    }


def _durable_owner_counts(host, worker_baseline: set[int]) -> dict[str, int]:
    owned_workers = [
        worker for worker in host.workers if id(worker) not in worker_baseline
    ]
    owned_tasks = [
        task
        for worker in owned_workers
        if (task := getattr(worker, "_task", None)) is not None
    ]
    return {
        "host_workers_before": len(worker_baseline),
        "host_workers_owned": len(owned_workers),
        "host_worker_leaks": sum(not worker.is_finished for worker in owned_workers),
        "host_task_leaks": sum(not task.done() for task in owned_tasks),
        "host_thread_worker_leaks": sum(
            bool(getattr(worker, "_thread_worker", False)) and not worker.is_finished
            for worker in owned_workers
        ),
    }


def _assert_durable_owner_cleanup(
    host, worker_baseline: set[int], facts: dict[str, object]
) -> None:
    counts = _durable_owner_counts(host, worker_baseline)
    facts["cleanup_owner_counts"] = counts
    assert not any(
        counts[key]
        for key in (
            "host_worker_leaks",
            "host_task_leaks",
            "host_thread_worker_leaks",
        )
    )


async def _focus_closeout_work_via_f6(
    screen, pilot, shell, destination: str
) -> tuple[str, str]:
    """Reach the active Work region through the app-owned visible F6 route."""
    # Prompt/Skills mode changes may recompose their shell after
    # ``_open_destination`` returns. Resolve the currently mounted owner so
    # focus evidence never compares a live control with a stale shell object.
    shell = screen.query_one(DESTINATION_CONTRACT[destination][1])
    available = _available_targets(screen, screen._library_workbench_focus_targets())
    assert any(
        pane is shell.work or shell.work in pane.ancestors
        for pane, _target in available
    ), f"{destination} has no reachable Work focus target"
    focused = screen.focused
    assert focused is None or focused is screen or focused.parent is not None, (
        f"{destination} recompose left detached focus owner {focused!r}"
    )
    for _target in range(len(available) + 1):
        await pilot.press("f6")
        await pilot.pause()
        if screen.focused is shell or shell.work in screen.focused.ancestors:
            break
    assert screen.focused is shell or shell in screen.focused.ancestors
    assert screen.focused is shell or shell.work in screen.focused.ancestors
    return "work", str(screen.focused.id)


async def _exercise_closeout_resize_is_presentation_only(
    destination: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], str, str]:
    app, prompt_db = await _seed_closeout_app(tmp_path / destination)
    host = LibraryProductionCSSHarness(app)
    worker_baseline = {id(worker) for worker in host.workers}
    facts: dict[str, object] | None = None
    svg = ""
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
            resize_sequence = (*SIZES[1:], SIZES[0])
            for width, height in resize_sequence:
                await pilot.resize_terminal(width, height)
                await _wait_for_condition(
                    pilot,
                    lambda width=width, height=height: (
                        screen.size.width == width and screen.size.height == height
                    ),
                    message=f"Resize to {width}x{height} did not settle",
                )
                current = _destination_state(screen, destination)
                assert current[:3] == before[:3]
                assert current[5] == before[5]
                assert set(service_counts.values()) == {0}
                assert seam_counts == {
                    "config": 0,
                    "write": 0,
                    "worker": 0,
                    "poll": 0,
                }
            after = _destination_state(screen, destination)
            assert after[:3] == before[:3]
            assert after[5] == before[5]
            assert set(service_counts.values()) == {0}
            assert seam_counts == {"config": 0, "write": 0, "worker": 0, "poll": 0}
            shell = screen.query_one(DESTINATION_CONTRACT[destination][1])
            facts = _durable_live_oracle(
                screen,
                shell,
                destination,
                SIZES[0],
                observations={
                    "resize_sequence": [list(size) for size in resize_sequence],
                    "widget_identity_retained": after[:3] == before[:3],
                    "semantic_state_retained": after[5] == before[5],
                    "service_calls_during_resize": service_counts,
                    "config_write_worker_poll_calls": seam_counts,
                },
            )
            svg = host.export_screenshot(simplify=True)
        assert facts is not None
        _assert_durable_owner_cleanup(host, worker_baseline, facts)
    finally:
        prompt_db.close()
    assert facts is not None
    return facts, str(facts["compositor_text"]), svg


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", DESTINATIONS)
async def test_closeout_resize_is_presentation_only(
    destination: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, _compositor, _svg = await _exercise_closeout_resize_is_presentation_only(
        destination, tmp_path, monkeypatch
    )
    assert facts["terminal_size"] == [160, 50]
    assert facts["observations"]["resize_sequence"] == [
        [120, 35],
        [100, 30],
        [80, 24],
        [160, 50],
    ]


async def _exercise_closeout_preferences_restore_in_fresh_screen(
    tmp_path: Path,
) -> tuple[dict[str, object], str, str]:
    expected = {
        "media": True,
        "collections": True,
        "conversations": False,
        "notes": True,
        "prompts": False,
        "skills": True,
    }
    first_config = load_settings(force_reload=True)
    first_app, first_prompt_db = await _seed_closeout_app(tmp_path / "first")
    first_app.app_config = first_config
    first_host = LibraryProductionCSSHarness(first_app)
    first_worker_baseline = {id(worker) for worker in first_host.workers}
    first_cleanup: dict[str, int] | None = None
    try:
        async with first_host.run_test(size=(160, 50)) as pilot:
            screen = _active_library_screen(first_host)
            await _wait_for_library_shell(screen, pilot)
            shell = None
            for destination, items_open in expected.items():
                shell = await _open_destination(screen, pilot, destination)
                preferences = operator.attrgetter(DESTINATION_CONTRACT[destination][3])(screen)
                if preferences.items_open is items_open:
                    continue
                authority = f"{destination}_items"
                generation = screen._library_reader_persistence_generations[authority]
                shell.items_grip.press()
                await _wait_for_condition(
                    pilot,
                    lambda authority=authority, generation=generation, destination=destination, items_open=items_open: (
                        operator.attrgetter(DESTINATION_CONTRACT[destination][3])(screen).items_open
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
                        not operator.attrgetter(contract[3])(screen).library_open
                        for contract in DESTINATION_CONTRACT.values()
                    )
                    and screen._library_reader_persistence_generations["library"]
                    > library_generation
                    and screen._library_reader_durable_generations["library"]
                    > library_generation
                ),
                message="Mounted screen did not persist shared Library choice",
            )
        first_cleanup = _durable_owner_counts(first_host, first_worker_baseline)
        assert not any(
            first_cleanup[key]
            for key in (
                "host_worker_leaks",
                "host_task_leaks",
                "host_thread_worker_leaks",
            )
        )
    finally:
        first_prompt_db.close()
    assert first_cleanup is not None

    fresh_config = load_settings(force_reload=True)
    fresh_app, fresh_prompt_db = await _seed_closeout_app(tmp_path / "fresh")
    fresh_app.app_config = fresh_config
    host = LibraryProductionCSSHarness(fresh_app)
    worker_baseline = {id(worker) for worker in host.workers}
    facts: dict[str, object] | None = None
    svg = ""
    try:
        async with host.run_test(size=(160, 50)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            for destination, items_open in expected.items():
                shell = await _open_destination(screen, pilot, destination)
                preferences = operator.attrgetter(DESTINATION_CONTRACT[destination][3])(screen)
                assert preferences.library_open is False
                assert preferences.items_open is items_open
            facts = _durable_live_oracle(
                screen,
                shell,
                destination,
                (160, 50),
                observations={
                    "fresh_screen": True,
                    "requested_library_open": False,
                    "requested_items_open": expected,
                    "first_host_cleanup_owner_counts": first_cleanup,
                },
            )
            facts["destination"] = "all"
            svg = host.export_screenshot(simplify=True)
        assert facts is not None
        _assert_durable_owner_cleanup(host, worker_baseline, facts)
    finally:
        fresh_prompt_db.close()
    assert facts is not None
    return facts, str(facts["compositor_text"]), svg


@pytest.mark.asyncio
async def test_closeout_preferences_restore_in_fresh_screen(tmp_path: Path) -> None:
    (
        facts,
        _compositor,
        _svg,
    ) = await _exercise_closeout_preferences_restore_in_fresh_screen(tmp_path)
    cleanup = facts["observations"]["first_host_cleanup_owner_counts"]
    assert set(cleanup) == {
        "host_workers_before",
        "host_workers_owned",
        "host_worker_leaks",
        "host_task_leaks",
        "host_thread_worker_leaks",
    }
    assert cleanup["host_worker_leaks"] == 0
    assert cleanup["host_task_leaks"] == 0
    assert cleanup["host_thread_worker_leaks"] == 0


async def _exercise_closeout_single_app_route_cycle(
    tmp_path: Path,
) -> tuple[dict[str, object], str, str]:
    app, prompt_db = await _seed_closeout_app(tmp_path / "cycle")
    host = LibraryGlobalKeyProductionCSSHarness(app)
    worker_baseline = {id(worker) for worker in host.workers}
    remembered: dict[str, tuple[object, ...]] = {}
    remembered_focus: dict[str, tuple[str, str]] = {}
    revisit_receipts: dict[str, dict[str, object]] = {}
    expected_items = {destination: True for destination in DESTINATIONS}
    notes_before = tuple(dict(note) for note in app.notes_scope_service.notes)
    prompts_before = tuple(prompt_db.get_prompt_by_id(index) for index in (1, 2))
    stale_service: _OutOfOrderConversationService | None = None
    stale_target_id: str | None = None
    facts: dict[str, object] | None = None
    svg = ""
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
                                not operator.attrgetter(contract[3])(screen).library_open
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
                elif destination == "collections":
                    screen.query_one("#library-collections-mode-info", Button).press()
                    await _wait_for_condition(
                        pilot,
                        lambda: screen._collections_state.reader_mode == "info",
                        message="Collections Info mode did not settle",
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
                            screen._conversations_state.reader_state.selected_id
                            == second_id
                            and screen._conversations_state.reader_state.loaded_id
                            == second_id
                            and not screen._conversations_state.reader_state.loading
                        ),
                        message=lambda: (
                            "Conversation B did not win rapid A-to-B selection: "
                            f"state={screen._conversations_state.reader_state!r}; "
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
                            and screen._conversations_state.reader_state.selected_id
                            == stale_target_id
                            and screen._conversations_state.reader_state.loaded_id
                            == stale_target_id
                            and not screen._conversations_state.reader_state.loading
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
                    operator.attrgetter(contract[3])(screen).library_open
                    for contract in DESTINATION_CONTRACT.values()
                } == {False}
                assert not screen._library_reader_durable_preferences["library"]
                assert (
                    operator.attrgetter(DESTINATION_CONTRACT[destination][3])(screen).items_open
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
                oracle = _durable_live_oracle(
                    screen, shell, destination, (160, 50), observations={}
                )
                if destination == "notes":
                    draft = {
                        "dirty": True,
                        "retained_without_save": True,
                        "value": screen.query_one("#library-note-body", TextArea).text,
                    }
                elif destination == "prompts":
                    draft = {
                        "dirty": True,
                        "retained_without_save": True,
                        "value": screen.query_one("#library-prompt-name", Input).value,
                    }
                else:
                    draft = {
                        "dirty": False,
                        "retained_without_save": False,
                        "value": None,
                    }
                revisit_receipts[destination] = {
                    "preferences": oracle["preferences"],
                    "record": oracle["record"],
                    "focus": {
                        "region": restored_focus[0],
                        "owner": restored_focus[1],
                    },
                    "identities": oracle["identities"],
                    "draft": draft,
                    "worker_fenced": destination != "conversations",
                }
                if destination == "notes" and "conversations" in revisit_receipts:
                    revisit_receipts["conversations"]["worker_fenced"] = True
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
            facts = _durable_live_oracle(
                screen,
                shell,
                destination,
                (160, 50),
                observations={
                    "route_order": list(DESTINATIONS),
                    "shared_library_open": False,
                    "destination_items_open": expected_items,
                    "focus_regions": {
                        name: region
                        for name, (region, _owner) in remembered_focus.items()
                    },
                    "notes_draft_retained_without_save": True,
                    "prompt_draft_retained_without_save": True,
                    "late_conversation_worker_fenced": True,
                    "revisit_receipts": revisit_receipts,
                },
            )
            facts["destination"] = "all"
            svg = host.export_screenshot(simplify=True)
        assert facts is not None
        _assert_durable_owner_cleanup(host, worker_baseline, facts)
    finally:
        if stale_service is not None:
            stale_service.release_first.set()
        prompt_db.close()
    assert facts is not None
    return facts, str(facts["compositor_text"]), svg


@pytest.mark.asyncio
async def test_closeout_single_app_route_cycle(tmp_path: Path) -> None:
    facts, _compositor, _svg = await _exercise_closeout_single_app_route_cycle(tmp_path)
    receipts = facts["observations"]["revisit_receipts"]
    assert set(receipts) == set(DESTINATIONS)
    for destination, receipt in receipts.items():
        assert set(receipt) == {
            "preferences",
            "record",
            "focus",
            "identities",
            "draft",
            "worker_fenced",
        }
        assert receipt["record"]["pending"] is None
        assert receipt["focus"]["region"] == "work"
        assert receipt["identities"]["shell"] == (
            DESTINATION_CONTRACT[destination][1].removeprefix("#")
        )
        assert receipt["worker_fenced"] is True


def _assert_inside_items(items, widget) -> None:
    assert widget.region.x >= items.region.x and widget.region.right <= items.region.right, (
        widget.id,
        widget.region,
        items.region,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_notes_branch_paging_is_contained_focusable_and_collapsible_in_production_shell(
    size: tuple[int, int],
) -> None:
    """Notes paging keeps its source-owned controls sound in the shared shell."""
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(), notes=[])
    service = _CloseoutPagedNotesService()
    app.notes_scope_service = service
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                # Twenty real roots plus the virtual Unfiled branch.
                len(screen.query(".library-notes-folder-row")) == 21
                and len(screen.query(".library-notes-tree-note-row")) == 20
                and len(screen.query(".library-notes-tree-pager")) == 2
            ),
            message=lambda: (
                f"Initial Notes branches did not settle at {size}: "
                f"folders={len(screen.query('.library-notes-folder-row'))}, "
                f"notes={len(screen.query('.library-notes-tree-note-row'))}, "
                f"pagers={len(screen.query('.library-notes-tree-pager'))}, "
                f"branches={screen._library_notes_tree_branches!r}"
            ),
        )
        shell = screen.query_one("#library-notes-reader-shell")
        items = shell.items
        notes_list = screen.query_one("#library-notes-list", Vertical)
        initial_pagers = list(screen.query(".library-notes-tree-pager"))
        assert {
            (pager.content_kind, pager.parent_folder_id, pager.paging_action)
            for pager in initial_pagers
        } == {
            ("folders", None, "more"),
            ("placements", None, "more"),
        }
        assert items.region.width > 0 and items.region.height > 0
        _assert_inside_items(items, notes_list)
        for widget in (*screen.query(".library-notes-folder-row"), *initial_pagers):
            _assert_inside_items(items, widget)
        assert all(widget.region.right <= items.region.right for widget in initial_pagers)

        identifying = next(
            row
            for row in screen.query(".library-notes-tree-note-row")
            if getattr(row, "note_id", "") == "note-19"
        )
        notes_list.scroll_to_widget(identifying, animate=False, force=True)
        await pilot.pause()
        painted = " ".join(
            "\n".join(strip.text for strip in screen._compositor.render_strips()).split()
        )
        assert "Long identifying Notes" in painted
        assert identifying.region.right <= items.region.right

        notes_list.scroll_to(y=4, animate=False, force=True, immediate=True)
        screen.query_one("#library-notes-filter", Input).focus()
        await pilot.pause()
        retained_scroll = notes_list.scroll_y
        screen._request_library_notes_tree_slice(
            NotesBranchKey(None, "folders"), direction="more", offset=20
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                len(screen.query(".library-notes-folder-row")) == 26
                and not screen._library_notes_tree_branches[
                    NotesBranchKey(None, "folders")
                ].loading
            ),
            message=f"Root folder continuation did not settle at {size}",
        )
        assert notes_list.scroll_y == retained_scroll

        personal = next(
            row
            for row in screen.query(".library-notes-folder-row")
            if getattr(row, "folder_id", "") == "personal"
        )
        personal.press()
        await _wait_for_condition(
            pilot,
            lambda: {
                (pager.content_kind, pager.parent_folder_id, pager.paging_action)
                for pager in screen.query(".library-notes-tree-pager")
            }
            >= {
                ("folders", "personal", "more"),
                ("placements", "personal", "more"),
            },
            message=f"Expanded Notes branch controls did not settle at {size}",
        )
        await pilot.pause()
        placement_pager = next(
            pager
            for pager in screen.query(".library-notes-tree-pager")
            if pager.content_kind == "placements"
            and pager.parent_folder_id == "personal"
        )
        stable_pager_id = placement_pager.id
        service.fail_next_personal_page = True
        screen.set_focus(placement_pager, scroll_visible=True)
        assert screen.focused is placement_pager
        placement_pager.press()
        await _wait_for_condition(
            pilot,
            service.failure_entered.is_set,
            message=f"Notes one-shot failure did not enter at {size}",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.focused is not None
                and screen.focused.id == stable_pager_id
                and screen.focused.disabled
            ),
            message=f"Notes loading pager did not retain focus at {size}",
        )
        service.failure_release.set()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.query_one(f"#{stable_pager_id}").paging_action == "retry"
                and screen.focused is screen.query_one(f"#{stable_pager_id}")
            ),
            message=f"Notes Retry did not retain pager focus at {size}",
        )
        retry = screen.query_one(f"#{stable_pager_id}", Button)
        service.gate_success = True
        retry.press()
        await _wait_for_condition(
            pilot,
            service.success_entered.is_set,
            message=f"Notes retry success did not enter at {size}",
        )
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is not None and screen.focused.id == stable_pager_id,
            message=f"Notes retry loading state did not retain focus at {size}",
        )
        service.success_release.set()
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "note_id", "") == "note-20",
            message=f"Successful Notes Retry did not focus the first added row at {size}",
        )
        assert len(
            [
                row
                for row in screen.query(".library-notes-tree-note-row")
                if getattr(row, "folder_id", None) == "personal"
            ]
        ) == 40
        for widget in screen.query(
            ".library-notes-folder-row, .library-notes-tree-note-row, "
            ".library-notes-tree-pager"
        ):
            assert widget.region.right <= items.region.right

        if not shell.effective_layout.library_open:
            shell.library_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: shell.effective_layout.library_open,
                message=f"Library pane did not open at {size}",
            )
        items_before_library_collapse = items.region.width
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                not shell.effective_layout.library_open
                and shell.effective_layout.items_open
                and items.region.width > items_before_library_collapse
            ),
            message=lambda: (
                f"Library pane did not collapse at {size}: "
                    f"shell={shell.effective_layout!r}, "
                    f"prefs={screen._library_notes_reader_preferences!r}, "
                    f"items_region={items.region!r}, "
                    f"selected={screen._library_selected_row_id!r}, "
                    f"view={screen._library_notes_view!r}, "
                    f"stage={screen._library_notes_stage!r}, "
                    f"shell_region={shell.region!r}, "
                    f"durable={screen._library_reader_durable_preferences!r}, "
                    f"generations={screen._library_reader_persistence_generations!r}"
                ),
        )
        assert items.region.width > items_before_library_collapse

        work_width_before_items_collapse = shell.work.region.width
        shell.items_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: not shell.effective_layout.items_open,
            message=lambda: (
                f"Items pane did not collapse at {size}: "
                f"shell={shell.effective_layout!r}, "
                f"screen={screen._library_notes_reader_layout!r}, "
                f"display={shell.items.display!r}"
            ),
        )
        assert shell.items.display is False
        assert shell.work.region.width > work_width_before_items_collapse

        assert tuple(DESTINATION_CONTRACT) == DESTINATIONS


@pytest.mark.asyncio
async def test_notes_explicit_items_close_survives_reconcile_resize_and_library_toggle() -> None:
    """An intentional Items close remains authoritative across later layout work."""
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(), notes=[])
    app.notes_scope_service = _CloseoutPagedNotesService()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        shell = await _wait_for_selector(
            screen, pilot, "#library-notes-reader-shell"
        )
        if not screen._library_notes_reader_layout.library_open:
            shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_notes_reader_layout.items_open,
            message="Notes Items pane did not open initially",
        )

        shell.items_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen._library_notes_reader_preferences.items_open
                and not screen._library_notes_reader_layout.items_open
            ),
            message="Explicit Notes Items close did not settle",
        )

        screen._sync_library_notes_reader_layout_from_shell()
        await pilot.pause()
        assert screen._library_notes_reader_preferences.items_open is False
        assert screen._library_notes_reader_layout.items_open is False

        await pilot.resize_terminal(120, 35)
        await _wait_for_condition(
            pilot,
            lambda: screen.size == (120, 35),
            message="Notes resize did not settle",
        )
        assert screen._library_notes_reader_preferences.items_open is False
        assert screen._library_notes_reader_layout.items_open is False

        if not screen._library_notes_reader_layout.library_open:
            shell.library_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_notes_reader_layout.library_open,
                message="Notes Library pane did not open",
            )
        assert screen._library_notes_reader_layout.items_open is False

        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_notes_reader_layout.library_open,
            message="Notes Library pane did not close",
        )
        assert screen._library_notes_reader_preferences.items_open is False
        assert screen._library_notes_reader_layout.items_open is False


@pytest.mark.asyncio
async def test_notes_explicit_close_never_resolves_against_stale_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hysteresis reset cannot make a decision from transient geometry."""
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(), notes=[])
    app.notes_scope_service = _CloseoutPagedNotesService()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        shell = await _wait_for_selector(
            screen, pilot, "#library-notes-reader-shell"
        )
        if not screen._library_notes_reader_layout.library_open:
            shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_notes_reader_layout.library_open,
            message="Notes Library pane did not open initially",
        )
        before = screen._library_notes_reader_layout
        monkeypatch.setattr(
            screen,
            "_library_adaptive_reader_allocation_is_current",
            lambda _shell: False,
        )

        shell.library_grip.press()
        await pilot.pause()

        assert screen._library_notes_reader_preferences.library_open is False
        assert screen._library_notes_reader_layout == before
