"""A Library resuming with nothing focused lands in Library, on EVERY route.

Review r11s Probe B and its follow-ups (r11n, r11n2): when a modal (F1 help,
the palette) closed over a Library with nothing focused, Textual's
``AUTO_FOCUS = "*"`` put focus on the nav bar's first tab, ``nav-home``,
where a blind Enter leaves Library. ``LibraryScreen.on_screen_resume``
suppresses that, and ``LibraryNavigationController.land_entry_focus_after_
resume`` lands the route's own entry, else the route's own rail row. The
per-route pins for the import, a list, Folder files and a slow Skills list
live beside the Escape pins in ``test_library_notes_import_escape_keeps_
running.py``; this file holds the hub's, a slow Collections page's, and the
two completeness checks: ``test_every_library_route_*`` walks every rail row
the shell state builds (and the hub), so a route the landing does not know
fails by name, and ``test_every_route_entry_selector_resolves_on_its_route``
keeps the controller's copied entry table honest.

Boots reuse that file's harness: the production-CSS Library over a real
ChaChaNotes database, the real F1 panel, real keys.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from Tests.UI.test_library_notes_import_escape_keeps_running import (
    HOLD,
    SIZE,
    _active_library_screen,
    _clear_focus,
    _click_focus,
    _f1_shortcuts,
    _press,
    _Profile,
    _until,
    _wait_for_library_shell,
)
from Tests.UI.test_library_prompts_canvas import (
    _open_prompt_editor,
    _real_prompt_scope_service,
)
from tldw_chatbook.Library.library_content_evidence import LibraryEvidenceStatus
from tldw_chatbook.Library.library_shell_state import build_library_shell_state
from tldw_chatbook.UI.Library_Modules import library_navigation_controller
from tldw_chatbook.UI.Library_Modules.library_artifacts_controller import (
    LibraryArtifactsController,
)
from tldw_chatbook.UI.Library_Modules.library_collections_controller import (
    LibraryCollectionsController,
)
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LIBRARY_ADAPTIVE_READER_GRIP_CLASS,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _focused_id(screen) -> str | None:
    return getattr(screen.focused, "id", None)


async def _select(screen, pilot, row: str) -> None:
    """Select ``row`` on the screen's own message pump, as a rail press does.

    Awaiting ``_select_library_rail_row`` from the test task ran its
    whole-screen recompose beside one the screen's pump started, and Textual
    raised ``DuplicateIds`` on ``screen-content`` (r11n4 load flake).
    """
    done = asyncio.Event()

    async def select() -> None:
        try:
            await screen._select_library_rail_row(row)
        finally:
            done.set()

    screen.call_next(select)
    await _until(pilot, done.is_set, f"{row or '(hub)'} to be selected")


# --- the hub: dev's landing-sync restore (review r11n2 N-3) -----------------


def _evidence_settled(screen, generation: int) -> bool:
    """The resume's onboarding-evidence read and its landing sync are done.

    Each resume starts one evidence read (a generation bump); its result
    syncs the landing, which clears focus and restores it after the
    landing's recompose -- the write this pair of pins is about.
    """
    if screen._library_onboarding_generation <= generation:
        return False
    if screen._library_onboarding_status is LibraryEvidenceStatus.LOADING:
        return False
    if any(
        worker.group == "library_onboarding_evidence" and not worker.is_finished
        for worker in screen.workers
    ):
        return False
    landing = screen.query("#library-landing-canvas")
    return not (landing and landing.first().has_pending_recompose_callback)


async def _hub(tmp_path, monkeypatch, focus_selector: str | None) -> str | None:
    """Boot on the hub, focus one action (or nothing), F1 open and close."""
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            assert screen._library_selected_row_id == ""
            await _until(
                pilot,
                lambda: _evidence_settled(screen, 0),
                "the boot's onboarding evidence to settle",
            )
            if focus_selector is None:
                await _clear_focus(screen, pilot)
            else:
                screen.set_focus(screen.query_one(focus_selector))
                await pilot.pause()
                assert _focused_id(screen) == focus_selector.lstrip("#")
            generation = screen._library_onboarding_generation
            await _f1_shortcuts(host, screen, pilot)
            await _until(
                pilot,
                lambda: _evidence_settled(screen, generation),
                "the resume's onboarding evidence to settle",
            )
            return _focused_id(screen)
    finally:
        profile.close()


async def test_f1_closing_over_the_unfocused_hub_lands_its_arrival_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hub lands where Esc from Ingest does, and it holds past the sync.

    Before the r11n fix the resume's evidence sync looked the landing's own
    action up in the rail only and moved focus to the rail's Import row.
    """
    landed = await _hub(tmp_path, monkeypatch, None)
    assert landed == "library-hub-action-import", landed


async def test_f1_closing_over_a_focused_hub_action_keeps_that_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A focused landing action survives any landing sync, not only a resume.

    Dev behaviour changed here (review r11n2 N-3): every landing sync with a
    hub action focused used to send focus to the rail's Import row. A second
    action, so "restored in place" cannot pass as "re-landed on the default".
    """
    landed = await _hub(tmp_path, monkeypatch, "#library-hub-action-new-note")
    assert landed == "library-hub-action-new-note", landed


# --- Collections whose capture page outlives the entry window (r11n2 N-1) --


async def test_f1_closing_over_a_slow_collections_page_lands_its_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collections' re-check lands its own entry, and Enter stays there.

    The copied entry ``#library-collection-name-input`` matched nothing, so
    the 2.1 s re-check fell to the rail's Import button and a blind Enter
    left for Ingest.
    """
    profile = _Profile(tmp_path, monkeypatch)
    page = threading.Event()
    hold = {"on": False}
    load = LibraryCollectionsController._load_library_collections_capture_entry

    async def held_load(self, *args, **kwargs):
        if hold["on"]:
            await asyncio.to_thread(page.wait, HOLD)
        return await load(self, *args, **kwargs)

    monkeypatch.setattr(
        LibraryCollectionsController,
        "_load_library_collections_capture_entry",
        held_load,
    )
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _press(host, screen, pilot, "#library-row-browse-collections")
            await _until(
                pilot,
                lambda: _focused_id(screen) == "library-collections-quick-capture",
                lambda: f"Collections' arrival, found {_focused_id(screen)!r}",
            )
            hold["on"] = True
            screen._disarm_library_list_entry_focus()
            await _clear_focus(screen, pilot)
            await _f1_shortcuts(host, screen, pilot)
            await _until(
                pilot,
                lambda: not screen._library_pending_list_entry_focus
                and screen.focused is not None,
                lambda: f"the re-check to land, found {_focused_id(screen)!r}",
            )
            assert _focused_id(screen) in {
                "library-collections-quick-capture",
                "library-collections-filter",
            }, _focused_id(screen)
            await pilot.press("enter")
            await pilot.pause()
            assert screen._library_selected_row_id == "browse-collections"
            assert host.screen is screen, host.screen_stack
    finally:
        page.set()
        profile.close()


# --- an open prompt, where the rail row's fresh entry closes it -----------


async def test_the_last_resort_over_an_open_prompt_lands_its_name_not_the_rail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the filter out of reach, an open prompt keeps its own field.

    A same-row rail press is a fresh entry: on a clean open prompt it closes
    the prompt (editor -> list), so the rail row is no safe landing there.
    The filter is made unreachable (a dead entry selector) to reach the step.
    """
    monkeypatch.setitem(
        library_navigation_controller._ROUTE_ENTRY_SELECTORS,
        "browse-prompts",
        "#library-no-such-entry",
    )
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _message = db.add_prompt(
        name="Release assistant", author="Ada", details="Prepares release notes"
    )
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    host.app_instance.prompt_scope_service = service
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            assert screen._prompts_state.view == "editor"
            assert not screen._prompts_state.dirty
            screen._disarm_library_list_entry_focus()
            await _clear_focus(screen, pilot)
            screen._navigation_controller._land_content_focus_if_unfocused()
            assert _focused_id(screen) == "library-prompt-name", _focused_id(screen)
            await pilot.press("enter")
            await pilot.pause()
            assert screen._library_selected_row_id == "browse-prompts"
            assert screen._prompts_state.view == "editor"
    finally:
        profile.close()


# --- an entry a recompose is removing (review r11n3 N3-1) -----------------


async def test_the_last_resort_moves_past_an_entry_being_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused write onto the route's entry is no landing; the row lands.

    ``BaseAppScreen.set_focus`` refuses a write onto a widget a recompose is
    removing (``_pruning``). The last resort reported that write as landed
    and stopped, so focus stayed on None for good.
    """
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _select(screen, pilot, "browse-conversations")
            await _until(
                pilot,
                lambda: bool(screen.query("#library-conversations-filter")),
                "the Conversations filter",
            )
            await pilot.pause()
            screen._disarm_library_list_entry_focus()
            await _clear_focus(screen, pilot)
            entry = screen.query_one("#library-conversations-filter")
            entry._pruning = True  # what ``App._prune`` marks while removing it
            try:
                screen._navigation_controller._land_content_focus_if_unfocused()
            finally:
                entry._pruning = False
            landed = _focused_id(screen)
            assert landed == "library-row-browse-conversations", landed
    finally:
        profile.close()


async def test_the_re_check_takes_focus_back_from_a_reset_parked_grip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grip Textual's reset parked focus on is no landing; a clicked one is.

    r11n4 load flake: Media's resume landed its empty-page Import, the
    page's reload removed it, and Textual's reset parked focus on the Library
    pane grip, where the re-check stood down because focus was not None.
    """
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _select(screen, pilot, "browse-media")
            await _until(
                pilot,
                lambda: bool(screen.query("#library-browse-library-grip")),
                "Media's Library pane grip",
            )
            await pilot.pause()
            screen._disarm_library_list_entry_focus()
            grip = screen.query_one("#library-browse-library-grip")
            controller = screen._navigation_controller
            screen.set_focus(grip)  # what Textual's reset does
            controller._land_content_focus_if_unfocused()
            landed = _focused_id(screen)
            assert landed == "library-media-empty-import", landed
            _click_focus(screen, grip)
            controller._land_content_focus_if_unfocused()
            assert screen.focused is grip, _focused_id(screen)
    finally:
        profile.close()


async def test_the_re_check_stays_on_the_route_it_was_armed_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A list route's re-check never takes a grip on the route switched to.

    Review r11n4 N4-1 (PROBE-B): F1 closing over an unfocused Conversations
    armed the 2.1 s re-check; inside the window the user switched to
    Artifacts, collapsed its Library pane and pressed F6 onto that pane's
    grip. F6 records no click or Tab, so the re-check read the grip as
    Textual's reset and moved focus to the Artifacts list. The re-check is
    held here and fired once the user has moved on, as the timer fires.
    """

    async def keep_in_memory(self, *args) -> None:
        return None  # the pane choice stays in app_config; nothing on disk

    monkeypatch.setattr(LibraryArtifactsController, "_persist", keep_in_memory)
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _press(host, screen, pilot, "#library-row-browse-conversations")
            await _until(
                pilot,
                lambda: bool(screen.query("#library-conversation-row-0")),
                "the Conversations rows",
            )
            await pilot.pause()
            screen._disarm_library_list_entry_focus()
            await _clear_focus(screen, pilot)
            rechecks: list[object] = []
            set_timer = screen.set_timer
            recheck_delay = library_navigation_controller._RECHECK_AFTER_ARM_SECONDS

            def hold_recheck(delay, callback=None, *args, **kwargs):
                if delay == recheck_delay:
                    rechecks.append(callback)
                    return set_timer(HOLD, lambda: None)
                return set_timer(delay, callback, *args, **kwargs)

            monkeypatch.setattr(screen, "set_timer", hold_recheck)
            await _f1_shortcuts(host, screen, pilot)
            monkeypatch.setattr(screen, "set_timer", set_timer)
            assert rechecks, "the resume armed no re-check"
            await _press(host, screen, pilot, "#library-row-artifacts-all")
            await _until(
                pilot,
                lambda: (c := screen._artifacts_controller) is not None
                and c.shell is not None
                and c.shell.is_mounted,
                "the Artifacts reader shell",
            )
            controller = screen._artifacts_controller
            if controller.layout.library_open:
                controller.toggle_pane("library")
            grip_id = "library-artifacts-library-grip"
            await _until(
                pilot,
                lambda: any(w.id == grip_id for w in screen.focus_chain),
                "the collapsed Library pane's grip",
            )
            for _ in range(4):  # F6 by its action, as the key runs it
                if _focused_id(screen) == grip_id:
                    break
                screen.action_focus_next_workbench_pane()
                await pilot.pause()
            assert _focused_id(screen) == grip_id, _focused_id(screen)
            await _until(
                pilot,
                lambda: not screen._library_pending_list_entry_focus,
                "the list-entry window to close, as it has when the timer fires",
            )
            for recheck in rechecks:
                recheck()
            await pilot.pause()
            assert _focused_id(screen) == grip_id, _focused_id(screen)
    finally:
        profile.close()


async def test_artifacts_never_syncs_a_page_into_a_shell_being_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page read that lands while a rail switch removes its shell is dropped.

    r11n4 load flake: switching between two Artifacts rows pruned the old
    reader shell while its page read was still current (same scope until the
    new shell attaches), and the read synced into the pruned children:
    ``NoMatches('#library-artifacts-heading')`` failed the worker.
    """
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _select(screen, pilot, "artifacts-all")
            controller = screen._artifacts_controller
            await _until(
                pilot,
                lambda: controller.shell is not None and controller.shell.is_mounted,
                "the Artifacts reader shell",
            )
            shell = controller.shell
            synced: list[bool] = []
            monkeypatch.setattr(shell, "sync", lambda: synced.append(True))
            shell._pruning = True  # what ``App._prune`` marks while removing it
            try:
                controller.sync()
            finally:
                shell._pruning = False
            assert not synced, "synced into a shell being removed"
            controller.sync()
            assert synced, "a live shell must still be synced"
    finally:
        profile.close()


# --- the copied entry table, and every route ------------------------------


async def test_every_route_entry_selector_resolves_on_its_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each row's entry in the controller's copied table is a live control.

    Review r11n2 item 3: the table is a second hand-kept copy; a selector
    that matches nothing silently falls through to rail chrome.
    """
    table = library_navigation_controller._ROUTE_ENTRY_SELECTORS
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    dead: list[str] = []
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            controller = screen._navigation_controller
            for row, selector in table.items():
                await _select(screen, pilot, row)
                await _until(
                    pilot, lambda row=row: screen._library_selected_row_id == row, row
                )
                await pilot.pause()
                screen._disarm_library_list_entry_focus()
                screen.set_focus(None)
                landed = controller._land_route_entry()
                landed_on = _focused_id(screen)
                if not landed or landed_on != selector.lstrip("#"):
                    dead.append(f"{row or '(hub)'}: {selector} -> {landed_on!r}")
    finally:
        profile.close()
    assert not dead, dead


#: What arriving on each route lands, owned here, NOT read from the
#: controller's table: an id, or a prefix ending in "-" for list rows. A
#: route that declares one must land it; only a route that declares none
#: may land its own rail row (the safe fallback, where Enter re-selects it).
_DECLARED_ENTRY = {
    "": ("library-hub-action-import",),
    "browse-media": (
        "library-media-empty-import",
        "library-media-type-filter",
        "library-media-filter",
        "library-media-row-",
    ),
    "browse-conversations": (
        "library-conversation-row-",
        "library-conversations-filter",
    ),
    "browse-notes": ("library-notes-filter", "library-notes-new", "library-notes-row-"),
    # Sort is Prompts' own "stable sort control": its page reload, started
    # by the resume while nothing was focused, restores there after the
    # landing when it finishes last (``_restore_library_prompts_focus``).
    "browse-prompts": (
        "library-prompts-filter",
        "library-prompt-row-",
        "library-prompts-sort",
    ),
    "browse-skills": ("library-skills-filter", "library-skill-row-"),
    "browse-collections": (
        "library-collections-quick-capture",
        "library-collections-filter",
    ),
    "browse-search": ("library-rag-query-input",),
    "artifacts-all": ("library-artifacts-list", "library-artifacts-search"),
    "artifacts-chatbooks": ("library-artifacts-list", "library-artifacts-search"),
    "artifacts-reports": ("library-artifacts-list", "library-artifacts-search"),
    "artifacts-dreams": ("library-artifacts-list", "library-artifacts-search"),
    "create-note": ("library-notes-create-blank",),
    "create-prompt": ("library-prompt-name",),
    "create-skill": ("library-skill-name",),
    # The study hand-offs' one control opens Study outside Library, so a
    # blind Enter there would leave: they land their own rail row only.
    "create-study": (),
    "create-flashcards": (),
    "create-quizzes": (),
    "ingest-import-media": ("library-ingest-path",),
    "ingest-export": ("library-export-name",),
}
#: A blind Enter on these routes' arrival controls -- the very controls a
#: rail press (or Esc from Ingest, for the hub) lands -- runs that control,
#: which moves the selection inside Library: the hub's and Media's empty-page
#: Import open Ingest, Create ▸ Blank note opens the new note on Notes.
_ENTER_MOVES_TO = {
    "": {"ingest-import-media"},
    "browse-media": {"ingest-import-media"},
    "create-note": {"browse-notes"},
}
#: The walk is split four ways so each census node stays inside the lane's
#: per-entry budget (about 7-10 s each); a literal so the census resolves it.
_GROUPS = (0, 1, 2, 3)


def _declares(row: str, widget_id: str | None) -> bool:
    if not widget_id:
        return False
    return any(
        widget_id.startswith(entry) if entry.endswith("-") else widget_id == entry
        for entry in _DECLARED_ENTRY.get(row, ())
    )


async def _walk_routes(tmp_path, monkeypatch, group: int) -> list[str]:
    """Resume over nothing focused on every row of one group; report misses.

    Each row is judged only after every landing check the resume queued has
    run: its own, and any re-check it set a timer for after the entry
    window. Prompts gets a real prompt source, so its page loads (without
    one its failed load rebuilds the canvas after the landing, a drift of
    its own that is not this walk's subject).
    """
    navigation = library_navigation_controller.LibraryNavigationController
    checks: list[str] = []
    rechecks: list[float] = []
    last_resort = navigation._land_content_focus_if_unfocused

    def counted(self, *args) -> None:
        checks.append(self.screen._library_selected_row_id)
        last_resort(self, *args)

    monkeypatch.setattr(navigation, "_land_content_focus_if_unfocused", counted)
    db, prompts = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Release assistant", author="Ada", details="Release notes")
    profile = _Profile(tmp_path, monkeypatch)
    host = profile.host()
    host.app_instance.prompt_scope_service = prompts
    misses: list[str] = []
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            set_timer = screen.set_timer

            def recorded_set_timer(delay, callback=None, *args, **kwargs):
                # The re-check is bound to its row (r11n4 N4-1): a partial.
                bound = getattr(callback, "func", callback)
                if getattr(bound, "__func__", None) is counted:
                    rechecks.append(delay)
                return set_timer(delay, callback, *args, **kwargs)

            monkeypatch.setattr(screen, "set_timer", recorded_set_timer)
            shell = build_library_shell_state(screen._build_library_shell_input())
            rows = [""]
            rows += [row.row_id for section in shell.sections for row in section.rows]
            # The source of truth must stay covered by this table too.
            assert set(rows) <= set(_DECLARED_ENTRY), set(rows) - set(_DECLARED_ENTRY)
            for row in rows[group :: len(_GROUPS)]:
                await _select(screen, pilot, row)
                await _until(
                    pilot, lambda row=row: screen._library_selected_row_id == row, row
                )
                await pilot.pause()
                screen._disarm_library_list_entry_focus()
                await _clear_focus(screen, pilot)
                before = len(checks) - len(rechecks)
                await _f1_shortcuts(host, screen, pilot)
                await _until(
                    pilot,
                    lambda before=before: len(checks) - len(rechecks) > before
                    and not screen._library_pending_list_entry_focus,
                    lambda row=row: f"{row or '(hub)'}: the landing checks to run",
                )
                await pilot.pause()
                focused = screen.focused
                if focused is None:
                    misses.append(f"{row or '(hub)'}: nothing focused")
                    continue
                widget_id = _focused_id(screen)
                rail = screen._active_library_rail()
                in_content = any(
                    node.id == "screen-content" for node in focused.ancestors
                )
                in_rail = rail is not None and rail in focused.ancestors_with_self
                own_row = widget_id == f"library-row-{row}"
                # Review r11n3 N3-3: only a list's Items-pane grip (the round-2
                # reload race); the Library ("Nav") pane grip is rail chrome.
                grip = (
                    focused.has_class(LIBRARY_ADAPTIVE_READER_GRIP_CLASS)
                    and getattr(focused, "pane", None) == "items"
                    and row in library_navigation_controller._LIST_ENTRY_ROWS
                )
                name = row or "(hub)"
                if not in_content:
                    misses.append(f"{name}: {widget_id!r} is outside Library")
                    continue
                if in_rail and not own_row:
                    misses.append(f"{name}: {widget_id!r} is rail chrome")
                    continue
                landed = _declares(row, widget_id) or grip
                if not (landed or (own_row and not _DECLARED_ENTRY[row])):
                    misses.append(f"{name}: {widget_id!r} is not its entry")
                    continue
                await pilot.press("enter")
                await pilot.pause()
                after = screen._library_selected_row_id
                if after != row and after not in _ENTER_MOVES_TO.get(row, set()):
                    misses.append(f"{name}: Enter on {widget_id!r} moved to {after!r}")
                if host.screen is not screen:
                    misses.append(f"{name}: Enter on {widget_id!r} left Library")
                    break
                print(f"resume landing: {name} -> {widget_id}")
    finally:
        profile.close()
    return misses


@pytest.mark.parametrize("group", _GROUPS)
async def test_every_library_route_lands_its_entry_or_own_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, group: int
) -> None:
    """EVERY rail row the shell builds (and the hub): F1 over nothing focused.

    Focus must land (never None), inside Library (never the nav bar), on the
    route's declared entry or its own rail row (never other rail chrome), and
    one blind Enter must keep the route.
    """
    misses = await _walk_routes(tmp_path, monkeypatch, group)
    assert not misses, misses
