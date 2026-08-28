"""Production-shaped live journeys for the TASK-23019 closeout."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static, TextArea

from Tests.Skills.test_skills_library_flow import (
    _real_skills_scope_service,
    _real_trust_service,
)
from Tests.UI.test_destination_shells import (
    _link_library_items_to_active_workspace,
)
from Tests.UI.test_library_adaptive_reader_closeout import (
    DESTINATION_CONTRACT,
    _exercise_closeout_preferences_restore_in_fresh_screen,
    _exercise_closeout_resize_is_presentation_only,
    _exercise_closeout_single_app_route_cycle,
    _open_destination,
    _seed_closeout_app,
)
from Tests.UI.test_library_conversation_reader import (
    _GatedFailureConversationService,
    _GatedFindRetryConversationService,
    _OutOfOrderConversationService,
    _ProgressiveConversationService,
)
from Tests.UI.test_library_prompts_reader import _structured_prompt_definition
from Tests.UI.test_library_shell import (
    LibraryGlobalKeyProductionCSSHarness,
    _active_library_screen,
    _bump_note_version_externally,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.workbench_focus import _available_targets

DESTINATIONS = ("media", "conversations", "notes", "prompts", "skills")
SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))


@dataclass
class ScenarioContext:
    """Own the one scratch evidence root supplied by the hermetic child."""

    root: Path
    cleanups: list[Any] = field(default_factory=list)

    @classmethod
    def from_environment(cls) -> "ScenarioContext":
        root = Path(os.environ["TASK23019_RAW_ROOT"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    def case_root(self, destination: str, terminal_size: tuple[int, int]) -> Path:
        width, height = terminal_size
        root = self.root / "cases" / f"{destination}-{width}x{height}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def add_cleanup(self, callback: Any) -> None:
        self.cleanups.append(callback)

    def capture(
        self,
        name: str,
        facts: dict[str, Any],
        compositor: str,
        svg: str,
    ) -> None:
        facts_root = self.root / "facts"
        captures_root = self.root / "captures"
        facts_root.mkdir(parents=True, exist_ok=True)
        captures_root.mkdir(parents=True, exist_ok=True)
        (facts_root / f"{name}.json").write_text(
            json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (captures_root / f"{name}.txt").write_text(compositor, encoding="utf-8")
        (captures_root / f"{name}.svg").write_text(svg, encoding="utf-8")

    def close(self) -> None:
        while self.cleanups:
            self.cleanups.pop()()


def _region(widget) -> dict[str, int]:
    region = widget.region
    return {
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height,
        "right": region.right,
        "bottom": region.bottom,
    }


def _host_owner_counts(host, worker_baseline: set[int]) -> dict[str, int]:
    """Count only workers and tasks created by this host after its baseline."""
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


def _capture_finished_case(
    context: ScenarioContext,
    name: str,
    host,
    worker_baseline: set[int],
    facts: dict[str, Any],
    compositor: str,
    svg: str,
) -> dict[str, Any]:
    """Capture one already-unmounted host after strict resource cleanup."""
    counts = _host_owner_counts(host, worker_baseline)
    facts["cleanup_owner_counts"] = counts
    assert not any(
        counts[key]
        for key in (
            "host_worker_leaks",
            "host_task_leaks",
            "host_thread_worker_leaks",
        )
    )
    context.capture(name, facts, compositor, svg)
    return facts


def _preferences(screen, destination: str):
    return getattr(screen, DESTINATION_CONTRACT[destination][3])


def _layout(screen, destination: str):
    return getattr(screen, DESTINATION_CONTRACT[destination][4])


def _identity_truth(screen, destination: str) -> dict[str, Any]:
    if destination == "media":
        state = screen._library_media_reader_session
        return {
            "selected": state.selected_id,
            "pending": getattr(state.pending_request, "requested_id", None),
            "loaded": state.loaded_id,
            "mode": state.mode,
        }
    if destination == "conversations":
        state = screen._library_conversation_reader_state
        return {
            "selected": state.selected_id,
            "pending": state.selected_id if state.loading else None,
            "loaded": state.loaded_id,
            "mode": state.mode,
        }
    if destination == "notes":
        snapshot = screen._library_note_session.snapshot
        mode = (
            "context"
            if screen._library_note_context
            else "preview"
            if screen._library_note_preview
            else "edit"
        )
        return {
            "selected": screen._selected_note_id,
            "pending": (
                screen._selected_note_id
                if screen._library_note_load_state == "loading"
                else None
            ),
            "loaded": snapshot.note_id if snapshot is not None else None,
            "mode": mode,
        }
    if destination == "prompts":
        return {
            "selected": screen._selected_prompt_id,
            "pending": (
                screen._selected_prompt_id
                if screen._library_prompt_detail_loading
                else None
            ),
            "loaded": screen._library_prompt_loaded_id,
            "mode": screen._library_prompt_editor_mode,
        }
    state = screen._library_skill_editor_state
    return {
        "selected": state.name,
        "pending": (
            screen._selected_skill_name
            if screen._library_skill_detail_loading
            else None
        ),
        "loaded": state.name if state is not None else None,
        "mode": screen._library_skill_reader_mode,
    }


async def _work_focus_target(screen, pilot, destination: str):
    candidates = {
        "media": (("#library-media-content-search", Input),),
        "conversations": (("#library-conversation-reader-find", Input),),
        "notes": (
            ("#library-note-title", Input),
            ("#library-note-preview-region", VerticalScroll),
        ),
        "prompts": (("#library-prompt-name", Input),),
        "skills": (("#library-skill-mode-overview", Button),),
    }[destination]

    def visible_target():
        for selector, widget_type in candidates:
            target = screen.query_one(selector, widget_type)
            if target.region.area > 0 and target.can_focus:
                return target
        return None

    await _wait_for_condition(
        pilot,
        lambda: visible_target() is not None,
        message=f"Work focus target did not settle: {destination}",
    )
    target = visible_target()
    assert target is not None
    return target


async def _exercise_modes(screen, pilot, destination: str) -> tuple[str, ...]:
    async def visible_button(selector: str) -> Button:
        await _wait_for_condition(
            pilot,
            lambda: (
                (button := screen.query_one(selector, Button)).region.area > 0
                and not button.disabled
            ),
            message=f"Visible mode control did not settle: {selector}",
        )
        return screen.query_one(selector, Button)

    controls: list[str] = []
    if destination == "media":
        modes = ("analysis", "highlights", "info", "read")
        current = screen._library_media_reader_session.mode
        for mode in (*[mode for mode in modes if mode != current], current):
            selector = f"#library-media-reader-select-{mode}"
            control = await visible_button(selector)
            control.press()
            await _wait_for_selector(
                screen, pilot, f"#library-media-reader-mode-{mode}"
            )
            controls.append(selector)
    elif destination == "conversations":
        for selector in (
            "#library-conversation-reader-info",
            "#library-conversation-reader-read",
        ):
            control = await visible_button(selector)
            control.press()
            await pilot.pause()
            controls.append(selector)
    elif destination == "notes":
        for selector in ("#library-note-preview", "#library-note-context"):
            control = await visible_button(selector)
            control.press()
            await pilot.pause()
            controls.append(selector)
        screen.query_one("#library-note-context-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_note_context,
            message="Note Info back control did not settle",
        )
        edit = await visible_button("#library-note-preview")
        edit.press()
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_note_preview,
            message="Note Edit control did not settle",
        )
        controls.append("#library-note-preview")
    elif destination == "prompts":
        for mode in ("advanced", "info", "basic"):
            selector = f"#library-prompt-mode-{mode}"
            control = await visible_button(selector)
            control.press()
            await pilot.pause()
            controls.append(selector)
    else:
        for mode in ("edit", "trust", "files", "overview"):
            selector = f"#library-skill-mode-{mode}"
            control = await visible_button(selector)
            control.press()
            await pilot.pause()
            controls.append(selector)
    return tuple(controls)


def _regions_do_not_intersect(regions: dict[str, dict[str, int]]) -> bool:
    painted = [region for region in regions.values() if region["width"]]
    return all(
        left["right"] <= right["x"] or right["right"] <= left["x"]
        for index, left in enumerate(painted)
        for right in painted[index + 1 :]
    )


def _visible_action_facts(shell) -> list[dict[str, Any]]:
    return [
        {
            "id": button.id,
            "label": str(button.label),
            "disabled": button.disabled,
            "region": _region(button),
        }
        for button in shell.query(Button)
        if button.region.area and button.display
    ]


def _selected_row_facts(screen, destination: str) -> dict[str, Any]:
    selector = {
        "media": ".library-media-row",
        "conversations": ".library-conversation-row",
        "notes": ".library-notes-row",
        "prompts": ".library-prompt-row",
        "skills": ".library-skill-row",
    }[destination]
    identity_name = {
        "media": "media_id",
        "conversations": "conversation_id",
        "notes": "note_id",
        "prompts": "prompt_id",
        "skills": "skill_name",
    }[destination]
    selected_id = str(_identity_truth(screen, destination)["selected"])
    selected = [
        row
        for row in screen.query(selector)
        if str(getattr(row, identity_name, "")) == selected_id
    ]
    assert len(selected) == 1
    row = selected[0]
    selected_class = {
        "media": "library-media-row-selected",
        "conversations": "library-conversation-row-selected",
        "notes": None,
        "prompts": None,
        "skills": "is-selected",
    }[destination]
    return {
        "identity": str(getattr(row, identity_name)),
        "widget_id": row.id,
        "selected": (
            row.has_class(selected_class) if selected_class is not None else True
        ),
        "region": _region(row),
    }


def _facts(
    screen,
    shell,
    destination: str,
    size: tuple[int, int],
    controls,
    f6_route,
):
    preferences = _preferences(screen, destination)
    layout = _layout(screen, destination)
    focus = screen.focused
    regions = {
        "library": _region(shell.library),
        "library_grip": _region(shell.library_grip),
        "items": _region(shell.items),
        "items_grip": _region(shell.items_grip),
        "work": _region(shell.work),
    }
    footer_shortcuts = tuple(screen._library_footer_shortcuts_for_current_state())
    return {
        "status": "PASS",
        "destination": destination,
        "final_destination": destination,
        "terminal_size": list(size),
        "contained": all(
            widget.region.x >= 0
            and widget.region.y >= 0
            and widget.region.right <= size[0]
            and widget.region.bottom <= size[1]
            for widget in (
                shell.library,
                shell.library_grip,
                shell.items,
                shell.items_grip,
                shell.work,
            )
            if widget.region.area
        ),
        "regions": regions,
        "regions_do_not_intersect": _regions_do_not_intersect(regions),
        "identities": {
            "shell": shell.id,
            "items": shell.items.id,
            "work": shell.work.id,
        },
        "preferences": {
            "requested_library_open": preferences.library_open,
            "requested_items_open": preferences.items_open,
            "requested_custom_widths_enabled": preferences.custom_widths_enabled,
            "requested_library_width": preferences.library_width,
            "requested_items_width": preferences.items_width,
            "effective_library_open": layout.library_open,
            "effective_items_open": layout.items_open,
            "effective_library_width": layout.library_width,
            "effective_items_width": layout.items_width,
            "effective_reader_width": layout.reader_width,
            "effective_priority_pane": layout.priority_pane,
        },
        "record": _identity_truth(screen, destination),
        "selected_row": _selected_row_facts(screen, destination),
        "focus_owner": getattr(focus, "id", None),
        "host_worker_groups": sorted(
            {str(worker.group) for worker in screen.workers if not worker.is_finished}
        ),
        "visible_controls": list(controls),
        "grips": {
            "library": {
                "id": shell.library_grip.id,
                "focusable": shell.library_grip.can_focus,
                "painted": bool(shell.library_grip.region.area),
            },
            "items": {
                "id": shell.items_grip.id,
                "focusable": shell.items_grip.can_focus,
                "painted": bool(shell.items_grip.region.area),
            },
        },
        "primary_actions": _visible_action_facts(shell),
        "footer_shortcuts": [list(shortcut) for shortcut in footer_shortcuts],
        "f6_route": list(f6_route),
        "compositor_text": "\n".join(
            strip.text for strip in screen._compositor.render_strips()
        ),
        "cleanup_owner_counts": {},
    }


async def _press_grip(screen, pilot, destination: str, pane: str, grip):
    authority = "library" if pane == "library" else f"{destination}_items"
    generation = screen._library_reader_persistence_generations[authority]
    grip.press()
    await _wait_for_condition(
        pilot,
        lambda: (
            screen._library_reader_persistence_generations[authority] > generation
            and screen._library_reader_durable_generations[authority] > generation
        ),
        message=f"{pane.title()} grip event did not settle",
    )
    preferences = _preferences(screen, destination)
    layout = _layout(screen, destination)
    return {
        "requested": getattr(preferences, f"{pane}_open"),
        "effective": getattr(layout, f"{pane}_open"),
    }


async def _exercise_grip(screen, pilot, destination: str, pane: str, grip):
    preferences = _preferences(screen, destination)
    layout = _layout(screen, destination)
    initial = {
        "requested": getattr(preferences, f"{pane}_open"),
        "effective": getattr(layout, f"{pane}_open"),
    }
    states = [initial, await _press_grip(screen, pilot, destination, pane, grip)]
    assert states[-1]["requested"] is (not initial["effective"])
    if initial["effective"]:
        assert states[-1] == {"requested": False, "effective": False}
        states.append(await _press_grip(screen, pilot, destination, pane, grip))
        assert states[-1]["requested"] is True
    elif states[-1]["effective"]:
        states.append(await _press_grip(screen, pilot, destination, pane, grip))
        assert states[-1] == {"requested": False, "effective": False}
        if initial["requested"]:
            states.append(await _press_grip(screen, pilot, destination, pane, grip))
            assert states[-1]["requested"] is True
    else:
        assert states[-1] == {"requested": True, "effective": False}
    return states


async def _exercise_f6(screen, pilot, destination: str) -> tuple[str | None, ...]:
    work = await _work_focus_target(screen, pilot, destination)
    work.focus()
    await _wait_for_condition(
        pilot, lambda: screen.focused is work, message="Work focus did not settle"
    )
    available = _available_targets(screen, screen._library_workbench_focus_targets())
    assert available, f"{destination} exposed no reachable F6 regions"
    work_index = next(
        (
            index
            for index, (pane, _target) in enumerate(available)
            if work is pane or pane in work.ancestors
        ),
        None,
    )
    assert work_index is not None, f"{destination} Work is absent from the F6 route"
    route = [f"{available[work_index][0].id}:{getattr(screen.focused, 'id', None)}"]
    for offset in range(1, len(available) + 1):
        expected_index = (work_index + offset) % len(available)
        expected_pane, expected_target = available[expected_index]
        await pilot.press("f6")
        await _wait_for_condition(
            pilot,
            lambda expected_target=expected_target: screen.focused is expected_target,
            message=(
                f"Global F6 did not reach {destination} region {expected_pane.id}"
            ),
        )
        route.append(f"{expected_pane.id}:{getattr(screen.focused, 'id', None)}")
    assert {entry.partition(":")[0] for entry in route[1:]} == {
        pane.id for pane, _target in available
    }
    shortcuts = screen._library_footer_shortcuts_for_current_state()
    assert any(str(key).casefold() == "f6" for key, _label in shortcuts)
    return tuple(route)


async def run_common_cell(
    destination: str,
    terminal_size: tuple[int, int],
    context: ScenarioContext,
) -> dict[str, Any]:
    """Exercise one destination/terminal cell through real visible controls."""
    app, prompt_db = await _seed_closeout_app(
        context.case_root(destination, terminal_size)
    )
    context.add_cleanup(prompt_db.close)
    host = LibraryGlobalKeyProductionCSSHarness(app)
    worker_baseline = {id(worker) for worker in host.workers}
    facts: dict[str, Any]
    compositor = svg = ""
    try:
        async with host.run_test(size=terminal_size) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            shell = await _open_destination(screen, pilot, destination)
            identities = (id(shell), id(shell.items), id(shell.work))
            truth = _identity_truth(screen, destination)
            expected = _selected_row_facts(screen, destination)["identity"]
            assert truth["selected"] is not None
            assert str(truth["selected"]) == expected
            assert truth["pending"] is None
            assert str(truth["loaded"]) == expected
            controls = await _exercise_modes(screen, pilot, destination)
            assert _identity_truth(screen, destination)["selected"] == truth["selected"]
            f6_route = await _exercise_f6(screen, pilot, destination)
            focus_owner = await _work_focus_target(screen, pilot, destination)
            focus_owner.focus()
            await _wait_for_condition(
                pilot,
                lambda: screen.focused is focus_owner,
                message="Work focus did not restore after F6 route",
            )

            initial_items_width = shell.items.region.width
            initial_library = {
                "requested": _preferences(screen, destination).library_open,
                "effective": _layout(screen, destination).library_open,
            }
            library_states = [initial_library]
            if initial_library["effective"]:
                collapsed_library = await _press_grip(
                    screen, pilot, destination, "library", shell.library_grip
                )
                assert collapsed_library == {
                    "requested": False,
                    "effective": False,
                }
                library_states.append(collapsed_library)
                collapsed_items_width = shell.items.region.width
                restored_library = await _press_grip(
                    screen, pilot, destination, "library", shell.library_grip
                )
                assert restored_library["requested"] is True
                library_states.append(restored_library)
            else:
                assert initial_library == {
                    "requested": True,
                    "effective": False,
                }
                collapsed_items_width = initial_items_width
                restore_attempt = await _press_grip(
                    screen, pilot, destination, "library", shell.library_grip
                )
                assert restore_attempt["requested"] is True
                library_states.append(restore_attempt)
                if restore_attempt["effective"]:
                    collapsed_library = await _press_grip(
                        screen, pilot, destination, "library", shell.library_grip
                    )
                    assert collapsed_library == {
                        "requested": False,
                        "effective": False,
                    }
                    library_states.append(collapsed_library)
                    collapsed_items_width = shell.items.region.width
                    restored_library = await _press_grip(
                        screen, pilot, destination, "library", shell.library_grip
                    )
                    assert restored_library["requested"] is True
                    library_states.append(restored_library)
                else:
                    assert restore_attempt == initial_library
            if terminal_size == (160, 50):
                assert collapsed_items_width > initial_items_width
            items_states = await _exercise_grip(
                screen, pilot, destination, "items", shell.items_grip
            )

            assert (id(shell), id(shell.items), id(shell.work)) == identities
            restored_truth = _identity_truth(screen, destination)
            assert restored_truth["selected"] == truth["selected"]
            assert restored_truth["pending"] is None
            assert restored_truth["loaded"] == truth["loaded"]
            assert screen.focused is focus_owner
            assert shell.work.is_mounted and shell.work.display
            assert shell.library_grip.region.area and shell.items_grip.region.area
            assert shell.library_grip.can_focus and shell.items_grip.can_focus
            facts = _facts(
                screen, shell, destination, terminal_size, controls, f6_route
            )
            facts["items_comfort_expansion"] = {
                "before": initial_items_width,
                "while_library_collapsed": collapsed_items_width,
            }
            facts["restoration_paths"] = {
                "library": library_states,
                "items": items_states,
            }
            assert facts["contained"] and facts["regions_do_not_intersect"]
            assert facts["primary_actions"]
            if terminal_size == (80, 24):
                visible = screen._compositor.visible_widgets
                assert shell.library_grip in visible and shell.items_grip in visible
            compositor = facts["compositor_text"]
            svg = host.export_screenshot(simplify=True)
    finally:
        context.close()
    return _capture_finished_case(
        context,
        f"{destination}-{terminal_size[0]}x{terminal_size[1]}",
        host,
        worker_baseline,
        facts,
        compositor,
        svg,
    )


async def run_common_matrix() -> dict[str, dict[str, Any]]:
    context = ScenarioContext.from_environment()
    results = {}
    for destination in DESTINATIONS:
        for size in SIZES:
            name = f"{destination}-{size[0]}x{size[1]}"
            try:
                results[name] = await run_common_cell(destination, size, context)
            except Exception as error:
                results[name] = {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
    return results


def _capability_facts(
    screen,
    shell,
    destination: str,
    size: tuple[int, int],
    observations: dict[str, Any],
) -> dict[str, Any]:
    facts = _facts(
        screen,
        shell,
        destination,
        size,
        tuple(
            button.id
            for button in shell.query(Button)
            if button.id and button.display and button.region.area
        ),
        (),
    )
    facts["observations"] = observations
    return facts


async def _settled_capability_facts(
    screen,
    shell,
    pilot,
    destination: str,
    size: tuple[int, int],
    observations: dict[str, Any],
) -> dict[str, Any]:
    focus_target = await _work_focus_target(screen, pilot, destination)
    focus_target.focus()
    await _wait_for_condition(
        pilot,
        lambda: screen.focused is focus_target,
        message=f"Work focus did not settle: {focus_target.id}",
    )
    return _capability_facts(screen, shell, destination, size, observations)


async def run_media_capability() -> dict[str, Any]:
    """ME-01/ME-02: Find, progress/mode continuity, and bulk delete preview."""
    size = (160, 50)
    context = ScenarioContext.from_environment()
    app, prompt_db = await _seed_closeout_app(context.case_root("media", size))
    context.add_cleanup(prompt_db.close)
    host = LibraryGlobalKeyProductionCSSHarness(app)
    worker_baseline = {id(worker) for worker in host.workers}
    facts: dict[str, Any]
    compositor = svg = ""
    try:
        async with host.run_test(size=size) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            shell = await _open_destination(screen, pilot, "media")
            work_id = id(shell.work)
            selected = screen._library_media_reader_session.selected_id
            loaded = screen._library_media_reader_session.loaded_id
            assert selected == loaded and selected is not None

            search = screen.query_one("#library-media-content-search", Input)
            search.value = "Transcript"
            search.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: (
                    "Match 1 of"
                    in str(
                        screen.query_one(
                            "#library-media-content-search-status", Static
                        ).renderable
                    )
                ),
                message="Media Find did not settle an exact transcript match",
            )
            find_status = str(
                screen.query_one(
                    "#library-media-content-search-status", Static
                ).renderable
            )
            screen.query_one("#library-media-reader-select-analysis", Button).press()
            await _wait_for_selector(
                screen, pilot, "#library-media-reader-mode-analysis"
            )
            screen.query_one("#library-media-reader-select-read", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-reader-mode-read")
            assert screen._library_media_reader_session.selected_id == selected
            assert screen._library_media_reader_session.loaded_id == loaded

            service = app.media_reading_scope_service
            item_count = len(service.media_items)
            screen.query_one("#library-media-select-toggle", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-selected-count")
            selected_row = next(
                row
                for row in screen.query(".library-media-row")
                if str(getattr(row, "media_id", "")) == str(selected)
            )
            selected_row.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    "1 selected"
                    in str(
                        screen.query_one(
                            "#library-media-selected-count", Static
                        ).renderable
                    )
                ),
                message="Media bulk selection count did not settle",
            )
            screen.query_one("#library-media-delete-selected", Button).press()
            preview = await _wait_for_selector(
                screen, pilot, "#library-media-bulk-delete-confirm-copy"
            )
            preview_copy = str(preview.renderable)
            assert str(screen._library_media_reader_session.loaded_id) == str(selected)
            screen.query_one("#library-media-bulk-delete-cancel", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: not screen.query("#library-media-bulk-delete-confirm-copy"),
                message="Media delete preview did not cancel",
            )
            assert len(service.media_items) == item_count
            assert id(shell.work) == work_id
            screen.query_one("#library-media-select-toggle", Button).press()

            def capture_is_settled() -> bool:
                session = screen._library_media_reader_session
                matching_rows = [
                    row
                    for row in screen.query(".library-media-row")
                    if str(getattr(row, "media_id", "")) == str(selected)
                ]
                return bool(
                    not screen._library_media_select_mode
                    and session.selected_id == session.loaded_id == selected
                    and session.pending_request is None
                    and len(matching_rows) == 1
                    and matching_rows[0].is_mounted
                    and matching_rows[0].display
                    and matching_rows[0].region.area
                    and matching_rows[0].has_class("library-media-row-selected")
                )

            await _wait_for_condition(
                pilot,
                capture_is_settled,
                message="Media selection did not settle after leaving bulk mode",
            )
            observations = {
                "catalogue_ids": ["ME-01", "ME-02"],
                "find_status": find_status,
                "selected_loaded_identity": str(selected),
                "mode_after_round_trip": screen._library_media_reader_session.mode,
                "bulk_preview_copy": preview_copy,
                "bulk_selected_count": 1,
                "item_count_after_cancel": len(service.media_items),
                "destructive_boundary": "truthful_preview_cancelled",
            }
            facts = await _settled_capability_facts(
                screen, shell, pilot, "media", size, observations
            )
            compositor = facts["compositor_text"]
            svg = host.export_screenshot(simplify=True)
    finally:
        context.close()
    return _capture_finished_case(
        context,
        "media-capability",
        host,
        worker_baseline,
        facts,
        compositor,
        svg,
    )


async def run_conversations_capability() -> dict[str, Any]:
    """CO-01/CO-02: progressive Find, stale fencing, Info, and handoff."""
    size = (160, 50)
    context = ScenarioContext.from_environment()
    app, prompt_db = await _seed_closeout_app(context.case_root("conversations", size))
    context.add_cleanup(prompt_db.close)
    app.chat_conversation_scope_service.conversations = tuple(
        {
            **record,
            "version": 4,
        }
        if record["id"] == "chat-a"
        else record
        for record in app.chat_conversation_scope_service.conversations
    )
    _link_library_items_to_active_workspace(
        app,
        (
            ("conversation", "chat-a", "Alpha planning"),
            ("conversation", "chat-b", "Beta review"),
        ),
    )
    host = LibraryGlobalKeyProductionCSSHarness(app)
    worker_baseline = {id(worker) for worker in host.workers}
    facts: dict[str, Any]
    compositor = svg = ""
    try:
        async with host.run_test(size=size) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            shell = await _open_destination(screen, pilot, "conversations")
            selected = screen._library_conversation_reader_state.selected_id
            if selected == "chat-a":
                next(
                    row
                    for row in screen.query(".library-conversation-row")
                    if getattr(row, "conversation_id", None) == "chat-b"
                ).press()
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen._library_conversation_reader_state.loaded_id == "chat-b"
                        and screen._library_conversation_reader_state.complete
                    ),
                    message="Conversation setup did not settle chat-b",
                )
            progressive = _ProgressiveConversationService()
            app.local_chat_conversation_service = progressive
            selected_row = next(
                row
                for row in screen.query(".library-conversation-row")
                if getattr(row, "conversation_id", None) == "chat-a"
            )
            selected_row.press()
            await asyncio.to_thread(progressive.second_started.wait, 10)
            await _wait_for_condition(
                pilot,
                lambda: (
                    len(screen._library_conversation_reader_state.messages) == 20
                    and not screen._library_conversation_reader_state.complete
                ),
                message=lambda: (
                    "Conversation first transcript page did not paint: "
                    f"state={screen._library_conversation_reader_state!r}; "
                    f"calls={progressive.calls!r}"
                ),
            )
            partial = screen._library_conversation_reader_state
            find = screen.query_one("#library-conversation-reader-find", Input)
            find.value = "needle"
            find.focus()
            await pilot.press("enter")
            assert not screen._library_conversation_reader_state.find_complete
            progressive.release_second.set()
            await screen.workers.wait_for_complete()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_conversation_reader_state.complete
                    and bool(screen._library_conversation_reader_state.find_matches)
                ),
                message="Progressive Conversation Find did not complete",
            )
            progressive_state = screen._library_conversation_reader_state

            stale = _OutOfOrderConversationService()
            app.local_chat_conversation_service = stale
            current_id = screen._library_conversation_reader_state.selected_id
            rows_by_id = {
                str(row.conversation_id): row
                for row in screen.query(".library-conversation-row")
            }
            first_row = next(
                row for row_id, row in rows_by_id.items() if row_id != current_id
            )
            first_row.press()
            await asyncio.to_thread(stale.first_started.wait, 10)
            first_id = str(stale.calls[0]["conversation_id"])
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_conversation_reader_state.selected_id == first_id
                    and screen._library_conversation_reader_state.loading
                ),
                message="Conversation stale candidate did not become pending",
            )
            target_id = "chat-b" if first_id == "chat-a" else "chat-a"
            await _wait_for_condition(
                pilot,
                lambda: any(
                    getattr(row, "conversation_id", None) == target_id
                    for row in screen.query(".library-conversation-row")
                ),
                message="Conversation target row did not remount",
            )
            target_row = next(
                row
                for row in screen.query(".library-conversation-row")
                if getattr(row, "conversation_id", None) == target_id
            )
            target_row.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_conversation_reader_state.selected_id == target_id
                    and len(stale.calls) == 2
                    and stale.calls[1]["conversation_id"] == target_id
                ),
                message="Conversation target request did not start",
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_conversation_reader_state.selected_id == target_id
                    and screen._library_conversation_reader_state.loaded_id == target_id
                    and screen._library_conversation_reader_state.complete
                ),
                message="Conversation B did not settle ahead of stale A",
            )
            stale.release_first.set()
            await screen.workers.wait_for_complete()
            assert screen._library_conversation_reader_state.selected_id == target_id
            assert screen._library_conversation_reader_state.loaded_id == target_id

            retry_failure = _GatedFailureConversationService("unavailable")
            app.local_chat_conversation_service = retry_failure
            next(
                row
                for row in screen.query(".library-conversation-row")
                if getattr(row, "conversation_id", None) == target_id
            ).press()
            await asyncio.to_thread(retry_failure.started.wait, 10)
            retry_failure.release.set()
            await screen.workers.wait_for_complete()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_conversation_reader_state.selected_id == target_id
                    and bool(screen._library_conversation_reader_state.error)
                    and bool(screen.query("#library-conversation-reader-retry"))
                ),
                message=lambda: (
                    "Conversation injected detail failure exposed no Retry: "
                    f"state={screen._library_conversation_reader_state!r}"
                ),
            )
            retry_error = str(screen._library_conversation_reader_state.error)

            retry_service = _GatedFindRetryConversationService()
            app.local_chat_conversation_service = retry_service
            screen.query_one("#library-conversation-reader-retry", Button).press()
            await asyncio.to_thread(retry_service.started.wait, 10)
            retry_service.release.set()
            await screen.workers.wait_for_complete()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_conversation_reader_state.selected_id == target_id
                    and screen._library_conversation_reader_state.loaded_id == target_id
                    and screen._library_conversation_reader_state.complete
                    and screen._library_conversation_reader_state.error is None
                ),
                message="Conversation visible Retry did not restore selected truth",
            )
            retry_message_ids = [
                message.message_id
                for message in screen._library_conversation_reader_state.messages
            ]
            assert retry_message_ids == ["message-retry"]

            screen.query_one("#library-conversation-reader-info", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_conversation_reader_state.mode == "info",
                message="Conversation Info mode did not settle",
            )
            info_copy = str(
                screen.query_one(
                    "#library-conversation-reader-info-body", Static
                ).renderable
            )
            screen.query_one("#library-conversation-reader-read", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_conversation_reader_state.mode == "read",
                message="Conversation Read mode did not settle",
            )
            handoffs = []
            app.open_chat_with_handoff = lambda payload, **kwargs: handoffs.append(
                (payload, kwargs)
            )
            screen.query_one("#library-conversation-open-console", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: len(handoffs) == 1,
                message="Conversation Open in Console handoff did not reach app seam",
            )
            payload, handoff_kwargs = handoffs[0]
            observations = {
                "catalogue_ids": ["CO-01", "CO-02"],
                "progressive_page_offsets": [
                    call["message_offset"] for call in progressive.calls
                ],
                "progressive_message_total": progressive_state.message_total,
                "find_match_ids": [
                    match.message_id for match in progressive_state.find_matches
                ],
                "stale_first_id": first_id,
                "settled_target_id": target_id,
                "retry_error": retry_error,
                "retry_recovered_id": (
                    screen._library_conversation_reader_state.loaded_id
                ),
                "retry_message_ids": retry_message_ids,
                "info_copy": info_copy,
                "handoff_source_id": str(payload.source_id),
                "handoff_action_label": handoff_kwargs["action_label"],
            }
            facts = await _settled_capability_facts(
                screen, shell, pilot, "conversations", size, observations
            )
            compositor = facts["compositor_text"]
            svg = host.export_screenshot(simplify=True)
    finally:
        if "stale" in locals():
            stale.release_first.set()
        if "progressive" in locals():
            progressive.release_second.set()
        if "retry_failure" in locals():
            retry_failure.release.set()
        if "retry_service" in locals():
            retry_service.release.set()
        context.close()
    return _capture_finished_case(
        context,
        "conversations-capability",
        host,
        worker_baseline,
        facts,
        compositor,
        svg,
    )


async def run_notes_capability() -> dict[str, Any]:
    """NO-01/NO-02: one draft through Preview/Info, conflict, and bulk preview."""
    size = (120, 35)
    context = ScenarioContext.from_environment()
    app, prompt_db = await _seed_closeout_app(context.case_root("notes", size))
    context.add_cleanup(prompt_db.close)
    host = LibraryGlobalKeyProductionCSSHarness(app)
    worker_baseline = {id(worker) for worker in host.workers}
    facts: dict[str, Any]
    compositor = svg = ""
    try:
        async with host.run_test(size=size) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            shell = await _open_destination(screen, pilot, "notes")
            items_id, work_id = id(shell.items), id(shell.work)
            note_id = screen._selected_note_id
            service = app.notes_scope_service
            save_calls_before = len(service.save_calls)
            body = screen.query_one("#library-note-body", TextArea)
            body.text = "one unsaved closeout draft"
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_note_session.snapshot is not None
                    and screen._library_note_session.snapshot.dirty
                ),
                message="Notes draft did not become dirty",
            )
            screen.query_one("#library-note-preview", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_note_preview,
                message="Notes Preview did not settle",
            )
            assert (
                "one unsaved closeout draft"
                in screen.query_one("#library-note-preview-body").source
            )
            screen.query_one("#library-note-context", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_note_context,
                message="Notes Info did not settle",
            )
            screen.query_one("#library-note-context-back", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_note_context,
                message="Notes Info back did not restore Preview",
            )
            screen.query_one("#library-note-preview", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_note_preview,
                message="Notes Edit did not restore",
            )
            snapshot = screen._library_note_session.snapshot
            assert snapshot is not None
            assert snapshot.note_id == note_id
            assert snapshot.body == "one unsaved closeout draft" and snapshot.dirty
            assert len(service.save_calls) == save_calls_before

            original_title = snapshot.title
            invalid_title = "x" * 301
            screen.query_one("#library-note-title", Input).value = invalid_title
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_note_session.snapshot is not None
                    and screen._library_note_session.snapshot.title == invalid_title
                ),
                message="Notes invalid draft did not reach the session coordinator",
            )
            attempted_row = next(
                row
                for row in screen.query(".library-notes-row")
                if str(getattr(row, "note_id", "")) != str(note_id)
            )
            attempted_note_id = str(attempted_row.note_id)
            attempted_row.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_note_autosave_state == "validation"
                    and getattr(screen.focused, "id", None) == "library-note-title"
                ),
                message="Dirty Notes navigation did not expose its validation veto",
            )
            vetoed_snapshot = screen._library_note_session.snapshot
            assert vetoed_snapshot is not None
            assert (
                vetoed_snapshot.note_id,
                vetoed_snapshot.title,
                vetoed_snapshot.body,
                vetoed_snapshot.dirty,
            ) == (
                note_id,
                invalid_title,
                "one unsaved closeout draft",
                True,
            )
            assert screen._selected_note_id == note_id
            assert screen._library_notes_view == "editor"
            assert len(service.save_calls) == save_calls_before
            screen.query_one("#library-note-title", Input).value = original_title
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_note_session.snapshot is not None
                    and screen._library_note_session.snapshot.title == original_title
                ),
                message="Notes valid title did not restore after navigation veto",
            )

            _bump_note_version_externally(
                service,
                str(note_id),
                title="Recovered server title",
                content="Recovered server body",
            )
            screen.query_one("#library-note-save", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_note_autosave_state == "conflict",
                message="Notes deterministic version conflict did not settle",
            )
            conflict_copy = str(
                screen.query_one("#library-note-conflict-copy", Static).renderable
            )
            screen.query_one("#library-note-conflict-reload", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_note_autosave_state == "idle"
                    and screen._library_note_session.snapshot is not None
                    and screen._library_note_session.snapshot.body
                    == "Recovered server body"
                ),
                message="Notes conflict reload did not adopt server truth",
            )

            screen.query_one("#library-notes-select-toggle", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen.query_one("#library-note-bulk-status", Static).display
                    and "Read-only preview"
                    in str(
                        screen.query_one("#library-note-bulk-status", Static).renderable
                    )
                ),
                message="Notes labelled bulk preview did not settle",
            )
            bulk_copy = str(
                screen.query_one("#library-note-bulk-status", Static).renderable
            )
            assert id(shell.items) == items_id and id(shell.work) == work_id
            observations = {
                "catalogue_ids": ["NO-01", "NO-02"],
                "draft_note_id": str(note_id),
                "preview_info_draft": "one unsaved closeout draft",
                "save_calls_before_conflict": save_calls_before,
                "dirty_navigation_veto": {
                    "attempted_note_id": attempted_note_id,
                    "retained_note_id": str(note_id),
                    "retained_mode": "edit",
                    "retained_draft": "one unsaved closeout draft",
                    "persistence_writes": len(service.save_calls) - save_calls_before,
                },
                "conflict_copy": conflict_copy,
                "recovered_title": screen._library_note_session.snapshot.title,
                "recovered_body": screen._library_note_session.snapshot.body,
                "bulk_preview_copy": bulk_copy,
            }
            facts = await _settled_capability_facts(
                screen, shell, pilot, "notes", size, observations
            )
            compositor = facts["compositor_text"]
            svg = host.export_screenshot(simplify=True)
    finally:
        context.close()
    return _capture_finished_case(
        context,
        "notes-capability",
        host,
        worker_baseline,
        facts,
        compositor,
        svg,
    )


async def run_prompts_capability() -> dict[str, Any]:
    """PR-01/PR-02: lossless modes, owning validation, history, and retry."""
    size = (100, 30)
    context = ScenarioContext.from_environment()
    case_root = context.case_root("prompts", size)
    app, prompt_db = await _seed_closeout_app(case_root)
    context.add_cleanup(prompt_db.close)
    definition = _structured_prompt_definition()
    prompt_id, _uuid, _message = prompt_db.add_prompt(
        name="Structured closeout prompt",
        author="Advanced Author",
        details="Advanced-only detail",
        system_prompt="# Specialized role\n\nBe exact.",
        user_prompt="Ship it.",
        keywords=["closeout", "structured"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="prompt",
    )
    prompt_db.update_prompt_by_id(
        prompt_id, {"details": "Advanced-only detail v2"}, expected_version=1
    )
    host = LibraryGlobalKeyProductionCSSHarness(app)
    worker_baseline = {id(worker) for worker in host.workers}
    facts: dict[str, Any]
    compositor = svg = ""
    try:
        async with host.run_test(size=size) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            shell = await _open_destination(screen, pilot, "prompts")
            items_id, work_id = id(shell.items), id(shell.work)

            screen.query_one("#library-prompts-import", Button).press()
            await _wait_for_selector(screen, pilot, "#library-prompts-import-path")
            screen.query_one("#library-prompts-import-cancel", Button).press()
            await _wait_for_selector(screen, pilot, "#library-prompt-name")

            screen.query_one("#library-prompts-select", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_prompt_select_mode
                    and bool(shell.work.query("#library-prompt-bulk-status"))
                    and shell.work.query_one(
                        "#library-prompt-bulk-status", Static
                    ).display
                    and shell.work.query_one("#library-prompt-name", Input).disabled
                ),
                message="Prompt bulk preview did not enter Select mode",
            )
            bulk_copy = str(
                shell.work.query_one("#library-prompt-bulk-status", Static).renderable
            )
            assert "Read-only preview" in bulk_copy
            shell.items.query_one("#library-prompts-selection-done", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    not screen._library_prompt_select_mode
                    and bool(shell.work.query("#library-prompt-bulk-status"))
                    and not shell.work.query_one(
                        "#library-prompt-bulk-status", Static
                    ).display
                    and not shell.work.query_one("#library-prompt-name", Input).disabled
                ),
                message="Prompt Select mode did not exit",
            )

            rows = list(screen.query(".library-prompt-row"))
            loaded_before_retry = int(screen._library_prompt_loaded_id)
            retry_target = next(
                row
                for row in rows
                if int(getattr(row, "prompt_id"))
                not in {int(prompt_id), loaded_before_retry}
            )
            retry_id = int(retry_target.prompt_id)
            service = app.prompt_scope_service
            original_get_prompt = service.get_prompt

            def fail_retry_target(*, prompt_identifier, **kwargs):
                if int(prompt_identifier) == retry_id:
                    raise RuntimeError("simulated closeout detail failure")
                return original_get_prompt(
                    prompt_identifier=prompt_identifier, **kwargs
                )

            service.get_prompt = fail_retry_target
            retry_target.press()
            await _wait_for_selector(screen, pilot, "#library-prompt-detail-retry")
            locked_loaded_id = screen._library_prompt_loaded_id
            assert screen._selected_prompt_id == retry_id
            assert locked_loaded_id != retry_id
            service.get_prompt = original_get_prompt
            screen.query_one("#library-prompt-detail-retry", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_prompt_loaded_id == retry_id
                    and screen._library_prompt_editor_armed
                    and bool(screen.query("#library-prompt-user"))
                ),
                message="Prompt detail retry did not settle selected identity",
            )
            basic_reason = screen._library_prompt_basic_unavailable_reason(
                screen._current_library_prompt_editor_state(),
                conflict=screen._library_prompt_conflict_snapshot is not None,
            )
            assert not basic_reason, (
                f"Prompt continuity fixture is not Basic-compatible: {basic_reason}"
            )
            if screen._library_prompt_editor_mode != "basic":
                screen.query_one("#library-prompt-mode-basic", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: screen._library_prompt_editor_mode == "basic",
                    message="Prompt Basic continuity fixture did not settle",
                )
            preserved_author = screen.query_one("#library-prompt-author", Input).value
            preserved_details = screen.query_one("#library-prompt-details", Input).value
            preserved_keywords = screen.query_one(
                "#library-prompt-keywords", Input
            ).value
            user = screen.query_one("#library-prompt-user", TextArea)
            user.load_text("Ship the verified closeout.")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_prompt_dirty
                    and screen._library_prompt_block_state is not None
                    and screen._library_prompt_block_state.definition.lanes[1]
                    .blocks[0]
                    .content
                    == "Ship the verified closeout."
                ),
                message="Prompt Basic edit did not reach the shared draft",
            )
            basic_draft = user.text
            screen.query_one("#library-prompt-mode-advanced", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompt_editor_mode == "advanced",
                message="Prompt Advanced mode did not settle",
            )
            assert screen.query_one("#library-prompt-author", Input).value == (
                preserved_author
            )
            assert screen.query_one("#library-prompt-details", Input).value == (
                preserved_details
            )
            assert screen.query_one("#library-prompt-keywords", Input).value == (
                preserved_keywords
            )
            screen.query_one("#library-prompt-mode-basic", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompt_editor_mode == "basic",
                message="Prompt Basic mode did not restore",
            )
            assert screen.query_one("#library-prompt-user", TextArea).text == (
                basic_draft
            )
            screen.query_one("#library-prompt-mode-advanced", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompt_editor_mode == "advanced",
                message="Prompt Advanced revisit did not settle",
            )
            assert screen.query_one("#library-prompt-author", Input).value == (
                preserved_author
            )
            assert screen.query_one("#library-prompt-details", Input).value == (
                preserved_details
            )
            assert screen.query_one("#library-prompt-keywords", Input).value == (
                preserved_keywords
            )
            screen.query_one("#library-prompt-discard", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_prompt_dirty,
                message="Prompt continuity draft did not discard",
            )

            target = next(
                row
                for row in screen.query(".library-prompt-row")
                if int(getattr(row, "prompt_id")) == int(prompt_id)
            )
            target.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_prompt_loaded_id == prompt_id
                    and screen._library_prompt_editor_armed
                    and bool(screen.query("#prompt-block-title-delivery"))
                ),
                message="Structured Prompt editor did not fully settle",
            )
            if screen._library_prompt_editor_mode != "advanced":
                screen.query_one("#library-prompt-mode-advanced", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: screen._library_prompt_editor_mode == "advanced",
                    message="Structured Prompt Advanced mode did not settle",
                )
            title = screen.query_one("#prompt-block-title-delivery", Input)
            preserved_title = title.value
            assert screen._library_prompt_block_state is not None
            preserved_mapping_hint = (
                screen._library_prompt_block_state.definition.lanes[1]
                .blocks[0]
                .mapping_hint
            )
            title.value = ""
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_prompt_block_state is not None
                    and not screen._library_prompt_block_state.definition.lanes[1]
                    .blocks[0]
                    .title
                ),
                message="Invalid Prompt title did not reach the Advanced block",
            )
            screen.query_one("#library-prompt-mode-info", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_prompt_editor_mode == "info",
                message="Prompt Info mode did not settle",
            )
            screen.query_one("#library-prompt-save", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_prompt_editor_mode == "advanced"
                    and screen.focused is title
                ),
                message="Invalid Prompt save did not focus its Advanced owner",
            )
            validation_status = screen._library_prompt_status
            assert validation_status == "Fix block validation errors before saving."

            history = screen.query_one("#library-prompt-history-collapsible")
            await _wait_for_condition(
                pilot,
                lambda: "(2)" in str(history.title),
                message="Prompt retained-history count did not settle",
            )
            observations = {
                "catalogue_ids": ["PR-01", "PR-02"],
                "import_boundary": "opened_and_cancelled",
                "bulk_preview_copy": bulk_copy,
                "retry_selected_id": retry_id,
                "retry_prior_loaded_id": locked_loaded_id,
                "structured_prompt_id": prompt_id,
                "basic_draft": basic_draft,
                "preserved_advanced_author": preserved_author,
                "preserved_advanced_details": preserved_details,
                "preserved_advanced_keywords": preserved_keywords,
                "preserved_advanced_title_before_validation": preserved_title,
                "preserved_advanced_mapping_hint": preserved_mapping_hint,
                "validation_status": validation_status,
                "validation_focus_owner": getattr(screen.focused, "id", None),
                "history_title": str(history.title),
            }
            assert id(shell.items) == items_id and id(shell.work) == work_id
            facts = await _settled_capability_facts(
                screen, shell, pilot, "prompts", size, observations
            )
            compositor = facts["compositor_text"]
            svg = host.export_screenshot(simplify=True)
    finally:
        if "service" in locals() and "original_get_prompt" in locals():
            service.get_prompt = original_get_prompt
        context.close()
    return _capture_finished_case(
        context,
        "prompts-capability",
        host,
        worker_baseline,
        facts,
        compositor,
        svg,
    )


async def run_skills_capability() -> dict[str, Any]:
    """SK-01/SK-02: draft, trust identity/staleness, Files, delete preview."""
    size = (80, 24)
    context = ScenarioContext.from_environment()
    case_root = context.case_root("skills", size)
    app, prompt_db = await _seed_closeout_app(case_root)
    context.add_cleanup(prompt_db.close)
    skills_root = case_root / "trusted-skills"
    trust = _real_trust_service(skills_root)
    trust.bootstrap_trust()
    local, scope = _real_skills_scope_service(
        skills_root, trust_service=trust, allow_untrusted=False
    )
    await local.create_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Release notes\n---\nBe exact.",
    )
    await local.create_skill(
        name="review-skill",
        content="---\nname: review-skill\ndescription: Review skill\n---\nReview exactly.",
        supporting_files={"references/guide.md": "Read this guide."},
    )
    binary_path = local.skills_dir / "review-skill" / "assets" / "logo.png"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00closeout")
    app.skills_scope_service = scope
    app.local_skill_trust_service = trust
    host = LibraryGlobalKeyProductionCSSHarness(app)
    worker_baseline = {id(worker) for worker in host.workers}
    facts: dict[str, Any]
    compositor = svg = ""
    try:
        async with host.run_test(size=size) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            shell = await _open_destination(screen, pilot, "skills")
            items_id, work_id = id(shell.items), id(shell.work)

            async def press_visible(selector: str) -> Button:
                button = screen.query_one(selector, Button)
                button.focus()
                await _wait_for_condition(
                    pilot,
                    lambda: button.region.area > 0 and not button.disabled,
                    message=lambda: (
                        f"Skills control did not become visible: {selector}; "
                        f"mode={screen._library_skill_reader_mode}; "
                        f"view={screen._library_skills_view}; "
                        f"dirty={screen._library_skill_dirty}; "
                        f"mutation={screen._library_skill_mutation_in_flight}; "
                        f"more={screen._library_skill_more_actions_open}; "
                        f"display={button.display}; region={button.region}; "
                        "workers="
                        f"{[(str(worker.group), worker.state.name) for worker in screen.workers if not worker.is_finished]}"
                    ),
                )
                button.press()
                return button

            await press_visible("#library-skill-mode-edit")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_skill_reader_mode == "edit"
                    and bool(screen.query("#library-skill-description"))
                ),
                message="Skills Edit mode did not settle",
            )
            description = screen.query_one("#library-skill-description", Input)
            description.value = "one live unsaved skill draft"
            await _wait_for_condition(
                pilot,
                lambda: screen._library_skill_dirty,
                message="Skills Edit draft did not become dirty",
            )
            await press_visible("#library-skill-mode-overview")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_skill_reader_mode == "overview",
                message="Skills Overview mode did not settle",
            )
            await press_visible("#library-skill-mode-edit")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_skill_reader_mode == "edit"
                    and bool(screen.query("#library-skill-description"))
                ),
                message="Skills Edit revisit did not settle",
            )
            draft_after_round_trip = screen.query_one(
                "#library-skill-description", Input
            ).value
            assert draft_after_round_trip == "one live unsaved skill draft"

            await press_visible("#library-skill-mode-trust")
            await _wait_for_selector(screen, pilot, "#library-skill-trust-region")
            await press_visible("#library-skill-trust-review")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_skill_active_review is not None,
                message="Skills trust review did not settle",
            )
            review = dict(screen._library_skill_active_review)
            identity_copy = str(
                screen.query_one(
                    "#library-skill-trust-review-identity", Static
                ).renderable
            )
            assert str(review["manifest_generation"]) in identity_copy
            assert str(review["current_digest"]) in identity_copy

            await local.update_skill(
                "release-notes",
                content=(
                    "---\nname: release-notes\ndescription: Changed elsewhere\n---\n"
                    "Be exact."
                ),
            )
            trust.trust_current_skill("release-notes")
            try:
                trust.trust_reviewed_snapshot(str(review["review_id"]))
            except ValueError as error:
                stale_rejection = str(error)
            else:
                raise AssertionError("A stale trust review was unexpectedly admitted")
            assert "snapshot_mismatch" in stale_rejection

            await press_visible("#library-skill-mode-files")
            await _wait_for_selector(screen, pilot, "#library-skill-files-region")
            files_copy = str(
                screen.query_one("#library-skill-supporting", Static).renderable
            )
            assert "references/guide.md" in files_copy
            assert "assets/logo.png" in files_copy and "binary" in files_copy
            files_region = screen.query_one("#library-skill-files-region")
            assert not files_region.query(Input) and not files_region.query(TextArea)

            await press_visible("#library-skill-mode-edit")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_skill_reader_mode == "edit"
                    and screen._library_skill_dirty
                    and bool(screen.query("#library-skill-discard"))
                ),
                message="Skills dirty Edit state did not restore from Files",
            )
            await press_visible("#library-skill-save")
            await _wait_for_condition(
                pilot,
                lambda: (
                    not screen._library_skill_dirty
                    and not screen._library_skill_mutation_in_flight
                    and screen._library_skills_view == "editor"
                    and bool(screen.query("#library-skill-more-actions"))
                ),
                message="Skills draft save did not restore clean actions",
            )
            await press_visible("#library-skill-more-actions")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_skill_more_actions_open
                    and (
                        delete := screen.query_one("#library-skill-delete", Button)
                    ).display
                    and delete.region.area > 0
                ),
                message="Skills More actions did not reveal Delete",
            )
            await press_visible("#library-skill-delete")
            preview = await _wait_for_selector(
                screen, pilot, "#library-skill-delete-confirm-copy"
            )
            preview_copy = str(preview.renderable)
            await press_visible("#library-skill-delete-cancel")
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_skill_confirming_delete,
                message="Skills delete preview did not cancel",
            )
            assert await local.get_skill("review-skill")
            assert id(shell.items) == items_id and id(shell.work) == work_id
            observations = {
                "catalogue_ids": ["SK-01", "SK-02"],
                "draft_after_overview_round_trip": draft_after_round_trip,
                "review_id": str(review["review_id"]),
                "review_manifest_generation": review["manifest_generation"],
                "review_digest": str(review["current_digest"]),
                "review_identity_copy": identity_copy,
                "stale_review_rejection": stale_rejection,
                "files_copy": files_copy,
                "delete_preview_copy": preview_copy,
                "destructive_boundary": "truthful_preview_cancelled",
            }
            facts = await _settled_capability_facts(
                screen, shell, pilot, "skills", size, observations
            )
            compositor = facts["compositor_text"]
            svg = host.export_screenshot(simplify=True)
    finally:
        context.close()
    return _capture_finished_case(
        context,
        "skills-capability",
        host,
        worker_baseline,
        facts,
        compositor,
        svg,
    )


async def run_resize_purity() -> dict[str, dict[str, Any]]:
    """SH-06: execute and retain one no-side-effect resize oracle per reader."""
    context = ScenarioContext.from_environment()
    results = {}
    for destination in DESTINATIONS:
        name = f"{destination}-resize-purity"
        monkeypatch = pytest.MonkeyPatch()
        try:
            (
                facts,
                compositor,
                svg,
            ) = await _exercise_closeout_resize_is_presentation_only(
                destination,
                context.case_root(destination, (160, 50)),
                monkeypatch,
            )
            context.capture(name, facts, compositor, svg)
            results[name] = facts
        except Exception as error:
            results[name] = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        finally:
            monkeypatch.undo()
    return results


async def run_preferences_fresh_reload() -> dict[str, dict[str, Any]]:
    """SH-03: retain requested preference truth from a fresh app screen."""
    context = ScenarioContext.from_environment()
    name = "preferences-fresh-reload"
    try:
        (
            facts,
            compositor,
            svg,
        ) = await _exercise_closeout_preferences_restore_in_fresh_screen(
            context.case_root("preferences", (160, 50))
        )
        context.capture(name, facts, compositor, svg)
        return {name: facts}
    except Exception as error:
        return {
            name: {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        }


async def run_single_app_route_cycle() -> dict[str, dict[str, Any]]:
    """SH-01/03/04/07: retain the sequential cross-reader isolation oracle."""
    context = ScenarioContext.from_environment()
    name = "single-app-route-cycle"
    try:
        facts, compositor, svg = await _exercise_closeout_single_app_route_cycle(
            context.case_root("route-cycle", (160, 50))
        )
        context.capture(name, facts, compositor, svg)
        return {name: facts}
    except Exception as error:
        return {
            name: {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        }


SCENARIOS = {
    "common_matrix": run_common_matrix,
    "media_capability": run_media_capability,
    "conversations_capability": run_conversations_capability,
    "notes_capability": run_notes_capability,
    "prompts_capability": run_prompts_capability,
    "skills_capability": run_skills_capability,
    "resize_purity": run_resize_purity,
    "preferences_fresh_reload": run_preferences_fresh_reload,
    "single_app_route_cycle": run_single_app_route_cycle,
}
