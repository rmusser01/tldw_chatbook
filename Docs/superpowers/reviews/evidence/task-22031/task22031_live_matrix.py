"""Production-shaped Textual evidence driver for TASK-22031.

This is a closeout evidence rig, not a test-suite replacement.  It mounts the
real ``LibraryScreen`` with the exact production stylesheet sequence and seeded,
bounded scope services so screenshots are deterministic and require no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace


WORKTREE = Path(__file__).resolve().parents[5]
EVIDENCE_DIR = Path(__file__).resolve().parent

# An ad-hoc harness gets no Tests/conftest.py isolation.  Refuse to import the
# app unless the caller supplied all profile boundaries explicitly.
required_environment = (
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "TLDW_CONFIG_PATH",
    "TLDW_TEST_MODE",
)
missing = [name for name in required_environment if not os.environ.get(name)]
if missing:
    raise SystemExit(f"refusing unisolated run; missing: {', '.join(missing)}")
config_path = Path(os.environ["TLDW_CONFIG_PATH"]).resolve()
home = Path(os.environ["HOME"]).resolve()
if home not in config_path.parents:
    raise SystemExit(f"config is outside scratch HOME: {config_path}")

sys.path.insert(0, str(WORKTREE))

import tldw_chatbook  # noqa: E402
from textual.widgets import Button, Input, Static  # noqa: E402

from Tests.UI.test_library_conversation_reader import (  # noqa: E402
    _GatedFailureConversationService,
    _ProgressiveConversationService,
    _active_conversations_screen,
    _conversation_records,
)
from Tests.UI.test_library_media_reader_shell import (  # noqa: E402
    _build_media_test_app,
    _open_media_shell,
)
from Tests.UI.test_library_shell import (  # noqa: E402
    LibraryHarness,
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen  # noqa: E402
from tldw_chatbook.UI.Screens import library_screen as library_screen_module  # noqa: E402
from tldw_chatbook.Widgets.Library import (  # noqa: E402
    LibraryAdaptiveReaderShell,
    LibraryConversationReader,
    LibraryConversationsCanvas,
)


SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))


class _CompleteConversationService:
    """Return one deterministic, complete transcript through the real seam."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_library_conversation_messages(self, conversation_id: str, **kwargs):
        self.calls.append({"conversation_id": conversation_id, **kwargs})
        text = (
            "A deterministic complete transcript proves the retained Work pane "
            "stays readable while Library and Items change around it."
        )
        version = 4 if conversation_id == "chat-a" else 7
        return {
            "id": conversation_id,
            "title": "Alpha planning" if conversation_id == "chat-a" else "Beta review",
            "version": version,
            "message_epoch": f"epoch-{conversation_id}",
            "message_total": 1,
            "message_offset": 0,
            "returned_message_count": 1,
            "has_more": False,
            "next_message_offset": None,
            "include_rag_context": False,
            "messages": [
                {
                    "id": f"message-{conversation_id}",
                    "sender": "user",
                    "timestamp": "2026-08-24T12:00:00Z",
                    "revision": f"revision-{conversation_id}",
                    "total_chars": len(text),
                    "char_start": 0,
                    "returned_chars": len(text),
                    "has_more": False,
                    "text": text,
                }
            ],
        }


def _screen_text(app) -> str:
    return "\n".join(
        strip.text.rstrip() for strip in app.screen._compositor.render_strips()
    )


def _region(region) -> dict[str, int]:
    return {
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height,
    }


def _capture(app, name: str, facts: dict[str, object]) -> None:
    svg_path = app.save_screenshot(filename=f"{name}.svg", path=str(EVIDENCE_DIR))
    (EVIDENCE_DIR / f"{name}.txt").write_text(
        _screen_text(app) + "\n", encoding="utf-8"
    )
    facts["svg"] = Path(svg_path).name
    (EVIDENCE_DIR / f"{name}.json").write_text(
        json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


async def _settle_reader(screen: LibraryScreen, pilot) -> LibraryConversationReader:
    reader = await _wait_for_selector(screen, pilot, "#library-conversation-reader")
    await screen.workers.wait_for_complete()
    await pilot.pause()
    return reader


async def _focus_and_wait(screen: LibraryScreen, pilot, selector: str):
    """Re-query, schedule focus, then prove event and paint readiness."""
    widget = screen.query_one(selector)
    widget.focus()
    await _wait_for_condition(
        pilot,
        lambda: (
            getattr(screen.focused, "id", None) == widget.id
            and widget in screen._compositor.visible_widgets
            and widget.region.width > 0
            and widget.region.height > 0
        ),
        message=f"{selector} did not become focused and painted",
    )
    return screen.query_one(selector)


async def conversations_geometry_matrix(summary: dict[str, object]) -> None:
    matrix: list[dict[str, object]] = []
    records = _conversation_records()
    records[0]["title"] = (
        "Alpha planning — an intentionally long conversation title whose full "
        "detail becomes easier to read when Library collapses"
    )
    for width, height in SIZES:
        app = _build_test_app()
        _seed_conversations(app, records)
        app.local_chat_conversation_service = _CompleteConversationService()
        host = LibraryHarness(app, screen=_active_conversations_screen(app))
        async with host.run_test(size=(width, height)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            reader = await _settle_reader(screen, pilot)
            shell = screen.query_one(
                "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
            )
            row = next(iter(screen.query(".library-conversation-row")))
            current = {
                "size": [width, height],
                "effective": {
                    "library_open": shell.effective_layout.library_open,
                    "items_open": shell.effective_layout.items_open,
                    "priority_pane": shell.effective_layout.priority_pane,
                },
                "regions": {
                    "shell": _region(shell.region),
                    "library": _region(shell.library.region),
                    "library_grip": _region(shell.library_grip.region),
                    "items": _region(shell.items.region),
                    "items_grip": _region(shell.items_grip.region),
                    "work": _region(shell.work.region),
                    "first_row": _region(row.region),
                },
                "work_mounted": shell.work.is_mounted and shell.work.display,
                "reader_complete": reader.state.complete,
                "message_epoch": reader.state.message_epoch,
                "grip_names": [shell.library_grip.name, shell.items_grip.name],
                "grip_widths": [
                    shell.library_grip.region.width,
                    shell.items_grip.region.width,
                ],
            }
            assert current["work_mounted"]
            assert current["grip_widths"] == [5, 5]
            assert shell.region.contains_region(shell.work.region)
            focused_grip = await _focus_and_wait(
                screen, pilot, f"#{shell.items_grip.id}"
            )
            owner, _ = host.get_widget_at(*focused_grip.region.center)
            current["focused_control"] = {
                "id": focused_grip.id,
                "region": _region(focused_grip.region),
                "paint_owner_id": getattr(owner, "id", None),
                "has_focus": focused_grip.has_focus,
            }
            assert current["focused_control"]["paint_owner_id"] == focused_grip.id
            _capture(host, f"conversations-{width}x{height}", current)
            matrix.append(current)
    summary["conversations_geometry"] = matrix


async def conversations_collapse_and_focus(summary: dict[str, object]) -> None:
    app = _build_test_app()
    records = _conversation_records()
    records[0]["title"] = (
        "Alpha planning — an intentionally long conversation title whose full "
        "detail becomes easier to read when Library collapses"
    )
    _seed_conversations(app, records)
    app.local_chat_conversation_service = _CompleteConversationService()
    host = LibraryHarness(app, screen=_active_conversations_screen(app))
    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        reader = await _settle_reader(screen, pilot)
        shell = screen.query_one(
            "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
        )
        items = shell.query_one(
            "#library-conversations-canvas", LibraryConversationsCanvas
        )
        row = next(iter(screen.query(".library-conversation-row")))
        saved_width = screen._library_conversation_reader_preferences.items_width
        before = {
            "items_region": _region(items.region),
            "row_region": _region(row.region),
            "stored_items_width": saved_width,
        }
        _capture(host, "conversations-160x50-expanded", before)

        shell.library_grip.press()
        await pilot.pause()
        row = next(iter(screen.query(".library-conversation-row")))
        after_library = {
            "library_open": shell.effective_layout.library_open,
            "items_open": shell.effective_layout.items_open,
            "items_region": _region(items.region),
            "row_region": _region(row.region),
            "stored_items_width": screen._library_conversation_reader_preferences.items_width,
            "library_restore_name": shell.library_grip.name,
        }
        assert not after_library["library_open"] and after_library["items_open"]
        assert after_library["row_region"]["width"] > before["row_region"]["width"]
        assert after_library["stored_items_width"] == before["stored_items_width"]
        _capture(host, "conversations-160x50-library-collapsed", after_library)

        shell.items_grip.press()
        await pilot.pause()
        both = {
            "library_open": shell.effective_layout.library_open,
            "items_open": shell.effective_layout.items_open,
            "work_mounted": shell.work.is_mounted and shell.work.display,
            "work_region": _region(shell.work.region),
            "restore_names": [shell.library_grip.name, shell.items_grip.name],
            "restore_focusable": [
                shell.library_grip.can_focus,
                shell.items_grip.can_focus,
            ],
        }
        assert not both["library_open"] and not both["items_open"]
        assert both["work_mounted"] and both["restore_focusable"] == [True, True]
        _capture(host, "conversations-160x50-both-collapsed", both)

        shell.items_grip.press()
        shell.library_grip.press()
        await pilot.pause()
        assert shell.effective_layout.library_open and shell.effective_layout.items_open

        rail = await _focus_and_wait(screen, pilot, "#library-search-input")
        item_filter = screen.query_one("#library-conversations-filter", Input)
        work_find = reader.query_one("#library-conversation-reader-find", Input)
        focus_cycle = []
        for expected_selector in (
            "#library-conversations-filter",
            "#library-conversation-reader-find",
            "#library-search-input",
        ):
            screen.action_focus_next_workbench_pane()
            await _wait_for_condition(
                pilot,
                lambda expected_selector=expected_selector: (
                    getattr(screen.focused, "id", None)
                    == expected_selector.removeprefix("#")
                ),
                message=f"F6 cycle did not settle on {expected_selector}",
            )
            focus_cycle.append(getattr(screen.focused, "id", None))
        assert focus_cycle == [item_filter.id, work_find.id, rail.id], focus_cycle
        work_find = await _focus_and_wait(
            screen, pilot, "#library-conversation-reader-find"
        )
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: (
                getattr(screen.focused, "id", None) == "library-conversations-filter"
            ),
            message="Escape did not settle on visible Items filter",
        )
        escape_target = getattr(screen.focused, "id", None)
        assert escape_target == item_filter.id
        await _focus_and_wait(screen, pilot, "#library-conversation-reader-read")
        await pilot.press("/")
        await _wait_for_condition(
            pilot,
            lambda: (
                getattr(screen.focused, "id", None) == "library-conversations-filter"
            ),
            message="Slash did not settle on visible Items filter",
        )
        slash_target = getattr(screen.focused, "id", None)
        assert slash_target == item_filter.id
        shortcuts = screen._library_route_shortcuts_for_current_state()
        focus_facts = {
            "f6_cycle": focus_cycle,
            "escape_target": escape_target,
            "slash_target": slash_target,
            "shortcuts": list(shortcuts),
            "focused": getattr(screen.focused, "id", None),
        }
        _capture(host, "conversations-160x50-focus-footer", focus_facts)
        summary["collapse_and_focus"] = {
            "before": before,
            "library_collapsed": after_library,
            "both_collapsed": both,
            "focus": focus_facts,
        }


async def conversations_progressive_find(summary: dict[str, object]) -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records()[:1])
    service = _ProgressiveConversationService()
    app.local_chat_conversation_service = service
    host = LibraryHarness(app, screen=_active_conversations_screen(app))
    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        reader = await _wait_for_selector(screen, pilot, "#library-conversation-reader")
        await asyncio.to_thread(service.second_started.wait, 10)
        try:
            await pilot.pause()
            first_page = {
                "message_count": len(
                    screen._library_conversation_reader_state.messages
                ),
                "message_epoch": screen._library_conversation_reader_state.message_epoch,
                "complete": screen._library_conversation_reader_state.complete,
                "status": str(
                    reader.query_one(
                        "#library-conversation-reader-status", Static
                    ).renderable
                ),
                "calls": list(service.calls),
            }
            assert first_page["message_count"] == 20 and not first_page["complete"]
            assert first_page["message_epoch"] == "epoch-chat-a"
            _capture(host, "conversations-progressive-first-page", first_page)
            find = reader.query_one("#library-conversation-reader-find", Input)
            find.value = "needle"
            find.focus()
            await pilot.press("enter")
            await pilot.pause()
            searching = str(
                reader.query_one(
                    "#library-conversation-reader-status", Static
                ).renderable
            )
            assert "Searching complete transcript" in searching
        finally:
            service.release_second.set()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        state = screen._library_conversation_reader_state
        status = str(
            reader.query_one("#library-conversation-reader-status", Static).renderable
        )
        complete = {
            "message_count": len(state.messages),
            "message_total": state.message_total,
            "message_epoch": state.message_epoch,
            "complete": state.complete,
            "find_complete": state.find_complete,
            "find_count": len(state.find_matches),
            "status": status,
            "focused": getattr(screen.focused, "message_id", None),
            "calls": list(service.calls),
        }
        assert complete["message_count"] == complete["message_total"] == 21
        assert (
            complete["message_epoch"] == first_page["message_epoch"] == "epoch-chat-a"
        )
        assert complete["complete"] and complete["find_complete"]
        assert complete["find_count"] == 1 and "1 exact match" in status
        _capture(host, "conversations-progressive-find-complete", complete)
        summary["progressive_find"] = {"first_page": first_page, "complete": complete}


async def conversations_bulk(summary: dict[str, object]) -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records())
    app.local_chat_conversation_service = _CompleteConversationService()
    host = LibraryHarness(app, screen=_active_conversations_screen(app))
    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        reader = await _settle_reader(screen, pilot)
        transcript = tuple(message.text for message in reader.state.messages)
        screen.query_one("#library-conversations-select-toggle", Button).press()
        await pilot.pause()
        bulk = {
            "bulk_active": reader.state.bulk_active,
            "bulk_selected_count": reader.state.bulk_selected_count,
            "transcript_preserved": tuple(
                message.text for message in reader.state.messages
            )
            == transcript,
            "preview_message_count": len(reader.state.messages),
            "message_epoch": reader.state.message_epoch,
            "status": str(
                reader.query_one(
                    "#library-conversation-reader-status", Static
                ).renderable
            ),
            "open_console_disabled": reader.query_one(
                "#library-conversation-open-console", Button
            ).disabled,
            "read_disabled": reader.query_one(
                "#library-conversation-reader-read", Button
            ).disabled,
            "info_disabled": reader.query_one(
                "#library-conversation-reader-info", Button
            ).disabled,
        }
        assert bulk == {
            "bulk_active": True,
            "bulk_selected_count": 0,
            "transcript_preserved": True,
            "preview_message_count": 1,
            "message_epoch": "epoch-chat-a",
            "status": "Bulk selection: 0 conversations. The retained transcript is not included and remains read-only.",
            "open_console_disabled": True,
            "read_disabled": False,
            "info_disabled": False,
        }, bulk
        _capture(host, "conversations-bulk-readonly-preview", bulk)
        summary["bulk"] = bulk


async def conversations_truthful_identity_and_deletion(
    summary: dict[str, object],
) -> None:
    """Drive A→B loading/error/retry and exact-locator deletion end to end."""
    app = _build_test_app()
    records = _conversation_records()
    _seed_conversations(app, records)
    app.local_chat_conversation_service = _CompleteConversationService()
    host = LibraryHarness(app, screen=_active_conversations_screen(app))
    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        reader = await _settle_reader(screen, pilot)
        assert reader.state.selected_id == reader.state.loaded_id == "chat-a"
        assert reader.state.message_epoch == "epoch-chat-a"

        failure_service = _GatedFailureConversationService("invalid")
        app.local_chat_conversation_service = failure_service
        target = next(
            row
            for row in screen.query(".library-conversation-row")
            if getattr(row, "conversation_id", None) == "chat-b"
        )
        target.press()
        await asyncio.to_thread(failure_service.started.wait, 10)
        try:
            await pilot.pause()
            loading = {
                "selected_id": reader.state.selected_id,
                "selected_version": reader.state.selected_version,
                "loaded_id": reader.state.loaded_id,
                "loaded_version": reader.state.loaded_version,
                "message_epoch": reader.state.message_epoch,
                "loading": reader.state.loading,
                "status": str(
                    reader.query_one(
                        "#library-conversation-reader-status", Static
                    ).renderable
                ),
            }
            assert loading["selected_id"] == "chat-b"
            assert loading["loaded_id"] == "chat-a" and loading["loading"]
            assert loading["message_epoch"] == "epoch-chat-a"
            _capture(host, "conversations-a-to-b-loading", loading)
        finally:
            failure_service.release.set()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        error = {
            "selected_id": reader.state.selected_id,
            "selected_version": reader.state.selected_version,
            "loaded_id": reader.state.loaded_id,
            "loaded_version": reader.state.loaded_version,
            "message_epoch": reader.state.message_epoch,
            "error": reader.state.error,
            "status": str(
                reader.query_one(
                    "#library-conversation-reader-status", Static
                ).renderable
            ),
            "open_console_disabled": reader.query_one(
                "#library-conversation-open-console", Button
            ).disabled,
        }
        assert error["selected_id"] == "chat-b" and error["loaded_id"] == "chat-a"
        assert error["message_epoch"] == "epoch-chat-a"
        assert error["error"] and error["open_console_disabled"]
        _capture(host, "conversations-a-to-b-error", error)

        app.local_chat_conversation_service = _CompleteConversationService()
        reader.query_one("#library-conversation-reader-retry", Button).press()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        recovered = {
            "selected_id": reader.state.selected_id,
            "selected_version": reader.state.selected_version,
            "loaded_id": reader.state.loaded_id,
            "loaded_version": reader.state.loaded_version,
            "message_epoch": reader.state.message_epoch,
            "complete": reader.state.complete,
            "error": reader.state.error,
            "status": str(
                reader.query_one(
                    "#library-conversation-reader-status", Static
                ).renderable
            ),
        }
        assert recovered["selected_id"] == recovered["loaded_id"] == "chat-b"
        assert recovered["message_epoch"] == "epoch-chat-b"
        assert recovered["complete"] and not recovered["error"]
        _capture(host, "conversations-a-to-b-retry", recovered)

        # The same public page-refresh boundary uses the production exact locator
        # before declaring the selected identity deleted.
        scope = app.chat_conversation_scope_service
        scope.conversations = (records[0],)
        _query, generation = screen._prepare_library_conversation_page_request("")
        await screen._load_library_conversation_page(1, "", generation)
        await pilot.pause()
        deleted = {
            "selected_id": reader.state.selected_id,
            "selected_version": reader.state.selected_version,
            "loaded_id": reader.state.loaded_id,
            "loaded_version": reader.state.loaded_version,
            "message_epoch": reader.state.message_epoch,
            "unavailable": reader.state.unavailable,
            "error": reader.state.error,
            "status": str(
                reader.query_one(
                    "#library-conversation-reader-status", Static
                ).renderable
            ),
            "locator_called_for_b": any(
                call.get("locator") and call.get("conversation_id") == "chat-b"
                for call in scope.calls
            ),
        }
        assert deleted["selected_id"] == "chat-b" and deleted["loaded_id"] is None
        assert deleted["message_epoch"] is None
        assert deleted["unavailable"] and deleted["error"] == "Conversation deleted."
        assert deleted["locator_called_for_b"]
        _capture(host, "conversations-b-deleted", deleted)
        summary["truthful_identity"] = {
            "loading": loading,
            "error": error,
            "recovered": recovered,
            "deleted": deleted,
        }


async def conversations_bulk_recovery_fence(summary: dict[str, object]) -> None:
    """Prove Select mode blocks Find/Retry from reviving invalidated work."""
    results: dict[str, object] = {}
    for trigger in ("find", "retry"):
        app = _build_test_app()
        _seed_conversations(app, _conversation_records()[:1])
        service = _ProgressiveConversationService()
        app.local_chat_conversation_service = service
        host = LibraryHarness(app, screen=_active_conversations_screen(app))
        async with host.run_test(size=(160, 50)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await asyncio.to_thread(service.second_started.wait, 10)
            try:
                screen.query_one("#library-conversations-select-toggle", Button).press()
                await pilot.pause()
                calls_before = len(service.calls)

                if trigger == "find":
                    find = screen.query_one("#library-conversation-reader-find", Input)
                    find.value = "needle"
                    find.focus()
                    await pilot.press("enter")
                else:
                    screen.retry_library_conversation_reader(
                        SimpleNamespace(stop=lambda: None)
                    )
                await pilot.pause()

                state = screen._library_conversation_reader_state
                during = {
                    "trigger": trigger,
                    "select_mode": screen._library_conversations_select_mode,
                    "bulk_active": state.bulk_active,
                    "loading": state.loading,
                    "loaded_actions_eligible": state.loaded_actions_eligible,
                    "message_epoch": state.message_epoch,
                    "service_calls_before": calls_before,
                    "service_calls_after": len(service.calls),
                    "status": str(
                        screen.query_one(
                            "#library-conversation-reader-status", Static
                        ).renderable
                    ),
                }
                assert during["select_mode"] and during["bulk_active"]
                assert not during["loading"] and not during["loaded_actions_eligible"]
                assert during["service_calls_after"] == during["service_calls_before"]
                assert "not included and remains read-only" in during["status"]
                _capture(host, f"conversations-select-{trigger}-fenced", during)
            finally:
                service.release_second.set()

            await screen.workers.wait_for_complete()
            await pilot.pause()
            settled = screen._library_conversation_reader_state
            assert screen._library_conversations_select_mode and settled.bulk_active
            assert not settled.loaded_actions_eligible
            results[trigger] = {
                "during": during,
                "settled_bulk_active": settled.bulk_active,
                "settled_loading": settled.loading,
                "settled_message_epoch": settled.message_epoch,
                "settled_service_calls": len(service.calls),
            }
    summary["bulk_recovery_fence"] = results


async def cross_destination_shared_library_latest_intent(
    summary: dict[str, object],
) -> None:
    """Prove an older Conversations write cannot beat newer Media intent."""
    app = _build_media_test_app()
    disk = {"library_open": True}
    writes: list[tuple[str, str, bool]] = []
    older_started = threading.Event()
    release_older = threading.Event()
    newer_started = threading.Event()

    def save_setting(section: str, key: str, value: bool) -> bool:
        if value is False:
            older_started.set()
            release_older.wait(timeout=10)
        else:
            newer_started.set()
        disk[key] = value
        writes.append((section, key, value))
        return True

    original_save = library_screen_module.save_setting_to_cli_config
    library_screen_module.save_setting_to_cli_config = save_setting
    try:
        host = LibraryProductionCSSHarness(app)
        async with host.run_test(size=(160, 50)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-conversations", Button).press()
            conversations = await _wait_for_selector(
                screen, pilot, "#library-conversations-reader-shell"
            )
            conversations.library_grip.press()
            await asyncio.to_thread(older_started.wait, 10)

            screen.query_one("#library-row-browse-media", Button).press()
            media = await _wait_for_selector(
                screen, pilot, "#library-media-reader-shell"
            )
            assert not media.effective_layout.library_open
            media.library_grip.press()
            try:
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen._library_reader_persistence_generations["library"] == 2
                    ),
                    message="Newer shared Library-pane intent was not claimed.",
                )
                await pilot.pause()
                assert not newer_started.is_set()
            finally:
                release_older.set()
            await screen.workers.wait_for_complete()

            facts = {
                "disk_library_open": disk["library_open"],
                "writes": [list(write) for write in writes],
                "conversation_preference_open": screen._library_conversation_reader_preferences.library_open,
                "media_preference_open": screen._library_media_reader_preferences.library_open,
                "media_effective_open": media.effective_layout.library_open,
                "shared_generation": screen._library_reader_persistence_generations[
                    "library"
                ],
            }
            assert newer_started.is_set()
            assert facts["disk_library_open"] is True
            assert facts["conversation_preference_open"] is True
            assert facts["media_preference_open"] is True
            assert facts["media_effective_open"] is True
            assert {tuple(write) for write in facts["writes"]} == {
                ("library.reader", "library_open", False),
                ("library.reader", "library_open", True),
            }
            _capture(host, "shared-library-latest-intent", facts)
            summary["shared_library_latest_intent"] = facts
    finally:
        library_screen_module.save_setting_to_cli_config = original_save


async def cross_destination_shared_library_double_failure(
    summary: dict[str, object],
) -> None:
    """Prove two failed overlapping writes restore durable shared True."""
    app = _build_media_test_app()
    writes: list[tuple[str, str, bool]] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def fail_save(section: str, key: str, value: bool) -> bool:
        writes.append((section, key, value))
        if len(writes) == 1:
            first_started.set()
            release_first.wait(timeout=10)
        return False

    original_save = library_screen_module.save_setting_to_cli_config
    library_screen_module.save_setting_to_cli_config = fail_save
    try:
        host = LibraryProductionCSSHarness(app)
        async with host.run_test(size=(160, 50)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-conversations", Button).press()
            conversations = await _wait_for_selector(
                screen, pilot, "#library-conversations-reader-shell"
            )
            conversations.library_grip.press()
            await asyncio.to_thread(first_started.wait, 10)
            try:
                screen.query_one("#library-row-browse-media", Button).press()
                media = await _wait_for_selector(
                    screen, pilot, "#library-media-reader-shell"
                )
                assert not media.effective_layout.library_open
                media.library_grip.press()
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen._library_reader_persistence_generations["library"] == 2
                    ),
                    message="Newer failed shared intent was not claimed.",
                )
            finally:
                release_first.set()

            await screen.workers.wait_for_complete()
            await pilot.pause()
            facts = {
                "writes": [list(write) for write in writes],
                "durable_library_open": screen._library_reader_durable_preferences[
                    "library"
                ],
                "config_library_open": app.app_config["library"]["reader"][
                    "library_open"
                ],
                "conversation_preference_open": screen._library_conversation_reader_preferences.library_open,
                "media_preference_open": screen._library_media_reader_preferences.library_open,
                "media_effective_open": media.effective_layout.library_open,
                "shared_generation": screen._library_reader_persistence_generations[
                    "library"
                ],
            }
            assert facts["writes"] == [
                ["library.reader", "library_open", False],
                ["library.reader", "library_open", True],
            ]
            assert facts["durable_library_open"] is True
            assert facts["config_library_open"] is True
            assert facts["conversation_preference_open"] is True
            assert facts["media_preference_open"] is True
            assert facts["media_effective_open"] is True
            _capture(host, "shared-library-double-failure", facts)
            summary["shared_library_double_failure"] = facts
    finally:
        library_screen_module.save_setting_to_cli_config = original_save


async def cross_destination_stale_skip_newer_failure(
    summary: dict[str, object],
) -> None:
    """Prove a queued stale close is skipped and newer failed open rolls back."""
    app = _build_media_test_app()
    writes: list[tuple[str, str, bool]] = []

    def fail_save(section: str, key: str, value: bool) -> bool:
        writes.append((section, key, value))
        return False

    original_save = library_screen_module.save_setting_to_cli_config
    library_screen_module.save_setting_to_cli_config = fail_save
    try:
        host = LibraryProductionCSSHarness(app)
        async with host.run_test(size=(160, 50)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            shared_lock = screen._library_conversation_reader_persistence_locks[
                "library"
            ]
            await shared_lock.acquire()
            try:
                screen.query_one("#library-row-browse-conversations", Button).press()
                conversations = await _wait_for_selector(
                    screen, pilot, "#library-conversations-reader-shell"
                )
                conversations.library_grip.press()
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen._library_reader_persistence_generations["library"] == 1
                    ),
                    message="Queued stale close was not claimed.",
                )

                screen.query_one("#library-row-browse-media", Button).press()
                media = await _wait_for_selector(
                    screen, pilot, "#library-media-reader-shell"
                )
                assert not media.effective_layout.library_open
                media.library_grip.press()
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen._library_reader_persistence_generations["library"] == 2
                    ),
                    message="Newer open was not claimed behind the shared lock.",
                )
            finally:
                shared_lock.release()

            await screen.workers.wait_for_complete()
            await pilot.pause()
            facts = {
                "writes": [list(write) for write in writes],
                "durable_library_open": screen._library_reader_durable_preferences[
                    "library"
                ],
                "config_library_open": app.app_config["library"]["reader"][
                    "library_open"
                ],
                "conversation_preference_open": screen._library_conversation_reader_preferences.library_open,
                "media_preference_open": screen._library_media_reader_preferences.library_open,
                "media_effective_open": media.effective_layout.library_open,
                "shared_generation": screen._library_reader_persistence_generations[
                    "library"
                ],
            }
            assert facts["writes"] == [["library.reader", "library_open", True]]
            assert facts["durable_library_open"] is True
            assert facts["config_library_open"] is True
            assert facts["conversation_preference_open"] is True
            assert facts["media_preference_open"] is True
            assert facts["media_effective_open"] is True
            _capture(host, "shared-library-stale-skip-newer-failure", facts)
            summary["shared_library_stale_skip_newer_failure"] = facts
    finally:
        library_screen_module.save_setting_to_cli_config = original_save


async def settings_refresh_repair_matrix(summary: dict[str, object]) -> None:
    """Prove Settings refresh repairs started stale writes for every authority."""
    cases = (
        ("media", "library", "reader", "library_open", "library"),
        ("conversations", "library", "reader", "library_open", "library"),
        ("media", "items", "media_reader", "items_open", "media_items"),
        (
            "conversations",
            "items",
            "conversations_reader",
            "items_open",
            "conversations_items",
        ),
    )
    results: list[dict[str, object]] = []
    for destination, pane, config_section, preference_key, authority in cases:
        app = _build_media_test_app()
        app.app_config["library"].setdefault(config_section, {})[preference_key] = True
        disk = {preference_key: True}
        writes: list[tuple[str, str, bool]] = []
        stale_started = threading.Event()
        release_stale = threading.Event()
        expected_section = f"library.{config_section}"

        def save_setting(section: str, key: str, value: bool) -> bool:
            if (section, key) != (expected_section, preference_key):
                return True
            writes.append((section, key, value))
            if len(writes) == 1:
                stale_started.set()
                release_stale.wait(timeout=10)
            disk[key] = value
            return True

        original_save = library_screen_module.save_setting_to_cli_config
        library_screen_module.save_setting_to_cli_config = save_setting
        try:
            host = LibraryProductionCSSHarness(app)
            async with host.run_test(size=(160, 50)) as pilot:
                if destination == "media":
                    screen, shell = await _open_media_shell(host, pilot)
                else:
                    screen = _active_library_screen(host)
                    await _wait_for_library_shell(screen, pilot)
                    screen.query_one(
                        "#library-row-browse-conversations", Button
                    ).press()
                    shell = await _wait_for_selector(
                        screen, pilot, "#library-conversations-reader-shell"
                    )
                getattr(shell, f"{pane}_grip").press()
                await asyncio.to_thread(stale_started.wait, 10)
                intent_generation = screen._library_reader_persistence_generations[
                    authority
                ]
                try:
                    app.app_config["library"].setdefault(config_section, {})[
                        preference_key
                    ] = True
                    screen.request_library_reader_layout_refresh(
                        screen._library_reader_layout_refresh_generation + 1
                    )
                    await _wait_for_condition(
                        pilot,
                        lambda: getattr(shell.effective_layout, preference_key),
                        message=(
                            f"Settings refresh did not paint {destination} {pane} open."
                        ),
                    )
                finally:
                    release_stale.set()

                await screen.workers.wait_for_complete()
                await pilot.pause()
                preferences = (
                    screen._library_conversation_reader_preferences
                    if destination == "conversations"
                    else screen._library_media_reader_preferences
                )
                facts: dict[str, object] = {
                    "destination": destination,
                    "pane": pane,
                    "authority": authority,
                    "writes": [list(write) for write in writes],
                    "disk_value": disk[preference_key],
                    "durable_value": screen._library_reader_durable_preferences[
                        authority
                    ],
                    "preference_value": getattr(preferences, preference_key),
                    "config_value": app.app_config["library"][config_section][
                        preference_key
                    ],
                    "mounted_effective_value": getattr(
                        shell.effective_layout, preference_key
                    ),
                    "generation_before_refresh": intent_generation,
                    "generation_after_repair": screen._library_reader_persistence_generations[
                        authority
                    ],
                }
                if pane == "library":
                    facts["conversation_preference_value"] = (
                        screen._library_conversation_reader_preferences.library_open
                    )
                    facts["media_preference_value"] = (
                        screen._library_media_reader_preferences.library_open
                    )
                assert facts["writes"] == [
                    [expected_section, preference_key, False],
                    [expected_section, preference_key, True],
                ]
                assert facts["disk_value"] is True
                assert facts["durable_value"] is True
                assert facts["preference_value"] is True
                assert facts["config_value"] is True
                assert facts["mounted_effective_value"] is True
                assert facts["generation_after_repair"] > intent_generation
                if pane == "library":
                    assert facts["conversation_preference_value"] is True
                    assert facts["media_preference_value"] is True
                artifact = f"settings-repair-{destination}-{pane}"
                _capture(host, artifact, facts)
                results.append(facts)
        finally:
            library_screen_module.save_setting_to_cli_config = original_save
    summary["settings_refresh_repair"] = results


async def settings_refresh_repair_failure(summary: dict[str, object]) -> None:
    """Prove failed repair tells the physical durable truth everywhere."""
    app = _build_media_test_app()
    disk = {"library_open": True}
    writes: list[tuple[str, str, bool]] = []
    stale_started = threading.Event()
    release_stale = threading.Event()
    notices: list[tuple[str, dict[str, object]]] = []

    def save_setting(section: str, key: str, value: bool) -> bool:
        if key != "library_open":
            return True
        writes.append((section, key, value))
        if len(writes) == 1:
            stale_started.set()
            release_stale.wait(timeout=10)
            disk[key] = value
            return True
        return False

    original_save = library_screen_module.save_setting_to_cli_config
    original_read = library_screen_module.read_cli_config_serialized
    original_notify = app.notify
    library_screen_module.save_setting_to_cli_config = save_setting
    library_screen_module.read_cli_config_serialized = lambda: (
        f"[library.media_reader]\nlibrary_open = {str(disk['library_open']).lower()}\n"
    )
    app.notify = lambda message, **kwargs: notices.append((message, kwargs))
    try:
        host = LibraryProductionCSSHarness(app)
        async with host.run_test(size=(160, 50)) as pilot:
            screen, shell = await _open_media_shell(host, pilot)
            shell.library_grip.press()
            await asyncio.to_thread(stale_started.wait, 10)
            try:
                app.app_config["library"]["reader"]["library_open"] = True
                screen.request_library_reader_layout_refresh(
                    screen._library_reader_layout_refresh_generation + 1
                )
            finally:
                release_stale.set()
            await screen.workers.wait_for_complete()
            await pilot.pause()
            facts = {
                "writes": [list(write) for write in writes],
                "disk_library_open": disk["library_open"],
                "durable_library_open": screen._library_reader_durable_preferences[
                    "library"
                ],
                "config_library_open": app.app_config["library"]["reader"][
                    "library_open"
                ],
                "conversation_preference_open": screen._library_conversation_reader_preferences.library_open,
                "media_preference_open": screen._library_media_reader_preferences.library_open,
                "mounted_effective_open": shell.effective_layout.library_open,
                "last_notice_severity": notices[-1][1].get("severity")
                if notices
                else None,
            }
            assert facts["writes"] == [
                ["library.reader", "library_open", False],
                ["library.reader", "library_open", True],
            ]
            assert facts["disk_library_open"] is False
            assert facts["durable_library_open"] is False
            assert facts["config_library_open"] is False
            assert facts["conversation_preference_open"] is False
            assert facts["media_preference_open"] is False
            assert facts["mounted_effective_open"] is False
            assert facts["last_notice_severity"] == "warning"
            _capture(host, "settings-repair-failure-truth", facts)
            summary["settings_refresh_repair_failure"] = facts
    finally:
        app.notify = original_notify
        library_screen_module.read_cli_config_serialized = original_read
        library_screen_module.save_setting_to_cli_config = original_save


async def delayed_settings_callback_repair_matrix(
    summary: dict[str, object],
) -> None:
    """Prove a delayed Settings callback repairs a grip write that already exited."""
    cases = (
        ("media", "library", "reader", "library_open", "library"),
        ("conversations", "library", "reader", "library_open", "library"),
        ("media", "items", "media_reader", "items_open", "media_items"),
        (
            "conversations",
            "items",
            "conversations_reader",
            "items_open",
            "conversations_items",
        ),
    )
    results: list[dict[str, object]] = []
    for destination, pane, config_section, preference_key, authority in cases:
        app = _build_media_test_app()
        app.app_config["library"].setdefault(config_section, {})[preference_key] = True
        disk = {preference_key: True}
        writes: list[tuple[str, str, bool]] = []
        grip_started = threading.Event()
        release_grip = threading.Event()
        expected_section = f"library.{config_section}"

        def save_setting(section: str, key: str, value: bool) -> bool:
            if (section, key) != (expected_section, preference_key):
                return True
            writes.append((section, key, value))
            if len(writes) == 1:
                grip_started.set()
                release_grip.wait(timeout=10)
            disk[key] = value
            return True

        original_save = library_screen_module.save_setting_to_cli_config
        library_screen_module.save_setting_to_cli_config = save_setting
        try:
            host = LibraryProductionCSSHarness(app)
            async with host.run_test(size=(160, 50)) as pilot:
                if destination == "media":
                    screen, shell = await _open_media_shell(host, pilot)
                else:
                    screen = _active_library_screen(host)
                    await _wait_for_library_shell(screen, pilot)
                    screen.query_one(
                        "#library-row-browse-conversations", Button
                    ).press()
                    shell = await _wait_for_selector(
                        screen, pilot, "#library-conversations-reader-shell"
                    )
                getattr(shell, f"{pane}_grip").press()
                await asyncio.to_thread(grip_started.wait, 10)

                # The Settings save has committed True, but its UI callback has not
                # run yet. Let the already-started stale grip False exit first.
                disk[preference_key] = True
                app.app_config["library"].setdefault(config_section, {})[
                    preference_key
                ] = True
                release_grip.set()
                await screen.workers.wait_for_complete()
                assert disk[preference_key] is False, (destination, pane, writes, disk)

                screen.request_library_reader_layout_refresh(
                    screen._library_reader_layout_refresh_generation + 1
                )
                await screen.workers.wait_for_complete()
                await pilot.pause()
                preferences = (
                    screen._library_conversation_reader_preferences
                    if destination == "conversations"
                    else screen._library_media_reader_preferences
                )
                facts: dict[str, object] = {
                    "destination": destination,
                    "pane": pane,
                    "authority": authority,
                    "schedule": [
                        "settings_true_committed",
                        "stale_grip_false_exited",
                        "settings_refresh_callback",
                    ],
                    "writes": [list(write) for write in writes],
                    "disk_value": disk[preference_key],
                    "durable_value": screen._library_reader_durable_preferences[
                        authority
                    ],
                    "preference_value": getattr(preferences, preference_key),
                    "config_value": app.app_config["library"][config_section][
                        preference_key
                    ],
                    "mounted_effective_value": getattr(
                        shell.effective_layout, preference_key
                    ),
                }
                if pane == "library":
                    facts["conversation_preference_value"] = (
                        screen._library_conversation_reader_preferences.library_open
                    )
                    facts["media_preference_value"] = (
                        screen._library_media_reader_preferences.library_open
                    )
                assert facts["writes"] == [
                    [expected_section, preference_key, False],
                    [expected_section, preference_key, True],
                ]
                assert all(
                    facts[key] is True
                    for key in (
                        "disk_value",
                        "durable_value",
                        "preference_value",
                        "config_value",
                        "mounted_effective_value",
                    )
                )
                if pane == "library":
                    assert facts["conversation_preference_value"] is True
                    assert facts["media_preference_value"] is True
                artifact = f"delayed-settings-repair-{destination}-{pane}"
                _capture(host, artifact, facts)
                results.append(facts)
        finally:
            library_screen_module.save_setting_to_cli_config = original_save
    summary["delayed_settings_callback_repair"] = results


async def delayed_settings_callback_repair_failure(
    summary: dict[str, object],
) -> None:
    """Prove failed delayed repair projects exact physical TOML truth."""
    app = _build_media_test_app()
    app.app_config["library"].setdefault("reader", {})["library_open"] = True
    disk = {"library_open": True}
    writes: list[tuple[str, str, bool]] = []
    grip_started = threading.Event()
    release_grip = threading.Event()
    notices: list[tuple[str, dict[str, object]]] = []

    def save_setting(section: str, key: str, value: bool) -> bool:
        if key != "library_open":
            return True
        writes.append((section, key, value))
        if len(writes) == 1:
            grip_started.set()
            release_grip.wait(timeout=10)
            disk[key] = value
            return True
        return False

    original_save = library_screen_module.save_setting_to_cli_config
    original_read = library_screen_module.read_cli_config_serialized
    original_notify = app.notify
    library_screen_module.save_setting_to_cli_config = save_setting
    library_screen_module.read_cli_config_serialized = lambda: (
        f"[library.reader]\nlibrary_open = {str(disk['library_open']).lower()}\n"
    )
    app.notify = lambda message, **kwargs: notices.append((message, kwargs))
    try:
        host = LibraryProductionCSSHarness(app)
        async with host.run_test(size=(160, 50)) as pilot:
            screen, shell = await _open_media_shell(host, pilot)
            shell.library_grip.press()
            await asyncio.to_thread(grip_started.wait, 10)
            disk["library_open"] = True
            app.app_config["library"]["reader"]["library_open"] = True
            release_grip.set()
            await screen.workers.wait_for_complete()
            assert disk["library_open"] is False

            screen.request_library_reader_layout_refresh(
                screen._library_reader_layout_refresh_generation + 1
            )
            await screen.workers.wait_for_complete()
            await pilot.pause()
            exact_toml = (
                "[library.reader]\n"
                f"library_open = {str(disk['library_open']).lower()}\n"
            )
            facts = {
                "writes": [list(write) for write in writes],
                "physical_toml": exact_toml,
                "disk_library_open": disk["library_open"],
                "durable_library_open": screen._library_reader_durable_preferences[
                    "library"
                ],
                "config_library_open": app.app_config["library"]["reader"][
                    "library_open"
                ],
                "conversation_preference_open": screen._library_conversation_reader_preferences.library_open,
                "media_preference_open": screen._library_media_reader_preferences.library_open,
                "mounted_effective_open": shell.effective_layout.library_open,
                "last_notice_severity": notices[-1][1].get("severity")
                if notices
                else None,
            }
            assert facts["writes"] == [
                ["library.reader", "library_open", False],
                ["library.reader", "library_open", True],
            ]
            assert all(
                facts[key] is False
                for key in (
                    "disk_library_open",
                    "durable_library_open",
                    "config_library_open",
                    "conversation_preference_open",
                    "media_preference_open",
                    "mounted_effective_open",
                )
            )
            assert facts["last_notice_severity"] == "warning"
            _capture(host, "delayed-settings-repair-failure-truth", facts)
            summary["delayed_settings_callback_repair_failure"] = facts
    finally:
        app.notify = original_notify
        library_screen_module.read_cli_config_serialized = original_read
        library_screen_module.save_setting_to_cli_config = original_save


async def persisted_preference_exact_toml_facts(
    summary: dict[str, object],
) -> None:
    """Record canonical, legacy, and absent-key defaults from exact TOML."""
    cases = (
        (
            "canonical_over_legacy",
            "[library.reader]\nlibrary_open = true\n"
            "[library.media_reader]\nlibrary_open = false\n",
            "library.reader",
            "library_open",
            True,
        ),
        (
            "legacy_shared_library",
            "[library.media_reader]\nlibrary_open = false\n",
            "library.reader",
            "library_open",
            False,
        ),
        (
            "absent_items_default",
            "[library.media_reader]\n",
            "library.media_reader",
            "items_open",
            True,
        ),
    )
    original_read = library_screen_module.read_cli_config_serialized
    facts: list[dict[str, object]] = []
    try:
        for name, serialized, section, key, expected in cases:
            library_screen_module.read_cli_config_serialized = lambda value=serialized: (
                value
            )
            actual = await LibraryScreen._read_library_reader_persisted_preference(
                object(), section, key
            )
            assert actual is expected
            facts.append(
                {
                    "name": name,
                    "physical_toml": serialized,
                    "section": section,
                    "key": key,
                    "expected": expected,
                    "actual": actual,
                }
            )
    finally:
        library_screen_module.read_cli_config_serialized = original_read
    summary["persisted_preference_exact_toml"] = facts


async def media_geometry_matrix(summary: dict[str, object]) -> None:
    matrix: list[dict[str, object]] = []
    for width, height in SIZES:
        app = _build_media_test_app()
        host = LibraryProductionCSSHarness(app)
        async with host.run_test(size=(width, height)) as pilot:
            screen, shell = await _open_media_shell(host, pilot)
            await pilot.pause()
            facts = {
                "size": [width, height],
                "effective": {
                    "library_open": shell.effective_layout.library_open,
                    "items_open": shell.effective_layout.items_open,
                    "priority_pane": shell.effective_layout.priority_pane,
                },
                "regions": {
                    "shell": _region(shell.region),
                    "library": _region(shell.library.region),
                    "library_grip": _region(shell.library_grip.region),
                    "items": _region(shell.items.region),
                    "items_grip": _region(shell.items_grip.region),
                    "work": _region(shell.work.region),
                },
                "work_mounted": shell.work.is_mounted and shell.work.display,
                "grip_names": [shell.library_grip.name, shell.items_grip.name],
                "grip_widths": [
                    shell.library_grip.region.width,
                    shell.items_grip.region.width,
                ],
            }
            assert facts["work_mounted"] and facts["grip_widths"] == [5, 5]
            assert shell.region.contains_region(shell.work.region)
            focused_grip = await _focus_and_wait(
                screen, pilot, f"#{shell.items_grip.id}"
            )
            owner, _ = host.get_widget_at(*focused_grip.region.center)
            facts["focused_control"] = {
                "id": focused_grip.id,
                "region": _region(focused_grip.region),
                "paint_owner_id": getattr(owner, "id", None),
                "has_focus": focused_grip.has_focus,
            }
            assert facts["focused_control"]["paint_owner_id"] == focused_grip.id
            _capture(host, f"media-{width}x{height}", facts)
            matrix.append(facts)
    summary["media_geometry"] = matrix


async def main() -> None:
    package_path = Path(tldw_chatbook.__file__).resolve()
    if WORKTREE not in package_path.parents:
        raise AssertionError(f"foreign package import: {package_path}")
    data_dir = Path(os.environ["TASK22031_DATA_DIR"]).resolve()
    if home not in data_dir.parents:
        raise AssertionError(f"data dir outside scratch HOME: {data_dir}")
    summary: dict[str, object] = {
        "worktree": str(WORKTREE),
        "package_import": str(package_path),
        "config_path": str(config_path),
        "scratch_home": str(home),
        "data_dir": str(data_dir),
        "sizes": [list(size) for size in SIZES],
        "surface": "real LibraryScreen + production Textual CSS + compositor",
        "data_authority": "deterministic seeded scope services; no network",
    }
    await conversations_geometry_matrix(summary)
    await conversations_collapse_and_focus(summary)
    await conversations_progressive_find(summary)
    await conversations_bulk(summary)
    await conversations_truthful_identity_and_deletion(summary)
    await conversations_bulk_recovery_fence(summary)
    await cross_destination_shared_library_latest_intent(summary)
    await cross_destination_shared_library_double_failure(summary)
    await cross_destination_stale_skip_newer_failure(summary)
    await settings_refresh_repair_matrix(summary)
    await settings_refresh_repair_failure(summary)
    await delayed_settings_callback_repair_matrix(summary)
    await delayed_settings_callback_repair_failure(summary)
    await persisted_preference_exact_toml_facts(summary)
    await media_geometry_matrix(summary)
    (EVIDENCE_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
