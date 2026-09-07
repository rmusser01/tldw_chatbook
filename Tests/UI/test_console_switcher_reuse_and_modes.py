"""Native-report regressions: exact History reuse and visible mode ownership."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_console_character_activation_presentation import (
    _seed,
    _until,
    activation_library,  # noqa: F401
)
from Tests.UI.test_console_character_switcher_geometry import _GeometryApp
from Tests.UI.test_console_native_chat_flow import _static_plain_text
from Tests.UI.test_console_scope_row import (
    _AlwaysExistsMediaDB,
    _open_inspector_and_get_row,
)
from Tests.UI.test_console_workbench_contract import ConsoleHarness
from Tests.UI.test_library_inspection_admission import library  # noqa: F401
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_switcher_state import SwitcherMode
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
    ConsoleSwitcherChoice,
)


def _luminance(rgb):
    channels = [value / 255 for value in rgb]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return sum(
        value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("inactive", [False, True], ids=["current", "other-tab"])
@pytest.mark.parametrize(
    "modes",
    [
        (SwitcherMode.CHARACTER_CHATS, SwitcherMode.CHARACTER_CHATS),
        (SwitcherMode.HISTORY, SwitcherMode.HISTORY),
        (SwitcherMode.CHARACTER_CHATS, SwitcherMode.HISTORY),
        (SwitcherMode.HISTORY, SwitcherMode.CHARACTER_CHATS),
    ],
    ids=[
        "character-character",
        "history-history",
        "character-history",
        "history-character",
    ],
)
async def test_reopening_exact_conversation_reuses_runtime(
    activation_library,  # noqa: F811
    modes,
    inactive,
):
    owner, _, db = activation_library
    _seed(owner, db)
    owner.local_chat_conversation_service = ChatConversationService(db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        chat = host.screen
        await _until(lambda: chat.query("#console-native-composer"))
        store = chat._ensure_console_chat_store()
        prior = store.active_session_id
        first_id = None
        for mode in modes:
            if first_id is not None and inactive:
                await chat._session._activate_native_console_session(prior)
                assert store.active_session_id == prior
            await chat.action_open_console_session_switcher(
                initial_mode=mode,
                initial_character_query="Exact",
            )
            await _until(lambda: isinstance(host.screen, ConsoleSessionSwitcherModal))
            modal = host.screen
            await _until(
                lambda modal=modal: bool(modal.query("#console-switcher-query"))
            )
            if mode is SwitcherMode.HISTORY:
                modal.query_one("#console-switcher-query", Input).value = "Exact"
            await _until(
                lambda modal=modal: (
                    not modal._query_pending
                    and bool(modal._entries)
                    and modal.query_one("#console-switcher-query", Input).value
                    == "Exact"
                )
            )
            assert modal._entries[0].target.conversation_id == "exact"
            assert modal._mode is mode
            await pilot.press("enter")
            await _until(
                lambda: (
                    host.screen is chat
                    and any(
                        s.id == store.active_session_id
                        and s.persisted_conversation_id == "exact"
                        for s in store.sessions()
                    )
                )
            )
            await pilot.pause()
            matches = [
                s.id for s in store.sessions() if s.persisted_conversation_id == "exact"
            ]
            assert len(matches) == 1, f"{mode}: duplicate runtimes {matches}"
            assert store.active_session_id == matches[0]
            if first_id is not None:
                assert matches[0] == first_id
            first_id = matches[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_current_mode_is_painted_independently_of_focus(size, tmp_path):
    app = _GeometryApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.screen
        for label in ("Character chats", "Active", "History"):
            await pilot.pause()
            buttons = list(screen.query(".console-switcher-mode").results(Button))
            selected = [
                b for b in buttons if b.has_class("console-switcher-mode-current")
            ]
            assert len(selected) == 1
            inactive = [b for b in buttons if b is not selected[0]]
            for focus in (screen.query_one("#console-switcher-query"), inactive[0]):
                screen.set_focus(focus)
                await pilot.pause()
                app.export_screenshot()
                frame = "\n".join(
                    strip.text for strip in screen._compositor.render_strips()
                )
                assert f"[{label}]" in frame, (
                    screen.query_one("#console-switcher-modal").border_title,
                    frame,
                )
                title_segments = [
                    segment
                    for strip in screen._compositor.render_strips()
                    for segment in strip
                    if f"[{label}]" in segment.text
                ]
                assert title_segments
                for segment in title_segments:
                    foreground = _luminance(segment.style.color.triplet)
                    background = _luminance(segment.style.bgcolor.triplet)
                    contrast = (max(foreground, background) + 0.05) / (
                        min(foreground, background) + 0.05
                    )
                    assert contrast >= 4.5, (
                        "mode text must contrast with its background"
                    )
                assert selected[0].styles.background != inactive[-1].styles.background
                assert screen.focused is focus
            (tmp_path / f"mode-{label.replace(' ', '-')}.svg").write_text(
                app.export_screenshot()
            )
            await pilot.press("f3")
        await pilot.pause()
        await pilot.press("escape")


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["deleted", "native-prefix"])
async def test_history_reuse_preserves_exact_saved_identity(activation_library, fault):  # noqa: F811
    owner, _, db = activation_library
    _seed(owner, db)
    owner.local_chat_conversation_service = ChatConversationService(db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        chat = host.screen
        store = chat._ensure_console_chat_store()
        prior = store.active_session_id
        target = "exact" if fault == "deleted" else f"native:{prior}"
        if fault == "native-prefix":
            db.add_conversation({"id": target, "title": "Prefix identity fixture"})
        assert await chat._workspace._resume_console_workspace_conversation(target)
        warm_id = store.active_session_id
        await chat._session._activate_native_console_session(prior)
        page = await chat._workspace.load_console_session_switcher_history(
            query="", offset=0, limit=50
        )
        entry = next(e for e in page.entries if e.conversation_id == target)
        if fault == "deleted":
            with db.transaction() as connection:
                connection.execute(
                    "UPDATE conversations SET deleted = 1 WHERE id = ?", (target,)
                )
        await chat._session._apply_console_switcher_choice(
            ConsoleSwitcherChoice("activate", entry)
        )
        await pilot.pause()
        assert store.active_session_id == (prior if fault == "deleted" else warm_id)
        assert (
            len([s for s in store.sessions() if s.persisted_conversation_id == target])
            == 1
        )


@pytest.mark.asyncio
async def test_history_reuse_refreshes_changed_workspace_scope(activation_library):  # noqa: F811
    owner, _, db = activation_library
    _seed(owner, db)
    owner.media_db = _AlwaysExistsMediaDB()
    owner.local_chat_conversation_service = ChatConversationService(db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(240, 64)) as pilot:
        chat = host.screen
        await _open_inspector_and_get_row(chat, pilot)
        store = chat._ensure_console_chat_store()
        prior = store.active_session_id
        assert await chat._workspace._resume_console_workspace_conversation("exact")
        target = next(s for s in store.sessions() if s.id == store.active_session_id)
        registry = owner.workspace_registry_service
        registry.set_workspace_scope(
            target.workspace_id,
            RagScope(
                items=(ScopeItem("media", "old"),), updated_at="2026-09-06T00:00:00Z"
            ),
        )
        await chat._retrieval._resolve_console_effective_scope_state(target)
        assert chat._console_effective_scope_cache["exact"].item_count == 1
        await chat._session._activate_native_console_session(prior)
        registry.set_workspace_scope(
            target.workspace_id,
            RagScope(
                items=(ScopeItem("media", "new-1"), ScopeItem("media", "new-2")),
                updated_at="2026-09-07T00:00:00Z",
            ),
        )
        page = await chat._workspace.load_console_session_switcher_history(
            query="Exact", offset=0, limit=50
        )
        entry = next(e for e in page.entries if e.conversation_id == "exact")
        await chat._session._apply_console_switcher_choice(
            ConsoleSwitcherChoice("activate", entry)
        )
        await pilot.pause()
        assert store.active_session_id == target.id
        assert chat._console_effective_scope_cache["exact"].item_count == 2
        assert _static_plain_text(chat.query_one("#console-scope-chip")) == "Scope: 2"


@pytest.mark.asyncio
async def test_history_reuse_still_presents_chat_when_scope_refresh_fails(
    activation_library,  # noqa: F811
    monkeypatch,
):
    owner, _, db = activation_library
    _seed(owner, db)
    owner.local_chat_conversation_service = ChatConversationService(db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        chat = host.screen
        store = chat._ensure_console_chat_store()
        prior = store.active_session_id
        assert await chat._workspace._resume_console_workspace_conversation("exact")
        warm_id = store.active_session_id
        await chat._session._activate_native_console_session(prior)

        async def fail_scope(_session):
            raise RuntimeError("synthetic scope-display outage")

        monkeypatch.setattr(
            chat._retrieval, "_refresh_console_effective_scope_and_sync", fail_scope
        )
        await chat.action_open_console_session_switcher(
            initial_mode=SwitcherMode.HISTORY
        )
        await _until(lambda: isinstance(host.screen, ConsoleSessionSwitcherModal))
        modal = host.screen
        await _until(lambda: bool(modal.query("#console-switcher-query")))
        await pilot.pause()
        modal.query_one("#console-switcher-query", Input).value = "Exact"
        await pilot.pause()
        await _until(lambda: not modal._query_pending and bool(modal._entries))
        assert modal._entries[0].target.conversation_id == "exact"
        await pilot.press("enter")
        await pilot.pause()
        assert host.screen is chat
        assert store.active_session_id == warm_id
        assert (
            len([s for s in store.sessions() if s.persisted_conversation_id == "exact"])
            == 1
        )
        assert chat.focused is chat.query_one("#console-native-composer")
