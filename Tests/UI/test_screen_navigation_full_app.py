from __future__ import annotations

import threading
from contextlib import asynccontextmanager

import pytest
from textual.widgets import OptionList

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_MODE,
    WATCHLISTS_NAV_CONTEXT_SECTION,
)
from tldw_chatbook.Home.dashboard_state import (
    HomeActiveWorkItem,
    HomeDashboardInput,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_BROWSE_SEARCH,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.screen_state_store import (
    RuntimeIdentity,
    ScreenStateStore,
)
from tldw_chatbook.UI.Screens.home_screen import HomeScreen
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.schedules_screen import SchedulesScreen
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen
from tldw_chatbook.Widgets.settings_splash_screen_viewer import (
    SettingsSplashScreenViewer,
)
from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor


class _RecordingRecentWorkAdapter:
    """Record the narrow Home adapter contract without impersonating the app."""

    def __init__(self) -> None:
        self.calls: list[tuple[bool, int]] = []

    def build_dashboard_input(
        self,
        *,
        providers_models,
        has_recent_work: bool,
    ) -> HomeDashboardInput:
        self.calls.append((has_recent_work, threading.get_ident()))
        return HomeDashboardInput(
            model_ready=bool(providers_models),
            has_recent_work=has_recent_work,
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="workflow:run:task-644",
                    title="Workflow run",
                    source="workflows",
                    status="running",
                    detail_route="workflows",
                    console_available=True,
                ),
                HomeActiveWorkItem(
                    item_id="schedule:run:task-644",
                    title="Schedule run",
                    source="Schedules",
                    status="running",
                    detail_route="schedules",
                    console_available=True,
                ),
            ),
        )


def _configure_startup(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
    route: str = "settings",
) -> None:
    app.app_config["_first_run"] = False
    app._initial_tab_value = route
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(
        app_module,
        "get_cli_setting",
        get_cli_setting_without_splash,
    )


@asynccontextmanager
async def _mounted_app(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
    route: str = "settings",
):
    _configure_startup(app, monkeypatch, route)
    _screen_name, canonical_route, screen_class = app._resolve_screen_navigation_target(
        route
    )
    assert screen_class is not None

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            if getattr(app, "_initial_screen_pushed", False) and isinstance(
                app.screen,
                screen_class,
            ):
                assert app.current_tab == canonical_route
                yield pilot
                return
            await pilot.pause(0.01)
        raise AssertionError("full app did not mount its configured production screen")


async def _navigate(
    app: TldwCli,
    pilot,
    route: str,
    context: dict | None = None,
) -> None:
    await app.handle_screen_navigation(NavigateToScreen(route, context))
    await pilot.pause()


@pytest.mark.parametrize(
    ("configured_route", "canonical_route", "screen_class"),
    [
        ("ccp", "personas", PersonasScreen),
        ("notes", "library", LibraryScreen),
        ("customize", "settings", SettingsScreen),
    ],
)
@pytest.mark.asyncio
async def test_full_app_alias_startup_keeps_canonical_snapshot_identity(
    monkeypatch: pytest.MonkeyPatch,
    configured_route: str,
    canonical_route: str,
    screen_class: type,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, configured_route) as pilot:
        for _ in range(150):
            if getattr(app, "_ui_ready", False):
                break
            await pilot.pause(0.01)
        assert app._ui_ready is True
        await pilot.pause()

        assert isinstance(app.screen, screen_class)
        assert app.current_tab == canonical_route

        identity = app._current_runtime_identity()
        app.screen.state_data = {"startup_alias": configured_route}

        await _navigate(app, pilot, "home")

        saved_state = app.screen_state_store.restore(canonical_route, identity)
        assert saved_state is not None
        assert saved_state["startup_alias"] == configured_route

        await _navigate(app, pilot, configured_route)

        assert isinstance(app.screen, screen_class)
        assert app.current_tab == canonical_route
        assert app.screen.state_data["startup_alias"] == configured_route


@pytest.mark.asyncio
async def test_full_app_constructs_screen_state_owner_from_runtime_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch):
        assert isinstance(app.screen_state_store, ScreenStateStore)
        assert app._current_runtime_identity() == RuntimeIdentity.from_state(
            app.runtime_policy.state
        )


@pytest.mark.parametrize("seed_incompatible", [False, True])
@pytest.mark.asyncio
async def test_full_app_home_recent_work_uses_runtime_compatible_store_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    seed_incompatible: bool,
) -> None:
    app = TldwCli()
    adapter = _RecordingRecentWorkAdapter()
    app.home_active_work_adapter = adapter
    current_identity = app._current_runtime_identity()
    if seed_incompatible:
        seeded_identity = (
            RuntimeIdentity("server", "https://task-644-server.example")
            if current_identity.active_source == "local"
            else RuntimeIdentity("local")
        )
    else:
        seeded_identity = current_identity
    app.screen_state_store.save(
        "chat",
        {"recent_work": "task-644"},
        seeded_identity,
    )

    async with _mounted_app(app, monkeypatch, "home"):
        assert adapter.calls
        assert {has_recent_work for has_recent_work, _thread_id in adapter.calls} == {
            not seed_incompatible
        }
        assert app.screen_state_store.has_snapshots(current_identity) is (
            not seed_incompatible
        )


@pytest.mark.parametrize(
    ("registered_route", "screen_class"),
    [
        ("workflows", WorkflowsScreen),
        (None, SchedulesScreen),
    ],
)
@pytest.mark.asyncio
async def test_full_app_threaded_recent_work_consumers_capture_on_app_thread(
    monkeypatch: pytest.MonkeyPatch,
    registered_route: str | None,
    screen_class: type,
) -> None:
    app_thread_id = threading.get_ident()
    store_call_threads: list[int] = []
    real_has_snapshots = ScreenStateStore.has_snapshots

    def recording_has_snapshots(
        store: ScreenStateStore,
        identity: RuntimeIdentity,
    ) -> bool:
        store_call_threads.append(threading.get_ident())
        return real_has_snapshots(store, identity)

    monkeypatch.setattr(
        ScreenStateStore,
        "has_snapshots",
        recording_has_snapshots,
    )
    app = TldwCli()
    adapter = _RecordingRecentWorkAdapter()
    app.home_active_work_adapter = adapter

    async with _mounted_app(app, monkeypatch) as pilot:
        app.screen_state_store.save(
            "chat",
            {"recent_work": "task-644"},
            app._current_runtime_identity(),
        )
        if registered_route is not None:
            await _navigate(app, pilot, registered_route)
        else:
            await app.switch_screen(screen_class(app))
            await pilot.pause()

        for _ in range(150):
            if getattr(app.screen, "_latest_console_context_loaded", False):
                break
            await pilot.pause(0.01)

        assert isinstance(app.screen, screen_class)
        assert app.screen._latest_console_context_loaded is True

    assert store_call_threads
    assert set(store_call_threads) == {app_thread_id}
    assert adapter.calls
    assert {has_recent_work for has_recent_work, _thread_id in adapter.calls} == {True}
    assert any(thread_id != app_thread_id for _recent, thread_id in adapter.calls)


@pytest.mark.asyncio
async def test_full_app_async_schedules_reads_recent_work_on_app_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_thread_id = threading.get_ident()
    store_call_threads: list[int] = []
    real_has_snapshots = ScreenStateStore.has_snapshots

    def recording_has_snapshots(
        store: ScreenStateStore,
        identity: RuntimeIdentity,
    ) -> bool:
        store_call_threads.append(threading.get_ident())
        return real_has_snapshots(store, identity)

    monkeypatch.setattr(
        ScreenStateStore,
        "has_snapshots",
        recording_has_snapshots,
    )
    app = TldwCli()
    adapter = _RecordingRecentWorkAdapter()
    app.home_active_work_adapter = adapter

    async with _mounted_app(app, monkeypatch) as pilot:
        app.screen_state_store.save(
            "chat",
            {"recent_work": "task-644"},
            app._current_runtime_identity(),
        )
        await _navigate(app, pilot, "schedules")

        screen = app.screen
        assert isinstance(screen, SchedulesWorkbench)
        item = await screen._latest_console_follow_item_from_adapter()

        assert getattr(item, "item_id", None) == "schedule:run:task-644"

    assert store_call_threads
    assert set(store_call_threads) == {app_thread_id}
    assert adapter.calls
    assert {has_recent_work for has_recent_work, _thread_id in adapter.calls} == {True}
    assert {thread_id for _recent, thread_id in adapter.calls} == {app_thread_id}


@pytest.mark.asyncio
async def test_full_app_navigation_preserves_order_and_canonical_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        identity = app._current_runtime_identity()
        app.screen_state_store.save(
            "library",
            {"library_selected_row_id": LIBRARY_ROW_BROWSE_SEARCH},
            identity,
        )
        events: list[str] = []
        outgoing = app.screen

        async def flush_pending_work() -> bool:
            events.append("flush")
            return True

        def save_state() -> dict[str, str]:
            events.append("save")
            return {"outgoing": "settings"}

        monkeypatch.setattr(
            outgoing,
            "flush_pending_work",
            flush_pending_work,
            raising=False,
        )
        monkeypatch.setattr(outgoing, "save_state", save_state)

        real_create = app._create_navigation_screen

        def create_navigation_screen(screen_name, screen_class):
            events.append("construct")
            return real_create(screen_name, screen_class)

        monkeypatch.setattr(
            app,
            "_create_navigation_screen",
            create_navigation_screen,
        )

        real_restore = LibraryScreen.restore_state

        def restore_state(screen, state) -> None:
            events.append("restore")
            real_restore(screen, state)

        monkeypatch.setattr(LibraryScreen, "restore_state", restore_state)
        real_apply = LibraryScreen.apply_navigation_context

        def apply_navigation_context(screen, context) -> None:
            events.append("apply_context")
            real_apply(screen, context)

        monkeypatch.setattr(
            LibraryScreen,
            "apply_navigation_context",
            apply_navigation_context,
        )
        real_switch = app.switch_screen

        async def switch_screen(screen):
            events.append("switch")
            return await real_switch(screen)

        monkeypatch.setattr(app, "switch_screen", switch_screen)

        await _navigate(
            app,
            pilot,
            "notes",
            {LIBRARY_NAV_CONTEXT_MODE: "notes"},
        )

        assert isinstance(app.screen, LibraryScreen)
        assert app.current_tab == "library"
        assert events == [
            "flush",
            "save",
            "construct",
            "restore",
            "apply_context",
            "switch",
        ]
        assert app.screen_state_store.restore("settings", identity) == {
            "outgoing": "settings"
        }


@pytest.mark.asyncio
async def test_full_app_ccp_and_personas_share_one_canonical_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        identity = app._current_runtime_identity()
        app.screen_state_store.save(
            "personas",
            {"marker": "seeded-for-personas"},
            identity,
        )

        await _navigate(app, pilot, "ccp")

        first_personas = app.screen
        assert isinstance(first_personas, PersonasScreen)
        assert app.current_tab == "personas"
        assert first_personas.state_data == {"marker": "seeded-for-personas"}
        first_personas.state_data = {"marker": "saved-from-ccp"}

        await _navigate(app, pilot, "settings")
        await _navigate(app, pilot, "personas")

        assert isinstance(app.screen, PersonasScreen)
        assert app.screen is not first_personas
        assert app.current_tab == "personas"
        assert app.screen.state_data == {"marker": "saved-from-ccp"}


@pytest.mark.asyncio
async def test_full_app_notes_alias_restores_library_then_applies_notes_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        identity = app._current_runtime_identity()
        app.screen_state_store.save(
            "library",
            {"library_selected_row_id": LIBRARY_ROW_BROWSE_SEARCH},
            identity,
        )

        await _navigate(
            app,
            pilot,
            "notes",
            {LIBRARY_NAV_CONTEXT_MODE: "notes"},
        )

        assert isinstance(app.screen, LibraryScreen)
        assert app.current_tab == "library"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES


@pytest.mark.asyncio
async def test_full_app_distinct_canonical_routes_ignore_shared_screen_owned_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        identity = app._current_runtime_identity()
        app.screen_state_store.save(
            "conversation",
            {"marker": "conversation-state"},
            identity,
        )
        app.screen_state_store.save(
            "library",
            {"marker": "library-state"},
            identity,
        )

        await _navigate(app, pilot, "conversation")

        assert app.screen.screen_name == "library"
        assert app.current_tab == "conversation"
        assert app.screen.state_data == {"marker": "conversation-state"}

        await _navigate(app, pilot, "library")

        assert isinstance(app.screen, LibraryScreen)
        assert app.screen.screen_name == "library"
        assert app.current_tab == "library"
        assert app.screen.state_data == {"marker": "library-state"}


@pytest.mark.asyncio
async def test_full_app_unregistered_outgoing_route_skips_snapshot_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        outgoing = app.screen
        app.current_tab = ""
        outgoing.screen_name = "unregistered-route"
        save_calls: list[str] = []

        def save_state() -> dict[str, str]:
            save_calls.append("save")
            return {"payload": "must-not-be-stored"}

        monkeypatch.setattr(outgoing, "save_state", save_state)

        await _navigate(app, pilot, "home")

        assert isinstance(app.screen, HomeScreen)
        assert save_calls == []
        assert (
            app.screen_state_store.has_snapshots(app._current_runtime_identity())
            is False
        )


@pytest.mark.asyncio
async def test_full_app_save_failures_are_bounded_and_navigation_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        exception_sentinel = "SCREEN-SAVE-EXCEPTION-SENTINEL-71c"
        payload_sentinel = "SCREEN-SAVE-PAYLOAD-SENTINEL-b34"
        warnings: list[str] = []
        sink = app_module.logger.add(
            warnings.append,
            level="WARNING",
            format="{message}",
        )
        try:

            def fail_save():
                raise RuntimeError(f"{exception_sentinel} {payload_sentinel}")

            monkeypatch.setattr(app.screen, "save_state", fail_save)
            await _navigate(app, pilot, "home")
            assert isinstance(app.screen, HomeScreen)

            monkeypatch.setattr(
                app.screen,
                "save_state",
                lambda: [payload_sentinel],
            )
            await _navigate(app, pilot, "settings")
            assert isinstance(app.screen, SettingsScreen)
        finally:
            app_module.logger.remove(sink)

        state_warnings = [
            warning for warning in warnings if "screen snapshot" in warning.lower()
        ]
        assert len(state_warnings) == 2
        assert "exception_category=RuntimeError" in state_warnings[0]
        assert "reason=non_mapping" in state_warnings[1]
        assert exception_sentinel not in "\n".join(state_warnings)
        assert payload_sentinel not in "\n".join(state_warnings)


@pytest.mark.asyncio
async def test_full_app_restore_failure_discards_only_incoming_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        identity = app._current_runtime_identity()
        app.screen_state_store.save("home", {"marker": "home"}, identity)
        exception_sentinel = "SCREEN-RESTORE-EXCEPTION-SENTINEL-d91"
        payload_sentinel = "SCREEN-RESTORE-PAYLOAD-SENTINEL-11a"

        def fail_restore(_screen, _state) -> None:
            raise RuntimeError(f"{exception_sentinel} {payload_sentinel}")

        monkeypatch.setattr(HomeScreen, "restore_state", fail_restore)
        warnings: list[str] = []
        sink = app_module.logger.add(
            warnings.append,
            level="WARNING",
            format="{message}",
        )
        try:
            await _navigate(app, pilot, "home")
        finally:
            app_module.logger.remove(sink)

        assert isinstance(app.screen, HomeScreen)
        assert app.screen_state_store.restore("home", identity) is None
        assert app.screen_state_store.restore("settings", identity) is not None
        restore_warnings = [
            warning
            for warning in warnings
            if "screen snapshot restore failed" in warning.lower()
        ]
        assert len(restore_warnings) == 1
        assert "exception_category=RuntimeError" in restore_warnings[0]
        assert exception_sentinel not in restore_warnings[0]
        assert payload_sentinel not in restore_warnings[0]


@pytest.mark.asyncio
async def test_full_app_flush_veto_and_failure_abort_before_save_or_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        outgoing = app.screen
        forbidden_calls: list[str] = []

        def forbidden_save():
            forbidden_calls.append("save")
            return {}

        def forbidden_create(*_args):
            forbidden_calls.append("construct")
            raise AssertionError("flush rejection must precede construction")

        monkeypatch.setattr(outgoing, "save_state", forbidden_save)
        monkeypatch.setattr(app, "_create_navigation_screen", forbidden_create)

        async def veto_flush() -> bool:
            return False

        monkeypatch.setattr(
            outgoing,
            "flush_pending_work",
            veto_flush,
            raising=False,
        )
        await _navigate(app, pilot, "home")

        assert app.screen is outgoing
        assert forbidden_calls == []

        exception_sentinel = "SCREEN-FLUSH-EXCEPTION-SENTINEL-f82"
        warnings: list[str] = []
        sink = app_module.logger.add(
            warnings.append,
            level="WARNING",
            format="{message}",
        )

        async def fail_flush() -> bool:
            raise RuntimeError(exception_sentinel)

        monkeypatch.setattr(outgoing, "flush_pending_work", fail_flush)
        try:
            await _navigate(app, pilot, "home")
        finally:
            app_module.logger.remove(sink)

        assert app.screen is outgoing
        assert forbidden_calls == []
        flush_warnings = [
            warning for warning in warnings if "screen flush failed" in warning.lower()
        ]
        assert len(flush_warnings) == 1
        assert "exception_category=RuntimeError" in flush_warnings[0]
        assert exception_sentinel not in flush_warnings[0]


@pytest.mark.asyncio
async def test_full_app_navigation_always_constructs_fresh_screens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        await _navigate(app, pilot, "home")
        first_home = app.screen
        assert isinstance(first_home, HomeScreen)

        await _navigate(app, pilot, "settings")
        await _navigate(app, pilot, "home")

        assert isinstance(app.screen, HomeScreen)
        assert app.screen is not first_home


@pytest.mark.parametrize(
    "category",
    [
        SettingsCategoryId.THEME,
        SettingsCategoryId.SPLASH_SCREEN,
    ],
)
@pytest.mark.asyncio
async def test_full_app_settings_self_managed_contexts_compose(
    monkeypatch: pytest.MonkeyPatch,
    category: SettingsCategoryId,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        await _navigate(app, pilot, "home")
        await _navigate(
            app,
            pilot,
            "settings",
            {"category": category.value},
        )

        assert isinstance(app.screen, SettingsScreen)
        assert app.screen.active_category == category.value
        if category is SettingsCategoryId.THEME:
            editor = app.screen.query_one(
                "#settings-theme-editor",
                SettingsThemeEditor,
            )
            for _ in range(150):
                if set(editor.color_inputs) == set(editor.BASE_COLORS):
                    break
                await pilot.pause(0.01)
            assert set(editor.color_inputs) == set(editor.BASE_COLORS)
        else:
            viewer = app.screen.query_one(
                "#settings-splash-screen-viewer",
                SettingsSplashScreenViewer,
            )
            card_list = viewer.query_one("#settings-splash-card-list", OptionList)
            if viewer._cards:
                for _ in range(150):
                    if card_list.highlighted == 0:
                        break
                    await pilot.pause(0.01)
                assert card_list.highlighted == 0


@pytest.mark.asyncio
async def test_full_app_settings_drafts_detach_nested_state_across_store_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        assert isinstance(app.screen, SettingsScreen)
        category = SettingsCategoryId.APPEARANCE
        producer_draft = SettingsDraft(
            category=category,
            originals={"palette": {"colors": ["original-blue"]}},
            values={"palette": {"colors": ["draft-blue"]}},
        )
        app.screen._settings_drafts = {category: producer_draft}
        identity = app._current_runtime_identity()

        await _navigate(app, pilot, "home")
        producer_draft.values["palette"]["colors"].append("producer-mutation")
        await _navigate(app, pilot, "settings")

        assert isinstance(app.screen, SettingsScreen)
        restored_draft = app.screen._settings_drafts[category]
        assert restored_draft.values == {
            "palette": {"colors": ["draft-blue"]},
        }
        restored_draft.values["palette"]["colors"].append("consumer-mutation")

        later_snapshot = app.screen_state_store.restore("settings", identity)
        assert later_snapshot is not None
        later_draft = later_snapshot["settings_drafts"][category]
        assert later_draft.values == {
            "palette": {"colors": ["draft-blue"]},
        }
        assert producer_draft.values == {
            "palette": {
                "colors": ["draft-blue", "producer-mutation"],
            },
        }


@pytest.mark.asyncio
async def test_full_app_explicit_destination_context_wins_after_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        identity = app._current_runtime_identity()
        app.screen_state_store.save(
            "library",
            {"library_selected_row_id": LIBRARY_ROW_BROWSE_SEARCH},
            identity,
        )
        await _navigate(
            app,
            pilot,
            "library",
            {LIBRARY_NAV_CONTEXT_MODE: "notes"},
        )
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES

        app.screen_state_store.save(
            "settings",
            {"active_category": SettingsCategoryId.APPEARANCE.value},
            identity,
        )
        await _navigate(
            app,
            pilot,
            "settings",
            {"category": SettingsCategoryId.DIAGNOSTICS.value},
        )
        assert isinstance(app.screen, SettingsScreen)
        assert app.screen.active_category == SettingsCategoryId.DIAGNOSTICS.value

        app.screen_state_store.save(
            "watchlists_collections",
            {"active_section": "sources"},
            identity,
        )

        def restore_watchlists(screen, state) -> None:
            screen.state_data = dict(state)
            screen.active_section = state["active_section"]

        monkeypatch.setattr(
            WatchlistsCollectionsScreen,
            "restore_state",
            restore_watchlists,
        )
        await _navigate(
            app,
            pilot,
            "watchlists_collections",
            {WATCHLISTS_NAV_CONTEXT_SECTION: "runs"},
        )

        assert isinstance(app.screen, WatchlistsCollectionsScreen)
        assert app.screen.active_section == "runs"
