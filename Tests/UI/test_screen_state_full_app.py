from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from typing import Any

import pytest

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_MODE,
    WATCHLISTS_NAV_CONTEXT_SECTION,
    WATCHLISTS_SECTION_RUNS,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.home_screen import HomeScreen
from tldw_chatbook.UI.Screens.library_screen import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LibraryScreen,
)
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)


def _configure_startup(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
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
    *,
    route: str = "home",
):
    _configure_startup(app, monkeypatch, route)
    _screen_name, canonical_route, screen_class = app._resolve_screen_navigation_target(
        route
    )
    assert screen_class is not None

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(300):
            if (
                getattr(app, "_initial_screen_pushed", False)
                and isinstance(app.screen, screen_class)
                and app.screen.is_mounted
            ):
                assert app.current_tab == canonical_route
                yield pilot
                return
            await pilot.pause(0.01)
        raise AssertionError("full app did not mount its configured production screen")


async def _wait_for_screen(
    app: TldwCli,
    pilot,
    screen_type: type,
    *,
    canonical_route: str,
):
    for _ in range(300):
        if (
            isinstance(app.screen, screen_type)
            and app.screen.is_mounted
            and app.current_tab == canonical_route
        ):
            await pilot.pause(0.01)
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(f"full app did not mount production {screen_type.__name__}")


@pytest.mark.asyncio
async def test_full_app_alias_startup_shares_canonical_snapshot_with_fresh_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="ccp") as pilot:
        original = app.screen
        assert isinstance(original, PersonasScreen)
        original.state_data = {"selection": "alias-owned-snapshot"}

        app.post_message(NavigateToScreen("home"))
        await _wait_for_screen(app, pilot, HomeScreen, canonical_route="home")
        assert app.screen_state_store.restore(
            "personas",
            app._current_runtime_identity(),
        ) == {"selection": "alias-owned-snapshot"}

        app.post_message(NavigateToScreen("personas"))
        restored = await _wait_for_screen(
            app,
            pilot,
            PersonasScreen,
            canonical_route="personas",
        )

        assert restored is not original
        assert restored.state_data == {"selection": "alias-owned-snapshot"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "canonical_route", "screen_type"),
    [
        ("notes", "library", LibraryScreen),
        ("customize", "settings", SettingsScreen),
    ],
)
async def test_full_app_legacy_alias_startup_publishes_canonical_tab(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    canonical_route: str,
    screen_type: type,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route=route):
        assert isinstance(app.screen, screen_type)
        assert app.current_tab == canonical_route


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["veto", "exception"])
async def test_full_app_pending_work_failure_keeps_exact_mounted_screen(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="library") as pilot:
        original = app.screen
        assert isinstance(original, LibraryScreen)

        if outcome == "veto":
            monkeypatch.setattr(original, "flush_pending_work", lambda: False)
        else:

            def fail_flush() -> None:
                raise RuntimeError("injected flush failure")

            monkeypatch.setattr(original, "flush_pending_work", fail_flush)

        app.post_message(NavigateToScreen("settings"))
        await pilot.pause(0.1)

        assert app.screen is original
        assert original.is_mounted
        assert app.current_tab == "library"
        assert (
            app.screen_state_store.has_snapshots(app._current_runtime_identity())
            is False
        )


@pytest.mark.asyncio
async def test_full_app_navigation_orders_real_screen_lifecycle_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="library") as pilot:
        outgoing = app.screen
        assert isinstance(outgoing, LibraryScreen)
        app.screen_state_store.save(
            "settings",
            {"active_category": SettingsCategoryId.STORAGE.value},
            app._current_runtime_identity(),
        )
        calls: list[str] = []

        real_flush = outgoing.flush_pending_work

        async def flush_pending_work():
            calls.append("flush")
            result = real_flush()
            if inspect.isawaitable(result):
                return await result
            return result

        real_save = outgoing.save_state

        def save_state():
            calls.append("save")
            return real_save()

        monkeypatch.setattr(outgoing, "flush_pending_work", flush_pending_work)
        monkeypatch.setattr(outgoing, "save_state", save_state)

        real_create = app._create_navigation_screen

        def create_with_lifecycle_observers(screen_name: str, screen_class: type):
            calls.append("construct")
            screen = real_create(screen_name, screen_class)
            if isinstance(screen, SettingsScreen):
                real_restore = screen.restore_state
                real_apply = screen.apply_navigation_context

                def restore_state(state: dict[str, Any]) -> None:
                    calls.append("restore")
                    real_restore(state)

                def apply_navigation_context(context: dict[str, Any]) -> None:
                    calls.append("context")
                    real_apply(context)

                monkeypatch.setattr(screen, "restore_state", restore_state)
                monkeypatch.setattr(
                    screen,
                    "apply_navigation_context",
                    apply_navigation_context,
                )
            return screen

        monkeypatch.setattr(
            app,
            "_create_navigation_screen",
            create_with_lifecycle_observers,
        )
        real_switch = app.switch_screen

        async def switch_screen(screen):
            calls.append("switch")
            return await real_switch(screen)

        monkeypatch.setattr(app, "switch_screen", switch_screen)

        app.post_message(
            NavigateToScreen(
                "settings",
                {"category": SettingsCategoryId.THEME.value},
            )
        )
        screen = await _wait_for_screen(
            app,
            pilot,
            SettingsScreen,
            canonical_route="settings",
        )

        assert calls == [
            "flush",
            "save",
            "construct",
            "restore",
            "context",
            "switch",
        ]
        assert screen.active_category == SettingsCategoryId.THEME.value


@pytest.mark.asyncio
async def test_full_app_restore_failure_discards_snapshot_and_mounts_fresh_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        app.screen_state_store.save(
            "settings",
            {"active_category": SettingsCategoryId.STORAGE.value},
            app._current_runtime_identity(),
        )
        real_create = app._create_navigation_screen
        created: list[SettingsScreen] = []

        def create_with_failing_restore(screen_name: str, screen_class: type):
            screen = real_create(screen_name, screen_class)
            if isinstance(screen, SettingsScreen):
                created.append(screen)

                def fail_restore(_state: dict[str, Any]) -> None:
                    raise ValueError("injected corrupt snapshot")

                monkeypatch.setattr(screen, "restore_state", fail_restore)
            return screen

        monkeypatch.setattr(
            app,
            "_create_navigation_screen",
            create_with_failing_restore,
        )

        app.post_message(NavigateToScreen("settings"))
        screen = await _wait_for_screen(
            app,
            pilot,
            SettingsScreen,
            canonical_route="settings",
        )

        assert created == [screen]
        assert screen.active_category == SettingsCategoryId.OVERVIEW.value
        assert (
            app.screen_state_store.restore(
                "settings",
                app._current_runtime_identity(),
            )
            is None
        )


@pytest.mark.asyncio
async def test_full_app_library_navigation_context_overrides_restored_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        app.screen_state_store.save(
            "library",
            {"library_selected_row_id": LIBRARY_ROW_BROWSE_MEDIA},
            app._current_runtime_identity(),
        )

        app.post_message(
            NavigateToScreen(
                "library",
                {LIBRARY_NAV_CONTEXT_MODE: "notes"},
            )
        )
        screen = await _wait_for_screen(
            app,
            pilot,
            LibraryScreen,
            canonical_route="library",
        )

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES


@pytest.mark.asyncio
async def test_full_app_settings_navigation_context_overrides_restored_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        app.screen_state_store.save(
            "settings",
            {"active_category": SettingsCategoryId.STORAGE.value},
            app._current_runtime_identity(),
        )

        app.post_message(
            NavigateToScreen(
                "settings",
                {"category": SettingsCategoryId.THEME.value},
            )
        )
        screen = await _wait_for_screen(
            app,
            pilot,
            SettingsScreen,
            canonical_route="settings",
        )

        assert screen.active_category == SettingsCategoryId.THEME.value


@pytest.mark.asyncio
async def test_full_app_settings_restore_detaches_nested_draft_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    draft = SettingsDraft(
        category=SettingsCategoryId.CONSOLE_BEHAVIOR,
        originals={"nested": {"items": ["original"]}},
        values={"nested": {"items": ["draft"]}},
    )
    restored_state = {
        "settings_drafts": {
            SettingsCategoryId.CONSOLE_BEHAVIOR: draft,
        }
    }

    async with _mounted_app(app, monkeypatch, route="settings") as pilot:
        screen = await _wait_for_screen(
            app,
            pilot,
            SettingsScreen,
            canonical_route="settings",
        )

        screen.restore_state(restored_state)
        retained = screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR]
        draft.values["nested"]["items"].append("producer-change")
        retained.values["nested"]["items"].append("consumer-change")

        assert retained.originals == {"nested": {"items": ["original"]}}
        assert retained.values == {"nested": {"items": ["draft", "consumer-change"]}}
        assert draft.values == {"nested": {"items": ["draft", "producer-change"]}}


@pytest.mark.asyncio
async def test_full_app_watchlists_restores_before_applying_navigation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        app.screen_state_store.save(
            "watchlists_collections",
            {"snapshot_marker": "restored"},
            app._current_runtime_identity(),
        )
        real_create = app._create_navigation_screen
        calls: list[str] = []

        def create_with_order_observers(screen_name: str, screen_class: type):
            screen = real_create(screen_name, screen_class)
            if isinstance(screen, WatchlistsCollectionsScreen):
                real_restore = screen.restore_state
                real_apply = screen.apply_navigation_context

                def restore_state(state: dict[str, Any]) -> None:
                    calls.append("restore")
                    real_restore(state)

                def apply_navigation_context(context: dict[str, Any]) -> None:
                    calls.append("context")
                    real_apply(context)

                monkeypatch.setattr(screen, "restore_state", restore_state)
                monkeypatch.setattr(
                    screen,
                    "apply_navigation_context",
                    apply_navigation_context,
                )
            return screen

        monkeypatch.setattr(
            app,
            "_create_navigation_screen",
            create_with_order_observers,
        )

        app.post_message(
            NavigateToScreen(
                "watchlists_collections",
                {WATCHLISTS_NAV_CONTEXT_SECTION: WATCHLISTS_SECTION_RUNS},
            )
        )
        screen = await _wait_for_screen(
            app,
            pilot,
            WatchlistsCollectionsScreen,
            canonical_route="watchlists_collections",
        )

        assert calls == ["restore", "context"]
        assert screen.state_data == {"snapshot_marker": "restored"}
        assert screen.active_section == WATCHLISTS_SECTION_RUNS
