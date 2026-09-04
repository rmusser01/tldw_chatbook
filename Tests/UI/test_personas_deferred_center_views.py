"""TASK-31215: heavy Personas center views mount only on first use.

TASK-2725 moved four hidden bodies past first paint, but still mounted all of
them during the initial load. TASK-31215 keeps stable slots in document order
and mounts only the body an editor/detail workflow first requests.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest
from textual.containers import Vertical
from textual.widgets import Input, Static

from tldw_chatbook.UI.Screens.personas_screen import (
    PersonasScreen,
    _CenterViewMountUnavailable,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail import (
    PersonasDictionaryDetailWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_detail import (
    PersonasLoreDetailWidget,
)

pytestmark = pytest.mark.asyncio

_DEFERRED_TYPES = (
    PersonasCharacterEditorWidget,
    PersonasDictionaryDetailWidget,
    PersonasLoreDetailWidget,
    PersonaProfileEditorWidget,
)

#: The stack's document order — scroll flow depends on it, so the lightweight
#: slots must occupy exactly the positions their bodies historically used.
_EXPECTED_STACK_ORDER = [
    "ccp-character-card-view",
    "personas-character-editor-slot",
    "personas-character-attachments",
    "ccp-persona-card-view",
    "personas-persona-editor-slot",
    "personas-conversation-actions",
    "personas-dictionary-detail-slot",
    "personas-lore-detail-slot",
    "personas-conversation-transcript-view",
    "personas-mode-placeholder",
    "personas-characters-empty",
]


async def test_first_paint_excludes_the_four_heavy_center_views(monkeypatch):
    """Compose alone must not mount the deferred views (the perf mechanism)."""
    from Tests.UI.app_factory import _build_test_app

    monkeypatch.setattr(PersonasScreen, "_load_after_mount", AsyncMock(), raising=True)

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        await pilot.pause()

        for deferred_type in _DEFERRED_TYPES:
            assert not list(screen.query(deferred_type)), (
                f"{deferred_type.__name__} mounted during compose — "
                "back on the click→paint critical path"
            )
        total = len(list(screen.query("*")))
        assert total < 300, (
            f"first-paint widget count regressed to {total} (pre-fix: 494)"
        )


async def test_settled_initial_load_keeps_heavy_views_unmounted():
    """The real load must settle without constructing inactive heavy bodies."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()

        for deferred_type in _DEFERRED_TYPES:
            assert not list(screen.query(deferred_type)), (
                f"{deferred_type.__name__} mounted during initial load — "
                "inactive bodies must wait for first use"
            )

        stack = screen.query_one("#personas-detail-stack")
        assert [child.id for child in stack.children] == _EXPECTED_STACK_ORDER


async def test_first_use_mounts_only_requested_view_and_caches_it():
    """One requested body mounts once and retains its in-screen form state."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()

        first = await screen._ensure_center_view("character-editor")
        assert isinstance(first, PersonasCharacterEditorWidget)
        name = first.query_one("#personas-char-editor-name", Input)
        name.value = "Keep this draft"

        second = await screen._ensure_center_view("character-editor")

        assert second is first
        assert second.query_one("#personas-char-editor-name", Input).value == (
            "Keep this draft"
        )
        assert not list(screen.query(PersonaProfileEditorWidget))
        assert not list(screen.query(PersonasDictionaryDetailWidget))
        assert not list(screen.query(PersonasLoreDetailWidget))


async def test_concurrent_first_use_builds_and_mounts_one_view(monkeypatch):
    """Concurrent requests coalesce behind one per-view mount operation."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        real_builder = screen._build_center_view
        build_count = 0

        def counted_builder(view_key: str):
            nonlocal build_count
            build_count += 1
            return real_builder(view_key)

        monkeypatch.setattr(screen, "_build_center_view", counted_builder)
        first, second = await asyncio.gather(
            screen._ensure_center_view("character-editor"),
            screen._ensure_center_view("character-editor"),
        )

        assert first is second
        assert build_count == 1
        assert len(list(screen.query(PersonasCharacterEditorWidget))) == 1


async def test_concurrent_caller_waits_until_mounted_view_is_hydrated(monkeypatch):
    """A queryable-but-unready body must not escape the per-view lock."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        slot = screen.query_one("#personas-character-editor-slot", Vertical)
        real_mount = slot.mount
        real_hydrate = screen._hydrate_center_view
        mounted = asyncio.Event()
        release = asyncio.Event()
        hydration_attempts = 0

        async def mount_then_pause(*widgets, **kwargs):
            result = await real_mount(*widgets, **kwargs)
            mounted.set()
            await release.wait()
            return result

        def fail_first_hydration(view_key: str, view):
            nonlocal hydration_attempts
            hydration_attempts += 1
            if hydration_attempts == 1:
                raise RuntimeError("first hydration failed")
            real_hydrate(view_key, view)

        monkeypatch.setattr(slot, "mount", mount_then_pause)
        monkeypatch.setattr(screen, "_hydrate_center_view", fail_first_hydration)
        first = asyncio.create_task(screen._ensure_center_view("character-editor"))
        await mounted.wait()
        second = asyncio.create_task(screen._ensure_center_view("character-editor"))
        await asyncio.sleep(0)
        escaped_before_hydration = second.done()
        release.set()

        assert await first is None
        ready = await second

        assert not escaped_before_hydration
        assert isinstance(ready, PersonasCharacterEditorWidget)
        assert ready.is_mounted
        assert hydration_attempts == 2
        assert len(list(screen.query(PersonasCharacterEditorWidget))) == 1


async def test_transient_mount_failure_remains_retryable(monkeypatch):
    """A failed attempt owns no ready state and a later attempt can succeed."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    notifications: list[tuple[str, str]] = []
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        slot = screen.query_one("#personas-character-editor-slot", Vertical)
        real_mount = slot.mount
        attempts = 0

        async def fail_once(*widgets, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient test failure")
            return await real_mount(*widgets, **kwargs)

        monkeypatch.setattr(slot, "mount", fail_once)
        monkeypatch.setattr(
            screen,
            "_notify",
            lambda message, severity="information", **_: notifications.append(
                (str(message), severity)
            ),
        )

        assert await screen._ensure_center_view("character-editor") is None
        second = await screen._ensure_center_view("character-editor")

        assert isinstance(second, PersonasCharacterEditorWidget)
        assert attempts == 2
        assert notifications == [
            ("Couldn't open this Personas view. Try again.", "error")
        ]
        assert len(list(screen.query(PersonasCharacterEditorWidget))) == 1


async def test_unmount_during_population_blocks_stale_hydration(monkeypatch):
    """A mount finishing after lifecycle invalidation is removed, not hydrated."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        slot = screen.query_one("#personas-persona-editor-slot", Vertical)
        real_mount = slot.mount
        started = asyncio.Event()
        release = asyncio.Event()
        hydrate = Mock(wraps=screen._hydrate_center_view)

        async def delayed_mount(*widgets, **kwargs):
            started.set()
            await release.wait()
            return await real_mount(*widgets, **kwargs)

        monkeypatch.setattr(slot, "mount", delayed_mount)
        monkeypatch.setattr(screen, "_hydrate_center_view", hydrate)
        population = asyncio.create_task(screen._ensure_center_view("persona-editor"))
        await started.wait()
        generation = screen._center_view_lifecycle_generation

        await screen.on_unmount()
        assert screen._center_view_lifecycle_generation == generation + 1
        release.set()
        assert await population is None

        hydrate.assert_not_called()
        assert not list(screen.query(PersonaProfileEditorWidget))


async def test_transient_restore_mount_failure_keeps_pending_intent(monkeypatch):
    """A view-mount failure is retryable restore state, not a deleted entity."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        pending = {"kind": "dictionary", "id": "7", "name": "Terms"}
        screen._pending_restore = pending
        screen._restored_from_saved_state = True
        screen.state.active_mode = "dictionaries"
        monkeypatch.setattr(screen, "_apply_mode", AsyncMock())
        monkeypatch.setattr(
            screen,
            "_select_dictionary",
            AsyncMock(side_effect=_CenterViewMountUnavailable("dictionary-detail")),
        )

        await screen._apply_pending_restore()

        assert screen._pending_restore is pending
        assert screen._restored_from_saved_state is True


async def test_inactive_editor_compose_cannot_stall_personas_navigation(monkeypatch):
    """An expensive inactive editor cannot block the default screen arrival."""
    from Tests.UI.app_factory import _build_test_app

    def blocking_character_editor_compose(self):
        time.sleep(0.35)
        yield Static("deliberately slow inactive editor")

    monkeypatch.setattr(
        PersonasCharacterEditorWidget, "compose", blocking_character_editor_compose
    )

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        gaps: list[float] = []
        running = True

        async def heartbeat() -> None:
            loop = asyncio.get_running_loop()
            previous = loop.time()
            while running:
                await asyncio.sleep(0.005)
                now = loop.time()
                gaps.append(now - previous)
                previous = now

        heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        for _ in range(8):
            await pilot.pause()
        await asyncio.sleep(0.05)
        running = False
        await heartbeat_task

        assert screen.query_one("#ccp-character-card-view").is_mounted
        assert not list(screen.query(PersonasCharacterEditorWidget))
        assert gaps
        assert max(gaps) < 0.25, f"event loop stalled for {max(gaps):.3f}s"
