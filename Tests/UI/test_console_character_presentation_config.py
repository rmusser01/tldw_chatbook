"""Character paint preserves checked rail refresh and lightweight consumers."""
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile

from tldw_chatbook.UI.Console_Modules.wiring import _sync_character_context_presentation


@pytest.mark.parametrize("paused", [False, True])
def test_character_snapshot_paints_before_checked_rail_refresh(paused):
    events = []
    state = object()
    widget = SimpleNamespace(sync_state=lambda value: events.append(("widget", value)))
    rail = SimpleNamespace(sync_character_context=lambda value: events.append(("rail", value)))
    screen = SimpleNamespace(
        is_attached=True,
        app=SimpleNamespace(screen_stack=[object()]),
        query_one=lambda selector, *args: widget if selector == "#console-character-context" else rail,
        _console_sync_maintenance_paused=paused,
        _request_console_control_bar_sync=lambda **kwargs: events.append(("retry", kwargs)),
        _current_console_rail_state=lambda: events.append(("read", None)) or "current",
        _sync_console_rail_visibility_if_changed=lambda value: events.append(("visibility", value)),
    )
    _sync_character_context_presentation(screen, state)
    assert events[:2] == [("widget", state), ("rail", state)]
    assert events[2:] == ([("retry", {"delayed": True})] if paused else [("read", None), ("visibility", "current")])


def test_character_lightweight_consumer_keeps_direct_rail_callback():
    events = []
    screen = SimpleNamespace(
        is_attached=True,
        app=SimpleNamespace(screen_stack=[object()]),
        query_one=lambda *args: SimpleNamespace(sync_state=lambda value: None),
        _current_console_rail_state=lambda: "current",
        _sync_console_rail_visibility_if_changed=events.append,
    )
    _sync_character_context_presentation(screen, object())
    assert events == ["current"]


@pytest.mark.parametrize("closed", ["_closing", "_console_sync_maintenance_paused"])
def test_character_closed_entry_defers_rail_then_replays_fresh_state(closed):
    events = []
    version = [1]
    screen = SimpleNamespace(
        is_attached=True,
        app=SimpleNamespace(screen_stack=[object()]),
        query_one=lambda *args: SimpleNamespace(sync_state=lambda value: events.append("paint")),
        _request_console_control_bar_sync=lambda **kwargs: events.append("retry"),
        _current_console_rail_state=lambda: events.append("read") or version[0],
        _sync_console_rail_visibility_if_changed=events.append,
    )
    setattr(screen, closed, True)
    _sync_character_context_presentation(screen, object())
    assert events == ["paint", "retry"]
    setattr(screen, closed, False)
    version[0] = 2
    _sync_character_context_presentation(screen, object())
    assert events == ["paint", "retry", "paint", "read", 2]


@pytest.mark.asyncio
async def test_original_mounted_character_restoration_with_bootstrap_profile(monkeypatch):
    # Keep the original mounted assertion body; this module uses the actual
    # collection-time private config participant rather than retargeting it.
    from Tests.UI.test_console_character_context import (
        test_character_mount_loads_groups_without_duplicate_cold_resume_worker,
    )
    await test_character_mount_loads_groups_without_duplicate_cold_resume_worker(monkeypatch)


@pytest.mark.asyncio
async def test_original_character_preference_writer_with_bootstrap_profile(monkeypatch):
    from Tests.UI.test_console_character_context import (
        test_canonical_writer_persists_first_use_close_reopen_and_legacy,
    )
    await test_canonical_writer_persists_first_use_close_reopen_and_legacy(monkeypatch)


@pytest.mark.asyncio
async def test_original_character_avatar_repaint_with_bootstrap_profile():
    from Tests.UI.test_console_character_context import (
        test_production_avatar_repaint_never_overwrites_identity_variants,
    )
    await test_production_avatar_repaint_never_overwrites_identity_variants()
