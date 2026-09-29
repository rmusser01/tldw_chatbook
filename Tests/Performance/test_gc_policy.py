"""ADR-198 / PERF-11 (TASK-33270): the long-lived boot heap is frozen.

Nothing used to call ``gc.freeze()``, so every automatic generation-2
collection walked the whole boot heap (~0.57M objects): 130-871 ms UI stalls on
screen switches in the 2026-09-27 audit, against ~0 ms for the freeze itself.
"""

from __future__ import annotations

import asyncio
import gc
import os
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test


def test_freeze_long_lived_heap_collects_young_generations_then_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A young collection first (cheap), never a full one, then the freeze."""
    from tldw_chatbook.Utils import ui_responsiveness

    calls: list[tuple] = []
    monkeypatch.setattr(
        ui_responsiveness.gc,
        "collect",
        lambda generation=2: calls.append(("collect", generation)) or 0,
    )
    monkeypatch.setattr(ui_responsiveness.gc, "freeze", lambda: calls.append(("freeze",)))

    ui_responsiveness.freeze_long_lived_heap("unit-test")

    assert calls == [("collect", 1), ("freeze",)]


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_the_boot_heap_is_frozen_once_the_ui_is_ready(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``_ui_ready`` most of the boot heap sits in the permanent generation.

    Args:
        request: pytest fixture the private-profile runner needs.
        monkeypatch: pytest fixture used for the scratch environment.
    """
    for name in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    config_file = Path(os.environ["TLDW_CONFIG_PATH"])
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[first_run]\nsetup_completed = true\n\n[splash_screen]\nenabled = false\n"
    )
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    from tldw_chatbook.config import load_settings

    load_settings(force_reload=True)
    from tldw_chatbook.app import TldwCli

    gc.unfreeze()
    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(200):
            if getattr(app, "_ui_ready", False):
                break
            await asyncio.sleep(0.05)
            await pilot.pause()
        assert app._ui_ready, "the app never reached _ui_ready"
        await pilot.pause()
        frozen = gc.get_freeze_count()

    assert frozen > 100_000, f"only {frozen} objects were frozen after _ui_ready"
