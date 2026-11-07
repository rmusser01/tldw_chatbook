"""Workflows mount defers its load sequence to an exclusive worker."""

import pytest

from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen

pytestmark = pytest.mark.bootstrap_profile


def test_on_mount_spawns_the_load_worker_and_returns():
    screen = WorkflowsScreen.__new__(WorkflowsScreen)
    spawned = []

    def capture(coro, **kwargs):
        spawned.append((kwargs.get("group"), kwargs.get("exclusive"), coro))
        coro.close()  # never awaited: close to avoid a warning

    screen.run_worker = capture
    screen.on_mount()

    assert len(spawned) == 1, "on_mount must not load inline"
    group, exclusive, _coro = spawned[0]
    assert group == "workflows-mount-load"
    assert exclusive is True
