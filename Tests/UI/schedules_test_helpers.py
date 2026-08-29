"""Shared stubs and paint oracles for the Schedules UI test files.

task-23106 review round (F15): the scheduling-service stub had been
copy-pasted into five test files while ``test_schedules_workbench.py``
already shipped reusable mixins, and the compositor center-probe
re-implemented the shared painted-region oracle. Everything lives here
once now.
"""

from __future__ import annotations

from typing import Any

import pytest


class MockServerClient:
    """Stub server client for test scheduling services."""

    def __init__(self, notifications_service: Any = None) -> None:
        self.notifications_service = notifications_service


class MockSchedulingDB:
    """Stub scheduled-tasks DB for test scheduling services."""

    def __init__(self, sync_state: dict | None = None, conflicts: list | None = None) -> None:
        self._sync_state = sync_state or {}
        self._conflicts = conflicts or []

    def get_sync_state(self, owner_id: str):
        return dict(self._sync_state)

    def update_sync_state(self, owner_id: str, **kwargs) -> None:
        self._sync_state.update(kwargs)

    def get_conflicts(self, owner_id: str, primitive=None):
        return self._conflicts


class MockSchedulingServiceMixin:
    """Common attributes expected by the SchedulesWorkbench UI.

    Subclass and implement ``list_tasks`` (and whichever mutation methods
    the test drives). ``server_client``/``db`` are class-level defaults;
    assign instance attributes to specialize.
    """

    owner_id = "local"
    server_client = MockServerClient()
    db = MockSchedulingDB()
    sync_engine = None

    def set_owner(self, owner_id: str) -> None:
        self.owner_id = owner_id

    async def sync_now(self, owner_id: str | None = None):
        return None


# --- compositor paint oracles ---------------------------------------------
#
# ``Widget.region`` is reported in an UNCLIPPED coordinate space -- a
# widget scrolled out of (or simply clipped inside) a scrollable ancestor
# still has a plausible region the ancestor never paints. Only the
# compositor (``App.get_widget_at``) answers what a live terminal actually
# renders (see lessons-live-verification.md).


def assert_painted_at_own_region(host, widget) -> None:
    """Fail unless the compositor paints ``widget`` at its own top-left."""
    region = widget.region
    try:
        hit_widget, _hit_region = host.get_widget_at(region.x + 1, region.y)
    except Exception as exc:  # textual.errors.NoWidget
        pytest.fail(
            f"nothing is painted at {widget!r}'s own region {region!r}: {exc}"
        )
    assert hit_widget is widget, (
        f"the compositor paints {hit_widget!r} at {region!r}, not {widget!r} "
        "itself -- the widget's display chain is all-True but it is not "
        "actually visible on screen"
    )


def painted_at_own_center(host, widget) -> bool:
    """Center-probe variant: True when the compositor paints ``widget``
    (or one of its descendants -- Select/TextArea paint through children)
    at the widget's own center cell."""
    region = widget.region
    if region.height <= 0 or region.width <= 0:
        return False
    cx, cy = region.center
    try:
        target, _ = host.get_widget_at(int(cx), int(cy))
    except Exception:
        return False
    return target is widget or widget in list(target.ancestors)
