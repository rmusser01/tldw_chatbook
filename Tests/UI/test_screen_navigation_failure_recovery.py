"""task-2720: an exception escaping the navigation worker must degrade loudly
and recoverably.

Live incident (2026-08-06 UAT, dev b0185749c): a transient PermissionError
inside `handle_screen_navigation` left the app with the nav-bar highlight on
the destination, the body on the old screen, no user-visible message, and the
destination unreachable for the rest of the session (the bar's already-active
check swallowed every retry click).
"""

from __future__ import annotations

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.UI.Navigation.main_navigation import (
    MainNavigationBar,
    NavigateToScreen,
)

pytestmark = pytest.mark.unit


class _RecordingBar:
    def __init__(self):
        self.restored: list[str] = []

    def restore_active(self, route: str) -> None:
        self.restored.append(route)


class _FakeOutgoingScreen:
    """Outgoing screen with its own nav bar, like every BaseAppScreen."""

    screen_name = "library"

    def __init__(self, bar: _RecordingBar):
        self._bar = bar

    def query_one(self, _selector):
        return self._bar


class _FakeOutgoingStudyScreen:
    """Outgoing screen mirroring StudyScreen (task-2854): ``screen_name``
    stays ``"study"`` (routing still folds it under Library via
    ``resolve_shell_route``), but ``nav_bar_active`` is ``""`` so the
    screen's own nav bar shows no highlight while it is on top -- see
    ``BaseAppScreen.nav_bar_active`` / ``StudyScreen.__init__``.
    """

    screen_name = "study"
    nav_bar_active = ""

    def __init__(self, bar: _RecordingBar):
        self._bar = bar

    def query_one(self, _selector):
        return self._bar


def _wire_failing_navigation(
    app,
    monkeypatch,
    bar: _RecordingBar,
    fail_flag: dict,
    outgoing_screen_cls=_FakeOutgoingScreen,
):
    """Point the app at fake screens and one injected unguarded failure."""

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            self.app_instance = app_instance

    monkeypatch.setattr(
        app,
        "_resolve_screen_navigation_target",
        lambda target: ("chat", "chat", FakeTargetScreen),
    )

    switched = []

    async def fake_switch_screen(screen):
        switched.append(screen)

    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)

    real_identity = type(app)._current_runtime_identity

    def maybe_failing_identity(self):
        if fail_flag["on"]:
            raise PermissionError("[Errno 13] injected: task-2720")
        return real_identity(self)

    # `_current_runtime_identity` is one of the genuinely unguarded steps the
    # incident's exception class could have escaped from.
    monkeypatch.setattr(
        type(app), "_current_runtime_identity", maybe_failing_identity
    )
    monkeypatch.setattr(
        type(app), "screen", property(lambda self: outgoing_screen_cls(bar))
    )
    app._initial_screen_pushed = True
    return switched


@pytest.mark.asyncio
async def test_escaped_navigation_exception_notifies_and_rolls_back_nav_bar(
    monkeypatch,
):
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    notifications: list[tuple] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs)),
    )
    bar = _RecordingBar()
    _wire_failing_navigation(app, monkeypatch, bar, {"on": True})

    with pytest.raises(PermissionError):
        await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert notifications, "user got no message about the failed navigation"
    assert bar.restored == ["library"], (
        "nav bar was not rolled back to the screen actually on the stack"
    )


@pytest.mark.asyncio
async def test_navigation_failure_while_study_screen_on_top_does_not_box_library(
    monkeypatch,
):
    """task-2854 review finding: StudyScreen clears its own nav-bar identity
    via ``nav_bar_active = ""`` (its ``screen_name`` still folds to Library's
    destination for routing purposes -- see ``BaseAppScreen.nav_bar_active``).
    The failure-recovery rollback in ``_notify_navigation_failure`` must
    consult ``nav_bar_active``, not ``screen_name``, when restoring the
    highlight -- otherwise ``restore_active("study")`` resolves through
    ``resolve_shell_route`` to Library's destination and re-boxes
    "⌃3 Library" while Study is still the screen actually on the stack,
    reintroducing the exact defect task-2854 fixed."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    monkeypatch.setattr(app, "notify", lambda message, **kwargs: None)
    bar = _RecordingBar()
    _wire_failing_navigation(
        app,
        monkeypatch,
        bar,
        {"on": True},
        outgoing_screen_cls=_FakeOutgoingStudyScreen,
    )

    with pytest.raises(PermissionError):
        await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert bar.restored == [""], (
        f"nav bar was restored with {bar.restored!r} instead of Study's "
        "empty nav_bar_active -- Library would get boxed while Study is "
        "still the screen on the stack"
    )


@pytest.mark.asyncio
async def test_dispatched_navigation_failure_still_records_worker_failed(
    monkeypatch,
):
    """The recovery guard must re-raise so the ADR-029 diagnostics line
    (`worker_failed operation=handle_screen_navigation`) keeps being written."""
    from Tests.UI.app_factory import _build_test_app

    recorded: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append(
            {"component": component, "event": event, **fields}
        ),
    )

    app = _build_test_app()
    notifications: list[str] = []

    async def failing_locked(message):
        raise PermissionError("[Errno 13] injected: task-2720")

    monkeypatch.setattr(app, "_handle_screen_navigation_locked", failing_locked)

    async with app.run_test(size=(120, 40)) as pilot:
        monkeypatch.setattr(
            app, "notify", lambda message, **kwargs: notifications.append(message)
        )
        app._initial_screen_pushed = True
        recorded.clear()
        app.post_message(NavigateToScreen("library"))
        await pilot.pause()
        await pilot.pause()

    assert [
        r
        for r in recorded
        if r["event"] == "worker_failed"
        and r["operation"] == "handle_screen_navigation"
        and r["exception_type"] == "PermissionError"
    ], f"worker_failed line lost, got {recorded}"
    assert notifications, "dispatched navigation failure showed the user nothing"


@pytest.mark.asyncio
async def test_restore_active_reverts_optimistic_highlight():
    """Widget-level: restore_active undoes the click-time optimistic state so
    the destination is re-clickable and the highlight matches reality."""

    class BarApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="library")

    app = BarApp()
    async with app.run_test(size=(200, 20)) as pilot:
        bar = app.query_one(MainNavigationBar)
        chat_button = bar.query_one("#nav-console")
        assert bar._activate_navigation_button(chat_button) is True
        assert bar.active_destination_id == "console"

        bar.restore_active("library")
        await pilot.pause()

        assert bar.active_destination_id == "library"
        assert not chat_button.has_class("is-active")
        library_button = bar.query_one("#nav-library")
        assert library_button.has_class("is-active")
        # The optimistic state is gone, so a fresh click on the failed
        # destination must request navigation again instead of no-opping.
        assert bar._activate_navigation_button(chat_button) is True
