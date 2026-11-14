"""Both Buddy management controls use the same app-owned opener."""

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from Tests.UI.test_persona_buddy_widget import (
    _BuddyApp,
    _compositor_text,
    _FakeController,
    _wait_until,
)
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    build_composer_menu_entries,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
    PersonaBuddyWidget,
)


def test_composer_menu_has_a_named_buddy_action():
    choices = [
        entry for entry in build_composer_menu_entries() if entry.action_id == "buddy"
    ]
    assert len(choices) == 1
    assert choices[0].label == "Buddy…"
    assert choices[0].enabled


@pytest.mark.asyncio
async def test_floating_settings_opens_the_shared_management_flow(monkeypatch):
    from tldw_chatbook.UI.Navigation import buddy_management

    opened = []
    monkeypatch.setattr(buddy_management, "open_buddy_management", opened.append)
    app = _BuddyApp(_FakeController())
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))
        assert buddy.query("#persona-buddy-settings"), (
            "Floating Buddy settings control is missing"
        )
        button = buddy.query_one("#persona-buddy-settings", Button)
        assert button.tooltip == "Buddy & Persona Management"
        await pilot.pause()
        assert button.display and button.region.width >= 3
        assert await pilot.click(button, offset=(1, 0)), (
            button.region,
            _compositor_text(app.screen),
        )
        assert opened == [app]


@pytest.mark.asyncio
async def test_body_click_and_enter_open_interaction_without_geometry_write(
    monkeypatch,
):
    from tldw_chatbook.UI.Navigation import buddy_management

    opened = []
    monkeypatch.setattr(buddy_management, "open_buddy_interaction", opened.append)
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))
        await pilot.pause()
        writes = []
        monkeypatch.setattr(buddy, "_schedule_geometry_persist", writes.append)
        await pilot.click(buddy.query_one("#persona-buddy-frame"), offset=(6, 3))
        assert opened == [app]
        await pilot.press("enter")
        assert opened == [app, app]
        assert writes == []


# --- TASK-21236: the ensure-fallback arm for explicit Buddy actions ------------


class _ConstructedBuddyController:
    """Fake controller with the management-surface seam methods."""

    def __init__(self) -> None:
        self.patches: list[dict] = []
        self.persisted: list[object] = []

    def current_preferences(self):
        from tldw_chatbook.Persona_Buddy.preferences import (
            PersonaBuddyPreferences,
        )

        return PersonaBuddyPreferences()

    def apply_preferences_patch(self, **changes):
        self.patches.append(changes)
        return {"revision": len(self.patches), **changes}

    async def persist_preferences_revision(self, revision):
        self.persisted.append(revision)
        return True

    def rollback_preferences_revision(self, revision, previous):
        pass


class _DisabledPreferencesApp(ConsolidatedCSSApp):
    """App double whose persisted Buddy preferences say disabled."""

    def __init__(self) -> None:
        super().__init__()
        self.app_config = {
            "persona_buddy": {"enabled": False},
            "buddy_interaction": {},
        }
        self.ensure_calls = 0
        self.constructed: _ConstructedBuddyController | None = None

    def ensure_persona_buddy_controller(self):
        """The ensure arm: constructs regardless of the persisted flag."""
        self.ensure_calls += 1
        self.constructed = _ConstructedBuddyController()
        return self.constructed

    def notify(self, *args, **kwargs):  # noqa: ARG002 - fake surface
        pass

    def reconcile_persona_buddy_view(self):
        return None


@private_profile_test
@pytest.mark.asyncio
async def test_buddy_action_from_disabled_profile_completes_via_ensure_fallback(
    request,
):
    """TASK-21236: the screen action constructs the controller from disabled.

    The old "Use for Buddy" arm now lives in `BuddyManagementCoordinator.
    controller`: read the passive property, and when it is None call
    `ensure_persona_buddy_controller()`. From a profile whose preferences
    still say disabled, the passive property returns None -- this ensure
    arm is the entire reason enabling from a disabled state works. The
    test drives a full visibility action through the coordinator and
    asserts it completed via the CONSTRUCTED controller; remove the ensure
    fallback (passive read only) and the action dies with
    "Buddy storage is unavailable".
    """
    from tldw_chatbook.UI.Navigation.buddy_management import (
        BuddyManagementCoordinator,
    )

    app = _DisabledPreferencesApp()
    async with app.run_test(size=(80, 24)) as pilot:
        coordinator = BuddyManagementCoordinator(app)
        from tldw_chatbook.Persona_Buddy.preferences import (
            parse_persona_buddy_preferences,
        )

        assert (
            parse_persona_buddy_preferences(app.app_config["persona_buddy"]).enabled
            is False
        ), "premise: the profile's persisted Buddy preferences say disabled"

        controller = coordinator.controller

        assert app.ensure_calls == 1, (
            "the coordinator did not go through the ensure-fallback arm"
        )
        assert controller is app.constructed

        # A full explicit Buddy action, end to end, through that controller.
        await coordinator._set_visibility("disable")
        await pilot.pause()

        assert app.constructed.patches == [{"enabled": False}]
        assert app.constructed.persisted, "the action never reached the controller"
