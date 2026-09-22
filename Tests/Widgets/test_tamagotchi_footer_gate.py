"""TASK-32905: the footer pet is reachable when enabled and absent when not.

``Widgets/Tamagotchi/`` shipped with its storage half wired into eight
backup/recovery and private-SQLite modules while the widget half had no host
at all -- no screen, no route, no ``compose()`` that mounted it. These tests
pin the host it now has, and both sides of its gate.

The host is the footer status bar, which is what the widget was written for:
"Status Bar Integration" is integration example #1 in
``Docs/Development/Textual-Tamagotchis.md``, ``CompactTamagotchi`` is
documented as "optimized for status bars", and the package's own example app
docks one in a footer under this exact id.

The gate defaults to OFF. A pet appearing unbidden in someone's terminal is
not a neutral change, so "disabled" has to mean nothing is constructed,
nothing is mounted, and -- the part that is easy to get wrong -- nothing is
ticking.
"""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Tamagotchi.base_tamagotchi import (
    BaseTamagotchi,
    CompactTamagotchi,
)
from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import MemoryStorage

#: The name `BaseTamagotchi.on_mount` gives its decay timer, for the id the
#: footer mounts the pet under.
_PET_TIMER = "tamagotchi-footer-tamagotchi-update"


class _FooterApp(App):
    """Bare host for one real ``AppFooterStatus``, with notifications recorded.

    Deliberately not the full app: booting ``TldwCli`` would rebuild the CSS
    bundle, and the question here is only what the footer composes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.notifications: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield AppFooterStatus(show_token_count=False)

    def notify(self, message: str, *, severity: str = "information", **kwargs) -> None:
        self.notifications.append((message, severity))


def _patch_gate(monkeypatch, **values) -> MemoryStorage:
    """Point ``[tamagotchi]`` at ``values`` and keep storage off the real disk.

    ``ConfigFileStorage()`` creates the config directory and seeds the pet file
    in its constructor, so a test that let it run would write to the developer's
    own ``~/.config``. Patching the class (rather than passing a storage in)
    keeps the assertion honest: the footer still has to ASK for the adapter
    that ``Widgets/Tamagotchi/recovery.py`` registers as the
    ``tamagotchi.config`` backup owner.
    """
    from tldw_chatbook import config
    from tldw_chatbook.Widgets.Tamagotchi import tamagotchi_storage as storage_module

    real = config.get_cli_setting

    def fake(section, key=None, default=config._CLI_SETTING_DEFAULT_UNSET):
        if section == "tamagotchi":
            return values.get(key, default)
        return real(section, key, default)

    monkeypatch.setattr(config, "get_cli_setting", fake)
    storage = MemoryStorage()
    monkeypatch.setattr(storage_module, "ConfigFileStorage", lambda: storage)
    return storage


def _live_timer_names(app: App) -> set[str]:
    """Every timer name currently held anywhere in the running app tree."""
    names: set[str] = set()
    pumps = [app, *app.screen.walk_children(with_self=True)]
    for pump in pumps:
        for timer in getattr(pump, "_timers", ()):
            names.add(timer.name)
    return names


@pytest.mark.asyncio
async def test_footer_pet_is_absent_and_idle_while_disabled(monkeypatch):
    """Disabled: no widget, no timer. The default for every user."""
    _patch_gate(monkeypatch, enabled=False)

    app = _FooterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        footer = app.query_one(AppFooterStatus)

        assert footer._tamagotchi is None
        assert not app.query(BaseTamagotchi)
        # The part a "does it render?" check would miss: nothing ticking.
        assert _PET_TIMER not in _live_timer_names(app)
        assert not [n for n in _live_timer_names(app) if "tamagotchi" in n]


@pytest.mark.asyncio
async def test_footer_pet_mounts_and_ticks_while_enabled(monkeypatch):
    """Enabled: the pet is really in the footer, persisted and ticking."""
    storage = _patch_gate(monkeypatch, enabled=True, name="Pixel")

    app = _FooterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        footer = app.query_one(AppFooterStatus)

        pets = app.query(CompactTamagotchi)
        assert len(pets) == 1
        pet = pets.first()
        assert pet.id == "footer-tamagotchi"
        assert pet.pet_name == "Pixel"
        assert pet.parent is footer
        assert footer._tamagotchi is pet
        # The backup-owned adapter, not the in-memory default the widget would
        # otherwise fall back to.
        assert pet.storage is storage
        assert _PET_TIMER in _live_timer_names(app)


def test_shipped_config_default_keeps_the_pet_off():
    """The gate ships off, so an untouched install never grows a pet."""
    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    section = CONFIG_TOML_CONTENT.split("[tamagotchi]", 1)[1].split("\n[", 1)[0]
    assert "enabled = false" in section


@pytest.mark.asyncio
async def test_a_real_app_screen_hosts_the_pet(monkeypatch):
    """The embedding resolves through the app's own chrome, not just a harness.

    Every destination in the app is a ``BaseAppScreen``, and ``BaseAppScreen.
    compose()`` mounts the ``AppFooterStatus`` the pet now rides in -- so this
    is the equivalent of resolving a route for an embedded surface.
    """
    from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen

    class _Screen(BaseAppScreen):
        def compose_content(self):
            return iter(())

    class _ShellApp(App):
        def compose(self) -> ComposeResult:
            return iter(())

    _patch_gate(monkeypatch, enabled=True, name="Pixel")

    app = _ShellApp()
    async with app.run_test() as pilot:
        await app.push_screen(_Screen(app, "home"))
        await pilot.pause()

        assert len(app.screen.query(AppFooterStatus)) == 1
        pet = app.screen.query_one(CompactTamagotchi)
        assert pet.pet_name == "Pixel"
        assert isinstance(pet.parent, AppFooterStatus)


@pytest.mark.asyncio
async def test_footer_pet_reports_critical_stats_and_death_to_the_user(monkeypatch):
    """The message protocol reaches a person, not just the message bus.

    ``TamagotchiStatCritical`` was declared but never posted by anything --
    the same way the widget itself was never mounted.
    """
    _patch_gate(monkeypatch, enabled=True, name="Pixel")

    app = _FooterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        pet = app.query_one(CompactTamagotchi)

        pet.health = 10  # under the 30 threshold
        pet._check_conditions()
        await pilot.pause()
        assert [s for m, s in app.notifications if "health" in m] == ["error"]

        # Edge-triggered: staying critical must not renotify every tick.
        pet._check_conditions()
        await pilot.pause()
        assert len([m for m, _ in app.notifications if "health" in m]) == 1

        app.notifications.clear()
        pet.hunger = 100  # starvation
        pet._check_conditions()
        await pilot.pause()
        deaths = [m for m, s in app.notifications if "has died" in m and s == "error"]
        assert len(deaths) == 1
        assert "starvation" in deaths[0]


@pytest.mark.asyncio
async def test_background_pet_neither_decays_nor_writes_back(monkeypatch):
    """A pet on a suspended screen must not clobber the visible pet's state.

    Textual suspends an installed screen instead of unmounting it, so every
    screen the user has visited keeps its own footer -- and its own pet --
    alive against ONE shared storage key. Without this guard the stalest
    writer wins and the pet the user can see loses its progress.
    """
    _patch_gate(monkeypatch, enabled=True)

    app = _FooterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        pet = app.query_one(CompactTamagotchi)
        pet.happiness = 42
        pet._state_authoritative = True

        # Pretend this pet's screen went to the background.
        monkeypatch.setattr(type(app.screen), "is_active", property(lambda self: False))
        pet._periodic_update()

        assert pet.happiness == 42, "a background pet kept decaying"
        assert not pet._state_authoritative, "a background pet still claims authority"


@pytest.mark.asyncio
async def test_pet_stays_off_for_a_config_that_predates_the_section(monkeypatch):
    """An existing user upgrading has no ``[tamagotchi]`` section at all.

    The other gate tests either set ``enabled`` explicitly or assert the shipped
    *template* says ``false`` -- neither exercises the code default, which is the
    only thing standing between an upgrading user and a pet appearing unbidden in
    their footer. Their ``config.toml`` was written before this section existed,
    so ``get_cli_setting`` falls through to the caller's default and the template
    is irrelevant to them.

    Passing no values to ``_patch_gate`` reproduces exactly that: the fake
    returns ``values.get(key, default)``, so the call site's own default decides.

    This test is born red against ``get_cli_setting("tamagotchi", "enabled", True)``
    -- verified by flipping it -- which the template-based check is not.
    """
    _patch_gate(monkeypatch)  # deliberately empty: no section, no keys

    app = _FooterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        footer = app.query_one(AppFooterStatus)

        assert footer._tamagotchi is None
        assert not app.query(BaseTamagotchi)
        assert not [n for n in _live_timer_names(app) if "tamagotchi" in n]
