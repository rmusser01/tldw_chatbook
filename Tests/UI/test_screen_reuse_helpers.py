"""Unit contracts for the screen-reuse cache helpers (TASK-24452, Qodo #2402).

`Tests/UI/test_screen_reuse.py` proves the behavior through the running app;
these pin each helper's own decision table in isolation -- cache hits and
misses, runtime-identity scoping, stale-entry disposal (including the
bounded in-stack leak), and the install-failure fallback -- against a stub
app, so a regression names the helper rather than a navigation symptom.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Navigation.screen_state_store import RuntimeIdentity
from tldw_chatbook.app import TldwCli

LOCAL = RuntimeIdentity(active_source="local")
SERVER = RuntimeIdentity(active_source="server", active_server_id="srv-1")


class _StubScreen:
    def __init__(self) -> None:
        self.removed = False

    def remove(self):
        self.removed = True


class _StubApp:
    """Just the surface the two helpers touch, recorded."""

    _reusable_navigation_screen = TldwCli._reusable_navigation_screen
    _retain_reusable_navigation_screen = TldwCli._retain_reusable_navigation_screen

    def __init__(self, *, install_raises: bool = False) -> None:
        self._screen_stacks: dict[str, list] = {"_default": []}
        self.installed: list[tuple] = []
        self.uninstalled: list = []
        self._install_raises = install_raises

    def install_screen(self, screen, name: str) -> None:
        if self._install_raises:
            raise RuntimeError("install refused")
        self.installed.append((screen, name))

    def uninstall_screen(self, screen) -> None:
        self.uninstalled.append(screen)


def test_lookup_misses_before_any_retention() -> None:
    app = _StubApp()
    assert app._reusable_navigation_screen("home", LOCAL) is None
    app._reusable_screen_instances = {}
    assert app._reusable_navigation_screen("home", LOCAL) is None


def test_retain_then_lookup_hits_for_the_same_identity() -> None:
    app = _StubApp()
    screen = _StubScreen()
    app._retain_reusable_navigation_screen("home", LOCAL, screen)
    assert len(app.installed) == 1
    installed_screen, name = app.installed[0]
    assert installed_screen is screen
    assert name.startswith("tldw-reusable:home:"), (
        "install names carry the instance id so a lingering stale install "
        "can never block a replacement's"
    )
    assert app._reusable_navigation_screen("home", LOCAL) is screen


def test_identity_mismatch_drops_uninstalls_and_removes() -> None:
    app = _StubApp()
    screen = _StubScreen()
    app._retain_reusable_navigation_screen("home", LOCAL, screen)
    assert app._reusable_navigation_screen("home", SERVER) is None
    assert "home" not in app._reusable_screen_instances
    assert app.uninstalled == [screen]
    assert screen.removed is True


def test_identity_mismatch_with_screen_in_stack_leaks_boundedly() -> None:
    """A current screen cannot be uninstalled; the entry still leaves."""
    app = _StubApp()
    screen = _StubScreen()
    app._retain_reusable_navigation_screen("home", LOCAL, screen)
    app._screen_stacks["_default"].append(screen)
    assert app._reusable_navigation_screen("home", SERVER) is None
    assert "home" not in app._reusable_screen_instances, (
        "the cache entry must leave even when disposal is deferred -- the "
        "stale instance may linger installed, but must never be REUSED"
    )
    assert not app.uninstalled
    assert screen.removed is False


def test_install_failure_falls_back_to_per_visit_construction() -> None:
    app = _StubApp(install_raises=True)
    screen = _StubScreen()
    app._retain_reusable_navigation_screen("home", LOCAL, screen)
    assert getattr(app, "_reusable_screen_instances", {}).get("home") is None, (
        "an uninstalled screen must never be cached: switch_screen would "
        "unmount it and the next visit would resume a torn-down instance "
        "-- the 2026-07-11 freeze class"
    )


def test_disposal_failure_still_returns_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _StubApp()
    screen = _StubScreen()
    app._retain_reusable_navigation_screen("home", LOCAL, screen)
    monkeypatch.setattr(
        _StubScreen,
        "remove",
        lambda self: (_ for _ in ()).throw(RuntimeError("remove exploded")),
    )
    assert app._reusable_navigation_screen("home", SERVER) is None
    assert "home" not in app._reusable_screen_instances
