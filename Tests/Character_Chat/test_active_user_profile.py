import pytest
from tldw_chatbook.Character_Chat.active_user_profile import (
    resolve_active_user_profile_name,
    set_active_user_profile,
    clear_active_user_profile,
    get_active_user_profile_pointer,
)


class _FakeService:
    def __init__(self, profiles):
        self._profiles = profiles

    def list_user_profiles(self, active_only: bool = False):
        return list(self._profiles)


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch):
    """Route the config read/write seam at an in-memory dict."""
    store = {}
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    monkeypatch.setattr(mod, "get_cli_setting", lambda section, key, default=None: store.get((section, key), default))
    def _save(section, key, value):
        store[(section, key)] = value
        return True
    monkeypatch.setattr(mod, "save_setting_to_cli_config", _save)
    return store


def test_unset_pointer_resolves_none():
    assert resolve_active_user_profile_name(_FakeService([{"name": "Sam"}])) is None


def test_set_then_resolve(_isolated_config):
    assert set_active_user_profile("Sam") is True
    svc = _FakeService([{"name": "Sam"}, {"name": "Kai"}])
    assert resolve_active_user_profile_name(svc) == "Sam"


def test_dangling_pointer_resolves_none(_isolated_config):
    set_active_user_profile("Ghost")
    assert resolve_active_user_profile_name(_FakeService([{"name": "Sam"}])) is None


def test_clear(_isolated_config):
    set_active_user_profile("Sam")
    assert clear_active_user_profile() is True
    assert get_active_user_profile_pointer() is None
    assert resolve_active_user_profile_name(_FakeService([{"name": "Sam"}])) is None


def test_resolver_never_raises_on_broken_service(_isolated_config):
    set_active_user_profile("Sam")
    class _Boom:
        def list_user_profiles(self, active_only: bool = False):
            raise RuntimeError("store unreadable")
    assert resolve_active_user_profile_name(_Boom()) is None
