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


class _FakeScopeService:
    def __init__(self, payload=None, exc=None):
        self.payload = payload
        self.exc = exc
        self.calls: list[dict] = []

    async def list_user_profiles(self, mode="local", **kwargs):
        self.calls.append({"mode": mode, **kwargs})
        if self.exc is not None:
            raise self.exc
        return self.payload


@pytest.mark.asyncio
async def test_async_resolver_server_mode_matches_server_profile(_isolated_config):
    set_active_user_profile("Sam")
    scope = _FakeScopeService(payload=[{"name": "Sam"}, {"name": "Rae"}])
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    assert await mod.resolve_active_user_profile_name_async(scope, mode="server") == "Sam"
    assert scope.calls == [{"mode": "server"}]


@pytest.mark.asyncio
async def test_async_resolver_server_mode_accepts_items_payload(_isolated_config):
    set_active_user_profile("Sam")
    scope = _FakeScopeService(payload={"items": [{"name": "Sam"}]})
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    assert await mod.resolve_active_user_profile_name_async(scope, mode="server") == "Sam"


@pytest.mark.asyncio
async def test_async_resolver_server_dangling_resolves_none(_isolated_config):
    set_active_user_profile("Ghost")
    scope = _FakeScopeService(payload=[{"name": "Sam"}])
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    assert await mod.resolve_active_user_profile_name_async(scope, mode="server") is None


@pytest.mark.asyncio
async def test_async_resolver_server_error_resolves_none(_isolated_config):
    set_active_user_profile("Sam")
    scope = _FakeScopeService(exc=RuntimeError("connection refused"))
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    assert await mod.resolve_active_user_profile_name_async(scope, mode="server") is None


@pytest.mark.asyncio
async def test_async_resolver_server_without_scope_service_resolves_none(_isolated_config):
    set_active_user_profile("Sam")
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    assert await mod.resolve_active_user_profile_name_async(None, mode="server") is None


@pytest.mark.asyncio
async def test_async_resolver_local_and_unknown_modes_delegate_to_sync(_isolated_config):
    set_active_user_profile("Sam")
    local = _FakeService([{"name": "Sam"}])
    scope = _FakeScopeService(payload=[{"name": "SERVER-ONLY"}])
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    for mode in ("local", None, "", "LOCAL", "garbage"):
        assert await mod.resolve_active_user_profile_name_async(
            scope, mode=mode, local_service=local
        ) == "Sam"
    assert scope.calls == []  # scope service never consulted off server mode


@pytest.mark.asyncio
async def test_async_resolver_no_pointer_short_circuits(_isolated_config):
    scope = _FakeScopeService(payload=[{"name": "Sam"}])
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    assert await mod.resolve_active_user_profile_name_async(scope, mode="server") is None
    assert scope.calls == []  # byte-compat: no backend call without a pointer


def test_resolve_runtime_backend_mode_guards():
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    class _App:
        def get_authoritative_runtime_source(self):
            return "SERVER"
    assert mod.resolve_runtime_backend_mode(_App()) == "server"
    class _Raises:
        def get_authoritative_runtime_source(self):
            raise RuntimeError("boom")
    assert mod.resolve_runtime_backend_mode(_Raises()) == "local"
    assert mod.resolve_runtime_backend_mode(object()) == "local"
    assert mod.resolve_runtime_backend_mode(None) == "local"
