from types import SimpleNamespace

from tldw_chatbook.Personal_Context.service import ProfileOperationalState
from tldw_chatbook.app import TldwCli


class _Service:
    def __init__(self, state: ProfileOperationalState) -> None:
        self._state = state

    def status(self):
        return SimpleNamespace(state=self._state)


def test_app_reuses_available_personal_context_service(monkeypatch) -> None:
    service = _Service(ProfileOperationalState.READY)
    calls: list[None] = []

    def bootstrap():
        calls.append(None)
        return service

    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.bootstrap.bootstrap_personal_context_service",
        bootstrap,
    )
    app = SimpleNamespace()

    first = TldwCli.get_personal_context_service(app)
    second = TldwCli.get_personal_context_service(app, retry_locked=True)

    assert first is service
    assert second is service
    assert calls == [None]


def test_app_explicit_retry_rebootstraps_a_locked_service(monkeypatch) -> None:
    locked = _Service(ProfileOperationalState.LOCKED)
    available = _Service(ProfileOperationalState.READY)
    services = iter((locked, available))
    calls: list[None] = []

    def bootstrap():
        calls.append(None)
        return next(services)

    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.bootstrap.bootstrap_personal_context_service",
        bootstrap,
    )
    app = SimpleNamespace()

    first = TldwCli.get_personal_context_service(app)
    cached = TldwCli.get_personal_context_service(app)
    retried = TldwCli.get_personal_context_service(app, retry_locked=True)
    reused = TldwCli.get_personal_context_service(app)

    assert first is locked
    assert cached is locked
    assert retried is available
    assert reused is available
    assert calls == [None, None]
