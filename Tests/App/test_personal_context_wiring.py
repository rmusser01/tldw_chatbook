from types import SimpleNamespace

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Personal_Context.service import ProfileOperationalState


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


def test_app_retry_and_interview_never_recreate_present_v2_profile(
    tmp_path, monkeypatch
):
    import pytest

    from Tests.Personal_Context.native_barrier_helpers import (
        install_v2_manifest,
        sql_state,
    )
    from tldw_chatbook.Personal_Context.bootstrap import (
        bootstrap_personal_context_service,
    )
    from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    protector = InMemoryProfileKeyProtector()
    repo = PersonalContextRepository(tmp_path / "profile.db", key_protector=protector)
    repo.create_provisional_profile()
    install_v2_manifest(repo)
    before = sql_state(repo)
    calls = []

    def bootstrap():
        calls.append(None)
        return bootstrap_personal_context_service(
            db_path=repo.db_path, key_protector=protector
        )

    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.bootstrap.bootstrap_personal_context_service",
        bootstrap,
    )
    app = SimpleNamespace()
    app.get_personal_context_service = lambda **kwargs: (
        TldwCli.get_personal_context_service(app, **kwargs)
    )
    first = app.get_personal_context_service()
    assert first.status().state is ProfileOperationalState.LOCKED
    assert app.get_personal_context_service() is first
    retried = app.get_personal_context_service(retry_locked=True)
    assert retried.status().state is ProfileOperationalState.LOCKED
    assert retried.status().profile_present
    with pytest.raises(ValueError, match="Personal Context is unavailable"):
        TldwCli.prepare_personal_context_interview_request(app, kind="personal")
    assert calls == [None, None, None]
    assert sql_state(repo) == before


def test_restored_personal_context_sync_loader_refuses_before_any_link_read(
    monkeypatch,
) -> None:
    """TASK-40134 AC#9: an unreviewed restored generation keeps Personal Context
    sync off before the loader reads link state or a keyring key."""
    import pytest

    import tldw_chatbook.Personal_Context.recovery as recovery

    seen = []
    monkeypatch.setattr(
        recovery, "sync_allowed", lambda path=None: seen.append(path) or False
    )

    class Links:
        def get_personal_context_link_state(self, **_kwargs):
            raise AssertionError("restored link state was read")

    app = SimpleNamespace(sync_state_repository=Links())
    with pytest.raises(ValueError, match=recovery.SYNC_REVIEW_REQUIRED):
        TldwCli._load_personal_context_sync_runtime(
            app, server_profile_id="server", authenticated_principal_id=None
        )
    assert seen == [None]


def test_restored_link_resumes_nothing_before_its_review(monkeypatch) -> None:
    """Review r14pca-rev1: neither link resume path, nor the Settings link flow
    for a stored link, reads link state or keyring keys while a restored
    generation is unreviewed."""
    import asyncio

    import pytest

    import tldw_chatbook.Personal_Context.link_key_custody as custody
    import tldw_chatbook.Personal_Context.recovery as recovery
    from tldw_chatbook import app_destinations
    from tldw_chatbook.Personal_Context.link_service import PersonalContextLinkService

    monkeypatch.setattr(recovery, "sync_allowed", lambda path=None: False)

    class State:
        def get_personal_context_link_state(self, **_kwargs):
            raise AssertionError("restored link state was read")

    # Bare instance holding only what a state read would need.
    service = PersonalContextLinkService.__new__(PersonalContextLinkService)
    service._state = State()
    service._server_profile_id, service._principal_id = "server", None
    for call in (
        service.resume(),
        service.resume_after_local_activation(rebaseline_version=1),
    ):
        with pytest.raises(ValueError, match=recovery.SYNC_REVIEW_REQUIRED):
            asyncio.run(call)

    class Custodian:
        def load(self, **_kwargs):
            raise AssertionError("staged integrity key was read")

    monkeypatch.setattr(custody, "KeyringPersonalContextLinkKeyCustodian", Custodian)
    monkeypatch.setattr(custody, "KeyringPersonalContextWrappingKeyProvider", object)
    notices = []
    app = SimpleNamespace(
        _server_notification_event_scope=lambda: {"server_profile_id": "server"},
        sync_state_repository=SimpleNamespace(
            get_personal_context_link_state=lambda **_kwargs: {
                "state": "applying",
                "profile_id": "profile",
            }
        ),
        notify=lambda message, **_kwargs: notices.append(message),
    )
    asyncio.run(app_destinations._run_personal_context_link(app))
    assert notices and "needs attention" in notices[0]


def test_an_unreadable_restore_review_is_named_only_while_sync_is_off(
    monkeypatch, tmp_path
) -> None:
    """Review r14pca-rev1: when the review cannot be read, Settings names why
    sync is off; an ordinary profile (gate open) shows nothing."""
    import tldw_chatbook.Personal_Context.recovery as recovery

    def unreadable(*_args, **_kwargs):
        raise RuntimeError("projection_generation_unavailable")

    monkeypatch.setattr(recovery, "_restore_reviews", unreadable)
    path = tmp_path / "tldw_chatbook_personal_context.db"
    monkeypatch.setattr(recovery, "sync_allowed", lambda path=None: False)
    assert recovery.restore_review(path) == recovery.REVIEW_UNAVAILABLE
    monkeypatch.setattr(recovery, "sync_allowed", lambda path=None: True)
    assert recovery.restore_review(path) is None
