"""Production-handler coverage for typed Personal Context bootstrap attention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Widgets.Settings_Widgets.personal_context_link_modal import (
    PersonalContextLinkModal,
    PersonalContextLinkReviewResult,
)
from tldw_chatbook.Personal_Context import link_key_custody
from tldw_chatbook.Personal_Context import link_service as link_service_module
from tldw_chatbook.Personal_Context.key_protector import ProfileLockedError
from tldw_chatbook.app import TldwCli
from tldw_chatbook.tldw_api.exceptions import (
    APIResponseError,
    PersonalContextBootstrapAttentionError,
)
from tldw_chatbook.tldw_api.sync_schemas import (
    SyncPersonalContextPurgeAttention,
    SyncPersonalContextQuotaAttention,
    SyncPersonalContextSchemaAttention,
)


ATTENTIONS = (
    SyncPersonalContextSchemaAttention(
        kind="schema_incompatible",
        required_schema_version=3,
        server_min_schema_version=1,
        server_max_schema_version=2,
    ),
    SyncPersonalContextQuotaAttention(
        kind="quota_incompatible",
        required_quotas={"max_record_bytes": 16_384},
        available_quotas={"max_record_bytes": 8_192},
        insufficient_quotas=["max_record_bytes"],
    ),
    SyncPersonalContextPurgeAttention(
        kind="purge_generation_mismatch",
        expected_purge_generation=4,
        current_purge_generation=5,
    ),
)


class _StateRepository:
    def get_personal_context_link_state(self, **_kwargs):
        return None


class _AttentionServer:
    def __init__(self, attention) -> None:
        self.attention = attention
        self.calls = 0

    async def bootstrap_personal_context_link(self, **_kwargs):
        self.calls += 1
        raise PersonalContextBootstrapAttentionError(self.attention)


class _MalformedAttentionServer:
    async def bootstrap_personal_context_link(self, **_kwargs):
        raise APIResponseError(
            409,
            "private server response",
            response_data={"detail": {"attention": {"kind": "unknown"}}},
        )


class _LinkAppHarness:
    def __init__(self, attention, modal_results) -> None:
        self.sync_state_repository = _StateRepository()
        self.server_sync_service = _AttentionServer(attention)
        self.local_first_sync_service = SimpleNamespace()
        self._profile = SimpleNamespace(
            get_manifest=lambda: SimpleNamespace(purge_generation=4)
        )
        self._modal_results = list(modal_results)
        self.modals: list[PersonalContextLinkModal] = []
        self.notifications: list[tuple[str, str | None]] = []
        self.reloads = 0

    def _server_notification_event_scope(self):
        return {
            "server_profile_id": "server-config-1",
            "authenticated_principal_id": "user-1",
        }

    def get_personal_context_service(self, *, retry_locked=False):
        return self._profile

    async def push_screen_wait(self, modal):
        self.modals.append(modal)
        return self._modal_results.pop(0)

    def notify(self, message, *, severity=None):
        self.notifications.append((message, severity))

    def _reload_personal_context_settings_panel(self):
        self.reloads += 1


@pytest.fixture
def secure_provider_stubs(monkeypatch):
    monkeypatch.setattr(
        link_key_custody,
        "KeyringPersonalContextWrappingKeyProvider",
        lambda: SimpleNamespace(public_key_pem="test-public-key"),
    )
    monkeypatch.setattr(
        link_key_custody,
        "KeyringPersonalContextLinkKeyCustodian",
        lambda: SimpleNamespace(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("attention", ATTENTIONS)
async def test_app_routes_each_typed_bootstrap_attention_to_canonical_modal(
    attention,
    secure_provider_stubs,
) -> None:
    app = _LinkAppHarness(attention, [None])

    await TldwCli._run_personal_context_link(app)

    assert len(app.modals) == 1, app.notifications
    assert app.modals[0]._bootstrap_attention is attention
    assert app.notifications == []
    assert app.reloads == 0


@pytest.mark.asyncio
async def test_app_retries_typed_bootstrap_attention_in_the_owning_worker(
    secure_provider_stubs,
) -> None:
    attention = ATTENTIONS[0]
    app = _LinkAppHarness(
        attention,
        [
            PersonalContextLinkReviewResult(
                plan_id=None,
                decisions={},
                unlinked_remote_scope_ids=(),
                retry=True,
            ),
            None,
        ],
    )

    await TldwCli._run_personal_context_link(app)

    assert app.server_sync_service.calls == 2, app.notifications
    assert len(app.modals) == 2
    assert app.notifications == []


@pytest.mark.asyncio
async def test_app_keeps_malformed_bootstrap_attention_out_of_the_review_surface(
    secure_provider_stubs,
) -> None:
    app = _LinkAppHarness(ATTENTIONS[0], [])
    app.server_sync_service = _MalformedAttentionServer()

    await TldwCli._run_personal_context_link(app)

    assert app.modals == []
    assert app.notifications == [
        (
            "Profile linking needs attention. No profile content was shown; "
            "retry from Settings.",
            "error",
        )
    ]
    assert "private server response" not in repr(app.notifications)


@pytest.mark.asyncio
async def test_app_abandons_restarted_apply_when_staged_key_is_unavailable(
    monkeypatch,
) -> None:
    existing = {
        "state": "applying",
        "server_profile_id": "server-config-1",
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "profile_id": "profile-server",
        "integrity_key_id": "integrity-1",
        "key_record_id": "record-1",
    }

    class ApplyingStateRepository:
        def get_personal_context_link_state(self, **_kwargs):
            return existing

    class MissingStagedKeyCustodian:
        def load(self, **_kwargs):
            raise ProfileLockedError("staged key unavailable")

    abandon_calls: list[bool] = []

    class RecoveryCoordinator:
        _key_binding = staticmethod(link_service_module.PersonalContextLinkService._key_binding)

        def __init__(self, **_kwargs):
            pass

        def abandon_uncommitted_apply(self):
            abandon_calls.append(True)
            return True

        async def plan(self, **_kwargs):
            raise RuntimeError("stop after recovery boundary")

    monkeypatch.setattr(
        link_key_custody,
        "KeyringPersonalContextWrappingKeyProvider",
        lambda: SimpleNamespace(public_key_pem="test-public-key"),
    )
    monkeypatch.setattr(
        link_key_custody,
        "KeyringPersonalContextLinkKeyCustodian",
        MissingStagedKeyCustodian,
    )
    monkeypatch.setattr(
        link_service_module,
        "PersonalContextLinkService",
        RecoveryCoordinator,
    )
    app = _LinkAppHarness(ATTENTIONS[0], [])
    app.sync_state_repository = ApplyingStateRepository()

    await TldwCli._run_personal_context_link(app)

    assert abandon_calls == [True]
    assert app.notifications == [
        (
            "Profile linking needs attention. No profile content was shown; "
            "retry from Settings.",
            "error",
        )
    ]


@pytest.mark.asyncio
async def test_app_does_not_replan_when_interrupted_apply_is_not_proven_uncommitted(
    monkeypatch,
) -> None:
    existing = {
        "state": "applying",
        "server_profile_id": "server-config-1",
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "profile_id": "profile-server",
        "integrity_key_id": "integrity-1",
        "key_record_id": "record-1",
    }

    class ApplyingStateRepository:
        def get_personal_context_link_state(self, **_kwargs):
            return existing

    class MissingStagedKeyCustodian:
        def load(self, **_kwargs):
            raise ProfileLockedError("transient staged-key load failure")

    plan_calls: list[bool] = []

    class RecoveryCoordinator:
        _key_binding = staticmethod(link_service_module.PersonalContextLinkService._key_binding)

        def __init__(self, **_kwargs):
            pass

        def abandon_uncommitted_apply(self):
            return False

        async def plan(self, **_kwargs):
            plan_calls.append(True)
            raise AssertionError("must not start a fresh plan")

    monkeypatch.setattr(
        link_key_custody,
        "KeyringPersonalContextWrappingKeyProvider",
        lambda: SimpleNamespace(public_key_pem="test-public-key"),
    )
    monkeypatch.setattr(
        link_key_custody,
        "KeyringPersonalContextLinkKeyCustodian",
        MissingStagedKeyCustodian,
    )
    monkeypatch.setattr(
        link_service_module,
        "PersonalContextLinkService",
        RecoveryCoordinator,
    )
    app = _LinkAppHarness(ATTENTIONS[0], [])
    app.sync_state_repository = ApplyingStateRepository()

    await TldwCli._run_personal_context_link(app)

    assert plan_calls == []
    assert app.notifications == [
        (
            "Profile linking needs attention. No profile content was shown; "
            "retry from Settings.",
            "error",
        )
    ]


def test_lazy_runtime_retries_complete_cleanup_before_enabling_sync(monkeypatch) -> None:
    link = {
        "state": "complete",
        "server_profile_id": "server-config-1",
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "profile_id": "profile-server",
        "integrity_key_id": "integrity-1",
        "key_record_id": "record-1",
        "plan_id": "plan-1",
        "rebaseline_version": 2,
    }

    class CompleteStateRepository:
        def get_personal_context_link_state(self, **_kwargs):
            return link

    events: list[str] = []

    class RetryCleanupCustodian:
        failures = 1

        def load_storage_key(self, **_kwargs):
            events.append("load-storage")
            return b"k" * 32

        def delete(self, **_kwargs):
            events.append("delete-staged")
            if self.failures:
                type(self).failures -= 1
                raise ProfileLockedError("transient cleanup failure")

    class ReadyProfile:
        def release_first_link_freeze(self, *, plan_id):
            assert plan_id == "plan-1"
            events.append("release-freeze")

        def clear_first_link_rebaseline_commit(self, **kwargs):
            assert kwargs == {
                "plan_id": "plan-1",
                "target_profile_id": "profile-server",
                "rebaseline_version": 2,
            }
            events.append("clear-marker")

        def build_personal_context_outbox_dispatcher(self, **_kwargs):
            events.append("build-dispatcher")
            return "dispatcher"

    monkeypatch.setattr(
        link_key_custody,
        "KeyringPersonalContextLinkKeyCustodian",
        RetryCleanupCustodian,
    )
    app = SimpleNamespace(
        sync_state_repository=CompleteStateRepository(),
        sync_v2_dataset_keys={},
        local_first_sync_service=SimpleNamespace(
            personal_context_outbox_dispatcher=None,
            personal_context_service=None,
        ),
        get_personal_context_service=lambda **_kwargs: ReadyProfile(),
    )

    with pytest.raises(ProfileLockedError, match="transient cleanup failure"):
        TldwCli._load_personal_context_sync_runtime(
            app,
            server_profile_id="server-config-1",
            authenticated_principal_id="user-1",
        )
    assert app.sync_v2_dataset_keys == {}
    assert app.local_first_sync_service.personal_context_outbox_dispatcher is None
    assert "clear-marker" not in events

    TldwCli._load_personal_context_sync_runtime(
        app,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
    )

    assert events[-4:] == [
        "delete-staged",
        "release-freeze",
        "clear-marker",
        "build-dispatcher",
    ]
    assert app.sync_v2_dataset_keys == {"dataset-1": b"k" * 32}
    assert app.local_first_sync_service.personal_context_outbox_dispatcher == (
        "dispatcher"
    )
