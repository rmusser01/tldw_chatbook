"""Production-handler coverage for typed Personal Context bootstrap attention."""

from __future__ import annotations

import sqlite3
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
    attention_calls: list[bool] = []

    class RecoveryCoordinator:
        _key_binding = staticmethod(link_service_module.PersonalContextLinkService._key_binding)

        def __init__(self, **_kwargs):
            pass

        def abandon_uncommitted_apply(self):
            return False

        def mark_ambiguous_apply_attention(self):
            attention_calls.append(True)
            return True

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
    assert attention_calls == [True]
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
        "purge_generation": 0,
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
                "target_integrity_key_id": "integrity-1",
                "target_key_record_id": "record-1",
                "target_purge_generation": 0,
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
    assert events[-3:] == ["clear-marker", "release-freeze", "delete-staged"]

    TldwCli._load_personal_context_sync_runtime(
        app,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
    )

    assert events[-4:] == [
        "clear-marker",
        "release-freeze",
        "delete-staged",
        "build-dispatcher",
    ]
    assert app.sync_v2_dataset_keys == {"dataset-1": b"k" * 32}
    assert app.local_first_sync_service.personal_context_outbox_dispatcher == (
        "dispatcher"
    )


@pytest.mark.parametrize(
    ("receipt_dataset_id", "receipt_key_record_id", "succeeds"),
    (
        ("dataset-1", "record-1", True),
        ("dataset-1", "foreign-record", False),
        ("foreign-dataset", "record-1", False),
    ),
)
def test_lazy_runtime_repairs_exact_v7_complete_marker_before_enabling_sync(
    monkeypatch, tmp_path, receipt_dataset_id, receipt_key_record_id, succeeds
) -> None:
    from tldw_profile_core import ProfileManifest, ProfileScope, ScopeKind

    from tldw_chatbook.Personal_Context.key_protector import (
        InMemoryProfileKeyProtector,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
    from tldw_chatbook.Personal_Context.service import PersonalContextService
    from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

    now = "2026-08-31T12:00:00.000Z"
    protector = InMemoryProfileKeyProtector()
    profile_db = tmp_path / "profile-v7-complete.db"
    repository = PersonalContextRepository(profile_db, key_protector=protector)
    repository.create_profile_with_global_scope(
        ProfileManifest(
            profile_id="profile-server",
            revision=1,
            purge_generation=0,
            created_at=now,
            updated_at=now,
            current_version_id="manifest-v1",
        ),
        ProfileScope(
            profile_id="profile-server",
            scope_id="scope-global",
            kind=ScopeKind.GLOBAL,
            version_id="scope-v1",
            created_at=now,
            updated_at=now,
        ),
    )
    with sqlite3.connect(profile_db) as connection:
        connection.execute(
            "INSERT INTO first_link_freeze VALUES (1, ?, ?, ?, ?, ?, ?)",
            (
                "plan-v7",
                "snapshot-v7",
                "profile-server",
                0,
                1,
                now,
            ),
        )
        connection.execute("DROP TABLE first_link_rebaseline_commit")
        connection.execute(
            """
            CREATE TABLE first_link_rebaseline_commit (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                plan_id TEXT NOT NULL,
                target_profile_id TEXT NOT NULL,
                target_integrity_key_id TEXT NOT NULL,
                target_purge_generation INTEGER NOT NULL,
                rebaseline_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO first_link_rebaseline_commit VALUES (1, ?, ?, ?, ?, ?, ?)",
            ("plan-v7", "profile-server", "integrity-1", 0, 1, now),
        )
        connection.execute(
            "UPDATE personal_context_schema SET version = 7 WHERE singleton = 1"
        )
    profile = PersonalContextService(
        PersonalContextRepository(profile_db, key_protector=protector)
    )
    state = SyncStateRepository(tmp_path / "sync.db")
    state.set_personal_context_link_state(
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
        state="complete",
        device_id="device-1",
        dataset_id=receipt_dataset_id,
        authority_id="authority-1",
        profile_id="profile-server",
        integrity_key_id="integrity-1",
        key_record_id=receipt_key_record_id,
        purge_generation=0,
        bootstrap_cursor="bootstrap-receipt",
        sync_transport_cursor="transport-bootstrap",
        confirmed_cursor="transport-confirmed",
        bootstrap_heads={},
        expected_heads={},
        plan_id="plan-v7",
        rebaseline_version=1,
        attention_code=None,
    )

    class FakeMacOSKeyring:
        priority = 1

        def __init__(self):
            self.values = {}

        def get_password(self, service, name):
            return self.values.get((service, name))

        def set_password(self, service, name, value):
            self.values[(service, name)] = value

        def delete_password(self, service, name):
            del self.values[(service, name)]

    FakeMacOSKeyring.__module__ = "keyring.backends.macOS"
    custodian = link_key_custody.KeyringPersonalContextLinkKeyCustodian(
        FakeMacOSKeyring()
    )
    staged_binding = {
        "server_profile_id": "server-config-1",
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "profile_id": "profile-server",
        "integrity_key_id": "integrity-1",
        "key_record_id": "record-1",
    }
    custodian.stage(
        **staged_binding,
        integrity_key=profile._repo()._require_keys().integrity_key,
    )
    receipt_binding = {
        **staged_binding,
        "dataset_id": receipt_dataset_id,
        "key_record_id": receipt_key_record_id,
    }
    storage_key = custodian.load_or_create_storage_key(**receipt_binding)

    monkeypatch.setattr(
        link_key_custody,
        "KeyringPersonalContextLinkKeyCustodian",
        lambda: custodian,
    )
    app = SimpleNamespace(
        sync_state_repository=state,
        sync_v2_dataset_keys={},
        local_first_sync_service=SimpleNamespace(
            personal_context_outbox_dispatcher=None,
            personal_context_service=None,
        ),
        get_personal_context_service=lambda **_kwargs: profile,
    )

    if not succeeds:
        with pytest.raises(ValueError, match="staged_integrity_key_binding_mismatch"):
            TldwCli._load_personal_context_sync_runtime(
                app,
                server_profile_id="server-config-1",
                authenticated_principal_id="user-1",
            )
        assert profile.first_link_freeze_plan_id() == "plan-v7"
        assert profile.first_link_rebaseline_commit_plan_id() == "plan-v7"
        assert custodian.load(**staged_binding) == (
            profile._repo()._require_keys().integrity_key
        )
        assert app.sync_v2_dataset_keys == {}
        return

    TldwCli._load_personal_context_sync_runtime(
        app,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
    )

    assert profile.first_link_freeze_plan_id() is None
    assert profile.first_link_rebaseline_commit_plan_id() is None
    assert app.sync_v2_dataset_keys == {"dataset-1": storage_key}
    with pytest.raises(ValueError, match="staged_integrity_key_binding_mismatch"):
        custodian.load(**staged_binding)
    TldwCli._load_personal_context_sync_runtime(
        app,
        server_profile_id="server-config-1",
        authenticated_principal_id="user-1",
    )
    assert profile.first_link_rebaseline_commit_plan_id() is None


def test_lazy_runtime_rejects_different_freeze_owner_before_custody_cleanup(
    monkeypatch,
) -> None:
    link = {
        "state": "complete",
        "server_profile_id": "server-config-1",
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "profile_id": "profile-server",
        "integrity_key_id": "integrity-1",
        "key_record_id": "record-1",
        "purge_generation": 0,
        "plan_id": "plan-1",
        "rebaseline_version": 2,
    }
    events: list[str] = []

    class CompleteStateRepository:
        def get_personal_context_link_state(self, **_kwargs):
            return link

    class Custodian:
        def load_storage_key(self, **_kwargs):
            return b"k" * 32

        def delete(self, **_kwargs):
            events.append("delete-staged")

    class WrongOwnerProfile:
        def first_link_freeze_plan_id(self):
            return "different-plan"

        def first_link_rebaseline_commit_plan_id(self):
            return None

        def build_personal_context_outbox_dispatcher(self, **_kwargs):
            events.append("build-dispatcher")
            return "dispatcher"

    monkeypatch.setattr(
        link_key_custody,
        "KeyringPersonalContextLinkKeyCustodian",
        Custodian,
    )
    app = SimpleNamespace(
        sync_state_repository=CompleteStateRepository(),
        sync_v2_dataset_keys={},
        local_first_sync_service=SimpleNamespace(
            personal_context_outbox_dispatcher=None,
            personal_context_service=None,
        ),
        get_personal_context_service=lambda **_kwargs: WrongOwnerProfile(),
    )

    with pytest.raises(ValueError, match="personal_context_link_cleanup_mismatch"):
        TldwCli._load_personal_context_sync_runtime(
            app,
            server_profile_id="server-config-1",
            authenticated_principal_id="user-1",
        )

    assert events == []
    assert app.sync_v2_dataset_keys == {}
