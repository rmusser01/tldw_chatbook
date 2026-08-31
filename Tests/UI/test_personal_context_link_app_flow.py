"""Production-handler coverage for typed Personal Context bootstrap attention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Widgets.Settings_Widgets.personal_context_link_modal import (
    PersonalContextLinkModal,
    PersonalContextLinkReviewResult,
)
from tldw_chatbook.Personal_Context import link_key_custody
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
