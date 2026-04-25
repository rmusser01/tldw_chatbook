from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    AuthorizeURLResponse,
    ConnectorAccount,
    ConnectorBrowseResponse,
    ConnectorProvider,
    ConnectorSource,
    ConnectorSourceCreateRequest,
    ConnectorSourcePatchRequest,
    ConnectorSourceSyncStatus,
    ConnectorSourceSyncTriggerResponse,
    ConnectorImportJob,
    TLDWAPIClient,
)


def _account_payload() -> dict:
    return {
        "id": 7,
        "provider": "drive",
        "display_name": "Drive Account",
        "created_at": "2026-04-25T12:00:00Z",
        "connected": True,
        "email": "reader@example.com",
    }


def _source_payload(**overrides) -> dict:
    payload = {
        "id": 11,
        "account_id": 7,
        "provider": "drive",
        "remote_id": "root",
        "type": "folder",
        "path": "/Research",
        "options": {"recursive": True, "include_types": ["pdf"], "exclude_patterns": []},
        "enabled": True,
        "last_synced_at": None,
        "sync": {
            "state": "idle",
            "sync_mode": "manual",
            "last_sync_succeeded_at": None,
            "last_sync_failed_at": None,
            "last_error": None,
            "webhook_status": None,
            "needs_full_rescan": False,
            "active_job_id": None,
            "tracked_item_count": 0,
            "degraded_item_count": 0,
            "duplicate_count": 0,
            "metadata_only_count": 0,
        },
    }
    payload.update(overrides)
    return payload


def _job_payload(job_id: str = "job-1") -> dict:
    return {
        "id": job_id,
        "source_id": 11,
        "type": "import",
        "status": "queued",
        "progress_pct": 0,
        "counts": {"processed": 0, "skipped": 0, "failed": 0},
        "started_at": None,
        "finished_at": None,
        "error": None,
    }


@pytest.mark.asyncio
async def test_connectors_client_routes_providers_accounts_sources_and_jobs(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            [{"name": "drive", "auth_type": "oauth2", "scopes_required": ["drive.readonly"]}],
            {"auth_url": "https://accounts.example.test/oauth", "state": "state-1"},
            _account_payload(),
            [_account_payload()],
            {"ok": True},
            {"items": [{"id": "root", "name": "Root"}], "next_cursor": "cursor-2"},
            _source_payload(),
            [_source_payload()],
            _source_payload(enabled=False),
            _job_payload("import-1"),
            {
                "source_id": 11,
                "provider": "drive",
                "enabled": True,
                "state": "queued",
                "sync_mode": "manual",
                "cursor": None,
                "cursor_kind": None,
                "last_bootstrap_at": None,
                "last_sync_started_at": None,
                "last_sync_succeeded_at": None,
                "last_sync_failed_at": None,
                "last_error": None,
                "retry_backoff_count": 0,
                "webhook_status": None,
                "webhook_expires_at": None,
                "needs_full_rescan": False,
                "active_job_id": "sync-1",
                "active_job_started_at": None,
                "active_job": {
                    "id": "sync-1",
                    "type": "incremental_sync",
                    "status": "queued",
                    "progress_pct": 0,
                    "counts": {"processed": 0, "skipped": 0, "failed": 0},
                },
                "tracked_item_count": 3,
                "degraded_item_count": 0,
                "duplicate_count": 0,
                "metadata_only_count": 1,
            },
            {"source_id": 11, "provider": "drive", "status": "queued", "job": _job_payload("sync-1")},
            {"id": 99, "status": "queued"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    providers = await client.list_connector_providers()
    authorize = await client.authorize_connector_provider("drive", state="state-1", scopes=["drive.readonly", "email"])
    callback_account = await client.complete_connector_oauth_callback("drive", code="code-1", state="state-1")
    accounts = await client.list_connector_accounts()
    deleted = await client.delete_connector_account(7)
    browse = await client.browse_connector_sources(
        "drive",
        account_id=7,
        parent_remote_id="root",
        page_size=25,
        cursor="cursor-1",
    )
    created = await client.create_connector_source(
        ConnectorSourceCreateRequest(
            account_id=7,
            provider="drive",
            remote_id="root",
            type="folder",
            path="/Research",
            options={"recursive": True},
        )
    )
    listed = await client.list_connector_sources()
    patched = await client.update_connector_source(
        11,
        ConnectorSourcePatchRequest(enabled=False, options={"recursive": False}),
    )
    imported = await client.import_connector_source(11)
    sync_status = await client.get_connector_source_sync_status(11)
    sync_trigger = await client.trigger_connector_source_sync(11)
    job_status = await client.get_connector_job_status(99)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/connectors/providers")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/connectors/providers/drive/authorize")
    assert mocked.await_args_list[1].kwargs["params"] == {"state": "state-1", "scopes": "drive.readonly,email"}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/connectors/providers/drive/callback")
    assert mocked.await_args_list[2].kwargs["params"] == {"code": "code-1", "state": "state-1"}
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/connectors/accounts")
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/connectors/accounts/7")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/connectors/providers/drive/sources/browse")
    assert mocked.await_args_list[5].kwargs["params"] == {
        "account_id": 7,
        "parent_remote_id": "root",
        "page_size": 25,
        "cursor": "cursor-1",
    }
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/connectors/sources")
    assert mocked.await_args_list[6].kwargs["json_data"] == {
        "account_id": 7,
        "provider": "drive",
        "remote_id": "root",
        "type": "folder",
        "path": "/Research",
        "options": {"recursive": True},
    }
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/connectors/sources")
    assert mocked.await_args_list[8].args[:2] == ("PATCH", "/api/v1/connectors/sources/11")
    assert mocked.await_args_list[8].kwargs["json_data"] == {"enabled": False, "options": {"recursive": False}}
    assert mocked.await_args_list[9].args[:2] == ("POST", "/api/v1/connectors/sources/11/import")
    assert mocked.await_args_list[10].args[:2] == ("GET", "/api/v1/connectors/sources/11/sync")
    assert mocked.await_args_list[11].args[:2] == ("POST", "/api/v1/connectors/sources/11/sync")
    assert mocked.await_args_list[12].args[:2] == ("GET", "/api/v1/connectors/jobs/99")

    assert isinstance(providers[0], ConnectorProvider)
    assert isinstance(authorize, AuthorizeURLResponse)
    assert isinstance(callback_account, ConnectorAccount)
    assert isinstance(accounts[0], ConnectorAccount)
    assert deleted is True
    assert isinstance(browse, ConnectorBrowseResponse)
    assert browse.next_cursor == "cursor-2"
    assert isinstance(created, ConnectorSource)
    assert isinstance(listed[0], ConnectorSource)
    assert patched.enabled is False
    assert isinstance(imported, ConnectorImportJob)
    assert isinstance(sync_status, ConnectorSourceSyncStatus)
    assert sync_status.active_job_id == "sync-1"
    assert isinstance(sync_trigger, ConnectorSourceSyncTriggerResponse)
    assert job_status["id"] == 99
