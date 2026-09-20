"""Isolated recovery publication and the existing store-to-catalog boundary."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.UI.MCP_Modules import mcp_workbench


@pytest.fixture
def publication(monkeypatch):
    """Call the real completion method without mounting a Textual application."""
    monkeypatch.setattr(
        mcp_workbench, "get_cli_setting", lambda section, key, default=None: default
    )
    service = SimpleNamespace(
        context=SimpleNamespace(selected_scope="personal", selected_scope_ref=None),
        approve_recovery_review=Mock(),
        local_external_catalog=AsyncMock(return_value=[]),
    )
    inspector = SimpleNamespace(show_tool=AsyncMock(), show_finding=AsyncMock())
    bench = SimpleNamespace(
        app=SimpleNamespace(notify=Mock()),
        query_one=Mock(return_value=inspector),
        _mcp_recovery_busy=True,
        _mcp_recovery_view=lambda: ("local", "permissions"),
        _rebind_inspector_advanced_context=Mock(),
        _sync_children=AsyncMock(),
    )
    token = (object(), service, object(), bench._mcp_recovery_view())
    bench._mcp_recovery_token = token
    bench._mcp_recovery_current = lambda candidate, **kwargs: (
        candidate is bench._mcp_recovery_token
    )
    return bench, service, inspector, token


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [[], ["alpha", "beta"]])
async def test_current_catalog_is_converted_before_publication(publication, names):
    bench, service, inspector, token = publication
    records = [
        {"profile_id": name, "command": "fixture-command", "discovery_snapshot": None}
        for name in names
    ]
    service.local_external_catalog.return_value = records
    review = object()

    await mcp_workbench.MCPWorkbench._record_mcp_recovery_review(bench, token, review)

    service.approve_recovery_review.assert_called_once_with(review)
    service.local_external_catalog.assert_awaited_once_with()
    assert list(bench._catalog_records) == names
    assert len(bench._snapshots) == len(names) + 1
    assert bench._snapshots[0].source == "builtin"
    assert [item.server_key for item in bench._snapshots[1:]] == [
        "local:" + name for name in names
    ]
    for record in records:
        assert bench._catalog_records[record["profile_id"]] == record
        assert bench._catalog_records[record["profile_id"]] is not record
    inspector.show_tool.assert_awaited_once_with(None)
    inspector.show_finding.assert_awaited_once_with(None)
    bench._sync_children.assert_awaited_once()
    assert bench.app.notify.call_args.kwargs == {}
    assert "Fresh MCP roots reviewed" in bench.app.notify.call_args.args[0]
    assert not bench._mcp_recovery_busy and bench._mcp_recovery_token is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "records",
    [
        None,
        {},
        [None],
        [{"command": "missing-id"}],
        [{"profile_id": "bad", "env_placeholders": 7}],
    ],
)
async def test_invalid_service_result_preserves_approval_and_offers_retry(
    publication, records
):
    bench, service, _, token = publication
    service.local_external_catalog.return_value = records

    await mcp_workbench.MCPWorkbench._record_mcp_recovery_review(bench, token, object())

    service.approve_recovery_review.assert_called_once()
    assert bench._catalog_records == {}
    assert len(bench._snapshots) == 1 and bench._snapshots[0].source == "builtin"
    bench._sync_children.assert_awaited_once()
    notice = bench.app.notify.call_args
    assert "Fresh MCP roots reviewed" in notice.args[0]
    assert "Press r to retry" in notice.args[0]
    assert notice.kwargs == {"severity": "warning"}
    assert not bench._mcp_recovery_busy and bench._mcp_recovery_token is None


@pytest.mark.asyncio
async def test_catalog_reader_failure_keeps_successful_approval_private(publication):
    bench, service, _, token = publication
    service.local_external_catalog.side_effect = OSError("/private/secret/catalog.json")

    await mcp_workbench.MCPWorkbench._record_mcp_recovery_review(bench, token, object())

    service.approve_recovery_review.assert_called_once()
    assert bench._catalog_records == {}
    notice = bench.app.notify.call_args.args[0]
    assert "Fresh MCP roots reviewed" in notice and "Press r to retry" in notice
    assert "secret" not in notice and "catalog.json" not in notice
    assert not bench._mcp_recovery_busy and bench._mcp_recovery_token is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reader_fails", [False, True])
async def test_late_catalog_does_not_publish_or_clear_successor_token(
    publication, reader_fails
):
    bench, service, inspector, token = publication
    successor = object()
    later_catalog, later_snapshots = {"later": object()}, [object()]

    async def read_after_navigation():
        bench._mcp_recovery_token = successor
        bench._catalog_records, bench._snapshots = later_catalog, later_snapshots
        if reader_fails:
            raise OSError("catalog read failed after navigation")
        return [{"profile_id": "old", "command": "fixture-command"}]

    service.local_external_catalog.side_effect = read_after_navigation
    await mcp_workbench.MCPWorkbench._record_mcp_recovery_review(bench, token, object())

    service.approve_recovery_review.assert_called_once()
    assert bench._catalog_records is later_catalog
    assert bench._snapshots is later_snapshots
    assert bench._mcp_recovery_token is successor
    assert not bench._mcp_recovery_busy
    bench._rebind_inspector_advanced_context.assert_not_called()
    inspector.show_tool.assert_not_awaited()
    inspector.show_finding.assert_not_awaited()
    bench._sync_children.assert_not_awaited()
    bench.app.notify.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", ["bad", ["bad"], 7, None])
async def test_store_normalizes_non_mapping_environment_before_passive_catalog(
    tmp_path, malformed
):
    """The reported malformed field never reaches readiness as raw input."""
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.MCP.readiness import local_profile_readiness
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    path = tmp_path / "local-store.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "profile_id": "malformed-env",
                        "command": "fixture-command",
                        "env_placeholders": malformed,
                    },
                    {
                        "profile_id": "valid-sibling",
                        "command": "fixture-command",
                        "env_placeholders": {"TOKEN": "$TOKEN"},
                    },
                ]
            }
        )
    )
    forbidden = Mock(
        side_effect=AssertionError("passive read attempted active operation")
    )
    client = SimpleNamespace(
        sessions={},
        connect_to_server=forbidden,
        list_tools=forbidden,
        execute_tool=forbidden,
    )
    local = LocalMCPControlService(
        store=LocalMCPStore(path), client=client, manifest_provider=dict
    )
    service = UnifiedMCPControlPlaneService(
        local_service=local, server_service=None, target_store=None, context_store=None
    )

    records = await service.local_external_catalog()

    assert [record["profile_id"] for record in records] == [
        "malformed-env",
        "valid-sibling",
    ]
    assert records[0]["env_placeholders"] == {}
    assert records[1]["env_placeholders"] == {"TOKEN": "$TOKEN"}
    snapshots = [local_profile_readiness(record, environ={}) for record in records]
    assert len(snapshots) == 2 and all(not item.is_connected for item in snapshots)
    forbidden.assert_not_called()
