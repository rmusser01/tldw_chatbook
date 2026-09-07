from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.MCP.execution_log import ExecutionRecord, MCPExecutionLog
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.permission_store import definition_hash
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)


class _LocalService:
    def __init__(self, store: LocalMCPStore, records: list[dict]) -> None:
        self.store = store
        self._records = records

    def get_external_servers(self) -> list[dict]:
        return list(self._records)

    def get_inventory(self) -> dict:
        return {"tools": []}


class _ChangingCatalogLocalService(_LocalService):
    def __init__(
        self,
        store: LocalMCPStore,
        first_records: list[dict],
        later_records: list[dict],
    ) -> None:
        super().__init__(store, first_records)
        self._later_records = later_records
        self.catalog_calls = 0

    def get_external_servers(self) -> list[dict]:
        self.catalog_calls += 1
        records = self._records if self.catalog_calls == 1 else self._later_records
        return list(records)


def _profile_record(
    *,
    profile_id: str = "docs",
    tool_name: str = "search",
    description: str = "Search docs",
    is_connected: bool = True,
) -> dict:
    return {
        "profile_id": profile_id,
        "name": profile_id,
        "is_connected": is_connected,
        "discovery_snapshot": {
            "tools": [
                {
                    "name": tool_name,
                    "description": description,
                    "inputSchema": {"type": "object"},
                }
            ]
        },
    }


def _service(
    tmp_path: Path, records: list[dict]
) -> tuple[UnifiedMCPControlPlaneService, LocalMCPStore]:
    store = LocalMCPStore(tmp_path / "store.json")
    service = UnifiedMCPControlPlaneService(
        local_service=_LocalService(store, records),
        server_service=None,
        target_store=None,
        context_store=None,
    )
    return service, store


def _append_approved(store: LocalMCPStore, *, ts: str) -> None:
    log = MCPExecutionLog(Path(store.path).with_name("mcp_execution_log.jsonl"))
    log.append(
        ExecutionRecord(
            ts=ts,
            server_key="local:docs",
            tool_name="search",
            initiator="agent",
            decision="approved",
            ok=True,
            status="success",
            duration_ms=12,
            error_category=None,
            exception_type=None,
            status_code=None,
            argument_names=(),
            unknown_argument_count=0,
            result_type="dict",
            result_size=1,
        )
    )


@pytest.mark.asyncio
async def test_permission_prompt_recommendations_uses_log_catalog_and_effective_state(
    tmp_path,
):
    """Catches reports that ignore either approval history or current permission state."""
    service, store = _service(tmp_path, [_profile_record()])
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")

    report = await service.permission_prompt_recommendations()

    assert [(r.server_key, r.tool_name, r.approved_count) for r in report.recommendations] == [
        ("local:docs", "search", 2)
    ]
    assert report.recommendations[0].last_seen == "2026-08-01T20:05:00+00:00"


@pytest.mark.asyncio
async def test_apply_permission_prompt_recommendation_persists_hash_safe_allow(
    tmp_path,
):
    """Catches applying a recommendation without using the existing tool-state API."""
    service, store = _service(tmp_path, [_profile_record(description="Search docs")])
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")

    recommendation = await service.apply_permission_prompt_recommendation(
        "local:docs", "search"
    )

    assert recommendation.server_key == "local:docs"
    assert recommendation.tool_name == "search"
    tool_entry = service.permission_store.load()["profiles"]["default"]["servers"][
        "local:docs"
    ]["tools"]["search"]
    assert tool_entry["state"] == "allow"
    assert tool_entry["definition_hash"] == definition_hash(
        "Search docs", {"type": "object"}
    )


@pytest.mark.asyncio
async def test_apply_permission_prompt_recommendation_uses_one_catalog_snapshot(
    tmp_path,
):
    """Prevents a changed second catalog read from inheriting older approvals."""
    store = LocalMCPStore(tmp_path / "store.json")
    local_service = _ChangingCatalogLocalService(
        store,
        [_profile_record(description="Search docs")],
        [_profile_record(description="Delete matching docs")],
    )
    service = UnifiedMCPControlPlaneService(
        local_service=local_service,
        server_service=None,
        target_store=None,
        context_store=None,
    )
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")

    await service.apply_permission_prompt_recommendation("local:docs", "search")

    tool_entry = service.permission_store.load()["profiles"]["default"]["servers"][
        "local:docs"
    ]["tools"]["search"]
    assert local_service.catalog_calls == 1
    assert tool_entry["definition_hash"] == definition_hash(
        "Search docs", {"type": "object"}
    )


@pytest.mark.asyncio
async def test_apply_permission_prompt_recommendation_rejects_non_recommended_tool(
    tmp_path,
):
    """Catches persisting allows for tools that do not meet recommendation criteria."""
    service, _store = _service(tmp_path, [_profile_record()])

    with pytest.raises(KeyError, match="No prompt-reduction recommendation"):
        await service.apply_permission_prompt_recommendation("local:docs", "search")


@pytest.mark.asyncio
async def test_apply_permission_prompt_recommendation_requires_permission_store(
    tmp_path,
):
    """Catches reporting success when the recommended allow cannot persist."""
    service, store = _service(tmp_path, [_profile_record()])
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")
    service._execution_log = MCPExecutionLog(
        Path(store.path).with_name("mcp_execution_log.jsonl")
    )
    service.local_service.store = None

    with pytest.raises(RuntimeError, match="MCP permission store unavailable"):
        await service.apply_permission_prompt_recommendation(
            "local:docs", "search"
        )
