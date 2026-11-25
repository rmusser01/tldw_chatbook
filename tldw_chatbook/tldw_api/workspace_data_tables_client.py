"""Authenticated native Data Tables admission and access for one Workspace."""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import quote
from uuid import UUID

from pydantic import BaseModel, Field, StrictBool, StrictInt

from .data_tables_schemas import (
    DataTableDetailResponse,
    DataTableGenerateResponse,
    DataTableJobStatus,
)
from .exceptions import APIResponseError


class WorkspaceTableCapability(BaseModel):
    contract: Literal["research-workspace-data-tables.v1"]
    provider: str
    model: str
    available: StrictBool
    reason: str | None = None
    max_sources: StrictInt = Field(ge=1, le=100)


class WorkspaceTableAdmission(DataTableGenerateResponse):
    job_id: StrictInt = Field(ge=1)
    job_uuid: str
    publication_status: str | None = None
    published_version: StrictInt | None = Field(default=None, ge=1)


def canonical_table_uuid(value: str) -> str:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("The Server returned an invalid native table or job ID")
    return value


class WorkspaceDataTablesClient:
    def __init__(self, client, workspace_id: str, principal_id: str) -> None:
        self.client = client
        self.workspace_tag = f"workspace:{workspace_id}"
        self.path = (
            f"/api/v1/workspaces/{quote(workspace_id, safe='')}/research-data-tables"
        )
        subject = principal_id.removeprefix("jwt-sub:")
        self.subject = (
            subject
            if subject != principal_id and re.fullmatch(r"[1-9][0-9]{0,18}", subject)
            else None
        )
        self.headers = (
            {"X-TLDW-Expected-User-ID": self.subject} if self.subject else None
        )

    async def capability(
        self, provider: str, model: str
    ) -> WorkspaceTableCapability | None:
        if not provider or not model or model.casefold() == "auto":
            return None
        try:
            raw = await self.client._request(
                "GET",
                self.path,
                params={"provider": provider, "model": model},
                headers=self.headers,
            )
        except APIResponseError as exc:
            if exc.status_code in {403, 404, 405, 422, 501}:
                return None
            raise
        if (
            not isinstance(raw, dict)
            or raw.get("contract") != "research-workspace-data-tables.v1"
        ):
            return None
        reply = WorkspaceTableCapability.model_validate(raw)
        return reply if (reply.provider, reply.model) == (provider, model) else None

    async def launch(self, payload: dict) -> WorkspaceTableAdmission:
        raw = await self.client._request(
            "POST", self.path, json_data=payload, headers=self.headers
        )
        reply = WorkspaceTableAdmission.model_validate(raw)
        canonical_table_uuid(reply.job_uuid)
        canonical_table_uuid(reply.table.uuid)
        if reply.table.workspace_tag != self.workspace_tag:
            raise ValueError("The Server returned a table from a different workspace")
        return reply

    async def job(self, receipt: WorkspaceTableAdmission) -> DataTableJobStatus:
        raw = await self.client._request(
            "GET", f"/api/v1/data-tables/jobs/{receipt.job_id}", headers=self.headers
        )
        state = DataTableJobStatus.model_validate(raw)
        if (
            (state.id, state.uuid, state.table_uuid)
            != (receipt.job_id, receipt.job_uuid, receipt.table.uuid)
            or state.job_type != "data_table_generate"
            or (self.subject and state.owner_user_id != self.subject)
        ):
            raise ValueError("The Server returned a different Data Tables job")
        return state

    async def cancel(self, receipt: WorkspaceTableAdmission) -> None:
        await self.client._request(
            "DELETE", f"/api/v1/data-tables/jobs/{receipt.job_id}", headers=self.headers
        )

    async def detail(
        self, table_id: str, *, rows_limit: int = 1
    ) -> DataTableDetailResponse | None:
        try:
            raw = await self.client._request(
                "GET",
                f"/api/v1/data-tables/{canonical_table_uuid(table_id)}",
                params={
                    "rows_limit": rows_limit,
                    "include_sources": "false",
                    "include_rows": str(rows_limit > 1).lower(),
                },
                headers=self.headers,
            )
        except APIResponseError as exc:
            if exc.status_code == 404:
                return None
            raise
        result = DataTableDetailResponse.model_validate(raw)
        if (
            result.table.uuid != table_id
            or result.table.workspace_tag != self.workspace_tag
        ):
            raise ValueError("The Server returned a different workspace table")
        return result

    async def export(self, table_id: str, format: str) -> bytes:
        if format not in {"csv", "json", "xlsx"}:
            raise ValueError("Choose CSV, JSON, or XLSX for a Data Table")
        return await self.client._request_bytes(
            "GET",
            f"/api/v1/data-tables/{canonical_table_uuid(table_id)}/export",
            params={"format": format, "async_mode": "sync", "download": "true"},
            headers=self.headers,
        )
