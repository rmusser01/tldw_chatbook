"""Read and publish native Research Study outputs on a bound Server client."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .exceptions import APIResponseError

_STUDY_CONTRACT = "research-study.v1"
_MAX_OWNER_ID = 2**63 - 1
StudyKind = Literal["flashcards", "quiz"]


class StudyOutputReply(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    workspace_id: str
    owner_id: Annotated[int, Field(strict=True, ge=1, le=_MAX_OWNER_ID)]
    version: Annotated[int, Field(strict=True, ge=1)]
    name: str
    created_at: datetime
    generation: dict


class StudyOutputPageReply(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    contract: Literal["research-study.v1"]
    items: list[StudyOutputReply] = Field(max_length=100)
    limit: Annotated[int, Field(strict=True, ge=1, le=100)]
    offset: Annotated[int, Field(strict=True, ge=0)]
    total: Annotated[int, Field(strict=True, ge=0)]


class WorkspaceStudyClient:
    """Use the existing authenticated transport for one exact workspace."""

    def __init__(self, client, workspace_id: str, principal_id: str) -> None:
        self.client = client
        self.workspace_id = workspace_id
        self.path = f"/api/v1/workspaces/{quote(workspace_id, safe='')}/study-outputs"
        subject = principal_id.removeprefix("jwt-sub:")
        self.headers = (
            {"X-TLDW-Expected-User-ID": subject}
            if subject != principal_id and re.fullmatch(r"[1-9][0-9]{0,18}", subject)
            else None
        )

    def _check_workspace(self, row: StudyOutputReply) -> StudyOutputReply:
        if row.workspace_id != self.workspace_id:
            raise ValueError("The Server returned an output from a different workspace")
        return row

    async def page(self, *, limit: int, offset: int) -> StudyOutputPageReply | None:
        try:
            raw = await self.client._request(
                "GET",
                self.path,
                params={"limit": limit, "offset": offset},
                headers=self.headers,
            )
        except APIResponseError as exc:
            if exc.status_code in {404, 405}:
                return None
            raise
        if isinstance(raw, dict) and raw.get("contract") != _STUDY_CONTRACT:
            return None
        try:
            page = StudyOutputPageReply.model_validate(raw)
        except ValidationError:
            raise ValueError(
                "The Server returned an invalid Study output page"
            ) from None
        if (page.limit, page.offset) != (limit, offset) or len(page.items) != min(
            limit, max(0, page.total - offset)
        ):
            raise ValueError("The Server returned a different Study output page")
        for item in page.items:
            self._check_workspace(item)
        return page

    async def save(self, payload: dict) -> StudyOutputReply:
        try:
            raw = await self.client._request(
                "POST",
                self.path,
                json_data=payload,
                headers=self.headers,
            )
        except APIResponseError as exc:
            if exc.status_code in {404, 405}:
                raise ValueError(
                    "This Server does not support Studio generation provenance for Study"
                ) from None
            raise
        return self._record(raw)

    async def get(self, kind: StudyKind, owner_id: str) -> StudyOutputReply | None:
        if kind not in {"flashcards", "quiz"} or not re.fullmatch(
            r"[1-9][0-9]{0,18}", owner_id
        ):
            raise ValueError("Choose an exact Server Study record")
        try:
            raw = await self.client._request(
                "GET",
                f"{self.path}/{kind}/{owner_id}",
                headers=self.headers,
            )
        except APIResponseError as exc:
            if exc.status_code == 404:
                return None
            raise
        row = self._record(raw)
        if str(row.owner_id) != owner_id or row.generation.get("output_kind") != kind:
            raise ValueError("The Server returned a different native Study record")
        return row

    def _record(self, raw: object) -> StudyOutputReply:
        try:
            row = StudyOutputReply.model_validate(raw)
        except ValidationError:
            raise ValueError(
                "The Server returned an invalid Study output receipt"
            ) from None
        return self._check_workspace(row)
