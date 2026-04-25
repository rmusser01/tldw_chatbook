from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


KanbanPriority = Literal["low", "medium", "high", "urgent"]


class KanbanDetailResponse(BaseModel):
    detail: str


class KanbanPaginationInfo(BaseModel):
    total: int
    limit: int
    offset: int
    has_more: bool


class KanbanBoardCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=5000)
    client_id: str = Field(..., min_length=1, max_length=100)
    activity_retention_days: int | None = Field(None, ge=7, le=365)
    metadata: dict[str, Any] | None = None


class KanbanBoardUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=5000)
    activity_retention_days: int | None = Field(None, ge=7, le=365)
    metadata: dict[str, Any] | None = None


class KanbanBoardResponse(BaseModel):
    id: int
    uuid: str
    user_id: str
    client_id: str
    name: str
    description: str | None = None
    archived: bool
    archived_at: datetime | None = None
    activity_retention_days: int | None = None
    created_at: datetime
    updated_at: datetime
    deleted: bool
    deleted_at: datetime | None = None
    version: int
    metadata: dict[str, Any] | None = None
    list_count: int | None = None
    card_count: int | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanBoardListResponse(BaseModel):
    boards: list[KanbanBoardResponse]
    pagination: KanbanPaginationInfo


class KanbanListCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    client_id: str = Field(..., min_length=1, max_length=100)
    position: int | None = Field(None, ge=0)


class KanbanListUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    position: int | None = Field(None, ge=0)


class KanbanListResponse(BaseModel):
    id: int
    uuid: str
    board_id: int
    client_id: str
    name: str
    position: int
    archived: bool
    archived_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    deleted: bool
    deleted_at: datetime | None = None
    version: int
    card_count: int | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanListsListResponse(BaseModel):
    lists: list[KanbanListResponse]


class KanbanListPositionItem(BaseModel):
    list_id: int
    position: int = Field(..., ge=0)


class KanbanCardPositionItem(BaseModel):
    card_id: int
    position: int = Field(..., ge=0)


class KanbanReorderRequest(BaseModel):
    ids: list[int] | None = Field(None, min_length=1)
    list_positions: list[KanbanListPositionItem] | None = Field(None, min_length=1)
    card_positions: list[KanbanCardPositionItem] | None = Field(None, min_length=1)

    @model_validator(mode="after")
    def normalize_legacy_payloads(self) -> "KanbanReorderRequest":
        if self.ids:
            return self
        if self.list_positions:
            ordered = sorted(self.list_positions, key=lambda item: item.position)
            self.ids = [item.list_id for item in ordered]
            return self
        if self.card_positions:
            ordered = sorted(self.card_positions, key=lambda item: item.position)
            self.ids = [item.card_id for item in ordered]
            return self
        raise ValueError("Provide one of: ids, list_positions, or card_positions")


class KanbanReorderResponse(BaseModel):
    success: bool
    message: str | None = None


class KanbanCardCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str | None = Field(None, max_length=50000)
    client_id: str = Field(..., min_length=1, max_length=100)
    position: int | None = Field(None, ge=0)
    due_date: datetime | None = None
    start_date: datetime | None = None
    priority: KanbanPriority | None = None
    label_ids: list[int] | None = None
    metadata: dict[str, Any] | None = None


class KanbanCardUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    description: str | None = Field(None, max_length=50000)
    due_date: datetime | None = None
    due_complete: bool | None = None
    start_date: datetime | None = None
    priority: KanbanPriority | None = None
    metadata: dict[str, Any] | None = None


class KanbanCardResponse(BaseModel):
    id: int
    uuid: str
    board_id: int
    list_id: int
    client_id: str
    title: str
    description: str | None = None
    position: int
    due_date: datetime | None = None
    due_complete: bool
    start_date: datetime | None = None
    priority: KanbanPriority | None = None
    archived: bool
    archived_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    deleted: bool
    deleted_at: datetime | None = None
    version: int
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanCardsListResponse(BaseModel):
    cards: list[KanbanCardResponse]


class KanbanCardMoveRequest(BaseModel):
    target_list_id: int
    position: int | None = Field(None, ge=0)


class KanbanCardCopyRequest(BaseModel):
    target_list_id: int
    new_client_id: str = Field(..., min_length=1, max_length=100)
    position: int | None = Field(None, ge=0)
    new_title: str | None = Field(None, min_length=1, max_length=500)


class KanbanCardInListResponse(KanbanCardResponse):
    labels: list[dict[str, Any]] = Field(default_factory=list)
    checklist_count: int = 0
    checklist_complete: int = 0
    checklist_total: int = 0
    comment_count: int = 0


class KanbanListWithCardsResponse(KanbanListResponse):
    cards: list[KanbanCardInListResponse] = Field(default_factory=list)


class KanbanBoardWithListsResponse(KanbanBoardResponse):
    labels: list[dict[str, Any]] = Field(default_factory=list)
    lists: list[KanbanListWithCardsResponse] = Field(default_factory=list)
    total_cards: int = 0


class KanbanCardWithDetailsResponse(KanbanCardResponse):
    labels: list[dict[str, Any]] = Field(default_factory=list)
    checklists: list[dict[str, Any]] = Field(default_factory=list)
    comment_count: int = 0
