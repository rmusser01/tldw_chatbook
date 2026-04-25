from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


KanbanPriority = Literal["low", "medium", "high", "urgent"]
KanbanLabelColor = Literal["red", "orange", "yellow", "green", "blue", "purple", "pink", "gray"]
KanbanSearchMode = Literal["fts", "vector", "hybrid"]


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


class KanbanActivityResponse(BaseModel):
    id: int
    uuid: str
    board_id: int
    list_id: int | None = None
    card_id: int | None = None
    user_id: str
    action_type: str
    entity_type: str
    entity_id: int | None = None
    details: dict[str, Any] | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanActivitiesListResponse(BaseModel):
    activities: list[KanbanActivityResponse]
    pagination: KanbanPaginationInfo


class KanbanLabelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    color: KanbanLabelColor


class KanbanLabelUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=50)
    color: KanbanLabelColor | None = None


class KanbanLabelResponse(BaseModel):
    id: int
    uuid: str
    board_id: int
    name: str
    color: KanbanLabelColor
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanLabelsListResponse(BaseModel):
    labels: list[KanbanLabelResponse]


class KanbanChecklistCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    position: int | None = Field(None, ge=0)


class KanbanChecklistUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)


class KanbanChecklistResponse(BaseModel):
    id: int
    uuid: str
    card_id: int
    name: str
    position: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanChecklistsListResponse(BaseModel):
    checklists: list[KanbanChecklistResponse]


class KanbanChecklistReorderRequest(BaseModel):
    checklist_ids: list[int] = Field(..., min_length=1)


class KanbanChecklistItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    position: int | None = Field(None, ge=0)
    checked: bool = False


class KanbanChecklistItemUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=500)
    checked: bool | None = None


class KanbanChecklistItemResponse(BaseModel):
    id: int
    uuid: str
    checklist_id: int
    name: str
    position: int
    checked: bool
    checked_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanChecklistItemsListResponse(BaseModel):
    items: list[KanbanChecklistItemResponse]


class KanbanChecklistItemReorderRequest(BaseModel):
    item_ids: list[int] = Field(..., min_length=1)


class KanbanChecklistWithItemsResponse(KanbanChecklistResponse):
    items: list[KanbanChecklistItemResponse] = Field(default_factory=list)
    total_items: int = 0
    checked_items: int = 0
    progress_percent: int = Field(0, ge=0, le=100)


class KanbanToggleAllChecklistItemsRequest(BaseModel):
    checked: bool


class KanbanCommentCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class KanbanCommentUpdate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class KanbanCommentResponse(BaseModel):
    id: int
    uuid: str
    card_id: int
    user_id: str
    content: str
    created_at: datetime
    updated_at: datetime
    deleted: bool = False

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanCommentsListResponse(BaseModel):
    comments: list[KanbanCommentResponse]
    pagination: KanbanPaginationInfo


class KanbanBoardExportRequest(BaseModel):
    include_archived: bool = False
    include_deleted: bool = False


class KanbanBoardExportResponse(BaseModel):
    format: str
    exported_at: str
    board: dict[str, Any]
    labels: list[dict[str, Any]]
    lists: list[dict[str, Any]]

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanBoardImportRequest(BaseModel):
    data: dict[str, Any]
    board_name: str | None = None


class KanbanImportStatsResponse(BaseModel):
    board_id: int
    lists_imported: int = 0
    cards_imported: int = 0
    labels_imported: int = 0
    checklists_imported: int = 0
    checklist_items_imported: int = 0
    comments_imported: int = 0


class KanbanBoardImportResponse(BaseModel):
    board: KanbanBoardResponse
    import_stats: KanbanImportStatsResponse


class KanbanBulkMoveCardsRequest(BaseModel):
    card_ids: list[int] = Field(..., min_length=1)
    target_list_id: int
    position: int | None = Field(None, ge=0)


class KanbanBulkMoveCardsResponse(BaseModel):
    success: bool
    moved_count: int
    cards: list[KanbanCardResponse]


class KanbanBulkArchiveCardsRequest(BaseModel):
    card_ids: list[int] = Field(..., min_length=1)


class KanbanBulkArchiveCardsResponse(BaseModel):
    success: bool
    archived_count: int


class KanbanBulkUnarchiveCardsResponse(BaseModel):
    success: bool
    unarchived_count: int


class KanbanBulkDeleteCardsRequest(BaseModel):
    card_ids: list[int] = Field(..., min_length=1)


class KanbanBulkDeleteCardsResponse(BaseModel):
    success: bool
    deleted_count: int


class KanbanBulkLabelCardsRequest(BaseModel):
    card_ids: list[int] = Field(..., min_length=1)
    add_label_ids: list[int] | None = None
    remove_label_ids: list[int] | None = None

    @field_validator("add_label_ids", "remove_label_ids")
    @classmethod
    def normalize_empty_label_lists(cls, value: list[int] | None) -> list[int] | None:
        return value or None


class KanbanBulkLabelCardsResponse(BaseModel):
    success: bool
    updated_count: int


class KanbanFilteredCardsResponse(BaseModel):
    cards: list[KanbanCardResponse]
    pagination: KanbanPaginationInfo


class KanbanCardCopyWithChecklistsRequest(BaseModel):
    target_list_id: int
    new_client_id: str = Field(..., min_length=1, max_length=100)
    position: int | None = Field(None, ge=0)
    new_title: str | None = Field(None, max_length=500)
    copy_checklists: bool = True
    copy_labels: bool = True


class KanbanCardSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    board_id: int | None = None
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)
    page: int | None = Field(None, ge=1)
    per_page: int | None = Field(None, ge=1, le=200)

    @model_validator(mode="after")
    def normalize_legacy_pagination(self) -> "KanbanCardSearchRequest":
        if self.per_page is not None:
            self.limit = self.per_page
        if self.page is not None:
            self.offset = (self.page - 1) * self.limit
        return self


class KanbanCardSearchResponse(BaseModel):
    cards: list[KanbanCardResponse]
    pagination: KanbanPaginationInfo


class KanbanSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    board_id: int | None = None
    label_ids: list[int] | None = None
    priority: str | None = None
    include_archived: bool = False
    search_mode: KanbanSearchMode = "fts"
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
    page: int | None = Field(None, ge=1)
    per_page: int | None = Field(None, ge=1, le=100)

    @model_validator(mode="after")
    def normalize_legacy_pagination(self) -> "KanbanSearchRequest":
        if self.per_page is not None:
            self.limit = self.per_page
        if self.page is not None:
            self.offset = (self.page - 1) * self.limit
        return self


class KanbanSearchResultCard(BaseModel):
    id: int
    uuid: str
    board_id: int
    board_name: str
    list_id: int
    list_name: str
    title: str
    description: str | None = None
    priority: str | None = None
    due_date: datetime | None = None
    labels: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    relevance_score: float | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanSearchResponse(BaseModel):
    query: str
    search_mode: str
    results: list[KanbanSearchResultCard]
    pagination: KanbanPaginationInfo


class KanbanCardLinkCreate(BaseModel):
    linked_type: str
    linked_id: str

    @field_validator("linked_type")
    @classmethod
    def validate_linked_type(cls, value: str) -> str:
        if value not in ("media", "note"):
            raise ValueError("linked_type must be 'media' or 'note'")
        return value


class KanbanCardLinkResponse(BaseModel):
    id: int
    card_id: int
    linked_type: str
    linked_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanCardLinksListResponse(BaseModel):
    links: list[KanbanCardLinkResponse]


class KanbanCardLinkCountsResponse(BaseModel):
    media: int = 0
    note: int = 0


class KanbanBulkCardLinksRequest(BaseModel):
    links: list[KanbanCardLinkCreate] = Field(..., min_length=1, max_length=100)


class KanbanBulkCardLinksAddResponse(BaseModel):
    added_count: int
    skipped_count: int
    links: list[KanbanCardLinkResponse]


class KanbanBulkCardLinksRemoveResponse(BaseModel):
    removed_count: int


class KanbanLinkedCardResponse(BaseModel):
    id: int
    title: str
    description: str | None = None
    board_id: int
    board_name: str
    list_id: int
    list_name: str
    position: int
    is_archived: bool = False
    is_deleted: bool = False
    link_id: int
    linked_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="allow")


class KanbanLinkedCardsListResponse(BaseModel):
    linked_type: str
    linked_id: str
    cards: list[KanbanLinkedCardResponse]
