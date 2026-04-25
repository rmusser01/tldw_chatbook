from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .media_reading_schemas import FileExportInfo


ColumnType = Literal["text", "number", "date", "url", "boolean", "currency"]
SourceType = Literal["chat", "document", "rag_query"]
DataTableExportFormat = Literal["csv", "json", "xlsx"]
DataTableRowData = dict[str, Any] | list[Any]


class DataTableColumnHint(BaseModel):
    name: str = Field(..., min_length=1)
    type: ColumnType | None = None
    description: str | None = None
    format: str | None = None


class DataTableSourceInput(BaseModel):
    source_type: SourceType
    source_id: str = Field(..., min_length=1)
    title: str | None = None
    snapshot: Any | None = None
    retrieval_params: dict[str, Any] | None = None


class DataTableGenerateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    description: str | None = None
    workspace_tag: str | None = None
    sources: list[DataTableSourceInput]
    column_hints: list[DataTableColumnHint] | None = None
    model: str | None = None
    max_rows: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_payload(self) -> DataTableGenerateRequest:
        if not self.sources:
            raise ValueError("sources are required")
        return self


class DataTableRegenerateRequest(BaseModel):
    prompt: str | None = None
    model: str | None = None
    max_rows: int | None = Field(default=None, ge=1)


class DataTableUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> DataTableUpdateRequest:
        if self.name is None and self.description is None:
            raise ValueError("at least one field is required")
        if self.name is not None and not self.name.strip():
            raise ValueError("name cannot be blank")
        return self


class DataTableColumn(BaseModel):
    column_id: str
    name: str
    type: ColumnType
    description: str | None = None
    format: str | None = None
    position: int


class DataTableColumnInput(BaseModel):
    column_id: str | None = None
    name: str = Field(..., min_length=1)
    type: ColumnType
    description: str | None = None
    format: str | None = None
    position: int | None = None


class DataTableRow(BaseModel):
    row_id: str
    row_index: int
    data: DataTableRowData
    row_hash: str | None = None


class DataTableSource(BaseModel):
    source_type: SourceType
    source_id: str
    title: str | None = None
    snapshot: Any | None = None
    retrieval_params: Any | None = None


class DataTableSummary(BaseModel):
    uuid: str
    name: str
    description: str | None = None
    workspace_tag: str | None = None
    prompt: str
    column_hints: Any | None = None
    status: str
    row_count: int
    column_count: int | None = None
    generation_model: str | None = None
    last_error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_modified: str | None = None
    version: int | None = None
    source_count: int | None = None


class DataTablesListResponse(BaseModel):
    tables: list[DataTableSummary]
    count: int
    limit: int
    offset: int
    total: int | None = None


class DataTableDetailResponse(BaseModel):
    table: DataTableSummary
    columns: list[DataTableColumn]
    rows: list[DataTableRow]
    sources: list[DataTableSource]
    rows_limit: int
    rows_offset: int


class DataTableContentUpdateRequest(BaseModel):
    columns: list[DataTableColumnInput]
    rows: list[dict[str, Any]]


class DataTableGenerateResponse(BaseModel):
    job_id: int
    job_uuid: str | None = None
    status: str
    table: DataTableSummary


class DataTableDeleteResponse(BaseModel):
    success: bool


class DataTableJobStatus(BaseModel):
    id: int
    uuid: str | None
    status: str
    job_type: str
    owner_user_id: str | None
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    cancelled_at: str | None
    cancellation_reason: str | None
    progress_percent: float | None
    progress_message: str | None
    result: dict[str, Any] | None
    error_message: str | None
    table_uuid: str | None = None


class DataTableJobCancelResponse(BaseModel):
    success: bool
    job_id: int
    status: str
    message: str | None = None


class DataTableExportResponse(BaseModel):
    table_uuid: str
    file_id: int
    export: FileExportInfo


__all__ = [
    "ColumnType",
    "DataTableColumn",
    "DataTableColumnHint",
    "DataTableColumnInput",
    "DataTableContentUpdateRequest",
    "DataTableDeleteResponse",
    "DataTableDetailResponse",
    "DataTableExportFormat",
    "DataTableExportResponse",
    "DataTableGenerateRequest",
    "DataTableGenerateResponse",
    "DataTableJobCancelResponse",
    "DataTableJobStatus",
    "DataTableRegenerateRequest",
    "DataTableRow",
    "DataTableRowData",
    "DataTablesListResponse",
    "DataTableSource",
    "DataTableSourceInput",
    "DataTableSummary",
    "DataTableUpdateRequest",
    "SourceType",
]
