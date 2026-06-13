from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ConnectorProviderName = Literal["drive", "notion", "gmail", "onedrive", "zotero"]
ConnectorAuthType = Literal["oauth1", "oauth2", "token"]
ConnectorSourceType = Literal["folder", "file", "page", "database", "link", "collection"]


def _validate_connector_source_type(provider: str, type_: str) -> None:
    normalized_provider = str(provider or "").strip().lower()
    normalized_type = str(type_ or "").strip().lower()
    if normalized_type == "collection" and normalized_provider != "zotero":
        raise ValueError("Connector source type 'collection' is only supported for provider 'zotero'.")
    if normalized_provider == "zotero" and normalized_type != "collection":
        raise ValueError("Zotero sources must use type 'collection'.")


class ConnectorProvider(BaseModel):
    name: ConnectorProviderName
    auth_type: ConnectorAuthType = "oauth2"
    scopes_required: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True, extra="allow")


class AuthorizeURLResponse(BaseModel):
    auth_url: str
    state: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ConnectorAccount(BaseModel):
    id: int
    provider: ConnectorProviderName
    display_name: str
    created_at: str | None = None
    connected: bool = True
    email: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class SyncOptions(BaseModel):
    recursive: bool = True
    include_types: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    export_format_overrides: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ConnectorSourceSyncSummary(BaseModel):
    state: str = "idle"
    sync_mode: str = "manual"
    last_sync_succeeded_at: str | None = None
    last_sync_failed_at: str | None = None
    last_error: str | None = None
    webhook_status: str | None = None
    needs_full_rescan: bool = False
    active_job_id: str | None = None
    tracked_item_count: int = 0
    degraded_item_count: int = 0
    duplicate_count: int = 0
    metadata_only_count: int = 0

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ConnectorSource(BaseModel):
    id: int
    account_id: int
    provider: ConnectorProviderName
    remote_id: str
    type: ConnectorSourceType
    path: str | None = None
    options: SyncOptions = Field(default_factory=SyncOptions)
    enabled: bool = True
    last_synced_at: str | None = None
    sync: ConnectorSourceSyncSummary | None = None

    @model_validator(mode="after")
    def validate_provider_type_pairing(self) -> "ConnectorSource":
        _validate_connector_source_type(self.provider, self.type)
        return self

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ConnectorImportJob(BaseModel):
    id: str
    source_id: int
    type: str = "import"
    status: Literal["queued", "running", "succeeded", "failed", "canceled"] = "queued"
    progress_pct: int = 0
    counts: dict[str, int] = Field(default_factory=lambda: {"processed": 0, "skipped": 0, "failed": 0})
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ConnectorSyncJobSummary(BaseModel):
    id: str
    type: str = "import"
    status: str
    progress_pct: int = 0
    counts: dict[str, int] = Field(default_factory=lambda: {"processed": 0, "skipped": 0, "failed": 0})

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ConnectorSourceSyncStatus(BaseModel):
    source_id: int
    provider: ConnectorProviderName
    enabled: bool = True
    state: str = "idle"
    sync_mode: str = "manual"
    cursor: str | None = None
    cursor_kind: str | None = None
    last_bootstrap_at: str | None = None
    last_sync_started_at: str | None = None
    last_sync_succeeded_at: str | None = None
    last_sync_failed_at: str | None = None
    last_error: str | None = None
    retry_backoff_count: int = 0
    webhook_status: str | None = None
    webhook_expires_at: str | None = None
    needs_full_rescan: bool = False
    active_job_id: str | None = None
    active_job_started_at: str | None = None
    active_job: ConnectorSyncJobSummary | None = None
    tracked_item_count: int = 0
    degraded_item_count: int = 0
    duplicate_count: int = 0
    metadata_only_count: int = 0

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ConnectorSourceSyncTriggerResponse(BaseModel):
    source_id: int
    provider: ConnectorProviderName
    status: Literal["queued"] = "queued"
    job: ConnectorImportJob

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ConnectorBrowseResponse(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    next_cursor: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ConnectorSourceCreateRequest(BaseModel):
    account_id: int
    provider: ConnectorProviderName
    remote_id: str
    type: ConnectorSourceType
    path: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider_type_pairing(self) -> "ConnectorSourceCreateRequest":
        _validate_connector_source_type(self.provider, self.type)
        return self

    model_config = ConfigDict(extra="forbid")


class ConnectorSourcePatchRequest(BaseModel):
    enabled: bool | None = None
    options: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")
