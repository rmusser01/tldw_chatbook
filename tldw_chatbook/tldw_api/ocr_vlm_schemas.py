from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel


class OCRBackendDiscoveryEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    available: bool | None = None
    mode: str | None = None
    configured_mode: str | None = None
    model: str | None = None
    configured: bool | None = None
    supports_structured_output: bool | None = None
    supports_json: bool | None = None
    configured_flags: str | None = None
    auto_eligible: bool | None = None
    auto_high_quality_eligible: bool | None = None
    url_configured: bool | None = None
    managed_configured: bool | None = None
    managed_running: bool | None = None
    allow_managed_start: bool | None = None
    cli_configured: bool | None = None
    backend_concurrency_cap: int | None = None
    healthcheck_url_configured: bool | None = None
    prompt: str | None = None
    prompt_preset: str | None = None
    text_format: str | None = None
    table_format: str | None = None
    remote_mode: str | None = None
    sglang_reachable: bool | None = None
    vllm_reachable: bool | None = None
    pdf_only: bool | None = None
    document_level: bool | None = None
    opt_in_only: bool | None = None
    supports_per_page_metrics: bool | None = None
    timeout_sec: int | None = None
    max_concurrency: int | None = None
    tmp_root: str | None = None
    debug_save_raw: bool | None = None
    model_id: str | None = None
    base_size: int | None = None
    image_size: int | None = None
    crop_mode: bool | str | None = None
    device: str | None = None
    dtype: str | None = None
    attn_impl: str | None = None
    error: str | None = None


class OCRBackendsResponse(RootModel[dict[str, OCRBackendDiscoveryEntry]]):
    pass


class OCRPointsPreloadResponse(BaseModel):
    status: str
    mode: str | None = None
    error: str | None = None


class VLMBackendsResponse(RootModel[dict[str, dict[str, Any]]]):
    pass


__all__ = [
    "OCRBackendDiscoveryEntry",
    "OCRBackendsResponse",
    "OCRPointsPreloadResponse",
    "VLMBackendsResponse",
]
