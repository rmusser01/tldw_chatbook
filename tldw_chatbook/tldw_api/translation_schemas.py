from __future__ import annotations

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    target_language: str = "English"
    source_language: str | None = None
    model: str | None = None
    provider: str | None = None


class TranslateResponse(BaseModel):
    translated_text: str
    detected_source_language: str | None = None
    target_language: str
    model_used: str


__all__ = ["TranslateRequest", "TranslateResponse"]
