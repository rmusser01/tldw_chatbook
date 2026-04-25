from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ChatGrammarValidationStatus = Literal["unchecked", "valid", "invalid"]


class ChatGrammarBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    grammar_text: str = Field(..., min_length=1, max_length=200_000)


class ChatGrammarCreate(ChatGrammarBase):
    pass


class ChatGrammarUpdate(BaseModel):
    version: int | None = Field(None, ge=1)
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    grammar_text: str | None = Field(None, min_length=1, max_length=200_000)
    validation_status: ChatGrammarValidationStatus | None = None
    validation_error: str | None = Field(None, max_length=4000)
    last_validated_at: datetime | None = None
    is_archived: bool | None = None

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> "ChatGrammarUpdate":
        if (
            self.name is None
            and self.description is None
            and self.grammar_text is None
            and self.validation_status is None
            and self.validation_error is None
            and self.last_validated_at is None
            and self.is_archived is None
        ):
            raise ValueError("At least one updatable field must be provided")
        return self


class ChatGrammarResponse(ChatGrammarBase):
    id: str
    validation_status: ChatGrammarValidationStatus
    validation_error: str | None = None
    last_validated_at: datetime | None = None
    is_archived: bool = False
    created_at: datetime
    updated_at: datetime
    version: int

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ChatGrammarListResponse(BaseModel):
    items: list[ChatGrammarResponse] = Field(default_factory=list)
    total: int

    model_config = ConfigDict(from_attributes=True, extra="allow")
