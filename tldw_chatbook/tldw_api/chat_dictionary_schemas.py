"""Request schemas for tldw_server chat dictionary APIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DictionaryEntryType = Literal["literal", "regex"]
BulkDictionaryEntryOperation = Literal["delete", "activate", "deactivate", "group"]


class ChatDictionaryCreateRequest(BaseModel):
    name: str
    description: str | None = None
    default_token_budget: int | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    included_dictionary_ids: list[int] = Field(default_factory=list)


class ChatDictionaryUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    category: str | None = None
    tags: list[str] | None = None
    included_dictionary_ids: list[int] | None = None
    is_active: bool | None = None
    default_token_budget: int | None = None
    version: int | None = None


class DictionaryEntryCreateRequest(BaseModel):
    pattern: str
    replacement: str
    probability: float = 1.0
    group: str | None = None
    timed_effects: dict[str, Any] | None = None
    max_replacements: int = 0
    type: DictionaryEntryType = "literal"
    enabled: bool = True
    case_sensitive: bool = False


class DictionaryEntryUpdateRequest(BaseModel):
    pattern: str | None = None
    replacement: str | None = None
    probability: float | None = None
    group: str | None = None
    timed_effects: dict[str, Any] | None = None
    max_replacements: int | None = None
    type: DictionaryEntryType | None = None
    enabled: bool | None = None
    case_sensitive: bool | None = None


class BulkDictionaryEntryOperationRequest(BaseModel):
    operation: BulkDictionaryEntryOperation
    entry_ids: list[int]
    group_name: str | None = None


class DictionaryEntryReorderRequest(BaseModel):
    entry_ids: list[int]


class ProcessChatDictionariesRequest(BaseModel):
    text: str
    dictionary_id: int | None = None
    group: str | None = None
    max_iterations: int | None = None
    token_budget: int | None = None
    chat_id: str | None = None


class ImportDictionaryMarkdownRequest(BaseModel):
    name: str
    content: str
    activate: bool = True


class ImportDictionaryJSONRequest(BaseModel):
    data: dict[str, Any]
    activate: bool = True


class ValidateDictionaryRequest(BaseModel):
    data: dict[str, Any]
