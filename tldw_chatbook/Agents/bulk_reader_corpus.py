"""Strict input models for the opt-in synthetic bulk-reader comparison."""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from tldw_chatbook.Utils.path_validation import validate_path_simple

NonblankText = Annotated[str, Field(min_length=1, pattern=r"\S")]


class BulkReaderCase(BaseModel):
    """One synthetic question, its confined sources, and manual answer facts."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    id: NonblankText
    category: str = ""
    question: NonblankText
    sources: dict[NonblankText, str] = Field(min_length=1)
    expected_facts: list[NonblankText]

    @field_validator("sources")
    @classmethod
    def _validate_sources(cls, sources: dict[str, str]) -> dict[str, str]:
        for value in sources:
            validated = validate_path_simple(value, probe_existing=False)
            candidate = PurePosixPath(validated.as_posix())
            if (
                candidate.is_absolute()
                or PureWindowsPath(value).drive
                or "\\" in value
                or ".." in candidate.parts
                or value != candidate.as_posix()
                or not candidate.parts
            ):
                raise ValueError("Corpus sources require a relative confined path.")
        return sources


class BulkReaderCorpus(BaseModel):
    """Versioned synthetic cases with a strictly typed manual grading rubric."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: StrictInt = Field(ge=1, le=1)
    id: NonblankText
    description: str = ""
    rubric: dict[NonblankText, NonblankText | list[NonblankText]] = Field(min_length=1)
    cases: list[BulkReaderCase] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def _validate_case_ids(cls, cases: list[BulkReaderCase]) -> list[BulkReaderCase]:
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("Bulk-reader case ids must be unique.")
        return cases
