import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, field_validator, model_validator

from .payloads import FrozenModel, reject_blank


def _bounded(max_length: int):
    return Annotated[
        str, Field(min_length=1, max_length=max_length), AfterValidator(reject_blank)
    ]


InterviewId = _bounded(128)
InterviewTopic = _bounded(128)
QuestionText = _bounded(1_000)
AnswerText = _bounded(16_384)

_COMPOUND_CLAUSE = re.compile(
    r"\band\s+(?:why|how|what|when|where|who|which)\b", re.IGNORECASE
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:gh[pousr]_|xox[baprs]-)[A-Za-z0-9_-]{20,}\b", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.IGNORECASE),
    re.compile(r"\b(?:my\s+)?password\s*(?:is|=|:)\s*\S{8,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token)\s*(?:is|=|:)\s*\S{12,}", re.IGNORECASE
    ),
)


def _reject_secret(value: str) -> str:
    if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
        raise ValueError("recognized secret material is not allowed")
    return value


class InterviewAudience(StrEnum):
    PERSONAL = "personal"
    WORKSPACE = "workspace"


class InterviewQuestion(FrozenModel):
    schema_version: Literal[1] = 1
    question_id: InterviewId
    topic: InterviewTopic
    text: QuestionText

    @field_validator("text")
    @classmethod
    def validate_question(cls, value: str) -> str:
        _reject_secret(value)
        if value.count("?") > 1 or ";" in value or _COMPOUND_CLAUSE.search(value):
            raise ValueError("question must ask one thing")
        return value


class InterviewTurn(FrozenModel):
    schema_version: Literal[1] = 1
    question_id: InterviewId
    answer: AnswerText

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        return _reject_secret(value)


class InterviewPack(FrozenModel):
    schema_version: Literal[1] = 1
    pack_id: InterviewId
    pack_version: Literal[1]
    audience: InterviewAudience
    coverage_version: Literal[1]
    coverage_topics: tuple[InterviewTopic, ...] = Field(min_length=1, max_length=32)
    questions: tuple[InterviewQuestion, ...] = Field(max_length=20)

    @model_validator(mode="after")
    def validate_pack(self):
        if len(set(self.coverage_topics)) != len(self.coverage_topics):
            raise ValueError("coverage topics must be unique")
        question_ids = [question.question_id for question in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("question IDs must be unique")
        if any(
            question.topic not in self.coverage_topics for question in self.questions
        ):
            raise ValueError("question topic is not covered by the pack")
        return self


class InterviewProposalBatch(FrozenModel):
    schema_version: Literal[1] = 1
    pack: InterviewPack
    turns: tuple[InterviewTurn, ...] = Field(max_length=20)

    @model_validator(mode="after")
    def validate_turns(self):
        valid_ids = {question.question_id for question in self.pack.questions}
        turn_ids = [turn.question_id for turn in self.turns]
        if any(question_id not in valid_ids for question_id in turn_ids):
            raise ValueError("turn references an unknown question")
        if len(set(turn_ids)) != len(turn_ids):
            raise ValueError("question may be answered at most once")
        return self
