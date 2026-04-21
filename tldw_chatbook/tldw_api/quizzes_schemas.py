from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


QuestionType = Literal["multiple_choice", "multi_select", "matching", "true_false", "fill_blank"]
AnswerValue = Union[int, str, list[int], dict[str, str]]


class QuizCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    time_limit_seconds: Optional[int] = Field(None, ge=1)
    passing_score: Optional[int] = Field(None, ge=0, le=100)


class QuizUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    time_limit_seconds: Optional[int] = Field(None, ge=1)
    passing_score: Optional[int] = Field(None, ge=0, le=100)
    expected_version: Optional[int] = None


class QuizResponse(BaseModel):
    id: int | str
    name: str
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    total_questions: int = 0
    time_limit_seconds: Optional[int] = None
    passing_score: Optional[int] = None
    deleted: bool = False
    client_id: Optional[str] = None
    version: int = 1
    created_at: Optional[str] = None
    last_modified: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class QuizListResponse(BaseModel):
    items: list[QuizResponse]
    count: int

    model_config = ConfigDict(from_attributes=True)


class QuizQuestionCreateRequest(BaseModel):
    question_type: QuestionType
    question_text: str
    options: Optional[list[str]] = None
    correct_answer: AnswerValue
    explanation: Optional[str] = None
    hint: Optional[str] = None
    hint_penalty_points: int = Field(0, ge=0)
    source_citations: Optional[list[dict[str, Any]]] = None
    points: int = Field(1, ge=0)
    order_index: int = 0
    tags: Optional[list[str]] = None


class QuizQuestionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None
    options: Optional[list[str]] = None
    correct_answer: Optional[AnswerValue] = None
    explanation: Optional[str] = None
    hint: Optional[str] = None
    hint_penalty_points: Optional[int] = Field(None, ge=0)
    source_citations: Optional[list[dict[str, Any]]] = None
    points: Optional[int] = Field(None, ge=0)
    order_index: Optional[int] = None
    tags: Optional[list[str]] = None
    expected_version: Optional[int] = None


class QuizQuestionResponse(BaseModel):
    id: int | str
    quiz_id: int | str
    question_type: QuestionType
    question_text: str
    options: Optional[list[str]] = None
    correct_answer: Optional[AnswerValue] = None
    explanation: Optional[str] = None
    hint: Optional[str] = None
    hint_penalty_points: int = Field(0, ge=0)
    source_citations: Optional[list[dict[str, Any]]] = None
    points: int = Field(1, ge=0)
    order_index: int = 0
    tags: Optional[list[str]] = None
    deleted: bool = False
    client_id: Optional[str] = None
    version: int = 1
    created_at: Optional[str] = None
    last_modified: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class QuizQuestionListResponse(BaseModel):
    items: list[QuizQuestionResponse]
    count: int

    model_config = ConfigDict(from_attributes=True)


class QuizAttemptAnswerInput(BaseModel):
    question_id: int | str
    user_answer: AnswerValue
    hint_used: Optional[bool] = None
    time_spent_ms: Optional[int] = None


class QuizAttemptSubmitRequest(BaseModel):
    answers: list[QuizAttemptAnswerInput]


class QuizAttemptAnswer(BaseModel):
    question_id: int | str
    user_answer: AnswerValue
    is_correct: bool
    correct_answer: Optional[AnswerValue] = None
    explanation: Optional[str] = None
    hint_used: Optional[bool] = None
    hint_penalty_points: Optional[int] = Field(None, ge=0)
    source_citations: Optional[list[dict[str, Any]]] = None
    points_awarded: Optional[int] = None
    time_spent_ms: Optional[int] = None


class QuizAttemptResponse(BaseModel):
    id: int | str
    quiz_id: int | str
    started_at: str
    completed_at: Optional[str] = None
    score: Optional[int] = None
    total_possible: int
    time_spent_seconds: Optional[int] = None
    answers: list[QuizAttemptAnswer] = Field(default_factory=list)
    questions: Optional[list[QuizQuestionResponse]] = None

    model_config = ConfigDict(from_attributes=True)


class QuizAttemptListResponse(BaseModel):
    items: list[QuizAttemptResponse]
    count: int

    model_config = ConfigDict(from_attributes=True)
