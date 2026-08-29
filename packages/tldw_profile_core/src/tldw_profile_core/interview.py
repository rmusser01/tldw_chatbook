from pydantic import Field, field_validator

from .payloads import FrozenModel


class InterviewQuestion(FrozenModel):
    question_id: str
    text: str

    @field_validator("text")
    @classmethod
    def single_question(cls, value: str) -> str:
        if any(token in value.lower() for token in (" and ", " or ", ";")):
            raise ValueError("question must ask one thing")
        return value


class InterviewTurn(FrozenModel):
    question_id: str
    answer: str


class InterviewPack(FrozenModel):
    pack_id: str
    questions: list[InterviewQuestion] = Field(max_length=20)


class InterviewProposalBatch(FrozenModel):
    pack_id: str
    turns: list[InterviewTurn]
