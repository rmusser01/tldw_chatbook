import pytest
from pydantic import ValidationError

from tldw_profile_core import (
    InterviewAudience,
    InterviewPack,
    InterviewProposalBatch,
    InterviewQuestion,
    InterviewTurn,
)


def question(question_id="q1", text="What response style helps you most?"):
    return InterviewQuestion(question_id=question_id, topic="communication", text=text)


def pack(**changes):
    values = dict(
        pack_id="personal-v1",
        pack_version=1,
        audience="personal",
        coverage_version=1,
        coverage_topics=("communication", "constraints"),
        questions=(question(),),
    )
    values.update(changes)
    return InterviewPack(**values)


def test_pack_has_versioned_metadata_topics_and_immutable_questions():
    value = pack()
    assert value.schema_version == value.pack_version == value.coverage_version == 1
    assert value.audience is InterviewAudience.PERSONAL
    assert isinstance(value.coverage_topics, tuple) and isinstance(
        value.questions, tuple
    )
    with pytest.raises(ValidationError):
        setattr(value, "questions", ())
    assert pack(audience="workspace").audience is InterviewAudience.WORKSPACE


def test_pack_limits_questions_and_requires_unique_ids_and_known_topics():
    with pytest.raises(ValidationError):
        pack(questions=tuple(question(str(i)) for i in range(21)))
    with pytest.raises(ValidationError):
        pack(questions=(question("same"), question("same")))
    with pytest.raises(ValidationError):
        pack(
            questions=(
                InterviewQuestion(question_id="q", topic="unknown", text="What helps?"),
            )
        )


def test_batch_limits_turns_and_validates_question_references():
    value = InterviewProposalBatch(
        pack=pack(), turns=(InterviewTurn(question_id="q1", answer="Short examples."),)
    )
    assert isinstance(value.turns, tuple)
    with pytest.raises(ValidationError):
        InterviewProposalBatch(
            pack=pack(), turns=(InterviewTurn(question_id="missing", answer="answer"),)
        )
    with pytest.raises(ValidationError):
        InterviewProposalBatch(
            pack=pack(),
            turns=tuple(
                InterviewTurn(question_id="q1", answer=str(i)) for i in range(21)
            ),
        )


@pytest.mark.parametrize(
    "text",
    [
        "What helps; why does it help?",
        "What helps? Why does it help?",
        "What helps and why?",
        "What helps and how should I apply it?",
    ],
)
def test_rejects_clear_compound_questions(text):
    with pytest.raises(ValidationError):
        question(text=text)


@pytest.mark.parametrize(
    "text",
    [
        "Do you prefer examples and concise explanations?",
        "Do you use a password manager?",
        "What token budget should answers use?",
        "Would examples or diagrams help?",
    ],
)
def test_allows_benign_and_or_security_vocabulary(text):
    assert question(text=text).text == text


@pytest.mark.parametrize(
    "secret",
    [
        "My password is hunter2-secret",
        "Use API key sk-abcdefghijklmnopqrstuvwxyz123456",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
        "-----BEGIN PRIVATE KEY-----",
    ],
)
def test_rejects_recognizable_secrets_at_question_and_answer_boundaries(secret):
    with pytest.raises(ValidationError):
        question(text=f"What should I remember: {secret}?")
    with pytest.raises(ValidationError):
        InterviewTurn(question_id="q1", answer=secret)
