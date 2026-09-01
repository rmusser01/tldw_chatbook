import pytest
from pydantic import ValidationError

from tldw_profile_core import (
    AgentVisibility,
    InterviewAudience,
    InterviewPack,
    InterviewProposalBatch,
    InterviewProposedChange,
    InterviewQuestion,
    InterviewTurn,
    LegacyUnclassifiedPayload,
    PreferencePayload,
    ProfileControls,
    ProposalOperation,
    SemanticKey,
    SyncMode,
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


@pytest.mark.parametrize("version", [True, "1"])
def test_interview_versions_reject_non_json_integer_values(version):
    with pytest.raises(ValidationError):
        InterviewQuestion(
            schema_version=version,
            question_id="q1",
            topic="communication",
            text="What helps?",
        )
    with pytest.raises(ValidationError):
        pack(schema_version=version)
    with pytest.raises(ValidationError):
        pack(pack_version=version)
    with pytest.raises(ValidationError):
        pack(coverage_version=version)


def test_interview_versions_accept_integral_float_one():
    value = pack(schema_version=1.0, pack_version=1.0, coverage_version=1.0)
    assert value.schema_version == value.pack_version == value.coverage_version == 1


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


CONTROLS = ProfileControls(
    sync_mode=SyncMode.SYNCABLE,
    agent_visibility=AgentVisibility.AGENT_VISIBLE,
)
PREFERENCE = PreferencePayload(subject="format", polarity="like", value="brief")
PREFERENCE_KEY = SemanticKey(namespace="preference", subject="format")


def proposed_change(operation, **changes):
    shapes = {
        ProposalOperation.CREATE: {
            "proposed_payload": PREFERENCE,
            "controls": CONTROLS,
            "semantic_key": PREFERENCE_KEY,
        },
        ProposalOperation.UPDATE: {
            "target_record_id": "record-1",
            "base_version_id": "version-1",
            "proposed_payload": PREFERENCE,
            "controls": CONTROLS,
            "semantic_key": PREFERENCE_KEY,
        },
        ProposalOperation.ARCHIVE: {
            "target_record_id": "record-1",
            "base_version_id": "version-1",
        },
        ProposalOperation.PROMOTE: {
            "target_record_id": "record-1",
            "base_version_id": "version-1",
        },
    }
    values = shapes[operation] | changes
    return InterviewProposedChange(operation=operation, **values)


@pytest.mark.parametrize("operation", list(ProposalOperation))
def test_proposed_change_accepts_all_exact_operation_shapes(operation):
    assert proposed_change(operation).operation is operation


@pytest.mark.parametrize(
    ("operation", "changes"),
    [
        (ProposalOperation.CREATE, {"target_record_id": "record-1"}),
        (ProposalOperation.UPDATE, {"base_version_id": None}),
        (ProposalOperation.ARCHIVE, {"proposed_payload": PREFERENCE}),
        (ProposalOperation.PROMOTE, {"controls": CONTROLS}),
    ],
)
def test_proposed_change_rejects_invalid_operation_shapes(operation, changes):
    with pytest.raises(ValidationError):
        proposed_change(operation, **changes)


def test_create_and_update_semantic_keys_follow_payload_kind():
    with pytest.raises(ValidationError):
        proposed_change(ProposalOperation.CREATE, semantic_key=None)
    with pytest.raises(ValidationError):
        proposed_change(ProposalOperation.UPDATE, semantic_key=None)

    legacy = LegacyUnclassifiedPayload(text="migrated text")
    assert (
        proposed_change(
            ProposalOperation.CREATE,
            proposed_payload=legacy,
            semantic_key=None,
        ).semantic_key
        is None
    )
    with pytest.raises(ValidationError):
        proposed_change(
            ProposalOperation.CREATE,
            proposed_payload=legacy,
            semantic_key=PREFERENCE_KEY,
        )


def test_proposed_changes_reject_secret_material_in_payloads():
    with pytest.raises(ValidationError):
        proposed_change(
            ProposalOperation.CREATE,
            proposed_payload=PreferencePayload(
                subject="credential",
                polarity="like",
                value="password: hunter2-secret",
            ),
        )


def test_batch_contains_only_metadata_and_at_most_twenty_changes():
    value = InterviewProposalBatch(
        pack_id="personal-v1",
        pack_version=1,
        audience="personal",
        changes=(proposed_change(ProposalOperation.CREATE),),
    )
    assert isinstance(value.changes, tuple)
    assert set(value.model_dump()) == {
        "pack_id",
        "pack_version",
        "audience",
        "changes",
    }
    with pytest.raises(ValidationError):
        InterviewProposalBatch(
            pack_id="personal-v1",
            pack_version=1,
            audience="personal",
            changes=(),
            turns=(),
        )
    with pytest.raises(ValidationError):
        InterviewProposalBatch(
            pack_id="personal-v1",
            pack_version=1,
            audience="personal",
            changes=tuple(
                proposed_change(ProposalOperation.ARCHIVE) for _ in range(21)
            ),
        )


def test_proposed_changes_and_batches_are_frozen():
    change = proposed_change(ProposalOperation.CREATE)
    batch = InterviewProposalBatch(
        pack_id="personal-v1",
        pack_version=1,
        audience="personal",
        changes=(change,),
    )
    with pytest.raises(ValidationError):
        change.semantic_key = None
    with pytest.raises(ValidationError):
        batch.changes = ()


@pytest.mark.parametrize(
    "text",
    [
        "What helps; why does it help?",
        "What helps? Why does it help?",
        "What helps and why?",
        "What helps and how should I apply it?",
        "State your preferred name and give your work history?",
        "What is your preferred name and your work role?",
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
    "text",
    [
        "What is your password?",
        "What's your password?",
        "What’s your password?",
        "Enter your password",
        "May I have your API key?",
        "Provide your API key",
        "Tell me your access token",
    ],
)
def test_rejects_questions_that_solicit_secret_material(text):
    with pytest.raises(ValidationError):
        question(text=text)


@pytest.mark.parametrize(
    "secret",
    [
        "My password is hunter2-secret",
        "Use API key sk-abcdefghijklmnopqrstuvwxyz123456",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
        "token: abcdefghijklmnopqrstuvwxyz",
        "token: 123456-abcdef",
        "credential: hunter2-secret",
        "-----BEGIN PRIVATE KEY-----",
    ],
)
def test_rejects_recognizable_secrets_at_question_and_answer_boundaries(secret):
    with pytest.raises(ValidationError):
        question(text=f"What should I remember: {secret}?")
    with pytest.raises(ValidationError):
        InterviewTurn(question_id="q1", answer=secret)


@pytest.mark.parametrize(
    "benign",
    [
        "The token is limited to 4096 characters",
        "token: 128000",
        "token = 128000",
    ],
)
def test_allows_benign_token_limit_wording_in_interview_answers(benign):
    assert InterviewTurn(question_id="q1", answer=benign).answer == benign
