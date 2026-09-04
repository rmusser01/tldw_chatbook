"""PRD Feature A1/A2/A6: bounds, validation, and result shapes for ask_user."""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.ask_user_questions import (
    ASK_USER_DESCRIPTION,
    ASK_USER_PARAMETERS,
    MAX_OTHER_TEXT_CHARS,
    AskUserValidationError,
    answered_result,
    busy_result,
    clean_other_text,
    empty_answers,
    unanswered_result,
    validate_answers,
    validate_questions,
)


def _q(**overrides):
    base = {
        "question": "Which database?",
        "header": "Database",
        "multiSelect": False,
        "options": [
            {"label": "Postgres", "description": "Managed, relational"},
            {"label": "SQLite", "description": "Embedded"},
        ],
    }
    base.update(overrides)
    return base


def test_valid_call_round_trips_cleaned_copies():
    source = [_q()]
    out = validate_questions(source)
    assert out == [_q()]
    assert out[0] is not source[0], "defensive copy, not the caller's dict"


def test_multiselect_defaults_false_and_description_optional():
    out = validate_questions(
        [_q(multiSelect=True, options=[{"label": "a"}, {"label": "b"}])]
    )
    assert out[0]["multiSelect"] is True
    assert out[0]["options"][0] == {"label": "a", "description": ""}
    out = validate_questions(
        [{"question": "q", "header": "h", "options": [{"label": "a"}, {"label": "b"}]}]
    )
    assert out[0]["multiSelect"] is False


@pytest.mark.parametrize(
    "raw, fragment",
    [
        ([], "at least 1 item"),
        ([_q()] * 5, "at most 4 items"),
        ([_q(options=[{"label": "only"}])], "question 1 options: List should have at least 2 items"),
        ([_q(header="thirteen chars")], "question 1 header: String should have at most 12 characters"),
        ([_q(question="x" * 501)], "at most 500 characters"),
        ([_q(question="   ")], "question 1 question: must not be blank"),
        ([_q(multiSelect="yes")], "question 1 multiSelect: Input should be a valid boolean"),
        ([_q(bogus=1)], "question 1 bogus: Extra inputs are not permitted"),
        ([_q(options=[{"label": "a"}, {"label": "A"}])], "repeats option label 'A'"),
        ([_q(options=[{"label": "a", "extra": 1}, {"label": "b"}])], "question 1 option 1 extra: Extra inputs are not permitted"),
        ("not a list", "valid list"),
        ([_q(question="\udcff bad")], "UTF-8"),
        ([_q(options=[{"label": 5}, {"label": "b"}])], "question 1 option 1 label: must be a string"),
    ],
)
def test_rejections_name_the_problem(raw, fragment):
    with pytest.raises(AskUserValidationError) as excinfo:
        validate_questions(raw)
    assert fragment in str(excinfo.value), str(excinfo.value)


def test_validation_is_pydantic_backed_and_uses_only_model_output():
    from tldw_chatbook.Agents.ask_user_questions import AskUserQuestion

    assert AskUserQuestion.model_config["extra"] == "forbid"
    assert AskUserQuestion.model_config["strict"] is True
    out = validate_questions([_q(options=[{"label": " a ", "description": "d\n"}, {"label": "b"}])])
    assert out[0]["options"] == [{"label": "a", "description": "d"}, {"label": "b", "description": ""}]


def test_control_characters_flatten_and_newlines_collapse():
    out = validate_questions([_q(question="line one\nline\ttwo\x07")])
    assert out[0]["question"] == "line one line two"


def test_schema_never_declares_other_and_pins_the_bounds():
    items = ASK_USER_PARAMETERS["properties"]["questions"]["items"]
    assert ASK_USER_PARAMETERS["properties"]["questions"]["maxItems"] == 4
    assert items["properties"]["options"]["minItems"] == 2
    assert items["properties"]["options"]["maxItems"] == 4
    assert "other" not in str(ASK_USER_PARAMETERS).lower()
    assert items["additionalProperties"] is False


def test_description_spends_its_words_on_restraint():
    text = ASK_USER_DESCRIPTION.lower()
    assert "do not ask" in text
    assert "conventional default" in text
    assert "busy" in text


def test_result_shapes():
    assert busy_result()["answered"] is False and busy_result()["reason"] == "busy"
    assert "instruction" in busy_result()
    assert unanswered_result("timeout") == {"answered": False, "reason": "timeout"}
    answers = empty_answers(validate_questions([_q()]))
    assert answers == [
        {"question": "Which database?", "selected": [], "other_text": None, "unanswered": True}
    ]
    assert answered_result(answers) == {"answered": True, "answers": answers}


# --- the answer boundary (Qodo #2379 findings 1 and 2) ---------------------


def test_clean_other_text_flattens_bounds_and_drops_blank_or_non_text():
    assert clean_other_text("  apac\nonly\x07 ") == "apac only"
    assert clean_other_text("x" * (MAX_OTHER_TEXT_CHARS + 50)) == "x" * MAX_OTHER_TEXT_CHARS
    assert clean_other_text("   ") is None
    assert clean_other_text(None) is None
    assert clean_other_text(42) is None


def test_validate_answers_accepts_the_card_shape_and_rejects_drift():
    good = [{"question": "q", "selected": ["a"], "other_text": None, "unanswered": False}]
    assert validate_answers(good) == good
    assert validate_answers([]) == []
    for bad, fragment in [
        ([{"question": "q", "selected": ["a"], "other_text": None, "unanswered": False, "extra": 1}], "Extra inputs"),
        ([{"question": "q", "selected": ["a"], "other_text": "x" * (MAX_OTHER_TEXT_CHARS + 1), "unanswered": False}], "at most 500 characters"),
        ([{"question": "q", "selected": "a", "other_text": None, "unanswered": False}], "valid list"),
        ("nope", "valid list"),
    ]:
        with pytest.raises(AskUserValidationError) as excinfo:
            validate_answers(bad)
        assert fragment in str(excinfo.value), str(excinfo.value)
