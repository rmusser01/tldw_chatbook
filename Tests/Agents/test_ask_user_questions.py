"""PRD Feature A1/A2/A6: bounds, validation, and result shapes for ask_user."""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.ask_user_questions import (
    ASK_USER_DESCRIPTION,
    ASK_USER_PARAMETERS,
    AskUserValidationError,
    answered_result,
    busy_result,
    empty_answers,
    unanswered_result,
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
        ([], "1 to 4"),
        ([_q()] * 5, "1 to 4"),
        ([_q(options=[{"label": "only"}])], "2 to 4"),
        ([_q(header="thirteen chars")], "12 characters"),
        ([_q(question="x" * 501)], "500 characters"),
        ([_q(question="   ")], "blank"),
        ([_q(multiSelect="yes")], "multiSelect"),
        ([_q(bogus=1)], "unknown keys"),
        ([_q(options=[{"label": "a"}, {"label": "A"}])], "repeats option label"),
        ([_q(options=[{"label": "a", "extra": 1}, {"label": "b"}])], "unknown keys"),
        ("not a list", "must be a list"),
        ([_q(question="\udcff bad")], "UTF-8"),
    ],
)
def test_rejections_name_the_problem(raw, fragment):
    with pytest.raises(AskUserValidationError) as excinfo:
        validate_questions(raw)
    assert fragment in str(excinfo.value)


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
