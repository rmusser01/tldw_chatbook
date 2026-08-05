"""Tests for expression generation prompt composition."""

import pytest

from tldw_chatbook.Character_Chat.expression_generation import (
    EXPRESSION_PROMPT_STATES,
    STATE_MODIFIERS,
    compose_expression_prompt,
)


@pytest.mark.parametrize("state", EXPRESSION_PROMPT_STATES)
def test_each_state_modifier_lands_in_prompt(state):
    prompt, negative, params = compose_expression_prompt(
        name="Sayori",
        description="short coral pink hair, red bow",
        state=state,
    )
    assert STATE_MODIFIERS[state] in prompt
    assert "Sayori" in prompt and "coral pink hair" in prompt
    assert negative == "" and params == {}


def test_personality_included_only_when_nonempty():
    with_p, _, _ = compose_expression_prompt(
        name="A",
        description="desc",
        personality="cheerful",
        state="avatar",
    )
    without_p, _, _ = compose_expression_prompt(
        name="A",
        description="desc",
        personality="   ",
        state="avatar",
    )
    assert "cheerful" in with_p and "cheerful" not in without_p


def test_blank_name_omitted():
    prompt, _, _ = compose_expression_prompt(
        name="  ", description="desc", state="avatar"
    )
    assert not prompt.startswith(",") and "desc" in prompt


def test_empty_description_raises():
    with pytest.raises(ValueError):
        compose_expression_prompt(name="A", description="   ", state="avatar")


def test_unknown_state_raises():
    with pytest.raises(ValueError):
        compose_expression_prompt(name="A", description="desc", state="angry")


def test_style_template_composes_and_keeps_user_text():
    from tldw_chatbook.Media_Creation.generation_templates import get_template

    template = get_template("style_anime")
    prompt, negative, params = compose_expression_prompt(
        name="Sayori",
        description="coral pink hair",
        state="thinking",
        style_template=template,
    )
    assert "coral pink hair" in prompt  # user text survives (P2b invariant)
    assert STATE_MODIFIERS["thinking"] in prompt
    assert negative == template.negative_prompt
    assert params == template.default_params
