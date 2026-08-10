from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import given, settings, strategies as st

from tldw_chatbook.Prompt_Management.prompt_variables import (
    MAX_VARIABLE_NAME_LENGTH,
    MAX_VARIABLES,
    PromptVariableValidationError,
    compile_prompt_variables,
)


@pytest.mark.parametrize(
    ("source", "values", "expected_names", "expected_rendered"),
    [
        ("{name}", {"name": "X"}, ("name",), "X"),
        (
            "{Name}{name}",
            {"Name": "UPPER", "name": "lower"},
            ("Name", "name"),
            "UPPERlower",
        ),
        ("{name}{name}", {"name": "X"}, ("name",), "XX"),
        ("{{name}}", {}, (), "{name}"),
        ("{{{name}}}", {"name": "X"}, ("name",), "{X}"),
        ("{{name}", {}, (), "{name}"),
        ("{name}}", {"name": "X"}, ("name",), "X}"),
        ("{}", {}, (), "{}"),
        ("{1name}", {}, (), "{1name}"),
        ("{first-name}", {}, (), "{first-name}"),
        ("{ name }", {}, (), "{ name }"),
        ('{{"key": "{name}"}}', {"name": "X"}, ("name",), '{"key": "X"}'),
        ('{"key": "value"}', {}, (), '{"key": "value"}'),
        (
            "<customer>{name}</customer>",
            {"name": "X"},
            ("name",),
            "<customer>X</customer>",
        ),
        ("{outer {name}}", {"name": "X"}, ("name",), "{outer X}"),
        ("{name", {}, (), "{name"),
        ("name}", {}, (), "name}"),
    ],
)
def test_adr_053_grammar_truth_table(
    source: str,
    values: dict[str, str],
    expected_names: tuple[str, ...],
    expected_rendered: str,
) -> None:
    plan = compile_prompt_variables(user_text=source)

    assert tuple(variable.name for variable in plan.variables) == expected_names
    assert plan.issues == ()
    assert plan.render(values).user_text == expected_rendered


def test_variables_are_shared_in_system_then_user_first_occurrence_order() -> None:
    plan = compile_prompt_variables(
        system_text="{shared} {SystemOnly} {shared}",
        user_text="{user_only} {shared} {systemonly}",
    )

    assert tuple((variable.name, variable.lanes) for variable in plan.variables) == (
        ("shared", ("system", "user")),
        ("SystemOnly", ("system",)),
        ("user_only", ("user",)),
        ("systemonly", ("user",)),
    )
    rendered = plan.render(
        {
            "shared": "S",
            "SystemOnly": "A",
            "user_only": "B",
            "systemonly": "C",
        }
    )
    assert rendered.system_text == "S A S"
    assert rendered.user_text == "B S C"


def test_blank_variable_values_are_valid() -> None:
    plan = compile_prompt_variables(system_text="before{name}after")

    assert plan.render({"name": ""}).system_text == "beforeafter"


def test_braces_introduced_by_a_value_are_never_reparsed() -> None:
    plan = compile_prompt_variables(user_text="{name}")

    rendered = plan.render({"name": "{other} and {{literal}}"})

    assert rendered.user_text == "{other} and {{literal}}"


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            blacklist_characters=("\x00",),
        ),
        max_size=256,
    )
)
def test_extraction_and_rendering_are_deterministic_and_non_mutating(
    source: str,
) -> None:
    source_before = source[:]
    first = compile_prompt_variables(user_text=source)
    second = compile_prompt_variables(user_text=source)
    values = {
        variable.name: f"<{index}>" for index, variable in enumerate(first.variables)
    }
    values_before = deepcopy(values)

    assert first == second
    if not first.issues:
        first_rendered = first.render(values)
        assert first_rendered == second.render(values)
    assert source == source_before
    assert values == values_before


_VALID_NAME = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,31}", fullmatch=True)


@settings(max_examples=80, deadline=None, derandomize=True)
@given(_VALID_NAME)
def test_escape_decoding_cannot_reveal_an_inner_placeholder(name: str) -> None:
    plan = compile_prompt_variables(user_text="{{" + name + "}}")

    assert plan.variables == ()
    assert plan.render({}).user_text == "{" + name + "}"


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            blacklist_characters=("\x00",),
        ),
        max_size=256,
    )
)
def test_rendered_values_are_opaque(value: str) -> None:
    plan = compile_prompt_variables(user_text="before {slot} after")

    assert plan.render({"slot": value}).user_text == f"before {value} after"


def test_exactly_sixty_four_unique_variables_are_renderable() -> None:
    source = "".join(f"{{v{index}}}" for index in range(MAX_VARIABLES))
    values = {f"v{index}": str(index) for index in range(MAX_VARIABLES)}
    plan = compile_prompt_variables(user_text=source)

    assert len(plan.variables) == MAX_VARIABLES
    assert plan.issues == ()
    assert plan.render(values).user_text == "".join(
        str(index) for index in range(MAX_VARIABLES)
    )


def test_first_excess_unique_variable_is_an_explicit_validation_issue() -> None:
    allowed = "".join(f"{{v{index}}}" for index in range(MAX_VARIABLES))
    source = allowed + "{excess}{later}"
    plan = compile_prompt_variables(user_text=source)

    assert tuple(issue.code for issue in plan.issues) == ("too_many_variables",)
    assert plan.issues[0].lane == "user"
    assert plan.issues[0].position == len(allowed)
    with pytest.raises(PromptVariableValidationError, match="validation issues"):
        plan.render({variable.name: "x" for variable in plan.variables})


def test_sixty_four_character_name_is_renderable() -> None:
    name = "a" * MAX_VARIABLE_NAME_LENGTH
    plan = compile_prompt_variables(user_text="{" + name + "}")

    assert tuple(variable.name for variable in plan.variables) == (name,)
    assert plan.issues == ()
    assert plan.render({name: "X"}).user_text == "X"


def test_first_overlong_name_is_an_explicit_validation_issue() -> None:
    name = "a" * (MAX_VARIABLE_NAME_LENGTH + 1)
    plan = compile_prompt_variables(user_text="{" + name + "}{valid}")

    assert tuple(issue.code for issue in plan.issues) == ("name_too_long",)
    assert plan.issues[0].lane == "user"
    assert plan.issues[0].position == 0
    with pytest.raises(PromptVariableValidationError, match="validation issues"):
        plan.render({})


def test_render_requires_an_explicit_value_for_every_variable() -> None:
    plan = compile_prompt_variables(user_text="{required}")

    with pytest.raises(PromptVariableValidationError, match="missing values"):
        plan.render({})


def test_inactive_lanes_are_absent_from_the_rendered_result() -> None:
    plan = compile_prompt_variables(user_text="hello")

    rendered = plan.render({})

    assert rendered.system_text is None
    assert rendered.user_text == "hello"


def test_plan_and_rendered_lane_representations_hide_prompt_content() -> None:
    plan = compile_prompt_variables(
        system_text="private source {secret_name}",
        user_text="another private source",
    )
    rendered = plan.render({"secret_name": "private value"})

    assert "secret_name" not in repr(plan)
    assert "private source" not in repr(plan)
    assert "private value" not in repr(rendered)
    assert "another private source" not in repr(rendered)
