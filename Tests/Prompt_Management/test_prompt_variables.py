from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import logging
import time
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from tldw_chatbook.Prompt_Management.prompt_variables import (
    APPLICATION_TTL_SECONDS,
    MAX_VARIABLE_NAME_LENGTH,
    MAX_VARIABLES,
    PromptVariableApplication,
    PromptVariableValidationError,
    compile_prompt_variables,
    fingerprint_system_text,
)


_COMPOSER_FINGERPRINT = "a" * 64
_SYSTEM_FINGERPRINT = "sha256:" + "b" * 64


def _application(**overrides: Any) -> PromptVariableApplication:
    values: dict[str, Any] = {
        "system_text": "private rendered system",
        "user_text": "private rendered user",
        "apply_system": True,
        "apply_user": True,
        "destination": "replace_snapshot",
        "target_session_id": "session-1",
        "composer_fingerprint": _COMPOSER_FINGERPRINT,
        "system_fingerprint": _SYSTEM_FINGERPRINT,
        "created_monotonic": 10.0,
    }
    values.update(overrides)
    return PromptVariableApplication(**values)


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
        ('{"key": "{name}"}', {"name": "X"}, ("name",), '{"key": "X"}'),
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


def test_prompt_application_contains_only_the_guarded_final_payload() -> None:
    application = _application()

    assert tuple(item.name for item in fields(application)) == (
        "system_text",
        "user_text",
        "apply_system",
        "apply_user",
        "destination",
        "target_session_id",
        "composer_fingerprint",
        "system_fingerprint",
        "created_monotonic",
        "expires_monotonic",
    )
    assert application.system_text == "private rendered system"
    assert application.user_text == "private rendered user"
    assert not hasattr(application, "__dict__")
    for forbidden in (
        "values",
        "variable_values",
        "source_body",
        "system_source",
        "user_source",
        "to_dict",
        "serialize",
        "persist",
    ):
        assert not hasattr(application, forbidden)
    with pytest.raises(FrozenInstanceError):
        application.apply_user = False  # type: ignore[misc]


def test_prompt_application_hides_payloads_and_fingerprints_from_repr() -> None:
    application = _application()
    rendered = repr(application)

    assert "private rendered system" not in rendered
    assert "private rendered user" not in rendered
    assert _COMPOSER_FINGERPRINT not in rendered
    assert _SYSTEM_FINGERPRINT not in rendered


def test_blank_selected_final_payloads_are_valid() -> None:
    application = _application(system_text="", user_text="")

    assert application.system_text == ""
    assert application.user_text == ""


def test_system_only_replace_is_a_valid_application() -> None:
    application = _application(
        user_text=None,
        apply_user=False,
    )

    assert application.apply_system is True
    assert application.system_text == "private rendered system"
    assert application.apply_user is False
    assert application.user_text is None
    assert application.composer_fingerprint == _COMPOSER_FINGERPRINT


@pytest.mark.parametrize("field_name", ["apply_system", "apply_user"])
@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_prompt_application_lane_flags_require_true_booleans(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match="lane flags must be booleans"):
        _application(**{field_name: value})


def test_prompt_application_requires_at_least_one_active_lane() -> None:
    with pytest.raises(ValueError, match="at least one lane"):
        _application(
            system_text=None,
            user_text=None,
            apply_system=False,
            apply_user=False,
            system_fingerprint=None,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"system_text": None}, "System payload"),
        (
            {
                "system_text": "inactive body",
                "apply_system": False,
                "system_fingerprint": None,
            },
            "System payload",
        ),
        ({"user_text": None}, "User payload"),
        ({"user_text": "inactive body", "apply_user": False}, "User payload"),
    ],
)
def test_prompt_application_payload_presence_matches_lane_flags(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _application(**overrides)


@pytest.mark.parametrize("payload_field", ["system_text", "user_text"])
def test_prompt_application_active_payloads_must_be_text(
    payload_field: str,
) -> None:
    with pytest.raises(TypeError, match="payload must be text"):
        _application(**{payload_field: object()})


@pytest.mark.parametrize("destination", ["", "replace", "append", None, 1, []])
def test_prompt_application_rejects_unknown_destinations(destination: object) -> None:
    with pytest.raises(ValueError, match="destination"):
        _application(destination=destination)


def test_replace_requires_a_composer_fingerprint() -> None:
    with pytest.raises(ValueError, match="composer fingerprint"):
        _application(composer_fingerprint=None)


def test_append_forbids_a_composer_fingerprint() -> None:
    with pytest.raises(ValueError, match="composer fingerprint"):
        _application(
            destination="append_active",
            composer_fingerprint=_COMPOSER_FINGERPRINT,
        )


def test_append_without_a_composer_fingerprint_is_valid() -> None:
    application = _application(
        destination="append_active",
        composer_fingerprint=None,
    )

    assert application.destination == "append_active"
    assert application.composer_fingerprint is None


def test_system_fingerprint_is_required_exactly_when_system_applies() -> None:
    with pytest.raises(ValueError, match="System fingerprint"):
        _application(system_fingerprint=None)
    with pytest.raises(ValueError, match="System fingerprint"):
        _application(
            system_text=None,
            apply_system=False,
            system_fingerprint=_SYSTEM_FINGERPRINT,
        )


@pytest.mark.parametrize(
    "target_session_id",
    ["", "   ", " session-1", "session-1 ", None, 1],
)
def test_prompt_application_requires_a_target_session_id(
    target_session_id: object,
) -> None:
    with pytest.raises(ValueError, match="target session"):
        _application(target_session_id=target_session_id)


@pytest.mark.parametrize(
    "composer_fingerprint",
    ["", "a" * 63, "a" * 65, "A" * 64, "sha256:" + "a" * 64, None, 1],
)
def test_replace_requires_the_real_composer_fingerprint_shape(
    composer_fingerprint: object,
) -> None:
    with pytest.raises(ValueError, match="composer fingerprint"):
        _application(composer_fingerprint=composer_fingerprint)


@pytest.mark.parametrize(
    "system_fingerprint",
    ["", "b" * 64, "sha256:" + "b" * 63, "sha256:" + "B" * 64, None, 1],
)
def test_system_lane_requires_the_real_system_fingerprint_shape(
    system_fingerprint: object,
) -> None:
    with pytest.raises(ValueError, match="System fingerprint"):
        _application(system_fingerprint=system_fingerprint)


@pytest.mark.parametrize(
    "created_monotonic",
    [float("nan"), float("inf"), float("-inf"), True, "10"],
)
def test_prompt_application_requires_finite_monotonic_creation_time(
    created_monotonic: object,
) -> None:
    with pytest.raises(ValueError, match="creation time"):
        _application(created_monotonic=created_monotonic)


def test_prompt_application_derives_exact_expiry_and_expires_at_boundary() -> None:
    application = _application(created_monotonic=75.25)

    assert application.expires_monotonic == 75.25 + APPLICATION_TTL_SECONDS
    assert application.is_expired(now_monotonic=195.249999) is False
    assert application.is_expired(now_monotonic=195.25) is True
    assert application.is_expired(now_monotonic=196.0) is True


def test_prompt_application_uses_current_monotonic_time_by_default() -> None:
    before = time.monotonic()
    application = PromptVariableApplication(
        system_text=None,
        user_text="payload",
        apply_system=False,
        apply_user=True,
        destination="append_active",
        target_session_id="session-1",
        composer_fingerprint=None,
        system_fingerprint=None,
    )
    after = time.monotonic()

    assert before <= application.created_monotonic <= after


def test_prompt_application_expiry_uses_current_or_injected_caller_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _application(created_monotonic=10.0)
    monkeypatch.setattr(
        "tldw_chatbook.Prompt_Management.prompt_variables.time.monotonic",
        lambda: 130.0,
    )

    assert application.is_expired() is True
    assert application.is_expired(now_monotonic=129.999) is False
    with pytest.raises(ValueError, match="current monotonic time"):
        application.is_expired(now_monotonic=float("nan"))


def test_system_fingerprint_is_one_way_and_uses_the_real_shape() -> None:
    secret = "SYSTEM-BODY-SECRET-4d781"
    fingerprint = fingerprint_system_text(secret)

    assert fingerprint.startswith("sha256:")
    assert len(fingerprint) == 71
    assert secret not in fingerprint
    with pytest.raises(TypeError, match="System fingerprint input"):
        fingerprint_system_text(1)  # type: ignore[arg-type]


def test_application_validation_and_expiry_never_log_or_raise_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "PROMPT-APPLICATION-SECRET-99cb"
    caplog.set_level(logging.DEBUG)
    application = _application(system_text=secret, user_text=secret)

    assert application.is_expired(now_monotonic=application.expires_monotonic)
    with pytest.raises(ValueError) as caught:
        replace(application, destination="invalid")

    assert secret not in str(caught.value)
    assert secret not in caplog.text
