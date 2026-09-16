"""Catch coercion, unsafe evaluation, lost detachment and missing references."""

import tracemalloc

import pytest

from tldw_chatbook.Workflows.expressions import ExpressionError, resolve_value


@pytest.mark.parametrize("value", [None, True, 7, 1.5, "s", ["a"], {"nested": 1}])
def test_pure_reference_preserves_json_type(value):
    result = resolve_value("{{ inputs.value }}", {"inputs": {"value": value}})
    assert result == value
    assert type(result) is type(value)


def test_mixed_text_renders_json_scalars_and_recursive_values():
    assert resolve_value(
        {"x": ["{{ inputs.a }}", "v={{ inputs.b }}; {{ inputs.c }}"]},
        {"inputs": {"a": [1, False], "b": True, "c": None}},
    ) == {"x": [[1, False], "v=true; null"]}


def test_resolved_values_are_detached_and_never_evaluated_twice():
    context = {"inputs": {"a": {"text": "{{ inputs.missing }}", "list": []}}}
    result = resolve_value("{{ inputs.a }}", context)
    result["list"].append(1)
    assert context["inputs"]["a"]["list"] == []
    assert result["text"] == "{{ inputs.missing }}"


@pytest.mark.parametrize(
    "template",
    [
        "{{ inputs.missing }}",
        "{{ inputs.a.upper() }}",
        "{{ inputs['a'] }}",
        "{{ inputs.a | safe }}",
        "{% for x in y %}",
        "{{ inputs.__class__ }}",
        "{{ inputs.a",
        "inputs.a }}",
        "{{ inputs.a + 1 }}",
        "{{ inputs }}",
    ],
)
def test_unsupported_expressions_fail_at_json_pointer(template):
    with pytest.raises(ExpressionError) as error:
        resolve_value({"a/b~": [template]}, {"inputs": {"a": "ok"}})
    assert error.value.pointer == "/a~1b~0/0"


@pytest.mark.parametrize("value", [[1], {"x": 1}])
def test_mixed_text_refuses_implicit_container_conversion(value):
    with pytest.raises(ExpressionError):
        resolve_value("value={{ inputs.a }}", {"inputs": {"a": value}})


@pytest.mark.parametrize("value", [object(), float("inf"), float("nan"), {1: "bad"}])
def test_python_objects_and_nonfinite_numbers_are_not_json(value):
    with pytest.raises(ExpressionError):
        resolve_value("{{ inputs.a }}", {"inputs": {"a": value}})


def test_repeated_references_are_bounded_before_concatenation():
    template = "{{ inputs.a }}" * 1024
    context = {"inputs": {"a": "x" * 16384}}
    tracemalloc.start()
    try:
        with pytest.raises(ExpressionError) as error:
            resolve_value(template, context)
        assert error.value.code == "byte_limit"
        assert tracemalloc.get_traced_memory()[1] < 4 * 1024 * 1024
    finally:
        tracemalloc.stop()


def test_expansion_bound_counts_json_escaping_and_aggregate_containers():
    # ["é","\""] is 11 UTF-8 JSON bytes, including escaping and punctuation.
    value = ["{{ inputs.a }}", "{{ inputs.b }}"]
    context = {"inputs": {"a": "é", "b": '"'}}
    assert resolve_value(value, context, byte_limit=11) == ["é", '"']
    with pytest.raises(ExpressionError) as error:
        resolve_value(value, context, byte_limit=10)
    assert error.value.pointer == "/1"


def test_pure_container_expansion_is_bounded_before_detached_copy():
    context = {"inputs": {"large": ["é" * 100] * 100}}
    with pytest.raises(ExpressionError):
        resolve_value("{{ inputs.large }}", context, byte_limit=100)
