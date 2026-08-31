"""Repair model tool arguments that arrive as JSON strings.

TASK-26005. `json.loads(call.arguments)` produced whatever the model emitted and
handed it straight to the provider. A common small-model failure is emitting a
stringified array where the schema declares an array -- `"[\\"a\\",\\"b\\"]"`
instead of `["a","b"]` -- which fails validation and costs a whole turn.

Coercion is schema-driven on purpose. Guessing from the value alone would
corrupt a legitimately string-typed field that happens to contain brackets, so
a string stays a string wherever the schema says string.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.tool_arg_coercion import coerce_tool_args

ARRAY_SCHEMA = {"type": "object", "properties": {"items": {"type": "array"}}}
STRING_SCHEMA = {"type": "object", "properties": {"note": {"type": "string"}}}
OBJECT_SCHEMA = {"type": "object", "properties": {"opts": {"type": "object"}}}


def test_stringified_array_is_coerced_where_the_schema_says_array():
    args, coerced = coerce_tool_args({"items": '["a", "b"]'}, ARRAY_SCHEMA)

    assert args == {"items": ["a", "b"]}
    assert coerced == ["items"]


def test_stringified_object_is_coerced_where_the_schema_says_object():
    args, coerced = coerce_tool_args({"opts": '{"deep": true}'}, OBJECT_SCHEMA)

    assert args == {"opts": {"deep": True}}
    assert coerced == ["opts"]


def test_a_string_typed_field_is_never_touched():
    """AC#2/#6: the bracket case. This is real prose, not a malformed array."""
    args, coerced = coerce_tool_args({"note": '["not", "an", "array"]'}, STRING_SCHEMA)

    assert args == {"note": '["not", "an", "array"]'}
    assert coerced == []


def test_untyped_field_is_left_alone():
    """No schema type means no basis to coerce; guessing would corrupt data."""
    args, coerced = coerce_tool_args(
        {"whatever": '["a"]'}, {"type": "object", "properties": {"whatever": {}}}
    )

    assert args == {"whatever": '["a"]'}
    assert coerced == []


def test_coercion_recurses_into_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "array"}},
            }
        },
    }

    args, coerced = coerce_tool_args({"outer": {"inner": '[1, 2]'}}, schema)

    assert args == {"outer": {"inner": [1, 2]}}
    assert coerced == ["outer.inner"]


def test_coercion_recurses_into_array_items():
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"tags": {"type": "array"}},
                },
            }
        },
    }

    args, coerced = coerce_tool_args({"rows": [{"tags": '["x"]'}]}, schema)

    assert args == {"rows": [{"tags": ["x"]}]}
    assert coerced == ["rows[0].tags"]


def test_double_encoded_json_is_unwrapped():
    """AC#6: the model encoded the argument twice."""
    args, coerced = coerce_tool_args({"items": '"[1, 2]"'}, ARRAY_SCHEMA)

    assert args == {"items": [1, 2]}
    assert coerced == ["items"]


def test_fenced_json_is_unwrapped():
    """AC#6: the model wrapped the value in a markdown code fence."""
    args, coerced = coerce_tool_args(
        {"items": '```json\n["a"]\n```'}, ARRAY_SCHEMA
    )

    assert args == {"items": ["a"]}
    assert coerced == ["items"]


@pytest.mark.parametrize(
    "value", ["definitely not json", "[1, 2", "", "   ", "{oops}"]
)
def test_uncoercible_value_is_left_for_normal_validation(value):
    """AC#4: never silently dispatch a wrong-typed value in its place."""
    args, coerced = coerce_tool_args({"items": value}, ARRAY_SCHEMA)

    assert args == {"items": value}
    assert coerced == []


def test_a_json_string_decoding_to_the_wrong_type_is_not_substituted():
    """`"42"` under an array schema decodes fine but is still not an array."""
    args, coerced = coerce_tool_args({"items": "42"}, ARRAY_SCHEMA)

    assert args == {"items": "42"}
    assert coerced == []


def test_already_correct_arguments_are_returned_unchanged():
    original = {"items": ["a", "b"]}

    args, coerced = coerce_tool_args(original, ARRAY_SCHEMA)

    assert args == original
    assert coerced == []


def test_the_input_mapping_is_not_mutated():
    original = {"items": '["a"]'}

    coerce_tool_args(original, ARRAY_SCHEMA)

    assert original == {"items": '["a"]'}, "coercion must not mutate its input"


def test_missing_or_malformed_schema_is_survivable():
    for schema in ({}, None, {"type": "object"}, {"properties": "not-a-dict"}):
        args, coerced = coerce_tool_args({"items": '["a"]'}, schema)
        assert args == {"items": '["a"]'}
        assert coerced == []


# --- the choke point, not just the pure function ----------------------------


def test_registry_repairs_arguments_before_the_provider_sees_them(caplog):
    """The unit tests above prove the function; this proves it is wired in.

    Goes through `invoke_by_name`, the one line every provider is reached
    through, so a provider added later inherits the repair without opting in.
    """
    import logging

    from tldw_chatbook.Agents.agent_models import (
        ToolCatalogEntry,
        ToolResult,
        ToolSchema,
    )
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    received: dict = {}

    class _Provider:
        source = "test"

        def list_catalog(self):
            return [
                ToolCatalogEntry(
                    id="test:echo",
                    name="echo",
                    one_line_description="d",
                    source="test",
                )
            ]

        def load_schema(self, tool_id):
            return ToolSchema(
                id=tool_id,
                name="echo",
                description="d",
                parameters={
                    "type": "object",
                    "properties": {"items": {"type": "array"}},
                },
            )

        def invoke(self, tool_id, args):
            received.update(args)
            return ToolResult(ok=True, content="ok")

    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider())

    with caplog.at_level(logging.WARNING):
        result = registry.invoke_by_name("echo", {"items": '["a", "b"]'})

    assert result.ok is True
    assert received["items"] == ["a", "b"], "provider saw the raw string"
