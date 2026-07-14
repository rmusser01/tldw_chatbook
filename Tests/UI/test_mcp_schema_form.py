# Tests/UI/test_mcp_schema_form.py
"""Task 4: JSON-schema-driven parameter form, with an honest raw-JSON
fallback whenever any property can't be rendered as a real form control.

`parse_schema` is pure (dict in, `list[SchemaField] | None` out) and is
tested standalone first; the widget tests mirror the small `App` subclass
harness style used throughout `Tests/UI/test_mcp_workbench.py` (no app
stylesheet is loaded in this harness, so nothing here asserts on styling).
"""
from __future__ import annotations

from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Input, Select, Static, TextArea

from tldw_chatbook.UI.MCP_Modules.mcp_schema_form import (
    MCPSchemaForm,
    SchemaField,
    parse_schema,
)

# -- parse_schema (pure) -----------------------------------------------------


def test_parse_schema_happy_path_string_number_bool_enum_required():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The widget's name"},
            "count": {"type": "number", "default": 1},
            "active": {"type": "boolean", "default": True},
            "mode": {"enum": ["fast", "slow"], "description": "Speed"},
        },
        "required": ["name", "mode"],
    }
    fields = parse_schema(schema)
    assert fields is not None
    by_name = {f.name: f for f in fields}

    assert by_name["name"] == SchemaField(
        name="name", kind="string", required=True,
        description="The widget's name", default=None,
    )
    assert by_name["count"] == SchemaField(
        name="count", kind="number", required=False,
        description="", default=1,
    )
    assert by_name["active"] == SchemaField(
        name="active", kind="boolean", required=False,
        description="", default=True,
    )
    assert by_name["mode"] == SchemaField(
        name="mode", kind="enum", required=True,
        description="Speed", default=None, choices=("fast", "slow"),
    )


def test_parse_schema_rejects_nested_object_property_entirely():
    """A single unrenderable property (nested object here) must fail the
    WHOLE schema, not just be skipped -- a partial form would lie about
    what the tool actually accepts."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "config": {"type": "object", "properties": {"nested": {"type": "string"}}},
        },
    }
    assert parse_schema(schema) is None


@pytest.mark.parametrize(
    "schema",
    [
        None,
        {},
        "not a dict",
        {"type": "array", "properties": {}},
        {"type": "object", "properties": {"items": {"type": "array"}}},
        {"type": "object", "properties": {"choice": {"oneOf": [{"type": "string"}]}}},
        {"type": "object", "properties": {"mystery": {}}},
    ],
)
def test_parse_schema_returns_none_for_unrenderable_or_missing_shapes(schema):
    assert parse_schema(schema) is None


# -- widget -------------------------------------------------------------------

FULL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "note": {"type": "string"},
        "ratio": {"type": "number"},
        "count": {"type": "integer", "default": 2},
        "active": {"type": "boolean", "default": True},
        "mode": {"enum": ["fast", "slow"], "default": "fast"},
        "level": {"enum": ["low", "high"], "default": "medium"},
    },
    "required": ["name"],
}

UNRENDERABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "config": {"type": "object", "properties": {}},
    },
}


class SchemaFormApp(App):
    def __init__(self, *, schema: dict | None) -> None:
        super().__init__()
        self._schema = schema

    def compose(self) -> ComposeResult:
        yield MCPSchemaForm(schema=self._schema, id="mcp-schema-form")


@pytest.mark.asyncio
async def test_widget_renders_input_checkbox_select_per_field_with_required_star():
    app = SchemaFormApp(schema=FULL_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)
        assert form.is_raw_mode is False

        # Field order mirrors dict insertion order in FULL_SCHEMA's properties.
        assert isinstance(app.query_one("#mcp-schema-field-0"), Input)  # name
        assert isinstance(app.query_one("#mcp-schema-field-1"), Input)  # note
        assert isinstance(app.query_one("#mcp-schema-field-2"), Input)  # ratio
        assert isinstance(app.query_one("#mcp-schema-field-3"), Input)  # count
        assert isinstance(app.query_one("#mcp-schema-field-4"), Checkbox)  # active
        assert isinstance(app.query_one("#mcp-schema-field-5"), Select)  # mode
        assert isinstance(app.query_one("#mcp-schema-field-6"), Select)  # level

        labels = [str(s.renderable) for s in app.query(".form-label")]
        assert "name *" in labels  # required field's label suffixed " *"
        assert "note" in labels
        assert "note *" not in labels

        # Enum default handling: "fast" is one of mode's choices -> becomes
        # the constructor value; "medium" is NOT one of level's choices ->
        # level is left unselected (Select.NULL).
        assert app.query_one("#mcp-schema-field-5", Select).value == "fast"
        assert app.query_one("#mcp-schema-field-6", Select).value is Select.NULL

        assert app.query_one("#mcp-schema-field-4", Checkbox).value is True


@pytest.mark.asyncio
async def test_collect_arguments_coerces_per_kind_and_omits_empty_optional():
    app = SchemaFormApp(schema=FULL_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)

        app.query_one("#mcp-schema-field-0", Input).value = "widget"  # name (required)
        # note (#1) left blank -- empty optional string must be omitted.
        app.query_one("#mcp-schema-field-2", Input).value = "3.5"  # ratio -> float
        # count (#3) left at its seeded default "2" -> coerced to int 2.
        # active (#4) left at its seeded default True.
        # mode (#5) left at its seeded default "fast".
        # level (#6) left unselected (Select.NULL) -> omitted.

        result = form.collect_arguments()
        assert result == {
            "name": "widget",
            "ratio": 3.5,
            "count": 2,
            "active": True,
            "mode": "fast",
        }


@pytest.mark.asyncio
async def test_collect_arguments_raises_for_bad_number_and_missing_required():
    app = SchemaFormApp(schema=FULL_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)

        # 'name' (index 0) is required and starts empty.
        with pytest.raises(ValueError) as exc_info:
            form.collect_arguments()
        assert str(exc_info.value) == "name: required."

        app.query_one("#mcp-schema-field-0", Input).value = "widget"
        app.query_one("#mcp-schema-field-2", Input).value = "not-a-number"
        with pytest.raises(ValueError) as exc_info:
            form.collect_arguments()
        assert str(exc_info.value) == "ratio: must be a number."


@pytest.mark.asyncio
async def test_raw_fallback_renders_textarea_and_note_for_unrenderable_schema():
    app = SchemaFormApp(schema=UNRENDERABLE_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)
        assert form.is_raw_mode is True

        textarea = app.query_one("#mcp-schema-raw", TextArea)
        assert textarea.text == "{}"

        note_text = " ".join(str(s.renderable) for s in app.query(Static))
        assert (
            "This tool's parameters can't be rendered as a form — edit raw JSON."
            in note_text
        )


@pytest.mark.asyncio
async def test_raw_mode_collect_arguments_parses_json_and_raises_on_invalid():
    app = SchemaFormApp(schema=UNRENDERABLE_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)

        app.query_one("#mcp-schema-raw", TextArea).text = '{"name": "widget"}'
        assert form.collect_arguments() == {"name": "widget"}

        app.query_one("#mcp-schema-raw", TextArea).text = "not json"
        with pytest.raises(ValueError, match="Not valid JSON"):
            form.collect_arguments()

        app.query_one("#mcp-schema-raw", TextArea).text = "[1, 2, 3]"
        with pytest.raises(ValueError):
            form.collect_arguments()
