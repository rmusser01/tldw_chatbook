# Tests/UI/test_mcp_schema_form.py
"""Task 4: JSON-schema-driven parameter form, with an honest raw-JSON
fallback whenever any property can't be rendered as a real form control.

`parse_schema` is pure (dict in, `list[SchemaField] | None` out) and is
tested standalone first; the widget tests mirror the small `App` subclass
harness style used throughout `Tests/UI/test_mcp_workbench.py` (no app
stylesheet is loaded in this harness, so nothing here asserts on styling).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Checkbox, Input, Select, Static, TextArea

import tldw_chatbook

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
        name="name",
        kind="string",
        required=True,
        description="The widget's name",
        default=None,
    )
    assert by_name["count"] == SchemaField(
        name="count",
        kind="number",
        required=False,
        description="",
        default=1,
    )
    assert by_name["active"] == SchemaField(
        name="active",
        kind="boolean",
        required=False,
        description="",
        default=True,
    )
    assert by_name["mode"] == SchemaField(
        name="mode",
        kind="enum",
        required=True,
        description="Speed",
        default=None,
        choices=("fast", "slow"),
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


class SchemaFormApp(ConsolidatedCSSApp):
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


@pytest.mark.asyncio
async def test_collect_arguments_raises_for_required_enum_left_unselected():
    """Required enum property left unselected (Select.NULL) must raise
    ValueError with exact message."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {"enum": ["fast", "slow"], "description": "Speed mode"},
        },
        "required": ["mode"],
    }
    app = SchemaFormApp(schema=schema)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)

        # mode Select is unselected (Select.NULL by default since default
        # is not in the enum choices).
        select_widget = app.query_one("#mcp-schema-field-0", Select)
        assert select_widget.value is Select.NULL

        with pytest.raises(ValueError) as exc_info:
            form.collect_arguments()
        assert str(exc_info.value) == "mode: required."


@pytest.mark.asyncio
async def test_collect_arguments_raises_for_integer_with_decimal_string():
    """Integer field given a decimal string (e.g., "3.5") must raise
    ValueError with exact message."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "description": "Count value"},
        },
        "required": ["count"],
    }
    app = SchemaFormApp(schema=schema)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)

        app.query_one("#mcp-schema-field-0", Input).value = "3.5"

        with pytest.raises(ValueError) as exc_info:
            form.collect_arguments()
        assert str(exc_info.value) == "count: must be a whole number."


def test_parse_schema_unwraps_pydantic_optional_anyof_null():
    """Pydantic v2 encodes Optional[T] as anyOf:[{type:T},{type:null}]; render
    it as an optional field of the underlying type rather than falling the whole
    schema back to raw JSON. (ATHF third-party UAT finding, 2026-07-21.)"""
    schema = {
        "type": "object",
        "properties": {
            "technique_id": {"type": "string"},
            "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
            "status": {"anyOf": [{"enum": ["open", "done"]}, {"type": "null"}]},
        },
    }
    fields = parse_schema(schema)
    assert fields is not None
    by_name = {f.name: f for f in fields}
    assert by_name["limit"].kind == "integer" and by_name["limit"].required is False
    assert by_name["status"].kind == "enum" and by_name["status"].choices == ("open", "done")


def test_parse_schema_unwraps_type_array_nullable():
    """The type:[T,"null"] spelling of Optional is unwrapped the same way."""
    schema = {
        "type": "object",
        "properties": {"note": {"type": ["string", "null"]}},
    }
    fields = parse_schema(schema)
    assert fields is not None and fields[0].kind == "string" and fields[0].required is False


def test_parse_schema_still_raw_for_genuine_multitype_union():
    """A real multi-type union (string|integer, no null) can't render as one
    field and must still trigger the honest raw fallback."""
    schema = {
        "type": "object",
        "properties": {"val": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    assert parse_schema(schema) is None


def test_parse_schema_required_nullable_field_stays_required():
    """A `T | None` param with no default lands in the schema's `required`
    list; unwrapping the nullable idiom must NOT silently make it optional,
    or the user could omit a parameter the tool requires."""
    schema = {
        "type": "object",
        "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["note"],
    }
    fields = parse_schema(schema)
    assert fields is not None
    assert fields[0].name == "note" and fields[0].kind == "string"
    assert fields[0].required is True


def test_parse_schema_type_array_multitype_union_stays_raw():
    """type:[string,integer] (no null) is a genuine union — still raw."""
    schema = {
        "type": "object",
        "properties": {"val": {"type": ["string", "integer"]}},
    }
    assert parse_schema(schema) is None


def test_parse_schema_enum_with_null_is_nullable_and_filters_null_choice():
    """An enum listing null among its values is nullable, and the null is
    dropped from the rendered choices (never a literal "null" option)."""
    schema = {
        "type": "object",
        "properties": {"mode": {"enum": ["fast", "slow", None]}},
    }
    fields = parse_schema(schema)
    assert fields is not None
    assert fields[0].kind == "enum" and fields[0].nullable is True
    assert fields[0].choices == ("fast", "slow")


# -- array support (RAG-48 part 3) -------------------------------------------

ARRAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["tags"],
}
OPTIONAL_ARRAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "media_types": {"type": ["array", "null"], "items": {"type": "string"}}
    },
    "required": [],
}
ANYOF_ARRAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "media_types": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ]
        }
    },
    "required": [],
}


def test_parse_schema_accepts_array_of_strings():
    fields = parse_schema(ARRAY_SCHEMA)
    assert fields is not None
    assert fields[0] == SchemaField(
        name="tags",
        kind="array",
        required=True,
        description="",
        default=None,
        item_kind="string",
    )


def test_parse_schema_accepts_optional_array_both_idioms():
    type_list_fields = parse_schema(OPTIONAL_ARRAY_SCHEMA)
    assert type_list_fields is not None
    assert type_list_fields[0].kind == "array"
    assert type_list_fields[0].required is False
    assert type_list_fields[0].nullable is True
    assert type_list_fields[0].item_kind == "string"

    anyof_fields = parse_schema(ANYOF_ARRAY_SCHEMA)
    assert anyof_fields is not None
    assert anyof_fields[0].kind == "array"
    assert anyof_fields[0].required is False
    assert anyof_fields[0].nullable is True
    assert anyof_fields[0].item_kind == "string"


def test_parse_schema_still_rejects_array_of_objects():
    nested = {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": {"type": "object"}}},
        "required": [],
    }
    assert parse_schema(nested) is None


def test_parse_schema_still_rejects_array_of_enum_items():
    """An item spec that is itself an enum can't be represented by a single
    comma-split Input either -- same honesty rule as array-of-objects."""
    nested = {
        "type": "object",
        "properties": {
            "modes": {"type": "array", "items": {"enum": ["fast", "slow"]}}
        },
        "required": [],
    }
    assert parse_schema(nested) is None


ARRAY_FORM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
        "media_types": {"type": ["array", "null"], "items": {"type": "string"}},
        "counts": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["tags"],
}


@pytest.mark.asyncio
async def test_widget_renders_array_field_as_comma_separated_input():
    app = SchemaFormApp(schema=ARRAY_FORM_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)
        assert form.is_raw_mode is False

        tags_input = app.query_one("#mcp-schema-field-0", Input)  # tags
        assert tags_input.placeholder == "comma-separated"
        media_types_input = app.query_one("#mcp-schema-field-1", Input)  # media_types
        assert media_types_input.placeholder == "comma-separated"

        labels = [str(s.renderable) for s in app.query(".form-label")]
        assert "tags *" in labels  # required
        assert "media_types" in labels
        assert "media_types *" not in labels  # optional


@pytest.mark.asyncio
async def test_collect_arguments_array_parses_comma_separated_values():
    app = SchemaFormApp(schema=ARRAY_FORM_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)
        app.query_one("#mcp-schema-field-0", Input).value = "a, b"
        result = form.collect_arguments()
        assert result["tags"] == ["a", "b"]


@pytest.mark.asyncio
async def test_collect_arguments_array_strips_whitespace_and_drops_empty_segments():
    app = SchemaFormApp(schema=ARRAY_FORM_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)
        app.query_one("#mcp-schema-field-0", Input).value = " a ,, b "
        result = form.collect_arguments()
        assert result["tags"] == ["a", "b"]


@pytest.mark.asyncio
async def test_collect_arguments_empty_optional_array_is_omitted():
    app = SchemaFormApp(schema=ARRAY_FORM_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)
        app.query_one("#mcp-schema-field-0", Input).value = "a"  # tags (required)
        # media_types (#1, optional+nullable) and counts (#2, optional) left blank.
        result = form.collect_arguments()
        assert "media_types" not in result
        assert "counts" not in result
        assert result["tags"] == ["a"]


@pytest.mark.asyncio
async def test_collect_arguments_empty_required_array_sends_empty_list():
    app = SchemaFormApp(schema=ARRAY_FORM_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)
        # tags (#0, required) left blank.
        result = form.collect_arguments()
        assert result["tags"] == []


@pytest.mark.asyncio
async def test_collect_arguments_array_casts_items_per_item_kind():
    app = SchemaFormApp(schema=ARRAY_FORM_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)
        app.query_one("#mcp-schema-field-0", Input).value = "a"
        app.query_one("#mcp-schema-field-2", Input).value = "1, 2, 3"
        result = form.collect_arguments()
        assert result["counts"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_collect_arguments_array_item_cast_failure_raises_value_error():
    app = SchemaFormApp(schema=ARRAY_FORM_SCHEMA)
    async with app.run_test() as pilot:
        await pilot.pause()
        form = app.query_one(MCPSchemaForm)
        app.query_one("#mcp-schema-field-0", Input).value = "a"
        app.query_one("#mcp-schema-field-2", Input).value = "1, not-a-number"
        with pytest.raises(ValueError):
            form.collect_arguments()


@pytest.mark.asyncio
async def test_collect_arguments_sends_null_for_blank_required_nullable_field():
    """A required-but-nullable field (Pydantic T|None, no default, in the
    schema's required list) left blank sends explicit JSON null instead of
    raising 'required.' — the honest way to satisfy a nullable requirement."""
    schema = {
        "type": "object",
        "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["note"],
    }
    app = SchemaFormApp(schema=schema)
    async with app.run_test():
        form = app.query_one(MCPSchemaForm)
        args = form.collect_arguments()
    assert args == {"note": None}


# -- Task 6 (PR-T3, task-2272 item 1): the boolean field must be READABLE ----
# The harness above loads no stylesheet, which is exactly why this shipped
# broken: under the production bundle, `css/features/_conversations.tcss`'s
# unscoped `Checkbox { width: 100%; height: 2; }` type selector fixes every
# checkbox app-wide at two rows -- both of which the widget's own border
# consumes, leaving ZERO content rows. `search_rag`'s `use_semantic` painted
# as an empty box: no toggle glyph, no label, state impossible to read.
# These tests mount with the real bundle so that rule is in play.

BOOLEAN_SCHEMA = {
    "type": "object",
    "properties": {
        "use_semantic": {
            "type": "boolean",
            "description": "Use semantic search",
            "default": True,
        },
    },
}


class StyledSchemaFormApp(ConsolidatedCSSApp):
    """Schema-form harness WITH the production stylesheet loaded."""

    CSS_PATH = str(
        Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
    )

    def compose(self) -> ComposeResult:
        yield MCPSchemaForm(schema=BOOLEAN_SCHEMA, id="mcp-inspector-test-form")


def _painted(checkbox: Checkbox) -> str:
    return "\n".join(
        checkbox.render_line(y).text for y in range(checkbox.size.height)
    )


@pytest.mark.asyncio
async def test_boolean_field_paints_glyph_and_label_under_the_app_stylesheet():
    app = StyledSchemaFormApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        checkbox = app.query_one(Checkbox)
        assert checkbox.size.height >= 1, (
            "the checkbox has no content rows at all -- a border-only husk"
        )
        painted = _painted(checkbox)
        assert "Use semantic search" in painted
        assert "X" in painted  # the toggle glyph itself


@pytest.mark.asyncio
async def test_boolean_field_state_is_readable_by_more_than_position():
    """On and off must differ in what is PAINTED, not only in the widget's
    internal `value` -- the reported symptom was a box whose state could not
    be read at all."""
    app = StyledSchemaFormApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        checkbox = app.query_one(Checkbox)

        def _glyph_style():
            for segment in checkbox.render_line(0):
                if "X" in segment.text:
                    return segment.style
            return None

        on_style = _glyph_style()
        checkbox.value = False
        await pilot.pause()
        off_style = _glyph_style()
        assert on_style is not None and off_style is not None
        assert on_style != off_style


@pytest.mark.asyncio
async def test_boolean_field_toggles_on_a_mouse_click():
    """Un-clickable was the other half of the report: only Tab+Space worked."""
    app = StyledSchemaFormApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        checkbox = app.query_one(Checkbox)
        before = checkbox.value
        await pilot.click(Checkbox)
        await pilot.pause()
        assert checkbox.value is not before


@pytest.mark.asyncio
async def test_boolean_field_keeps_its_content_rows_while_focused():
    """The `_evals.tcss` precedent: a focus-only decoration that grows or
    overpaints a fixed-height checkbox erases the very label it is meant to
    highlight, and shifts every sibling below it mid-click."""
    app = StyledSchemaFormApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        checkbox = app.query_one(Checkbox)
        unfocused_region = checkbox.region
        checkbox.focus()
        await pilot.pause()
        assert checkbox.region.height == unfocused_region.height
        assert "Use semantic search" in _painted(checkbox)
