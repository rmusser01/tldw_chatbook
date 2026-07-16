# tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py
"""JSON-schema-driven parameter form for one MCP tool call, with an honest
raw-JSON fallback.

`parse_schema()` is pure: it turns a JSON-Schema "object" schema into a list
of `SchemaField`s the widget can render as real controls. If ANY declared
property can't be rendered faithfully (a nested object/array, `oneOf`, a
missing/unsupported `type`), the WHOLE parse fails (returns `None`) rather
than silently dropping that property -- a form missing a parameter the tool
actually requires would lie to the user. `MCPSchemaForm` falls back to a raw
JSON `TextArea` in that case, so every tool call remains possible even when
its schema can't be rendered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Checkbox, Input, Select, Static, TextArea

_SIMPLE_KINDS = ("string", "number", "integer", "boolean")

_RAW_MODE_NOTE = (
    "This tool's parameters can't be rendered as a form — edit raw JSON."
)


@dataclass(frozen=True)
class SchemaField:
    """One renderable form field derived from a JSON-Schema property."""

    name: str
    kind: str  # string|number|integer|boolean|enum
    required: bool
    description: str
    default: object | None
    choices: tuple[str, ...] = field(default_factory=tuple)


def parse_schema(schema: dict | None) -> list[SchemaField] | None:
    """Turn a JSON-Schema `object` schema into a list of `SchemaField`s.

    PURE -- no widget/Textual dependency.

    Returns:
        `None` when `schema` is falsy/not a dict, its `type` isn't
        `"object"`, `properties` isn't a mapping, or ANY property is
        unrenderable (nested object/array, `oneOf`/`anyOf`, missing/
        unsupported `type`) -- the raw-JSON-fallback trigger. Otherwise the
        parsed fields, in `properties` iteration order.
    """
    if not schema or not isinstance(schema, dict):
        return None
    if schema.get("type") != "object":
        return None
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return None

    required_raw = schema.get("required") or []
    if not isinstance(required_raw, list):
        return None
    required_names = {str(name) for name in required_raw}

    fields: list[SchemaField] = []
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            return None
        description = str(spec.get("description") or "")
        default = spec.get("default")
        field_name = str(name)
        required = field_name in required_names

        enum_values = spec.get("enum")
        if isinstance(enum_values, list) and enum_values:
            fields.append(SchemaField(
                name=field_name,
                kind="enum",
                required=required,
                description=description,
                default=default,
                choices=tuple(str(value) for value in enum_values),
            ))
            continue

        prop_type = spec.get("type")
        if prop_type in _SIMPLE_KINDS:
            fields.append(SchemaField(
                name=field_name,
                kind=str(prop_type),
                required=required,
                description=description,
                default=default,
            ))
            continue

        # Nested object/array, oneOf/anyOf, missing/unsupported type, etc.
        # -- one unrenderable property fails the WHOLE schema.
        return None

    return fields


class MCPSchemaForm(Vertical):
    """Renders `schema`'s properties as form controls, or a raw-JSON
    `TextArea` when `parse_schema()` can't render the schema faithfully.
    """

    DEFAULT_CSS = """
    MCPSchemaForm {
        height: auto;
        min-height: 0;
    }
    MCPSchemaForm .form-label {
        height: auto;
    }
    #mcp-schema-raw {
        height: 8;
        min-height: 4;
    }
    """

    def __init__(self, *, schema: dict | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._schema = schema
        self._fields: list[SchemaField] | None = parse_schema(schema)

    @property
    def is_raw_mode(self) -> bool:
        return self._fields is None

    def compose(self) -> ComposeResult:
        if self._fields is None:
            yield Static(_RAW_MODE_NOTE, id="mcp-schema-raw-note",
                         classes="ds-field-row", markup=False)
            yield TextArea("{}", id="mcp-schema-raw")
            return

        for index, schema_field in enumerate(self._fields):
            label = schema_field.name + (" *" if schema_field.required else "")
            yield Static(label, classes="form-label", markup=False)
            widget_id = f"mcp-schema-field-{index}"

            if schema_field.kind == "boolean":
                default_value = (
                    bool(schema_field.default) if schema_field.default is not None else False
                )
                yield Checkbox(
                    schema_field.description or schema_field.name,
                    value=default_value, id=widget_id,
                )
            elif schema_field.kind == "enum":
                options = [(choice, choice) for choice in schema_field.choices]
                default_str = (
                    str(schema_field.default) if schema_field.default is not None else None
                )
                if default_str is not None and default_str in schema_field.choices:
                    yield Select(options, id=widget_id, value=default_str)
                else:
                    yield Select(options, id=widget_id)
            else:
                # string/number/integer -> Input. number/integer get a
                # type hint via the placeholder (not Input's native
                # `type=` validator -- collect_arguments() does its own
                # coercion, so a restrictive native validator would only
                # get in the way of typing intermediate values like "-").
                placeholder = (
                    schema_field.kind if schema_field.kind in ("number", "integer")
                    else (schema_field.description or "")
                )
                default_str = (
                    "" if schema_field.default is None else str(schema_field.default)
                )
                yield Input(value=default_str, placeholder=placeholder, id=widget_id)

    def collect_arguments(self) -> dict:
        """Collect this form's current values into a tool-call argument dict.

        Returns:
            Raw mode: the parsed JSON object from `#mcp-schema-raw`.
            Form mode: one entry per field, coerced per kind (number ->
            float, integer -> int, boolean -> the Checkbox's value); empty
            optional strings/numbers and unselected optional enums are
            omitted entirely.

        Raises:
            ValueError: Raw mode -- invalid JSON, or valid JSON that isn't
                a JSON object. Form mode -- a required field with no value
                (`"<field>: required."`), or a number/integer field whose
                text can't be coerced (`"<field>: must be a number."`).
        """
        if self.is_raw_mode:
            return self._collect_raw()

        result: dict[str, Any] = {}
        assert self._fields is not None
        for index, schema_field in enumerate(self._fields):
            widget_id = f"#mcp-schema-field-{index}"
            if schema_field.kind == "boolean":
                result[schema_field.name] = self.query_one(widget_id, Checkbox).value
                continue
            if schema_field.kind == "enum":
                value = self.query_one(widget_id, Select).value
                if value is Select.NULL:
                    if schema_field.required:
                        raise ValueError(f"{schema_field.name}: required.")
                    continue
                result[schema_field.name] = value
                continue

            raw_value = self.query_one(widget_id, Input).value
            text_value = raw_value.strip()
            if not text_value:
                if schema_field.required:
                    raise ValueError(f"{schema_field.name}: required.")
                continue
            if schema_field.kind == "number":
                try:
                    result[schema_field.name] = float(text_value)
                except ValueError:
                    raise ValueError(f"{schema_field.name}: must be a number.")
            elif schema_field.kind == "integer":
                try:
                    result[schema_field.name] = int(text_value)
                except ValueError:
                    raise ValueError(f"{schema_field.name}: must be a number.")
            else:
                result[schema_field.name] = raw_value
        return result

    def _collect_raw(self) -> dict:
        raw_text = self.query_one("#mcp-schema-raw", TextArea).text
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Not valid JSON: expected a JSON object.")
        return data
