# tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py
"""JSON-schema-driven parameter form for one MCP tool call, with an honest
raw-JSON fallback.

`parse_schema()` is pure: it turns a JSON-Schema "object" schema into a list
of `SchemaField`s the widget can render as real controls. Simple/enum types
render directly, arrays of simple items render as one comma-separated
`Input`, and the Optional[T] idiom (Pydantic v2's `anyOf: [T, null]` /
`type: [T, "null"]`) is unwrapped to the underlying type -- including
`Optional[List[T]]`. If ANY declared property can't be rendered faithfully
(a nested object, an array of non-simple items, a real multi-type union,
`oneOf`, a missing/unsupported `type`), the WHOLE parse fails (returns
`None`) rather than silently dropping that property -- a form missing a
parameter the tool actually requires would lie to the user. `MCPSchemaForm`
falls back to a raw JSON `TextArea` in that case, so every tool call remains
possible even when its schema can't be rendered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Checkbox, Input, Select, Static, TextArea

_SIMPLE_KINDS = ("string", "number", "integer", "boolean")

_RAW_MODE_NOTE = "This tool's parameters can't be rendered as a form — edit raw JSON."


@dataclass(frozen=True)
class SchemaField:
    """One renderable form field derived from a JSON-Schema property."""

    name: str
    kind: str  # string|number|integer|boolean|enum|array
    required: bool
    description: str
    default: object | None
    choices: tuple[str, ...] = field(default_factory=tuple)
    nullable: bool = False  # accepts null (Pydantic Optional[T]) — blank sends null
    item_kind: str | None = None  # kind == "array" only: the item type for casting


def _resolve_array_item_kind(items_spec: object) -> str | None:
    """Resolve an `array` property's `items` sub-schema to a simple item kind.

    Reuses `_resolve_property` on `items_spec` so the item gets the same
    enum/Optional handling as a top-level property, then rejects anything
    that isn't one of `_SIMPLE_KINDS` -- an item that is itself an enum,
    nested array, or object can't be represented by a single comma-split
    `Input`, so the array (and therefore the whole schema) must fall back
    to raw JSON rather than lying about what it can render.
    """
    if not isinstance(items_spec, dict):
        return None
    resolved = _resolve_property(items_spec)
    if resolved is None:
        return None
    item_kind = resolved[0]
    if item_kind not in _SIMPLE_KINDS:
        return None
    return item_kind


def _resolve_property(spec: dict) -> tuple[str, tuple[str, ...], bool, str | None] | None:
    """Resolve a JSON-Schema property to `(kind, enum_choices, nullable, item_kind)`.

    Renders the simple/enum/array-of-simple types and unwraps the
    Optional[T] idiom that Pydantic v2 (the most common third-party MCP
    server framework) emits: `anyOf: [{...}, {"type": "null"}]` and
    `type: [T, "null"]` become an optional field of the underlying type.
    `nullable` is True when `null` was one of the allowed options -- the
    caller marks such fields not-required (blank == the accepted null).
    `item_kind` is only set when `kind == "array"` -- the item type used to
    cast comma-split values back on collection.

    Returns `None` for genuinely unrenderable shapes -- nested object, an
    array of non-simple items (enum/array/object), a real multi-type union
    (e.g. `string|integer`), `oneOf`, or a missing type -- preserving the
    honest raw-JSON fallback (a partial form that silently drops a
    parameter the tool needs would lie to the user).
    """
    enum_values = spec.get("enum")
    if isinstance(enum_values, list) and enum_values:
        # An enum listing null/None among its values is nullable; drop those
        # so they don't render as literal "null"/"None" choices.
        nullable = None in enum_values or "null" in enum_values
        choices = tuple(
            str(value)
            for value in enum_values
            if value is not None and value != "null"
        )
        return ("enum", choices, nullable, None)

    prop_type = spec.get("type")
    if prop_type in _SIMPLE_KINDS:
        return (str(prop_type), (), False, None)
    if prop_type == "array":
        item_kind = _resolve_array_item_kind(spec.get("items"))
        if item_kind is None:
            return None
        return ("array", (), False, item_kind)
    if isinstance(prop_type, list):
        nullable = "null" in prop_type
        non_null = [t for t in prop_type if t != "null"]
        if len(non_null) == 1:
            only = non_null[0]
            if only in _SIMPLE_KINDS:
                return (str(only), (), nullable, None)
            if only == "array":
                item_kind = _resolve_array_item_kind(spec.get("items"))
                if item_kind is None:
                    return None
                return ("array", (), nullable, item_kind)
        return None

    branches = spec.get("anyOf")
    if isinstance(branches, list):
        subs = [b for b in branches if isinstance(b, dict)]
        nullable = any(b.get("type") == "null" for b in subs)
        non_null = [b for b in subs if b.get("type") != "null"]
        if len(non_null) == 1:
            inner = _resolve_property(non_null[0])
            if inner is None:
                return None
            kind, choices, inner_nullable, item_kind = inner
            return (kind, choices, nullable or inner_nullable, item_kind)
        return None

    return None


def parse_schema(schema: dict | None) -> list[SchemaField] | None:
    """Turn a JSON-Schema `object` schema into a list of `SchemaField`s.

    PURE -- no widget/Textual dependency.

    Renders simple/enum/array-of-simple-items properties and the Optional[T]
    idiom (Pydantic v2's `anyOf: [T, null]` / `type: [T, "null"]`, including
    `Optional[List[T]]`) via `_resolve_property`.

    Args:
        schema: A JSON-Schema dict for one tool's parameters (a top-level
            `"object"` schema with a `properties` mapping), or `None`.

    Returns:
        `None` when `schema` is falsy/not a dict, its `type` isn't
        `"object"`, `properties` isn't a mapping, or ANY property is
        unrenderable (nested object, array of non-simple items, a real
        multi-type union, `oneOf`, missing/unsupported `type`) -- the
        raw-JSON-fallback trigger. Otherwise the parsed fields, in
        `properties` iteration order.
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

        resolved = _resolve_property(spec)
        if resolved is None:
            # Nested object, array-of-non-simple-items, real multi-type
            # union, oneOf, missing type, etc. -- one unrenderable property
            # fails the WHOLE schema.
            return None
        kind, choices, nullable, item_kind = resolved
        # The schema's `required` list is authoritative: a nullable field
        # WITH a default isn't in it (renders optional), but `T | None` with
        # NO default IS in it and the tool wants it. Rather than silently
        # dropping such a param OR forcing a non-null value, `collect_arguments`
        # sends explicit JSON null when a nullable field is left blank.
        required = field_name in required_names
        fields.append(
            SchemaField(
                name=field_name,
                kind=kind,
                required=required,
                description=description,
                default=default,
                choices=choices,
                nullable=nullable,
                item_kind=item_kind,
            )
        )

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
            yield Static(
                _RAW_MODE_NOTE,
                id="mcp-schema-raw-note",
                classes="ds-field-row",
                markup=False,
            )
            yield TextArea("{}", id="mcp-schema-raw")
            return

        for index, schema_field in enumerate(self._fields):
            label = schema_field.name + (" *" if schema_field.required else "")
            yield Static(label, classes="form-label", markup=False)
            widget_id = f"mcp-schema-field-{index}"

            if schema_field.kind == "boolean":
                default_value = (
                    bool(schema_field.default)
                    if schema_field.default is not None
                    else False
                )
                yield Checkbox(
                    schema_field.description or schema_field.name,
                    value=default_value,
                    id=widget_id,
                )
            elif schema_field.kind == "enum":
                options = [(choice, choice) for choice in schema_field.choices]
                default_str = (
                    str(schema_field.default)
                    if schema_field.default is not None
                    else None
                )
                if default_str is not None and default_str in schema_field.choices:
                    yield Select(options, id=widget_id, value=default_str)
                else:
                    yield Select(options, id=widget_id)
            elif schema_field.kind == "array":
                default_items = (
                    schema_field.default
                    if isinstance(schema_field.default, list)
                    else None
                )
                default_str = (
                    ", ".join(str(item) for item in default_items)
                    if default_items
                    else ""
                )
                yield Input(
                    value=default_str, placeholder="comma-separated", id=widget_id
                )
            else:
                # string/number/integer -> Input. number/integer get a
                # type hint via the placeholder (not Input's native
                # `type=` validator -- collect_arguments() does its own
                # coercion, so a restrictive native validator would only
                # get in the way of typing intermediate values like "-").
                placeholder = (
                    schema_field.kind
                    if schema_field.kind in ("number", "integer")
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
            float, integer -> int, boolean -> the Checkbox's value, array ->
            the comma-split `Input` text with each item cast per the
            array's item kind); a blank nullable field (Pydantic
            Optional[T]) sends explicit JSON null; empty non-nullable
            optional strings/numbers and unselected optional enums are
            omitted entirely. Arrays have no null-sending idiom: a blank
            array `Input` sends `[]` when required, or is omitted when not.

        Raises:
            ValueError: Raw mode -- invalid JSON, or valid JSON that isn't
                a JSON object. Form mode -- a required field with no value
                (`"<field>: required."`), a number field whose text can't
                be coerced (`"<field>: must be a number."`), an integer
                field whose text can't be coerced
                (`"<field>: must be a whole number."`), or an array item
                whose text can't be coerced per the array's item kind
                (`"<field>: items must be numbers."` etc.).
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
                    if schema_field.nullable:
                        result[schema_field.name] = None
                    elif schema_field.required:
                        raise ValueError(f"{schema_field.name}: required.")
                    continue
                result[schema_field.name] = value
                continue
            if schema_field.kind == "array":
                raw_value = self.query_one(widget_id, Input).value
                items = [part.strip() for part in raw_value.split(",")]
                items = [part for part in items if part]
                if not items:
                    # No null-sending idiom for arrays (unlike scalars): an
                    # empty comma-list is either "no items" (required -> [])
                    # or "not provided" (optional -> omitted).
                    if schema_field.required:
                        result[schema_field.name] = []
                    continue
                result[schema_field.name] = [
                    self._cast_array_item(schema_field.name, part, schema_field.item_kind)
                    for part in items
                ]
                continue

            raw_value = self.query_one(widget_id, Input).value
            text_value = raw_value.strip()
            if not text_value:
                if schema_field.nullable:
                    result[schema_field.name] = None
                elif schema_field.required:
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
                    raise ValueError(f"{schema_field.name}: must be a whole number.")
            else:
                result[schema_field.name] = raw_value
        return result

    @staticmethod
    def _cast_array_item(field_name: str, text: str, item_kind: str | None) -> Any:
        """Cast one comma-split array item per `item_kind`.

        Mirrors the scalar coercion rules directly above (`number` ->
        float, `integer` -> int) with the same error-message wording,
        pluralized since the failure is about one item among several.
        `boolean` items have no scalar-Input precedent in this file (a
        top-level boolean renders as a Checkbox, never free text) so this
        defines the convention for array-of-bool: case-insensitive
        true/false/1/0/yes/no.
        """
        if item_kind == "number":
            try:
                return float(text)
            except ValueError:
                raise ValueError(f"{field_name}: items must be numbers.")
        if item_kind == "integer":
            try:
                return int(text)
            except ValueError:
                raise ValueError(f"{field_name}: items must be whole numbers.")
        if item_kind == "boolean":
            lowered = text.lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
            raise ValueError(f"{field_name}: items must be true/false values.")
        return text

    def _collect_raw(self) -> dict:
        raw_text = self.query_one("#mcp-schema-raw", TextArea).text
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Not valid JSON: expected a JSON object.")
        return data
