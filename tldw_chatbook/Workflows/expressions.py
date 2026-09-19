"""Dotted dictionary references, never Python attributes or template execution."""

import json
import math
import re

IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
_EXPRESSION = re.compile(
    r"{{\s*([A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z][A-Za-z0-9_-]*)+)\s*}}"
)
_DELIMITERS = ("{{", "}}", "{%", "%}", "{#", "#}")
DEFAULT_EXPRESSION_BYTES = 10 * 1024 * 1024


class _Budget:
    def __init__(self, maximum: int) -> None:
        if type(maximum) is not int or not 0 < maximum <= 2**63 - 1:
            raise ExpressionError("", "byte_limit")
        self.remaining = maximum

    def consume(self, size: int, pointer: str) -> None:
        if size > self.remaining:
            raise ExpressionError(
                pointer, "byte_limit", "Resolved JSON exceeds the captured byte limit."
            )
        self.remaining -= size

    def string_content(self, text: str, pointer: str) -> None:
        # Bounded pieces avoid allocating a complete escaped/UTF-8 duplicate.
        for start in range(0, len(text), 2048):
            encoded = json.dumps(text[start : start + 2048], ensure_ascii=False)
            self.consume(len(encoded.encode("utf-8")) - 2, pointer)


class ExpressionError(ValueError):
    """A bounded, payload-free diagnostic with a JSON pointer and stable code."""

    def __init__(
        self,
        pointer: str,
        code: str = "expression",
        message: str = "Unsupported or unresolved JSON reference.",
    ) -> None:
        self.pointer = pointer[:512]
        self.code = code
        super().__init__(message)


def pointer_child(pointer: str, key: str | int) -> str:
    """Append an escaped JSON-pointer segment."""
    return pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")


def json_copy(
    value: object, pointer: str = "", *, byte_limit: int | None = None
) -> object:
    """Detach exact JSON runtime types; refuse objects, non-finite numbers/depth."""
    try:
        if byte_limit is not None:
            return _resolve(value, {}, pointer, _Budget(byte_limit), False)
        if value is None or type(value) in (bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if type(value) is str:
            value.encode("utf-8")
            return value
        if type(value) is list:
            return [
                json_copy(item, pointer_child(pointer, i))
                for i, item in enumerate(value)
            ]
        if type(value) is dict and all(type(key) is str for key in value):
            return {
                key: json_copy(item, pointer_child(pointer, key))
                for key, item in value.items()
            }
    except (RecursionError, UnicodeError):
        pass
    raise ExpressionError(
        pointer,
        "invalid_json",
        "Expected finite, UTF-8 JSON values with bounded nesting.",
    )


def reference_paths(text: str, pointer: str = "") -> tuple[tuple[str, ...], ...]:
    """Parse the admitted grammar, returning dictionary paths in source order."""
    remainder = _EXPRESSION.sub("", text)
    if any(delimiter in remainder for delimiter in _DELIMITERS):
        raise ExpressionError(pointer)
    return tuple(
        tuple(match.group(1).split(".")) for match in _EXPRESSION.finditer(text)
    )


def _resolve(
    value: object, context: dict, pointer: str, budget: _Budget, evaluate: bool = True
) -> object:
    if type(value) is dict:
        budget.consume(2 + max(0, len(value) - 1), pointer)
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ExpressionError(pointer, "invalid_json")
            child = pointer_child(pointer, key)
            budget.consume(3, child)  # quotes and colon
            budget.string_content(key, child)
            result[key] = _resolve(item, context, child, budget, evaluate)
        return result
    if type(value) is list:
        budget.consume(2 + max(0, len(value) - 1), pointer)
        return [
            _resolve(item, context, pointer_child(pointer, i), budget, evaluate)
            for i, item in enumerate(value)
        ]
    if type(value) is not str:
        json_copy(value, pointer)
        budget.consume(len(json.dumps(value, allow_nan=False)), pointer)
        return value
    if not evaluate:
        budget.consume(2, pointer)
        budget.string_content(value, pointer)
        return value
    reference_paths(value, pointer)

    def lookup(match: re.Match) -> object:
        result = context
        for key in match.group(1).split("."):
            if type(result) is not dict or key not in result:
                raise ExpressionError(pointer, "missing_reference")
            result = result[key]
        return result

    pure = _EXPRESSION.fullmatch(value)
    if pure:
        return _resolve(lookup(pure), context, pointer, budget, False)
    budget.consume(2, pointer)
    pieces = []
    end = 0
    for match in _EXPRESSION.finditer(value):
        literal = value[end : match.start()]
        budget.string_content(literal, pointer)
        pieces.append(literal)
        item = lookup(match)
        if type(item) not in (str, bool, int, float, type(None)):
            raise ExpressionError(
                pointer, "unsafe_conversion", "Containers require a pure reference."
            )
        if type(item) is not str:
            json_copy(item, pointer)
        rendered = item if type(item) is str else json.dumps(item, allow_nan=False)
        budget.string_content(rendered, pointer)
        pieces.append(rendered)
        end = match.end()
    literal = value[end:]
    budget.string_content(literal, pointer)
    pieces.append(literal)
    return "".join(pieces)


def resolve_value(
    value: object, context: dict, *, byte_limit: int = DEFAULT_EXPRESSION_BYTES
) -> object:
    """Resolve JSON recursively once, preserving pure-reference runtime types.

    Mixed text renders scalar values using JSON spelling (true/null); implicit
    list/map-to-text conversion is refused. Keys are literals. Returned values
    are detached; strings obtained from references are never evaluated again.
    byte_limit bounds aggregate canonical UTF-8 JSON, including escaping and
    punctuation, before concatenation or container copying. Two-argument calls
    use 10 MiB. Admission/runtime callers pass captured, possibly lower remaining
    byte limits; increasing WorkflowLimits alone does not increase this default.

    Raises:
        ExpressionError: Missing dictionary keys, unsupported syntax or non-JSON.
    """
    try:
        if type(context) is not dict:
            raise ExpressionError("", "invalid_json")
        return _resolve(value, context, "", _Budget(byte_limit))
    except (RecursionError, ValueError, OverflowError, UnicodeError) as error:
        if isinstance(error, ExpressionError):
            raise
        raise ExpressionError("", "invalid_json") from None
