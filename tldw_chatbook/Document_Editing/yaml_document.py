"""Qualified YAML event graphs and token-preserving whitespace edits.

No load/compose/dump API, application constructors, or alias expansion is used.
The dependency is imported only when a YAML request reaches this adapter.
"""

import re
from decimal import Decimal, InvalidOperation

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.nodes import ScalarNode
from ruamel.yaml.resolver import VersionedResolver

from .models import Diagnostic
from .source_positions import MAX_DIAGNOSTICS, Budget, DocumentError

_PREFIX = "tag:yaml.org,2002:"
_CORE_INT = re.compile(r"(?:[+-]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)\Z")
_CORE_FLOAT = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|\.(?:inf|Inf|INF))\Z"
)
_TAGS = {"str", "null", "bool", "int", "float", "map", "seq", "merge"}


def _unsupported(message: str, event: object) -> DocumentError:
    return DocumentError(
        "unsupported", message, event.start_mark.index, event.end_mark.index
    )


def _scalar_tag(event: object, version: tuple[int, int]) -> str:
    if event.tag is not None and event.tag != "!":
        return event.tag
    if event.style is not None or event.tag == "!":
        return _PREFIX + "str"
    return _plain_tag(event.value, version)


def _plain_tag(value: str, version: tuple[int, int]) -> str:
    if version == (1, 1):
        return str(
            VersionedResolver(version=version).resolve(ScalarNode, value, (True, False))
        )
    # YAML 1.2 Core, deliberately excluding the library's implicit timestamp and
    # 1.1 extensions. Merge is the programme's explicit compatibility extension.
    if value == "<<":
        kind = "merge"
    elif value in ("", "~", "null", "Null", "NULL"):
        kind = "null"
    elif value in ("true", "True", "TRUE", "false", "False", "FALSE"):
        kind = "bool"
    elif _CORE_INT.fullmatch(value):
        kind = "int"
    elif _CORE_FLOAT.fullmatch(value) or value in (".nan", ".NaN", ".NAN"):
        kind = "float"
    else:
        kind = "str"
    return _PREFIX + kind


def _key(event: object, tag: str, version: tuple[int, int]) -> tuple[str, object]:
    """Canonical scalar-key identity without constructing an application graph."""
    kind = tag.removeprefix(_PREFIX)
    value = event.value
    try:
        if kind in ("str", "merge"):
            identity = value
        elif kind == "null":
            identity = None
        elif kind == "bool":
            identity = value.lower() in ("true", "yes", "on", "y")
        elif kind in ("int", "float"):
            number = value.replace("_", "")
            sign = -1 if number.startswith("-") else 1
            unsigned = number.lstrip("+-")
            if kind == "float":
                if unsigned.lower() == ".nan":
                    identity = "nan"
                elif unsigned.lower() == ".inf":
                    identity = sign * Decimal("Infinity")
                elif ":" in number:
                    # 1.1 sexagesimal, computed exactly without Decimal context
                    # rounding (the fractional final segment stays a decimal).
                    segments = number.lstrip("+-").split(":")
                    whole = 0
                    for segment in segments[:-1]:
                        whole = whole * 60 + int(segment)
                    final = segments[-1].split(".")
                    digits = str(whole * 60 + int(final[0]))
                    identity = Decimal(
                        ("-" if sign < 0 else "")
                        + digits
                        + ("." + final[1] if len(final) > 1 else "")
                    )
                else:
                    identity = Decimal(number)
            elif ":" in unsigned:
                identity = 0
                for segment in unsigned.split(":"):
                    identity = identity * 60 + int(segment)
                identity *= sign
            elif unsigned.startswith(("0x", "0o", "0b")):
                identity = sign * int(
                    unsigned[2:], {"x": 16, "o": 8, "b": 2}[unsigned[1]]
                )
            elif version == (1, 1) and unsigned.startswith("0"):
                identity = sign * int(unsigned, 8)
            else:
                identity = int(number)
        else:
            raise _unsupported("This YAML scalar key type is not qualified.", event)
    except (ValueError, InvalidOperation, OverflowError):
        raise DocumentError(
            "resource_limited",
            "Numeric key exceeds canonicalization limits.",
            event.start_mark.index,
            event.end_mark.index,
        ) from None
    return tag, identity


def _scan(text: str, budget: Budget) -> list[object]:
    scanned = []
    depth = 0
    for token in YAML(typ="rt", pure=True).scan(text):
        kind = type(token).__name__
        if kind in (
            "BlockMappingStartToken",
            "BlockSequenceStartToken",
            "FlowMappingStartToken",
            "FlowSequenceStartToken",
        ):
            depth += 1
        elif kind in ("BlockEndToken", "FlowMappingEndToken", "FlowSequenceEndToken"):
            depth -= 1
        budget.check(depth)
        if (
            kind == "DirectiveToken"
            and token.name == "YAML"
            and token.value not in ((1, 1), (1, 2))
        ):
            raise _unsupported("Only YAML 1.1 and 1.2 directives are qualified.", token)
        scanned.append(token)
    return scanned


def _graph(text: str, budget: Budget) -> tuple[list[tuple], list[Diagnostic]]:
    graph = []
    diagnostics = []
    scopes = []
    anchors = {}
    versions = set()
    version = (1, 2)
    for event in YAML(typ="rt", pure=True).parse(text):
        kind = type(event).__name__
        budget.check(len(scopes))
        if kind == "DocumentStartEvent":
            version = event.version or (1, 2)
            versions.add(version)
            if len(versions) > 1:
                raise _unsupported(
                    "Mixed YAML-version streams are not qualified.", event
                )
            anchors = {}
        tag = getattr(event, "tag", None)
        if kind == "ScalarEvent":
            tag = _scalar_tag(event, version)
        if (
            tag is not None
            and tag != "!"
            and (not tag.startswith(_PREFIX) or tag.removeprefix(_PREFIX) not in _TAGS)
        ):
            raise _unsupported(
                "This YAML tag is not qualified; no constructors were executed.", event
            )
        if tag not in (None, "!"):
            tag_kind = tag.removeprefix(_PREFIX)
            if kind == "ScalarEvent":
                if tag_kind in ("map", "seq") or (
                    event.tag is not None
                    and tag_kind != "str"
                    and _plain_tag(event.value, version) != tag
                    and not (tag_kind == "float" and _CORE_FLOAT.fullmatch(event.value))
                ):
                    raise _unsupported(
                        "The explicit scalar tag and value are not qualified together.",
                        event,
                    )
            elif kind in ("MappingStartEvent", "SequenceStartEvent"):
                expected = "map" if kind == "MappingStartEvent" else "seq"
                if tag_kind != expected:
                    raise _unsupported(
                        "This tag does not match the YAML collection kind.", event
                    )
        if kind in (
            "MappingStartEvent",
            "SequenceStartEvent",
            "ScalarEvent",
            "AliasEvent",
        ):
            merge_sequence = False
            is_key = bool(scopes and scopes[-1]["mapping"] and scopes[-1]["key"])
            if is_key:
                if kind != "ScalarEvent":
                    raise _unsupported(
                        "Complex or alias mapping keys are not qualified.", event
                    )
                identity = _key(event, tag, version)
                names = scopes[-1]["names"]
                if identity in names and len(diagnostics) <= MAX_DIAGNOSTICS:
                    diagnostics.append(
                        Diagnostic(
                            "duplicate_key",
                            "error",
                            "Duplicate YAML mapping key.",
                            event.start_mark.index,
                            event.end_mark.index,
                        )
                    )
                names.add(identity)
                scopes[-1]["merge"] = tag == _PREFIX + "merge"
            elif scopes and (
                scopes[-1].get("merge", False) or scopes[-1].get("merge_items", False)
            ):
                merge_sequence = scopes[-1]["mapping"] and kind == "SequenceStartEvent"
                if not (
                    merge_sequence
                    or kind == "MappingStartEvent"
                    or (
                        kind == "AliasEvent"
                        and anchors.get(event.anchor) == "MappingStartEvent"
                    )
                ):
                    raise _unsupported(
                        "Merge values must be mappings or sequences of mappings; aliases are not expanded.",
                        event,
                    )
            if scopes and scopes[-1]["mapping"]:
                scopes[-1]["key"] = not scopes[-1]["key"]
            anchor = getattr(event, "anchor", None)
            if kind == "AliasEvent":
                if anchor not in anchors:
                    raise DocumentError(
                        "syntax",
                        "YAML alias has no preceding anchor in this document.",
                        event.start_mark.index,
                        event.end_mark.index,
                    )
            elif anchor is not None:
                anchors[anchor] = kind
            if kind in ("MappingStartEvent", "SequenceStartEvent"):
                scopes.append(
                    {
                        "mapping": kind == "MappingStartEvent",
                        "key": True,
                        "names": set(),
                        "merge_items": merge_sequence,
                    }
                )
                budget.check(len(scopes))
        elif kind in ("MappingEndEvent", "SequenceEndEvent"):
            scopes.pop()
        graph.append(
            (
                kind,
                tag,
                *(
                    getattr(event, field, None)
                    for field in (
                        "value",
                        "anchor",
                        "style",
                        "flow_style",
                        "implicit",
                        "explicit",
                        "version",
                        "tags",
                    )
                ),
            )
        )
    return graph, diagnostics


def _edits(
    text: str, scanned: list[object], budget: Budget
) -> list[tuple[int, int, str]]:
    """Edit only whitespace outside source tokens and comments."""
    edits = {}
    blocks = []
    flow_depth = 0
    protected_lines = set()
    scalar_headers = set()
    for token in scanned:
        if (
            type(token).__name__ == "ScalarToken"
            and token.end_mark.line > token.start_mark.line
        ):
            scalar_headers.add(token.start_mark.line)
            protected_lines.update(
                range(
                    token.start_mark.line + 1,
                    token.end_mark.line + bool(token.end_mark.column),
                )
            )
    for token in scanned:
        budget.check()
        kind = type(token).__name__
        if kind in ("BlockMappingStartToken", "BlockSequenceStartToken"):
            blocks.append(kind)
        elif kind == "BlockEndToken":
            blocks.pop()
        elif kind == "BlockEntryToken" and (
            not blocks or blocks[-1] != "BlockSequenceStartToken"
        ):
            raise _unsupported(
                "Indentless block sequences are not qualified for formatting.", token
            )
        block_depth = len(blocks)
        start, end = token.start_mark.index, token.end_mark.index
        line, column = token.start_mark.line, token.start_mark.column
        line_start = start - column
        if (
            not flow_depth
            and block_depth
            and kind not in ("BlockEndToken", "StreamEndToken")
            and line not in protected_lines
            and text[line_start:start] == " " * column
        ):
            wanted = "  " * (block_depth - 1)
            if wanted != text[line_start:start]:
                if line in scalar_headers:
                    raise _unsupported(
                        "Reindenting this multiline scalar would change its source token; keep its current indentation.",
                        token,
                    )
                edits[(line_start, start)] = wanted
        if kind in ("FlowMappingStartToken", "FlowSequenceStartToken"):
            flow_depth += 1
        elif kind in ("FlowMappingEndToken", "FlowSequenceEndToken"):
            flow_depth -= 1
        if (
            kind in ("ValueToken", "FlowEntryToken", "BlockEntryToken", "KeyToken")
            and end > start
        ):
            stop = end
            while stop < len(text) and text[stop] in " \t":
                stop += 1
            if (
                stop > end
                and stop < len(text)
                and text[stop] not in "#\r\n"
                and text[end:stop] != " "
            ):
                edits[(end, stop)] = " "
    return [(start, end, value) for (start, end), value in sorted(edits.items())]


def _token_signature(text: str, scanned: list[object]) -> list[tuple[str, str]]:
    signature = []
    position = 0
    for token in scanned:
        start, end = token.start_mark.index, token.end_mark.index
        if start == end:
            continue
        # Comments live in trivia between tokens (or inside a block scalar's
        # raw token). Retain their exact text and position in the token stream.
        signature.extend(
            ("trivia", match.group())
            for match in re.finditer(r"#[^\r\n]*|[^\s]+", text[position:start])
        )
        signature.append((type(token).__name__, text[start:end]))
        position = end
    signature.extend(
        ("trivia", match.group())
        for match in re.finditer(r"#[^\r\n]*|[^\s]+", text[position:])
    )
    return signature


def analyze(
    text: str, format_requested: bool, budget: Budget
) -> tuple[list[Diagnostic], str | None]:
    """Check source and verify symbolic graph and every token before proposing it."""
    try:
        scanned = _scan(text, budget)
        graph, diagnostics = _graph(text, budget)
        if diagnostics:
            return diagnostics, None
        edits = _edits(text, scanned, budget)
        if not format_requested:
            return [], None
        parts = []
        position = 0
        for start, end, value in edits:
            parts.extend((text[position:start], value))
            position = end
        parts.append(text[position:])
        candidate = "".join(parts)
        if candidate != text:
            try:
                verified = _scan(candidate, budget)
                candidate_graph, problems = _graph(candidate, budget)
            except (YAMLError, DocumentError) as error:
                if (
                    isinstance(error, DocumentError)
                    and error.code == "resource_limited"
                ):
                    raise
                raise DocumentError(
                    "failed",
                    "YAML formatter produced unqualified source; keep the original draft.",
                ) from None
            if (
                problems
                or graph != candidate_graph
                or _token_signature(text, scanned)
                != _token_signature(candidate, verified)
            ):
                raise DocumentError(
                    "failed",
                    "YAML formatter preservation check failed; keep the original draft.",
                )
        return [], candidate
    except YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        offset = min(mark.index, len(text)) if mark is not None else 0
        raise DocumentError(
            "syntax", "Invalid YAML syntax.", offset, min(offset + 1, len(text))
        ) from None
