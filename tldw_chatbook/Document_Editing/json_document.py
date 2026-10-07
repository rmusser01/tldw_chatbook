"""Strict stdlib validation followed only by source-token whitespace edits."""

import json
import re

from .models import Diagnostic
from .source_positions import MAX_BYTES, MAX_DIAGNOSTICS, Budget, DocumentError

_TOKEN = re.compile(
    r"\s+|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null|[{}\[\]:,]",
    re.ASCII,
)


def tokens(text: str, budget: Budget) -> list[tuple[str, int, int]]:
    """Record raw token slices and bound depth before stdlib object creation."""
    result = []
    position = depth = 0
    while position < len(text):
        budget.check(depth)
        if text[position] == '"':
            try:
                _decoded, end = json.decoder.scanstring(text, position + 1, True)
            except json.JSONDecodeError as error:
                raise DocumentError(
                    "syntax",
                    "Invalid JSON string.",
                    error.pos,
                    min(error.pos + 1, len(text)),
                ) from None
            result.append((text[position:end], position, end))
            position = end
            continue
        match = _TOKEN.match(text, position)
        if match is None:
            raise DocumentError(
                "syntax", "Invalid JSON syntax.", position, position + 1
            )
        value = match.group()
        if value[0] not in " \t\r\n":
            if value in ("{", "["):
                depth += 1
                budget.check(depth)
            elif value in ("}", "]"):
                depth -= 1
            result.append((value, position, match.end()))
        position = match.end()
    return result


def _reject_constant(_value: str) -> None:
    raise DocumentError("syntax", "JSON constants must be finite JSON literals.")


def _pairs(pairs: list[tuple[str, object]]) -> tuple[bool, ...]:
    # Keep duplicate awareness in validation, without building dictionaries that
    # silently discard members. Locations come from the validated token stream.
    names = set()
    duplicates = False
    for name, _value in pairs:
        duplicates |= name in names
        names.add(name)
    return (duplicates,)


def _record(
    text: str, budget: Budget
) -> tuple[list[Diagnostic], list[tuple[str, int, int]]]:
    raw = tokens(text, budget)
    try:
        json.loads(
            text,
            parse_int=str,
            parse_float=str,
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs,
        )
    except json.JSONDecodeError as error:
        raise DocumentError(
            "syntax", "Invalid JSON syntax.", error.pos, min(error.pos + 1, len(text))
        ) from None
    budget.check()
    scopes: list[set[str] | None] = []
    diagnostics = []
    for index, (value, start, end) in enumerate(raw):
        if index % 256 == 0:
            budget.check()
        if value in ("{", "["):
            scopes.append(set() if value == "{" else None)
        elif value in ("}", "]"):
            scopes.pop()
        elif (
            value.startswith('"') and index + 1 < len(raw) and raw[index + 1][0] == ":"
        ):
            name = json.loads(value)
            scope = scopes[-1]
            if name in scope and len(diagnostics) <= MAX_DIAGNOSTICS:
                diagnostics.append(
                    Diagnostic(
                        "duplicate_key",
                        "warning",
                        "Duplicate object name; formatting is blocked.",
                        start,
                        end,
                    )
                )
            scope.add(name)
    return diagnostics, raw


def _layout(
    raw: list[tuple[str, int, int]], newline: str, compact: bool, budget: Budget
) -> str:
    parts = []
    depth = size = 0
    for index, (value, _start, _end) in enumerate(raw):
        if index % 256 == 0:
            budget.check()
        if compact:
            piece = value
        elif value in ("{", "["):
            depth += 1
            piece = value
            if raw[index + 1][0] not in ("}", "]"):
                piece += newline + "  " * depth
        elif value in ("}", "]"):
            depth -= 1
            piece = (
                value
                if raw[index - 1][0] in ("{", "[")
                else newline + "  " * depth + value
            )
        elif value == ",":
            piece = "," + newline + "  " * depth
        elif value == ":":
            piece = ": "
        else:
            piece = value
        size += len(piece)
        if size > MAX_BYTES:
            raise DocumentError(
                "resource_limited", "Formatted source exceeds the size limit."
            )
        parts.append(piece)
    return "".join(parts)


def analyze(
    text: str, language: str, newline: str, format_requested: bool, budget: Budget
) -> tuple[list[Diagnostic], str | None]:
    """Check a document or each physical JSONL record, retaining raw lexemes."""
    records = text.split("\n")
    if language == "json":
        records = [text]
    elif text == "":
        records = []
    elif text.endswith("\n"):
        records.pop()
    diagnostics: list[Diagnostic] = []
    output = []
    offset = 0
    for index, record in enumerate(records):
        ending = ""
        if language == "jsonl" and (index < len(records) - 1 or text.endswith("\n")):
            ending = "\r\n" if record.endswith("\r") else "\n"
            if ending == "\r\n":
                record = record[:-1]
        try:
            if language == "jsonl" and not record.strip(" \t\r"):
                raise DocumentError(
                    "blank_record",
                    "Blank JSONL records are not allowed.",
                    0,
                    len(record),
                )
            found, raw = _record(record, budget)
            diagnostics.extend(
                Diagnostic(
                    d.code, d.severity, d.message, d.start + offset, d.end + offset
                )
                for d in found[: MAX_DIAGNOSTICS + 1 - len(diagnostics)]
            )
            if format_requested and not diagnostics:
                formatted = _layout(raw, newline, language == "jsonl", budget)
                if [t[0] for t in tokens(formatted, budget)] != [t[0] for t in raw]:
                    raise DocumentError(
                        "failed",
                        "Formatter token verification failed; keep the original draft.",
                    )
                if language == "json" and text.endswith(("\n", "\r")):
                    formatted += newline
                output.append(formatted + ending)
        except DocumentError as error:
            if error.code not in ("syntax", "blank_record"):
                raise
            diagnostics.append(
                Diagnostic(
                    error.code,
                    "error",
                    str(error),
                    offset + error.start,
                    offset + error.end,
                )
            )
        if len(diagnostics) > MAX_DIAGNOSTICS:
            break
        offset += len(record) + len(ending)
    return diagnostics, (
        "".join(output) if format_requested and not diagnostics else None
    )
