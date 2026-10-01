"""Bounded field-aware lexical matching over canonical profile content."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from tldw_profile_core import ProfileRecord

_MAX_QUERY_CHARS = 4_096
_MAX_QUERY_TERMS = 32
_MAX_FIELD_CHARS = 16_384
_COMPONENT_RE = re.compile(r"[._-]+")


@dataclass(frozen=True, slots=True, repr=False)
class LexicalQuery:
    """Normalized distinct terms and bounded ordered phrase tokens."""

    terms: tuple[str, ...]
    phrase: tuple[str, ...]


@dataclass(frozen=True, slots=True, repr=False)
class LexicalMatch:
    """Search ranking evidence from one canonical record's content fields."""

    distinct_terms: int
    subject_terms: int
    phrase: bool


def _tokens(value: str, maximum: int) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value[:maximum]).casefold()[:maximum]
    tokens: list[str] = []
    index = 0
    length = len(normalized)
    while index < length:
        start = index
        if (
            normalized[index] == "."
            and index + 1 < length
            and _is_base(normalized[index + 1])
        ):
            index += 1
        elif not _is_base(normalized[index]):
            index += 1
            continue
        index += 1
        while index < length:
            character = normalized[index]
            if unicodedata.category(character)[0] in "LNM" or (
                character in "._-"
                and index + 1 < length
                and _is_base(normalized[index + 1])
            ):
                index += 1
            else:
                break
        while index < length and normalized[index] in "+#":
            index += 1
        tokens.append(normalized[start:index])
    return tuple(tokens)


def _is_base(character: str) -> bool:
    return unicodedata.category(character)[0] in "LN"


def compile_query(text: str) -> LexicalQuery:
    """Compile at most 32 distinct Unicode terms from a bounded query."""

    tokens = _tokens(text, _MAX_QUERY_CHARS)
    return LexicalQuery(
        tuple(dict.fromkeys(tokens))[:_MAX_QUERY_TERMS], tokens[:_MAX_QUERY_TERMS]
    )


def _field_terms(tokens: tuple[str, ...]) -> set[str]:
    terms = set(tokens)
    for token in tokens:
        if any(mark in token for mark in "._-"):
            terms.update(part for part in _COMPONENT_RE.split(token) if part)
    return terms


def _has_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    return bool(size) and any(
        tokens[index : index + size] == phrase
        for index in range(len(tokens) - size + 1)
    )


def match_record(record: ProfileRecord, query: LexicalQuery) -> LexicalMatch | None:
    """Score only semantic-key and human-readable payload values."""

    if not query.terms:
        return None
    semantic = record.semantic_key
    subject_values = ((semantic.subject,) if semantic is not None else ()) + (
        (record.payload.subject,) if hasattr(record.payload, "subject") else ()
    )
    values = tuple(
        value
        for name in ("value", "outcome", "text")
        if isinstance(value := getattr(record.payload, name, None), str)
    )
    namespace = (
        (semantic.namespace,)
        if semantic is not None and semantic.namespace != record.kind.value
        else ()
    )
    all_terms: set[str] = set()
    subject_terms: set[str] = set()
    phrase = False
    for value in subject_values + values + namespace:
        tokens = _tokens(value, _MAX_FIELD_CHARS)
        field_terms = _field_terms(tokens)
        all_terms.update(field_terms)
        if value in subject_values:
            subject_terms.update(field_terms)
        phrase = phrase or _has_phrase(tokens, query.phrase)
    hits = all_terms.intersection(query.terms)
    if not hits:
        return None
    return LexicalMatch(len(hits), len(subject_terms.intersection(query.terms)), phrase)
