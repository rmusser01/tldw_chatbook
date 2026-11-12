"""Frozen field-aware lexical behavior for Personal Context records."""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from statistics import median
from time import perf_counter

import pytest
from tldw_profile_core import (
    PreferencePayload,
    ProfileControls,
    ProfileProvenance,
    ProfileRecord,
    SemanticKey,
)

from tldw_chatbook.Personal_Context.lexical_match import compile_query, match_record

NOW = datetime(2026, 9, 25, tzinfo=UTC)


def _record(
    value: str,
    *,
    subject: str = "toolchain",
    namespace: str = "preference",
    record_id: str = "synthetic-record",
) -> ProfileRecord:
    return ProfileRecord(
        profile_id="synthetic-profile",
        record_id=record_id,
        scope_id="synthetic-scope",
        kind="preference",
        payload=PreferencePayload(subject=subject, polarity="like", value=value),
        semantic_key=SemanticKey(namespace=namespace, subject=subject),
        state="active",
        controls=ProfileControls(
            sync_mode="syncable", agent_visibility="agent_visible"
        ),
        provenance=ProfileProvenance(
            source="manual",
            actor="user",
            reason_code="settings_edit",
            source_references=("source-canary",),
            source_hashes=("a" * 64,),
        ),
        version_id="version-canary",
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
        expires_at=None,
    )


@pytest.mark.parametrize(
    ("query", "value", "matches"),
    [
        ("C", "C", True),
        ("C", "C++", False),
        ("C++", "C++", True),
        ("C++", "C#", False),
        ("C#", "C#", True),
        (".NET", ".NET", True),
        ("NET", ".NET", True),
        ("café", "cafe\u0301", True),
        ("cafe", "café", False),
        ("i", "İstanbul", False),
        ("İstanbul", "İstanbul", True),
        ("q", "q\u0301", False),
        ("q\u0301", "q\u0301", True),
        ("café", "café\u0307", False),
        ("東京", "東京", True),
        ("東京", "東京駅", False),
        ("strasse", "Straße", True),
    ],
)
def test_technical_and_international_terms_match_whole_tokens(
    query: str, value: str, matches: bool
) -> None:
    assert (match_record(_record(value), compile_query(query)) is not None) is matches


def test_compound_subject_component_matches_without_metadata() -> None:
    record = _record("concise", subject="response.detail")

    match = match_record(record, compile_query("detail"))

    assert match is not None
    assert (match.distinct_terms, match.subject_terms, match.phrase) == (1, 1, False)
    for query in ("settings_edit", "source-canary", "version-canary", "schema_version"):
        assert match_record(record, compile_query(query)) is None


def test_nondefault_semantic_namespace_is_content_but_default_kind_is_not() -> None:
    assert match_record(
        _record("value", namespace="travel-style"), compile_query("travel")
    )
    assert match_record(_record("value"), compile_query("preference")) is None


def test_distinct_term_subject_and_same_field_phrase_scores() -> None:
    query = compile_query("concise replies")
    two_terms = match_record(_record("concise replies"), query)
    subject = match_record(_record("replies", subject="concise"), query)
    cross_field = match_record(_record("replies", subject="concise"), query)
    one_term = match_record(_record("concise"), query)

    assert two_terms is not None and subject is not None
    assert cross_field is not None and one_term is not None
    assert (two_terms.distinct_terms, two_terms.subject_terms, two_terms.phrase) == (
        2,
        0,
        True,
    )
    assert (subject.distinct_terms, subject.subject_terms, subject.phrase) == (
        2,
        1,
        False,
    )
    assert cross_field.phrase is False
    assert one_term.distinct_terms == 1


def test_reordered_terms_qualify_without_phrase_bonus() -> None:
    match = match_record(_record("concise replies"), compile_query("replies concise"))

    assert match is not None
    assert (match.distinct_terms, match.phrase) == (2, False)


def test_query_bounds_and_no_usable_terms() -> None:
    query = compile_query(" ".join(f"word{i:02d}" for i in range(33)))
    assert len(query.terms) == 32
    assert "word32" not in query.terms
    assert match_record(_record("word32"), query) is None
    assert match_record(_record("value"), compile_query(" / + # ... ")) is None
    assert (
        match_record(_record("needle"), compile_query("x" * 4096 + " needle")) is None
    )


def test_large_canonical_text_still_matches_within_field_bound() -> None:
    record = _record("x" * 16_000 + " needle")

    assert match_record(record, compile_query("needle")) is not None


def test_bounded_synthetic_match_cost() -> None:
    records = tuple(
        _record("x" * 1_017 + " needle", record_id=f"bounded-{index:03d}")
        for index in range(128)
    )
    query = compile_query("needle")
    timings = []
    for _ in range(5):
        start = perf_counter()
        matched = sum(match_record(record, query) is not None for record in records)
        timings.append(perf_counter() - start)
        assert matched == 128
    print(
        f"LEXICAL_COST python={platform.python_version()} records=128 "
        f"field_chars=1024 median_seconds={median(timings):.6f}"
    )
