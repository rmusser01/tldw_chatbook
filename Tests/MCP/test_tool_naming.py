from __future__ import annotations

import hashlib
import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_chatbook.MCP.tool_naming import (
    dedupe_names,
    llm_tool_name,
    sanitize_component,
)

_ALLOWED_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_LEN = 64


# ---------------------------------------------------------------------------
# sanitize_component
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("simple", "simple"),
        ("with space", "with_space"),
        ("with.dot.path", "with_dot_path"),
        ("under_score-kept", "under_score-kept"),
        ("  leading and trailing  ", "leading_and_trailing"),
        ("___only_underscores___", "only_underscores"),
        ("...", "x"),
        ("", "x"),
        ("   ", "x"),
        ("héllo wörld", "h_llo_w_rld"),
        ("🎉tool🎉", "tool"),
        ("Hello-World_123", "Hello-World_123"),
    ],
)
def test_sanitize_component_table(text, expected):
    assert sanitize_component(text) == expected


def test_sanitize_component_result_always_matches_allowed_charset():
    for text in ["", "   ", "a/b\\c", "日本語", "café_menu.v2", "---"]:
        result = sanitize_component(text)
        assert result != ""
        assert _ALLOWED_RE.match(result)


def test_sanitize_component_collapses_runs_not_individual_chars():
    # Three consecutive invalid chars collapse to a single "_", not three.
    assert sanitize_component("a...b") == "a_b"
    assert sanitize_component("a   b") == "a_b"


# ---------------------------------------------------------------------------
# llm_tool_name - basic assembly + prefix stripping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("server_key", "tool_name", "expected"),
    [
        ("myserver", "list_files", "mcp__myserver__list_files"),
        ("local:myserver", "list_files", "mcp__myserver__list_files"),
        ("builtin:core", "calculate", "mcp__core__calculate"),
        ("server:api", "search", "mcp__api__search"),  # defensive <word>: prefix
        ("local:web search", "find.docs", "mcp__web_search__find_docs"),
    ],
)
def test_llm_tool_name_basic_assembly(server_key, tool_name, expected):
    assert llm_tool_name(server_key, tool_name) == expected


def test_llm_tool_name_only_strips_server_key_prefix_not_tool_name_prefix():
    # tool_name looking like it has a "word:" prefix must NOT be stripped,
    # only sanitized (":" becomes "_").
    result = llm_tool_name("local:myserver", "builtin:eval")
    assert result == "mcp__myserver__builtin_eval"


def test_llm_tool_name_result_always_matches_allowed_charset_and_length():
    for server_key, tool_name in [
        ("local:my server", "do thing!"),
        ("", ""),
        ("builtin:", "x"),
        ("weird/chars\\here", "tool🎉name"),
    ]:
        result = llm_tool_name(server_key, tool_name)
        assert len(result) <= MAX_LEN
        assert _ALLOWED_RE.match(result)


# ---------------------------------------------------------------------------
# llm_tool_name - 64-char truncate-with-hash determinism
# ---------------------------------------------------------------------------

def test_llm_tool_name_truncates_and_appends_hash_when_over_64_chars():
    server_key = "local:" + ("server" * 20)
    tool_name = "tool" * 20
    result = llm_tool_name(server_key, tool_name)

    assert len(result) == MAX_LEN

    label = "server" * 20  # prefix stripped, all-valid chars so unchanged
    untruncated = f"mcp__{label}__{tool_name}"
    assert result[:55] == untruncated[:55]

    expected_hash = hashlib.sha256(
        f"{server_key}::{tool_name}".encode("utf-8")
    ).hexdigest()[:8]
    assert result == f"{untruncated[:55]}_{expected_hash}"


def test_llm_tool_name_truncation_is_deterministic():
    server_key = "local:" + ("abc" * 30)
    tool_name = "def" * 30
    first = llm_tool_name(server_key, tool_name)
    second = llm_tool_name(server_key, tool_name)
    assert first == second


def test_llm_tool_name_truncation_disambiguates_shared_visible_prefix():
    # Two long inputs whose first 55 visible characters are identical (the
    # difference is only in tool_name, which lands past character 55) must
    # still resolve to different final names because the hash is computed
    # from the *original* server_key/tool_name identity, not the truncated
    # visible text.
    label_raw = "s" * 100
    server_key = f"local:{label_raw}"
    result_a = llm_tool_name(server_key, "toolA")
    result_b = llm_tool_name(server_key, "toolB")

    assert result_a[:55] == result_b[:55]
    assert result_a != result_b
    assert len(result_a) == MAX_LEN
    assert len(result_b) == MAX_LEN


def test_llm_tool_name_exactly_64_chars_is_not_truncated():
    # Craft a name that lands at exactly the boundary to make sure the
    # off-by-one comparison (<=64 keeps, >64 truncates) is correct.
    # len("mcp__") + len(label) + len("__") + len(tool) == 7 + 27 + 30 == 64
    label = "l" * 27
    tool = "t" * 30
    server_key = f"local:{label}"
    name = f"mcp__{label}__{tool}"
    assert len(name) == MAX_LEN
    result = llm_tool_name(server_key, tool)
    assert result == name


def test_llm_tool_name_at_65_chars_is_truncated():
    # 7 + 27 + 31 == 65, one char over the 64 boundary.
    label = "l" * 27
    tool = "t" * 31
    server_key = f"local:{label}"
    name = f"mcp__{label}__{tool}"
    assert len(name) == MAX_LEN + 1
    result = llm_tool_name(server_key, tool)
    assert len(result) == MAX_LEN
    assert result != name


# ---------------------------------------------------------------------------
# dedupe_names - stable collision suffixes
# ---------------------------------------------------------------------------

def test_dedupe_names_leaves_unique_names_untouched():
    names = ["mcp__a__x", "mcp__b__y", "mcp__c__z"]
    assert dedupe_names(names) == names


def test_dedupe_names_suffixes_duplicates_in_order():
    names = ["mcp__a__x", "mcp__a__x", "mcp__a__x"]
    result = dedupe_names(names)
    assert result == ["mcp__a__x", "mcp__a__x_2", "mcp__a__x_3"]


def test_dedupe_names_is_stable_across_repeated_calls():
    names = ["dup", "dup", "unique", "dup"]
    first = dedupe_names(names)
    second = dedupe_names(names)
    assert first == second


def test_dedupe_names_avoids_colliding_with_a_literal_pre_existing_suffix():
    # The natural "_2" suffix for the second "foo" collides with a literal
    # "foo_2" already present in the input, so it must be pushed further.
    names = ["foo", "foo", "foo_2"]
    result = dedupe_names(names)
    assert len(result) == len(set(result)) == 3
    assert result[0] == "foo"


def test_dedupe_names_all_outputs_match_allowed_charset_and_length():
    names = ["mcp__srv__tool"] * 5 + ["mcp__other__tool"]
    result = dedupe_names(names)
    for name in result:
        assert len(name) <= MAX_LEN
        assert _ALLOWED_RE.match(name)


def test_dedupe_names_retruncates_with_hash_when_suffix_overflows_64():
    # A name already at the 64-char ceiling: appending "_2" would overflow,
    # so the implementation must re-truncate (using the hash mechanism)
    # rather than exceed MAX_LEN.
    base = "a" * 55 + "_" + "0" * 8  # 64 chars, already at the ceiling
    assert len(base) == MAX_LEN
    names = [base, base, base]
    result = dedupe_names(names)
    for name in result:
        assert len(name) <= MAX_LEN
    assert len(result) == len(set(result)) == 3
    assert result[0] == base


def test_dedupe_names_empty_list():
    assert dedupe_names([]) == []


# ---------------------------------------------------------------------------
# Property-based: round-trip uniqueness for distinct inputs
# ---------------------------------------------------------------------------

_key_strategy = st.text(min_size=0, max_size=30)
_pair_strategy = st.tuples(_key_strategy, _key_strategy)


@given(st.lists(_pair_strategy, min_size=1, max_size=25, unique=True))
def test_llm_tool_name_then_dedupe_names_is_always_unique(pairs):
    names = [llm_tool_name(server_key, tool_name) for server_key, tool_name in pairs]
    result = dedupe_names(names)

    assert len(result) == len(names)
    assert len(set(result)) == len(result)
    for name in result:
        assert len(name) <= MAX_LEN
        assert _ALLOWED_RE.match(name)


@given(_key_strategy)
def test_sanitize_component_is_idempotent(text):
    once = sanitize_component(text)
    twice = sanitize_component(once)
    assert once == twice
