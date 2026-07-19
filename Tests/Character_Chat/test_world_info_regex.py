import pytest

from tldw_chatbook.Character_Chat.world_info_regex import (
    validate_regex_pattern,
    regex_search,
    MAX_REGEX_PATTERN_LENGTH,
)


@pytest.mark.parametrize("pattern", ["w[ao]rden", r"https?://", r"(\d{3}-)+\d{4}", "(a|b)*", "hello"])
def test_valid_patterns_pass(pattern):
    validate_regex_pattern(pattern)  # must not raise


def test_over_length_rejected():
    with pytest.raises(ValueError, match="too long"):
        validate_regex_pattern("a" * (MAX_REGEX_PATTERN_LENGTH + 1))


def test_syntax_error_rejected():
    with pytest.raises(ValueError, match="Invalid regex"):
        validate_regex_pattern("(unclosed")


@pytest.mark.parametrize("pattern", ["(a+)+", "(a*)*", "(a+)*$", "(a|a)*"])
def test_catastrophic_patterns_rejected(pattern):
    with pytest.raises(ValueError, match="too complex"):
        validate_regex_pattern(pattern)


@pytest.mark.parametrize("pattern", [
    "((a+))+",       # nesting must not evade the nested-quantifier detector
    "(?:(a+))+",     # non-capturing wrapper
    "(((a+)))+",     # deeper nesting
    "(a|a|a)+",      # N-way identical alternation (not just 2-way)
    "(?:x|x)*",      # identical alternation behind a non-capturing group
])
def test_catastrophic_variants_rejected(pattern):
    with pytest.raises(ValueError, match="too complex"):
        validate_regex_pattern(pattern)


@pytest.mark.parametrize("pattern", [
    "(?P<name>x+)",     # inner quantifier but NOT externally quantified — safe
    "(?:abc){2,5}",     # bounded outer quantifier — safe
    r"(\d{3}-)+\d{4}",  # bounded inner quantifier — safe
    "(a|b|c)+",         # distinct alternatives — safe
    "(cat|dog)*",       # distinct alternatives — safe
])
def test_safe_variants_still_pass(pattern):
    validate_regex_pattern(pattern)  # must not raise


def test_regex_search_matches_case_insensitive():
    assert regex_search("w[ao]rden", "The WARDEN speaks", ignore_case=True) is True
    assert regex_search("w[ao]rden", "The WARDEN speaks", ignore_case=False) is False
    assert regex_search("w[ao]rden", "the warden", ignore_case=False) is True


def test_regex_search_never_raises_on_bad_pattern():
    # An uncompilable pattern must return False, not raise.
    assert regex_search("(unclosed", "anything", ignore_case=True) is False


def test_regex_search_returns_bool():
    assert regex_search("x", "no match here", ignore_case=True) is False
