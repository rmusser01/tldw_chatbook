"""Tests for the pure slash-command suggestion provider."""

from tldw_chatbook.Chat.console_command_grammar import default_console_registry
from tldw_chatbook.Chat.console_command_suggestions import suggestions_for_draft
from tldw_chatbook.Chat.console_skill_resolver import SkillCommandCandidate

SKILLS = (
    SkillCommandCandidate(name="web-search", description="Search the web"),
    SkillCommandCandidate(name="summarize", description="Summarize text"),
)


def _labels(result):
    return [s.label for s in result]


def test_bare_slash_lists_commands_then_skills():
    result = suggestions_for_draft("/", default_console_registry(), SKILLS)
    assert _labels(result) == ["/prompt", "/system", "/skills", "/web-search", "/summarize"]


def test_prefix_filters_case_insensitively():
    result = suggestions_for_draft("/SK", default_console_registry(), SKILLS)
    assert _labels(result) == ["/skills"]


def test_skill_entries_insert_bare_slash_name():
    result = suggestions_for_draft("/w", default_console_registry(), SKILLS)
    assert _labels(result) == ["/web-search"]
    assert result[0].insert_text == "/web-search "
    assert result[0].description == "Search the web"


def test_non_command_drafts_return_none():
    registry = default_console_registry()
    assert suggestions_for_draft("hello", registry, SKILLS) is None
    assert suggestions_for_draft("/prompt foo", registry, SKILLS) is None
    assert suggestions_for_draft(" /", registry, SKILLS) is None


def test_empty_filter_returns_empty_list():
    assert suggestions_for_draft("/zzz", default_console_registry(), SKILLS) == []


def test_skills_arg_mode_filters_and_builds_full_replacement():
    result = suggestions_for_draft("/skills w", default_console_registry(), SKILLS)
    assert _labels(result) == ["web-search"]
    assert result[0].insert_text == "/skills web-search "


def test_skills_arg_mode_ends_after_second_argument():
    assert suggestions_for_draft("/skills web-search extra", default_console_registry(), SKILLS) is None


def test_skill_named_like_a_command_is_deduplicated():
    skills = (SkillCommandCandidate(name="prompt", description="clash"),)
    result = suggestions_for_draft("/", default_console_registry(), skills)
    assert _labels(result) == ["/prompt", "/system", "/skills"]


def test_trailing_newline_leaves_completion_contexts():
    registry = default_console_registry()
    assert suggestions_for_draft("/sy\n", registry, SKILLS) is None
    assert suggestions_for_draft("/skills \n", registry, SKILLS) is None
