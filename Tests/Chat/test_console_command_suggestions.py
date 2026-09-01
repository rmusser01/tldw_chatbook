"""Tests for the pure slash-command suggestion provider."""

from tldw_chatbook.Chat.console_command_grammar import (
    COMMAND_PREFIX,
    ConsoleCommand,
    ConsoleCommandRegistry,
    default_console_registry,
)
from tldw_chatbook.Chat.console_command_suggestions import (
    COMMAND_DESCRIPTION_FALLBACK,
    completion_context_for_draft,
    suggestions_for_draft,
)
from tldw_chatbook.Chat.console_skill_resolver import SkillCommandCandidate

SKILLS = (
    SkillCommandCandidate(name="web-search", description="Search the web"),
    SkillCommandCandidate(name="summarize", description="Summarize text"),
)


def _labels(result):
    return [s.label for s in result]


COMMANDS = [
    "/prompt",
    "/system",
    "/skills",
    "/fewer-permission-prompts",
    "/prefill",
    "/generate-image",
    "/generate-video",
    "/stream-video",
    "/steer",
    "/redirect",
    "/emergency-stop",
    "/rewind",
    "/research",
]


def test_bare_slash_lists_commands_then_skills():
    result = suggestions_for_draft("/", default_console_registry(), SKILLS)
    assert _labels(result) == COMMANDS + ["/skills web-search", "/skills summarize"]


def test_prefix_filters_case_insensitively():
    result = suggestions_for_draft("/SK", default_console_registry(), SKILLS)
    assert _labels(result) == ["/skills"]


def test_skill_entries_complete_to_skills_invocation():
    # Bare `/skill-name` is not dispatchable (fallback resolver removed), so
    # skill entries insert the canonical `/skills <name> ` form.
    result = suggestions_for_draft("/w", default_console_registry(), SKILLS)
    assert _labels(result) == ["/skills web-search"]
    assert result[0].insert_text == "/skills web-search "
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
    assert (
        suggestions_for_draft(
            "/skills web-search extra", default_console_registry(), SKILLS
        )
        is None
    )


def test_skill_named_like_a_command_is_deduplicated():
    skills = (SkillCommandCandidate(name="prompt", description="clash"),)
    result = suggestions_for_draft("/", default_console_registry(), skills)
    assert _labels(result) == COMMANDS


def test_trailing_newline_leaves_completion_contexts():
    registry = default_console_registry()
    assert suggestions_for_draft("/sy\n", registry, SKILLS) is None
    assert suggestions_for_draft("/skills \n", registry, SKILLS) is None


def test_max_results_caps_both_modes():
    skills = tuple(SkillCommandCandidate(name=f"skill-{i:03d}") for i in range(300))
    registry = default_console_registry()
    command_result = suggestions_for_draft("/s", registry, skills)
    assert len(command_result) == 200  # 2 matching commands + skills, capped
    arg_result = suggestions_for_draft("/skills skill", registry, skills)
    assert len(arg_result) == 200
    capped = suggestions_for_draft("/skills skill", registry, skills, max_results=5)
    assert len(capped) == 5


# --- TX-05 (TASK-2154.12): jargon-free, never-empty descriptions ----------


def test_builtin_command_descriptions_are_plain_language():
    result = suggestions_for_draft("/", default_console_registry(), ())
    descriptions = {s.label: s.description for s in result}
    assert descriptions["/prefill"] == "Prepare the start of the assistant's reply"
    # Every listed row carries a description; none is empty, none leaks the
    # old "Arm"/"prefill" jargon.
    for suggestion in result:
        assert suggestion.description.strip(), suggestion.label
    assert "Arm" not in descriptions["/prefill"]
    assert "prefill" not in descriptions["/prefill"].lower()


def test_no_builtin_command_falls_back_to_the_extension_description():
    """`COMMAND_DESCRIPTION_FALLBACK` is for registrations we do not ship.

    `console_command_suggestions` documents the fallback as reachable "only
    [by] non-built-in registrations (extensions, test doubles): every
    built-in has an entry above". Nothing pinned that, so `/generate-video`
    (task-3401.5) and `/stream-video` (task-3401.11) shipped rendering
    "Custom command" in the popup. Assert the invariant itself rather than a
    list of names, so the next built-in cannot repeat it.
    """
    result = suggestions_for_draft("/", default_console_registry(), ())
    undescribed = [
        s.label for s in result if s.description == COMMAND_DESCRIPTION_FALLBACK
    ]
    assert undescribed == []


def test_registered_command_without_dict_entry_gets_fallback_description():
    """A command registered outside the built-in set has no
    ``_COMMAND_DESCRIPTIONS`` entry; the popup must not render an empty
    description for it (TX-05)."""
    registry = ConsoleCommandRegistry()
    registry.register(
        ConsoleCommand(name="frobnicate", argument_hint="", handler_id="frob")
    )
    result = suggestions_for_draft(f"{COMMAND_PREFIX}f", registry, ())
    assert [s.label for s in result] == ["/frobnicate"]
    assert result[0].description == "Custom command"


def test_skill_rows_fall_back_when_snapshot_description_is_empty():
    """``SkillCommandCandidate.description`` defaults to ""; the popup must
    still never render an empty description (TX-05)."""
    skills = (SkillCommandCandidate(name="mystery"),)
    command_mode = suggestions_for_draft("/m", default_console_registry(), skills)
    assert command_mode[0].description == "Run this skill"
    arg_mode = suggestions_for_draft("/skills m", default_console_registry(), skills)
    assert arg_mode[0].description == "Run this skill"


# ---------------------------------------------------------------------------
# completion_context_for_draft (TASK-24416): the pure context/prefix parser
# the screen keys popup etiquette on. Direct unit coverage for every
# classification branch, alongside the mounted etiquette tests.
# ---------------------------------------------------------------------------


def test_context_bare_slash_is_command_mode_with_empty_prefix():
    assert completion_context_for_draft("/") == ("command", "")


def test_context_filtered_command_token_carries_its_prefix():
    assert completion_context_for_draft("/pro") == ("command", "pro")
    # No space yet: even a full command name is still command MODE with its
    # own name as the prefix.
    assert completion_context_for_draft("/skills") == ("command", "skills")


def test_context_skills_arg_mode_and_its_prefixes():
    assert completion_context_for_draft("/skills ") == ("skills_arg", "")
    assert completion_context_for_draft("/skills web") == ("skills_arg", "web")
    # The arg separator is case-insensitive on the command name (the
    # suggestion pattern is), and the prefix is reported as typed -- the
    # caller owns any normalization.
    assert completion_context_for_draft("/SKILLS Web") == ("skills_arg", "Web")


def test_context_none_outside_completion_contexts():
    assert completion_context_for_draft("") is None
    assert completion_context_for_draft("hello") is None
    # Not anchored at the start.
    assert completion_context_for_draft("hello /prompt") is None
    # A space ends the bare-command token (argument territory).
    assert completion_context_for_draft("/prompt foo") is None
    # A trailing newline (Shift+Enter multiline draft) leaves the context --
    # the pattern's `\Z` (not `$`) exists for exactly this.
    assert completion_context_for_draft("/prompt\n") is None
