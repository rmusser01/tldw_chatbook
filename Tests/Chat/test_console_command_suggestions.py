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
    "/stop",
    "/emergency-stop",
    "/rewind",
    "/research",
    "/help",
    "/doctor",
    "/model",
    "/sessions",
    "/workspace",
    "/new",
    "/temp",
    "/settings",
    "/context",
    "/endpoint",
    "/goal",
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
    # `/web` uniquely prefixes the web-search skill (no command starts with
    # "web"); `/w` alone now also matches the /workspace action command.
    result = suggestions_for_draft("/web", default_console_registry(), SKILLS)
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
    # `/my` uniquely prefixes the skill; `/m` now also matches /model.
    command_mode = suggestions_for_draft("/my", default_console_registry(), skills)
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


def test_goal_command_completion_preserves_description_mode():
    result = suggestions_for_draft("/go", default_console_registry(), SKILLS)
    assert _labels(result) == ["/goal"]
    assert (
        suggestions_for_draft(
            "/goal repair fixture", default_console_registry(), SKILLS
        )
        is None
    )

# ---------------------------------------------------------------------------
# TASK-18921: usage ranking (display-only; matching and filtering unchanged)
# ---------------------------------------------------------------------------

from tldw_chatbook.Chat.console_command_suggestions import (
    CommandSuggestion,
    count_suggestion_usage,
    rank_suggestions_by_usage,
    suggestion_usage_key,
)


def _suggestion(label: str, insert_text: str | None = None) -> CommandSuggestion:
    return CommandSuggestion(
        insert_text=insert_text if insert_text is not None else f"{label} ",
        label=label,
    )


def test_usage_keys_for_every_row_kind():
    # Command row.
    assert suggestion_usage_key(_suggestion("/model")) == "command:model"
    # The /skills command row itself is a command, not a skill.
    assert suggestion_usage_key(_suggestion("/skills")) == "command:skills"
    # Command-mode skill row carries the skill identity.
    assert (
        suggestion_usage_key(_suggestion("/skills web-search", "/skills web-search "))
        == "skill:web-search"
    )
    # Skills-arg mode rows are bare names.
    assert suggestion_usage_key(_suggestion("web-search")) == "skill:web-search"
    # Keys are case-folded.
    assert suggestion_usage_key(_suggestion("/Model")) == "command:model"


def test_count_mines_slash_commands_and_skills_invocations():
    counts = count_suggestion_usage(
        [
            "/model gpt-5",
            "/stop",
            "/model",
            "/skills web-search",
            "/SKILLS Summarize",
            "an ordinary prompt",
            "",
        ]
    )
    assert counts["command:model"] == 2
    assert counts["command:stop"] == 1
    assert counts["command:skills"] == 2
    assert counts["skill:web-search"] == 1
    assert counts["skill:summarize"] == 1
    # No other keys: ordinary prompts carry no command counts.
    assert set(counts) == {
        "command:model",
        "command:stop",
        "command:skills",
        "skill:web-search",
        "skill:summarize",
    }


def test_count_mines_dollar_mentions_once_per_text():
    counts = count_suggestion_usage(
        [
            "run $web-search then $web-search again",
            "and $summarize too",
            "$HOME and $PATH stay literal (no skill carries these keys)",
        ]
    )
    assert counts["skill:web-search"] == 1
    assert counts["skill:summarize"] == 1
    # Non-skill sigil tokens count toward keys no suggestion ever carries.
    assert counts["skill:home"] == 1
    assert counts["skill:path"] == 1


def test_count_ignores_leading_sigil_inside_command_texts():
    # A slash draft is a command count; a $ inside it never double-counts a
    # skill (the mention scan only runs on non-slash texts).
    counts = count_suggestion_usage(["/prompt use $web-search now"])
    assert counts == {"command:prompt": 1}


def test_rank_with_no_usage_data_returns_input_order():
    rows = [_suggestion("/prompt"), _suggestion("/model"), _suggestion("/skills a")]
    # Empty counts: byte-identical order.
    assert rank_suggestions_by_usage(rows, {}) == rows
    # Counts that cover none of the set's keys: still today's order.
    assert rank_suggestions_by_usage(rows, {"command:doctor": 9}) == rows
    # All-zero counts for the set's keys: still today's order.
    assert rank_suggestions_by_usage(rows, {"command:model": 0}) == rows


def test_rank_most_used_first_with_alphabetical_ties():
    rows = [
        _suggestion("/prompt"),
        _suggestion("/model"),
        _suggestion("/help"),
        _suggestion("/stop"),
        _suggestion("/skills web-search"),
    ]
    ranked = rank_suggestions_by_usage(
        rows, {"command:stop": 1, "command:model": 3, "skill:web-search": 2}
    )
    assert [s.label for s in ranked] == [
        "/model",
        "/skills web-search",
        "/stop",
        "/help",  # zero-count rows tie-break alphabetically
        "/prompt",
    ]


def test_rank_is_a_permutation_of_the_filtered_set():
    rows = [
        _suggestion("/prompt"),
        _suggestion("/model"),
        _suggestion("web-search"),
        _suggestion("summarize"),
    ]
    ranked = rank_suggestions_by_usage(
        rows, {"skill:summarize": 4, "command:prompt": 1, "skill:web-search": 1}
    )
    assert sorted(r.label for r in ranked) == sorted(r.label for r in rows)
    # The most-used row leads.
    assert ranked[0].label == "summarize"


def test_ranked_suggestions_keep_prefix_filtering_semantics():
    # Filtering happens upstream and is untouched: ranking a filtered set
    # never reintroduces rows the filter dropped.
    registry = default_console_registry()
    filtered = suggestions_for_draft("/s", registry, SKILLS)
    ranked = rank_suggestions_by_usage(
        filtered, {"command:settings": 5, "command:prompt": 5, "skill:nope": 5}
    )
    assert {s.label for s in ranked} == {s.label for s in filtered}
    assert ranked[0].label == "/settings"  # /s filter + usage beats registry order
