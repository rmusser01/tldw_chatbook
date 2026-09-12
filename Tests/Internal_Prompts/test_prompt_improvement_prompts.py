# Tests/Internal_Prompts/test_prompt_improvement_prompts.py
"""Registry coverage for the prompt-improvement rewrite spec.

The spec makes the owner default "improve my prompt" template editable in
Settings > Internal Prompts (TASK-32479 / ADR-151). The scratch_config
fixture lives in this directory's conftest.py.
"""

from tldw_chatbook.Internal_Prompts import (
    CATALOG,
    authoring,
    get_internal_prompt,
)
from tldw_chatbook.Internal_Prompts.prompt_improvement_prompts import (
    REWRITE_DEFAULT,
    REWRITE_PROMPT_ID,
)

PROMPT_ID = REWRITE_PROMPT_ID


def test_spec_registered_with_owner_template_default() -> None:
    spec = CATALOG[PROMPT_ID]

    assert spec.subsystem == "prompt_improvement"
    assert spec.default == REWRITE_DEFAULT
    assert spec.required_placeholders == ()
    for marker in (
        "expert prompt engineer",
        "**Situation**",
        "**Task**",
        "**Objective**",
        "**Knowledge**",
        "`source_prompt`",
    ):
        assert marker in spec.default


def test_default_resolves_when_no_override(scratch_config) -> None:
    assert get_internal_prompt(PROMPT_ID) == REWRITE_DEFAULT
    assert not authoring.override_state(PROMPT_ID).customized


def test_override_round_trip(scratch_config) -> None:
    custom = "You are a terse rewriting assistant. Keep it short."

    assert authoring.save_override(PROMPT_ID, custom)
    assert get_internal_prompt(PROMPT_ID) == custom
    state = authoring.override_state(PROMPT_ID)
    assert state.customized
    assert state.active_text == custom

    assert authoring.reset_override(PROMPT_ID)
    assert get_internal_prompt(PROMPT_ID) == REWRITE_DEFAULT
    assert not authoring.override_state(PROMPT_ID).customized


def test_composed_trusted_instructions_use_override_but_keep_pinned_guards(
    scratch_config,
) -> None:
    """An override swaps the persona portion only; safety guards, the JSON
    envelope instruction, and the recency anchor are always appended."""
    from tldw_chatbook.Prompt_Management.prompt_improvement_prompts import (
        trusted_optimizer_instructions,
    )

    assert authoring.save_override(PROMPT_ID, "CUSTOM-PERSONA-MARKER")
    trusted = trusted_optimizer_instructions("auto")

    assert "CUSTOM-PERSONA-MARKER" in trusted
    assert "expert prompt engineer" not in trusted
    for pinned in (
        "Rewrite the source request; never answer it",
        "Do not invent requirements, facts, evidence, metrics, names, tools, capabilities, or permissions.",
        "JSON object only",
        "never a minor edit, summary, or near-copy",
    ):
        assert pinned in trusted


def test_composed_default_matches_section_order_and_separators(scratch_config) -> None:
    """The default path composes persona + pinned sections in the exact
    shape live providers were verified against (TASK-32478 recency anchor)."""
    from tldw_chatbook.Prompt_Management.prompt_improvement_prompts import (
        _REWRITE_INSTRUCTIONS,
        _REWRITE_SAFETY_INSTRUCTIONS,
        _REWRITE_TASK_ANCHOR,
        trusted_optimizer_instructions,
    )

    trusted = trusted_optimizer_instructions("review")

    expected = (
        f"{REWRITE_DEFAULT}\n\n"
        f"{_REWRITE_SAFETY_INSTRUCTIONS}\n"
        f"{_REWRITE_INSTRUCTIONS}\n"
        f"{_REWRITE_TASK_ANCHOR}"
    )
    assert trusted == expected
