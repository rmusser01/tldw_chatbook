"""Console /research flag parsing (task-16793)."""

import pytest

from tldw_chatbook.UI.Console_Modules.research_command import (
    ResearchCommandIntent,
    parse_research_command,
)


def test_plain_question():
    intent = parse_research_command("what is RAG?")
    assert intent.question == "what is RAG?"
    assert intent.source_policy == "balanced"
    assert intent.providers is None


def test_policy_flag():
    intent = parse_research_command("--policy academic_only why do neurons die?")
    assert intent.source_policy == "academic_only"
    assert intent.question == "why do neurons die?"


def test_providers_flag_with_categories():
    intent = parse_research_command(
        "--policy web_first --providers biomedical,zenodo tau aggregation"
    )
    assert intent.source_policy == "web_first"
    assert intent.providers == ["biomedical", "zenodo"]
    assert intent.question == "tau aggregation"


def test_flags_after_question():
    intent = parse_research_command("how do gnn work --policy web_only")
    assert intent.source_policy == "web_only"
    assert intent.question == "how do gnn work"


def test_invalid_policy_rejected():
    with pytest.raises(ValueError, match="policy"):
        parse_research_command("--policy sideways a question")


def test_empty_question_rejected():
    with pytest.raises(ValueError, match="question"):
        parse_research_command("--policy web_only")


def test_intent_shape():
    intent = ResearchCommandIntent(question="q", source_policy="balanced")
    assert intent.provider_overrides() is None
    intent = ResearchCommandIntent(
        question="q", source_policy="web_first", providers=["biomedical"]
    )
    assert intent.provider_overrides() == {"academic_providers": ["biomedical"]}


def test_dangling_flag_token_is_a_usage_error():
    with pytest.raises(ValueError, match="--policy needs a value"):
        parse_research_command("--policy")  # flag with no value, no question
    with pytest.raises(ValueError, match="--providers needs a value"):
        parse_research_command("a question --providers")  # dangling at end


def test_oversized_args_rejected_by_shared_validator():
    with pytest.raises(ValueError, match="too long"):
        parse_research_command("x" * 5000 + " --policy web_only")
