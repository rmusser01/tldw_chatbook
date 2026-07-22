# Tests/Internal_Prompts/test_document_generation_prompt_parity.py
"""Registry defaults must match the document-generation sources byte-for-
byte. The *_system literals below are verbatim copies of the one-line system
prompts hardcoded in Chat/document_generator.py (generate_timeline:219,
generate_study_guide:317, generate_briefing:415). The *_user defaults are
asserted against config.py's DEFAULT_CONFIG_FROM_TOML (the shipped TOML
[prompts.document_generation.<type>].prompt values) rather than the shorter
inline dict literals in DocumentGenerator.__init__ — see the module
docstring on document_generation_prompts.py for why the TOML text is the
canonical default. config is imported inside the test functions only, never
at module scope (Internal_Prompts must stay off the config import chain)."""

from tldw_chatbook.Internal_Prompts import CATALOG

ORIGINAL_TIMELINE_SYSTEM = "You are an expert at creating clear, chronological timelines from conversations and content."
ORIGINAL_STUDY_GUIDE_SYSTEM = "You are an educational expert specializing in creating comprehensive study guides."
ORIGINAL_BRIEFING_SYSTEM = "You are an expert at creating executive briefing documents with actionable insights."


def test_timeline_system_matches_source_literal():
    assert (
        CATALOG["document_generation.timeline_system"].default
        == ORIGINAL_TIMELINE_SYSTEM
    )


def test_study_guide_system_matches_source_literal():
    assert (
        CATALOG["document_generation.study_guide_system"].default
        == ORIGINAL_STUDY_GUIDE_SYSTEM
    )


def test_briefing_system_matches_source_literal():
    assert (
        CATALOG["document_generation.briefing_system"].default
        == ORIGINAL_BRIEFING_SYSTEM
    )


def test_timeline_user_matches_shipped_toml_default():
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    assert (
        CATALOG["document_generation.timeline_user"].default
        == DEFAULT_CONFIG_FROM_TOML["prompts"]["document_generation"]["timeline"]["prompt"]
    )


def test_study_guide_user_matches_shipped_toml_default():
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    assert (
        CATALOG["document_generation.study_guide_user"].default
        == DEFAULT_CONFIG_FROM_TOML["prompts"]["document_generation"]["study_guide"]["prompt"]
    )


def test_briefing_user_matches_shipped_toml_default():
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    assert (
        CATALOG["document_generation.briefing_user"].default
        == DEFAULT_CONFIG_FROM_TOML["prompts"]["document_generation"]["briefing"]["prompt"]
    )


def test_user_prompts_differ_from_document_generator_inline_dict_fallback():
    # Pins the TOML-canonical decision: the registry defaults are the RICHER
    # shipped TOML strings, not the shorter inline dict literals at
    # document_generator.py:70-95 (which only fire when the whole
    # [prompts.document_generation.<type>] table is absent).
    inline_fallbacks = {
        "document_generation.timeline_user": (
            "Create a detailed text-based timeline based on our "
            "conversation/materials being referenced."
        ),
        "document_generation.study_guide_user": (
            "Create a detailed and well produced study guide based on the "
            "current focus of our conversation/materials in reference."
        ),
        "document_generation.briefing_user": (
            "Create a detailed and well produced executive briefing document "
            "regarding this conversation and the subject material."
        ),
    }
    for prompt_id, inline_literal in inline_fallbacks.items():
        assert CATALOG[prompt_id].default != inline_literal
        assert CATALOG[prompt_id].default.startswith(inline_literal)


def test_user_prompts_have_matching_legacy_config_path():
    expected = {
        "document_generation.timeline_user": "prompts.document_generation.timeline.prompt",
        "document_generation.study_guide_user": "prompts.document_generation.study_guide.prompt",
        "document_generation.briefing_user": "prompts.document_generation.briefing.prompt",
    }
    for prompt_id, path in expected.items():
        assert CATALOG[prompt_id].legacy_config_path == path


def test_no_placeholders_on_any_of_the_six_specs():
    for suffix in (
        "timeline_system",
        "timeline_user",
        "study_guide_system",
        "study_guide_user",
        "briefing_system",
        "briefing_user",
    ):
        spec = CATALOG[f"document_generation.{suffix}"]
        assert spec.required_placeholders == ()
        assert spec.optional_placeholders == ()
