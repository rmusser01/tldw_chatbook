# Tests/Internal_Prompts/test_summarization_prompt_parity.py
"""Registry defaults must match the summarization-source literals
byte-for-byte. The two multi-line prompts below are verbatim copies of the
pre-migration source (Summarization_General_Lib.py:528-550 and
Local_Summarization_Lib.py:39-56), copied before the migration touched those
files."""

from tldw_chatbook.Internal_Prompts import CATALOG

ORIGINAL_ANALYZE_DEFAULT_SYSTEM = 'You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.\n**Bulleted Note Creation Guidelines**\n\n**Headings**:\n- Based on referenced topics, not categories like quotes or terms\n- Surrounded by **bold** formatting\n- Not listed as bullet points\n- No space between headings and list items underneath\n\n**Emphasis**:\n- **Important terms** set in bold font\n- **Text ending in a colon**: also bolded\n\n**Review**:\n- Ensure adherence to specified format\n- Do not reference these instructions in your response.'

ORIGINAL_LOCAL_SUMMARIZER_TEMPLATE = '\n                    <s>You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.\n                        **Bulleted Note Creation Guidelines**\n\n                        **Headings**:\n                        - Based on referenced topics, not categories like quotes or terms\n                        - Surrounded by **bold** formatting\n                        - Not listed as bullet points\n                        - No space between headings and list items underneath\n\n                        **Emphasis**:\n                        - **Important terms** set in bold font\n                        - **Text ending in a colon**: also bolded\n\n                        **Review**:\n                        - Ensure adherence to specified format\n                        - Do not reference these instructions in your response.</s> {{ .Prompt }}\n                    '


def test_analyze_default_system_matches_source_literal():
    assert (
        CATALOG["summarization.analyze_default_system"].default
        == ORIGINAL_ANALYZE_DEFAULT_SYSTEM
    )


def test_local_summarizer_template_matches_source_literal():
    assert (
        CATALOG["summarization.local_summarizer_template"].default
        == ORIGINAL_LOCAL_SUMMARIZER_TEMPLATE
    )


def test_local_summarizer_template_ships_ollama_cruft_unchanged():
    # Pins the cruft-ships-unchanged decision: the doubled-brace
    # "{{ .Prompt }}" Ollama token is left in place (callers concatenate
    # this template ahead of the input text rather than render it).
    assert (
        "{{ .Prompt }}"
        in CATALOG["summarization.local_summarizer_template"].default
    )
    assert (
        "<s>" in CATALOG["summarization.local_summarizer_template"].default
    )
    assert (
        "</s>" in CATALOG["summarization.local_summarizer_template"].default
    )


def test_rolling_summarize_system_matches_default_literal():
    assert (
        CATALOG["summarization.rolling_summarize_system"].default
        == "Rewrite this text in summarized form."
    )


def test_rolling_summarize_system_legacy_config_path():
    assert (
        CATALOG["summarization.rolling_summarize_system"].legacy_config_path
        == "chunking_config.summarize_system_prompt"
    )
