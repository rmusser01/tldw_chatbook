# tldw_chatbook/Internal_Prompts/summarization_prompts.py
"""Summarization prompt specs. Defaults moved verbatim from
LLM_Calls/Summarization_General_Lib.py (analyze()'s default system_message),
LLM_Calls/Local_Summarization_Lib.py (module constant summarizer_prompt), and
Chunking/Chunk_Lib.py (the "summarize" chunk method's system_prompt_content
fallback). Parity tests compare the registry defaults against literals copied
from source pre-migration. Note: rolling_summarize_system's legacy key
[chunking_config].summarize_system_prompt has no entry in the shipped default
TOML, so _shipped_default_for returns None and any user-set value is treated
as a customization (the intended behavior)."""

from .catalog import PromptSpec, register

register(
    PromptSpec(
        id="summarization.analyze_default_system",
        subsystem="summarization",
        title="Bulleted notes summarizer — default system prompt",
        description=(
            "Default system prompt used for LLM-based summarization when no "
            "custom system message is supplied."
        ),
        used_in="LLM_Calls/Summarization_General_Lib.py (analyze())",
        default='You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.\n**Bulleted Note Creation Guidelines**\n\n**Headings**:\n- Based on referenced topics, not categories like quotes or terms\n- Surrounded by **bold** formatting\n- Not listed as bullet points\n- No space between headings and list items underneath\n\n**Emphasis**:\n- **Important terms** set in bold font\n- **Text ending in a colon**: also bolded\n\n**Review**:\n- Ensure adherence to specified format\n- Do not reference these instructions in your response.',
        contract_note=(
            "'Based on the content between backticks' refers to the "
            "embedded ``` block — keep them together."
        ),
    )
)

register(
    PromptSpec(
        id="summarization.local_summarizer_template",
        subsystem="summarization",
        title="Local-backend summarizer prompt (legacy Ollama template)",
        description=(
            "Module-level bulleted-notes prompt concatenated ahead of the "
            "input text for local inference backends."
        ),
        used_in=(
            "LLM_Calls/Local_Summarization_Lib.py (module constant "
            "summarizer_prompt; concatenated with input text in "
            "summarize_with_llama, summarize_with_kobold, "
            "summarize_with_tabbyapi, summarize_with_custom_openai, and "
            "summarize_with_custom_openai_2)"
        ),
        default='\n                    You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.\n                        **Bulleted Note Creation Guidelines**\n\n                        **Headings**:\n                        - Based on referenced topics, not categories like quotes or terms\n                        - Surrounded by **bold** formatting\n                        - Not listed as bullet points\n                        - No space between headings and list items underneath\n\n                        **Emphasis**:\n                        - **Important terms** set in bold font\n                        - **Text ending in a colon**: also bolded\n\n                        **Review**:\n                        - Ensure adherence to specified format\n                        - Do not reference these instructions in your response.\n                    ',
        contract_note=(
            "The leftover Ollama-modelfile cruft (the <s>/</s> sentinel tags "
            "and the trailing {{ .Prompt }} token) was removed in task 452 — "
            "it was sent verbatim to models but was never intended prompt "
            "content. Callers concatenate this ahead of the input text."
        ),
    )
)

register(
    PromptSpec(
        id="summarization.rolling_summarize_system",
        subsystem="summarization",
        title="Rolling summarize — default system prompt",
        description=(
            "Default system prompt for the chunk-by-chunk rolling "
            "summarization strategy."
        ),
        used_in="Chunking/Chunk_Lib.py (chunk_text's \"summarize\" method, via _rolling_summarize)",
        default="Rewrite this text in summarized form.",
        legacy_config_path="chunking_config.summarize_system_prompt",
        applies="process restart (frozen at Chunk_Lib import; overrides apply on next app start for default-option callers)",
        contract_note=(
            "This legacy key has no entry in the shipped default TOML, so "
            "_shipped_default_for returns None and any user-set value at "
            "[chunking_config] summarize_system_prompt counts as "
            "customized — intentional."
        ),
    )
)
