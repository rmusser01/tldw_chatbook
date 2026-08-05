# tldw_chatbook/Internal_Prompts/document_generation_prompts.py
"""Document-generation prompt specs (timeline / study guide / briefing).

The 3 *_system prompts are the one-line literals hardcoded in
Chat/document_generator.py (generate_timeline:219, generate_study_guide:317,
generate_briefing:415) — copied verbatim, never read from config.

The 3 *_user prompts are the "instruction" half of each document type's
config-driven prompt. Their canonical default is the SHIPPED TOML string in
config.py's CONFIG_TOML_CONTENT (~2824-2837: [prompts.document_generation.
timeline]/.study_guide/.briefing, the `prompt = "..."` value), NOT the
shorter inline dict literal at document_generator.py:70-95. Rationale: on
first run, config.py writes CONFIG_TOML_CONTENT to every user's config.toml,
so the richer TOML text is what real users actually run; the inline dict in
DocumentGenerator.__init__ only fires via get_cli_setting's default argument
when the whole [prompts.document_generation.<type>] table is absent from the
user's file entirely (never true after first run). Each *_user spec carries
legacy_config_path="prompts.document_generation.<type>.prompt" so the
resolver's differs-from-shipped rule treats a user-edited TOML value as an
override and an untouched one as the (TOML-sourced) default.

Zero placeholders on any of the six: document_generator.py builds the final
user prompt by concatenating this text with "\n\nConversation Context:\n"
plus the formatted conversation in code — not via token substitution.
"""

from .catalog import PromptSpec, register

register(
    PromptSpec(
        id="document_generation.timeline_system",
        subsystem="document_generation",
        title="Timeline generator — system prompt",
        description="System prompt for generating a chronological timeline from a conversation.",
        used_in="Chat/document_generator.py (DocumentGenerator.generate_timeline, system_prompt)",
        default="You are an expert at creating clear, chronological timelines from conversations and content.",
    )
)

register(
    PromptSpec(
        id="document_generation.timeline_user",
        subsystem="document_generation",
        title="Timeline generator — user instruction",
        description="Instruction text prefixed to the conversation context when generating a timeline.",
        used_in=(
            "Chat/document_generator.py (DocumentGenerator.generate_timeline, "
            "user_prompt via self.timeline_config['prompt']); shipped default "
            "from config.py CONFIG_TOML_CONTENT [prompts.document_generation.timeline].prompt"
        ),
        default=(
            "Create a detailed text-based timeline based on our "
            "conversation/materials being referenced. Include key dates, "
            "events, and their relationships in chronological order."
        ),
        legacy_config_path="prompts.document_generation.timeline.prompt",
        contract_note=(
            "Code appends \"\\n\\nConversation Context:\\n{context}\" after "
            "this text in Python (string concatenation, not token "
            "substitution) — no placeholders here."
        ),
    )
)

register(
    PromptSpec(
        id="document_generation.study_guide_system",
        subsystem="document_generation",
        title="Study guide generator — system prompt",
        description="System prompt for generating a study guide from a conversation.",
        used_in="Chat/document_generator.py (DocumentGenerator.generate_study_guide, system_prompt)",
        default="You are an educational expert specializing in creating comprehensive study guides.",
    )
)

register(
    PromptSpec(
        id="document_generation.study_guide_user",
        subsystem="document_generation",
        title="Study guide generator — user instruction",
        description="Instruction text prefixed to the conversation context when generating a study guide.",
        used_in=(
            "Chat/document_generator.py (DocumentGenerator.generate_study_guide, "
            "user_prompt via self.study_guide_config['prompt']); shipped default "
            "from config.py CONFIG_TOML_CONTENT [prompts.document_generation.study_guide].prompt"
        ),
        default=(
            "Create a detailed and well produced study guide based on the "
            "current focus of our conversation/materials in reference. "
            "Include key concepts, definitions, learning objectives, and "
            "potential exam questions."
        ),
        legacy_config_path="prompts.document_generation.study_guide.prompt",
        contract_note=(
            "Code appends \"\\n\\nConversation Context:\\n{context}\" after "
            "this text in Python (string concatenation, not token "
            "substitution) — no placeholders here."
        ),
    )
)

register(
    PromptSpec(
        id="document_generation.briefing_system",
        subsystem="document_generation",
        title="Briefing generator — system prompt",
        description="System prompt for generating an executive briefing from a conversation.",
        used_in="Chat/document_generator.py (DocumentGenerator.generate_briefing, system_prompt)",
        default="You are an expert at creating executive briefing documents with actionable insights.",
    )
)

register(
    PromptSpec(
        id="document_generation.briefing_user",
        subsystem="document_generation",
        title="Briefing generator — user instruction",
        description="Instruction text prefixed to the conversation context when generating a briefing.",
        used_in=(
            "Chat/document_generator.py (DocumentGenerator.generate_briefing, "
            "user_prompt via self.briefing_config['prompt']); shipped default "
            "from config.py CONFIG_TOML_CONTENT [prompts.document_generation.briefing].prompt"
        ),
        default=(
            "Create a detailed and well produced executive briefing document "
            "regarding this conversation and the subject material. Include "
            "key points, actionable insights, strategic implications, and "
            "recommendations."
        ),
        legacy_config_path="prompts.document_generation.briefing.prompt",
        contract_note=(
            "Code appends \"\\n\\nConversation Context:\\n{context}\" after "
            "this text in Python (string concatenation, not token "
            "substitution) — no placeholders here."
        ),
    )
)
