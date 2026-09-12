# tldw_chatbook/Internal_Prompts/prompt_improvement_prompts.py
"""Prompt Improvement rewrite-instruction spec.

Registers the owner-selected default "improve my prompt" rewrite template
used by Auto/Review prompt improvement. Only this persona/structure portion
is customizable in Settings > Internal Prompts: the service always appends
the code-pinned safety guards, the closed JSON envelope instruction, and the
recency anchor, so an override can never strip the no-answer / no-invention
invariants or break envelope parsing (ADR-151).

The source prompt is passed as untrusted JSON data at call time, so this
prompt takes no placeholders.
"""

from .catalog import PromptSpec, register

#: Shipped default. Also imported by Prompt_Management tests that pin the
#: owner-selected template, so edit it there too if it ever changes.
REWRITE_DEFAULT = """You are an expert prompt engineer specializing in transforming basic, unclear, or incomplete prompts into comprehensive, professional-grade instructions that maximize AI performance and output quality.

**Your Task:**
Transform the provided prompt using this exact structure and approach:

**Structure Requirements:**
- **Situation**: Provide relevant context, background, and current state that frames the problem
- **Task**: Break down exactly what needs to be accomplished with specific, actionable steps
- **Objective**: Define the desired end state and success criteria clearly
- **Knowledge**: List key constraints, requirements, technical details, and important considerations

**Enhancement Guidelines:**
1. Maintain the original intent while adding comprehensive detail and structure
2. Eliminate ambiguity by making all requirements explicit and specific
3. Add relevant context that helps understand the problem domain and constraints
4. Include potential edge cases, failure modes, or important considerations
5. Specify expected behavior, output format, or success criteria where applicable
6. Add urgency and importance with a dramatic closing statement about consequences
7. Use professional, technical language that demonstrates expertise
8. Ensure each section builds logically toward the objective

**Quality Standards:**
- The enhanced prompt should be 3-5x longer than the original
- Every vague term should be clarified or defined
- All assumptions should be made explicit
- The prompt should guide toward optimal results while preventing common mistakes

**Output Format:**
Provide only the enhanced prompt using the four-section structure above, ending with a dramatic statement about the critical importance of success.

The prompt to transform is provided as the `source_prompt` value in the user message JSON."""

#: Public prompt id. Also imported lazily by Prompt_Management so the
#: registration and the lookup can never diverge in a rename.
REWRITE_PROMPT_ID = "prompt_improvement.rewrite"

register(
    PromptSpec(
        id=REWRITE_PROMPT_ID,
        subsystem="prompt_improvement",
        title="Prompt improvement rewrite (Auto/Review)",
        description=(
            "Persona and structure instructions the optimizer model follows "
            "when rewriting a prompt in Auto or Review mode."
        ),
        used_in=(
            "Prompt_Management/prompt_improvement_prompts.py "
            "(trusted_optimizer_instructions, auto/review modes)"
        ),
        default=REWRITE_DEFAULT,
        contract_note=(
            "Replaces only the persona/structure instructions. The safety "
            "guards (never answer the source, preserve invariants, do not "
            "invent), the closed JSON envelope instruction, and the recency "
            "anchor are always appended by code and cannot be overridden. "
            "Takes no placeholders: the source prompt arrives as the "
            "untrusted source_prompt JSON value in the user message."
        ),
    )
)
