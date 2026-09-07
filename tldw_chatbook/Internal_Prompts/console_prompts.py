# tldw_chatbook/Internal_Prompts/console_prompts.py
"""Native Console prompt specs.

Registers the shared Console conversation-summary instruction used by both
automatic conversation-memory compaction and `/rewind` "Summarize up to here",
plus the user-facing note variant used by the Console message More-menu
"Summarize up to here as note" action. The transcript span is passed as
untrusted user data at call time, so these prompts take no placeholders.
Editable in Settings like any other internal prompt.
"""

from .catalog import PromptSpec, register

register(
    PromptSpec(
        id="console.rewind_summarize",
        subsystem="console",
        title="Console conversation summary",
        description=(
            "Summarizes earlier conversation turns into carried-forward "
            "memory for automatic compaction and the Console /rewind action."
        ),
        used_in=(
            "Chat/console_chat_controller.py (automatic compaction and summarize_up_to)"
        ),
        default=(
            "You are compacting an earlier portion of a conversation into "
            "carried-forward context. Summarize the transcript below into a "
            "compact, factual briefing the assistant can rely on to continue "
            "the conversation. Preserve every decision that was made, the key "
            "facts, names, and values, any preferences the user expressed, and "
            "the current state of the task in progress. Omit greetings, "
            "acknowledgements, and other pleasantries. Write plain prose with "
            "no headers, lists, or preamble. Keep it as short as the content "
            "allows."
        ),
        contract_note=(
            "The conversation transcript to summarize is supplied as the user "
            "message at call time; this prompt is the system instruction and "
            "takes no placeholders."
        ),
    )
)

register(
    PromptSpec(
        id="console.summarize_note",
        subsystem="console",
        title="Console summarize-to-note summary",
        description=(
            "Summarizes the conversation up to a selected message into a "
            "readable note saved to the notes library via the Console "
            "message More-menu action."
        ),
        used_in=(
            "Chat/console_chat_controller.py (summarize_span_as_note)"
        ),
        default=(
            "You are summarizing a conversation transcript for the user's "
            "own reference; the summary will be saved as a note. Write a "
            "clear, factual account of what was discussed and decided: the "
            "user's goal, the key questions and answers, decisions made, "
            "important facts, names, and values, and where things stand at "
            "the end of the transcript. Use plain prose or short bullets as "
            "content demands. Omit greetings and pleasantries. Do not "
            "invent details that are not in the transcript."
        ),
        contract_note=(
            "The conversation transcript to summarize is supplied as the "
            "user message at call time; this prompt is the system "
            "instruction and takes no placeholders."
        ),
    )
)
