# tldw_chatbook/Internal_Prompts/console_prompts.py
"""Native Console prompt specs.

Currently registers the `/rewind` "Summarize up to here" instruction (SP2):
the system prompt used when compacting an earlier portion of a Console
conversation into a carried-forward context summary. The transcript span is
passed as the user message at call time (see
``Chat/console_chat_controller.py::summarize_up_to``), so this prompt takes no
placeholders. Editable in Settings like any other internal prompt.
"""

from .catalog import PromptSpec, register

register(
    PromptSpec(
        id="console.rewind_summarize",
        subsystem="console",
        title="Console rewind summary",
        description=(
            "Summarizes earlier conversation turns into carried-forward "
            "context for the Console /rewind 'Summarize up to here' action."
        ),
        used_in="Chat/console_chat_controller.py (summarize_up_to)",
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
