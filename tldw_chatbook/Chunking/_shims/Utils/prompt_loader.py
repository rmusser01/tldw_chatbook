# tldw_chatbook/Chunking/_shims/Utils/prompt_loader.py
"""Replaces tldw_Server_API.app.core.Utils.prompt_loader (spec §5.3).

Server IDs are ("category", "Human Title") pairs; chatbook's resolver keys
are dotted. The known phase-1 mapping is chunking/Rolling Summarization →
summarization.rolling_summarize_system (verified against both trees).

This module lives under the capital-``U`` ``Utils`` package because that is
the dotted path the vendored engine imports unguarded
(engine/strategies/rolling_summarize.py:13).
"""
from ....Internal_Prompts.resolver import get_internal_prompt

_KNOWN = {
    ("chunking", "Rolling Summarization"): "summarization.rolling_summarize_system",
}


def load_prompt(category: str, name: str) -> str:
    prompt_id = _KNOWN.get((category, name))
    if prompt_id is None:
        # Unknown pairing: raise loudly rather than returning "" (a silent
        # empty system prompt would degrade every downstream LLM call).
        raise KeyError(
            f"No prompt mapping for ('{category}', '{name}'); add it to "
            f"_shims/Utils/prompt_loader._KNOWN or Internal_Prompts."
        )
    return get_internal_prompt(prompt_id)
