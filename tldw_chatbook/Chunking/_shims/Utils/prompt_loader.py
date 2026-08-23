# tldw_chatbook/Chunking/_shims/Utils/prompt_loader.py
"""Replaces tldw_Server_API.app.core.Utils.prompt_loader (spec §5.3).

Server IDs are ("category", "name") pairs; chatbook's resolver keys
are dotted. Two kinds of pairing live here:

- REQUIRED prompts (rolling_summarize) map to a resolver id whose catalog
  default is the effective prompt — verified against both trees.
- OPTIONAL overrides (the proposition profiles, verified at the pin:
  engine/strategies/propositions.py:321/334/347 call
  ``load_prompt("chunking", "proposition_claimify" | "proposition_gemma_aps"
  | "proposition_generic")`` and use the result only ``if override:``) map
  to "" — a known pairing with no shipped override, so the strategy's
  built-in instruction applies, byte-faithful to upstream's
  absent-override behavior. A user override for these would ride the
  Internal_Prompts catalog (nothing ships there today, by the 2026-08-23
  descope scope ruling).

This module lives under the capital-``U`` ``Utils`` package because that is
the dotted path the vendored engine imports unguarded
(engine/strategies/rolling_summarize.py:13).
"""
from ....Internal_Prompts.resolver import get_internal_prompt

_KNOWN = {
    ("chunking", "Rolling Summarization"): "summarization.rolling_summarize_system",
    # optional-override pairings: known, nothing shipped (see module docstring)
    ("chunking", "proposition_claimify"): "",
    ("chunking", "proposition_gemma_aps"): "",
    ("chunking", "proposition_generic"): "",
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
    if not prompt_id:
        # Known pairing whose upstream contract is an OPTIONAL override:
        # "" is the honest answer (no override shipped), and the caller's
        # own built-in default applies.
        return ""
    return get_internal_prompt(prompt_id)
