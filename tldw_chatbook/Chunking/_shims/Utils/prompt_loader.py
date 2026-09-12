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
  to "". Chatbook does not carry the server's Prompts runtime
  (Config_Files/Prompts/chunking.prompts.yaml), so the strategy's
  in-code defaults are chatbook's effective instructions. Upstream at
  the pin DOES ship YAML entries for all three pairs, and for
  claimify/gemma_aps that YAML wording differs from the in-code
  defaults — a recorded divergence, and a candidate for
  Internal_Prompts catalog entries if true parity is ever wanted.
  Because the _KNOWN value is "", the resolver is never consulted for
  these pairs: a user override cannot ride the Internal_Prompts catalog
  today (nothing ships there, by the 2026-08-23 descope scope ruling);
  a future override mechanism changes the map VALUES, not the keys.

This module lives under the capital-``U`` ``Utils`` package because that is
the dotted path the vendored engine imports unguarded
(engine/strategies/rolling_summarize.py:13).
"""
from ....Internal_Prompts.resolver import get_internal_prompt

_KNOWN = {
    ("chunking", "Rolling Summarization"): "summarization.rolling_summarize_system",
    # optional-override pairings: "" = chatbook carries no Prompts runtime,
    # so the engine's in-code defaults are effective (see module docstring
    # for the recorded upstream-YAML divergence)
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
        # Known pairing whose engine contract is an OPTIONAL override:
        # "" means chatbook ships no override, so the caller's own
        # built-in instruction applies (see module docstring).
        return ""
    return get_internal_prompt(prompt_id)
