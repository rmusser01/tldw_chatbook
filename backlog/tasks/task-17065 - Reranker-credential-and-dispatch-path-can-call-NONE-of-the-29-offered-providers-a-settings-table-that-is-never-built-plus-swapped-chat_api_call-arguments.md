---
id: TASK-17065
title: >-
  Reranker credential and dispatch path can call NONE of the 29 offered
  providers -- a settings table that is never built, plus swapped chat_api_call
  arguments
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-16'
updated_date: '2026-08-17 05:41'
labels:
  - rag
  - settings
  - config
dependencies:
  - TASK-3502
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-3502 AC#1 gave Settings ▸ RAG's Reranking fold a provider Select whose
options are enumerated from `Chat_Functions.API_CALL_HANDLERS` -- 29 rows, the
exact dispatch table `chat_api_call` looks the reranker's `model_provider` up
in, so no newly registered chat provider can silently go missing and no
undispatchable name can be offered. The enumeration is right. What is behind
it is not.

**Verified coverage (measured, not inferred): credential resolution reaches 1
of 29 providers (`deepseek` only), and end-to-end dispatch reaches 0 of 29.**
The earlier filing said "4 of 29"; that was read off the shape of the
`if/elif` chain rather than off a run. There are two independent defects,
stacked.

### Defect 1 -- the reranker reads a settings table `load_settings()` never builds

`BaseReranker.__init__` does `self._settings = load_settings()`
(`RAG_Search/reranker.py:128`). `_call_llm_impl`
(`RAG_Search/reranker.py:183-206`) then resolves the credential from a
hand-rolled `if/elif`:

- `openai` / `anthropic` / `groq` read `self._settings["API"]["<p>_api_key"]`;
- `deepseek` reads `self._settings["api_settings"]["deepseek"]["api_key"]`;
- everything else falls past `# Add other providers as needed` into
  `raise ValueError(f"No API key found for provider: {...}")`.

**The dict `load_settings()` returns has no `"API"` key at all.** `config_dict`
(`config.py:1433`) projects `api_settings`, the per-provider `<provider>_api`
dicts and ~30 other sections, but never the raw `[API]` table (the only three
`"API"` occurrences in `config.py` are local `get_toml_section` reads at
`:1242`, `:1360` and `get_cli_setting` at `:6107`). So three of the four
"covered" branches read `None` unconditionally, for every user, always. Only
the `deepseek` branch reads a table that exists.

No environment variable rescues it: env credentials land in `api_settings` and
the legacy `<provider>_api` dicts (`_normalize_legacy_provider_api_key`,
`config.py:1133-1166`), never in a top-level `"API"` key of the returned dict.

### Defect 2 -- `chat_api_call` is invoked with the wrong positional arguments

Even the one provider whose credential resolves cannot complete a call.
`_call_llm_impl` dispatches positionally (`RAG_Search/reranker.py:228-237`):

```python
response = await loop.run_in_executor(
    None, chat_api_call,
    api_key,                     # -> api_endpoint
    messages_payload,            # -> messages_payload
    self.config.model_provider,  # -> api_key
    self.config.model_name,      # -> temp
    self.config.temperature,     # -> system_message
    self.config.max_tokens,      # -> streaming
)
```

but the real signature is
`chat_api_call(api_endpoint, messages_payload, api_key, temp, system_message,
streaming, minp, maxp, model, ...)` (`Chat/Chat_Functions.py:809-821`). The
API KEY is passed where the ENDPOINT belongs, so the call dies at the routing
step with `Unsupported API endpoint: <the api key>` -- and the key is echoed
into an ERROR log line on the way (`Chat_Functions.py:925/931`), which is its
own disclosure problem.

Why ~2,500 green tests never caught this: every reranker test fakes the seam
with a stub whose parameter list mirrors the CALLER's wrong assumption
(`Tests/RAG_Search/test_reranker_degraded_paths.py:75`:
`def fake_chat_api_call(api_key, messages_payload, provider, model, temp, maxp)`),
so the fake agrees with the bug. A fake at this seam must match the REAL
signature.

### Reproduction (isolated `TLDW_CONFIG_PATH`, no live provider call)

Config containing BOTH the legacy and the modern credential for openai, plus a
deepseek key:

```toml
[API]
openai_api_key = "REDACTED-legacy-openai-value"
[api_settings.openai]
api_key = "REDACTED-modern-openai-value"
[api_settings.deepseek]
api_key = "REDACTED-deepseek-value"
```

```
API section present in load_settings(): False
openai lookup via _call_llm_impl's read: None
api_settings.openai.api_key: REDACTED-modern-openai-value          <- the credential IS loaded
RAW loader sees API.openai_api_key: REDACTED-legacy-openai   <- and so is the legacy one
get_cli_setting('API','openai_api_key'): REDACTED-legacy-openai

openai:    RESULT ValueError No API key found for provider: openai
anthropic: RESULT ValueError No API key found for provider: anthropic
groq:      RESULT ValueError No API key found for provider: groq
deepseek:  RESULT ValueError Unsupported API endpoint: REDACTED-deepseek-value   <- defect 2
```

(`await PointwiseReranker(RerankingConfig(model_provider=p))._call_llm_impl(...)`
for each `p`; the deepseek line is the real `chat_api_call` refusing the key it
was handed as an endpoint.)

A second run with ONLY `[API] openai_api_key` set confirms the fix shape:
`api_settings.openai.api_key == "REDACTED-legacy-only"` -- the loader has already
normalised the legacy value into the modern table.

### Consequences

1. Every reranking-enabled profile has silently no-opped since the feature
   existed. This is PRE-EXISTING and total: before TASK-3502, Settings could
   only ever produce `RerankingConfig()` = openai, and openai is one of the
   three branches that read the phantom table.
2. Selecting any provider produces a hard failure on the first search even
   with a perfectly valid credential configured.
3. Local providers that need no key at all (`ollama`, `llama_cpp`, `vllm`,
   `koboldcpp`, `mlx_lm`, ...) fail for a MISSING KEY they never require.
4. The lookup bypasses the precedence rules CLAUDE.md documents (explicit
   `api_settings.<provider>.api_key` outranks the env var, legacy `[API]`
   lowest, every source validity-checked so a placeholder is never accepted).

TASK-3502 note-(a) shipped the first UI consumer of the reranker's disclosure
tags, so the failure is at least now VISIBLE: the Library RAG results surface
renders "Reranking was skipped (No API key found for provider: X) -- these
results are in their original retrieval order." That is disclosure, not a fix.

### Fix shape

Route the lookup through the config-precedence family CLAUDE.md documents
rather than per-provider hardcoded reads: take the credential from the
normalised `api_settings.<provider>.api_key` table (which the loader has
already filled from env and legacy `[API]`) and pass it through
`resolve_provider_api_key` (`config.py:844`) so a placeholder or a
whitespace-padded value is never accepted. NOTE the shared helper's real
contract: it is a per-VALUE validity check, not a provider→key resolver -- the
lookup itself still has to name the right table. Simply not resolving at all
and letting the provider handler resolve (as `chat_api_call` callers with
`api_key=None` do) is an acceptable variant. Fix defect 2 with keyword
arguments, not a re-ordered positional list.

### Spend consequence -- this fix must land ON TOP of TASK-3502's cost surface

Fixing this converts a silent, disclosed no-op into real provider SPEND: every
user who has ever ticked "Enable reranking" starts paying one extra provider
call per candidate result on their next search, with no further action from
them. That is why TASK-3502 shipped the cost disclosure and the skipped/
degraded notice FIRST and left this open: the disclosure is the prerequisite
groundwork, and this change must not land on a build that lacks it.

### Also in scope: an unrecognised stored provider is mis-displayed and unrepairable (final-review F5)

A profile carrying a provider this build cannot dispatch (hand-edited file, or
a profile written by a newer build) displays as `openai`:
`normalise_library_rag_reranker_provider`
(`UI/Screens/settings_library_rag_defaults.py:304-322`) folds unknown →
default. Selecting `openai` to repair it hits the effective-provider guard
(`settings_screen.py:18783-18786`: `normalise(loaded) == normalise(chosen)` →
`chosen = loaded`), so the invalid value is re-staged unchanged, the draft
stays clean, and `apply_defaults_to_profile` writes the invalid name straight
back. The control then shows a provider that is not the one that will be used
-- which also makes that function's docstring claim ("so the control shows the
provider that would really be billed") false in exactly this branch. Decide
display AND repair here: either do not fold an unrecognised loaded value into
the guard's equality, or normalise on save, or show the stored value as an
explicit invalid row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is implemented, either arm acceptable: the reranker resolves credentials through the normalised config path for every provider it offers, OR the Settings provider Select is bounded to the providers the reranker can actually call
- [ ] #2 The reranker's credential read targets a table `load_settings()` actually builds -- `self._settings["API"]` is gone, and a configured openai/anthropic/groq credential is found rather than silently read as `None`
- [ ] #3 `chat_api_call` is invoked with arguments its real signature accepts (keyword, not the current positional list) -- no call routes the API KEY into `api_endpoint`, and no credential can reach a log line as an "endpoint"
- [ ] #4 A provider offered in the Reranking fold, with a valid credential configured for it, completes a scoring call instead of failing the first search with `No API key found for provider: X` or `Unsupported API endpoint: <key>`
- [ ] #5 Providers needing no credential (local `ollama`/`llama_cpp`/`vllm`/`koboldcpp`/`mlx_lm`) either rerank successfully or are absent from the picker -- they are never rejected for a missing key they do not need
- [ ] #6 The reranker's credential lookup obeys the documented precedence (explicit `api_settings.<provider>.api_key` over env var over legacy `[API]`, every source validity-checked through `resolve_provider_api_key`)
- [ ] #7 The chosen arm is pinned by tests at BOTH the credential and the dispatch seam, with no live provider calls -- and any `chat_api_call` fake matches the REAL signature, since today's fakes mirror the caller's wrong positional order and is why the defect survived a green suite
- [ ] #8 A stored provider this build cannot dispatch is displayed honestly and is repairable from the UI: picking a valid provider over it stages and saves, rather than being folded back to the invalid stored value by the effective-provider guard
- [ ] #9 The picker's enumeration stays derived, not hand-listed: whichever arm ships, adding a chat provider must not silently desynchronise Settings from the engine
- [ ] #10 The spend consequence is handled deliberately: the change lands on a build carrying TASK-3502's cost disclosure and skipped/degraded notice, and the release note states that reranking-enabled profiles begin spending real provider calls
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep and list every reranker-seam chat_api_call fake under Tests/.
2. RED: rewrite test_reranker_dispatch_binding_against_the_real_chat_api_call_signature so it OBSERVES the real caller through a signature-binding fake and asserts api_endpoint<-model_provider, model<-model_name, temp<-temperature, no credential argument. Must fail against today's caller.
3. RED: assert the reranker no longer reads a settings table (no _settings; load_settings not called during __init__).
4. RED: per-provider parametrised test (keyless local ollama + remote openai + anthropic/groq/deepseek) - _call_llm_impl completes, fake records api_endpoint == the provider, no 'No API key found for provider:'.
5. Implement: delete the if/elif credential block and the load_settings() read; call chat_api_call with KEYWORDS (functools.partial in run_in_executor): api_endpoint, messages_payload, model, temp, max_tokens. Pass NO api_key. Remove dead imports.
6. GREEN all; correct every seam fake found in step 1 to bind via inspect.signature(chat_api_call).bind; batteries: Tests/RAG_Search + Tests/Chat/test_chat_functions.py + the redaction file; ruff.
7. Gate (RAG_EVAL=1 Tests/RAG_Eval/) with the vacuity caveat; commit + push.
<!-- SECTION:PLAN:END -->
