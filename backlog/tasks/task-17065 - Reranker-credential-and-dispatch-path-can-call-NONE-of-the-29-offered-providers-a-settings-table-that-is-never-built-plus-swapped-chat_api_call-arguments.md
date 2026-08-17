---
id: TASK-17065
title: >-
  Reranker credential and dispatch path can call NONE of the 29 offered
  providers -- a settings table that is never built, plus swapped chat_api_call
  arguments
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
updated_date: '2026-08-17 06:02'
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
- [x] #1 A decision is implemented, either arm acceptable: the reranker resolves credentials through the normalised config path for every provider it offers, OR the Settings provider Select is bounded to the providers the reranker can actually call
- [x] #2 The reranker's credential read targets a table `load_settings()` actually builds -- `self._settings["API"]` is gone, and a configured openai/anthropic/groq credential is found rather than silently read as `None`
- [x] #3 `chat_api_call` is invoked with arguments its real signature accepts (keyword, not the current positional list) -- no call routes the API KEY into `api_endpoint`, and no credential can reach a log line as an "endpoint"
- [x] #4 A provider offered in the Reranking fold, with a valid credential configured for it, completes a scoring call instead of failing the first search with `No API key found for provider: X` or `Unsupported API endpoint: <key>`
- [x] #5 Providers needing no credential (local `ollama`/`llama_cpp`/`vllm`/`koboldcpp`/`mlx_lm`) either rerank successfully or are absent from the picker -- they are never rejected for a missing key they do not need
- [x] #6 The reranker's credential lookup obeys the documented precedence (explicit `api_settings.<provider>.api_key` over env var over legacy `[API]`, every source validity-checked through `resolve_provider_api_key`)
- [x] #7 The chosen arm is pinned by tests at BOTH the credential and the dispatch seam, with no live provider calls -- and any `chat_api_call` fake matches the REAL signature, since today's fakes mirror the caller's wrong positional order and is why the defect survived a green suite
- [x] #8 A stored provider this build cannot dispatch is displayed honestly and is repairable from the UI: picking a valid provider over it stages and saves, rather than being folded back to the invalid stored value by the effective-provider guard
- [x] #9 The picker's enumeration stays derived, not hand-listed: whichever arm ships, adding a chat provider must not silently desynchronise Settings from the engine
- [x] #10 The spend consequence is handled deliberately: the change lands on a build carrying TASK-3502's cost disclosure and skipped/degraded notice, and the release note states that reranking-enabled profiles begin spending real provider calls
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The fix is a DELETION, not a rewrite.** Both defects lived in code the reranker
should never have had. `RAG_Search/reranker.py` now holds no credential logic and no
settings read at all: `from ..config import load_settings`, `self._settings =
load_settings()`, the whole `if/elif` chain and its `raise ValueError(f"No API key
found for provider: ...")` are gone, and the dispatch is

```python
functools.partial(
    chat_api_call,
    api_endpoint=self.config.model_provider,
    messages_payload=messages_payload,
    model=self.config.model_name,
    temp=self.config.temperature,
    max_tokens=self.config.max_tokens,
)
```

handed to `run_in_executor` (which forwards positionals only -- a positional list at
this signature is the defect, so the `partial` is load-bearing; do not "simplify" it
back). **No `api_key` is passed.**

**Why deleting the lookup is safe (measured, 29/29).** Every row of
`API_CALL_HANDLERS` resolves its own credential or needs none: 22 use the
`api_key or <config>` idiom; the remaining 7 were regex false-positives --
`qwencloud` (`resolve_qwencloud_api_key`), `custom-openai-api` (`api_key_resolved`),
`moonshot`/`zai` (`explicit_api_key` -> `api_settings` -> `resolve_provider_api_key`),
and `mlx_lm`/`local_mlx_lm`/`local-llm` (keyless by design). Also checked: all 29
`PROVIDER_PARAM_MAP` entries map `max_tokens` (openai->`max_tokens`,
ollama->`num_predict`), and the dispatcher DROPS `None` generics, so omitting
`api_key`/`streaming` leaves each handler's own resolution and non-streaming default
in force.

**The sole outlier is what broke.** Every other `chat_api_call` caller in the repo
already passes keywords and omits `api_key` -- `UI/Tools_Settings_Window.py:4157`/
`:4296`, `UI/Screens/evals_screen.py:193`, `Chat/console_provider_gateway.py:2534`.
The reranker was the only caller that hand-rolled a credential path and the only one
that dispatched positionally, and being the only one is exactly why nothing else
caught it.

**Extra finding (the third defect, unfiled).** A signature-binding probe over the OLD
call (no live call; the legacy settings shape planted only to get past the credential
gate that fires first) showed where the arguments actually landed:
`api_endpoint='THE-API-KEY'`, `api_key='openai'`, `temp='gpt-4o-mini'`,
`system_message=0.25`, **`streaming=128`**. The old positional list also silently
switched STREAMING ON (truthy `max_tokens`), so every scoring call would have STREAMED
had a credential ever resolved. Scoring calls are now non-streaming (handler default),
and `RerankingConfig`'s `max_tokens=100` / `temperature=0.0` reach a provider for the
first time.

**Evidence per AC**

- **#1** (arm A: fix the engine, keep the picker) -- the reranker now reaches every
  provider `chat_api_call` can route.
  `test_every_sampled_provider_reaches_the_seam_without_a_credential_gate`,
  9 parametrised cells.
- **#2** -- `self._settings` and the `["API"]` read are deleted outright;
  `test_reranker_does_not_read_a_settings_table` monkeypatches `load_settings` to
  raise and construction still succeeds. The openai/anthropic/groq credential is now
  FOUND -- by the provider handler, which is the acceptable variant the task's "Fix
  shape" blesses ("simply not resolving at all and letting the provider handler
  resolve ... is an acceptable variant").
- **#3** -- keyword dispatch, pinned by
  `test_reranker_dispatch_binding_against_the_real_chat_api_call_signature`, which now
  DRIVES the real `_call_llm_impl` and binds what lands through
  `inspect.signature(chat_api_call)`: `api_endpoint <- model_provider`,
  `model <- model_name`, `temp <- temperature`, `max_tokens <- max_tokens`, and NO
  argument carrying a credential. A credential can no longer reach the
  `Unsupported API endpoint: <value>` log line because no credential is passed at all.
- **#4** -- both named failure modes are now structurally impossible: `No API key
  found for provider: X` because the raise is deleted, `Unsupported API endpoint:
  <key>` because `api_endpoint` carries the provider name. Pinned at the seam by the
  9 provider cells. **Honest bound: no live provider call was made anywhere in this
  arc (a hard constraint), so "completes a scoring call" is evidenced up to the
  dispatcher, not over the wire.**
- **#5** -- the five keyless locals (`ollama`, `llama_cpp`, `vllm`, `koboldcpp`,
  `mlx_lm`) are among the 9 parametrised cells; each reaches the seam with its own
  name in `api_endpoint` and no `api_key` argument. They stay in the picker.
- **#6** -- the reranker performs no lookup, so it cannot violate the precedence: it
  now uses the same path as every other chat call, and each handler applies
  `api_settings.<provider>.api_key` -> env -> legacy `[API]`, validity-checked through
  `resolve_provider_api_key`. CLAUDE.md's known open caveat (env-only credentials for
  `google`) is inherited by every chat path and is not introduced or worsened here.
- **#7** -- pinned at BOTH seams (the settings-table test and the dispatch guard),
  no live calls. Every reranker-seam fake corrected: `_install_fake_provider` in
  `Tests/RAG_Search/test_reranker_degraded_paths.py` (it declared the CALLER's wrong
  positional order and planted `reranker._settings` -- now binds through the real
  signature and plants nothing; 8 call sites), the guard itself (it re-typed the
  caller's argument list as a literal tuple, so it guarded a copy; it now drives the
  real caller), and 3 fakes in `test_reranker_construction.py` raising an error the
  code can no longer produce. Other `chat_api_call` fakes in the repo (Console
  gateway, Evals, Tools, Web_Scraping) are keyword-only and not at this seam.
- **#8** -- ALREADY SATISFIED ON DEV, no code written here: the picker-repair half
  shipped with TASK-3502's Qodo remediation (the reranker-provider change guard folds
  back only a loaded value the fold-back PRESERVES -- blank or registered -- so an
  unrecognised loaded value no longer swallows the corrective pick,
  `settings_screen.py:18790-18798`). Verified by running
  `Tests/UI/test_settings_rag_profile_region.py::
  test_an_unrecognised_stored_provider_is_repairable_from_the_picker` -> **1 passed**
  (the test is present at the merge-base, dev `1c328e1a7`). Display side, stated
  plainly: an unrecognised stored name still RENDERS as the default row (the Select
  cannot show a value it has no option for without raising out of `compose()`), and
  the honesty is delivered at the point of use -- the stored name goes to
  `chat_api_call`, which cannot route it, and the results screen discloses
  "Reranking was skipped (...)". `normalise_library_rag_reranker_provider`'s docstring
  claimed this branch was unrepairable and pointed at this task; corrected here.
- **#9** -- the picker was NOT TOUCHED. `library_rag_reranker_providers()` still
  derives its options from `Chat_Functions.API_CALL_HANDLERS`; the only change in
  `settings_library_rag_defaults.py` is docstring text. Adding a chat provider still
  adds a picker row and, now, a provider the reranker can actually dispatch to.
- **#10** -- release note added to `CHANGELOG.md` (Unreleased -> Changed), plus the
  user-facing copy in `Docs/User_Guide/settings/rag.md`, which had documented the
  defect as an open gap ("its credential path currently reaches far fewer of them
  than the list offers ... TASK-17065") and is now corrected in all three places it
  said so. The note states that an already-ticked profile begins spending real
  provider calls on its NEXT SEARCH with no further user action (one call per
  candidate up to the configured top-k), names TASK-3502's cost line and
  skipped/degraded notice as the surfaces that make it visible, names the
  streaming-128 consequence, and names `max_tokens=100`/`temperature=0.0` reaching
  providers for the first time. The build carries both TASK-3502 surfaces.

**Tests.** New/rewritten in `Tests/RAG_Search/test_reranker_degraded_paths.py`: the
dispatch guard (rewritten), `test_reranker_does_not_read_a_settings_table`,
`test_every_sampled_provider_reaches_the_seam_without_a_credential_gate` (9 cells).
11 RED before the fix -- the guard's RED was `ValueError: No API key found for
provider: openai` at `reranker.py:204`, i.e. defect 1 firing before defect 2 is even
reachable -- and 11 GREEN after. Batteries: `Tests/RAG_Search/` 358 passed / 12
skipped; `Tests/Chat/test_chat_functions.py` + `Tests/RAG/` + the reranker prompt
parity file 815 passed (4 failures are missing optional deps `tomli`/`nltk` in the
pinned venv, in files that never import the reranker); redaction + settings + library
files 354 passed; RAG UI files 215 passed. Gate `RAG_EVAL=1 ... Tests/RAG_Eval/` 307
passed -- **VACUOUS for the reranker** (no gated cell constructs or runs one), stated
as established in TASK-3502; the repair's evidence is the guard and the 9 provider
cells.

**Lesson.** `backlog/docs/lessons-testing-evidence.md` -- extended the existing entry
"A fake written to match your call site validates the mistake" with this incident
rather than filing a duplicate: a module that grew BOTH its own credential path and
its own dispatch convention, with a seam fake that mirrored both, so ~2,500 green
tests could not see zero-of-29 coverage. Two rules: a fake at a shared seam must bind
against the real signature (even the guard written to catch this first re-typed the
caller's argument list and caught nothing); and a feature that resolves credentials
itself is a divergence to justify, not a default.

**Files.** `tldw_chatbook/RAG_Search/reranker.py`,
`tldw_chatbook/UI/Screens/settings_library_rag_defaults.py` (docstrings only),
`Tests/RAG_Search/test_reranker_degraded_paths.py`,
`Tests/RAG_Search/test_reranker_construction.py`, `CHANGELOG.md`,
`Docs/User_Guide/settings/rag.md`, `backlog/docs/lessons-testing-evidence.md`, plus
the spec/plan under `Docs/superpowers/`.
<!-- SECTION:NOTES:END -->
