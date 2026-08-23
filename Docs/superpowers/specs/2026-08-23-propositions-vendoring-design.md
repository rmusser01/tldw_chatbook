# Propositions Vendoring & Program Descope — Design Spec

**Date:** 2026-08-23
**Status:** Draft, maintainer-approved in brainstorming (the scope ruling); awaiting review gate.
**Sub-project:** 6 of 6 — the closing act of the Chunking Parity & Agent Tools program.
**Depends on:** #1-#5, all merged (#5 = PR #1984, `c56eab813b`). Branches off `origin/dev`.
**Author:** brainstormed with the maintainer. Upstream facts verified against the same
unmoved pin (`dev` @ `385afa951922c8a9dc2002c675bb6cad65e4ac23`, via the local
`tldw_server2` checkout); chatbook-side facts verified against `origin/dev` at `c56eab813b`.

---

## 1. Why

The program decomposed six sub-projects in #1's spec; five are merged. What remains
of "LLM-dependent extras" after honest discovery is **one cheap vendoring plus two
descopes worth more than their implementations**:

- **`strategies/propositions.py` is a manifest move.** 728 lines whose imports are
  stdlib + the already-vendored `..base` + the already-shimmed `prompt_loader`.
  Chatbook already routes the method end-to-end — verified by execution today:
  `improved_chunking_process(..., {"method": "propositions"})` raises
  `InvalidChunkingMethodError: No module named …strategies.propositions`. The
  sync lands the file; the routing starts working; the dead-skipped upstream
  suite (10 tests) un-skips.
- **`auto_boundary_assistant.py` and `async_chunker.py` cost more than they
  return.** The assistant imports the server's entire provider stack
  (`AuthNZ.llm_provider_overrides`, `AuthNZ.providerCredentialRuntime`,
  `Chat.bounded_daemon`, `Chat.chat_helpers`, `LLM_Calls.adapter_registry`,
  plus function-level `api.v1.schemas` — verified); `async_chunker` needs
  `http_client` (RetryPolicy/afetch) + `core.exceptions.ValidationError`.
  Chatbook has no equivalent seams and — decisively — **no consumer**: the
  capability space is already covered (#3's auto-selection for the non-LLM
  case; #4's agent surface for the model-driven case; chatbook chunks
  in-process, never via HTTP). Vendoring would build 4+ load-bearing shims
  with zero callers.
- **Telemetry depth would reverse a recorded ruling.** The engine's Metrics are
  vendored-and-no-op'd deliberately (#1 §5.3); deepening them has no consumer.

**Scope ruling (maintainer, 2026-08-23): propositions-only; the rest descoped
with recorded rulings.**

## 2. Goals

1. `propositions` works as a chunking method in chatbook — heuristic (default),
   spacy (optional), and llm engines, through the existing tool/shim surfaces.
2. The manifest's `excluded` list becomes an honest ledger: every entry carries
   its reason, and no "deferred to #6" residue remains.
3. The program closes with the descope decisions recorded where a future
   revisit will find them.

## 3. Non-goals

- **Vendoring `auto_boundary_assistant.py` or `async_chunker.py`** — descoped (§4).
- **A Metrics/telemetry shim** — the no-op ruling stands (§4).
- **Wiring `propositions` into the tool surface** — nothing to wire; the
  method rides the existing routing (the shim's method map has passed
  `propositions` since #1; the auto-selection vocabulary needs no entry —
  it is a chunking method, not a media type).
- **Upstream changes** — the pin does not move.

## 4. The descope rulings (permanent)

Recorded in three places — this spec's decisions (§8), the manifest's
`excluded`-list comments, and the sync script's `TESTS_MODULE_SKIPPED` reason
strings (edited in the script, regenerated into the two ported test files on
sync):

1. **`auto_boundary_assistant.py` — NOT VENDORED.** Server-stack shims
   (AuthNZ ×2, Chat ×2, LLM_Calls adapter registry, api.v1 schemas at
   function level) with no chatbook consumer; the capability is covered by
   #3's auto-selection and #4's agent surface. Revisit only if a consumer
   appears — as a fresh decision against this cost record.
2. **`async_chunker.py` — NOT VENDORED.** Requires `http_client`
   (RetryPolicy/afetch) + `core.exceptions.ValidationError`; chatbook chunks
   in-process and calls no async entry point. Same revisit condition.
3. **Telemetry depth — NO-OP REAFFIRMED.** #1 §5.3's ruling stands; no
   Metrics shim is ever built without a consumer.
4. **The #1 §0 drift obligation for the two files closes as moot** — the
   files drifted from the codex branch; the descope makes re-verification
   unnecessary. Recorded here so the obligation isn't left dangling.

## 5. Vendoring

- `strategies/propositions.py` moves `excluded` → `vendored` (the 39th file,
  in both the manifest and the sync script's list — the move, never both).
- Imports resolve with **zero new shims**: `from .base import …` (vendored),
  `from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt`
  rewrites to the existing `_shims/Utils/prompt_loader`, whose `_KNOWN` gains
  the verified mapping `("chunking", "Proposition-based Chunking")` (read the
  pin for the exact `load_prompt` call site; propositions.py:31 imports it —
  find where it is invoked and add the pair the call uses).
- **spacy stays optional** (#2 §5.5's standing note): the engine's spacy path
  imports inside functions and degrades to heuristics; chatbook declares it
  nowhere new.
- Byte-identical vendoring modulo the rewrite rules; the sync contract tests
  update 38→39.

### 5.1 The LLM engine contract (the #1 precedent, restated)

`PropositionChunkingStrategy.__init__(language, llm_call_func, llm_config)`
and `_propositions_via_llm` calls positionally: `llm_call_func(api_name,
prompt, None, api_key, system_message, temp, False, False, False,
model_override=…, **snapshot_kwargs)` (verified at the pin, :367-378). This is
the same analyze-style family as `rolling_summarize._call_llm`, and the same
ruling applies: **the shim's adapter translates chatbook's payload-dict
callback to the positional contract; callers keep their signature** — the
established `test_chunk_lib_shim.py` rolling_summarize pins are the pattern.
Two upstream behaviors preserved as parity (recorded so nobody "fixes" them):

- **LLM failure falls back to heuristics** (`_propositions_via_llm` failure →
  warning → heuristic propositions — verified :271-286). This is deliberately
  different from rolling_summarize's fail-close; the contrast is upstream
  design, not an oversight.
- The server-only config keys (`app_config`, `credentials_resolved`,
  `provider_credentials`) are `if 'x' in config`-guarded (verified :360-366) —
  their absence in chatbook's `llm_config` is benign; no shim provides them.

### 5.2 Method surface

Nothing new wires. The shim's method-map passthrough set already contains
`"propositions"` (Chunk_Lib.py:348 — carried since #1's Task 3 for exactly
this moment). The upstream suite
`test_propositions_strategy.py` (10 tests, dead-skipped with the "deferred to
#6" reason) un-skips via the sync-script table edit and must pass;
`test_upstream_chunking_templates.py`'s skip reason gets its final wording
(the §11-item-8 residue note in #2's spec closes here — both files now carry
  terminal dispositions).

## 6. Testing

1. The un-skipped upstream suite green (10 tests).
2. Heuristic engine: deterministic unit coverage (input → propositions →
   packed chunks; the `aggressiveness`/`min_proposition_length` knobs).
3. spacy engine: skip-if-absent parity (no spacy in the venv; the skip is
   upstream's own, reasons verified).
4. LLM engine: a stubbed positional callback through the shim (payload-dict
   caller → adapter → positional — the #1 pattern), including the
   fallback-to-heuristics leg (stub raises → heuristic output, no raise).
5. Parity fixtures: propositions cases join the golden corpus (heuristic
   engine — the LLM engine is stub-only by design; golden LLM outputs are
   not reproducible across model versions and are out).
6. Descope pins: the two files absent from the tree; the manifest reasons
   present and grep-clean of "deferred to #6" anywhere; the regenerated test
   files carry the terminal wording.
7. The standing suites green (Chunking incl. the story test; nothing else
   moves).

## 7. Acceptance criteria

- [ ] #1 `strategies/propositions.py` vendored from the existing pin (manifest
      move, 38→39 files, byte-faithful modulo rewrite rules, zero new shims)
- [ ] #2 `improved_chunking_process(..., {"method": "propositions"})` returns
      chunks (the heuristic engine, no extra deps) instead of raising
- [ ] #3 The un-skipped upstream suite passes; both formerly-deferred test
      files carry terminal dispositions (no "deferred" residue repo-wide)
- [ ] #4 The LLM engine works through the shim's payload-dict→positional
      adapter, and its failure falls back to heuristics (pinned)
- [ ] #5 spacy remains optional and skip-clean; no new dependency declared
- [ ] #6 The manifest `excluded` entries for `auto_boundary_assistant.py` and
      `async_chunker.py` carry the not-vendored rulings; telemetry no-op
      reaffirmed; the #1 §0 drift obligation recorded as closed
- [ ] #7 Parity fixtures extended with propositions (heuristic) cases
- [ ] #8 CHANGELOG + the user-guide method list gain `propositions`
- [ ] #9 Targeted suites green (Chunking incl. story; sync contract at 39);
      zero new failures vs dev baseline

## 8. Decisions taken

1. **Scope (maintainer): propositions-only; `auto_boundary_assistant`,
   `async_chunker`, telemetry descoped with permanent rulings** (§4).
2. **The LLM contract follows the #1 rolling_summarize precedent** (shim
   adapter translates; callers unchanged) — stated in brainstorm, not asked.
3. **The fallback-to-heuristics and guarded-config behaviors are parity,
   not bugs** — preserved and pinned as-is.
4. **Golden fixtures cover the heuristic engine only** — LLM outputs are not
   reproducible across model versions; the LLM engine's coverage is
   stub-based (§6.5).
5. **Nothing wires** — the routing has been waiting since #1; the sync
   activates it.
