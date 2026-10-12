---
id: TASK-33245
title: Collapse per-provider summarizers into one shared routine
status: In Progress
assignee:
  - '@robert'
created_date: '2026-09-27 19:46'
labels: [refactor, providers]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nineteen near-identical summarize_with_* functions across LLM_Calls/Summarization_General_Lib.py and LLM_Calls/Local_Summarization_Lib.py (roughly 4k LOC) each repeat the same sequence per provider: key-from-param-or-config, text extraction, default temperature, headers, payload assembly, retry, and SSE parsing — dispatched by a ~200-line elif chain. Each new provider currently costs another ~250-line function. One shared OpenAI-shaped request routine plus a per-provider config row (config key, default model, endpoint quirks) removes an entire class of per-provider edits; providers with genuinely unique flows (Anthropic, Google wire format, huggingface quirks) keep adapter hooks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single shared routine executes the common request lifecycle for all supported providers.
- [ ] #2 Each provider is expressed as a config row covering config key, default model, and endpoint quirks.
- [ ] #3 The elif dispatch chain is replaced by a table lookup.
- [ ] #4 Existing summarization behavior is preserved per provider, verified by targeted tests plus golden-output checks.
- [ ] #5 The net LOC change is recorded in Implementation Notes.
<!-- AC:END -->

## Implementation Plan

Premise re-verified against origin/dev (2026-10-12): all 19 functions still present; the transport layer is already shared (TASK-32853 hosted engine), so the duplication is the surrounding per-provider boilerplate.

1. Add `_OpenAIShapedSummarizerSpec` + one shared `_summarize_openai_shaped()` lifecycle routine to Summarization_General_Lib (key resolution, input extraction, defaults, payload, hosted transport, stream parse, spec-templated error strings; config-read order preserved: api_key?/model/retries/retry_delay/base_url?).
2. Convert the three cleanest OpenAI-shaped providers first (groq, deepseek, mistral) into decorated thin wrappers over spec rows; verify with the LLM_Calls battery before converting more.
3. Replace the ~200-line elif dispatch with `_SUMMARIZER_DISPATCH`, a name-to-handler table with small adapters for the local/mock argument-order families.
4. Providers with genuinely unique flows (anthropic, google, cohere, huggingface, local llama/kobold/ooba/tabby/ollama/custom pair) stay as their existing functions routed by the table this pass; converting them is follow-up work under the same spec pattern.
5. Run the targeted battery (LLM_Calls summarizer tests, custom-openai execution keys, cohere v2, prompt parity/migration) and record net LOC.

ADR required: no
ADR path: N/A
Reason: internal refactor preserving the public dispatch contract and per-provider behavior; no boundary, storage, or protocol decision changes.

## Implementation Notes (partial — 2026-10-12)

**Landed (AC #3):** the ~200-line `_dispatch_to_api` elif chain is now `_SUMMARIZER_DISPATCH`, a 19-entry table with small adapters preserving each local/mock function's original argument order. The entries MUST stay late-binding (lambdas resolving module globals at call time): direct function references captured at import broke 15 tests that monkeypatch provider functions on the module. Net −73 LOC in Summarization_General_Lib.py. Battery at exact dev baseline (17F+6E environmental, zero regressions); ledger reconciliation green.

**Blocked (ACs #1/#2/#4) by diagnostic-review governance — owner decision required.** The full shared-lifecycle conversion was implemented (groq/deepseek/mistral over an `_OpenAIShapedSummarizerSpec` row) and reached complete behavioral parity — every privacy pin passed, including the four-path stream unification (`missing_choices_rejected` flag distinguishing groq/mistral's "rejected" from deepseek's "missing required field" for choice-less events) and groq's single-`str()` eager-conversion pin. It was then reverted, because the diagnostic-review ledger (Tests/fixtures/summarization_diagnostic_review.json, 523 sites; guard: Tests/LLM_Calls/summarization_diagnostic_guard.py) freezes 10–19 reviewed_safe sites per provider, and its outcome vocabulary {pending, frozen, metadata, deleted} admits no consolidation transition — frozen sites must stay byte-identical, and retirement-by-deletion is private-only. Every one of the 19 providers carries frozen sites, so no subset can convert legally either.

Options for the owner:
- **A. Extend the review system** with a sanctioned consolidation outcome (guard rule: N frozen sites converge into one parameterized site, severity preserved, expressions from an approved set; fixture migration + count/hash constants). Unblocks the full collapse as designed (~3k LOC).
- **B. Non-diagnostic core extraction**: share config reads/payload/transport/parsing while keeping each provider's log call sites in place — a larger redesign that respects the freeze.
- **C. Stop at the dispatch table** (this PR).

Task remains In Progress pending the decision.



## Renumbering provenance

Filed 2026-09-27 as task-33085. origin/dev minted its own task-33085 before this branch merged, so per the landed-keeps-id rule this task moved to task-33245; every inbound reference moved with it.

Second renumber: dev minted its own task-33130 (second move: dev minted its own task-33130 before merge); per landed-keeps-id this task moved to task-33245 as the last commit before push.
