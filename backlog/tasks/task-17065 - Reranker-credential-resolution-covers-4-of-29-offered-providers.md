---
id: TASK-17065
title: >-
  Reranker credential resolution covers 4 of 29 offered providers -- unify on
  resolve_provider_api_key or bound the picker
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - rag
  - settings
  - config
dependencies:
  - TASK-3502
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-3502 AC#1 gave Settings ▸ RAG's Reranking fold a provider Select whose
options are enumerated from `Chat_Functions.API_CALL_HANDLERS` -- 29 rows, the
exact dispatch table `chat_api_call` looks the reranker's `model_provider` up
in, so no newly registered chat provider can silently go missing and no
undispatchable name can be offered. The enumeration is right. What is behind
it is not.

`BaseReranker._call_llm_impl` (`RAG_Search/reranker.py:187-206`) resolves the
credential with a hand-rolled `if/elif` chain covering exactly four providers
-- `openai`, `anthropic`, `groq` read `API.<provider>_api_key`; `deepseek`
reads `api_settings.deepseek.api_key` -- followed by `# Add other providers as
needed` and `raise ValueError(f"No API key found for provider:
{self.config.model_provider}")`. It never calls `resolve_provider_api_key`
(`config.py:844`). Consequences, in order of how badly they read to a user:

1. Selecting any of the other 25 providers produces a hard runtime failure on
   the first search, even with a perfectly valid credential configured.
2. Local providers that need no key at all (`ollama`, `llama_cpp`, `vllm`,
   `koboldcpp`, `mlx_lm`, ...) fail for a MISSING KEY they never require.
3. Even the four covered providers bypass the precedence rules CLAUDE.md
   documents (explicit `api_settings.<provider>.api_key` outranks the env var,
   legacy `[API]` lowest, every source validity-checked so a placeholder is
   never accepted) -- each reads one hardcoded location, and the four do not
   even agree on which location that is.

TASK-3502 note-(a) shipped the first UI consumer of the reranker's disclosure
tags, so this failure is at least now VISIBLE: the Library RAG results surface
renders "Reranking was skipped (No API key found for provider: X) -- these
results are in their original retrieval order." That is disclosure, not a fix
-- the picker still offers 29 rows of which 25 are known to fail.

This belongs with the config-precedence family (CLAUDE.md's `Configuration`
section, `resolve_provider_api_key` uniformity across the ~9 bridged chat
providers). It is a DECISION task with two acceptable arms: make the
reranker's lookup go through the shared resolver for everything it offers, or
bound the picker to what the reranker can actually call. Widening coverage is
the better end state; narrowing the picker is honest and cheap. Either closes
the gap between what Settings offers and what the engine can do.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A decision is implemented, either arm acceptable: the reranker resolves credentials through the shared `resolve_provider_api_key` path for every provider it offers, OR the Settings provider Select is bounded to the providers the reranker can actually call
- [ ] #2 A provider offered in the Reranking fold, with a valid credential configured for it, does not fail the first search with `No API key found for provider: X`
- [ ] #3 Providers needing no credential (local `ollama`/`llama_cpp`/`vllm`/`koboldcpp`/`mlx_lm`) either rerank successfully or are absent from the picker -- they are never rejected for a missing key they do not need
- [ ] #4 The reranker's credential lookup obeys the documented precedence (explicit `api_settings.<provider>.api_key` over env var over legacy `[API]`, every source validity-checked), rather than one hardcoded read per provider that the four covered providers do not even agree on
- [ ] #5 The chosen arm is pinned by tests at the credential seam, with no live provider calls
- [ ] #6 The picker's enumeration stays derived, not hand-listed: whichever arm ships, adding a chat provider must not silently desynchronise Settings from the engine
<!-- AC:END -->
