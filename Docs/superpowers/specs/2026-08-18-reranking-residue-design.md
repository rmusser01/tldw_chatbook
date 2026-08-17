# Reranking residue batch: TASK-17265 + TASK-17365 + TASK-17600

Date: 2026-08-18
Programme: RAG server-port (seventeen merged; last: 16965 cross_encoder #1775)
Worktree: `.worktrees/rag-rerank-residue`, branch `chore/rag-reranking-residue`, off dev `22d156155`.

The three tasks the reranking thread left behind. All small, all the same
family, one arc. Each has a pre-registered decision below.

## TASK-17265 — the system prompt never reaches anthropic or google

The reranker builds its instruction as an **in-band** `{"role": "system"}`
entry in `messages_payload` (`reranker.py:315-326`) and never passes
`system_message=`. `chat_with_anthropic` discards a system-role message in
`messages` (Anthropic wants a top-level `system`), and `chat_with_google`
skips non-user/assistant roles by construction — so on those two providers
the JSON contract the reranker depends on **never arrives**, which is a
plausible contributor to malformed responses there.

**Decision (verified): pass it as `system_message=`, not in-band.** All
**29 of 29** providers map `system_message` in `PROVIDER_PARAM_MAP`
(measured this session), so the shared dispatcher already knows where each
provider wants it. This is TASK-17065's lesson applied a second time: the
reranker's job is not to know provider shapes. Removing the in-band entry
also satisfies AC#2 by construction — exactly one system instruction can
reach the wire, because only one is sent.

Risk to check in the plan: whether any provider's handler ONLY honours an
in-band system turn and ignores its mapped parameter. That is the one way
this fix could regress a currently-working provider, so it is verified
per-provider at the payload boundary (AC#1's "assembled payload, not just
the call site"), with a fake transport.

## TASK-17365 — cloned profiles keep `include_reasoning=true`

TASK-16965 AC#11 turned the flag off on the two shipped profiles, but
built-ins never persist, so a profile a user CLONED earlier keeps it — and
with `max_tokens=100` the reasoning text truncates the JSON, producing a
**billed-but-unscored** call (listwise: the whole rerank fails).

**Decision: make `max_tokens` reasoning-aware, not a migration.** A
migration mutates a user's saved profile behind their back and must guess
whether a large `max_tokens` was deliberate; a floor cannot. When
`include_reasoning` is on, the effective token budget is raised to a
constant that fits reasoning plus the JSON (the task's own suggestion:
>= 400). Deliberate large values are untouched because the rule is a floor,
not an assignment. Applies to every strategy and to profiles this arc
cannot see.

## TASK-17600 — `result_reranking` is an enabled no-op (+ F3)

`Config_Files/rag_pipelines.toml` declares a `result_reranking` middleware,
listed by `high_accuracy` with `enabled = true`, and
`_apply_after_middleware` handles the name with a bare `pass`. Its sibling
finding: `reranking_strategy` has **zero readers repo-wide** while
RAG-DESIGN.md tells users to select a strategy with it.

**Decision: DELETE both, do not wire.** Wiring `result_reranking` would
switch reranking ON for anyone using `high_accuracy` — and TASK-16965
measured reranking as net-harmful on the averaged row. Shipping a stage
that spends provider calls to make retrieval worse, on the strength of a
name, is the opposite of what the measurement licenses. The honest move is
to remove the promise; `RerankingConfig.strategy` remains the real,
documented control. **AC#4's "a test fails if a declared-enabled middleware
name has no implementation" still ships** — that guard is what stops the
class recurring, and it is the part worth keeping regardless of which arm
the individual name takes.

## Out of scope
- Any change to reranking's default state (it stays off; 16965 stands).
- The cross-encoder strategy itself.
- Re-opening 16965's verdict.

## Plan-phase verification
1. Per-provider: does any handler honour ONLY an in-band system turn?
   (the one regression risk in 17265's fix).
2. Where the reranker's `max_tokens` is consumed, so the reasoning floor
   lands in one place rather than per strategy.
3. Every middleware name `_apply_after_middleware`/its before-equivalent
   accepts, so 17600's guard covers the whole set rather than one name.
4. Whether `reranking_strategy` has a non-code consumer (docs, examples,
   a user's config) that a deletion would strand.
