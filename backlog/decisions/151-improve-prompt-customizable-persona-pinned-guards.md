# ADR-151: Improve-My-Prompt rewrite prompt — customizable persona, code-pinned guards

Status: Accepted
Date: 2026-09-12
Related Task: [TASK-32479](../tasks/task-32479%20-%20Expose-Improve-My-Prompt-rewrite-prompt-in-Settings-Internal-Prompts.md)
Builds on: [ADR-029](029-versioned-prompt-artifacts-and-safe-improvement-transactions.md) (structured prompt-improvement transactions); the Internal_Prompts registry pattern used by [ADR-052](052-console-conversation-memory-and-compaction-policy.md)

## Context

The owner-selected default "improve my prompt" rewrite template (TASK-32478)
shipped as a hardcoded constant in
`Prompt_Management/prompt_improvement_prompts.py`. Every other long-lived
internal prompt (console, summarization, RAG reranker, agents, websearch,
character, document generation) is registered in the `Internal_Prompts`
catalog and is editable in Settings > Internal Prompts with save/reset
overrides stored under `[internal_prompts.<subsystem>.<key>]`. The rewrite
template was the odd one out: invisible in Settings and not customizable
without a code change.

The assembled system message for Auto/Review improvement has four parts with
different trust requirements:

1. the persona/structure template (taste — the owner wants to edit it);
2. the safety guards (never answer the source, preserve invariants, do not
   invent semantic content);
3. the closed JSON envelope instruction (response parsing depends on it);
4. the recency anchor (live verification in TASK-32478 showed that without
   it, providers return schema-valid near-copies instead of transforming).

## Decision

Register exactly one spec, `prompt_improvement.rewrite`, whose `default` is
the persona/structure template (`Internal_Prompts/prompt_improvement_prompts.REWRITE_DEFAULT`).
`trusted_optimizer_instructions()` resolves that portion through
`get_internal_prompt()` (lazy import, mirroring the Console controller's
boot-hygiene pattern) and then always appends parts 2–4 from code-pinned
constants.

An override therefore can only replace the persona/structure instructions.
The safety guards, envelope instruction, and recency anchor are
non-negotiable: a hand-edited or careless override cannot strip the
no-answer / no-invention invariants, cannot relax the closed response
envelope, and cannot remove the recency anchor that keeps providers
task-compliant.

Recipe mode is unchanged: its instructions stay fully code-pinned because a
Recipe fill is a structural operation (block IDs, fingerprint, closed fill
schema), not a stylistic one.

## Alternatives considered

- **Expose the entire assembled system message as one editable default.**
  Rejected: the override editor would let users delete the safety guards or
  the JSON-envelope instruction, silently degrading both the security
  posture and parse reliability. The resolver's `required_placeholders`
  check can only guard `{tokens}`, not prose invariants.
- **Register four separate specs (persona, safety, envelope, anchor).**
  Rejected: three of the four must never change; listing them as editable
  rows invites exactly the edits we need to forbid, and repeats the
  TASK-1220 lesson (specs badged customizable that should not be).
- **Keep the prompt code-only.** Rejected by owner request: the template is
  explicitly meant to be tuned like the other internal prompts.

## Consequences

- Settings > Internal Prompts gains a "Prompt improvement" group with one
  row; overrides land in `[internal_prompts.prompt_improvement.rewrite]` and
  get the standard customized / default-changed badges.
- `Prompt_Management` now depends on the `Internal_Prompts` registry at call
  time (lazy import). Import-hygiene tests still gate the registry off the
  config import chain at module load.
- Tests that pin the shipped template run against a scratch config so a host
  override can never flip them.
