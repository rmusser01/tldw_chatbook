---
id: TASK-31266
title: Adopt verified vLLM targets into Console
status: Done
assignee: []
created_date: '2026-09-03 22:33'
updated_date: '2026-09-04 10:15'
labels:
  - vllm
  - lab
  - console
  - handoff
dependencies:
  - TASK-31265
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the Lab workflow by applying a verified vLLM provider, canonical endpoint, and served model to Console with explicit session or durable scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Use in Console is enabled only for the current verified vLLM generation.
- [x] #2 Session adoption updates the active Console provider, endpoint, model, and readiness without writing durable configuration.
- [x] #3 The durable option delegates to the established Settings/provider persistence path and never silently replaces a different configured endpoint.
- [x] #4 Wildcard bind addresses are converted to an explicit usable client endpoint without weakening exposure warnings.
- [x] #5 Mounted Lab-to-Console and persistence regression tests cover session, durable, stale, and rollback paths.
- [x] #6 Session-only vLLM endpoints remain process-local across existing-chat adoption, later first persistence/messages, and reload; failed compensation restores exact metadata or blocks endpoint use observably.
- [x] #7 A late Settings acknowledgment failure restores the complete pre-stage provider draft and visible presentation while provider edits remain fenced during compensation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the Console session-only persistence leak with real SQLite persisted-chat coverage before adoption, immediately after adoption, after later first persistence/new message, and after reload/reopen; include forced post-mutation failure and a non-vLLM ordinary-settings persistence control.
2. Add the smallest Console-owned ephemeral provider-endpoint override that participates in active provider/model/readiness projection but is excluded from every conversation serializer and persistence path; make adoption compensation restore exact in-memory ownership and durable metadata or fail closed with an observable outcome.
3. Run the focused Console RED node, implement only the Console fix, then rerun that node GREEN before starting Settings work.
4. Reproduce the mounted Settings late-ack failure after staging and assert authoritative draft plus provider/model/endpoint, placeholder, credential/profile, save-result, and dirty presentation all return to the pre-claim snapshot while edits are fenced.
5. Add complete Settings presentation snapshot/restore or authoritative rehydration around vLLM default intent staging, release the claim only after compensation settles, and rerun the focused Settings RED node GREEN.
6. Run focused handoff, Console persistence, Settings, compatibility, Ruff/format where feasible, py_compile, diagnostic inventory, and diff checks; record exact RED/GREEN evidence and self-review.

ADR required: no new ADR
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already assigns ephemeral session adoption to Console, excludes api_url from conversation metadata, and assigns durable defaults to Settings; this fix restores that accepted boundary without changing ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-117 verified-target handoff with exact secret-free Console and Settings intents, strict detached-store reconstruction, and normal navigation from Lab.

Console performs one generation-fenced active-session replacement, preserves differing durable endpoints as Endpoint not saved, and releases stale, detached, failed, or inactive-session claims for replay. Settings stages provider/model/endpoint drafts only, shows endpoint-difference review copy, rolls back without changing config bytes, and leaves the existing Save action as the sole durable writer.

Modified the vLLM setup view, LLMScreen, pending handoff store, Console, canonical Settings screen, and focused Console/Lab/session tests; no app.py change was required because TldwCli already owns PendingHandoffStore and Task 2 already installs the app-scoped readiness owner.

Verification: focused Lab, Console/Settings/provider-persistence, pending-store, and upstream vLLM suites pass; Ruff, focused handoff/store mypy, py_compile, and git diff --check pass. The broad legacy screen mypy invocation still reports its existing baseline errors; no scoped seam errors remain. ADR required: yes. ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md.

Fix Round 1 plan: reproduce post-mutation Console sync failure and exact-text boundary defects; add compensating rollback and exact built-in string validation; rerun Task 3 and upstream focused evidence before returning Done.

Fix Round 1: compensation now begins before the existing active-session replacement owner is invoked, so exceptions from either downstream projection sync restore the captured session and resynchronize the active controller/summary before releasing the claim for exact retry. The Task 3 model boundary now accepts only exact built-in str values, preventing detached intents from retaining subclass state. Added focused regression coverage for post-mutation failure, projection restoration, replay success, and mutable model text rejection at construction, stage, and claim. All Task 3 focused, pending-store, upstream vLLM, Ruff, focused mypy, py_compile, and diff checks pass; ADR-117 remains the governing contract.

Fix Round 2 plan: add RED mounted regressions for restoring a pristine session's has_user_work ownership and for concrete controller/summary recovery when public rollback sync methods fail; implement the smallest store-owned rollback and projection-owner fallback; rerun all focused Task 3 gates before returning Done.

Fix Round 2: Console now snapshots the active session's ownership flag plus its concrete controller and summary projections before replacement. Failed adoption uses a store-owned exact replacement rollback to restore both settings and has_user_work, then falls back from either failing public projection sync to the captured controller/summary owner seams before releasing the claim. Mounted regressions cover forward mutation, repeated core/summary rollback-sync failure, projection restoration, and successful replay. All focused Task 3, pending-store, upstream vLLM, exact Ruff/py_compile, focused mypy, and diff checks pass; ADR-117 remains governing.

Task 6 integration reconciliation updated one feature-owned assertion to the
current Console summary contract shipped on `origin/dev`: display rows use the
canonical `vLLM` label and compact model text, while endpoint persistence remains
machine-readable as `blocker=endpoint_not_saved` with
`recovery_action=save_endpoint`. Production behavior was unchanged.

Fix Round 3 implements the final handoff security review under the existing
ADR-117 ownership boundary; no new ADR is required. Console now keeps the
verified target URL in an explicit process-local endpoint policy, while ordinary
session settings and every durable serializer receive only endpoint-safe state.
Existing-chat adoption writes only canonical endpoint-free generation metadata
and returns an optimistic receipt. Post-mutation failure either restores the
exact previous metadata bytes and in-memory ownership or publishes an observable
blocked policy that cannot supply an endpoint. Ordinary non-vLLM full-settings
persistence remains unchanged, and successful adoption still updates provider,
model, readiness, current-provider routing, and `has_user_work`.

Settings now snapshots and restores the authoritative provider draft plus all
mounted provider inputs, selects, placeholders, credential/profile controls,
save-result copy, dirty status, and action states. The provider card and global
Save/Revert actions remain disabled until acknowledgment succeeds or complete
compensation finishes, so late ownership loss cannot expose a staged endpoint.

Modified `console_session_endpoint_policy.py`, `console_chat_store.py`,
`chat_persistence_service.py`, `chat_screen.py`, `settings_screen.py`, and the
focused provider/handoff flow tests. Verification: 12 focused vLLM Console and
Settings nodes passed; the endpoint/rollback compatibility matrix passed 86;
vLLM Lab plus pending-handoff coverage passed 105; the provider-flow file passed
21 with one known baseline node excluded. Focused real-SQLite lifecycle,
first-persistence, exact rollback, fail-closed, non-vLLM persistence, and mounted
Settings compensation tests pass. Pycompile, new-module Ruff/format, changed-file
Ruff undefined-name checks, and `git diff --check` pass. The excluded unchanged
direct-provider node still fails in terminal-generation persistence before the
vLLM adoption seam (`Terminal generation persistence did not commit`); broader
diagnostics also retain pre-existing roleplay-cleanup, loopback-network, legacy
hydration, and lazy probe-monkeypatch failures unrelated to this diff.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-31217. The Task 6 collision correction moved
the full dependent vLLM sequence into one collision-free monotonic block so no
task depends on a future/higher task id. It therefore moved to TASK-31266 after
TASK-31264 preflight and TASK-31265 readiness. The record was originally added
by `ffc4f9d8f8343169097dcac40d3ba4ed0a2177c0`.
