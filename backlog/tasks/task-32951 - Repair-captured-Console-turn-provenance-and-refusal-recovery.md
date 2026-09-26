---
id: TASK-32951
title: Repair captured Console turn provenance and refusal recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-25 16:16'
updated_date: '2026-09-25 16:45'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/issues/2829'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and address GitHub issue #2829 so captured sends preserve durable message identity and refused accepted turns remain recoverable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Captured saved assistant history with verified thinking sidecars retains revision ownership, and stopped tool-run successors send without rewriting earlier trace content.
- [x] #2 Surface refusals retain fail-closed checks and expose content-free counts, domains, span, and refusal-kind diagnostics.
- [x] #3 Accepted refused sends expose existing explicit recovery without duplicating accepted messages; explicit Cancel retains the existing durable soft-delete semantics.
- [x] #4 Unsaved effective system prompts and lease-free RAG callbacks follow the captured-send contracts without provenance or missing-argument failures.
- [x] #5 Targeted regressions cover the defects, immutable prior traces, and mismatched saved-source rejection; checks and self-review are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Repair recording and recovery within the existing revision, sidecar, frozen projection, and explicit capture bypass contracts; do not widen surface replacement permissions.

1. Reproduce thinking identity loss and companion captured-send failures with real SQLite/controller/provider-boundary tests.
2. Preserve exact saved message ownership for verified sidecar attachment; classify rendered system context correctly and honor the frozen RAG launch callback signature.
3. Add safe structural refusal diagnostics at both existing surface fences without storing content or private identifiers.
4. Exercise existing refusal recovery and stopped-tool successors, asserting one accepted user and immutable historical traces; coordinate mounted recovery coverage with the parallel recovery investigation.
5. Run only related test modules and targeted static checks, review the delta against the pre-existing working tree, then record findings and limitations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Captured assistant history now retains its saved revision when agent transport
renames the private thinking-owner marker. The controller already preserved the
revision at admission: the demonstrated loss occurred in
`ConsoleAgentTraceRequestFactory.build`. Normalize only the marker spelling;
owner values, other message fields, and frozen historical projections remain
exact. Changed content or a changed owner cannot reuse the saved descriptor.

- `console_agent_bridge.py` preserves admitted revision identity through the
  private marker rename. Real SQLite tests cover failed tool loops, exhausted
  model-turn budgets with a failed wrap-up, warm/cold successors, and immutable
  previously captured requests.
- `console_chat_controller.py` classifies unsaved system rows as rendered system
  context and calls the actual runtime RAG callback with explicit `launch=None`
  for lease-free sends. No later staged evidence is silently substituted.
- `console_trace_service.py` emits safe counts, domains, replacement span, and
  refusal kind at the existing count/span and sequence-gap fences;
  `persistent_diagnostics.py` admits those structural fields. Tests read actual
  diagnostic sinks and both Logs copy projections and reject content canaries.
- `console_trace_errors.py`, `console_trace_runtime.py`, and
  `console_provider_gateway.py` classify only a known surface refusal before
  reservation, after successful transaction unwinding. The ValueError/token
  contract is retained. Capture retry consumes the same single-use live first-call
  ownership proof; controller rebinding removes the exception placeholder.
  Unknown failures, foreign owners/gateways, forged errors, rebound scopes, later
  calls, and tool-loop calls still cannot authorize Capture-On recovery.
- The mounted 80×24 production-stylesheet test exercises Retry capture, Send
  without capture, and Cancel send with zero initial provider entries. Sending
  retains the exact accepted user; Cancel follows existing Discard semantics by
  removing the pending branch and retaining its one durable soft-deleted row.

Historical limitation: existing wrongly recorded ACTIVE_REQUEST artifacts are
not migrated or reinterpreted. A real persisted legacy ledger can still refuse
its successor, including repeated Retry. Explicit Send without capture sends
the same accepted turn once; Cancel safely discards it. Both leave prior trace
reconstruction unchanged. No surface fence was widened.

Verification:

- Pre-fix regressions demonstrated wrong artifact ownership in real agent
  requests, rejected unsaved system provenance, the missing runtime RAG launch
  argument, absent structural refusal diagnostics, and inert/refused Retry and
  Cancel after a first surface failure. A final pre-edit snapshot replay confirms
  all six identity/system/RAG cases fail for those exact causes:
  `/tmp/tldw-2829-red-proof.log`.
- 127 passed: trace runtime, discarded tool-run history, new known-refusal
  ownership matrix, and existing mounted constructor-failure recovery. Log:
  `/tmp/tldw-2829-final-guards.log`.
- 15 passed: new identity/legacy-history, refusal diagnostics, and mounted surface
  recovery tests. Log: `/tmp/tldw-2829-final-new-2.log`.
- The broader targeted run was 174 passed / 19 failed: 18 existing stale dictionary
  callback fixture failures in sidecar/current-turn tests and one existing MCP
  diagnostic guard failure. Loading the pre-edit source snapshots reproduces
  the exact same 18 dictionary failures (43 passed), recorded in
  `/tmp/tldw-2829-baseline-all.log`; the MCP diagnostic failure was independently
  confirmed unchanged from HEAD. These unrelated failures were not repaired.
- Four new test files pass Ruff check and formatting. All touched Python sources
  compile; comparison against pre-edit snapshots finds no new Ruff diagnostics.
  Whole-file source lint has inherited failures; broad reformatting was avoided.
- Self-review and an independent read-only review completed. No external live
  provider was contacted: real SQLite/runtime/mounted UI used deterministic
  provider adapters. No full test sweep was run.

ADR required: no new ADR. This implements existing
`backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md` contracts.
The owner-marker incident is recorded in
`backlog/docs/lessons-console-wiring.md`. Pre-edit snapshots and isolated diffs
are under `/tmp/tldw-2829-baseline/`; pre-existing workspace changes are preserved.
<!-- SECTION:NOTES:END -->
