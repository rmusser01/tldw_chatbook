---
id: TASK-575
title: 'Console /rewind: add a Summarize-from-here complement to Summarize-up-to-here'
status: Done
assignee: []
created_date: '2026-07-25'
updated_date: '2026-08-29 15:54'
labels:
  - console
  - chat
  - rewind
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred v1 scope cut from the `/rewind` menu (SP2, PR #844, decision D2: v1 = Restore + Summarize-up-to-here only). Up-to-here compresses the OLDEST turns and keeps the recent tail verbatim; the complementary gesture — keep the early framing verbatim and compress a recent tangent — has no affordance. Design decision needed first: exact semantics of "from here" (compress from the selected turn through the current leaf into a summary that sits at the boundary), how it composes with an existing up-to-here boundary (two boundaries vs replace), and whether the existing single `(context_summary, summary_boundary_message_id)` conversation-field pair can represent it or a second pair/shape is required. Reuses the SP2 machinery: session-provider summarization through the gateway, editable Internal_Prompts entry, render-derived banner (never a tree node), and the id-anchored leak rule (compaction only when the boundary row is in the payload).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Semantics for Summarize-from-here (including interaction with an existing up-to-here boundary) are decided and written down before implementation
- [x] #2 The /rewind menu offers the new option and the resulting compaction follows the same leak rule as up-to-here (pre-boundary sends never receive a summary of turns they precede)
- [x] #3 The summary banner renders derived-only (never a tree node) and survives resume
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: [ADR-052](../decisions/052-console-conversation-memory-and-compaction-policy.md)
Reason: The task changes durable memory scope, branch selection, atomic replacement, provider request projection, and the long-lived `/rewind` UX.

Detailed plan: [Console `/rewind` Summarize-from-here implementation](../../Docs/superpowers/plans/2026-08-28-console-rewind-summarize-from-here-implementation.md)

1. Add the local-only memory-scope and append-mostly branch-selection schema with deterministic backfill, deletion constraints, and migration foreign-key auditing.
2. Implement one typed effective-memory selector and exact repository CAS for select, reset, undo, and reset-all without record-global branch mutation.
3. Unify complete durable-unit grouping and exact one-call manual prefix/range planning with canonical idle progress accounting.
4. Route both manual directions through the existing bounded auxiliary service and project effective prefix/range memory through every preview/direct/agent request path without leaks.
5. Add ordered range-to-prefix automatic compaction plus Context & memory lifecycle controls.
6. Add the `/rewind` choice, exclusive guarded worker flow, derived banner, and restart/branch lifecycle restoration.
7. Run focused migration, repository, planner, provider, controller, and mounted UI verification; complete static/privacy/self-review and record exact evidence before marking the task Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented branch-safe manual prefix/range memory across the Console context
repository, planner/service, provider projection, `/rewind` UI, derived banner,
and Context & memory lifecycle controls. The additive v55 migration stores
one-to-one scope plus append-mostly branch selections, uses exact CAS fences,
keeps legacy memory as a validated baseline, and fails open to raw history when
identity validation cannot prove a safe range. Manual prefix and range actions
share the complete-durable-unit planner and make at most one auxiliary
completion; generated memory remains separate app context and is never a
transcript row or user-role fallback.

The first whole-branch audit found and corrected three cross-layer defects:
effective selection, branch fences, and Undo now scan complete history beyond
the default 100-row page; manual summary input preserves bounded ephemeral
visual content while durable state retains only safe digest/provenance; and
the local summary projection is always checked against the requested output
cap even when provider usage is under-reported. Each correction received an
independent focused review with no new Critical or Important issue.

A fresh final audit then found the same output/visual contracts were incomplete
in the parallel automatic and range-to-prefix paths. Ordinary and range
automatic compaction now measure candidate-versus-empty memory locally, reject
local or reported output above the requested cap, and persist the conservative
larger output count. Selected attachments must have an exact frozen visual
representation or planning refuses before auxiliary preparation/dispatch;
supported images use the active model's canonical vision/count policy and the
real provider request for capacity accounting. Text-only range envelopes keep
complete durable-unit rows byte-stable, while multimodal units use explicit
indexed unit/message frames so each image immediately follows its owning
message around the sealed-memory marker. Three adversarial review/fix cycles
ended with zero Critical, Important, or Minor findings.

Rebased cleanly onto `origin/dev` `6a3092ad39` with no conflicts; the upstream
commit is the exact merge base. Import provenance passed and resolved this
worktree's package. Before that upstream merge, the final focused 15-file
DB/Chat/provider/controller/mounted-UI suite passed `1097 passed, 2 skipped,
2 warnings` in 328.12 seconds. After the rebase, the same matrix passed all
TASK-575-relevant coverage with `1090 passed, 2 skipped, 7 deselected,
2 warnings` in 343.05 seconds. The seven deselected settings tests were each
reproduced failing unchanged on a detached checkout of `origin/dev`; they use
the settings API retired by upstream PR #2201 and are not TASK-575 regressions.
The skips were two host-denied loopback-listener cases. Ruff over all changed
task production/test owners and the provider gateway passed. Production
compileall, schema allowlist (75 declared ChaChaNotes tables), privacy/dead-seam
searches, and both diff guards passed. The full repository suite was not run
because repository policy requires explicit opt-in and no opt-in was given.

Fresh isolated mounted verification used
`.superpowers/sdd/2026-08-28-console-rewind-summarize-from-here-implementation/task-10-live-scratch-final-post-i3/`.
At 120x40, visible-send `/rewind` opened the actual modal and both directions
completed through their production action paths with exactly one auxiliary
completion each. The actual close/confirmation plus production
`open_console_workspace_conversation` route restored range scope, derived
banner, and next-send projection. The real serializer emitted exactly one app
memory with no user fallback or private identifiers. Separate resumes proved
regeneration before/inside/after as raw/raw/one-memory; an 80x24 resume verified
the narrow banner and reset/Undo/reset-all. The retained isolated DB is schema
55 with zero FK violations, two memories (one prefix and one range), three
selection events, two succeeded auxiliary rows, no Console sync-log entities,
and private `0600` DB/config files. Shared config/database fingerprints were
unchanged. External-provider success is not claimed because the isolated
profile intentionally had no API key; visual attachment behavior is covered by
the focused real-controller/provider tests rather than the mounted run.

No lesson was added: the disposable harness configuration issue is already
covered by the repository's testing/live-verification lessons. Non-blocking
legacy seams identified during review remain outside TASK-575 scope.

ADR required: yes. Canonical ADR:
`backlog/decisions/052-console-conversation-memory-and-compaction-policy.md`.
<!-- SECTION:NOTES:END -->

## Design Decision

- Approved design: [Console `/rewind` Summarize-from-here](../../Docs/superpowers/specs/2026-08-28-console-rewind-summarize-from-here-design.md)
- ADR required: yes
- ADR path: [ADR-052](../decisions/052-console-conversation-memory-and-compaction-policy.md), amended 2026-08-28
- Reason: TASK-575 extends the durable memory scope, atomic replacement, provider-context projection, and long-lived `/rewind` UX governed by ADR-052.
- Design review resolution: branch-local select/reset events replace record-global deactivation; exact CAS, legacy-baseline overrides, monotonic event ordering, migration foreign-key auditing, manual-prefix parity, canonical idle progress, and range-to-prefix automatic planning are specified and independently re-reviewed.
