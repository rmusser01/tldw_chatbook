---
id: TASK-575
title: 'Console /rewind: add a Summarize-from-here complement to Summarize-up-to-here'
status: In Progress
assignee: []
created_date: '2026-07-25'
updated_date: '2026-08-29 12:38'
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

Rebased cleanly onto `origin/dev` `4da99a8849` (dev schema v54); TASK-575 remains
the additive v55 migration, runner, fetched foreign-key audit, tests, and
allowlist update. Import provenance passed 1 test and printed this worktree's
`tldw_chatbook/__init__.py`. The exact focused 15-file suite passed
`1057 passed, 2 skipped` in 321.58 seconds; the skips were two host-denied
loopback-listener cases. The plan's exact Ruff scope passed, the schema guard
reported 75 declared ChaChaNotes tables all present, and diff guards were
clean. After independent review reopened the task, the directly affected
modal/controller/resume suite passed `177 passed` with 2 dependency warnings
in 136.75 seconds. No tracked production/test code changed in that evidence
fix, so no new Ruff scope was applicable. The full repository suite was not
run because it is opt-in and was not authorized.

Corrected live verification used isolated `TLDW_CONFIG_PATH` and
`[paths].data_dir` beneath
`.superpowers/sdd/2026-08-28-console-rewind-summarize-from-here-implementation/task-10-live-scratch-modal-resume-v6/`.
It printed and verified the exact pre-mount SQLite path
`.superpowers/sdd/2026-08-28-console-rewind-summarize-from-here-implementation/task-10-live-scratch-modal-resume-v6/data/task575_live/tldw_chatbook_ChaChaNotes.db`.
The exit-zero flow mounted 120x40 and drove visible-send `/rewind` into the
actual modal row/action path for both directions, exactly one deterministic
auxiliary call each. It then used the actual close-tab button and confirmation
plus the production `open_console_workspace_conversation` route; the resumed
screen-owned session restored range scope, banner, and next-send payload. The
real provider serializer produced one memory, no user fallback, and no private
IDs. Three separately mounted production resumes proved regenerate
before/inside/after as raw/raw/one-memory. An 80x24 production resume passed
reset/Undo/reset-all. External OpenAI was unavailable under an explicitly
empty isolated key and was not invoked; no external-provider success is
claimed. Shared config/database fingerprints were unchanged. The retained DB
is schema 55 with no FK violations, two succeeded content-free auxiliary rows,
one prefix and one range scope, and no `console_%` sync-log entity.

The first closeout probe's direct controller and helper-resume evidence was
withdrawn after review; it is not used for these claims. Fixing the retained
probe required no production change. Modified feature owners remain the four
Console Chat context/controller modules, ChaChaNotes v55 migration/runner/
allowlist, and the existing rewind, context-controls, settings, transcript,
and screen owners, with focused DB/Chat/provider/mounted-UI tests and approved
design/plan documentation. No lesson was added: the probe issues were
disposable harness wiring/configuration mistakes already covered by existing
testing/live-verification lessons.

ADR required: yes. Canonical ADR:
`backlog/decisions/052-console-conversation-memory-and-compaction-policy.md`.
<!-- SECTION:NOTES:END -->

## Design Decision

- Approved design: [Console `/rewind` Summarize-from-here](../../Docs/superpowers/specs/2026-08-28-console-rewind-summarize-from-here-design.md)
- ADR required: yes
- ADR path: [ADR-052](../decisions/052-console-conversation-memory-and-compaction-policy.md), amended 2026-08-28
- Reason: TASK-575 extends the durable memory scope, atomic replacement, provider-context projection, and long-lived `/rewind` UX governed by ADR-052.
- Design review resolution: branch-local select/reset events replace record-global deactivation; exact CAS, legacy-baseline overrides, monotonic event ordering, migration foreign-key auditing, manual-prefix parity, canonical idle progress, and range-to-prefix automatic planning are specified and independently re-reviewed.
