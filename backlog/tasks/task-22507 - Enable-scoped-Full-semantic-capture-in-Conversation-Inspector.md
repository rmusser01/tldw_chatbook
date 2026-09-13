---
id: TASK-22507
title: Enable scoped Full semantic capture in Conversation Inspector
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 14:34'
updated_date: '2026-08-26 23:48'
labels:
  - console
  - privacy
  - ui
  - db
  - transparency
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-26-console-full-semantic-capture-design.md
  - backlog/decisions/092-console-full-semantic-capture-policy.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users deliberately retain complete semantic provider exchanges for one eligible send, one conversation, or all Console conversations from the Inspector or live Trace screen so injected context, tool traffic, and provider-specific payload content can be diagnosed without weakening the default privacy boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Conversation Inspector and live Trace screen expose one shared Safe/Full capture flow for the next eligible send, the inspected conversation, and the global default, with deterministic precedence and active-run freezing; imported Traces remain read-only.
- [x] #2 Full capture retains semantic provider inputs and outputs, including Anthropic system content, project and workspace instructions, RAG context, tool schemas, tool calls, and tool results, while structured credentials remain excluded, request/response binary data becomes bounded stubs, and in-memory/compressed/decompression limits remain enforced.
- [x] #3 Capture policy changes use scope-appropriate confirmation, visible lifecycle and Capture Off/resume states, immutable inspected-conversation targeting, honest partial-write recovery, and fail closed without changing an admitted run.
- [x] #4 Each persisted exchange records consistent queryable capture detail, historical exchanges remain backward compatible as Safe, and corrupt provenance mismatches fail closed.
- [x] #5 Users can delete stored Full captures across every branch and soft-deleted message of one quiescent conversation without deleting Safe captures, messages, usage, exports, backups, or changing capture policy; deleted captures cannot be re-persisted or exported from a stale Inspector.
- [x] #6 Capture detail and export profile remain distinct, with confirmed Full clipboard and filesystem exports and accurate per-call provenance in the Exchange view.
- [x] #7 Targeted automated tests cover policy precedence and consumption, provider and injected-context capture, persistence migration and purge, concurrency and ephemeral behavior, export safety, and production-shaped 80x24 keyboard and focus behavior.
- [x] #8 The governing privacy and storage ADR, user documentation, and implementation notes describe retention, compression-not-encryption, logical deletion and WAL/free-page limits, provider-boundary caveats, and the default Safe behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Treat Docs/superpowers/specs/2026-08-26-console-full-semantic-capture-design.md and ADR-092 as the approved contract.
2. Execute TASK-22507.1: add Safe-first capture construction, bounds, provenance, schema migration, and local policy persistence without exposing Full in the UI.
3. Execute TASK-22507.2 after TASK-22507.1: resolve and consume scoped policy at admission, freeze it on provider signals, and cover direct/retry/tool/fleet/Anthropic/llama.cpp paths.
4. Execute TASK-22507.3 after TASK-22507.1 and TASK-22507.2: add conversation-wide Full-capture count/purge under quiescence with staged cache replacement and capture-revision fences.
5. Execute TASK-22507.4 after TASK-22507.1, TASK-22507.2, and TASK-22507.3: expose the shared Inspector/live Trace/F9 flow, governed per-call export, responsive styling, documentation, and production-shaped verification.
6. Follow Docs/superpowers/plans/2026-08-26-console-full-semantic-capture.md task-by-task, close each child only with its focused evidence, then run the final integration gate and close this parent.

ADR required: yes
ADR path: backlog/decisions/092-console-full-semantic-capture-policy.md
Reason: ADR-092 governs the persisted privacy metadata, provider/runtime capture boundary, logical deletion semantics, and shared UI/storage contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented ADR-092 across four reviewed child tasks: Safe-first v51 exchange provenance/persistence, frozen scoped next-send/conversation/global policy, quiescent conversation-wide Full purge, and shared Inspector/live Trace/F9 controls with governed per-call export.
- Full capture now includes Anthropic system content, project/workspace instructions, RAG context, tool schemas/calls/results, and provider semantic bodies while structured credentials, paths, and binary/base64 content remain excluded across chunks, caches, SQLite, exports, logs, and exception graphs.
- Final whole-branch review found and closed policy hydration, stream aggregation, Safe-before-disk, config projection, purge truthfulness, admission fallback, and DB exception-chain defects. Final reviewer verdict: Ready to merge — Yes, with no Critical or Important findings.
- Final verification at `c5d18abc50`: fresh exact privacy/UI gate `886 passed, 2 sandbox-only loopback skips`; complete DB gate `1831 passed, 1 Windows-only skip`; Task 2 gate `570 passed, 2 sandbox-only skips`; 80x24 `114 passed`; Settings/config/layout `381 passed`; lowest DB boundary `12 passed`; Ruff, py_compile, all-five CSS sync, and diff checks clean. The repository-wide absolute screen-size ceiling remains a pre-existing stale failure, while the feature delta passes at 20,093 lines/633 methods versus 20,099/633 base.
- Documentation, ADR-092, implementation reports, and the aggregate-stream sanitation lesson are updated. No new ADR was required for the final corrections because they enforce the existing Safe-first and content-free boundary.
- The final `dev` rebase preserved upstream's v49-to-v50 Console Library tombstone cleanup, moved Full-capture persistence to v50-to-v51, restored the approved governing spec and plan from the former stacked base, and renumbered the now-colliding capture decision from ADR-089 to ADR-092.
- The post-rebase Qodo review produced 11 actionable findings, all addressed: transactional policy reads, complete public API documentation, owner-only export files with atomic no-clobber semantics, generation-safe config publication, store-level purge/write exclusion, and Inspector recovery after quiescent startup. One nullable-type comment was declined because purge bindings intentionally require a concrete revision; the separately typed Inspector freshness callback remains nullable and is now documented explicitly.
- Final post-Qodo evidence: the eight new regressions failed before their fixes and passed afterward; the focused fix gate passed 90 tests; the complete 16-file privacy/UI matrix passed 890 tests with 2 sandbox-only loopback skips; changed-file Ruff, production py_compile, and `git diff --check` passed. These corrections enforce ADR-092, so no new ADR or lesson entry was required.
- Post-push Derived Artifacts CI found two missing review records. Statement-level inspection of all eight changed diagnostic owners confirmed that added messages carry only stable phases, identifiers, and exception classes; raw exception/path-bearing calls were removed and no persistent sink topology changed, so the reviewed diagnostic inventory was regenerated. The v51 capture-detail index now has a no-`sqlite_stat1` `EXPLAIN QUERY PLAN` regression against both exact Full inventory and delete queries plus a `plan-pinned` census row. Both artifact checkers, the 35 affected DB/guard tests, Ruff, and diff checks pass locally.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Scoped Full semantic capture is complete on `codex/full-semantic-capture`. Safe remains default; Full is deliberately scoped and acknowledged; injected context and tool/provider exchanges are inspectable; credentials/binary/path data remain excluded; purge/export are revision-fenced and truthful; imported traces remain read-only. Independent final review and targeted integration verification are clean.
<!-- SECTION:FINAL_SUMMARY:END -->
