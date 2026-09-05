---
id: TASK-31737
title: Fix durable trace ownership for resumed tool-loop calls
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:50'
updated_date: '2026-09-05 20:11'
labels: []
dependencies: []
documentation:
  - backlog/docs/validation-31737-tool-chain-snapshot-handoff.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow approved tool continuations to preserve the active turn in durable capture without guessing an earlier message or weakening fresh-send ownership checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual tool-loop calls and resumed calculator approval complete through the production trace boundary with the correct turn and chain.
- [x] #2 Missing, mismatched and stale chain ownership fail closed before provider dispatch; ordinary fresh-send and historical reconstruction behavior remain intact.
- [x] #3 Targeted regression and real mounted snapshot/approval validation are recorded with production support still gated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the tool-loop failure with real SQLite, controller, agent, gateway and production boundary; trace existing chain metadata and call ownership. 2. Resolve continuation ownership from its recorded first agent call, verify the currently supplied saved turn and frozen policy/owner, and preserve fresh-send fail-closed rules. Add missing/foreign/stale-chain negative controls before implementation. 3. Run whole targeted trace/controller modules, scoped lint/format/security and independent review. 4. Rerun actual mounted calculator approval with and without save/restart/restore, measure native warm/cold reuse, document limits and clean owned fixtures. ADR required: no new ADR. ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Reason: implement existing turn/run/call-sequence ownership for tool calls using the durable ledger; no new schema, authority, or capture policy. Reassess if a new cross-module identity contract is necessary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All remote refs (142) and worktrees swept before filing: maximum31736. CLI offered31715, so only this new untracked record was immediately moved to31737.

Independent review found stale-run admission when a newer chain used identical saved prompt surface. Added a failing same-surface/new-run regression, then atomically recorded the existing call_boundary event at reservation and required the prior logical call to be the latest boundary. This implements ADR097 ordered call ownership without schema changes. Re-review approved; pre-fix in-flight traces without boundaries remain fail-closed. Query-only100k-row schema-clone measurement: origin scan median2.199ms/p952.610ms; not an end-to-end/disk guarantee. New global index deferred as a scaling consideration, not claimed bounded by LIMIT2.

Implemented recorded tool-chain ownership in console_trace_runtime plus bounded-result origin/latest-call lookups in repository. Existing call_boundary event now commits atomically with reservation; no schema change. Real controller/agent regression and seven runtime ownership scenarios cover valid/deferred settlement, missing/foreign/policy/ambiguous/stale chains, including newer same-surface runs. TDD failures observed before each correction. Final whole targeted modules374passed1knownbaselinefailure1warning; catalog count29vs26 was independently reproduced unchanged dev in TASK31714. Storage release gates2passed84.43s: five fresh databases/scenario,200turns, median293280/316591bytes, linear growth below2MiB. No introduced Ruff/security findings (23 inherited), touched-range format and diff checks pass. Independent re-review approved; old in-flight traces without ordered boundaries remain fail-closed and origin scan scaling is measured/documented. Final real mounted calculator approval across6085-token save/restart/restore completed323; warm6054reused/73processed versus cold0/6127. No-lifecycle approval control also passed. Messages/durable records/settings unchanged by lifecycle; Pause/Resume unchanged, no autosends. Actual trace calls complete on same turn/run seq0/1, exception list empty. Production allowlist empty. Both owned processes/ports clear; zero copies14receipts, synthetic evidence retained. Documentation: backlog/docs/validation-31737-tool-chain-snapshot-handoff.md and updated testing lesson. Existing ADR097 applies. No push/merge or dependency installation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tool-loop ownership fixed under ADR097, with durable call-order supersession checks and fail-closed mismatches. Actual calculator approval now completes after restart/restore with measured cache reuse; production enablement remains gated. Targeted regressions, linear-storage gates and independent review completed; baseline catalog assertion and upgrade/scaling caveats documented.
<!-- SECTION:FINAL_SUMMARY:END -->
