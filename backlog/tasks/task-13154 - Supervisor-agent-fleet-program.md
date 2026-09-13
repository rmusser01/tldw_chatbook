---
id: TASK-13154
title: Supervisor agent fleet program
status: Done
assignee: []
created_date: '2026-08-09 13:57'
updated_date: '2026-09-13 04:33'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Named sub-agent definitions, background/parallel execution, steering, Console fleet panel. Spec: Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All approved fleet outcomes are delivered and verified: definitions, concurrency runtime, fleet panel, cross-turn lifetime, wake and notification, steering and continuation, and phase-four polish; every remaining TASK-13154 child is Done.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md plus current accounting/admission and implementation decisions. Reason: records-only reconciliation after all implementation children complete. Execute Docs/superpowers/plans/2026-09-12-agent-fleet-program-closeout.md last: verify six historical core deliveries and the seventh polish slice, checked child criteria/reviews, prior and fresh evidence with limits, current authority/remaining optional extensions, independent records and final branch review, then Backlog Done. Do not invent retroactive children or erase unfinished outcomes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Historical PR-1 review triage is preserved below; it is not a claim that the remaining work is complete. Current child tasks and the closeout plan reconcile each item.

### Deferred to later phases (final-review triage, 2026-08-09)

The final whole-branch review of PR-1 (named agent definitions) accepted
these as real but out of scope for the fix wave that landed the review's
findings. Recorded here because the SDD ledger that carried the full triage
(`.superpowers/sdd/2026-08-08-supervisor-fleet-pr1-agent-definitions/`) is
gitignored and does not survive merge.

- **PR-2a** — convert the spawn-closure disjoint-path `assert` to a `raise`
  (an `assert` is stripped under `python -O`, silently turning a real
  invariant violation into undefined behavior); add a load-once-per-turn
  call-count guard where `run_turn` changes.
- **PR-2b** — memoize/close the Settings ▸ Agents panel's `AgentRunsDB`
  handle. It is currently opened fresh on every category visit and relies
  on garbage collection to close the underlying connection rather than an
  explicit lifecycle.
- **Phase-4 polish** — give per-save feedback when `RUNTIME_TOOL_NAMES`
  entries are silently dropped from a typed tool list, so a user who lists
  `spawn_subagent` sees why it didn't stick.
- **Owner taste call** — where the Agents category belongs in Settings
  navigation (Troubleshooting vs. Expert) is a placement judgment call, not
  a defect; left for the owner to decide.

Interim reconciliation (2026-09-12, superseded by final closure below): all seven actual TASK13154 children are Done with checked acceptance criteria and implementation notes. This supersedes the historical deferred labels above, while preserving their original review context. The parent remains In Progress, with its acceptance unchecked, because final integration and the broader requested worktree outcomes are unresolved.

Six core deliveries are present in reachable local history: definitions PR1461 f24f8c6921; concurrency PR1477 7625968469; fleet panel PR1498 2ff4c27084; cross-turn lifetime PR1557 d5445a4c10; wake/notification PR1609 b456263894; steering/continuation PR1816 230acdaac0. These are historical ancestry checks recorded at 770a1735d5, not fresh runtime or live-provider tests. PR2631 merged the 25-item audit; PR2641 merged the five subsequent reliability/verification follow-ups at d66908a69. No duplicate historical children were created and wake delivery is counted once.

The seventh polish slice is implemented and independently reviewed locally: approval verification TASK13154.4 (686ca653cc/abf4a0d879), owned Settings DB cleanup and filtered-tool feedback TASK13154.5 (97c3bce2e0/1b9cff9776), researcher/critic/ingest-runner plus retained bulk-reader presets TASK13154.6 (6f05d7399a), and definition caps TASK13154.7 (b46807d547/be709a648f, bbfc233aad, c0d419d420). Named spawn now raises ValueError for agent plus allowed_tools, and the real load-count/no-reread guards are present; the current 122-pass fleet runtime gate covers the explicit disjoint-path and once-per-turn tests. Each child record contains exact scoped evidence and qualifications. Canonical Settings category placement remains unchanged as a product preference.

Current communication is bounded process-local steering and child progress, explicit supervisor collection/relay, explicit finished-child continuation and shared versioned session tasks. This does not claim durable inboxes, arbitrary direct peer routing or progress-triggered wakes. Existing ADR129/131/134/135/136 govern communication, durable accounting, aggregate admission, delivery and scoped progress. Supplemental ADR153 governs best-effort bounded webhook delivery (drops remain possible), ADR154 denial boundaries, ADR156 provider/local live usage, and ADR157/158 capped continuation plus migration order.

Historical interim checkpoint (superseded below): worktree safeguards temporarily refused create/merge/discard and retained existing checkouts/branches. The proposed execution qualification requirement was later withdrawn in favor of the user-approved ordinary local Git scope in ADR155. TASK31210 and TASK31211 retained unchecked functional criteria. TASK18929 was reopened for this wave's ordinary budget-stop continuation regression; two positive worktree continuation tests and a deferred per-chunk usage extraction observation remained final review obligations. No functional recovery, durable inbox, full-suite run or live-provider certification was claimed. Existing dependency/static warnings, the earlier preset-probe user-config read before sandboxed write failure, and the cap Task 1 automatic foreign pytest-cleanup attempt remain explicitly qualified in child/current-status records.

ADR required: no new ADR for this records-only reconciliation. Existing backlog/decisions/129-fleet-mailbox-and-wake-reliability.md and the exact supplemental decision paths in the linked closeout plan govern delivered behavior. The implementation plan was narrowed to allow honest partial reconciliation without prematurely closing the parent. Independent records review and combined branch review are pending. Work remains local on codex/agent-orchestration-remaining; no new PR, push or merge was performed.

Final integration checkpoint: all four broad-review findings were repaired in 7852cf47ba and the single scoped re-review approved all four with no new Critical/Important breakage. TASK18929 and TASK18923 are Done again with their new regression criteria checked. The continuation module now passes 45 cases without teardown errors, and provider/error plus denial/cancel selections pass. The partial records review was also approved. Complete review, evidence qualifications and all 50 controller rulings are durably preserved in Docs/superpowers/reviews/2026-09-12-agent-orchestration-remaining.md. No additional parent child or retroactive completion was invented; no PR/push/merge is implied.

Interim restoration checkpoint (superseded by final closure below): ordinary selected-authority worktree creation, schema20 durable ownership and actual physical-owner drain integration are implemented and independently reviewed (a11db916ff/d63d4f2a19, 4756f88793/94f7110db1, 8efca6ee05/c8e1d119b8/eed41ac5d2). There is no missing external backend. Confirmed apply/merge/discard and the visible Console recovery flow are being completed under Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md; until those outcomes pass review, TASK31210, TASK31211 and this parent remain In Progress. The current evidence and restoration rulings are in Docs/superpowers/reviews/2026-09-12-agent-worktree-restoration.md. Automatic root deletion remains disabled; ordinary Git has the documented concurrent external metadata-replacement limitation.

Final closure checkpoint (supersedes interim remaining-work labels above): all approved fleet outcomes and the requested worktree restoration are implemented and independently reviewed through e135a085f2. TASK-31210 and TASK-31211 now have all acceptance criteria checked and Done status. Fresh local records verification confirmed all seven actual TASK-13154 children are Done with zero unchecked acceptance criteria and implementation notes; no retroactive children or historical test results were invented.

The restoration includes ordinary selected-authority Git creation, schema 20 durable ownership and positive physical completion, shared confirmed apply/merge/discard, a visible retained card and paged earlier-turn recovery. Final integration found two startup-lifetime gaps; both are corrected and independently re-reviewed. The final fix has11 pre-source RED failures,88 passing affected/capacity cases and2 passing actual card/Git flows. Exact earlier selections and process/guard qualifications are preserved without summing overlapping counts.

Verification is targeted, not a full-suite or live-provider claim. The inherited Requests warning remains; UI-ready passes at 973/973 with zero headroom. Existing ChatScreen size/no-growth guards (including the disclosed 21-line/3-method addition) and eight stale historical diagnostic-label expectations remain failures. The current diagnostic inventory/sink guard passes, and changed-file static comparisons add no diagnostic identities with edited-range formatting passing.

Current communication remains bounded process-local steering/progress, explicit supervisor relay and continuation, and versioned session tasks. Durable inboxes, arbitrary direct peer routing and progress-triggered wakes are optional designs outside this completed scope. Recovery retains uncertain/legacy work and its disclosed baseline checkout; ordinary Git does not promise atomic protection against concurrent external metadata replacement.

ADR check: the existing decisions named above plus ADR-155/158 govern delivery; no new ADR for records-only closure. The complete restoration review, evidence and rulings are in Docs/superpowers/reviews/2026-09-12-agent-worktree-restoration.md; the earlier remaining-wave review stays linked and preserved. Source and closeout records remain committed locally on codex/agent-orchestration-remaining.
<!-- SECTION:NOTES:END -->
