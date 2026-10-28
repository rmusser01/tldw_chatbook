---
id: TASK-37173
title: Prepare copyable bug reports from retained diagnostics
status: Done
created_date: 2026-10-28 17:02
assignee:
- '@codex'
labels:
- diagnostics
- privacy
- ux
references:
- Docs/superpowers/specs/2026-10-28-report-a-bug-design.md
documentation:
- backlog/decisions/252-bug-report-evidence-and-sharing.md
updated_date: 2026-10-28 22:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users a readable report containing their description and retained diagnostic evidence so they can share it manually for diagnosis without reconstructing the incident by hand.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can open one report flow from the command palette, Logs, and Settings About without changing their active work.
- [x] #2 The report includes available session logs, captured exception traces, environment facts, and selected Console events, timing, and errors with explicit source scope and missing-evidence notices.
- [x] #3 Tool inputs and results are excluded by default and included only through explicit opt-in after source governance and diagnostic redaction; source messages, provider exchanges, and reasoning fields remain excluded.
- [x] #4 The editable reviewed Markdown is the exact validated clipboard payload and stays within 60000 UTF-8 bytes including formatting and manual edits.
- [x] #5 Source failures, read deadlines, concurrent logging, runtime changes, and dialog cancellation preserve the draft and cannot mix evidence or apply stale results.
- [x] #6 Report creation performs no upload, LLM call, domain/config mutation, automatic draft write, or new capture/retention change.
- [x] #7 Focused privacy, repository, rendering, clipboard, UI, packaging, and design-governance checks pass, with live evidence of the report and pasted bytes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/252-bug-report-evidence-and-sharing.md
Reason: User-controlled diagnostic export contract; existing ADR-029, ADR-080 and ADR-097 remain binding. The shared-check follow-up implements existing modal, CSS and profile contracts and requires no new ADR.

1. Implement and verify the pure report privacy projection, Markdown budgets and exact-copy validation.
2. Add bounded read-only diagnostic adapters with source deadlines and frozen references.
3. Build the editable describe/review modal, opt-in controls and guarded dismissal/copy flow.
4. Wire palette, Logs and Settings About; document and verify the feature and native pasted bytes.
5. Resolve shared checks requested on 2026-10-28: normalize governance paths/encoding and pin-script imports, align focus/provider tests with canonical token/Settings contracts, give uncovered guarded modals real quit hooks, and repair type-only annotation imports.
6. Rerun the named affected checks, self-review changes, update verification notes, and check AC #7/mark Done only after those pass. Keep full-suite execution outside this targeted verification scope.

Detailed plan: Docs/superpowers/plans/2026-10-28-report-a-bug.md
Approved spec: Docs/superpowers/specs/2026-10-28-report-a-bug-design.md
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the shared editable, memory-only Report a bug flow from Quick Actions, Logs and Settings About. Reports include retained app-wide logs/exception traces and selected Console events, timing and errors. Tool inputs/results require opt-in and immutable source governance. Rendering and exact edited clipboard candidates are redacted and bounded to 60000 UTF-8 bytes; no upload, new recorder, dependency, schema or automatic draft persistence was added.

Bounded adapters retain timed-out work through retirement. The modal preserves drafts, guards dismissal/quit, rejects stale generation/copy delivery and truthfully says Copy requested. The feature review's four substantive findings were reproduced and fixed: off-loop alias projection, production surface-event tool fields, separate step/run outcomes and per-run causal ordering.

Resolved all requested shared failures: Windows governance paths/UTF-8 reads, standalone pin imports, canonical Settings/token/provider contracts, type-only annotation imports, three missing quit hooks and reasoned census exemptions. The focused shared review exposed goal completion under a covering quit prompt; a mounted RED-to-GREEN regression verifies exact-owner delivery after resume plus retained configuration/error updates. The palette navigation fixture owns its profile and awaits finite Skills workers before retirement; native storage gates and retirement budgets are unchanged.

Final verification at 478fd5b328: 605 unique passing targeted cases (395 combined report/privacy/UI/focus/palette/census; 31 design/component/bundle; 179 lifecycle/goal/exchange/extraction). Fatal lint across all changed Python files passes; default Ruff checks on 13 report/shared-repair modules and new tests pass; 19 formatting targets pass. Task uniqueness/readability and ADR guards pass. No full suite ran. Self-review and independent review findings are resolved, with no deferred minors.

Native WindowsDriver/TTY journeys cover all three entry points, redacted failure evidence, real captured Calculator results, manual review edits, copy requests and discard guards. Decoded OSC52 payloads match reviewed bytes and stored SHA-256 receipts (Palette 2762 bytes; About 2429); no OS clipboard acknowledgement is claimed.

ADR required: yes for the feature; backlog/decisions/252-bug-report-evidence-and-sharing.md governs. Shared-check follow-up requires no new ADR because it enforces existing contracts. Plan: Docs/superpowers/plans/2026-10-28-report-a-bug.md. Final evidence/rulings: Docs/superpowers/verification/2026-10-28-report-a-bug.md and its linked receipts. Updated lessons record exact-top-screen ownership and awaiting native screen work before test teardown.

Core files: Utils/bug_report.py, Utils/bug_report_capture.py, Widgets/bug_report_dialog.py, bounded log/trace/DB readers, shared app action and three entry points, focused tests and user/architecture guidance. Shared repairs cover governance/focus/provider/census tests and goal/response-rule/portrait quit hooks. Local commits and the managed worktree are preserved for review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
