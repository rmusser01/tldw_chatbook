---
id: TASK-37173
title: Prepare copyable bug reports from retained diagnostics
status: In Progress
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
updated_date: 2026-10-28 21:31
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
- [ ] #7 Focused privacy, repository, rendering, clipboard, UI, packaging, and design-governance checks pass, with live evidence of the report and pasted bytes.
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
Implemented a shared editable, memory-only bug report from Quick Actions, Logs and Settings About. Defaults include retained app-wide logs/exception traces and selected Console events/timing/errors; tool inputs/results require opt-in and immutable source governance. Reports and edited copy candidates are redacted and bounded to 60,000 UTF-8 bytes. No uploader, new recorder, schema, dependency or draft persistence was added.

Bounded repository reads retain timed-out work through retirement. The modal preserves edits, guards dismissal/quit, rejects stale generation/copy results and truthfully says Copy requested. A fresh whole-branch review found four substantive issues; all were reproduced and fixed: off-loop alias projection, real surface-event tool projection, step/run outcome separation and causal agent ordering.

Verification: 57 affected tests pass after fixes; 5 post-fix entry/import checks pass; renderer/sanitizer 163 pass; task/ADR guards pass. Native WindowsDriver/TTY journeys cover all three entry points, redacted exception evidence, Calculator results through the real capture service, manual edits, copy requests and discard guards. Received OSC52 packets match reviewed bytes (Palette 2,762; About 2,429) and recorded SHA-256 hashes. No OS clipboard acknowledgement is claimed.

ADR required: yes. ADR: backlog/decisions/252-bug-report-evidence-and-sharing.md. Evidence and limitations: Docs/superpowers/verification/2026-10-28-report-a-bug.md. Plan: Docs/superpowers/plans/2026-10-28-report-a-bug.md. Core files: Utils/bug_report.py, Utils/bug_report_capture.py, Widgets/bug_report_dialog.py, bounded log/DB/trace readers, shared action/entry points, focused tests and user/architecture guidance.

AC #7 stays unchecked and task remains In Progress: existing shared governance/focus/provider/profile-fixture failures prevent claiming the repository Definition of Done. No full suite ran. New helpers/tests format clean; two fatal lint annotation findings reproduce unchanged on dev. Unrelated test debt was left visible.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
