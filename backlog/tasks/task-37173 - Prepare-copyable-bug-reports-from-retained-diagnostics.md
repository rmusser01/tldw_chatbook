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
updated_date: 2026-10-28 17:04
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users a readable report containing their description and retained diagnostic evidence so they can share it manually for diagnosis without reconstructing the incident by hand.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can open one report flow from the command palette, Logs, and Settings About without changing their active work.
- [ ] #2 The report includes available session logs, captured exception traces, environment facts, and selected Console events, timing, and errors with explicit source scope and missing-evidence notices.
- [ ] #3 Tool inputs and results are excluded by default and included only through explicit opt-in after source governance and diagnostic redaction; source messages, provider exchanges, and reasoning fields remain excluded.
- [ ] #4 The editable reviewed Markdown is the exact validated clipboard payload and stays within 60000 UTF-8 bytes including formatting and manual edits.
- [ ] #5 Source failures, read deadlines, concurrent logging, runtime changes, and dialog cancellation preserve the draft and cannot mix evidence or apply stale results.
- [ ] #6 Report creation performs no upload, LLM call, domain/config mutation, automatic draft write, or new capture/retention change.
- [ ] #7 Focused privacy, repository, rendering, clipboard, UI, packaging, and design-governance checks pass, with live evidence of the report and pasted bytes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/252-bug-report-evidence-and-sharing.md
Reason: Introduces a user-controlled diagnostic export contract across logs, governed Console evidence, and clipboard output; existing ADR-029, ADR-080, and ADR-097 remain binding.

1. Implement and verify the pure report data, privacy projection, Markdown budgets, and exact-copy validation.
2. Add bounded read-only log/Console/agent metadata adapters and governed tool-field reads with source deadlines and frozen references.
3. Build the shared describe/review modal with opt-in controls, guarded dismissal, draft preservation, and truthful client clipboard delivery.
4. Wire the command palette, Logs, and Settings About; update user/architecture guidance; run focused checks and live paste verification.

Detailed plan: Docs/superpowers/plans/2026-10-28-report-a-bug.md
Approved spec: Docs/superpowers/specs/2026-10-28-report-a-bug-design.md
Implementation has not started; execution method is selected after review of the detailed plan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
