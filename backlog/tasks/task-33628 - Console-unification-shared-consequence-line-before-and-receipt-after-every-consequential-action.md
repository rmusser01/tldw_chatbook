---
id: TASK-33628
title: >-
  Console unification: shared consequence line before and receipt after every
  consequential action
status: To Do
assignee: []
created_date: '2026-09-30 03:04'
labels:
  - console
  - ux-review-2026-09-29
  - unification
dependencies: []
references:
  - qa/console-ux-review-2026-09-29/report.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: many Console actions change history, spend money, discard work or grant authority without warning first or offering a way back. Delete removes the selected message and every later turn under singular copy with no undo (P1). Discard leaves the prompt in model context. Summarize, Comment and Continue spend tokens silently. /rewind leaves an empty transcript with no banner. A character swap relabels earlier replies and silently replaces a custom system prompt. One Enter on the Hands-free switch starts the microphone pipeline (P1). Ctrl+Q quits without warning and loses the Temporary chat, drafts and parked unsent turns (P1), and the quit dialog, when shown, reports zero unsent work (P1). Single-call 'Approve once' ignores an 'Always' selection. File approvals do not name the root folder or distinguish read from write. Trace records Stop and timeout denials as approvals. Exports land in the working directory, Downloads, the clipboard or Notes depending on the feature.

What unifying means here: one consequence-and-receipt model, built on the Archive and Fork pattern that already works. Before any action that sends to the model, spends, deletes, changes history or grants authority, a shared feed-forward line states what will be sent, the estimated cost, the scope and whether it can be undone. Afterwards, a shared receipt states what happened and offers Undo and open-location where they apply. Tab close, leaving the Console and quit share one loss projection. Approval cards resolve paths to the bound folder and the access mode. All exports go through one destination resolver that never writes to the working directory. This umbrella is the parent of the theme's P0/P1 fixes (G1-02, G2-01, G4-11, GAP2-15).

Evidence: qa/console-ux-review-2026-09-29/report.md (theme 'Consequential actions') and qa/console-ux-review-2026-09-29/findings.md / findings.json.

Related P2/P3 in the ledger: G1-11, G1-40, GAP5-03, GAP5-04, GAP5-12, G1-32, G1-16, G2-10, GAP4-10, GAP4-12, G3-14, G3-39, GAP2-05, G1-38, G4-63, GAP5-16, G1-20, G4-51, G3-06, GAP4-08, GAP4-09, GAP4-14, GAP4-16, GAP5-10, GAP5-14.

Source: Console UX review 2026-09-29 — qa/console-ux-review-2026-09-29/report.md (themes) and findings.md (ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Console action that sends to the model, spends, deletes, changes history or grants authority shows, before it runs, one shared consequence line stating what it sends or affects, its estimated cost where applicable, its scope, and whether it can be undone
- [ ] #2 Every such action ends with one shared receipt stating what happened and offering Undo or an open-location action where one applies, and destructive history changes (delete, rewind) can be undone from that receipt
- [ ] #3 Closing a tab, leaving the Console and quitting all report the same accurate count of work that would be lost (drafts, Temporary chats, unsent turns, live runs), and none of them discards that work without confirmation
- [ ] #4 File-tool approval cards name the bound folder a path resolves into and whether the call reads or writes, and the recorded Trace outcome matches the decision actually taken (approved, denied, timed out or stopped)
- [ ] #5 Every Console export or saved artifact goes through one destination resolver, and none is written to the process working directory
- [ ] #6 All P0/P1 child tasks under this umbrella are Done
<!-- AC:END -->
