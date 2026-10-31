---
id: TASK-36610
title: >-
  Console: a send's start window can end early or never open, and its
  Inspector-visible path is unmeasured
status: To Do
assignee: []
created_date: '2026-10-24 07:50'
labels:
  - console
  - performance
  - ux-review-2026-09-29
dependencies:
  - TASK-33620.15.4
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.4 proposes keeping three Console display reads on the UI loop during a send's start, from acceptance to the provider request and 3 s at most: the Library activity rows and the Inspector's registry and appearance reads. Off the loop, each read opened its own SQLite connection and delayed first sends by ~66 ms. That exception still needs the owner's approval. If the owner rejects it, this task is void.

Its review found gaps in that window. All of them cost time only; none shows wrong data:
- The provider-request mark is process-wide, so another session's provider request ends this send's start early, and its reads go off the loop for the rest of the start.
- Queued-prompt dispatch and automatic submissions never note acceptance, so their reads go off the loop during their start.
- If acceptance is ever noted after the provider request, only the 3 s limit ends the window.

The live measurement (2026-10-23, Anthropic claude-haiku-4-5, 160x45, dev 1381e70735 as the base) ran with the Inspector and Context rails collapsed. The Inspector-visible path, whose reads the window exists for, was not measured. First sends alone showed no detectable difference (16 per arm: paired median +1.1 ms, mean +9.7 ms, branch faster in 8/16). That is weaker than a proven "no later".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A send's start ends at its own provider request, not at another session's
- [ ] #2 Queued-prompt dispatches and automatic submissions open a send's start at their acceptance, or measured evidence shows they need not
- [ ] #3 With the Inspector rail open, a send reaches the provider no later than on the base build (Enter to the provider request, live with a real provider, paired and interleaved, at least 30 sends per arm)
- [ ] #4 First sends alone reach the provider no later than on the base build (at least 30 first sends per arm, paired and interleaved, median and paired difference)
- [ ] #5 Deterministic tests fail on the base build for #1 and #2
<!-- AC:END -->
