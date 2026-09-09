---
id: TASK-21500
title: 'Console: expose Improve current draft and recommend Review'
status: To Do
assignee: []
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 04:46'
labels:
  - console
  - prompts
  - ux
  - uat
dependencies: []
references:
  - .impeccable/critique/2026-08-24T04-39-32Z__chatbook-widgets-console-console-prompts-modal-py.md
  - Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md
  - backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user with an unsent Console draft begin improvement directly, without first navigating through the Prompt Library. Preserve Library browsing as a separate destination and make Review the visibly recommended, safe first-use improvement mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the composer contains a nonblank unsent message, its menu exposes a distinct `Improve current draft…` item that opens the improvement mode chooser directly.
- [ ] #2 `Browse Prompt Library…` remains a separate, plainly named destination; an empty composer does not offer an actionable improvement path and still provides direct Library access.
- [ ] #3 The improvement chooser visibly marks Review as `Recommended`, gives it initial keyboard focus on first entry, and does not start any provider request until the user chooses a mode.
- [ ] #4 Direct entry captures the current unsent message, optional current System prompt, and current Console provider/model exactly as the existing workbench does; it does not silently select another provider or model.
- [ ] #5 The user can include or exclude the System prompt before starting any of the three improvement modes, with the choice preserved when entering Recipe mode.
- [ ] #6 Menu order, focus restoration, Escape behavior, and the 140x40, 100x30, and 80x24 layouts remain keyboard-reachable and free of clipped labels or actions.
<!-- AC:END -->

