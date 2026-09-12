---
id: TASK-25821
title: First-run wizard's final step removes every exit affordance
status: Done
assignee: []
created_date: '2026-08-31 05:08'
updated_date: '2026-08-31 13:50'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Steps one through five advertise Esc to leave setup and show it in the footer hint. The final step silently withdraws it: Esc stops working, the footer shows only Back, and no control is labelled Finish or Done. The only ways out are three body buttons whose names describe destinations rather than completion, so users cannot tell that setup is over.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Esc behaves consistently on every wizard step or its removal is explained on screen
- [ ] #2 The final step offers a clearly labelled completion action
- [ ] #3 The footer hint line matches the keys that actually work on the current step
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the genuine half. Steps 1-5 teach 'Esc skip setup'/'Esc exit setup' and Esc works; on the summary the cancel button is hidden and Esc goes inert, but the hint line just dropped the exit vocabulary -- so the key the wizard taught for five screens stopped working with no explanation and nothing said how setup ends. New SUMMARY_KEY_HINTS = 'Ctrl+B back · choose an action below to finish'. Deliberately does NOT mention Esc: it does not exit there, and a footer must only advertise keys that work (same rule as the Console footer's setup-blocked variant).

DECLINED: 'the final step offers no clearly labelled completion action'. It does -- #setup-exit-chat is dynamic, reading 'Start chatting' once provider and model are configured and 'Review provider setup' when they are not (pinned by test_first_run_wizard_live_contract). I saw the latter only because my provider was unconfigured, which is honest. 807 wizard tests pass.
<!-- SECTION:NOTES:END -->
