---
id: TASK-39122
title: "The response-rules manager's status line raises NoMatches when a write settles during teardown, and the region editor's opener loses the close when its worker is cancelled before mount"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- ui
- console
- personas
priority: low
references:
- tldw_chatbook/Widgets/Console/response_rules_modal.py
- tldw_chatbook/UI/Persona_Modules/portrait_regions.py
- Tests/UI/test_response_rules_modal.py
dependencies:
- TASK-38979
- TASK-38980
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-modals, which lands TASK-38979 and TASK-38980. Line numbers are dev's.

**The status line.**
- `ResponseRulesModal._status` (`Widgets/Console/response_rules_modal.py:193-195`) guards its `query_one("#rr-status")` on `is_mounted`. Textual 8.2.8 sets that flag True on mount and never resets it.
- So a write that settles while the dialog's children are being removed raises `NoMatches`. `_buttons` (`:446-447`) already guards by querying, with the comment "Owned workers can finish while Textual removes the dialog's children."
- It was seen in a round-11 run: `test_inherited_exclusion_masks_and_exact_promotion_preview` raised `WorkerFailed: … NoMatches("No nodes match '#rr-status' …")` from `_action` after `await self._reload(...)`. That path is unchanged from dev.

**The region editor's opener.**
- `edit_regions`' `finally` (`UI/Persona_Modules/portrait_regions.py:127-131`) retires the editor only `if modal.is_mounted`, and clears the controller's reference either way.
- If the worker is cancelled after the push but before mount, the `finally` skips the retire. A probe of that shape leaves the editor on the stack once it mounts, and a later `discard()` no longer finds it.
- On fu/r11-modals `PortraitRegionEditor.retire` handles a pre-mount call, but this `finally` never makes one.
- This path was not reproduced through the real Personas controller.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A response-rules action that settles while the manager's children are being removed raises nothing; a deterministic test is red on dev
- [ ] #2 When the regions worker is cancelled after the push and before mount, the editor still closes and the controller's `close()` returns; a test that holds the mount is red on dev
<!-- AC:END -->
