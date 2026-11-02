---
id: TASK-38641
title: Two portrait modals keep their 120-column width on full-screen terminals
status: To Do
created_date: 2026-11-02 21:52
labels:
- follow-up
- ui
- modals
- css
priority: low
references:
- Tests/UI/modal_wide_tier_registry.py
- Tests/UI/test_modal_wide_tier.py
- tldw_chatbook/Widgets/Persona_Widgets/portrait_composition_review.py
- tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py
dependencies:
- TASK-36382
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36382's re-sync (PR #3227, 2026-10-30). The wide-tier reverse sweep named two modals that dev had added after TASK-36522 was triaged:
- `PortraitCompositionReview` (Widgets/Persona_Widgets/portrait_composition_review.py)
- `PortraitRegionEditor` (Widgets/Persona_Widgets/portrait_region_editor.py)

Both size their dialog with `$ds-inspector-modal-width`, which is 120 columns. They went into `MODAL_WIDE_TIER_SKIPPED` marked "follow-up wave", which the registry comment calls a queued gap, not a verdict.

The tier's ladder gives a base width of 100 or more a cap of 170 at `WIDE_VIEWPORT_COLUMNS` (150) and above. The app is used full-screen (211x44 and 235x52), where both modals open at 120 columns. Both show artwork: one reviews eye-row pairs and source crops, the other edits regions over the source.

This is the same work as TASK-36522's ten modals, filed apart so that task's scope stays as triaged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 211x44 and 235x52, each portrait modal opens at the 170-column wide-tier cap, or the registry records why it stays at 120
- [ ] #2 Below the wide viewport, each modal's geometry is unchanged
- [ ] #3 Each tiered modal has a registry anchor and no skip entry, and the wide-tier coverage contract passes for them
- [ ] #4 The boot CSS byte budget still passes
<!-- AC:END -->
