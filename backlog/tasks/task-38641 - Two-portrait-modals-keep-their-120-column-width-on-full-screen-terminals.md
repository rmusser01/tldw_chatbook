---
id: TASK-38641
title: Two portrait modals keep their 120-column width on full-screen terminals
status: Done
assignee:
- '@claude'
created_date: 2026-11-02 21:52
updated_date: 2026-11-03 00:32
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

The tier's ladder (`modal_wide_tier_registry.py`: base width >= 104 -> 170) gives a 120-column base a cap of 170 at `WIDE_VIEWPORT_COLUMNS` (150) and above. The app is used full-screen (211x44 and 235x52), where both modals open at 120 columns. Both show artwork: one reviews eye-row pairs and source crops, the other edits regions over the source.

This is the same work as TASK-36522's ten modals, filed apart so that task's scope stays as triaged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 211x44 and 235x52, each portrait modal opens at the 170-column wide-tier cap, or the registry records why it stays at 120
- [x] #2 Below the wide viewport, each modal's geometry is unchanged
- [x] #3 Each tiered modal has a registry anchor and no skip entry, and the wide-tier coverage contract passes for them
- [x] #4 The boot CSS byte budget still passes
<!-- AC:END -->

## Implementation Plan

1. Tier both dialogs the way wave 2 tiered its 100-120 column modals: one `App.-wide-viewport <anchor> { width: $ds-percent-85; max-width: $ds-size-170; }` rule each in the wide-tier block of `css/components/_agentic_terminal.tcss`, anchored on each dialog's root class (`.portrait-pairs-dialog`, `.portrait-regions-dialog`); rebuild the bundle with `build_css.py`.
2. Registry: add both anchors under cap 170 with their owner modules; delete the two "follow-up wave" rows from `MODAL_WIDE_TIER_SKIPPED`.
3. Tests (`Tests/UI/test_modal_wide_tier.py`): add both dialogs to the spot representatives (base `120 / 100%`, cap 170) and measure them with `run_test(size=...)` at 211x44 and 235x52 (cap 170) and below the wide viewport (base 120 / viewport width); show the new cases red before the CSS lands.
4. Check the coverage contract (the two names leave the reverse sweep's output, nothing new joins it), the boot CSS byte budget, bundle sync via preflight, and the size ratchets.

## Implementation Notes

Both portrait dialogs joined the wide tier the way wave 2 tiered its 100-120 column modals. Each has one `App.-wide-viewport` rule (`width: $ds-percent-85; max-width: $ds-size-170`) in the cap-170 group of the wide-tier block in `css/components/_agentic_terminal.tcss`, and the bundle was rebuilt with `build_css.py`. The anchors are the dialogs' root classes, `.portrait-pairs-dialog` and `.portrait-regions-dialog`. Each class is unique to its dialog, so no type prefix is needed, and the class-only selector is the shortest in boot bytes. The app-tier rule outranks the dialogs' widget-default base rule (`width: 120; max-width: 100%`) only when the App has `-wide-viewport`. Below 150 columns nothing changes.

- Registry (`Tests/UI/modal_wide_tier_registry.py`): two cap-170 anchors with their owner modules; the two "follow-up wave" skip rows are deleted.
- Tests (`Tests/UI/test_modal_wide_tier.py`): both dialogs are spot representatives (base `120 / 100%`, cap 170), each built idle (`is_current` False, so no artwork job runs). They are measured with `run_test(size=...)` at 200x50, 211x44 and 235x52 (cap 170), at 149x44 and 120x40 (base), and across live resizes in both directions.
- Red/green: before the CSS, the 8 wide cases (200x50, 211x44, 235x52 and grow-past-threshold, for each dialog) failed `assert 120 == 170`. The 6 narrow cases (120x40, 149x44 and shrink-below-threshold) passed on dev and still pass, which is the AC#2 evidence that base geometry is unchanged. After the CSS, all 14 portrait cases pass.
- Coverage contract: the static contracts pass. The reverse sweep is still red only at its `missing` assert, with the same 7 unrelated names as on dev (TASK-38230). Neither portrait class is among them. With those 7 temporarily skip-listed (reverted), the whole sweep passed, stale-skip checks included.
- Boot CSS: 595,812 -> 596,011 B (+199 B) against the 608,090 B limit.
- Not caused by this change: `test_component_pattern_governance` (dimension literal, Python style, sheet ceiling), the fastpath bare-type ratchet (Console recovery buttons 8 != 7) and the `chat_screen.py` screen-size ratchet are red. The first four were run at dev with the same command and were red with the same messages there. The `chat_screen.py` ratchet reads only files this change leaves identical to dev.
- User Guide: `roleplay-chat-dictionaries/characters-and-personas.md` describes both dialogs but not their width, so no page change.
- The new mounted spot cases are not on a PR lane. The census runs `test_modal_wide_tier.py` by node, listing only its static contracts, which cover AC#3 on the lane. Every earlier spot case is in the same position.

