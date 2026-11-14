---
id: TASK-39372
title: Prepare the real checked display before the same-tick attachment focus regression
status: Done
created_date: 2026-11-08 21:59
labels:
- testing
- console
- ci
priority: high
updated_date: 2026-11-08 22:23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The original attachment Switch-model keyboard regression must reach its immediate render and focus assertions even when the native checked configuration display has aged during setup. Diagnose the real refusal before changing the test, and keep production ownership, proof lifetime, and the original same-tick focus ordering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A controlled native proof-age refusal reproduces the original immediate-sync assertion without replacing configuration getters, proof verdicts, or focus behavior.
- [x] #2 The test establishes its real checked-display precondition before the original same-tick capability change, dismissal restore and immediate sync, preserves every original assertion and original 5-second/180-second bounds, and still detects the historical hidden-button focus defect.
- [x] #3 The whole affected file and original UI31 selectors pass under unchanged limits; scoped static checks and independent review cover the correction and honest evidence, while exact committed-head review and its own hosted PR checks remain separate landing gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing Console checked-presentation and keyboard-focus contracts apply.
Reason: a bounded test-precondition correction, without changing production ownership or presentation behavior.

1. Retain the original hosted UI31 log, Test/production preimages, current HEAD/index and original assertion ASTs. Use the fresh all-ref/worktree/live-remote frontier's unused ID39372. Leave the original three Done tasks and QA README untouched.
2. Diagnose with the original test unchanged. An ignored, target-only pytest probe performs the existing real checked-display warm-up, holds the native config rebuild lock in a real thread while its genuine one-second proof ages, then physically releases the holder before returning from the existing final setup pause without another UI yield. Passively record the actual stock refusal/age/identity; count only a failure of the original immediate-True assertion with the expected native expiry branch. Preserve the 5-second preparation/render budget and case180; bounded cleanup must join the holder.
3. If proven, prepare the real registered checked presentation before the capability change. Preserve every original assertion and the no-await capability-change -> real dismissal restore -> immediate sync sequence, without retries, fake getters, new runtime limits, or production edits. Repeat the identical controlled probe once to verify the correction. For a separate semantic negative, temporarily replace only row.disabled = not show with row.disabled = False. This retains genuine visible-row enabling despite the disabled=True constructor and removes only hidden-row disabling. It must keep the original focused-Switch precondition and immediate sync True, then fail an original post-yield focus/draft assertion. Restore exact production bytes only after physical reaping.
4. Run the whole affected file plus all original UI31 selectors once under unchanged profiles/case limits, replacing any selected nodes from that file with the whole file once. Preserve UI19's completed204-case run through explicit unchanged consumed-input/AST continuity. Compare scoped static checks against the baseline, inspect original exceptions, and request independent working-source review.
5. After local ACs, record truthful notes/incident lesson, check AC via native Backlog and verify raw Markdown before Done. Normal repair commit/hooks and independent exact-new-head review/publication/its OWN hosted gates remain mandatory before landing. Do not restart completed UAT: retain only its measured unchanged runtime/package/UAT-source projection and explicit existing limitations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The attachment focus regression now prepares its actual stock checked configuration display once before changing model capability. Stage, Switch focus, its existing pause and native preparation share one five-second timeout. All six original assertion ASTs remain, with one preparation assertion added; every other module AST is identical. The capability change, real dismissal-focus restoration, immediate sync-True assertion and subsequent hidden-row/focus/draft/image assertions are byte-identical. Production proof lifetime, ownership, focus behavior and case180 remain unchanged.

Controlled RED811 ran the original test and failed its original immediate-True assertion: the same genuinely issued proof was 1.051790875 seconds old, and the real expiry branch returned False. The native holder was physically closed before the critical sequence. The identical probe passed812 after preparation. Negative813 changed only hidden-row disabling while retaining visible-row enabling: the original Switch-focus and immediate-sync checks passed, then the original post-yield composer-focus assertion failed on a hidden button. Exact production bytes were restored only after reaping.

Uninstrumented814 passed 384 unique cases, no skips/errors/failures, in 479.98 seconds: all 374 original hosted UI31 case IDs once plus the whole 11-case affected file. Only its one selected node was expanded; the other 64 selector arguments kept their order. No diagnostic probe was loaded. All owned processes reaped, source/mode/membership/HEAD/index guards stayed unchanged, and no thread/unraisable/holder-timeout marker was recorded. Ruff has zero findings and changed-function formatting passes. Independent working-source and completed-control reviews found no actionable issue; final local full-run review is retained separately.

ADR required: no new ADR; this test-only setup correction preserves existing checked-presentation and keyboard-focus contracts. The testing-evidence lesson records the native expiry and focus-negative incidents. Original logs, sources, native phase receipts, temporary negative source, JUnit and closure proofs remain under completion-metadata-20261108/UI31-failure. These controls establish a mechanism, not the unrecorded refusal branch of the hosted failure.

All runtime/package/browser/mounted-module/helper inputs remain unchanged. Original browser795/mounted796 UAT retains its bounded V1/native-child-origin-unobserved scope; no later UAT or complete old Test-map identity is claimed. Normal repair commit, independent exact-new-head review and its own full automatic hosted gates are separate mandatory landing requirements. The original three Done tasks and QA README remain untouched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared the genuine checked display before the original same-tick attachment focus regression, preserving all assertions and limits.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
