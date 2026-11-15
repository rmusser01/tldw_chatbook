---
id: TASK-39503
title: Keep FULL Console sync responsive during a held config rebuild
status: Done
created_date: 2026-11-08 23:50
labels:
- console
- performance
- ci
priority: high
updated_date: 2026-11-09 00:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3345's own UI8 full-ui-held case waited until its existing watchdog released the real rebuild hold. Restore the original responsive FULL contract while retaining native checked ownership and fresh replay.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A controlled native held-lock run reproduces the original assertion and identifies the exact suspended FULL await, native reader and holder under unchanged watchdog and case limits.
- [x] #2 FULL defers the measured busy native admission and converges after release, while admitted reads retain their checked leases, retirement and current-owner rules. Original assertions, action paths and limits remain.
- [x] #3 Original UI8 and directly affected coverage, original software UAT as required by changed inputs, scoped static checks and independent correction review qualify. Exact committed-head review and own hosted CI remain separate landing gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md and the existing Console checked-presentation contract in Docs/Architecture/console.md.
Reason: preserve the existing native read owner and screen-by-screen deferral, with no new authority, data or timeout contract.

1. Retain the original UI8 failure and the actual watchdog stack. Keep the separate dictionary-backend warning separate. Reproduce the original full-ui-held assertion under its genuine rebuild holder and unchanged two-second watchdog; passively bind the test coroutine's await chain to the exact native reader and holder. Use a real issued display proof and bounded preparation, not a patched getter, fake lock, proof clock or verdict.
2. If native lock admission is the cause, make only the stock display reader decline busy native REBUILD/FILE admission before entering the original checked operation. Keep exact lock order, native operation/body/retirement, source and visit fences, same existing replay/backoff, and awaiting/cancellation of reads that already own their leases. Do not change live action or custom reads, original assertions or any runtime limit.
3. Verify the identical controlled RED/GREEN, real lock contention/partial release and original admitted-read/cancellation controls. Qualify all original UI8 selectors plus directly affected files with original limits; preserve earlier repairs by measured continuity. Audit affected generated artifacts and rerun the original browser/mounted software UAT if production inputs change.
4. Complete static checks, notes and independent review; normal commit/hooks, exact tested blob/mode binding, independent exact-head review and one fresh actual-ref exact lease while auto is off. Successor OWN full automatic hosted gates remain mandatory. Keep all original completion records, earlier repairs, evidence and artifact files intact.
Post-attach custody fixture follow-up: guarded 817/818 preserved the original failed gate. The passive 818 trace proved the first post-attach read refused real REBUILD before entering native scope, then existing replay published a proof for the exact replacement config after the one-shot fixture had disarmed. Keep the native observer armed across those real retries until the second identity check of the same projection and replacement snapshot, then run the original heartbeat, active-lease and retirement assertions under their original deadlines. No product change beyond step 2, no fallback or new test wait.
Expanded response-rules evidence 820: all 138 original UI8 cases pass; one additional case asserts legacy controller copy before the first held provider chunk. TASK-38977 already records the same dev failure and the deliberate 077e5bfa09 UX. Keep the original transport/draft/count assertions and replace the superseded equality with the exact current waiting copy, proving an earlier verdict exists. A focused precedence negative must still fail at the exact copy assertion. Do not edit or claim completion of the other TASK-38977 prompts cases.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stock Console presentation refuses a busy native REBUILD or FILE lock before entering its checked operation and uses the existing replay. An admitted read keeps both locks, its checked source proof, storage leases and physical retirement. ADR-126's native ownership still applies. The original held-FULL and FILE controls reproduced and now pass under the original limits. The post-attach test now follows the same actual replacement snapshot through a refused first read and replay, and only holds a successful second native identity check. The uninstrumented original UI8 roster passed all 138 IDs. Its one supplemental stale assertion is separately covered by the retained real prior-verdict negative and the final native positive; the four unrelated TASK-38977 nodes stay outside this repair. The original software UAT and affected official checks are documented in the final qualification evidence. Exact committed-head review and the successor's own hosted checks remain separate landing gates. Browser UAT passed all 15 cases. The mounted parent and all three children passed (operation 4b7c8ad160454c98b44920be1372e2eb); all nine artifact pairs matched, guards stayed empty, and the ordinary reopen and disposal checks passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored responsive FULL config admission and passed original software UAT without changing limits or admitted ownership.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
