---
id: TASK-21236
title: Persona Buddy lazy-controller residue - unlocked setter and an untested ensure-fallback
status: Done
assignee: []
created_date: '2026-08-23'
updated_date: 2026-10-12 01:17
labels:
- personas
- concurrency
- test-coverage
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: close-out of the 2026-08-22 holistic performance review burn-down; the two Minors the
TASK-21103 review ledgered rather than fixed. Both confirmed still present on dev `b2b1e2e0d`.

TASK-21103 (PR #2002, `6c0abdba7`) made the Persona Buddy controller lazy, taking PIL and 93%
of `Persona_Visual` off the boot path — **−80 modules**, and the PIL chain alone was
**1.276 s of the 3.10 s cold import** measured by the review.

1. `_build_persona_buddy_controller` guards its check-then-build with
   `_persona_buddy_controller_lock` (`app.py:6213`). The public property setter
   (`@persona_buddy_controller.setter`, `app.py:6289`) writes `self._persona_buddy_controller`
   with **no lock**, so a setter call concurrent with a build can be lost, or can replace a
   controller another caller already holds a reference to. Today the setter's only callers are
   tests and skeletal doubles on the UI thread — so this is theoretical, but it is theoretical
   only because of a caller property nothing enforces.
2. The ensure-fallback arm added for explicit Buddy actions
   (`personas_screen.py:6666-6668` — read the passive property, and when it is None call
   `ensure_persona_buddy_controller()`) has **no screen-level test**. That arm is the entire
   reason "Use for Buddy" works from a profile whose preferences still say disabled.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The controller slot cannot be written outside the lock that guards its construction, or the setter is shown unreachable from any concurrent context and that constraint is enforced rather than assumed
- [x] #2 A screen-level test drives the current Manage Buddy action (successor to Use for Buddy) from a profile whose preferences say disabled and asserts the management form opens through the constructed controller
- [x] #3 That test fails when explicit management stops ensuring a controller for a disabled profile
- [x] #4 TASK-21103's import-closure guard stays green
- [x] #5 Actor Pack and Buddy-conversion activation retain atomic source/UUID/visual authority checks while performing profile-directory preflight outside the database-only operation scope.
- [x] #6 Persona Actor Packs preserve validated policy rules on export/import, reject malformed rules without dropping them, and exclude local authority fields.
- [x] #7 The unchanged screen pre-import module budget stays green after Buddy startup laziness is restored, and character generation retains its existing requested-action controller.
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md and backlog/decisions/079-workspace-assistant-defaults.md and backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md and backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md and backlog/decisions/126-complete-local-backup-and-recovery.md (existing)
Reason: retain the existing lazy controller, lock, independent management and storage-admission authority. The routine activation repair separates ordinary directory preflight from unchanged atomic source/UUID/visual revalidation under existing ADR-126; no new durable ownership or storage policy.
1. Reproduce setter publication while the construction lock is held using real thread synchronization.
2. Acquire that same lock in the existing setter; retain lazy construction and injection/clear behavior.
3. Add a mounted Roleplay Manage Buddy journey from disabled saved settings; prove the existing ensure path matters with a temporary negative control.
4. The initial focused run exposed two startup-fixture profile admission failures and an eager Actor_Packs attribution import that violates the unchanged boot guard. Retain the startup tests on their admitted bootstrap profile and defer attribution imports inside the existing validation and archive-reading functions, preserving all validation and original assertions. Verify attribution contracts as well as the focused setter, startup, management and import-closure checks, scoped static comparison and independent review, then record evidence and close the task. The broader attribution controls exposed an inherited transaction-scope refusal. Separate ordinary profile-directory preflight from atomic revalidation; retain character and Persona checks and the conversion source guard. Verify unchanged round trips, rollback and stale-source controls before closing. The remaining Persona export refusal is the ADR-079 policy_rules field added to every local profile. Carry that existing Persona data through the portable allowlist with strict validation from the existing PersonaPolicyRule model; verify valid/invalid contracts and a real export/import round trip. ADR-079 already requires policy rules to travel with Persona Actor Packs; no new policy or schema is introduced.

5. Repair the CI pre-import ratchet under existing ADR-097 by moving the same generation controller/runner imports into the existing explicit-action helper after its test override. Preserve all budget constants. Verify unchanged census, existing generation behavior and a real default-controller self-check; retain actual CI RED and updated source hashes.
The selected legacy generation suites use collection-bound config/internal prompts and fail raw_source_selection_changed under per-test profile replacement; both representative failures reproduce before the screen import move. Keep these two modules on their admitted private bootstrap profile, preserve original bodies/assertions, and rerun all affected generation checks.

6. Current-head CI follow-up: supply the existing constructor-owned threading.Lock in the two skeletal TldwCli injection fixtures (Console shutdown and Actor Pack refresh). Preserve the locked public setter and every original lifecycle assertion. Capture both missing-lock REDs, verify corrected tests plus setter/startup controls, compare scoped static diagnostics with the previous head, review independently and retain evidence. Existing ADRs apply; fixture repair only.
The adjacent Actor Pack shutdown test also omits its existing synchronous raw CLI owner and shared shutdown-task slot. Its function body and production lifecycle match the previous head, and the admitted cohort control reaches the missing slot. A standalone attempt was refused during profile setup and is excluded from defect attribution. Provide a no-op synchronous owner and the task slot while preserving the original Actor Pack teardown order assertion; retain the intermediate failure and baseline control.
Once the raw CLI drain runs, the same baseline skeleton also lacks the constructor-owned empty meeting slot. Retain that intermediate result and initialize _meeting_session_owner to None, matching the unstarted production owner. No lifecycle method or ordering assertion changes.

7. The unchanged Library folder handoff fails again in UI shard 2 on fixture head ac8f1d83. Capture mounted/disabled/parent and processed folder press, expanded IDs and branch/worker state before choosing a correction. Reproduce with controlled root-response scheduling using the real app/services where needed. Preserve the delayed-root exact-folder readiness control and all Delete/Undo assertions. Correct only the proven handoff seam with existing mount-settling/response authority patterns; keep every timeout and required gate unchanged. Record RED, final results, source attribution and independent review. ADR required: no for a routine handoff/fixture repair within existing Library ownership; reassess if evidence requires a new boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-11: Resumed under the requester-approved Buddy burn-down. The service wiring moved to app_service_wiring.py and the explicit action is now Manage Buddy under ADR-139. Retain the existing lazy import and ownership contracts; serialize setter publication with the existing construction lock and qualify the current mounted action from disabled saved preferences. Focused tests only; accepted human/native/voice UAT is preserved.
Completed controller setter locking and mounted disabled-profile Manage Buddy coverage, with a failing ensure-path mutation. Deferred attribution imports preserve the boot guard. Repaired inherited database-only activation scope refusal by keeping directory preflight outside SQL and exact source/UUID/visual revalidation inside Character and Persona commits; conversion before-insert and final source guards remain. Carried validated Persona policy_rules through reviewed Actor Packs under ADR-079 without dropping malformed rules or local authority exclusion. 104 affected activation/conversion/policy cases and 26 final boundary cases pass, 129 unique identities; expected RED/mutation controls retained. Zero introduced Ruff/Bandit findings; ten files formatted, existing service-wiring formatting debt unchanged, compile/whitespace pass. Independent review has no actionable findings. Report and exact evidence: Docs/Reviews/2026-10-11-chatbook-buddy-closeout.md and artifacts/buddy-closeout-20261011/verification.json. ADR-074/079/126/139 apply; no new ADR. No full suite, accepted UAT repetition, provider/audio call or normal-profile mutation.
Recorded the traced stale-review/storage-scope incident and its verified boundary repair in backlog/docs/lessons-testing-evidence.md.
Current-head CI exposed the adjacent pre-import ratchet: 559 modules against558 because Persona_Visual moved from eager boot into the screen pass. Under existing ADR097, defer the existing CharacterGenerationController/build_gateway_runner import until its explicit action (all runtime callers are the same screen helper; its test override remains first). Keep module/LOC limits unchanged. Verify the exact budget and existing character-generation controller/UI checks, then retain updated source attribution and fresh independent review. Disarmed normal auto-merge before modifying this PR.
The CI pre-import failure is repaired under ADR-097 by deferring the existing generation controller imports until requested. All 22 budget, controller and UI cases pass at 558/558 modules and 417835/425347 lines of code; the headroom warning is recorded. Two profile-admission failures reproduce before the import move. Keeping those modules on their admitted private bootstrap profile preserves all original function/class bodies. The final six-case gate passes both mounted Buddy journeys, three startup guards and the actual default constructor/override check. Across the retained runs, 151 unique repository tests and the constructor check pass. Follow-up static checks introduce no Ruff or Bandit findings; compilation and whitespace pass, with existing formatting debt preserved. Independent review found no import or caller issue. The October 11 report and receipt retain final source sets, the CI failure and controls. Protected current-head CI remains required before merge.
CI fixture follow-up: both skeletal Buddy setter callers now supply the constructor-owned lock. The unchanged Actor Pack teardown skeleton receives its synchronous no-op raw CLI owner/task and empty meeting slot. All original assertions and production lifecycle methods remain intact. Both missing-lock RED failures and admitted cohort missing-owner failures are retained; the standalone profile setup refusal is explicitly excluded from defect attribution. Final targeted gate 10/10 passes, with no introduced scoped Ruff/Bandit findings and preserved Console formatting debt. Whole-module AST proof excludes only the exact constructor prerequisites/import. Across retained successful runs, 158 unique repository tests plus the constructor check pass. Independent code and final receipt reviews have no remaining findings. Fresh protected CI remains required before merge.
Library CI follow-up: repeated current-head timeout traced with a controlled real canvas/prune lifecycle. The original exact-folder helper finds a mounted but pruning/hidden row; its press sends no event and no child load starts. One pilot.pause before the existing fresh domain query repairs this helper. Same lifecycle control passes 1/1 and both original real Delete/Undo cases pass 2/2; all assertions and timeout constants remain intact and production source is unchanged. Whole-module AST proof excludes only that await. Ruff 0/0, Bandit 16/16, formatting/compile/whitespace pass. Retained exact source and byte-preserving events/logs/XML in the closeout receipt; 160 unique repository tests plus the constructor check pass across recorded runs. The CI schedule itself was not captured; fresh required CI still qualifies publication. Lesson records this distinction. Existing Library ownership applies; no new ADR.
Final independent code and evidence review has no findings. All nine Library control artifacts and exact raw bytes, source/static hashes, XML results and 161=160 repository plus constructor union verified. All AC/DoD checks complete; publish through required CI and the normal merge queue.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->


## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Controller publication, disabled-profile management, atomic Actor Pack/conversion guards, policy portability and unchanged startup/pre-import budgets are verified. Constructor fixtures and real Library test handoff retain original assertions. Across recorded runs 160 unique repository tests and the constructor check pass; accepted historical UAT is preserved.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done

<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests and negative controls recorded
- [x] #3 Documentation updated
- [x] #4 Scoped static and Bandit baseline checks complete
- [x] #5 Independent review and final summary recorded
- [x] #6 Known coverage limits documented
<!-- DOD:END -->
