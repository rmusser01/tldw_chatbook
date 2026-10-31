---
id: TASK-624
title: Wire keyring convenience so skill trust auto-unlocks
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-25 11:20'
updated_date: '2026-10-31 10:04'
labels:
  - skills
  - trust
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`SkillTrustService.unlock_from_keyring_convenience()` exists, is unit-tested, and has **zero callers in the application**. The feature it implements is therefore dead: enabling "keyring convenience" persists the derived trust keys to the OS keychain specifically so the passphrase does not have to be re-entered, but nothing ever loads them back.

The user-visible result is that every launch shows *"Skill trust is locked for this session"* in Library ▸ Skills and demands a manual Unlock, even for a user who explicitly enabled convenience. The keys sit in the keychain unused.

`app.py` constructs the trust service with a `key_cache` wired **and** `keyring_convenience_enabled=False` hardcoded, then never attempts the cached-key unlock, so the flag can only ever become True inside a process that just called `enable_keyring_convenience()` itself.

Found by live TUI verification (task-579). Unit tests cannot catch this class of defect: the method passes its own tests in isolation precisely because those tests call it directly — the missing thing is the call site.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With keyring convenience previously enabled, a fresh app launch reaches a usable trust state without the user entering a passphrase
- [x] #2 The Library ▸ Skills trust header does not report "locked for this session" when cached keys are available and valid
- [x] #3 A failed or absent cached-key load still falls back cleanly to the existing locked/Unlock path — never a crash and never a silent unlocked-looking state
- [x] #4 A stale cache (salt no longer matching the manifest) does not unlock, matching the existing salt-bound behaviour
- [x] #5 Whatever surface enables convenience persists it, so the setting survives a restart rather than being reset to False on every construction
- [x] #6 A regression test asserts the auto-unlock is actually invoked at startup, not merely that the method works when called
- [x] #7 Repeated failed automatic unlocks avoid salt/keyring reads for 60 seconds after failure completes, then retry; all backend failure variants remain fail-closed
- [x] #8 Explicit Library Retry and direct convenience unlock can recover immediately; passphrase unlock, reset and recovery activation guards remain effective
- [x] #9 A successful explicit Retry refreshes the current Skills page so previously blocked rows and totals become usable immediately; cancelled, navigated-away, and superseded requests cannot refresh it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wired the cached-key unlock that existed but was never called.

Root cause: unlock_from_keyring_convenience() was unit-tested yet had ZERO production callers, so every lockedness decision saw self._keys is None and reported 'locked for this session' while the derived keys sat unused in the keychain. The method passed its own tests precisely because those tests supplied the call site production lacked.

Fix: a single memoized _try_cached_unlock() called at the four points that decide lockedness — trust_posture(), status_for_skill(), trusted_file_paths(), _require_keys(). Putting it behind the choke points rather than at app startup means every consumer benefits (Library UI, agent worker thread, Console skill paths) with no startup keychain cost, and it stays lazy.

Design notes: the attempt is latched to once per trust-state generation because keychain reads are not free and posture is queried on every Library render (pinned by a counting-cache test asserting exactly 1 call across 10 queries); the latch clears in reset_trust so a later re-bootstrap can cache again. AC#5 needed no config key — the keychain entry IS the persistence, and unlock_from_keyring_convenience already flips keyring_convenience_enabled on success. Failure is silent by design: a missing, stale, or exploding cache leaves the session locked so the existing Unlock path remains the fallback. Also added the module's missing loguru import.

LIVE-VERIFIED in a real app (the point, since unit tests could not catch this): same QA profile, before = 'Skill trust is locked for this session.' with warning glyphs on both skills; after = 'Skill trust: ready.' with checkmarks, no unlock and no passphrase. Capture at Docs/superpowers/qa/skills-script-execution-2026-07-25/after-fix-trust-ready.txt.

Tests: Tests/Skills/test_skill_trust_keyring_autounlock.py (8 tests) RED-first — 5 failed against the old code, 3 (locked/stale/absent-cache fallbacks) passed throughout, proving no false baseline. Skills 357, Agents 261, ruff clean.

Files: tldw_chatbook/Skills_Interop/skill_trust_service.py, Tests/Skills/test_skill_trust_keyring_autounlock.py


### PR #3134 repair (2026-10-31)

Supersedes the historical negative-latch description above. Failed automatic
unlock attempts now wait 60 seconds from completion, including missing keys,
salt errors, RuntimeError and KeyringError. Direct unlock remains immediate;
Library Retry uses the existing posture worker and refreshes both in-memory
keyring read caches under the existing recovery guards. Stored credentials and
trust authority do not change. Passphrase unlock, success and reset clear the
cooldown. Governing decision: [ADR-009](../decisions/009-local-skill-trust-boundary.md);
no new ADR required.

Rebased the original fork head 0334fa4ca84008e5a06980807e1a06e7987042a7 onto
198eca8b93a04b9399ac8d26f310833a82a959a0. Modified the trust service/store,
Library posture action, focused Skills/UI/activation tests, and Library
architecture/evidence notes. The activation test's TOML fixture now writes a
portable path; its unescaped Windows backslashes previously failed parsing
before the tested guard ran.

Verification: the service regressions first produced 10 expected failures and
10 passing controls. All 31 targeted Skills/keyring cases then passed (39
unrelated cases deselected). The mounted Retry test holds both clocks fixed,
uses the real button and worker, and verifies fresh off-thread key/marker reads.
Removing only retry=True made it fail with unavailable instead of ready; the
exact file was restored. Retry and ordinary posture both passed the restored
activation test; the Library size checks passed at 37757 lines/1361 methods.
No full suite was run. Native keyring behavior was represented by the secure
backend test double, not a live OS keyring.

Two additional pre-existing Library tests could not reach their bodies in this
profile setup: test_missing_trust_service_supersedes_in_flight_posture_worker
and test_stale_skills_posture_cannot_project_after_route_switch raised
raw_source_selection_changed while creating the app. The attempted baseline
comparison lost its console result; no baseline pass/fail claim is made.
Touched-file lint has no new findings (63 existing UI findings, exact comparison
to dev); existing formatter debt remains outside the changed blocks. All touched
Python compiles and git diff --check passes. Initial overlapping automatic
workers are not serialized; the cooldown bounds subsequent attempts.

Task stays In Progress because backlog-py is broken on this host and independent
review/remote integration are handled at handoff. No push, PR comment, auto-merge
or merge was performed.

### Independent review correction (2026-10-31)

The first mounted test seeded a healthy row, so it missed a stale blocked page
behind the recovered header. The revised test creates and lists a real local
skill while the keyring is failing; it failed at blocked_total=1 after Retry
reached ready. Explicit Retry now carries its intent through the existing
posture worker and, only after success and the existing route/generation/view
fences, reuses the scoped committed-mutation refresh. The real row becomes
trusted, its blocked total clears, actions become usable, and query/sort remain
intact. Cancelled, navigated-away, superseded, and still-failing retries retain
their previous page.

Fresh verification: 14 passed, 2 deselected in 43.84s across the five mounted
Retry cases, all seven Skills browse-controller tests, and both Library size
checks. The negative run failed at the intended stale blocked-count assertion.
One intermediate cancellation control raised Textual's expected WorkerCancelled;
the test now explicitly awaits that cancellation before checking retained state.
The production files and new test block compile, git diff --check passes, and
baseline comparison finds zero new lint findings (67 existing across the three
touched Python files). The controller formats cleanly; the canvas test keeps
only its two pre-existing formatter hunks. No full sweep, native OS keyring run,
or remote operation was performed. ADR-009 remains the governing decision.

<!-- SECTION:NOTES:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR #3134 repair (2026-10-31):

ADR required: no
ADR path: backlog/decisions/009-local-skill-trust-boundary.md
Reason: restores the existing recoverable, fail-closed convenience contract without changing trust authority or storage.

1. Rebase the existing PR onto dev 198eca8b93a04b9399ac8d26f310833a82a959a0 and trace automatic decisions plus the actual Library Retry worker.
2. Add failing behavioral tests for salt/backend retry suppression, expiry after slow failure, generic backend errors, explicit retry, passphrase/reset, and the mounted Library action.
3. Limit the failure cooldown to automatic attempts and make explicit Retry invalidate transient keyring read caches within the existing activation guards; preserve worker dispatch.
4. Run targeted Skills and Library/activation checks plus touched-file static analysis, review the diff and record exact evidence. No full suite.
5. Review follow-up: reproduce Retry with a genuinely blocked source page, refresh that page only after a successful current Retry, and verify cancellation/navigation/generation fences before committing the correction.

The installed backlog-py CLI currently fails before execution (uv trampoline cannot canonicalize its script path); this repair updates the existing record directly and leaves it In Progress pending CLI closeout.
<!-- SECTION:PLAN:END -->
