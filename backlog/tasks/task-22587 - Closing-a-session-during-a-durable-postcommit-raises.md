---
id: TASK-22587
title: Closing a session during a durable postcommit raises
status: Done
assignee:
  - '@claude'
created_date: ''
updated_date: '2026-08-26 21:42'
labels:
  - console
  - durable-turns
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Closing a Console session while a durable turn is still collecting raises
`RuntimeError: Durable postcommit fingerprint changed.` out of the effect
release path.

Mechanism: closing retires the durable preparation, so
`_durable_fingerprint_by_preparation` no longer holds the fingerprint the
in-flight effect captured. `_require_durable_fingerprint_locked`
(`console_chat_store.py:3853`) then raises, and the release path has no handler
for it.

Found while converting `test_console_local_citation_boundary` to durable
sessions (review follow-up for #2104). Two tests there —
`test_citation_repair_session_close_privacy_sentinels` and
`test_citation_repair_close_during_collection_never_resurrects_session_or_message`
— close the session mid-collection and hit it immediately once the session is
durable rather than ephemeral. They pass today only because that module uses
ephemeral sessions throughout.

The conversion was reverted rather than worked around, so this is currently
latent in tests but reachable in production: any user closing a chat while a
durable turn is in flight takes the same path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Closing a session during an in-flight durable turn does not raise
- [x] #2 The in-flight effect is released (or deliberately abandoned) without an
      unhandled exception
- [x] #3 A test covers close-during-collection on a DURABLE session
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce at store level: claim an effect, retire the preparation, release the claim -> RuntimeError (DONE: confirmed on dev)
2. Distinguish the two causes the guard currently conflates. retire_durable_acceptance already leaves a tombstone carrying the SAME fingerprint, so 'retired' is decidable: tombstone present AND fingerprint matches = ordinary retirement; anything else = genuine mismatch.
3. Add a distinct exception type for retirement so callers can handle it without string-matching, and keep the existing RuntimeError for a real fingerprint change.
4. Make the RELEASE path tolerate retirement: abandon_durable_postcommit_effect on a retired preparation is a no-op, not a raise. There is nothing left to protect.
5. Fix the masking bug in _run_durable_postcommit_effect: its 'except BaseException: abandon(); raise' arm currently loses the ORIGINAL exception when abandon raises.
6. Mutation-prove each behaviour change, then A/B the Chat/Console test set against the merge-base.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
--- ROUND 2 (Qodo review of PR #2123) ---

Both findings were real and both were verified empirically before fixing. The
diagnosis: round 1 fixed the HELPER, not the ORCHESTRATION. Suppressing
retirement inside one effect only moved the raise to the next
fingerprint-validated effect, or to the unconditional retire ending the sequence.

Retirement is now terminal-benign for the whole orchestration:
1. the early durable_postcommit_effects_for lookup is guarded -- it sits BEFORE
   the try block, so guarding only the sequence left it raising (this is what
   actually fires first);
2. a sequence-level except arm covering all 8 effects;
3. retire_durable_acceptance is idempotent for the SAME acceptance (keyed on the
   tombstone fingerprint, with a negative control proving a different acceptance
   on a retired id still raises);
4. new durable_completed_effects_for reads completed names from ledger OR
   tombstone, so the failure handler's provider_started question survives a
   close instead of raising inside the handler.
Both paths share _postcommit_stopped_by_close, which runs the normal cleanup
tail minus the owner-changed check.

MUTATION TESTING CAUGHT UNPROVEN CODE A SECOND TIME: after the round-2 fix,
3 of the changes stayed GREEN under mutation, because every test closed the chat
BEFORE resume and the early guard short-circuited the rest. Added a mid-sequence
close (inside effect #1 of 8) plus store-level tests; all 8 mutations now go red.

Round-2 verification: 13 tests; A/B of Tests/Chat vs merge-base 65cf855371 ->
collected 7682->7695 (+13), failures 111->111, 0 newly broken; preflight green
(+1 diagnostic, a constant string with no interpolation).
<!-- SECTION:NOTES:END -->
