---
id: TASK-22587
title: Closing a session during a durable postcommit raises
status: Done
assignee:
  - '@claude'
created_date: ''
updated_date: '2026-08-26 21:11'
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
Reproduced at store level first: claim a postcommit effect, close the session, release the claim -> RuntimeError('Durable postcommit fingerprint changed.').

The bug was worse than filed. That raise comes out of the RELEASE path, which runs inside `except BaseException:`, so it REPLACED the exception that sent us there -- a genuine failure during a turn the user happened to close was reported as a fingerprint message.

Fix rests on a fact already in the code rather than a guess: retire_durable_acceptance leaves a tombstone carrying the SAME fingerprint, so 'retired' is decidable from 'changed'. New `_durable_retired_locked` reads it; retirement raises the distinct `ConsoleDurableAcceptanceRetired`; a real mismatch still raises the generic RuntimeError (pinned by a negative-control test, since narrowing a guard is only safe if it still fires for its original case).

Release-after-retirement is a no-op. Safe rather than merely quiet: retirement has already dropped every in-flight key for the preparation, so there is nothing left to release -- asserted in a test so the no-op cannot silently start leaking claims if that stops holding.

Controller: the release call can no longer replace the original failure, and retirement during a SUCCESSFUL effect is a non-event (work done, no ledger left to record it in).

MUTATION TESTING CAUGHT A TEST THAT COULD NOT FAIL -- mine. The first 'not masked by the release path' test passed with the controller guard removed, because the store fix already stops abandon from raising there. Split into one honest end-to-end test (which documents what it does NOT cover) and one that forces abandon to raise and does go red without the guard. That also made a '# pragma: no cover' I had written false, so it was removed. All 4 changes are now individually mutation-proven.

Verification: A/B of Tests/Chat against merge-base 65cf855371 -- 0 newly broken, +7 collected, 111 failures both sides. Preflight green; the +2 diagnostics were reviewed before regenerating (internal effect names only, exception TYPE not message, per TASK-22251).

Files: tldw_chatbook/Chat/console_chat_store.py, tldw_chatbook/Chat/console_chat_controller.py, Tests/Chat/test_console_close_during_durable_postcommit.py (new), Docs/security/production-diagnostic-inventory.json
<!-- SECTION:NOTES:END -->
