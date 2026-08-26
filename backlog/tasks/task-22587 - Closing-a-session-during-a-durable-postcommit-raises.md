---
id: TASK-22587
title: Closing a session during a durable postcommit raises
status: In Progress
assignee:
  - '@claude'
created_date: ''
updated_date: '2026-08-26 20:48'
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
- [ ] #1 Closing a session during an in-flight durable turn does not raise
- [ ] #2 The in-flight effect is released (or deliberately abandoned) without an
      unhandled exception
- [ ] #3 A test covers close-during-collection on a DURABLE session
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
The fingerprint guard is correct in intent — an effect must not commit against a
retired preparation. What is missing is a defined outcome for "the preparation
was retired underneath me", distinct from "the fingerprint changed
unexpectedly". A retired preparation is an ordinary consequence of the user
closing a chat; a *changed* one is a bug. Those two currently share a raise.

Related: TASK-22301 (converting that module's assertions to row queries needs
durable sessions, so this blocks it).

## Renumbering provenance

Filed as task-22303; renumbered to task-22587 under the 2026-08-21 owner rule
(TASK-19601) after a same-id collision with
`task-22303 - Restore-priced-Console-cost-chip-harness-readiness`, which
arrived first and keeps the id. No dependencies or doc references pointed at the
old number.
<!-- SECTION:NOTES:END -->
