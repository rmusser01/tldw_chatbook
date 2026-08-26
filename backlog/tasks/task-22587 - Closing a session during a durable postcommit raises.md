---
id: task-22587
title: Closing a session during a durable postcommit raises
status: To Do
labels:
  - console
  - durable-turns
priority: medium
---

## Description

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

## Acceptance Criteria

- [ ] Closing a session during an in-flight durable turn does not raise
- [ ] The in-flight effect is released (or deliberately abandoned) without an
      unhandled exception
- [ ] A test covers close-during-collection on a DURABLE session

## Implementation Notes

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
