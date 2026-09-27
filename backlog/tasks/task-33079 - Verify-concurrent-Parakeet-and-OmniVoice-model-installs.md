---
id: TASK-33079
title: Verify concurrent Parakeet and OmniVoice model installs
status: To Do
assignee: []
created_date: '2026-09-27 18:30'
labels:
  - wizard
  - model-artifacts
  - tts
  - stt
dependencies:
  - TASK-32958
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first-run wizard can now install two curated models into the same managed artifact store: Parakeet in the Speech step and OmniVoice (1.1 GB) in the Voice step. Both installs run on background workers and neither blocks Next, so a Full-track user can start one while the other is still downloading. The design relies on the store's per-artifact locks to keep the two installs apart, but TASK-32958's live UAT never ran them at the same time (it would have needed a second 1.1 GB download). This task closes that gap before a user finds it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With both installs running at once on a fresh profile, both finish, both models verify, and each step then reports its model as ready
- [ ] #2 Cancelling or failing one install while the other runs leaves the other install unaffected and leaves no partial files for the failed one
- [ ] #3 An automated test pins concurrent provisioning of two different artifacts in one store, using fake sources (no real download)
<!-- AC:END -->
