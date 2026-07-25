---
id: TASK-639
title: >-
  First-run healing branch re-flips deliberate profile switches away from
  Imported settings
status: To Do
assignee: []
created_date: '2026-07-25 21:54'
labels:
  - followup
  - uat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-634/635 review (Minor): ensure_imported_profile()'s self-healing branch (active_config.py:367-370) re-activates imported_settings on EVERY first RAG-touch-in-process where the active pointer differs from imported_settings and the profile already exists on disk -- it cannot distinguish a genuinely half-done first run (crashed before activating) from a user who deliberately switched away to a different profile afterward. A user who explicitly Set-active'd to a different profile, then restarted the app, would get silently switched back to Imported settings the next time anything touches get_shared_rag_service() for the first time in that new process. This fix should also account for cleaning up configs already damaged by the pre-635 always-import bug (a fresh user who got an unwanted imported_settings profile created and activated before this fix shipped).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The healing branch only re-activates imported_settings when the active pointer is still the default builtin (i.e. a genuine half-done first run), never when the user has since deliberately activated a different profile
- [ ] #2 A migration/cleanup path exists (or is explicitly scoped out with rationale) for configs already carrying an unwanted imported_settings profile + pointer from before task-635 shipped
- [ ] #3 Existing half-done-first-run healing regression coverage (test_ensure_imported_profile_heals_half_done_first_run) still passes
<!-- AC:END -->
