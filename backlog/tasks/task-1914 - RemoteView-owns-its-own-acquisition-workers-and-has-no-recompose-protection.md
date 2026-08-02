---
id: TASK-1914
title: RemoteView owns its own acquisition workers and has no recompose protection
status: To Do
assignee: []
created_date: '2026-08-02 14:57'
labels:
  - models
  - architecture
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
model_remote_view.py (added by PR #1190, TASK-596.1) drives preflight/provision itself via _preflight_model/_provision_model/_confirm_install and imports ArtifactAcquisitionService at module scope. This is the same boundary violation TASK-1803 just fixed in CuratedView -- views post intents, the host screen owns the worker -- but RemoteView has no compensating delivery logic at all, so a screen-level recompose mid-install orphans the worker and progress stops reaching the UI with nothing to catch it. The module-scope acquisition import also sits against the rule that only functions may import acquisition/fetch. Fix by mirroring TASK-1803: move the workers to LLMScreen, have RemoteView post intents.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 RemoteView posts intents; LLMScreen owns the preflight/provision workers
- [ ] #2 A screen-level recompose mid-install does not orphan the worker or lose progress, proven by a test
- [ ] #3 acquisition is no longer imported at module scope in model_remote_view.py
<!-- AC:END -->
