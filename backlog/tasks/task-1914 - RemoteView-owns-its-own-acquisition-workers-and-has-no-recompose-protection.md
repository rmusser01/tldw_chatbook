---
id: TASK-1914
title: RemoteView owns its own acquisition workers and has no recompose protection
status: Done
assignee: []
created_date: '2026-08-02 14:57'
updated_date: '2026-08-03 00:47'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Merged as PR #1245 on 2026-08-03. LLMScreen owns remote preflight/provision; RemoteView posts intents; remote search stayed on the view (read-only listing, matching _load_curated's precedent). One shared install lock across curated+remote with _model_install_kind routing progress to the right view (proven both directions). Review round 1: the AC3 substring scan was circumventable by 'from ...Model_Artifacts import acquisition' -- replaced with an AST walk, applied to model_curated_view.py too (same gap), sabotage-verified against exactly that bypass. Review round 2 (Qodo + prior reviewer converging): the consent-window race we had annotated-only is now closed by construction -- _install_in_progress() guards on _model_install_kind, which spans request->consent->provision; 6-case phase x flow test asserts refusal + is-identical state survival.
<!-- SECTION:NOTES:END -->
