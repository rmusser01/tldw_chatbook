---
id: TASK-13212
title: Complete audio.cpp community recipes and release accounting
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
labels:
  - tts
  - audio-cpp
  - recipes
  - compatibility
  - release
dependencies:
  - TASK-13206
  - TASK-13207
  - TASK-13208
  - TASK-13211
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
  - backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add exact release-0.5.1 recipes for glm_tts, inflect_v2, outetts, and vietneu_tts and close the 21-family 67-package accounting matrix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The four declared release-0.5.1 community package entries—glm_tts, inflect_v2, outetts, and vietneu_tts—have exact immutable recipes, typed tts/clone capability where upstream declares it, adversarial recognition tests, and tuple-scoped real-process evidence.
- [ ] #2 The canonical inventory accounts exactly once for all 21 pinned families and all 67 declared package entries, with every entry Supported, Unsupported with reviewed reason, or an explicit release blocker and no unknown, duplicated, or unapproved gap.
- [ ] #3 Every Supported entry has exact package signals, safe generated model projection, catalog/task cross-check, voice/reference rules, backend/platform compatibility state, recipe revision, and withdrawal/upgrade behavior tied to the pinned audio.cpp release.
- [ ] #4 Public Settings, Model Library, Speech Lab, documentation, and diagnostics derive support claims from the same accounting data and name the exact evidenced family/package/OS/backend subset rather than claiming blanket support.
- [ ] #5 Provisioned compatibility gates cover every Verified tuple, at minimum CPU on macOS, Linux, and Windows, while accelerated tuples require their own evidence and normal CI remains hermetic with no binary/model/network/audio-hardware dependency.
- [ ] #6 Clean-profile release UAT records exact sanitized app/server/recipe/artifact/runtime/WAV/shutdown evidence and human audible confirmation for local selection, Model Library, text, clone/profile/roleplay, multi-model lazy use, staged restart, failure recovery, source regressions, portability, and keyboard/narrow-layout journeys.
- [ ] #7 Existing External and user-provided server.json users retain exact behavior and ownership, generated/clone failures leave no child, handle, client, task, artifact, or materialization, and complete-WAV async adapter behavior remains unchanged.
- [ ] #8 Release documentation declares the guided-model workstream complete only after all blockers are resolved and the 21-family/67-package matrix, platform evidence, recipe revisions, rollback behavior, and sanitized UAT are traceable to the exact commit.
<!-- AC:END -->
