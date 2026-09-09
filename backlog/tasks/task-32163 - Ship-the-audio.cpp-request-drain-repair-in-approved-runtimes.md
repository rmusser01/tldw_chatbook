---
id: TASK-32163
title: Ship the audio.cpp request-drain repair in approved runtimes
status: To Do
assignee: []
created_date: '2026-09-09 06:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The active-request shutdown repair is qualified in isolated patched macOS binaries. Normal Chatbook provisioning still uses the approved upstream baseline; distributing the fix requires an accepted upstream revision or an explicitly reviewed runtime recipe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reviewed upstream revision or owned build recipe includes the retained-request shutdown repair, with matching source and binary provenance.
- [ ] #2 Approved supported runtime manifests and provisioning install that repair without silently substituting an unqualified model/runtime tuple.
- [ ] #3 Installed-runtime CPU and Metal active-inference shutdown and successor playback pass, with remaining Windows and Linux device qualification linked.
<!-- AC:END -->
