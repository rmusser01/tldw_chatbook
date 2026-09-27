---
id: TASK-33086
title: Hoist shared hosted-chat helpers out of provider adapters
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [refactor, providers]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
hosted_chat.py is the intended shared framework for hosted OpenAI-shaped providers, yet moonshot.py, zai.py, and qwencloud.py each privately re-implement the same helper inventory (_resolve_api_key, the _normalize_* family, _json_shape_is_bounded, retry policy, _strict_json_loads). The copies are already drifting in behavior between adapters. Hoisting the helpers into hosted_chat leaves each adapter with only its payload builders and finish policies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shared normalize, resolve, and retry helpers live in hosted_chat.py and adapters keep only payload builders and finish policies.
- [ ] #2 Each hoisted helper exists exactly once repo-wide.
- [ ] #3 Targeted adapter tests pass with no behavioral change.
<!-- AC:END -->
