---
id: TASK-41533
title: "Local and CI disagree on the first-send warm-up module census since PR #3446"
status: To Do
assignee: []
created_date: 2026-11-17 05:39
labels:
- follow-up
- tests
- performance
priority: medium
references:
- Tests/Performance/test_screen_preimport_payload_budget.py
- tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py
dependencies:
- TASK-33620.15.53
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reported by the coordinator at the filing review (round 15). Since PR #3446 (merge 3917e2d0cb, TASK-33620.15.53) the warm-up test in `Tests/Performance/test_screen_preimport_payload_budget.py` fails locally and deterministically: 95 modules against a limit of 94. The new module is `tldw_chatbook.RAG_Search.scope_evidence`, imported lazily by `Event_Handlers/Chat_Events/chat_rag_events.py`. It passes at cc6006fcf5 and every merge before #3446. CI's boot budget step passes on dev, so the two disagree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The reason local and CI differ is named (environment, import order, snapshot, or step coverage)
- [ ] #2 The snapshot and the census agree in both local and CI runs, on 3 of 3 runs each
- [ ] #3 If the lazy import is intended to stay, the pinned budget or allowlist records `scope_evidence` with its reason
<!-- AC:END -->
