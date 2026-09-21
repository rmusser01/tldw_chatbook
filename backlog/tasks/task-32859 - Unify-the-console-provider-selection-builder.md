---
id: TASK-32859
title: Unify the console provider-selection builder and resolver helpers
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies: []
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
  - backlog/decisions/006-provider-aware-generation-settings.md
  - backlog/decisions/146-console-custom-endpoint-registry.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console's provider/model selection algorithm is written twice, near-verbatim: `Chat/console_chat_controller.py:593-720` `build_console_provider_selection_from_settings` (~128 LOC) and the 4-deep chain in `UI/Screens/chat_screen.py:9618→9635→9670→9691` (~140 LOC). The drift hazard is demonstrated, not hypothetical: the identical "PR-2668 CE-001" fix (dashed custom-endpoint slugs vs underscore config keys) had to be applied at both copies (`console_chat_controller.py:609-614`, `chat_screen.py:9700-9704`).

Around the twins: the `model/api_model/default_model` fallback chain spelled ×5 (gateway :4068, controller :621, session settings :1271, chat_screen :7997 and :9716); same-named `_provider_settings` ×3 with divergent behavior — only `console_session_settings.py:1996` is ADR-146 registry-aware; the gateway copy (:7171) raises instead of swallowing; the 20-field `ConsoleProviderSelection` spelled ×7. The screen's copy is a superset (endpoint policy, workspace, identity) — the superset must survive the merge. ADR required: no — consolidation is ADR-006-compliant ("Console owns effective session resolution"); the unified `_provider_settings` must be the registry-aware variant per ADR-146, with the gateway's raising policy preserved at its call sites or explicitly changed.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exactly one implementation of the selection algorithm exists (the superset survives); a grep for the algorithm's distinctive steps finds one copy
- [ ] #2 The unified `_provider_settings` is registry-aware (ADR-146); the gateway's error policy is preserved or the change is an explicit recorded decision
- [ ] #3 The model-fallback chain and the selection-construction are centralized; the ×5/×7 spellings are gone
- [ ] #4 ADR-006 precedence order unchanged and pinned by the existing console settings tests
- [ ] #5 The PR-2668 fix exists in exactly one place
<!-- AC:END -->
