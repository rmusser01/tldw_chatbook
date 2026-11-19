---
id: TASK-41512
title: "The UI-ready module census overruns at base about two runs in three"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- flaky
priority: medium
references:
- Tests/UI
dependencies:
- TASK-41141
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41141's review (r15ti-rev2, round 15) and settled: base fails 2 of 3 and the branch fails 2 of 3, at 1041 against a limit of 1040 modules. The extra modules are `DB.character_conversation_search`, reached via `character_context._load_recent_sync` on an executor thread (timing-dependent), and `Provider_Contract.provider_setup_facts`, reached via `endpoint_refusal` then `_route_owning_presets` (a lazy import; TASK-40300).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census gives the same module count on 10 of 10 runs, by removing the timing-dependent import or by pinning it in the allowlist with its reason
- [ ] #2 The cause of each extra module is named in the notes
<!-- AC:END -->
