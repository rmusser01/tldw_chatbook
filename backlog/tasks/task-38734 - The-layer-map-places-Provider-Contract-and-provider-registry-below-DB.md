---
id: TASK-38734
title: "The layer map places Provider_Contract and provider_registry below DB"
status: To Do
created_date: 2026-11-04 02:55
labels:
- follow-up
- architecture-review-2026-10-25
- architecture
priority: low
references:
- Docs/Architecture/layer-map.md
- Tests/Architecture/test_layer_direction.py
- Tests/Architecture/layer_direction_allowlist.tsv
dependencies:
- TASK-36944.1
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36944.1 created tldw_chatbook/Provider_Contract/, a pure package: it imports only the stdlib and the root module provider_registry.py, and an outbound-import pin enforces that. The layer map does not list either one, so both default to the domain layer (rank 2). DB modules import provider types from it, which leaves three upward-edge rows in layer_direction_allowlist.tsv: DB.AgentRuns_DB to Provider_Contract.sampling_params, and DB.ChaChaNotes_DB to Provider_Contract.provider_continuation and Provider_Contract.thinking_blocks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Docs/Architecture/layer-map.md and the layer test classify Provider_Contract and provider_registry.py at a rank no higher than DB, with the reason stated in the map
- [ ] #2 The three DB to Provider_Contract allowlist rows are deleted, and test_allowlist_has_no_stale_rows and test_no_new_upward_import pass
- [ ] #3 Neither module gains an import from a higher layer; the Provider_Contract outbound-import pin still passes
<!-- AC:END -->
