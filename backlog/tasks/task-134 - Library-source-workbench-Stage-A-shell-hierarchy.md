---
id: TASK-134
title: Library source workbench Stage A shell hierarchy
status: Done
labels:
- library
- ux
- source-workbench
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adapt the Library content hub into the accepted Source Workbench shell hierarchy so users can distinguish Source Map, Workspace Context, Active Workbench, Quick Actions, and Inspector responsibilities while keeping unsupported Collections item workflows visibly disabled until later stages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library mode column titles and visible section labels match the accepted Source Workbench Stage A hierarchy without adding new service calls.
- [x] #2 Collections copy describes stored collection content review/read workflows rather than reusable source groups or workspace folders.
- [x] #3 Unsupported collection-scoped actions remain disabled with visible reasons and recovery copy, including Search/RAG, Study, Console handoff, and server sync promotion.
- [x] #4 Mounted regressions cover Source Map hierarchy, Workspace Context, Quick Actions, selected/empty Collections copy, and disabled capability reasons.
- [x] #5 QA evidence includes a rendered CDP/Textual-web screenshot of the updated Library screen and a manual note that no tldw_server runtime dependency was introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Stage A changes Library shell hierarchy, labels, disabled-state copy, mounted tests, and QA evidence only. It does not change storage/schema, sync/conflict policy, data ownership, provider/runtime boundaries, service contracts, security policy, or server integration.

1. Add failing mounted regressions in Tests/UI/test_library_content_hub.py for Source Map, Workspace Context, Quick Actions, Active Workbench, Collections reader copy, and disabled capability reasons.
2. Update tldw_chatbook/UI/Screens/library_screen.py column titles and left-rail hierarchy while preserving route IDs, mode chip IDs, existing services, and disabled action behavior.
3. Update tldw_chatbook/Library/library_collections_state.py and Collections inspector/action copy so Collections is framed as stored content reading/review with local item-reader limitations, not reusable source grouping.
4. Add minimal source TCSS support only if the Python hierarchy does not render cleanly; regenerate tldw_chatbook/css/tldw_cli_modular.tcss via python tldw_chatbook/css/build_css.py only after source TCSS changes.
5. Run focused Library/design-system regressions and git diff --check.
6. Capture actual textual-web/CDP screenshots for Hub and Collections modes, create QA evidence, update the product maturity roadmap, complete task hygiene, then commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Stage A Source Workbench shell closeout as a narrow hierarchy/copy slice. The Library left rail now keeps Workspace Context and Source Map visible while adding a visible Quick Actions section, and Collections action copy now explains that item-level reading/review depends on a future local item adapter while Search/RAG, Study, Console handoff, and server sync promotion remain disabled.

Modified files:

- `tldw_chatbook/UI/Screens/library_screen.py`
- `Tests/UI/test_library_content_hub.py`
- `Docs/superpowers/qa/library-source-workbench-stage-a/2026-06-24-library-source-workbench-stage-a.md`
- `Docs/superpowers/qa/library-source-workbench-stage-a/library-stage-a-source-workbench-cdp-2026-06-24.png`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- `backlog/tasks/task-134 - Library-source-workbench-Stage-A-shell-hierarchy.md`

Verification:

- `python -m pytest -q Tests/UI/test_library_content_hub.py --tb=short` -> `18 passed, 1 warning`
- `python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short` -> `11 passed, 1 warning`
- `git diff --check` -> passed with no output

QA evidence: rendered Textual-web/CDP screenshot approved by the user at `Docs/superpowers/qa/library-source-workbench-stage-a/library-stage-a-source-workbench-cdp-2026-06-24.png`.

Residual risks: collection item reader, local item capability flags, collection-scoped Search/RAG, collection-scoped Console handoff, and server sync promotion remain future stages. No tldw_server runtime dependency or API call was introduced.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Library Source Workbench Stage A hierarchy is verified and documented. Collections remains locally manageable/readiness-focused without implying unavailable item services are wired.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
