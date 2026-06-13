---
id: TASK-89.2
title: Make Library default route a content hub landing page
status: Done
dependencies:
- TASK-89.1
labels:
- library
- content-hub
- ux
priority: high
parent_task_id: TASK-89
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the default Library route into a landing page and center hub for ingested content. The page should summarize real Notes, Media, and Conversations inventory, route users to the owning modules for deeper work, and keep Console/RAG handoff secondary and policy-gated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The default Library mode presents a content hub landing page, not a selected-source staging view.
- [x] #2 Hub cards summarize Notes, Media, Conversations, Search/RAG, Import/Export, Collections, and Study using real local counts and recent sample titles where available.
- [x] #3 The inspector explains module ownership, workspace visibility versus staging eligibility, and why Console handoff is secondary.
- [x] #4 Empty/loading/error states remain stable and teach how to add or find content without showing misleading selected-source actions.
- [x] #5 Existing route/action IDs remain keyboard reachable for owner module navigation and downstream handoffs.
- [x] #6 Focused regressions cover hub cards, real-count/recent content rendering, empty copy, owner-route actions, and inspector ownership guidance.
- [x] #7 Actual CDP/Textual-web screenshot QA is approved before PR creation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This slice changes Library UI presentation and mounted tests only. It does not change storage/schema, sync policy, provider/runtime boundaries, security posture, or service ownership.

1. Add focused failing regressions for Library content-hub cards, empty-state guidance, owner-route actions, and ownership inspector copy.
2. Reuse existing local source snapshot services to build hub summaries for Notes, Media, and Conversations.
3. Replace default selected-source detail/inspector copy with content-hub rows and module ownership guidance.
4. Preserve existing action IDs and workspace-gated Console handoff behavior.
5. Update older Library contract tests from source-workbench wording to content-hub wording.
6. Run focused Library tests and diff hygiene, then capture actual Textual-web screenshots for approval.
<!-- SECTION:PLAN:END -->

## Non-Goals

<!-- SECTION:NON_GOALS:BEGIN -->
- Do not add full source editing in this slice.
- Do not add selectable per-source detail rows as the default Library landing-page behavior.
- Do not implement full Search/RAG retrieval or downstream citation persistence here.
<!-- SECTION:NON_GOALS:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reworked the default Library route into a content hub landing page that summarizes Notes, Media, Conversations, Search/RAG, Import/Export, Collections, and Study instead of defaulting to selected-source staging.
- Preserved existing Library route/action IDs while clarifying that owner modules handle deeper editing and that Console/RAG handoff is secondary and workspace-gated.
- Added content-hub empty-state and inspector copy for module ownership, global browse/search visibility, current-workspace staging eligibility, and recovery guidance.
- Updated Library layout/contract regressions and added `Tests/UI/test_library_content_hub.py`, including a visual separation regression for taller, spaced left-column module actions.
- Regenerated `tldw_chatbook/css/tldw_cli_modular.tcss` from source TCSS after changing Library action sizing tokens and spacer styling.
- Approved CDP screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/library/content-hub-buttons-spaced-cdp-2026-06-09.png`.
- Verification: `python -m pytest -q Tests/UI/test_library_content_hub.py Tests/UI/test_destination_shells.py -k library Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_post_release_workspaces_library_depth.py --tb=short` passed with 30 passed and 92 deselected.
- Verification: `python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py -k library --tb=short` passed with 17 passed and 72 deselected.
- Verification: `python tldw_chatbook/css/build_css.py` completed; it reported the existing optional missing `features/_evaluation_v2.tcss` warning.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Default Library now behaves as an ingested-content hub with owner-module navigation, workspace-aware handoff guidance, focused regressions, and approved CDP screenshot evidence.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
