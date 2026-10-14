---
id: TASK-34972
title: >-
  Dreams stories cannot be opened: no route in the app reaches the screen that
  lists them
status: In Progress
assignee: []
created_date: '2026-10-12 02:33'
updated_date: '2026-10-13 22:02'
labels:
  - dreams
  - navigation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams writes a daily collection of stories and the user guide says they appear on the Artifacts screen, where a story opens for feedback, Keep, Track, Watch, Ingest, Export, Dive and the goals editor. Since the Artifacts destination was folded into Library (ADR-172, September 2026) the artifacts route resolves to the Library screen, the old Artifacts screen is never mounted anywhere in the product, and the Library artifact catalog knows only chatbooks and reports. So a user who enables Dreams gets cycles that spend the daily search and model budgets and stories that cannot be listed or opened, and every action that starts from a story is unreachable. The test suite did not catch it because the UI test harness maps the artifacts destination to the old screen itself. Found by reading and by executing the route lookup during the review of PR 2921; not yet confirmed in a running app. It is already true on dev for the Dreams phases that have shipped.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Dreams enabled and a collection present, the user can reach the story list from the app's own navigation and open a story
- [ ] #2 Every action the user guide lists for a story is reachable from there, including the goals editor
- [ ] #3 The Dreams UI tests mount the same screen the app's navigation mounts, so an unreachable surface fails a test
- [ ] #4 The user guide and the Settings hints name the real location and key
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use actual production Library routing/catalog/controller to expose Dreams stories, all existing actions and goals editor; update tests to mount actual destination and guide/Settings real location/key. Preserve shell authority/layout/token/owner contracts and current PC/Goal. Direct ADR172/196 integration, no new ADR.

Source update admits34815 persisted degradation reasons alongside the real Library-owned story/list/goals navigation; compose existing token-backed views, preserve all original actions and avoid a second artifact shell.

R3 PR3020 required UI4 qualification on clean752: ADR required: no; ADR path: backlog/decisions/172-library-artifacts-browse-and-navigation.md and backlog/decisions/196-dreams-daily-discovery-and-tracking.md. Investigate missing Dreams story modal after real production Library click under original five-second budget. Record actual click result, button mounted/visible/enabled/geometry, current controller detail and fresh action read, worker lifecycle, running host/modal stack/compose/error chronology; compare accepted84861 owners. Reproduce controlled causal failure before minimum owner or fixture fix. Preserve all literal-markup/security/story-actions/goals/held-compose/native cleanup and font/R1/R2/census/caps. Qualify original journey and connected readiness/ownership/static on frozen source, new R3 report/seal and normal commit for independent scoped review. No new task ID/Done or unrelated Prompts sweep; root owns final actualbase/publication/CI/queue.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Source author follow-up admitted into the user-authorized complete PR2921 review/remediation. Historical source description/notes retained. Current integration plan targets frozen updated author6c074d6eed/v6; actual post-Buddy dev and fresh author lease remain required before implementation. No product change or completion claimed.

Current source update: author0c941ba2ec984d960ba83376aeace0e830452bbc, 113 nonmerge commits/88 feature paths after actual merged Goal9595facbbb. Independent source review preserves the real shared Toast fix and stronger reconcile/suspension work; remaining integrated proof and actual post-Buddy/latest-dev publication gate are pending. Prior source/plan/qualification claims remain historical. No product change or completion is claimed here.

PR2921 current private integration qualification (2026-10-12):
Dreams now uses actual Library catalog/controller/reader and Story actions; every existing action and goals editor remains reachable. All 16 former source route scenarios now mount production Library, with native focus/late-generation/selection ownership. Final modal actions 38 pass; Library routes/catalog/layout/ownership/import 29 pass. Missing-origin tracking retains mechanism/query/last-check Details. Guide and F4 hints identify the actual destination. ADR-172/196 amendments apply.
Full source-bound outcomes, original failures, native owners and tradeoffs: .superpowers/sdd/2026-10-11-pr2921-current-dev-integration/task-1-report.md; external pr-review-wave2/pr2921-current-integration. Local implementation is ready for independent review. Actual-base restack, complete independent review and hosted required gates remain pending; preserve In Progress and the existing acceptance checklist until the full DoD is met.

Hosted publication Task2 plan (2026-10-12): Correct only the real Library test story/Goals readiness fences to require expected modal, mounted state and expected control within the existing five-second loop. Preserve every original action/assertion, marker, private child, fixture and deadline. Add a bounded held real-compose witness for both modal types, with a failing type-only control then corrected success. Qualify only amended nodes, UI census membership and static checks; production remains unchanged. ADR required: no, routine synchronization under the existing Library/Textual mount contract. Earlier original-node local pass remains non-reproduction; hosted UI4 failure is retained. No Done; scoped Task2 review/new-head hosted gates remain pending.

Hosted publication Task2 results (2026-10-12): Corrected only the Library test story/Goals waits to require expected modal, mounted state and expected control within the existing five-second/0.03-second loop. All 9 original assertions, actions, markers and fixture parameters retained; production unchanged. Two controlled real held-compose witnesses fail the original type-only predicate and pass the correction; the corrected original Library journey also passes (3 cases total). Initial RED witness cleanup produced secondary NoMatches and is preserved separately; corrected cleanup awaits actual Textual mount, yielding clean 2-case early-return RED. Ruff/format/full actual-base whitespace and unchanged required UI census188 checks pass. Source-bound report/handoff: external pr-review-wave2/pr2921-hosted-readiness-task2-report.md and -handoff.json. Prior local pass remains non-reproduction, not confirmed hosted flake. Scoped Task2 review, separate root-owned Fast CI failure investigation and new-head hosted qualification/publication remain pending; preserve In Progress/AC.

R3 PR3020 UI4 qualification: actual held Textual batch reproduces a mounted/displayed/enabled Dreams action button with zero geometry. Pilot caches that origin before its own pause; actual layout release later cannot change cached coordinates, real click returns False and no Dreams action/compose starts. Original local pass remains non-reproduction; relevant owners/fixture equal accepted84861. Paired old-helper RED and corrected production journey GREEN preserve original five-second budgets. Existing loaded helper now observes requested actual mouse target readiness; both story/goals single real clicks assert True. All literal notice/title/security/actions/goals/composition/native cleanup remain, no product/cap/authority/timeout change or click retry. Existing ADR172/196 apply, no new ADR/ID. Connected and exact committed controls/new R3 seal/independent review and normal hosted gates remain root-owned; status and acceptance open.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Renumbering provenance

Renumbered from task-34804 to TASK-34972 (2026-10-13): this ID collided
with a concurrently filed task after the branches merged; per the
2026-08-21 owner rule (TASK-19601) the older arrival kept the id.
