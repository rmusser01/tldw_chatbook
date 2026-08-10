---
id: TASK-198
title: Prompt collections
status: Done
assignee:
  - '@codex'
created_date: '2026-07-12 13:16'
updated_date: '2026-08-09 23:46'
labels:
  - ux
  - library
  - prompts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the 2026-07-12 Library Prompts spec: the server prompt service already models collections; local has a parallel seam. Surface collections (group/browse/assign) in Library Prompts once core CRUD ships.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library Prompt list uses an exact local `PromptScopeService` browse contract for search, collection filter, whitelisted sort, and bounded stable pagination; displayed totals/pages remain truthful beyond 100 active Prompts and Recipes, and existing Console/picker search behavior is unchanged.
- [x] #2 Users can create and rename named local Prompt collections and browse the complete catalog through bounded pages, collection search, and explicit Load more; case-fold collisions are rejected transactionally, pre-existing colliding names remain distinguishable by ID, and this task does not add collection deletion.
- [x] #3 A Prompt or Recipe can belong to multiple collections; the editor shows current memberships and applies one atomic membership update independently from Prompt content Save, with no partial change when any Prompt or collection identity is inactive, foreign, or invalid.
- [x] #4 Local and server collection operations remain routed through `PromptScopeService`, while the Library collection UI is explicitly local-only and does not expose a source selector or imply mixed/server results.
- [x] #5 Immutable browse state and request fingerprints distinguish loading, empty library, empty collection, no matches, and service failure with Retry; debounced workers reject late success/error results after scope changes and restore focus after page/result replacement.
- [x] #6 Collection, browse, paging, and membership controls have stable keyboard order, remain usable in narrow terminals and large catalogs, render markup-looking names literally, and report content Save and membership Apply outcomes separately.
- [x] #7 Automated database, service, state, and Textual UI tests cover exact counts/paging, sort injection rejection, case-fold races, multi-membership rollback, stale workers, truthful empty/error states, and responsive geometry; the user guide documents the final local-only workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.

ADR path: N/A.

Reason: TASK-198 surfaces the existing local collection ownership and PromptScopeService boundary without changing schema, sync policy, or durable ownership.

1. Define immutable exact-browse scope/result state and service contracts with RED tests.
2. Add whitelisted SQLite browse/count paging with deterministic ordering and injection-safe sort selection.
3. Complete the local collection catalog, case-fold validation, and atomic multi-membership services while preserving existing server routes.
4. Replace sampled Library Prompt filtering with request-tokened local browse workers and truthful loading/empty/error states.
5. Add collection management and membership controls with separate Prompt Save and membership Apply outcomes.
6. Update documentation, run focused/full verification and visual QA, request independent review, and close the task before PR merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact local Prompt browsing and collection management end to end. PromptScopeService now owns bounded local DB/service routing; immutable request fingerprints reject stale browse, catalog, and membership settlements. The shared local-only manager supports searched/paged catalogs, create/rename, literal collision labels, and one-or-many staged memberships; membership Apply remains independent from Prompt Save.

Updated the Prompts DB/service/state/controller/modal/canvas/screen paths and their focused database, service, reducer, and mounted Textual coverage. Updated Docs/User_Guide/library/prompts.md with the exact list, collection-manager, membership, Retry, empty/error, paging, and keyboard workflows. No collection deletion, source selector, server collection UI, schema change, or shared TCSS/bundle change was added; the manager's responsive styles remain widget-scoped.

Post-rebase verification: focused DB/service/state/Task 6 suite 383 passed; full Prompt canvas 205 passed; Prompt shell targets 2 passed. Ruff lint, changed-range formatting, py_compile, generated CSS source/bundle sync, git diff --check, and the final Impeccable detector passed. Real Textual compositor QA produced 16 ignored captures at 64x24, 80x24, 100x30, and 140x40 under .superpowers/sdd/2026-08-02-task-198-prompt-collections/visual-closeout; representative rasters showed literal markup, reachable actions, one manager scroll owner, and no clipping/contrast defect. The existing editor guide image remains more useful than adding a state-specific manager screenshot.

ADR required: no; ADR path: N/A. This implements the approved local ownership and PromptScopeService boundary without a new durable architecture decision. Lessons: no new entry; the only observed flake is the state-before-DOM Textual timing class already recorded in lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
