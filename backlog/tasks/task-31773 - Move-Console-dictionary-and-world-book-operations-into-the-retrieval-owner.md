---
id: TASK-31773
title: Move Console dictionary and world-book operations into the retrieval owner
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:38'
updated_date: '2026-09-05 19:56'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Console dictionary and world-book retrieval operations with their existing retrieval owner while preserving picker, send, and screen lifecycle behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dictionary and world-book send appliers and picker outcomes preserve current behavior, including error containment and guard release.
- [x] #2 Screen event admission and Textual worker scheduling remain on ChatScreen, and all moved dependencies are explicit and late-bound.
- [x] #3 Whole affected dictionary, world-book, retrieval, and native integration tests pass without increasing architecture ceilings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; implements existing DESIGN.md section 7 and Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md controller ownership. 1. Census production and test callers, constructor dependencies, and baseline bodies. 2. Add ownership characterization and verify RED; run affected baseline tests. 3. Move the two send appliers and four picker workers into ConsoleRetrievalController with named completion callbacks; preserve screen event scheduling. 4. Verify normalized AST body equivalence, complete affected test files, lint, and exact size counts. 5. Obtain root diff review and commit scoped files only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification so far: six moved method ASTs equal HEAD after only current-conversation, same-owner summary refresh, and dialog-completion rewrites. New ownership/guard tests were RED (10 expected failures); complete affected files plus bare-shell Architecture guard now184/184 pass109.41s (XML /private/tmp/console-retrieval-31738-final.xml). Hook construction remains deferred through two explicit synchronous lambdas; regression proves no owner read at hook construction and resolves both first/replacement owners with identical positional arguments and history identity. Pre-move baseline25pass2fail identified separate dictionary fixture TASK31740. Root reviewed production move and fixtures with no actionable findings; native349 run still pending. No screen cap or shared architecture inventory changed.

Complete native-flow file349passed624.28s (XML /private/tmp/console-native-flow-31738.xml). That run imported the extraction before the final hook-lambda repair; the final184run and explicit first/replacement-owner regression verify that subsequent delta. Ruff lint all affected files passes; changed-region formatting passes (pre-existing unrelated formatting differences left alone). Final screen17256lines/571methods versus17541/577 before this move:285line/6method net reduction, including eight lines to retain deferred hook binding. Root reviewed and approved scoped commits conditional on these green results. Remaining absolute screen deficit383lines/12methods and upstream Environment wildcard-worker guard are separate work; no claim of overall Architecture green.

Final range-format check wrapped the two long hook lambdas without changing their AST; corrected final screen census17260lines/571methods, net281lines/6methods, remaining387lines/12methods. Shared caps remain untouched.
<!-- SECTION:NOTES:END -->
