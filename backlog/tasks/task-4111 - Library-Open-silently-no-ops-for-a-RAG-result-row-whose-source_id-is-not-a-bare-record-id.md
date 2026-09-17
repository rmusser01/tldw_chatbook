---
id: TASK-4111
title: >-
  Library Open silently no-ops for a RAG result row whose source_id is not a
  bare record id
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 20:23'
updated_date: '2026-09-17 04:09'
labels:
  - library
  - rag
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found reviewing plan Tasks 4 and 5 of the hybrid-fusion cluster; PRE-EXISTING, not introduced by that work. The Library RAG panel's Open action routes through _open_library_item_by_id (UI/Screens/library_screen.py), which hands the row's source_id straight to the per-type detail route. Every one of those routes assumes a bare record id: the prompt branch does int(record_id) inside a try/except (TypeError, ValueError) that simply returns, and the media branch passes the value to the media detail fetch whose own broad except Exception logs and sets detail to None. A row whose source_id carries the retrieval layer's prefixed document id (media_15 rather than 15) therefore produces no navigation, no error, and no message - the user presses Open and the screen does not change. The engine's keyword leg used to emit exactly that shape, because SearchResult.id is a document id and the source_id key the row mappers read was never populated; TASK-3996 fixed that one instance by stamping a bare source_id on keyword rows, and the eval harness's canonicalizer compensates for the same mismatch on its own side. The general fragility is untouched: any future row builder, any fallback path that stamps a composite id (_fusion_doc_key still falls back to the row id when metadata is missing), and any id space that is not an integer reaches the same silent dead end. The remedy is for the open route to resolve or reject explicitly rather than swallow - normalise a prefixed id, or tell the user the row cannot be opened - so the failure is never invisible.
<!-- SECTION:DESCRIPTION:END -->

## Additional audit evidence — 2026-09-16

TASK-2530's targeted Search/RAG run reproduces
`test_library_search_rag_o_on_focused_card_opens_like_button`: a fixture result
with `source_id="media-1"` never reaches the expected Media viewer. The same test
fails with the unchanged `d6b8ea2566` controller, so it is not caused by the
Run/disclosure fix. Investigate this fixture/route contract alongside the prefixed
ID path; this reproduction alone does not establish that a real stored record
with a canonical ID fails. Evidence:
`Docs/superpowers/qa/2026-09-16-rag-query-gate/baseline-failures.txt`.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening a RAG result row whose source_id carries a source-type prefix lands on the item's detail surface
- [x] #2 A row that genuinely cannot be resolved reports why instead of doing nothing
- [x] #3 No open path swallows a parse failure into a silent return or a None detail
- [x] #4 A regression test covers a prefixed source_id and an unresolvable id, for media and for prompts
- [x] #5 Recognized local media/prompt IDs preserve source type and authority; malformed, mismatched, or server identities cannot silently open an unrelated local record.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/003-settings-library-rag-defaults.md; backlog/decisions/084-library-media-reader-ia.md; backlog/decisions/150-design-token-system-and-design-language.md.
Reason: repair the existing local Search/RAG source-open boundary without changing reader ownership, backend authority, persistence, or route contracts.
1. Reproduce recognized prefixed media/prompt IDs and invalid IDs through production-CSS result Open and focused-card keyboard actions. Distinguish the old canonical-selection test pin from real missing detail.
2. Normalize only recognized matching local prefixes before invoking existing routes. Report invalid or nonlocal identities without changing the result list or routing to a guessed record. Keep existing missing-record errors, dirty-save vetoes, and generation fences.
3. Verify targeted model, result-opening, retained-reader, dirty-edit, and token tests. Exercise real private-profile Media/Prompt reads and invalid-ID feedback in native dark/light wide/compact views, with no model request.
4. Obtain independent review, record evidence and limitations, update audit/task documentation, and commit locally.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Search/RAG Open now resolves recognized matching local Media/Prompt ID wrappers before entering existing reader routes. Invalid, mismatched, server and display-sanitized identities report a warning while preserving Search and focus. Citation IDs, shared deep-link behavior, dirty-save vetoes and reader generation fences are unchanged.

Changed the result model/controller, added 66 identity and 24 mounted UI regressions, and repaired two legacy Open test contracts. Independent review found a display-sanitization collision; 13 red regressions reproduced it, the refusal guard fixes it, and follow-up review found no remaining issues. Recorded the lesson in lessons-testing-evidence.md.

466 targeted tests passed; TASK-15390 heading test deselected. New files pass lint/format, changed production ranges pass format, and existing files add no lint diagnostics. Four native dark/light 170x48/80x24 cells perform real local keyword retrieval with an explicit ID-shape adapter: eight exact stored-record opens and eight visible refusals pass. Sixteen captures inspected, records/default profile hashes unchanged, ten private databases healthy, clean exit verified. Missing Prompt records retain the existing generic load-error/Retry surface. No provider call, full suite, push or merge.

ADR check: no new ADR; existing backlog/decisions/003-settings-library-rag-defaults.md, 084-library-media-reader-ia.md and 150-design-token-system-and-design-language.md apply. Evidence and limits: Docs/superpowers/qa/2026-09-17-rag-source-open/README.md. Continuation ledger updated. Two unrelated fixture failures reproduce with unchanged production modules and remain separately documented.
<!-- SECTION:NOTES:END -->
