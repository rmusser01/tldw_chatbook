---
id: TASK-20980
title: 'Folder-filtered library_list_notes (conditional candidate — only if search-based re-runs trip on false positives)'
status: To Do
assignee: []
created_date: '2026-08-23'
updated_date: '2026-08-23'
labels:
  - library
  - agent-tools
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Third-review finding behind the student-workflow spec's re-run disambiguation
ruling (§5): the cross-session re-run convention for `library_save_note` is
**search-based** — `library_search_notes(query=<note title>)`, agent
disambiguates by reading — because `library_list_notes` has **no folder
filter** (its schema is limit/offset only) and its payloads carry **no folder
info**, so "which notes are in this study folder" is not expressible through
the list tool. Search-by-title can false-positive: any note whose content
contains the title string matches, and note search is substring + FTS over
content, not title-exact.

This task is **conditionally valuable, not unconditional**: only worth taking
if agents actually trip over those false positives in practice (mis-picked
match, duplicate minted anyway) or the maintainer rules the risk live. The
fix when warranted is a `folder` parameter on `library_list_notes` — folders
are already the save tool's grouping affordance, ensured in the notes UI's
local scope — plus the folder info the disambiguation needs in note briefs.
If the evidence never arrives, the search-based convention stands and this
task closes as "not needed" with that recorded.

Source spec: `Docs/superpowers/specs/2026-08-23-student-workflow-design.md` §5
(re-run disambiguation ruling) and §4.4 (the accepted duplicate window).
Discovery record: `.superpowers/sdd/2026-08-23-student-workflow/task-2-report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] An evidence gate is recorded either way: documented instances of agents mis-disambiguating a title search (wrong note updated, or a duplicate minted despite the convention), or a maintainer ruling that the risk is live — without one, the task closes as not-needed with the search-based convention reaffirmed
- [ ] If taken: `library_list_notes` accepts a `folder` filter (one-level name, matching the save tool's affordance) and returns only notes filed there, with contract + service tests covering the empty-folder, unknown-folder, and unfiled-notes cases
- [ ] If taken: the re-run documentation (Docs/Development/Agent-Tools/local-library-tools.md, the save-note section) is updated from the search-based convention to the folder-filtered one, stating why it changed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
(To be added when the task is picked up.)
<!-- SECTION:PLAN:END -->
