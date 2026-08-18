---
id: TASK-17964
title: >-
  Project-skills follow-ups: offer-modal footer tests, checkbox-escape
  assertion, off-thread discovery in _add_folder
status: To Do
assignee: []
created_date: '2026-08-18 00:00'
labels:
  - skills
  - workspaces
  - testing
priority: low
dependencies:
  - TASK-17651
---

## Description (the why)

TASK-17651's close-out review and live verification found a handful of
small, non-blocking gaps left by the project-skills import feature (Tasks
1-5 of `Docs/superpowers/plans/2026-08-17-project-skills-import.md`). None of
these affect the shipped behavior's correctness for the common paths already
covered by `Tests/Skills/test_project_skills_import_modal.py` and
`Tests/Workspaces/test_workspace_create_modal.py`, but they are real,
identified gaps worth closing.

## Acceptance Criteria (the what)

- [ ] `ProjectSkillsImportModal`'s offer-phase footer (`_offer_footer_lines`
      in `tldw_chatbook/Widgets/project_skills_import_modal.py`) has test
      coverage for both its branches: a discovery with `skipped` entries
      renders `Skipped: <names>`, and a discovery with `truncated > 0`
      renders `N more not shown` — neither branch is exercised by any
      existing test today
- [ ] A test asserts Checkbox-label escaping in the offer phase: a
      malicious/markup-hostile skill `description` (e.g. containing
      `[bold]`/`[/bold]` or other Rich markup) renders as literal text in
      the Checkbox label, not interpreted markup — the discovery-layer
      fixtures already used elsewhere in `Tests/Skills/` for
      markup-hostile names/descriptions can be reused; today only the
      `escape_markup()` call site in `_compose_offer_phase` defends this,
      with no assertion pinning it
- [ ] Fix the "1 project skill(s)" pluralization in
      `tldw_chatbook/Widgets/workspace_create_modal.py` (`compose()`'s
      folder-row label and `_add_folder`'s discovery-count usage) so a
      single discovered skill reads "1 project skill" and multiple read "N
      project skills" (live-verified as-is during TASK-17651's Step 4:
      `... — contains 1 project skill(s)`, grammatically wrong for the
      singular case)
- [ ] Fix the stale `_folder_discoveries` entry: in
      `WorkspaceCreateModal._add_folder`, a folder that is removed and later
      re-added at a moment when its `.SKILLS/` folder no longer has entries
      (deleted, emptied, or now-invalid between the two adds) keeps showing
      the OLD discovery from the first add — the assignment
      `if discovery is not None and discovery.entries:
      self._folder_discoveries[resolved_locator] = discovery` only ever
      sets, never clears, so a folder's entry outlives a rescan that finds
      nothing. Either explicitly pop the stale entry when the rescan is
      empty, or clear `_folder_discoveries[locator]` on `_remove_folder` (the
      entry is currently left in the dict, unreachable via `self._folders`
      but not deleted, when a folder is removed at all)
- [ ] Decide (fix, or explicitly accept and document) whether
      `discover_project_skills()` inside `_add_folder`'s synchronous button
      handler should move off the main thread — it is a bounded scan (50
      entries / 64 KiB reads, non-recursive) but still a blocking
      filesystem walk on the UI thread, and a `.SKILLS/` folder that
      resolves onto a slow or hung network mount would stall the whole
      modal for as long as the OS filesystem call takes to time out or
      return

## Notes

Source: TASK-17651 close-out (ADR/docs/live-verification task). Related:
TASK-17961 (pre-existing, separately-tracked focused-widget rendering defect
in the same `WorkspaceCreateModal`, reproduced again during this task's live
verification in both the Name/folder Input fields and, in a new sighting, the
`ProjectSkillsImportModal`'s offer-phase Checkboxes and Library's
"Search Library…"/"Filter skills…" inputs — not re-filed here since
TASK-17961 already tracks the underlying rendering characteristic across
surfaces).
