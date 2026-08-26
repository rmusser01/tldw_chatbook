---
id: TASK-17964
title: >-
  Project-skills follow-ups: offer-modal footer tests, checkbox-escape
  assertion, off-thread discovery in _add_folder
status: Done
assignee: []
created_date: '2026-08-18 00:00'
labels:
  - skills
  - workspaces
  - testing
priority: low
dependencies:
  - TASK-18705
---

## Description (the why)

TASK-18705's close-out review and live verification found a handful of
small, non-blocking gaps left by the project-skills import feature (Tasks
1-5 of `Docs/superpowers/plans/2026-08-17-project-skills-import.md`). None of
these affect the shipped behavior's correctness for the common paths already
covered by `Tests/Skills/test_project_skills_import_modal.py` and
`Tests/Workspaces/test_workspace_create_modal.py`, but they are real,
identified gaps worth closing.

## Acceptance Criteria (the what)

- [x] `ProjectSkillsImportModal`'s offer-phase footer (`_offer_footer_lines`
      in `tldw_chatbook/Widgets/project_skills_import_modal.py`) has test
      coverage for both its branches: a discovery with `skipped` entries
      renders `Skipped: <names>`, and a discovery with `truncated > 0`
      renders `N more not shown` — neither branch is exercised by any
      existing test today
- [x] A test asserts Checkbox-label escaping in the offer phase: a
      malicious/markup-hostile skill `description` (e.g. containing
      `[bold]`/`[/bold]` or other Rich markup) renders as literal text in
      the Checkbox label, not interpreted markup — the discovery-layer
      fixtures already used elsewhere in `Tests/Skills/` for
      markup-hostile names/descriptions can be reused; today only the
      `escape_markup()` call site in `_compose_offer_phase` defends this,
      with no assertion pinning it
- [x] Fix the "1 project skill(s)" pluralization in
      `tldw_chatbook/Widgets/workspace_create_modal.py` (`compose()`'s
      folder-row label and `_add_folder`'s discovery-count usage) so a
      single discovered skill reads "1 project skill" and multiple read "N
      project skills" (live-verified as-is during TASK-18705's Step 4:
      `... — contains 1 project skill(s)`, grammatically wrong for the
      singular case)
- [x] Fix the stale `_folder_discoveries` entry: in
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
- [x] Decide (fix, or explicitly accept and document) whether
      `discover_project_skills()` inside `_add_folder`'s synchronous button
      handler should move off the main thread — it is a bounded scan (50
      entries / 64 KiB reads, non-recursive) but still a blocking
      filesystem walk on the UI thread, and a `.SKILLS/` folder that
      resolves onto a slow or hung network mount would stall the whole
      modal for as long as the OS filesystem call takes to time out or
      return

## Notes

Source: TASK-18705 close-out (ADR/docs/live-verification task). Related:
TASK-17961 (pre-existing, separately-tracked focused-widget rendering defect
in the same `WorkspaceCreateModal`, reproduced again during this task's live
verification in both the Name/folder Input fields and, in a new sighting, the
`ProjectSkillsImportModal`'s offer-phase Checkboxes and Library's
"Search Library…"/"Filter skills…" inputs — not re-filed here since
TASK-17961 already tracks the underlying rendering characteristic across
surfaces).
- [x] Decide (timeout vs. accepted trade-off) the in-flight-import dismissal posture: post final-fix, Escape/Not now/Never are inert while an import runs, so a permanently-hung importer leaves no in-modal exit (final-review named risk (c), noted by the fix-wave re-review as inherent to not discarding partial imports silently)

## Implementation Plan (the how)

1. Reconcile each AC against the current tree first: the stale
   `_folder_discoveries` bug was reportedly fixed by PR #1810's final
   fix-wave — verify with a passing test before assuming any of the rest is
   still open.
2. Add the two missing `_offer_footer_lines` branch tests (skipped names,
   truncated count) to `Tests/Skills/test_project_skills_import_modal.py`.
3. Add the checkbox-escape assertion, using a YAML-quoted hostile
   description (unquoted brackets break the frontmatter's YAML grammar and
   degrade to an empty description — verified against the existing
   `test_project_skills_discovery.py` fixtures for this exact failure
   mode) rather than this file's own `_discovery()` helper, which uses an
   unquoted hostile string and would silently test nothing.
4. Fix the "1 project skill(s)" pluralization in
   `workspace_create_modal.py`'s folder-row label; strengthen the two
   existing tests that substring-matched the old buggy string to assert
   the exact, corrected grammar.
5. Record the two controller-ruled accepted-tradeoff decisions
   (off-thread discovery, hung-import dismissal posture) as comments next
   to the code they describe, per the controller's rulings — no behavior
   change for either.
6. Run the full gate and a repo-wide `--collect-only` sweep.

## Implementation Notes

**Reconciliation (ticked-by-evidence vs. newly implemented):**

- Footer tests (skipped/truncated): NOT covered before this task —
  implemented `test_offer_footer_renders_skipped_and_truncated_lines`
  (`Tests/Skills/test_project_skills_import_modal.py`), using
  `dataclasses.replace()` on a real discovery to set `skipped`/`truncated`
  without depending on the scan's own caps.
- Checkbox-escape assertion: NOT covered before this task — implemented
  `test_offer_phase_checkbox_label_escapes_markup_literally`. Trap hit:
  this file's shared `_discovery()` fixture uses an **unquoted**
  `[red]desc[/red] for {name}` description, which breaks the frontmatter's
  YAML grammar and silently degrades to an **empty** description (per
  `ProjectSkillEntry`'s own docstring and
  `test_project_skills_discovery.py::test_unparseable_frontmatter_
  degrades_to_empty_description`) — so reusing it here would have produced
  a vacuous pass (asserting markup escaping on an empty string). Built a
  dedicated YAML-quoted fixture (`description: "[red]evil[/red]"`,
  matching `test_hostile_description_survives_as_plain_data`'s working
  pattern) instead, with a sanity assertion on `entry.description` before
  checking the rendered `Checkbox.label` (a Textual `Content` object,
  whose `.plain`/`.spans` show the escaped brackets rendered literally
  with no color span).
- Pluralization: NOT fixed before this task — fixed in `compose()`'s
  folder-row label (`workspace_create_modal.py`); there was only one live
  code call site (the task's AC text implied two, but `_add_folder`'s
  own reference was a comment describing the same rendering, not a second
  formatting site — the comment was updated too for accuracy). Strengthened
  the two existing tests that substring-matched the old "1 project
  skill(s)" text (`test_folder_with_skills_annotated_and_carried_on_result`,
  `test_removed_and_rescanned_folder_clears_stale_discovery`) to assert the
  exact corrected row text.
- Stale `_folder_discoveries` after remove: **already fixed** — PR #1810's
  final fix-wave (Finding 9) made `_remove_folder` pop the locator and
  `_add_folder` assign-or-pop on a rescan that finds nothing, and it's
  pinned by the existing
  `test_removed_and_rescanned_folder_clears_stale_discovery`
  (`Tests/Workspaces/test_workspace_create_modal.py:480-523`) — ticked
  with evidence, no code change needed.
- Off-thread discovery — **controller ruling, accepted and documented, not
  implemented**: recorded as a comment directly above the
  `discover_project_skills()` call in `_add_folder`
  (`workspace_create_modal.py`). Rationale: the scan is hard-bounded (500
  scanned children / 50 recognized entries / 64 KiB reads, non-recursive);
  the synchronous `validate_folder_binding_path()` call immediately above
  it already stats the same filesystem, synchronously, on the same UI
  thread, before discovery ever runs — a dead network mount hangs the
  modal at validation regardless of what discovery does, so moving only
  discovery off-thread cannot fix the named hang. Async validation (and
  discovery) together is a separate, differently-scoped task if the hang
  is ever actually observed.
- Hung-import dismissal posture — **controller ruling, accepted and
  documented, not implemented**: recorded in `_import_in_flight`'s
  docstring in `project_skills_import_modal.py`. Rationale: imports here
  are local-filesystem only, so a hung importer implies a hung filesystem
  (the same failure class as the discovery decision above); discarding a
  partial import silently — the only alternative while the worker is
  mid-flight — is worse than leaving no escape hatch for a
  previously-unseen pathological case, since it risks a half-imported
  skill directory the user has no way to detect.

**Files touched:** `tldw_chatbook/Widgets/workspace_create_modal.py`,
`tldw_chatbook/Widgets/project_skills_import_modal.py`,
`Tests/Skills/test_project_skills_import_modal.py`,
`Tests/Workspaces/test_workspace_create_modal.py` (pluralization assertion
strengthening, shared with TASK-17962's edits to the same file).

**Gate:** `Tests/Skills/` 436 passed; `Tests/Workspaces/` 279 passed;
repo-wide `--collect-only` sweep: 52305 tests collected, zero collection
errors.
