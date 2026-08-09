# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-202 and its absorbed TASK-2700/TASK-2701 defects in the first of six merge-gated PRs: keep every Prompt editor action visible, group actions by purpose, make Copy Markdown real, and confirm single-item deletion.

**Architecture:** Preserve `LibraryScreen` as the interaction coordinator and `LibraryPromptsListCanvas` as a state-driven view. Split the editor into a scrollable content region plus an auto-height action region; route live working-copy Markdown through the existing renderer and app clipboard seam; add one reusable Prompt deletion confirmation modal that TASK-203 can reuse. No service or storage boundary changes.

**Tech Stack:** Python 3.11+, Textual 8, pytest with `App.run_test()`, TCSS source plus generated bundle, Backlog.md CLI.

---

## Merge Gate and Scope

- This is PR 1 of 6. Reuse this isolated design worktree so its approved,
  already-committed specification and plan commits remain in the branch; fetch
  and rebase those commits onto the latest merged `origin/dev`, then rename the
  branch for TASK-202 before any Backlog or production edit.
- Carry the approved umbrella spec `Docs/superpowers/specs/2026-08-02-library-prompt-enhancement-series-design.md` and all six plan documents in this PR; do not open a separate design PR.
- Close TASK-202, TASK-2700, and TASK-2701 in this one PR. Do not include TASK-2702 or any TASK-196 implementation.
- ADR required: no.
- ADR path: N/A.
- Reason: this is UI grouping and defect repair within ADR-011 and ADR-040.

## File Responsibility Map

- Modify `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`: editor shell, stable action groups, conflict action layout.
- Create `tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py`: reusable single/bulk confirmation view and typed result.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: Copy Markdown handler, clipboard outcomes, confirmed single delete.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerate `tldw_chatbook/css/tldw_cli_modular.tcss`: fixed footer geometry, narrow stacking, primary/danger treatment.
- Modify `Tests/UI/test_library_prompts_canvas.py`: grouping, focus order, live copy, clipboard failures, delete confirmation, geometry.
- Create `Tests/UI/test_prompt_delete_confirmation_modal.py`: modal contract in isolation.
- Modify `Tests/UI/test_css_build_integrity.py` and/or `Tests/UI/test_css_bundle_sync_guard.py` only if the new selectors need guard coverage.
- Modify `Docs/User_Guide/library/prompts.md`: Copy Markdown and confirmed deletion behavior.
- Modify the three task records in `backlog/tasks/` with complete, separate hygiene.

## Task 1: Convert the Approved Design Worktree into the PR Baseline

- [ ] In `.worktrees/prompt-library-enhancement-series-design`, verify the worktree is clean and that the approved spec and all six plans are committed.
- [ ] Fetch `origin/dev`, rebase the design commits onto its latest tip, and rename the current branch for TASK-202. Do not create a second worktree or copy untracked design files between worktrees.

```bash
git status --short
git fetch origin dev
git log -1 --oneline origin/dev
git rebase origin/dev
git branch -m codex/task-202-prompt-editor-actions
git merge-base --is-ancestor origin/dev HEAD
git status --short
```

Expected: the existing isolated worktree is clean on
`codex/task-202-prompt-editor-actions`; `origin/dev` is an ancestor; the
approved spec and six plan files remain tracked.

- [ ] Put all three records In Progress and add task-specific plans/acceptance criteria before production edits.

```bash
backlog task edit 202 -a @codex -s "In Progress" --plan "Implement grouped, always-visible editor actions; wire live Copy Markdown; add shared delete confirmation; verify TCSS geometry. ADR required: no; ADR path: N/A; reason: UI-only change under ADR-011/ADR-040."
backlog task edit 2700 -a @codex -s "In Progress" --plan "Implemented in the TASK-202 PR; add the missing handler through LibraryScreen and reuse the canonical Prompt Markdown renderer. ADR required: no; ADR path: N/A; reason: UI-only defect repair under ADR-011/ADR-040."
backlog task edit 2701 -a @codex -s "In Progress" --plan "Implemented in the TASK-202 PR; split the editor into a scrollable body and auto-height footer and add geometry regressions. ADR required: no; ADR path: N/A; reason: UI-only defect repair under ADR-011/ADR-040."
```

- [ ] Add each acceptance criterion as a separate checkbox line in the corresponding task file; do not pass comma-separated criteria to `backlog task edit --ac`.
- [ ] Expand TASK-202's acceptance criteria from the approved spec, including stable action IDs, confirmation, and narrow layouts.
- [ ] Confirm the umbrella spec and all six plan documents are present in the rebased history, then commit the three Backlog start-state changes before code.

## Task 2: Characterize the Broken Geometry and Dead Copy Action

- [ ] Add failing tests in `Tests/UI/test_library_prompts_canvas.py` that mount an editor at 80x24, 100x30, 140x40, and 200x50 and assert the action region has a visible nonzero region and does not overlap the last body control.
- [ ] Assert the normal action order by stable IDs: Save, Use in Console, Export, Copy Markdown, Duplicate, Delete. Assert conflict order: Save as new, Reload.
- [ ] Add failing tests that edit the live System/User controls without saving, press `#library-prompt-copy`, and expect the exact output of `render_prompt_markdown` for that working copy.
- [ ] Add failing tests for missing clipboard support and a clipboard exception; both must avoid success copy.
- [ ] Run the focused red tests.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -k "geometry or copy or action_group" -q
```

Expected: failures demonstrate the clipped footer, missing handler, and missing grouping.

## Task 3: Add the Reusable Delete Confirmation Contract

- [ ] Write failing tests in `Tests/UI/test_prompt_delete_confirmation_modal.py` for single Prompt, single Recipe, dirty working copy, bounded bulk preview, Cancel, Confirm, and markup-looking names rendered literally.
- [ ] Implement `PromptDeleteConfirmationModal` with immutable constructor data and one result type such as `PromptDeleteDecision(confirmed: bool, fingerprint: str | None)`.
- [ ] Keep the modal free of DB and scope-service calls; hosts own execution. Expose singular/plural copy and a bounded preview so TASK-203 can reuse it without a second modal.
- [ ] Ensure dirty single-delete copy says both the saved artifact and unsaved working copy will be discarded.
- [ ] Run the modal tests green.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_prompt_delete_confirmation_modal.py -q
```

Expected: all modal tests pass.

## Task 4: Rebuild the Editor Shell and Group Actions

- [ ] Refactor `_compose_editor` in `library_prompts_canvas.py` to yield a bounded shell with a `VerticalScroll` body and an auto-height action area; preserve every existing field/action ID.
- [ ] Add semantic containers/classes for `primary`, `content`, and `lifecycle` groups. Keep DOM and keyboard order identical to the approved order.
- [ ] Render conflict actions in the same always-visible action area rather than in the scroll body.
- [ ] Update `_agentic_terminal.tcss` with `min-height: 0`, `height: 1fr`, and `height: auto` ownership; stack groups at narrow widths without reordering them. Apply existing primary and danger tokens rather than new colors.
- [ ] Regenerate the compiled bundle.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.css.check_bundle_sync
```

Expected: bundle sync succeeds.

## Task 5: Wire Live Copy Markdown and Confirmed Single Delete

- [ ] Add a `#library-prompt-copy` handler in `library_screen.py` that reads `_read_library_prompt_editor_fields()` plus the mounted block state, builds the current working-copy record, and calls the same `render_prompt_markdown` used by individual export.
- [ ] Call the established app clipboard seam. Announce success only after it returns successfully; distinguish unavailable support from an exception without logging Prompt bodies.
- [ ] Change the current immediate `#library-prompt-delete` handler to open `PromptDeleteConfirmationModal` and execute the existing delete worker only on Confirm.
- [ ] Capture the prompt identity before opening the modal and reject a result if the editor identity changed before settlement.
- [ ] Preserve existing delete soft-delete/service routing and refresh behavior.
- [ ] Run the focused UI tests green.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_prompt_delete_confirmation_modal.py Tests/UI/test_library_prompts_canvas.py -k "copy or delete or geometry or action" -q
```

Expected: all selected tests pass.

## Task 6: Document and Visually Inspect the Result

- [ ] Update `Docs/User_Guide/library/prompts.md`: rename the user-facing action to Copy Markdown, state that it copies unsaved edits, and replace immediate-delete wording with confirmation behavior.
- [ ] Use a small Textual harness or existing pilots to render 80x24, 100x30, 140x40, and 200x50 in normal, dirty, conflict, and confirmation states.
- [ ] Inspect for clipping, body/footer overlap, nested-scroll traps, illogical tab order, unreadable grouping, and literal-text rendering. Revise and rerun tests if any defect appears.

## Task 7: Verify, Review, and Close All Three Records

- [ ] Run affected suites and static checks.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py Tests/UI/test_prompt_delete_confirmation_modal.py Tests/UI/test_css_build_integrity.py Tests/UI/test_css_bundle_sync_guard.py -q
git diff --check
```

- [ ] Run the full suite before merge and record the exact result in each task's notes.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
```

- [ ] Self-review for action loss, stale modal results, accidental body logging, generated CSS drift, and regressions to Recipe/conflict actions. Request independent code review and address every valid finding.
- [ ] Check every acceptance criterion and add separate concise implementation notes to TASK-202, TASK-2700, and TASK-2701, including tests, docs, and ADR check.
- [ ] Mark all three tasks Done only after checks and review pass.

```bash
backlog task edit 202 -s Done --notes "Grouped the always-visible Prompt editor actions, wired live Copy Markdown, reused confirmed deletion, updated docs, and verified focused/full suites. ADR required: no; ADR path: N/A."
backlog task edit 2700 -s Done --notes "Added the Copy Markdown handler for live unsaved content with honest clipboard outcomes; verified UI regression tests. ADR required: no; ADR path: N/A."
backlog task edit 2701 -s Done --notes "Made the editor body scroll independently of an auto-height action area and verified required terminal sizes. ADR required: no; ADR path: N/A."
```

- [ ] Open one ready PR against `dev`, address CI/review, merge, and confirm the merge commit is present on `origin/dev`. Do not begin TASK-196 implementation before that confirmation.
