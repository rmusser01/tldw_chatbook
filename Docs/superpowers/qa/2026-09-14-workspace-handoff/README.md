# Workspace handoff refresh evidence

TASK-32462.1; baseline `cee85b4a27`, branch `feat/component-pattern-library`.

After a source snapshot, Library now refreshes the mounted Active/Handoff rows,
Use in Console state and tooltip, and empty-source Import action from the same
cached workspace projection. The controls remain mounted. The existing handler
still blocks cross-workspace handoff and explains how to copy or link sources.

The rail also remembers disclosure changes in its cached preferences and accepts
Diagnostics rows mounted after its initial snapshot. A wrapped status can move
the focused button; the rail reveals it after its virtual size changes. No
permission rules, storage schema, token values or stylesheets changed.

## Product repair versus test corrections

- Product: four registered sources already produced the correct projection, but
  the mounted Workspace body retained the initial no-sources state.
- Product: opening Details updated the DOM without updating cached disclosure
  preferences; lazily mounted Diagnostics also invalidated the old shape check.
  Both triggered unnecessary control replacement on the next source refresh.
- Product: the ready-to-blocked status expansion could leave Use in Console
  focused on the rail border. A compositor assertion exposed this after ordinary
  focus and identity assertions passed. A screen refresh callback ran before the
  changed layout; the rail's existing virtual-size/fold update is the correct
  boundary for revealing the retained focus.
- Harness: all thirteen original cases now load `TldwCli.CSS_PATH`. This alone
  repaired the mouse-create and scroll geometry cases in the prior audit.
- Assertions: two original cases expected eligible/blocked count wording that
  task-32357 had replaced. They now assert the shipped blocker and remedy copy.

## Verification

- Entire Workspace depth file: **15 passed**. The two new theme cases cover
  empty → eligible → mixed/blocked → failure → recovery; retained control
  identity, disclosure, actual button paint, focus, action state, tooltip,
  Import visibility and a renamed workspace containing literal markup.
- Neighboring selection: **41 passed**, 79 deselected. Rail focus, rail Details
  and fold behavior, token governance, and source/landing/handoff reconciliation.
- Two disposable production-styled capture cases passed. Both SVGs were rendered
  through Quick Look and visually inspected: blocker/remedy and focused action
  are readable above the cue in both themes.
- Native app at 120×45, with all ten configured DB paths and data/user directories
  isolated: a real saved audit note linked to Handoff review displays `1 eligible`
  and Use in Console. Final native exit was **0**. No model request was made.
- Ruff check and format pass for the changed test and rail files and archived
  probe. The large pre-existing Library screen has **205 Ruff diagnostics before
  and after**, with no added diagnostic; all three changed/new method fragments
  pass formatting. `git diff --check` passes.

The two pytest warnings concern removal of old temporary Kokoro test directories.
Native startup is not entirely clean: it logs the existing missing
`#app-log-display` widget and, on restart, a Console sidebar load error for
`_sidebar_state_save_timer`. Both occur before Library entry in code unchanged by
this repair. Optional dependency and project-skills worker warnings are retained
in `native-log-review.txt`/`startup-exit.ansi`; these are not Workspace failures.

The broader TASK-32462 remains open. Its footer, other entry-compose failures and
Prompts canvas debt are outside this slice; this branch has not been merged into
dev. No full test sweep was run.

## Reproduce

```sh
.venv/bin/python -m pytest -q Tests/UI/test_post_release_workspaces_library_depth.py
.venv/bin/python -m pytest -q Tests/UI/test_library_rail_focus_visibility.py Tests/UI/test_library_crit9_rail.py Tests/UI/test_design_token_governance.py Tests/UI/test_library_entry_compose_once.py -k 'not test_library_entry_compose_once or source_snapshot or landing_snapshot_sync or study_handoff_snapshot or warm_landing_fresh_reconcile'
```

To reproduce the two captures, copy `probes/_workspace_capture.py` into `Tests/UI/`,
run it with pytest, then remove that copied file. It uses test service fixtures
and real workspace registry state. TXT files are compositor strips; SVGs are
Textual exports; ANSI files are the independent native run. Private config and
DB files remain in ignored scratch. `test-results.json` records the red/green
summaries; raw logs are in `.superpowers/sdd/2026-09-14-workspace-handoff/`.

ADR required: no. Existing ADR-150, ADR-161 and ADR-086 govern this routine
presentation repair; permissions and provider/storage boundaries are unchanged.
