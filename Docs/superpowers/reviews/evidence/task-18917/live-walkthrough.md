# TASK-18917 live walkthrough evidence

Date: 2026-08-29

## Production-shaped mounted matrix

Command:

```text
../../.venv/bin/python -m pytest Tests/UI/test_library_adaptive_reader_closeout.py -q -k 'notes_branch_paging_is_contained_focusable_and_collapsible_in_production_shell'
```

Result: `4 passed, 7 deselected`.

The mounted harness uses the real `LibraryScreen`, adaptive-reader shell,
Notes canvas, and the repository stylesheet hierarchy. At 160×50, 120×35,
100×30, and 80×24 it exercised long identifying titles, independent root and
expanded-folder folder/note pagers, branch-local scroll preservation, a gated
one-shot failure, stable Retry focus, focus on the first appended note,
Library/Items collapse, compositor text, and horizontal containment. The
unchanged closeout cases continue to cover Conversations, Media, Prompts, and
Skills destination identities and reader behavior.

### RED and correction evidence

- At 160×50 the real compositor initially allocated the Items owner at
  `x=36 width=40`, but the Notes canvas/list painted at `x=38 width=40`, two
  cells beyond the Items content edge. Constraining the Notes canvas/list to
  `1fr`, `min-width: 0`, and `max-width: 100%` produced `x=38 width=36`, fully
  inside the Items edge at column 76.
- Explicitly closing Items was immediately undone by Notes list-priority
  reconciliation. The explicit close now suppresses that automatic priority,
  leaving Items closed and expanding Work.
- At 100×30 and 80×24, opening Library evicted Items as intended, but closing
  Library could leave both navigation panes closed: the explicit Items
  priority fit at the breakpoint and was then rejected by stale hysteresis.
  The explicit Notes interaction now resolves without the prior automatic
  layout, reopening Items and giving the reclaimed width to titles.
- At 80×24 the identifying long title is intentionally ellipsized by the
  compositor; its identifying text remains painted and contained. This was a
  test-expectation correction, not a production defect.

## Isolated real-repository walkthrough

Command:

```text
../../.venv/bin/python -m pytest Tests/Live/test_library_notes_tree_paging_live.py -q -s
```

Result: `1 passed`.

The test creates a `tmp_path` ChaChaNotes SQLite database and drives
`LocalNoteFolderRepository` through `NotesScopeService` in the mounted
production-shaped Notes reader. It seeds 25 roots, 25 Unfiled notes, one
parent with 25 children and 45 visible placements, a deep managed descendant,
a duplicate manual/managed placement, and a managed placement shadowed at its
ancestor. Repository assertions confirm exact totals of 25 root folders, 25
children, 25 Unfiled placements, 45 primary placements, and one visible
deepest placement.

The Pilot walkthrough loads root and child folder continuations, expands the
primary folder, loads placement continuation, gates a one-shot page failure,
observes branch-local Retry, retries to 40 loaded placements, locates and
focuses a placement in the 20–39 middle page, loads Earlier back to offset 0,
renames and refreshes a real child folder, collapses/re-expands the branch,
and walks every required size. Exact final geometry observations were:

| Terminal | Items width | Work width | Observation |
|---|---:|---:|---|
| 160×50 | 40 | 77 | Long titles and both pagers contained |
| 120×35 | 56 | 50 | Items receives available comfort width |
| 100×30 | 42 | 48 | Items remains reachable at the breakpoint |
| 80×24 | 32 | 38 | Identifying text ellipsizes without overflow |

All four compositors painted the Notes/Library navigation context, the Notes
list stayed within the Items owner, and no horizontal overflow was observed.

## Targeted and static closeout

The exact Task 8 feature/cross-reader command completed with `695 passed in
218.03s`. This includes the Notes models, real repository, scope service,
pure paging and tree state, mounted navigator/canvas/shell/closeout coverage,
and the isolated live walkthrough; it is not the full repository suite.

`tldw_chatbook/css/build_css.py` completed successfully and regenerated the
modular bundle plus scoped widget defaults from the corrected source. Ruff on
the changed Python files, `git diff --check`, and the requested `compileall`
over Notes, Library, Library widgets, and `library_screen.py` all exited 0.
The post-commit rebuild changed only the modular bundle's designed wall-clock
`Generated:` header; a diff ignoring only that header exited 0. The refreshed
header was recorded in the closeout commit, after which all tracked generated
files and the worktree were clean. No stylesheet content drift was present.
